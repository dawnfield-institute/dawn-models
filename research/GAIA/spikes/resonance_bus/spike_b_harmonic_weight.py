"""Spike B — Harmonic-Weighted ResonanceBus.

Hypothesis: pure delta-coherence (magnitude * sign_alignment) only rewards
modules that make LARGE, SIGN-ALIGNED changes. This misses modules that:
  1. Reshape the frequency spectrum (Reasoning/Mobius) without flipping signs
  2. Smooth/predict at fine scales (Language) with subtle structure
  3. Add small but structurally rich overlays (Memory context blend)

Fix: replace phase_coherence with harmonic_resonance — multi-scale cosine
similarity with golden-ratio weighting. This captures structural alignment
at multiple frequency scales, not just sign agreement.

Weight = delta_magnitude * harmonic_resonance(delta, input)

This should distribute weight more evenly because:
- Safety: large delta, moderate harmonic resonance => still high weight
- Reasoning: moderate delta, HIGH harmonic resonance (Mobius = multi-scale) => gains weight
- Memory: small delta, HIGH harmonic resonance (context = structurally similar) => gains weight
- Language: small delta, moderate harmonic resonance => some weight
- Observability: zero delta => zero weight (correct)

Additionally, this spike adds a "resonance floor" — a minimum weight
for any module with harmonic_resonance > threshold, preventing total
suppression of structurally-aligned modules.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import torch

import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
_gaia_root = os.path.join(_here, "..", "..")
sys.path.insert(0, os.path.join(_gaia_root, "src"))
sys.path.insert(0, os.path.join(_here, "..", "..", "..", "..", "..", "fracton"))

from gaia.core.exceptions import ConservationViolation, ModuleRegistrationError
from gaia.core.protocol import GAIAModule
from gaia.core.sec_router import SECRouter
from gaia.core.types import ConservationResult, FieldState, SECPhase

# Local implementations - Fracton harmonic_resonance has odd-length bug
# from fracton.field.resonance import harmonic_resonance, phase_coherence
if True:
    def phase_coherence(a: torch.Tensor, b: torch.Tensor) -> float:
        signs_a = torch.sign(a.flatten().float())
        signs_b = torch.sign(b.flatten().float())
        return float((signs_a == signs_b).float().mean().item())

    def harmonic_resonance(a: torch.Tensor, b: torch.Tensor, harmonics: int = 3) -> float:
        PHI = 1.618033988749895
        total, weight_sum = 0.0, 0.0
        ca, cb = a.flatten().float(), b.flatten().float()
        for h in range(harmonics):
            w = 1.0 / (PHI ** h)
            na, nb = torch.norm(ca), torch.norm(cb)
            if na > 1e-10 and nb > 1e-10:
                total += w * float(torch.dot(ca, cb) / (na * nb))
            weight_sum += w
            if ca.shape[0] >= 2:
                n = ca.shape[0] - (ca.shape[0] % 2)
                ca = (ca[:n:2] + ca[1:n:2]) / 2
                cb = (cb[:n:2] + cb[1:n:2]) / 2
            else:
                break
        return total / weight_sum if weight_sum > 0 else 0.0

logger = logging.getLogger(__name__)


@dataclass
class HarmonicWeight:
    """Per-module harmonic weight from one dispatch cycle."""
    module_name: str
    delta_magnitude: float       # ||output - input||
    harmonic_resonance: float    # multi-scale cosine(delta, input)
    phase_coherence: float       # sign alignment (for comparison)
    raw_weight: float            # delta_magnitude * harmonic_resonance
    floor_applied: bool          # was the resonance floor used?
    normalized_weight: float


class HarmonicBus:
    """Spike B: Harmonic-weighted resonance bus.

    Same broadcast + superpose architecture as ResonanceBus, but uses
    harmonic_resonance instead of phase_coherence for weighting.

    Also adds a resonance floor: any module with harmonic_resonance above
    a threshold gets at least `floor_weight` in the raw weighting, even
    if its delta magnitude is small.
    """

    def __init__(
        self,
        enforcement: str = "hard",
        tolerance: float = 1e-6,
        rbf_lambda: float = 1.0,
        rbf_alpha: float = 0.1,
        rbf_suppression_threshold: float = 0.0,
        min_weight_epsilon: float = 1e-6,
        resonance_floor_threshold: float = 0.5,
        floor_weight: float = 0.05,
        harmonics: int = 3,
    ) -> None:
        if enforcement not in ("hard", "soft", "monitor"):
            raise ValueError(f"enforcement must be 'hard', 'soft', or 'monitor', got '{enforcement}'")

        self._enforcement = enforcement
        self._tolerance = tolerance
        self._rbf_lambda = rbf_lambda
        self._rbf_alpha = rbf_alpha
        self._rbf_suppression_threshold = rbf_suppression_threshold
        self._min_weight_epsilon = min_weight_epsilon
        self._resonance_floor_threshold = resonance_floor_threshold
        self._floor_weight = floor_weight
        self._harmonics = harmonics
        self._router = SECRouter()
        self._modules: dict[str, GAIAModule] = {}
        self._violation_log: list[ConservationResult] = []
        self._harmonic_log: list[list[HarmonicWeight]] = []

    def register_module(self, module: GAIAModule, phases: list[SECPhase] | None = None) -> None:
        if not isinstance(module, GAIAModule):
            raise ModuleRegistrationError(f"Not a GAIAModule: {type(module).__name__}")
        self._modules[module.name] = module
        self._router.register(module, phases)

    def process(self, field_state: FieldState) -> FieldState:
        """Harmonic dispatch: broadcast -> harmonic weight -> superpose -> PAC."""

        # 1. Classify + route
        input_state = field_state.clone()
        phase = self._router.classify(input_state.entropy)
        input_state.phase = phase
        modules = self._router.route(input_state)
        if not modules:
            return input_state
        modules = self._regulate(modules)
        if not modules:
            return input_state

        # 2. BROADCAST
        outputs: list[tuple[GAIAModule, FieldState]] = []
        for module in modules:
            module_input = input_state.clone()
            output = module.process(module_input)
            outputs.append((module, output))

        # 3. HARMONIC WEIGHT
        weights = self._compute_weights(input_state, outputs)
        self._harmonic_log.append(weights)

        # 4. SUPERPOSE — weighted sum of outputs
        merged_tensor = torch.zeros_like(input_state.tensor)
        for (module, output), w in zip(outputs, weights):
            merged_tensor = merged_tensor + w.normalized_weight * output.tensor

        # 5. PAC BOUNDARY
        input_energy = input_state.total_energy()
        merged_energy = float(torch.sum(merged_tensor).item())

        if abs(merged_energy) > 1e-10:
            merged_tensor = merged_tensor * (input_energy / merged_energy)
        elif abs(input_energy) > 1e-10:
            merged_tensor = input_state.tensor.clone()

        final_energy = float(torch.sum(merged_tensor).item())
        result = ConservationResult(
            conserved=abs(final_energy - input_energy) < self._tolerance * max(abs(input_energy), 1e-10),
            input_energy=input_energy,
            output_energy=final_energy,
            residual=abs(final_energy - input_energy),
            module_name="harmonic_merge",
        )
        if not result.conserved:
            self._violation_log.append(result)
            self._handle_violation(result)

        # 6. Recompute
        entropy = self._shannon_entropy(merged_tensor)
        new_phase = self._router.classify(entropy)
        provenance = [w.module_name for w in weights if w.normalized_weight > self._min_weight_epsilon]

        return FieldState(
            tensor=merged_tensor,
            entropy=entropy,
            phase=new_phase,
            conservation_budget=input_state.conservation_budget,
            provenance=provenance,
            timestamp=time.time(),
        )

    def _compute_weights(
        self,
        input_state: FieldState,
        outputs: list[tuple[GAIAModule, FieldState]],
    ) -> list[HarmonicWeight]:
        """Weight by delta_magnitude * harmonic_resonance(delta, input).

        Adds resonance floor for structurally-aligned modules with small deltas.
        """
        weights: list[HarmonicWeight] = []

        for module, output in outputs:
            delta = output.tensor - input_state.tensor
            mag = float(torch.norm(delta).item())

            if mag < self._min_weight_epsilon:
                # Identity module
                hr = 0.0
                pc = 1.0
                raw = self._min_weight_epsilon
                floor_applied = False
            else:
                hr = harmonic_resonance(delta, input_state.tensor, harmonics=self._harmonics)
                pc = phase_coherence(delta, input_state.tensor)

                # Core weight: magnitude * harmonic resonance
                # Harmonic resonance can be negative (anti-resonant) — clamp to 0
                hr_clamped = max(0.0, hr)
                raw = mag * hr_clamped

                # Resonance floor: if harmonically aligned but small delta,
                # ensure minimum participation
                floor_applied = False
                if hr > self._resonance_floor_threshold and raw < self._floor_weight:
                    raw = self._floor_weight
                    floor_applied = True

            weights.append(HarmonicWeight(
                module_name=module.name,
                delta_magnitude=mag,
                harmonic_resonance=hr,
                phase_coherence=pc,
                raw_weight=raw,
                floor_applied=floor_applied,
                normalized_weight=0.0,
            ))

        # Normalize
        total_raw = sum(w.raw_weight for w in weights)
        if total_raw > 1e-10:
            for w in weights:
                w.normalized_weight = w.raw_weight / total_raw
        else:
            n = len(weights)
            for w in weights:
                w.normalized_weight = 1.0 / n if n > 0 else 0.0

        return weights

    @staticmethod
    def _shannon_entropy(tensor: torch.Tensor) -> float:
        flat = tensor.flatten().float().abs()
        total = flat.sum()
        if total < 1e-12 or flat.numel() == 0:
            return 0.0
        probs = flat / total
        mask = probs > 1e-12
        probs = probs[mask]
        return float(-(probs * probs.log()).sum().item())

    def _regulate(self, modules: list[GAIAModule]) -> list[GAIAModule]:
        healthy = []
        for module in modules:
            rbf = module.health()
            if rbf.balance >= self._rbf_suppression_threshold:
                healthy.append(module)
        return healthy

    def _handle_violation(self, result: ConservationResult) -> None:
        msg = (f"PAC violation in '{result.module_name}': "
               f"residual={result.residual:.6e}")
        if self._enforcement == "hard":
            raise ConservationViolation(result, result.module_name)
        elif self._enforcement == "soft":
            logger.warning("SOFT: %s", msg)

    @property
    def violation_log(self): return list(self._violation_log)
    @property
    def harmonic_log(self): return self._harmonic_log
    @property
    def enforcement(self): return self._enforcement
    def get_metrics(self):
        return {
            "modules_registered": len(self._modules),
            "module_names": list(self._modules.keys()),
            "enforcement": self._enforcement,
            "tolerance": self._tolerance,
            "total_violations": len(self._violation_log),
            "harmonic_ticks": len(self._harmonic_log),
            "dispatch_mode": "harmonic",
            "resonance_floor_threshold": self._resonance_floor_threshold,
            "floor_weight": self._floor_weight,
        }
