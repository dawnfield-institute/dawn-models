"""Spike C — Perspective-Weighted ResonanceBus.

Insight (Peter, 2026-03-22): "What if harmonic IS identity?"

You and your dad look at the same field and see different things because
your PAC trees — your accumulated histories — normalize your perception
differently. Neither is more "correct." The harmonic resonance between
a module's output and the input field measures how deeply that module's
perspective aligns with what's there.

This is fundamentally different from delta-coherence:
  - Delta-coherence: "how much did you change, and was it constructive?"
  - Perspective: "how do YOU see this field, given everything you've seen before?"

Key consequences:
  - Identity modules (observability) get harmonic_resonance = 1.0 (PERFECT
    resonance). They see the field clearly and say "no change needed." That
    voice was previously silenced.
  - Modules that radically transform (Safety, Reasoning) get lower harmonic
    resonance with the input — they see something DIFFERENT. That difference
    IS their contribution, but it's weighted by how much of the original
    structure they preserved.
  - The PAC tree behind each module conditions its perspective. A module
    with deep history might produce an output that barely resembles the
    input (low harmonic resonance) — a minority perspective. Just because
    the normalization says "this is what I see" doesn't mean it's right.

Weight formula:
    w = harmonic_resonance(module_output, input)

No delta magnitude. No floor. Just: how deeply do you resonate with
what's here? The field decides who gets heard.
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

# Local implementations — Fracton's harmonic_resonance has odd-length bug
def phase_coherence(a: torch.Tensor, b: torch.Tensor) -> float:
    """Sign alignment measure [0, 1]."""
    signs_a = torch.sign(a.flatten().float())
    signs_b = torch.sign(b.flatten().float())
    return float((signs_a == signs_b).float().mean().item())


def harmonic_resonance(a: torch.Tensor, b: torch.Tensor, harmonics: int = 3) -> float:
    """Multi-scale cosine similarity with golden-ratio weighting."""
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
class PerspectiveWeight:
    """Per-module perspective weight from one dispatch cycle."""
    module_name: str
    output_input_resonance: float   # harmonic_resonance(output, input)
    delta_magnitude: float          # ||output - input|| (for diagnostics only)
    raw_weight: float               # max(0, output_input_resonance)
    normalized_weight: float


class PerspectiveBus:
    """Spike C: Perspective-weighted resonance bus.

    Weight = harmonic_resonance(module_output, input_field)

    Each module broadcasts its perspective on the field. The bus
    weights these perspectives by how harmonically they resonate
    with the raw input. Identity = perfect resonance. Radical
    transformation = lower resonance (different perspective).

    The merged output is a harmonic superposition of all perspectives,
    PAC-scaled at the outer boundary.
    """

    def __init__(
        self,
        enforcement: str = "hard",
        tolerance: float = 1e-6,
        rbf_lambda: float = 1.0,
        rbf_alpha: float = 0.1,
        rbf_suppression_threshold: float = 0.0,
        min_weight_epsilon: float = 1e-6,
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
        self._harmonics = harmonics
        self._router = SECRouter()
        self._modules: dict[str, GAIAModule] = {}
        self._violation_log: list[ConservationResult] = []
        self._perspective_log: list[list[PerspectiveWeight]] = []

    def register_module(self, module: GAIAModule, phases: list[SECPhase] | None = None) -> None:
        if not isinstance(module, GAIAModule):
            raise ModuleRegistrationError(f"Not a GAIAModule: {type(module).__name__}")
        self._modules[module.name] = module
        self._router.register(module, phases)

    def process(self, field_state: FieldState) -> FieldState:
        """Perspective dispatch: broadcast -> resonate -> superpose -> PAC."""

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

        # 2. BROADCAST — each module sees same input, produces its perspective
        outputs: list[tuple[GAIAModule, FieldState]] = []
        for module in modules:
            module_input = input_state.clone()
            output = module.process(module_input)
            outputs.append((module, output))

        # 3. PERSPECTIVE WEIGHT — resonance of output with input
        weights = self._compute_weights(input_state, outputs)
        self._perspective_log.append(weights)

        # 4. SUPERPOSE — weighted sum of perspectives
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
            module_name="perspective_merge",
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
    ) -> list[PerspectiveWeight]:
        """Weight = harmonic_resonance(output, input).

        No delta magnitude. No floor. Just perspective alignment.
        The field decides who gets heard.
        """
        weights: list[PerspectiveWeight] = []

        for module, output in outputs:
            # How deeply does this module's perspective resonate with the input?
            resonance = harmonic_resonance(
                output.tensor, input_state.tensor, harmonics=self._harmonics
            )

            # Delta magnitude — tracked for diagnostics, NOT used in weighting
            delta_mag = float(torch.norm(output.tensor - input_state.tensor).item())

            # Raw weight is just the resonance, clamped to non-negative
            # (anti-resonant perspectives get silenced — they contradict the field)
            raw = max(0.0, resonance)

            weights.append(PerspectiveWeight(
                module_name=module.name,
                output_input_resonance=resonance,
                delta_magnitude=delta_mag,
                raw_weight=raw,
                normalized_weight=0.0,
            ))

        # Normalize
        total_raw = sum(w.raw_weight for w in weights)
        if total_raw > 1e-10:
            for w in weights:
                w.normalized_weight = w.raw_weight / total_raw
        else:
            # All anti-resonant — equal weight fallback
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
    def perspective_log(self): return self._perspective_log
    @property
    def enforcement(self): return self._enforcement
    def get_metrics(self):
        return {
            "modules_registered": len(self._modules),
            "module_names": list(self._modules.keys()),
            "enforcement": self._enforcement,
            "tolerance": self._tolerance,
            "total_violations": len(self._violation_log),
            "perspective_ticks": len(self._perspective_log),
            "dispatch_mode": "perspective",
        }
