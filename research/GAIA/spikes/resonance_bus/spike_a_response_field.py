"""Spike A — Response-Field ResonanceBus.

Hypothesis: modules fail in broadcast mode because they produce *attenuated
versions* of the input (output = input * 0.9). The delta is negative/small,
so delta-coherence correctly assigns near-zero weight.

Fix: wrap each module so it produces a *response field* — the difference
between its output and its input. The bus then combines:

    merged = input + sum(w_i * response_i)

where weights come from the same delta-coherence mechanism.

This way:
- Safety's ConcentrationGate redistributes energy => large response, high weight
- Reasoning's Mobius transform reshapes structure => response captures the reshape
- Memory's context blend adds small overlay => response IS the overlay
- Language's prediction blend adds correction => response IS the correction
- Observability's identity => zero response, zero weight (correct!)

PAC boundary enforcement still happens at the outer boundary after merge.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import torch

import sys, os
# Add src and fracton to path
_here = os.path.dirname(os.path.abspath(__file__))
_gaia_root = os.path.join(_here, "..", "..")
sys.path.insert(0, os.path.join(_gaia_root, "src"))
sys.path.insert(0, os.path.join(_here, "..", "..", "..", "..", "..", "fracton"))

from gaia.core.exceptions import ConservationViolation, ModuleRegistrationError
from gaia.core.protocol import GAIAModule
from gaia.core.sec_router import SECRouter
from gaia.core.types import ConservationResult, FieldState, SECPhase

try:
    from fracton.field.resonance import harmonic_resonance, phase_coherence
except ImportError:
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
class ResponseWeight:
    """Per-module response weight from one dispatch cycle."""
    module_name: str
    response_magnitude: float    # ||response||
    phase_coherence: float       # alignment of response with input
    raw_weight: float            # response_magnitude * phase_coherence
    normalized_weight: float


class ResponseFieldBus:
    """Spike A: Response-field resonance bus.

    Key difference from ResonanceBus v1:
    - Modules still produce full outputs (same interface, no module changes)
    - The bus extracts the RESPONSE (output - input) from each module
    - Responses are weighted by delta-coherence
    - Merged = input + sum(w_i * response_i) * response_scale
    - response_scale controls how much total module influence is applied
    - PAC at outer boundary

    The response_scale parameter is critical: it controls the "gain" of
    module responses. At 1.0, the full weighted response is applied.
    Lower values make the bus more conservative.
    """

    def __init__(
        self,
        enforcement: str = "hard",
        tolerance: float = 1e-6,
        rbf_lambda: float = 1.0,
        rbf_alpha: float = 0.1,
        rbf_suppression_threshold: float = 0.0,
        min_weight_epsilon: float = 1e-6,
        response_scale: float = 1.0,
    ) -> None:
        if enforcement not in ("hard", "soft", "monitor"):
            raise ValueError(f"enforcement must be 'hard', 'soft', or 'monitor', got '{enforcement}'")

        self._enforcement = enforcement
        self._tolerance = tolerance
        self._rbf_lambda = rbf_lambda
        self._rbf_alpha = rbf_alpha
        self._rbf_suppression_threshold = rbf_suppression_threshold
        self._min_weight_epsilon = min_weight_epsilon
        self._response_scale = response_scale
        self._router = SECRouter()
        self._modules: dict[str, GAIAModule] = {}
        self._violation_log: list[ConservationResult] = []
        self._response_log: list[list[ResponseWeight]] = []

    def register_module(self, module: GAIAModule, phases: list[SECPhase] | None = None) -> None:
        if not isinstance(module, GAIAModule):
            raise ModuleRegistrationError(f"Not a GAIAModule: {type(module).__name__}")
        self._modules[module.name] = module
        self._router.register(module, phases)

    def process(self, field_state: FieldState) -> FieldState:
        """Response-field dispatch: broadcast -> extract responses -> weight -> add -> PAC."""

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

        # 2. BROADCAST — each module sees same input
        responses: list[tuple[GAIAModule, torch.Tensor]] = []
        for module in modules:
            module_input = input_state.clone()
            output = module.process(module_input)
            # Extract RESPONSE = output - input
            response = output.tensor - input_state.tensor
            responses.append((module, response))

        # 3. WEIGHT — delta-coherence on responses (response IS the delta)
        weights = self._compute_weights(input_state, responses)
        self._response_log.append(weights)

        # 4. SUPERPOSE — input + weighted sum of responses
        response_sum = torch.zeros_like(input_state.tensor)
        for (module, response), w in zip(responses, weights):
            response_sum = response_sum + w.normalized_weight * response

        merged_tensor = input_state.tensor + self._response_scale * response_sum

        # 5. PAC BOUNDARY — scale to match input energy
        input_energy = input_state.total_energy()
        merged_energy = float(torch.sum(merged_tensor).item())

        if abs(merged_energy) > 1e-10:
            merged_tensor = merged_tensor * (input_energy / merged_energy)
        elif abs(input_energy) > 1e-10:
            merged_tensor = input_state.tensor.clone()

        # Validate
        final_energy = float(torch.sum(merged_tensor).item())
        result = ConservationResult(
            conserved=abs(final_energy - input_energy) < self._tolerance * max(abs(input_energy), 1e-10),
            input_energy=input_energy,
            output_energy=final_energy,
            residual=abs(final_energy - input_energy),
            module_name="response_merge",
        )
        if not result.conserved:
            self._violation_log.append(result)
            self._handle_violation(result)

        # 6. Recompute entropy/phase, provenance
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
        responses: list[tuple[GAIAModule, torch.Tensor]],
    ) -> list[ResponseWeight]:
        """Weight responses by magnitude * phase-coherence with input."""
        weights: list[ResponseWeight] = []

        for module, response in responses:
            mag = float(torch.norm(response).item())

            if mag < self._min_weight_epsilon:
                # Zero response (identity module) — minimal weight
                coh = 1.0
                raw = self._min_weight_epsilon
            else:
                coh = phase_coherence(response, input_state.tensor)
                raw = mag * coh

            weights.append(ResponseWeight(
                module_name=module.name,
                response_magnitude=mag,
                phase_coherence=coh,
                raw_weight=raw,
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
    def response_log(self): return self._response_log
    @property
    def enforcement(self): return self._enforcement
    def get_metrics(self):
        return {
            "modules_registered": len(self._modules),
            "module_names": list(self._modules.keys()),
            "enforcement": self._enforcement,
            "tolerance": self._tolerance,
            "total_violations": len(self._violation_log),
            "response_ticks": len(self._response_log),
            "dispatch_mode": "response_field",
            "response_scale": self._response_scale,
        }
