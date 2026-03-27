"""ResonanceBus — QSocket-inspired field coupling for post-symbolic modules.

Replaces sequential dispatch with broadcast -> resonance-gated parallel
processing -> superposition merge. Modules see the same input simultaneously
and their outputs are weighted by delta-coherence: how much they changed
the field, modulated by structural alignment with the input.

Uses Fracton field primitives for resonance computation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from .exceptions import ConservationViolation, ModuleRegistrationError
from .protocol import GAIAModule
from .sec_router import SECRouter
from .types import ConservationResult, FieldState, SECPhase

try:
    from fracton.field.resonance import harmonic_resonance, phase_coherence
except ImportError:
    # Fallback if Fracton not on PYTHONPATH — pure-torch implementations
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
                ca = (ca[::2] + ca[1::2]) / 2
                cb = (cb[::2] + cb[1::2]) / 2
            else:
                break
        return total / weight_sum if weight_sum > 0 else 0.0

logger = logging.getLogger(__name__)


@dataclass
class ResonanceWeight:
    """Per-module resonance weight from one dispatch cycle."""

    module_name: str
    delta_magnitude: float
    phase_coherence: float
    raw_weight: float
    normalized_weight: float


class ResonanceBus:
    """QSocket-inspired resonance dispatch for GAIA v2.

    Broadcast -> resonance gate -> superposition merge.

    Each module receives the SAME input field. Their outputs are weighted
    by delta-coherence: large, structurally-aligned changes get high weight.
    Merged output is PAC-scaled to match input energy.

    Same interface as ConservationBus — drop-in replacement.
    """

    def __init__(
        self,
        enforcement: str = "hard",
        tolerance: float = 1e-6,
        rbf_lambda: float = 1.0,
        rbf_alpha: float = 0.1,
        rbf_suppression_threshold: float = 0.0,
        min_weight_epsilon: float = 1e-6,
    ) -> None:
        if enforcement not in ("hard", "soft", "monitor"):
            raise ValueError(f"enforcement must be 'hard', 'soft', or 'monitor', got '{enforcement}'")

        self._enforcement = enforcement
        self._tolerance = tolerance
        self._rbf_lambda = rbf_lambda
        self._rbf_alpha = rbf_alpha
        self._rbf_suppression_threshold = rbf_suppression_threshold
        self._min_weight_epsilon = min_weight_epsilon
        self._router = SECRouter()
        self._modules: dict[str, GAIAModule] = {}
        self._violation_log: list[ConservationResult] = []
        self._resonance_log: list[list[ResonanceWeight]] = []

    def register_module(
        self,
        module: GAIAModule,
        phases: list[SECPhase] | None = None,
    ) -> None:
        """Register a module with the bus."""
        if not isinstance(module, GAIAModule):
            raise ModuleRegistrationError(
                f"Object does not satisfy GAIAModule protocol: {type(module).__name__}"
            )
        self._modules[module.name] = module
        self._router.register(module, phases)

    def process(self, field_state: FieldState) -> FieldState:
        """Resonance dispatch: broadcast -> weight -> superpose -> PAC.

        1. Classify SEC phase
        2. Route to modules by phase
        3. Regulate by RBF health
        4. BROADCAST: each module gets clone of same input
        5. RESONANCE GATE: delta-coherence weights
        6. SUPERPOSE: weighted sum of output tensors
        7. PAC BOUNDARY: scale to match input energy
        8. Recompute entropy/phase, build provenance
        """
        # 1. Classify
        input_state = field_state.clone()
        phase = self._router.classify(input_state.entropy)
        input_state.phase = phase

        # 2. Route
        modules = self._router.route(input_state)
        if not modules:
            return input_state

        # 3. Regulate
        modules = self._regulate(modules)
        if not modules:
            return input_state

        # 4. BROADCAST — each module sees the same input
        outputs: list[tuple[GAIAModule, FieldState]] = []
        for module in modules:
            module_input = input_state.clone()
            output = module.process(module_input)
            outputs.append((module, output))

        # 5. RESONANCE GATE — delta-coherence weights
        weights = self._compute_weights(input_state, outputs)
        self._resonance_log.append(weights)

        # 6. SUPERPOSE — weighted sum of output tensors
        merged_tensor = torch.zeros_like(input_state.tensor)
        for (module, output), w in zip(outputs, weights):
            merged_tensor = merged_tensor + w.normalized_weight * output.tensor

        # 7. PAC BOUNDARY — scale to match input energy
        input_energy = input_state.total_energy()
        merged_energy = float(torch.sum(merged_tensor).item())

        pac_scale = 1.0
        if abs(merged_energy) > 1e-10:
            pac_scale = input_energy / merged_energy
            merged_tensor = merged_tensor * pac_scale
        elif abs(input_energy) > 1e-10:
            # Merged to zero but input wasn't — restore input
            merged_tensor = input_state.tensor.clone()

        # Validate PAC at outer boundary
        final_energy = float(torch.sum(merged_tensor).item())
        result = ConservationResult(
            conserved=abs(final_energy - input_energy) < self._tolerance * max(abs(input_energy), 1e-10),
            input_energy=input_energy,
            output_energy=final_energy,
            residual=abs(final_energy - input_energy),
            module_name="resonance_merge",
        )
        if not result.conserved:
            self._violation_log.append(result)
            self._handle_violation(result)

        # 8. Recompute entropy/phase, build provenance
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
    ) -> list[ResonanceWeight]:
        """Compute delta-coherence weights for each module output."""
        weights: list[ResonanceWeight] = []

        for module, output in outputs:
            delta = output.tensor - input_state.tensor
            mag = float(torch.norm(delta).item())

            if mag < self._min_weight_epsilon:
                # Identity-like module — minimal contribution
                coh = 1.0
                raw = self._min_weight_epsilon
            else:
                coh = phase_coherence(delta, input_state.tensor)
                raw = mag * coh

            weights.append(ResonanceWeight(
                module_name=module.name,
                delta_magnitude=mag,
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
        """Compute Shannon entropy of tensor's value distribution."""
        flat = tensor.flatten().float().abs()
        total = flat.sum()
        if total < 1e-12 or flat.numel() == 0:
            return 0.0
        probs = flat / total
        mask = probs > 1e-12
        probs = probs[mask]
        return float(-(probs * probs.log()).sum().item())

    def _regulate(self, modules: list[GAIAModule]) -> list[GAIAModule]:
        """Filter modules by RBF health."""
        healthy = []
        for module in modules:
            rbf = module.health()
            if rbf.balance >= self._rbf_suppression_threshold:
                healthy.append(module)
            else:
                logger.info(
                    "Module '%s' suppressed: RBF balance %.4f < threshold %.4f",
                    module.name, rbf.balance, self._rbf_suppression_threshold,
                )
        return healthy

    def _handle_violation(self, result: ConservationResult) -> None:
        """Handle a conservation violation based on enforcement mode."""
        msg = (
            f"PAC violation in '{result.module_name}': "
            f"residual={result.residual:.6e} "
            f"(in={result.input_energy:.6f}, out={result.output_energy:.6f})"
        )
        if self._enforcement == "hard":
            raise ConservationViolation(result, result.module_name)
        elif self._enforcement == "soft":
            logger.warning("SOFT: %s", msg)
        else:
            logger.info("MONITOR: %s", msg)

    @property
    def violation_log(self) -> list[ConservationResult]:
        """All conservation violations recorded during processing."""
        return list(self._violation_log)

    @property
    def resonance_log(self) -> list[list[ResonanceWeight]]:
        """Per-tick resonance weight history."""
        return self._resonance_log

    @property
    def enforcement(self) -> str:
        """Current enforcement mode."""
        return self._enforcement

    def get_metrics(self) -> dict:
        """Bus operational metrics."""
        return {
            "modules_registered": len(self._modules),
            "module_names": list(self._modules.keys()),
            "enforcement": self._enforcement,
            "tolerance": self._tolerance,
            "total_violations": len(self._violation_log),
            "resonance_ticks": len(self._resonance_log),
            "dispatch_mode": "resonance",
        }
