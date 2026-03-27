"""Spike D — Continuous-Flow Mobius Field Bus.

Insight (Peter, 2026-03-22): "It's a websocket, not REST."

The brain doesn't get requests and responses. It's always on. Modules
don't wake up, process, and return — they're standing waves in the field.
The prediction/correction loop isn't a separate step; it IS the field
evolving. One continuous process, not an iteration of many processes.
The Mobius in action.

Architecture:
    - The field persists between ticks (not discarded after each process)
    - Each module maintains a RESONANCE STATE: its running perspective
      on the field, shaped by everything it's ever seen
    - On each tick, the field flows through all modules simultaneously
    - Each module's output feeds back into its own resonance state
    - The bus merges perspectives using harmonic resonance of each
      module's resonance state with the current field
    - Prediction error (delta between last output and new input) is
      itself a signal — the "surprise" that drives adaptation

Key differences from all previous spikes:
    1. Field persists — output of tick N becomes context for tick N+1
    2. Module resonance state — each module accumulates a running
       average of its outputs, weighted by phi-decay (like Mobius
       fixed points encoding memory)
    3. Prediction error drives weight — modules whose resonance state
       PREDICTED the current input well get more weight (they "saw it
       coming"). This is the continuous feedback loop.
    4. No discrete predict/compare/update — the resonance state
       evolves continuously as a Mobius transform of itself

The Mobius connection:
    M(z) = (az+b)/(cz+d)
    Fixed points solve M(z) = z
    The resonance state IS the fixed point — it's where the module
    settles given its accumulated field history. When new input arrives,
    it perturbs the fixed point. How much it moves = how surprised
    the module is. How quickly it re-stabilizes = how well it adapts.

    We don't need literal Mobius neurons here — the concept is:
    resonance_state(t+1) = blend(resonance_state(t), module_output)
    This IS M(z) where the parameters (a,b,c,d) are implicit in
    the blend rate and module transform.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
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

# Golden ratio — the natural blend rate
PHI = 1.618033988749895
PHI_INV = 1.0 / PHI  # 0.618... — the blend coefficient


def harmonic_resonance(a: torch.Tensor, b: torch.Tensor, harmonics: int = 3) -> float:
    """Multi-scale cosine similarity with golden-ratio weighting."""
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
class ModuleResonanceState:
    """Running resonance state for one module — its accumulated perspective.

    This is the module's "fixed point" — where it settles given its history.
    Updated each tick via phi-blended exponential moving average.
    """
    tensor: torch.Tensor          # Running average of module outputs
    prediction_error: float       # How well this state predicted current input
    adaptation_rate: float        # How quickly the state is currently shifting
    ticks_alive: int              # How long this state has been evolving
    surprise_history: list[float] = field(default_factory=list)


@dataclass
class ContinuousWeight:
    """Per-module weight from one tick of continuous flow."""
    module_name: str
    prediction_accuracy: float    # harmonic_resonance(resonance_state, input)
    output_resonance: float       # harmonic_resonance(output, input) (perspective)
    surprise: float               # ||input - resonance_state||
    combined_weight: float        # prediction_accuracy * output_resonance
    normalized_weight: float


class ContinuousFieldBus:
    """Spike D: Continuous-flow Mobius field bus.

    The field persists. Modules accumulate resonance states.
    Prediction accuracy drives weight. Output feeds back as
    the next tick's context. One continuous stream.

    Weight = prediction_accuracy * perspective_resonance

    prediction_accuracy: harmonic_resonance(module_resonance_state, input)
        = "how well did your accumulated perspective predict THIS input?"
        = modules that have been tracking the field's trajectory get more weight

    perspective_resonance: harmonic_resonance(module_output, input)
        = "how deeply do you see this field?" (Spike C insight)
        = modules whose perspective aligns with reality get credibility

    The product means: you need BOTH to have tracked the field well
    AND to currently resonate with it. A module that predicted well
    but now sees something totally different (high surprise) might
    be detecting novelty — that's captured separately in the surprise
    metric.

    Resonance state update (the Mobius iteration):
        state(t+1) = phi_inv * state(t) + (1 - phi_inv) * output(t)

    phi_inv = 0.618... means recent outputs dominate, but there's always
    a 61.8% memory of accumulated history. This IS the Mobius transform
    with implicit parameters — the fixed point IS the resonance state.
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
        blend_rate: float = PHI_INV,
        surprise_window: int = 10,
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
        self._blend_rate = blend_rate  # phi_inv by default
        self._surprise_window = surprise_window
        self._router = SECRouter()
        self._modules: dict[str, GAIAModule] = {}
        self._resonance_states: dict[str, ModuleResonanceState] = {}
        self._violation_log: list[ConservationResult] = []
        self._continuous_log: list[list[ContinuousWeight]] = []

        # The persistent field — carries forward between ticks
        self._last_output: Optional[torch.Tensor] = None

    def register_module(self, module: GAIAModule, phases: list[SECPhase] | None = None) -> None:
        if not isinstance(module, GAIAModule):
            raise ModuleRegistrationError(f"Not a GAIAModule: {type(module).__name__}")
        self._modules[module.name] = module
        self._router.register(module, phases)

    def process(self, field_state: FieldState) -> FieldState:
        """Continuous-flow dispatch.

        1. Measure prediction error (how well did resonance states predict this?)
        2. Broadcast to modules
        3. Weight by prediction_accuracy * perspective_resonance
        4. Superpose
        5. Update resonance states (Mobius iteration)
        6. PAC boundary
        """

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

        # Initialize resonance states for new modules
        for module in modules:
            if module.name not in self._resonance_states:
                self._resonance_states[module.name] = ModuleResonanceState(
                    tensor=input_state.tensor.clone(),
                    prediction_error=0.0,
                    adaptation_rate=1.0,
                    ticks_alive=0,
                )

        # 2. BROADCAST — each module sees same input
        outputs: list[tuple[GAIAModule, FieldState]] = []
        for module in modules:
            module_input = input_state.clone()
            output = module.process(module_input)
            outputs.append((module, output))

        # 3. WEIGHT — prediction accuracy * perspective resonance
        weights = self._compute_weights(input_state, outputs)
        self._continuous_log.append(weights)

        # 4. SUPERPOSE — weighted sum of perspectives
        merged_tensor = torch.zeros_like(input_state.tensor)
        for (module, output), w in zip(outputs, weights):
            merged_tensor = merged_tensor + w.normalized_weight * output.tensor

        # 5. UPDATE RESONANCE STATES (the Mobius iteration)
        for (module, output), w in zip(outputs, weights):
            state = self._resonance_states[module.name]
            # phi-blended exponential moving average
            # state = phi_inv * old_state + (1 - phi_inv) * new_output
            state.tensor = (
                self._blend_rate * state.tensor +
                (1.0 - self._blend_rate) * output.tensor
            )
            state.ticks_alive += 1

            # Track surprise (prediction error magnitude)
            surprise = float(torch.norm(input_state.tensor - state.tensor).item())
            state.surprise_history.append(surprise)
            if len(state.surprise_history) > self._surprise_window:
                state.surprise_history.pop(0)

            # Adaptation rate: how much is the state currently moving?
            if state.ticks_alive > 1:
                delta_norm = float(torch.norm(
                    output.tensor - state.tensor
                ).item())
                state.adaptation_rate = (
                    0.9 * state.adaptation_rate + 0.1 * delta_norm
                )

        # 6. PAC BOUNDARY
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
            module_name="continuous_merge",
        )
        if not result.conserved:
            self._violation_log.append(result)
            self._handle_violation(result)

        # Recompute entropy/phase
        entropy = self._shannon_entropy(merged_tensor)
        new_phase = self._router.classify(entropy)
        provenance = [w.module_name for w in weights if w.normalized_weight > self._min_weight_epsilon]

        # Remember this output for next tick's context
        self._last_output = merged_tensor.clone().detach()

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
    ) -> list[ContinuousWeight]:
        """Weight = prediction_accuracy * perspective_resonance.

        prediction_accuracy: how well the module's resonance state
            (accumulated perspective) predicted the current input.
        perspective_resonance: how deeply the module's output
            resonates with the current input (from Spike C).
        """
        weights: list[ContinuousWeight] = []

        for module, output in outputs:
            state = self._resonance_states[module.name]

            # Prediction accuracy: did your resonance state predict this input?
            prediction_accuracy = harmonic_resonance(
                state.tensor, input_state.tensor, harmonics=self._harmonics
            )

            # Perspective resonance: how does your output align with input?
            output_resonance = harmonic_resonance(
                output.tensor, input_state.tensor, harmonics=self._harmonics
            )

            # Surprise: how far is the input from your expectation?
            surprise = float(torch.norm(
                input_state.tensor - state.tensor
            ).item())

            # Update prediction error in state
            state.prediction_error = 1.0 - max(0.0, prediction_accuracy)

            # Combined weight: both prediction and perspective matter
            # Clamp negatives (anti-resonant = no contribution)
            pa = max(0.0, prediction_accuracy)
            pr = max(0.0, output_resonance)
            combined = pa * pr

            weights.append(ContinuousWeight(
                module_name=module.name,
                prediction_accuracy=prediction_accuracy,
                output_resonance=output_resonance,
                surprise=surprise,
                combined_weight=combined,
                normalized_weight=0.0,
            ))

        # Normalize
        total = sum(w.combined_weight for w in weights)
        if total > 1e-10:
            for w in weights:
                w.normalized_weight = w.combined_weight / total
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
    def continuous_log(self): return self._continuous_log
    @property
    def resonance_states(self): return dict(self._resonance_states)
    @property
    def enforcement(self): return self._enforcement
    def get_metrics(self):
        state_info = {}
        for name, state in self._resonance_states.items():
            mean_surprise = (
                sum(state.surprise_history) / len(state.surprise_history)
                if state.surprise_history else 0.0
            )
            state_info[name] = {
                "ticks_alive": state.ticks_alive,
                "prediction_error": state.prediction_error,
                "adaptation_rate": state.adaptation_rate,
                "mean_surprise": mean_surprise,
            }
        return {
            "modules_registered": len(self._modules),
            "module_names": list(self._modules.keys()),
            "enforcement": self._enforcement,
            "tolerance": self._tolerance,
            "total_violations": len(self._violation_log),
            "continuous_ticks": len(self._continuous_log),
            "dispatch_mode": "continuous",
            "blend_rate": self._blend_rate,
            "resonance_states": state_info,
        }
