"""Spike F — Coupled Fields Bus.

Insight (Peter, 2026-03-23): "What if it's like WiFi — we need different
channels or frequency domains?"

The problem with Spikes C-E: broadcast+superpose homogenizes the signal.
All modules receive the same merged output, so Memory's PAC tree can't
differentiate stimulus classes. Retrieval resonance = 1.000 for all classes,
zero preference signal. The superposition acts as a low-pass filter,
averaging away class-specific features.

The fix: each module is a standing wave in its own frequency domain.
The bus COUPLES these fields (like tuning forks on a table) rather than
MERGING them into one tone. Each module's accumulated resonance state
acts as a spectral lens — filtering the raw input to emphasize dimensions
that module has historically found important.

Architecture:
    1. Per-module INPUT LENS — resonance state normalized to unit mean,
       multiplicatively filters the raw input. Memory sees input through
       Memory's lens, Reasoning through Reasoning's lens. Same stimulus,
       different perspectives (Peter's dad insight from Spike C).

    2. Inter-module COUPLING — coupled oscillator dynamics. Modules
       influence each other's state evolution via a coupling matrix:
       C[i,j] = harmonic_resonance(state_i, state_j). The coupling force
       is proportional to displacement: (state_j - state_i). Similar
       modules synchronize, dissimilar ones stay independent.

    3. QBE-regulated COUPLING STRENGTH — QPL oscillation modulates the
       global coupling constant. QPL > 0 → stronger coupling (exploit,
       synchronize). QPL < 0 → weaker coupling (explore, diverge).
       The system breathes between coherent action and independent
       exploration.

    4. SPECTRUM OUTPUT — each module's output was produced from a
       different lensed input, so the superposition retains genuinely
       different information (unlike Spikes C-E where all modules saw
       the same input and the merge was lossy).

Key metric: lens_contrast = std(lens). High contrast = specialized filter
(module has developed preferences). Low contrast = flat filter (no bias).
If coupling works, Memory's lens should diverge from Reasoning's lens
over time, enabling internal preference formation.

Physics grounding:
    - Each module = standing wave in its own frequency mode
    - Bus = coupling medium (the string/table)
    - QBE regulates inter-mode energy transfer
    - Lens = spectral filter (like a resonant cavity)
    - PAC conservation at the boundary (total energy across all modes)
    - Coupling force ∝ displacement → symmetric → energy-conserving
"""

from __future__ import annotations

import logging
import math
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
from gaia.core.types import ConservationResult, FieldState, RBFBalance, SECPhase

# DFT constants
PHI = 1.618033988749895
PHI_INV = 1.0 / PHI  # 0.618...

# QBE constants
QBE_OMEGA = 0.020       # Universal frequency (Hz)
QBE_LAMBDA = 1.0        # QBE coupling constant
COUPLING_KAPPA = 0.05   # Base coupling strength (small — coupling is a perturbation)

# Lens safety bounds
LENS_CLAMP_MIN = 0.1
LENS_CLAMP_MAX = 10.0


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
class CoupledFieldState:
    """Per-module field in the coupled system."""
    tensor: torch.Tensor       # Running resonance state (the module's persistent field)
    lens: torch.Tensor         # Normalized filter derived from state
    prediction_error: float
    adaptation_rate: float
    ticks_alive: int
    surprise_history: list[float] = field(default_factory=list)


@dataclass
class CoupledWeight:
    """Per-module weight from one tick of coupled-field dispatch."""
    module_name: str
    prediction_accuracy: float    # harmonic_resonance(state, lensed_input)
    perspective_resonance: float  # harmonic_resonance(output, lensed_input)
    combined_weight: float        # prediction_accuracy * perspective_resonance
    normalized_weight: float
    # Coupling diagnostics
    coupling_received: float      # total coupling force magnitude received
    lens_contrast: float          # std(lens) — how specialized is this module's filter?


class CoupledFieldsBus:
    """Spike F: Coupled standing-wave field bus.

    Each module maintains its own frequency domain (resonance state).
    The state acts as a spectral lens that filters the raw input,
    giving each module a unique perspective. Modules are coupled
    like oscillators on a shared medium — they influence each other's
    evolution but maintain distinct identities.

    QBE regulates coupling strength (not individual weights):
        coupling = kappa * (1 + lambda * cos(omega * tick))
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
        coupling_kappa: float = COUPLING_KAPPA,
        qbe_omega: float = QBE_OMEGA,
        qbe_lambda: float = QBE_LAMBDA,
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
        self._blend_rate = blend_rate
        self._surprise_window = surprise_window
        self._coupling_kappa = coupling_kappa
        self._qbe_omega = qbe_omega
        self._qbe_lambda = qbe_lambda
        self._router = SECRouter()
        self._modules: dict[str, GAIAModule] = {}
        self._field_states: dict[str, CoupledFieldState] = {}
        self._violation_log: list[ConservationResult] = []
        self._coupled_log: list[list[CoupledWeight]] = []
        self._tick: int = 0

    def register_module(self, module: GAIAModule, phases: list[SECPhase] | None = None) -> None:
        if not isinstance(module, GAIAModule):
            raise ModuleRegistrationError(f"Not a GAIAModule: {type(module).__name__}")
        self._modules[module.name] = module
        self._router.register(module, phases)

    def _compute_lens(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Compute spectral lens from resonance state.

        Normalize to unit mean so the filter is energy-neutral on average.
        Clamp to [LENS_CLAMP_MIN, LENS_CLAMP_MAX] to prevent dimension collapse.
        """
        abs_state = state_tensor.abs()
        mean_abs = abs_state.mean()
        if mean_abs < 1e-10:
            return torch.ones_like(state_tensor)
        lens = state_tensor / mean_abs
        return lens.clamp(LENS_CLAMP_MIN, LENS_CLAMP_MAX)

    def _compute_coupling_strength(self) -> float:
        """QBE-regulated coupling strength.

        coupling = kappa * (1 + lambda * cos(omega * tick))
        QPL > 0 → stronger coupling (synchronize/exploit)
        QPL < 0 → weaker coupling (diverge/explore)
        """
        qpl = math.cos(self._qbe_omega * self._tick)
        return self._coupling_kappa * (1.0 + self._qbe_lambda * qpl)

    def process(self, field_state: FieldState) -> FieldState:
        """Coupled-field dispatch.

        1. Classify + route (SEC)
        2. Init field states for new modules (lens = ones)
        3. Compute lenses from resonance states
        4. Lens + broadcast: module_input_i = input * lens_i
        5. Process: output_i = module_i.process(lensed_input_i)
        6. Weight: prediction_accuracy * perspective_resonance
        7. Superpose weighted outputs
        8. Compute coupling matrix
        9. Update states (Mobius + coupling force)
        10. PAC boundary
        """
        # 1. CLASSIFY + ROUTE
        input_state = field_state.clone()
        phase = self._router.classify(input_state.entropy)
        input_state.phase = phase
        modules = self._router.route(input_state)
        if not modules:
            self._tick += 1
            return input_state
        modules = self._regulate(modules)
        if not modules:
            self._tick += 1
            return input_state

        # 2. INIT FIELD STATES for new modules
        for module in modules:
            if module.name not in self._field_states:
                ones_lens = torch.ones_like(input_state.tensor)
                self._field_states[module.name] = CoupledFieldState(
                    tensor=input_state.tensor.clone(),
                    lens=ones_lens,
                    prediction_error=0.0,
                    adaptation_rate=1.0,
                    ticks_alive=0,
                )

        # 3. COMPUTE LENSES from current resonance states
        for module in modules:
            fs = self._field_states[module.name]
            fs.lens = self._compute_lens(fs.tensor)

        # 4. LENS + BROADCAST — each module gets input through its own filter
        lensed_inputs: dict[str, FieldState] = {}
        for module in modules:
            fs = self._field_states[module.name]
            lensed_tensor = input_state.tensor * fs.lens

            # Re-scale lensed input to preserve energy (lens is energy-neutral on average
            # but not exactly — enforce it here)
            input_energy = input_state.total_energy()
            lensed_energy = float(torch.sum(lensed_tensor).item())
            if abs(lensed_energy) > 1e-10:
                lensed_tensor = lensed_tensor * (input_energy / lensed_energy)

            lensed_state = FieldState(
                tensor=lensed_tensor,
                entropy=self._shannon_entropy(lensed_tensor),
                phase=input_state.phase,
                conservation_budget=input_state.conservation_budget,
                provenance=list(input_state.provenance),
                timestamp=input_state.timestamp,
            )
            lensed_inputs[module.name] = lensed_state

        # 5. PROCESS — each module transforms its own lensed input
        outputs: list[tuple[GAIAModule, FieldState]] = []
        for module in modules:
            output = module.process(lensed_inputs[module.name])
            outputs.append((module, output))

        # 6. WEIGHT — prediction accuracy * perspective resonance
        weights = self._compute_weights(input_state, outputs, lensed_inputs)
        self._coupled_log.append(weights)

        # 7. SUPERPOSE — weighted sum of channel outputs
        merged_tensor = torch.zeros_like(input_state.tensor)
        for (module, output), w in zip(outputs, weights):
            merged_tensor = merged_tensor + w.normalized_weight * output.tensor

        # 8. COMPUTE COUPLING MATRIX
        coupling_strength = self._compute_coupling_strength()
        module_names = [m.name for m in modules]
        coupling_matrix = self._compute_coupling_matrix(module_names)

        # 9. UPDATE STATES (Mobius iteration + coupling force)
        coupling_forces: dict[str, float] = {}
        for i, module in enumerate(modules):
            state = self._field_states[module.name]
            output_tensor = outputs[i][1].tensor

            # Mobius iteration (same as Spike D)
            new_state = (
                self._blend_rate * state.tensor +
                (1.0 - self._blend_rate) * output_tensor
            )

            # Coupling force: sum_j(C[i,j] * (state_j - state_i))
            coupling_force = torch.zeros_like(state.tensor)
            for j, other_module in enumerate(modules):
                if i == j:
                    continue
                c_ij = coupling_matrix.get((module.name, other_module.name), 0.0)
                other_state = self._field_states[other_module.name].tensor
                coupling_force = coupling_force + c_ij * (other_state - state.tensor)

            coupling_force_mag = float(torch.norm(coupling_force).item())
            coupling_forces[module.name] = coupling_force_mag

            # Apply coupling (scaled by QBE-regulated strength)
            new_state = new_state + coupling_strength * coupling_force

            state.tensor = new_state
            state.ticks_alive += 1

            # Track surprise and adaptation
            surprise = float(torch.norm(
                lensed_inputs[module.name].tensor - state.tensor
            ).item())
            state.surprise_history.append(surprise)
            if len(state.surprise_history) > self._surprise_window:
                state.surprise_history.pop(0)

            if state.ticks_alive > 1:
                delta_norm = float(torch.norm(output_tensor - state.tensor).item())
                state.adaptation_rate = 0.9 * state.adaptation_rate + 0.1 * delta_norm

        # Update coupling_received in weight objects
        for w in weights:
            w.coupling_received = coupling_forces.get(w.module_name, 0.0)

        # 10. PAC BOUNDARY
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
            module_name="coupled_merge",
        )
        if not result.conserved:
            self._violation_log.append(result)
            self._handle_violation(result)

        entropy = self._shannon_entropy(merged_tensor)
        new_phase = self._router.classify(entropy)
        provenance = [w.module_name for w in weights if w.normalized_weight > self._min_weight_epsilon]

        self._tick += 1

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
        lensed_inputs: dict[str, FieldState],
    ) -> list[CoupledWeight]:
        """Weight = prediction_accuracy * perspective_resonance.

        prediction_accuracy: harmonic_resonance(resonance_state, lensed_input)
            "Did your accumulated perspective predict what you'd see through your lens?"

        perspective_resonance: harmonic_resonance(output, lensed_input)
            "How deeply do you see the field through your lens?"
        """
        weights: list[CoupledWeight] = []

        for module, output in outputs:
            state = self._field_states[module.name]
            lensed = lensed_inputs[module.name]

            # Prediction accuracy — did this module's state predict its lensed input?
            prediction = harmonic_resonance(
                state.tensor, lensed.tensor, harmonics=self._harmonics
            )

            # Perspective resonance — does the output resonate with the lensed view?
            perspective = harmonic_resonance(
                output.tensor, lensed.tensor, harmonics=self._harmonics
            )

            # Lens contrast — how specialized is this module's filter?
            lens_std = float(state.lens.std().item())

            pred = max(0.0, prediction)
            persp = max(0.0, perspective)
            combined = pred * persp

            weights.append(CoupledWeight(
                module_name=module.name,
                prediction_accuracy=prediction,
                perspective_resonance=perspective,
                combined_weight=combined,
                normalized_weight=0.0,
                coupling_received=0.0,  # filled in after coupling computation
                lens_contrast=lens_std,
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

    def _compute_coupling_matrix(
        self, module_names: list[str]
    ) -> dict[tuple[str, str], float]:
        """Compute coupling between all module pairs.

        C[i,j] = harmonic_resonance(state_i, state_j)
        Symmetric: C[i,j] = C[j,i] (energy-conserving).
        """
        matrix: dict[tuple[str, str], float] = {}
        for i, name_i in enumerate(module_names):
            state_i = self._field_states[name_i].tensor
            for j, name_j in enumerate(module_names):
                if i == j:
                    continue
                if (name_j, name_i) in matrix:
                    # Symmetric
                    matrix[(name_i, name_j)] = matrix[(name_j, name_i)]
                else:
                    c = harmonic_resonance(
                        state_i, self._field_states[name_j].tensor,
                        harmonics=self._harmonics
                    )
                    matrix[(name_i, name_j)] = c
        return matrix

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
    def coupled_log(self): return self._coupled_log
    @property
    def field_states(self): return dict(self._field_states)
    @property
    def enforcement(self): return self._enforcement

    def get_metrics(self):
        state_info = {}
        for name, fs in self._field_states.items():
            mean_surprise = (
                sum(fs.surprise_history) / len(fs.surprise_history)
                if fs.surprise_history else 0.0
            )
            state_info[name] = {
                "ticks_alive": fs.ticks_alive,
                "prediction_error": fs.prediction_error,
                "adaptation_rate": fs.adaptation_rate,
                "mean_surprise": mean_surprise,
                "lens_contrast": float(fs.lens.std().item()),
            }

        coupling_info = {}
        coupling_strength = self._compute_coupling_strength()
        coupling_info["strength"] = coupling_strength
        coupling_info["qpl"] = math.cos(self._qbe_omega * self._tick)

        return {
            "modules_registered": len(self._modules),
            "module_names": list(self._modules.keys()),
            "enforcement": self._enforcement,
            "tolerance": self._tolerance,
            "total_violations": len(self._violation_log),
            "coupled_ticks": len(self._coupled_log),
            "dispatch_mode": "coupled_fields",
            "blend_rate": self._blend_rate,
            "coupling_kappa": self._coupling_kappa,
            "current_tick": self._tick,
            "resonance_states": state_info,
            "coupling": coupling_info,
        }
