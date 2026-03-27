"""Spike E — QBE-Regulated Continuous Field Bus.

Insight (Peter, 2026-03-22): "What if we use the quantum balance equation
for auto weight scaling? The system can decide how important things are."

The QBE constraint: dI/dt + dE/dt = λ*QPL(t)
Where QPL(t) = cos(ω*t), ω = 0.020 Hz (universal frequency from DFT)

Each module already reports health() -> RBFBalance with:
    B = λ(E-I)/(1+αM)
    Positive B = energy-dominant (overactive)
    Negative B = information-dominant (underactive)

The bus uses this to AUTO-SCALE weights:
    1. Compute system-wide E-I balance from all module health reports
    2. QBE provides the equilibrium TARGET via QPL(t) — the system
       doesn't aim for zero balance, it aims for the QPL oscillation
    3. Modules whose RBF balance would push the system TOWARD the
       QBE target get amplified; modules pushing AWAY get dampened
    4. The system naturally oscillates between energy and information
       phases — some ticks favor Safety (energy-dominant), others
       favor Memory (information-rich), driven by QPL's cosmic clock

Why this is different from all previous spikes:
    - Spikes A-C: weight = f(tensor similarity) — purely geometric
    - Spike D: weight = f(prediction accuracy) — purely temporal
    - Spike E: weight = f(tensor similarity) * f(E-I balance) — PHYSICS
      The system uses its own conservation law to decide importance.
      Not arbitrary, not learned — derived from the field dynamics.

The QBE coupling means the bus literally runs on the same equation
that governs E↔I equivalence in DFT. The brain's "attention" IS
the quantum balance equation deciding what matters right now.

Architecture:
    - Builds on Spike D (continuous flow, persistent resonance states)
    - Adds QBE regulation layer on top of perspective weights
    - QPL oscillation creates natural attentional rhythms
    - Module health() values drive the regulation — zero new parameters
      (all constants come from DFT: ω=0.020, λ=1.0, α=0.1)
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

# Golden ratio — the natural blend rate
PHI = 1.618033988749895
PHI_INV = 1.0 / PHI  # 0.618...

# QBE constants from DFT
QBE_OMEGA = 0.020      # Universal frequency (Hz) — from dawn-field-theory experiments
QBE_LAMBDA = 1.0       # QBE coupling constant (dimensionless)
QBE_KAPPA = 0.5        # Weight modulation strength (how much QBE affects weights)


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
    """Running resonance state for one module — its accumulated perspective."""
    tensor: torch.Tensor
    prediction_error: float
    adaptation_rate: float
    ticks_alive: int
    surprise_history: list[float] = field(default_factory=list)


@dataclass
class QBEWeight:
    """Per-module weight from one tick of QBE-regulated dispatch."""
    module_name: str
    perspective_resonance: float   # harmonic_resonance(output, input)
    rbf_balance: float             # module's RBF health balance
    qbe_scale: float               # QBE regulation multiplier
    combined_weight: float         # perspective * qbe_scale
    normalized_weight: float
    # Diagnostics
    system_balance: float          # system-wide E-I balance this tick
    qbe_target: float             # QPL(t) target this tick
    system_need: float            # qbe_target - system_balance


class QBEFieldBus:
    """Spike E: QBE-regulated continuous field bus.

    Combines Spike D's continuous flow with QBE-driven weight scaling.
    The system uses its own physics to decide module importance.

    Weight = perspective_resonance * qbe_scale

    perspective_resonance: harmonic_resonance(output, input)
        = how deeply does this module see the field? (from Spike C)

    qbe_scale: 1 + κ * alignment(module_balance, system_need)
        = does this module's E-I balance help the system reach QPL target?
        κ = QBE_KAPPA (modulation strength)
        system_need = QPL(t) - system_balance
        alignment = tanh(module_balance * system_need)
            positive when module balance and system need have same sign
            (energy-dominant module when system needs energy, etc.)

    The QPL oscillation (cos(0.020*t)) creates natural attentional rhythms:
    - When QPL > 0: system target is positive balance → favor energy-dominant modules
    - When QPL < 0: system target is negative → favor information-dominant modules
    - Modules with the RIGHT balance for the current phase get amplified
    - The system breathes between energy and information focus

    All constants from DFT — zero arbitrary parameters.
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
        qbe_omega: float = QBE_OMEGA,
        qbe_lambda: float = QBE_LAMBDA,
        qbe_kappa: float = QBE_KAPPA,
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
        self._qbe_omega = qbe_omega
        self._qbe_lambda = qbe_lambda
        self._qbe_kappa = qbe_kappa
        self._router = SECRouter()
        self._modules: dict[str, GAIAModule] = {}
        self._resonance_states: dict[str, ModuleResonanceState] = {}
        self._violation_log: list[ConservationResult] = []
        self._qbe_log: list[list[QBEWeight]] = []

        # Time counter for QPL oscillation
        self._tick: int = 0
        self._last_output: Optional[torch.Tensor] = None

    def register_module(self, module: GAIAModule, phases: list[SECPhase] | None = None) -> None:
        if not isinstance(module, GAIAModule):
            raise ModuleRegistrationError(f"Not a GAIAModule: {type(module).__name__}")
        self._modules[module.name] = module
        self._router.register(module, phases)

    def _compute_qpl(self) -> float:
        """QPL(t) = cos(ω*t) — the quantum potential layer at current tick."""
        return math.cos(self._qbe_omega * self._tick)

    def process(self, field_state: FieldState) -> FieldState:
        """QBE-regulated continuous-flow dispatch.

        1. Classify + route
        2. Broadcast to modules
        3. Query module health (RBF balance)
        4. Compute QBE-regulated weights
        5. Superpose
        6. Update resonance states
        7. PAC boundary
        """

        # 1. Classify + route
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

        # 3. QUERY MODULE HEALTH — RBF balance for each
        module_balances: dict[str, RBFBalance] = {}
        for module, _ in outputs:
            module_balances[module.name] = module.health()

        # 4. QBE-REGULATED WEIGHTS
        weights = self._compute_weights(input_state, outputs, module_balances)
        self._qbe_log.append(weights)

        # 5. SUPERPOSE — weighted sum of perspectives
        merged_tensor = torch.zeros_like(input_state.tensor)
        for (module, output), w in zip(outputs, weights):
            merged_tensor = merged_tensor + w.normalized_weight * output.tensor

        # 6. UPDATE RESONANCE STATES (Mobius iteration from Spike D)
        for (module, output), w in zip(outputs, weights):
            state = self._resonance_states[module.name]
            state.tensor = (
                self._blend_rate * state.tensor +
                (1.0 - self._blend_rate) * output.tensor
            )
            state.ticks_alive += 1

            surprise = float(torch.norm(input_state.tensor - state.tensor).item())
            state.surprise_history.append(surprise)
            if len(state.surprise_history) > self._surprise_window:
                state.surprise_history.pop(0)

            if state.ticks_alive > 1:
                delta_norm = float(torch.norm(output.tensor - state.tensor).item())
                state.adaptation_rate = 0.9 * state.adaptation_rate + 0.1 * delta_norm

        # 7. PAC BOUNDARY
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
            module_name="qbe_merge",
        )
        if not result.conserved:
            self._violation_log.append(result)
            self._handle_violation(result)

        entropy = self._shannon_entropy(merged_tensor)
        new_phase = self._router.classify(entropy)
        provenance = [w.module_name for w in weights if w.normalized_weight > self._min_weight_epsilon]

        self._last_output = merged_tensor.clone().detach()
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
        module_balances: dict[str, RBFBalance],
    ) -> list[QBEWeight]:
        """Weight = perspective_resonance * qbe_scale.

        perspective_resonance: harmonic_resonance(output, input)
        qbe_scale: 1 + κ * tanh(module_balance * system_need)

        system_need = QPL(t) - system_balance
        system_balance = mean of all module RBF balances
        """
        # System-wide E-I balance
        balances = [rb.balance for rb in module_balances.values()]
        system_balance = sum(balances) / len(balances) if balances else 0.0

        # QBE target from QPL oscillation
        qbe_target = self._qbe_lambda * self._compute_qpl()

        # What the system needs to reach equilibrium
        system_need = qbe_target - system_balance

        weights: list[QBEWeight] = []

        for module, output in outputs:
            # Perspective resonance (from Spike C)
            perspective = harmonic_resonance(
                output.tensor, input_state.tensor, harmonics=self._harmonics
            )

            # Module's RBF balance
            rb = module_balances[module.name]

            # QBE alignment: does this module's balance help?
            # tanh(balance * need) is positive when they have same sign
            # (energy-dominant module when system needs energy, etc.)
            alignment = math.tanh(rb.balance * system_need)
            qbe_scale = 1.0 + self._qbe_kappa * alignment

            # Clamp perspective (anti-resonant = no contribution)
            pr = max(0.0, perspective)

            # Combined weight
            combined = pr * max(0.0, qbe_scale)

            weights.append(QBEWeight(
                module_name=module.name,
                perspective_resonance=perspective,
                rbf_balance=rb.balance,
                qbe_scale=qbe_scale,
                combined_weight=combined,
                normalized_weight=0.0,
                system_balance=system_balance,
                qbe_target=qbe_target,
                system_need=system_need,
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
    def qbe_log(self): return self._qbe_log
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

        # QBE diagnostics
        qbe_info = {}
        if self._qbe_log:
            last_tick = self._qbe_log[-1]
            if last_tick:
                qbe_info = {
                    "system_balance": last_tick[0].system_balance,
                    "qbe_target": last_tick[0].qbe_target,
                    "system_need": last_tick[0].system_need,
                    "qbe_scales": {w.module_name: w.qbe_scale for w in last_tick},
                }

        return {
            "modules_registered": len(self._modules),
            "module_names": list(self._modules.keys()),
            "enforcement": self._enforcement,
            "tolerance": self._tolerance,
            "total_violations": len(self._violation_log),
            "qbe_ticks": len(self._qbe_log),
            "dispatch_mode": "qbe_regulated",
            "blend_rate": self._blend_rate,
            "qbe_omega": self._qbe_omega,
            "qbe_kappa": self._qbe_kappa,
            "current_tick": self._tick,
            "resonance_states": state_info,
            "qbe": qbe_info,
        }
