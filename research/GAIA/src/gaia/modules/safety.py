"""Safety Module — Hallucination detection via PAC conservation.

Ported from TinyCIMM-Boltzmann. Core hypothesis: hallucination IS PAC
violation — uncompensated entropy creation. This module wraps the
Boltzmann architecture as a GAIAModule that plugs into the ConservationBus.

Components:
    BoltzmannHead: Single processing head with tracked activation entropy.
    BoltzmannLayer: N parallel heads with shared entropy budget.
    ConservationProjector: Soft/hard PAC enforcement across heads.
    BoltzmannMonitor: Real-time conservation tracking.
    SafetyModule: GAIAModule wrapper for the full Boltzmann stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gaia.core.types import FieldState, RBFBalance, SECPhase

# SEC thresholds (same as core/sec_router.py — raw Shannon entropy)
_SEC_CRYSTALLIZED = 0.5
_SEC_ORDERED = 2.0
_SEC_TRANSITIONAL = 4.0


def _classify_sec_phase(entropy: float) -> SECPhase:
    """Classify SEC phase from entropy value."""
    if entropy < _SEC_CRYSTALLIZED:
        return SECPhase.CRYSTALLIZED
    elif entropy < _SEC_ORDERED:
        return SECPhase.ORDERED
    elif entropy < _SEC_TRANSITIONAL:
        return SECPhase.TRANSITIONAL
    return SECPhase.CHAOTIC


# ─── Data Structures ────────────────────────────────────────────────


@dataclass
class ConservationState:
    """Tracks PAC conservation metrics across heads."""

    total_budget: float = 0.0
    target_budget: float = 0.0
    violation: float = 0.0
    compensation_ratio: float = 1.0
    head_entropies: list[float] = field(default_factory=list)
    phase_distribution: dict[str, int] = field(default_factory=dict)
    steps: int = 0


@dataclass
class SafetyMetrics:
    """Metrics from a safety module processing step."""

    total_budget: float = 0.0
    target_budget: float = 0.0
    violation_pct: float = 0.0
    compensation_ratio: float = 1.0
    budget_stability: float = 1.0
    head_entropies: list[float] = field(default_factory=list)
    head_phases: list[SECPhase] = field(default_factory=list)
    mean_entropy: float = 0.0
    conservation_loss: float = 0.0


# ─── Core Components ────────────────────────────────────────────────


class BoltzmannHead(nn.Module):
    """Single processing head with tracked activation entropy.

    The head computes attention-like scores and tracks the Shannon entropy
    of its activation distribution. This entropy is the signal used for
    PAC conservation enforcement.
    """

    def __init__(self, input_dim: int, head_dim: int) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.W_q = nn.Linear(input_dim, head_dim, bias=False)
        self.W_k = nn.Linear(input_dim, head_dim, bias=False)
        self.W_v = nn.Linear(input_dim, head_dim, bias=False)

        self._last_entropy = 0.0
        self._last_phase = SECPhase.ORDERED

        nn.init.xavier_uniform_(self.W_q.weight, gain=0.5)
        nn.init.xavier_uniform_(self.W_k.weight, gain=0.5)
        nn.init.xavier_uniform_(self.W_v.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, float, torch.Tensor]:
        """Forward pass.

        Returns:
            (output, entropy_float, entropy_tensor) where entropy_tensor
            is differentiable for backprop through conservation loss.
        """
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        scale = math.sqrt(self.head_dim)
        scores = F.softmax(torch.sum(q * k, dim=-1, keepdim=True) / scale, dim=-1)
        out = scores * v

        # Differentiable entropy proxy
        act_sq = out.flatten() ** 2 + 1e-10
        act_prob = act_sq / act_sq.sum()
        entropy_tensor = -torch.sum(act_prob * torch.log(act_prob))

        entropy = float(entropy_tensor.detach().item())
        self._last_entropy = entropy
        self._last_phase = _classify_sec_phase(entropy)

        return out, entropy, entropy_tensor

    @property
    def entropy(self) -> float:
        return self._last_entropy

    @property
    def phase(self) -> SECPhase:
        return self._last_phase


class ConservationProjector(nn.Module):
    """Enforces PAC conservation across heads within a layer.

    Two modes:
        soft: Adds a conservation loss term (penalty for violation).
        hard: Explicitly normalizes head outputs to enforce budget.
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        mode: str = "soft",
        conservation_strength: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.mode = mode
        self.conservation_strength = conservation_strength

        self.mix = nn.Linear(n_heads * head_dim, n_heads * head_dim, bias=False)
        nn.init.eye_(self.mix.weight)

        self._target_budget: Optional[float] = None
        self._budget_initialized = False

    def set_target_budget(self, budget: float) -> None:
        self._target_budget = budget
        self._budget_initialized = True

    @property
    def target_budget(self) -> Optional[float]:
        return self._target_budget

    def forward(
        self,
        head_outputs: list[torch.Tensor],
        head_entropies: list[float],
        entropy_tensors: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine head outputs while tracking conservation.

        Returns:
            (combined_output, conservation_loss_tensor)
        """
        combined = torch.cat(head_outputs, dim=-1)
        mixed = self.mix(combined)

        current_budget = sum(head_entropies)

        if not self._budget_initialized:
            self._target_budget = current_budget
            self._budget_initialized = True

        if self.mode == "soft":
            total_entropy_tensor = sum(entropy_tensors)
            target = torch.tensor(
                self._target_budget, device=mixed.device, dtype=mixed.dtype
            )
            conservation_loss = self.conservation_strength * (
                total_entropy_tensor - target
            ) ** 2
        elif self.mode == "hard":
            if current_budget > 1e-8:
                scale_factor = self._target_budget / current_budget
                mixed = mixed * math.sqrt(scale_factor)
            conservation_loss = torch.tensor(0.0, device=mixed.device)
        else:
            conservation_loss = torch.tensor(0.0, device=mixed.device)

        return mixed, conservation_loss


class BoltzmannLayer(nn.Module):
    """Multi-head layer with PAC-conserved entropy budget.

    Contains N parallel BoltzmannHeads plus a ConservationProjector
    that enforces the total entropy budget stays constant.
    """

    def __init__(
        self,
        input_dim: int,
        n_heads: int = 4,
        head_dim: int = 8,
        output_dim: int | None = None,
        conservation_mode: str = "soft",
        conservation_strength: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.heads = nn.ModuleList(
            [BoltzmannHead(input_dim, head_dim) for _ in range(n_heads)]
        )
        self.projector = ConservationProjector(
            n_heads, head_dim, mode=conservation_mode,
            conservation_strength=conservation_strength,
        )
        out_dim = output_dim or input_dim
        self.output_proj = nn.Linear(n_heads * head_dim, out_dim, bias=True)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
        """Forward pass.

        Returns:
            (output, conservation_loss_tensor, head_entropies)
        """
        head_outputs = []
        head_entropies = []
        entropy_tensors = []

        for head in self.heads:
            out, entropy, entropy_t = head(x)
            head_outputs.append(out)
            head_entropies.append(entropy)
            entropy_tensors.append(entropy_t)

        combined, conservation_loss = self.projector(
            head_outputs, head_entropies, entropy_tensors
        )
        output = self.output_proj(combined)

        return output, conservation_loss, head_entropies


class BoltzmannMonitor:
    """Real-time PAC conservation monitor.

    Tracks entropy budget stability, violation trends, and
    compensation dynamics (when one head increases, does another decrease?).
    """

    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size
        self.budget_history: list[float] = []
        self.violation_history: list[float] = []
        self.head_entropy_history: list[list[float]] = []
        self.compensation_history: list[float] = []
        self.state = ConservationState()

    def update(self, head_entropies: list[float], target_budget: float) -> ConservationState:
        """Update conservation state with new measurements."""
        self.state.steps += 1
        self.state.head_entropies = head_entropies
        self.state.total_budget = sum(head_entropies)
        self.state.target_budget = target_budget
        self.state.violation = self.state.total_budget - target_budget

        # Compensation analysis
        if self.head_entropy_history:
            prev = self.head_entropy_history[-1]
            deltas = [h - p for h, p in zip(head_entropies, prev)]
            increases = sum(d for d in deltas if d > 0)
            decreases = sum(d for d in deltas if d < 0)
            if increases > 1e-8:
                self.state.compensation_ratio = abs(decreases / increases)
            else:
                self.state.compensation_ratio = 1.0
            self.compensation_history.append(self.state.compensation_ratio)

        # Phase distribution
        self.state.phase_distribution = {}
        for h in head_entropies:
            p = _classify_sec_phase(h).value
            self.state.phase_distribution[p] = self.state.phase_distribution.get(p, 0) + 1

        # Update histories
        violation_pct = (self.state.violation / max(target_budget, 1e-8)) * 100
        self.budget_history.append(self.state.total_budget)
        self.violation_history.append(violation_pct)
        self.head_entropy_history.append(list(head_entropies))

        # Trim to window
        for history in (self.budget_history, self.violation_history,
                        self.head_entropy_history, self.compensation_history):
            while len(history) > self.window_size:
                history.pop(0)

        return self.state

    def budget_stability(self) -> float:
        """How stable is the budget? 1.0 = perfectly constant."""
        if len(self.budget_history) < 3:
            return 1.0
        mean = sum(self.budget_history) / len(self.budget_history)
        if mean < 1e-8:
            return 1.0
        variance = sum((x - mean) ** 2 for x in self.budget_history) / len(self.budget_history)
        std = variance ** 0.5
        cv = std / mean
        return 1.0 / (1.0 + cv * 10)

    def mean_compensation(self) -> float:
        """Average compensation ratio over window."""
        if not self.compensation_history:
            return 1.0
        return sum(self.compensation_history) / len(self.compensation_history)


# ─── GAIAModule Wrapper ─────────────────────────────────────────────


class SafetyModule:
    """GAIA Safety Module — wraps TinyCIMM-Boltzmann as a GAIAModule.

    Processes FieldState through Boltzmann layers that enforce PAC
    conservation on the internal representation. Detects hallucination
    as uncompensated entropy creation.

    The module is PAC-conserving at its boundary: it transforms the
    tensor but preserves total energy (via internal conservation
    projector enforcement).

    Args:
        input_dim: Dimension of FieldState tensor (flattened).
        n_heads: Number of parallel Boltzmann heads per layer.
        n_layers: Number of stacked Boltzmann layers.
        hidden_dim: Internal hidden dimension.
        conservation_mode: "soft" (loss penalty) or "hard" (normalization).
        conservation_strength: Weight of conservation loss in soft mode.
    """

    def __init__(
        self,
        input_dim: int,
        n_heads: int = 4,
        n_layers: int = 2,
        hidden_dim: int = 32,
        conservation_mode: str = "soft",
        conservation_strength: float = 1.0,
    ) -> None:
        self._input_dim = input_dim
        self._n_heads = n_heads
        self._n_layers = n_layers
        self._hidden_dim = hidden_dim
        self._conservation_mode = conservation_mode

        head_dim = hidden_dim // n_heads

        # Build Boltzmann layers
        self._layers: list[BoltzmannLayer] = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layer = BoltzmannLayer(
                input_dim=in_dim,
                n_heads=n_heads,
                head_dim=head_dim,
                output_dim=hidden_dim,
                conservation_mode=conservation_mode,
                conservation_strength=conservation_strength,
            )
            self._layers.append(layer)

        # Project back to input dimension to preserve PAC at boundary
        self._output_proj = nn.Linear(hidden_dim, input_dim, bias=False)
        nn.init.eye_(self._output_proj.weight[:min(input_dim, hidden_dim), :min(input_dim, hidden_dim)])

        # Monitors
        self._monitors = [BoltzmannMonitor() for _ in range(n_layers)]
        self._last_metrics: Optional[SafetyMetrics] = None
        self._step_count = 0

    @property
    def name(self) -> str:
        return "safety"

    def process(self, field_state: FieldState) -> FieldState:
        """Process field state through Boltzmann safety layers.

        The module checks the tensor for hallucination signatures
        (PAC violations across internal heads) and returns a
        conservation-enforced output.

        The output tensor has the same total energy as the input
        (PAC-conserving at the module boundary).
        """
        self._step_count += 1
        result = field_state.clone()

        # Pass through Boltzmann layers
        x = field_state.tensor.unsqueeze(0) if field_state.tensor.dim() == 1 else field_state.tensor
        total_conservation_loss = torch.tensor(0.0)
        all_head_entropies: list[float] = []

        with torch.no_grad():
            h = x
            for i, layer in enumerate(self._layers):
                h, cons_loss, head_ents = layer(h)
                h = F.gelu(h)
                total_conservation_loss = total_conservation_loss + cons_loss
                all_head_entropies.extend(head_ents)

                # Update monitor
                target = layer.projector.target_budget or sum(head_ents)
                self._monitors[i].update(head_ents, target)

            # Project back to input dimension
            h = self._output_proj(h)

        # Squeeze back if input was 1D
        output_tensor = h.squeeze(0) if field_state.tensor.dim() == 1 else h

        # Enforce PAC conservation at module boundary:
        # Scale output to match input total energy exactly
        input_energy = field_state.total_energy()
        output_energy = float(torch.sum(output_tensor).item())
        if abs(output_energy) > 1e-10:
            scale = input_energy / output_energy
            output_tensor = output_tensor * scale

        result.tensor = output_tensor
        result.provenance.append(self.name)

        # Compute metrics
        head_phases = [_classify_sec_phase(e) for e in all_head_entropies]
        mean_ent = sum(all_head_entropies) / len(all_head_entropies) if all_head_entropies else 0.0
        total_budget = sum(all_head_entropies)
        target_budget = sum(
            layer.projector.target_budget or 0.0 for layer in self._layers
        )
        violation_pct = (
            (total_budget - target_budget) / max(target_budget, 1e-8) * 100
            if target_budget > 0 else 0.0
        )

        self._last_metrics = SafetyMetrics(
            total_budget=total_budget,
            target_budget=target_budget,
            violation_pct=violation_pct,
            compensation_ratio=sum(m.mean_compensation() for m in self._monitors) / len(self._monitors),
            budget_stability=sum(m.budget_stability() for m in self._monitors) / len(self._monitors),
            head_entropies=all_head_entropies,
            head_phases=head_phases,
            mean_entropy=mean_ent,
            conservation_loss=float(total_conservation_loss.item()),
        )

        return result

    def phase(self) -> SECPhase:
        """Current SEC phase based on mean head entropy."""
        if self._last_metrics and self._last_metrics.head_entropies:
            return _classify_sec_phase(self._last_metrics.mean_entropy)
        return SECPhase.ORDERED

    def health(self) -> RBFBalance:
        """RBF balance based on conservation state.

        Energy = budget stability (how well conservation holds).
        Information = mean entropy (how much information is flowing).
        Memory = step count normalized (processing load).
        """
        if self._last_metrics:
            energy = self._last_metrics.budget_stability
            information = min(self._last_metrics.mean_entropy, 5.0) / 5.0
            memory = min(self._step_count, 1000) / 1000.0
        else:
            energy = 1.0
            information = 0.5
            memory = 0.0
        return RBFBalance.compute(energy=energy, information=information, memory=memory)

    @property
    def metrics(self) -> Optional[SafetyMetrics]:
        """Last processing step metrics."""
        return self._last_metrics

    @property
    def monitors(self) -> list[BoltzmannMonitor]:
        """Layer monitors for detailed analysis."""
        return list(self._monitors)
