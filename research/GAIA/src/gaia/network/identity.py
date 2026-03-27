"""AgentIdentity — persistent identity derived from accumulated resonance state.

An agent's identity IS its spectral lens — the filter it has developed through
processing signals over time. An agent that processes many safety-related
signals develops a lens that amplifies safety-relevant tensor dimensions.
This IS specialization.

Maps directly to CoupledFieldState from coupled_fields_bus.py:
    - resonance_field = CoupledFieldState.tensor
    - spectral_lens = CoupledFieldState.lens
    - specialization = std(lens) = lens_contrast
    - experience = CoupledFieldState.ticks_alive
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..core.coupled_fields_bus import CoupledFieldState


@dataclass
class AgentIdentity:
    """Persistent identity derived from accumulated resonance state.

    The network-level CoupledFieldsBus maintains a CoupledFieldState for
    each agent. This class syncs from that state and provides accessors
    for identity-level properties (specialization, aggregate field for
    inter-agent coupling).
    """

    agent_name: str
    field_dim: int
    resonance_field: torch.Tensor = field(default=None)  # type: ignore[assignment]
    spectral_lens: torch.Tensor = field(default=None)  # type: ignore[assignment]
    experience: int = 0
    surprise_history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.resonance_field is None:
            self.resonance_field = torch.ones(self.field_dim)
        if self.spectral_lens is None:
            self.spectral_lens = torch.ones(self.field_dim)

    @property
    def specialization(self) -> float:
        """How specialized is this agent? std(lens).

        High specialization = the agent has developed strong preferences
        about which tensor dimensions matter. Low = flat, unbiased lens.
        """
        return float(self.spectral_lens.std().item())

    def aggregate_field(self) -> torch.Tensor:
        """The agent's state for inter-agent coupling computation.

        This is what the network bus uses to compute harmonic_resonance
        between agents — the same way CoupledFieldsBus uses module
        states for inter-module coupling.
        """
        return self.resonance_field

    def update_from_bus_state(self, coupled_state: CoupledFieldState) -> None:
        """Sync identity from the network bus's internal tracking."""
        self.resonance_field = coupled_state.tensor.clone()
        self.spectral_lens = coupled_state.lens.clone()
        self.experience = coupled_state.ticks_alive
        self.surprise_history = list(coupled_state.surprise_history)

    def lens_divergence(self, other: AgentIdentity) -> float:
        """Measure how different two agents' lenses are (L2 distance)."""
        return float(torch.norm(self.spectral_lens - other.spectral_lens).item())
