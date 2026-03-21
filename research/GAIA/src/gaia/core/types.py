"""Core data types for the GAIA v2 conservation bus."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch


class SECPhase(Enum):
    """SEC entropy phase classification.

    DFT-derived thresholds (zero-parameter):
        CRYSTALLIZED:  H < 0.5   — routine, stable
        ORDERED:       0.5 <= H < 2.0 — structured
        TRANSITIONAL:  2.0 <= H < 4.0 — edge of chaos
        CHAOTIC:       H >= 4.0  — novel, requires full orchestration
    """

    CRYSTALLIZED = "crystallized"
    ORDERED = "ordered"
    TRANSITIONAL = "transitional"
    CHAOTIC = "chaotic"


@dataclass
class FieldState:
    """Entropy-encoded state flowing between modules.

    The universal data type on the conservation bus. Every module
    receives and returns a FieldState, with PAC conservation
    enforced at each boundary.
    """

    tensor: torch.Tensor
    entropy: float
    phase: SECPhase = SECPhase.ORDERED
    conservation_budget: float = 0.0
    provenance: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def total_energy(self) -> float:
        """Conserved scalar quantity — sum of all tensor elements."""
        return float(torch.sum(self.tensor).item())

    def clone(self) -> FieldState:
        """Deep copy for immutable processing."""
        return FieldState(
            tensor=self.tensor.clone(),
            entropy=self.entropy,
            phase=self.phase,
            conservation_budget=self.conservation_budget,
            provenance=list(self.provenance),
            timestamp=self.timestamp,
        )


@dataclass
class RBFBalance:
    """Energy-information balance for a module.

    B = lambda * (E - I) / (1 + alpha * M)

    Positive B: module producing more energy than information (overactive)
    Negative B: module consuming more energy than producing (underactive)
    Zero B: balanced
    """

    energy: float
    information: float
    memory: float
    balance: float

    @staticmethod
    def compute(
        energy: float,
        information: float,
        memory: float,
        rbf_lambda: float = 1.0,
        rbf_alpha: float = 0.1,
    ) -> RBFBalance:
        """Compute RBF balance from components."""
        balance = rbf_lambda * (energy - information) / (1.0 + rbf_alpha * memory)
        return RBFBalance(
            energy=energy,
            information=information,
            memory=memory,
            balance=balance,
        )


@dataclass
class ConservationResult:
    """Result of PAC conservation validation at a module boundary."""

    conserved: bool
    input_energy: float
    output_energy: float
    residual: float
    module_name: str
    violation_type: Optional[str] = None  # "hard" | "soft" | None
