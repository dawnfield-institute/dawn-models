"""GAIAModule protocol — the contract for pluggable modules."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import FieldState, RBFBalance, SECPhase


@runtime_checkable
class GAIAModule(Protocol):
    """Contract for pluggable GAIA modules.

    Every module that plugs into the conservation bus must implement
    these three methods. The bus validates PAC conservation between
    the input and output FieldState of each process() call.
    """

    @property
    def name(self) -> str:
        """Module identifier for provenance tracking and logging."""
        ...

    def process(self, field_state: FieldState) -> FieldState:
        """Transform input field state to output field state.

        Contract: PAC conservation must hold at the boundary.
        |input.total_energy() - output.total_energy()| < tolerance
        Unless explicit compensation is declared via conservation_budget.
        """
        ...

    def phase(self) -> SECPhase:
        """Current SEC phase of this module.

        Used by the conservation bus for routing decisions.
        """
        ...

    def health(self) -> RBFBalance:
        """Current energy-information balance.

        Used by the bus to suppress overactive or unhealthy modules.
        """
        ...
