"""SEC phase router — zero-parameter routing based on entropy classification."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from .types import FieldState, SECPhase

if TYPE_CHECKING:
    from .protocol import GAIAModule


# DFT-derived thresholds (zero-parameter)
SEC_THRESHOLDS: list[tuple[float, SECPhase]] = [
    (0.5, SECPhase.CRYSTALLIZED),
    (2.0, SECPhase.ORDERED),
    (4.0, SECPhase.TRANSITIONAL),
    # >= 4.0 is CHAOTIC (handled as default)
]


class SECRouter:
    """Routes FieldState to modules based on SEC phase classification.

    Uses DFT-derived entropy thresholds for zero-parameter routing:
        CRYSTALLIZED:  H < 0.5
        ORDERED:       0.5 <= H < 2.0
        TRANSITIONAL:  2.0 <= H < 4.0
        CHAOTIC:       H >= 4.0
    """

    def __init__(self) -> None:
        self._phase_handlers: dict[SECPhase, list[GAIAModule]] = defaultdict(list)

    def classify(self, entropy: float) -> SECPhase:
        """Classify entropy value into SEC phase.

        Args:
            entropy: Raw Shannon entropy (nats), unbounded.

        Returns:
            The SEC phase for the given entropy level.
        """
        for threshold, phase in SEC_THRESHOLDS:
            if entropy < threshold:
                return phase
        return SECPhase.CHAOTIC

    def register(self, module: GAIAModule, phases: list[SECPhase] | None = None) -> None:
        """Register a module to handle specific SEC phases.

        Args:
            module: Module satisfying the GAIAModule protocol.
            phases: Which phases this module handles. None = all phases.
        """
        if phases is None:
            phases = list(SECPhase)
        for phase in phases:
            if module not in self._phase_handlers[phase]:
                self._phase_handlers[phase].append(module)

    def route(self, field_state: FieldState) -> list[GAIAModule]:
        """Determine which modules should process this FieldState.

        Args:
            field_state: Input state to route.

        Returns:
            List of modules registered for the input's SEC phase.
            Empty list if no modules are registered for that phase.
        """
        phase = self.classify(field_state.entropy)
        return list(self._phase_handlers[phase])

    @property
    def registered_modules(self) -> dict[SECPhase, list[GAIAModule]]:
        """View of all registered phase-module mappings."""
        return dict(self._phase_handlers)
