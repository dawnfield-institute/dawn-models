"""RecursiveEntity — wraps a CoupledFieldsBus as a GAIAModule.

This is the recursion enabler. A CoupledFieldsBus with registered modules
becomes a single GAIAModule that can be registered in a higher-level bus.
Same physics at every scale — PAC, SEC, RBF, QBE all apply recursively.

Example:
    # Level 1: modules in a bus
    inner_bus = CoupledFieldsBus(enforcement="soft")
    inner_bus.register_module(SafetyModule())
    inner_bus.register_module(ReasoningModule())

    # Level 2: that bus becomes a module in a network bus
    entity = RecursiveEntity("agent_alpha", inner_bus)
    network_bus = CoupledFieldsBus(enforcement="soft")
    network_bus.register_module(entity)  # works because entity IS a GAIAModule
"""

from __future__ import annotations

from typing import Optional

from ..core.coupled_fields_bus import CoupledFieldsBus
from ..core.types import FieldState, RBFBalance, SECPhase


class RecursiveEntity:
    """A CoupledFieldsBus wrapped as a GAIAModule.

    Satisfies the GAIAModule protocol (name, process, phase, health)
    so it can be registered as a module in any bus — enabling fractal
    nesting of coupled-field architectures.
    """

    def __init__(self, name: str, bus: CoupledFieldsBus) -> None:
        self._name = name
        self._bus = bus
        self._last_output: Optional[FieldState] = None
        self._tick_count: int = 0

    @property
    def name(self) -> str:
        return self._name

    def process(self, field_state: FieldState) -> FieldState:
        """Run full internal bus dispatch. Same interface as any module."""
        self._last_output = self._bus.process(field_state)
        self._tick_count += 1
        return self._last_output

    def phase(self) -> SECPhase:
        """Dominant phase = phase of last output, or ORDERED if no output yet."""
        if self._last_output is not None:
            return self._last_output.phase
        return SECPhase.ORDERED

    def health(self) -> RBFBalance:
        """Aggregate health = mean of all internal module healths."""
        modules = list(self._bus._modules.values())
        if not modules:
            return RBFBalance.compute(energy=1.0, information=1.0, memory=0.0)

        total_e, total_i, total_m = 0.0, 0.0, 0.0
        for module in modules:
            rbf = module.health()
            total_e += rbf.energy
            total_i += rbf.information
            total_m += rbf.memory

        n = len(modules)
        return RBFBalance.compute(
            energy=total_e / n,
            information=total_i / n,
            memory=total_m / n,
        )

    @property
    def bus(self) -> CoupledFieldsBus:
        """Access to the internal bus (for identity sync, metrics, etc.)."""
        return self._bus

    @property
    def tick_count(self) -> int:
        return self._tick_count
