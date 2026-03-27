"""GAIAAgent — a GAIA Core elevated to an autonomous agent.

A GAIAAgent wraps a CoupledFieldsBus (via RecursiveEntity) and adds:
    - Persistent identity (spectral lens, specialization tracking)
    - Self-modification (add/remove modules at runtime)
    - Sub-agent spawning (fractal nesting)

Implements GAIAModule so it can participate in a network-level bus.
Same physics at every scale.
"""

from __future__ import annotations

from typing import Optional

from ..core.coupled_fields_bus import CoupledFieldsBus
from ..core.protocol import GAIAModule
from ..core.types import FieldState, SECPhase

from .identity import AgentIdentity
from .recursive_entity import RecursiveEntity


class GAIAAgent(RecursiveEntity):
    """A GAIA Core with identity, self-modification, and sub-agent spawning.

    Usage:
        agent = GAIAAgent("alpha", [safety_mod, reasoning_mod])
        result = agent.process(field_state)  # runs internal coupled-field dispatch
        print(agent.identity.specialization)  # how specialized is this agent?

    The agent implements GAIAModule so it can be registered in a
    network-level CoupledFieldsBus alongside other agents.
    """

    def __init__(
        self,
        name: str,
        modules: list[GAIAModule],
        field_dim: int = 64,
        enforcement: str = "soft",
        **bus_kwargs,
    ) -> None:
        bus = CoupledFieldsBus(enforcement=enforcement, **bus_kwargs)
        for module in modules:
            bus.register_module(module)
        super().__init__(name, bus)

        self._identity = AgentIdentity(agent_name=name, field_dim=field_dim)
        self._field_dim = field_dim
        self._children: dict[str, GAIAAgent] = {}

    def process(self, field_state: FieldState) -> FieldState:
        """Process through internal bus, then sync identity from bus state."""
        result = super().process(field_state)

        # Sync identity from the network-level bus's tracking of this agent.
        # The network bus (if this agent is registered in one) maintains a
        # CoupledFieldState for this agent. But we can also track from our
        # own internal bus state — the aggregate of our modules' evolution.
        self._sync_identity_from_internal()

        return result

    def _sync_identity_from_internal(self) -> None:
        """Update identity from internal bus module states.

        The agent's identity aggregates its internal modules' resonance
        states into a single field — the "emergent perspective" of the
        whole agent, which is what the network-level bus couples on.
        """
        states = self._bus.field_states
        if not states:
            return

        # Aggregate: mean of all internal module resonance states
        tensors = [s.tensor for s in states.values()]
        mean_tensor = sum(tensors) / len(tensors)  # type: ignore[arg-type]
        self._identity.resonance_field = mean_tensor.clone()

        # Lens: mean of all internal module lenses
        lenses = [s.lens for s in states.values()]
        mean_lens = sum(lenses) / len(lenses)  # type: ignore[arg-type]
        self._identity.spectral_lens = mean_lens.clone()

        # Experience: max ticks of any internal module
        self._identity.experience = max(
            (s.ticks_alive for s in states.values()), default=0
        )

    @property
    def identity(self) -> AgentIdentity:
        return self._identity

    @property
    def field_dim(self) -> int:
        return self._field_dim

    @property
    def module_names(self) -> list[str]:
        return list(self._bus._modules.keys())

    @property
    def children(self) -> dict[str, GAIAAgent]:
        return dict(self._children)

    # --- Self-modification ---

    def add_module(self, module: GAIAModule, phases: list[SECPhase] | None = None) -> None:
        """Add a module at runtime. Agent gains new capability."""
        self._bus.register_module(module, phases)

    def remove_module(self, name: str) -> None:
        """Remove a module at runtime. Agent sheds unused capability.

        Removes the module from the bus internals (router, field states).
        """
        if name in self._bus._modules:
            del self._bus._modules[name]
        if name in self._bus._field_states:
            del self._bus._field_states[name]
        # Remove from router's internal tracking
        for phase, handlers in self._bus._router._phase_handlers.items():
            self._bus._router._phase_handlers[phase] = [
                m for m in handlers if m.name != name
            ]

    def spawn_sub_agent(
        self,
        name: str,
        modules: list[GAIAModule],
        field_dim: int | None = None,
        **bus_kwargs,
    ) -> GAIAAgent:
        """Create a sub-agent and register it as a module in this agent's bus.

        Fractal nesting: the child agent IS a GAIAModule in the parent's
        bus, participating in coupled-field dispatch alongside other modules.
        """
        child = GAIAAgent(
            name=name,
            modules=modules,
            field_dim=field_dim or self._field_dim,
            **bus_kwargs,
        )
        # Register the child as a module in this agent's bus
        self._bus.register_module(child)
        self._children[name] = child
        return child

    def get_metrics(self) -> dict:
        """Agent-level metrics including identity and internal bus state."""
        bus_metrics = self._bus.get_metrics()
        return {
            "agent_name": self._name,
            "identity": {
                "specialization": self._identity.specialization,
                "experience": self._identity.experience,
                "field_dim": self._field_dim,
            },
            "modules": bus_metrics["module_names"],
            "children": list(self._children.keys()),
            "internal_bus": bus_metrics,
        }
