"""GAIANetwork — multiple agents coupled via CoupledFieldsBus.

The network IS a CoupledFieldsBus where each registered "module" is a
GAIAAgent. Same physics at every scale:

    Module level:  CoupledFieldsBus couples Safety, Reasoning, Memory, ...
    Agent level:   CoupledFieldsBus couples Agent_A, Agent_B, Agent_C, ...

Each agent develops a spectral lens on the network signal space.
QBE regulates inter-agent coupling strength. SEC routes signals to
agents whose entropy phase matches. RBF suppresses unhealthy agents.

The QSocket concept from the design is realized by the network bus itself —
no separate communication layer needed because CoupledFieldsBus already
handles lensing, coupling, QBE modulation, and PAC conservation.
"""

from __future__ import annotations

from ..core.coupled_fields_bus import CoupledFieldsBus, CoupledFieldState
from ..core.types import FieldState, SECPhase

from .agent import GAIAAgent


class GAIANetwork:
    """Network of GAIA Agents coupled via CoupledFieldsBus.

    Usage:
        agents = [
            GAIAAgent("alpha", [SafetyModule(64), ReasoningModule(64)]),
            GAIAAgent("beta", [MemoryModule(64), LanguageModule(64)]),
            GAIAAgent("gamma", [SafetyModule(64), MemoryModule(64)]),
        ]
        network = GAIANetwork(network_dim=64)
        for agent in agents:
            network.add_agent(agent)

        # Process signals — agents specialize over time
        for signal in signals:
            output = network.process(signal)

        # Observe emergent specialization
        print(network.get_specializations())
    """

    def __init__(
        self,
        network_dim: int = 64,
        enforcement: str = "soft",
        **bus_kwargs,
    ) -> None:
        self._network_dim = network_dim
        self._bus = CoupledFieldsBus(enforcement=enforcement, **bus_kwargs)
        self._agents: dict[str, GAIAAgent] = {}

    def add_agent(
        self,
        agent: GAIAAgent,
        phases: list[SECPhase] | None = None,
    ) -> None:
        """Add an agent to the network.

        The agent becomes a module in the network-level CoupledFieldsBus,
        participating in lensed broadcast, coupling, and QBE regulation.
        """
        self._agents[agent.name] = agent
        self._bus.register_module(agent, phases)

    def remove_agent(self, name: str) -> None:
        """Remove an agent from the network."""
        if name in self._agents:
            del self._agents[name]
        if name in self._bus._modules:
            del self._bus._modules[name]
        if name in self._bus._field_states:
            del self._bus._field_states[name]

    def process(self, signal: FieldState) -> FieldState:
        """Process signal through the network.

        The CoupledFieldsBus handles everything:
        1. Each agent gets the signal through its spectral lens
        2. Each agent processes internally (full coupled-field dispatch)
        3. Outputs weighted by prediction_accuracy * perspective_resonance
        4. Coupling forces synchronize/diverge agent states
        5. QBE modulates coupling strength
        6. PAC conservation enforced at network boundary
        """
        result = self._bus.process(signal)

        # Sync agent identities from the network bus's tracking
        for name, agent in self._agents.items():
            if name in self._bus._field_states:
                agent.identity.update_from_bus_state(self._bus._field_states[name])

        return result

    def get_specializations(self) -> dict[str, float]:
        """Report each agent's specialization score (lens_contrast = std(lens))."""
        return {
            name: agent.identity.specialization
            for name, agent in self._agents.items()
        }

    def get_coupling_matrix(self) -> dict[tuple[str, str], float]:
        """Report inter-agent harmonic resonance coupling strengths."""
        names = list(self._agents.keys())
        return self._bus._compute_coupling_matrix(names)

    def get_agent(self, name: str) -> GAIAAgent:
        """Retrieve an agent by name."""
        return self._agents[name]

    @property
    def agents(self) -> dict[str, GAIAAgent]:
        return dict(self._agents)

    @property
    def bus(self) -> CoupledFieldsBus:
        return self._bus

    @property
    def network_dim(self) -> int:
        return self._network_dim

    def get_metrics(self) -> dict:
        """Network-level metrics: specializations, coupling, per-agent health."""
        bus_metrics = self._bus.get_metrics()
        agent_metrics = {}
        for name, agent in self._agents.items():
            agent_metrics[name] = {
                "specialization": agent.identity.specialization,
                "experience": agent.identity.experience,
                "modules": agent.module_names,
                "children": list(agent.children.keys()),
                "health": agent.health().balance,
            }
        return {
            "network_dim": self._network_dim,
            "agent_count": len(self._agents),
            "agents": agent_metrics,
            "bus": bus_metrics,
            "specializations": self.get_specializations(),
        }
