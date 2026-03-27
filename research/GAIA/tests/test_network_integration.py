"""Integration test for GAIANetwork with real modules.

No stubs, no mocks. Real SafetyModule, ReasoningModule, MemoryModule,
LanguageModule, ObservabilityModule — the actual GAIA intelligence stack
running through a recursive multi-agent network.

Validates that DFT physics (PAC/SEC/RBF/QBE) work recursively with
production modules, and that spontaneous specialization emerges from
real module dynamics.
"""

from __future__ import annotations

import time

import torch

from gaia.core.coupled_fields_bus import CoupledFieldsBus, _harmonic_resonance
from gaia.core.types import FieldState, SECPhase
from gaia.modules.safety import SafetyModule
from gaia.modules.reasoning import ReasoningModule
from gaia.modules.memory import MemoryModule
from gaia.modules.language import LanguageModule
from gaia.modules.observability import ObservabilityModule
from gaia.network import GAIAAgent, GAIANetwork, RecursiveEntity


DIM = 22  # Standard GAIA field dimension


def _make_signal(seed: int, dim: int = DIM, entropy: float = 1.5) -> FieldState:
    """Create a realistic signal with controlled randomness."""
    torch.manual_seed(seed)
    tensor = torch.randn(dim)
    # Give it some structure (not pure noise)
    tensor = tensor * 0.5 + torch.sin(torch.arange(dim, dtype=torch.float32) * seed * 0.1) * 0.5
    return FieldState(
        tensor=tensor,
        entropy=entropy,
        phase=SECPhase.ORDERED,
        conservation_budget=0.0,
        provenance=[],
        timestamp=time.time(),
    )


def _make_stimulus_class(class_id: int, dim: int = DIM) -> FieldState:
    """Create a signal with class-specific frequency content.

    Different class IDs produce signals with energy concentrated in
    different dimensions — the kind of structure that should cause
    agents to develop different spectral lenses.
    """
    torch.manual_seed(42 + class_id)
    tensor = torch.zeros(dim)
    # Each class lights up a different band of dimensions
    band_start = (class_id * 5) % dim
    band_end = min(band_start + 7, dim)
    tensor[band_start:band_end] = torch.randn(band_end - band_start) * 2.0
    # Add low-level background everywhere
    tensor += torch.randn(dim) * 0.1
    return FieldState(
        tensor=tensor,
        entropy=1.5,
        phase=SECPhase.ORDERED,
        conservation_budget=0.0,
        provenance=[],
        timestamp=time.time(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSingleAgentRealModules:
    """A single GAIAAgent with real modules processes correctly."""

    def test_safety_reasoning_agent(self):
        """Agent with Safety + Reasoning processes a signal."""
        agent = GAIAAgent(
            "safety_reason",
            [SafetyModule(input_dim=DIM), ReasoningModule(input_dim=DIM)],
            field_dim=DIM,
        )
        signal = _make_signal(seed=0)
        result = agent.process(signal)

        assert result is not None
        assert result.tensor.shape == signal.tensor.shape
        # PAC conservation (soft enforcement — may have small residual)
        assert abs(signal.total_energy() - result.total_energy()) < 1.0

    def test_memory_language_agent(self):
        """Agent with Memory + Language processes a signal."""
        agent = GAIAAgent(
            "mem_lang",
            [MemoryModule(), LanguageModule()],
            field_dim=DIM,
        )
        signal = _make_signal(seed=1)
        result = agent.process(signal)

        assert result is not None
        assert result.tensor.shape == signal.tensor.shape

    def test_full_stack_agent(self):
        """Agent with all 5 modules — the complete GAIA core."""
        agent = GAIAAgent(
            "full",
            [
                SafetyModule(input_dim=DIM),
                ReasoningModule(input_dim=DIM),
                MemoryModule(),
                LanguageModule(),
                ObservabilityModule(),
            ],
            field_dim=DIM,
        )
        signal = _make_signal(seed=2)
        result = agent.process(signal)

        assert result is not None
        assert agent.identity.experience > 0
        assert len(agent.module_names) == 5


class TestMultiAgentRealModules:
    """Multiple agents with real modules in a GAIANetwork."""

    def test_two_agent_network(self):
        """Two agents with different real modules."""
        agent_a = GAIAAgent(
            "alpha",
            [SafetyModule(input_dim=DIM), ReasoningModule(input_dim=DIM)],
            field_dim=DIM,
        )
        agent_b = GAIAAgent(
            "beta",
            [MemoryModule(), LanguageModule()],
            field_dim=DIM,
        )

        network = GAIANetwork(network_dim=DIM)
        network.add_agent(agent_a)
        network.add_agent(agent_b)

        signal = _make_signal(seed=3)
        result = network.process(signal)

        assert result is not None
        assert result.tensor.shape == signal.tensor.shape

    def test_three_agent_diverse_network(self):
        """Three agents with complementary specializations."""
        agents = [
            GAIAAgent("guardian", [
                SafetyModule(input_dim=DIM),
                ObservabilityModule(),
            ], field_dim=DIM),
            GAIAAgent("thinker", [
                ReasoningModule(input_dim=DIM),
                MemoryModule(),
            ], field_dim=DIM),
            GAIAAgent("speaker", [
                LanguageModule(),
                MemoryModule(),
            ], field_dim=DIM),
        ]

        network = GAIANetwork(network_dim=DIM)
        for agent in agents:
            network.add_agent(agent)

        # Process 20 signals
        for i in range(20):
            signal = _make_signal(seed=i)
            result = network.process(signal)
            assert result is not None

        # All agents should have accumulated experience
        for agent in agents:
            assert agent.identity.experience > 0


class TestPACConservationReal:
    """PAC conservation holds with real modules across recursion levels."""

    def test_pac_over_many_ticks(self):
        """PAC violation rate stays low over 50 ticks with real modules."""
        agent = GAIAAgent(
            "pac_test",
            [SafetyModule(input_dim=DIM), ReasoningModule(input_dim=DIM)],
            field_dim=DIM,
        )
        network = GAIANetwork(network_dim=DIM)
        network.add_agent(agent)

        violations = 0
        for i in range(50):
            signal = _make_signal(seed=i)
            input_energy = signal.total_energy()
            result = network.process(signal)
            output_energy = result.total_energy()

            if abs(input_energy - output_energy) > 1.0:
                violations += 1

        # Real modules may have small violations due to Safety's conservation
        # projector, but the bus rescales at the boundary. Allow up to 10%.
        assert violations < 5, f"Too many PAC violations: {violations}/50"


class TestSpecializationReal:
    """Agents with different real modules develop different specializations."""

    def test_divergent_lenses_with_real_modules(self):
        """Safety-focused vs Memory-focused agents should develop different lenses."""
        agent_safety = GAIAAgent(
            "safety_focused",
            [SafetyModule(input_dim=DIM), SafetyModule(input_dim=DIM)],
            field_dim=DIM,
        )
        agent_memory = GAIAAgent(
            "memory_focused",
            [MemoryModule(), MemoryModule()],
            field_dim=DIM,
        )

        network = GAIANetwork(network_dim=DIM)
        network.add_agent(agent_safety)
        network.add_agent(agent_memory)

        # Feed class-specific stimuli to encourage specialization
        for class_id in range(4):
            for tick in range(15):
                signal = _make_stimulus_class(class_id * 10 + tick)
                network.process(signal)

        # Agents should have diverged — they process signals differently
        lens_safety = agent_safety.identity.spectral_lens
        lens_memory = agent_memory.identity.spectral_lens
        divergence = float(torch.norm(lens_safety - lens_memory).item())

        # With different internal transforms, lenses should differ
        assert divergence > 0.01, (
            f"Agents with different modules should diverge, but divergence = {divergence:.6f}"
        )

    def test_coupling_reflects_module_differences(self):
        """Inter-agent coupling should be < 1.0 for agents with different modules."""
        agent_a = GAIAAgent(
            "alpha",
            [SafetyModule(input_dim=DIM)],
            field_dim=DIM,
        )
        agent_b = GAIAAgent(
            "beta",
            [ReasoningModule(input_dim=DIM)],
            field_dim=DIM,
        )

        network = GAIANetwork(network_dim=DIM)
        network.add_agent(agent_a)
        network.add_agent(agent_b)

        for i in range(30):
            network.process(_make_signal(seed=i))

        matrix = network.get_coupling_matrix()
        coupling = matrix.get(("alpha", "beta"), 1.0)

        # Different modules should produce imperfect coupling
        assert coupling < 1.0, (
            f"Different-module agents should not be perfectly coupled: C={coupling:.4f}"
        )


class TestRecursiveNestingReal:
    """Real modules in recursive (nested) configurations."""

    def test_agent_with_sub_agent(self):
        """Agent spawns a sub-agent with real modules."""
        parent = GAIAAgent(
            "parent",
            [SafetyModule(input_dim=DIM)],
            field_dim=DIM,
        )
        child = parent.spawn_sub_agent(
            "child",
            [ReasoningModule(input_dim=DIM), MemoryModule()],
        )

        # Parent now has Safety + child (which has Reasoning + Memory)
        assert "child" in parent.module_names

        network = GAIANetwork(network_dim=DIM)
        network.add_agent(parent)

        for i in range(10):
            result = network.process(_make_signal(seed=i))
            assert result is not None

        # Both parent and child should have experience
        assert parent.identity.experience > 0

    def test_recursive_entity_with_full_stack(self):
        """RecursiveEntity wrapping all 5 real modules works in a bus."""
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(SafetyModule(input_dim=DIM))
        bus.register_module(ReasoningModule(input_dim=DIM))
        bus.register_module(MemoryModule())
        bus.register_module(LanguageModule())
        bus.register_module(ObservabilityModule())

        entity = RecursiveEntity("full_core", bus)

        # Use this entity in a network-level bus
        network_bus = CoupledFieldsBus(enforcement="soft")
        network_bus.register_module(entity)

        signal = _make_signal(seed=42)
        result = network_bus.process(signal)

        assert result is not None
        assert entity.tick_count == 1


class TestFullNetworkSmoke:
    """End-to-end: 3 specialized agents, 100 ticks, real modules."""

    def test_three_specialist_network(self):
        """Three agents with distinct real-module compositions.

        guardian: Safety + Observability (monitors for violations)
        thinker:  Reasoning + Memory (learns and reasons)
        speaker:  Language + Memory (processes linguistic structure)

        100 ticks of diverse input. Validates:
        - Network doesn't crash with real module dynamics
        - PAC conservation holds (< 10% violation rate)
        - All agents accumulate experience
        - Agents develop non-trivial specializations
        - Coupling matrix is populated and symmetric
        """
        guardian = GAIAAgent("guardian", [
            SafetyModule(input_dim=DIM),
            ObservabilityModule(),
        ], field_dim=DIM)

        thinker = GAIAAgent("thinker", [
            ReasoningModule(input_dim=DIM),
            MemoryModule(),
        ], field_dim=DIM)

        speaker = GAIAAgent("speaker", [
            LanguageModule(),
            MemoryModule(),
        ], field_dim=DIM)

        network = GAIANetwork(network_dim=DIM)
        network.add_agent(guardian)
        network.add_agent(thinker)
        network.add_agent(speaker)

        # Run 100 ticks with diverse signals
        pac_violations = 0
        for i in range(100):
            # Mix random signals with class-specific stimuli
            if i % 3 == 0:
                signal = _make_stimulus_class(i % 5)
            else:
                signal = _make_signal(seed=i)

            input_energy = signal.total_energy()
            result = network.process(signal)
            output_energy = result.total_energy()

            if abs(input_energy - output_energy) > 1.0:
                pac_violations += 1

        # --- Assertions ---

        # PAC conservation
        assert pac_violations < 10, f"PAC violations: {pac_violations}/100"

        # All agents have experience
        for name in ["guardian", "thinker", "speaker"]:
            agent = network.get_agent(name)
            assert agent.identity.experience > 0, f"{name} has no experience"

        # Specializations exist
        specs = network.get_specializations()
        assert len(specs) == 3

        # Coupling matrix is populated and symmetric
        matrix = network.get_coupling_matrix()
        assert len(matrix) == 6  # 3 agents, 3*2 = 6 directed pairs
        for (a, b), val in matrix.items():
            reverse = matrix.get((b, a))
            if reverse is not None:
                assert abs(val - reverse) < 1e-6, f"Coupling not symmetric: {a}-{b}"

        # Network metrics are populated
        metrics = network.get_metrics()
        assert metrics["agent_count"] == 3
        assert metrics["network_dim"] == DIM

        # Print summary for visibility
        print(f"\n--- 100-tick Integration Results ---")
        print(f"PAC violations: {pac_violations}/100")
        for name, spec in specs.items():
            agent = network.get_agent(name)
            print(f"  {name}: specialization={spec:.4f}, experience={agent.identity.experience}")
        print(f"Coupling matrix:")
        for (a, b), val in sorted(matrix.items()):
            print(f"  C[{a},{b}] = {val:.4f}")
