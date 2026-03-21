"""Integration tests — MemoryModule through the ConservationBus."""

from __future__ import annotations

import pytest
import torch

from gaia.core.bus import ConservationBus
from gaia.core.types import SECPhase
from gaia.modules.memory import MemoryModule
from gaia.modules.reasoning import ReasoningModule
from gaia.modules.safety import SafetyModule

from tests.conftest import IdentityModule, make_field_state


class TestMemoryThroughBus:
    """MemoryModule registered with ConservationBus, end-to-end."""

    def test_memory_passes_hard_enforcement(self):
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(MemoryModule())
        state = make_field_state(tensor=torch.ones(10), entropy=1.0)
        result = bus.process(state)
        assert result.total_energy() == pytest.approx(state.total_energy(), rel=1e-3)
        assert "memory" in result.provenance
        assert len(bus.violation_log) == 0

    def test_memory_with_identity(self):
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(MemoryModule())
        bus.register_module(IdentityModule())
        state = make_field_state(tensor=torch.ones(10), entropy=1.0)
        result = bus.process(state)
        assert "memory" in result.provenance
        assert "identity" in result.provenance

    def test_memory_with_safety_and_reasoning(self):
        """Three real modules composing through the bus."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=10))
        bus.register_module(ReasoningModule(input_dim=10))
        bus.register_module(MemoryModule())
        state = make_field_state(tensor=torch.ones(10), entropy=1.0)
        result = bus.process(state)
        assert "safety" in result.provenance
        assert "reasoning" in result.provenance
        assert "memory" in result.provenance
        assert len(bus.violation_log) == 0

    def test_memory_accumulates_across_bus_calls(self):
        memory = MemoryModule()
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(memory)
        for _ in range(5):
            state = make_field_state(tensor=torch.randn(10).abs() + 0.1, entropy=1.0)
            bus.process(state)
        assert memory.tree.size >= 5

    def test_memory_metrics_after_bus(self):
        memory = MemoryModule()
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(memory)
        bus.process(make_field_state(tensor=torch.ones(10), entropy=1.0))
        assert memory.metrics is not None
        assert memory.metrics.n_nodes >= 1

    def test_memory_multiple_inputs_no_violations(self):
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(MemoryModule())
        for _ in range(15):
            state = make_field_state(tensor=torch.randn(10).abs() + 0.1, entropy=1.0)
            result = bus.process(state)
            assert result.total_energy() == pytest.approx(state.total_energy(), rel=1e-3)
        assert len(bus.violation_log) == 0
