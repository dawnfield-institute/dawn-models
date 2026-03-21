"""Integration tests — ReasoningModule through the ConservationBus."""

from __future__ import annotations

import pytest
import torch

from gaia.core.bus import ConservationBus
from gaia.core.types import SECPhase
from gaia.modules.reasoning import ReasoningModule
from gaia.modules.safety import SafetyModule

from tests.conftest import IdentityModule, make_field_state


class TestReasoningThroughBus:
    """ReasoningModule registered with ConservationBus, end-to-end."""

    def test_reasoning_passes_hard_enforcement(self):
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(ReasoningModule(input_dim=10))
        state = make_field_state(tensor=torch.ones(10), entropy=1.0)
        result = bus.process(state)
        assert result.total_energy() == pytest.approx(state.total_energy(), rel=1e-3)
        assert "reasoning" in result.provenance
        assert len(bus.violation_log) == 0

    def test_reasoning_with_identity(self):
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(ReasoningModule(input_dim=10))
        bus.register_module(IdentityModule())
        state = make_field_state(tensor=torch.ones(10), entropy=1.0)
        result = bus.process(state)
        assert "reasoning" in result.provenance
        assert "identity" in result.provenance

    def test_reasoning_with_safety(self):
        """Two real modules through the bus sequentially."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=10))
        bus.register_module(ReasoningModule(input_dim=10))
        state = make_field_state(tensor=torch.ones(10), entropy=1.0)
        result = bus.process(state)
        assert "safety" in result.provenance
        assert "reasoning" in result.provenance
        assert len(bus.violation_log) == 0

    def test_reasoning_phase_routing(self):
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(ReasoningModule(input_dim=10), phases=[SECPhase.ORDERED])

        ordered = make_field_state(tensor=torch.ones(10), entropy=1.0)
        result = bus.process(ordered)
        assert "reasoning" in result.provenance

        chaotic = make_field_state(tensor=torch.ones(10), entropy=5.0)
        result2 = bus.process(chaotic)
        assert "reasoning" not in result2.provenance

    def test_reasoning_multiple_inputs(self):
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(ReasoningModule(input_dim=10))
        for _ in range(10):
            state = make_field_state(tensor=torch.randn(10).abs() + 0.1, entropy=1.0)
            result = bus.process(state)
            assert result.total_energy() == pytest.approx(state.total_energy(), rel=1e-3)
        assert len(bus.violation_log) == 0

    def test_reasoning_metrics_after_bus(self):
        reasoning = ReasoningModule(input_dim=10)
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(reasoning)
        bus.process(make_field_state(tensor=torch.ones(10), entropy=1.0))
        assert reasoning.metrics is not None
        assert reasoning.metrics.phi_frequency > 0
