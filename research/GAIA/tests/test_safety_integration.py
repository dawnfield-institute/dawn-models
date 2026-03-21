"""Integration tests — SafetyModule through the ConservationBus."""

from __future__ import annotations

import pytest
import torch

from gaia.core.bus import ConservationBus
from gaia.core.types import SECPhase
from gaia.modules.safety import SafetyModule

from tests.conftest import IdentityModule, make_field_state


class TestSafetyThroughBus:
    """SafetyModule registered with ConservationBus, end-to-end."""

    def test_safety_passes_hard_enforcement(self):
        """SafetyModule is PAC-conserving — should pass hard enforcement."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=10))
        state = make_field_state(tensor=torch.ones(10), entropy=1.0)
        result = bus.process(state)
        assert result.total_energy() == pytest.approx(state.total_energy(), rel=1e-3)
        assert "safety" in result.provenance
        assert len(bus.violation_log) == 0

    def test_safety_with_identity(self):
        """Safety + Identity modules process sequentially."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=10))
        bus.register_module(IdentityModule())
        state = make_field_state(tensor=torch.ones(10), entropy=1.0)
        result = bus.process(state)
        assert "safety" in result.provenance
        assert "identity" in result.provenance

    def test_safety_phase_routing(self):
        """SafetyModule registered for specific phases only activates for those."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=10), phases=[SECPhase.ORDERED])

        # Ordered entropy — should activate
        ordered_state = make_field_state(tensor=torch.ones(10), entropy=1.0)
        result = bus.process(ordered_state)
        assert "safety" in result.provenance

        # Chaotic entropy — should NOT activate
        chaotic_state = make_field_state(tensor=torch.ones(10), entropy=5.0)
        result2 = bus.process(chaotic_state)
        assert "safety" not in result2.provenance

    def test_safety_metrics_available_after_bus(self):
        """Can access SafetyModule metrics after bus processing."""
        safety = SafetyModule(input_dim=10)
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(safety)
        state = make_field_state(tensor=torch.ones(10), entropy=1.0)
        bus.process(state)
        assert safety.metrics is not None
        assert len(safety.metrics.head_entropies) > 0

    def test_safety_multiple_inputs(self):
        """Safety handles multiple different inputs without accumulating violations."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=10))
        for _ in range(10):
            state = make_field_state(tensor=torch.randn(10).abs() + 0.1, entropy=1.0)
            result = bus.process(state)
            assert result.total_energy() == pytest.approx(state.total_energy(), rel=1e-3)
        assert len(bus.violation_log) == 0

    def test_safety_with_negative_tensor(self):
        """Safety handles tensors with negative values."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=10))
        state = make_field_state(tensor=torch.randn(10), entropy=1.0)
        result = bus.process(state)
        assert result.total_energy() == pytest.approx(state.total_energy(), rel=1e-3)

    def test_safety_soft_mode_bus_hard_mode(self):
        """Internal soft conservation + external hard enforcement."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        safety = SafetyModule(input_dim=10, conservation_mode="soft")
        bus.register_module(safety)
        state = make_field_state(tensor=torch.ones(10), entropy=1.0)
        result = bus.process(state)
        assert len(bus.violation_log) == 0
