"""Integration tests — ObservabilityModule through the ConservationBus."""

from __future__ import annotations

import pytest
import torch

from gaia.core.bus import ConservationBus
from gaia.core.types import SECPhase
from gaia.modules.memory import MemoryModule
from gaia.modules.observability import ObservabilityModule
from gaia.modules.reasoning import ReasoningModule
from gaia.modules.safety import SafetyModule

from tests.conftest import IdentityModule, make_field_state


class TestObservabilityThroughBus:
    """ObservabilityModule registered with ConservationBus, end-to-end."""

    def test_observability_passes_hard_enforcement(self):
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(ObservabilityModule())
        state = make_field_state(tensor=torch.ones(10), entropy=1.0)
        result = bus.process(state)
        assert result.total_energy() == pytest.approx(state.total_energy(), abs=1e-6)
        assert "observability" in result.provenance
        assert len(bus.violation_log) == 0

    def test_observability_with_identity(self):
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(ObservabilityModule())
        bus.register_module(IdentityModule())
        state = make_field_state(tensor=torch.ones(10), entropy=1.0)
        result = bus.process(state)
        assert "observability" in result.provenance
        assert "identity" in result.provenance

    def test_full_stack_with_observability(self):
        """All 4 modules + observability composing through the bus."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=10))
        bus.register_module(ReasoningModule(input_dim=10))
        bus.register_module(MemoryModule())
        bus.register_module(ObservabilityModule())
        state = make_field_state(tensor=torch.ones(10), entropy=1.0)
        result = bus.process(state)
        assert "safety" in result.provenance
        assert "reasoning" in result.provenance
        assert "memory" in result.provenance
        assert "observability" in result.provenance
        assert len(bus.violation_log) == 0

    def test_metrics_accessible_after_bus(self):
        obs = ObservabilityModule()
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(obs)
        bus.process(make_field_state(tensor=torch.ones(10), entropy=1.0))
        assert obs.metrics is not None
        assert obs.metrics.scbf.entropy_collapse >= 0

    def test_multiple_inputs_no_violations(self):
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(ObservabilityModule())
        for _ in range(20):
            state = make_field_state(tensor=torch.randn(10).abs() + 0.1, entropy=1.0)
            result = bus.process(state)
            assert result.total_energy() == pytest.approx(state.total_energy(), abs=1e-6)
        assert len(bus.violation_log) == 0

    def test_observability_phase_reflects_qbe(self):
        obs = ObservabilityModule()
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(obs)
        bus.process(make_field_state(tensor=torch.ones(10), entropy=1.0))
        phase = obs.phase()
        assert isinstance(phase, SECPhase)

    def test_tight_tolerance_zero_violations(self):
        """Observability is identity — should have zero violations even at 1e-10."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-10)
        bus.register_module(ObservabilityModule())
        for _ in range(10):
            state = make_field_state(tensor=torch.randn(10).abs() + 0.1, entropy=1.0)
            bus.process(state)
        assert len(bus.violation_log) == 0
