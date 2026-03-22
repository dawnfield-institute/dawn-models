"""Integration tests for the Language Module through ConservationBus.

Tests composition with other modules, PAC enforcement at bus boundaries,
and multi-step accumulation.
"""

import torch
import pytest

from gaia.core.bus import ConservationBus
from gaia.core.types import FieldState, SECPhase
from gaia.modules.language import LanguageModule, EmbeddingStore
from gaia.modules.memory import MemoryModule
from gaia.modules.observability import ObservabilityModule
from gaia.modules.reasoning import ReasoningModule
from gaia.modules.safety import SafetyModule

from tests.conftest import IdentityModule, make_field_state


def _make_state(dim: int = 16, entropy: float = 1.0) -> FieldState:
    return FieldState(
        tensor=torch.randn(dim).abs() + 0.1,
        entropy=entropy,
    )


class TestLanguageBusIntegration:

    def test_language_passes_hard_enforcement(self):
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(LanguageModule())

        for _ in range(10):
            state = _make_state()
            result = bus.process(state)
            assert torch.isfinite(result.tensor).all()
        assert len(bus.violation_log) == 0

    def test_language_with_identity(self):
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(LanguageModule())
        bus.register_module(IdentityModule())

        state = _make_state()
        result = bus.process(state)
        assert "language" in result.provenance
        assert "identity" in result.provenance

    def test_language_with_safety_and_reasoning(self):
        dim = 16
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=dim))
        bus.register_module(ReasoningModule(input_dim=dim))
        bus.register_module(LanguageModule())

        for _ in range(5):
            state = _make_state(dim=dim)
            result = bus.process(state)
            assert torch.isfinite(result.tensor).all()
        assert len(bus.violation_log) == 0

    def test_language_with_memory(self):
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(LanguageModule())
        bus.register_module(MemoryModule())

        for _ in range(10):
            state = _make_state()
            result = bus.process(state)
            assert torch.isfinite(result.tensor).all()
        assert len(bus.violation_log) == 0

    def test_full_five_module_stack(self):
        """All 5 modules through bus — the real integration test."""
        dim = 16
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=dim))
        bus.register_module(ReasoningModule(input_dim=dim))
        bus.register_module(MemoryModule())
        bus.register_module(ObservabilityModule())
        bus.register_module(LanguageModule())

        for _ in range(20):
            state = _make_state(dim=dim)
            result = bus.process(state)
            assert torch.isfinite(result.tensor).all()
            assert len(result.provenance) == 5

        assert len(bus.violation_log) == 0

    def test_accumulates_across_bus_calls(self):
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        lang = LanguageModule()
        bus.register_module(lang)

        for _ in range(15):
            state = _make_state()
            bus.process(state)

        assert lang.counter.stats.total_transitions > 0
        assert lang.metrics.step_count == 15

    def test_metrics_accessible_after_bus(self):
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        lang = LanguageModule()
        bus.register_module(lang)

        state = _make_state()
        bus.process(state)

        m = lang.metrics
        assert m is not None
        assert m.step_count == 1

    def test_language_ordering_before_safety(self):
        dim = 16
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(LanguageModule())
        bus.register_module(SafetyModule(input_dim=dim))

        state = _make_state(dim=dim)
        result = bus.process(state)
        assert result.provenance == ["language", "safety"]
        assert len(bus.violation_log) == 0

    def test_high_entropy_input(self):
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(LanguageModule())

        state = _make_state(entropy=5.0)  # CHAOTIC phase
        result = bus.process(state)
        assert torch.isfinite(result.tensor).all()

    def test_tight_tolerance(self):
        """Even at 1e-10, language module (quantize + rescale) should conserve."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-10)
        lang = LanguageModule()
        bus.register_module(lang)

        # First call — no blending (no learned transitions yet)
        state = FieldState(tensor=torch.ones(10) * 2.0, entropy=1.0)
        result = bus.process(state)
        ie = state.total_energy()
        oe = result.total_energy()
        # Should be exact (no blending on first call → pass-through + rescale)
        assert abs(ie - oe) < 1e-10
