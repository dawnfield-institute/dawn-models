"""Tests for the ConservationBus."""

from __future__ import annotations

import logging

import pytest
import torch

from gaia.core.bus import ConservationBus
from gaia.core.exceptions import ConservationViolation, ModuleRegistrationError
from gaia.core.types import FieldState, RBFBalance, SECPhase

from tests.conftest import IdentityModule, ScalingModule, UnhealthyModule, make_field_state


# --- Registration ---


class TestRegistration:
    def test_register_valid_module(self):
        bus = ConservationBus()
        module = IdentityModule()
        bus.register_module(module)
        assert "identity" in bus.get_metrics()["module_names"]

    def test_register_invalid_object(self):
        bus = ConservationBus()
        with pytest.raises(ModuleRegistrationError):
            bus.register_module("not a module")

    def test_register_with_specific_phases(self):
        bus = ConservationBus()
        module = IdentityModule()
        bus.register_module(module, phases=[SECPhase.ORDERED, SECPhase.CHAOTIC])
        metrics = bus.get_metrics()
        assert metrics["modules_registered"] == 1


# --- Identity / passthrough ---


class TestIdentityProcessing:
    def test_identity_module_conserves(self):
        bus = ConservationBus()
        bus.register_module(IdentityModule())
        state = make_field_state(entropy=1.0)
        result = bus.process(state)
        assert result.total_energy() == pytest.approx(state.total_energy())

    def test_no_modules_passthrough(self):
        """Bus with no modules returns cloned input unchanged."""
        bus = ConservationBus()
        state = make_field_state()
        result = bus.process(state)
        assert result.total_energy() == pytest.approx(state.total_energy())
        assert result is not state  # cloned

    def test_no_matching_modules_passthrough(self):
        """Modules registered for different phase than input entropy."""
        bus = ConservationBus()
        module = IdentityModule()
        # Register only for CHAOTIC (entropy >= 4.0)
        bus.register_module(module, phases=[SECPhase.CHAOTIC])
        # Input has entropy=1.0 → ORDERED phase, no match
        state = make_field_state(entropy=1.0)
        result = bus.process(state)
        assert "identity" not in result.provenance


# --- Violation detection ---


class TestViolationDetection:
    def test_hard_enforcement_raises(self):
        bus = ConservationBus(enforcement="hard")
        bus.register_module(ScalingModule(factor=2.0))
        state = make_field_state(entropy=1.0)
        with pytest.raises(ConservationViolation) as exc_info:
            bus.process(state)
        assert "scaler(2.0)" in str(exc_info.value)

    def test_soft_enforcement_logs(self, caplog):
        bus = ConservationBus(enforcement="soft")
        bus.register_module(ScalingModule(factor=2.0))
        state = make_field_state(entropy=1.0)
        with caplog.at_level(logging.WARNING):
            result = bus.process(state)
        # Should complete without raising
        assert result.total_energy() == pytest.approx(20.0)  # 10 * 2.0
        assert len(bus.violation_log) == 1

    def test_monitor_enforcement_logs(self, caplog):
        bus = ConservationBus(enforcement="monitor")
        bus.register_module(ScalingModule(factor=2.0))
        state = make_field_state(entropy=1.0)
        with caplog.at_level(logging.INFO):
            result = bus.process(state)
        assert result.total_energy() == pytest.approx(20.0)
        assert len(bus.violation_log) == 1

    def test_invalid_enforcement_mode(self):
        with pytest.raises(ValueError, match="enforcement must be"):
            ConservationBus(enforcement="yolo")


# --- Tolerance ---


class TestTolerance:
    def test_within_tolerance_passes(self):
        """A tiny scaling factor within tolerance should not trigger violation."""
        bus = ConservationBus(enforcement="hard", tolerance=0.1)
        # Factor of 1.001 on tensor sum=10 → residual = 0.01 < 0.1
        bus.register_module(ScalingModule(factor=1.001))
        state = make_field_state(entropy=1.0)
        result = bus.process(state)  # should not raise
        assert len(bus.violation_log) == 0

    def test_exceeds_tolerance_fails(self):
        bus = ConservationBus(enforcement="hard", tolerance=1e-6)
        bus.register_module(ScalingModule(factor=1.1))
        state = make_field_state(entropy=1.0)
        with pytest.raises(ConservationViolation):
            bus.process(state)


# --- Violation log ---


class TestViolationLog:
    def test_violation_log_accumulates(self):
        bus = ConservationBus(enforcement="soft")
        bus.register_module(ScalingModule(factor=2.0))
        for _ in range(3):
            bus.process(make_field_state(entropy=1.0))
        assert len(bus.violation_log) == 3

    def test_violation_log_is_copy(self):
        bus = ConservationBus(enforcement="soft")
        bus.register_module(ScalingModule(factor=2.0))
        bus.process(make_field_state(entropy=1.0))
        log = bus.violation_log
        log.clear()
        assert len(bus.violation_log) == 1  # original unaffected


# --- Sequential multi-module ---


class TestMultiModule:
    def test_sequential_dispatch_order(self):
        """Modules process sequentially — output of first feeds second."""
        bus = ConservationBus(enforcement="hard")
        bus.register_module(IdentityModule())

        class AppenderModule:
            @property
            def name(self) -> str:
                return "appender"

            def process(self, field_state: FieldState) -> FieldState:
                result = field_state.clone()
                result.provenance.append(self.name)
                return result

            def phase(self) -> SECPhase:
                return SECPhase.ORDERED

            def health(self) -> RBFBalance:
                return RBFBalance.compute(1.0, 1.0, 0.0)

        bus.register_module(AppenderModule())
        state = make_field_state(entropy=1.0)
        result = bus.process(state)
        assert result.provenance == ["identity", "appender"]


# --- RBF suppression ---


class TestRBFSuppression:
    def test_unhealthy_module_suppressed(self):
        """Module with negative RBF balance is suppressed (threshold=0)."""
        bus = ConservationBus(rbf_suppression_threshold=0.0)
        bus.register_module(UnhealthyModule())
        state = make_field_state(entropy=1.0)
        result = bus.process(state)
        # Unhealthy module has balance < 0, should be suppressed
        assert "unhealthy" not in result.provenance

    def test_healthy_module_not_suppressed(self):
        bus = ConservationBus(rbf_suppression_threshold=0.0)
        bus.register_module(IdentityModule())
        state = make_field_state(entropy=1.0)
        result = bus.process(state)
        assert "identity" in result.provenance


# --- SEC phase classification ---


class TestSECPhaseInBus:
    def test_phase_set_on_output(self):
        bus = ConservationBus()
        bus.register_module(IdentityModule())
        state = make_field_state(entropy=1.0)
        result = bus.process(state)
        assert result.phase == SECPhase.ORDERED

    def test_chaotic_phase(self):
        bus = ConservationBus()
        bus.register_module(IdentityModule())
        state = make_field_state(entropy=5.0)
        result = bus.process(state)
        assert result.phase == SECPhase.CHAOTIC


# --- Metrics ---


class TestMetrics:
    def test_metrics_content(self):
        bus = ConservationBus(enforcement="soft", tolerance=0.01)
        bus.register_module(IdentityModule())
        bus.register_module(ScalingModule(factor=2.0))
        metrics = bus.get_metrics()
        assert metrics["modules_registered"] == 2
        assert metrics["enforcement"] == "soft"
        assert metrics["tolerance"] == 0.01
        assert metrics["total_violations"] == 0

    def test_metrics_after_violation(self):
        bus = ConservationBus(enforcement="soft")
        bus.register_module(ScalingModule(factor=2.0))
        bus.process(make_field_state(entropy=1.0))
        assert bus.get_metrics()["total_violations"] == 1
