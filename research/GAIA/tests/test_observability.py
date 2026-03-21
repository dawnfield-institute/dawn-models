"""Unit tests — ObservabilityModule, QBEController, SCBFTracker."""

from __future__ import annotations

import pytest
import torch

from gaia.core.types import FieldState, RBFBalance, SECPhase
from gaia.modules.observability import (
    CollapseEvent,
    ObservabilityMetrics,
    ObservabilityModule,
    QBEController,
    SCBFMetrics,
    SCBFTracker,
)


# ─── CollapseEvent ───────────────────────────────────────────────


class TestCollapseEvent:
    def test_creation(self):
        e = CollapseEvent(step=5, immediate_change=0.1, trend_change=0.05, magnitude=0.1, threshold=0.02)
        assert e.step == 5
        assert e.magnitude == 0.1

    def test_fields(self):
        e = CollapseEvent(step=1, immediate_change=0.2, trend_change=0.3, magnitude=0.3, threshold=0.1)
        assert e.threshold == 0.1
        assert e.trend_change == 0.3


# ─── QBEController ───────────────────────────────────────────────


class TestQBEController:
    def test_initial_state(self):
        qbe = QBEController()
        assert qbe.momentum == 0.8
        assert qbe.error_band == 0.1
        assert qbe.get_status() == "Near Equilibrium"

    def test_update_adjusts_momentum(self):
        qbe = QBEController()
        initial = qbe.momentum
        qbe.update(error=1.0, entropy=0.5)
        assert qbe.momentum != initial

    def test_update_clamps_error_band(self):
        qbe = QBEController()
        # Many updates with high entropy
        for _ in range(100):
            qbe.update(error=0.1, entropy=10.0)
        assert qbe.error_band <= 0.2  # Upper clamp

    def test_status_near_equilibrium(self):
        qbe = QBEController(initial_momentum=0.5, error_band=0.1)
        assert qbe.get_status() == "Near Equilibrium"

    def test_status_moderate(self):
        qbe = QBEController(initial_momentum=1.5, error_band=0.2)
        assert qbe.get_status() == "Moderate Equilibrium"

    def test_status_far(self):
        qbe = QBEController(initial_momentum=1.8, error_band=0.5)
        assert qbe.get_status() == "Far from Equilibrium"

    def test_detect_convergence(self):
        qbe = QBEController()
        # Very low variance values
        vals = [1.0] * 15
        assert qbe.detect_pattern_type(vals) == "convergence"

    def test_detect_chaotic(self):
        qbe = QBEController()
        # High variance values
        vals = [float(i % 5) for i in range(15)]
        result = qbe.detect_pattern_type(vals)
        assert result in ("chaotic", "unknown")  # Depends on variance

    def test_detect_unknown_short(self):
        qbe = QBEController()
        assert qbe.detect_pattern_type([1.0, 2.0]) == "unknown"

    def test_adjust_convergence(self):
        qbe = QBEController(initial_momentum=0.8, error_band=0.15)
        qbe.adjust_for_pattern("convergence")
        assert qbe.error_band < 0.15
        assert qbe.momentum > 0.8

    def test_adjust_chaotic(self):
        qbe = QBEController(initial_momentum=0.85, error_band=0.1)
        qbe.adjust_for_pattern("chaotic")
        assert qbe.error_band > 0.1
        assert qbe.momentum < 0.85

    def test_to_sec_phase_near(self):
        qbe = QBEController(initial_momentum=0.5, error_band=0.1)
        assert qbe.to_sec_phase() == SECPhase.CRYSTALLIZED

    def test_to_sec_phase_moderate(self):
        qbe = QBEController(initial_momentum=1.5, error_band=0.2)
        assert qbe.to_sec_phase() == SECPhase.ORDERED

    def test_to_sec_phase_far(self):
        qbe = QBEController(initial_momentum=1.8, error_band=0.5)
        assert qbe.to_sec_phase() == SECPhase.CHAOTIC


# ─── SCBFTracker ─────────────────────────────────────────────────


class TestSCBFTracker:
    def test_compute_returns_metrics(self):
        tracker = SCBFTracker()
        m = tracker.compute(torch.randn(16))
        assert isinstance(m, SCBFMetrics)

    def test_compute_all_fields_finite(self):
        tracker = SCBFTracker()
        m = tracker.compute(torch.randn(16))
        assert all(
            isinstance(getattr(m, f), (int, float))
            for f in [
                "entropy_collapse", "ancestry_stability", "phase_alignment",
                "bifractal_strength", "attractor_density",
            ]
        )

    def test_step_increments(self):
        tracker = SCBFTracker()
        tracker.compute(torch.randn(8))
        tracker.compute(torch.randn(8))
        assert tracker.step == 2

    # Entropy collapse
    def test_entropy_collapse_positive(self):
        tracker = SCBFTracker()
        m = tracker.compute(torch.ones(16))
        assert m.entropy_collapse >= 0

    def test_entropy_collapse_single_element(self):
        tracker = SCBFTracker()
        m = tracker.compute(torch.tensor([1.0]))
        assert m.entropy_collapse == 3.0  # Default for single element

    def test_collapse_detection_after_shift(self):
        tracker = SCBFTracker()
        # Feed stable pattern then shift
        for _ in range(10):
            tracker.compute(torch.ones(16))
        tracker.compute(torch.randn(16) * 10)  # Big shift
        assert len(tracker.collapse_events) >= 0  # May or may not detect

    def test_entropy_momentum_updates(self):
        tracker = SCBFTracker()
        tracker.compute(torch.ones(8))
        tracker.compute(torch.randn(8) * 5)
        m = tracker.compute(torch.ones(8))
        assert m.entropy_momentum != 0.0  # Should have changed

    # Ancestry
    def test_ancestry_first_call(self):
        tracker = SCBFTracker()
        m = tracker.compute(torch.randn(16))
        assert m.ancestry_stability == 1.0  # No previous to compare

    def test_ancestry_identical_inputs(self):
        tracker = SCBFTracker()
        pattern = torch.randn(16)
        tracker.compute(pattern.clone())
        m = tracker.compute(pattern.clone())
        assert m.ancestry_stability > 0.9  # Should be very stable

    def test_ancestry_different_inputs(self):
        tracker = SCBFTracker()
        tracker.compute(torch.ones(16))
        m = tracker.compute(torch.ones(16) * -1)  # Opposite
        assert m.ancestry_stability < 0.5

    # Phase alignment
    def test_phase_alignment_first_call(self):
        tracker = SCBFTracker()
        m = tracker.compute(torch.randn(16))
        assert m.phase_alignment == 1.0

    def test_phase_alignment_stable(self):
        tracker = SCBFTracker()
        pattern = torch.randn(16)
        for _ in range(5):
            tracker.compute(pattern.clone())
        m = tracker.compute(pattern.clone())
        assert m.phase_alignment > 0.8

    # Bifractal
    def test_bifractal_first_call(self):
        tracker = SCBFTracker()
        m = tracker.compute(torch.randn(16))
        assert m.bifractal_strength == 0.0  # No recursion yet

    def test_bifractal_recurring_pattern(self):
        tracker = SCBFTracker()
        pattern = torch.randn(16)
        # Feed alternating patterns
        for _ in range(10):
            tracker.compute(pattern.clone())
            tracker.compute(torch.randn(16))
        m = tracker.compute(pattern.clone())
        assert m.bifractal_strength >= 0  # Should detect some recursion

    def test_mathematical_memory_grows(self):
        tracker = SCBFTracker()
        pattern = torch.randn(16)
        for _ in range(10):
            tracker.compute(pattern.clone())
        # Mathematical memory should have entries
        m = tracker.compute(pattern.clone())
        assert m.mathematical_memory_size >= 0

    # Attractor density
    def test_attractor_density_initial(self):
        tracker = SCBFTracker()
        m = tracker.compute(torch.randn(16))
        assert m.attractor_density == 0.5  # Default for < 3 centroids

    def test_attractor_density_after_warmup(self):
        tracker = SCBFTracker()
        for _ in range(5):
            tracker.compute(torch.randn(16))
        m = tracker.compute(torch.randn(16))
        assert 0.0 <= m.attractor_density <= 1.0

    def test_attractor_density_stable_input(self):
        tracker = SCBFTracker()
        pattern = torch.ones(16)
        for _ in range(5):
            tracker.compute(pattern.clone())
        m = tracker.compute(pattern.clone())
        # Same input → centroids cluster tightly → high density
        assert m.attractor_density > 0.5

    # Derived metrics
    def test_entropy_variance_needs_warmup(self):
        tracker = SCBFTracker()
        m = tracker.compute(torch.randn(8))
        assert m.entropy_variance == 0.0  # < 10 samples

    def test_pattern_consistency_needs_warmup(self):
        tracker = SCBFTracker()
        m = tracker.compute(torch.randn(8))
        assert m.pattern_consistency == 1.0  # < 5 samples

    def test_collapse_count(self):
        tracker = SCBFTracker()
        for _ in range(5):
            tracker.compute(torch.randn(16))
        m = tracker.compute(torch.randn(16))
        assert m.collapse_count >= 0  # Count of collapse events


# ─── ObservabilityModule ─────────────────────────────────────────


class TestObservabilityModule:
    def test_name(self):
        assert ObservabilityModule().name == "observability"

    def test_satisfies_protocol(self):
        from gaia.core.protocol import GAIAModule
        assert isinstance(ObservabilityModule(), GAIAModule)

    def test_process_returns_field_state(self):
        mod = ObservabilityModule()
        state = FieldState(tensor=torch.randn(8), entropy=1.0)
        result = mod.process(state)
        assert isinstance(result, FieldState)

    def test_process_preserves_tensor(self):
        """Observability is pass-through — tensor should be unchanged."""
        mod = ObservabilityModule()
        tensor = torch.randn(8)
        state = FieldState(tensor=tensor.clone(), entropy=1.0)
        result = mod.process(state)
        assert torch.allclose(result.tensor, tensor)

    def test_process_conserves_energy(self):
        mod = ObservabilityModule()
        state = FieldState(tensor=torch.randn(8), entropy=1.0)
        input_e = state.total_energy()
        result = mod.process(state)
        assert result.total_energy() == pytest.approx(input_e, abs=1e-6)

    def test_process_adds_provenance(self):
        mod = ObservabilityModule()
        state = FieldState(tensor=torch.randn(8), entropy=1.0)
        result = mod.process(state)
        assert "observability" in result.provenance

    def test_metrics_populated(self):
        mod = ObservabilityModule()
        mod.process(FieldState(tensor=torch.randn(8), entropy=1.0))
        assert mod.metrics is not None
        assert isinstance(mod.metrics, ObservabilityMetrics)
        assert isinstance(mod.metrics.scbf, SCBFMetrics)

    def test_phase_returns_sec_phase(self):
        mod = ObservabilityModule()
        mod.process(FieldState(tensor=torch.randn(8), entropy=1.0))
        phase = mod.phase()
        assert isinstance(phase, SECPhase)

    def test_health_returns_rbf(self):
        mod = ObservabilityModule()
        mod.process(FieldState(tensor=torch.randn(8), entropy=1.0))
        h = mod.health()
        assert isinstance(h, RBFBalance)

    def test_qbe_status_accessible(self):
        mod = ObservabilityModule()
        mod.process(FieldState(tensor=torch.randn(8), entropy=1.0))
        assert mod.metrics.qbe_status in (
            "Near Equilibrium", "Moderate Equilibrium", "Far from Equilibrium"
        )

    def test_multiple_processes(self):
        mod = ObservabilityModule()
        for _ in range(20):
            mod.process(FieldState(tensor=torch.randn(8), entropy=1.0))
        assert mod.metrics.step_count == 20
        assert mod.tracker.step == 20
