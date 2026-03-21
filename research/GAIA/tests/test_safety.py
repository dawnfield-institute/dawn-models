"""Tests for the Safety Module (TinyCIMM-Boltzmann port)."""

from __future__ import annotations

import pytest
import torch

from gaia.core.protocol import GAIAModule
from gaia.core.types import FieldState, RBFBalance, SECPhase
from gaia.modules.safety import (
    BoltzmannHead,
    BoltzmannLayer,
    BoltzmannMonitor,
    ConservationProjector,
    ConservationState,
    SafetyMetrics,
    SafetyModule,
    _classify_sec_phase,
)

from tests.conftest import make_field_state


# ─── SEC Classification ────────────────────────────────────────────


class TestSECClassification:
    def test_crystallized(self):
        assert _classify_sec_phase(0.3) == SECPhase.CRYSTALLIZED

    def test_ordered(self):
        assert _classify_sec_phase(1.0) == SECPhase.ORDERED

    def test_transitional(self):
        assert _classify_sec_phase(3.0) == SECPhase.TRANSITIONAL

    def test_chaotic(self):
        assert _classify_sec_phase(5.0) == SECPhase.CHAOTIC


# ─── BoltzmannHead ─────────────────────────────────────────────────


class TestBoltzmannHead:
    def test_output_shape(self):
        head = BoltzmannHead(input_dim=10, head_dim=8)
        x = torch.randn(1, 10)
        out, entropy, entropy_t = head(x)
        assert out.shape == (1, 8)

    def test_entropy_is_positive(self):
        head = BoltzmannHead(input_dim=10, head_dim=8)
        x = torch.randn(1, 10)
        _, entropy, _ = head(x)
        assert entropy >= 0.0

    def test_entropy_tensor_is_differentiable(self):
        head = BoltzmannHead(input_dim=10, head_dim=8)
        x = torch.randn(1, 10, requires_grad=True)
        _, _, entropy_t = head(x)
        assert entropy_t.requires_grad

    def test_entropy_property_updated(self):
        head = BoltzmannHead(input_dim=10, head_dim=8)
        x = torch.randn(1, 10)
        _, entropy, _ = head(x)
        assert head.entropy == pytest.approx(entropy)

    def test_phase_property_updated(self):
        head = BoltzmannHead(input_dim=10, head_dim=8)
        x = torch.randn(1, 10)
        head(x)
        assert isinstance(head.phase, SECPhase)


# ─── ConservationProjector ─────────────────────────────────────────


class TestConservationProjector:
    def test_soft_mode_produces_loss(self):
        proj = ConservationProjector(n_heads=2, head_dim=4, mode="soft")
        outputs = [torch.randn(1, 4), torch.randn(1, 4)]
        entropies = [1.0, 1.5]
        entropy_ts = [torch.tensor(1.0, requires_grad=True), torch.tensor(1.5, requires_grad=True)]
        # Initialize budget
        proj.set_target_budget(2.5)
        combined, loss = proj(outputs, entropies, entropy_ts)
        assert combined.shape == (1, 8)
        # Loss should be ~0 since current = target
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_soft_mode_nonzero_loss_on_violation(self):
        proj = ConservationProjector(n_heads=2, head_dim=4, mode="soft")
        proj.set_target_budget(1.0)  # Target much lower than actual
        outputs = [torch.randn(1, 4), torch.randn(1, 4)]
        entropies = [2.0, 2.0]  # Total = 4.0, target = 1.0
        entropy_ts = [torch.tensor(2.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)]
        _, loss = proj(outputs, entropies, entropy_ts)
        assert loss.item() > 0.0

    def test_hard_mode_zero_loss(self):
        proj = ConservationProjector(n_heads=2, head_dim=4, mode="hard")
        proj.set_target_budget(2.0)
        outputs = [torch.randn(1, 4), torch.randn(1, 4)]
        entropies = [1.0, 1.5]
        entropy_ts = [torch.tensor(1.0), torch.tensor(1.5)]
        _, loss = proj(outputs, entropies, entropy_ts)
        assert loss.item() == pytest.approx(0.0)

    def test_auto_budget_initialization(self):
        proj = ConservationProjector(n_heads=2, head_dim=4, mode="soft")
        assert proj.target_budget is None
        outputs = [torch.randn(1, 4), torch.randn(1, 4)]
        entropies = [1.0, 2.0]
        entropy_ts = [torch.tensor(1.0), torch.tensor(2.0)]
        proj(outputs, entropies, entropy_ts)
        assert proj.target_budget == pytest.approx(3.0)


# ─── BoltzmannLayer ────────────────────────────────────────────────


class TestBoltzmannLayer:
    def test_output_shape(self):
        layer = BoltzmannLayer(input_dim=10, n_heads=4, head_dim=8, output_dim=16)
        x = torch.randn(1, 10)
        out, loss, entropies = layer(x)
        assert out.shape == (1, 16)
        assert len(entropies) == 4

    def test_conservation_loss_is_tensor(self):
        layer = BoltzmannLayer(input_dim=10, n_heads=4, head_dim=8)
        x = torch.randn(1, 10)
        _, loss, _ = layer(x)
        assert isinstance(loss, torch.Tensor)

    def test_default_output_dim_matches_input(self):
        layer = BoltzmannLayer(input_dim=10, n_heads=2, head_dim=4)
        x = torch.randn(1, 10)
        out, _, _ = layer(x)
        assert out.shape == (1, 10)

    def test_head_entropies_all_positive(self):
        layer = BoltzmannLayer(input_dim=10, n_heads=4, head_dim=8)
        x = torch.randn(1, 10)
        _, _, entropies = layer(x)
        assert all(e >= 0.0 for e in entropies)


# ─── BoltzmannMonitor ──────────────────────────────────────────────


class TestBoltzmannMonitor:
    def test_initial_state(self):
        monitor = BoltzmannMonitor()
        assert monitor.state.steps == 0
        assert monitor.budget_stability() == 1.0
        assert monitor.mean_compensation() == 1.0

    def test_update_increments_steps(self):
        monitor = BoltzmannMonitor()
        monitor.update([1.0, 1.5], target_budget=2.5)
        assert monitor.state.steps == 1

    def test_violation_tracking(self):
        monitor = BoltzmannMonitor()
        monitor.update([2.0, 2.0], target_budget=3.0)
        assert monitor.state.violation == pytest.approx(1.0)

    def test_compensation_ratio(self):
        monitor = BoltzmannMonitor()
        monitor.update([1.0, 1.0], target_budget=2.0)
        # Second update: head 0 up, head 1 down
        monitor.update([1.5, 0.5], target_budget=2.0)
        # Perfect compensation: |decrease| / increase = 0.5 / 0.5 = 1.0
        assert monitor.state.compensation_ratio == pytest.approx(1.0)

    def test_window_trimming(self):
        monitor = BoltzmannMonitor(window_size=5)
        for i in range(10):
            monitor.update([1.0], target_budget=1.0)
        assert len(monitor.budget_history) == 5

    def test_budget_stability_perfect(self):
        monitor = BoltzmannMonitor()
        for _ in range(5):
            monitor.update([1.0, 1.0], target_budget=2.0)
        assert monitor.budget_stability() == pytest.approx(1.0)

    def test_budget_stability_degrades_with_variance(self):
        monitor = BoltzmannMonitor()
        budgets = [1.0, 5.0, 1.0, 5.0, 1.0]
        for b in budgets:
            monitor.update([b], target_budget=3.0)
        assert monitor.budget_stability() < 1.0


# ─── SafetyModule ──────────────────────────────────────────────────


class TestSafetyModule:
    def test_satisfies_gaia_protocol(self):
        module = SafetyModule(input_dim=10)
        assert isinstance(module, GAIAModule)

    def test_name(self):
        module = SafetyModule(input_dim=10)
        assert module.name == "safety"

    def test_process_returns_field_state(self):
        module = SafetyModule(input_dim=10)
        state = make_field_state(tensor=torch.randn(10), entropy=1.0)
        result = module.process(state)
        assert isinstance(result, FieldState)

    def test_process_conserves_energy(self):
        module = SafetyModule(input_dim=10)
        state = make_field_state(tensor=torch.ones(10), entropy=1.0)
        result = module.process(state)
        assert result.total_energy() == pytest.approx(state.total_energy(), rel=1e-4)

    def test_process_adds_provenance(self):
        module = SafetyModule(input_dim=10)
        state = make_field_state(entropy=1.0)
        result = module.process(state)
        assert "safety" in result.provenance

    def test_process_preserves_tensor_shape(self):
        module = SafetyModule(input_dim=10)
        state = make_field_state(tensor=torch.randn(10), entropy=1.0)
        result = module.process(state)
        assert result.tensor.shape == state.tensor.shape

    def test_phase_returns_sec_phase(self):
        module = SafetyModule(input_dim=10)
        state = make_field_state(entropy=1.0)
        module.process(state)
        assert isinstance(module.phase(), SECPhase)

    def test_health_returns_rbf_balance(self):
        module = SafetyModule(input_dim=10)
        state = make_field_state(entropy=1.0)
        module.process(state)
        health = module.health()
        assert isinstance(health, RBFBalance)

    def test_metrics_populated_after_process(self):
        module = SafetyModule(input_dim=10)
        assert module.metrics is None
        state = make_field_state(entropy=1.0)
        module.process(state)
        assert module.metrics is not None
        assert isinstance(module.metrics, SafetyMetrics)
        assert len(module.metrics.head_entropies) > 0

    def test_multiple_processes(self):
        module = SafetyModule(input_dim=10)
        for _ in range(5):
            state = make_field_state(tensor=torch.randn(10), entropy=1.0)
            result = module.process(state)
            assert result.total_energy() == pytest.approx(state.total_energy(), rel=1e-4)

    def test_monitors_accessible(self):
        module = SafetyModule(input_dim=10, n_layers=2)
        state = make_field_state(entropy=1.0)
        module.process(state)
        assert len(module.monitors) == 2

    def test_conservation_mode_hard(self):
        module = SafetyModule(input_dim=10, conservation_mode="hard")
        state = make_field_state(tensor=torch.ones(10), entropy=1.0)
        result = module.process(state)
        assert result.total_energy() == pytest.approx(state.total_energy(), rel=1e-4)
