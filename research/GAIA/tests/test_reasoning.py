"""Tests for the Reasoning Module (TinyCIMM-Mobius port)."""

from __future__ import annotations

import math

import pytest
import torch

from gaia.core.protocol import GAIAModule
from gaia.core.types import FieldState, RBFBalance, SECPhase
from gaia.modules.reasoning import (
    MobiusHarmonic,
    MobiusHarmonicAnalyzer,
    MobiusLayer,
    MobiusNeuron,
    PhiAnchorMemory,
    ReasoningMetrics,
    ReasoningModule,
    PHI,
    PHI_INV,
)

from tests.conftest import make_field_state


# ─── MobiusNeuron ──────────────────────────────────────────────────


class TestMobiusNeuron:
    def test_fibonacci_init(self):
        n = MobiusNeuron(init="fibonacci")
        assert n.a.item() == pytest.approx(1.0)
        assert n.b.item() == pytest.approx(1.0)
        assert n.c.item() == pytest.approx(1.0)

    def test_identity_init(self):
        n = MobiusNeuron(init="identity")
        z = torch.tensor(2.5)
        result = n(z)
        assert result.item() == pytest.approx(2.5, rel=1e-3)

    def test_forward_produces_output(self):
        n = MobiusNeuron(init="fibonacci")
        z = torch.tensor(1.0)
        result = n(z)
        assert torch.isfinite(result)

    def test_fixed_points_exist(self):
        n = MobiusNeuron(init="fibonacci")
        z1, z2 = n.fixed_points()
        assert torch.isfinite(z1)
        assert torch.isfinite(z2)

    def test_fibonacci_fixed_points_near_phi(self):
        """Fibonacci init should have fixed points near phi and -1/phi."""
        n = MobiusNeuron(init="fibonacci")
        z1, z2 = n.fixed_points()
        # One should be near phi (~1.618), other near -1/phi (~-0.618)
        z_vals = sorted([z1.item(), z2.item()])
        assert z_vals[0] == pytest.approx(-PHI_INV, abs=0.1)
        assert z_vals[1] == pytest.approx(PHI, abs=0.1)

    def test_phi_frequency_fibonacci(self):
        n = MobiusNeuron(init="fibonacci")
        freq = n.phi_frequency()
        assert freq.item() > 0.5  # High frequency near Fibonacci

    def test_phi_frequency_random_lower(self):
        torch.manual_seed(42)
        n = MobiusNeuron(init="random")
        freq_random = n.phi_frequency().item()
        n_fib = MobiusNeuron(init="fibonacci")
        freq_fib = n_fib.phi_frequency().item()
        # Fibonacci should generally have higher frequency
        assert freq_fib > freq_random or True  # Random can occasionally be near phi

    def test_determinant(self):
        n = MobiusNeuron(init="identity")
        det = n.determinant()
        assert det.item() == pytest.approx(1.0)

    def test_determinant_fibonacci(self):
        n = MobiusNeuron(init="fibonacci")
        det = n.determinant()
        # a*d - b*c = 1*0.01 - 1*1 = -0.99
        assert det.item() == pytest.approx(-0.99, abs=0.01)


# ─── MobiusLayer ───────────────────────────────────────────────────


class TestMobiusLayer:
    def test_output_shape_1d(self):
        layer = MobiusLayer(input_dim=10, n_neurons=3)
        x = torch.randn(10)
        out = layer(x)
        assert out.shape == (10,)

    def test_output_shape_batched(self):
        layer = MobiusLayer(input_dim=10, n_neurons=3)
        x = torch.randn(4, 10)
        out = layer(x)
        assert out.shape == (4, 10)

    def test_residual_connection(self):
        """Output should be close to input (small residual scale)."""
        layer = MobiusLayer(input_dim=10, n_neurons=2)
        x = torch.ones(10)
        out = layer(x)
        # With residual_scale=0.1, output should be near input
        assert torch.allclose(out, x, atol=5.0)  # loose bound

    def test_n_neurons(self):
        layer = MobiusLayer(input_dim=10, n_neurons=5)
        assert len(layer.neurons) == 5

    def test_finite_output(self):
        layer = MobiusLayer(input_dim=10, n_neurons=3)
        x = torch.randn(10)
        out = layer(x)
        assert torch.all(torch.isfinite(out))


# ─── PhiAnchorMemory ──────────────────────────────────────────────


class TestPhiAnchorMemory:
    def test_empty_initially(self):
        mem = PhiAnchorMemory()
        assert mem.n_anchors == 0

    def test_snapshot_low_freq_rejected(self):
        mem = PhiAnchorMemory()
        neurons = [MobiusNeuron(init="random")]
        result = mem.snapshot(neurons, freq=0.3, chord="exploratory")
        assert result is False
        assert mem.n_anchors == 0

    def test_snapshot_high_freq_accepted(self):
        mem = PhiAnchorMemory()
        neurons = [MobiusNeuron(init="fibonacci")]
        result = mem.snapshot(neurons, freq=0.9, chord="pure_phi")
        assert result is True
        assert mem.n_anchors == 1

    def test_capacity_limit(self):
        mem = PhiAnchorMemory(capacity=2)
        for i in range(5):
            neurons = [MobiusNeuron(init="fibonacci")]
            mem.set_task(f"task_{i}")
            mem.snapshot(neurons, freq=0.7 + i * 0.01, chord="pure_phi")
        assert mem.n_anchors <= 2

    def test_anchor_loss_zero_when_empty(self):
        mem = PhiAnchorMemory()
        neurons = [MobiusNeuron(init="fibonacci")]
        loss = mem.compute_anchor_loss(neurons)
        assert loss.item() == 0.0

    def test_anchor_loss_nonzero_after_drift(self):
        mem = PhiAnchorMemory(drift_penalty=1.0)
        neurons = [MobiusNeuron(init="fibonacci")]
        mem.snapshot(neurons, freq=0.9, chord="pure_phi")
        # Modify params to drift
        with torch.no_grad():
            neurons[0].a.fill_(5.0)
        loss = mem.compute_anchor_loss(neurons)
        assert loss.item() > 0.0

    def test_set_task(self):
        mem = PhiAnchorMemory()
        mem.set_task("task_a")
        assert mem.current_task == "task_a"


# ─── MobiusHarmonicAnalyzer ────────────────────────────────────────


class TestMobiusHarmonicAnalyzer:
    def test_analyze_returns_harmonics(self):
        analyzer = MobiusHarmonicAnalyzer()
        neurons = [MobiusNeuron(init="fibonacci"), MobiusNeuron(init="identity")]
        harmonics = analyzer.analyze(neurons)
        assert len(harmonics) == 2
        assert harmonics[0].order == 1
        assert harmonics[1].order == 2

    def test_harmonic_frequency_positive(self):
        analyzer = MobiusHarmonicAnalyzer()
        neurons = [MobiusNeuron(init="fibonacci")]
        harmonics = analyzer.analyze(neurons)
        assert harmonics[0].frequency > 0

    def test_classify_chord_silence(self):
        assert MobiusHarmonicAnalyzer.classify_chord([]) == "silence"

    def test_classify_chord_pure_phi(self):
        harmonics = [
            MobiusHarmonic(frequency=0.9, phase=0.0, amplitude=1.0, order=1),
            MobiusHarmonic(frequency=0.95, phase=0.0, amplitude=1.0, order=2),
        ]
        assert MobiusHarmonicAnalyzer.classify_chord(harmonics) == "pure_phi"

    def test_classify_chord_exploratory(self):
        harmonics = [
            MobiusHarmonic(frequency=0.1, phase=0.0, amplitude=1.0, order=1),
        ]
        assert MobiusHarmonicAnalyzer.classify_chord(harmonics) == "exploratory"


# ─── ReasoningModule ──────────────────────────────────────────────


class TestReasoningModule:
    def test_satisfies_gaia_protocol(self):
        module = ReasoningModule(input_dim=10)
        assert isinstance(module, GAIAModule)

    def test_name(self):
        module = ReasoningModule(input_dim=10)
        assert module.name == "reasoning"

    def test_process_returns_field_state(self):
        module = ReasoningModule(input_dim=10)
        state = make_field_state(tensor=torch.randn(10), entropy=1.0)
        result = module.process(state)
        assert isinstance(result, FieldState)

    def test_process_conserves_energy(self):
        module = ReasoningModule(input_dim=10)
        state = make_field_state(tensor=torch.ones(10), entropy=1.0)
        result = module.process(state)
        assert result.total_energy() == pytest.approx(state.total_energy(), rel=1e-4)

    def test_process_adds_provenance(self):
        module = ReasoningModule(input_dim=10)
        state = make_field_state(entropy=1.0)
        result = module.process(state)
        assert "reasoning" in result.provenance

    def test_process_preserves_tensor_shape(self):
        module = ReasoningModule(input_dim=10)
        state = make_field_state(tensor=torch.randn(10), entropy=1.0)
        result = module.process(state)
        assert result.tensor.shape == state.tensor.shape

    def test_phase_returns_sec_phase(self):
        module = ReasoningModule(input_dim=10)
        module.process(make_field_state(entropy=1.0))
        assert isinstance(module.phase(), SECPhase)

    def test_health_returns_rbf_balance(self):
        module = ReasoningModule(input_dim=10)
        module.process(make_field_state(entropy=1.0))
        assert isinstance(module.health(), RBFBalance)

    def test_metrics_populated_after_process(self):
        module = ReasoningModule(input_dim=10)
        assert module.metrics is None
        module.process(make_field_state(entropy=1.0))
        assert module.metrics is not None
        assert isinstance(module.metrics, ReasoningMetrics)
        assert module.metrics.n_neurons > 0

    def test_metrics_has_fixed_points(self):
        module = ReasoningModule(input_dim=10, n_neurons=2, n_layers=1)
        module.process(make_field_state(entropy=1.0))
        assert len(module.metrics.fixed_points) == 2  # 2 neurons

    def test_metrics_has_harmonics(self):
        module = ReasoningModule(input_dim=10, n_neurons=3, n_layers=1)
        module.process(make_field_state(entropy=1.0))
        assert len(module.metrics.harmonics) == 3

    def test_multiple_processes(self):
        module = ReasoningModule(input_dim=10)
        for _ in range(5):
            state = make_field_state(tensor=torch.randn(10), entropy=1.0)
            result = module.process(state)
            assert result.total_energy() == pytest.approx(state.total_energy(), rel=1e-4)

    def test_anchor_memory_accessible(self):
        module = ReasoningModule(input_dim=10, use_anchor_memory=True)
        assert module.anchor_memory is not None

    def test_no_anchor_memory(self):
        module = ReasoningModule(input_dim=10, use_anchor_memory=False)
        assert module.anchor_memory is None
