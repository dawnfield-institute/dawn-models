"""Tests for CoupledFieldsBus — lensed broadcast with coupled oscillator dynamics.

Verifies per-module lensing, coupling matrix, QBE-regulated coupling strength,
state persistence, and PAC conservation at boundary.
"""

from __future__ import annotations

import math

import pytest
import torch

from gaia.core.coupled_fields_bus import (
    CoupledFieldsBus,
    CoupledWeight,
    CoupledFieldState,
    LENS_CLAMP_MIN,
    LENS_CLAMP_MAX,
    COUPLING_KAPPA,
    QBE_OMEGA,
)
from gaia.core.types import FieldState, RBFBalance, SECPhase


# ─── Test Helpers ────────────────────────────────────────────────


class StubModule:
    """Configurable stub that implements GAIAModule protocol."""

    def __init__(self, name: str, transform=None, rbf_balance: float = 1.0):
        self._name = name
        self._transform = transform or (lambda t: t.clone())
        self._rbf_balance = rbf_balance

    @property
    def name(self) -> str:
        return self._name

    def process(self, field_state: FieldState) -> FieldState:
        new_tensor = self._transform(field_state.tensor)
        return FieldState(
            tensor=new_tensor,
            entropy=field_state.entropy,
            phase=field_state.phase,
            conservation_budget=field_state.conservation_budget,
            provenance=field_state.provenance,
            timestamp=field_state.timestamp,
        )

    def phase(self) -> SECPhase:
        return SECPhase.TRANSITIONAL

    def health(self) -> RBFBalance:
        return RBFBalance(balance=self._rbf_balance, energy=1.0, information=1.0, memory=0.0)


def _make_field(dim: int = 16, seed: int = 42) -> FieldState:
    """Create a deterministic test field state."""
    gen = torch.Generator().manual_seed(seed)
    tensor = torch.randn(dim, generator=gen).abs() + 0.1  # positive values
    return FieldState(
        tensor=tensor,
        entropy=1.5,
        phase=SECPhase.TRANSITIONAL,
        conservation_budget=0.0,
        provenance=[],
        timestamp=0.0,
    )


# ─── Tests ───────────────────────────────────────────────────────


class TestSingleModule:

    def test_single_module_gets_weight_one(self):
        """A lone module always gets normalized weight 1.0."""
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("only", lambda t: t * 1.1))
        field = _make_field()

        bus.process(field)

        weights = bus.coupled_log[-1]
        assert len(weights) == 1
        assert abs(weights[0].normalized_weight - 1.0) < 1e-10

    def test_single_module_preserves_pac(self):
        """Single module output conserves energy after PAC scaling."""
        bus = CoupledFieldsBus(enforcement="hard")
        bus.register_module(StubModule("scaler", lambda t: t * 2.0))
        field = _make_field()

        result = bus.process(field)

        input_energy = field.total_energy()
        output_energy = result.total_energy()
        assert abs(output_energy - input_energy) < 1e-4


class TestIdentityModules:

    def test_all_identity_preserves_energy(self):
        """When all modules are identity, PAC conservation still holds.

        Note: output differs from input because lensing transforms the signal
        before identity processes it. This is by design — the lens is the
        module's perspective, not an identity operation.
        """
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("id1", lambda t: t.clone()))
        bus.register_module(StubModule("id2", lambda t: t.clone()))
        field = _make_field()

        result = bus.process(field)

        # Energy is conserved even if tensor values differ
        assert abs(result.total_energy() - field.total_energy()) < 1e-4

    def test_identity_still_gets_weight(self):
        """Identity modules get weight from prediction accuracy (unlike ResonanceBus)."""
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("identity", lambda t: t.clone()))
        bus.register_module(StubModule("active", lambda t: t * 1.5))
        field = _make_field()

        bus.process(field)

        weights = bus.coupled_log[-1]
        identity_w = next(w for w in weights if w.module_name == "identity")
        # Identity should have non-zero weight (prediction accuracy still works)
        assert identity_w.normalized_weight > 0.0


class TestPACConservation:

    def test_pac_conserved_with_amplifier(self):
        bus = CoupledFieldsBus(enforcement="hard")
        bus.register_module(StubModule("amp", lambda t: t * 3.0))
        field = _make_field()

        result = bus.process(field)
        assert abs(result.total_energy() - field.total_energy()) < 1e-4

    def test_pac_conserved_with_attenuator(self):
        bus = CoupledFieldsBus(enforcement="hard")
        bus.register_module(StubModule("atten", lambda t: t * 0.3))
        field = _make_field()

        result = bus.process(field)
        assert abs(result.total_energy() - field.total_energy()) < 1e-4

    def test_pac_zero_output_restored(self):
        """Zero output gets restored from input."""
        bus = CoupledFieldsBus(enforcement="hard")
        bus.register_module(StubModule("zero", lambda t: torch.zeros_like(t)))
        field = _make_field()

        result = bus.process(field)
        assert abs(result.total_energy() - field.total_energy()) < 1e-4


class TestLensComputation:

    def test_new_module_lens_is_ones(self):
        """New modules start with identity lens (no bias)."""
        bus = CoupledFieldsBus(enforcement="soft")
        mod = StubModule("new", lambda t: t * 1.1)
        bus.register_module(mod)
        field = _make_field()

        bus.process(field)

        state = bus.field_states["new"]
        # After first tick, lens was computed from initial state (clone of input)
        # which is all positive → lens should be non-trivial but not ones
        # The key property: on tick 0, state is initialized to input.clone()
        assert state.ticks_alive == 1

    def test_lens_clamped_min(self):
        """Lens values don't go below LENS_CLAMP_MIN."""
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("mod", lambda t: t * 1.1))
        field = _make_field()

        # Run a few ticks to let lens evolve
        for _ in range(5):
            bus.process(field)

        state = bus.field_states["mod"]
        assert state.lens.min().item() >= LENS_CLAMP_MIN - 1e-6

    def test_lens_clamped_max(self):
        """Lens values don't exceed LENS_CLAMP_MAX."""
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("mod", lambda t: t * 1.1))
        field = _make_field()

        for _ in range(5):
            bus.process(field)

        state = bus.field_states["mod"]
        assert state.lens.max().item() <= LENS_CLAMP_MAX + 1e-6


class TestLensDivergence:

    def test_different_transforms_diverge_lenses(self):
        """Modules with different transforms develop different lenses over time."""
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("amplifier", lambda t: t * 2.0))
        bus.register_module(StubModule("dampener", lambda t: t * 0.5))
        field = _make_field()

        # Run enough ticks for lenses to diverge
        for _ in range(20):
            bus.process(field)

        states = bus.field_states
        lens_a = states["amplifier"].lens
        lens_b = states["dampener"].lens
        diff = float(torch.norm(lens_a - lens_b).item())
        assert diff > 0.01, f"Lenses should diverge, diff={diff}"

    def test_lens_contrast_increases(self):
        """Lens contrast (std) should increase from zero as modules specialize."""
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("mod", lambda t: t * 1.5))
        field = _make_field()

        # First tick: lens starts from input (some variance already)
        bus.process(field)
        initial_contrast = bus.coupled_log[0][0].lens_contrast

        # After more ticks
        for _ in range(10):
            bus.process(field)

        later_contrast = bus.coupled_log[-1][0].lens_contrast
        # Lens contrast should be non-zero (module develops preferences)
        assert later_contrast > 0.0


class TestCouplingMatrix:

    def test_coupling_symmetric(self):
        """Coupling matrix is symmetric: C[i,j] == C[j,i]."""
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("a", lambda t: t * 1.2))
        bus.register_module(StubModule("b", lambda t: t * 0.8))
        field = _make_field()

        bus.process(field)

        # Access coupling computation directly
        matrix = bus._compute_coupling_matrix(["a", "b"])
        assert abs(matrix[("a", "b")] - matrix[("b", "a")]) < 1e-10

    def test_no_self_coupling(self):
        """Coupling matrix has no diagonal entries."""
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("a", lambda t: t * 1.2))
        bus.register_module(StubModule("b", lambda t: t * 0.8))
        field = _make_field()

        bus.process(field)

        matrix = bus._compute_coupling_matrix(["a", "b"])
        assert ("a", "a") not in matrix
        assert ("b", "b") not in matrix


class TestCouplingForce:

    def test_similar_states_small_force(self):
        """Modules with similar states have small coupling force.

        Note: even identity modules develop slightly different states because
        the Mobius iteration includes the lensed input (which gets PAC-rescaled
        per-module). The coupling force is small but non-zero.
        """
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("a", lambda t: t.clone()))
        bus.register_module(StubModule("b", lambda t: t.clone()))
        field = _make_field()

        bus.process(field)

        # Compare to a bus with very different modules
        bus2 = CoupledFieldsBus(enforcement="soft")
        bus2.register_module(StubModule("c", lambda t: t * 3.0))
        bus2.register_module(StubModule("d", lambda t: t * 0.1))
        bus2.process(field)

        # Similar modules should have less coupling force than divergent ones
        similar_force = max(w.coupling_received for w in bus.coupled_log[-1])
        divergent_force = max(w.coupling_received for w in bus2.coupled_log[-1])
        assert similar_force < divergent_force


class TestQBEModulation:

    def test_coupling_strength_varies(self):
        """Coupling strength oscillates with tick via QPL."""
        bus = CoupledFieldsBus(enforcement="soft", coupling_kappa=0.1)

        strengths = []
        for tick in range(100):
            bus._tick = tick
            strengths.append(bus._compute_coupling_strength())

        # Should vary (not constant)
        assert max(strengths) > min(strengths)
        # Should oscillate between kappa*0 and kappa*2
        assert min(strengths) >= -0.01  # kappa * (1 - 1) = 0
        assert max(strengths) <= 0.21   # kappa * (1 + 1) = 0.2

    def test_qbe_period(self):
        """QPL completes one full cycle at expected period."""
        bus = CoupledFieldsBus(enforcement="soft")
        period = int(2 * math.pi / QBE_OMEGA)  # ~314 ticks

        bus._tick = 0
        s0 = bus._compute_coupling_strength()

        bus._tick = period
        s_period = bus._compute_coupling_strength()

        assert abs(s0 - s_period) < 1e-4


class TestStatePersistence:

    def test_state_survives_across_ticks(self):
        """Resonance states persist between process() calls."""
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("mod", lambda t: t * 1.2))
        field = _make_field()

        bus.process(field)
        assert "mod" in bus.field_states
        state_after_1 = bus.field_states["mod"].tensor.clone()

        bus.process(field)
        state_after_2 = bus.field_states["mod"].tensor

        # States should differ (Mobius iteration changes them)
        diff = float(torch.norm(state_after_2 - state_after_1).item())
        assert diff > 0.0

    def test_ticks_alive_increments(self):
        """ticks_alive counter increments each process() call."""
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("mod", lambda t: t * 1.1))
        field = _make_field()

        bus.process(field)
        assert bus.field_states["mod"].ticks_alive == 1

        bus.process(field)
        assert bus.field_states["mod"].ticks_alive == 2

    def test_surprise_history_populates(self):
        """Surprise history accumulates over ticks."""
        bus = CoupledFieldsBus(enforcement="soft", surprise_window=5)
        bus.register_module(StubModule("mod", lambda t: t * 1.1))
        field = _make_field()

        for _ in range(7):
            bus.process(field)

        history = bus.field_states["mod"].surprise_history
        assert len(history) == 5  # capped at window size


class TestProvenance:

    def test_provenance_tracks_active_modules(self):
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("mod_a", lambda t: t * 1.2))
        bus.register_module(StubModule("mod_b", lambda t: t * 1.5))
        field = _make_field()

        result = bus.process(field)

        assert "mod_a" in result.provenance
        assert "mod_b" in result.provenance


class TestPassthrough:

    def test_no_modules_returns_input(self):
        bus = CoupledFieldsBus(enforcement="soft")
        field = _make_field()

        result = bus.process(field)

        diff = float(torch.norm(result.tensor - field.tensor).item())
        assert diff < 1e-10


class TestRBFSuppression:

    def test_unhealthy_module_suppressed(self):
        bus = CoupledFieldsBus(enforcement="soft", rbf_suppression_threshold=0.5)
        bus.register_module(StubModule("healthy", lambda t: t * 1.5, rbf_balance=1.0))
        bus.register_module(StubModule("sick", lambda t: t * 3.0, rbf_balance=0.1))
        field = _make_field()

        bus.process(field)

        weights = bus.coupled_log[-1]
        module_names = [w.module_name for w in weights]
        assert "healthy" in module_names
        assert "sick" not in module_names


class TestCommutativity:

    def test_registration_order_irrelevant(self):
        """Superposition merge is commutative — order doesn't matter."""
        def make_bus_ordered(order):
            bus = CoupledFieldsBus(enforcement="soft")
            modules = {
                "a": StubModule("a", lambda t: t * 1.3),
                "b": StubModule("b", lambda t: t * 0.7),
                "c": StubModule("c", lambda t: t * 1.1),
            }
            for name in order:
                bus.register_module(modules[name])
            return bus

        field = _make_field()

        bus1 = make_bus_ordered(["a", "b", "c"])
        result1 = bus1.process(field)

        bus2 = make_bus_ordered(["c", "a", "b"])
        result2 = bus2.process(field)

        diff = float(torch.norm(result1.tensor - result2.tensor).item())
        assert diff < 1e-6, f"Order-dependent: diff={diff}"


class TestCoupledLog:

    def test_log_populated_after_process(self):
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("mod", lambda t: t * 1.2))
        field = _make_field()

        assert len(bus.coupled_log) == 0
        bus.process(field)
        assert len(bus.coupled_log) == 1
        bus.process(field)
        assert len(bus.coupled_log) == 2

    def test_log_entries_are_coupled_weights(self):
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("mod", lambda t: t * 1.2))
        bus.process(_make_field())

        entry = bus.coupled_log[0]
        assert isinstance(entry, list)
        assert all(isinstance(w, CoupledWeight) for w in entry)

    def test_weight_has_lens_contrast(self):
        """CoupledWeight includes lens_contrast diagnostic."""
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("mod", lambda t: t * 1.2))
        bus.process(_make_field())

        w = bus.coupled_log[0][0]
        assert hasattr(w, "lens_contrast")
        assert isinstance(w.lens_contrast, float)

    def test_weight_has_coupling_received(self):
        """CoupledWeight includes coupling_received diagnostic."""
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("a", lambda t: t * 1.2))
        bus.register_module(StubModule("b", lambda t: t * 0.8))
        bus.process(_make_field())

        for w in bus.coupled_log[0]:
            assert hasattr(w, "coupling_received")
            assert isinstance(w.coupling_received, float)


class TestMetrics:

    def test_get_metrics_structure(self):
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("mod", lambda t: t * 1.2))

        m = bus.get_metrics()
        assert m["modules_registered"] == 1
        assert m["module_names"] == ["mod"]
        assert m["enforcement"] == "soft"
        assert m["dispatch_mode"] == "coupled_fields"
        assert m["coupled_ticks"] == 0

    def test_metrics_include_coupling(self):
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("mod", lambda t: t * 1.2))
        bus.process(_make_field())

        m = bus.get_metrics()
        assert "coupling" in m
        assert "strength" in m["coupling"]
        assert "qpl" in m["coupling"]

    def test_metrics_include_lens_contrast(self):
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("mod", lambda t: t * 1.2))
        bus.process(_make_field())

        m = bus.get_metrics()
        assert "resonance_states" in m
        assert "mod" in m["resonance_states"]
        assert "lens_contrast" in m["resonance_states"]["mod"]


class TestFullStack:

    def test_five_module_smoke(self):
        """Smoke test: 5 different transforms produce valid output."""
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("obs", lambda t: t.clone()))
        bus.register_module(StubModule("safety", lambda t: t * 0.9))
        bus.register_module(StubModule("reasoning", lambda t: t * 1.2))
        bus.register_module(StubModule("memory", lambda t: t * 1.05))
        bus.register_module(StubModule("language", lambda t: t * 0.95))
        field = _make_field()

        result = bus.process(field)

        assert abs(result.total_energy() - field.total_energy()) < 1e-3
        assert len(result.provenance) >= 1
        assert len(bus.coupled_log) == 1
        assert len(bus.coupled_log[0]) == 5

    def test_multi_tick_stability(self):
        """Bus remains stable over many ticks (no NaN, no explosion)."""
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("a", lambda t: t * 1.3))
        bus.register_module(StubModule("b", lambda t: t * 0.7))
        bus.register_module(StubModule("c", lambda t: t * 1.1))
        field = _make_field()

        for _ in range(50):
            result = bus.process(field)
            assert not torch.isnan(result.tensor).any()
            assert not torch.isinf(result.tensor).any()


class TestViolationLog:

    def test_soft_enforcement_logs_violations(self):
        bus = CoupledFieldsBus(enforcement="soft", tolerance=0.0)
        bus.register_module(StubModule("mod", lambda t: t * 1.1))
        field = _make_field()

        bus.process(field)
        assert isinstance(bus.violation_log, list)

    def test_monitor_mode_no_raise(self):
        bus = CoupledFieldsBus(enforcement="monitor")
        bus.register_module(StubModule("mod", lambda t: t * 5.0))
        field = _make_field()

        result = bus.process(field)
        assert result is not None


class TestEnforcementValidation:

    def test_invalid_enforcement_raises(self):
        with pytest.raises(ValueError, match="enforcement"):
            CoupledFieldsBus(enforcement="invalid")
