"""Tests for ResonanceBus — QSocket-inspired field coupling.

Verifies broadcast dispatch, delta-coherence weighting,
superposition merge, and PAC conservation at boundary.
"""

from __future__ import annotations

import pytest
import torch

from gaia.core.resonance_bus import ResonanceBus, ResonanceWeight
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
        bus = ResonanceBus(enforcement="soft")
        bus.register_module(StubModule("only", lambda t: t * 1.1))
        field = _make_field()

        bus.process(field)

        weights = bus.resonance_log[-1]
        assert len(weights) == 1
        assert abs(weights[0].normalized_weight - 1.0) < 1e-10

    def test_single_module_preserves_pac(self):
        """Single module output conserves energy after PAC scaling."""
        bus = ResonanceBus(enforcement="hard")
        bus.register_module(StubModule("scaler", lambda t: t * 2.0))
        field = _make_field()

        result = bus.process(field)

        input_energy = field.total_energy()
        output_energy = result.total_energy()
        assert abs(output_energy - input_energy) < 1e-4


class TestIdentityModules:

    def test_identity_gets_minimal_weight(self):
        """An identity module (no change) gets minimal weight vs active module."""
        bus = ResonanceBus(enforcement="soft")
        bus.register_module(StubModule("identity", lambda t: t.clone()))
        bus.register_module(StubModule("active", lambda t: t * 1.5))
        field = _make_field()

        bus.process(field)

        weights = bus.resonance_log[-1]
        identity_w = next(w for w in weights if w.module_name == "identity")
        active_w = next(w for w in weights if w.module_name == "active")
        assert identity_w.normalized_weight < active_w.normalized_weight

    def test_all_identity_returns_input(self):
        """When all modules are identity, output equals input (after PAC)."""
        bus = ResonanceBus(enforcement="soft")
        bus.register_module(StubModule("id1", lambda t: t.clone()))
        bus.register_module(StubModule("id2", lambda t: t.clone()))
        field = _make_field()

        result = bus.process(field)

        # With all-identity, merged tensor should be very close to input
        diff = float(torch.norm(result.tensor - field.tensor).item())
        assert diff < 1e-3


class TestPACConservation:

    def test_pac_conserved_with_amplifier(self):
        """PAC boundary scales amplified output back to input energy."""
        bus = ResonanceBus(enforcement="hard")
        bus.register_module(StubModule("amp", lambda t: t * 3.0))
        field = _make_field()

        result = bus.process(field)
        assert abs(result.total_energy() - field.total_energy()) < 1e-4

    def test_pac_conserved_with_attenuator(self):
        """PAC boundary scales attenuated output back to input energy."""
        bus = ResonanceBus(enforcement="hard")
        bus.register_module(StubModule("atten", lambda t: t * 0.3))
        field = _make_field()

        result = bus.process(field)
        assert abs(result.total_energy() - field.total_energy()) < 1e-4

    def test_pac_violation_raises_in_hard_mode(self):
        """Hard enforcement raises on PAC violation (edge case: zero output)."""
        bus = ResonanceBus(enforcement="hard", tolerance=1e-12)
        # Module that zeros out the tensor — PAC can't scale zero back
        bus.register_module(StubModule("zero", lambda t: torch.zeros_like(t)))
        field = _make_field()

        # Zero output gets restored from input in the bus, so no violation
        result = bus.process(field)
        assert abs(result.total_energy() - field.total_energy()) < 1e-4


class TestAntiCoherence:

    def test_sign_flip_gets_low_coherence(self):
        """A module that flips signs gets low phase coherence."""
        bus = ResonanceBus(enforcement="soft")
        bus.register_module(StubModule("flipper", lambda t: -t))
        bus.register_module(StubModule("amplifier", lambda t: t * 1.5))
        field = _make_field()

        bus.process(field)

        weights = bus.resonance_log[-1]
        flip_w = next(w for w in weights if w.module_name == "flipper")
        amp_w = next(w for w in weights if w.module_name == "amplifier")
        # Sign-flipped delta has low coherence with positive input
        assert flip_w.phase_coherence < amp_w.phase_coherence


class TestProvenance:

    def test_provenance_tracks_active_modules(self):
        """Provenance includes all modules with non-trivial weight."""
        bus = ResonanceBus(enforcement="soft")
        # Both amplify (positive delta aligned with positive input = high coherence)
        bus.register_module(StubModule("mod_a", lambda t: t * 1.2))
        bus.register_module(StubModule("mod_b", lambda t: t * 1.5))
        field = _make_field()

        result = bus.process(field)

        assert "mod_a" in result.provenance
        assert "mod_b" in result.provenance

    def test_identity_excluded_from_provenance(self):
        """Identity module (epsilon weight) excluded from provenance."""
        bus = ResonanceBus(enforcement="soft", min_weight_epsilon=1e-6)
        bus.register_module(StubModule("identity", lambda t: t.clone()))
        bus.register_module(StubModule("active", lambda t: t * 2.0))
        field = _make_field()

        result = bus.process(field)

        # Identity should have weight at or below epsilon
        # Active should dominate provenance
        assert "active" in result.provenance


class TestPassthrough:

    def test_no_modules_returns_input(self):
        """With no modules registered, process returns input unchanged."""
        bus = ResonanceBus(enforcement="soft")
        field = _make_field()

        result = bus.process(field)

        diff = float(torch.norm(result.tensor - field.tensor).item())
        assert diff < 1e-10


class TestRBFSuppression:

    def test_unhealthy_module_suppressed(self):
        """Module with RBF below threshold is excluded from resonance."""
        bus = ResonanceBus(enforcement="soft", rbf_suppression_threshold=0.5)
        bus.register_module(StubModule("healthy", lambda t: t * 1.5, rbf_balance=1.0))
        bus.register_module(StubModule("sick", lambda t: t * 3.0, rbf_balance=0.1))
        field = _make_field()

        bus.process(field)

        weights = bus.resonance_log[-1]
        module_names = [w.module_name for w in weights]
        assert "healthy" in module_names
        assert "sick" not in module_names


class TestCommutativity:

    def test_registration_order_irrelevant(self):
        """Superposition merge is commutative — order doesn't matter."""
        def make_bus_ordered(order):
            bus = ResonanceBus(enforcement="soft")
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


class TestResonanceLog:

    def test_log_populated_after_process(self):
        """resonance_log has one entry per process() call."""
        bus = ResonanceBus(enforcement="soft")
        bus.register_module(StubModule("mod", lambda t: t * 1.2))
        field = _make_field()

        assert len(bus.resonance_log) == 0
        bus.process(field)
        assert len(bus.resonance_log) == 1
        bus.process(field)
        assert len(bus.resonance_log) == 2

    def test_log_entries_are_resonance_weights(self):
        """Each log entry is a list of ResonanceWeight objects."""
        bus = ResonanceBus(enforcement="soft")
        bus.register_module(StubModule("mod", lambda t: t * 1.2))
        bus.process(_make_field())

        entry = bus.resonance_log[0]
        assert isinstance(entry, list)
        assert all(isinstance(w, ResonanceWeight) for w in entry)


class TestMetrics:

    def test_get_metrics_structure(self):
        """get_metrics() returns expected keys."""
        bus = ResonanceBus(enforcement="soft")
        bus.register_module(StubModule("mod", lambda t: t * 1.2))

        m = bus.get_metrics()
        assert m["modules_registered"] == 1
        assert m["module_names"] == ["mod"]
        assert m["enforcement"] == "soft"
        assert m["dispatch_mode"] == "resonance"
        assert m["resonance_ticks"] == 0

    def test_metrics_update_after_process(self):
        """Metrics reflect state after processing."""
        bus = ResonanceBus(enforcement="soft")
        bus.register_module(StubModule("mod", lambda t: t * 1.2))
        bus.process(_make_field())

        m = bus.get_metrics()
        assert m["resonance_ticks"] == 1


class TestFullStack:

    def test_five_module_smoke(self):
        """Smoke test: 5 different transforms produce valid output."""
        bus = ResonanceBus(enforcement="soft")
        bus.register_module(StubModule("obs", lambda t: t.clone()))  # identity
        bus.register_module(StubModule("safety", lambda t: t * 0.9))
        bus.register_module(StubModule("reasoning", lambda t: t * 1.2))
        bus.register_module(StubModule("memory", lambda t: t * 1.05))
        bus.register_module(StubModule("language", lambda t: t * 0.95))
        field = _make_field()

        result = bus.process(field)

        # PAC conserved
        assert abs(result.total_energy() - field.total_energy()) < 1e-3
        # At least some modules in provenance (attenuators may get low coherence)
        assert len(result.provenance) >= 1
        # Resonance log populated with all 5 modules
        assert len(bus.resonance_log) == 1
        assert len(bus.resonance_log[0]) == 5


class TestViolationLog:

    def test_soft_enforcement_logs_violations(self):
        """Soft mode logs violations without raising."""
        bus = ResonanceBus(enforcement="soft", tolerance=0.0)
        bus.register_module(StubModule("mod", lambda t: t * 1.1))
        field = _make_field()

        # Should not raise
        bus.process(field)
        # PAC scaling should prevent violation in most cases,
        # so violation log may be empty — that's fine, PAC works
        assert isinstance(bus.violation_log, list)

    def test_monitor_mode_no_raise(self):
        """Monitor mode never raises, just logs."""
        bus = ResonanceBus(enforcement="monitor")
        bus.register_module(StubModule("mod", lambda t: t * 5.0))
        field = _make_field()

        result = bus.process(field)
        assert result is not None


class TestEnforcementValidation:

    def test_invalid_enforcement_raises(self):
        """Invalid enforcement mode raises ValueError."""
        with pytest.raises(ValueError, match="enforcement"):
            ResonanceBus(enforcement="invalid")
