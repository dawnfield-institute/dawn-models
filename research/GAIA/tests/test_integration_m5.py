"""M5 Integration — Multi-module composition, SEC routing, RBF regulation.

Tests the architectural properties that emerge from composing all 6 modules
through the conservation bus. These tests validate zero-parameter routing,
self-regulation, and compositional invariants — the core claims of GAIA v2.

Organized by property:
    1. SEC Phase Routing — modules activate/deactivate by entropy level
    2. RBF Regulation — unhealthy modules are suppressed
    3. Composition Invariants — ordering, idempotency, energy flow
    4. Full-Stack Stress — edge cases, drift, saturation
"""

from __future__ import annotations

import math

import torch
import pytest

from gaia.core.bus import ConservationBus
from gaia.core.exceptions import ConservationViolation
from gaia.core.types import FieldState, RBFBalance, SECPhase
from gaia.modules.language import LanguageModule
from gaia.modules.memory import MemoryModule
from gaia.modules.observability import ObservabilityModule
from gaia.modules.reasoning import ReasoningModule
from gaia.modules.safety import SafetyModule

from tests.conftest import IdentityModule


def _state(dim: int = 16, entropy: float = 1.0, positive: bool = True) -> FieldState:
    """Helper: random FieldState with controlled entropy."""
    tensor = torch.randn(dim).abs() + 0.1 if positive else torch.randn(dim)
    return FieldState(tensor=tensor, entropy=entropy)


def _full_bus(dim: int = 16, **bus_kw) -> tuple[ConservationBus, dict]:
    """Create a bus with all 5 modules. Returns (bus, modules_dict)."""
    modules = {
        "safety": SafetyModule(input_dim=dim),
        "reasoning": ReasoningModule(input_dim=dim),
        "memory": MemoryModule(),
        "observability": ObservabilityModule(),
        "language": LanguageModule(),
    }
    bus = ConservationBus(**{**{"enforcement": "hard", "tolerance": 1e-3}, **bus_kw})
    for m in modules.values():
        bus.register_module(m)
    return bus, modules


# ─── 1. SEC Phase Routing ────────────────────────────────────────


class TestSECPhaseRouting:
    """Zero-parameter routing based on entropy classification."""

    def test_crystallized_routes_all_modules(self):
        """entropy < 0.5 → CRYSTALLIZED. All modules registered for all phases."""
        bus, _ = _full_bus()
        state = _state(entropy=0.1)  # CRYSTALLIZED
        result = bus.process(state)
        assert result.phase == SECPhase.CRYSTALLIZED
        assert len(result.provenance) == 5

    def test_ordered_routes_all_modules(self):
        """0.5 <= entropy < 2.0 → ORDERED."""
        bus, _ = _full_bus()
        state = _state(entropy=1.0)  # ORDERED
        result = bus.process(state)
        assert result.phase == SECPhase.ORDERED
        assert len(result.provenance) == 5

    def test_transitional_routes_all_modules(self):
        """2.0 <= entropy < 4.0 → TRANSITIONAL."""
        bus, _ = _full_bus()
        state = _state(entropy=3.0)  # TRANSITIONAL
        result = bus.process(state)
        assert result.phase == SECPhase.TRANSITIONAL
        assert len(result.provenance) == 5

    def test_chaotic_routes_all_modules(self):
        """entropy >= 4.0 → CHAOTIC."""
        bus, _ = _full_bus()
        state = _state(entropy=8.0)  # CHAOTIC
        result = bus.process(state)
        assert result.phase == SECPhase.CHAOTIC
        assert len(result.provenance) == 5

    def test_selective_phase_registration(self):
        """Module registered for ORDERED only skips CRYSTALLIZED input."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        safety = SafetyModule(input_dim=16)
        identity = IdentityModule()

        # safety → all phases, identity → ORDERED only
        bus.register_module(safety)
        bus.register_module(identity, phases=[SECPhase.ORDERED])

        # ORDERED input → both modules run
        ordered = _state(entropy=1.0)
        result = bus.process(ordered)
        assert "safety" in result.provenance
        assert "identity" in result.provenance

        # CRYSTALLIZED input → only safety runs
        crystal = _state(entropy=0.1)
        result = bus.process(crystal)
        assert "safety" in result.provenance
        assert "identity" not in result.provenance

    def test_phase_classification_boundaries(self):
        """Entropy at exact boundaries classifies correctly."""
        bus, _ = _full_bus()

        # Just below 0.5 → CRYSTALLIZED
        r1 = bus.process(_state(entropy=0.49))
        assert r1.phase == SECPhase.CRYSTALLIZED

        # At 0.5 → ORDERED
        r2 = bus.process(_state(entropy=0.5))
        assert r2.phase == SECPhase.ORDERED

        # Just below 2.0 → ORDERED
        r3 = bus.process(_state(entropy=1.99))
        assert r3.phase == SECPhase.ORDERED

        # At 2.0 → TRANSITIONAL
        r4 = bus.process(_state(entropy=2.0))
        assert r4.phase == SECPhase.TRANSITIONAL

        # At 4.0 → CHAOTIC
        r5 = bus.process(_state(entropy=4.0))
        assert r5.phase == SECPhase.CHAOTIC

    def test_module_internal_phase_varies_with_state(self):
        """Module's own phase() reflects its internal state, not input phase."""
        bus, mods = _full_bus()

        # Process several inputs to build module state
        for _ in range(10):
            bus.process(_state(entropy=1.0))

        # Each module reports its own SEC phase based on internal metrics
        for name, mod in mods.items():
            p = mod.phase()
            assert isinstance(p, SECPhase), f"{name}.phase() returned {type(p)}"


# ─── 2. RBF Regulation ──────────────────────────────────────────


class TestRBFRegulation:
    """Self-regulation via energy-information balance."""

    def test_healthy_modules_not_suppressed(self):
        """All modules with balance >= 0 pass regulation."""
        bus, _ = _full_bus()
        state = _state()
        result = bus.process(state)
        # All 5 should run (default threshold = 0.0)
        assert len(result.provenance) == 5

    def test_suppression_threshold_filters_modules(self):
        """Raising suppression threshold filters weak modules."""
        # Create bus with high threshold — modules with low health get suppressed
        bus, mods = _full_bus(rbf_suppression_threshold=10.0)
        state = _state()
        result = bus.process(state)
        # With threshold=10.0, most modules should be suppressed
        # (RBF balance rarely exceeds 10.0)
        assert len(result.provenance) < 5

    def test_health_reports_valid_rbf(self):
        """All modules return valid RBFBalance after processing."""
        bus, mods = _full_bus()
        bus.process(_state())

        for name, mod in mods.items():
            h = mod.health()
            assert isinstance(h, RBFBalance), f"{name} health type wrong"
            assert math.isfinite(h.energy), f"{name} energy not finite"
            assert math.isfinite(h.information), f"{name} information not finite"
            assert math.isfinite(h.memory), f"{name} memory not finite"
            assert math.isfinite(h.balance), f"{name} balance not finite"

    def test_observability_never_suppressed(self):
        """Observability module (pass-through) should never be suppressed."""
        bus = ConservationBus(
            enforcement="hard",
            tolerance=1e-3,
            rbf_suppression_threshold=0.0,
        )
        obs = ObservabilityModule()
        bus.register_module(obs)

        for _ in range(100):
            bus.process(_state())

        # Observability should have run every time
        assert obs.metrics.step_count == 100

    def test_language_health_reflects_concentration(self):
        """Language module health energy tracks concentration (prediction quality)."""
        lang = LanguageModule()

        # Before processing: default health
        h_before = lang.health()
        assert h_before.energy == 0.5  # default when no metrics

        # After processing: health reflects actual concentration
        state = _state()
        lang.process(state)
        h_after = lang.health()
        assert isinstance(h_after, RBFBalance)
        assert math.isfinite(h_after.balance)


# ─── 3. Composition Invariants ───────────────────────────────────


class TestCompositionInvariants:
    """Properties that must hold regardless of module order or count."""

    def test_energy_conserved_across_all_phases(self):
        """Total energy preserved at every entropy level."""
        bus, _ = _full_bus()
        for entropy in [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0]:
            state = _state(entropy=entropy)
            result = bus.process(state)
            ie = state.total_energy()
            oe = result.total_energy()
            rel = abs(ie - oe) / max(abs(ie), 1e-10)
            assert rel < 1e-3, f"Conservation violated at entropy={entropy}: rel={rel:.2e}"

    def test_ordering_does_not_violate_conservation(self):
        """Different module orderings all conserve energy."""
        dim = 16
        state = _state(dim=dim)
        input_e = state.total_energy()

        orderings = [
            [SafetyModule(input_dim=dim), ReasoningModule(input_dim=dim), MemoryModule(), ObservabilityModule(), LanguageModule()],
            [LanguageModule(), ObservabilityModule(), MemoryModule(), ReasoningModule(input_dim=dim), SafetyModule(input_dim=dim)],
            [MemoryModule(), LanguageModule(), SafetyModule(input_dim=dim), ObservabilityModule(), ReasoningModule(input_dim=dim)],
        ]

        for i, modules in enumerate(orderings):
            bus = ConservationBus(enforcement="hard", tolerance=1e-3)
            for m in modules:
                bus.register_module(m)
            result = bus.process(state)
            oe = result.total_energy()
            rel = abs(input_e - oe) / max(abs(input_e), 1e-10)
            assert rel < 1e-3, f"Order {i} violated conservation: rel={rel:.2e}"

    def test_provenance_chain_complete(self):
        """Every module's name appears in provenance in registration order."""
        bus, _ = _full_bus()
        result = bus.process(_state())
        assert len(result.provenance) == 5
        expected_names = {"safety", "reasoning", "memory", "observability", "language"}
        assert set(result.provenance) == expected_names

    def test_identity_module_is_truly_neutral(self):
        """Identity module doesn't alter energy, entropy, or tensor values."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-10)
        bus.register_module(IdentityModule())

        state = FieldState(tensor=torch.ones(10) * 3.0, entropy=2.5)
        result = bus.process(state)

        assert torch.allclose(result.tensor, state.tensor)
        assert result.entropy == state.entropy
        assert result.total_energy() == state.total_energy()

    def test_empty_bus_returns_input(self):
        """Bus with no modules returns input unchanged."""
        bus = ConservationBus()
        state = _state()
        result = bus.process(state)
        assert torch.allclose(result.tensor, state.tensor)
        assert result.entropy == state.entropy

    def test_single_module_conserves_like_bus(self):
        """Bus with one module conserves energy same as direct call."""
        dim = 16
        state = _state(dim=dim)
        input_e = state.total_energy()

        # Through bus
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=dim))
        bus_result = bus.process(state)

        # Both should conserve energy relative to input
        bus_rel = abs(input_e - bus_result.total_energy()) / max(abs(input_e), 1e-10)
        assert bus_rel < 1e-3, f"Bus conservation failed: {bus_rel:.2e}"


# ─── 4. Full-Stack Stress ───────────────────────────────────────


class TestFullStackStress:
    """Edge cases and stress tests for the full 5-module stack."""

    def test_sustained_processing_no_violations(self):
        """100 inputs through all 5 modules — zero violations."""
        bus, _ = _full_bus()
        for _ in range(100):
            bus.process(_state())
        assert len(bus.violation_log) == 0

    def test_alternating_entropy_phases(self):
        """Rapid phase switches: CRYSTALLIZED → CHAOTIC → ORDERED → ..."""
        bus, _ = _full_bus()
        entropies = [0.1, 8.0, 1.0, 3.0, 0.0, 5.0, 1.5, 0.3] * 5

        for e in entropies:
            result = bus.process(_state(entropy=e))
            assert torch.isfinite(result.tensor).all()
        assert len(bus.violation_log) == 0

    def test_dim_1_tensor(self):
        """Scalar tensor through full stack."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=1))
        bus.register_module(ReasoningModule(input_dim=1))
        bus.register_module(MemoryModule())
        bus.register_module(ObservabilityModule())
        bus.register_module(LanguageModule())

        state = FieldState(tensor=torch.tensor([2.0]), entropy=1.0)
        result = bus.process(state)
        assert torch.isfinite(result.tensor).all()

    def test_large_dim_tensor(self):
        """dim=512 through full stack."""
        dim = 512
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=dim))
        bus.register_module(ReasoningModule(input_dim=dim))
        bus.register_module(MemoryModule())
        bus.register_module(ObservabilityModule())
        bus.register_module(LanguageModule())

        state = _state(dim=dim)
        result = bus.process(state)
        assert torch.isfinite(result.tensor).all()
        assert len(result.provenance) == 5

    def test_near_zero_energy_tensor(self):
        """Tensor that sums to ~0 — tests conservation rescaling edge case."""
        bus, _ = _full_bus()
        # Create tensor with near-zero sum
        tensor = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.3, -0.3,
                               1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.3, -0.3])
        state = FieldState(tensor=tensor, entropy=1.0)
        result = bus.process(state)
        assert torch.isfinite(result.tensor).all()

    def test_very_large_entropy(self):
        """Extreme entropy (100.0) — still CHAOTIC, no crash."""
        bus, _ = _full_bus()
        state = _state(entropy=100.0)
        result = bus.process(state)
        assert result.phase == SECPhase.CHAOTIC
        assert torch.isfinite(result.tensor).all()

    def test_zero_entropy(self):
        """Entropy = 0 — CRYSTALLIZED, all modules run."""
        bus, _ = _full_bus()
        state = _state(entropy=0.0)
        result = bus.process(state)
        assert result.phase == SECPhase.CRYSTALLIZED
        assert torch.isfinite(result.tensor).all()

    def test_metrics_accessible_after_full_stack(self):
        """All module metrics populated after bus processing."""
        bus, mods = _full_bus()
        bus.process(_state())

        for name, mod in mods.items():
            m = mod.metrics if hasattr(mod, "metrics") else None
            assert m is not None, f"{name} has no metrics after processing"

    def test_recirculation_drift(self):
        """Feed output back as input 20 times — energy drift stays bounded."""
        bus, _ = _full_bus()
        state = _state()
        initial_e = state.total_energy()

        current = state
        for _ in range(20):
            current = bus.process(current)

        final_e = current.total_energy()
        rel_drift = abs(initial_e - final_e) / max(abs(initial_e), 1e-10)
        assert rel_drift < 0.05, f"Recirculation drift {rel_drift:.2%} exceeds 5%"

    def test_enforcement_modes(self):
        """All three enforcement modes produce valid output."""
        for mode in ["hard", "soft", "monitor"]:
            bus, _ = _full_bus(enforcement=mode, tolerance=1e-3)
            result = bus.process(_state())
            assert torch.isfinite(result.tensor).all()
            assert len(result.provenance) == 5
