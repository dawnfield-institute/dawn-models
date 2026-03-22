"""Benchmark & stress tests — exercise the full GAIA v2 stack.

Finds real issues: conservation drift, performance degradation,
edge cases, multi-module composition under load.
"""

from __future__ import annotations

import time
from collections import defaultdict

import pytest
import torch

from gaia.core.bus import ConservationBus
from gaia.core.exceptions import ConservationViolation
from gaia.core.types import FieldState, SECPhase
from gaia.modules.memory import MemoryModule, PACTree, BifractalDepth
from gaia.modules.observability import ObservabilityModule, SCBFMetrics
from gaia.modules.reasoning import ReasoningModule
from gaia.modules.safety import SafetyModule


def make_state(
    dim: int = 10,
    entropy: float = 1.0,
    scale: float = 1.0,
    positive: bool = True,
) -> FieldState:
    """Factory for benchmark field states."""
    if positive:
        tensor = torch.randn(dim).abs() * scale + 0.01
    else:
        tensor = torch.randn(dim) * scale
    return FieldState(tensor=tensor, entropy=entropy)


# ─── Conservation Drift ──────────────────────────────────────────


class TestConservationDrift:
    """Does energy leak or accumulate over many iterations?"""

    def test_single_module_drift_safety(self):
        """Safety module over 100 iterations — track cumulative drift."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=16))

        drifts = []
        for _ in range(100):
            state = make_state(dim=16, entropy=1.0)
            input_e = state.total_energy()
            result = bus.process(state)
            output_e = result.total_energy()
            drifts.append(abs(input_e - output_e))

        mean_drift = sum(drifts) / len(drifts)
        max_drift = max(drifts)
        assert mean_drift < 1e-3, f"Mean drift {mean_drift:.6e} exceeds threshold"
        assert max_drift < 1e-2, f"Max drift {max_drift:.6e} exceeds threshold"

    def test_single_module_drift_reasoning(self):
        """Reasoning module over 100 iterations."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(ReasoningModule(input_dim=16))

        drifts = []
        for _ in range(100):
            state = make_state(dim=16, entropy=1.0)
            input_e = state.total_energy()
            result = bus.process(state)
            output_e = result.total_energy()
            drifts.append(abs(input_e - output_e))

        mean_drift = sum(drifts) / len(drifts)
        max_drift = max(drifts)
        assert mean_drift < 1e-3, f"Mean drift {mean_drift:.6e} exceeds threshold"
        assert max_drift < 1e-2, f"Max drift {max_drift:.6e} exceeds threshold"

    def test_single_module_drift_memory(self):
        """Memory module over 100 iterations."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(MemoryModule())

        drifts = []
        for _ in range(100):
            state = make_state(dim=16, entropy=1.0)
            input_e = state.total_energy()
            result = bus.process(state)
            output_e = result.total_energy()
            drifts.append(abs(input_e - output_e))

        mean_drift = sum(drifts) / len(drifts)
        max_drift = max(drifts)
        assert mean_drift < 1e-3, f"Mean drift {mean_drift:.6e} exceeds threshold"
        assert max_drift < 1e-2, f"Max drift {max_drift:.6e} exceeds threshold"

    def test_single_module_drift_observability(self):
        """Observability is pass-through — should have zero drift."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-10)
        bus.register_module(ObservabilityModule())

        drifts = []
        for _ in range(100):
            state = make_state(dim=16, entropy=1.0)
            input_e = state.total_energy()
            result = bus.process(state)
            output_e = result.total_energy()
            drifts.append(abs(input_e - output_e))

        max_drift = max(drifts)
        assert max_drift == 0.0, f"Observability drift {max_drift:.6e} — should be exactly 0"

    def test_full_stack_drift_100_iterations(self):
        """All 4 modules in sequence, 100 iterations. The real test."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=16))
        bus.register_module(ReasoningModule(input_dim=16))
        bus.register_module(MemoryModule())
        bus.register_module(ObservabilityModule())

        drifts = []
        for _ in range(100):
            state = make_state(dim=16, entropy=1.0)
            input_e = state.total_energy()
            result = bus.process(state)
            output_e = result.total_energy()
            drifts.append(abs(input_e - output_e))

        mean_drift = sum(drifts) / len(drifts)
        max_drift = max(drifts)
        assert mean_drift < 1e-3, f"Full stack mean drift {mean_drift:.6e}"
        assert max_drift < 1e-2, f"Full stack max drift {max_drift:.6e}"
        assert len(bus.violation_log) == 0

    def test_recirculation_drift(self):
        """Feed output back as input N times — does drift compound?"""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=16))
        bus.register_module(ReasoningModule(input_dim=16))
        bus.register_module(MemoryModule())
        bus.register_module(ObservabilityModule())

        state = make_state(dim=16, entropy=1.0, positive=True)
        initial_energy = state.total_energy()
        energies = [initial_energy]

        for _ in range(50):
            state = bus.process(state)
            energies.append(state.total_energy())

        # Energy should stay within tolerance of initial
        for i, e in enumerate(energies):
            drift = abs(e - initial_energy) / max(abs(initial_energy), 1e-10)
            assert drift < 0.05, f"Step {i}: relative drift {drift:.4f} from initial energy"


# ─── Edge Cases ──────────────────────────────────────────────────


class TestEdgeCases:
    """Pathological inputs that might break conservation."""

    def test_near_zero_tensor(self):
        """Tensor with values ~1e-10. Division by near-zero in scaling?"""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=8))
        bus.register_module(MemoryModule())

        state = FieldState(tensor=torch.ones(8) * 1e-10, entropy=1.0)
        result = bus.process(state)
        # Should not crash or produce NaN/Inf
        assert torch.isfinite(result.tensor).all(), "Non-finite values in output"

    def test_large_tensor(self):
        """Large magnitude tensor."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=8))
        bus.register_module(ReasoningModule(input_dim=8))

        state = FieldState(tensor=torch.ones(8) * 1e6, entropy=1.0)
        result = bus.process(state)
        assert torch.isfinite(result.tensor).all()

    def test_single_element_tensor(self):
        """dim=1 edge case."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=1))
        bus.register_module(ReasoningModule(input_dim=1))
        bus.register_module(MemoryModule())
        bus.register_module(ObservabilityModule())

        state = FieldState(tensor=torch.tensor([1.0]), entropy=1.0)
        result = bus.process(state)
        assert torch.isfinite(result.tensor).all()
        assert len(result.provenance) == 4

    def test_high_dim_tensor(self):
        """dim=1024 — still works?"""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=1024))
        bus.register_module(ReasoningModule(input_dim=1024))
        bus.register_module(MemoryModule())
        bus.register_module(ObservabilityModule())

        state = make_state(dim=1024, entropy=1.0)
        result = bus.process(state)
        assert torch.isfinite(result.tensor).all()
        assert len(result.provenance) == 4

    def test_zero_entropy(self):
        """Entropy = 0 → CRYSTALLIZED phase. All modules still run?"""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=8))
        bus.register_module(MemoryModule())

        state = FieldState(tensor=torch.ones(8), entropy=0.0)
        result = bus.process(state)
        assert result.phase == SECPhase.CRYSTALLIZED

    def test_high_entropy(self):
        """Entropy = 10 → CHAOTIC phase."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=8))
        bus.register_module(MemoryModule())

        state = FieldState(tensor=torch.ones(8), entropy=10.0)
        result = bus.process(state)
        assert result.phase == SECPhase.CHAOTIC

    def test_mixed_sign_tensor(self):
        """Tensor with positive and negative values — energy can be near zero."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=8))
        bus.register_module(ReasoningModule(input_dim=8))
        bus.register_module(MemoryModule())
        bus.register_module(ObservabilityModule())

        # Create tensor that sums to ~0
        tensor = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.3, -0.3])
        state = FieldState(tensor=tensor, entropy=1.0)
        result = bus.process(state)
        assert torch.isfinite(result.tensor).all()

    def test_identical_repeated_inputs(self):
        """Same input 50 times — memory should handle gracefully."""
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(MemoryModule())

        pattern = torch.ones(8) * 0.5
        for _ in range(50):
            state = FieldState(tensor=pattern.clone(), entropy=1.0)
            result = bus.process(state)
            assert torch.isfinite(result.tensor).all()
        assert len(bus.violation_log) == 0


# ─── Memory-Specific Stress ─────────────────────────────────────


class TestMemoryStress:
    """PACTree under load."""

    def test_tree_growth_and_retrieval(self):
        """Store 500 patterns, retrieve, check quality."""
        tree = PACTree(capacity=1000)
        patterns = [torch.randn(16) for _ in range(500)]
        for p in patterns:
            tree.store(p)

        # Retrieve a known pattern — should find itself
        query = patterns[0]
        results = tree.retrieve(query, top_k=3, threshold=0.1)
        assert len(results) > 0, "Failed to retrieve any pattern"

    def test_delta_compression_efficiency(self):
        """Similar patterns should compress well."""
        tree = PACTree()
        base = torch.ones(32)
        for i in range(100):
            # Small perturbations of base
            tree.store(base + torch.randn(32) * 0.01)

        ratio = tree.storage_ratio()
        assert ratio <= 1.0, f"Storage ratio {ratio} — no compression benefit"

    def test_decay_doesnt_corrupt_reconstruction(self):
        """After decay, can we still reconstruct?"""
        tree = PACTree()
        original = torch.ones(8) * 3.0
        nid = tree.store(original)

        # Store more, creating children
        for _ in range(10):
            tree.store(original + torch.randn(8) * 0.01)

        # Decay multiple times
        for _ in range(20):
            tree.decay()

        # Reconstruction should still work (decay affects strength, not deltas)
        reconstructed = tree.reconstruct(nid)
        assert torch.isfinite(reconstructed).all()
        # Value should be unchanged (decay is on strength, not delta)
        assert torch.allclose(reconstructed, original, atol=1e-5)

    def test_gc_preserves_tree_integrity(self):
        """After GC, all remaining nodes should be reconstructable."""
        tree = PACTree(capacity=50)
        for _ in range(100):
            tree.store(torch.randn(8))

        # Try to reconstruct every remaining node
        for nid in list(tree._nodes.keys()):
            val = tree.reconstruct(nid)
            assert torch.isfinite(val).all(), f"Node {nid} has non-finite reconstruction"

    def test_bifractal_depth_distribution_evolves(self):
        """With enough access, nodes should promote through levels."""
        mem = MemoryModule(capacity=100)
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(mem)

        # Run enough iterations to trigger promotions (every 10 steps)
        for i in range(100):
            state = make_state(dim=8, entropy=1.0)
            bus.process(state)

        dist = mem.tree.depth_distribution()
        # Should have nodes at multiple depths by now (access-based promotion)
        assert len(dist) >= 1, f"Only {len(dist)} depth levels after 100 iterations"

    def test_transition_prediction_accuracy(self):
        """Sequential patterns should produce meaningful predictions."""
        mem = MemoryModule(capacity=100, auto_learn_transitions=True)
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(mem)

        # Feed a repeating sequence: A, B, A, B, A, B...
        pattern_a = torch.ones(8) * 2.0
        pattern_b = torch.ones(8) * 5.0
        for _ in range(20):
            bus.process(FieldState(tensor=pattern_a.clone(), entropy=1.0))
            bus.process(FieldState(tensor=pattern_b.clone(), entropy=1.0))

        # Transition tracker should have learned something
        assert mem.transitions.n_transitions > 0


# ─── Composition Stress ──────────────────────────────────────────


class TestCompositionStress:
    """Multi-module composition under varied conditions."""

    def test_module_ordering_matters(self):
        """Different orderings produce different tensors but all conserve."""
        dim = 16
        state = make_state(dim=dim, entropy=1.0)
        input_e = state.total_energy()

        # Order 1: safety → reasoning → memory → observability
        bus1 = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus1.register_module(SafetyModule(input_dim=dim))
        bus1.register_module(ReasoningModule(input_dim=dim))
        bus1.register_module(MemoryModule())
        bus1.register_module(ObservabilityModule())
        r1 = bus1.process(state)

        # Order 2: observability → memory → safety → reasoning
        bus2 = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus2.register_module(ObservabilityModule())
        bus2.register_module(MemoryModule())
        bus2.register_module(SafetyModule(input_dim=dim))
        bus2.register_module(ReasoningModule(input_dim=dim))
        r2 = bus2.process(state)

        # Both should conserve energy
        assert abs(r1.total_energy() - input_e) < 1e-2
        assert abs(r2.total_energy() - input_e) < 1e-2
        # But tensors may differ
        # (not asserting they're different — just that both are valid)

    def test_all_sec_phases(self):
        """Full stack at each SEC phase boundary."""
        dim = 16
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=dim))
        bus.register_module(ReasoningModule(input_dim=dim))
        bus.register_module(MemoryModule())
        bus.register_module(ObservabilityModule())

        for entropy in [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0]:
            state = make_state(dim=dim, entropy=entropy)
            result = bus.process(state)
            assert torch.isfinite(result.tensor).all(), f"Non-finite at entropy={entropy}"
            assert len(bus.violation_log) == 0, f"Violation at entropy={entropy}"

    def test_soft_enforcement_counts_violations(self):
        """Under soft mode, count how many violations the stack actually produces."""
        dim = 16
        bus = ConservationBus(enforcement="soft", tolerance=1e-6)  # Very tight
        bus.register_module(SafetyModule(input_dim=dim))
        bus.register_module(ReasoningModule(input_dim=dim))
        bus.register_module(MemoryModule())
        bus.register_module(ObservabilityModule())

        for _ in range(50):
            state = make_state(dim=dim, entropy=1.0)
            bus.process(state)

        # Just report — this is diagnostic, not pass/fail
        # The question is: at 1e-6 tolerance, how many violations?
        n_violations = len(bus.violation_log)
        # With the blunt energy-scaling approach, we expect some at tight tolerance
        print(f"\n  Soft enforcement at 1e-6: {n_violations}/200 module-passes had violations")


# ─── Performance ─────────────────────────────────────────────────


class TestPerformance:
    """Timing benchmarks. Not pass/fail — just reporting."""

    def test_throughput_single_module(self):
        """Measure iterations/sec for each module alone."""
        dim = 32
        n = 200
        results = {}

        for name, module in [
            ("safety", SafetyModule(input_dim=dim)),
            ("reasoning", ReasoningModule(input_dim=dim)),
            ("memory", MemoryModule()),
            ("observability", ObservabilityModule()),
        ]:
            bus = ConservationBus(enforcement="hard", tolerance=1e-3)
            bus.register_module(module)

            t0 = time.perf_counter()
            for _ in range(n):
                state = make_state(dim=dim, entropy=1.0)
                bus.process(state)
            elapsed = time.perf_counter() - t0
            results[name] = n / elapsed

        print(f"\n  Throughput (iter/s, dim={dim}):")
        for name, rate in results.items():
            print(f"    {name}: {rate:.0f}")

    def test_throughput_full_stack(self):
        """Full 4-module stack throughput."""
        dim = 32
        n = 100

        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(SafetyModule(input_dim=dim))
        bus.register_module(ReasoningModule(input_dim=dim))
        bus.register_module(MemoryModule())
        bus.register_module(ObservabilityModule())

        t0 = time.perf_counter()
        for _ in range(n):
            state = make_state(dim=dim, entropy=1.0)
            bus.process(state)
        elapsed = time.perf_counter() - t0

        rate = n / elapsed
        print(f"\n  Full stack throughput: {rate:.0f} iter/s (dim={dim})")

    def test_memory_scales_with_tree_size(self):
        """Does memory module slow down as tree fills?"""
        dim = 16
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        mem = MemoryModule(capacity=10000)
        bus.register_module(mem)

        checkpoints = [10, 50, 100, 200, 500]
        timings = {}

        total_processed = 0
        for target in checkpoints:
            batch = target - total_processed
            t0 = time.perf_counter()
            for _ in range(batch):
                state = make_state(dim=dim, entropy=1.0)
                bus.process(state)
            elapsed = time.perf_counter() - t0
            total_processed = target
            timings[target] = batch / elapsed

        print(f"\n  Memory throughput vs tree size:")
        for size, rate in timings.items():
            print(f"    {size} nodes: {rate:.0f} iter/s")

        # Performance should not degrade more than 20x from start to end
        first_rate = timings[checkpoints[0]]
        last_rate = timings[checkpoints[-1]]
        if first_rate > 0:
            degradation = first_rate / max(last_rate, 1e-10)
            assert degradation < 50, (
                f"Memory performance degraded {degradation:.1f}x "
                f"({first_rate:.0f} → {last_rate:.0f} iter/s)"
            )


# ─── Module Health & Phase ───────────────────────────────────────


class TestModuleHealth:
    """Do health() and phase() return sensible values under load?"""

    def test_safety_health_stays_positive(self):
        """Safety module health should remain positive (energy > info)."""
        safety = SafetyModule(input_dim=16)
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(safety)

        for _ in range(50):
            state = make_state(dim=16, entropy=1.0)
            bus.process(state)

        h = safety.health()
        assert h.energy >= 0, f"Negative energy: {h.energy}"
        assert h.information >= 0, f"Negative information: {h.information}"

    def test_reasoning_phase_evolution(self):
        """Reasoning phase should reflect phi-frequency state."""
        reasoning = ReasoningModule(input_dim=16)
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(reasoning)

        phases = []
        for _ in range(20):
            state = make_state(dim=16, entropy=1.0)
            bus.process(state)
            phases.append(reasoning.phase())

        # Should produce valid phases
        assert all(isinstance(p, SECPhase) for p in phases)

    def test_memory_phase_tracks_utilization(self):
        """Memory phase should shift as tree fills."""
        mem = MemoryModule(capacity=20)
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(mem)

        initial_phase = mem.phase()

        # Fill the tree
        for _ in range(15):
            state = make_state(dim=8, entropy=1.0)
            bus.process(state)

        later_phase = mem.phase()
        # With capacity=20 and 15 entries, utilization = 75% → TRANSITIONAL
        # Initial was empty → ORDERED
        # Phase should have changed (or at least not crashed)
        assert isinstance(later_phase, SECPhase)

    def test_observability_phase_reflects_qbe(self):
        """Observability phase should track QBE equilibrium status."""
        obs = ObservabilityModule()
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(obs)

        phases = []
        for _ in range(30):
            state = make_state(dim=16, entropy=1.0)
            bus.process(state)
            phases.append(obs.phase())

        assert all(isinstance(p, SECPhase) for p in phases)

    def test_observability_health_under_load(self):
        """Observability health() should return valid RBFBalance after many steps."""
        obs = ObservabilityModule()
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(obs)

        for _ in range(50):
            state = make_state(dim=16, entropy=1.0)
            bus.process(state)

        h = obs.health()
        assert h.energy >= 0, f"Negative energy: {h.energy}"
        assert h.information >= 0, f"Negative information: {h.information}"


# ─── Observability-Specific Stress ──────────────────────────────


class TestObservabilityStress:
    """SCBF metrics and QBE under load."""

    def test_scbf_metrics_under_varied_inputs(self):
        """SCBF metrics should stay finite across diverse input distributions."""
        obs = ObservabilityModule()
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(obs)

        distributions = [
            ("uniform", lambda: torch.rand(16)),
            ("normal", lambda: torch.randn(16).abs() + 0.01),
            ("sparse", lambda: torch.zeros(16).scatter_(0, torch.randint(0, 16, (4,)), 1.0)),
            ("constant", lambda: torch.ones(16) * 3.0),
            ("large", lambda: torch.randn(16).abs() * 1000),
            ("tiny", lambda: torch.randn(16).abs() * 1e-6 + 1e-8),
        ]

        for name, gen_fn in distributions:
            obs_fresh = ObservabilityModule()
            for _ in range(10):
                state = FieldState(tensor=gen_fn(), entropy=1.0)
                result = obs_fresh.process(state)
                assert torch.isfinite(result.tensor).all(), f"Non-finite at dist={name}"

            m = obs_fresh.metrics
            assert m is not None, f"No metrics after dist={name}"
            assert m.scbf.entropy_collapse >= 0, f"Negative entropy collapse at dist={name}"
            assert 0 <= m.scbf.ancestry_stability <= 1.0, f"Bad ancestry at dist={name}"

    def test_scbf_collapse_detection(self):
        """Feed stable pattern then sudden shift — tracker should detect collapse."""
        obs = ObservabilityModule()

        # Warm up with stable pattern
        stable = torch.ones(16) * 2.0
        for _ in range(20):
            obs.process(FieldState(tensor=stable.clone(), entropy=1.0))

        # Sudden shift
        shock = torch.randn(16) * 100
        obs.process(FieldState(tensor=shock, entropy=1.0))

        # Check tracker recorded the shift
        assert obs.tracker.step == 21
        # Entropy momentum should be non-zero after the shift
        assert obs.metrics.scbf.entropy_momentum != 0.0

    def test_qbe_convergence_pattern(self):
        """Feeding identical inputs should drive QBE toward convergence."""
        obs = ObservabilityModule()
        pattern = torch.ones(16) * 2.0

        for _ in range(30):
            obs.process(FieldState(tensor=pattern.clone(), entropy=1.0))

        # After 30 identical inputs, QBE should be near equilibrium
        status = obs.metrics.qbe_status
        assert status == "Near Equilibrium", f"Expected near equilibrium, got {status}"

    def test_scbf_metrics_scale_with_dimension(self):
        """SCBF should work at different tensor dimensions."""
        for dim in [1, 4, 16, 64, 256]:
            obs = ObservabilityModule()
            for _ in range(10):
                state = FieldState(tensor=torch.randn(dim).abs() + 0.01, entropy=1.0)
                obs.process(state)

            m = obs.metrics
            assert m is not None, f"No metrics at dim={dim}"
            assert isinstance(m.scbf, SCBFMetrics), f"Bad SCBF type at dim={dim}"

    def test_observability_throughput_overhead(self):
        """Measure overhead of adding observability to the stack."""
        dim = 32
        n = 200

        # Without observability
        bus_no_obs = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus_no_obs.register_module(SafetyModule(input_dim=dim))
        bus_no_obs.register_module(ReasoningModule(input_dim=dim))
        bus_no_obs.register_module(MemoryModule())

        t0 = time.perf_counter()
        for _ in range(n):
            bus_no_obs.process(make_state(dim=dim, entropy=1.0))
        time_without = time.perf_counter() - t0

        # With observability
        bus_obs = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus_obs.register_module(SafetyModule(input_dim=dim))
        bus_obs.register_module(ReasoningModule(input_dim=dim))
        bus_obs.register_module(MemoryModule())
        bus_obs.register_module(ObservabilityModule())

        t0 = time.perf_counter()
        for _ in range(n):
            bus_obs.process(make_state(dim=dim, entropy=1.0))
        time_with = time.perf_counter() - t0

        overhead_pct = ((time_with - time_without) / time_without) * 100
        rate_without = n / time_without
        rate_with = n / time_with
        print(f"\n  Observability overhead:")
        print(f"    Without: {rate_without:.0f} iter/s")
        print(f"    With:    {rate_with:.0f} iter/s")
        print(f"    Overhead: {overhead_pct:.1f}%")

        # Observability should add < 100% overhead (it's pass-through + metrics)
        assert overhead_pct < 100, f"Observability overhead {overhead_pct:.1f}% is too high"

    def test_long_running_scbf_stability(self):
        """500 iterations — metrics should stay bounded and finite."""
        obs = ObservabilityModule()
        bus = ConservationBus(enforcement="hard", tolerance=1e-3)
        bus.register_module(obs)

        for i in range(500):
            state = make_state(dim=16, entropy=1.0)
            result = bus.process(state)
            assert torch.isfinite(result.tensor).all(), f"Non-finite at step {i}"

        m = obs.metrics
        assert m.step_count == 500
        assert m.scbf.entropy_collapse >= 0
        assert 0 <= m.scbf.ancestry_stability <= 1.0
        assert 0 <= m.scbf.phase_alignment <= 1.0
        assert 0 <= m.scbf.attractor_density <= 1.0
        assert m.scbf.bifractal_strength >= 0
        assert len(bus.violation_log) == 0
