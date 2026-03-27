"""Phase 3 tests — ablation framework, behavioral probes, cross-module coherence.

Tests module registry, configuration generation, ablation runner,
probe results, and cross-module analysis functions.
"""

from __future__ import annotations

import pytest
import torch

from gaia.body.ablation import (
    ALL_MODULES,
    AblationReport,
    AblationResult,
    ModuleConfig,
    make_bus,
    run_ablation_config,
    run_full_ablation,
    standard_configs,
)
from gaia.body.probes import (
    ProbeResult,
    energy_quadrant_analysis,
    phase_trajectory,
    probe_language_predictor,
    probe_memory_context,
    probe_reasoning_attractor,
    probe_safety_stabilizer,
    provenance_frequency,
    run_instrumented,
)


# ─── Module Registry ──────────────────────────────────────────────


class TestModuleRegistry:

    def test_all_five_modules_in_registry(self):
        """Registry contains all 5 GAIA modules."""
        from gaia.body.ablation import MODULE_REGISTRY

        assert len(MODULE_REGISTRY) == 5
        for name in ["observability", "safety", "reasoning", "memory", "language"]:
            assert name in MODULE_REGISTRY

    def test_make_bus_creates_specified_modules(self):
        """make_bus registers only the requested modules."""
        bus = make_bus(["observability", "safety"], input_dim=22)
        assert len(bus._modules) == 2
        assert "observability" in bus._modules
        assert "safety" in bus._modules

    def test_make_bus_full(self):
        """make_bus with all modules creates 5-module bus."""
        bus = make_bus(ALL_MODULES, input_dim=22)
        assert len(bus._modules) == 5


# ─── Ablation Configs ─────────────────────────────────────────────


class TestAblationConfigs:

    def test_standard_configs_count(self):
        """standard_configs() returns exactly 14 configurations."""
        configs = standard_configs()
        assert len(configs) == 14

    def test_full_config_has_all_modules(self):
        """'full' config includes all 5 modules."""
        configs = standard_configs()
        full = next(c for c in configs if c.name == "full")
        assert set(full.modules) == set(ALL_MODULES)

    def test_observability_in_every_config(self):
        """Every config includes observability."""
        configs = standard_configs()
        for config in configs:
            assert "observability" in config.modules, f"{config.name} missing observability"

    def test_leave_one_out_has_four_modules(self):
        """Each leave-one-out config has exactly 4 modules."""
        configs = standard_configs()
        loo = [c for c in configs if c.name.startswith("no_")]
        assert len(loo) == 4
        for config in loo:
            assert len(config.modules) == 4

    def test_config_names_unique(self):
        """All config names are unique."""
        configs = standard_configs()
        names = [c.name for c in configs]
        assert len(names) == len(set(names))


# ─── Ablation Runner ──────────────────────────────────────────────


class TestAblationRunner:

    def test_single_config_produces_all_metrics(self):
        """Running a single config produces all expected metric keys."""
        config = ModuleConfig(name="test", modules=["observability"])
        result = run_ablation_config(config, seed=42)
        # Should have metrics from all 4 scenarios
        assert "habituation_rate" in result.metrics
        assert "surprise_ratio" in result.metrics
        assert "adaptation_latency" in result.metrics
        assert "mean_preference_divergence" in result.metrics

    def test_observability_only_runs(self):
        """Control config (observability-only) runs without error."""
        config = ModuleConfig(name="control", modules=["observability"])
        result = run_ablation_config(config, seed=42)
        assert isinstance(result, AblationResult)
        assert len(result.metrics) > 0

    def test_full_config_runs(self):
        """Full 5-module config runs without error."""
        config = ModuleConfig(name="full", modules=list(ALL_MODULES))
        result = run_ablation_config(config, seed=42)
        assert isinstance(result, AblationResult)


class TestFullAblation:

    @pytest.fixture(scope="class")
    def report(self) -> AblationReport:
        """Run a minimal ablation (3 configs) for testing."""
        configs = [
            ModuleConfig("full", list(ALL_MODULES)),
            ModuleConfig("observability_only", ["observability"]),
            ModuleConfig("no_safety", [m for m in ALL_MODULES if m != "safety"]),
        ]
        return run_full_ablation(configs=configs, seed=42)

    def test_identifies_baseline(self, report: AblationReport):
        """Full ablation correctly identifies the 'full' config as baseline."""
        assert report.baseline is not None
        assert report.baseline.config.name == "full"

    def test_delta_matrix_excludes_baseline(self, report: AblationReport):
        """Delta matrix does not include the baseline config."""
        deltas = report.delta_matrix()
        assert "full" not in deltas
        assert "observability_only" in deltas
        assert "no_safety" in deltas

    def test_metric_matrix_has_all_configs(self, report: AblationReport):
        """Metric matrix has an entry for every config."""
        matrix = report.metric_matrix()
        assert len(matrix) == 3

    def test_delta_is_difference_from_baseline(self, report: AblationReport):
        """Delta values equal metric - baseline for each cell."""
        deltas = report.delta_matrix()
        matrix = report.metric_matrix()
        baseline = matrix["full"]
        for config_name, row in deltas.items():
            for metric, delta_val in row.items():
                expected = matrix[config_name][metric] - baseline[metric]
                assert abs(delta_val - expected) < 1e-10, (
                    f"{config_name}/{metric}: delta={delta_val}, expected={expected}"
                )


# ─── Probes ───────────────────────────────────────────────────────


class TestProbes:

    def test_probe_safety_returns_result(self):
        """Safety probe returns a valid ProbeResult."""
        result = probe_safety_stabilizer(seed=42)
        assert isinstance(result, ProbeResult)
        assert result.name == "safety_stabilizer"
        assert len(result.evidence) > 0

    def test_probe_memory_returns_result(self):
        """Memory probe returns a valid ProbeResult."""
        result = probe_memory_context(seed=42)
        assert isinstance(result, ProbeResult)
        assert result.name == "memory_context"
        assert len(result.evidence) > 0

    def test_probe_reasoning_returns_result(self):
        """Reasoning probe returns a valid ProbeResult."""
        result = probe_reasoning_attractor(seed=42)
        assert isinstance(result, ProbeResult)
        assert result.name == "reasoning_attractor"
        assert len(result.evidence) > 0

    def test_probe_language_returns_result(self):
        """Language probe returns a valid ProbeResult."""
        result = probe_language_predictor(seed=42)
        assert isinstance(result, ProbeResult)
        assert result.name == "language_predictor"
        assert len(result.evidence) > 0

    def test_all_probes_have_deltas(self):
        """Every probe result has non-empty delta dict."""
        for probe_fn in [
            probe_safety_stabilizer,
            probe_memory_context,
            probe_reasoning_attractor,
            probe_language_predictor,
        ]:
            result = probe_fn(seed=42)
            assert len(result.delta) > 0, f"{result.name} has empty deltas"


# ─── Instrumented Run ─────────────────────────────────────────────


class TestInstrumentedRun:

    def test_returns_trajectory_and_health(self):
        """Instrumented run returns trajectory + health snapshots."""
        bus = make_bus(ALL_MODULES, input_dim=22)
        traj, health = run_instrumented(bus, n_ticks=10, seed=42)
        assert len(traj) == 10
        assert len(health) == 10

    def test_health_has_all_modules(self):
        """Each health snapshot includes all registered modules."""
        bus = make_bus(ALL_MODULES, input_dim=22)
        _, health = run_instrumented(bus, n_ticks=5, seed=42)
        for snapshot in health:
            for mod in ALL_MODULES:
                assert mod in snapshot, f"Missing {mod} in health snapshot"


# ─── Cross-Module Coherence ───────────────────────────────────────


class TestCrossModuleCoherence:

    @pytest.fixture(scope="class")
    def trajectory_data(self):
        """Run 20 ticks for coherence analysis."""
        bus = make_bus(ALL_MODULES, input_dim=22)
        traj, health = run_instrumented(bus, n_ticks=20, seed=42)
        return traj, health

    def test_energy_quadrant_returns_four_lists(self, trajectory_data):
        """Energy analysis returns 4 lists matching tick count."""
        traj, _ = trajectory_data
        quads = energy_quadrant_analysis(traj)
        assert set(quads.keys()) == {"top", "bottom", "left", "right"}
        for key in quads:
            assert len(quads[key]) == 20

    def test_phase_trajectory_length(self, trajectory_data):
        """Phase trajectory has one entry per tick."""
        traj, _ = trajectory_data
        phases = phase_trajectory(traj)
        assert len(phases) == 20
        # All phases should be valid SEC phase names
        valid = {"crystallized", "ordered", "transitional", "chaotic"}
        for p in phases:
            assert p in valid, f"Invalid phase: {p}"

    def test_provenance_frequency_bounds(self, trajectory_data):
        """Provenance frequencies are in [0, 1]."""
        traj, _ = trajectory_data
        freqs = provenance_frequency(traj)
        for name, freq in freqs.items():
            assert 0.0 <= freq <= 1.0, f"{name}: frequency {freq} out of bounds"

    def test_observability_always_active(self, trajectory_data):
        """Observability should appear in provenance for every tick."""
        traj, _ = trajectory_data
        freqs = provenance_frequency(traj)
        assert "observability" in freqs
        assert freqs["observability"] == 1.0


# ─── Behavioral Assertions ────────────────────────────────────────


class TestBehavioralAssertions:

    @pytest.fixture(scope="class")
    def mini_report(self) -> AblationReport:
        """Run full vs. control for behavioral comparison."""
        configs = [
            ModuleConfig("full", list(ALL_MODULES)),
            ModuleConfig("observability_only", ["observability"]),
        ]
        return run_full_ablation(configs=configs, seed=42)

    def test_control_produces_nonzero_metrics(self, mini_report: AblationReport):
        """Even observability-only produces valid (non-degenerate) metrics."""
        control = next(r for r in mini_report.results if r.config.name == "observability_only")
        assert control.metrics["surprise_ratio"] > 0.0
        assert control.metrics["adaptation_latency"] > 0.0

    def test_full_and_control_differ(self, mini_report: AblationReport):
        """Full brain and control produce different behavioral signatures."""
        matrix = mini_report.metric_matrix()
        full = matrix["full"]
        control = matrix["observability_only"]
        # At least some metrics should differ
        diffs = sum(1 for k in full if abs(full[k] - control[k]) > 1e-6)
        assert diffs > 0, "Full brain and control have identical metrics"
