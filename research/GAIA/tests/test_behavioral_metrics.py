"""Phase 2 tests — behavioral metrics and scenarios.

Tests metric computation on synthetic trajectories and validates
that scenarios run and produce expected metric keys.
"""

from __future__ import annotations

import pytest
import torch

from gaia.body.environment import GridWorld, Observation
from gaia.body.loop import BodyLoop, TickRecord, Trajectory
from gaia.body.metrics import TrajectoryAnalyzer
from gaia.body.motor import Action, GridMotorDecoder
from gaia.body.senses import ProprioceptiveChannel, VisualChannel
from gaia.core.bus import ConservationBus
from gaia.core.types import FieldState, SECPhase
from gaia.modules.observability import ObservabilityModule


def _make_tick(tick: int, direction: list[float], magnitude: float = 1.0) -> TickRecord:
    """Create a synthetic TickRecord for metric testing."""
    tensor = torch.tensor(direction + [magnitude, 0.0])
    return TickRecord(
        tick=tick,
        observation=Observation(
            visual=torch.zeros(5, 5),
            proprioceptive=torch.zeros(4),
        ),
        field_state_in=FieldState(tensor=tensor, entropy=1.0),
        field_state_out=FieldState(tensor=tensor * magnitude, entropy=1.0),
        action=Action(
            direction=torch.tensor(direction),
            magnitude=magnitude,
        ),
    )


def _make_trajectory(directions: list[list[float]], magnitudes: list[float] | None = None) -> Trajectory:
    """Build a trajectory from direction/magnitude lists."""
    if magnitudes is None:
        magnitudes = [1.0] * len(directions)
    ticks = [_make_tick(i, d, m) for i, (d, m) in enumerate(zip(directions, magnitudes))]
    return Trajectory(ticks=ticks)


# ─── Coherence Index ──────────────────────────────────────────────


class TestCoherenceIndex:

    def test_constant_direction_high_coherence(self):
        """Same direction every tick → CI near 1.0."""
        dirs = [[1.0, 0.0]] * 20
        traj = _make_trajectory(dirs)
        ci = TrajectoryAnalyzer.coherence_index(traj)
        assert ci > 0.95

    def test_alternating_direction_low_coherence(self):
        """Alternating opposite directions → CI near -1.0."""
        dirs = [[1.0, 0.0], [-1.0, 0.0]] * 10
        traj = _make_trajectory(dirs)
        ci = TrajectoryAnalyzer.coherence_index(traj)
        assert ci < -0.9

    def test_orthogonal_directions_zero_coherence(self):
        """Alternating orthogonal directions → CI near 0.0."""
        dirs = [[1.0, 0.0], [0.0, 1.0]] * 10
        traj = _make_trajectory(dirs)
        ci = TrajectoryAnalyzer.coherence_index(traj)
        assert abs(ci) < 0.1

    def test_empty_trajectory(self):
        """Empty trajectory → CI = 0.0."""
        traj = Trajectory(ticks=[])
        ci = TrajectoryAnalyzer.coherence_index(traj)
        assert ci == 0.0


# ─── Habituation Rate ─────────────────────────────────────────────


class TestHabituationRate:

    def test_decaying_magnitudes(self):
        """Exponentially decaying response → positive habituation rate."""
        import math
        magnitudes = [10.0 * math.exp(-0.1 * t) for t in range(30)]
        rate = TrajectoryAnalyzer.habituation_rate(magnitudes)
        assert rate > 0.0

    def test_constant_magnitudes_no_habituation(self):
        """Constant response → zero habituation rate."""
        magnitudes = [5.0] * 30
        rate = TrajectoryAnalyzer.habituation_rate(magnitudes)
        assert rate == 0.0

    def test_increasing_magnitudes_no_habituation(self):
        """Increasing response → zero habituation rate (no decay)."""
        magnitudes = [float(i) for i in range(1, 31)]
        rate = TrajectoryAnalyzer.habituation_rate(magnitudes)
        assert rate == 0.0


# ─── Surprise Response ────────────────────────────────────────────


class TestSurpriseResponse:

    def test_magnitude_spike_detected(self):
        """A spike in response magnitude is detected as surprise > 1.0."""
        magnitudes = [1.0] * 10 + [5.0, 5.0, 5.0] + [1.0] * 10
        surprise = TrajectoryAnalyzer.surprise_response(magnitudes, spike_tick=10, window=3)
        assert surprise > 1.0

    def test_no_spike_ratio_near_one(self):
        """No change → surprise ratio near 1.0."""
        magnitudes = [2.0] * 20
        surprise = TrajectoryAnalyzer.surprise_response(magnitudes, spike_tick=10, window=3)
        assert surprise == pytest.approx(1.0)


# ─── Adaptation Latency ───────────────────────────────────────────


class TestAdaptationLatency:

    def test_immediate_direction_change(self):
        """Immediate direction shift → latency = 1."""
        dirs = [[1.0, 0.0]] * 10 + [[-1.0, 0.0]] * 10
        traj = _make_trajectory(dirs)
        # change_tick=9 is the last [1,0] action; tick 10 is the first [-1,0]
        latency = TrajectoryAnalyzer.adaptation_latency(traj, change_tick=9, threshold=0.5)
        assert latency == 1

    def test_slow_adaptation(self):
        """Gradual direction change → latency > 1."""
        dirs = [[1.0, 0.0]] * 10 + [[0.9, 0.1]] * 5 + [[-1.0, 0.0]] * 5
        traj = _make_trajectory(dirs)
        latency = TrajectoryAnalyzer.adaptation_latency(traj, change_tick=10, threshold=0.3)
        assert latency > 1


# ─── Preference Divergence ────────────────────────────────────────


class TestPreferenceDivergence:

    def test_different_magnitudes_positive_divergence(self):
        """Different mean magnitudes → positive divergence."""
        mags_a = [5.0] * 10
        mags_b = [1.0] * 10
        div = TrajectoryAnalyzer.preference_divergence(mags_a, mags_b)
        assert div > 0.0

    def test_same_magnitudes_zero_divergence(self):
        """Same mean magnitudes → zero divergence."""
        mags = [3.0] * 10
        div = TrajectoryAnalyzer.preference_divergence(mags, mags)
        assert div == pytest.approx(0.0)


# ─── Scenarios ─────────────────────────────────────────────────────


def _make_test_bus() -> ConservationBus:
    """Lightweight bus for scenario smoke tests."""
    bus = ConservationBus(enforcement="soft")
    bus.register_module(ObservabilityModule())
    return bus


class TestScenarios:

    def test_habituation_scenario_runs(self):
        """Habituation scenario produces expected metric keys."""
        from gaia.body.scenarios import run_habituation
        results = run_habituation(_make_test_bus(), n_ticks=20)
        assert "habituation_rate" in results
        assert "habituation_first_half_mean" in results
        assert "habituation_second_half_mean" in results

    def test_novelty_scenario_runs(self):
        """Novelty scenario produces expected metric keys."""
        from gaia.body.scenarios import run_novelty
        results = run_novelty(_make_test_bus(), n_familiar=10)
        assert "surprise_ratio" in results
        assert "pre_novelty_coherence" in results

    def test_adaptation_scenario_runs(self):
        """Adaptation scenario produces expected metric keys."""
        from gaia.body.scenarios import run_adaptation
        results = run_adaptation(_make_test_bus(), n_before=10, n_after=10)
        assert "adaptation_latency" in results
        assert "coherence_before" in results
        assert "coherence_after" in results

    def test_preference_scenario_runs(self):
        """Preference scenario produces expected metric keys."""
        from gaia.body.scenarios import run_preference
        results = run_preference(_make_test_bus(), n_ticks_per_class=10)
        assert "mean_preference_divergence" in results
        assert "preference_divergence_ab" in results


# ─── Behavioral Benchmark ─────────────────────────────────────────


class TestBehavioralBenchmark:

    def test_benchmark_runs_and_produces_all_metrics(self):
        """Full behavioral benchmark produces all expected metric keys."""
        from benchmarks.behavioral import bench_behavioral
        results = bench_behavioral(seed=42)
        expected_keys = [
            "habituation_rate", "surprise_ratio", "adaptation_latency",
            "coherence_before", "mean_preference_divergence",
        ]
        for key in expected_keys:
            assert key in results, f"Missing metric: {key}"
