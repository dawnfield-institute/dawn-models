"""Standardized behavioral test scenarios.

Each scenario runs a body loop under controlled conditions and returns
scored metrics. Scenarios are deterministic (seeded) and designed to
test specific brain-like properties.
"""

from __future__ import annotations

import torch

from gaia.core.bus import ConservationBus

from .environment import GridWorld
from .loop import BodyLoop
from .metrics import TrajectoryAnalyzer
from .motor import Action, GridMotorDecoder
from .senses import ProprioceptiveChannel, VisualChannel


def _make_standard_loop(
    bus: ConservationBus,
    grid_size: int = 5,
    seed: int = 42,
) -> BodyLoop:
    """Create a standard body loop with visual + proprioceptive channels."""
    env = GridWorld(size=grid_size, n_stimuli=3, seed=seed)
    channels = [VisualChannel(), ProprioceptiveChannel()]
    decoder = GridMotorDecoder()
    return BodyLoop(bus, channels, decoder, env)


def run_habituation(bus: ConservationBus, n_ticks: int = 50, seed: int = 42) -> dict[str, float]:
    """Habituation scenario: same environment, no changes, measure response decay.

    A system with memory should show diminishing response to repeated
    identical stimuli as patterns become familiar.

    Returns:
        habituation_rate: Exponential decay constant (higher = faster habituation)
        first_half_mean: Mean response in first half
        second_half_mean: Mean response in second half
    """
    loop = _make_standard_loop(bus, seed=seed)
    trajectory = loop.run(n_ticks)
    magnitudes = TrajectoryAnalyzer.response_magnitudes(trajectory)

    mid = len(magnitudes) // 2
    first_half = sum(magnitudes[:mid]) / max(mid, 1)
    second_half = sum(magnitudes[mid:]) / max(len(magnitudes) - mid, 1)

    return {
        "habituation_rate": TrajectoryAnalyzer.habituation_rate(magnitudes),
        "habituation_first_half_mean": first_half,
        "habituation_second_half_mean": second_half,
    }


def run_novelty(
    bus: ConservationBus,
    n_familiar: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    """Novelty scenario: familiar environment then sudden stimulus change.

    Run n_familiar ticks in one environment, then swap to a different
    environment and measure the surprise response.

    Returns:
        surprise_ratio: Output magnitude after/before novelty (>1 = surprise)
        pre_novelty_coherence: Coherence index before change
    """
    # Phase 1: Familiar environment
    env1 = GridWorld(size=5, n_stimuli=3, seed=seed)
    channels = [VisualChannel(), ProprioceptiveChannel()]
    decoder = GridMotorDecoder()
    loop = BodyLoop(bus, channels, decoder, env1)
    traj_familiar = loop.run(n_familiar)

    # Phase 2: Novel environment (different stimuli)
    env2 = GridWorld(size=5, n_stimuli=3, seed=seed + 1000)
    loop2 = BodyLoop(bus, channels, decoder, env2)
    traj_novel = loop2.run(10)

    magnitudes_familiar = TrajectoryAnalyzer.response_magnitudes(traj_familiar)
    magnitudes_novel = TrajectoryAnalyzer.response_magnitudes(traj_novel)

    # Compare last few familiar ticks to first few novel ticks
    window = min(5, len(magnitudes_familiar), len(magnitudes_novel))
    mean_familiar = sum(magnitudes_familiar[-window:]) / max(window, 1)
    mean_novel = sum(magnitudes_novel[:window]) / max(window, 1)

    surprise_ratio = mean_novel / max(mean_familiar, 1e-10)

    return {
        "surprise_ratio": surprise_ratio,
        "pre_novelty_coherence": TrajectoryAnalyzer.coherence_index(traj_familiar),
    }


def run_adaptation(
    bus: ConservationBus,
    n_before: int = 25,
    n_after: int = 25,
    seed: int = 42,
) -> dict[str, float]:
    """Adaptation scenario: run in env A, then switch to env B, measure latency.

    Tests how quickly the system adjusts its behavior after an environmental
    shift. Lower adaptation_latency = more adaptive.

    Returns:
        adaptation_latency: Ticks until behavior shifts
        coherence_before: Coherence index in env A
        coherence_after: Coherence index in env B
    """
    channels = [VisualChannel(), ProprioceptiveChannel()]
    decoder = GridMotorDecoder()

    # Phase A
    env_a = GridWorld(size=5, n_stimuli=3, seed=seed)
    loop_a = BodyLoop(bus, channels, decoder, env_a)
    traj_a = loop_a.run(n_before)

    # Phase B (different environment)
    env_b = GridWorld(size=5, n_stimuli=3, seed=seed + 500)
    loop_b = BodyLoop(bus, channels, decoder, env_b)
    traj_b = loop_b.run(n_after)

    coherence_a = TrajectoryAnalyzer.coherence_index(traj_a)
    coherence_b = TrajectoryAnalyzer.coherence_index(traj_b)

    # Measure adaptation: how quickly actions in B diverge from last action in A
    from .loop import Trajectory, TickRecord

    combined_ticks = traj_a.ticks + traj_b.ticks
    combined = Trajectory(ticks=combined_ticks)
    latency = TrajectoryAnalyzer.adaptation_latency(combined, change_tick=n_before)

    return {
        "adaptation_latency": float(latency),
        "coherence_before": coherence_a,
        "coherence_after": coherence_b,
    }


def run_preference(
    bus: ConservationBus,
    n_ticks_per_class: int = 30,
    seed: int = 42,
) -> dict[str, float]:
    """Preference scenario: alternate between 3 stimulus classes.

    After exposure to different environments, measure whether the system
    develops asymmetric responses (preference formation).

    Returns:
        preference_divergence_ab: KL proxy between class A and B responses
        preference_divergence_ac: KL proxy between class A and C responses
        mean_divergence: Average divergence (higher = stronger preferences)
    """
    channels = [VisualChannel(), ProprioceptiveChannel()]
    decoder = GridMotorDecoder()
    class_magnitudes: dict[str, list[float]] = {"a": [], "b": [], "c": []}

    for label, class_seed in [("a", seed), ("b", seed + 100), ("c", seed + 200)]:
        env = GridWorld(size=5, n_stimuli=3, seed=class_seed)
        loop = BodyLoop(bus, channels, decoder, env)
        traj = loop.run(n_ticks_per_class)
        class_magnitudes[label] = TrajectoryAnalyzer.response_magnitudes(traj)

    div_ab = TrajectoryAnalyzer.preference_divergence(class_magnitudes["a"], class_magnitudes["b"])
    div_ac = TrajectoryAnalyzer.preference_divergence(class_magnitudes["a"], class_magnitudes["c"])

    return {
        "preference_divergence_ab": div_ab,
        "preference_divergence_ac": div_ac,
        "mean_preference_divergence": (div_ab + div_ac) / 2.0,
    }
