"""Behavioral metrics — measure coherence, adaptation, and habituation from trajectories.

These replace prediction-accuracy benchmarks with brain-like behavioral measures.
All metrics operate on Trajectory objects from the BodyLoop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .loop import Trajectory


@dataclass
class BehavioralScores:
    """Aggregate behavioral scores from a trajectory."""

    coherence_index: float       # [0, 1] — consistency of consecutive actions
    adaptation_rate: float       # ticks until behavior shifts after env change
    habituation_rate: float      # exponential decay constant of repeated-stimulus response
    surprise_magnitude: float    # output spike when novel stimulus appears
    preference_divergence: float  # KL divergence between stimulus-class responses


class TrajectoryAnalyzer:
    """Computes behavioral metrics from body-loop trajectories."""

    @staticmethod
    def coherence_index(trajectory: Trajectory) -> float:
        """Consistency of consecutive actions (autocorrelation).

        CI = mean(cosine_similarity(action[t], action[t+1])) over the trajectory.
        High CI = consistent, purposeful behavior.
        Low CI = erratic, random behavior.

        Returns:
            Float in [-1, 1], where 1 = perfectly consistent.
        """
        actions = trajectory.actions
        if len(actions) < 2:
            return 0.0

        similarities = []
        for i in range(len(actions) - 1):
            d1 = actions[i].direction.float()
            d2 = actions[i + 1].direction.float()
            norm1 = d1.norm()
            norm2 = d2.norm()
            if norm1 < 1e-8 or norm2 < 1e-8:
                similarities.append(0.0)
            else:
                cos_sim = float(torch.dot(d1, d2) / (norm1 * norm2))
                similarities.append(cos_sim)

        return sum(similarities) / len(similarities)

    @staticmethod
    def response_magnitudes(trajectory: Trajectory) -> list[float]:
        """Extract output magnitude (mean abs tensor value) per tick."""
        return [
            float(t.field_state_out.tensor.abs().mean().item())
            for t in trajectory.ticks
        ]

    @staticmethod
    def habituation_rate(magnitudes: list[float]) -> float:
        """Fit exponential decay to response magnitudes.

        Models magnitude[t] ~ a * exp(-rate * t) + c.
        Returns the decay rate. Higher = faster habituation.
        Returns 0.0 if no decay detected.
        """
        if len(magnitudes) < 3:
            return 0.0

        # Simple estimate: ratio of first-half mean to second-half mean
        mid = len(magnitudes) // 2
        first_half = sum(magnitudes[:mid]) / max(mid, 1)
        second_half = sum(magnitudes[mid:]) / max(len(magnitudes) - mid, 1)

        if first_half < 1e-10:
            return 0.0

        ratio = second_half / first_half
        if ratio >= 1.0:
            return 0.0  # No decay

        # rate ≈ -ln(ratio) / mid
        return -math.log(max(ratio, 1e-10)) / mid

    @staticmethod
    def surprise_response(
        magnitudes: list[float],
        spike_tick: int,
        window: int = 3,
    ) -> float:
        """Measure response magnitude spike at a novel stimulus.

        Compares mean magnitude in the window after spike_tick to
        the mean magnitude in the window before. Returns the ratio
        (values > 1.0 indicate surprise response).
        """
        if spike_tick < window or spike_tick + window >= len(magnitudes):
            return 1.0  # Can't measure

        before = sum(magnitudes[spike_tick - window : spike_tick]) / window
        after = sum(magnitudes[spike_tick : spike_tick + window]) / window

        if before < 1e-10:
            return 1.0

        return after / before

    @staticmethod
    def adaptation_latency(
        trajectory: Trajectory,
        change_tick: int,
        threshold: float = 0.3,
    ) -> int:
        """Ticks until behavior shifts after an environment change.

        Measures when cosine similarity between pre-change and post-change
        actions drops below threshold. Lower = more adaptive.

        Returns:
            Number of ticks until adaptation, or remaining ticks if never adapts.
        """
        actions = trajectory.actions
        if change_tick >= len(actions) - 1:
            return 0

        # Reference: last action before change
        ref_dir = actions[change_tick].direction.float()
        ref_norm = ref_dir.norm()
        if ref_norm < 1e-8:
            return 0

        for t in range(change_tick + 1, len(actions)):
            d = actions[t].direction.float()
            d_norm = d.norm()
            if d_norm < 1e-8:
                continue
            sim = float(torch.dot(ref_dir, d) / (ref_norm * d_norm))
            if sim < threshold:
                return t - change_tick

        return len(actions) - change_tick  # Never adapted

    @staticmethod
    def preference_divergence(
        magnitudes_a: list[float],
        magnitudes_b: list[float],
    ) -> float:
        """KL divergence between response distributions to two stimulus classes.

        Higher values indicate stronger preference formation (asymmetric responses).
        """
        if not magnitudes_a or not magnitudes_b:
            return 0.0

        # Normalize to probability distributions
        sum_a = sum(magnitudes_a)
        sum_b = sum(magnitudes_b)
        if sum_a < 1e-10 or sum_b < 1e-10:
            return 0.0

        mean_a = sum_a / len(magnitudes_a)
        mean_b = sum_b / len(magnitudes_b)

        # Simple symmetric KL proxy: |log(mean_a / mean_b)|
        if mean_a < 1e-10 or mean_b < 1e-10:
            return 0.0

        return abs(math.log(mean_a / mean_b))
