"""Observability Module — SCBF metrics + QBE equilibrium monitoring.

Ported from TinyCIMM-Euler. Pure instrumentation layer — observes
FieldState tensors flowing through the bus and computes 5 SCBF
(Symbolic Collapse Benchmarking Framework) metrics without
transforming the tensor. PAC conservation is trivial (pass-through).

Components:
    QBEController: Dynamic equilibrium management (momentum + error band).
    SCBFTracker: 5-metric symbolic collapse tracker.
    ObservabilityModule: GAIAModule wrapper for the full instrumentation stack.

Metrics tracked:
    1. Symbolic Entropy Collapse (SEC): multi-level entropy analysis
    2. Activation Ancestry Trace: neuron identity stability
    3. Collapse Phase Alignment: temporal coherence of collapse
    4. Bifractal Lineage: recursive reactivation patterns
    5. Semantic Attractor Density: clustering of activation centroids
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch

from gaia.core.types import FieldState, RBFBalance, SECPhase


# ─── Data Structures ────────────────────────────────────────────────


@dataclass
class CollapseEvent:
    """A detected entropy collapse event."""

    step: int
    immediate_change: float
    trend_change: float
    magnitude: float
    threshold: float


@dataclass
class SCBFMetrics:
    """5-metric SCBF measurement from a single observation step."""

    entropy_collapse: float = 0.0
    ancestry_stability: float = 0.0
    phase_alignment: float = 0.0
    bifractal_strength: float = 0.0
    attractor_density: float = 0.0

    # Derived
    entropy_variance: float = 0.0
    pattern_consistency: float = 1.0
    recursive_activity: float = 0.0
    mathematical_memory_size: int = 0
    collapse_count: int = 0
    entropy_momentum: float = 0.0


@dataclass
class ObservabilityMetrics:
    """Full observability output — SCBF + QBE + step info."""

    scbf: SCBFMetrics
    qbe_status: str = "Near Equilibrium"
    qbe_energy_balance: float = 1.0
    step_count: int = 0


# ─── QBE Controller ─────────────────────────────────────────────────


class QBEController:
    """Quantum Bifractal Equilibrium controller for dynamic adaptation.

    Manages momentum, error bands, and energy balance for real-time
    learning state monitoring. Maps to SEC phases:
        Near Equilibrium    → CRYSTALLIZED
        Moderate Equilibrium → ORDERED
        Far from Equilibrium → CHAOTIC
    """

    def __init__(
        self,
        initial_momentum: float = 0.8,
        error_band: float = 0.1,
    ) -> None:
        self.momentum = initial_momentum
        self.error_band = error_band
        self.energy_balance = initial_momentum + error_band

    def update(self, error: float, entropy: float) -> None:
        """Update QBE state from error and entropy signals."""
        self.momentum = 0.9 * self.momentum + 0.1 * abs(error)
        self.error_band = max(0.05, min(0.2, self.error_band + 0.01 * entropy))
        self.energy_balance = self.momentum + self.error_band

    def get_status(self) -> str:
        """Equilibrium status based on energy balance."""
        if self.energy_balance < 1.5:
            return "Near Equilibrium"
        elif self.energy_balance < 2.0:
            return "Moderate Equilibrium"
        return "Far from Equilibrium"

    def detect_pattern_type(self, recent_values: list[float]) -> str:
        """Detect convergence/chaotic/unknown from recent values."""
        if len(recent_values) > 10:
            vals = torch.tensor(recent_values[-10:])
            variance = torch.var(vals).item()
            if variance < 0.01:
                return "convergence"
            elif variance > 0.5:
                return "chaotic"
        return "unknown"

    def adjust_for_pattern(self, pattern_type: str) -> None:
        """Tune QBE settings based on detected pattern type."""
        if pattern_type == "convergence":
            self.error_band = max(0.05, self.error_band - 0.01)
            self.momentum = min(0.9, self.momentum + 0.05)
        elif pattern_type == "chaotic":
            self.error_band = min(0.2, self.error_band + 0.01)
            self.momentum = max(0.7, self.momentum - 0.05)

    def to_sec_phase(self) -> SECPhase:
        """Map QBE status to SEC phase."""
        status = self.get_status()
        if status == "Near Equilibrium":
            return SECPhase.CRYSTALLIZED
        elif status == "Moderate Equilibrium":
            return SECPhase.ORDERED
        return SECPhase.CHAOTIC


# ─── SCBF Tracker ────────────────────────────────────────────────────


class SCBFTracker:
    """Unified SCBF tracker — 5 symbolic collapse metrics.

    Consolidates TinyCIMM-Euler's UnifiedSymbolicCollapseTracker
    into a clean, rolling-window design. All metrics computed from
    activation tensors (no weight access needed).

    Metrics:
        1. Entropy collapse: multi-level Shannon entropy with collapse detection
        2. Ancestry stability: top-K neuron consistency + pattern correlation
        3. Phase alignment: temporal coherence via phase vectors
        4. Bifractal lineage: recursive pattern detection
        5. Attractor density: centroid clustering analysis
    """

    def __init__(
        self,
        memory_window: int = 30,
        sensitivity_factor: float = 2.0,
    ) -> None:
        self._window = memory_window
        self._sensitivity = sensitivity_factor
        self._step = 0

        # Entropy collapse state
        self._raw_entropies: deque[float] = deque(maxlen=memory_window)
        self._smoothed_entropies: deque[float] = deque(maxlen=memory_window)
        self._entropy_momentum = 0.0
        self._collapse_events: list[CollapseEvent] = []

        # Ancestry state
        self._prev_normalized: Optional[torch.Tensor] = None
        self._prev_top_neurons: Optional[set[int]] = None
        self._stability_scores: deque[float] = deque(maxlen=memory_window)
        self._top_consistency: deque[float] = deque(maxlen=memory_window)

        # Phase alignment state
        self._phase_vectors: deque[torch.Tensor] = deque(maxlen=memory_window)
        self._coherence_scores: deque[float] = deque(maxlen=memory_window)

        # Bifractal state
        self._fingerprints: deque[torch.Tensor] = deque(maxlen=memory_window)
        self._recursion_scores: deque[float] = deque(maxlen=memory_window)
        self._mathematical_memory: list[dict] = []

        # Attractor state
        self._centroids: deque[torch.Tensor] = deque(maxlen=memory_window)
        self._densities: deque[float] = deque(maxlen=memory_window)

    def compute(self, tensor: torch.Tensor) -> SCBFMetrics:
        """Compute all 5 SCBF metrics from an activation tensor.

        This is the main entry point — calls all sub-metrics and
        assembles derived metrics.
        """
        self._step += 1

        entropy_collapse = self._compute_entropy_collapse(tensor)
        ancestry_stability = self._track_ancestry(tensor)
        phase_alignment = self._compute_phase_alignment(tensor)
        bifractal_strength = self._track_bifractal(tensor)
        attractor_density = self._compute_attractor_density(tensor)

        # Derived metrics
        entropy_variance = 0.0
        if len(self._smoothed_entropies) >= 10:
            vals = list(self._smoothed_entropies)[-10:]
            mean = sum(vals) / len(vals)
            entropy_variance = sum((v - mean) ** 2 for v in vals) / len(vals)

        pattern_consistency = 1.0
        if len(self._stability_scores) >= 5:
            recent = list(self._stability_scores)[-5:]
            pattern_consistency = sum(recent) / len(recent)

        recursive_activity = 0.0
        if len(self._recursion_scores) >= 5:
            recent = list(self._recursion_scores)[-5:]
            recursive_activity = sum(recent) / len(recent)

        return SCBFMetrics(
            entropy_collapse=entropy_collapse,
            ancestry_stability=ancestry_stability,
            phase_alignment=phase_alignment,
            bifractal_strength=bifractal_strength,
            attractor_density=attractor_density,
            entropy_variance=entropy_variance,
            pattern_consistency=pattern_consistency,
            recursive_activity=recursive_activity,
            mathematical_memory_size=len(self._mathematical_memory),
            collapse_count=len(self._collapse_events),
            entropy_momentum=self._entropy_momentum,
        )

    def _compute_entropy_collapse(self, tensor: torch.Tensor) -> float:
        """Multi-level entropy analysis with adaptive collapse detection.

        Three entropy components:
            1. Raw Shannon entropy of |activations|
            2. Gradient entropy (pattern sharpness)
            3. Order entropy (mathematical structure)
        Combined: 0.5 * raw + 0.3 * gradient + 0.2 * order
        """
        flat = tensor.flatten().float()

        if len(flat) <= 1:
            self._raw_entropies.append(3.0)
            self._smoothed_entropies.append(3.0)
            return 3.0

        # 1. Raw activation entropy
        abs_act = torch.abs(flat) + 1e-9
        norm_act = abs_act / torch.sum(abs_act)
        raw_entropy = -torch.sum(norm_act * torch.log(norm_act + 1e-9)).item()

        # 2. Gradient entropy (pattern sharpness)
        if len(flat) > 2:
            gradients = torch.diff(flat)
            grad_mag = torch.abs(gradients) + 1e-9
            grad_probs = grad_mag / torch.sum(grad_mag)
            gradient_entropy = -torch.sum(grad_probs * torch.log(grad_probs + 1e-9)).item()
        else:
            gradient_entropy = raw_entropy

        # 3. Order entropy (mathematical structure)
        sorted_vals, _ = torch.sort(torch.abs(flat), descending=True)
        total = torch.sum(sorted_vals)
        if total > 1e-9:
            order_probs = sorted_vals / total
            order_entropy = -torch.sum(order_probs * torch.log(order_probs + 1e-9)).item()
        else:
            order_entropy = raw_entropy

        # Combined with mathematical weighting
        combined = 0.5 * raw_entropy + 0.3 * gradient_entropy + 0.2 * order_entropy
        self._raw_entropies.append(combined)

        # Smoothed entropy with momentum
        if len(self._raw_entropies) > 1:
            prev = list(self._raw_entropies)[-2]
            self._entropy_momentum = 0.7 * self._entropy_momentum + 0.3 * (combined - prev)
            smoothed = 0.8 * combined + 0.2 * prev
        else:
            self._entropy_momentum = 0.0
            smoothed = combined
        self._smoothed_entropies.append(smoothed)

        # Collapse detection
        if len(self._smoothed_entropies) > 3:
            smoothed_list = list(self._smoothed_entropies)
            immediate_change = abs(smoothed_list[-1] - smoothed_list[-2])
            if len(smoothed_list) >= 4:
                recent = sum(smoothed_list[-2:]) / 2
                earlier = sum(smoothed_list[-4:-2]) / 2
                trend_change = abs(recent - earlier)
            else:
                trend_change = 0.0

            # Adaptive threshold from recent variance
            if len(smoothed_list) >= 5:
                last5 = smoothed_list[-5:]
                mean5 = sum(last5) / 5
                var5 = sum((v - mean5) ** 2 for v in last5) / 5
            else:
                var5 = 0.1
            threshold = max(0.005, min(0.2, var5 * self._sensitivity))

            if immediate_change > threshold or trend_change > threshold * 0.7:
                self._collapse_events.append(
                    CollapseEvent(
                        step=self._step,
                        immediate_change=immediate_change,
                        trend_change=trend_change,
                        magnitude=max(immediate_change, trend_change),
                        threshold=threshold,
                    )
                )

        return combined

    def _track_ancestry(self, tensor: torch.Tensor) -> float:
        """Track neuron identity stability over time.

        Measures top-K neuron consistency and pattern correlation
        between consecutive activations.
        """
        flat = tensor.flatten().float()

        if len(flat) < 2:
            return 1.0

        # Normalized pattern
        norm = torch.norm(flat)
        normalized = flat / (norm + 1e-9)

        # Top-K neurons
        top_k = min(5, len(flat))
        magnitude = torch.abs(flat)
        top_indices = torch.argsort(magnitude, descending=True)[:top_k]
        current_top = set(top_indices.tolist())

        stability = 1.0

        if self._prev_normalized is not None and self._prev_top_neurons is not None:
            # Top neuron consistency
            overlap = len(current_top.intersection(self._prev_top_neurons)) / top_k
            self._top_consistency.append(overlap)

            # Pattern correlation
            min_size = min(len(self._prev_normalized), len(normalized))
            if min_size > 1:
                prev = self._prev_normalized[:min_size]
                curr = normalized[:min_size]
                dot = torch.dot(prev, curr)
                stability = max(0.0, min(1.0, dot.item()))
            else:
                stability = 1.0

            self._stability_scores.append(stability)

        self._prev_normalized = normalized.detach().clone()
        self._prev_top_neurons = current_top

        return stability

    def _compute_phase_alignment(self, tensor: torch.Tensor) -> float:
        """Temporal coherence of activation collapse.

        Computes phase vectors from top-4 activations and measures
        correlation between consecutive phase vectors.
        """
        flat = tensor.flatten().float()
        top_k = min(4, len(flat))
        if top_k < 2:
            return 1.0

        # Phase vector from top activations
        top_vals, _ = torch.topk(torch.abs(flat), top_k)
        phase_vec = top_vals / (torch.norm(top_vals) + 1e-9)
        self._phase_vectors.append(phase_vec.detach().clone())

        if len(self._phase_vectors) < 2:
            return 1.0

        # Correlation between consecutive phase vectors
        vectors = list(self._phase_vectors)
        correlations = []
        for i in range(max(0, len(vectors) - 3), len(vectors) - 1):
            v1 = vectors[i]
            v2 = vectors[i + 1]
            min_len = min(len(v1), len(v2))
            corr = torch.dot(v1[:min_len], v2[:min_len]).item()
            correlations.append(max(0.0, min(1.0, corr)))

        coherence = sum(correlations) / len(correlations) if correlations else 1.0
        self._coherence_scores.append(coherence)
        return coherence

    def _track_bifractal(self, tensor: torch.Tensor) -> float:
        """Recursive reactivation pattern detection.

        Creates an 8-D fingerprint (6 activation stats + 2 derivative stats)
        and looks for recurring patterns in a 5-step window.
        """
        flat = tensor.flatten().float()

        if len(flat) < 3:
            self._recursion_scores.append(0.0)
            return 0.0

        # 8-D fingerprint: [mean, std, max, min, median, sum] + [diff_mean, diff_std]
        stats = torch.tensor([
            torch.mean(flat).item(),
            torch.std(flat).item() if len(flat) > 1 else 0.0,
            torch.max(flat).item(),
            torch.min(flat).item(),
            torch.median(flat).item(),
            torch.sum(torch.abs(flat)).item(),
        ])

        diffs = torch.diff(flat)
        diff_stats = torch.tensor([
            torch.mean(diffs).item(),
            torch.std(diffs).item() if len(diffs) > 1 else 0.0,
        ])

        fingerprint = torch.cat([stats, diff_stats])
        self._fingerprints.append(fingerprint.detach().clone())

        # Look for recursive patterns in last 5 fingerprints
        recursion_strength = 0.0
        fps = list(self._fingerprints)
        if len(fps) >= 2:
            current = fps[-1]
            window = fps[max(0, len(fps) - 6):-1]  # Last 5 (excluding current)

            for i, prev_fp in enumerate(window):
                # Cosine similarity
                norm_curr = torch.norm(current)
                norm_prev = torch.norm(prev_fp)
                if norm_curr > 1e-8 and norm_prev > 1e-8:
                    similarity = torch.dot(current, prev_fp) / (norm_curr * norm_prev)
                    sim_val = similarity.item()

                    if sim_val > 0.7:
                        # Temporal weighting — recent matches score higher
                        recency = (i + 1) / len(window)
                        recursion_strength = max(recursion_strength, sim_val * recency)

            # Track in mathematical memory
            if recursion_strength > 0.7:
                matched = False
                for mem in self._mathematical_memory:
                    mem_fp = mem["fingerprint"]
                    if torch.norm(current - mem_fp) < 0.5:
                        mem["recurrence_count"] += 1
                        mem["strength"] = max(mem["strength"], recursion_strength)
                        matched = True
                        break
                if not matched and len(self._mathematical_memory) < self._window:
                    self._mathematical_memory.append({
                        "fingerprint": current.detach().clone(),
                        "strength": recursion_strength,
                        "first_seen": self._step,
                        "recurrence_count": 1,
                    })

        self._recursion_scores.append(recursion_strength)
        return recursion_strength

    def _compute_attractor_density(self, tensor: torch.Tensor) -> float:
        """Centroid clustering analysis for activation attractors.

        Measures how tightly clustered recent activation centroids are.
        High density = distinct, stable attractors. Low = dispersed.
        """
        flat = tensor.flatten().float()

        # Centroid: [mean, std, max, min]
        centroid = torch.tensor([
            torch.mean(flat).item(),
            torch.std(flat).item() if len(flat) > 1 else 0.0,
            torch.max(flat).item(),
            torch.min(flat).item(),
        ])
        self._centroids.append(centroid.detach().clone())

        if len(self._centroids) < 3:
            density = 0.5
            self._densities.append(density)
            return density

        # Pairwise distances between recent 3 centroids
        recent = list(self._centroids)[-3:]
        distances = []
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                dist = torch.norm(recent[i] - recent[j]).item()
                distances.append(dist)

        avg_dist = sum(distances) / len(distances) if distances else 0.0

        # Density = 1 / (1 + avg_distance)
        base_density = 1.0 / (1.0 + avg_dist)

        # Stability factor from density history
        if len(self._densities) >= 2:
            prev = list(self._densities)[-1]
            stability = 1.0 - min(abs(base_density - prev), 1.0)
        else:
            stability = 0.5

        density = 0.7 * base_density + 0.3 * stability
        self._densities.append(density)
        return density

    @property
    def collapse_events(self) -> list[CollapseEvent]:
        return list(self._collapse_events)

    @property
    def step(self) -> int:
        return self._step


# ─── GAIAModule Wrapper ──────────────────────────────────────────────


class ObservabilityModule:
    """GAIA Observability Module — SCBF instrumentation + QBE monitoring.

    Pure observation: computes metrics from the FieldState tensor without
    modifying it. PAC conservation is trivially satisfied (identity on tensor).

    The module reports system health via QBE equilibrium status, which maps
    to SEC phases for bus-level routing decisions.

    Args:
        memory_window: Rolling window size for all metric histories.
        sensitivity_factor: Collapse detection sensitivity multiplier.
    """

    def __init__(
        self,
        memory_window: int = 30,
        sensitivity_factor: float = 2.0,
    ) -> None:
        self._tracker = SCBFTracker(memory_window, sensitivity_factor)
        self._qbe = QBEController()
        self._last_metrics: Optional[ObservabilityMetrics] = None
        self._step_count = 0

    @property
    def name(self) -> str:
        return "observability"

    def process(self, field_state: FieldState) -> FieldState:
        """Observe field state — compute metrics, pass tensor through unchanged."""
        self._step_count += 1
        result = field_state.clone()

        # Compute SCBF metrics from the tensor
        scbf = self._tracker.compute(field_state.tensor)

        # Update QBE from entropy collapse signal
        error = scbf.entropy_collapse - 2.0  # Deviation from "ordered" entropy
        self._qbe.update(error, scbf.entropy_collapse)

        # Adjust QBE based on detected patterns
        if len(self._tracker._raw_entropies) > 10:
            pattern = self._qbe.detect_pattern_type(
                list(self._tracker._raw_entropies)
            )
            self._qbe.adjust_for_pattern(pattern)

        # Store metrics
        self._last_metrics = ObservabilityMetrics(
            scbf=scbf,
            qbe_status=self._qbe.get_status(),
            qbe_energy_balance=self._qbe.energy_balance,
            step_count=self._step_count,
        )

        # Tensor passes through unchanged — PAC trivially conserved
        result.provenance.append(self.name)
        return result

    def phase(self) -> SECPhase:
        """SEC phase from QBE equilibrium status."""
        return self._qbe.to_sec_phase()

    def health(self) -> RBFBalance:
        """RBF balance from SCBF metrics.

        Energy = pattern consistency (how stable are the patterns).
        Information = attractor density (how structured is the space).
        Memory = mathematical memory utilization.
        """
        if self._last_metrics:
            scbf = self._last_metrics.scbf
            energy = scbf.pattern_consistency
            information = scbf.attractor_density
            memory = min(scbf.mathematical_memory_size, 30) / 30.0
        else:
            energy = 1.0
            information = 0.5
            memory = 0.0
        return RBFBalance.compute(energy=energy, information=information, memory=memory)

    @property
    def metrics(self) -> Optional[ObservabilityMetrics]:
        return self._last_metrics

    @property
    def tracker(self) -> SCBFTracker:
        return self._tracker

    @property
    def qbe(self) -> QBEController:
        return self._qbe
