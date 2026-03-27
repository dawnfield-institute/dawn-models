"""Behavioral probes — hypothesis-driven experiments for module contributions.

Each probe tests a specific causal claim about a module's role by comparing
behavioral metrics with vs. without the target module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gaia.core.bus import ConservationBus

from .ablation import ALL_MODULES, make_bus
from .environment import GridWorld
from .loop import BodyLoop, Trajectory
from .metrics import TrajectoryAnalyzer
from .motor import GridMotorDecoder
from .scenarios import run_adaptation, run_habituation, run_novelty
from .senses import ProprioceptiveChannel, VisualChannel


# ─── Probe Result ─────────────────────────────────────────────────


@dataclass
class ProbeResult:
    """Result from a behavioral probe."""

    name: str
    hypothesis: str
    with_module: dict[str, float]
    without_module: dict[str, float]
    delta: dict[str, float]
    supported: bool
    evidence: str


# ─── Probes ───────────────────────────────────────────────────────


def probe_safety_stabilizer(seed: int = 42, input_dim: int = 22) -> ProbeResult:
    """Hypothesis: Safety module increases behavioral stability.

    Compares coherence in the adaptation scenario with vs. without safety.
    """
    bus_with = make_bus(ALL_MODULES, input_dim=input_dim)
    metrics_with = run_adaptation(bus_with, seed=seed)

    without = [m for m in ALL_MODULES if m != "safety"]
    bus_without = make_bus(without, input_dim=input_dim)
    metrics_without = run_adaptation(bus_without, seed=seed)

    delta = {k: metrics_with[k] - metrics_without.get(k, 0.0) for k in metrics_with}

    # Safety should increase coherence (more stable behavior)
    key = "coherence_before"
    supported = metrics_with[key] >= metrics_without[key]

    return ProbeResult(
        name="safety_stabilizer",
        hypothesis="Safety module increases behavioral stability (coherence)",
        with_module=metrics_with,
        without_module=metrics_without,
        delta=delta,
        supported=supported,
        evidence=(
            f"coherence_before: {metrics_with[key]:.4f} (with) vs "
            f"{metrics_without[key]:.4f} (without). "
            f"{'Supported' if supported else 'Refuted'}: safety "
            f"{'increases' if supported else 'does not increase'} coherence."
        ),
    )


def probe_memory_context(seed: int = 42, input_dim: int = 22) -> ProbeResult:
    """Hypothesis: Memory module enables faster adaptation.

    Compares adaptation latency with vs. without memory.
    """
    bus_with = make_bus(ALL_MODULES, input_dim=input_dim)
    metrics_with = run_adaptation(bus_with, seed=seed)

    without = [m for m in ALL_MODULES if m != "memory"]
    bus_without = make_bus(without, input_dim=input_dim)
    metrics_without = run_adaptation(bus_without, seed=seed)

    delta = {k: metrics_with[k] - metrics_without.get(k, 0.0) for k in metrics_with}

    # Memory should reduce adaptation latency (faster response)
    key = "adaptation_latency"
    supported = metrics_with[key] <= metrics_without[key]

    return ProbeResult(
        name="memory_context",
        hypothesis="Memory module enables faster adaptation (lower latency)",
        with_module=metrics_with,
        without_module=metrics_without,
        delta=delta,
        supported=supported,
        evidence=(
            f"adaptation_latency: {metrics_with[key]:.1f} (with) vs "
            f"{metrics_without[key]:.1f} (without). "
            f"{'Supported' if supported else 'Refuted'}: memory "
            f"{'reduces' if supported else 'does not reduce'} adaptation latency."
        ),
    )


def probe_reasoning_attractor(seed: int = 42, input_dim: int = 22) -> ProbeResult:
    """Hypothesis: Reasoning module's phi-convergence creates behavioral attractors.

    Compares coherence in the habituation scenario with vs. without reasoning.
    Higher coherence = stronger attractor behavior.
    """
    bus_with = make_bus(ALL_MODULES, input_dim=input_dim)
    metrics_with = run_habituation(bus_with, seed=seed)

    without = [m for m in ALL_MODULES if m != "reasoning"]
    bus_without = make_bus(without, input_dim=input_dim)
    metrics_without = run_habituation(bus_without, seed=seed)

    # Add coherence from a separate habituation run
    bus_c1 = make_bus(ALL_MODULES, input_dim=input_dim)
    loop1 = BodyLoop(
        bus_c1,
        [VisualChannel(), ProprioceptiveChannel()],
        GridMotorDecoder(),
        GridWorld(size=5, n_stimuli=3, seed=seed),
    )
    traj1 = loop1.run(50)
    ci_with = TrajectoryAnalyzer.coherence_index(traj1)

    bus_c2 = make_bus(without, input_dim=input_dim)
    loop2 = BodyLoop(
        bus_c2,
        [VisualChannel(), ProprioceptiveChannel()],
        GridMotorDecoder(),
        GridWorld(size=5, n_stimuli=3, seed=seed),
    )
    traj2 = loop2.run(50)
    ci_without = TrajectoryAnalyzer.coherence_index(traj2)

    metrics_with["coherence_index"] = ci_with
    metrics_without["coherence_index"] = ci_without

    delta = {k: metrics_with[k] - metrics_without.get(k, 0.0) for k in metrics_with}

    supported = ci_with >= ci_without

    return ProbeResult(
        name="reasoning_attractor",
        hypothesis="Reasoning module creates behavioral attractors (higher coherence)",
        with_module=metrics_with,
        without_module=metrics_without,
        delta=delta,
        supported=supported,
        evidence=(
            f"coherence_index: {ci_with:.4f} (with) vs {ci_without:.4f} (without). "
            f"{'Supported' if supported else 'Refuted'}: reasoning "
            f"{'creates' if supported else 'does not create'} attractor behavior."
        ),
    )


def probe_language_predictor(seed: int = 42, input_dim: int = 22) -> ProbeResult:
    """Hypothesis: Language module's predictions create expectations, producing bigger surprise.

    Compares surprise_ratio in the novelty scenario with vs. without language.
    A predictive system should show larger surprise when expectations are violated.
    """
    bus_with = make_bus(ALL_MODULES, input_dim=input_dim)
    metrics_with = run_novelty(bus_with, seed=seed)

    without = [m for m in ALL_MODULES if m != "language"]
    bus_without = make_bus(without, input_dim=input_dim)
    metrics_without = run_novelty(bus_without, seed=seed)

    delta = {k: metrics_with[k] - metrics_without.get(k, 0.0) for k in metrics_with}

    key = "surprise_ratio"
    supported = metrics_with[key] > metrics_without[key]

    return ProbeResult(
        name="language_predictor",
        hypothesis="Language module creates expectations, producing larger surprise on novelty",
        with_module=metrics_with,
        without_module=metrics_without,
        delta=delta,
        supported=supported,
        evidence=(
            f"surprise_ratio: {metrics_with[key]:.4f} (with) vs "
            f"{metrics_without[key]:.4f} (without). "
            f"{'Supported' if supported else 'Refuted'}: language "
            f"{'amplifies' if supported else 'does not amplify'} surprise response."
        ),
    )


# ─── Instrumented Run ─────────────────────────────────────────────


def run_instrumented(
    bus: ConservationBus,
    n_ticks: int = 50,
    seed: int = 42,
) -> tuple[Trajectory, list[dict[str, float]]]:
    """Run a body loop and capture per-tick module health snapshots.

    Returns:
        trajectory: Full trajectory from the run.
        health_snapshots: Per-tick list of {module_name: RBF balance}.
    """
    loop = BodyLoop(
        bus,
        [VisualChannel(), ProprioceptiveChannel()],
        GridMotorDecoder(),
        GridWorld(size=5, n_stimuli=3, seed=seed),
    )

    health_snapshots: list[dict[str, float]] = []
    for _ in range(n_ticks):
        loop.tick()
        # Capture health from all registered modules
        snapshot: dict[str, float] = {}
        for name, module in bus._modules.items():
            rbf = module.health()
            snapshot[name] = rbf.balance
        health_snapshots.append(snapshot)

    return loop.trajectory, health_snapshots


# ─── Cross-Module Analysis ────────────────────────────────────────


def energy_quadrant_analysis(trajectory: Trajectory) -> dict[str, list[float]]:
    """Per-tick energy distribution in tensor halves/interleaved.

    Maps directly to GridMotorDecoder's dx/dy computation:
    - top_energy / bottom_energy → dy
    - left_energy (even) / right_energy (odd) → dx
    """
    top: list[float] = []
    bottom: list[float] = []
    left: list[float] = []
    right: list[float] = []

    for tick in trajectory.ticks:
        flat = tick.field_state_out.tensor.flatten().float()
        n = flat.numel()
        half = n // 2

        top.append(float(flat[:half].sum().item()))
        bottom.append(float(flat[half:].sum().item()))
        left.append(float(flat[::2].sum().item()))
        right.append(float(flat[1::2].sum().item()))

    return {"top": top, "bottom": bottom, "left": left, "right": right}


def phase_trajectory(trajectory: Trajectory) -> list[str]:
    """SEC phase at each tick's output FieldState."""
    return [tick.field_state_out.phase.value for tick in trajectory.ticks]


def provenance_frequency(trajectory: Trajectory) -> dict[str, float]:
    """Fraction of ticks where each module appears in provenance.

    A module with frequency < 1.0 was sometimes RBF-suppressed.
    """
    if not trajectory.ticks:
        return {}

    counts: dict[str, int] = {}
    n = len(trajectory.ticks)

    for tick in trajectory.ticks:
        for mod_name in tick.field_state_out.provenance:
            counts[mod_name] = counts.get(mod_name, 0) + 1

    return {name: count / n for name, count in counts.items()}
