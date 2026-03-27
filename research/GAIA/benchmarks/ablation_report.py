"""Ablation report runner — prints the full ablation matrix, probes, and coherence analysis.

Usage:
    cd research/GAIA
    PYTHONPATH="src:../.." python benchmarks/ablation_report.py
"""

from __future__ import annotations

from gaia.body.ablation import ALL_MODULES, AblationReport, make_bus, run_full_ablation
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

# Columns to display (subset of all metrics for readability)
DISPLAY_METRICS = [
    "habituation_rate",
    "surprise_ratio",
    "adaptation_latency",
    "coherence_before",
    "mean_preference_divergence",
]

SHORT_NAMES = {
    "habituation_rate": "hab_rate",
    "surprise_ratio": "surprise",
    "adaptation_latency": "adapt_lat",
    "coherence_before": "coherence",
    "mean_preference_divergence": "pref_div",
}


def print_ablation_matrix(report: AblationReport) -> None:
    """Print the full ablation matrix: rows = configs, columns = metrics."""
    title = "GAIA v2 ABLATION MATRIX"
    print(f"\n{title}")
    print("=" * len(title))

    matrix = report.metric_matrix()
    cols = DISPLAY_METRICS

    # Header
    header = f"  {'config':28s}"
    for col in cols:
        header += f"  {SHORT_NAMES.get(col, col):>10s}"
    print(header)
    print("  " + "-" * (28 + len(cols) * 12))

    # Rows
    for result in report.results:
        row = f"  {result.config.name:28s}"
        for col in cols:
            val = result.metrics.get(col, float("nan"))
            row += f"  {val:>10.4f}"
        print(row)


def print_delta_matrix(report: AblationReport) -> None:
    """Print deltas from baseline."""
    title = "DELTAS FROM BASELINE (full)"
    print(f"\n{title}")
    print("=" * len(title))

    deltas = report.delta_matrix()
    cols = DISPLAY_METRICS

    # Header
    header = f"  {'config':28s}"
    for col in cols:
        header += f"  {SHORT_NAMES.get(col, col):>10s}"
    print(header)
    print("  " + "-" * (28 + len(cols) * 12))

    for config_name, row in deltas.items():
        line = f"  {config_name:28s}"
        for col in cols:
            val = row.get(col, float("nan"))
            sign = "+" if val > 0 else ""
            line += f"  {sign}{val:>9.4f}"
        print(line)


def print_probe_results(results: list[ProbeResult]) -> None:
    """Print probe hypotheses and verdicts."""
    title = "BEHAVIORAL PROBES"
    print(f"\n{title}")
    print("=" * len(title))

    for r in results:
        status = "SUPPORTED" if r.supported else "REFUTED"
        print(f"\n  [{status}] {r.name}")
        print(f"    Hypothesis: {r.hypothesis}")
        print(f"    Evidence:   {r.evidence}")


def print_cross_module_summary(
    traj_data: tuple,
) -> None:
    """Print energy redistribution, provenance frequency, phase trajectory."""
    traj, health = traj_data

    title = "CROSS-MODULE COHERENCE"
    print(f"\n{title}")
    print("=" * len(title))

    # Provenance frequency
    print("\n  Module Provenance Frequency:")
    freqs = provenance_frequency(traj)
    for name in sorted(freqs.keys()):
        bar = "#" * int(freqs[name] * 20)
        print(f"    {name:20s} {freqs[name]:.2f}  {bar}")

    # Phase trajectory summary
    phases = phase_trajectory(traj)
    phase_counts: dict[str, int] = {}
    for p in phases:
        phase_counts[p] = phase_counts.get(p, 0) + 1
    print("\n  SEC Phase Distribution:")
    for phase, count in sorted(phase_counts.items()):
        pct = count / len(phases) * 100
        print(f"    {phase:20s} {count:3d} ticks ({pct:.0f}%)")

    # Energy quadrant summary
    quads = energy_quadrant_analysis(traj)
    print("\n  Energy Quadrants (mean over trajectory):")
    for quad_name in ["top", "bottom", "left", "right"]:
        vals = quads[quad_name]
        mean_val = sum(vals) / len(vals) if vals else 0.0
        print(f"    {quad_name:10s} {mean_val:>10.4f}")

    # Module health trajectory
    print("\n  Module Health (RBF Balance) — first/mid/last tick:")
    if health:
        first = health[0]
        mid = health[len(health) // 2]
        last = health[-1]
        for mod in sorted(first.keys()):
            print(
                f"    {mod:20s}  t=0: {first[mod]:>7.4f}  "
                f"t={len(health)//2}: {mid[mod]:>7.4f}  "
                f"t={len(health)-1}: {last[mod]:>7.4f}"
            )


def main() -> None:
    """Run the full Phase 3 ablation report."""
    print("Running GAIA v2 Phase 3: Multi-Module Ablation Study\n")

    # 1. Full ablation
    print("Running ablation matrix (14 configurations x 4 scenarios)...")
    report = run_full_ablation(seed=42)
    print_ablation_matrix(report)
    print_delta_matrix(report)

    # 2. Behavioral probes
    print("\nRunning behavioral probes...")
    probes = [
        probe_safety_stabilizer(seed=42),
        probe_memory_context(seed=42),
        probe_reasoning_attractor(seed=42),
        probe_language_predictor(seed=42),
    ]
    print_probe_results(probes)

    # 3. Cross-module coherence (full brain, instrumented)
    print("\nRunning instrumented coherence analysis...")
    bus = make_bus(ALL_MODULES, input_dim=22)
    traj_data = run_instrumented(bus, n_ticks=50, seed=42)
    print_cross_module_summary(traj_data)

    # Summary
    n_supported = sum(1 for p in probes if p.supported)
    print(f"\n{'=' * 40}")
    print(f"PROBES: {n_supported}/{len(probes)} hypotheses supported")
    print(f"CONFIGS: {len(report.results)} ablation configurations evaluated")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
