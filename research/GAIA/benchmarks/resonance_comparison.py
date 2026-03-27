"""Resonance vs Sequential — side-by-side behavioral comparison.

Runs the same 4 scenarios with the same seed through both ConservationBus
(sequential pipeline) and ResonanceBus (broadcast + delta-coherence merge),
then prints comparative scorecards and resonance weight distributions.

Usage:
    cd research/GAIA
    PYTHONPATH="src;c:/Users/peter/repos/core_workspace/fracton" python benchmarks/resonance_comparison.py
"""

from __future__ import annotations

from gaia.body.ablation import ALL_MODULES, ModuleConfig, make_bus, make_resonance_bus
from gaia.body.probes import provenance_frequency, run_instrumented
from gaia.body.scenarios import run_adaptation, run_habituation, run_novelty, run_preference

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

CONFIGS = [
    ModuleConfig("full", list(ALL_MODULES), "All 5 modules"),
    ModuleConfig("no_safety", [m for m in ALL_MODULES if m != "safety"], "Without safety"),
    ModuleConfig("memory+language", ["observability", "memory", "language"], "Obs + Memory + Language"),
    ModuleConfig("observability_only", ["observability"], "Identity baseline"),
]


def run_scenarios(bus, seed: int = 42) -> dict[str, float]:
    """Run all 4 behavioral scenarios on a bus, return metrics."""
    metrics: dict[str, float] = {}
    # Each scenario needs a fresh bus state, but we reuse same modules.
    # The bus itself is stateless between process() calls (no hidden state).
    metrics.update(run_habituation(bus, n_ticks=50, seed=seed))

    # Rebuild bus for each scenario to reset any internal state
    return metrics


def run_all_scenarios(bus_factory, module_names, seed=42, **kwargs) -> dict[str, float]:
    """Run all 4 scenarios with fresh buses from the factory."""
    metrics: dict[str, float] = {}

    bus = bus_factory(module_names, **kwargs)
    metrics.update(run_habituation(bus, n_ticks=50, seed=seed))

    bus = bus_factory(module_names, **kwargs)
    metrics.update(run_novelty(bus, n_familiar=20, seed=seed))

    bus = bus_factory(module_names, **kwargs)
    metrics.update(run_adaptation(bus, n_before=25, n_after=25, seed=seed))

    bus = bus_factory(module_names, **kwargs)
    metrics.update(run_preference(bus, n_ticks_per_class=30, seed=seed))

    return metrics


def print_comparison(
    seq_results: dict[str, dict[str, float]],
    res_results: dict[str, dict[str, float]],
) -> None:
    """Print side-by-side comparison table."""
    title = "SEQUENTIAL vs RESONANCE DISPATCH"
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    cols = DISPLAY_METRICS

    for config_name in seq_results:
        seq = seq_results[config_name]
        res = res_results[config_name]

        print(f"\n  Config: {config_name}")
        print(f"  {'metric':>15s}  {'sequential':>12s}  {'resonance':>12s}  {'delta':>10s}")
        print(f"  {'-' * 55}")

        for col in cols:
            s_val = seq.get(col, float("nan"))
            r_val = res.get(col, float("nan"))
            delta = r_val - s_val
            sign = "+" if delta > 0 else ""
            print(f"  {SHORT_NAMES.get(col, col):>15s}  {s_val:>12.4f}  {r_val:>12.4f}  {sign}{delta:>9.4f}")


def print_resonance_weights(res_bus, config_name: str) -> None:
    """Print the resonance weight distribution from a bus."""
    if not res_bus.resonance_log:
        return

    print(f"\n  Resonance Weights ({config_name}) — last 5 ticks:")

    log = res_bus.resonance_log
    show = log[-5:] if len(log) > 5 else log
    for i, weights in enumerate(show):
        tick = len(log) - len(show) + i
        parts = []
        for w in sorted(weights, key=lambda x: -x.normalized_weight):
            parts.append(f"{w.module_name}={w.normalized_weight:.3f}")
        print(f"    t={tick:3d}: {', '.join(parts)}")


def print_provenance_comparison(seq_bus, res_bus, config_name: str) -> None:
    """Compare provenance frequencies between sequential and resonance."""
    print(f"\n  Provenance Frequency ({config_name}):")

    seq_traj, _ = run_instrumented(seq_bus, n_ticks=30, seed=42)
    res_traj, _ = run_instrumented(res_bus, n_ticks=30, seed=42)

    seq_freq = provenance_frequency(seq_traj)
    res_freq = provenance_frequency(res_traj)

    all_names = sorted(set(list(seq_freq.keys()) + list(res_freq.keys())))
    print(f"  {'module':>20s}  {'sequential':>12s}  {'resonance':>12s}")
    print(f"  {'-' * 48}")
    for name in all_names:
        s = seq_freq.get(name, 0.0)
        r = res_freq.get(name, 0.0)
        print(f"  {name:>20s}  {s:>12.2f}  {r:>12.2f}")


def main() -> None:
    """Run the full comparison."""
    print("GAIA v2 — Sequential vs Resonance Dispatch Comparison")
    print("=" * 55)

    seed = 42
    seq_results: dict[str, dict[str, float]] = {}
    res_results: dict[str, dict[str, float]] = {}

    for config in CONFIGS:
        print(f"\n  Running {config.name}...", end=" ", flush=True)

        seq_metrics = run_all_scenarios(make_bus, config.modules, seed=seed)
        seq_results[config.name] = seq_metrics

        res_metrics = run_all_scenarios(make_resonance_bus, config.modules, seed=seed)
        res_results[config.name] = res_metrics

        print("done")

    # Side-by-side metrics
    print_comparison(seq_results, res_results)

    # Resonance weight distributions for the full config
    print(f"\n{'=' * 80}")
    print("  RESONANCE WEIGHT DISTRIBUTIONS")
    print(f"{'=' * 80}")

    for config in CONFIGS:
        res_bus = make_resonance_bus(config.modules)
        # Run 50 ticks to populate resonance log
        from gaia.body.scenarios import run_habituation
        run_habituation(res_bus, n_ticks=50, seed=seed)
        print_resonance_weights(res_bus, config.name)

    # Provenance comparison for key configs
    print(f"\n{'=' * 80}")
    print("  PROVENANCE FREQUENCY COMPARISON")
    print(f"{'=' * 80}")

    for config in [CONFIGS[0], CONFIGS[2]]:  # full and memory+language
        seq_bus = make_bus(config.modules)
        res_bus = make_resonance_bus(config.modules)
        print_provenance_comparison(seq_bus, res_bus, config.name)

    # Summary
    print(f"\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}")

    # Count how many metrics differ between sequential and resonance
    total_diffs = 0
    total_metrics = 0
    for config_name in seq_results:
        for col in DISPLAY_METRICS:
            s = seq_results[config_name].get(col, 0)
            r = res_results[config_name].get(col, 0)
            total_metrics += 1
            if abs(s - r) > 1e-6:
                total_diffs += 1

    print(f"\n  Metrics that differ: {total_diffs}/{total_metrics}")
    print(f"  Configs tested: {len(CONFIGS)}")
    print(f"  Dispatch modes: sequential (ConservationBus) vs resonance (ResonanceBus)")
    print()


if __name__ == "__main__":
    main()
