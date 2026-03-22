"""GAIA v2 Benchmark Runner.

Usage:
    cd research/GAIA
    python -m benchmarks.run_benchmarks [--real] [--verbose]
"""

from __future__ import annotations

import argparse
import sys

from .scorecard import print_scorecard, THRESHOLDS
from .synthetic import (
    bench_efficiency,
    bench_continuous_learning,
    bench_hallucination_detection,
    bench_hallucination_e2e,
    bench_memory,
)


def run_synthetic() -> dict[str, float]:
    """Run all synthetic benchmarks. Always works, no external deps."""
    results: dict[str, float] = {}

    print("\n--- Efficiency ---")
    results.update(bench_efficiency())

    print("\n--- Continuous Learning ---")
    results.update(bench_continuous_learning())

    print("\n--- Hallucination Detection ---")
    results.update(bench_hallucination_detection())

    print("\n--- Memory ---")
    results.update(bench_memory())

    print("\n--- E2E Hallucination ---")
    results.update(bench_hallucination_e2e())

    return results


def run_real() -> dict[str, float] | None:
    """Run real data benchmarks. Requires transformers + datasets."""
    from .real_data import _has_deps

    if not _has_deps():
        print("\nSkipping real benchmarks (install: pip install transformers datasets)")
        return None

    from .real_data import bench_wikitext2

    print("\n--- WikiText-2 ---")
    return bench_wikitext2()


def main():
    parser = argparse.ArgumentParser(description="GAIA v2 Benchmark Suite")
    parser.add_argument("--real", action="store_true", help="Run real data benchmarks")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Synthetic (always)
    synthetic_results = run_synthetic()
    all_pass = print_scorecard(synthetic_results, title="SYNTHETIC SCORECARD")

    # Real data (optional)
    if args.real:
        real_results = run_real()
        if real_results:
            print_scorecard(
                real_results,
                thresholds={"wikitext2_hit_rate": (0.55, "gt")},
                title="REAL DATA SCORECARD",
            )

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
