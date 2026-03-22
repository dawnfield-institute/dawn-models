"""Behavioral benchmark runner — brain-like evaluation for GAIA v2.

Runs standardized scenarios through a full 5-module ConservationBus
and measures behavioral coherence rather than prediction accuracy.
"""

from __future__ import annotations

from gaia.body.scenarios import (
    run_adaptation,
    run_habituation,
    run_novelty,
    run_preference,
)
from gaia.core.bus import ConservationBus
from gaia.modules.language import LanguageModule
from gaia.modules.memory import MemoryModule
from gaia.modules.observability import ObservabilityModule
from gaia.modules.reasoning import ReasoningModule
from gaia.modules.safety import SafetyModule


# Behavioral thresholds — these are initial baselines, will be tuned
# as we understand the system's behavioral signatures better.
BEHAVIORAL_THRESHOLDS: dict[str, tuple[float, str]] = {
    "habituation_rate": (0.0, "gt"),               # Must show some decay
    "surprise_ratio": (0.5, "gt"),                  # Must respond to novelty
    "pre_novelty_coherence": (-0.5, "gt"),          # Not purely random
    "adaptation_latency": (26.0, "lt"),             # Must respond within window (baseline: 25)
    "coherence_before": (-0.5, "gt"),               # Coherent in familiar env
    "mean_preference_divergence": (0.0, "gt"),      # Must develop some preference
}


def _make_full_bus(input_dim: int = 22) -> ConservationBus:
    """Create bus with all 5 modules. input_dim = 3*3 + 9 visual + 4 proprio = 22."""
    bus = ConservationBus(enforcement="soft")
    bus.register_module(ObservabilityModule())
    bus.register_module(LanguageModule())
    bus.register_module(MemoryModule())
    bus.register_module(SafetyModule(input_dim=input_dim))
    bus.register_module(ReasoningModule(input_dim=input_dim))
    return bus


def bench_behavioral(seed: int = 42) -> dict[str, float]:
    """Run all behavioral scenarios and return combined metrics."""
    results: dict[str, float] = {}

    print("  Habituation...")
    bus = _make_full_bus()
    results.update(run_habituation(bus, n_ticks=50, seed=seed))

    print("  Novelty...")
    bus = _make_full_bus()
    results.update(run_novelty(bus, n_familiar=20, seed=seed))

    print("  Adaptation...")
    bus = _make_full_bus()
    results.update(run_adaptation(bus, n_before=25, n_after=25, seed=seed))

    print("  Preference...")
    bus = _make_full_bus()
    results.update(run_preference(bus, n_ticks_per_class=30, seed=seed))

    return results


def print_behavioral_scorecard(results: dict[str, float]) -> bool:
    """Print formatted behavioral scorecard. Returns True if all pass."""
    title = "GAIA v2 BEHAVIORAL SCORECARD"
    print(f"\n{title}")
    print("=" * len(title))

    all_pass = True
    n_pass = 0
    n_total = 0

    for key in sorted(results.keys()):
        value = results[key]
        if key in BEHAVIORAL_THRESHOLDS:
            threshold, direction = BEHAVIORAL_THRESHOLDS[key]
            passed = (value > threshold) if direction == "gt" else (value < threshold)
            sym = ">" if direction == "gt" else "<"
            status = "PASS" if passed else "FAIL"
            print(f"  {key + ':':34s}{value:>12.4f}  {status}  (threshold: {sym}{threshold:g})")
            if not passed:
                all_pass = False
            n_total += 1
            if passed:
                n_pass += 1
        else:
            print(f"  {key + ':':34s}{value:>12.4f}")

    print(f"\nRESULT: {n_pass}/{n_total} PASS")
    return all_pass
