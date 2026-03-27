"""Comparison benchmark: Sequential vs Resonance v1 vs Response-Field vs Harmonic.

Runs all 4 buses through the same behavioral scenarios and compares:
1. Weight distributions per module (who gets heard?)
2. Behavioral metrics (habituation, novelty, adaptation, preference)
3. Provenance patterns (which modules appear in output?)

Usage:
    cd dawn-models/research/GAIA
    PYTHONPATH="src;c:/Users/peter/repos/core_workspace/fracton" python spikes/resonance_bus/run_comparison.py
"""

from __future__ import annotations

import sys
import os

# Path setup
_here = os.path.dirname(os.path.abspath(__file__))
_gaia_root = os.path.join(_here, "..", "..")
sys.path.insert(0, os.path.join(_gaia_root, "src"))
sys.path.insert(0, os.path.join(_here, "..", "..", "..", "..", "..", "fracton"))

import torch

from gaia.core.bus import ConservationBus
from gaia.core.resonance_bus import ResonanceBus
from gaia.core.types import FieldState
from gaia.modules.language import LanguageModule
from gaia.modules.memory import MemoryModule
from gaia.modules.observability import ObservabilityModule
from gaia.modules.reasoning import ReasoningModule
from gaia.modules.safety import SafetyModule
from gaia.body.scenarios import run_habituation, run_novelty, run_adaptation, run_preference

from spike_a_response_field import ResponseFieldBus
from spike_b_harmonic_weight import HarmonicBus
from spike_c_perspective import PerspectiveBus
from spike_d_continuous import ContinuousFieldBus
from spike_e_qbe_regulated import QBEFieldBus
from spike_f_coupled_fields import CoupledFieldsBus

# ─── Module factories ────────────────────────────────────────────

MODULE_FACTORIES = {
    "observability": lambda dim: ObservabilityModule(),
    "safety": lambda dim: SafetyModule(input_dim=dim),
    "reasoning": lambda dim: ReasoningModule(input_dim=dim),
    "memory": lambda dim: MemoryModule(),
    "language": lambda dim: LanguageModule(),
}

ALL_MODULES = list(MODULE_FACTORIES.keys())
INPUT_DIM = 22
SEED = 42
N_WEIGHT_PROBES = 20  # ticks to collect weight data


def make_bus(bus_class, module_names=None, **kwargs):
    """Create a bus with modules registered."""
    if module_names is None:
        module_names = ALL_MODULES
    bus = bus_class(enforcement="soft", **kwargs)
    for name in module_names:
        bus.register_module(MODULE_FACTORIES[name](INPUT_DIM))
    return bus


# ─── Weight Distribution Probe ──────────────────────────────────

def probe_weights(bus, n_ticks=N_WEIGHT_PROBES, seed=SEED):
    """Run n_ticks through the bus and collect weight distributions.

    For ConservationBus (sequential), we can't get weights — just provenance.
    For resonance variants, we extract from their weight logs.
    """
    torch.manual_seed(seed)

    for _ in range(n_ticks):
        tensor = torch.randn(INPUT_DIM) * 5.0 + 2.0  # Non-trivial energy
        flat_abs = tensor.flatten().float().abs(); total_e = flat_abs.sum(); entropy = float(-(flat_abs[flat_abs > 1e-12] / total_e * (flat_abs[flat_abs > 1e-12] / total_e).log()).sum().item()) if total_e > 1e-12 else 0.0; state = FieldState(tensor=tensor, entropy=entropy)
        bus.process(state)

    # Extract weight history
    if hasattr(bus, '_resonance_log'):
        return _extract_weight_history(bus._resonance_log)
    elif hasattr(bus, '_response_log'):
        return _extract_weight_history(bus._response_log)
    elif hasattr(bus, '_harmonic_log'):
        return _extract_weight_history(bus._harmonic_log)
    elif hasattr(bus, '_perspective_log'):
        return _extract_weight_history(bus._perspective_log)
    elif hasattr(bus, '_continuous_log'):
        return _extract_weight_history(bus._continuous_log)
    elif hasattr(bus, '_qbe_log'):
        return _extract_weight_history(bus._qbe_log)
    elif hasattr(bus, '_coupled_log'):
        return _extract_weight_history(bus._coupled_log)
    else:
        return None  # Sequential bus has no weight log


def _extract_weight_history(log):
    """Extract per-module mean weights from a weight log."""
    if not log:
        return {}

    module_weights = {}
    for tick_weights in log:
        for w in tick_weights:
            if w.module_name not in module_weights:
                module_weights[w.module_name] = []
            module_weights[w.module_name].append(w.normalized_weight)

    return {
        name: {
            "mean": sum(ws) / len(ws),
            "min": min(ws),
            "max": max(ws),
            "std": (sum((x - sum(ws)/len(ws))**2 for x in ws) / len(ws)) ** 0.5,
        }
        for name, ws in module_weights.items()
    }


# ─── Behavioral Metrics ─────────────────────────────────────────

def run_behavioral(bus_factory):
    """Run all 4 behavioral scenarios with fresh buses."""
    metrics = {}

    bus = bus_factory()
    metrics.update(run_habituation(bus, n_ticks=50, seed=SEED))

    bus = bus_factory()
    metrics.update(run_novelty(bus, n_familiar=20, seed=SEED))

    bus = bus_factory()
    metrics.update(run_adaptation(bus, n_before=25, n_after=25, seed=SEED))

    bus = bus_factory()
    metrics.update(run_preference(bus, n_ticks_per_class=30, seed=SEED))

    return metrics


# ─── Display ────────────────────────────────────────────────────

def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_weights(weights_by_bus):
    """Print weight distribution comparison table."""
    print_header("WEIGHT DISTRIBUTIONS (mean over 20 ticks)")

    # Get all module names
    all_names = set()
    for weights in weights_by_bus.values():
        if weights:
            all_names.update(weights.keys())
    all_names = sorted(all_names)

    # Header
    bus_names = list(weights_by_bus.keys())
    header = f"{'Module':<15}" + "".join(f"{b:>18}" for b in bus_names)
    print(header)
    print("-" * len(header))

    for mod in all_names:
        row = f"{mod:<15}"
        for bus_name in bus_names:
            weights = weights_by_bus[bus_name]
            if weights and mod in weights:
                w = weights[mod]
                row += f"  {w['mean']:6.1%} +/- {w['std']:5.3f}"
            elif bus_name == "Sequential":
                row += f"  {'(no weights)':>15}"
            else:
                row += f"  {'---':>15}"
        print(row)


def print_weight_details(weights_by_bus):
    """Print detailed weight breakdown for resonance buses."""
    print_header("WEIGHT DETAIL (per module)")

    for bus_name, weights in weights_by_bus.items():
        if bus_name == "Sequential" or not weights:
            continue
        print(f"\n  {bus_name}:")
        print(f"  {'Module':<15} {'Mean':>8} {'Min':>8} {'Max':>8} {'Std':>8}")
        print(f"  {'-'*47}")
        for mod in sorted(weights.keys()):
            w = weights[mod]
            print(f"  {mod:<15} {w['mean']:8.4f} {w['min']:8.4f} {w['max']:8.4f} {w['std']:8.4f}")


def print_behavioral(metrics_by_bus):
    """Print behavioral metrics comparison."""
    print_header("BEHAVIORAL METRICS")

    # Metrics to display (short name -> key)
    display = [
        ("Hab Rate", "habituation_rate"),
        ("Hab 1st", "habituation_first_half_mean"),
        ("Hab 2nd", "habituation_second_half_mean"),
        ("Surprise", "surprise_ratio"),
        ("Pre-Nov Coh", "pre_novelty_coherence"),
        ("Adapt Lat", "adaptation_latency"),
        ("Coh Before", "coherence_before"),
        ("Coh After", "coherence_after"),
        ("Pref AB", "preference_divergence_ab"),
        ("Pref AC", "preference_divergence_ac"),
        ("Mean Pref", "mean_preference_divergence"),
    ]

    bus_names = list(metrics_by_bus.keys())
    header = f"{'Metric':<15}" + "".join(f"{b:>15}" for b in bus_names)
    print(header)
    print("-" * len(header))

    for short, key in display:
        row = f"{short:<15}"
        for bus_name in bus_names:
            val = metrics_by_bus[bus_name].get(key, float('nan'))
            row += f"{val:15.4f}"
        print(row)


def print_behavioral_deltas(metrics_by_bus):
    """Print deltas vs sequential baseline."""
    print_header("BEHAVIORAL DELTAS (vs Sequential)")

    baseline_key = "Sequential"
    if baseline_key not in metrics_by_bus:
        print("  No sequential baseline found")
        return

    baseline = metrics_by_bus[baseline_key]
    display = [
        ("Hab Rate", "habituation_rate"),
        ("Surprise", "surprise_ratio"),
        ("Adapt Lat", "adaptation_latency"),
        ("Mean Pref", "mean_preference_divergence"),
    ]

    bus_names = [b for b in metrics_by_bus if b != baseline_key]
    header = f"{'Metric':<15}" + "".join(f"{b:>18}" for b in bus_names)
    print(header)
    print("-" * len(header))

    for short, key in display:
        row = f"{short:<15}"
        base_val = baseline.get(key, 0.0)
        for bus_name in bus_names:
            val = metrics_by_bus[bus_name].get(key, 0.0)
            delta = val - base_val
            pct = (delta / abs(base_val) * 100) if abs(base_val) > 1e-10 else 0.0
            sign = "+" if delta >= 0 else ""
            row += f"  {sign}{delta:7.4f} ({sign}{pct:5.1f}%)"
        print(row)


# ─── Main ───────────────────────────────────────────────────────

def main():
    print("Resonance Bus Spike Comparison")
    print(f"Modules: {', '.join(ALL_MODULES)}")
    print(f"Input dim: {INPUT_DIM}, Seed: {SEED}")

    # Define bus configurations
    configs = {
        "Sequential": lambda: make_bus(ConservationBus),
        "Resonance v1": lambda: make_bus(ResonanceBus),
        "Response-Field": lambda: make_bus(ResponseFieldBus, response_scale=1.0),
        "Harmonic": lambda: make_bus(HarmonicBus, resonance_floor_threshold=0.5, floor_weight=0.05),
        "Perspective": lambda: make_bus(PerspectiveBus),
        "Continuous": lambda: make_bus(ContinuousFieldBus),
        "QBE": lambda: make_bus(QBEFieldBus),
        "Coupled": lambda: make_bus(CoupledFieldsBus),
    }

    # 1. Weight distribution probe
    print("\nProbing weight distributions...")
    weights_by_bus = {}
    for name, factory in configs.items():
        bus = factory()
        weights_by_bus[name] = probe_weights(bus)
        print(f"  {name}: done")

    print_weights(weights_by_bus)
    print_weight_details(weights_by_bus)

    # 2. Behavioral metrics
    print("\nRunning behavioral scenarios...")
    metrics_by_bus = {}
    for name, factory in configs.items():
        print(f"  {name}...", end=" ", flush=True)
        metrics_by_bus[name] = run_behavioral(factory)
        print("done")

    print_behavioral(metrics_by_bus)
    print_behavioral_deltas(metrics_by_bus)

    # 3. Summary
    print_header("SUMMARY")
    for name, weights in weights_by_bus.items():
        if weights:
            n_active = sum(1 for w in weights.values() if w["mean"] > 0.01)
            max_mod = max(weights.items(), key=lambda x: x[1]["mean"])
            print(f"  {name}: {n_active}/{len(weights)} modules active (>1%), "
                  f"dominant: {max_mod[0]} ({max_mod[1]['mean']:.1%})")
        else:
            print(f"  {name}: sequential (no weight data)")


if __name__ == "__main__":
    main()
