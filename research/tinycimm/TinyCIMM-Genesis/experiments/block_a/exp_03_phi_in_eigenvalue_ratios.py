"""
Genesis Exp 03: Phi in Eigenvalue Ratios

After 1000+ steps of anti-Hebbian modulation, do consecutive eigenvalue
ratios cluster near phi?

M10 exp_11 showed >15% phi enrichment in the SelfApplicator's eigenvalue
spectrum. Here we test whether random symmetric W systems show the same
enrichment. Negative control: Hebbian modulation (strengthen dominant modes
instead of weakening them) should NOT show phi enrichment.

Tests:
  T1: Phi enrichment exists — enrichment > 15% after dynamics
  T2: Enrichment increases with time — later timesteps show more phi
  T3: Anti-Hebbian required — Hebbian modulation shows no enrichment
  T4: N-dependence — enrichment is consistent across N values
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
GENESIS_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(GENESIS_ROOT))

from spectral_utils import (
    PHI, PHI_INV, LN_PHI, SCOPE_RATIO,
    eigenvalue_ratios, phi_enrichment,
)

N_VALUES = [8, 12, 16, 24, 32]
N_SEEDS = 30
RESULTS_DIR = GENESIS_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def run_and_measure_ratios(N, seed, n_steps=1500, target_sr=1.2,
                            weak_factor=None, anti_hebbian=True):
    """
    Run Genesis system and measure phi enrichment at intervals.
    Returns list of (step, enrichment) pairs.
    """
    rng = np.random.RandomState(seed)
    if weak_factor is None:
        weak_factor = PHI ** (-1.0 / N)
    strong_factor = 1.01

    W = rng.randn(N, N) / np.sqrt(N)
    W = (W + W.T) / 2
    eigvals = np.linalg.eigvalsh(W)
    sr = np.max(np.abs(eigvals))
    if sr > 1e-10:
        W = W * (target_sr / sr)

    state = rng.randn(N) * 0.5
    measurements = []

    for t in range(n_steps):
        state = np.tanh(W @ state)

        eigvals, eigvecs = np.linalg.eigh(W)
        projections = (eigvecs.T @ state) ** 2
        total = np.sum(projections) + 1e-10
        activities = projections / total
        mean_act = 1.0 / N

        modulation = np.ones(N)
        if anti_hebbian:
            # Anti-Hebbian: weaken dominant, strengthen inactive
            modulation[activities > 2.0 * mean_act] = weak_factor
            modulation[activities < 0.5 * mean_act] = strong_factor
        else:
            # Hebbian (negative control): strengthen dominant, weaken inactive
            modulation[activities > 2.0 * mean_act] = strong_factor
            modulation[activities < 0.5 * mean_act] = weak_factor

        new_eigvals = eigvals * modulation

        post_sr = np.max(np.abs(new_eigvals))
        if post_sr > 1e-10:
            new_eigvals = new_eigvals * (target_sr / post_sr)

        W = eigvecs @ np.diag(new_eigvals) @ eigvecs.T

        # Measure every 100 steps
        if (t + 1) % 100 == 0:
            ratios = eigenvalue_ratios(W)
            enrich = phi_enrichment(ratios) if len(ratios) > 0 else 0.0
            measurements.append({'step': t + 1, 'enrichment': float(enrich)})

    return measurements


# ============================================================
# Test 1: Phi Enrichment Exists
# ============================================================
def test1_enrichment_exists():
    """T1: Enrichment > 15% after dynamics."""
    print("\n=== T1: Phi Enrichment Exists ===")

    N = 16
    enrichments = []

    for seed in range(N_SEEDS):
        measurements = run_and_measure_ratios(N, seed)
        final_enrich = measurements[-1]['enrichment']
        enrichments.append(final_enrich)

    mean_enrich = float(np.mean(enrichments))
    std_enrich = float(np.std(enrichments))
    frac_above_15 = sum(1 for e in enrichments if e > 0.15) / len(enrichments)

    # Also measure baseline (random matrix, no dynamics)
    baseline = []
    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed + 1000)
        W = rng.randn(N, N) / np.sqrt(N)
        W = (W + W.T) / 2
        ratios = eigenvalue_ratios(W)
        baseline.append(phi_enrichment(ratios) if len(ratios) > 0 else 0.0)
    mean_baseline = float(np.mean(baseline))

    passed = mean_enrich > 0.15
    print(f"  Mean enrichment: {mean_enrich:.4f} +/- {std_enrich:.4f}")
    print(f"  Baseline (random): {mean_baseline:.4f}")
    print(f"  Seeds above 15%: {frac_above_15:.0%}")
    print(f"  PASS: {passed} (need mean > 15%)")

    return {
        'test': 'enrichment_exists',
        'passed': bool(passed),
        'mean_enrichment': mean_enrich,
        'std_enrichment': std_enrich,
        'baseline': mean_baseline,
        'frac_above_15': float(frac_above_15),
        'N': N,
    }


# ============================================================
# Test 2: Enrichment Increases with Time
# ============================================================
def test2_enrichment_grows():
    """T2: Later timesteps show more phi enrichment."""
    print("\n=== T2: Enrichment Increases with Time ===")

    N = 16
    # Collect enrichment at each measurement point across seeds
    step_enrichments = {}

    for seed in range(N_SEEDS):
        measurements = run_and_measure_ratios(N, seed)
        for m in measurements:
            step = m['step']
            if step not in step_enrichments:
                step_enrichments[step] = []
            step_enrichments[step].append(m['enrichment'])

    steps = sorted(step_enrichments.keys())
    means = [float(np.mean(step_enrichments[s])) for s in steps]

    for s, m in zip(steps, means):
        print(f"  step {s:5d}: mean enrichment = {m:.4f}")

    # Check: last 3 means > first 3 means
    early_mean = float(np.mean(means[:3]))
    late_mean = float(np.mean(means[-3:]))
    grows = late_mean > early_mean

    passed = grows
    print(f"  Early mean: {early_mean:.4f}, Late mean: {late_mean:.4f}")
    print(f"  PASS: {passed}")

    return {
        'test': 'enrichment_grows',
        'passed': bool(passed),
        'early_mean': early_mean,
        'late_mean': late_mean,
        'timeline': {str(s): float(m) for s, m in zip(steps, means)},
    }


# ============================================================
# Test 3: Anti-Hebbian Required (Negative Control)
# ============================================================
def test3_negative_control():
    """T3: Hebbian modulation shows no phi enrichment."""
    print("\n=== T3: Anti-Hebbian Required ===")

    N = 16
    anti_hebb_enrichments = []
    hebb_enrichments = []

    for seed in range(N_SEEDS):
        # Anti-Hebbian
        ah_measurements = run_and_measure_ratios(N, seed, anti_hebbian=True)
        anti_hebb_enrichments.append(ah_measurements[-1]['enrichment'])

        # Hebbian (negative control)
        h_measurements = run_and_measure_ratios(N, seed, anti_hebbian=False)
        hebb_enrichments.append(h_measurements[-1]['enrichment'])

    mean_ah = float(np.mean(anti_hebb_enrichments))
    mean_h = float(np.mean(hebb_enrichments))

    # Anti-Hebbian should be higher
    ratio = mean_ah / mean_h if mean_h > 0 else float('inf')

    passed = mean_ah > mean_h and mean_ah > 0.10
    print(f"  Anti-Hebbian enrichment: {mean_ah:.4f}")
    print(f"  Hebbian enrichment: {mean_h:.4f}")
    print(f"  Ratio: {ratio:.2f}x")
    print(f"  PASS: {passed}")

    return {
        'test': 'negative_control',
        'passed': bool(passed),
        'anti_hebbian_enrichment': mean_ah,
        'hebbian_enrichment': mean_h,
        'ratio': float(ratio),
    }


# ============================================================
# Test 4: N-Dependence
# ============================================================
def test4_n_dependence():
    """T4: Enrichment is consistent across N values."""
    print("\n=== T4: N-Dependence ===")

    enrichments_by_N = {}
    n_above_10 = 0

    for N in N_VALUES:
        enrichments = []
        for seed in range(N_SEEDS):
            measurements = run_and_measure_ratios(N, seed)
            enrichments.append(measurements[-1]['enrichment'])

        mean_e = float(np.mean(enrichments))
        enrichments_by_N[N] = mean_e

        above = mean_e > 0.10
        if above:
            n_above_10 += 1
        print(f"  N={N:3d}: mean enrichment = {mean_e:.4f} "
              f"{'OK' if above else 'LOW'}")

    passed = n_above_10 >= 3
    print(f"  PASS: {passed} ({n_above_10}/{len(N_VALUES)} above 10%)")

    return {
        'test': 'n_dependence',
        'passed': bool(passed),
        'n_above_10': n_above_10,
        'enrichments': {str(k): v for k, v in enrichments_by_N.items()},
    }


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Genesis Exp 03: Phi in Eigenvalue Ratios")
    print("  Do consecutive eigenvalue ratios cluster near phi?")
    print("=" * 70)

    tests = [test1_enrichment_exists, test2_enrichment_grows,
             test3_negative_control, test4_n_dependence]

    results = []
    n_passed = 0

    for test_fn in tests:
        result = test_fn()
        results.append(result)
        if result['passed']:
            n_passed += 1

    print("\n" + "=" * 70)
    print(f"SCORE: {n_passed}/{len(tests)}")
    print("=" * 70)

    for r in results:
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  [{status}] {r['test']}")

    output = {
        'experiment': 'exp_03_phi_in_eigenvalue_ratios',
        'variant': 'TinyCIMM-Genesis',
        'description': 'Do consecutive eigenvalue ratios cluster near phi '
                       'after anti-Hebbian modulation?',
        'score': f'{n_passed}/{len(tests)}',
        'n_passed': n_passed,
        'n_tests': len(tests),
        'tests': results,
        'config': {
            'N_values': N_VALUES,
            'n_seeds': N_SEEDS,
        },
        'timestamp': datetime.now().isoformat(),
    }

    fname = RESULTS_DIR / f"exp_03_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {fname}")
