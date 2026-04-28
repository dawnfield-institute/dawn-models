"""
Genesis Exp 02: Spectral Radius and Viability

Does sr = gamma/ln(phi) = 1.1995 emerge as the critical spectral radius?

Exp_01 showed the critical modulation rate is phi^(-1/N). Here we fix
weak_factor near the critical rate and ask: what spectral radius does
the system need to survive?

M10 exp_16 showed a "complexity valley" at sr = gamma/ln(phi) = 1.1995.
In the Genesis system, this manifests as a viability transition: below
the critical sr, the system dies (tanh contraction wins); above it, the
system lives (eigenvalue expansion sustains dynamics). The critical sr
should match gamma/ln(phi).

Tests:
  T1: SR viability transition — there is a sharp alive/dead transition in sr
  T2: Critical sr matches scope ratio — boundary within 5% of gamma/ln(phi)
  T3: N-independence — critical sr is consistent across N values
  T4: Two-parameter consistency — the (sr, weak_factor) boundary is self-consistent
      with exp_01 (at sr=1.2, boundary is phi^(-1/N); at weak=phi^(-1/N), boundary is ~1.2)
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
    PHI, PHI_INV, LN_PHI, GAMMA_EM, SCOPE_RATIO,
    hierarchy_entropy, cascade_depth,
)

N_VALUES = [8, 12, 16, 24, 32]
N_SEEDS = 50
N_STEPS = 500
RESULTS_DIR = GENESIS_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def run_at_sr(N, seed, target_sr, weak_factor, strong_factor=1.01, n_steps=500):
    """
    Run a Genesis-style system at a FIXED sr with a FIXED weak_factor.
    Returns whether the system is alive.
    """
    rng = np.random.RandomState(seed)

    # Random symmetric W
    W = rng.randn(N, N) / np.sqrt(N)
    W = (W + W.T) / 2
    eigvals = np.linalg.eigvalsh(W)
    sr = np.max(np.abs(eigvals))
    if sr > 1e-10:
        W = W * (target_sr / sr)

    state = rng.randn(N) * 0.5
    alive_steps = 0

    for t in range(n_steps):
        state = np.tanh(W @ state)

        eigvals, eigvecs = np.linalg.eigh(W)
        projections = (eigvecs.T @ state) ** 2
        total = np.sum(projections) + 1e-10
        activities = projections / total
        mean_act = 1.0 / N

        modulation = np.ones(N)
        modulation[activities > 2.0 * mean_act] = weak_factor
        modulation[activities < 0.5 * mean_act] = strong_factor
        new_eigvals = eigvals * modulation

        # Re-normalize sr to target
        post_sr = np.max(np.abs(new_eigvals))
        if post_sr > 1e-10:
            new_eigvals = new_eigvals * (target_sr / post_sr)

        W = eigvecs @ np.diag(new_eigvals) @ eigvecs.T

        if np.linalg.norm(state) > 0.01:
            alive_steps += 1

    alive_frac = alive_steps / n_steps
    return alive_frac > 0.5


def bisect_sr_boundary(N, weak_factor, n_seeds=50, sr_low=0.8, sr_high=2.0,
                        tol=0.01, max_iter=25):
    """Bisect for the critical sr where ~50% of seeds are alive."""
    def alive_fraction(sr):
        n_alive = sum(1 for seed in range(n_seeds)
                      if run_at_sr(N, seed, sr, weak_factor))
        return n_alive / n_seeds

    for _ in range(max_iter):
        mid = (sr_low + sr_high) / 2
        frac = alive_fraction(mid)

        if frac < 0.5:
            sr_low = mid  # Too low, increase
        else:
            sr_high = mid  # Alive, can decrease

        if sr_high - sr_low < tol:
            break

    return (sr_low + sr_high) / 2


# ============================================================
# Test 1: SR Viability Transition
# ============================================================
def test1_sr_transition():
    """T1: There is a sharp alive/dead transition in sr."""
    print("\n=== T1: SR Viability Transition ===")

    N = 16
    weak_factor = PHI ** (-1.0 / N)
    sr_values = np.linspace(0.9, 1.6, 30)
    alive_fracs = []

    for sr in sr_values:
        n_alive = sum(1 for seed in range(N_SEEDS)
                      if run_at_sr(N, seed, float(sr), weak_factor))
        alive_fracs.append(n_alive / N_SEEDS)

    # Find transition width: from 10% to 90% alive
    above_10 = [i for i, f in enumerate(alive_fracs) if f > 0.1]
    above_90 = [i for i, f in enumerate(alive_fracs) if f > 0.9]

    if above_10 and above_90:
        transition_width = sr_values[above_90[0]] - sr_values[above_10[0]]
    else:
        transition_width = 1.0

    is_sharp = transition_width < 0.15
    has_both = min(alive_fracs) < 0.3 and max(alive_fracs) > 0.7

    # Print a few key points
    for i in range(0, len(sr_values), 3):
        print(f"  sr={sr_values[i]:.3f}: alive={alive_fracs[i]:.0%}")

    print(f"  Transition width: {transition_width:.4f}")
    print(f"  Sharp: {is_sharp}")

    passed = has_both and is_sharp
    print(f"  PASS: {passed}")

    return {
        'test': 'sr_transition',
        'passed': bool(passed),
        'transition_width': float(transition_width),
        'min_alive': float(min(alive_fracs)),
        'max_alive': float(max(alive_fracs)),
        'N': N,
        'weak_factor': float(weak_factor),
    }


# ============================================================
# Test 2: Critical SR Matches Scope Ratio
# ============================================================
def test2_scope_ratio():
    """T2: Critical sr within 5% of gamma/ln(phi) = 1.1995."""
    print("\n=== T2: Critical SR Matches Scope Ratio ===")

    results = {}
    n_within = 0

    for N in N_VALUES:
        weak_factor = PHI ** (-1.0 / N)
        critical_sr = bisect_sr_boundary(N, weak_factor, n_seeds=N_SEEDS)
        error_pct = abs(critical_sr - SCOPE_RATIO) / SCOPE_RATIO * 100

        results[N] = {
            'critical_sr': float(critical_sr),
            'predicted': float(SCOPE_RATIO),
            'error_pct': float(error_pct),
            'weak_factor': float(weak_factor),
        }

        within = error_pct < 5.0
        if within:
            n_within += 1
        print(f"  N={N:3d}: critical_sr={critical_sr:.4f}, "
              f"target={SCOPE_RATIO:.4f}, error={error_pct:.2f}% "
              f"{'OK' if within else 'MISS'}")

    mean_error = float(np.mean([r['error_pct'] for r in results.values()]))
    passed = n_within >= 3
    print(f"  Mean error: {mean_error:.2f}%")
    print(f"  PASS: {passed} ({n_within}/{len(N_VALUES)} within 5%)")

    return {
        'test': 'scope_ratio',
        'passed': bool(passed),
        'n_within': n_within,
        'mean_error_pct': mean_error,
        'results': {str(k): v for k, v in results.items()},
    }


# ============================================================
# Test 3: N-Independence
# ============================================================
def test3_n_independence():
    """T3: Critical sr is consistent across N values."""
    print("\n=== T3: N-Independence ===")

    critical_srs = []
    for N in N_VALUES:
        weak_factor = PHI ** (-1.0 / N)
        critical_sr = bisect_sr_boundary(N, weak_factor, n_seeds=N_SEEDS)
        critical_srs.append(float(critical_sr))
        print(f"  N={N:3d}: critical_sr={critical_sr:.4f}")

    mean_sr = float(np.mean(critical_srs))
    std_sr = float(np.std(critical_srs))
    cv = std_sr / mean_sr if mean_sr > 0 else float('inf')

    passed = cv < 0.10
    print(f"  Mean: {mean_sr:.4f}, Std: {std_sr:.4f}, CV: {cv:.4f}")
    print(f"  PASS: {passed} (CV < 0.10)")

    return {
        'test': 'n_independence',
        'passed': bool(passed),
        'critical_srs': critical_srs,
        'mean_sr': mean_sr,
        'std_sr': std_sr,
        'cv': float(cv),
    }


# ============================================================
# Test 4: Two-Parameter Consistency
# ============================================================
def test4_two_parameter():
    """
    T4: The viability boundary is self-consistent.
    At sr=1.2, exp_01 found weak_crit = phi^(-1/N).
    At weak=phi^(-1/N), this exp should find sr_crit = ~1.2.
    Cross-check: run both directions for N=16.
    """
    print("\n=== T4: Two-Parameter Consistency ===")

    N = 16

    # Direction 1: fix sr=1.2, find weak_crit (repeating exp_01 logic)
    def alive_fraction_weak(weak):
        n_alive = sum(1 for seed in range(N_SEEDS)
                      if run_at_sr(N, seed, 1.2, weak))
        return n_alive / N_SEEDS

    # Bisect for weak_crit at sr=1.2
    wlo, whi = 0.90, 0.999
    for _ in range(20):
        mid = (wlo + whi) / 2
        if alive_fraction_weak(mid) < 0.5:
            wlo = mid
        else:
            whi = mid
        if whi - wlo < 0.001:
            break
    weak_crit_at_sr12 = (wlo + whi) / 2
    weak_predicted = PHI ** (-1.0 / N)

    # Direction 2: fix weak=phi^(-1/N), find sr_crit
    weak_factor = PHI ** (-1.0 / N)
    sr_crit_at_phi_weak = bisect_sr_boundary(N, weak_factor, n_seeds=N_SEEDS)

    # Consistency check
    weak_error = abs(weak_crit_at_sr12 - weak_predicted) / weak_predicted * 100
    sr_error = abs(sr_crit_at_phi_weak - SCOPE_RATIO) / SCOPE_RATIO * 100

    print(f"  Direction 1 (sr=1.2): weak_crit={weak_crit_at_sr12:.4f}, "
          f"predicted={weak_predicted:.4f}, error={weak_error:.2f}%")
    print(f"  Direction 2 (weak=phi^(-1/N)): sr_crit={sr_crit_at_phi_weak:.4f}, "
          f"predicted={SCOPE_RATIO:.4f}, error={sr_error:.2f}%")

    passed = weak_error < 5.0 and sr_error < 10.0
    print(f"  PASS: {passed}")

    return {
        'test': 'two_parameter_consistency',
        'passed': bool(passed),
        'weak_crit_at_sr12': float(weak_crit_at_sr12),
        'weak_predicted': float(weak_predicted),
        'weak_error_pct': float(weak_error),
        'sr_crit_at_phi_weak': float(sr_crit_at_phi_weak),
        'sr_predicted': float(SCOPE_RATIO),
        'sr_error_pct': float(sr_error),
        'N': N,
    }


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Genesis Exp 02: Spectral Radius and Viability")
    print(f"  Does the critical sr match gamma/ln(phi) = {SCOPE_RATIO:.4f}?")
    print("=" * 70)

    tests = [test1_sr_transition, test2_scope_ratio,
             test3_n_independence, test4_two_parameter]

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
        'experiment': 'exp_02_spectral_radius_viability',
        'variant': 'TinyCIMM-Genesis',
        'description': f'Does the critical sr match gamma/ln(phi) = {SCOPE_RATIO:.4f}?',
        'score': f'{n_passed}/{len(tests)}',
        'n_passed': n_passed,
        'n_tests': len(tests),
        'tests': results,
        'config': {
            'N_values': N_VALUES,
            'n_seeds': N_SEEDS,
            'n_steps': N_STEPS,
        },
        'timestamp': datetime.now().isoformat(),
    }

    fname = RESULTS_DIR / f"exp_02_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {fname}")
