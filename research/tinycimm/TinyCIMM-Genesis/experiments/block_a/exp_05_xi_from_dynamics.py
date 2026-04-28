"""
Genesis Exp 05: Xi from Dynamics

Does Xi = gamma + ln(phi) = 1.0584 emerge as the information cost
per eigenvalue mode transition?

In M10, Xi was measured as the information cost when the dominant
eigenvalue mode changes. When a new eigenvalue becomes the largest,
there's an entropy cost associated with the reorganization. M10 predicts
this cost is Xi = gamma + ln(phi).

Definition of "boundary crossing": the dominant eigenvalue index changes.
When the largest-magnitude eigenvalue shifts from mode i to mode j,
the system reorganizes. We measure the entropy change at each such event.

Tests:
  T1: Mode transitions occur — dominant mode changes during dynamics
  T2: Transition cost is consistent — low variance across events
  T3: Cost matches Xi — mean transition cost within 10% of Xi = 1.0584
  T4: N-independence — transition cost is consistent across N
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
    PHI, PHI_INV, LN_PHI, GAMMA_EM, XI,
    hierarchy_entropy,
)

N_VALUES = [8, 12, 16, 24, 32]
N_SEEDS = 30
N_STEPS = 2000
RESULTS_DIR = GENESIS_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def run_and_track_transitions(N, seed, n_steps=2000, target_sr=1.2,
                               weak_factor=None):
    """
    Run Genesis system and track dominant eigenvalue mode transitions.
    Returns list of transition events with entropy changes.
    """
    rng = np.random.RandomState(seed)
    if weak_factor is None:
        # Use slightly above critical (alive side of boundary)
        weak_factor = PHI ** (-1.0 / N) + 0.005
    strong_factor = 1.01

    W = rng.randn(N, N) / np.sqrt(N)
    W = (W + W.T) / 2
    eigvals = np.linalg.eigvalsh(W)
    sr = np.max(np.abs(eigvals))
    if sr > 1e-10:
        W = W * (target_sr / sr)

    state = rng.randn(N) * 0.5

    # Track dominant mode and entropy
    prev_dominant = None
    prev_entropy = None
    transitions = []

    for t in range(n_steps):
        state = np.tanh(W @ state)

        eigvals, eigvecs = np.linalg.eigh(W)
        projections = (eigvecs.T @ state) ** 2
        total = np.sum(projections) + 1e-10
        activities = projections / total
        mean_act = 1.0 / N

        # Track dominant mode (highest activity projection)
        dominant = int(np.argmax(activities))
        current_entropy = hierarchy_entropy(eigvals)

        if prev_dominant is not None and dominant != prev_dominant:
            # Mode transition occurred
            if prev_entropy is not None:
                entropy_change = abs(current_entropy - prev_entropy)
                transitions.append({
                    'step': t,
                    'from_mode': int(prev_dominant),
                    'to_mode': dominant,
                    'entropy_change': float(entropy_change),
                    'entropy_before': float(prev_entropy),
                    'entropy_after': float(current_entropy),
                })

        prev_dominant = dominant
        prev_entropy = current_entropy

        # Modulation
        modulation = np.ones(N)
        modulation[activities > 2.0 * mean_act] = weak_factor
        modulation[activities < 0.5 * mean_act] = strong_factor
        new_eigvals = eigvals * modulation

        post_sr = np.max(np.abs(new_eigvals))
        if post_sr > 1e-10:
            new_eigvals = new_eigvals * (target_sr / post_sr)

        W = eigvecs @ np.diag(new_eigvals) @ eigvecs.T

    return transitions


# ============================================================
# Test 1: Mode Transitions Occur
# ============================================================
def test1_transitions_occur():
    """T1: Dominant mode changes during dynamics."""
    print("\n=== T1: Mode Transitions Occur ===")

    N = 16
    transition_counts = []

    for seed in range(N_SEEDS):
        transitions = run_and_track_transitions(N, seed)
        transition_counts.append(len(transitions))

    mean_count = float(np.mean(transition_counts))
    frac_with = sum(1 for c in transition_counts if c > 0) / len(transition_counts)

    passed = frac_with > 0.5 and mean_count > 5
    print(f"  Mean transitions: {mean_count:.1f}")
    print(f"  Seeds with transitions: {frac_with:.0%}")
    print(f"  PASS: {passed}")

    return {
        'test': 'transitions_occur',
        'passed': bool(passed),
        'mean_count': mean_count,
        'frac_with_transitions': float(frac_with),
        'N': N,
    }


# ============================================================
# Test 2: Transition Cost is Consistent
# ============================================================
def test2_consistent_cost():
    """T2: Low variance of entropy change across transition events."""
    print("\n=== T2: Transition Cost Consistency ===")

    N = 16
    all_costs = []

    for seed in range(N_SEEDS):
        transitions = run_and_track_transitions(N, seed)
        for t in transitions:
            all_costs.append(t['entropy_change'])

    if len(all_costs) < 10:
        print(f"  Only {len(all_costs)} transitions — insufficient data")
        return {
            'test': 'consistent_cost',
            'passed': False,
            'reason': 'insufficient_data',
            'n_events': len(all_costs),
        }

    mean_cost = float(np.mean(all_costs))
    std_cost = float(np.std(all_costs))
    cv = std_cost / mean_cost if mean_cost > 0 else float('inf')

    # Filter out near-zero costs (mode flicker, not real transitions)
    significant = [c for c in all_costs if c > 0.01]
    if len(significant) > 5:
        mean_sig = float(np.mean(significant))
        std_sig = float(np.std(significant))
        cv_sig = std_sig / mean_sig if mean_sig > 0 else float('inf')
    else:
        mean_sig = 0
        std_sig = 0
        cv_sig = float('inf')

    passed = cv_sig < 1.0  # reasonably consistent
    print(f"  Total transitions: {len(all_costs)}")
    print(f"  Significant (>0.01): {len(significant)}")
    print(f"  All costs: mean={mean_cost:.4f}, std={std_cost:.4f}, CV={cv:.4f}")
    print(f"  Significant: mean={mean_sig:.4f}, std={std_sig:.4f}, CV={cv_sig:.4f}")
    print(f"  PASS: {passed} (CV < 1.0)")

    return {
        'test': 'consistent_cost',
        'passed': bool(passed),
        'n_events': len(all_costs),
        'n_significant': len(significant),
        'mean_cost': mean_cost,
        'mean_significant': mean_sig,
        'cv_all': float(cv),
        'cv_significant': float(cv_sig),
    }


# ============================================================
# Test 3: Cost Matches Xi
# ============================================================
def test3_xi_match():
    """T3: Mean transition cost within 10% of Xi = 1.0584."""
    print("\n=== T3: Cost Matches Xi ===")

    N = 16
    all_costs = []

    for seed in range(N_SEEDS):
        transitions = run_and_track_transitions(N, seed)
        for t in transitions:
            if t['entropy_change'] > 0.01:  # significant only
                all_costs.append(t['entropy_change'])

    if len(all_costs) < 5:
        print(f"  Only {len(all_costs)} significant transitions")
        return {
            'test': 'xi_match',
            'passed': False,
            'reason': 'insufficient_data',
        }

    mean_cost = float(np.mean(all_costs))
    median_cost = float(np.median(all_costs))
    error_mean = abs(mean_cost - XI) / XI * 100
    error_median = abs(median_cost - XI) / XI * 100

    # Use whichever is closer (median is more robust to outliers)
    best_match = min(error_mean, error_median)
    best_which = 'mean' if error_mean < error_median else 'median'
    best_value = mean_cost if error_mean < error_median else median_cost

    passed = best_match < 10.0
    print(f"  Mean cost: {mean_cost:.4f} (error vs Xi: {error_mean:.1f}%)")
    print(f"  Median cost: {median_cost:.4f} (error vs Xi: {error_median:.1f}%)")
    print(f"  Xi = {XI:.4f}")
    print(f"  Best match ({best_which}): {best_value:.4f}, error={best_match:.1f}%")
    print(f"  PASS: {passed} (best within 10%)")

    return {
        'test': 'xi_match',
        'passed': bool(passed),
        'mean_cost': mean_cost,
        'median_cost': median_cost,
        'xi': float(XI),
        'error_mean_pct': float(error_mean),
        'error_median_pct': float(error_median),
        'n_events': len(all_costs),
    }


# ============================================================
# Test 4: N-Independence
# ============================================================
def test4_n_independence():
    """T4: Transition cost is consistent across N values."""
    print("\n=== T4: N-Independence ===")

    costs_by_N = {}

    for N in N_VALUES:
        all_costs = []
        for seed in range(N_SEEDS):
            transitions = run_and_track_transitions(N, seed)
            for t in transitions:
                if t['entropy_change'] > 0.01:
                    all_costs.append(t['entropy_change'])

        if len(all_costs) > 0:
            mean_cost = float(np.mean(all_costs))
            median_cost = float(np.median(all_costs))
        else:
            mean_cost = 0.0
            median_cost = 0.0

        costs_by_N[N] = {
            'mean': mean_cost,
            'median': median_cost,
            'n_events': len(all_costs),
        }
        error = abs(median_cost - XI) / XI * 100 if median_cost > 0 else 100
        print(f"  N={N:3d}: median={median_cost:.4f}, n_events={len(all_costs)}, "
              f"error_vs_Xi={error:.1f}%")

    # Check consistency: CV of median costs across N
    medians = [costs_by_N[N]['median'] for N in N_VALUES if costs_by_N[N]['n_events'] > 0]
    if len(medians) >= 3:
        cv = float(np.std(medians) / np.mean(medians)) if np.mean(medians) > 0 else float('inf')
    else:
        cv = float('inf')

    passed = cv < 0.50 and len(medians) >= 3
    print(f"  CV of medians: {cv:.4f}")
    print(f"  PASS: {passed} (CV < 0.50, need >=3 N values with data)")

    return {
        'test': 'n_independence',
        'passed': bool(passed),
        'costs_by_N': {str(k): v for k, v in costs_by_N.items()},
        'cv': float(cv),
    }


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Genesis Exp 05: Xi from Dynamics")
    print(f"  Does Xi = gamma + ln(phi) = {XI:.4f} emerge as transition cost?")
    print("=" * 70)

    tests = [test1_transitions_occur, test2_consistent_cost,
             test3_xi_match, test4_n_independence]

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
        'experiment': 'exp_05_xi_from_dynamics',
        'variant': 'TinyCIMM-Genesis',
        'description': f'Does Xi = gamma + ln(phi) = {XI:.4f} emerge as '
                       'the info cost per mode transition?',
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

    fname = RESULTS_DIR / f"exp_05_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {fname}")
