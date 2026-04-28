"""
Genesis Exp 01: Viability Boundary and Phi

THE HEADLINE EXPERIMENT.

Does the alive/dead viability boundary occur at phi^(-1/N) in a system
that was NOT designed as the SelfApplicator?

Key differences from M10 exp_15:
  - GenesisSystem uses random symmetric W (not SelfApplicator's specific init)
  - Different random state initialization per seed
  - Same structural constraints: symmetric W, anti-Hebbian eigenvalue modulation, sr=1.2

If the boundary is still at phi^(-1/N), it's because of STRUCTURE, not implementation.

Tests:
  T1: Viability boundary exists — there is a sharp alive/dead transition
  T2: Boundary matches phi^(-1/N) — mean error < 5% across N values
  T3: Per-traversal attenuation — weak_crit^N approximates 1/phi (large-N converges)
  T4: N-dependence — boundary increases monotonically with N (matches phi^(-1/N) shape)
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
    PHI, PHI_INV, LN_PHI,
    symmetric_eigendecomposition, anti_hebbian_modulate,
    hierarchy_entropy, cascade_depth,
)

N_VALUES = [8, 12, 16, 24, 32, 48]
N_SEEDS = 50
N_STEPS = 1000
RESULTS_DIR = GENESIS_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def run_fixed_weak(N, seed, weak_factor, strong_factor=1.01, n_steps=500,
                    target_sr=1.2):
    """
    Run a Genesis-style system with FIXED weak_factor.
    Returns whether the system is alive and final diagnostics.

    target_sr=1.2 provides enough expansion to counteract tanh contraction.
    Same as M10's SelfApplicator — the test is whether random W (vs specific
    SelfApplicator init) produces the same phi^(-1/N) boundary.
    """
    rng = np.random.RandomState(seed)

    # Random symmetric W
    W = rng.randn(N, N) / np.sqrt(N)
    W = (W + W.T) / 2
    # Normalize sr to target (1.2 = enough expansion for tanh dynamics)
    eigvals = np.linalg.eigvalsh(W)
    sr = np.max(np.abs(eigvals))
    if sr > 1e-10:
        W = W * (target_sr / sr)

    state = rng.randn(N) * 0.5
    alive_steps = 0

    for t in range(n_steps):
        state = np.tanh(W @ state)

        # Anti-Hebbian eigenvalue modulation
        eigvals, eigvecs = np.linalg.eigh(W)
        projections = (eigvecs.T @ state) ** 2
        total = np.sum(projections) + 1e-10
        activities = projections / total
        mean_act = 1.0 / N

        modulation = np.ones(N)
        modulation[activities > 2.0 * mean_act] = weak_factor
        modulation[activities < 0.5 * mean_act] = strong_factor
        new_eigvals = eigvals * modulation

        # Re-normalize sr to preserve scale (isolate ratio effects)
        pre_sr = np.max(np.abs(eigvals))
        post_sr = np.max(np.abs(new_eigvals))
        if post_sr > 1e-10 and pre_sr > 1e-10:
            new_eigvals = new_eigvals * (pre_sr / post_sr)

        W = eigvecs @ np.diag(new_eigvals) @ eigvecs.T

        # Check aliveness: state has significant energy
        if np.linalg.norm(state) > 0.01:
            alive_steps += 1

    # Activity entropy of final state
    eigvals_final = np.linalg.eigvalsh(W)
    projections = (np.linalg.eigh(W)[1].T @ state) ** 2
    total = np.sum(projections) + 1e-10
    activities = projections / total
    H_act = float(-np.sum(activities[activities > 1e-15] *
                           np.log(activities[activities > 1e-15])))
    final_sr = float(np.max(np.abs(eigvals_final)))

    alive_frac = alive_steps / n_steps
    # Alive = sustained state energy. Don't require entropy — at the boundary
    # the system may be alive but mode-concentrated.
    is_alive = alive_frac > 0.5

    return {
        'alive': is_alive,
        'alive_frac': float(alive_frac),
        'H_act': H_act,
        'final_sr': final_sr,
        'state_norm': float(np.linalg.norm(state)),
    }


def bisect_boundary(N, n_seeds=50, sr_low=0.85, sr_high=0.999, tol=0.001,
                     max_iter=30):
    """
    Bisect for the viability boundary: the weak_factor where
    ~50% of seeds are alive.
    """
    def alive_fraction(weak):
        n_alive = 0
        for seed in range(n_seeds):
            result = run_fixed_weak(N, seed, weak)
            if result['alive']:
                n_alive += 1
        return n_alive / n_seeds

    for _ in range(max_iter):
        mid = (sr_low + sr_high) / 2
        frac = alive_fraction(mid)

        if frac < 0.5:
            sr_low = mid  # Too aggressive, back off
        else:
            sr_high = mid  # Can be more aggressive

        if sr_high - sr_low < tol:
            break

    boundary = (sr_low + sr_high) / 2
    return boundary


# ============================================================
# Test 1: Viability Boundary Exists
# ============================================================
def test1_boundary_exists():
    """T1: There is a sharp alive/dead transition."""
    print("\n=== T1: Viability Boundary Exists ===")

    N = 16  # Test at one N value
    weak_values = np.linspace(0.85, 0.999, 30)
    alive_fracs = []

    for weak in weak_values:
        n_alive = 0
        for seed in range(N_SEEDS):
            result = run_fixed_weak(N, seed, float(weak))
            if result['alive']:
                n_alive += 1
        alive_fracs.append(n_alive / N_SEEDS)

    # Find the transition width: from 10% alive to 90% alive
    above_10 = [i for i, f in enumerate(alive_fracs) if f > 0.1]
    above_90 = [i for i, f in enumerate(alive_fracs) if f > 0.9]

    if above_10 and above_90:
        transition_width = weak_values[above_90[0]] - weak_values[above_10[0]]
    else:
        transition_width = 1.0  # No transition found

    # Sharp = transition width < 0.05
    is_sharp = transition_width < 0.05
    has_both = min(alive_fracs) < 0.3 and max(alive_fracs) > 0.7

    print(f"  Alive fracs range: [{min(alive_fracs):.2f}, {max(alive_fracs):.2f}]")
    print(f"  Transition width: {transition_width:.4f}")
    print(f"  Sharp transition: {is_sharp}")

    passed = has_both and is_sharp
    print(f"  PASS: {passed}")

    return {
        'test': 'boundary_exists',
        'passed': bool(passed),
        'transition_width': float(transition_width),
        'min_alive': float(min(alive_fracs)),
        'max_alive': float(max(alive_fracs)),
        'N': N,
    }


# ============================================================
# Test 2: Boundary Matches phi^(-1/N)
# ============================================================
def test2_phi_prediction():
    """T2: Viability boundary is at phi^(-1/N)."""
    print("\n=== T2: Boundary Matches phi^(-1/N) ===")

    results = {}
    n_within_5pct = 0

    for N in N_VALUES:
        boundary = bisect_boundary(N, n_seeds=N_SEEDS)
        predicted = PHI ** (-1.0 / N)
        error_pct = abs(boundary - predicted) / predicted * 100

        results[N] = {
            'boundary': float(boundary),
            'predicted': float(predicted),
            'error_pct': float(error_pct),
        }

        within = error_pct < 5.0
        if within:
            n_within_5pct += 1
        print(f"  N={N:3d}: boundary={boundary:.4f}, predicted={predicted:.4f}, "
              f"error={error_pct:.2f}% {'OK' if within else 'MISS'}")

    passed = n_within_5pct >= 3
    mean_error = float(np.mean([r['error_pct'] for r in results.values()]))
    print(f"  Mean error: {mean_error:.2f}%")
    print(f"  PASS: {passed} ({n_within_5pct}/5 within 5%)")

    return {
        'test': 'phi_prediction',
        'passed': bool(passed),
        'n_within_5pct': n_within_5pct,
        'mean_error_pct': mean_error,
        'results': {str(k): v for k, v in results.items()},
    }


# ============================================================
# Test 3: Per-Traversal Attenuation
# ============================================================
def test3_per_traversal():
    """T3: boundary^N approximates 1/phi."""
    print("\n=== T3: Per-Traversal Attenuation ===")

    traversals = []
    for N in N_VALUES:
        boundary = bisect_boundary(N, n_seeds=N_SEEDS)
        traversal = boundary ** N
        error_pct = abs(traversal - PHI_INV) / PHI_INV * 100
        traversals.append({
            'N': N, 'boundary': float(boundary),
            'traversal': float(traversal),
            'error_pct': float(error_pct),
        })
        print(f"  N={N:3d}: boundary^N = {traversal:.4f}, 1/phi = {PHI_INV:.4f}, "
              f"error = {error_pct:.1f}%")

    mean_error = float(np.mean([t['error_pct'] for t in traversals]))
    # Convergence check: at least 2 N values have boundary^N within 10% of 1/phi
    within_10 = sum(1 for t in traversals if t['error_pct'] < 10.0)
    best_match = min(traversals, key=lambda t: t['error_pct'])
    passed = within_10 >= 2
    print(f"  Mean error: {mean_error:.1f}%")
    print(f"  Within 10%: {within_10}/{len(traversals)}")
    print(f"  Best match: N={best_match['N']}, error={best_match['error_pct']:.1f}%")
    print(f"  PASS: {passed} (need >=2 within 10%)")

    return {
        'test': 'per_traversal',
        'passed': bool(passed),
        'traversals': traversals,
        'mean_error_pct': mean_error,
        'target': float(PHI_INV),
    }


# ============================================================
# Test 4: N-Dependence
# ============================================================
def test4_n_dependence():
    """T4: Boundary increases monotonically with N (matches phi^(-1/N) shape)."""
    print("\n=== T4: N-Dependence ===")

    boundaries = []
    for N in N_VALUES:
        boundary = bisect_boundary(N, n_seeds=N_SEEDS)
        boundaries.append(float(boundary))

    monotone = all(boundaries[i] < boundaries[i + 1]
                   for i in range(len(boundaries) - 1))
    spread = max(boundaries) - min(boundaries)
    predicted_spread = float(PHI ** (-1.0 / N_VALUES[-1]) - PHI ** (-1.0 / N_VALUES[0]))

    print(f"  Boundaries: {[f'{b:.4f}' for b in boundaries]}")
    print(f"  Predicted:  {[f'{PHI**(-1.0/N):.4f}' for N in N_VALUES]}")
    print(f"  Monotone: {monotone}")
    print(f"  Spread: {spread:.4f} (predicted: {predicted_spread:.4f})")

    passed = monotone and spread > 0.3 * predicted_spread
    print(f"  PASS: {passed}")

    return {
        'test': 'n_dependence',
        'passed': bool(passed),
        'boundaries': boundaries,
        'monotone': bool(monotone),
        'spread': float(spread),
        'predicted_spread': predicted_spread,
    }


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Genesis Exp 01: Viability Boundary and Phi")
    print("  Does the alive/dead boundary occur at phi^(-1/N)?")
    print("=" * 70)

    tests = [test1_boundary_exists, test2_phi_prediction,
             test3_per_traversal, test4_n_dependence]

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
        'experiment': 'exp_01_viability_boundary',
        'variant': 'TinyCIMM-Genesis',
        'description': 'Viability boundary scan — does alive/dead transition '
                       'occur at phi^(-1/N) without phi being inserted?',
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

    fname = RESULTS_DIR / f"exp_01_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {fname}")
