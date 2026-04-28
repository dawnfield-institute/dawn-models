"""
Genesis Exp 04: Cascade Depth Floor

Does the anti-Hebbian modulation produce a first-order transition in
cascade depth? Is there a minimum eigenvalue spacing (Planck-scale analogue)?

M10 exp_15 showed a first-order transition with a 1.58 nat gap — modes
don't die gradually, they drop out in a discontinuous jump. Here we
reproduce this with random symmetric W.

PLANCK THREAD: the minimum eigenvalue spacing at the cascade depth floor
may connect to the minimum resolution of self-referential dynamics. If
there's a gap below which the system can't distinguish eigenvalue modes,
that's a dynamical Planck scale.

Tests:
  T1: First-order transition — cascade depth drops discontinuously
  T2: Gap size — entropy gap at transition > 1.0 nats
  T3: Floor exists — minimum depth is > 0 and < N (not trivial)
  T4: Eigenvalue spacing floor — minimum spacing between active modes
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
    hierarchy_entropy, cascade_depth,
)

N_VALUES = [8, 12, 16, 24, 32]
N_SEEDS = 30
N_STEPS = 500
RESULTS_DIR = GENESIS_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def run_at_params(N, seed, weak_factor, target_sr=1.2, n_steps=500):
    """
    Run Genesis system at fixed parameters.
    Returns cascade depth, entropy, eigenvalue spectrum, and eigenvalue spacing.
    """
    rng = np.random.RandomState(seed)
    strong_factor = 1.01

    W = rng.randn(N, N) / np.sqrt(N)
    W = (W + W.T) / 2
    eigvals = np.linalg.eigvalsh(W)
    sr = np.max(np.abs(eigvals))
    if sr > 1e-10:
        W = W * (target_sr / sr)

    state = rng.randn(N) * 0.5

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

        post_sr = np.max(np.abs(new_eigvals))
        if post_sr > 1e-10:
            new_eigvals = new_eigvals * (target_sr / post_sr)

        W = eigvecs @ np.diag(new_eigvals) @ eigvecs.T

    # Final measurements
    eigvals_final = np.linalg.eigvalsh(W)
    depth = cascade_depth(W)
    H = hierarchy_entropy(eigvals_final)

    # Eigenvalue spacing of active modes
    abs_eigs = np.sort(np.abs(eigvals_final))[::-1]
    max_eig = abs_eigs[0] if len(abs_eigs) > 0 else 1.0
    active = abs_eigs[abs_eigs > 0.01 * max_eig]
    if len(active) >= 2:
        spacings = np.diff(active[::-1])  # ascending order, take diffs
        min_spacing = float(np.min(np.abs(spacings)))
        mean_spacing = float(np.mean(np.abs(spacings)))
    else:
        min_spacing = 0.0
        mean_spacing = 0.0

    return {
        'depth': depth,
        'entropy': float(H),
        'min_spacing': min_spacing,
        'mean_spacing': mean_spacing,
        'n_active': len(active),
        'alive': float(np.linalg.norm(state)) > 0.01,
    }


def scan_weak_factor(N, n_points=40, n_seeds=30):
    """Scan weak_factor from 0.80 to 0.999, measure cascade depth."""
    weak_values = np.linspace(0.80, 0.999, n_points)
    results = {}

    for weak in weak_values:
        depths = []
        entropies = []
        spacings = []
        for seed in range(n_seeds):
            r = run_at_params(N, seed, float(weak))
            depths.append(r['depth'])
            entropies.append(r['entropy'])
            spacings.append(r['min_spacing'])

        results[float(weak)] = {
            'mean_depth': float(np.mean(depths)),
            'std_depth': float(np.std(depths)),
            'mean_entropy': float(np.mean(entropies)),
            'mean_min_spacing': float(np.mean(spacings)),
        }

    return results, weak_values


# ============================================================
# Test 1: First-Order Transition
# ============================================================
def test1_first_order():
    """T1: Cascade depth drops discontinuously as weak_factor decreases."""
    print("\n=== T1: First-Order Transition ===")

    N = 16
    results, weak_values = scan_weak_factor(N, n_points=40, n_seeds=N_SEEDS)

    depths = [results[float(w)]['mean_depth'] for w in weak_values]

    # Find the maximum jump in depth between consecutive weak_factor values
    jumps = [abs(depths[i+1] - depths[i]) for i in range(len(depths)-1)]
    max_jump = max(jumps) if jumps else 0
    max_jump_idx = jumps.index(max_jump) if jumps else 0

    # Print around the transition
    start = max(0, max_jump_idx - 3)
    end = min(len(weak_values), max_jump_idx + 5)
    for i in range(start, end):
        w = float(weak_values[i])
        d = results[w]['mean_depth']
        marker = " <-- transition" if i == max_jump_idx else ""
        print(f"  weak={w:.4f}: depth={d:.1f}{marker}")

    # First-order: max jump > 3 modes
    is_first_order = max_jump > 3.0

    passed = is_first_order
    print(f"  Max depth jump: {max_jump:.1f} modes")
    print(f"  PASS: {passed} (need > 3 modes)")

    return {
        'test': 'first_order_transition',
        'passed': bool(passed),
        'max_jump': float(max_jump),
        'transition_weak': float(weak_values[max_jump_idx]),
        'N': N,
    }


# ============================================================
# Test 2: Entropy Gap
# ============================================================
def test2_entropy_gap():
    """T2: Entropy gap at transition > 1.0 nats."""
    print("\n=== T2: Entropy Gap ===")

    N = 16
    results, weak_values = scan_weak_factor(N, n_points=40, n_seeds=N_SEEDS)

    entropies = [results[float(w)]['mean_entropy'] for w in weak_values]

    # Find max entropy jump
    jumps = [abs(entropies[i+1] - entropies[i]) for i in range(len(entropies)-1)]
    max_jump = max(jumps) if jumps else 0
    max_jump_idx = jumps.index(max_jump) if jumps else 0

    # Report
    start = max(0, max_jump_idx - 2)
    end = min(len(weak_values), max_jump_idx + 4)
    for i in range(start, end):
        w = float(weak_values[i])
        e = results[w]['mean_entropy']
        marker = " <-- gap" if i == max_jump_idx else ""
        print(f"  weak={w:.4f}: entropy={e:.4f}{marker}")

    passed = max_jump > 1.0
    print(f"  Max entropy gap: {max_jump:.4f} nats")
    print(f"  PASS: {passed} (need > 1.0 nats)")

    return {
        'test': 'entropy_gap',
        'passed': bool(passed),
        'max_entropy_gap': float(max_jump),
        'transition_weak': float(weak_values[max_jump_idx]),
    }


# ============================================================
# Test 3: Floor Exists
# ============================================================
def test3_floor_exists():
    """T3: Minimum cascade depth is > 0 and < N."""
    print("\n=== T3: Floor Exists ===")

    floor_depths = {}

    for N in N_VALUES:
        # Use aggressive weak_factor (below boundary) to find the floor
        min_depth = N
        for seed in range(N_SEEDS):
            r = run_at_params(N, seed, 0.85)  # Very aggressive
            min_depth = min(min_depth, r['depth'])

        # Also check at the boundary
        weak_boundary = PHI ** (-1.0 / N)
        boundary_depths = []
        for seed in range(N_SEEDS):
            r = run_at_params(N, seed, weak_boundary)
            boundary_depths.append(r['depth'])

        mean_boundary = float(np.mean(boundary_depths))
        floor_depths[N] = {
            'min_depth_aggressive': min_depth,
            'mean_depth_boundary': mean_boundary,
        }
        print(f"  N={N:3d}: floor={min_depth}, boundary_mean={mean_boundary:.1f}")

    # Check: floors are non-trivial
    all_floors = [floor_depths[N]['min_depth_aggressive'] for N in N_VALUES]
    has_nontrivial_floor = any(0 < f < N for f, N in zip(all_floors, N_VALUES))

    passed = has_nontrivial_floor
    print(f"  Non-trivial floor exists: {has_nontrivial_floor}")
    print(f"  PASS: {passed}")

    return {
        'test': 'floor_exists',
        'passed': bool(passed),
        'floor_depths': {str(k): v for k, v in floor_depths.items()},
    }


# ============================================================
# Test 4: Eigenvalue Spacing Floor (Planck Thread)
# ============================================================
def test4_spacing_floor():
    """T4: Minimum eigenvalue spacing at the cascade depth floor."""
    print("\n=== T4: Eigenvalue Spacing Floor (Planck) ===")

    spacing_data = {}

    for N in N_VALUES:
        # At various weak_factors, measure min spacing
        weak_values = [0.85, 0.90, 0.95, PHI ** (-1.0 / N), 0.99]
        for weak in weak_values:
            spacings = []
            for seed in range(N_SEEDS):
                r = run_at_params(N, seed, weak)
                spacings.append(r['min_spacing'])
            mean_sp = float(np.mean(spacings))

        # Focus on boundary spacing
        boundary_weak = PHI ** (-1.0 / N)
        boundary_spacings = []
        for seed in range(N_SEEDS):
            r = run_at_params(N, seed, boundary_weak)
            boundary_spacings.append(r['min_spacing'])

        mean_boundary_sp = float(np.mean(boundary_spacings))
        spacing_data[N] = {
            'mean_min_spacing': mean_boundary_sp,
            'spacing_per_sr': mean_boundary_sp / 1.2 if 1.2 > 0 else 0,
        }
        print(f"  N={N:3d}: min_spacing={mean_boundary_sp:.6f}, "
              f"spacing/sr={mean_boundary_sp/1.2:.6f}")

    # Check if spacing scales with 1/N (Planck-like)
    spacings = [spacing_data[N]['mean_min_spacing'] for N in N_VALUES]
    if len(spacings) >= 2 and spacings[0] > 0 and spacings[-1] > 0:
        ratio = spacings[0] / spacings[-1]
        n_ratio = N_VALUES[-1] / N_VALUES[0]
        # If spacing ~ 1/N, ratio should be ~ n_ratio
        scaling_match = abs(ratio - n_ratio) / n_ratio < 0.5
    else:
        scaling_match = False
        ratio = 0
        n_ratio = 0

    # Also check if spacing decreases with N (basic requirement)
    decreasing = all(spacings[i] >= spacings[i+1] - 0.001
                     for i in range(len(spacings)-1))

    passed = decreasing  # Spacing floor exists and shrinks with N
    print(f"  Spacing decreases with N: {decreasing}")
    print(f"  Spacing ratio (N_min/N_max): {ratio:.2f} "
          f"(1/N scaling would give {n_ratio:.2f})")
    print(f"  PASS: {passed}")

    return {
        'test': 'spacing_floor',
        'passed': bool(passed),
        'spacing_data': {str(k): v for k, v in spacing_data.items()},
        'decreasing': bool(decreasing),
        'scaling_match': bool(scaling_match),
    }


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Genesis Exp 04: Cascade Depth Floor")
    print("  First-order transition and eigenvalue spacing floor")
    print("=" * 70)

    tests = [test1_first_order, test2_entropy_gap,
             test3_floor_exists, test4_spacing_floor]

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
        'experiment': 'exp_04_cascade_depth_floor',
        'variant': 'TinyCIMM-Genesis',
        'description': 'Cascade depth transition and eigenvalue spacing floor',
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

    fname = RESULTS_DIR / f"exp_04_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {fname}")
