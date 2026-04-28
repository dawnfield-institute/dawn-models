"""
Genesis Exp 06: Self-Consistency Attractor

THE FAILURE INVESTIGATION.

Genesis exp_03 found: SR normalization kills phi enrichment in eigenvalue
ratios. SR normalization rescales max(|lambda|) back to target every step,
which actively pushes all eigenvalues toward uniform spacing (ratios -> 1.0).

But M10's SelfApplicator HAS phi-structured ratios. The hypothesis: phi
ratios aren't an artifact of the SelfApplicator's specific initialization —
they're a FIXED POINT of the anti-Hebbian dynamics. The SelfApplicator
reaches that fixed point because its construction IS the fixed point. Random
W should approach it too, IF the normalization doesn't destroy ratio structure.

We test three normalization schemes:
  A) SR normalization (control — kills ratio structure, proven in exp_03)
  B) Frobenius normalization (preserves ||W||_F, allows sr and ratios to float)
  C) No normalization (eigenvalue clipping at [0.01, 10.0] only)

Key insight: Frobenius normalization preserves total "energy" in the spectrum
(sum of lambda^2) but lets individual eigenvalues redistribute. The maximum
can drift freely. This is the minimal stabilization that still allows
ratio structure to develop.

Tests:
  T1: Frobenius-normed sr stays bounded ([0.5, 5.0]) over 5000 steps
  T2: Frobenius phi enrichment > SR enrichment (by at least 5 percentage points)
  T3: Frobenius phi enrichment > random baseline (>5% absolute)
  T4: Modulation frequency decreases over time (system approaches fixed point)
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
GENESIS_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(GENESIS_ROOT))

from spectral_utils import PHI, PHI_INV, LN_PHI, SCOPE_RATIO, phi_enrichment

N_VALUES = [8, 16, 32]
N_SEEDS = 30
N_STEPS = 5000
RECORD_EVERY = 100
RESULTS_DIR = GENESIS_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def compute_ratios(eigvals):
    """Consecutive ratios of sorted descending |eigenvalues|."""
    abs_eigs = np.sort(np.abs(eigvals))[::-1]
    ratios = []
    for k in range(len(abs_eigs) - 1):
        if abs_eigs[k + 1] > 1e-12:
            ratios.append(abs_eigs[k] / abs_eigs[k + 1])
    return np.array(ratios)


def run_dynamics(N, seed, normalization='frobenius', n_steps=N_STEPS,
                 weak_factor=None, strong_factor=1.01, initial_sr=1.2):
    """
    Run anti-Hebbian dynamics with specified normalization.

    normalization:
        'sr' — rescale max(|lambda|) to initial_sr every step (control)
        'frobenius' — preserve sqrt(sum(lambda^2)) (allows ratio drift)
        'none' — clip eigenvalues to [0.01, 10.0] only

    Returns list of snapshots recorded every RECORD_EVERY steps.
    """
    rng = np.random.RandomState(seed)

    if weak_factor is None:
        weak_factor = PHI ** (-1.0 / N)

    # Random symmetric W, normalized to initial_sr
    W = rng.randn(N, N) / np.sqrt(N)
    W = (W + W.T) / 2
    eigvals = np.linalg.eigvalsh(W)
    sr = np.max(np.abs(eigvals))
    if sr > 1e-10:
        W = W * (initial_sr / sr)

    state = rng.randn(N) * 0.5
    history = []

    for t in range(n_steps):
        state = np.tanh(W @ state)
        eigvals, eigvecs = np.linalg.eigh(W)

        # Activity-based modulation
        projections = (eigvecs.T @ state) ** 2
        total = np.sum(projections) + 1e-10
        activities = projections / total
        mean_act = 1.0 / N

        n_weak = int(np.sum(activities > 2.0 * mean_act))
        n_strong = int(np.sum(activities < 0.5 * mean_act))

        modulation = np.ones(N)
        modulation[activities > 2.0 * mean_act] = weak_factor
        modulation[activities < 0.5 * mean_act] = strong_factor
        new_eigvals = eigvals * modulation

        # Apply normalization
        if normalization == 'sr':
            post_sr = np.max(np.abs(new_eigvals))
            if post_sr > 1e-10:
                new_eigvals = new_eigvals * (initial_sr / post_sr)

        elif normalization == 'frobenius':
            old_frob = np.sqrt(np.sum(eigvals ** 2))
            new_frob = np.sqrt(np.sum(new_eigvals ** 2))
            if new_frob > 1e-10:
                new_eigvals = new_eigvals * (old_frob / new_frob)

        elif normalization == 'none':
            # Soft clipping: scale down if max exceeds 10
            max_abs = np.max(np.abs(new_eigvals))
            if max_abs > 10.0:
                new_eigvals = new_eigvals * (10.0 / max_abs)

        W = eigvecs @ np.diag(new_eigvals) @ eigvecs.T

        if t % RECORD_EVERY == 0:
            ratios = compute_ratios(new_eigvals)
            enrich = phi_enrichment(ratios) if len(ratios) > 0 else 0.0

            history.append({
                'step': t,
                'sr': float(np.max(np.abs(new_eigvals))),
                'n_modulated': n_weak + n_strong,
                'phi_enrichment': enrich,
                'mean_ratio': float(np.mean(ratios)) if len(ratios) > 0 else 0.0,
                'ratio_std': float(np.std(ratios)) if len(ratios) > 0 else 0.0,
                'state_norm': float(np.linalg.norm(state)),
            })

    return history


def random_baseline_enrichment(N, n_samples=200):
    """Phi enrichment of GOE matrices (no dynamics, just random)."""
    enrichments = []
    for seed in range(n_samples):
        rng = np.random.RandomState(seed + 10000)
        W = rng.randn(N, N) / np.sqrt(N)
        W = (W + W.T) / 2
        eigvals = np.linalg.eigvalsh(W)
        ratios = compute_ratios(eigvals)
        enrichments.append(phi_enrichment(ratios) if len(ratios) > 0 else 0.0)
    return float(np.mean(enrichments)), float(np.std(enrichments))


# ============================================================
# Test 1: Frobenius-normed sr stays bounded
# ============================================================
def test1_stability():
    """T1: sr remains in [0.5, 5.0] under Frobenius normalization."""
    print("\n=== T1: Stability Under Frobenius Normalization ===")

    all_bounded = True
    for N in N_VALUES:
        sr_min, sr_max = float('inf'), 0.0
        for seed in range(N_SEEDS):
            history = run_dynamics(N, seed, normalization='frobenius')
            srs = [h['sr'] for h in history]
            sr_min = min(sr_min, min(srs))
            sr_max = max(sr_max, max(srs))

        bounded = 0.5 <= sr_min and sr_max <= 5.0
        if not bounded:
            all_bounded = False
        print(f"  N={N:3d}: sr in [{sr_min:.3f}, {sr_max:.3f}] "
              f"{'OK' if bounded else 'OUT OF BOUNDS'}")

    print(f"  PASS: {all_bounded}")
    return {
        'test': 'stability',
        'passed': all_bounded,
    }


# ============================================================
# Test 2: Frobenius phi enrichment > SR enrichment
# ============================================================
def test2_frobenius_vs_sr():
    """T2: Frobenius normalization produces more phi enrichment than SR."""
    print("\n=== T2: Frobenius vs SR Normalization ===")

    results_by_N = {}
    all_better = True

    for N in N_VALUES:
        frob_enrichments = []
        sr_enrichments = []

        for seed in range(N_SEEDS):
            h_frob = run_dynamics(N, seed, normalization='frobenius')
            h_sr = run_dynamics(N, seed, normalization='sr')

            # Take final enrichment (last 10 snapshots)
            frob_final = np.mean([h['phi_enrichment'] for h in h_frob[-10:]])
            sr_final = np.mean([h['phi_enrichment'] for h in h_sr[-10:]])

            frob_enrichments.append(frob_final)
            sr_enrichments.append(sr_final)

        mean_frob = float(np.mean(frob_enrichments))
        mean_sr = float(np.mean(sr_enrichments))
        advantage = mean_frob - mean_sr

        better = advantage > 0.05  # 5 percentage points
        if not better:
            all_better = False

        results_by_N[N] = {
            'frob_enrichment': mean_frob,
            'sr_enrichment': mean_sr,
            'advantage': advantage,
        }
        print(f"  N={N:3d}: Frobenius={mean_frob:.4f}, SR={mean_sr:.4f}, "
              f"advantage={advantage:+.4f} "
              f"{'OK' if better else 'MISS'}")

    print(f"  PASS: {all_better}")
    return {
        'test': 'frobenius_vs_sr',
        'passed': all_better,
        'results': {str(k): v for k, v in results_by_N.items()},
    }


# ============================================================
# Test 3: Frobenius enrichment > random baseline
# ============================================================
def test3_above_baseline():
    """T3: Frobenius phi enrichment exceeds random GOE baseline."""
    print("\n=== T3: Frobenius vs Random Baseline ===")

    any_above = False
    results_by_N = {}

    for N in N_VALUES:
        baseline_mean, baseline_std = random_baseline_enrichment(N)

        frob_enrichments = []
        for seed in range(N_SEEDS):
            h = run_dynamics(N, seed, normalization='frobenius')
            frob_enrichments.append(np.mean([s['phi_enrichment'] for s in h[-10:]]))

        mean_frob = float(np.mean(frob_enrichments))
        above = mean_frob > baseline_mean + 0.05  # 5% above baseline

        if above:
            any_above = True

        results_by_N[N] = {
            'frob_enrichment': mean_frob,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'excess': mean_frob - baseline_mean,
        }
        print(f"  N={N:3d}: Frobenius={mean_frob:.4f}, "
              f"baseline={baseline_mean:.4f}+/-{baseline_std:.4f}, "
              f"excess={mean_frob - baseline_mean:+.4f} "
              f"{'OK' if above else 'MISS'}")

    # Pass if ANY N shows excess (phi structure may need sufficient N)
    print(f"  PASS: {any_above}")
    return {
        'test': 'above_baseline',
        'passed': any_above,
        'results': {str(k): v for k, v in results_by_N.items()},
    }


# ============================================================
# Test 4: Modulation frequency decreases (approaching fixed point)
# ============================================================
def test4_convergence():
    """T4: Modulation count decreases over time."""
    print("\n=== T4: Convergence Toward Fixed Point ===")

    converging_count = 0
    total_count = 0

    for N in N_VALUES:
        for seed in range(N_SEEDS):
            h = run_dynamics(N, seed, normalization='frobenius')
            mods = [s['n_modulated'] for s in h]

            # Compare first quarter to last quarter
            q = len(mods) // 4
            if q < 2:
                continue
            early = np.mean(mods[:q])
            late = np.mean(mods[-q:])
            if late < early:
                converging_count += 1
            total_count += 1

    frac_converging = converging_count / total_count if total_count > 0 else 0
    passed = frac_converging > 0.6  # Majority should be converging

    print(f"  Converging: {converging_count}/{total_count} "
          f"({frac_converging:.1%})")
    print(f"  PASS: {passed}")

    return {
        'test': 'convergence',
        'passed': passed,
        'frac_converging': frac_converging,
        'converging_count': converging_count,
        'total_count': total_count,
    }


# ============================================================
# Bonus: Detailed ratio evolution for the best case
# ============================================================
def bonus_ratio_trajectory():
    """Print detailed ratio evolution for N=32 to understand dynamics."""
    print("\n=== Bonus: Ratio Evolution (N=32, seed=0) ===")

    for norm in ['sr', 'frobenius', 'none']:
        h = run_dynamics(32, 0, normalization=norm)
        print(f"\n  {norm.upper()} normalization:")
        for snap in h[::10]:  # Every 1000 steps
            print(f"    step={snap['step']:5d}: sr={snap['sr']:.3f}, "
                  f"phi_enrich={snap['phi_enrichment']:.4f}, "
                  f"mean_ratio={snap['mean_ratio']:.4f}, "
                  f"n_mod={snap['n_modulated']}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Genesis Exp 06: Self-Consistency Attractor")
    print("  Does removing SR normalization allow phi ratios to emerge?")
    print("=" * 70)

    tests = [test1_stability, test2_frobenius_vs_sr,
             test3_above_baseline, test4_convergence]

    results = []
    n_passed = 0

    for test_fn in tests:
        result = test_fn()
        results.append(result)
        if result['passed']:
            n_passed += 1

    bonus_ratio_trajectory()

    print("\n" + "=" * 70)
    print(f"SCORE: {n_passed}/{len(tests)}")
    print("=" * 70)
    for r in results:
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  [{status}] {r['test']}")

    output = {
        'experiment': 'exp_06_self_consistency_attractor',
        'variant': 'TinyCIMM-Genesis',
        'description': 'Does Frobenius normalization allow phi ratio emergence?',
        'score': f'{n_passed}/{len(tests)}',
        'n_passed': n_passed,
        'n_tests': len(tests),
        'tests': results,
        'config': {
            'N_values': N_VALUES,
            'n_seeds': N_SEEDS,
            'n_steps': N_STEPS,
            'record_every': RECORD_EVERY,
        },
        'timestamp': datetime.now().isoformat(),
    }

    fname = RESULTS_DIR / f"exp_06_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {fname}")
