"""
Genesis Exp 08: Basin of Attraction for Phi-Structured Fixed Point

THE SHARPEST QUESTION FROM EXP 06.

Exp 06 showed: anti-Hebbian dynamics don't CREATE phi ratios from random
initial conditions. But the SelfApplicator HAS phi ratios as a fixed point.
Is this fixed point STABLE (perturbations return to it) or UNSTABLE
(perturbations drive away)?

If stable: phi structure is a genuine attractor — the SelfApplicator
    construction just finds the attractor faster than random search.
    The basin of attraction may be small, explaining why random W misses it.

If unstable: phi structure is a property of the CONSTRUCTION, not the
    DYNAMICS. The SelfApplicator maintains phi ratios only because it's
    initialized exactly at the fixed point. Any perturbation destroys it.

Method: Initialize W with eigenvalue ratios NEAR phi, then run anti-Hebbian
dynamics. Track whether ratios drift toward phi (stable) or toward 1.0
(unstable).

Specifically:
  1. Construct W with eigvals that form a geometric series with ratio phi
  2. Perturb the eigenvalues by varying amounts (1%, 5%, 10%, 20%)
  3. Run anti-Hebbian + SR normalization (same as M10)
  4. Track phi enrichment over time

Tests:
  T1: Unperturbed phi-structured W maintains enrichment > 15%
  T2: Small perturbation (5%) returns to high enrichment (>10%)
  T3: Large perturbation (20%) — does enrichment increase or decrease?
  T4: Compare to geometric series with ratio e (non-phi) — does that maintain too?
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

N = 16
N_SEEDS = 30
N_STEPS = 3000
RECORD_EVERY = 50
TARGET_SR = 1.2
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


def construct_phi_structured_W(N, seed, target_sr=TARGET_SR, ratio=PHI,
                                perturbation=0.0):
    """
    Construct a symmetric W whose eigenvalue ratios are approximately `ratio`.

    The eigenvalues form a geometric series: lambda_k = target_sr * ratio^(-k)
    for k=0..N-1, with signs alternating (to fill both positive and negative).

    Then we perturb each eigenvalue by a random fraction of `perturbation`.
    """
    rng = np.random.RandomState(seed)

    # Geometric series eigenvalues
    eigvals = np.array([target_sr * ratio ** (-k) for k in range(N)])

    # Alternate signs so we have both positive and negative eigenvalues
    # (symmetric matrices typically have both)
    signs = np.array([(-1) ** k for k in range(N)])
    eigvals = eigvals * signs

    # Perturbation
    if perturbation > 0:
        noise = rng.randn(N) * perturbation
        eigvals = eigvals * (1 + noise)

    # Random orthogonal eigenvectors
    Q = rng.randn(N, N)
    Q, _ = np.linalg.qr(Q)

    # Construct W = Q @ diag(eigvals) @ Q.T
    W = Q @ np.diag(eigvals) @ Q.T

    # Normalize to target sr
    sr = np.max(np.abs(eigvals))
    if sr > 1e-10:
        W = W * (target_sr / sr)

    return W, Q


def run_from_structured_W(W_init, seed, n_steps=N_STEPS):
    """
    Run anti-Hebbian dynamics starting from a pre-structured W.
    Uses SR normalization (same as M10 SelfApplicator).
    """
    rng = np.random.RandomState(seed + 1000)
    N_loc = W_init.shape[0]
    W = W_init.copy()

    weak_factor = PHI ** (-1.0 / N_loc)
    strong_factor = 1.01

    state = rng.randn(N_loc) * 0.5
    history = []

    for t in range(n_steps):
        state = np.tanh(W @ state)
        eigvals, eigvecs = np.linalg.eigh(W)

        projections = (eigvecs.T @ state) ** 2
        total = np.sum(projections) + 1e-10
        activities = projections / total
        mean_act = 1.0 / N_loc

        modulation = np.ones(N_loc)
        modulation[activities > 2.0 * mean_act] = weak_factor
        modulation[activities < 0.5 * mean_act] = strong_factor
        new_eigvals = eigvals * modulation

        # SR normalization (same as M10)
        post_sr = np.max(np.abs(new_eigvals))
        if post_sr > 1e-10:
            new_eigvals = new_eigvals * (TARGET_SR / post_sr)

        W = eigvecs @ np.diag(new_eigvals) @ eigvecs.T

        if t % RECORD_EVERY == 0:
            ratios = compute_ratios(new_eigvals)
            enrich = phi_enrichment(ratios) if len(ratios) > 0 else 0.0
            mean_r = float(np.mean(ratios)) if len(ratios) > 0 else 0.0
            history.append({
                'step': t,
                'phi_enrichment': enrich,
                'mean_ratio': mean_r,
                'sr': float(np.max(np.abs(new_eigvals))),
            })

    return history


# ============================================================
# Test 1: Unperturbed phi-structured W
# ============================================================
def test1_unperturbed():
    """T1: Phi-structured W maintains enrichment > 15%."""
    print("\n=== T1: Unperturbed Phi-Structured W ===")

    final_enrichments = []

    for seed in range(N_SEEDS):
        W, _ = construct_phi_structured_W(N, seed, ratio=PHI, perturbation=0.0)

        # Check initial enrichment
        init_ratios = compute_ratios(np.linalg.eigvalsh(W))
        init_enrich = phi_enrichment(init_ratios)

        h = run_from_structured_W(W, seed)
        final_enrich = np.mean([s['phi_enrichment'] for s in h[-10:]])
        final_enrichments.append(final_enrich)

        if seed < 3:
            print(f"  seed={seed}: init={init_enrich:.4f} -> final={final_enrich:.4f}")

    mean_final = float(np.mean(final_enrichments))
    passed = mean_final > 0.15

    print(f"  Mean final enrichment: {mean_final:.4f}")
    print(f"  PASS: {passed} (need > 0.15)")

    return {
        'test': 'unperturbed_phi',
        'passed': passed,
        'mean_final_enrichment': mean_final,
        'std_final': float(np.std(final_enrichments)),
    }


# ============================================================
# Test 2: Small perturbation recovery
# ============================================================
def test2_small_perturbation():
    """T2: 5% perturbation — does enrichment recover?"""
    print("\n=== T2: Small Perturbation (5%) ===")

    trajectories = {'enrichment_over_time': []}
    final_enrichments = []

    for seed in range(N_SEEDS):
        W, _ = construct_phi_structured_W(N, seed, ratio=PHI, perturbation=0.05)
        h = run_from_structured_W(W, seed)

        enrichments_t = [s['phi_enrichment'] for s in h]
        trajectories['enrichment_over_time'].append(enrichments_t)
        final_enrichments.append(np.mean([s['phi_enrichment'] for s in h[-10:]]))

    mean_final = float(np.mean(final_enrichments))

    # Check trajectory: does enrichment increase from initial perturbed value?
    mean_trajectory = np.mean(trajectories['enrichment_over_time'], axis=0)
    initial = mean_trajectory[0]
    final_traj = mean_trajectory[-1]
    increases = final_traj > initial

    passed = mean_final > 0.10
    print(f"  Mean final enrichment: {mean_final:.4f}")
    print(f"  Trajectory: {initial:.4f} -> {final_traj:.4f} "
          f"({'increases' if increases else 'decreases'})")
    print(f"  PASS: {passed} (need > 0.10)")

    return {
        'test': 'small_perturbation',
        'passed': passed,
        'mean_final_enrichment': mean_final,
        'trajectory_direction': 'increase' if increases else 'decrease',
    }


# ============================================================
# Test 3: Large perturbation — which way does enrichment go?
# ============================================================
def test3_large_perturbation():
    """T3: 20% perturbation — does enrichment increase or decrease?"""
    print("\n=== T3: Large Perturbation (20%) ===")

    perturbation_levels = [0.01, 0.05, 0.10, 0.20, 0.50]
    results_by_pert = {}

    for pert in perturbation_levels:
        final_enrichments = []
        for seed in range(N_SEEDS):
            W, _ = construct_phi_structured_W(N, seed, ratio=PHI, perturbation=pert)
            h = run_from_structured_W(W, seed)
            final_enrichments.append(np.mean([s['phi_enrichment'] for s in h[-10:]]))

        mean_final = float(np.mean(final_enrichments))
        results_by_pert[pert] = mean_final
        print(f"  perturbation={pert:.0%}: final enrichment={mean_final:.4f}")

    # Is there a monotonic relationship? More perturbation → less enrichment?
    values = list(results_by_pert.values())
    monotone_decrease = all(values[i] >= values[i + 1] - 0.02
                            for i in range(len(values) - 1))

    # Does 20% perturbation still maintain enrichment > 5%?
    survives = results_by_pert[0.20] > 0.05

    passed = survives
    print(f"  Monotone decrease: {monotone_decrease}")
    print(f"  Survives 20% perturbation: {survives}")
    print(f"  PASS: {passed}")

    return {
        'test': 'large_perturbation',
        'passed': passed,
        'results_by_perturbation': {str(k): v for k, v in results_by_pert.items()},
        'monotone_decrease': monotone_decrease,
    }


# ============================================================
# Test 4: Non-phi geometric ratio (control)
# ============================================================
def test4_non_phi_control():
    """T4: Does a non-phi geometric ratio (e=2.718) also maintain?"""
    print("\n=== T4: Non-Phi Control (ratio=e) ===")

    import math
    E = math.e

    phi_enrichments = []
    e_enrichments = []

    for seed in range(N_SEEDS):
        # Phi-structured
        W_phi, _ = construct_phi_structured_W(N, seed, ratio=PHI, perturbation=0.0)
        h_phi = run_from_structured_W(W_phi, seed)
        phi_enrichments.append(np.mean([s['phi_enrichment'] for s in h_phi[-10:]]))

        # E-structured
        W_e, _ = construct_phi_structured_W(N, seed, ratio=E, perturbation=0.0)
        h_e = run_from_structured_W(W_e, seed)
        e_enrichments.append(np.mean([s['phi_enrichment'] for s in h_e[-10:]]))

    mean_phi = float(np.mean(phi_enrichments))
    mean_e = float(np.mean(e_enrichments))

    # Also check mean ratio convergence
    h_phi_ex = run_from_structured_W(
        construct_phi_structured_W(N, 0, ratio=PHI)[0], 0)
    h_e_ex = run_from_structured_W(
        construct_phi_structured_W(N, 0, ratio=E)[0], 0)

    phi_mean_ratio_final = h_phi_ex[-1]['mean_ratio']
    e_mean_ratio_final = h_e_ex[-1]['mean_ratio']

    print(f"  Phi-structured: enrichment={mean_phi:.4f}, "
          f"mean_ratio_final={phi_mean_ratio_final:.4f}")
    print(f"  E-structured:   enrichment={mean_e:.4f}, "
          f"mean_ratio_final={e_mean_ratio_final:.4f}")

    # Phi should maintain MORE enrichment than e (since phi is the predicted attractor)
    phi_advantage = mean_phi > mean_e + 0.05
    print(f"  Phi advantage: {phi_advantage} "
          f"(difference={mean_phi - mean_e:+.4f})")

    passed = phi_advantage
    print(f"  PASS: {passed}")

    return {
        'test': 'non_phi_control',
        'passed': passed,
        'phi_enrichment': mean_phi,
        'e_enrichment': mean_e,
        'phi_mean_ratio': phi_mean_ratio_final,
        'e_mean_ratio': e_mean_ratio_final,
    }


# ============================================================
# Bonus: Enrichment trajectory comparison
# ============================================================
def bonus_trajectories():
    """Print enrichment over time for phi vs e vs random init."""
    print("\n=== Bonus: Enrichment Trajectories (seed=0) ===")

    import math

    # Phi-structured
    W_phi, _ = construct_phi_structured_W(N, 0, ratio=PHI, perturbation=0.0)
    h_phi = run_from_structured_W(W_phi, 0)

    # E-structured
    W_e, _ = construct_phi_structured_W(N, 0, ratio=math.e, perturbation=0.0)
    h_e = run_from_structured_W(W_e, 0)

    # 2-structured
    W_2, _ = construct_phi_structured_W(N, 0, ratio=2.0, perturbation=0.0)
    h_2 = run_from_structured_W(W_2, 0)

    # Random (from exp_06)
    rng = np.random.RandomState(0)
    W_rand = rng.randn(N, N) / np.sqrt(N)
    W_rand = (W_rand + W_rand.T) / 2
    sr = np.max(np.abs(np.linalg.eigvalsh(W_rand)))
    if sr > 1e-10:
        W_rand = W_rand * (TARGET_SR / sr)
    h_rand = run_from_structured_W(W_rand, 0)

    print(f"\n  {'step':>6}  {'phi':>8}  {'e':>8}  {'2.0':>8}  {'random':>8}")
    for i in range(0, len(h_phi), 10):  # Every 500 steps
        print(f"  {h_phi[i]['step']:6d}  "
              f"{h_phi[i]['phi_enrichment']:8.4f}  "
              f"{h_e[i]['phi_enrichment']:8.4f}  "
              f"{h_2[i]['phi_enrichment']:8.4f}  "
              f"{h_rand[i]['phi_enrichment']:8.4f}")

    print(f"\n  Mean ratio at end:")
    print(f"    phi-init:    {h_phi[-1]['mean_ratio']:.4f}")
    print(f"    e-init:      {h_e[-1]['mean_ratio']:.4f}")
    print(f"    2.0-init:    {h_2[-1]['mean_ratio']:.4f}")
    print(f"    random-init: {h_rand[-1]['mean_ratio']:.4f}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Genesis Exp 08: Basin of Attraction for Phi-Structured Fixed Point")
    print("  Is phi-ratio structure stable or unstable under anti-Hebbian?")
    print("=" * 70)

    tests = [test1_unperturbed, test2_small_perturbation,
             test3_large_perturbation, test4_non_phi_control]

    results = []
    n_passed = 0

    for test_fn in tests:
        result = test_fn()
        results.append(result)
        if result['passed']:
            n_passed += 1

    bonus_trajectories()

    print("\n" + "=" * 70)
    print(f"SCORE: {n_passed}/{len(tests)}")
    print("=" * 70)
    for r in results:
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  [{status}] {r['test']}")

    output = {
        'experiment': 'exp_08_phi_basin_of_attraction',
        'variant': 'TinyCIMM-Genesis',
        'description': 'Is phi eigenvalue structure a stable attractor?',
        'score': f'{n_passed}/{len(tests)}',
        'n_passed': n_passed,
        'n_tests': len(tests),
        'tests': results,
        'config': {
            'N': N,
            'n_seeds': N_SEEDS,
            'n_steps': N_STEPS,
            'target_sr': TARGET_SR,
        },
        'timestamp': datetime.now().isoformat(),
    }

    fname = RESULTS_DIR / f"exp_08_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {fname}")
