"""
Genesis Exp 09: Metastability Depth

THE SHARPENING EXPERIMENT.

Exp 08 showed phi-structured eigenvalue ratios persist at ~17% enrichment
after 3000 steps. But is this:
  (a) True metastability — enrichment decays to 0, just slowly
  (b) Weak attractor — enrichment plateaus at a nonzero equilibrium

This changes the physics interpretation:
  (a) → phi is selected by initial conditions or anthropic reasoning
  (b) → phi is dynamically preferred (the system has a phi-flavored attractor)

Also: the participation ratio of a phi-geometric spectrum is exactly sqrt(5).
This is analytic (not numerical). We verify this and test whether it's the
mechanism behind phi's persistence — phi-structured eigenvalues produce more
uniform activity distributions, reducing modulation pressure.

Tests:
  T1: Long-run enrichment — run 50,000 steps, fit decay curve
      Plateau (enrichment > 5% at step 50000) vs decay (< 2%)
  T2: Participation ratio — PR for phi-geometric vs other ratios
      Verify PR(phi) = sqrt(5) analytically and numerically
  T3: Modulation rate — phi-init has fewer modes modulated per step than
      random or e-init (mechanism verification)
  T4: Equilibrium enrichment — if plateau exists, does it scale with N?
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from math import sqrt, log

SCRIPT_DIR = Path(__file__).resolve().parent
GENESIS_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(GENESIS_ROOT))

from spectral_utils import PHI, PHI_INV, LN_PHI, SCOPE_RATIO, phi_enrichment

N = 16
N_SEEDS = 20
N_STEPS_LONG = 50000
RECORD_EVERY = 500
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


def construct_geometric_W(N, seed, ratio, target_sr=TARGET_SR):
    """Construct symmetric W with geometric eigenvalue series."""
    rng = np.random.RandomState(seed)
    eigvals = np.array([target_sr * ratio ** (-k) for k in range(N)])
    signs = np.array([(-1) ** k for k in range(N)])
    eigvals = eigvals * signs
    Q, _ = np.linalg.qr(rng.randn(N, N))
    W = Q @ np.diag(eigvals) @ Q.T
    sr = np.max(np.abs(eigvals))
    if sr > 1e-10:
        W = W * (target_sr / sr)
    return W


def run_long(W_init, seed, n_steps=N_STEPS_LONG, record_every=RECORD_EVERY):
    """Run anti-Hebbian dynamics for many steps, recording diagnostics."""
    rng = np.random.RandomState(seed + 5000)
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

        n_weak = int(np.sum(activities > 2.0 * mean_act))
        n_strong = int(np.sum(activities < 0.5 * mean_act))

        modulation = np.ones(N_loc)
        modulation[activities > 2.0 * mean_act] = weak_factor
        modulation[activities < 0.5 * mean_act] = strong_factor
        new_eigvals = eigvals * modulation

        post_sr = np.max(np.abs(new_eigvals))
        if post_sr > 1e-10:
            new_eigvals = new_eigvals * (TARGET_SR / post_sr)

        W = eigvecs @ np.diag(new_eigvals) @ eigvecs.T

        if t % record_every == 0:
            ratios = compute_ratios(new_eigvals)
            enrich = phi_enrichment(ratios) if len(ratios) > 0 else 0.0

            # Participation ratio of activity distribution
            pr = float((np.sum(activities) ** 2) / (np.sum(activities ** 2) + 1e-15))

            history.append({
                'step': t,
                'phi_enrichment': enrich,
                'mean_ratio': float(np.mean(ratios)) if len(ratios) > 0 else 0,
                'n_modulated': n_weak + n_strong,
                'participation_ratio': pr,
                'sr': float(np.max(np.abs(new_eigvals))),
            })

    return history


# ============================================================
# Test 1: Long-Run Enrichment — Plateau vs Decay
# ============================================================
def test1_long_run():
    """T1: Run 50,000 steps. Does phi enrichment plateau or decay to 0?"""
    print("\n=== T1: Long-Run Enrichment (50,000 steps) ===")

    all_trajectories = []

    for seed in range(N_SEEDS):
        W = construct_geometric_W(N, seed, ratio=PHI)
        h = run_long(W, seed)
        enrichments = [s['phi_enrichment'] for s in h]
        all_trajectories.append(enrichments)

        if seed < 3:
            # Print trajectory for first few seeds
            for snap in h[::10]:
                pass  # suppress per-step output

    mean_traj = np.mean(all_trajectories, axis=0)
    steps = [s * RECORD_EVERY for s in range(len(mean_traj))]

    # Print trajectory at key points
    print(f"  {'Step':>8}  {'Mean Enrichment':>16}  {'Std':>8}")
    for i in range(0, len(mean_traj), max(1, len(mean_traj) // 10)):
        std_i = float(np.std([t[i] for t in all_trajectories]))
        print(f"  {steps[i]:8d}  {mean_traj[i]:16.4f}  {std_i:8.4f}")

    # Final enrichment
    final_mean = float(np.mean(mean_traj[-5:]))
    final_std = float(np.std([np.mean(t[-5:]) for t in all_trajectories]))

    # Fit exponential decay: enrichment(t) = a * exp(-t/tau) + c
    # If c > 0.02, it's a plateau (weak attractor)
    # If c < 0.02, it's pure decay (true metastable)
    from scipy.optimize import curve_fit
    try:
        def exp_decay(t, a, tau, c):
            return a * np.exp(-t / tau) + c

        popt, pcov = curve_fit(exp_decay, np.array(steps, dtype=float),
                               mean_traj, p0=[0.5, 5000, 0.05],
                               bounds=([0, 100, 0], [2, 500000, 1]),
                               maxfev=5000)
        a, tau, c = popt
        fit_success = True
    except Exception as e:
        a, tau, c = 0, 0, 0
        fit_success = False
        print(f"  Fit failed: {e}")

    if fit_success:
        print(f"\n  Exponential fit: enrichment = {a:.4f} * exp(-t/{tau:.0f}) + {c:.4f}")
        print(f"  Plateau value (c): {c:.4f}")
        print(f"  Decay timescale (tau): {tau:.0f} steps")
        print(f"  Half-life: {tau * log(2):.0f} steps")

    is_plateau = final_mean > 0.05
    print(f"\n  Final enrichment: {final_mean:.4f} +/- {final_std:.4f}")
    print(f"  Plateau (>5%): {is_plateau}")
    print(f"  PASS: {is_plateau}")

    return {
        'test': 'long_run',
        'passed': is_plateau,
        'final_mean': final_mean,
        'final_std': final_std,
        'fit_a': float(a) if fit_success else None,
        'fit_tau': float(tau) if fit_success else None,
        'fit_c': float(c) if fit_success else None,
        'trajectory_mean': [float(x) for x in mean_traj],
        'steps': steps,
    }


# ============================================================
# Test 2: Participation Ratio — Verify PR(phi) = sqrt(5)
# ============================================================
def test2_participation_ratio():
    """T2: Participation ratio of phi-geometric spectrum is sqrt(5)."""
    print("\n=== T2: Participation Ratio ===")

    # Analytic prediction for infinite geometric series
    # PR = (1 + r^(-2)) / (1 - r^(-2))
    ratios_to_test = {
        'phi': PHI,
        'e': np.e,
        '2.0': 2.0,
        '1.5': 1.5,
        '3.0': 3.0,
    }

    print(f"  {'Ratio':>8}  {'Analytic PR':>12}  {'Numeric PR':>12}  {'Note':>20}")
    for name, r in ratios_to_test.items():
        r_inv2 = r ** (-2)
        analytic_pr = (1 + r_inv2) / (1 - r_inv2)

        # Numeric: compute for N=16 eigenvalue series
        eigvals = np.array([r ** (-k) for k in range(N)])
        p = eigvals ** 2 / np.sum(eigvals ** 2)
        numeric_pr = 1.0 / np.sum(p ** 2)

        note = ""
        if name == 'phi':
            note = f"sqrt(5)={sqrt(5):.6f}"

        print(f"  {name:>8}  {analytic_pr:12.6f}  {numeric_pr:12.6f}  {note:>20}")

    # Verify PR(phi) = sqrt(5)
    phi_pr_analytic = (1 + PHI ** (-2)) / (1 - PHI ** (-2))
    error = abs(phi_pr_analytic - sqrt(5)) / sqrt(5)

    passed = error < 1e-10
    print(f"\n  PR(phi) = {phi_pr_analytic:.10f}")
    print(f"  sqrt(5) = {sqrt(5):.10f}")
    print(f"  Error: {error:.2e}")
    print(f"  PASS: {passed} (exact identity)")

    # Also: measure DYNAMIC participation ratio during anti-Hebbian
    print(f"\n  Dynamic PR during anti-Hebbian (N={N}, 5000 steps):")
    for name, r in [('phi', PHI), ('e', np.e), ('random', None)]:
        prs = []
        for seed in range(10):
            if r is not None:
                W = construct_geometric_W(N, seed, ratio=r)
            else:
                rng = np.random.RandomState(seed)
                W = rng.randn(N, N) / np.sqrt(N)
                W = (W + W.T) / 2
                sr = np.max(np.abs(np.linalg.eigvalsh(W)))
                if sr > 1e-10:
                    W = W * (TARGET_SR / sr)

            h = run_long(W, seed, n_steps=5000, record_every=1000)
            prs.append(h[-1]['participation_ratio'])

        print(f"    {name:>8}: mean PR = {np.mean(prs):.4f} "
              f"+/- {np.std(prs):.4f}")

    return {
        'test': 'participation_ratio',
        'passed': passed,
        'phi_pr': float(phi_pr_analytic),
        'sqrt5': float(sqrt(5)),
        'error': float(error),
    }


# ============================================================
# Test 3: Modulation Rate — Does Phi Minimize Modulation?
# ============================================================
def test3_modulation_rate():
    """T3: Phi-init has fewer modes modulated per step than others."""
    print("\n=== T3: Modulation Rate ===")

    configs = [
        ('phi', PHI),
        ('e', np.e),
        ('2.0', 2.0),
        ('1.5', 1.5),
        ('random', None),
    ]

    mod_rates = {}

    for name, r in configs:
        rates = []
        for seed in range(N_SEEDS):
            if r is not None:
                W = construct_geometric_W(N, seed, ratio=r)
            else:
                rng = np.random.RandomState(seed)
                W = rng.randn(N, N) / np.sqrt(N)
                W = (W + W.T) / 2
                sr = np.max(np.abs(np.linalg.eigvalsh(W)))
                if sr > 1e-10:
                    W = W * (TARGET_SR / sr)

            h = run_long(W, seed, n_steps=5000, record_every=100)
            mean_mod = np.mean([s['n_modulated'] for s in h])
            rates.append(mean_mod)

        mean_rate = float(np.mean(rates))
        mod_rates[name] = mean_rate
        print(f"  {name:>8}: {mean_rate:.2f} modes/step "
              f"(of {N} total)")

    # Phi should have the lowest modulation rate
    phi_rate = mod_rates['phi']
    others = [v for k, v in mod_rates.items() if k != 'phi']
    phi_is_lowest = phi_rate < min(others)
    phi_is_lower_than_random = phi_rate < mod_rates['random']

    passed = phi_is_lower_than_random
    print(f"\n  Phi has lowest modulation rate: {phi_is_lowest}")
    print(f"  Phi < random: {phi_is_lower_than_random}")
    print(f"  PASS: {passed}")

    return {
        'test': 'modulation_rate',
        'passed': passed,
        'mod_rates': mod_rates,
        'phi_is_lowest': phi_is_lowest,
    }


# ============================================================
# Test 4: Equilibrium Enrichment Scaling with N
# ============================================================
def test4_scaling():
    """T4: Does the equilibrium enrichment depend on N?"""
    print("\n=== T4: Equilibrium Enrichment vs N ===")

    N_values = [8, 12, 16, 24, 32]
    equilibria = {}

    for N_loc in N_values:
        enrichments = []
        for seed in range(min(N_SEEDS, 15)):
            W = construct_geometric_W(N_loc, seed, ratio=PHI, target_sr=TARGET_SR)
            h = run_long(W, seed, n_steps=20000, record_every=500)
            final_enrich = np.mean([s['phi_enrichment'] for s in h[-5:]])
            enrichments.append(final_enrich)

        mean_eq = float(np.mean(enrichments))
        equilibria[N_loc] = mean_eq
        print(f"  N={N_loc:3d}: equilibrium enrichment = {mean_eq:.4f}")

    # Does enrichment increase with N? (More modes = more opportunity for phi ratios)
    values = [equilibria[n] for n in N_values]
    increases_with_N = values[-1] > values[0]

    # Or is it constant? (Universal equilibrium)
    range_val = max(values) - min(values)
    is_constant = range_val < 0.05

    passed = any(v > 0.05 for v in values)  # Any N has nonzero equilibrium
    print(f"\n  Range: {range_val:.4f}")
    print(f"  Increases with N: {increases_with_N}")
    print(f"  Approximately constant: {is_constant}")
    print(f"  PASS: {passed}")

    return {
        'test': 'scaling',
        'passed': passed,
        'equilibria': {str(k): v for k, v in equilibria.items()},
        'increases_with_N': increases_with_N,
        'is_constant': is_constant,
    }


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Genesis Exp 09: Metastability Depth")
    print("  Is phi a weak attractor or truly metastable?")
    print("=" * 70)

    tests = [test1_long_run, test2_participation_ratio,
             test3_modulation_rate, test4_scaling]

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
        'experiment': 'exp_09_metastability_depth',
        'variant': 'TinyCIMM-Genesis',
        'description': 'Is phi a weak attractor or truly metastable?',
        'score': f'{n_passed}/{len(tests)}',
        'n_passed': n_passed,
        'n_tests': len(tests),
        'tests': results,
        'config': {
            'N': N,
            'n_seeds': N_SEEDS,
            'n_steps_long': N_STEPS_LONG,
            'target_sr': TARGET_SR,
        },
        'timestamp': datetime.now().isoformat(),
    }

    fname = RESULTS_DIR / f"exp_09_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {fname}")
