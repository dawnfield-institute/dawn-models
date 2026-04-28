"""
Genesis Exp 07: RMT vs DFT in Eigenvalue Spacing

PLANCK THREAD INVESTIGATION.

Genesis exp_04 found: minimum eigenvalue spacing scales ~1/N^2.3, with
spacing at N=8 being 20x spacing at N=32. Is this DFT or random matrix
theory (RMT)?

For GOE (Gaussian Orthogonal Ensemble — the universality class of random
symmetric matrices), eigenvalue repulsion follows the Wigner surmise:
  p(s) = (pi/2) * s * exp(-pi*s^2/4)
where s = spacing/mean_spacing.

Anti-Hebbian modulation changes the spectrum from GOE. The question: HOW
does it change it? Three possibilities:

1. Modulated spacing matches GOE scaling -> spacing floor is just RMT
2. Modulated spacing EXCEEDS GOE (more repulsion) -> anti-Hebbian adds repulsion
3. Modulated spacing DEVIATES from GOE in phi-structured way -> DFT signature

This experiment computes spacing statistics for:
  A) Raw GOE matrices (no dynamics)
  B) GOE + SR normalization (same as Genesis exp_04)
  C) GOE + anti-Hebbian modulation + SR normalization
  D) GOE + anti-Hebbian + Frobenius normalization

Tests:
  T1: GOE spacing scaling — confirm 1/N scaling in raw GOE
  T2: Modulated spacing deviates from GOE — anti-Hebbian changes spacing structure
  T3: Scaling exponent — measure N-scaling exponent precisely
  T4: Phi structure in spacing — does min_spacing / mean_spacing relate to phi?
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
GENESIS_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(GENESIS_ROOT))

from spectral_utils import PHI, PHI_INV, LN_PHI, SCOPE_RATIO

N_VALUES = [8, 12, 16, 24, 32, 48, 64]
N_SAMPLES = 200
N_STEPS = 1000
RESULTS_DIR = GENESIS_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def compute_spacing_stats(eigvals):
    """
    Compute nearest-neighbor spacing statistics for a set of eigenvalues.
    Returns min, mean, std of spacings, and the normalized spacing distribution.
    """
    sorted_eigs = np.sort(eigvals)
    spacings = np.diff(sorted_eigs)

    if len(spacings) == 0:
        return {'min': 0, 'mean': 0, 'std': 0, 'spacings': []}

    mean_sp = np.mean(spacings)
    normalized = spacings / mean_sp if mean_sp > 1e-15 else spacings

    return {
        'min': float(np.min(spacings)),
        'mean': float(mean_sp),
        'std': float(np.std(spacings)),
        'min_normalized': float(np.min(normalized)),
        'mean_normalized': float(np.mean(normalized)),
    }


def goe_spacing(N, n_samples=N_SAMPLES):
    """Spacing statistics for raw GOE matrices."""
    min_spacings = []
    mean_spacings = []
    min_normalized = []

    for seed in range(n_samples):
        rng = np.random.RandomState(seed)
        W = rng.randn(N, N) / np.sqrt(N)
        W = (W + W.T) / 2
        eigvals = np.linalg.eigvalsh(W)

        stats = compute_spacing_stats(eigvals)
        min_spacings.append(stats['min'])
        mean_spacings.append(stats['mean'])
        min_normalized.append(stats['min_normalized'])

    return {
        'min_spacing': float(np.mean(min_spacings)),
        'min_spacing_std': float(np.std(min_spacings)),
        'mean_spacing': float(np.mean(mean_spacings)),
        'min_normalized': float(np.mean(min_normalized)),
        'min_normalized_std': float(np.std(min_normalized)),
    }


def modulated_spacing(N, n_samples=N_SAMPLES, normalization='sr'):
    """
    Spacing statistics after anti-Hebbian modulation.
    """
    weak_factor = PHI ** (-1.0 / N)
    strong_factor = 1.01

    min_spacings = []
    mean_spacings = []
    min_normalized = []

    for seed in range(n_samples):
        rng = np.random.RandomState(seed)
        W = rng.randn(N, N) / np.sqrt(N)
        W = (W + W.T) / 2
        eigvals = np.linalg.eigvalsh(W)
        sr = np.max(np.abs(eigvals))
        if sr > 1e-10:
            W = W * (1.2 / sr)

        state = rng.randn(N) * 0.5

        for t in range(N_STEPS):
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

            if normalization == 'sr':
                post_sr = np.max(np.abs(new_eigvals))
                if post_sr > 1e-10:
                    new_eigvals = new_eigvals * (1.2 / post_sr)
            elif normalization == 'frobenius':
                old_frob = np.sqrt(np.sum(eigvals ** 2))
                new_frob = np.sqrt(np.sum(new_eigvals ** 2))
                if new_frob > 1e-10:
                    new_eigvals = new_eigvals * (old_frob / new_frob)

            W = eigvecs @ np.diag(new_eigvals) @ eigvecs.T

        # Final spacing
        final_eigvals = np.linalg.eigvalsh(W)
        stats = compute_spacing_stats(final_eigvals)
        min_spacings.append(stats['min'])
        mean_spacings.append(stats['mean'])
        min_normalized.append(stats['min_normalized'])

    return {
        'min_spacing': float(np.mean(min_spacings)),
        'min_spacing_std': float(np.std(min_spacings)),
        'mean_spacing': float(np.mean(mean_spacings)),
        'min_normalized': float(np.mean(min_normalized)),
        'min_normalized_std': float(np.std(min_normalized)),
    }


def fit_power_law(N_vals, spacings):
    """Fit min_spacing = a * N^(-alpha). Returns alpha and R^2."""
    log_N = np.log(np.array(N_vals, dtype=float))
    log_sp = np.log(np.array(spacings, dtype=float) + 1e-15)

    # Linear regression in log space
    A = np.vstack([log_N, np.ones(len(log_N))]).T
    result = np.linalg.lstsq(A, log_sp, rcond=None)
    slope, intercept = result[0]

    # R^2
    predicted = slope * log_N + intercept
    ss_res = np.sum((log_sp - predicted) ** 2)
    ss_tot = np.sum((log_sp - np.mean(log_sp)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 0

    return -slope, r_squared  # alpha (positive), R^2


# ============================================================
# Test 1: GOE spacing scaling
# ============================================================
def test1_goe_scaling():
    """T1: Confirm GOE min spacing scales as 1/N^alpha."""
    print("\n=== T1: GOE Spacing Scaling ===")

    goe_data = {}
    min_spacings = []

    for N in N_VALUES:
        stats = goe_spacing(N)
        goe_data[N] = stats
        min_spacings.append(stats['min_spacing'])
        print(f"  N={N:3d}: min_spacing={stats['min_spacing']:.6f}, "
              f"mean_spacing={stats['mean_spacing']:.4f}")

    alpha, r2 = fit_power_law(N_VALUES, min_spacings)
    print(f"  Power law fit: min_spacing ~ N^(-{alpha:.2f}), R^2={r2:.4f}")

    # GOE theory predicts min spacing ~ 1/N^2 for the bulk
    passed = r2 > 0.95 and alpha > 1.0
    print(f"  PASS: {passed} (need R^2>0.95, alpha>1.0)")

    return {
        'test': 'goe_scaling',
        'passed': passed,
        'alpha': alpha,
        'r_squared': r2,
        'data': {str(k): v for k, v in goe_data.items()},
    }


# ============================================================
# Test 2: Anti-Hebbian modulation changes spacing
# ============================================================
def test2_modulation_effect():
    """T2: Modulated spacing differs from GOE."""
    print("\n=== T2: Modulation Effect on Spacing ===")

    deviations = {}
    any_significant = False

    for N in [8, 16, 32]:
        goe = goe_spacing(N)
        mod_sr = modulated_spacing(N, normalization='sr')
        mod_frob = modulated_spacing(N, normalization='frobenius')

        # Ratio of modulated to GOE min spacing
        ratio_sr = mod_sr['min_spacing'] / goe['min_spacing'] if goe['min_spacing'] > 1e-15 else 0
        ratio_frob = mod_frob['min_spacing'] / goe['min_spacing'] if goe['min_spacing'] > 1e-15 else 0

        # Significant if ratio differs from 1.0 by >20%
        sig_sr = abs(ratio_sr - 1.0) > 0.2
        sig_frob = abs(ratio_frob - 1.0) > 0.2
        if sig_sr or sig_frob:
            any_significant = True

        deviations[N] = {
            'goe_min': goe['min_spacing'],
            'sr_min': mod_sr['min_spacing'],
            'frob_min': mod_frob['min_spacing'],
            'ratio_sr': ratio_sr,
            'ratio_frob': ratio_frob,
        }
        print(f"  N={N:3d}: GOE={goe['min_spacing']:.6f}, "
              f"SR={mod_sr['min_spacing']:.6f} ({ratio_sr:.2f}x), "
              f"Frob={mod_frob['min_spacing']:.6f} ({ratio_frob:.2f}x)")

    passed = any_significant
    print(f"  PASS: {passed} (modulation changes spacing structure)")

    return {
        'test': 'modulation_effect',
        'passed': passed,
        'deviations': {str(k): v for k, v in deviations.items()},
    }


# ============================================================
# Test 3: Precise scaling exponent for modulated systems
# ============================================================
def test3_scaling_exponent():
    """T3: What is the exact N-scaling exponent for modulated spacing?"""
    print("\n=== T3: Scaling Exponent ===")

    # GOE scaling
    goe_spacings = []
    for N in N_VALUES:
        stats = goe_spacing(N, n_samples=100)
        goe_spacings.append(stats['min_spacing'])

    goe_alpha, goe_r2 = fit_power_law(N_VALUES, goe_spacings)

    # Modulated (SR) scaling
    sr_spacings = []
    for N in N_VALUES:
        stats = modulated_spacing(N, n_samples=100, normalization='sr')
        sr_spacings.append(stats['min_spacing'])

    sr_alpha, sr_r2 = fit_power_law(N_VALUES, sr_spacings)

    # Modulated (Frobenius) scaling
    frob_spacings = []
    for N in N_VALUES:
        stats = modulated_spacing(N, n_samples=100, normalization='frobenius')
        frob_spacings.append(stats['min_spacing'])

    frob_alpha, frob_r2 = fit_power_law(N_VALUES, frob_spacings)

    print(f"  GOE:       alpha={goe_alpha:.3f}, R^2={goe_r2:.4f}")
    print(f"  SR-mod:    alpha={sr_alpha:.3f}, R^2={sr_r2:.4f}")
    print(f"  Frob-mod:  alpha={frob_alpha:.3f}, R^2={frob_r2:.4f}")

    # Is the exponent DIFFERENT for modulated vs GOE?
    exponent_differs = abs(sr_alpha - goe_alpha) > 0.3 or abs(frob_alpha - goe_alpha) > 0.3

    passed = exponent_differs
    print(f"  Exponents differ: {passed} (GOE={goe_alpha:.2f}, "
          f"SR={sr_alpha:.2f}, Frob={frob_alpha:.2f})")
    print(f"  PASS: {passed}")

    return {
        'test': 'scaling_exponent',
        'passed': passed,
        'goe_alpha': goe_alpha,
        'goe_r2': goe_r2,
        'sr_alpha': sr_alpha,
        'sr_r2': sr_r2,
        'frob_alpha': frob_alpha,
        'frob_r2': frob_r2,
    }


# ============================================================
# Test 4: Phi structure in spacing
# ============================================================
def test4_phi_in_spacing():
    """T4: Does the spacing structure relate to phi?"""
    print("\n=== T4: Phi Structure in Spacing ===")

    phi_connections = {}

    for N in [8, 16, 32]:
        stats = modulated_spacing(N, n_samples=150, normalization='sr')

        # Check if min_spacing / mean_spacing relates to phi
        ratio = stats['min_spacing'] / stats['mean_spacing'] if stats['mean_spacing'] > 1e-15 else 0

        # Possible phi connections:
        # 1/phi = 0.618
        # 1/phi^2 = 0.382
        # phi^(-1/N) for various N
        near_phi_inv = abs(ratio - PHI_INV) / PHI_INV < 0.15
        near_phi_inv2 = abs(ratio - PHI_INV ** 2) / PHI_INV ** 2 < 0.15
        near_phi_inv_N = abs(ratio - PHI ** (-1.0 / N)) / PHI ** (-1.0 / N) < 0.15

        phi_connections[N] = {
            'min_over_mean': ratio,
            'near_1/phi': near_phi_inv,
            'near_1/phi^2': near_phi_inv2,
            'near_phi^(-1/N)': near_phi_inv_N,
        }
        print(f"  N={N:3d}: min/mean={ratio:.4f} "
              f"(1/phi={PHI_INV:.4f}, 1/phi^2={PHI_INV**2:.4f}, "
              f"phi^(-1/{N})={PHI**(-1.0/N):.4f})")

    # Any phi connection found
    any_phi = any(
        v['near_1/phi'] or v['near_1/phi^2'] or v['near_phi^(-1/N)']
        for v in phi_connections.values()
    )

    passed = any_phi
    print(f"  PASS: {passed}")

    return {
        'test': 'phi_in_spacing',
        'passed': passed,
        'connections': {str(k): v for k, v in phi_connections.items()},
    }


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Genesis Exp 07: RMT vs DFT in Eigenvalue Spacing")
    print("  Is the spacing floor random matrix theory or physics?")
    print("=" * 70)

    tests = [test1_goe_scaling, test2_modulation_effect,
             test3_scaling_exponent, test4_phi_in_spacing]

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
        'experiment': 'exp_07_rmt_spacing_comparison',
        'variant': 'TinyCIMM-Genesis',
        'description': 'Compare eigenvalue spacing to GOE predictions',
        'score': f'{n_passed}/{len(tests)}',
        'n_passed': n_passed,
        'n_tests': len(tests),
        'tests': results,
        'config': {
            'N_values': N_VALUES,
            'n_samples': N_SAMPLES,
            'n_steps': N_STEPS,
        },
        'timestamp': datetime.now().isoformat(),
    }

    fname = RESULTS_DIR / f"exp_07_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {fname}")
