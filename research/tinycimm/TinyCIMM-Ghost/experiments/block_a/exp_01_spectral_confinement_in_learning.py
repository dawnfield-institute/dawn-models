"""
Ghost Exp 01: Spectral Confinement in Learning

Does W_core maintain eigenvector fixity during training?

M10 showed eigenvector drift < 2.4e-15 in the SelfApplicator. Ghost's core
is a symmetric recurrent system with anti-Hebbian modulation. During training,
the encoder/decoder weights change via gradients, but the CORE should maintain
spectral confinement: eigenvectors fixed, only eigenvalues change.

Negative control: asymmetric core (no symmetry constraint) should show drift.

Tests:
  T1: Eigenvector drift < 1e-10 after training
  T2: Drift stays low throughout training (not just at end)
  T3: Asymmetric core shows significant drift (negative control)
  T4: Core spectral radius stays near gamma/ln(phi) during training
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
GHOST_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(GHOST_ROOT))

from spectral_utils import PHI, SCOPE_RATIO
from ghost_network import GhostNetwork, GhostConfig
from ghost_core import SymmetricRecurrentCore, CoreConfig

N_SEEDS = 10
N_EPOCHS = 200
RESULTS_DIR = GHOST_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def generate_regression_data(n_samples=200, input_dim=5, seed=0):
    """Simple regression data for training."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, input_dim)
    # Power-law relationship
    Y = np.sum(X ** 2, axis=1, keepdims=True) * 0.5
    return X, Y


# ============================================================
# Test 1: Eigenvector Drift After Training
# ============================================================
def test1_drift_after_training():
    """T1: Eigenvector drift < 1e-10 after 200 epochs."""
    print("\n=== T1: Eigenvector Drift After Training ===")

    drifts = []
    for seed in range(N_SEEDS):
        config = GhostConfig(input_dim=5, output_dim=1, core_dim=13,
                             seed=seed)
        net = GhostNetwork(config)
        X, Y = generate_regression_data(seed=seed)
        net.train(X, Y, epochs=N_EPOCHS)

        drift = net.core.eigenvector_drift()
        drifts.append(drift)
        print(f"  seed={seed}: drift={drift:.2e}")

    mean_drift = float(np.mean(drifts))
    max_drift = float(np.max(drifts))

    passed = max_drift < 1e-10
    print(f"  Mean drift: {mean_drift:.2e}")
    print(f"  Max drift: {max_drift:.2e}")
    print(f"  PASS: {passed} (max < 1e-10)")

    return {
        'test': 'drift_after_training',
        'passed': bool(passed),
        'mean_drift': mean_drift,
        'max_drift': max_drift,
        'drifts': [float(d) for d in drifts],
    }


# ============================================================
# Test 2: Drift Throughout Training
# ============================================================
def test2_drift_during_training():
    """T2: Drift stays low throughout training."""
    print("\n=== T2: Drift Throughout Training ===")

    config = GhostConfig(input_dim=5, output_dim=1, core_dim=13,
                         K=3, seed=42)
    net = GhostNetwork(config)
    X, Y = generate_regression_data(seed=42)

    drift_timeline = []
    for epoch in range(N_EPOCHS):
        metrics = net.update(X, Y)
        if epoch % 20 == 0 or epoch == N_EPOCHS - 1:
            drift = net.core.eigenvector_drift()
            drift_timeline.append({'epoch': epoch, 'drift': float(drift)})
            print(f"  epoch {epoch:4d}: drift={drift:.2e}")

    max_drift = max(d['drift'] for d in drift_timeline)
    always_low = all(d['drift'] < 1e-10 for d in drift_timeline)

    passed = always_low
    print(f"  Max drift across training: {max_drift:.2e}")
    print(f"  PASS: {passed} (always < 1e-10)")

    return {
        'test': 'drift_during_training',
        'passed': bool(passed),
        'max_drift': float(max_drift),
        'always_low': bool(always_low),
        'timeline': drift_timeline,
    }


# ============================================================
# Test 3: Negative Control (Asymmetric Core)
# ============================================================
def test3_asymmetric_control():
    """T3: Asymmetric core shows significant eigenvector drift."""
    print("\n=== T3: Negative Control (Asymmetric Core) ===")

    drifts_sym = []
    drifts_asym = []

    for seed in range(N_SEEDS):
        # Symmetric core
        config = GhostConfig(input_dim=5, output_dim=1, core_dim=13,
                             seed=seed)
        net_sym = GhostNetwork(config)
        X, Y = generate_regression_data(seed=seed)
        net_sym.train(X, Y, epochs=N_EPOCHS)
        drifts_sym.append(net_sym.core.eigenvector_drift())

        # Asymmetric core: break symmetry by adding noise
        net_asym = GhostNetwork(config)
        rng = np.random.RandomState(seed + 1000)
        noise = rng.randn(13, 13) * 0.01
        net_asym.core.W = net_asym.core.W + noise  # Break symmetry
        # Reset initial eigvecs to current (after symmetry breaking)
        _, net_asym.core.initial_eigvecs = np.linalg.eigh(
            (net_asym.core.W + net_asym.core.W.T) / 2)
        net_asym.train(X, Y, epochs=N_EPOCHS)
        # Measure drift against the ORIGINAL symmetric eigvecs
        _, current_vecs = np.linalg.eigh(
            (net_asym.core.W + net_asym.core.W.T) / 2)
        alignment = np.abs(np.diag(
            current_vecs.T @ net_asym.core.initial_eigvecs))
        asym_drift = float(1.0 - np.mean(alignment))
        drifts_asym.append(asym_drift)

    mean_sym = float(np.mean(drifts_sym))
    mean_asym = float(np.mean(drifts_asym))
    ratio = mean_asym / mean_sym if mean_sym > 0 else float('inf')

    passed = mean_asym > 10 * mean_sym  # Asymmetric should drift much more
    print(f"  Symmetric drift: {mean_sym:.2e}")
    print(f"  Asymmetric drift: {mean_asym:.2e}")
    print(f"  Ratio: {ratio:.1f}x")
    print(f"  PASS: {passed} (asymmetric > 10x symmetric)")

    return {
        'test': 'asymmetric_control',
        'passed': bool(passed),
        'mean_symmetric': mean_sym,
        'mean_asymmetric': mean_asym,
        'ratio': float(ratio),
    }


# ============================================================
# Test 4: SR Stability During Training
# ============================================================
def test4_sr_stability():
    """T4: Core spectral radius stays near gamma/ln(phi) during training."""
    print("\n=== T4: SR Stability ===")

    config = GhostConfig(input_dim=5, output_dim=1, core_dim=13,
                         K=3, seed=42)
    net = GhostNetwork(config)
    X, Y = generate_regression_data(seed=42)

    sr_timeline = []
    for epoch in range(N_EPOCHS):
        net.update(X, Y)
        if epoch % 20 == 0 or epoch == N_EPOCHS - 1:
            sr = net.core.spectral_radius()
            sr_timeline.append({'epoch': epoch, 'sr': float(sr)})

    srs = [s['sr'] for s in sr_timeline]
    mean_sr = float(np.mean(srs))
    std_sr = float(np.std(srs))
    error_pct = abs(mean_sr - SCOPE_RATIO) / SCOPE_RATIO * 100

    for s in sr_timeline[::2]:
        print(f"  epoch {s['epoch']:4d}: sr={s['sr']:.4f}")

    passed = error_pct < 1.0 and std_sr < 0.01
    print(f"  Mean sr: {mean_sr:.4f} (target: {SCOPE_RATIO:.4f})")
    print(f"  Error: {error_pct:.2f}%, Std: {std_sr:.6f}")
    print(f"  PASS: {passed}")

    return {
        'test': 'sr_stability',
        'passed': bool(passed),
        'mean_sr': mean_sr,
        'std_sr': std_sr,
        'error_pct': float(error_pct),
        'timeline': sr_timeline,
    }


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Ghost Exp 01: Spectral Confinement in Learning")
    print("  Does W_core maintain eigenvector fixity during training?")
    print("=" * 70)

    tests = [test1_drift_after_training, test2_drift_during_training,
             test3_asymmetric_control, test4_sr_stability]

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
        'experiment': 'exp_01_spectral_confinement_in_learning',
        'variant': 'TinyCIMM-Ghost',
        'score': f'{n_passed}/{len(tests)}',
        'n_passed': n_passed,
        'n_tests': len(tests),
        'tests': results,
        'timestamp': datetime.now().isoformat(),
    }

    fname = RESULTS_DIR / f"exp_01_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {fname}")
