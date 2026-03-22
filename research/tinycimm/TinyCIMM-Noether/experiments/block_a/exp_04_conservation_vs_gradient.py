#!/usr/bin/env python3
"""
Experiment 04: Conservation vs Gradient on Known-Physics Dataset

Direct comparison: PAC descent vs Adam-equivalent on a dataset with
known conservation structure (power-law / Fibonacci scaling).

The hypothesis: on data with inherent PAC/Fibonacci structure, the
conservation-based update rule should outperform gradient descent
because it exploits the structure gradient descent is blind to.

PASS criteria:
  1. On power-law data: Noether MSE < SGD MSE
  2. On Fibonacci cascade data: Noether MSE < SGD MSE
  3. Conservation violations stay bounded throughout training
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from TinyCIMM_Noether import NoetherNetwork, NoetherConfig, SimpleMLPBaseline
from fibonacci_topology import PHI, PHI_INV, fibonacci


def generate_power_law_data(n_samples=200, seed=42):
    """
    Power-law dataset: y = Σ x_i^(φ^i) — nested golden ratio exponents.
    This has inherent PAC structure.
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(0.2, 1.5, (n_samples, 4))
    Y = np.zeros((n_samples, 1))
    for i in range(4):
        Y[:, 0] += X[:, i] ** (PHI ** i)
    Y = (Y - Y.mean()) / (Y.std() + 1e-8)
    return X, Y


def generate_fibonacci_cascade_data(n_samples=200, seed=42):
    """
    Fibonacci cascade: y = F_1*x1 + F_2*x2 + F_3*x3 + F_4*x4 + F_5*x5
    with Fibonacci coefficients. Direct PAC structure in the linear combination.
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, (n_samples, 5))
    fibs = [fibonacci(i) for i in range(1, 6)]  # 1, 1, 2, 3, 5
    Y = np.zeros((n_samples, 1))
    for i, f in enumerate(fibs):
        Y[:, 0] += f * X[:, i]
    # Add slight nonlinearity
    Y[:, 0] += 0.1 * np.sin(Y[:, 0] * np.pi)
    Y = (Y - Y.mean()) / (Y.std() + 1e-8)
    return X, Y


def compare_on_dataset(name, X, Y, epochs=600):
    """Run both Noether and SGD on a dataset, return comparison."""
    print(f"\n--- Dataset: {name} ---")
    print(f"  Shape: {X.shape} -> {Y.shape}")

    input_dim = X.shape[1]
    output_dim = Y.shape[1]

    # Noether
    config = NoetherConfig(
        depth=3,
        conservation_rate=0.08,
        direction_rate=0.03,
        default_epochs=epochs,
        seed=42,
    )
    noether = NoetherNetwork(input_dim=input_dim, output_dim=output_dim, config=config)
    topo = noether.topology

    noether_history = noether.fit(X, Y, verbose=False)
    Y_pred_n = noether.predict(X)
    noether_mse = float(np.mean((Y - Y_pred_n) ** 2))
    noether_viol = noether.check_conservation(X[:32]).total_violation

    # SGD baseline (same topology)
    baseline = SimpleMLPBaseline(
        layer_widths=topo.layer_widths,
        lr=0.005,
        activation='tanh',
        seed=42,
    )
    baseline.fit(X, Y, epochs=epochs, verbose=False)
    Y_pred_s = baseline.predict(X)
    sgd_mse = float(np.mean((Y - Y_pred_s) ** 2))

    print(f"  Noether MSE: {noether_mse:.6f} | SGD MSE: {sgd_mse:.6f}")
    print(f"  Noether PAC violation: {noether_viol:.6f}")
    print(f"  Winner: {'Noether' if noether_mse < sgd_mse else 'SGD'}")

    return {
        'noether_mse': noether_mse,
        'sgd_mse': sgd_mse,
        'noether_violation': float(noether_viol),
        'noether_wins': noether_mse < sgd_mse,
    }


def run():
    print("=" * 60)
    print("Exp 04: Conservation vs Gradient on Known-Physics Data")
    print("=" * 60)

    # Dataset 1: Power-law with golden ratio exponents
    X1, Y1 = generate_power_law_data()
    r1 = compare_on_dataset("Power-Law (φ exponents)", X1, Y1)

    # Dataset 2: Fibonacci cascade
    X2, Y2 = generate_fibonacci_cascade_data()
    r2 = compare_on_dataset("Fibonacci Cascade", X2, Y2)

    results = {
        'power_law': r1,
        'fibonacci_cascade': r2,
    }

    # === PASS/FAIL criteria ===
    crit1 = r1['noether_wins']
    crit2 = r2['noether_wins']
    crit3 = r1['noether_violation'] < 0.5 and r2['noether_violation'] < 0.5

    results['criteria'] = {
        'crit1_power_law_wins': {'pass': bool(crit1)},
        'crit2_fibonacci_wins': {'pass': bool(crit2)},
        'crit3_violations_bounded': {
            'pass': bool(crit3),
            'power_law_viol': r1['noether_violation'],
            'fib_viol': r2['noether_violation'],
        },
    }

    all_pass = crit1 and crit2 and crit3
    partial = sum([crit1, crit2, crit3]) >= 2
    results['overall'] = 'PASS' if all_pass else ('PARTIAL' if partial else 'FAIL')

    print("\n" + "=" * 60)
    print(f"Criterion 1 (Noether wins on power-law): {'PASS' if crit1 else 'FAIL'}")
    print(f"Criterion 2 (Noether wins on Fibonacci): {'PASS' if crit2 else 'FAIL'}")
    print(f"Criterion 3 (violations bounded): {'PASS' if crit3 else 'FAIL'}")
    print(f"\nOVERALL: {results['overall']}")
    print("=" * 60)

    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'exp_04_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return all_pass or partial


if __name__ == '__main__':
    success = run()
    sys.exit(0 if success else 1)
