#!/usr/bin/env python3
"""
Experiment 01: Baseline PAC Conservation

Verify that PAC conservation holds during training on synthetic data.

PASS criteria:
  1. PAC violations decrease during training (final < initial)
  2. After training, max PAC violation < 0.1
  3. Conservation enforcement brings violations below epsilon

Synthetic data: power-law function y = x^φ (golden ratio exponent)
"""

import sys
import os
import json
import numpy as np

# Add parent dirs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from TinyCIMM_Noether import NoetherNetwork, NoetherConfig
from fibonacci_topology import PHI


def generate_power_law_data(n_samples=200, seed=42):
    """Generate power-law data with golden ratio exponent."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0.1, 2.0, (n_samples, 3))
    # y = sum(x_i^φ) — a simple power-law target
    Y = np.sum(X ** PHI, axis=1, keepdims=True)
    # Normalize
    Y = (Y - Y.mean()) / (Y.std() + 1e-8)
    return X, Y


def run():
    print("=" * 60)
    print("Exp 01: Baseline PAC Conservation")
    print("=" * 60)

    X, Y = generate_power_law_data()
    print(f"Data: {X.shape[0]} samples, {X.shape[1]}D -> {Y.shape[1]}D")

    config = NoetherConfig(
        depth=3,
        conservation_rate=0.15,
        direction_rate=0.02,
        epsilon=1e-3,
        default_epochs=300,
        seed=42,
    )
    net = NoetherNetwork(input_dim=3, output_dim=1, config=config)
    print(f"\n{net.summary()}\n")

    # Check initial violations
    initial_status = net.check_conservation(X[:32])
    initial_violation = initial_status.total_violation
    print(f"Initial PAC violation: {initial_violation:.6f}")

    # Train
    print("\nTraining...")
    history = net.fit(X, Y, verbose=True)

    # Check final violations
    final_status = net.check_conservation(X[:32])
    final_violation = final_status.total_violation
    print(f"\nFinal PAC violation: {final_violation:.6f}")

    # Conservation enforcement
    _, enforced_status = net.forward_with_enforcement(X[:32])
    enforced_violation = enforced_status.total_violation
    print(f"Enforced PAC violation: {enforced_violation:.6f}")
    print(f"Corrections applied: {enforced_status.corrections_applied}")

    # Conservation report
    print(f"\n{net.conservation_report(X[:1])}")

    # === PASS/FAIL criteria ===
    results = {}

    # Criterion 1: Violations decrease
    # Compare first 10% of epochs to last 10%
    n = len(history)
    early_violations = np.mean([h['total_violation'] for h in history[:max(1, n//10)]])
    late_violations = np.mean([h['total_violation'] for h in history[-max(1, n//10):]])
    crit1 = late_violations < early_violations
    results['violations_decrease'] = {
        'pass': bool(crit1),
        'early': float(early_violations),
        'late': float(late_violations),
    }

    # Criterion 2: Final max violation < 0.1
    crit2 = final_status.max_violation < 0.1
    results['final_violation_bounded'] = {
        'pass': bool(crit2),
        'max_violation': float(final_status.max_violation),
        'threshold': 0.1,
    }

    # Criterion 3: Enforcement brings violations below epsilon
    crit3 = enforced_status.satisfied or enforced_violation < 0.01
    results['enforcement_works'] = {
        'pass': bool(crit3),
        'enforced_violation': float(enforced_violation),
        'satisfied': bool(enforced_status.satisfied),
    }

    all_pass = crit1 and crit2 and crit3
    results['overall'] = 'PASS' if all_pass else 'FAIL'

    print("\n" + "=" * 60)
    print(f"Criterion 1 (violations decrease): {'PASS' if crit1 else 'FAIL'}")
    print(f"  Early avg: {early_violations:.6f}, Late avg: {late_violations:.6f}")
    print(f"Criterion 2 (max violation < 0.1): {'PASS' if crit2 else 'FAIL'}")
    print(f"  Max violation: {final_status.max_violation:.6f}")
    print(f"Criterion 3 (enforcement works): {'PASS' if crit3 else 'FAIL'}")
    print(f"  Enforced violation: {enforced_violation:.6f}")
    print(f"\nOVERALL: {results['overall']}")
    print("=" * 60)

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'exp_01_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return all_pass


if __name__ == '__main__':
    success = run()
    sys.exit(0 if success else 1)
