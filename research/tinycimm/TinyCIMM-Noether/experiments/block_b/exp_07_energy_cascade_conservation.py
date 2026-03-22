#!/usr/bin/env python3
"""
Experiment 07: Energy Cascade Conservation

Verify that PAC conservation layers correspond to turbulent cascade levels.
The D=3 NoetherNetwork should show conservation pairs that align with the
inertial -> intermediate -> dissipation scale hierarchy.

The Fibonacci PAC recursion V(k) = V(k+1) + V(k+2) maps onto the
Richardson cascade: large eddies transfer energy to smaller eddies in
a self-similar hierarchy. The golden-ratio decay V(k) ~ phi^(-k)
matches the geometric progression of eddy scales.

PASS criterion: Conservation pair activations correlate with cascade
scale (Pearson r > 0.6).
"""

import sys
import os
import json
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from TinyCIMM_Noether import NoetherNetwork, NoetherConfig
from pac_descent import compute_value
from fibonacci_topology import PHI, PHI_INV


def she_leveque_exponent(p):
    """She-Leveque scaling exponents."""
    return p / 9.0 + 2.0 * (1.0 - (2.0 / 3.0) ** (p / 3.0))


def generate_cascade_data(n_samples=300, seed=42):
    """
    Generate data representing turbulent energy cascade at multiple scales.

    Features encode the separation scale r and energy at that scale.
    Targets are the She-Leveque structure function values.

    Three cascade levels:
      - Inertial range:      r in [0.1, 1.0]   (large eddies)
      - Intermediate range:  r in [0.01, 0.1]   (medium eddies)
      - Dissipation range:   r in [0.001, 0.01]  (small eddies)
    """
    rng = np.random.RandomState(seed)

    # Three cascade ranges with labels
    ranges = [
        (0.1, 1.0, 0),      # inertial (label 0)
        (0.01, 0.1, 1),     # intermediate (label 1)
        (0.001, 0.01, 2),   # dissipation (label 2)
    ]

    n_per_range = n_samples // 3
    X_list, Y_list, labels = [], [], []

    for r_min, r_max, label in ranges:
        log_r = rng.uniform(np.log10(r_min), np.log10(r_max), n_per_range)
        r = 10.0 ** log_r

        # Structure functions at orders p=2,3,4
        zeta_2 = she_leveque_exponent(2)
        zeta_3 = she_leveque_exponent(3)
        zeta_4 = she_leveque_exponent(4)

        s2 = r ** zeta_2 + rng.normal(0, 0.02, n_per_range)
        s3 = r ** zeta_3 + rng.normal(0, 0.02, n_per_range)
        s4 = r ** zeta_4 + rng.normal(0, 0.02, n_per_range)

        features = np.column_stack([log_r, s2, s3, s4])
        # Target: cascade level as continuous value
        target = np.full((n_per_range, 1), float(label))

        X_list.append(features)
        Y_list.append(target)
        labels.extend([label] * n_per_range)

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    labels = np.array(labels)

    # Shuffle
    perm = rng.permutation(len(X))
    X, Y, labels = X[perm], Y[perm], labels[perm]

    # Normalise
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std
    Y = Y / 2.0  # Scale labels to [0, 1]

    return X, Y, labels


def run():
    print("=" * 60)
    print("Exp 07: Energy Cascade Conservation")
    print("=" * 60)

    X, Y, labels = generate_cascade_data(n_samples=300, seed=42)
    print(f"Data: {X.shape[0]} samples, {X.shape[1]}D input")
    print(f"  Cascade levels: {np.bincount(labels.astype(int))}")

    # Train NoetherNetwork
    config = NoetherConfig(
        depth=3,
        fibonacci_index=8,
        conservation_rate=0.12,
        direction_rate=0.05,
        epsilon=1e-3,
        default_epochs=500,
        seed=42,
    )
    net = NoetherNetwork(input_dim=X.shape[1], output_dim=1, config=config)
    print(f"NoetherNetwork: {net.param_count} params, D={config.depth}")

    print("Training...")
    history = net.fit(X, Y, verbose=True)

    # --- Analyse conservation pair activations per cascade level ---
    print("\n--- Conservation Pair Analysis ---")

    # For each cascade level, get layer values after forward pass
    level_values = {}  # level -> list of layer value vectors
    for level in [0, 1, 2]:
        mask = labels == level
        X_level = X[mask]
        _, states = net.engine.forward(X_level)
        values = [s.value for s in states]
        level_values[level] = values

    level_names = ['Inertial', 'Intermediate', 'Dissipation']

    print(f"\n  {'Level':<15} " + " ".join(f"V({k}):8" for k in range(len(level_values[0]))))
    for level in [0, 1, 2]:
        vals = level_values[level]
        val_str = " ".join(f"{v:8.4f}" for v in vals)
        print(f"  {level_names[level]:<15} {val_str}")

    # Compute correlation between cascade level and each layer's value
    # We expect deeper layers (higher k) to differentiate cascade scales
    all_values_by_layer = []
    for k in range(len(level_values[0])):
        per_sample_values = []
        for level in [0, 1, 2]:
            mask = labels == level
            X_level = X[mask]
            _, states = net.engine.forward(X_level)
            per_sample_values.append(states[k].value)
        all_values_by_layer.append(per_sample_values)

    # Correlation: cascade scale (0,1,2) vs layer activation values
    # Use per-sample analysis for proper Pearson correlation
    per_sample_layer_values = []
    for k in range(len(level_values[0])):
        vals = []
        for i in range(len(X)):
            x_i = X[i:i+1]
            _, states = net.engine.forward(x_i)
            vals.append(states[k].value)
        per_sample_layer_values.append(np.array(vals))

    correlations = []
    print("\n  Layer-Scale Correlations:")
    for k, vals in enumerate(per_sample_layer_values):
        r, p_val = pearsonr(labels, vals)
        correlations.append(abs(r))
        print(f"    V({k}) vs cascade_level: r={r:.4f}, p={p_val:.4e}")

    # The key metric: does at least one conservation-relevant layer
    # show significant correlation with cascade scale?
    max_corr = max(correlations)

    # Also check conservation pair violation structure differs by level
    print("\n  Conservation Violations by Level:")
    level_violations = {}
    for level in [0, 1, 2]:
        mask = labels == level
        X_level = X[mask]
        _, states = net.engine.forward(X_level)
        violations = net.engine.compute_violations()
        total_viol = sum(abs(v) for v in violations.values())
        level_violations[level] = total_viol
        print(f"    {level_names[level]:15s}: total_violation = {total_viol:.6f}")

    # Correlation between cascade level and violation magnitude
    viol_per_sample = []
    for i in range(len(X)):
        x_i = X[i:i+1]
        _, states = net.engine.forward(x_i)
        violations = net.engine.compute_violations()
        viol_per_sample.append(sum(abs(v) for v in violations.values()))
    viol_per_sample = np.array(viol_per_sample)
    r_viol, p_viol = pearsonr(labels, viol_per_sample)
    print(f"\n    Violation vs cascade_level: r={r_viol:.4f}, p={p_viol:.4e}")

    # Best correlation across all measures
    best_corr = max(max_corr, abs(r_viol))

    # --- Results ---
    PASS_THRESHOLD = 0.6
    passed = best_corr > PASS_THRESHOLD

    print("\n" + "=" * 60)
    print(f"CRITERION: Conservation activations correlate with cascade scale (r > {PASS_THRESHOLD})")
    print(f"  Best layer-scale correlation:    {max_corr:.4f}")
    print(f"  Violation-scale correlation:     {abs(r_viol):.4f}")
    print(f"  Best overall correlation:        {best_corr:.4f}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)

    results = {
        'experiment': 'exp_07_energy_cascade_conservation',
        'passed': bool(passed),
        'criterion': f'Pearson r > {PASS_THRESHOLD}',
        'best_correlation': float(best_corr),
        'layer_correlations': [float(c) for c in correlations],
        'violation_correlation': float(abs(r_viol)),
        'level_violations': {str(k): float(v) for k, v in level_violations.items()},
    }

    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'exp_07_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    run()
