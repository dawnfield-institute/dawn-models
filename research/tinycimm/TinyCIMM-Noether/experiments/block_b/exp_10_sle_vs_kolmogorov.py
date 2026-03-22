#!/usr/bin/env python3
"""
Experiment 10: She-Leveque vs Kolmogorov K41

Compare She-Leveque (intermittent) vs Kolmogorov K41 (non-intermittent)
turbulence. The NoetherNetwork should distinguish them via conservation
structure differences.

K41 predicts zeta_p = p/3 (no intermittency corrections).
SLE predicts zeta_p = p/9 + 2*(1 - (2/3)^(p/3)).

These produce identical results at p=3 (both give zeta_3 = 1), but
diverge at higher orders. The conservation structure in a PAC network
should encode this difference — intermittent turbulence requires
different conservation pair ratios than non-intermittent.

PASS criterion: Conservation signatures are statistically distinguishable
(p < 0.05, Mann-Whitney U test on conservation violation distributions).
"""

import sys
import os
import json
import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from TinyCIMM_Noether import NoetherNetwork, NoetherConfig
from pac_descent import compute_value


def she_leveque_exponent(p):
    """She-Leveque scaling exponents."""
    return p / 9.0 + 2.0 * (1.0 - (2.0 / 3.0) ** (p / 3.0))


def kolmogorov_exponent(p):
    """Kolmogorov K41 scaling exponents (non-intermittent)."""
    return p / 3.0


def generate_turbulence_data(exponent_fn, n_samples=300, orders=None, seed=42):
    """
    Generate synthetic structure function data using given exponent model.

    Parameters:
        exponent_fn: Function p -> zeta_p
        n_samples: Number of samples
        orders: List of structure function orders
        seed: Random seed
    """
    rng = np.random.RandomState(seed)
    if orders is None:
        orders = [2, 3, 4, 5, 6, 7, 8]

    n_per_order = n_samples // len(orders)
    X_list, Y_list = [], []

    for p in orders:
        zeta_p = exponent_fn(p)
        log_r = rng.uniform(-3, 0, n_per_order)
        r = 10.0 ** log_r

        # Structure function
        S_p = r ** zeta_p + rng.normal(0, 0.03, n_per_order)

        features = np.column_stack([
            log_r,
            np.full(n_per_order, p / 8.0),
            np.log(np.abs(S_p) + 1e-10),
            np.full(n_per_order, zeta_p),
        ])

        # Target: the exponent itself
        targets = np.full((n_per_order, 1), zeta_p)

        X_list.append(features)
        Y_list.append(targets)

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    perm = rng.permutation(len(X))
    return X[perm], Y[perm]


def extract_conservation_signatures(net, X):
    """
    Extract conservation signature for each sample:
    - Per-layer values
    - Per-pair violations
    - Value ratios between layers
    """
    signatures = []
    for i in range(len(X)):
        x_i = X[i:i+1]
        _, states = net.engine.forward(x_i)
        violations = net.engine.compute_violations()

        values = [s.value for s in states]
        # Signature vector: [V(0), V(1), ..., V(D), delta_0, delta_1, ...,
        #                    ratio_01, ratio_12, ...]
        sig = list(values)
        sig.extend([abs(violations.get(k, 0.0))
                    for k in sorted(violations.keys())])
        # Value ratios
        for k in range(len(values) - 1):
            if values[k + 1] > 1e-10:
                sig.append(values[k] / values[k + 1])
            else:
                sig.append(0.0)
        signatures.append(sig)

    return np.array(signatures)


def run():
    print("=" * 60)
    print("Exp 10: She-Leveque vs Kolmogorov K41")
    print("=" * 60)

    # Show exponent differences
    print("\n  Exponent Comparison:")
    print(f"  {'p':>3}  {'SLE':>8}  {'K41':>8}  {'diff':>8}")
    for p in range(1, 9):
        z_sle = she_leveque_exponent(p)
        z_k41 = kolmogorov_exponent(p)
        print(f"  {p:3d}  {z_sle:8.4f}  {z_k41:8.4f}  {z_sle-z_k41:8.4f}")

    # Generate data for both models
    X_sle, Y_sle = generate_turbulence_data(
        she_leveque_exponent, n_samples=350, seed=42)
    X_k41, Y_k41 = generate_turbulence_data(
        kolmogorov_exponent, n_samples=350, seed=43)

    # Normalise jointly
    X_all = np.vstack([X_sle, X_k41])
    X_mean, X_std = X_all.mean(axis=0), X_all.std(axis=0) + 1e-8
    X_sle_norm = (X_sle - X_mean) / X_std
    X_k41_norm = (X_k41 - X_mean) / X_std

    Y_all = np.vstack([Y_sle, Y_k41])
    Y_mean, Y_std = Y_all.mean(), Y_all.std() + 1e-8
    Y_sle_norm = (Y_sle - Y_mean) / Y_std
    Y_k41_norm = (Y_k41 - Y_mean) / Y_std

    print(f"\nSLE data: {X_sle.shape[0]} samples")
    print(f"K41 data: {X_k41.shape[0]} samples")

    # Train NoetherNetwork on combined data with labels
    # Combine and add a label feature
    X_combined = np.vstack([X_sle_norm, X_k41_norm])
    Y_combined = np.vstack([Y_sle_norm, Y_k41_norm])

    config = NoetherConfig(
        depth=3,
        fibonacci_index=8,
        conservation_rate=0.12,
        direction_rate=0.05,
        epsilon=1e-3,
        default_epochs=500,
        seed=42,
    )
    net = NoetherNetwork(input_dim=X_combined.shape[1], output_dim=1,
                         config=config)
    print(f"\nNoetherNetwork: {net.param_count} params")

    # Shuffle combined data
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(X_combined))
    X_combined = X_combined[perm]
    Y_combined = Y_combined[perm]

    print("Training on combined SLE+K41 data...")
    history = net.fit(X_combined, Y_combined, verbose=True)

    # --- Extract conservation signatures for each model ---
    print("\n--- Conservation Signature Analysis ---")

    sigs_sle = extract_conservation_signatures(net, X_sle_norm)
    sigs_k41 = extract_conservation_signatures(net, X_k41_norm)

    print(f"  Signature dimension: {sigs_sle.shape[1]}")
    print(f"  SLE samples: {sigs_sle.shape[0]}")
    print(f"  K41 samples: {sigs_k41.shape[0]}")

    # --- Statistical test: are the signatures distinguishable? ---
    # Mann-Whitney U test on each signature component
    p_values = []
    u_stats = []
    sig_names = []

    n_layers = len(net.topology.layer_widths)
    n_pairs = len(net.topology.conservation_pairs)

    for j in range(sigs_sle.shape[1]):
        stat, pval = mannwhitneyu(sigs_sle[:, j], sigs_k41[:, j],
                                   alternative='two-sided')
        p_values.append(pval)
        u_stats.append(stat)

        if j < n_layers:
            sig_names.append(f"V({j})")
        elif j < n_layers + n_pairs:
            sig_names.append(f"delta_{j - n_layers}")
        else:
            sig_names.append(f"ratio_{j - n_layers - n_pairs}")

    print("\n  Per-Component Mann-Whitney U Tests:")
    print(f"  {'Component':<12} {'U-stat':>10} {'p-value':>12} {'Significant':>12}")
    for name, u, p in zip(sig_names, u_stats, p_values):
        sig_mark = "***" if p < 0.001 else ("**" if p < 0.01 else
                   ("*" if p < 0.05 else ""))
        print(f"  {name:<12} {u:10.0f} {p:12.4e} {sig_mark:>12}")

    # Overall: use Bonferroni-corrected minimum p-value
    min_p = min(p_values)
    n_tests = len(p_values)
    bonferroni_p = min(min_p * n_tests, 1.0)

    # Also test aggregate signature distance
    mean_sle = sigs_sle.mean(axis=0)
    mean_k41 = sigs_k41.mean(axis=0)
    l2_dist = np.sqrt(np.sum((mean_sle - mean_k41) ** 2))
    print(f"\n  Mean signature L2 distance: {l2_dist:.6f}")

    # Count individually significant components
    n_significant = sum(1 for p in p_values if p < 0.05)
    print(f"  Significant components (p<0.05): {n_significant}/{n_tests}")

    # --- Results ---
    PASS_THRESHOLD = 0.05
    passed = bonferroni_p < PASS_THRESHOLD

    print("\n" + "=" * 60)
    print(f"CRITERION: Conservation signatures statistically distinguishable "
          f"(p < {PASS_THRESHOLD})")
    print(f"  Min component p-value:      {min_p:.4e}")
    print(f"  Bonferroni corrected:        {bonferroni_p:.4e}")
    print(f"  Significant components:      {n_significant}/{n_tests}")
    print(f"  Mean signature distance:     {l2_dist:.6f}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)

    results = {
        'experiment': 'exp_10_sle_vs_kolmogorov',
        'passed': bool(passed),
        'criterion': f'Bonferroni p < {PASS_THRESHOLD}',
        'min_p_value': float(min_p),
        'bonferroni_p': float(bonferroni_p),
        'n_significant': n_significant,
        'n_tests': n_tests,
        'mean_signature_distance': float(l2_dist),
        'component_p_values': {name: float(p) for name, p in
                                zip(sig_names, p_values)},
    }

    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'exp_10_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    run()
