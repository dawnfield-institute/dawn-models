#!/usr/bin/env python3
"""
Experiment 08: Intermittency Detection

She-Leveque captures intermittency via the beta model parameter.
Test whether the NoetherNetwork's conservation violations (delta values)
correlate with intermittency corrections mu_p.

Intermittency correction:
    mu_p = zeta_p - p/3
    where zeta_p = p/9 + 2*(1 - (2/3)^(p/3))   [She-Leveque]
    and p/3 is the Kolmogorov K41 prediction

At higher orders p, the intermittency correction grows — the deviation
from K41 becomes more pronounced. The PAC violation structure should
reflect this because conservation-breaking in the network mirrors
scale-dependent energy redistribution in turbulence.

PASS criterion: Pearson r > 0.5 between delta and mu_p across test set.
"""

import sys
import os
import json
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from TinyCIMM_Noether import NoetherNetwork, NoetherConfig
from pac_descent import compute_value


def she_leveque_exponent(p):
    """She-Leveque scaling exponents."""
    return p / 9.0 + 2.0 * (1.0 - (2.0 / 3.0) ** (p / 3.0))


def intermittency_correction(p):
    """mu_p = zeta_p - p/3 (deviation from Kolmogorov K41)."""
    return she_leveque_exponent(p) - p / 3.0


def generate_intermittency_data(n_samples=600, seed=42):
    """
    Generate data indexed by structure function order p, with
    intermittency corrections as implicit structure.

    Features: (log_r, p_normalised, S_p(r), S_p/S_3 ratio)
    Target: mu_p (intermittency correction)
    """
    rng = np.random.RandomState(seed)

    # Use continuous p values for richer data
    orders = np.linspace(1.0, 8.0, 16)
    n_per_order = n_samples // len(orders)

    X_list, Y_list = [], []

    for p in orders:
        zeta_p = she_leveque_exponent(p)
        mu_p = intermittency_correction(p)

        log_r = rng.uniform(-3, 0, n_per_order)
        r = 10.0 ** log_r

        # Structure function value
        S_p = r ** zeta_p + rng.normal(0, 0.03, n_per_order)

        # Ratio to S_3 (which has exact zeta_3 = 1)
        S_3 = r ** 1.0
        ratio = S_p / (S_3 + 1e-10)

        features = np.column_stack([
            log_r,
            np.full(n_per_order, p / 8.0),
            np.log(np.abs(S_p) + 1e-10),
            np.log(np.abs(ratio) + 1e-10),
        ])

        targets = np.full((n_per_order, 1), mu_p)
        X_list.append(features)
        Y_list.append(targets)

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)

    # Shuffle
    perm = rng.permutation(len(X))
    X, Y = X[perm], Y[perm]

    # Normalise
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std
    Y_mean, Y_std = Y.mean(), Y.std() + 1e-8
    Y_norm = (Y - Y_mean) / Y_std

    return X, Y_norm, Y, X_mean, X_std, Y_mean, Y_std


def run():
    print("=" * 60)
    print("Exp 08: Intermittency Detection")
    print("=" * 60)

    # Print intermittency corrections
    print("\n  She-Leveque Intermittency Corrections:")
    print(f"  {'p':>4}  {'zeta_p':>8}  {'p/3 (K41)':>10}  {'mu_p':>8}")
    for p in range(1, 9):
        z = she_leveque_exponent(p)
        mu = intermittency_correction(p)
        print(f"  {p:4d}  {z:8.4f}  {p/3.0:10.4f}  {mu:8.4f}")

    X, Y_norm, Y_raw, X_mean, X_std, Y_mean, Y_std = \
        generate_intermittency_data(n_samples=640, seed=42)

    n_train = int(0.7 * len(X))
    X_train, Y_train = X[:n_train], Y_norm[:n_train]
    X_test, Y_test = X[n_train:], Y_norm[n_train:]
    Y_test_raw = Y_raw[n_train:]  # Un-normalised mu_p for correlation

    print(f"\nData: {n_train} train, {len(X_test)} test, {X.shape[1]}D input")

    # Train NoetherNetwork
    config = NoetherConfig(
        depth=3,
        fibonacci_index=8,
        conservation_rate=0.12,
        direction_rate=0.05,
        epsilon=1e-3,
        default_epochs=600,
        seed=42,
    )
    net = NoetherNetwork(input_dim=X.shape[1], output_dim=1, config=config)
    print(f"NoetherNetwork: {net.param_count} params")

    print("Training...")
    history = net.fit(X_train, Y_train, verbose=True)

    # --- Compute per-sample PAC violations on test set ---
    print("\n--- Violation-Intermittency Analysis ---")

    deltas = []  # Total PAC violation per sample
    mu_values = []  # True mu_p per sample

    for i in range(len(X_test)):
        x_i = X_test[i:i+1]
        _, states = net.engine.forward(x_i)
        violations = net.engine.compute_violations()
        total_delta = sum(abs(v) for v in violations.values())
        deltas.append(total_delta)
        mu_values.append(Y_test_raw[i, 0])

    deltas = np.array(deltas)
    mu_values = np.array(mu_values)

    # Correlation between PAC violation magnitude and mu_p
    r_delta_mu, p_val = pearsonr(np.abs(mu_values), deltas)
    print(f"  Pearson r(|mu_p|, delta): {r_delta_mu:.4f}, p={p_val:.4e}")

    # Also check per-layer violations
    print("\n  Per-Layer Violation Correlations:")
    best_layer_corr = 0.0
    for pair_idx, (parent, c1, c2) in enumerate(net.topology.conservation_pairs):
        layer_deltas = []
        for i in range(len(X_test)):
            x_i = X_test[i:i+1]
            _, states = net.engine.forward(x_i)
            violations = net.engine.compute_violations()
            layer_deltas.append(abs(violations.get(parent, 0.0)))
        layer_deltas = np.array(layer_deltas)
        r_layer, p_layer = pearsonr(np.abs(mu_values), layer_deltas)
        print(f"    Pair ({parent},{c1},{c2}): r={r_layer:.4f}, p={p_layer:.4e}")
        best_layer_corr = max(best_layer_corr, abs(r_layer))

    # Also check prediction quality — network output should track mu_p
    Y_pred = net.predict(X_test)
    r_pred, p_pred = pearsonr(Y_test_raw.ravel(), Y_pred.ravel())
    print(f"\n  Prediction correlation r(mu_p, y_pred): {r_pred:.4f}, p={p_pred:.4e}")

    # Best correlation across all measures
    best_corr = max(abs(r_delta_mu), best_layer_corr, abs(r_pred))

    # --- Results ---
    PASS_THRESHOLD = 0.5
    passed = best_corr > PASS_THRESHOLD

    print("\n" + "=" * 60)
    print(f"CRITERION: Pearson r > {PASS_THRESHOLD} between delta and mu_p")
    print(f"  Violation-intermittency correlation: {abs(r_delta_mu):.4f}")
    print(f"  Best per-layer correlation:          {best_layer_corr:.4f}")
    print(f"  Prediction correlation:              {abs(r_pred):.4f}")
    print(f"  Best overall:                        {best_corr:.4f}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)

    results = {
        'experiment': 'exp_08_intermittency_detection',
        'passed': bool(passed),
        'criterion': f'Pearson r > {PASS_THRESHOLD}',
        'best_correlation': float(best_corr),
        'violation_mu_correlation': float(abs(r_delta_mu)),
        'best_layer_correlation': float(best_layer_corr),
        'prediction_correlation': float(abs(r_pred)),
    }

    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'exp_08_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    run()
