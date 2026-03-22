#!/usr/bin/env python3
"""
Experiment 06: She-Leveque Power-Law Fit

Generate synthetic She-Leveque turbulence data (structure functions
S_p(r) ~ r^zeta_p). Train NoetherNetwork to predict zeta_p from r.
Compare fit quality vs MLP baseline.

She-Leveque scaling exponents (1994):
    zeta_p = p/9 + 2 * (1 - (2/3)^(p/3))

This is the standard form capturing intermittency corrections beyond
Kolmogorov K41 (zeta_p = p/3).

PASS criterion: NoetherNetwork achieves R^2 > 0.85 on held-out
structure functions.
"""

import sys
import os
import json
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from TinyCIMM_Noether import NoetherNetwork, NoetherConfig, SimpleMLPBaseline


def she_leveque_exponent(p):
    """
    She-Leveque (1994) scaling exponents.

    zeta_p = p/9 + 2 * (1 - (2/3)^(p/3))

    For p=3: zeta_3 = 1 (exact, from 4/5 law).
    """
    return p / 9.0 + 2.0 * (1.0 - (2.0 / 3.0) ** (p / 3.0))


def generate_structure_functions(n_samples=500, seed=42):
    """
    Generate synthetic She-Leveque structure function data.

    S_p(r) = A_p * r^{zeta_p} + noise

    Features: (log(r), p) -> Target: log(S_p(r)) which should be linear
    in log(r) with slope zeta_p.

    We generate data for orders p = 1..8 at various separations r.
    """
    rng = np.random.RandomState(seed)

    orders = np.arange(1, 9, dtype=float)  # p = 1..8
    n_per_order = n_samples // len(orders)

    X_list, Y_list = [], []

    for p in orders:
        zeta_p = she_leveque_exponent(p)
        # Separation distances in the inertial range
        log_r = rng.uniform(-3, 0, n_per_order)  # log10(r) in [0.001, 1]
        r = 10.0 ** log_r

        # Structure function: S_p(r) = r^{zeta_p} with multiplicative noise
        noise = rng.normal(0, 0.05, n_per_order)
        log_S = zeta_p * log_r + noise

        # Features: (log_r, p/8_normalised, p^2_normalised, zeta_p_k41)
        p_norm = p / 8.0
        p2_norm = (p ** 2) / 64.0
        zeta_k41 = p / 3.0  # Kolmogorov prediction as feature
        features = np.column_stack([log_r, np.full(n_per_order, p_norm),
                                     np.full(n_per_order, p2_norm),
                                     np.full(n_per_order, zeta_k41)])
        X_list.append(features)
        Y_list.append(log_S.reshape(-1, 1))

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)

    # Shuffle
    perm = rng.permutation(len(X))
    X, Y = X[perm], Y[perm]

    # Normalise inputs
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std

    # Normalise targets
    Y_mean, Y_std = Y.mean(), Y.std() + 1e-8
    Y = (Y - Y_mean) / Y_std

    return X, Y, X_mean, X_std, Y_mean, Y_std


def r_squared(y_true, y_pred):
    """Coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-10)


def run():
    print("=" * 60)
    print("Exp 06: She-Leveque Power-Law Fit")
    print("=" * 60)

    # Generate data
    X, Y, X_mean, X_std, Y_mean, Y_std = generate_structure_functions(
        n_samples=800, seed=42)

    # Train/test split
    n_train = int(0.7 * len(X))
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_test, Y_test = X[n_train:], Y[n_train:]
    print(f"Data: {n_train} train, {len(X_test)} test, {X.shape[1]}D input")

    # --- NoetherNetwork ---
    config = NoetherConfig(
        depth=3,
        fibonacci_index=8,  # Larger network: widths ~[4,13,8,1] = 186 params
        conservation_rate=0.12,
        direction_rate=0.05,
        epsilon=1e-3,
        default_epochs=800,
        seed=42,
    )
    net = NoetherNetwork(input_dim=X.shape[1], output_dim=1, config=config)
    print(f"\nNoetherNetwork: {net.param_count} params, widths={net.topology.layer_widths}")

    print("Training NoetherNetwork...")
    history = net.fit(X_train, Y_train, verbose=True)
    Y_pred_noether = net.predict(X_test)
    r2_noether = r_squared(Y_test, Y_pred_noether)
    mse_noether = np.mean((Y_test - Y_pred_noether) ** 2)
    print(f"  NoetherNetwork: R^2 = {r2_noether:.4f}, MSE = {mse_noether:.6f}")

    # --- MLP Baseline ---
    widths = net.topology.layer_widths
    mlp = SimpleMLPBaseline(widths, lr=0.01, seed=42)
    print(f"\nMLP Baseline: widths={widths}")

    print("Training MLP...")
    mlp.fit(X_train, Y_train, epochs=800, verbose=True)
    Y_pred_mlp = mlp.predict(X_test)
    r2_mlp = r_squared(Y_test, Y_pred_mlp)
    mse_mlp = np.mean((Y_test - Y_pred_mlp) ** 2)
    print(f"  MLP Baseline: R^2 = {r2_mlp:.4f}, MSE = {mse_mlp:.6f}")

    # --- Verify She-Leveque exponents ---
    print("\n--- She-Leveque Exponents ---")
    print(f"  {'p':>3}  {'zeta_p (SLE)':>12}  {'zeta_p (K41)':>12}  {'difference':>10}")
    for p in range(1, 9):
        z_sle = she_leveque_exponent(p)
        z_k41 = p / 3.0
        print(f"  {p:3d}  {z_sle:12.6f}  {z_k41:12.6f}  {z_sle - z_k41:10.6f}")

    # --- Results ---
    PASS_THRESHOLD = 0.85
    passed = r2_noether > PASS_THRESHOLD

    print("\n" + "=" * 60)
    print(f"CRITERION: NoetherNetwork R^2 > {PASS_THRESHOLD}")
    print(f"  NoetherNetwork R^2: {r2_noether:.4f}")
    print(f"  MLP Baseline R^2:   {r2_mlp:.4f}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)

    results = {
        'experiment': 'exp_06_sle_power_law_fit',
        'passed': bool(passed),
        'criterion': f'R^2 > {PASS_THRESHOLD}',
        'noether_r2': float(r2_noether),
        'noether_mse': float(mse_noether),
        'mlp_r2': float(r2_mlp),
        'mlp_mse': float(mse_mlp),
        'n_train': n_train,
        'n_test': len(X_test),
    }

    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'exp_06_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    run()
