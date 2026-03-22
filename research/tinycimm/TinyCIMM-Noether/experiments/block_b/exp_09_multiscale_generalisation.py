#!/usr/bin/env python3
"""
Experiment 09: Multiscale Generalisation

Train at one Reynolds number range, test at another. The claim is that
PAC-conserved representations generalise across Re better than
gradient-trained MLPs because the conserved structure is scale-invariant.

Reynolds number affects turbulence through:
  - Inertial range extent: L_inertial ~ Re^(3/4)
  - Dissipation scale: eta ~ Re^(-3/4)
  - Energy spectrum: E(k) ~ k^{-5/3} (universal in inertial range)

The She-Leveque exponents zeta_p are Re-independent (universal),
but the scales at which they apply shift with Re.

PASS criterion: NoetherNetwork Re-generalisation gap < 50% of MLP gap.

Generalisation gap = |MSE_test_Re - MSE_train_Re| / MSE_train_Re
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from TinyCIMM_Noether import NoetherNetwork, NoetherConfig, SimpleMLPBaseline


def she_leveque_exponent(p):
    """She-Leveque scaling exponents."""
    return p / 9.0 + 2.0 * (1.0 - (2.0 / 3.0) ** (p / 3.0))


def generate_turbulence_at_re(Re, n_samples=200, orders=None, seed=42):
    """
    Generate synthetic turbulence structure function data at a given
    Reynolds number.

    Reynolds number affects:
      - The extent of the inertial range
      - The dissipation cutoff scale
      - The amplitude of structure functions (but NOT the exponents)

    The She-Leveque exponents zeta_p are universal — they don't depend
    on Re. What changes is:
      - Kolmogorov scale: eta ~ Re^{-3/4}
      - Integral scale: L ~ 1
      - Inertial range: eta < r < L
    """
    rng = np.random.RandomState(seed)
    if orders is None:
        orders = [2, 3, 4, 5, 6]

    # Kolmogorov scale depends on Re
    eta = Re ** (-3.0 / 4.0)
    L = 1.0  # Integral scale

    n_per_order = n_samples // len(orders)
    X_list, Y_list = [], []

    for p in orders:
        zeta_p = she_leveque_exponent(p)

        # Sample separations in the inertial range for this Re
        log_r_min = np.log10(eta * 10)  # Above dissipation
        log_r_max = np.log10(L * 0.5)   # Below integral scale
        if log_r_min >= log_r_max:
            log_r_min = log_r_max - 2  # Fallback

        log_r = rng.uniform(log_r_min, log_r_max, n_per_order)
        r = 10.0 ** log_r

        # Structure function: S_p(r) = C_p * (r/L)^{zeta_p}
        # Amplitude C_p depends on Re: C_p ~ Re^{zeta_p/2}
        C_p = Re ** (zeta_p / 4.0)
        S_p = C_p * (r / L) ** zeta_p
        noise = rng.normal(0, 0.05 * np.abs(S_p).mean(), n_per_order)
        S_p += noise

        # Features: normalised log_r, normalised p, log|S_p|, Re_normalised
        features = np.column_stack([
            (log_r - log_r_min) / (log_r_max - log_r_min + 1e-10),
            np.full(n_per_order, p / 6.0),
            np.log(np.abs(S_p) + 1e-10),
            np.full(n_per_order, np.log10(Re) / 6.0),  # Normalise Re
        ])

        # Target: zeta_p (the scaling exponent itself)
        targets = np.full((n_per_order, 1), zeta_p)

        X_list.append(features)
        Y_list.append(targets)

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)

    perm = rng.permutation(len(X))
    return X[perm], Y[perm]


def run():
    print("=" * 60)
    print("Exp 09: Multiscale Generalisation")
    print("=" * 60)

    # Train on moderate Re, test on high Re
    Re_train = 1e4   # Moderate turbulence
    Re_test = 1e6    # Fully developed turbulence

    print(f"Training Re = {Re_train:.0e}")
    print(f"Testing  Re = {Re_test:.0e}")

    X_train, Y_train = generate_turbulence_at_re(
        Re_train, n_samples=500, seed=42)
    X_test, Y_test = generate_turbulence_at_re(
        Re_test, n_samples=200, seed=123)

    # Normalise jointly
    X_all = np.vstack([X_train, X_test])
    X_mean, X_std = X_all.mean(axis=0), X_all.std(axis=0) + 1e-8
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    Y_mean, Y_std = Y_train.mean(), Y_train.std() + 1e-8
    Y_train_norm = (Y_train - Y_mean) / Y_std
    Y_test_norm = (Y_test - Y_mean) / Y_std

    print(f"Data: {len(X_train)} train @ Re={Re_train:.0e}, "
          f"{len(X_test)} test @ Re={Re_test:.0e}")

    # --- NoetherNetwork ---
    config = NoetherConfig(
        depth=3,
        fibonacci_index=8,
        conservation_rate=0.12,
        direction_rate=0.05,
        epsilon=1e-3,
        default_epochs=600,
        seed=42,
    )
    net = NoetherNetwork(input_dim=X_train.shape[1], output_dim=1, config=config)
    widths = net.topology.layer_widths

    print(f"\nNoetherNetwork: {net.param_count} params, widths={widths}")
    print("Training NoetherNetwork...")
    history = net.fit(X_train, Y_train_norm, verbose=True)

    Y_pred_train_n = net.predict(X_train)
    Y_pred_test_n = net.predict(X_test)

    mse_train_n = np.mean((Y_train_norm - Y_pred_train_n) ** 2)
    mse_test_n = np.mean((Y_test_norm - Y_pred_test_n) ** 2)
    gap_n = abs(mse_test_n - mse_train_n) / (mse_train_n + 1e-10)

    print(f"  NoetherNetwork: MSE_train={mse_train_n:.6f}, "
          f"MSE_test={mse_test_n:.6f}, gap={gap_n:.4f}")

    # --- MLP Baseline ---
    mlp = SimpleMLPBaseline(widths, lr=0.01, seed=42)

    print(f"\nMLP Baseline: widths={widths}")
    print("Training MLP...")
    mlp.fit(X_train, Y_train_norm, epochs=600, verbose=True)

    Y_pred_train_m = mlp.predict(X_train)
    Y_pred_test_m = mlp.predict(X_test)

    mse_train_m = np.mean((Y_train_norm - Y_pred_train_m) ** 2)
    mse_test_m = np.mean((Y_test_norm - Y_pred_test_m) ** 2)
    gap_m = abs(mse_test_m - mse_train_m) / (mse_train_m + 1e-10)

    print(f"  MLP: MSE_train={mse_train_m:.6f}, "
          f"MSE_test={mse_test_m:.6f}, gap={gap_m:.4f}")

    # --- Results ---
    # PASS: NoetherNetwork gap < 50% of MLP gap
    ratio = gap_n / (gap_m + 1e-10)
    PASS_THRESHOLD = 0.50
    passed = ratio < PASS_THRESHOLD

    print("\n" + "=" * 60)
    print(f"CRITERION: NoetherNetwork gap < 50% of MLP gap")
    print(f"  NoetherNetwork generalisation gap: {gap_n:.4f}")
    print(f"  MLP generalisation gap:            {gap_m:.4f}")
    print(f"  Ratio (Noether/MLP):               {ratio:.4f}")
    print(f"  Threshold:                         {PASS_THRESHOLD}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)

    results = {
        'experiment': 'exp_09_multiscale_generalisation',
        'passed': bool(passed),
        'criterion': f'gap_ratio < {PASS_THRESHOLD}',
        'noether_mse_train': float(mse_train_n),
        'noether_mse_test': float(mse_test_n),
        'noether_gap': float(gap_n),
        'mlp_mse_train': float(mse_train_m),
        'mlp_mse_test': float(mse_test_m),
        'mlp_gap': float(gap_m),
        'gap_ratio': float(ratio),
        'Re_train': Re_train,
        'Re_test': Re_test,
    }

    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'exp_09_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    run()
