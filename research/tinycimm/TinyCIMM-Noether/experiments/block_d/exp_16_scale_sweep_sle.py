#!/usr/bin/env python3
"""
Experiment 16: Scale Sweep — She-Leveque Turbulence

Sweep PAC conservation descent (NoetherNetwork) vs backprop MLP across
five Fibonacci scale tiers on the She-Leveque turbulence task (Exp 06).

Hypothesis: At small scale (n=6), conservation enforcement is overhead —
the network wastes capacity maintaining structure. At larger scale (n≥11),
the conservation structure becomes load-bearing and PAC descent outperforms
backprop because it operates on the actual conservation law.

PASS criterion: Noether outperforms MLP at tier M or above (MSE_noether <
MSE_mlp at n ≥ 11).
"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scale_sweep_utils import (
    SCALE_TIERS, XL_TIMEOUT_SECONDS,
    generate_sle_data, r_squared,
    train_noether_at_scale, train_mlp_at_scale,
    build_matched_mlp_widths, get_tier_widths,
    count_mlp_params, save_results,
)
from fibonacci_topology import build_topology


def run():
    print("=" * 70)
    print("Exp 16: Scale Sweep — She-Leveque Turbulence")
    print("=" * 70)

    # Generate data
    X, Y = generate_sle_data(n_samples=800, seed=42)
    n_train = int(0.7 * len(X))
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_test, Y_test = X[n_train:], Y[n_train:]
    input_dim = X.shape[1]
    output_dim = 1
    print(f"Data: {n_train} train, {len(X_test)} test, {input_dim}D input\n")

    tier_results = []
    epochs = 400

    for tier in SCALE_TIERS:
        name = tier['name']
        fib_idx = tier['fib_index']

        # Get Noether topology
        noether_widths = get_tier_widths(fib_idx, depth=3,
                                         input_dim=input_dim,
                                         output_dim=output_dim)
        topo = build_topology(fib_idx, depth=3,
                              input_dim=input_dim, output_dim=output_dim)
        n_params_noether = topo.total_params

        # Build matched MLP
        mlp_widths = build_matched_mlp_widths(n_params_noether,
                                              input_dim, output_dim)
        n_params_mlp = count_mlp_params(mlp_widths)

        print(f"--- Tier {name} (n={fib_idx}) ---")
        print(f"  Noether widths: {noether_widths}, params: {n_params_noether}")
        print(f"  MLP widths:     {mlp_widths}, params: {n_params_mlp}")

        # Train Noether
        t0 = time.time()
        try:
            net, hist_n, elapsed_n = train_noether_at_scale(
                fib_idx, input_dim, output_dim, X_train, Y_train,
                epochs=epochs, seed=42)
            Y_pred_n = net.predict(X_test)
            mse_n = float(np.mean((Y_test - Y_pred_n) ** 2))
            r2_n = float(r_squared(Y_test, Y_pred_n))
            skipped_n = False
        except Exception as e:
            print(f"  Noether FAILED: {e}")
            mse_n, r2_n, elapsed_n = float('inf'), -1.0, 0.0
            skipped_n = True

        if elapsed_n > XL_TIMEOUT_SECONDS and name == 'XL':
            print(f"  Noether timed out ({elapsed_n:.1f}s > {XL_TIMEOUT_SECONDS}s)")
            skipped_n = True

        # Train MLP
        try:
            mlp, hist_m, elapsed_m = train_mlp_at_scale(
                mlp_widths, X_train, Y_train, epochs=epochs, seed=42)
            Y_pred_m = mlp.predict(X_test)
            mse_m = float(np.mean((Y_test - Y_pred_m) ** 2))
            r2_m = float(r_squared(Y_test, Y_pred_m))
            skipped_m = False
        except Exception as e:
            print(f"  MLP FAILED: {e}")
            mse_m, r2_m, elapsed_m = float('inf'), -1.0, 0.0
            skipped_m = True

        if elapsed_m > XL_TIMEOUT_SECONDS and name == 'XL':
            print(f"  MLP timed out ({elapsed_m:.1f}s > {XL_TIMEOUT_SECONDS}s)")
            skipped_m = True

        noether_wins = mse_n < mse_m and not skipped_n and not skipped_m
        ratio = mse_n / mse_m if mse_m > 0 else float('inf')

        print(f"  Noether: MSE={mse_n:.6f}, R²={r2_n:.4f} ({elapsed_n:.1f}s)")
        print(f"  MLP:     MSE={mse_m:.6f}, R²={r2_m:.4f} ({elapsed_m:.1f}s)")
        print(f"  Ratio (N/M): {ratio:.4f}  {'← Noether wins' if noether_wins else ''}")
        print()

        tier_results.append({
            'tier': name,
            'fib_index': fib_idx,
            'noether_widths': noether_widths,
            'noether_params': n_params_noether,
            'mlp_widths': mlp_widths,
            'mlp_params': n_params_mlp,
            'noether_mse': mse_n,
            'noether_r2': r2_n,
            'noether_time_s': elapsed_n,
            'noether_skipped': skipped_n,
            'mlp_mse': mse_m,
            'mlp_r2': r2_m,
            'mlp_time_s': elapsed_m,
            'mlp_skipped': skipped_m,
            'mse_ratio': ratio,
            'noether_wins': noether_wins,
        })

    # ── Verdict ──────────────────────────────────────────────────────────
    # PASS: Noether outperforms MLP at tier M (n≥11) or above
    wins_at_m_or_above = [
        r for r in tier_results
        if r['fib_index'] >= 11 and r['noether_wins']
    ]
    passed = len(wins_at_m_or_above) > 0

    # Find first crossover tier (first tier where Noether wins)
    crossover_tier = None
    for r in tier_results:
        if r['noether_wins']:
            crossover_tier = r['tier']
            break

    print("=" * 70)
    print(f"CRITERION: Noether MSE < MLP MSE at tier M (n≥11) or above")
    print(f"  Wins at M+: {[r['tier'] for r in wins_at_m_or_above]}")
    print(f"  First crossover: {crossover_tier or 'not found in range'}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

    results = {
        'experiment': 'exp_16_scale_sweep_sle',
        'passed': bool(passed),
        'criterion': 'Noether MSE < MLP MSE at n >= 11',
        'crossover_tier': crossover_tier,
        'tiers': tier_results,
    }

    save_results(results, 'exp_16_results.json')
    return results


if __name__ == '__main__':
    results = run()
    sys.exit(0 if results['passed'] else 1)
