#!/usr/bin/env python3
"""
Experiment 17: Scale Sweep — Riemann Zero Prediction

Sweep PAC conservation descent vs backprop MLP across five Fibonacci
scale tiers on Riemann zero spacing prediction (Exp 11 task).

Exp 11 found Noether competitive but not dominant at tiny scale. The
hypothesis is that the arithmetic regularity of zero spacings is better
captured by conservation structure at larger scale.

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
    generate_riemann_data, r_squared,
    train_noether_at_scale, train_mlp_at_scale,
    build_matched_mlp_widths, get_tier_widths,
    count_mlp_params, save_results,
)
from fibonacci_topology import build_topology


def run():
    print("=" * 70)
    print("Exp 17: Scale Sweep — Riemann Zero Prediction")
    print("=" * 70)

    lookback = 5
    X, Y = generate_riemann_data(n_zeros=300, lookback=lookback, seed=42)
    split = int(0.8 * len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_test, Y_test = X[split:], Y[split:]
    input_dim = lookback
    output_dim = 1
    print(f"Data: {len(X_train)} train, {len(X_test)} test, {input_dim}D input\n")

    tier_results = []
    epochs = 400

    for tier in SCALE_TIERS:
        name = tier['name']
        fib_idx = tier['fib_index']

        noether_widths = get_tier_widths(fib_idx, depth=3,
                                         input_dim=input_dim,
                                         output_dim=output_dim)
        topo = build_topology(fib_idx, depth=3,
                              input_dim=input_dim, output_dim=output_dim)
        n_params_noether = topo.total_params

        mlp_widths = build_matched_mlp_widths(n_params_noether,
                                              input_dim, output_dim)
        n_params_mlp = count_mlp_params(mlp_widths)

        print(f"--- Tier {name} (n={fib_idx}) ---")
        print(f"  Noether: {noether_widths}, {n_params_noether} params")
        print(f"  MLP:     {mlp_widths}, {n_params_mlp} params")

        # Average over 3 seeds for robustness (Riemann data is noisy)
        seeds = [42, 137, 256]
        n_mses, m_mses = [], []
        n_elapsed_total, m_elapsed_total = 0.0, 0.0
        skipped_n = skipped_m = False

        for seed in seeds:
            # Noether
            try:
                net, _, elapsed_n = train_noether_at_scale(
                    fib_idx, input_dim, output_dim, X_train, Y_train,
                    epochs=epochs, seed=seed)
                Y_pred = net.predict(X_test)
                n_mses.append(float(np.mean((Y_test - Y_pred) ** 2)))
                n_elapsed_total += elapsed_n
            except Exception as e:
                print(f"    Noether seed={seed} failed: {e}")
                skipped_n = True

            # MLP
            try:
                mlp, _, elapsed_m = train_mlp_at_scale(
                    mlp_widths, X_train, Y_train, epochs=epochs, seed=seed)
                Y_pred = mlp.predict(X_test)
                m_mses.append(float(np.mean((Y_test - Y_pred) ** 2)))
                m_elapsed_total += elapsed_m
            except Exception as e:
                print(f"    MLP seed={seed} failed: {e}")
                skipped_m = True

        if n_elapsed_total > XL_TIMEOUT_SECONDS and name == 'XL':
            print(f"  Noether timed out at XL ({n_elapsed_total:.1f}s)")
            skipped_n = True

        mse_n = float(np.mean(n_mses)) if n_mses else float('inf')
        mse_m = float(np.mean(m_mses)) if m_mses else float('inf')
        r2_n = float(r_squared(Y_test, net.predict(X_test))) if n_mses else -1.0
        r2_m = float(r_squared(Y_test, mlp.predict(X_test))) if m_mses else -1.0

        noether_wins = mse_n < mse_m and not skipped_n and not skipped_m
        ratio = mse_n / mse_m if mse_m > 0 else float('inf')

        print(f"  Noether: MSE={mse_n:.6f} (mean of {len(n_mses)} seeds)")
        print(f"  MLP:     MSE={mse_m:.6f} (mean of {len(m_mses)} seeds)")
        print(f"  Ratio:   {ratio:.4f}  {'← Noether wins' if noether_wins else ''}")
        print()

        tier_results.append({
            'tier': name,
            'fib_index': fib_idx,
            'noether_params': n_params_noether,
            'mlp_params': n_params_mlp,
            'noether_mse': mse_n,
            'noether_r2': r2_n,
            'mlp_mse': mse_m,
            'mlp_r2': r2_m,
            'mse_ratio': ratio,
            'noether_wins': noether_wins,
            'noether_skipped': skipped_n,
            'mlp_skipped': skipped_m,
            'per_seed_noether': n_mses,
            'per_seed_mlp': m_mses,
        })

    # ── Verdict ──────────────────────────────────────────────────────────
    wins_at_m_or_above = [
        r for r in tier_results
        if r['fib_index'] >= 11 and r['noether_wins']
    ]
    passed = len(wins_at_m_or_above) > 0

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
        'experiment': 'exp_17_scale_sweep_riemann',
        'passed': bool(passed),
        'criterion': 'Noether MSE < MLP MSE at n >= 11',
        'crossover_tier': crossover_tier,
        'tiers': tier_results,
    }

    save_results(results, 'exp_17_results.json')
    return results


if __name__ == '__main__':
    results = run()
    sys.exit(0 if results['passed'] else 1)
