#!/usr/bin/env python3
"""
Experiment 19: Sample Efficiency at Scale M

At scale M (n=11, ~12k params), sweep training set size and compare
how many samples Noether vs MLP need to reach a given MSE threshold.

Hypothesis: Conservation structure provides an inductive bias that reduces
sample requirements. PAC descent should reach backprop's final MSE with
fewer training examples because conservation constrains the hypothesis space.

PASS criterion: Noether reaches backprop's final MSE (at 10k samples)
using ≤ 50% of the training data.
"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scale_sweep_utils import (
    generate_sle_data, r_squared,
    train_noether_at_scale, train_mlp_at_scale,
    build_matched_mlp_widths, get_tier_widths,
    count_mlp_params, save_results,
)
from fibonacci_topology import build_topology


SCALE_M_FIB_INDEX = 11
SAMPLE_SIZES = [100, 500, 1000, 5000, 10000]


def run():
    print("=" * 70)
    print("Exp 19: Sample Efficiency at Scale M (n=11)")
    print("=" * 70)

    # Generate a large pool of SLE data
    X_pool, Y_pool = generate_sle_data(n_samples=12000, seed=42)
    # Fixed test set (last 2000)
    X_test = X_pool[-2000:]
    Y_test = Y_pool[-2000:]
    X_pool = X_pool[:-2000]
    Y_pool = Y_pool[:-2000]

    input_dim = X_pool.shape[1]
    output_dim = 1

    topo = build_topology(SCALE_M_FIB_INDEX, depth=3,
                          input_dim=input_dim, output_dim=output_dim)
    n_params = topo.total_params
    noether_widths = get_tier_widths(SCALE_M_FIB_INDEX, depth=3,
                                     input_dim=input_dim, output_dim=output_dim)
    mlp_widths = build_matched_mlp_widths(n_params, input_dim, output_dim)

    print(f"Noether: {noether_widths}, {n_params} params")
    print(f"MLP:     {mlp_widths}, {count_mlp_params(mlp_widths)} params")
    print(f"Test set: {len(X_test)} samples\n")

    epochs = 400
    sweep_results = []

    for n_samples in SAMPLE_SIZES:
        X_train = X_pool[:n_samples]
        Y_train = Y_pool[:n_samples]

        print(f"--- N_train = {n_samples} ---")

        # Noether
        try:
            net, _, elapsed_n = train_noether_at_scale(
                SCALE_M_FIB_INDEX, input_dim, output_dim,
                X_train, Y_train, epochs=epochs, seed=42)
            Y_pred_n = net.predict(X_test)
            mse_n = float(np.mean((Y_test - Y_pred_n) ** 2))
            r2_n = float(r_squared(Y_test, Y_pred_n))
        except Exception as e:
            print(f"  Noether failed: {e}")
            mse_n, r2_n, elapsed_n = float('inf'), -1.0, 0.0

        # MLP
        try:
            mlp, _, elapsed_m = train_mlp_at_scale(
                mlp_widths, X_train, Y_train, epochs=epochs, seed=42)
            Y_pred_m = mlp.predict(X_test)
            mse_m = float(np.mean((Y_test - Y_pred_m) ** 2))
            r2_m = float(r_squared(Y_test, Y_pred_m))
        except Exception as e:
            print(f"  MLP failed: {e}")
            mse_m, r2_m, elapsed_m = float('inf'), -1.0, 0.0

        ratio = mse_n / mse_m if mse_m > 0 else float('inf')
        print(f"  Noether: MSE={mse_n:.6f}, R²={r2_n:.4f} ({elapsed_n:.1f}s)")
        print(f"  MLP:     MSE={mse_m:.6f}, R²={r2_m:.4f} ({elapsed_m:.1f}s)")
        print(f"  Ratio:   {ratio:.4f}")
        print()

        sweep_results.append({
            'n_samples': n_samples,
            'noether_mse': mse_n,
            'noether_r2': r2_n,
            'noether_time_s': elapsed_n,
            'mlp_mse': mse_m,
            'mlp_r2': r2_m,
            'mlp_time_s': elapsed_m,
            'mse_ratio': ratio,
        })

    # ── Find sample efficiency crossover ─────────────────────────────────
    # MLP's final MSE at max samples
    mlp_final_mse = sweep_results[-1]['mlp_mse']
    print(f"MLP final MSE (at {SAMPLE_SIZES[-1]} samples): {mlp_final_mse:.6f}")

    # Find smallest N where Noether MSE ≤ mlp_final_mse
    noether_match_n = None
    for r in sweep_results:
        if r['noether_mse'] <= mlp_final_mse:
            noether_match_n = r['n_samples']
            break

    if noether_match_n is not None:
        efficiency_ratio = noether_match_n / SAMPLE_SIZES[-1]
        print(f"Noether matches at N={noether_match_n} "
              f"({efficiency_ratio * 100:.0f}% of max)")
    else:
        efficiency_ratio = None
        # Check if Noether at max samples is close
        noether_final_mse = sweep_results[-1]['noether_mse']
        print(f"Noether did not match MLP final MSE")
        print(f"  Noether final MSE: {noether_final_mse:.6f}")
        print(f"  MLP final MSE:     {mlp_final_mse:.6f}")

    # PASS: Noether matches MLP's final MSE with ≤ 50% data
    passed = (noether_match_n is not None and
              efficiency_ratio is not None and
              efficiency_ratio <= 0.50)

    print("\n" + "=" * 70)
    print("SAMPLE EFFICIENCY SUMMARY")
    print("-" * 70)
    print(f"{'N_train':>8} {'Noether MSE':>12} {'MLP MSE':>12} {'Ratio':>8}")
    for r in sweep_results:
        flag = " ★" if r['noether_mse'] <= mlp_final_mse else ""
        print(f"{r['n_samples']:>8} {r['noether_mse']:>12.6f} "
              f"{r['mlp_mse']:>12.6f} {r['mse_ratio']:>8.4f}{flag}")
    print("-" * 70)
    print(f"MLP target MSE: {mlp_final_mse:.6f} (at N={SAMPLE_SIZES[-1]})")
    print(f"Noether matches at: N={noether_match_n or '?'}")
    print(f"Efficiency ratio: {efficiency_ratio * 100:.0f}%" if efficiency_ratio else
          "Efficiency ratio: N/A")
    print(f"CRITERION: Noether reaches MLP final MSE with ≤ 50% data")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

    results = {
        'experiment': 'exp_19_sample_efficiency',
        'passed': bool(passed),
        'criterion': 'Noether matches MLP final MSE with <= 50% data',
        'fib_index': SCALE_M_FIB_INDEX,
        'mlp_final_mse': mlp_final_mse,
        'noether_match_n': noether_match_n,
        'efficiency_ratio': float(efficiency_ratio) if efficiency_ratio else None,
        'max_samples': SAMPLE_SIZES[-1],
        'sweep': sweep_results,
    }

    save_results(results, 'exp_19_results.json')
    return results


if __name__ == '__main__':
    results = run()
    sys.exit(0 if results['passed'] else 1)
