#!/usr/bin/env python3
"""
Experiment 18: Formal Crossover Characterisation

At each scale tier, compute:
  (a) PAC conservation quality — mean |δ| across conservation pairs
  (b) Generalisation gap — |MSE_test - MSE_train| / MSE_train
  (c) MSE ratio — Noether_MSE / MLP_MSE

Find scale N* where the MSE ratio crosses 1.0 (Noether first beats MLP).

PASS criterion: N* exists within the tested range (XS to XL).
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
from pac_descent import compute_value, compute_pac_violations


def run():
    print("=" * 70)
    print("Exp 18: Formal Crossover Characterisation")
    print("=" * 70)

    # Use SLE data as the canonical task
    X, Y = generate_sle_data(n_samples=800, seed=42)
    n_train = int(0.7 * len(X))
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_test, Y_test = X[n_train:], Y[n_train:]
    input_dim = X.shape[1]
    output_dim = 1
    print(f"Data: {n_train} train, {len(X_test)} test\n")

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
        n_params = topo.total_params

        mlp_widths = build_matched_mlp_widths(n_params, input_dim, output_dim)

        print(f"--- Tier {name} (n={fib_idx}, ~{n_params} params) ---")

        skipped = False

        # ── Train Noether ────────────────────────────────────────────────
        try:
            net, hist_n, elapsed_n = train_noether_at_scale(
                fib_idx, input_dim, output_dim, X_train, Y_train,
                epochs=epochs, seed=42)

            Y_pred_train_n = net.predict(X_train)
            Y_pred_test_n = net.predict(X_test)
            mse_train_n = float(np.mean((Y_train - Y_pred_train_n) ** 2))
            mse_test_n = float(np.mean((Y_test - Y_pred_test_n) ** 2))
            gap_n = abs(mse_test_n - mse_train_n) / (mse_train_n + 1e-10)

            # (a) PAC conservation quality
            _, states = net.engine.forward(X_test[:100])
            values = [s.value for s in states]
            violations = compute_pac_violations(
                values, topo.conservation_pairs)
            mean_delta = float(np.mean([abs(v) for v in violations.values()]))

        except Exception as e:
            print(f"  Noether FAILED: {e}")
            mse_train_n = mse_test_n = float('inf')
            gap_n = float('inf')
            mean_delta = float('inf')
            elapsed_n = 0.0
            skipped = True

        if elapsed_n > XL_TIMEOUT_SECONDS and name == 'XL':
            print(f"  Skipping XL — timeout ({elapsed_n:.1f}s)")
            skipped = True

        # ── Train MLP ────────────────────────────────────────────────────
        try:
            mlp, hist_m, elapsed_m = train_mlp_at_scale(
                mlp_widths, X_train, Y_train, epochs=epochs, seed=42)

            Y_pred_train_m = mlp.predict(X_train)
            Y_pred_test_m = mlp.predict(X_test)
            mse_train_m = float(np.mean((Y_train - Y_pred_train_m) ** 2))
            mse_test_m = float(np.mean((Y_test - Y_pred_test_m) ** 2))
            gap_m = abs(mse_test_m - mse_train_m) / (mse_train_m + 1e-10)

        except Exception as e:
            print(f"  MLP FAILED: {e}")
            mse_train_m = mse_test_m = float('inf')
            gap_m = float('inf')
            elapsed_m = 0.0
            skipped = True

        # ── Metrics ──────────────────────────────────────────────────────
        mse_ratio = mse_test_n / mse_test_m if mse_test_m > 0 else float('inf')
        gap_ratio = gap_n / (gap_m + 1e-10)

        print(f"  PAC conservation δ:  {mean_delta:.6f}")
        print(f"  Noether: train={mse_train_n:.6f}, test={mse_test_n:.6f}, gap={gap_n:.4f}")
        print(f"  MLP:     train={mse_train_m:.6f}, test={mse_test_m:.6f}, gap={gap_m:.4f}")
        print(f"  MSE ratio (N/M): {mse_ratio:.4f}")
        print(f"  Gap ratio (N/M): {gap_ratio:.4f}")
        print()

        tier_results.append({
            'tier': name,
            'fib_index': fib_idx,
            'n_params': n_params,
            'mean_pac_delta': mean_delta,
            'noether_mse_train': mse_train_n,
            'noether_mse_test': mse_test_n,
            'noether_gap': gap_n,
            'mlp_mse_train': mse_train_m,
            'mlp_mse_test': mse_test_m,
            'mlp_gap': gap_m,
            'mse_ratio': mse_ratio,
            'gap_ratio': gap_ratio,
            'skipped': skipped,
        })

    # ── Find crossover N* ────────────────────────────────────────────────
    # N* = first scale where mse_ratio < 1.0 (Noether beats MLP)
    n_star = None
    n_star_tier = None
    for r in tier_results:
        if not r['skipped'] and r['mse_ratio'] < 1.0:
            n_star = r['fib_index']
            n_star_tier = r['tier']
            break

    # If not found, estimate via linear interpolation of log(ratio)
    n_star_interpolated = None
    if n_star is None:
        valid = [r for r in tier_results if not r['skipped']]
        if len(valid) >= 2:
            fibs = [r['fib_index'] for r in valid]
            log_ratios = [np.log(r['mse_ratio']) for r in valid
                          if r['mse_ratio'] > 0 and r['mse_ratio'] != float('inf')]
            if len(log_ratios) >= 2:
                # Linear fit: log(ratio) = a * n + b; solve for log(ratio) = 0
                coeffs = np.polyfit(fibs[:len(log_ratios)], log_ratios, 1)
                if coeffs[0] != 0:
                    n_star_interpolated = -coeffs[1] / coeffs[0]
                    print(f"Interpolated N* ≈ {n_star_interpolated:.1f}")

    passed = n_star is not None

    print("=" * 70)
    print("CROSSOVER ANALYSIS SUMMARY")
    print("-" * 70)
    print(f"{'Tier':<5} {'n':>3} {'Params':>8} {'MSE ratio':>10} {'Gap ratio':>10} {'δ_PAC':>10}")
    for r in tier_results:
        flag = " ★" if r.get('mse_ratio', 99) < 1.0 else ""
        print(f"{r['tier']:<5} {r['fib_index']:>3} {r['n_params']:>8} "
              f"{r['mse_ratio']:>10.4f} {r['gap_ratio']:>10.4f} "
              f"{r['mean_pac_delta']:>10.6f}{flag}")
    print("-" * 70)
    print(f"N* (crossover): {n_star_tier or 'not found'} (n={n_star or '?'})")
    if n_star_interpolated and n_star is None:
        print(f"N* (interpolated): n ≈ {n_star_interpolated:.1f}")
    print(f"CRITERION: N* exists within tested range")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

    results = {
        'experiment': 'exp_18_crossover_analysis',
        'passed': bool(passed),
        'criterion': 'N* exists within tested range',
        'n_star': n_star,
        'n_star_tier': n_star_tier,
        'n_star_interpolated': float(n_star_interpolated) if n_star_interpolated else None,
        'tiers': tier_results,
    }

    save_results(results, 'exp_18_results.json')
    return results


if __name__ == '__main__':
    results = run()
    sys.exit(0 if results['passed'] else 1)
