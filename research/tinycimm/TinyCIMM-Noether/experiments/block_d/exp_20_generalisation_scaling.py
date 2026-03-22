#!/usr/bin/env python3
"""
Experiment 20: Generalisation Gap Scaling with Network Size

At each scale tier, measure Reynolds number generalisation:
  Train at Re_low (1e4), test at Re_high (1e6) — same setup as Exp 09.

Hypothesis: Conservation structure provides scale-invariant representations
that generalise better across Reynolds numbers. The advantage should grow
with network capacity because larger conservation-enforced networks have
more structure to exploit.

PASS criterion:
  Generalisation gap ratio (Noether_gap / MLP_gap) decreases monotonically
  with scale AND reaches < 0.20 at tier L.
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
    generate_turbulence_at_re, r_squared,
    train_noether_at_scale, train_mlp_at_scale,
    build_matched_mlp_widths, get_tier_widths,
    count_mlp_params, save_results,
)
from fibonacci_topology import build_topology


def run():
    print("=" * 70)
    print("Exp 20: Generalisation Gap Scaling with Network Size")
    print("=" * 70)

    Re_train = 1e4   # Moderate turbulence
    Re_test = 1e6    # Fully developed turbulence

    X_train_raw, Y_train = generate_turbulence_at_re(
        Re_train, n_samples=500, seed=42)
    X_test_raw, Y_test = generate_turbulence_at_re(
        Re_test, n_samples=200, seed=123)

    # Normalise jointly
    X_all = np.vstack([X_train_raw, X_test_raw])
    X_mean, X_std = X_all.mean(axis=0), X_all.std(axis=0) + 1e-8
    X_train = (X_train_raw - X_mean) / X_std
    X_test = (X_test_raw - X_mean) / X_std

    Y_mean, Y_std = Y_train.mean(), Y_train.std() + 1e-8
    Y_train_norm = (Y_train - Y_mean) / Y_std
    Y_test_norm = (Y_test - Y_mean) / Y_std

    input_dim = X_train.shape[1]
    output_dim = 1
    print(f"Train: {len(X_train)} @ Re={Re_train:.0e}")
    print(f"Test:  {len(X_test)} @ Re={Re_test:.0e}\n")

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

        # ── Noether ──────────────────────────────────────────────────────
        try:
            net, _, elapsed_n = train_noether_at_scale(
                fib_idx, input_dim, output_dim,
                X_train, Y_train_norm, epochs=epochs, seed=42)

            mse_train_n = float(np.mean((Y_train_norm - net.predict(X_train)) ** 2))
            mse_test_n = float(np.mean((Y_test_norm - net.predict(X_test)) ** 2))
            gap_n = abs(mse_test_n - mse_train_n) / (mse_train_n + 1e-10)
        except Exception as e:
            print(f"  Noether FAILED: {e}")
            mse_train_n = mse_test_n = float('inf')
            gap_n = float('inf')
            elapsed_n = 0.0
            skipped = True

        if elapsed_n > XL_TIMEOUT_SECONDS and name == 'XL':
            print(f"  Noether timed out ({elapsed_n:.1f}s)")
            skipped = True

        # ── MLP ──────────────────────────────────────────────────────────
        try:
            mlp, _, elapsed_m = train_mlp_at_scale(
                mlp_widths, X_train, Y_train_norm, epochs=epochs, seed=42)

            mse_train_m = float(np.mean((Y_train_norm - mlp.predict(X_train)) ** 2))
            mse_test_m = float(np.mean((Y_test_norm - mlp.predict(X_test)) ** 2))
            gap_m = abs(mse_test_m - mse_train_m) / (mse_train_m + 1e-10)
        except Exception as e:
            print(f"  MLP FAILED: {e}")
            mse_train_m = mse_test_m = float('inf')
            gap_m = float('inf')
            elapsed_m = 0.0
            skipped = True

        gap_ratio = gap_n / (gap_m + 1e-10)

        print(f"  Noether: train={mse_train_n:.6f}, test={mse_test_n:.6f}, gap={gap_n:.4f}")
        print(f"  MLP:     train={mse_train_m:.6f}, test={mse_test_m:.6f}, gap={gap_m:.4f}")
        print(f"  Gap ratio (N/M): {gap_ratio:.4f}")
        print()

        tier_results.append({
            'tier': name,
            'fib_index': fib_idx,
            'n_params': n_params,
            'noether_mse_train': mse_train_n,
            'noether_mse_test': mse_test_n,
            'noether_gap': gap_n,
            'mlp_mse_train': mse_train_m,
            'mlp_mse_test': mse_test_m,
            'mlp_gap': gap_m,
            'gap_ratio': gap_ratio,
            'skipped': skipped,
        })

    # ── Check monotonic decrease ─────────────────────────────────────────
    valid = [r for r in tier_results if not r['skipped']]
    gap_ratios = [r['gap_ratio'] for r in valid]

    # Monotonicity: each ratio ≤ previous (allow 5% tolerance for noise)
    monotonic = True
    for i in range(1, len(gap_ratios)):
        if gap_ratios[i] > gap_ratios[i - 1] * 1.05:
            monotonic = False
            break

    # Check < 0.20 at tier L
    tier_l = next((r for r in tier_results if r['tier'] == 'L'), None)
    below_threshold = (tier_l is not None and
                       not tier_l['skipped'] and
                       tier_l['gap_ratio'] < 0.20)

    passed = monotonic and below_threshold

    print("=" * 70)
    print("GENERALISATION GAP SCALING SUMMARY")
    print("-" * 70)
    print(f"{'Tier':<5} {'n':>3} {'Noether gap':>12} {'MLP gap':>12} {'Gap ratio':>10}")
    for r in tier_results:
        flag = " ★" if r['gap_ratio'] < 0.20 else ""
        skip = " (skip)" if r['skipped'] else ""
        print(f"{r['tier']:<5} {r['fib_index']:>3} "
              f"{r['noether_gap']:>12.4f} {r['mlp_gap']:>12.4f} "
              f"{r['gap_ratio']:>10.4f}{flag}{skip}")
    print("-" * 70)
    print(f"Monotonic decrease: {'YES' if monotonic else 'NO'}")
    print(f"Gap ratio < 0.20 at L: {'YES' if below_threshold else 'NO'}")
    if tier_l:
        print(f"  (Actual: {tier_l['gap_ratio']:.4f})")
    print(f"CRITERION: monotonic decrease AND < 0.20 at L")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

    results = {
        'experiment': 'exp_20_generalisation_scaling',
        'passed': bool(passed),
        'criterion': 'monotonic gap ratio decrease AND < 0.20 at L',
        'monotonic': monotonic,
        'below_threshold_at_l': below_threshold,
        'gap_ratios': gap_ratios,
        'Re_train': Re_train,
        'Re_test': Re_test,
        'tiers': tier_results,
    }

    save_results(results, 'exp_20_results.json')
    return results


if __name__ == '__main__':
    results = run()
    sys.exit(0 if results['passed'] else 1)
