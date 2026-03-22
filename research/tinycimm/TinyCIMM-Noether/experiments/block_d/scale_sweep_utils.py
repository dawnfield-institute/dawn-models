"""
Shared utilities for Block D scale sweep experiments.

Scale tiers:
  XS  n=6   widths ~[8,5,3]       ~100 params
  S   n=9   widths ~[34,21,13]    ~1,800 params
  M   n=11  widths ~[89,55,34]    ~12,000 params
  L   n=13  widths ~[233,144,89]  ~82,000 params
  XL  n=15  widths ~[610,377,233] ~560,000 params
"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from TinyCIMM_Noether import NoetherNetwork, NoetherConfig, SimpleMLPBaseline
from fibonacci_topology import fibonacci, build_topology


# ── Scale tiers ──────────────────────────────────────────────────────────────

SCALE_TIERS = [
    {'name': 'XS', 'fib_index': 6},
    {'name': 'S',  'fib_index': 9},
    {'name': 'M',  'fib_index': 11},
    {'name': 'L',  'fib_index': 13},
    {'name': 'XL', 'fib_index': 15},
]

XL_TIMEOUT_SECONDS = 300  # 5 minutes per model per tier


def get_tier_widths(fib_index, depth=3, input_dim=None, output_dim=None):
    """Get layer widths for a given Fibonacci index."""
    topo = build_topology(fib_index, depth=depth,
                          input_dim=input_dim, output_dim=output_dim)
    return topo.layer_widths


def count_mlp_params(widths):
    """Count parameters in an MLP with given widths (weights + biases)."""
    total = 0
    for i in range(len(widths) - 1):
        total += widths[i] * widths[i + 1] + widths[i + 1]
    return total


def build_matched_mlp_widths(noether_params, input_dim, output_dim, depth=3):
    """
    Build MLP widths that match Noether's parameter count.

    Strategy: uniform hidden width across `depth` hidden layers, chosen so
    total params ≈ noether_params.
    """
    # For depth=3: input → h → h → h → output
    # params = input*h + h + h*h + h + h*output + output
    # Solve for h numerically
    best_h = 1
    best_diff = float('inf')
    for h in range(1, 2000):
        widths = [input_dim] + [h] * (depth - 1) + [output_dim]
        p = count_mlp_params(widths)
        diff = abs(p - noether_params)
        if diff < best_diff:
            best_diff = diff
            best_h = h
        if p > noether_params * 1.5:
            break
    return [input_dim] + [best_h] * (depth - 1) + [output_dim]


# ── Data generators (reused from Block B/C) ─────────────────────────────────

def she_leveque_exponent(p):
    """She-Leveque (1994) scaling exponents."""
    return p / 9.0 + 2.0 * (1.0 - (2.0 / 3.0) ** (p / 3.0))


def generate_sle_data(n_samples=800, seed=42):
    """Generate She-Leveque structure function data (same as Exp 06)."""
    rng = np.random.RandomState(seed)
    orders = np.arange(1, 9, dtype=float)
    n_per_order = n_samples // len(orders)

    X_list, Y_list = [], []
    for p in orders:
        zeta_p = she_leveque_exponent(p)
        log_r = rng.uniform(-3, 0, n_per_order)
        noise = rng.normal(0, 0.05, n_per_order)
        log_S = zeta_p * log_r + noise

        p_norm = p / 8.0
        p2_norm = (p ** 2) / 64.0
        zeta_k41 = p / 3.0
        features = np.column_stack([
            log_r,
            np.full(n_per_order, p_norm),
            np.full(n_per_order, p2_norm),
            np.full(n_per_order, zeta_k41),
        ])
        X_list.append(features)
        Y_list.append(log_S.reshape(-1, 1))

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    perm = rng.permutation(len(X))
    X, Y = X[perm], Y[perm]

    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std
    Y_mean, Y_std = Y.mean(), Y.std() + 1e-8
    Y = (Y - Y_mean) / Y_std

    return X, Y


RIEMANN_ZEROS_30 = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
]


def extend_zeros_synthetically(known_zeros, target_count=200, seed=42):
    """Extend Riemann zeros using asymptotic density + GUE fluctuations."""
    rng = np.random.RandomState(seed)
    known = np.array(known_zeros)
    spacings = np.diff(known)
    mean_spacing = np.mean(spacings)
    std_spacing = np.std(spacings)
    zeros = list(known_zeros)
    for n in range(len(known_zeros) + 1, target_count + 1):
        t_approx = 2 * np.pi * n / np.log(max(n / (2 * np.pi), 2))
        fluctuation = rng.normal(0, std_spacing * 0.3)
        t_n = t_approx + fluctuation
        if t_n <= zeros[-1]:
            t_n = zeros[-1] + abs(rng.normal(mean_spacing, std_spacing * 0.2))
        zeros.append(t_n)
    return np.array(zeros)


def generate_riemann_data(n_zeros=200, lookback=5, seed=42):
    """Generate Riemann zero spacing prediction data (same as Exp 11)."""
    zeros = extend_zeros_synthetically(RIEMANN_ZEROS_30,
                                        target_count=n_zeros, seed=seed)
    spacings = np.diff(zeros)
    sp_mean = np.mean(spacings)
    sp_std = np.std(spacings) + 1e-8
    spacings_norm = (spacings - sp_mean) / sp_std

    X, Y = [], []
    for i in range(lookback, len(spacings_norm)):
        X.append(spacings_norm[i - lookback:i])
        Y.append(spacings_norm[i])

    return np.array(X), np.array(Y).reshape(-1, 1)


def generate_turbulence_at_re(Re, n_samples=200, orders=None, seed=42):
    """Generate turbulence data at a given Reynolds number (same as Exp 09)."""
    rng = np.random.RandomState(seed)
    if orders is None:
        orders = [2, 3, 4, 5, 6]

    eta = Re ** (-3.0 / 4.0)
    L = 1.0
    n_per_order = n_samples // len(orders)
    X_list, Y_list = [], []

    for p in orders:
        zeta_p = she_leveque_exponent(p)
        log_r_min = np.log10(eta * 10)
        log_r_max = np.log10(L * 0.5)
        if log_r_min >= log_r_max:
            log_r_min = log_r_max - 2

        log_r = rng.uniform(log_r_min, log_r_max, n_per_order)
        r = 10.0 ** log_r
        C_p = Re ** (zeta_p / 4.0)
        S_p = C_p * (r / L) ** zeta_p
        noise = rng.normal(0, 0.05 * np.abs(S_p).mean(), n_per_order)
        S_p += noise

        features = np.column_stack([
            (log_r - log_r_min) / (log_r_max - log_r_min + 1e-10),
            np.full(n_per_order, p / 6.0),
            np.log(np.abs(S_p) + 1e-10),
            np.full(n_per_order, np.log10(Re) / 6.0),
        ])
        targets = np.full((n_per_order, 1), zeta_p)
        X_list.append(features)
        Y_list.append(targets)

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    perm = rng.permutation(len(X))
    return X[perm], Y[perm]


# ── Training helpers ─────────────────────────────────────────────────────────

def r_squared(y_true, y_pred):
    """Coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-10)


def _scale_epochs(base_epochs, n_params):
    """
    Scale epochs with network size.

    Larger networks need more iterations to converge at reasonable LR.
    """
    if n_params <= 200:
        return base_epochs
    elif n_params <= 3000:
        return int(base_epochs * 1.5)
    elif n_params <= 15000:
        return int(base_epochs * 2.0)
    else:
        return int(base_epochs * 2.5)


def train_noether_at_scale(fib_index, input_dim, output_dim, X_train, Y_train,
                           epochs=400, seed=42, depth=3):
    """
    Train a NoetherNetwork at a given scale tier.

    Learning rate strategy:
      - Conservation rate: fixed at 0.12 (controls PAC enforcement strength)
      - Direction rate: gentle scaling, 0.05 / n^(1/4) where n = param_ratio
        At XS (~100 params): 0.05
        At M  (~12k params): ~0.015
        At XL (~560k params): ~0.006
      This keeps the direction signal strong enough to learn at scale.

    Returns (network, history, elapsed_seconds).
    """
    topo = build_topology(fib_index, depth=depth,
                          input_dim=input_dim, output_dim=output_dim)
    param_ratio = max(1.0, topo.total_params / 100.0)

    # Gentle fourth-root scaling preserves learning signal at large scale
    conservation_rate = 0.12
    direction_rate = 0.05 / (param_ratio ** 0.25)

    actual_epochs = _scale_epochs(epochs, topo.total_params)

    config = NoetherConfig(
        depth=depth,
        fibonacci_index=fib_index,
        conservation_rate=conservation_rate,
        direction_rate=direction_rate,
        epsilon=1e-3,
        default_epochs=actual_epochs,
        seed=seed,
    )
    net = NoetherNetwork(input_dim=input_dim, output_dim=output_dim, config=config)

    t0 = time.time()
    history = net.fit(X_train, Y_train, verbose=False)
    elapsed = time.time() - t0

    return net, history, elapsed


def train_mlp_at_scale(widths, X_train, Y_train, epochs=400, seed=42):
    """
    Train MLP baseline at matching scale.

    Learning rate: 0.01 / n^(1/4) — same scaling law as Noether
    for fair comparison.

    Returns (mlp, history, elapsed_seconds).
    """
    n_params = count_mlp_params(widths)
    param_ratio = max(1.0, n_params / 100.0)
    lr = 0.01 / (param_ratio ** 0.25)

    actual_epochs = _scale_epochs(epochs, n_params)

    mlp = SimpleMLPBaseline(widths, lr=lr, activation='tanh', seed=seed)

    t0 = time.time()
    history = mlp.fit(X_train, Y_train, epochs=actual_epochs, verbose=False)
    elapsed = time.time() - t0

    return mlp, history, elapsed


def save_results(results, filename):
    """Save results JSON to the standard results directory."""
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")
    return path
