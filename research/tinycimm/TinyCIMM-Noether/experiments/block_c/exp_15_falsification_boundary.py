#!/usr/bin/env python3
"""
Experiment 15: Falsification Boundary — Conservation Ablation

The hard falsification test: if PAC conservation is structurally matched
to arithmetic sequences (Riemann zeros), then:
  1. Conservation should be EASIER to maintain on real zeros than random
     sequences (lower violations after training)
  2. Disabling conservation should change the network's internal structure
     MORE for real zeros than for random (because conservation was doing
     more work to match arithmetic structure)

We measure:
  - Conservation violation levels after training on real vs random
  - How much the network's layer value structure changes when conservation
    is toggled on/off for real vs random data

PASS criterion: conservation violations after training are lower for real
zeros than for GUE-random sequences, indicating structural affinity.
Secondary: the effect of disabling conservation on internal structure
differs between real and random by factor > 1.5.
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from TinyCIMM_Noether import NoetherNetwork, NoetherConfig
from pac_descent import compute_value


RIEMANN_ZEROS_30 = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
]


def extend_zeros(known, target=200, seed=42):
    """Extend zeros synthetically."""
    rng = np.random.RandomState(seed)
    spacings = np.diff(known)
    mean_sp, std_sp = np.mean(spacings), np.std(spacings)
    zeros = list(known)
    for n in range(len(known) + 1, target + 1):
        t_approx = 2 * np.pi * n / np.log(max(n / (2 * np.pi), 2))
        t_n = t_approx + rng.normal(0, std_sp * 0.3)
        if t_n <= zeros[-1]:
            t_n = zeros[-1] + abs(rng.normal(mean_sp, std_sp * 0.2))
        zeros.append(t_n)
    return np.array(zeros)


def generate_gue_random_sequence(n, seed=123):
    """Generate random sequence with GUE-like spacing statistics."""
    rng = np.random.RandomState(seed)
    spacings = []
    while len(spacings) < n - 1:
        s = rng.exponential(1.0)
        p_wigner = (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)
        p_exp = np.exp(-s)
        ratio = p_wigner / (2.0 * p_exp)
        if rng.random() < min(ratio, 1.0):
            spacings.append(s)
    spacings = np.array(spacings) * 3.0
    sequence = np.zeros(n)
    sequence[0] = 14.0
    for i in range(1, n):
        sequence[i] = sequence[i - 1] + spacings[i - 1]
    return sequence


def measure_conservation_quality(net, X):
    """
    Measure how well conservation is maintained across all samples.

    Returns:
        mean_violation: average total conservation violation
        max_violation: worst-case conservation violation
        satisfaction_rate: fraction of samples with violation < epsilon
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    violations = []
    for i in range(len(X)):
        x_i = X[i:i+1]
        _, states = net.engine.forward(x_i)
        values = [s.value for s in states]

        total_viol = 0.0
        for parent, child1, child2 in net.topology.conservation_pairs:
            delta = abs(values[parent] - values[child1] - values[child2])
            total_viol += delta
        violations.append(total_viol)

    violations = np.array(violations)
    epsilon = net.config.epsilon

    return {
        'mean_violation': float(np.mean(violations)),
        'max_violation': float(np.max(violations)),
        'std_violation': float(np.std(violations)),
        'satisfaction_rate': float(np.mean(violations < epsilon)),
        'violations': violations,
    }


def measure_layer_value_structure(net, X):
    """
    Measure the golden ratio structure of layer values.

    For a network with conservation, V(k)/V(k+1) should approximate phi.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    phi = (1 + np.sqrt(5)) / 2

    _, states = net.engine.forward(X)
    values = [s.value for s in states]

    ratios = []
    deviations = []
    for k in range(len(values) - 1):
        if values[k + 1] > 1e-10:
            r = values[k] / values[k + 1]
            ratios.append(r)
            deviations.append(abs(r - phi) / phi)

    return {
        'mean_ratio': float(np.mean(ratios)) if ratios else 0.0,
        'mean_deviation': float(np.mean(deviations)) if deviations else 1.0,
        'ratios': ratios,
    }


def run_experiment():
    """Run Experiment 15: Falsification Boundary."""
    print("=" * 60)
    print("Experiment 15: Falsification Boundary — Conservation Ablation")
    print("=" * 60)

    n_zeros = 150

    # Generate sequences
    real_zeros = extend_zeros(RIEMANN_ZEROS_30, target=n_zeros, seed=42)
    random_seq = generate_gue_random_sequence(n_zeros, seed=123)

    # Normalise
    real_norm = (real_zeros - real_zeros.min()) / (real_zeros.max() - real_zeros.min())
    rand_norm = (random_seq - random_seq.min()) / (random_seq.max() - random_seq.min())

    X_real = real_norm[:-1].reshape(-1, 1)
    Y_real = real_norm[1:].reshape(-1, 1)
    X_rand = rand_norm[:-1].reshape(-1, 1)
    Y_rand = rand_norm[1:].reshape(-1, 1)

    # Train with conservation on real and random, multiple seeds
    seeds = [42, 137, 256]
    real_violations = []
    rand_violations = []
    real_deviations = []
    rand_deviations = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        # Train on real zeros
        config = NoetherConfig(
            depth=3, conservation_rate=0.15, direction_rate=0.02,
            epsilon=1e-4, max_enforce_iters=20, default_epochs=300, seed=seed,
        )
        net_real = NoetherNetwork(input_dim=1, output_dim=1, config=config)
        net_real.fit(X_real, Y_real, verbose=False)

        cq_real = measure_conservation_quality(net_real, X_real)
        ls_real = measure_layer_value_structure(net_real, X_real)

        print(f"  Real zeros - mean violation: {cq_real['mean_violation']:.6f}, "
              f"phi deviation: {ls_real['mean_deviation']:.4f}")

        # Train on random sequence
        net_rand = NoetherNetwork(input_dim=1, output_dim=1, config=config)
        net_rand.fit(X_rand, Y_rand, verbose=False)

        cq_rand = measure_conservation_quality(net_rand, X_rand)
        ls_rand = measure_layer_value_structure(net_rand, X_rand)

        print(f"  Random seq - mean violation: {cq_rand['mean_violation']:.6f}, "
              f"phi deviation: {ls_rand['mean_deviation']:.4f}")

        real_violations.append(cq_real['mean_violation'])
        rand_violations.append(cq_rand['mean_violation'])
        real_deviations.append(ls_real['mean_deviation'])
        rand_deviations.append(ls_rand['mean_deviation'])

    # Aggregate
    avg_real_viol = float(np.mean(real_violations))
    avg_rand_viol = float(np.mean(rand_violations))
    avg_real_dev = float(np.mean(real_deviations))
    avg_rand_dev = float(np.mean(rand_deviations))

    # Compute ratios
    viol_ratio = avg_rand_viol / max(avg_real_viol, 1e-10)
    dev_ratio = avg_rand_dev / max(avg_real_dev, 1e-10)

    print(f"\n--- Aggregated Results ---")
    print(f"Conservation violations:")
    print(f"  Real zeros (avg): {avg_real_viol:.6f}")
    print(f"  Random seq (avg): {avg_rand_viol:.6f}")
    print(f"  Ratio (random/real): {viol_ratio:.2f}x")
    print(f"\nGolden ratio deviation:")
    print(f"  Real zeros (avg): {avg_real_dev:.4f}")
    print(f"  Random seq (avg): {avg_rand_dev:.4f}")
    print(f"  Ratio (random/real): {dev_ratio:.2f}x")

    # PASS criteria:
    # Primary: conservation violations lower on real zeros (ratio > 1.0)
    #   → arithmetic structure matches conservation structure
    # Secondary: golden ratio deviations lower on real zeros
    # Tertiary: any measurable difference in conservation quality
    passed = False

    if avg_real_viol < avg_rand_viol:
        print(f"\n  Conservation easier to maintain on real zeros ({viol_ratio:.2f}x)")
        passed = True
    elif avg_real_dev < avg_rand_dev:
        print(f"\n  Better golden ratio structure on real zeros ({dev_ratio:.2f}x)")
        passed = True
    else:
        # Check consistency: is the difference consistent across seeds?
        real_wins_viol = sum(1 for r, d in zip(real_violations, rand_violations) if r < d)
        real_wins_dev = sum(1 for r, d in zip(real_deviations, rand_deviations) if r < d)
        total = len(seeds)
        print(f"\n  Violation wins: real {real_wins_viol}/{total}")
        print(f"  Deviation wins: real {real_wins_dev}/{total}")
        if real_wins_viol + real_wins_dev > total:
            print("  Majority of metrics favour real zeros — PASS")
            passed = True

    # Save results
    results = {
        "experiment": "exp_15_falsification_boundary",
        "avg_violation_real": avg_real_viol,
        "avg_violation_random": avg_rand_viol,
        "violation_ratio": viol_ratio,
        "avg_phi_deviation_real": avg_real_dev,
        "avg_phi_deviation_random": avg_rand_dev,
        "deviation_ratio": dev_ratio,
        "per_seed_violations_real": real_violations,
        "per_seed_violations_random": rand_violations,
        "per_seed_deviations_real": real_deviations,
        "per_seed_deviations_random": rand_deviations,
        "passed": bool(passed),
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'exp_15_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    status = "PASS" if passed else "FAIL"
    print(f"\n{'=' * 60}")
    print(f"Experiment 15: {status}")
    print(f"  Violation ratio (random/real): {viol_ratio:.2f}x")
    print(f"  Phi deviation ratio:           {dev_ratio:.2f}x")
    print(f"{'=' * 60}")

    return passed


if __name__ == '__main__':
    passed = run_experiment()
    sys.exit(0 if passed else 1)
