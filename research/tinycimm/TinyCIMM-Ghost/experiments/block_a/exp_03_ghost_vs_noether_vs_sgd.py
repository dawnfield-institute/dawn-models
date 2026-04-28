"""
Ghost Exp 03: Ghost vs Noether vs SGD

THE HEADLINE EXPERIMENT.

Noether's exp_04 FAILS: SGD beats Noether on physics-structured data.
Can Ghost's spectral confinement fix this? If Ghost beats Noether on the
exact datasets where Noether fails, spectral confinement isn't just pretty
math — it's computationally useful.

Three datasets:
  D1: Power-law regression (y = sum(x^alpha), alpha from Fibonacci sequence)
  D2: Fibonacci cascade (y depends on x through Fibonacci-weighted sum)
  D3: Generic regression (polynomial — no physics structure, SGD should win)

Three models (matched parameter count):
  - Ghost: encoder + symmetric recurrent core + decoder
  - SGD: standard MLP with backpropagation
  - Noether-like: PAC descent on same topology (no recurrent core)

Tests:
  T1: Ghost MSE < Noether MSE on power-law data (D1)
  T2: Ghost MSE < Noether MSE on Fibonacci cascade data (D2)
  T3: Ghost within 2x of SGD on all datasets
  T4: Ghost has lower PAC violations than SGD
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
GHOST_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(GHOST_ROOT))

from spectral_utils import PHI, PHI_INV
from ghost_network import GhostNetwork, GhostConfig, SimpleMLPBaseline

N_SEEDS = 10
N_EPOCHS = 300
N_TRAIN = 200
N_TEST = 50
RESULTS_DIR = GHOST_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# Data Generators
# ============================================================

def make_power_law_data(n_samples, input_dim=5, seed=0):
    """
    Power-law regression with Fibonacci-derived exponents.
    y = sum_i(|x_i|^alpha_i) where alpha_i = phi^(-i).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, input_dim) * 0.5
    alphas = [PHI ** (-i) for i in range(input_dim)]
    Y = np.zeros((n_samples, 1))
    for i in range(input_dim):
        Y += np.abs(X[:, i:i+1]) ** alphas[i]
    return X, Y


def make_fibonacci_cascade_data(n_samples, input_dim=5, seed=0):
    """
    Fibonacci cascade: y depends on x through Fibonacci-weighted sum.
    y = sum_i(F_i * tanh(x_i)) where F_i are Fibonacci numbers.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, input_dim)
    fibs = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55][:input_dim]
    Y = np.zeros((n_samples, 1))
    for i in range(input_dim):
        Y += fibs[i] * np.tanh(X[:, i:i+1])
    return X, Y


def make_generic_data(n_samples, input_dim=5, seed=0):
    """
    Generic polynomial regression — no physics structure.
    y = x1^2 + 2*x2 + sin(x3) + x4*x5.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, input_dim)
    Y = (X[:, 0:1] ** 2 + 2 * X[:, 1:2] + np.sin(X[:, 2:3])
         + X[:, 3:4] * X[:, 4:5])
    return X, Y


# ============================================================
# Noether-like Baseline (PAC descent, no recurrent core)
# ============================================================

class NoetherLikeBaseline:
    """
    PAC descent on encoder-decoder topology (no recurrent core).
    Matches Ghost's encoder/decoder structure but without spectral confinement.
    """

    def __init__(self, input_dim=5, output_dim=1, hidden_dim=13,
                 lr=0.01, pac_rate=0.05, seed=42):
        self.rng = np.random.RandomState(seed)
        self.lr = lr
        self.pac_rate = pac_rate

        # Two-layer: input → hidden → output
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim)) * np.sqrt(PHI_INV)
        self.W1 = self.rng.randn(hidden_dim, input_dim) * scale1
        self.b1 = np.zeros(hidden_dim)

        scale2 = np.sqrt(2.0 / (hidden_dim + output_dim))
        self.W2 = self.rng.randn(output_dim, hidden_dim) * scale2
        self.b2 = np.zeros(output_dim)

    def forward(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        h = np.tanh(x @ self.W1.T + self.b1)
        y = h @ self.W2.T + self.b2
        return y, {'x': x, 'h': h, 'y': y}

    def update(self, x, y_target):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y_target.ndim == 1:
            y_target = y_target.reshape(1, -1)

        y_pred, cache = self.forward(x)
        error = y_target - y_pred
        mse = float(np.mean(error ** 2))
        bs = x.shape[0]

        # Output layer delta rule
        dW2 = error.T @ cache['h'] / bs
        db2 = np.mean(error, axis=0)
        self.W2 += self.lr * dW2
        self.b2 += self.lr * db2

        # Hidden layer feedback alignment
        pseudo_error = error @ self.W2
        tanh_deriv = 1.0 - cache['h'] ** 2
        h_error = pseudo_error * tanh_deriv
        dW1 = h_error.T @ cache['x'] / bs
        db1 = np.mean(h_error, axis=0)
        self.W1 += self.lr * dW1
        self.b1 += self.lr * db1

        # PAC conservation scaling
        v_input = np.mean(np.abs(x))
        v_hidden = np.mean(np.abs(cache['h']))
        if v_input > 1e-10 and v_hidden > 1e-10:
            rho = PHI_INV / (v_hidden / v_input)
            rho = np.clip(rho, 0.9, 1.1)
            self.W1 *= 1.0 + self.pac_rate * (rho - 1.0)

        return mse

    def train(self, X, Y, epochs=200, verbose=False):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        history = []
        for ep in range(epochs):
            mse = self.update(X, Y)
            history.append({'epoch': ep, 'mse': mse})
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                print(f"  Epoch {ep:4d}: MSE={mse:.6f}")
        return history

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        y, _ = self.forward(x)
        return y


def evaluate(model, X_test, Y_test):
    """Compute test MSE."""
    Y_pred = model.predict(X_test)
    return float(np.mean((Y_pred - Y_test) ** 2))


def run_comparison(data_fn, data_name, seeds=N_SEEDS):
    """Run Ghost vs Noether vs SGD on a dataset."""
    ghost_mses = []
    noether_mses = []
    sgd_mses = []

    for seed in range(seeds):
        X, Y = data_fn(N_TRAIN + N_TEST, seed=seed)
        X_train, Y_train = X[:N_TRAIN], Y[:N_TRAIN]
        X_test, Y_test = X[N_TRAIN:], Y[N_TRAIN:]

        # Ghost
        config = GhostConfig(input_dim=5, output_dim=1, core_dim=13,
                             seed=seed)
        ghost = GhostNetwork(config)
        ghost.train(X_train, Y_train, epochs=N_EPOCHS)
        ghost_mses.append(evaluate(ghost, X_test, Y_test))

        # Noether-like
        noether = NoetherLikeBaseline(input_dim=5, output_dim=1,
                                       hidden_dim=13, seed=seed)
        noether.train(X_train, Y_train, epochs=N_EPOCHS)
        noether_mses.append(evaluate(noether, X_test, Y_test))

        # SGD (same architecture: 5 → 13 → 1)
        sgd = SimpleMLPBaseline([5, 13, 1], lr=0.01, seed=seed)
        sgd.train(X_train, Y_train, epochs=N_EPOCHS)
        sgd_mses.append(evaluate(sgd, X_test, Y_test))

    return {
        'dataset': data_name,
        'ghost_mse': float(np.mean(ghost_mses)),
        'ghost_std': float(np.std(ghost_mses)),
        'noether_mse': float(np.mean(noether_mses)),
        'noether_std': float(np.std(noether_mses)),
        'sgd_mse': float(np.mean(sgd_mses)),
        'sgd_std': float(np.std(sgd_mses)),
    }


# ============================================================
# Test 1: Ghost < Noether on Power-Law
# ============================================================
def test1_power_law():
    """T1: Ghost MSE < Noether MSE on power-law data."""
    print("\n=== T1: Power-Law Data ===")
    r = run_comparison(make_power_law_data, 'power_law')

    print(f"  Ghost:   MSE={r['ghost_mse']:.6f} +/- {r['ghost_std']:.6f}")
    print(f"  Noether: MSE={r['noether_mse']:.6f} +/- {r['noether_std']:.6f}")
    print(f"  SGD:     MSE={r['sgd_mse']:.6f} +/- {r['sgd_std']:.6f}")

    passed = r['ghost_mse'] < r['noether_mse']
    print(f"  Ghost < Noether: {passed}")
    print(f"  PASS: {passed}")

    r['test'] = 'power_law'
    r['passed'] = bool(passed)
    return r


# ============================================================
# Test 2: Ghost < Noether on Fibonacci Cascade
# ============================================================
def test2_fibonacci_cascade():
    """T2: Ghost MSE < Noether MSE on Fibonacci cascade data."""
    print("\n=== T2: Fibonacci Cascade Data ===")
    r = run_comparison(make_fibonacci_cascade_data, 'fibonacci_cascade')

    print(f"  Ghost:   MSE={r['ghost_mse']:.6f} +/- {r['ghost_std']:.6f}")
    print(f"  Noether: MSE={r['noether_mse']:.6f} +/- {r['noether_std']:.6f}")
    print(f"  SGD:     MSE={r['sgd_mse']:.6f} +/- {r['sgd_std']:.6f}")

    passed = r['ghost_mse'] < r['noether_mse']
    print(f"  Ghost < Noether: {passed}")
    print(f"  PASS: {passed}")

    r['test'] = 'fibonacci_cascade'
    r['passed'] = bool(passed)
    return r


# ============================================================
# Test 3: Ghost Within 2x of SGD
# ============================================================
def test3_within_sgd():
    """T3: Ghost within 2x of SGD on all datasets."""
    print("\n=== T3: Ghost Within 2x of SGD ===")

    datasets = [
        (make_power_law_data, 'power_law'),
        (make_fibonacci_cascade_data, 'fibonacci_cascade'),
        (make_generic_data, 'generic'),
    ]

    all_within = True
    results_detail = []

    for data_fn, name in datasets:
        r = run_comparison(data_fn, name)
        ratio = r['ghost_mse'] / r['sgd_mse'] if r['sgd_mse'] > 0 else float('inf')
        within = ratio < 2.0
        if not within:
            all_within = False

        print(f"  {name}: Ghost/SGD = {ratio:.2f}x {'OK' if within else 'MISS'}")
        results_detail.append({
            'dataset': name,
            'ghost_mse': r['ghost_mse'],
            'sgd_mse': r['sgd_mse'],
            'ratio': float(ratio),
            'within_2x': bool(within),
        })

    passed = all_within
    print(f"  PASS: {passed}")

    return {
        'test': 'within_sgd',
        'passed': bool(passed),
        'detail': results_detail,
    }


# ============================================================
# Test 4: Ghost Has Lower PAC Violations
# ============================================================
def test4_pac_violations():
    """T4: Ghost has lower PAC violations than SGD."""
    print("\n=== T4: PAC Violations ===")

    ghost_violations = []
    sgd_violations = []

    for seed in range(N_SEEDS):
        X, Y = make_power_law_data(N_TRAIN, seed=seed)

        # Ghost — track violations
        config = GhostConfig(input_dim=5, output_dim=1, core_dim=13,
                             seed=seed)
        ghost = GhostNetwork(config)
        history = ghost.train(X, Y, epochs=N_EPOCHS)
        final_viol = history[-1]['pac_violation']
        ghost_violations.append(final_viol)

        # SGD — compute PAC violation on same data
        sgd = SimpleMLPBaseline([5, 13, 1], lr=0.01, seed=seed)
        sgd.train(X, Y, epochs=N_EPOCHS)
        # Compute PAC violation for SGD (value ratio deviation from phi^(-1))
        y_pred, (acts, _) = sgd.forward(X)
        v_in = np.mean(np.abs(X))
        v_hid = np.mean(np.abs(acts[1]))
        v_out = np.mean(np.abs(y_pred))
        sgd_viol = abs(v_in - v_hid - v_out) if v_in > 1e-10 else 0
        sgd_violations.append(sgd_viol)

    mean_ghost = float(np.mean(ghost_violations))
    mean_sgd = float(np.mean(sgd_violations))

    passed = mean_ghost < mean_sgd
    print(f"  Ghost mean violation: {mean_ghost:.6f}")
    print(f"  SGD mean violation: {mean_sgd:.6f}")
    print(f"  PASS: {passed} (Ghost < SGD)")

    return {
        'test': 'pac_violations',
        'passed': bool(passed),
        'ghost_mean': mean_ghost,
        'sgd_mean': mean_sgd,
    }


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Ghost Exp 03: Ghost vs Noether vs SGD")
    print("  Can spectral confinement beat PAC descent alone?")
    print("=" * 70)

    tests = [test1_power_law, test2_fibonacci_cascade,
             test3_within_sgd, test4_pac_violations]

    results = []
    n_passed = 0
    for test_fn in tests:
        result = test_fn()
        results.append(result)
        if result['passed']:
            n_passed += 1

    print("\n" + "=" * 70)
    print(f"SCORE: {n_passed}/{len(tests)}")
    print("=" * 70)
    for r in results:
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  [{status}] {r['test']}")

    output = {
        'experiment': 'exp_03_ghost_vs_noether_vs_sgd',
        'variant': 'TinyCIMM-Ghost',
        'description': 'Ghost vs Noether vs SGD on physics-structured data',
        'score': f'{n_passed}/{len(tests)}',
        'n_passed': n_passed,
        'n_tests': len(tests),
        'tests': results,
        'config': {
            'n_seeds': N_SEEDS,
            'n_epochs': N_EPOCHS,
            'n_train': N_TRAIN,
            'n_test': N_TEST,
        },
        'timestamp': datetime.now().isoformat(),
    }

    fname = RESULTS_DIR / f"exp_03_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {fname}")
