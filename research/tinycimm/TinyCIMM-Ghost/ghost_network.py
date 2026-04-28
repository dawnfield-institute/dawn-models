"""
TinyCIMM-Ghost: Constrained Learner

Encoder-Core-Decoder architecture:
  - ENCODER: W_enc [core_dim, input_dim], tanh — learns via PAC descent
  - CORE: K recurrent steps of tanh(W_core @ h), anti-Hebbian modulation
           W_core is symmetric (core_dim x core_dim), sr = gamma/ln(phi)
  - DECODER: W_dec [output_dim, core_dim], linear — learns via PAC descent

Learning rule (dual-domain):
  - Encoder/decoder: local error-based updates with PAC conservation scaling
  - Core: spectral modulation only (NO gradient through recurrent steps)

This tests whether M10's spectral confinement helps a LEARNING system.
The headline experiment: Ghost vs Noether vs SGD on physics-structured data.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from spectral_utils import PHI, PHI_INV, SCOPE_RATIO
from ghost_core import SymmetricRecurrentCore, CoreConfig


@dataclass
class GhostConfig:
    """Configuration for the Ghost network."""
    input_dim: int = 5
    output_dim: int = 1
    core_dim: int = 13              # Fibonacci number via MED
    K: int = 1                      # Recurrent steps in core
    enc_lr: float = 0.01            # Encoder learning rate
    dec_lr: float = 0.01            # Decoder learning rate
    pac_conservation_rate: float = 0.05  # PAC scaling correction
    activation: str = 'tanh'
    seed: int = 42


class GhostNetwork:
    """
    TinyCIMM-Ghost: Encoder-Core-Decoder with spectral confinement.
    """

    def __init__(self, config: Optional[GhostConfig] = None):
        self.config = config or GhostConfig()
        self.rng = np.random.RandomState(self.config.seed)

        input_dim = self.config.input_dim
        core_dim = self.config.core_dim
        output_dim = self.config.output_dim

        # Encoder: input_dim → core_dim
        scale_enc = np.sqrt(2.0 / (input_dim + core_dim)) * np.sqrt(PHI_INV)
        self.W_enc = self.rng.randn(core_dim, input_dim) * scale_enc
        self.b_enc = np.zeros(core_dim)

        # Core: symmetric recurrent
        core_config = CoreConfig(
            core_dim=core_dim,
            K=self.config.K,
            target_sr=SCOPE_RATIO,
            seed=self.config.seed + 100,
        )
        self.core = SymmetricRecurrentCore(core_config)

        # Decoder: core_dim → output_dim
        scale_dec = np.sqrt(2.0 / (core_dim + output_dim))
        self.W_dec = self.rng.randn(output_dim, core_dim) * scale_dec
        self.b_dec = np.zeros(output_dim)

        self.step_count = 0

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Forward pass: encode → recurrent core → decode.

        Parameters:
            x: Input [batch_size, input_dim]

        Returns:
            y_pred: Output [batch_size, output_dim]
            cache: Intermediate states for learning
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Encode
        h_enc = np.tanh(x @ self.W_enc.T + self.b_enc)

        # Core (K recurrent steps with spectral modulation)
        h_core = self.core.forward(h_enc)

        # Decode (linear)
        y_pred = h_core @ self.W_dec.T + self.b_dec

        # PAC values at each stage
        v_input = np.mean(np.abs(x))
        v_enc = np.mean(np.abs(h_enc))
        v_core = np.mean(np.abs(h_core))
        v_output = np.mean(np.abs(y_pred))

        cache = {
            'x': x,
            'h_enc': h_enc,
            'h_core': h_core,
            'y_pred': y_pred,
            'v_input': v_input,
            'v_enc': v_enc,
            'v_core': v_core,
            'v_output': v_output,
        }

        return y_pred, cache

    def update(self, x: np.ndarray, y_target: np.ndarray) -> Dict[str, float]:
        """
        One training step.

        Encoder/decoder learn via local error signals.
        Core learns via spectral modulation (happens in forward pass).
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y_target.ndim == 1:
            y_target = y_target.reshape(1, -1)

        y_pred, cache = self.forward(x)
        batch_size = x.shape[0]

        # Output error
        error = y_target - y_pred
        mse = float(np.mean(error ** 2))

        # === Decoder update (direct delta rule) ===
        dW_dec = error.T @ cache['h_core'] / batch_size
        db_dec = np.mean(error, axis=0)
        self.W_dec += self.config.dec_lr * dW_dec
        self.b_dec += self.config.dec_lr * db_dec

        # === Encoder update (feedback alignment) ===
        # Project error back through decoder weights (one-hop, not backprop)
        pseudo_error = error @ self.W_dec  # [batch, core_dim]
        # Apply through core (approximate — use identity since core is recurrent)
        # Multiply by tanh derivative of encoder
        tanh_deriv = 1.0 - cache['h_enc'] ** 2
        enc_error = pseudo_error * tanh_deriv

        dW_enc = enc_error.T @ cache['x'] / batch_size
        db_enc = np.mean(enc_error, axis=0)
        self.W_enc += self.config.enc_lr * dW_enc
        self.b_enc += self.config.enc_lr * db_enc

        # === PAC conservation scaling ===
        # Scale encoder weights to maintain PAC value ratios
        # Target: v_enc / v_input ≈ phi^(-1)
        if cache['v_input'] > 1e-10 and cache['v_enc'] > 1e-10:
            actual_ratio = cache['v_enc'] / cache['v_input']
            target_ratio = PHI_INV
            rho = target_ratio / actual_ratio
            rho = np.clip(rho, 0.9, 1.1)
            correction = 1.0 + self.config.pac_conservation_rate * (rho - 1.0)
            self.W_enc *= correction

        # PAC violation
        total_violation = 0.0
        if cache['v_enc'] > 1e-10:
            # V(input) should ≈ V(enc) + V(core)
            delta1 = cache['v_input'] - cache['v_enc'] - cache['v_core']
            total_violation += abs(delta1)

        self.step_count += 1

        return {
            'mse': mse,
            'pac_violation': total_violation,
            'v_input': cache['v_input'],
            'v_enc': cache['v_enc'],
            'v_core': cache['v_core'],
            'v_output': cache['v_output'],
            'step': self.step_count,
        }

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 200,
              batch_size: Optional[int] = None,
              verbose: bool = False) -> List[Dict]:
        """Train the Ghost network."""
        n = X.shape[0]
        if batch_size is None:
            batch_size = n

        history = []
        for epoch in range(epochs):
            perm = self.rng.permutation(n)
            epoch_mse = 0.0
            epoch_viol = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                xb = X[perm[start:end]]
                yb = Y[perm[start:end]]
                metrics = self.update(xb, yb)
                epoch_mse += metrics['mse']
                epoch_viol += metrics['pac_violation']
                n_batches += 1

            epoch_mse /= n_batches
            epoch_viol /= n_batches
            record = {'epoch': epoch, 'mse': epoch_mse, 'pac_violation': epoch_viol}
            history.append(record)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch:4d}: MSE={epoch_mse:.6f}, "
                      f"PAC viol={epoch_viol:.6f}")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict (same forward pass as training)."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y, _ = self.forward(X)
        return y

    def core_snapshot(self) -> dict:
        """Get core diagnostic snapshot."""
        return self.core.snapshot()


# === Baselines ===

class SimpleMLPBaseline:
    """Standard MLP with SGD for fair comparison."""

    def __init__(self, layer_widths: List[int], lr: float = 0.01,
                 seed: int = 42):
        self.layer_widths = layer_widths
        self.lr = lr
        self.rng = np.random.RandomState(seed)

        self.weights = []
        self.biases = []
        for i in range(len(layer_widths) - 1):
            fan_in = layer_widths[i]
            fan_out = layer_widths[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(self.rng.randn(fan_out, fan_in) * scale)
            self.biases.append(np.zeros(fan_out))

    def forward(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        activations = [x]
        pre_acts = [x]
        current = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ W.T + b
            pre_acts.append(z)
            if i < len(self.weights) - 1:
                current = np.tanh(z)
            else:
                current = z
            activations.append(current)
        return current, (activations, pre_acts)

    def train_step(self, x, y):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        y_pred, (activations, pre_acts) = self.forward(x)
        error = y_pred - y
        mse = float(np.mean(error ** 2))
        delta = error / x.shape[0]
        for i in range(len(self.weights) - 1, -1, -1):
            dW = delta.T @ activations[i]
            db = np.sum(delta, axis=0)
            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db
            if i > 0:
                delta = (delta @ self.weights[i]) * (1.0 - np.tanh(pre_acts[i]) ** 2)
        return mse

    def train(self, X, Y, epochs=200, verbose=False):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        history = []
        for ep in range(epochs):
            mse = self.train_step(X, Y)
            history.append({'epoch': ep, 'mse': mse})
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                print(f"  Epoch {ep:4d}: MSE={mse:.6f}")
        return history

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        out, _ = self.forward(x)
        return out
