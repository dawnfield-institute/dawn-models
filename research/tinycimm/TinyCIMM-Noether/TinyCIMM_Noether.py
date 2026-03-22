"""
TinyCIMM-Noether: PAC Conservation as Learning Rule

A neural architecture where PAC conservation IS the learning rule,
replacing gradient descent with local PAC violation reduction.

Named after Emmy Noether: every conservation law corresponds to a symmetry;
every symmetry in the data produces a conserved structure in the weights.

Phase A (this implementation):
  - Fibonacci-derived topology (D=3, widths from F_n)
  - PAC conservation descent (local violation reduction, no backprop)
  - Hard conservation enforcement (not a soft penalty)
  - Forward-only training (no separate training/inference phases)

Usage:
    from TinyCIMM_Noether import NoetherNetwork

    # Create network for 5-input, 1-output regression
    net = NoetherNetwork(input_dim=5, output_dim=1)

    # Train using PAC conservation descent
    history = net.fit(X_train, Y_train, epochs=200)

    # Predict (same forward pass as training)
    Y_pred = net.predict(X_test)

    # Check conservation status
    report = net.conservation_report(X_test[:1])
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from fibonacci_topology import (
    FibonacciTopology, build_topology, build_topology_for_data,
    topology_summary, PHI, PHI_INV,
)
from pac_descent import (
    PACDescentEngine, PACDescentConfig, LayerState,
    compute_value, compute_pac_violations,
)
from conservation_enforcement import (
    ConservationEnforcer, EnforcementConfig, ConservationStatus,
)


@dataclass
class NoetherConfig:
    """Full configuration for a TinyCIMM-Noether network."""
    # Topology
    depth: int = 3
    fibonacci_index: Optional[int] = None  # Auto-selected if None

    # PAC descent
    conservation_rate: float = 0.1
    direction_rate: float = 0.01
    max_correction: float = 0.5
    value_fn: str = 'l1'
    activation: str = 'tanh'

    # Conservation enforcement
    epsilon: float = 1e-4
    enforce_every: int = 1        # Enforce conservation every N steps
    max_enforce_iters: int = 20

    # Training
    default_epochs: int = 200
    batch_size: Optional[int] = None
    seed: int = 42


class NoetherNetwork:
    """
    TinyCIMM-Noether: PAC Conservation as a Complete Learning Rule.

    This network replaces gradient descent with PAC conservation descent.
    Topology is derived from Fibonacci structure (not searched).
    Training and inference use the same forward pass.
    Conservation is hard-enforced, not softly penalized.
    """

    def __init__(self, input_dim: int, output_dim: int,
                 config: Optional[NoetherConfig] = None):
        self.config = config or NoetherConfig()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build Fibonacci topology
        if self.config.fibonacci_index is not None:
            self.topology = build_topology(
                self.config.fibonacci_index,
                depth=self.config.depth,
                input_dim=input_dim,
                output_dim=output_dim,
            )
        else:
            self.topology = build_topology_for_data(
                input_dim, output_dim,
                depth=self.config.depth,
            )

        # Create PAC descent engine
        pac_config = PACDescentConfig(
            conservation_rate=self.config.conservation_rate,
            direction_rate=self.config.direction_rate,
            max_correction=self.config.max_correction,
            value_fn=self.config.value_fn,
            activation=self.config.activation,
        )
        self.engine = PACDescentEngine(
            self.topology, pac_config, seed=self.config.seed,
        )

        # Create conservation enforcer
        enforce_config = EnforcementConfig(
            epsilon=self.config.epsilon,
            max_iterations=self.config.max_enforce_iters,
            value_fn=self.config.value_fn,
        )
        self.enforcer = ConservationEnforcer(self.topology, enforce_config)

        # Training history
        self.history: List[Dict] = []

    def fit(self, X: np.ndarray, Y: np.ndarray,
            epochs: Optional[int] = None,
            verbose: bool = False) -> List[Dict]:
        """
        Train using PAC conservation descent with hard enforcement.

        Parameters:
            X: Training inputs [n_samples, input_dim]
            Y: Training targets [n_samples, output_dim]
            epochs: Number of training epochs
            verbose: Print progress

        Returns:
            Training history (list of per-epoch metrics)
        """
        epochs = epochs or self.config.default_epochs

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        history = self.engine.train(
            X, Y,
            epochs=epochs,
            batch_size=self.config.batch_size,
            verbose=verbose,
        )

        # Post-training conservation enforcement on a sample
        sample = X[:min(32, len(X))]
        _, states = self.engine.forward(sample)
        status = self.enforcer.enforce(states)

        self.history = history
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using forward pass (same pass as training).

        Parameters:
            X: Input data [n_samples, input_dim]

        Returns:
            Predictions [n_samples, output_dim]
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.engine.predict(X)

    def forward_with_enforcement(self, X: np.ndarray
                                  ) -> Tuple[np.ndarray, ConservationStatus]:
        """
        Forward pass with hard conservation enforcement.

        Parameters:
            X: Input data

        Returns:
            (predictions, conservation_status)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        output, states = self.engine.forward(X)
        status = self.enforcer.enforce(states)
        # Return corrected output
        corrected_output = states[-1].activations
        return corrected_output, status

    def check_conservation(self, X: np.ndarray) -> ConservationStatus:
        """Check PAC conservation for given input (without enforcing)."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        _, states = self.engine.forward(X)
        return self.enforcer.check_conservation(states)

    def conservation_report(self, X: np.ndarray) -> str:
        """Generate human-readable conservation report."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        _, states = self.engine.forward(X)
        return self.enforcer.conservation_report(states)

    def get_violations(self, X: np.ndarray) -> Dict[int, float]:
        """Get PAC violations for given input."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        _, states = self.engine.forward(X)
        return self.engine.compute_violations()

    @property
    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return self.topology.total_params

    def summary(self) -> str:
        """Human-readable network summary."""
        lines = [
            "TinyCIMM-Noether Network",
            "=" * 40,
            topology_summary(self.topology),
            "",
            f"Update rule: PAC conservation descent",
            f"  Conservation rate: {self.config.conservation_rate}",
            f"  Direction rate: {self.config.direction_rate}",
            f"  Activation: {self.config.activation}",
            f"  Value function: {self.config.value_fn}",
            "",
            f"Conservation enforcement:",
            f"  Epsilon: {self.config.epsilon}",
            f"  Max iterations: {self.config.max_enforce_iters}",
        ]
        return '\n'.join(lines)


# === Simple SGD baseline for comparison ===

class SimpleMLPBaseline:
    """
    Standard MLP with SGD for fair comparison.
    Same topology, same parameter count — different learning rule.
    """

    def __init__(self, layer_widths: List[int], lr: float = 0.01,
                 activation: str = 'tanh', seed: int = 42):
        self.layer_widths = layer_widths
        self.lr = lr
        self.activation = activation
        self.rng = np.random.RandomState(seed)

        self.weights = []
        self.biases = []
        for i in range(len(layer_widths) - 1):
            fan_in = layer_widths[i]
            fan_out = layer_widths[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(self.rng.randn(fan_out, fan_in) * scale)
            self.biases.append(np.zeros(fan_out))

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        return x

    def _activate_deriv(self, x: np.ndarray) -> np.ndarray:
        if self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation == 'relu':
            return (x > 0).astype(float)
        return np.ones_like(x)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List]:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        activations = [x]
        pre_acts = [x]
        current = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ W.T + b
            pre_acts.append(z)
            if i < len(self.weights) - 1:
                current = self._activate(z)
            else:
                current = z  # Linear output
            activations.append(current)
        return current, (activations, pre_acts)

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """Standard backpropagation + SGD step."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)

        y_pred, (activations, pre_acts) = self.forward(x)
        error = y_pred - y
        mse = np.mean(error ** 2)

        # Backprop
        delta = error / x.shape[0]  # [batch, output_dim]
        for i in range(len(self.weights) - 1, -1, -1):
            dW = delta.T @ activations[i]
            db = np.sum(delta, axis=0)
            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db
            if i > 0:
                delta = (delta @ self.weights[i]) * self._activate_deriv(pre_acts[i])

        return mse

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 200,
            verbose: bool = False) -> List[float]:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        history = []
        for ep in range(epochs):
            mse = self.train_step(X, Y)
            history.append(mse)
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                print(f"  Epoch {ep:4d}: MSE={mse:.6f}")
        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        out, _ = self.forward(x)
        return out
