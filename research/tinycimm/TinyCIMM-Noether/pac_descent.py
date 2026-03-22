"""
PAC Conservation Descent — the update rule for TinyCIMM-Noether

Instead of gradient descent (minimize loss via backpropagation), PAC descent
reduces PAC violations locally at each layer. No global gradient, no optimizer,
no learning rate schedule.

Core principle:
  At each layer k, PAC conservation requires:
    V(k) = V(k+1) + V(k+2)    [Fibonacci recursion]

  where V(k) is the total "value" (L1 norm of activations) at layer k.

  PAC violation: δ_k = V(k) - V(k+1) - V(k+2)

  The update rule adjusts weights to reduce |δ_k| toward zero.

Update mechanism:
  1. Conservation correction: scale weights to match PAC value ratios
  2. Direction correction: Hebbian update using local error signals
  3. All layers update in parallel — no sequential dependency

This is forward-only: training and inference use the same pass.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from fibonacci_topology import PHI, PHI_INV, FibonacciTopology


@dataclass
class LayerState:
    """Cached state of a single layer during a forward pass."""
    activations: np.ndarray          # Layer output activations
    pre_activations: np.ndarray      # Before activation function
    value: float                     # V(k) = ||activations||_1
    target_value: float = 0.0        # PAC-prescribed target value
    violation: float = 0.0           # δ_k = V(k) - target


@dataclass
class PACDescentConfig:
    """Configuration for PAC conservation descent."""
    conservation_rate: float = 0.1    # Step size for conservation corrections
    direction_rate: float = 0.01      # Step size for directional updates
    max_correction: float = 0.5       # Maximum relative weight change per step
    value_fn: str = 'l1'              # Value function: 'l1' or 'l2'
    activation: str = 'tanh'          # Activation function


def compute_value(activations: np.ndarray, value_fn: str = 'l1') -> float:
    """
    Compute the PAC value of a layer's activations.

    The value function maps activations to a scalar representing the
    "conserved quantity" that PAC tracks through the network.

    Parameters:
        activations: Layer output activations [batch_size, width]
        value_fn: 'l1' (sum of absolutes) or 'l2' (sum of squares)

    Returns:
        Scalar value V(k)
    """
    if value_fn == 'l1':
        return np.mean(np.abs(activations))
    elif value_fn == 'l2':
        return np.mean(activations ** 2)
    else:
        raise ValueError(f"Unknown value function: {value_fn}")


def compute_pac_violations(layer_values: List[float],
                           conservation_pairs: List[Tuple[int, int, int]]
                           ) -> Dict[int, float]:
    """
    Compute PAC violations for all conservation pairs.

    For Fibonacci recursion: δ_k = V(k) - V(k+1) - V(k+2)

    Parameters:
        layer_values: V(k) for each layer
        conservation_pairs: List of (parent, child1, child2) index triples

    Returns:
        Dict mapping parent layer index to violation δ_k
    """
    violations = {}
    for parent, child1, child2 in conservation_pairs:
        delta = layer_values[parent] - layer_values[child1] - layer_values[child2]
        violations[parent] = delta
    return violations


def compute_target_values(input_value: float, num_layers: int) -> List[float]:
    """
    Compute PAC-prescribed target values for each layer.

    From the Fibonacci recursion, V(k) = V(0) * φ^(-k).

    Parameters:
        input_value: V(0) — the value at the input layer
        num_layers: Total number of layers

    Returns:
        List of target values for each layer
    """
    return [input_value * (PHI_INV ** k) for k in range(num_layers)]


def activation_fn(x: np.ndarray, name: str = 'tanh') -> np.ndarray:
    """Apply activation function."""
    if name == 'tanh':
        return np.tanh(x)
    elif name == 'relu':
        return np.maximum(0, x)
    elif name == 'sigmoid':
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    elif name == 'identity':
        return x
    else:
        raise ValueError(f"Unknown activation: {name}")


def activation_derivative(x: np.ndarray, name: str = 'tanh') -> np.ndarray:
    """Derivative of activation function (for local Hebbian updates)."""
    if name == 'tanh':
        return 1.0 - np.tanh(x) ** 2
    elif name == 'relu':
        return (x > 0).astype(float)
    elif name == 'sigmoid':
        s = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return s * (1 - s)
    elif name == 'identity':
        return np.ones_like(x)
    else:
        raise ValueError(f"Unknown activation: {name}")


class PACDescentEngine:
    """
    PAC Conservation Descent engine.

    Manages the forward pass and local weight updates for a
    TinyCIMM-Noether network. Each layer updates independently
    based on its local PAC violation — no backpropagation.
    """

    def __init__(self, topology: FibonacciTopology,
                 config: Optional[PACDescentConfig] = None,
                 seed: int = 42):
        self.topology = topology
        self.config = config or PACDescentConfig()
        self.rng = np.random.RandomState(seed)
        self.config_seed = seed

        # Initialize weights and biases
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._init_weights()

        # Layer states (populated during forward pass)
        self.layer_states: List[LayerState] = []

        # Training statistics
        self.step_count = 0
        self.violation_history: List[float] = []

    def _init_weights(self):
        """
        Initialize weights using PAC-consistent initialization.

        Weights are scaled so that the expected PAC value ratio between
        adjacent layers matches φ^(-1). This gives the network a head
        start on conservation.
        """
        self.weights = []
        self.biases = []
        widths = self.topology.layer_widths

        for i in range(len(widths) - 1):
            fan_in = widths[i]
            fan_out = widths[i + 1]

            # Xavier-like init, but scaled by φ^(-1) for PAC consistency
            # The scaling ensures V(k+1) ≈ V(k) * φ^(-1) at initialization
            scale = np.sqrt(2.0 / (fan_in + fan_out)) * np.sqrt(PHI_INV)
            W = self.rng.randn(fan_out, fan_in) * scale
            b = np.zeros(fan_out)

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[LayerState]]:
        """
        Forward pass through the network.

        This is simultaneously inference and a conservation check.
        No separate training/inference modes.

        Parameters:
            x: Input data [batch_size, input_dim]

        Returns:
            output: Network output [batch_size, output_dim]
            layer_states: States at each layer for PAC analysis
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        states = []
        current = x

        # Input layer state
        v_input = compute_value(current, self.config.value_fn)
        targets = compute_target_values(v_input, self.topology.num_layers)

        states.append(LayerState(
            activations=current.copy(),
            pre_activations=current.copy(),
            value=v_input,
            target_value=targets[0],
            violation=0.0,
        ))

        # Hidden and output layers
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            pre_act = current @ W.T + b
            if i < len(self.weights) - 1:
                current = activation_fn(pre_act, self.config.activation)
            else:
                # Output layer: identity activation for regression
                current = pre_act

            v = compute_value(current, self.config.value_fn)
            states.append(LayerState(
                activations=current.copy(),
                pre_activations=pre_act.copy(),
                value=v,
                target_value=targets[i + 1],
                violation=v - targets[i + 1],
            ))

        self.layer_states = states
        return current, states

    def compute_violations(self) -> Dict[int, float]:
        """Compute PAC violations from current layer states."""
        values = [s.value for s in self.layer_states]
        return compute_pac_violations(values, self.topology.conservation_pairs)

    def update(self, x: np.ndarray, y_target: np.ndarray) -> Dict[str, float]:
        """
        Perform one PAC conservation descent step.

        Each layer updates independently using:
        1. Conservation correction — scale weights to match PAC ratios
        2. Direction correction — Hebbian update toward target

        Parameters:
            x: Input data [batch_size, input_dim]
            y_target: Target output [batch_size, output_dim]

        Returns:
            Dict of training metrics
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y_target.ndim == 1:
            y_target = y_target.reshape(1, -1)

        # Forward pass
        y_pred, states = self.forward(x)

        # Compute PAC violations
        violations = self.compute_violations()
        total_violation = sum(abs(v) for v in violations.values())

        # Output error
        output_error = y_target - y_pred
        mse = np.mean(output_error ** 2)

        # === PAC Target Propagation ===
        # Distribute output error through PAC structure (forward, not backward).
        # Each hidden layer gets a pseudo-target derived from:
        #   1. Its PAC-prescribed value ratio (how much value it should carry)
        #   2. The output error distributed proportionally to φ^(-k)
        # This gives every layer a meaningful directional learning signal
        # without backpropagating gradients through the chain rule.

        n_layers = len(self.weights)
        output_error_magnitude = np.mean(np.abs(output_error))

        # Compute PAC-weighted pseudo-targets for each hidden layer.
        # The output error is distributed forward: layer k gets a share
        # proportional to φ^(-(n-k)) — deeper layers get more of the error.
        layer_errors = []
        for layer_idx in range(n_layers):
            # Error share: deeper layers (closer to output) get more
            depth_weight = PHI_INV ** (n_layers - 1 - layer_idx)
            layer_errors.append(depth_weight)
        total_weight = sum(layer_errors) + 1e-10
        layer_errors = [e / total_weight for e in layer_errors]

        # === Layer updates (all parallel — no dependency between layers) ===

        for layer_idx in range(n_layers):
            W = self.weights[layer_idx]
            b = self.biases[layer_idx]
            a_in = states[layer_idx].activations      # Input to this weight matrix
            a_out = states[layer_idx + 1].activations  # Output of this weight matrix

            # --- Conservation correction ---
            # Two-pronged: gentle weight scaling (preserves direction) +
            # bias adjustment (controls magnitude). Split the conservation
            # rate: 20% to weights, 80% to biases. This keeps weights
            # near conservation while allowing direction correction to work.
            v_actual = states[layer_idx + 1].value
            v_target = states[layer_idx + 1].target_value

            if v_actual > 1e-10:
                rho = v_target / v_actual
                rho = np.clip(rho, 1.0 - self.config.max_correction,
                              1.0 + self.config.max_correction)
                # Gentle multiplicative weight correction (20% of rate)
                w_factor = 1.0 + 0.2 * self.config.conservation_rate * (rho - 1.0)
                W *= w_factor
                # Additive bias correction (80% of rate)
                bias_correction = 0.8 * self.config.conservation_rate * (rho - 1.0)
                b += bias_correction * np.sign(b + 1e-8)

            # --- Direction correction ---
            if layer_idx == n_layers - 1:
                # Output layer: direct error correction (delta rule)
                dW = output_error.T @ a_in / x.shape[0]
                db = np.mean(output_error, axis=0)
                W += self.config.direction_rate * dW
                b += self.config.direction_rate * db
            else:
                # Hidden layers: PAC-structured direction correction
                # Two signals combined:
                #   A) Conservation violation → Hebbian scale adjustment
                #   B) Direct feedback from output error (one-hop, not backprop)
                error_share = layer_errors[layer_idx]

                # Signal A: conservation violation → scale adjustment
                v_err = states[layer_idx + 1].violation
                if abs(v_err) > 1e-10:
                    sign = -np.sign(v_err)
                    hebbian = a_out.T @ a_in / x.shape[0]
                    scale = min(abs(v_err), self.config.max_correction)
                    W += self.config.direction_rate * sign * scale * hebbian

                # Signal B: Direct feedback alignment
                # Project output error into hidden layer's space using
                # the downstream weight matrices (single hop per layer,
                # NOT recursive chain rule — this is direct feedback).
                # For the penultimate layer: use output weights directly.
                # For deeper layers: use PAC-scaled fixed projection.
                hid_dim = a_out.shape[1]
                hops_to_output = n_layers - 1 - layer_idx

                if hops_to_output == 1:
                    # Penultimate layer: project through output weights
                    # pseudo_error = output_error @ W_output (direct feedback)
                    W_next = self.weights[layer_idx + 1]
                    pseudo_error = output_error @ W_next  # [batch, hid_dim]
                else:
                    # Deeper layers: project through a PAC-scaled fixed matrix
                    # This avoids recursive chain rule while still providing
                    # a directional signal.
                    proj_rng = np.random.RandomState(
                        self.config_seed + layer_idx)
                    out_dim = output_error.shape[1]
                    proj = proj_rng.randn(out_dim, hid_dim)
                    proj *= np.sqrt(PHI_INV / max(out_dim, 1))
                    pseudo_error = output_error @ proj

                # Apply activation derivative for proper scaling
                pre_act = states[layer_idx + 1].pre_activations
                act_deriv = activation_derivative(
                    pre_act, self.config.activation)
                pseudo_error = pseudo_error * act_deriv

                # Delta rule with PAC-weighted error share
                dW = (error_share * pseudo_error).T @ a_in / x.shape[0]
                db = error_share * np.mean(pseudo_error, axis=0)
                W += self.config.direction_rate * dW
                b += self.config.direction_rate * db

            self.weights[layer_idx] = W
            self.biases[layer_idx] = b

        self.step_count += 1
        self.violation_history.append(total_violation)

        return {
            'mse': mse,
            'total_violation': total_violation,
            'violations': violations,
            'step': self.step_count,
        }

    def train(self, X: np.ndarray, Y: np.ndarray,
              epochs: int = 100,
              batch_size: Optional[int] = None,
              verbose: bool = False) -> List[Dict[str, float]]:
        """
        Train the network using PAC conservation descent.

        Parameters:
            X: Training inputs [n_samples, input_dim]
            Y: Training targets [n_samples, output_dim]
            epochs: Number of passes through the data
            batch_size: Mini-batch size (None = full batch)
            verbose: Print progress

        Returns:
            List of training metrics per epoch
        """
        n_samples = X.shape[0]
        if batch_size is None:
            batch_size = n_samples

        history = []
        for epoch in range(epochs):
            # Shuffle
            perm = self.rng.permutation(n_samples)
            X_shuffled = X[perm]
            Y_shuffled = Y[perm]

            epoch_mse = 0.0
            epoch_violation = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                x_batch = X_shuffled[start:end]
                y_batch = Y_shuffled[start:end]

                metrics = self.update(x_batch, y_batch)
                epoch_mse += metrics['mse']
                epoch_violation += metrics['total_violation']
                n_batches += 1

            epoch_mse /= n_batches
            epoch_violation /= n_batches

            record = {
                'epoch': epoch,
                'mse': epoch_mse,
                'total_violation': epoch_violation,
            }
            history.append(record)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch:4d}: MSE={epoch_mse:.6f}, "
                      f"PAC violation={epoch_violation:.6f}")

        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for inference (same as training — no mode switch)."""
        output, _ = self.forward(x)
        return output
