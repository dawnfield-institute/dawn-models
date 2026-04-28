"""
Ghost Core: Symmetric Recurrent Core with Spectral Confinement

The core is a square symmetric matrix W_core with:
  - sr = gamma/ln(phi) = 1.1995 (from M10/Genesis exp_02)
  - Anti-Hebbian eigenvalue modulation at phi^(-1/N) rate (from M10/Genesis exp_01)
  - K recurrent steps of tanh(W @ h)
  - Eigenvectors are FIXED (spectral confinement / PAC)

The core does NOT learn via gradients. It learns via spectral modulation:
the eigenvalue spectrum reorganizes through anti-Hebbian dynamics while
eigenvectors remain fixed. This is a fundamentally different learning
mechanism from backpropagation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from spectral_utils import (
    PHI, PHI_INV, LN_PHI, GAMMA_EM, SCOPE_RATIO,
    symmetric_eigendecomposition, anti_hebbian_modulate,
    hierarchy_entropy, cascade_depth,
)


@dataclass
class CoreConfig:
    """Configuration for the symmetric recurrent core."""
    core_dim: int = 16
    K: int = 3                        # Recurrent steps
    target_sr: float = SCOPE_RATIO    # gamma/ln(phi) = 1.1995
    weak_factor: Optional[float] = None  # phi^(-1/N), auto-computed if None
    strong_factor: float = 1.01
    seed: int = 42


class SymmetricRecurrentCore:
    """
    Symmetric recurrent core with spectral confinement.

    Forward pass: h → K steps of tanh(W @ h) with anti-Hebbian modulation.
    No gradient flows through the core — only spectral modulation.
    """

    def __init__(self, config: Optional[CoreConfig] = None):
        self.config = config or CoreConfig()
        self.N = self.config.core_dim
        self.K = self.config.K
        self.rng = np.random.RandomState(self.config.seed)

        # Compute weak_factor from N if not specified
        if self.config.weak_factor is None:
            self.weak_factor = PHI ** (-1.0 / self.N)
        else:
            self.weak_factor = self.config.weak_factor

        # Initialize symmetric W
        W = self.rng.randn(self.N, self.N) / np.sqrt(self.N)
        W = (W + W.T) / 2
        eigvals = np.linalg.eigvalsh(W)
        sr = np.max(np.abs(eigvals))
        if sr > 1e-10:
            W = W * (self.config.target_sr / sr)
        self.W = W

        # Track eigenvectors for confinement verification
        _, self.initial_eigvecs = np.linalg.eigh(self.W)

    def forward(self, h: np.ndarray, modulate: bool = False) -> np.ndarray:
        """
        Forward pass: K recurrent steps through the symmetric core.

        Parameters:
            h: Input state [batch_size, core_dim]
            modulate: If True, apply anti-Hebbian modulation (for self-org).
                      If False, just pass through fixed W (for data processing).

        Returns:
            h_out: Output state [batch_size, core_dim]
        """
        if h.ndim == 1:
            h = h.reshape(1, -1)

        # K recurrent steps
        for k in range(self.K):
            h = np.tanh(h @ self.W.T)

            if modulate:
                mean_state = np.mean(h, axis=0)
                self._modulate(mean_state)

        return h

    def _modulate(self, state: np.ndarray):
        """Anti-Hebbian eigenvalue modulation."""
        eigvals, eigvecs = np.linalg.eigh(self.W)

        # Activity projections
        projections = (eigvecs.T @ state) ** 2
        total = np.sum(projections) + 1e-10
        activities = projections / total
        mean_act = 1.0 / self.N

        # Modulate eigenvalues
        modulation = np.ones(self.N)
        modulation[activities > 2.0 * mean_act] = self.weak_factor
        modulation[activities < 0.5 * mean_act] = self.config.strong_factor
        new_eigvals = eigvals * modulation

        # Re-normalize sr
        post_sr = np.max(np.abs(new_eigvals))
        if post_sr > 1e-10:
            new_eigvals = new_eigvals * (self.config.target_sr / post_sr)

        self.W = eigvecs @ np.diag(new_eigvals) @ eigvecs.T

    def eigenvector_drift(self) -> float:
        """
        Measure eigenvector drift from initialization (should be ~0).
        Uses best-match alignment (not positional) because eigh sorts by
        eigenvalue, and modulation changes eigenvalue ordering.
        """
        _, current_eigvecs = np.linalg.eigh(self.W)
        alignment_matrix = np.abs(current_eigvecs.T @ self.initial_eigvecs)
        best_alignment = np.max(alignment_matrix, axis=1)
        return float(1.0 - np.mean(best_alignment))

    def spectral_radius(self) -> float:
        """Current spectral radius."""
        return float(np.max(np.abs(np.linalg.eigvalsh(self.W))))

    def snapshot(self) -> dict:
        """Diagnostic snapshot."""
        eigvals = np.linalg.eigvalsh(self.W)
        return {
            'spectral_radius': float(np.max(np.abs(eigvals))),
            'eigvec_drift': self.eigenvector_drift(),
            'cascade_depth': cascade_depth(self.W),
            'hierarchy_entropy': hierarchy_entropy(eigvals),
            'weak_factor': float(self.weak_factor),
        }
