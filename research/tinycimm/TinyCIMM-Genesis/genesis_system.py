"""
TinyCIMM-Genesis: Self-Organizing Dynamical System

NOT a neural network. A self-referential dynamical system with measurement
instruments. No data, no loss function, no target. Only structural constraints:
symmetry + self-application.

Initialize randomly, run, measure what constants emerge from dynamics alone.
The M10 derivation chain predicts: phi, Xi, gamma/ln(phi) should appear
without being inserted.

Key design: modulation rates EVOLVE via meta-modulation. Not fixed at
phi^(-1/N) — that would insert the answer. The entropy-seeking meta-rule
finds the sweet spot. M10 predicts it converges to phi^(-1/N).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from spectral_utils import (
    symmetric_eigendecomposition,
    anti_hebbian_modulate,
    hierarchy_entropy,
    cascade_depth,
)


@dataclass
class GenesisConfig:
    """Configuration for a Genesis self-organizing system."""
    N: int = 16                        # System size
    initial_sr: float = 1.0            # Initial spectral radius (NOT 1.2)
    initial_weak: float = 0.9          # Initial weak factor (NOT phi-derived)
    initial_strong: float = 1.02       # Initial strong factor
    meta_rate: float = 0.001           # Meta-modulation step size
    meta_threshold: float = 0.01       # Entropy change threshold for meta-modulation
    weak_clamp: tuple = (0.80, 0.999)  # Clamp range for weak factor
    eigval_clamp: tuple = (0.01, 10.0) # Prevent blow-up
    seed: int = 42


class GenesisSystem:
    """
    Self-organizing dynamical system with symmetric self-application.

    Dynamics:
        state(t+1) = tanh(W(t) @ state(t))
        W(t+1) = anti-Hebbian modulation of W(t) eigenvalues
        weak_factor evolves via meta-modulation (entropy-seeking)

    No data, no loss, no target. Measure what emerges.
    """

    def __init__(self, config: Optional[GenesisConfig] = None):
        self.config = config or GenesisConfig()
        self.N = self.config.N
        self.rng = np.random.RandomState(self.config.seed)

        # Random symmetric coupling matrix
        W = self.rng.randn(self.N, self.N) / np.sqrt(self.N)
        W = (W + W.T) / 2
        # Normalize to initial spectral radius
        eigvals = np.linalg.eigvalsh(W)
        sr = np.max(np.abs(eigvals))
        if sr > 1e-10:
            W = W * (self.config.initial_sr / sr)
        self.W = W

        # Random initial state
        self.state = self.rng.randn(self.N) * 0.5

        # Evolving modulation rates
        self.weak_factor = self.config.initial_weak
        self.strong_factor = self.config.initial_strong

        # Evolving target spectral radius
        self.target_sr = self.config.initial_sr

        # Tracking for meta-modulation
        self.prev_depth = cascade_depth(self.W)
        self.prev_entropy = hierarchy_entropy(np.linalg.eigvalsh(self.W))

        # History
        self.step_count = 0
        self.history: List[Dict] = []

    def step(self):
        """One step of self-referential dynamics + meta-modulation."""
        # 1. State update
        self.state = np.tanh(self.W @ self.state)

        # 2. Eigendecompose
        eigvals, eigvecs = symmetric_eigendecomposition(self.W)

        # 3. Record pre-modulation spectral radius
        pre_sr = float(np.max(np.abs(eigvals)))

        # 4. Anti-Hebbian modulation (with evolving rates)
        #    No target_sr here — we handle sr separately
        self.W, new_eigvals = anti_hebbian_modulate(
            eigvals, eigvecs, self.state,
            self.weak_factor, self.strong_factor,
            target_sr=None,
        )

        # 5. Re-normalize sr to evolving target (isolate ratio effects)
        #    Modulation should change eigenvalue RATIOS, not overall SCALE.
        #    Without this, scale drift dominates and meta-modulation chases
        #    scale effects instead of finding the viability threshold.
        post_sr = float(np.max(np.abs(new_eigvals)))
        if post_sr > 1e-10 and self.target_sr > 1e-10:
            scale = self.target_sr / post_sr
            new_eigvals = new_eigvals * scale
            self.W = eigvecs @ np.diag(new_eigvals) @ eigvecs.T

        # 6. Meta-modulate: evolve weak_factor AND target_sr
        self._meta_modulate()

        self.step_count += 1

    def _meta_modulate(self):
        """
        Evolve weak_factor and target_sr based on cascade depth.

        Cascade depth (number of active eigenvalue modes) is the direct
        measure of system viability. If modes are dying → back off.
        If modes are stable → can be more aggressive.

        Two evolving parameters:
          - weak_factor: how aggressively to weaken dominant modes
          - target_sr: the overall spectral scale

        M10 predictions: weak_factor → phi^(-1/N), target_sr → gamma/ln(phi).
        """
        current_depth = cascade_depth(self.W)
        current_entropy = hierarchy_entropy(np.linalg.eigvalsh(self.W))
        depth_delta = current_depth - self.prev_depth
        entropy_delta = current_entropy - self.prev_entropy

        # Weak factor: evolve based on cascade depth
        if depth_delta < 0:
            # Losing modes — back off (less aggressive)
            self.weak_factor += self.config.meta_rate * 2
        elif depth_delta == 0 and entropy_delta < -self.config.meta_threshold:
            # Modes stable but entropy dropping — mild back off
            self.weak_factor += self.config.meta_rate * 0.5
        elif depth_delta >= 0 and entropy_delta > self.config.meta_threshold:
            # Healthy — can be more aggressive
            self.weak_factor -= self.config.meta_rate

        # Target sr: evolve based on hierarchy health
        # If entropy is low (modes concentrated), increase sr to spread energy
        # If entropy is high (modes diffuse), decrease sr to concentrate
        max_entropy = np.log(self.N)  # Maximum possible entropy
        entropy_frac = current_entropy / max_entropy if max_entropy > 0 else 0.5
        if entropy_frac < 0.3:
            self.target_sr += self.config.meta_rate * 0.5
        elif entropy_frac > 0.7:
            self.target_sr -= self.config.meta_rate * 0.5

        # Clamp both
        lo, hi = self.config.weak_clamp
        self.weak_factor = max(lo, min(hi, self.weak_factor))
        self.target_sr = max(0.5, min(3.0, self.target_sr))

        self.prev_depth = current_depth
        self.prev_entropy = current_entropy

    def run(self, n_steps, record_every=1):
        """
        Run for n_steps, recording state and diagnostics.

        Returns trajectory [n_recorded, N] and diagnostics list.
        """
        trajectory = []
        diagnostics = []

        for t in range(n_steps):
            self.step()

            if t % record_every == 0:
                trajectory.append(self.state.copy())
                diagnostics.append(self.snapshot())

        return np.array(trajectory), diagnostics

    def snapshot(self):
        """Current state diagnostics."""
        eigvals = np.linalg.eigvalsh(self.W)
        sr = float(np.max(np.abs(eigvals)))
        depth = cascade_depth(self.W)
        entropy = hierarchy_entropy(eigvals)
        state_norm = float(np.linalg.norm(self.state))

        return {
            'step': self.step_count,
            'weak_factor': float(self.weak_factor),
            'target_sr': float(self.target_sr),
            'spectral_radius': sr,
            'cascade_depth': depth,
            'hierarchy_entropy': entropy,
            'state_norm': state_norm,
            'eigenvalues': eigvals.tolist(),
        }

    def reset(self, seed=None):
        """Reset to fresh random state."""
        if seed is not None:
            self.config.seed = seed
        self.__init__(self.config)
