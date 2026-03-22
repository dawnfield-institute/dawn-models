"""Sensory channels — encode raw observations into FieldState tensors.

Each channel maps a specific modality (visual, proprioceptive) to a
FieldState with naturally-computed entropy and SEC phase classification.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch

from gaia.core.types import FieldState, SECPhase


# DFT-derived SEC thresholds (matching sec_router.py)
_SEC_BOUNDS = [(0.5, SECPhase.CRYSTALLIZED), (2.0, SECPhase.ORDERED), (4.0, SECPhase.TRANSITIONAL)]


def _classify_phase(entropy: float) -> SECPhase:
    """Classify entropy into SEC phase."""
    for threshold, phase in _SEC_BOUNDS:
        if entropy < threshold:
            return phase
    return SECPhase.CHAOTIC


def _shannon_entropy(tensor: torch.Tensor) -> float:
    """Compute Shannon entropy from a tensor's value distribution.

    Normalizes absolute values to a probability distribution, then
    computes H = -sum(p * log(p)). Returns 0.0 for zero/empty tensors.
    """
    flat = tensor.flatten().float().abs()
    total = flat.sum()
    if total < 1e-12 or flat.numel() == 0:
        return 0.0
    probs = flat / total
    # Filter zeros to avoid log(0)
    mask = probs > 1e-12
    probs = probs[mask]
    return float(-(probs * probs.log()).sum().item())


class SensoryChannel(Protocol):
    """Protocol for encoding raw observations into FieldState."""

    def encode(self, raw: torch.Tensor) -> FieldState:
        """Map raw sensory data to a FieldState on the bus."""
        ...


class VisualChannel:
    """Encodes 2D visual patterns (grids) into FieldState.

    The tensor is the flattened grid. Entropy is Shannon entropy of
    the pixel distribution — structured patterns (gradients) produce
    lower entropy than noise.
    """

    def encode(self, raw: torch.Tensor) -> FieldState:
        """Encode a visual observation (any shape) into FieldState.

        Args:
            raw: Visual data tensor (e.g., 2D grid of intensities).

        Returns:
            FieldState with flattened tensor and computed entropy/phase.
        """
        flat = raw.flatten().float()
        entropy = _shannon_entropy(flat)
        phase = _classify_phase(entropy)
        return FieldState(tensor=flat, entropy=entropy, phase=phase)


class ProprioceptiveChannel:
    """Encodes position/velocity state into FieldState.

    Low entropy for stationary states (near-zero velocity),
    high entropy for rapid/varied movement.
    """

    def encode(self, raw: torch.Tensor) -> FieldState:
        """Encode proprioceptive state (position + velocity) into FieldState.

        Args:
            raw: 1D tensor of [x, y, vx, vy, ...] state values.

        Returns:
            FieldState with the state vector and computed entropy/phase.
        """
        flat = raw.flatten().float()
        entropy = _shannon_entropy(flat)
        phase = _classify_phase(entropy)
        return FieldState(tensor=flat, entropy=entropy, phase=phase)
