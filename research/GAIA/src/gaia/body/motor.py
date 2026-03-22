"""Motor decoders — extract actions from output FieldState tensors.

Decoders read the bus output but do NOT modify it — they translate
the processed FieldState into actions the environment can execute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from gaia.core.types import FieldState


@dataclass
class Action:
    """An action the body can perform in the environment."""

    direction: torch.Tensor  # unit-ish movement vector (2D for grid)
    magnitude: float = 0.0
    metadata: dict = field(default_factory=dict)


class MotorDecoder(Protocol):
    """Protocol for decoding FieldState into environment actions."""

    def decode(self, state: FieldState) -> Action:
        """Extract an action from a processed FieldState."""
        ...


class GridMotorDecoder:
    """Decodes FieldState into 2D grid movement.

    Reads the full tensor's energy distribution to determine direction.
    Splits the tensor into halves/interleaved to extract vertical and
    horizontal energy gradients. Module transformations (safety, reasoning,
    memory) all influence the motor output through energy redistribution.
    """

    def decode(self, state: FieldState) -> Action:
        """Decode a FieldState into a grid movement action.

        Compares energy in tensor halves (vertical) and interleaved
        elements (horizontal) to produce a continuous direction vector.
        """
        flat = state.tensor.flatten().float()
        n = flat.numel()

        if n < 2:
            return Action(direction=torch.tensor([0.0, 0.0]), magnitude=0.0)

        half = n // 2

        # Vertical axis: first half vs second half energy
        top_energy = float(flat[:half].sum().item())
        bottom_energy = float(flat[half:].sum().item())
        dy = top_energy - bottom_energy

        # Horizontal axis: even indices vs odd indices energy
        left_energy = float(flat[::2].sum().item())
        right_energy = float(flat[1::2].sum().item())
        dx = left_energy - right_energy

        direction = torch.tensor([dx, dy])
        norm = float(direction.norm().item())
        if norm > 1e-8:
            direction = direction / norm

        magnitude = float(flat.abs().mean().item())

        return Action(direction=direction, magnitude=magnitude)
