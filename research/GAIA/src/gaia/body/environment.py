"""Environments — closed-loop worlds that respond to motor actions.

Each environment produces observations and accepts actions, forming
the outer loop of the body harness. All environments are deterministic
(seeded RNG) for reproducible behavioral tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .motor import Action


@dataclass
class Observation:
    """Sensory observation from the environment."""

    visual: torch.Tensor        # 2D grid or flattened visual field
    proprioceptive: torch.Tensor  # [x, y, vx, vy] agent state


class Environment(Protocol):
    """Protocol for closed-loop environments."""

    def reset(self) -> Observation:
        """Reset to initial state and return first observation."""
        ...

    def step(self, action: Action) -> Observation:
        """Apply action and return resulting observation."""
        ...


class GridWorld:
    """Simple 2D grid with an agent and stimulus objects.

    The agent moves on a grid. Stimulus objects at fixed positions
    generate visual patterns based on proximity — closer objects
    produce stronger signals. Fully deterministic.

    Args:
        size: Grid dimensions (size x size).
        n_stimuli: Number of stimulus objects placed on the grid.
        seed: Random seed for stimulus placement.
    """

    def __init__(self, size: int = 10, n_stimuli: int = 3, seed: int = 42) -> None:
        self._size = size
        self._n_stimuli = n_stimuli
        self._rng = torch.Generator().manual_seed(seed)

        # Place stimuli at random grid positions with random intensities
        self._stimuli_pos = torch.randint(0, size, (n_stimuli, 2), generator=self._rng).float()
        self._stimuli_intensity = torch.rand(n_stimuli, generator=self._rng) * 2.0 + 0.5

        # Agent state
        self._pos = torch.tensor([size / 2.0, size / 2.0])
        self._vel = torch.zeros(2)
        self._tick = 0

    def reset(self) -> Observation:
        """Reset agent to center, zero velocity."""
        self._pos = torch.tensor([self._size / 2.0, self._size / 2.0])
        self._vel = torch.zeros(2)
        self._tick = 0
        return self._observe()

    def step(self, action: Action) -> Observation:
        """Move agent by action direction * magnitude, bouncing off walls."""
        # Update velocity (exponential smoothing with action)
        move = action.direction.float() * min(action.magnitude, 2.0)
        self._vel = 0.3 * self._vel + 0.7 * move

        # Update position
        self._pos = self._pos + self._vel

        # Bounce off walls (reflect velocity on collision)
        bound = float(self._size - 1)
        for dim in range(2):
            if self._pos[dim] < 0:
                self._pos[dim] = -self._pos[dim]
                self._vel[dim] = -self._vel[dim]
            elif self._pos[dim] > bound:
                self._pos[dim] = 2 * bound - self._pos[dim]
                self._vel[dim] = -self._vel[dim]
        # Safety clamp (in case of large velocities causing double-bounce)
        self._pos = self._pos.clamp(0.0, bound)

        self._tick += 1
        return self._observe()

    def _observe(self) -> Observation:
        """Generate egocentric observation from current state.

        Visual field is agent-relative:
        - Per-stimulus: (dx, dy, perceived_intensity) relative to agent
        - Local receptive field: 3x3 grid centered on agent position
        Total visual tensor: 3*n_stimuli + 9 elements.

        Proprioceptive: [x, y, vx, vy].
        """
        ax, ay = self._pos[0].item(), self._pos[1].item()

        # Stimulus proximity vector (egocentric)
        stim_features = []
        for i in range(self._n_stimuli):
            sx, sy = self._stimuli_pos[i]
            dx = float(sx) - ax
            dy = float(sy) - ay
            dist_sq = dx * dx + dy * dy
            perceived = float(self._stimuli_intensity[i]) / (1.0 + dist_sq)
            stim_features.extend([dx, dy, perceived])

        # Local 3x3 receptive field centered on agent
        receptive = torch.zeros(3, 3)
        cx, cy = int(round(ax)), int(round(ay))
        for di in range(-1, 2):
            for dj in range(-1, 2):
                gx, gy = cx + di, cy + dj
                if 0 <= gx < self._size and 0 <= gy < self._size:
                    # Sum stimulus contributions at this cell
                    val = 0.0
                    for k in range(self._n_stimuli):
                        sx, sy = self._stimuli_pos[k]
                        d2 = (gx - float(sx)) ** 2 + (gy - float(sy)) ** 2
                        val += float(self._stimuli_intensity[k]) / (1.0 + d2)
                    receptive[di + 1, dj + 1] = val

        visual = torch.cat([torch.tensor(stim_features), receptive.flatten()])

        # Proprioceptive state
        proprio = torch.tensor([ax, ay, self._vel[0].item(), self._vel[1].item()])

        return Observation(visual=visual, proprioceptive=proprio)

    @property
    def agent_position(self) -> torch.Tensor:
        """Current agent position (read-only)."""
        return self._pos.clone()

    @property
    def tick(self) -> int:
        """Current tick count."""
        return self._tick
