"""BodyLoop — closed-loop orchestrator for brain-in-a-body evaluation.

Each tick: observe environment -> encode via sensory channels ->
merge into single FieldState -> bus.process() -> decode motor action ->
step environment. Records full trajectory for behavioral analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from gaia.core.types import FieldState

from .environment import Environment, Observation
from .motor import Action, MotorDecoder
from .senses import SensoryChannel

if TYPE_CHECKING:
    from gaia.core.bus import ConservationBus


@dataclass
class TickRecord:
    """Record of a single body-loop tick."""

    tick: int
    observation: Observation
    field_state_in: FieldState
    field_state_out: FieldState
    action: Action
    bus_violations: int = 0


@dataclass
class Trajectory:
    """Full trajectory of a body-loop run."""

    ticks: list[TickRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ticks)

    @property
    def actions(self) -> list[Action]:
        """All actions taken during the trajectory."""
        return [t.action for t in self.ticks]

    @property
    def observations(self) -> list[Observation]:
        """All observations received during the trajectory."""
        return [t.observation for t in self.ticks]


def _merge_field_states(states: list[FieldState]) -> FieldState:
    """Merge multiple sensory FieldStates into a single bus input.

    Concatenates tensors, averages entropy, takes the highest phase.
    """
    if len(states) == 1:
        return states[0]

    tensors = [s.tensor.flatten() for s in states]
    merged_tensor = torch.cat(tensors)
    avg_entropy = sum(s.entropy for s in states) / len(states)

    # Take the most energetic phase (highest ordinal)
    phase_order = {p: i for i, p in enumerate(
        ["crystallized", "ordered", "transitional", "chaotic"]
    )}
    max_phase = max(states, key=lambda s: phase_order.get(s.phase.value, 0)).phase

    return FieldState(
        tensor=merged_tensor,
        entropy=avg_entropy,
        phase=max_phase,
    )


class BodyLoop:
    """Closed-loop orchestrator: environment <-> sensory <-> bus <-> motor.

    Args:
        bus: ConservationBus with registered GAIA modules.
        channels: Sensory channels for encoding observations.
        decoder: Motor decoder for extracting actions.
        env: Environment that produces observations and accepts actions.
    """

    def __init__(
        self,
        bus: ConservationBus,
        channels: list[SensoryChannel],
        decoder: MotorDecoder,
        env: Environment,
    ) -> None:
        self._bus = bus
        self._channels = channels
        self._decoder = decoder
        self._env = env
        self._tick_count = 0
        self._trajectory = Trajectory()

    def tick(self) -> TickRecord:
        """Execute one body-loop cycle.

        1. Get observation from environment (or reset on first tick)
        2. Encode via sensory channels
        3. Merge into single FieldState
        4. Process through ConservationBus
        5. Decode motor action
        6. Step environment
        7. Record and return tick
        """
        # 1. Observe
        if self._tick_count == 0:
            obs = self._env.reset()
        else:
            # Use the previous action to step (already done at end of last tick)
            # On first tick after reset, we just observe
            obs = self._last_obs

        # 2. Encode each sensory channel
        encoded: list[FieldState] = []
        for i, channel in enumerate(self._channels):
            if i == 0:
                raw = obs.visual
            else:
                raw = obs.proprioceptive
            encoded.append(channel.encode(raw))

        # 3. Merge
        field_state_in = _merge_field_states(encoded)

        # 4. Process through bus
        violations_before = len(self._bus.violation_log)
        field_state_out = self._bus.process(field_state_in)
        violations_after = len(self._bus.violation_log)

        # 5. Decode action
        action = self._decoder.decode(field_state_out)

        # 6. Step environment
        next_obs = self._env.step(action)
        self._last_obs = next_obs

        # 7. Record
        record = TickRecord(
            tick=self._tick_count,
            observation=obs,
            field_state_in=field_state_in,
            field_state_out=field_state_out,
            action=action,
            bus_violations=violations_after - violations_before,
        )
        self._trajectory.ticks.append(record)
        self._tick_count += 1

        return record

    def run(self, n_ticks: int) -> Trajectory:
        """Run the body loop for n_ticks and return the trajectory."""
        for _ in range(n_ticks):
            self.tick()
        return self._trajectory

    @property
    def trajectory(self) -> Trajectory:
        """Current trajectory (accumulated across all ticks)."""
        return self._trajectory

    def reset(self) -> None:
        """Reset the loop for a new run."""
        self._tick_count = 0
        self._trajectory = Trajectory()
        self._env.reset()
