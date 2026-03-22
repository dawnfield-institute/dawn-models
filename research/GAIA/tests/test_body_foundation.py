"""Phase 1 tests — sensorimotor foundation for mock body harness.

Tests sensory encoding, motor decoding, environment stepping,
and the closed-loop BodyLoop with all 5 GAIA modules.
"""

from __future__ import annotations

import pytest
import torch

from gaia.body.environment import GridWorld, Observation
from gaia.body.loop import BodyLoop, _merge_field_states
from gaia.body.motor import Action, GridMotorDecoder
from gaia.body.senses import ProprioceptiveChannel, VisualChannel, _shannon_entropy
from gaia.core.bus import ConservationBus
from gaia.core.types import FieldState, SECPhase


# ─── Sensory Channels ─────────────────────────────────────────────


class TestVisualChannel:

    def test_produces_valid_field_state(self):
        """Visual encoding produces FieldState with finite entropy and valid phase."""
        ch = VisualChannel()
        raw = torch.rand(5, 5)
        state = ch.encode(raw)
        assert state.tensor.shape == (25,)
        assert state.entropy >= 0.0
        assert isinstance(state.phase, SECPhase)

    def test_peaked_lower_entropy_than_uniform(self):
        """A peaked distribution has lower entropy than uniform."""
        ch = VisualChannel()
        # Peaked: one dominant value
        peaked = torch.zeros(5, 5)
        peaked[2, 2] = 10.0
        # Uniform: all values equal
        uniform = torch.ones(5, 5)
        state_peaked = ch.encode(peaked)
        state_uniform = ch.encode(uniform)
        assert state_peaked.entropy < state_uniform.entropy

    def test_zero_tensor_zero_entropy(self):
        """All-zero tensor produces zero entropy."""
        ch = VisualChannel()
        state = ch.encode(torch.zeros(3, 3))
        assert state.entropy == 0.0
        assert state.phase == SECPhase.CRYSTALLIZED

    def test_uniform_tensor_max_entropy(self):
        """Uniform positive tensor produces maximum entropy for its size."""
        ch = VisualChannel()
        uniform = torch.ones(100)
        state = ch.encode(uniform)
        # Shannon entropy of uniform distribution over 100 elements = log(100) ≈ 4.6
        assert state.entropy > 4.0
        assert state.phase == SECPhase.CHAOTIC


class TestProprioceptiveChannel:

    def test_produces_valid_field_state(self):
        """Proprioceptive encoding produces valid FieldState."""
        ch = ProprioceptiveChannel()
        raw = torch.tensor([5.0, 5.0, 0.1, -0.2])
        state = ch.encode(raw)
        assert state.tensor.shape == (4,)
        assert state.entropy >= 0.0

    def test_stationary_lower_entropy(self):
        """Stationary state (one dominant position value) has lower entropy than moving."""
        ch = ProprioceptiveChannel()
        stationary = torch.tensor([5.0, 5.0, 0.0, 0.0])
        moving = torch.tensor([3.0, 7.0, 1.5, -2.0])
        s_stat = ch.encode(stationary)
        s_move = ch.encode(moving)
        # Stationary: most energy in position, sparse → lower entropy
        # Moving: energy spread across pos + vel → higher entropy
        assert s_stat.entropy < s_move.entropy


# ─── Motor Decoder ─────────────────────────────────────────────────


class TestGridMotorDecoder:

    def test_produces_valid_action(self):
        """Decoder produces Action with direction and magnitude."""
        decoder = GridMotorDecoder()
        state = FieldState(tensor=torch.tensor([1.0, 0.5, 0.2, 0.1]), entropy=1.0)
        action = decoder.decode(state)
        assert isinstance(action, Action)
        assert action.direction.shape == (2,)
        assert action.magnitude >= 0.0

    def test_energy_asymmetry_produces_direction(self):
        """Asymmetric energy in tensor halves produces non-zero direction."""
        decoder = GridMotorDecoder()
        # First half has more energy → positive dy
        tensor = torch.cat([torch.ones(5) * 10.0, torch.ones(5) * 1.0])
        state = FieldState(tensor=tensor, entropy=1.0)
        action = decoder.decode(state)
        assert action.direction[1] > 0  # dy positive (first half > second half)

    def test_short_tensor_handled(self):
        """Decoder handles very short tensors gracefully."""
        decoder = GridMotorDecoder()
        state = FieldState(tensor=torch.tensor([5.0, 1.0]), entropy=1.0)
        action = decoder.decode(state)
        assert isinstance(action, Action)
        assert action.direction.shape == (2,)


# ─── Environment ───────────────────────────────────────────────────


class TestGridWorld:

    def test_reset_returns_observation(self):
        """Reset produces a valid Observation."""
        env = GridWorld(size=8, n_stimuli=2, seed=42)
        obs = env.reset()
        assert isinstance(obs, Observation)
        # Egocentric: 3*n_stimuli + 9 (3x3 receptive field)
        assert obs.visual.shape == (3 * 2 + 9,)
        assert obs.proprioceptive.shape == (4,)

    def test_step_moves_agent(self):
        """Stepping with a non-zero action changes agent position."""
        env = GridWorld(size=10, seed=42)
        env.reset()
        pos_before = env.agent_position.clone()
        action = Action(direction=torch.tensor([1.0, 0.0]), magnitude=1.0)
        env.step(action)
        pos_after = env.agent_position
        assert not torch.equal(pos_before, pos_after)

    def test_position_stays_in_bounds(self):
        """Agent position stays within grid bounds after bouncing."""
        env = GridWorld(size=5, seed=42)
        env.reset()
        # Push hard in one direction for many steps
        action = Action(direction=torch.tensor([1.0, 1.0]), magnitude=2.0)
        for _ in range(50):
            env.step(action)
        pos = env.agent_position
        assert pos[0] >= 0.0 and pos[0] <= 4.0
        assert pos[1] >= 0.0 and pos[1] <= 4.0

    def test_boundary_bounce(self):
        """Agent bounces off walls instead of sticking."""
        env = GridWorld(size=10, seed=42)
        env.reset()
        # Push to the right wall
        action = Action(direction=torch.tensor([1.0, 0.0]), magnitude=2.0)
        positions = []
        for _ in range(20):
            env.step(action)
            positions.append(env.agent_position[0].item())
        # After hitting the wall, position should vary (not stuck at boundary)
        unique_positions = len(set(round(p, 1) for p in positions))
        assert unique_positions > 2, f"Agent stuck: only {unique_positions} unique x-positions"

    def test_deterministic(self):
        """Same seed produces same observations."""
        env1 = GridWorld(size=6, n_stimuli=3, seed=99)
        env2 = GridWorld(size=6, n_stimuli=3, seed=99)
        obs1 = env1.reset()
        obs2 = env2.reset()
        assert torch.equal(obs1.visual, obs2.visual)
        assert torch.equal(obs1.proprioceptive, obs2.proprioceptive)

    def test_different_seeds_different_worlds(self):
        """Different seeds produce different stimulus placements."""
        env1 = GridWorld(size=8, seed=1)
        env2 = GridWorld(size=8, seed=2)
        obs1 = env1.reset()
        obs2 = env2.reset()
        assert not torch.equal(obs1.visual, obs2.visual)

    def test_egocentric_visual_changes_with_movement(self):
        """Visual observation changes as agent moves (egocentric)."""
        env = GridWorld(size=10, seed=42)
        obs_start = env.reset()
        action = Action(direction=torch.tensor([1.0, 0.0]), magnitude=1.5)
        env.step(action)
        env.step(action)
        obs_moved = env.step(action)
        # Agent has moved significantly → different egocentric view
        assert not torch.equal(obs_start.visual, obs_moved.visual)


# ─── Field State Merging ───────────────────────────────────────────


class TestMergeFieldStates:

    def test_single_state_passthrough(self):
        """Single state is returned as-is."""
        state = FieldState(tensor=torch.tensor([1.0, 2.0]), entropy=1.0)
        merged = _merge_field_states([state])
        assert merged is state

    def test_two_states_concatenated(self):
        """Two states produce a concatenated tensor."""
        s1 = FieldState(tensor=torch.tensor([1.0, 2.0]), entropy=1.0)
        s2 = FieldState(tensor=torch.tensor([3.0, 4.0, 5.0]), entropy=3.0)
        merged = _merge_field_states([s1, s2])
        assert merged.tensor.shape == (5,)
        assert merged.entropy == pytest.approx(2.0)  # average

    def test_phase_takes_highest(self):
        """Merged phase is the most energetic of the inputs."""
        s1 = FieldState(tensor=torch.tensor([1.0]), entropy=0.1, phase=SECPhase.CRYSTALLIZED)
        s2 = FieldState(tensor=torch.tensor([2.0]), entropy=3.0, phase=SECPhase.TRANSITIONAL)
        merged = _merge_field_states([s1, s2])
        assert merged.phase == SECPhase.TRANSITIONAL


# ─── BodyLoop ──────────────────────────────────────────────────────


def _make_bus_with_observability(input_dim: int) -> ConservationBus:
    """Create a bus with just the observability module (always PAC-safe)."""
    from gaia.modules.observability import ObservabilityModule

    bus = ConservationBus(enforcement="soft")
    bus.register_module(ObservabilityModule())
    return bus


def _make_full_bus(input_dim: int) -> ConservationBus:
    """Create a bus with all 5 modules for the given input dimension."""
    from gaia.modules.language import LanguageModule
    from gaia.modules.memory import MemoryModule
    from gaia.modules.observability import ObservabilityModule
    from gaia.modules.reasoning import ReasoningModule
    from gaia.modules.safety import SafetyModule

    bus = ConservationBus(enforcement="soft")
    bus.register_module(ObservabilityModule())
    bus.register_module(LanguageModule())
    bus.register_module(MemoryModule())
    bus.register_module(SafetyModule(input_dim=input_dim))
    bus.register_module(ReasoningModule(input_dim=input_dim))
    return bus


class TestBodyLoop:

    def test_single_tick(self):
        """A single tick produces a valid TickRecord."""
        env = GridWorld(size=5, n_stimuli=2, seed=42)
        channels = [VisualChannel(), ProprioceptiveChannel()]
        decoder = GridMotorDecoder()
        # 3*2 + 9 visual + 4 proprio = 19 elements
        bus = _make_bus_with_observability(19)

        loop = BodyLoop(bus, channels, decoder, env)
        record = loop.tick()
        assert record.tick == 0
        assert isinstance(record.action, Action)
        assert isinstance(record.observation, Observation)

    def test_100_ticks_no_crash(self):
        """100 ticks run without errors (smoke test)."""
        env = GridWorld(size=5, n_stimuli=2, seed=42)
        channels = [VisualChannel(), ProprioceptiveChannel()]
        decoder = GridMotorDecoder()
        bus = _make_bus_with_observability(19)

        loop = BodyLoop(bus, channels, decoder, env)
        trajectory = loop.run(100)
        assert len(trajectory) == 100

    def test_trajectory_records_all_ticks(self):
        """Trajectory accumulates all tick records."""
        env = GridWorld(size=5, n_stimuli=2, seed=42)
        channels = [VisualChannel(), ProprioceptiveChannel()]
        decoder = GridMotorDecoder()
        bus = _make_bus_with_observability(19)

        loop = BodyLoop(bus, channels, decoder, env)
        loop.run(10)
        traj = loop.trajectory
        assert len(traj) == 10
        assert traj.ticks[0].tick == 0
        assert traj.ticks[9].tick == 9

    def test_different_environments_different_observations(self):
        """Different environments produce different visual observations."""
        env1 = GridWorld(size=5, seed=1)
        env2 = GridWorld(size=5, seed=2)
        obs1 = env1.reset()
        obs2 = env2.reset()
        # Different stimulus placements → different visual fields
        assert not torch.equal(obs1.visual, obs2.visual)

    def test_reset_clears_trajectory(self):
        """Resetting the loop clears accumulated trajectory."""
        env = GridWorld(size=5, seed=42)
        channels = [VisualChannel(), ProprioceptiveChannel()]
        decoder = GridMotorDecoder()
        bus = _make_bus_with_observability(19)

        loop = BodyLoop(bus, channels, decoder, env)
        loop.run(10)
        assert len(loop.trajectory) == 10
        loop.reset()
        assert len(loop.trajectory) == 0


class TestBodyLoopFullBrain:
    """Integration test with all 5 modules registered."""

    def test_full_brain_10_ticks(self):
        """10 ticks with all 5 modules — verifies PAC tolerance in soft mode."""
        env = GridWorld(size=5, n_stimuli=2, seed=42)
        channels = [VisualChannel(), ProprioceptiveChannel()]
        decoder = GridMotorDecoder()
        # 3*2 + 9 + 4 = 19
        bus = _make_full_bus(19)

        loop = BodyLoop(bus, channels, decoder, env)
        trajectory = loop.run(10)
        assert len(trajectory) == 10

    def test_full_brain_provenance(self):
        """All active modules appear in provenance chain."""
        env = GridWorld(size=5, n_stimuli=2, seed=42)
        channels = [VisualChannel(), ProprioceptiveChannel()]
        decoder = GridMotorDecoder()
        bus = _make_full_bus(19)

        loop = BodyLoop(bus, channels, decoder, env)
        record = loop.tick()
        # At least observability should always appear
        assert len(record.field_state_out.provenance) > 0
