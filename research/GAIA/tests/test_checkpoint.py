"""Tests for colony checkpoint save/load.

Validates that save_colony/load_colony round-trips all state:
module weights, learned transitions, PAC trees, trust networks,
voice vectors, identity, and bus field states.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from gaia.core.coupled_fields_bus import _harmonic_resonance
from gaia.core.types import FieldState, SECPhase
from gaia.modules.safety import SafetyModule
from gaia.modules.reasoning import ReasoningModule
from gaia.modules.memory import MemoryModule
from gaia.modules.language import LanguageModule
from gaia.modules.observability import ObservabilityModule
from gaia.network import GAIAAgent, save_colony, load_colony, checkpoint_info

DIM = 22


# ─── Test Colony Infrastructure ──────────────────────────────────
# Minimal Colony/Organism/Signal that mirrors spike_h_organisms
# but lives in the test so there's no import path issues.

@dataclass
class Signal:
    sender: str
    tensor: torch.Tensor
    tick: int


def _make_agent(name: str) -> GAIAAgent:
    return GAIAAgent(
        name,
        [
            SafetyModule(input_dim=DIM),
            ReasoningModule(input_dim=DIM),
            MemoryModule(),
            LanguageModule(),
            ObservabilityModule(),
        ],
        field_dim=DIM,
    )


class Organism:
    def __init__(self, name: str):
        self.agent = _make_agent(name)
        self.voice = torch.zeros(DIM)
        self.listening: dict[str, float] = {}
        self.tick = 0
        self.last_signal: Signal | None = None
        self.spec_history: list[dict[str, float]] = []

    @property
    def name(self) -> str:
        return self.agent.name

    def perceive(self, env: FieldState, signals: list[Signal]) -> FieldState:
        combined = env.tensor.clone()
        env_energy = env.total_energy()
        others = [s for s in signals if s.sender != self.name]
        if others:
            signal_sum = torch.zeros(DIM)
            total_w = 0.0
            for sig in others:
                if torch.norm(self.voice) > 1e-6:
                    resonance = max(0.0, _harmonic_resonance(sig.tensor, self.voice))
                else:
                    resonance = 1.0 / len(others)
                trust = self.listening.get(sig.sender, 0.5)
                w = resonance * trust
                signal_sum += w * sig.tensor
                total_w += w
            if total_w > 1e-10:
                signal_sum /= total_w
                phi_inv = 0.6180339887498949
                combined = phi_inv * combined + (1.0 - phi_inv) * signal_sum
        ce = float(torch.sum(combined).item())
        if abs(ce) > 1e-10:
            combined = combined * (env_energy / ce)
        return FieldState(
            tensor=combined, entropy=env.entropy, phase=env.phase,
            conservation_budget=0.0, provenance=[], timestamp=time.time(),
        )

    def process(self, input_state: FieldState) -> Signal:
        output = self.agent.process(input_state)
        phi_inv = 0.6180339887498949
        self.voice = phi_inv * self.voice + (1.0 - phi_inv) * output.tensor
        self.tick += 1
        bus_states = self.agent.bus.field_states
        if bus_states:
            self.spec_history.append({
                name: float(state.lens.std().item())
                for name, state in bus_states.items()
            })
        sig = Signal(sender=self.name, tensor=output.tensor.clone(), tick=self.tick)
        self.last_signal = sig
        return sig

    def update_trust(self, signals: list[Signal], my_signal: Signal):
        for sig in signals:
            if sig.sender == self.name:
                continue
            coherence = _harmonic_resonance(sig.tensor, my_signal.tensor)
            old = self.listening.get(sig.sender, 0.5)
            self.listening[sig.sender] = 0.95 * old + 0.05 * max(0.0, coherence)


class Colony:
    def __init__(self, n_organisms: int = 0):
        if n_organisms > 0:
            self.organisms = {
                f"cell_{i}": Organism(f"cell_{i}")
                for i in range(n_organisms)
            }
        else:
            self.organisms = {}
        self.tick = 0

    def step(self, env: FieldState) -> list[Signal]:
        prev = [o.last_signal for o in self.organisms.values() if o.last_signal]
        inputs = {
            name: org.perceive(env, prev)
            for name, org in self.organisms.items()
        }
        signals = {}
        for name, org in self.organisms.items():
            signals[name] = org.process(inputs[name])
        sig_list = list(signals.values())
        for org in self.organisms.values():
            org.update_trust(sig_list, signals[org.name])
        self.tick += 1
        return sig_list


def _make_env(tick: int) -> FieldState:
    torch.manual_seed(tick * 17 + 7)
    tensor = torch.randn(DIM)
    freq = (tick % 5) + 1
    tensor += torch.sin(torch.arange(DIM, dtype=torch.float32) * freq * 0.3) * 0.5
    return FieldState(
        tensor=tensor, entropy=1.5, phase=SECPhase.ORDERED,
        conservation_budget=0.0, provenance=[], timestamp=time.time(),
    )


# ─── Tests ────────────────────────────────────────────────────────

class TestSaveLoad:
    """Core round-trip tests."""

    def test_save_and_load_basic(self, tmp_path):
        """Colony state survives save/load."""
        colony = Colony(n_organisms=3)

        # Run 10 ticks to accumulate state
        for t in range(10):
            colony.step(_make_env(t))

        path = tmp_path / "colony.pt"
        save_colony(colony, path)
        assert path.exists()

        loaded = load_colony(path, Organism, Colony, Signal)

        assert loaded.tick == colony.tick
        assert set(loaded.organisms.keys()) == set(colony.organisms.keys())

    def test_voice_preserved(self, tmp_path):
        """Voice vectors match after round-trip."""
        colony = Colony(n_organisms=2)
        for t in range(15):
            colony.step(_make_env(t))

        path = tmp_path / "voice.pt"
        save_colony(colony, path)
        loaded = load_colony(path, Organism, Colony, Signal)

        for name in colony.organisms:
            orig_voice = colony.organisms[name].voice
            loaded_voice = loaded.organisms[name].voice
            assert torch.allclose(orig_voice, loaded_voice, atol=1e-6), (
                f"{name} voice mismatch"
            )

    def test_trust_preserved(self, tmp_path):
        """Listening weights match after round-trip."""
        colony = Colony(n_organisms=3)
        for t in range(20):
            colony.step(_make_env(t))

        path = tmp_path / "trust.pt"
        save_colony(colony, path)
        loaded = load_colony(path, Organism, Colony, Signal)

        for name in colony.organisms:
            orig = colony.organisms[name].listening
            rest = loaded.organisms[name].listening
            assert set(orig.keys()) == set(rest.keys()), f"{name} trust keys differ"
            for k in orig:
                assert abs(orig[k] - rest[k]) < 1e-6, f"{name} trust[{k}] mismatch"

    def test_identity_preserved(self, tmp_path):
        """Agent identity (lens, resonance, experience) preserved."""
        colony = Colony(n_organisms=2)
        for t in range(20):
            colony.step(_make_env(t))

        path = tmp_path / "identity.pt"
        save_colony(colony, path)
        loaded = load_colony(path, Organism, Colony, Signal)

        for name in colony.organisms:
            orig_id = colony.organisms[name].agent.identity
            load_id = loaded.organisms[name].agent.identity
            assert torch.allclose(orig_id.resonance_field, load_id.resonance_field, atol=1e-6)
            assert torch.allclose(orig_id.spectral_lens, load_id.spectral_lens, atol=1e-6)
            assert orig_id.experience == load_id.experience

    def test_last_signal_preserved(self, tmp_path):
        """Last signal is saved and restored."""
        colony = Colony(n_organisms=2)
        for t in range(5):
            colony.step(_make_env(t))

        path = tmp_path / "signal.pt"
        save_colony(colony, path)
        loaded = load_colony(path, Organism, Colony, Signal)

        for name in colony.organisms:
            orig = colony.organisms[name].last_signal
            rest = loaded.organisms[name].last_signal
            assert orig is not None
            assert rest is not None
            assert orig.sender == rest.sender
            assert orig.tick == rest.tick
            assert torch.allclose(orig.tensor, rest.tensor, atol=1e-6)


class TestBusState:
    """Bus internal state (field states, module weights) preserved."""

    def test_bus_tick_preserved(self, tmp_path):
        """Bus tick counter matches after round-trip."""
        colony = Colony(n_organisms=2)
        for t in range(10):
            colony.step(_make_env(t))

        path = tmp_path / "bus.pt"
        save_colony(colony, path)
        loaded = load_colony(path, Organism, Colony, Signal)

        for name in colony.organisms:
            orig_tick = colony.organisms[name].agent.bus._tick
            load_tick = loaded.organisms[name].agent.bus._tick
            assert orig_tick == load_tick, f"{name} bus tick mismatch"

    def test_field_states_preserved(self, tmp_path):
        """Per-module CoupledFieldState (tensor, lens) preserved."""
        colony = Colony(n_organisms=2)
        for t in range(15):
            colony.step(_make_env(t))

        path = tmp_path / "fs.pt"
        save_colony(colony, path)
        loaded = load_colony(path, Organism, Colony, Signal)

        for name in colony.organisms:
            orig_fs = colony.organisms[name].agent.bus._field_states
            load_fs = loaded.organisms[name].agent.bus._field_states
            assert set(orig_fs.keys()) == set(load_fs.keys()), (
                f"{name} field state keys differ"
            )
            for mod_name in orig_fs:
                assert torch.allclose(
                    orig_fs[mod_name].tensor,
                    load_fs[mod_name].tensor,
                    atol=1e-6,
                ), f"{name}/{mod_name} tensor mismatch"
                assert torch.allclose(
                    orig_fs[mod_name].lens,
                    load_fs[mod_name].lens,
                    atol=1e-6,
                ), f"{name}/{mod_name} lens mismatch"


class TestModuleState:
    """Individual module internal state preserved."""

    def test_language_counter_preserved(self, tmp_path):
        """Language module's TransitionCounter state survives round-trip."""
        colony = Colony(n_organisms=2)
        for t in range(30):
            colony.step(_make_env(t))

        path = tmp_path / "lang.pt"
        save_colony(colony, path)
        loaded = load_colony(path, Organism, Colony, Signal)

        for name in colony.organisms:
            orig_mods = colony.organisms[name].agent.bus._modules
            load_mods = loaded.organisms[name].agent.bus._modules
            for mod_name, mod in orig_mods.items():
                if hasattr(mod, '_counter'):
                    orig_stats = mod._counter.stats
                    load_stats = load_mods[mod_name]._counter.stats
                    assert orig_stats.total_transitions == load_stats.total_transitions, (
                        f"{name}/{mod_name} transition count mismatch"
                    )
                    assert orig_stats.unique_contexts == load_stats.unique_contexts

    def test_observability_tracker_preserved(self, tmp_path):
        """Observability module's SCBF tracker state survives round-trip."""
        colony = Colony(n_organisms=2)
        for t in range(20):
            colony.step(_make_env(t))

        path = tmp_path / "obs.pt"
        save_colony(colony, path)
        loaded = load_colony(path, Organism, Colony, Signal)

        for name in colony.organisms:
            orig_mods = colony.organisms[name].agent.bus._modules
            load_mods = loaded.organisms[name].agent.bus._modules
            for mod_name, mod in orig_mods.items():
                if hasattr(mod, '_tracker'):
                    assert mod._tracker._step == load_mods[mod_name]._tracker._step
                    assert list(mod._tracker._raw_entropies) == list(
                        load_mods[mod_name]._tracker._raw_entropies
                    )


class TestContinuation:
    """Colony continues correctly after load."""

    def test_loaded_colony_continues(self, tmp_path):
        """Loaded colony can continue processing without errors."""
        colony = Colony(n_organisms=3)
        for t in range(10):
            colony.step(_make_env(t))

        path = tmp_path / "cont.pt"
        save_colony(colony, path)
        loaded = load_colony(path, Organism, Colony, Signal)

        # Continue running for 10 more ticks -- should not crash
        for t in range(10, 20):
            signals = loaded.step(_make_env(t))
            assert len(signals) == 3

        assert loaded.tick == 20

    def test_state_diverges_from_fresh(self, tmp_path):
        """A loaded colony produces different outputs than a fresh one at the same tick."""
        colony = Colony(n_organisms=2)
        for t in range(20):
            colony.step(_make_env(t))

        path = tmp_path / "diverge.pt"
        save_colony(colony, path)
        loaded = load_colony(path, Organism, Colony, Signal)

        # Fresh colony at tick 0
        fresh = Colony(n_organisms=2)

        # Process same signal through both
        env = _make_env(99)
        loaded_sigs = loaded.step(env)
        fresh_sigs = fresh.step(env)

        # Outputs should differ because loaded has 20 ticks of learned state
        loaded_out = loaded_sigs[0].tensor
        fresh_out = fresh_sigs[0].tensor
        diff = float(torch.norm(loaded_out - fresh_out).item())
        assert diff > 0.01, "Loaded colony should produce different output than fresh"


class TestCheckpointInfo:
    """checkpoint_info reads metadata without loading full state."""

    def test_info_basic(self, tmp_path):
        colony = Colony(n_organisms=3)
        for t in range(10):
            colony.step(_make_env(t))

        path = tmp_path / "info.pt"
        save_colony(colony, path)

        info = checkpoint_info(path)
        assert info["version"] == 1
        assert info["tick"] == 10
        assert len(info["organisms"]) == 3
        for name, org_info in info["organisms"].items():
            assert "modules" in org_info
            assert "experience" in org_info
            assert len(org_info["modules"]) == 5  # all 5 module types
