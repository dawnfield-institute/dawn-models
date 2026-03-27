"""Spike H -- Population of Identical GAIA Organisms.

Each organism is a complete GAIA core -- all 5 modules (Safety, Reasoning,
Memory, Language, Observability) working together through a CoupledFieldsBus.
Like single-cell organisms, each has all the machinery for intelligence.

Multiple identical organisms interact through field resonance. The question:
do they differentiate? Like cells in an embryo -- same DNA, different fates
based on position and signals from neighbors.

Usage:
    cd dawn-models/research/GAIA
    PYTHONPATH="src;../../fracton" python spikes/population/spike_h_organisms.py
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

_root = Path(__file__).resolve().parents[2]
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))
_fracton = _root.parents[0] / "fracton"
if str(_fracton) not in sys.path:
    sys.path.insert(0, str(_fracton))

from gaia.core.coupled_fields_bus import CoupledFieldsBus, _harmonic_resonance
from gaia.core.types import FieldState, RBFBalance, SECPhase
from gaia.modules.safety import SafetyModule
from gaia.modules.reasoning import ReasoningModule
from gaia.modules.memory import MemoryModule
from gaia.modules.language import LanguageModule
from gaia.modules.observability import ObservabilityModule
from gaia.network import GAIAAgent

DIM = 22
PHI_INV = 0.6180339887498949


def make_organism(name: str) -> GAIAAgent:
    """Create a complete GAIA organism -- all 5 modules.

    Every organism is identical at birth. Differentiation
    comes purely from interaction dynamics.
    """
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


@dataclass
class Signal:
    """A broadcast from one organism."""
    sender: str
    tensor: torch.Tensor
    tick: int


class Organism:
    """A complete GAIA cell that communicates with neighbors."""

    def __init__(self, name: str):
        self.agent = make_organism(name)
        self.voice = torch.zeros(DIM)
        self.listening: dict[str, float] = {}
        self.tick = 0
        self.last_signal: Signal | None = None
        # Track internal module specializations over time
        self.spec_history: list[dict[str, float]] = []

    @property
    def name(self) -> str:
        return self.agent.name

    def perceive(self, env: FieldState, signals: list[Signal]) -> FieldState:
        """Combine environment with signals from other organisms."""
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
                combined = PHI_INV * combined + (1.0 - PHI_INV) * signal_sum

        # PAC-scale
        ce = float(torch.sum(combined).item())
        if abs(ce) > 1e-10:
            combined = combined * (env_energy / ce)

        return FieldState(
            tensor=combined, entropy=env.entropy, phase=env.phase,
            conservation_budget=0.0, provenance=[], timestamp=time.time(),
        )

    def process(self, input_state: FieldState) -> Signal:
        """Full GAIA core processes input, produces signal."""
        output = self.agent.process(input_state)
        self.voice = PHI_INV * self.voice + (1.0 - PHI_INV) * output.tensor
        self.tick += 1

        # Record internal module lens specializations
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
        """Update listening weights based on coherence."""
        for sig in signals:
            if sig.sender == self.name:
                continue
            coherence = _harmonic_resonance(sig.tensor, my_signal.tensor)
            old = self.listening.get(sig.sender, 0.5)
            self.listening[sig.sender] = 0.95 * old + 0.05 * max(0.0, coherence)


class Colony:
    """A colony of identical GAIA organisms."""

    def __init__(self, n_organisms: int):
        self.organisms = {
            f"cell_{i}": Organism(f"cell_{i}")
            for i in range(n_organisms)
        }
        self.tick = 0

    def step(self, env: FieldState) -> list[Signal]:
        """One colony tick -- all organisms act simultaneously."""
        # Gather last signals
        prev = [o.last_signal for o in self.organisms.values() if o.last_signal]

        # Perceive
        inputs = {
            name: org.perceive(env, prev)
            for name, org in self.organisms.items()
        }

        # Process
        signals = {}
        for name, org in self.organisms.items():
            signals[name] = org.process(inputs[name])

        # Update trust
        sig_list = list(signals.values())
        for org in self.organisms.values():
            org.update_trust(sig_list, signals[org.name])

        self.tick += 1
        return sig_list

    def run(self, n_ticks: int, env_fn, print_every: int = 25):
        for t in range(n_ticks):
            env = env_fn(self.tick + t)
            self.step(env)
            if (t + 1) % print_every == 0:
                self.print_status()

    def print_status(self):
        print(f"\n{'='*70}")
        print(f"  TICK {self.tick}")
        print(f"{'='*70}")

        names = list(self.organisms.keys())

        # Identity divergence: how different are the organisms?
        print(f"\n  ORGANISM IDENTITIES:")
        for name, org in self.organisms.items():
            spec = org.agent.identity.specialization
            exp = org.agent.identity.experience

            # Who does this organism listen to most?
            if org.listening:
                top = max(org.listening, key=org.listening.get)
                top_w = org.listening[top]
            else:
                top, top_w = "none", 0.0

            # Internal module diversity (are internal modules specializing differently?)
            if org.spec_history:
                latest = org.spec_history[-1]
                mod_specs = ", ".join(f"{k}={v:.2f}" for k, v in latest.items())
            else:
                mod_specs = "n/a"

            print(f"    {name}: spec={spec:.3f}, exp={exp}, "
                  f"trusts={top}({top_w:.3f})")
            print(f"      modules: {mod_specs}")

        # Voice divergence matrix
        print(f"\n  VOICE DIVERGENCE (cross-resonance):")
        header = "            " + "  ".join(f"{n:>8s}" for n in names)
        print(header)
        for i, ni in enumerate(names):
            row = f"    {ni:8s}"
            for j, nj in enumerate(names):
                if i == j:
                    row += "     --- "
                else:
                    r = _harmonic_resonance(
                        self.organisms[ni].voice,
                        self.organisms[nj].voice,
                    )
                    row += f"  {r:+6.3f}"
            print(row)

        # Language learning comparison
        print(f"\n  LANGUAGE LEARNING:")
        for name, org in self.organisms.items():
            for mod_name, mod in org.agent.bus._modules.items():
                if hasattr(mod, '_counter'):
                    s = mod._counter.stats
                    print(f"    {name}/{mod_name}: "
                          f"{s.total_transitions} trans, {s.unique_contexts} ctx")

    def report(self):
        """Final differentiation report."""
        names = list(self.organisms.keys())
        n = len(names)

        print(f"\n{'='*70}")
        print(f"  DIFFERENTIATION REPORT -- {self.tick} ticks, {n} organisms")
        print(f"{'='*70}")

        # Did organisms differentiate?
        specs = [self.organisms[name].agent.identity.specialization for name in names]
        voices = [self.organisms[name].voice for name in names]

        print(f"\n  SPECIALIZATION SCORES:")
        for name, spec in zip(names, specs):
            bar = "#" * int(spec * 10)
            print(f"    {name}: {spec:.4f}  {bar}")

        # Mean pairwise voice divergence
        divergences = []
        for i in range(n):
            for j in range(i + 1, n):
                d = float(torch.norm(voices[i] - voices[j]).item())
                r = _harmonic_resonance(voices[i], voices[j])
                divergences.append((names[i], names[j], d, r))

        print(f"\n  PAIRWISE DIVERGENCE:")
        for ni, nj, d, r in divergences:
            print(f"    {ni} vs {nj}: L2={d:.4f}, resonance={r:+.4f}")

        mean_d = sum(x[2] for x in divergences) / len(divergences) if divergences else 0
        mean_r = sum(x[3] for x in divergences) / len(divergences) if divergences else 0
        print(f"\n  MEAN: L2={mean_d:.4f}, resonance={mean_r:+.4f}")

        if mean_r < 0.9:
            print(f"\n  >> DIFFERENTIATION DETECTED: organisms developed distinct voices")
        else:
            print(f"\n  >> MINIMAL DIFFERENTIATION: organisms remain similar")

        # Communication graph -- who trusts whom?
        print(f"\n  TRUST NETWORK:")
        for name, org in self.organisms.items():
            if org.listening:
                sorted_w = sorted(org.listening.items(), key=lambda x: -x[1])
                top3 = sorted_w[:3]
                prefs = ", ".join(f"{k}={v:.3f}" for k, v in top3)
                print(f"    {name} -> {prefs}")

        # Internal module specialization comparison
        print(f"\n  INTERNAL MODULE LENSES (final):")
        for name, org in self.organisms.items():
            if org.spec_history:
                latest = org.spec_history[-1]
                mods = "  ".join(f"{k}={v:.3f}" for k, v in latest.items())
                print(f"    {name}: {mods}")

        # Did internal modules specialize differently ACROSS organisms?
        if all(org.spec_history for org in self.organisms.values()):
            print(f"\n  CROSS-ORGANISM MODULE COMPARISON:")
            all_latest = {
                name: org.spec_history[-1]
                for name, org in self.organisms.items()
            }
            # For each module type, compare across organisms
            mod_names = list(next(iter(all_latest.values())).keys())
            for mod in mod_names:
                vals = [all_latest[name].get(mod, 0.0) for name in names]
                mean_v = sum(vals) / len(vals)
                std_v = (sum((v - mean_v)**2 for v in vals) / len(vals)) ** 0.5
                print(f"    {mod:20s}: mean={mean_v:.3f}, std={std_v:.3f}  "
                      f"{'** DIVERGED' if std_v > 0.1 else ''}")


# ─── Environments ──────────────────────────────────────────────────

def env_varied(tick: int) -> FieldState:
    """Diverse signals -- different frequency content each tick."""
    torch.manual_seed(tick * 17 + 7)
    tensor = torch.randn(DIM)
    # Add some structure
    freq = (tick % 5) + 1
    tensor += torch.sin(torch.arange(DIM, dtype=torch.float32) * freq * 0.3) * 0.5
    return FieldState(
        tensor=tensor, entropy=1.5, phase=SECPhase.ORDERED,
        conservation_budget=0.0, provenance=[], timestamp=time.time(),
    )


def env_constant(tick: int) -> FieldState:
    """Same signal every tick -- tests: does differentiation survive without novelty?"""
    torch.manual_seed(42)
    return FieldState(
        tensor=torch.randn(DIM), entropy=1.0, phase=SECPhase.ORDERED,
        conservation_budget=0.0, provenance=[], timestamp=time.time(),
    )


def env_adversarial(tick: int) -> FieldState:
    """Sudden regime shifts every 20 ticks."""
    regime = tick // 20
    torch.manual_seed(regime * 1000 + tick)
    tensor = torch.randn(DIM) * (1.0 + regime * 0.5)
    # Each regime lights up different dimensions
    start = (regime * 7) % DIM
    end = min(start + 8, DIM)
    tensor[start:end] *= 3.0
    entropy = 1.0 + regime * 0.5
    return FieldState(
        tensor=tensor, entropy=min(entropy, 5.0), phase=SECPhase.ORDERED,
        conservation_budget=0.0, provenance=[], timestamp=time.time(),
    )


# ─── Main ──────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  SPIKE H -- Identical Organism Differentiation")
    print("  Same DNA (all 5 modules), different fates from interaction")
    print("=" * 70)

    # 5 identical organisms
    colony = Colony(n_organisms=5)

    # Phase 1: diverse environment (100 ticks)
    print(f"\n--- Phase 1: Diverse Signals (100 ticks) ---")
    colony.run(100, env_fn=env_varied, print_every=50)

    # Phase 2: constant signal (50 ticks) -- does structure persist?
    print(f"\n--- Phase 2: Constant Signal (50 ticks) ---")
    colony.run(50, env_fn=env_constant, print_every=50)

    # Phase 3: adversarial regime shifts (50 ticks) -- stress test
    print(f"\n--- Phase 3: Regime Shifts (50 ticks) ---")
    colony.run(50, env_fn=env_adversarial, print_every=50)

    colony.report()


if __name__ == "__main__":
    main()
