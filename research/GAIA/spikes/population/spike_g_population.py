"""Spike G -- Population of Independent GAIA Entities.

NOT a GAIANetwork (shared bus). Independent entities that:
1. Perceive a shared environment signal
2. Process through their own GAIA core
3. Broadcast their output as a "message" to the population
4. Receive other entities' messages, filtered through their lens
5. Combine environment + messages -> next input
6. Repeat

The question: do they develop communication patterns?
Do they specialize? Does collective behavior emerge that
no single entity could produce alone?

Usage:
    cd dawn-models/research/GAIA
    PYTHONPATH="src;../../fracton" python spikes/population/spike_g_population.py
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

# Add src to path for standalone running
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

DIM = 22  # Standard GAIA field dimension
PHI_INV = 0.6180339887498949


# ─── Population Entity ─────────────────────────────────────────────

@dataclass
class Message:
    """A broadcast from one entity to the population."""
    sender: str
    tensor: torch.Tensor
    entropy: float
    tick: int


@dataclass
class EntityState:
    """Observable state of an entity at a given tick."""
    name: str
    tick: int
    output_energy: float
    specialization: float
    phase: SECPhase
    messages_received: int
    resonance_with_others: dict[str, float] = field(default_factory=dict)


class PopulationEntity:
    """An independent GAIA agent that communicates with a population.

    Each entity:
    - Has its own CoupledFieldsBus with real modules
    - Maintains a "voice" (accumulated output pattern)
    - Listens to others through its spectral lens
    - Develops communication preferences (who to listen to)
    """

    def __init__(self, name: str, modules: list, field_dim: int = DIM):
        self.agent = GAIAAgent(name, modules, field_dim=field_dim)
        self._voice = torch.zeros(field_dim)  # accumulated output signature
        self._listening_weights: dict[str, float] = {}  # who do I listen to?
        self._tick = 0
        self._message_history: list[Message] = []

    @property
    def name(self) -> str:
        return self.agent.name

    @property
    def voice(self) -> torch.Tensor:
        return self._voice

    def perceive(
        self,
        environment: FieldState,
        messages: list[Message],
    ) -> FieldState:
        """Combine environment signal with messages from other entities.

        Each message is weighted by:
        1. Harmonic resonance between message and this entity's voice
           (do I resonate with what you're saying?)
        2. Listening weight history (have I found you useful before?)

        The combined signal = environment + blended messages, PAC-scaled.
        """
        combined = environment.tensor.clone()
        env_energy = environment.total_energy()

        if messages:
            # Filter messages through this entity's perspective
            message_sum = torch.zeros_like(combined)
            total_weight = 0.0

            for msg in messages:
                if msg.sender == self.name:
                    continue  # don't listen to yourself

                # Resonance-based attention: how much does this message
                # resonate with my accumulated voice?
                if torch.norm(self._voice) > 1e-6:
                    resonance = max(0.0, _harmonic_resonance(msg.tensor, self._voice))
                else:
                    resonance = 1.0 / len(messages)  # flat attention initially

                # Historical listening weight (trust built over time)
                history_weight = self._listening_weights.get(msg.sender, 0.5)

                weight = resonance * history_weight
                message_sum += weight * msg.tensor
                total_weight += weight

            if total_weight > 1e-10:
                message_sum = message_sum / total_weight

                # Blend: environment is primary, messages are secondary
                # PHI_INV of attention goes to environment, (1-PHI_INV) to messages
                combined = PHI_INV * combined + (1.0 - PHI_INV) * message_sum

        # PAC-scale to preserve environment energy
        combined_energy = float(torch.sum(combined).item())
        if abs(combined_energy) > 1e-10:
            combined = combined * (env_energy / combined_energy)

        return FieldState(
            tensor=combined,
            entropy=environment.entropy,
            phase=environment.phase,
            conservation_budget=environment.conservation_budget,
            provenance=[],
            timestamp=time.time(),
        )

    def process(self, input_state: FieldState) -> Message:
        """Process input and produce a broadcast message."""
        output = self.agent.process(input_state)

        # Update voice (running blend of outputs)
        self._voice = PHI_INV * self._voice + (1.0 - PHI_INV) * output.tensor

        self._tick += 1

        msg = Message(
            sender=self.name,
            tensor=output.tensor.clone(),
            entropy=output.entropy,
            tick=self._tick,
        )
        self._message_history.append(msg)
        return msg

    def update_listening_weights(self, messages: list[Message], my_output: Message):
        """Update who I listen to based on how useful their messages were.

        If listening to entity X made my output more coherent (lower entropy),
        increase X's weight. Otherwise decrease it.
        """
        for msg in messages:
            if msg.sender == self.name:
                continue
            # Coherence = harmonic resonance between their message and my output
            coherence = _harmonic_resonance(msg.tensor, my_output.tensor)
            old = self._listening_weights.get(msg.sender, 0.5)
            # Slow update toward coherence
            self._listening_weights[msg.sender] = 0.95 * old + 0.05 * max(0.0, coherence)

    def get_state(self, all_entities: list[PopulationEntity]) -> EntityState:
        """Snapshot of this entity's current state."""
        resonances = {}
        for other in all_entities:
            if other.name != self.name and torch.norm(other.voice) > 1e-6:
                resonances[other.name] = _harmonic_resonance(self._voice, other.voice)

        return EntityState(
            name=self.name,
            tick=self._tick,
            output_energy=float(torch.sum(self._voice).item()),
            specialization=self.agent.identity.specialization,
            phase=self.agent.phase(),
            messages_received=len(self._listening_weights),
            resonance_with_others=resonances,
        )


# ─── Population ────────────────────────────────────────────────────

class Population:
    """A collection of independent GAIA entities that interact."""

    def __init__(self, entities: list[PopulationEntity]):
        self.entities = {e.name: e for e in entities}
        self._tick = 0
        self._history: list[dict[str, EntityState]] = []

    def step(self, environment: FieldState) -> dict[str, Message]:
        """One population tick:

        1. All entities perceive (environment + last messages)
        2. All entities process (produce output)
        3. All entities broadcast (share output as message)
        4. All entities update listening weights

        Entities act simultaneously -- no ordering bias.
        """
        # Collect last tick's messages (empty on first tick)
        last_messages = []
        for entity in self.entities.values():
            if entity._message_history:
                last_messages.append(entity._message_history[-1])

        # 1. Perceive: each entity combines environment + messages
        inputs: dict[str, FieldState] = {}
        for entity in self.entities.values():
            inputs[entity.name] = entity.perceive(environment, last_messages)

        # 2+3. Process + broadcast
        messages: dict[str, Message] = {}
        for entity in self.entities.values():
            msg = entity.process(inputs[entity.name])
            messages[entity.name] = msg

        # 4. Update listening weights
        msg_list = list(messages.values())
        for entity in self.entities.values():
            entity.update_listening_weights(msg_list, messages[entity.name])

        # Record state
        states = {
            name: entity.get_state(list(self.entities.values()))
            for name, entity in self.entities.items()
        }
        self._history.append(states)
        self._tick += 1

        return messages

    def run(self, n_ticks: int, env_fn=None, print_every: int = 10):
        """Run the population for n_ticks.

        env_fn: callable(tick) -> FieldState. If None, uses random signals.
        """
        for tick in range(n_ticks):
            if env_fn:
                env = env_fn(tick)
            else:
                torch.manual_seed(tick * 7 + 13)
                env = FieldState(
                    tensor=torch.randn(DIM),
                    entropy=1.5,
                    phase=SECPhase.ORDERED,
                    conservation_budget=0.0,
                    provenance=[],
                    timestamp=time.time(),
                )

            self.step(env)

            if (tick + 1) % print_every == 0:
                self._print_status(tick + 1)

    def _print_status(self, tick: int):
        """Print population status."""
        states = self._history[-1]
        print(f"\n{'='*60}")
        print(f"  TICK {tick}")
        print(f"{'='*60}")

        for name, state in states.items():
            entity = self.entities[name]
            listening = entity._listening_weights
            top_listener = max(listening, key=listening.get) if listening else "none"
            top_weight = listening.get(top_listener, 0.0) if listening else 0.0

            print(f"\n  {name}:")
            print(f"    specialization: {state.specialization:.4f}")
            print(f"    phase:          {state.phase.value}")
            print(f"    listens to:     {top_listener} ({top_weight:.3f})")

            if state.resonance_with_others:
                res_str = ", ".join(
                    f"{k}={v:.3f}" for k, v in state.resonance_with_others.items()
                )
                print(f"    resonance:      {res_str}")

    def report(self):
        """Final population report."""
        if not self._history:
            print("No ticks run yet.")
            return

        states = self._history[-1]
        print(f"\n{'='*60}")
        print(f"  POPULATION REPORT -- {self._tick} ticks")
        print(f"{'='*60}")

        # Specialization
        print(f"\n  SPECIALIZATION:")
        for name, state in states.items():
            bar = "#" * int(state.specialization * 10)
            print(f"    {name:20s}: {state.specialization:.4f} {bar}")

        # Communication preferences (who listens to whom)
        print(f"\n  COMMUNICATION GRAPH:")
        for name, entity in self.entities.items():
            weights = entity._listening_weights
            if weights:
                sorted_w = sorted(weights.items(), key=lambda x: -x[1])
                prefs = ", ".join(f"{k}={v:.3f}" for k, v in sorted_w)
                print(f"    {name:20s} -> {prefs}")

        # Voice resonance matrix (who sounds like whom)
        print(f"\n  VOICE RESONANCE MATRIX:")
        names = list(self.entities.keys())
        header = "                      " + "  ".join(f"{n[:8]:>8s}" for n in names)
        print(header)
        for i, name_i in enumerate(names):
            voice_i = self.entities[name_i].voice
            row = f"    {name_i:20s}"
            for j, name_j in enumerate(names):
                if i == j:
                    row += "     --- "
                else:
                    voice_j = self.entities[name_j].voice
                    r = _harmonic_resonance(voice_i, voice_j)
                    row += f"   {r:+.3f}"
            print(row)

        # Communication emergence: do listening weights diverge from uniform?
        print(f"\n  COMMUNICATION EMERGENCE:")
        for name, entity in self.entities.items():
            weights = entity._listening_weights
            if len(weights) > 1:
                vals = list(weights.values())
                mean_w = sum(vals) / len(vals)
                std_w = (sum((v - mean_w) ** 2 for v in vals) / len(vals)) ** 0.5
                uniform = 1.0 / len(vals)
                divergence = std_w / uniform if uniform > 0 else 0.0
                print(f"    {name:20s}: weight_std={std_w:.4f}, divergence_from_uniform={divergence:.4f}")

        # Memory state (did Language modules learn transitions?)
        print(f"\n  LANGUAGE LEARNING:")
        for name, entity in self.entities.items():
            for mod_name, mod in entity.agent._bus._modules.items():
                if hasattr(mod, '_counter'):
                    stats = mod._counter.stats
                    print(f"    {name}/{mod_name}: {stats.total_transitions} transitions, "
                          f"{stats.unique_contexts} contexts")
                elif hasattr(mod, '_tree'):
                    tree_size = getattr(mod._tree, '_size', 0)
                    if hasattr(mod._tree, 'root') and mod._tree.root:
                        tree_size = _count_tree(mod._tree.root)
                    print(f"    {name}/{mod_name}: tree_size={tree_size}")


def _count_tree(node, depth=0):
    """Count nodes in a PACTree."""
    count = 1
    if hasattr(node, 'children'):
        for child in node.children:
            count += _count_tree(child, depth + 1)
    return count


# ─── Environments ──────────────────────────────────────────────────

def env_class_rotation(tick: int) -> FieldState:
    """Rotate through 3 stimulus classes -- tests differentiation."""
    class_id = tick % 3
    torch.manual_seed(42 + class_id * 1000 + tick)
    tensor = torch.zeros(DIM)
    # Each class has energy in different bands
    if class_id == 0:
        tensor[:7] = torch.randn(7) * 2.0  # low band
    elif class_id == 1:
        tensor[7:15] = torch.randn(8) * 2.0  # mid band
    else:
        tensor[15:] = torch.randn(DIM - 15) * 2.0  # high band
    tensor += torch.randn(DIM) * 0.1  # background noise
    return FieldState(
        tensor=tensor, entropy=1.5, phase=SECPhase.ORDERED,
        conservation_budget=0.0, provenance=[], timestamp=time.time(),
    )


def env_challenge(tick: int) -> FieldState:
    """Gradually increasing complexity -- tests adaptation."""
    complexity = 1.0 + tick * 0.05  # entropy increases over time
    torch.manual_seed(tick * 31)
    tensor = torch.randn(DIM) * (1.0 + tick * 0.01)
    return FieldState(
        tensor=tensor, entropy=min(complexity, 5.0), phase=SECPhase.ORDERED,
        conservation_budget=0.0, provenance=[], timestamp=time.time(),
    )


# ─── Main ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SPIKE G -- GAIA Population Interaction")
    print("  Independent entities communicating through field resonance")
    print("=" * 60)

    # Create a population of 4 specialized entities
    entities = [
        PopulationEntity("guardian", [
            SafetyModule(input_dim=DIM),
            ObservabilityModule(),
        ]),
        PopulationEntity("thinker", [
            ReasoningModule(input_dim=DIM),
            MemoryModule(),
        ]),
        PopulationEntity("speaker", [
            LanguageModule(),
            MemoryModule(),
        ]),
        PopulationEntity("watcher", [
            ObservabilityModule(),
            MemoryModule(),
            LanguageModule(),
        ]),
    ]

    pop = Population(entities)

    # Phase 1: Class rotation (30 ticks) -- do entities specialize?
    print(f"\n--- Phase 1: Class Rotation (30 ticks) ---")
    pop.run(30, env_fn=env_class_rotation, print_every=15)

    # Phase 2: Challenge (30 ticks) -- do they cooperate under pressure?
    print(f"\n--- Phase 2: Increasing Challenge (30 ticks) ---")
    pop.run(30, env_fn=env_challenge, print_every=15)

    # Phase 3: Free interaction (40 ticks) -- what emerges?
    print(f"\n--- Phase 3: Free Interaction (40 ticks) ---")
    pop.run(40, print_every=20)

    # Final report
    pop.report()


if __name__ == "__main__":
    main()
