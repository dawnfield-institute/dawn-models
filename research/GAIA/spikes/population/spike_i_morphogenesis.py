"""Spike I -- Morphogenesis: PACTree-Driven Colony Growth.

A colony that GROWS its own structure. Uses PACTree's branching/deepening
logic as the morphogenesis blueprint:

  - PACTree root     = new lobe (functional cluster)
  - PACTree child    = new organism budded from parent
  - Bifractal depth  = organism maturity (surface dies fast, deep persists)
  - Access count     = experience -> promotion to deeper stability
  - GC               = organism death (inactive cells pruned)
  - Trust clustering = emergent lobe boundaries

The colony starts with ONE organism. Growth happens organically:
  - When environment signal has LOW resonance with existing organisms,
    a new root organism spawns (new lobe -- novel capability needed)
  - When signal has HIGH resonance with an existing organism,
    that organism buds a child (specialization -- similar but different)
  - Organisms that process many signals get promoted to deeper stability
  - Organisms that go unused decay and die

This is PACTree logic applied to organism spawning instead of memory storage.
The tree IS the connectivity graph. Parent-child = structural connection.
Siblings in the same subtree = lobe members.

Usage:
    cd dawn-models/research/GAIA
    PYTHONPATH="src;../../fracton" python spikes/population/spike_i_morphogenesis.py
"""

from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import torch

_root = Path(__file__).resolve().parents[2]
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))
_fracton = _root.parents[0] / "fracton"
if str(_fracton) not in sys.path:
    sys.path.insert(0, str(_fracton))

from gaia.core.coupled_fields_bus import CoupledFieldsBus, _harmonic_resonance
from gaia.core.types import FieldState, SECPhase
from gaia.modules.safety import SafetyModule
from gaia.modules.reasoning import ReasoningModule
from gaia.modules.memory import MemoryModule
from gaia.modules.language import LanguageModule
from gaia.modules.observability import ObservabilityModule
from gaia.network import GAIAAgent

DIM = 22
PHI_INV = 0.6180339887498949

# Growth thresholds -- PACTree-derived
BRANCH_THRESHOLD = 0.5   # resonance > this -> bud child (deepen)
ROOT_THRESHOLD = 0.15    # resonance < this -> new root (new lobe)
TRUST_WIRE_THRESHOLD = 0.6  # trust > this -> structural connection
MAX_COLONY_SIZE = 40     # capacity limit (like PACTree capacity)

# Selective activation -- not every neuron fires every cycle
ACTIVATION_THRESHOLD = 0.15  # minimum resonance to fire
MIN_ACTIVATION_RATIO = 0.1   # at least 10% of cells fire per tick


# --- Maturity (mirrors BifractalDepth) ---------------------------------

class Maturity(IntEnum):
    """Organism maturity -- mirrors BifractalDepth."""
    SURFACE = 0     # Ephemeral, dies fast
    SHALLOW = 1     # Short-lived, still proving itself
    STABLE = 2      # Established, moderate persistence
    DEEP = 3        # Core identity, very persistent
    CRYSTALLIZED = 4  # Permanent -- lobe anchor

# Promotion thresholds (access counts needed -- harder than before)
PROMOTION_THRESHOLDS = {
    Maturity.SURFACE: 10,
    Maturity.SHALLOW: 30,
    Maturity.STABLE: 80,
    Maturity.DEEP: 200,
}

# Voice decay rate per maturity (deeper = more stable voice)
VOICE_DECAY = {
    Maturity.SURFACE: 0.50,      # fast drift
    Maturity.SHALLOW: PHI_INV,   # standard
    Maturity.STABLE: 0.75,       # slower drift
    Maturity.DEEP: 0.90,         # very stable
    Maturity.CRYSTALLIZED: 0.98, # near-permanent
}


# --- Organism ---------------------------------------------------------

@dataclass
class Signal:
    sender: str
    tensor: torch.Tensor
    tick: int


def make_organism(name: str) -> GAIAAgent:
    """Create a complete GAIA organism -- all 5 modules."""
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


class Cell:
    """A single organism in the growing colony.

    Like a MemoryNode in PACTree: has parent, children, depth,
    strength, access_count. But it's alive -- it processes signals.
    """

    def __init__(
        self,
        name: str,
        parent: Cell | None = None,
        initial_voice: torch.Tensor | None = None,
    ):
        self.agent = make_organism(name)
        self.parent = parent
        self.children: list[Cell] = []
        self.maturity = Maturity.SURFACE
        self.access_count = 0      # how many signals processed
        self.idle_ticks = 0        # ticks since last activation
        self.birth_tick = 0

        # Voice (resonance state)
        if initial_voice is not None:
            self.voice = initial_voice.clone()
        else:
            self.voice = torch.zeros(DIM)

        # Structural connections (trust-derived wiring)
        self.connections: dict[str, float] = {}  # name -> strength

        # Last output
        self.last_signal: Signal | None = None
        self.spec_history: list[float] = []  # specialization over time
        self.activation_history: list[bool] = []  # fired or not each tick
        self.total_activations: int = 0

    @property
    def name(self) -> str:
        return self.agent.name

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def lobe_id(self) -> str:
        """Which lobe (root ancestor) does this cell belong to?"""
        cell = self
        while cell.parent is not None:
            cell = cell.parent
        return cell.name

    @property
    def tree_depth(self) -> int:
        """How deep in the tree."""
        depth = 0
        cell = self
        while cell.parent is not None:
            cell = cell.parent
            depth += 1
        return depth

    def should_activate(self, env_tensor: torch.Tensor) -> float:
        """How strongly does this cell resonate with its input?

        Root cells resonate with raw environment (sensory neurons).
        Child cells resonate with parent's voice (layered processing).
        Like a neuron -- needs enough input to reach action potential.
        """
        if torch.norm(self.voice) < 1e-6:
            return 1.0  # uninitialized cells always fire (need to learn)
        # Children check resonance with parent, not raw env
        if self.parent is not None and torch.norm(self.parent.voice) > 1e-6:
            return max(0.0, _harmonic_resonance(self.parent.voice, self.voice))
        return max(0.0, _harmonic_resonance(env_tensor, self.voice))

    def perceive(self, env: FieldState, signals: list[Signal]) -> FieldState:
        """Perceive environment + signals from CONNECTED cells only.

        Isolated cells (no connections) get weak broadcast perception --
        otherwise they'd be deaf and could never discover who to connect to.
        """
        combined = env.tensor.clone()
        env_energy = env.total_energy()

        # Filter to connected signals
        connected_signals = [
            s for s in signals
            if s.sender in self.connections and s.sender != self.name
        ]

        # Isolated cells: weak broadcast of ALL signals (discovery mode)
        if not connected_signals and len(self.connections) == 0:
            other_signals = [s for s in signals if s.sender != self.name]
            if other_signals:
                avg = sum(s.tensor for s in other_signals) / len(other_signals)
                # Very weak blend -- just enough to learn who's out there
                combined = 0.9 * combined + 0.1 * avg

        elif connected_signals:
            signal_sum = torch.zeros(DIM)
            total_w = 0.0
            for sig in connected_signals:
                if torch.norm(self.voice) > 1e-6:
                    resonance = max(0.0, _harmonic_resonance(sig.tensor, self.voice))
                else:
                    resonance = 1.0 / len(connected_signals)
                conn_strength = self.connections.get(sig.sender, 0.5)
                w = resonance * conn_strength
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
        """Process input, update voice with maturity-dependent decay."""
        output = self.agent.process(input_state)

        decay = VOICE_DECAY[self.maturity]
        self.voice = decay * self.voice + (1.0 - decay) * output.tensor

        self.access_count += 1
        self.idle_ticks = 0
        self.total_activations += 1
        self.activation_history.append(True)

        sig = Signal(sender=self.name, tensor=output.tensor.clone(), tick=self.access_count)
        self.last_signal = sig
        self.spec_history.append(self.agent.identity.specialization)
        return sig

    def skip_tick(self):
        """Cell didn't fire this tick -- idle increases, no processing."""
        self.idle_ticks += 1
        self.activation_history.append(False)

    def update_connections(self, signals: list[Signal], my_signal: Signal):
        """Update structural connection strengths based on coherence.

        Discovery mode: cells with < 2 connections evaluate ALL signals
        for potential new connections (not just existing ones).
        """
        discovery_mode = len(self.connections) < 2  # isolated cell seeks wiring

        for sig in signals:
            if sig.sender == self.name:
                continue
            coherence = _harmonic_resonance(sig.tensor, my_signal.tensor)
            old = self.connections.get(sig.sender, 0.0)
            new_w = 0.95 * old + 0.05 * max(0.0, coherence)

            if new_w > TRUST_WIRE_THRESHOLD:
                self.connections[sig.sender] = new_w
            elif sig.sender in self.connections:
                # Weak connections decay away
                if new_w < 0.1:
                    del self.connections[sig.sender]
                else:
                    self.connections[sig.sender] = new_w
            elif discovery_mode and coherence > 0.3:
                # New connection discovered through resonance!
                self.connections[sig.sender] = coherence * 0.5

    def try_promote(self) -> bool:
        """Check if this cell should be promoted to deeper maturity."""
        threshold = PROMOTION_THRESHOLDS.get(self.maturity)
        if threshold is None:
            return False  # already CRYSTALLIZED
        if self.access_count >= threshold:
            self.maturity = Maturity(self.maturity + 1)
            self.access_count = 0  # reset after promotion
            return True
        return False


# --- Growing Colony ---------------------------------------------------

class GrowingColony:
    """A colony that grows its own structure using PACTree logic.

    Starts with a single cell. New cells are spawned when:
    - Environment doesn't resonate with existing cells -> new root (new lobe)
    - Environment strongly resonates with one cell -> bud child (specialize)

    Cells that go unused die (GC). Cells that process many signals
    get promoted to deeper maturity (more stable identity).

    The tree structure IS the connectivity graph. Parent-child pairs
    always have connections. Trust-based wiring adds lateral connections
    between siblings and across lobes.
    """

    def __init__(self, seed_name: str = "cell_0"):
        seed = Cell(seed_name)
        self.cells: dict[str, Cell] = {seed.name: seed}
        self.roots: list[str] = [seed.name]  # lobe anchors
        self.tick = 0
        self._next_id = 1

        # History
        self.growth_log: list[dict] = []  # births, deaths, promotions
        self.population_history: list[int] = []
        self.lobe_history: list[int] = []
        self._last_activation_ratio = 1.0

    def _resonance_to_colony(self, signal: torch.Tensor) -> tuple[str, float]:
        """Find the most resonant cell in the colony (tree-guided search).

        Mirrors PACTree._find_best_match: score roots, drill down best subtree.
        """
        best_name = ""
        best_score = 0.0

        # Score all roots (lobe anchors)
        for root_name in self.roots:
            cell = self.cells.get(root_name)
            if cell is None:
                continue
            if torch.norm(cell.voice) < 1e-6:
                score = 0.1  # uninitialized gets small baseline
            else:
                score = max(0.0, _harmonic_resonance(signal, cell.voice))
            if score > best_score:
                best_score = score
                best_name = root_name

        if not best_name:
            return "", 0.0

        # Drill down: check children, keep going if we improve
        current = self.cells[best_name]
        while current.children:
            improved = False
            for child in current.children:
                if torch.norm(child.voice) < 1e-6:
                    continue
                score = max(0.0, _harmonic_resonance(signal, child.voice))
                if score > best_score:
                    best_score = score
                    best_name = child.name
                    improved = True
            if improved:
                current = self.cells[best_name]
            else:
                break

        return best_name, best_score

    def _spawn_root(self, env: FieldState) -> Cell:
        """Spawn a new root cell -- new lobe."""
        name = f"cell_{self._next_id}"
        self._next_id += 1
        cell = Cell(name, parent=None, initial_voice=env.tensor.clone() * 0.1)
        cell.birth_tick = self.tick
        self.cells[name] = cell
        self.roots.append(name)

        # No free wiring -- inter-lobe connections must be EARNED via trust
        # New root starts isolated, discovers connections through coherence

        self.growth_log.append({
            "tick": self.tick, "event": "birth_root",
            "cell": name, "lobe": name,
        })
        return cell

    def _spawn_child(self, parent: Cell, env: FieldState) -> Cell:
        """Bud a child cell from parent -- specialization within lobe."""
        name = f"cell_{self._next_id}"
        self._next_id += 1

        # Child's initial voice = parent + small perturbation (delta)
        delta = env.tensor - parent.voice
        initial_voice = parent.voice + 0.3 * delta  # partial step toward signal

        cell = Cell(name, parent=parent, initial_voice=initial_voice)
        cell.birth_tick = self.tick
        parent.children.append(cell)
        self.cells[name] = cell

        # Wire parent <-> child (structural, always connected)
        cell.connections[parent.name] = 0.8
        parent.connections[name] = 0.8

        # Wire to siblings (same lobe, lateral connections)
        for sibling in parent.children:
            if sibling.name != name:
                cell.connections[sibling.name] = 0.4
                sibling.connections[name] = 0.4

        self.growth_log.append({
            "tick": self.tick, "event": "birth_child",
            "cell": name, "parent": parent.name, "lobe": cell.lobe_id,
        })
        return cell

    def _gc(self):
        """Garbage collect dead cells -- idle leaves at surface/shallow depth."""
        dead: list[str] = []
        for name, cell in self.cells.items():
            if len(self.cells) - len(dead) <= 3:
                break  # keep minimum viable colony
            # SURFACE leaves die after 15 idle ticks
            if (cell.maturity == Maturity.SURFACE and cell.is_leaf
                    and cell.idle_ticks > 15):
                dead.append(name)
            # SHALLOW leaves die after 30 idle ticks
            elif (cell.maturity == Maturity.SHALLOW and cell.is_leaf
                    and cell.idle_ticks > 30):
                dead.append(name)

        for name in dead:
            cell = self.cells[name]
            # Remove from parent's children
            if cell.parent is not None:
                cell.parent.children = [c for c in cell.parent.children if c.name != name]
            # Remove from roots
            if name in self.roots:
                self.roots = [r for r in self.roots if r != name]
            # Remove connections
            for other_name in list(cell.connections.keys()):
                if other_name in self.cells:
                    self.cells[other_name].connections.pop(name, None)
            del self.cells[name]
            self.growth_log.append({
                "tick": self.tick, "event": "death",
                "cell": name,
            })

    def _get_descendants(self, cell: Cell) -> list[Cell]:
        """Iterative tree traversal returning all descendants of cell."""
        descendants = []
        stack = list(cell.children)
        while stack:
            c = stack.pop()
            descendants.append(c)
            stack.extend(c.children)
        return descendants

    def _gc_subtrees(self):
        """Prune entire subtrees where node AND all descendants are idle > 30 ticks.

        Skips cells at DEEP+ maturity (stable identity worth preserving).
        """
        pruned: list[str] = []
        for name, cell in list(self.cells.items()):
            if name in [p for p in pruned]:
                continue
            if cell.is_leaf:
                continue  # regular _gc handles leaves
            if cell.maturity >= Maturity.DEEP:
                continue  # stable identity
            descendants = self._get_descendants(cell)
            # Check if entire subtree is idle
            all_idle = (cell.idle_ticks > 30 and
                        all(d.idle_ticks > 30 for d in descendants))
            if not all_idle:
                continue
            # Don't prune if it would kill the colony
            subtree_size = 1 + len(descendants)
            if len(self.cells) - len(pruned) - subtree_size < 3:
                continue
            # Prune entire subtree
            all_names = [cell.name] + [d.name for d in descendants]
            for n in all_names:
                # Sever connections
                c = self.cells[n]
                for other_name in list(c.connections.keys()):
                    if other_name in self.cells and other_name not in all_names:
                        self.cells[other_name].connections.pop(n, None)
                # Remove from parent
                if c.parent is not None and c.parent.name not in all_names:
                    c.parent.children = [
                        ch for ch in c.parent.children if ch.name != n
                    ]
                if n in self.roots:
                    self.roots = [r for r in self.roots if r != n]
                del self.cells[n]
                pruned.append(n)
            self.growth_log.append({
                "tick": self.tick, "event": "subtree_death",
                "cell": cell.name, "subtree_size": subtree_size,
            })

    def _check_lobe_merges(self):
        """Merge lobes whose root voices have sustained high resonance."""
        if not hasattr(self, '_merge_candidates'):
            self._merge_candidates: dict[tuple[str, str], int] = {}

        # Evaluate all root pairs
        current_pairs: set[tuple[str, str]] = set()
        for i in range(len(self.roots)):
            for j in range(i + 1, len(self.roots)):
                ri, rj = self.roots[i], self.roots[j]
                if ri not in self.cells or rj not in self.cells:
                    continue
                key = (min(ri, rj), max(ri, rj))
                current_pairs.add(key)
                r = _harmonic_resonance(self.cells[ri].voice, self.cells[rj].voice)
                if r > 0.7:
                    self._merge_candidates[key] = self._merge_candidates.get(key, 0) + 1
                else:
                    self._merge_candidates.pop(key, None)

        # Clean stale entries
        for key in list(self._merge_candidates.keys()):
            if key not in current_pairs:
                del self._merge_candidates[key]

        # Merge pairs that hit threshold (5 consecutive ticks above 0.7)
        for (ri, rj), count in list(self._merge_candidates.items()):
            if count < 5:
                continue
            if ri not in self.cells or rj not in self.cells:
                self._merge_candidates.pop((ri, rj), None)
                continue
            # Merge smaller into larger
            size_i = 1 + len(self._get_descendants(self.cells[ri]))
            size_j = 1 + len(self._get_descendants(self.cells[rj]))
            if size_i >= size_j:
                keep, merge = ri, rj
            else:
                keep, merge = rj, ri
            self._merge_lobes(keep, merge)
            self._merge_candidates.pop((ri, rj), None)

    def _merge_lobes(self, keep: str, merge: str):
        """Reparent merge root under keep root, wire at 0.8."""
        keep_cell = self.cells[keep]
        merge_cell = self.cells[merge]
        # Reparent (is_root and tree_depth are computed properties)
        merge_cell.parent = keep_cell
        keep_cell.children.append(merge_cell)
        # Wire strong connection
        keep_cell.connections[merge] = 0.8
        merge_cell.connections[keep] = 0.8
        # Remove from roots
        self.roots = [r for r in self.roots if r != merge]
        self.growth_log.append({
            "tick": self.tick, "event": "lobe_merge",
            "keep": keep, "merged": merge,
            "new_lobe_count": len(self.roots),
        })

    def step(self, env: FieldState, skip_growth: bool = False,
             skip_maintenance: bool = False):
        """One colony tick with selective activation + growth/death/promotion.

        Not every cell fires every tick. Like neurons: cells compute their
        resonance with the input, and only fire if above threshold.
        This creates genuine idle time, real deaths, and sparse processing.

        Args:
            skip_growth: If True, skip the internal growth decision (caller
                         handles growth externally, e.g. PredictiveColony).
            skip_maintenance: If True, skip death/promotion/pruning/merge
                         (caller handles maintenance, e.g. PhysicsFirstColony).
        """
        # --- Growth decision ---
        if not skip_growth:
            best_name, best_score = self._resonance_to_colony(env.tensor)

            if len(self.cells) < MAX_COLONY_SIZE:
                if best_score < ROOT_THRESHOLD:
                    # Low resonance everywhere -> new lobe needed
                    self._spawn_root(env)
                elif best_score > BRANCH_THRESHOLD and best_name:
                    # High resonance with existing cell -> bud child
                    parent = self.cells[best_name]
                    if len(parent.children) < 5:  # branching factor limit
                        self._spawn_child(parent, env)

        # --- Selective activation ---
        # Score all cells against environment. Only fire above threshold.
        activation_scores: dict[str, float] = {}
        for name, cell in self.cells.items():
            activation_scores[name] = cell.should_activate(env.tensor)

        # Sort by activation score (strongest first)
        ranked = sorted(activation_scores.items(), key=lambda x: -x[1])

        # Determine who fires: above threshold, but guarantee minimum ratio
        min_active = max(1, int(len(self.cells) * MIN_ACTIVATION_RATIO))
        active_names: set[str] = set()

        for name, score in ranked:
            if score >= ACTIVATION_THRESHOLD or len(active_names) < min_active:
                active_names.add(name)

        # --- Layered processing (depth-ordered) ---
        # Roots see raw environment (sensory layer).
        # Children see parent's output from THIS tick (information hierarchy).
        prev_signals = [
            c.last_signal for c in self.cells.values()
            if c.last_signal is not None
        ]

        # Group active cells by tree depth
        depth_groups: dict[int, list[str]] = defaultdict(list)
        for name in active_names:
            depth_groups[self.cells[name].tree_depth].append(name)

        # Process layer by layer: roots first, then depth 1, depth 2, etc.
        signals: dict[str, Signal] = {}
        for depth in sorted(depth_groups.keys()):
            for name in depth_groups[depth]:
                cell = self.cells[name]
                if cell.is_root:
                    # Root cells: raw environment (sensory neurons)
                    inp = cell.perceive(env, prev_signals)
                else:
                    # Child cells: parent's output from THIS tick
                    parent_sig = signals.get(cell.parent.name) if cell.parent else None
                    if parent_sig is not None:
                        parent_env = FieldState(
                            tensor=parent_sig.tensor, entropy=env.entropy,
                            phase=env.phase, conservation_budget=0.0,
                            provenance=[], timestamp=time.time(),
                        )
                        inp = cell.perceive(parent_env, prev_signals)
                    else:
                        # Parent wasn't active this tick -- fall back to raw env
                        inp = cell.perceive(env, prev_signals)
                signals[name] = cell.process(inp)

        # Inactive cells just idle
        for name, cell in self.cells.items():
            if name not in active_names:
                cell.skip_tick()

        # Update connections (only active cells update their wiring)
        sig_list = list(signals.values())
        for name in active_names:
            self.cells[name].update_connections(sig_list, signals[name])

        # Track activation ratio and store signals for external access
        self._last_activation_ratio = len(active_names) / max(1, len(self.cells))
        self._last_active_signals = dict(signals)

        # --- Maintenance ---
        if not skip_maintenance:
            # Promotion checks (only active cells can promote)
            for name in active_names:
                cell = self.cells[name]
                if cell.try_promote():
                    self.growth_log.append({
                        "tick": self.tick, "event": "promotion",
                        "cell": cell.name,
                        "new_maturity": cell.maturity.name,
                    })

            # Garbage collection
            self._gc()
            # Subtree pruning every 10 ticks
            if self.tick % 10 == 0:
                self._gc_subtrees()
            # Lobe merge check every 20 ticks
            if self.tick % 20 == 0:
                self._check_lobe_merges()

            # Capacity limit -- if over, cull weakest surface leaves
            while len(self.cells) > MAX_COLONY_SIZE:
                self._gc()
                if len(self.cells) > MAX_COLONY_SIZE:
                    # Force kill the weakest surface cell
                    surface_cells = [
                        (name, c.access_count)
                        for name, c in self.cells.items()
                        if c.maturity == Maturity.SURFACE and c.is_leaf
                    ]
                    if surface_cells:
                        weakest = min(surface_cells, key=lambda x: x[1])[0]
                        cell = self.cells[weakest]
                        if cell.parent:
                            cell.parent.children = [
                                c for c in cell.parent.children if c.name != weakest
                            ]
                        if weakest in self.roots:
                            self.roots.remove(weakest)
                        for other in self.cells.values():
                            other.connections.pop(weakest, None)
                        del self.cells[weakest]
                    else:
                        break  # can't shrink further

        self.tick += 1
        self.population_history.append(len(self.cells))
        self.lobe_history.append(len(self.roots))

    def run(self, n_ticks: int, env_fn, print_every: int = 25):
        for t in range(n_ticks):
            # Support env_fn(tick) or env_fn(tick, colony)
            try:
                env = env_fn(self.tick, colony=self)
            except TypeError:
                env = env_fn(self.tick)
            self.step(env)
            if (t + 1) % print_every == 0:
                self.print_status()

    def print_status(self):
        n_cells = len(self.cells)
        n_lobes = len(self.roots)
        maturities = {}
        for cell in self.cells.values():
            m = cell.maturity.name
            maturities[m] = maturities.get(m, 0) + 1

        connections = sum(len(c.connections) for c in self.cells.values()) // 2
        act_ratio = getattr(self, '_last_activation_ratio', 1.0)

        print(f"\n{'='*70}")
        print(f"  TICK {self.tick}  |  {n_cells} cells  |  {n_lobes} lobes  |  "
              f"{connections} conn  |  {act_ratio:.0%} active")
        print(f"{'='*70}")

        # Maturity distribution
        mat_str = "  ".join(f"{k}={v}" for k, v in sorted(maturities.items()))
        print(f"  maturity: {mat_str}")

        # Lobe structure
        print(f"\n  LOBE STRUCTURE:")
        for root_name in self.roots:
            root = self.cells.get(root_name)
            if root is None:
                continue
            lobe_cells = [c for c in self.cells.values() if c.lobe_id == root_name]
            depths = [c.tree_depth for c in lobe_cells]
            max_depth = max(depths) if depths else 0
            print(f"    {root_name}: {len(lobe_cells)} cells, depth={max_depth}, "
                  f"maturity={root.maturity.name}")
            # Show tree structure (compact)
            self._print_subtree(root, indent=6, max_depth=3)

        # Recent events
        recent = [e for e in self.growth_log if e["tick"] > self.tick - 10]
        if recent:
            print(f"\n  RECENT EVENTS (last 10 ticks):")
            for e in recent[-5:]:
                print(f"    t={e['tick']}: {e['event']} {e['cell']}")

    def _print_subtree(self, cell: Cell, indent: int = 0, max_depth: int = 3):
        """Print a compact tree visualization."""
        if cell.tree_depth > max_depth:
            return
        prefix = " " * indent
        conn_count = len(cell.connections)
        mat = cell.maturity.name[0]  # S/S/S/D/C
        print(f"{prefix}|- {cell.name} [{mat}] "
              f"acc={cell.access_count} conn={conn_count} "
              f"spec={cell.spec_history[-1]:.3f}" if cell.spec_history else
              f"{prefix}|- {cell.name} [{mat}] acc={cell.access_count} conn={conn_count}")
        for child in cell.children:
            self._print_subtree(child, indent + 3, max_depth)

    def report(self):
        """Final morphogenesis report."""
        print(f"\n{'='*70}")
        print(f"  MORPHOGENESIS REPORT")
        print(f"  {self.tick} ticks | started: 1 cell -> {len(self.cells)} cells, "
              f"{len(self.roots)} lobes")
        print(f"{'='*70}")

        # Growth timeline
        births = [e for e in self.growth_log if "birth" in e["event"]]
        deaths = [e for e in self.growth_log if e["event"] == "death"]
        promotions = [e for e in self.growth_log if e["event"] == "promotion"]
        print(f"\n  LIFECYCLE: {len(births)} births, {len(deaths)} deaths, "
              f"{len(promotions)} promotions")

        # Population over time (compact)
        if self.population_history:
            h = self.population_history
            quarter = len(h) // 4
            if quarter > 0:
                print(f"  population: "
                      f"t=0:{h[0]} -> "
                      f"t={quarter}:{h[quarter]} -> "
                      f"t={quarter*2}:{h[quarter*2]} -> "
                      f"t={quarter*3}:{h[quarter*3]} -> "
                      f"t={len(h)}:{h[-1]}")

        # Maturity distribution
        print(f"\n  MATURITY DISTRIBUTION:")
        for mat in Maturity:
            cells_at = [c for c in self.cells.values() if c.maturity == mat]
            if cells_at:
                names = ", ".join(c.name for c in cells_at[:5])
                extra = f" (+{len(cells_at)-5} more)" if len(cells_at) > 5 else ""
                print(f"    {mat.name:15s}: {len(cells_at):3d}  [{names}{extra}]")

        # Activation analysis
        print(f"\n  ACTIVATION ANALYSIS:")
        for cell in sorted(self.cells.values(), key=lambda c: c.total_activations, reverse=True)[:10]:
            n_ticks = len(cell.activation_history) or 1
            fire_rate = cell.total_activations / n_ticks
            print(f"    {cell.name:10s}: fired {cell.total_activations:4d}/{n_ticks} "
                  f"({fire_rate:.0%})  idle={cell.idle_ticks}  mat={cell.maturity.name}")

        # Lobe analysis
        print(f"\n  LOBE ANALYSIS:")
        lobe_members: dict[str, list[Cell]] = {}
        for cell in self.cells.values():
            lid = cell.lobe_id
            lobe_members.setdefault(lid, []).append(cell)

        for lobe_id, members in lobe_members.items():
            n = len(members)
            depths = [c.tree_depth for c in members]
            mats = [c.maturity.value for c in members]
            # Intra-lobe connection density
            intra_conn = 0
            for c in members:
                for conn_name in c.connections:
                    if conn_name in [m.name for m in members]:
                        intra_conn += 1
            density = intra_conn / (n * (n - 1)) if n > 1 else 0
            print(f"    {lobe_id}: {n} cells, max_depth={max(depths)}, "
                  f"mean_maturity={sum(mats)/n:.1f}, "
                  f"internal_density={density:.2f}")

        # Cross-lobe connections (inter-lobe wiring)
        print(f"\n  INTER-LOBE WIRING:")
        for li, members_i in lobe_members.items():
            for lj, members_j in lobe_members.items():
                if li >= lj:
                    continue
                cross = 0
                for ci in members_i:
                    for cj in members_j:
                        if cj.name in ci.connections:
                            cross += 1
                if cross > 0:
                    print(f"    {li} <-> {lj}: {cross} connections")

        # Voice divergence between lobes
        print(f"\n  LOBE VOICE RESONANCE:")
        lobe_voices = {}
        for lid, members in lobe_members.items():
            voices = [c.voice for c in members if torch.norm(c.voice) > 1e-6]
            if voices:
                lobe_voices[lid] = sum(voices) / len(voices)
        lobe_ids = list(lobe_voices.keys())
        for i, li in enumerate(lobe_ids):
            for j in range(i + 1, len(lobe_ids)):
                lj = lobe_ids[j]
                r = _harmonic_resonance(lobe_voices[li], lobe_voices[lj])
                print(f"    {li} vs {lj}: resonance={r:+.3f}")

        # Connection graph (who connects to whom)
        print(f"\n  FULL CONNECTIVITY ({sum(len(c.connections) for c in self.cells.values())//2} edges):")
        for cell in sorted(self.cells.values(), key=lambda c: c.name):
            if cell.connections:
                conns = sorted(cell.connections.items(), key=lambda x: -x[1])[:5]
                conn_str = ", ".join(f"{n}={w:.2f}" for n, w in conns)
                print(f"    {cell.name} [{cell.maturity.name[0]}] -> {conn_str}")


# --- Environments -------------------------------------------------------

def env_diverse(tick: int) -> FieldState:
    """Diverse signals -- stimulates lobe formation."""
    torch.manual_seed(tick * 17 + 7)
    tensor = torch.randn(DIM)
    # Different frequency bands per tick class
    class_id = tick % 5
    band_start = (class_id * 4) % DIM
    band_end = min(band_start + 6, DIM)
    tensor[band_start:band_end] *= 2.5  # amplify one band
    return FieldState(
        tensor=tensor, entropy=1.5, phase=SECPhase.ORDERED,
        conservation_budget=0.0, provenance=[], timestamp=time.time(),
    )


def env_shifting(tick: int) -> FieldState:
    """Regime shifts every 30 ticks -- drives lobe formation."""
    regime = tick // 30
    torch.manual_seed(regime * 1000 + tick)
    tensor = torch.randn(DIM)
    # Each regime lights up different dimensions
    start = (regime * 7) % DIM
    end = min(start + 8, DIM)
    tensor[start:end] *= 3.0
    entropy = 1.0 + regime * 0.3
    return FieldState(
        tensor=tensor, entropy=min(entropy, 5.0), phase=SECPhase.ORDERED,
        conservation_budget=0.0, provenance=[], timestamp=time.time(),
    )


def env_gradual(tick: int) -> FieldState:
    """Smoothly changing signal -- tests organic growth vs rapid expansion."""
    torch.manual_seed(tick)
    phase = tick * 0.05
    tensor = torch.zeros(DIM)
    for i in range(DIM):
        tensor[i] = math.sin(phase + i * 0.3) + 0.2 * torch.randn(1).item()
    return FieldState(
        tensor=tensor, entropy=1.5, phase=SECPhase.ORDERED,
        conservation_budget=0.0, provenance=[], timestamp=time.time(),
    )


# Precompute 5 orthogonal token vectors for env_sequential
_TOKEN_VECTORS: list[torch.Tensor] = []
torch.manual_seed(42_000)
for _i in range(5):
    v = torch.randn(DIM)
    # Gram-Schmidt against previous tokens
    for prev in _TOKEN_VECTORS:
        v = v - torch.dot(v, prev) * prev
    v = v / (torch.norm(v) + 1e-8)
    _TOKEN_VECTORS.append(v)


def env_sequential(tick: int) -> FieldState:
    """Repeating token sequence A->B->C with surprise D injection.

    Exercises: Language learns transitions, Safety flags surprise tokens.
    """
    # Cycle A(0)->B(1)->C(2) repeating
    token_idx = tick % 3
    # After tick 50, every 7th tick swap C for D (surprise)
    if tick > 50 and tick % 7 == 0 and token_idx == 2:
        token_idx = 3  # D = surprise token
    tensor = _TOKEN_VECTORS[token_idx].clone()
    # Small noise overlay
    torch.manual_seed(tick * 31 + 11)
    tensor = tensor + 0.1 * torch.randn(DIM)
    return FieldState(
        tensor=tensor, entropy=1.0, phase=SECPhase.ORDERED,
        conservation_budget=0.0, provenance=[], timestamp=time.time(),
    )


def env_hierarchical(tick: int) -> FieldState:
    """Multi-scale structure: slow base wave + fast overlay + periodic pulses.

    Exercises: Layered processing -- roots catch base, children catch fast,
    grandchildren catch pulses.
    """
    tensor = torch.zeros(DIM)
    # Base: slow sine across all dims
    base_phase = 2.0 * math.pi * tick / 100.0
    tensor += math.sin(base_phase)
    # Fast overlay on dims 0-7 only
    fast_phase = 2.0 * math.pi * tick / 10.0
    tensor[:8] += 0.5 * math.sin(fast_phase)
    # Pulse: spike on dims 15-22 every 50 ticks
    if tick % 50 < 3:  # 3-tick pulse window
        tensor[15:DIM] += 3.0
    return FieldState(
        tensor=tensor, entropy=1.5, phase=SECPhase.ORDERED,
        conservation_budget=0.0, provenance=[], timestamp=time.time(),
    )


def env_ecosystem(tick: int, colony=None) -> FieldState:
    """Adversarial: generates signal orthogonal to colony's mean voice.

    Forces colony to keep adapting. Falls back to env_diverse if no colony.
    """
    if colony is None or len(colony.cells) == 0:
        return env_diverse(tick)

    # Compute colony's mean voice
    voices = [c.voice for c in colony.cells.values() if torch.norm(c.voice) > 1e-6]
    if not voices:
        return env_diverse(tick)

    mean_voice = sum(voices) / len(voices)
    mean_norm = torch.norm(mean_voice)
    if mean_norm < 1e-6:
        return env_diverse(tick)

    mean_voice = mean_voice / mean_norm

    # Generate random vector and make it orthogonal to mean voice
    torch.manual_seed(tick * 41 + 13)
    rand_vec = torch.randn(DIM)
    # Gram-Schmidt: remove projection onto mean voice
    rand_vec = rand_vec - torch.dot(rand_vec, mean_voice) * mean_voice
    rand_norm = torch.norm(rand_vec)
    if rand_norm < 1e-6:
        return env_diverse(tick)
    tensor = rand_vec / rand_norm * 2.0  # scale to reasonable magnitude
    return FieldState(
        tensor=tensor, entropy=2.0, phase=SECPhase.ORDERED,
        conservation_budget=0.0, provenance=[], timestamp=time.time(),
    )


# --- Main ---------------------------------------------------------------

def main():
    import tempfile
    import os

    print("=" * 70)
    print("  SPIKE I -- Morphogenesis: PACTree-Driven Colony Growth")
    print("  One cell -> many cells, lobes form organically")
    print("=" * 70)

    colony = GrowingColony(seed_name="cell_0")

    # Phase 1: Diverse signals -- initial growth + lobe formation
    print(f"\n--- Phase 1: Diverse Stimuli (150 ticks) ---")
    print(f"  Starting: 1 cell, 1 lobe")
    colony.run(150, env_fn=env_diverse, print_every=50)

    # Phase 2: Sequential tokens -- exercises language transitions
    print(f"\n--- Phase 2: Sequential Tokens (200 ticks) ---")
    colony.run(200, env_fn=env_sequential, print_every=50)

    # Phase 3: Hierarchical signals -- exercises layered processing
    print(f"\n--- Phase 3: Hierarchical Signals (200 ticks) ---")
    colony.run(200, env_fn=env_hierarchical, print_every=50)

    # --- CHECKPOINT: save, load, verify, continue ---
    print(f"\n{'='*70}")
    print(f"  CHECKPOINT TEST: save + load + verify continuity")
    print(f"{'='*70}")

    from gaia.network.checkpoint import save_colony, load_colony, checkpoint_info

    checkpoint_dir = Path(_root) / "checkpoints"
    checkpoint_path = checkpoint_dir / "colony_morphogenesis.pt"

    # Save
    save_colony(colony, checkpoint_path)
    pre_tick = colony.tick
    pre_cells = len(colony.cells)
    pre_lobes = len(colony.roots)
    pre_births = len([e for e in colony.growth_log if "birth" in e["event"]])
    pre_deaths = len([e for e in colony.growth_log
                      if e["event"] in ("death", "subtree_death")])

    # Inspect checkpoint metadata
    info = checkpoint_info(checkpoint_path)
    file_size = os.path.getsize(checkpoint_path)
    print(f"  saved: {file_size/1024:.0f} KB, version={info['version']}, "
          f"{info['n_cells']} cells, {info['n_lobes']} lobes, tick={info['tick']}")
    print(f"  lifecycle: {info['total_births']} births, {info['total_deaths']} deaths")

    # Load into fresh colony
    loaded = load_colony(
        checkpoint_path,
        cell_class=Cell,
        colony_class=GrowingColony,
        make_organism_fn=make_organism,
        signal_class=Signal,
    )

    # Verify state
    assert loaded.tick == pre_tick, f"tick mismatch: {loaded.tick} vs {pre_tick}"
    assert len(loaded.cells) == pre_cells, f"cell count mismatch"
    assert len(loaded.roots) == pre_lobes, f"lobe count mismatch"
    loaded_births = len([e for e in loaded.growth_log if "birth" in e["event"]])
    loaded_deaths = len([e for e in loaded.growth_log
                         if e["event"] in ("death", "subtree_death")])
    assert loaded_births == pre_births, f"birth count mismatch"
    assert loaded_deaths == pre_deaths, f"death count mismatch"

    # Verify tree topology preserved
    for name, cell in loaded.cells.items():
        orig = colony.cells[name]
        assert cell.maturity == orig.maturity, f"{name} maturity mismatch"
        assert cell.access_count == orig.access_count, f"{name} access_count mismatch"
        assert cell.total_activations == orig.total_activations, f"{name} activations mismatch"
        assert (cell.parent is None) == (orig.parent is None), f"{name} parent mismatch"
        if cell.parent is not None:
            assert cell.parent.name == orig.parent.name, f"{name} parent name mismatch"
        assert len(cell.children) == len(orig.children), f"{name} children count mismatch"
        # Voice should be close (float precision)
        voice_diff = float(torch.norm(cell.voice - orig.voice).item())
        assert voice_diff < 1e-5, f"{name} voice drift: {voice_diff}"
        # Connections preserved
        assert set(cell.connections.keys()) == set(orig.connections.keys()), \
            f"{name} connection keys mismatch"

    print(f"  VERIFIED: {pre_cells} cells, {pre_lobes} lobes, tree topology, "
          f"voices, connections, maturity, activation history -- all match")

    # Phase 4: Ecosystem (adversarial) -- forces adaptation from checkpoint
    print(f"\n--- Phase 4: Ecosystem Pressure (200 ticks, FROM CHECKPOINT) ---")
    loaded.run(200, env_fn=env_ecosystem, print_every=50)

    # Phase 5: Gradual evolution -- settle into final shape
    print(f"\n--- Phase 5: Gradual Evolution (150 ticks) ---")
    loaded.run(150, env_fn=env_gradual, print_every=50)

    loaded.report()

    # Summary of v2 dynamics
    merges = [e for e in loaded.growth_log if e["event"] == "lobe_merge"]
    subtree_deaths = [e for e in loaded.growth_log if e["event"] == "subtree_death"]
    print(f"\n  V2 DYNAMICS: {len(merges)} lobe merges, "
          f"{len(subtree_deaths)} subtree deaths")

    # Save final state
    final_path = checkpoint_dir / "colony_morphogenesis_final.pt"
    save_colony(loaded, final_path)
    final_size = os.path.getsize(final_path)
    print(f"\n  Final checkpoint saved: {final_size/1024:.0f} KB, tick={loaded.tick}")
    print(f"  Ready to resume: load_colony('{final_path.name}', ...)")


if __name__ == "__main__":
    main()
