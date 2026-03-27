"""Spike K Cascade -- Event-Driven Bottom-Up Actualization.

Replaces the sequential tick loop with an event-driven cascade where:
  - Nodes are ACTIVE: they decide when to actualize, not a scheduler
  - Time is EMERGENT: each actualization increments a monotonic event counter
  - Flow is BOTTOM-UP: leaves fire first, parents synthesize children's signals
  - Parallelism is NATURAL: siblings at the same depth are independent (GPU batch)

The tick loop was a Newtonian clock. This is relativistic — time is local to each
node's causal history. Global time is just the aggregate count of actualizations.

Physics preserved:
  - PAC conservation (crystal filter at each node, residual-based growth)
  - SEC collapse (crystallization-dependent sharpness at each gate)
  - Landauer epochs (entropy reinjection between passes)
  - All constants from DFT (XI_SEC, PHI_INV, LAMBDA_STAR, GAMMA)

Usage:
    cd dawn-models/research/GAIA/spikes/population
    PYTHONIOENCODING=utf-8 PYTHONPATH="../../src;path/to/fracton" python spike_k_cascade.py
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# ==========================================================================
#  DFT Constants (zero heuristic -- all from information theory)
# ==========================================================================

XI_SEC = 0.0618033988749895      # SEC collapse threshold
PHI_INV = 0.618033988749895      # Golden ratio inverse
LAMBDA_STAR = 0.9816             # Voice decay (crystallized)
GAMMA = 1.0 - LAMBDA_STAR       # Dissipation rate = 0.0184
LN_PHI = math.log((1 + math.sqrt(5)) / 2)  # 0.4812
PHI_SQ_FLOOR = 2                 # Max growth per cascade
DIM = 64                         # Tensor dimensionality

# Force CPU — CUDA kernel launch overhead kills small-tensor perf (12x slower)
# GPU batching only wins when leaf count > ~1000 (not this spike's scale)
DEVICE = torch.device("cpu")


# ==========================================================================
#  FastCodebook (from spike_k_cuda -- GPU-resident character encoding)
# ==========================================================================

class FastCodebook:
    """Character codebook with GPU-resident matrix for vectorized decode."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self._char_to_idx: dict[str, int] = {}
        self._idx_to_char: list[str] = []
        self._vectors: list[torch.Tensor] = []
        self._matrix: torch.Tensor | None = None
        self._build()

    def _build(self):
        classes = {
            "vowel": list("aeiou"),
            "consonant": list("bcdfghjklmnpqrstvwxyz"),
            "digit": list("0123456789"),
            "space": [" ", "\n", "\t"],
            "punct": list(".,!?;:'\"()-"),
        }

        for class_idx, (class_name, chars) in enumerate(classes.items()):
            torch.manual_seed(class_idx * 10000 + 42)
            class_dir = torch.randn(self.dim, device="cpu")
            class_dir = class_dir / (torch.norm(class_dir) + 1e-8)

            vecs = []
            for i, ch in enumerate(chars):
                torch.manual_seed(class_idx * 10000 + i * 100 + 7)
                v = torch.randn(self.dim, device="cpu")
                v = 0.4 * class_dir + 0.6 * v
                for prev in vecs:
                    v = v - torch.dot(v, prev) * prev
                norm = torch.norm(v)
                if norm < 1e-8:
                    torch.manual_seed(class_idx * 10000 + i * 100 + 999)
                    v = torch.randn(self.dim, device="cpu")
                v = v / (torch.norm(v) + 1e-8)
                vecs.append(v)
                self._char_to_idx[ch] = len(self._idx_to_char)
                self._idx_to_char.append(ch)
                self._vectors.append(v)

        # Uppercase
        torch.manual_seed(99999)
        offset = 0.15 * torch.randn(self.dim, device="cpu")
        for ch in "abcdefghijklmnopqrstuvwxyz":
            upper = ch.upper()
            if ch in self._char_to_idx:
                v = self._vectors[self._char_to_idx[ch]] + offset
                v = v / (torch.norm(v) + 1e-8)
                self._char_to_idx[upper] = len(self._idx_to_char)
                self._idx_to_char.append(upper)
                self._vectors.append(v)

        self._matrix = torch.stack(self._vectors).to(DEVICE)

    def encode(self, char: str) -> torch.Tensor:
        if char in self._char_to_idx:
            return self._matrix[self._char_to_idx[char]].clone()
        torch.manual_seed(hash(char) % 2**31)
        v = torch.randn(self.dim, device=DEVICE)
        return v / (torch.norm(v) + 1e-8)

    def decode_nearest(self, tensor: torch.Tensor) -> tuple[str, float]:
        t = tensor.flatten()[:self.dim].to(DEVICE)
        t_norm = torch.norm(t)
        if t_norm < 1e-8:
            return "?", 0.0
        sims = F.cosine_similarity(self._matrix, t.unsqueeze(0), dim=1)
        idx = int(torch.argmax(sims))
        return self._idx_to_char[idx], float(sims[idx])


# ==========================================================================
#  CascadeNode
# ==========================================================================

@dataclass
class CascadeNode:
    """Active node in the cascade tree.

    Each node has a fixed 'axis' (birth direction) for consistent crystal
    filtering, and an evolving 'voice' for resonance matching and decode.
    Nodes decide for themselves whether to actualize — no external scheduler.
    """
    name: str
    voice: torch.Tensor
    axis: torch.Tensor
    parent_name: str | None = None
    child_names: list[str] = field(default_factory=list)

    # Event-driven state
    last_actualization: int = 0
    actualization_count: int = 0
    birth_event: int = 0
    birth_input: int = 0  # input counter at birth (for grace period)

    # Entropy for crystallization
    entropy_history: list[float] = field(default_factory=list)

    # Potential field
    potential: float = 0.0

    # Cascade round state (reset each cascade)
    cascade_signal: torch.Tensor | None = None
    actualized_this_round: bool = False


# ==========================================================================
#  CascadeTree
# ==========================================================================

class CascadeTree:
    """Event-driven actualization cascade. The tree walks itself.

    Time is emergent: event_counter increments with each actualization.
    Parallelism is natural: all nodes at the same depth are independent.
    """

    def __init__(self, codebook: FastCodebook, max_nodes: int = 500):
        self.codebook = codebook
        self.max_nodes = max_nodes

        # THE CLOCK — monotonic, incremented by each actualization
        self.event_counter = 0
        self.input_counter = 0  # counts input characters (for grace period)

        # Node storage
        self.nodes: dict[str, CascadeNode] = {}
        self.root_names: list[str] = []
        self._next_id = 0

        # Depth index: depth -> [node names]
        self._depth_index: dict[int, list[str]] = defaultdict(list)
        self._node_depth: dict[str, int] = {}  # cached depth per node
        self._max_depth = 0
        self._depth_dirty = False  # lazy rebuild flag

        # Cascade state (per-round)
        self._node_inputs: dict[str, torch.Tensor] = {}

        # Growth queue (deferred by one input)
        self._growth_queue: list[str] = []  # leaf names from previous cascade
        self._growth_spawned_this_round = 0

        # Prediction state
        self._last_pred_tensor: torch.Tensor | None = None
        self._last_target_char: str | None = None

        # Metrics
        self.accuracy_history: list[bool] = []
        self.phase_accuracy: dict[str, list[bool]] = defaultdict(list)
        self.char_accuracy: dict[str, list[bool]] = defaultdict(list)
        self._current_phase = ""

        # Leaf residual tracking (for Landauer)
        self.leaf_residuals: dict[str, torch.Tensor] = {}
        self.leaf_residual_counts: dict[str, int] = {}

        # Diagnostics
        self._events_per_cascade: list[int] = []
        self._cascade_depths: list[int] = []
        self._leaf_fire_rates: list[float] = []

        # Seed tree: one root per unique character in corpus
        # (wide, shallow tree — matches original Crystal Colony structure)
        self._seed_roots = True  # flag for deferred seeding

    # ------------------------------------------------------------------
    #  Node lifecycle
    # ------------------------------------------------------------------

    def _make_axis(self, voice: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(voice)
        if norm > 1e-8:
            return (voice / norm).clone()
        axis = torch.randn(DIM, device=DEVICE)
        return axis / (torch.norm(axis) + 1e-8)

    def _spawn_root(self, voice: torch.Tensor) -> CascadeNode:
        name = f"n{self._next_id}"
        self._next_id += 1
        v = voice.to(DEVICE).clone()
        node = CascadeNode(
            name=name, voice=v, axis=self._make_axis(v),
            birth_event=self.event_counter, birth_input=self.input_counter,
            potential=float(torch.norm(voice)) + 0.1,
        )
        self.nodes[name] = node
        self.root_names.append(name)
        self._add_to_depth_index(name, 0)
        return node

    def _spawn_child(self, parent_name: str, voice: torch.Tensor) -> CascadeNode | None:
        if len(self.nodes) >= self.max_nodes:
            return None
        parent = self.nodes.get(parent_name)
        if parent is None:
            return None
        name = f"n{self._next_id}"
        self._next_id += 1
        v = voice.to(DEVICE).clone()
        node = CascadeNode(
            name=name, voice=v, axis=self._make_axis(v),
            parent_name=parent_name,
            birth_event=self.event_counter, birth_input=self.input_counter,
            potential=float(torch.norm(voice)),
        )
        parent.child_names.append(name)
        self.nodes[name] = node
        parent_depth = self._get_depth(parent_name)
        self._add_to_depth_index(name, parent_depth + 1)
        return node

    def _dissolve_node(self, name: str):
        node = self.nodes.get(name)
        if node is None:
            return
        has_reparented = bool(node.child_names)

        # Reparent children
        for child_name in node.child_names:
            child = self.nodes.get(child_name)
            if child is None:
                continue
            if node.parent_name and node.parent_name in self.nodes:
                child.parent_name = node.parent_name
                self.nodes[node.parent_name].child_names.append(child_name)
            else:
                # Adopt into most resonant root
                best_root, best_sim = None, -1.0
                for rname in self.root_names:
                    if rname == name or rname == child_name or rname not in self.nodes:
                        continue
                    sim = float(F.cosine_similarity(
                        child.voice.unsqueeze(0),
                        self.nodes[rname].voice.unsqueeze(0)
                    ))
                    if sim > best_sim:
                        best_sim = sim
                        best_root = rname
                if best_root and best_sim > XI_SEC:
                    child.parent_name = best_root
                    self.nodes[best_root].child_names.append(child_name)
                else:
                    child.parent_name = None
                    self.root_names.append(child_name)

        # Remove from parent
        if node.parent_name and node.parent_name in self.nodes:
            parent = self.nodes[node.parent_name]
            parent.child_names = [c for c in parent.child_names if c != name]

        # Remove from roots
        if name in self.root_names:
            self.root_names = [r for r in self.root_names if r != name]

        self._remove_from_depth_index(name)
        del self.nodes[name]
        self.leaf_residuals.pop(name, None)
        self.leaf_residual_counts.pop(name, None)

        # Reparenting changes child depths — need full rebuild
        if has_reparented:
            self._depth_dirty = True

    def _rebuild_depth_index(self):
        """Rebuild depth index from tree topology (full rebuild, called lazily)."""
        self._depth_index.clear()
        self._node_depth.clear()
        for name in self.nodes:
            d = self._compute_depth(name)
            self._node_depth[name] = d
            self._depth_index[d].append(name)
        self._max_depth = max(self._depth_index.keys()) if self._depth_index else 0
        self._depth_dirty = False

    def _ensure_depth_index(self):
        """Lazy rebuild — only when dirty."""
        if self._depth_dirty:
            self._rebuild_depth_index()

    def _compute_depth(self, name: str) -> int:
        d = 0
        node = self.nodes.get(name)
        while node and node.parent_name:
            d += 1
            node = self.nodes.get(node.parent_name)
        return d

    def _get_depth(self, name: str) -> int:
        if name in self._node_depth:
            return self._node_depth[name]
        d = self._compute_depth(name)
        self._node_depth[name] = d
        return d

    def _add_to_depth_index(self, name: str, depth: int):
        """Incremental add — no full rebuild needed."""
        self._node_depth[name] = depth
        self._depth_index[depth].append(name)
        if depth > self._max_depth:
            self._max_depth = depth

    def _remove_from_depth_index(self, name: str):
        """Incremental remove."""
        d = self._node_depth.pop(name, None)
        if d is not None and name in self._depth_index.get(d, []):
            self._depth_index[d].remove(name)

    def _get_leaves(self) -> list[str]:
        return [n for n, node in self.nodes.items() if not node.child_names]

    # ------------------------------------------------------------------
    #  Entropy / crystallization
    # ------------------------------------------------------------------

    def _get_entropy_var(self, name: str) -> float:
        node = self.nodes.get(name)
        if node is None:
            return 1.0
        hist = node.entropy_history
        if len(hist) < 5:
            return XI_SEC  # neutral for young nodes
        recent = hist[-20:]
        mean_h = sum(recent) / len(recent)
        return sum((h - mean_h) ** 2 for h in recent) / len(recent)

    def _compute_entropy(self, voice: torch.Tensor) -> float:
        v = torch.abs(voice) + 1e-10
        p = v / v.sum()
        return -float(torch.sum(p * torch.log(p)))

    # ------------------------------------------------------------------
    #  Phase 0: Evaluate last prediction
    # ------------------------------------------------------------------

    def _evaluate_prediction(self):
        if self._last_pred_tensor is not None and self._last_target_char is not None:
            pred_char, _ = self.codebook.decode_nearest(self._last_pred_tensor)
            correct = (pred_char == self._last_target_char)
            self.accuracy_history.append(correct)
            self.phase_accuracy[self._current_phase].append(correct)
            self.char_accuracy[self._last_target_char].append(correct)

    # ------------------------------------------------------------------
    #  Phase 1: Broadcast (top-down filter preparation)
    # ------------------------------------------------------------------

    def _broadcast(self, input_tensor: torch.Tensor):
        """Distribute filtered input to all nodes, level by level.

        Each node's children see the input filtered through the node's
        crystal axis. This is the same layered filtering as the current
        navigation, but applied to ALL branches, not just one path.
        """
        self._node_inputs.clear()
        self._ensure_depth_index()

        # Reset cascade state
        for node in self.nodes.values():
            node.actualized_this_round = False
            node.cascade_signal = None

        # Roots see raw input
        for rname in self.root_names:
            if rname in self.nodes:
                self._node_inputs[rname] = input_tensor.clone()

        # Level by level: filter through each node's axis for its children
        for d in range(self._max_depth + 1):
            names_at_d = self._depth_index.get(d, [])
            for name in names_at_d:
                node = self.nodes[name]
                node_input = self._node_inputs.get(name, input_tensor)
                if not node.child_names:
                    continue  # leaf, no children to filter for
                # Crystal filter: project through fixed axis
                proj = torch.dot(node_input, node.axis) * node.axis
                orth = node_input - proj
                filtered = proj + PHI_INV * orth
                for child_name in node.child_names:
                    if child_name in self.nodes:
                        self._node_inputs[child_name] = filtered

    # ------------------------------------------------------------------
    #  Phase 2: Leaf actualization (batch GPU parallel)
    # ------------------------------------------------------------------

    def _actualize_leaves(self) -> int:
        """All leaves check resonance simultaneously. Batch GPU op."""
        leaves = self._get_leaves()
        if not leaves:
            return 0

        # Batch tensors
        leaf_inputs = torch.stack([
            self._node_inputs.get(n, torch.zeros(DIM, device=DEVICE))
            for n in leaves
        ])  # [L, DIM]
        leaf_voices = torch.stack([self.nodes[n].voice for n in leaves])  # [L, DIM]
        leaf_axes = torch.stack([self.nodes[n].axis for n in leaves])    # [L, DIM]

        # Batch resonance: cosine similarity [L]
        voice_norms = torch.norm(leaf_voices, dim=1)
        cos_sim = F.cosine_similarity(leaf_inputs, leaf_voices, dim=1)
        cos_sim = torch.clamp(cos_sim, min=0.0)

        # Actualization mask: resonate > XI_SEC or zero-voice (needs to learn)
        actualize_mask = (cos_sim >= XI_SEC) | (voice_norms < 1e-6)

        # Batch crystal filter
        dots = torch.sum(leaf_inputs * leaf_axes, dim=1, keepdim=True)  # [L, 1]
        projs = dots * leaf_axes                                         # [L, DIM]
        orths = leaf_inputs - projs                                      # [L, DIM]
        signals = projs + PHI_INV * orths                                # [L, DIM]

        # Batch entropy variance for crystallization
        entropy_vars = torch.tensor(
            [self._get_entropy_var(n) for n in leaves], device=DEVICE
        )
        cryst = torch.clamp(1.0 - entropy_vars / XI_SEC, min=0.0, max=1.0)
        decay = LAMBDA_STAR + (1.0 - LAMBDA_STAR) * cryst               # [L]

        # Batch voice update
        new_voices = decay.unsqueeze(1) * leaf_voices + \
                     (1.0 - decay.unsqueeze(1)) * signals                # [L, DIM]

        # Batch residual
        residuals = leaf_inputs - signals
        residual_energies = torch.norm(residuals, dim=1)
        input_energies = torch.norm(leaf_inputs, dim=1) + 1e-8
        absorptions = 1.0 - residual_energies / input_energies

        # Apply updates
        events = 0
        for i, name in enumerate(leaves):
            if not actualize_mask[i]:
                continue
            node = self.nodes[name]
            self.event_counter += 1
            events += 1

            node.voice = new_voices[i].clone()
            node.cascade_signal = signals[i].clone()
            node.actualized_this_round = True
            node.last_actualization = self.event_counter
            node.actualization_count += 1

            # Entropy tracking
            entropy = self._compute_entropy(node.voice)
            node.entropy_history.append(entropy)
            if len(node.entropy_history) > 30:
                node.entropy_history.pop(0)

            # Track residual for Landauer
            if float(residual_energies[i]) > XI_SEC:
                if name not in self.leaf_residuals:
                    self.leaf_residuals[name] = torch.zeros(DIM, device=DEVICE)
                    self.leaf_residual_counts[name] = 0
                self.leaf_residuals[name] += residuals[i]
                self.leaf_residual_counts[name] += 1

        # Track leaf fire rate
        if leaves:
            self._leaf_fire_rates.append(events / len(leaves))

        return events

    # ------------------------------------------------------------------
    #  Phase 3: Cascade upward (level by level)
    # ------------------------------------------------------------------

    def _cascade_upward(self) -> int:
        """Parents synthesize children's signals. Bottom-up, batch per level."""
        events = 0
        max_cascade_depth = 0

        for d in range(self._max_depth - 1, -1, -1):
            names_at_d = self._depth_index.get(d, [])
            if not names_at_d:
                continue

            for name in names_at_d:
                node = self.nodes[name]
                if not node.child_names:
                    continue  # leaf at this depth, already handled

                # Which children actualized?
                act_children = [
                    cn for cn in node.child_names
                    if cn in self.nodes and self.nodes[cn].actualized_this_round
                ]
                if not act_children:
                    continue  # no children fired -> parent stays as potential

                ev = self._get_entropy_var(name)
                cryst = max(0.0, min(1.0, 1.0 - ev / XI_SEC))

                # Fast path: single actualized child (most common)
                if len(act_children) == 1:
                    synthesized = self.nodes[act_children[0]].cascade_signal
                else:
                    # Gather children's cascade signals
                    child_signals = torch.stack([
                        self.nodes[cn].cascade_signal for cn in act_children
                    ])  # [K, DIM]
                    child_voices = torch.stack([
                        self.nodes[cn].voice for cn in act_children
                    ])  # [K, DIM]

                    # SEC-weighted synthesis
                    resonances = F.cosine_similarity(
                        child_voices, node.voice.unsqueeze(0), dim=1
                    )
                    resonances = torch.clamp(resonances, min=XI_SEC)

                    collapse_exp = 1.0 + cryst * (1.0 / XI_SEC - 1.0)
                    weights = resonances ** collapse_exp
                    weights = weights / (weights.sum() + 1e-10)

                    # Weighted synthesis of children's signals
                    synthesized = (weights.unsqueeze(1) * child_signals).sum(dim=0)

                # Crystal filter on synthesized signal
                proj = torch.dot(synthesized, node.axis) * node.axis
                orth = synthesized - proj
                node_signal = proj + PHI_INV * orth

                # Voice update
                decay = LAMBDA_STAR + (1.0 - LAMBDA_STAR) * cryst
                old_voice = node.voice.clone()
                node.voice = decay * old_voice + (1.0 - decay) * node_signal

                # Actualize
                node.cascade_signal = node_signal
                node.actualized_this_round = True
                self.event_counter += 1
                events += 1
                node.last_actualization = self.event_counter
                node.actualization_count += 1
                max_cascade_depth = max(max_cascade_depth, d + 1)

                # Entropy tracking
                entropy = self._compute_entropy(node.voice)
                node.entropy_history.append(entropy)
                if len(node.entropy_history) > 30:
                    node.entropy_history.pop(0)

        if max_cascade_depth > 0:
            self._cascade_depths.append(max_cascade_depth)

        return events

    # ------------------------------------------------------------------
    #  Phase 4: Root collapse -> prediction
    # ------------------------------------------------------------------

    def _root_collapse_and_predict(self) -> tuple[str, torch.Tensor]:
        """Prediction via cumulative potential field.

        1. Find strongest-firing leaf (highest resonance with filtered input)
        2. Walk its path to root, redistribute potential to siblings at each level
        3. Predict = highest-potential LOCAL candidate (siblings, children, ancestors)
        """
        # Find the strongest-firing leaf
        best_leaf = None
        best_resonance = -1.0
        for name, node in self.nodes.items():
            if not node.actualized_this_round or node.child_names:
                continue  # only actualized leaves
            node_input = self._node_inputs.get(name)
            if node_input is not None:
                res = float(F.cosine_similarity(
                    node.voice.unsqueeze(0), node_input.unsqueeze(0)
                ))
                if res > best_resonance:
                    best_resonance = res
                    best_leaf = name

        if best_leaf is None:
            if self.root_names:
                best = max(
                    (n for n in self.root_names if n in self.nodes),
                    key=lambda n: self.nodes[n].potential, default=None
                )
                if best:
                    ch, _ = self.codebook.decode_nearest(self.nodes[best].voice)
                    return ch, self.nodes[best].voice.clone()
            return "?", torch.zeros(DIM, device=DEVICE)

        # Build path from leaf to root
        path = []
        cur = best_leaf
        while cur and cur in self.nodes:
            path.append(cur)
            cur = self.nodes[cur].parent_name
        path.reverse()  # root → ... → leaf

        # Potential redistribution (cumulative, from original Crystal Colony)
        leaf_node = self.nodes[best_leaf]
        leaf_pot = leaf_node.potential
        consumed = leaf_pot * 0.5
        remaining = leaf_pot - consumed
        leaf_node.potential = consumed

        for i in range(len(path) - 1, -1, -1):
            cell_name = path[i]
            node = self.nodes.get(cell_name)
            if node is None:
                continue

            # Find siblings
            if node.parent_name and node.parent_name in self.nodes:
                parent = self.nodes[node.parent_name]
                siblings = [cn for cn in parent.child_names
                           if cn != cell_name and cn in self.nodes]
            else:
                siblings = [rn for rn in self.root_names
                           if rn != cell_name and rn in self.nodes]

            if not siblings:
                continue

            # Fraction at this level: deeper = more local flow
            level_frac = PHI_INV if i == len(path) - 1 else PHI_INV ** 2
            level_amount = remaining * level_frac
            remaining -= level_amount

            # SEC-shaped distribution: resonance with leaf voice
            sib_voices = torch.stack([self.nodes[s].voice for s in siblings])
            resonances = F.cosine_similarity(
                sib_voices, leaf_node.voice.unsqueeze(0), dim=1
            )
            resonances = torch.clamp(resonances, min=XI_SEC)
            weights = resonances / (resonances.sum() + 1e-10)

            for j, sib_name in enumerate(siblings):
                self.nodes[sib_name].potential += level_amount * float(weights[j])

        if remaining > 1e-8:
            leaf_node.potential += remaining

        # Prediction: highest potential among LOCAL candidates
        candidates = []

        # Leaf's children
        candidates.extend(cn for cn in leaf_node.child_names if cn in self.nodes)

        # Walk up path: siblings at each level
        for cell_name in path:
            node = self.nodes.get(cell_name)
            if node is None:
                continue
            if node.parent_name and node.parent_name in self.nodes:
                parent = self.nodes[node.parent_name]
                for cn in parent.child_names:
                    if cn != cell_name and cn in self.nodes:
                        candidates.append(cn)
            else:
                for rn in self.root_names:
                    if rn != cell_name and rn in self.nodes:
                        candidates.append(rn)

        if not candidates:
            return "?", torch.zeros(DIM, device=DEVICE)

        # Deduplicate
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        best_name = max(unique_candidates, key=lambda n: self.nodes[n].potential)
        pred_voice = self.nodes[best_name].voice
        ch, _ = self.codebook.decode_nearest(pred_voice)
        return ch, pred_voice.clone()

    # ------------------------------------------------------------------
    #  Phase 5: Growth (deferred by one input)
    # ------------------------------------------------------------------

    def _process_growth(self, input_tensor: torch.Tensor):
        """Deferred growth: last cascade's leaves -> this cascade's children."""
        self._growth_spawned_this_round = 0

        # Depth cap: breadth over depth. Original Crystal Colony was depth 3-5.
        # log_phi(n_roots) ≈ log(25)/0.48 ≈ 6.7 — cap at 6
        n_roots = max(1, len(self.root_names))
        max_depth = min(6, int(math.log(max(2, self.max_nodes / n_roots)) / LN_PHI))

        # Process queue from PREVIOUS cascade
        for parent_name in self._growth_queue:
            if self._growth_spawned_this_round >= PHI_SQ_FLOOR:
                break
            if parent_name not in self.nodes or len(self.nodes) >= self.max_nodes:
                continue
            parent = self.nodes[parent_name]
            parent_depth = self._get_depth(parent_name)
            if parent_depth >= max_depth:
                continue  # depth cap
            pv_norm = float(torch.norm(parent.voice))
            if pv_norm < 1e-6:
                continue
            # Transition encoding: child voice = what follows parent's context
            # This is what makes prediction work — children decode to the
            # character that typically follows the parent's character
            transition = input_tensor - parent.voice
            trans_energy = float(torch.norm(transition))
            in_energy = float(torch.norm(input_tensor)) + 1e-8
            absorption = 1.0 - (trans_energy / in_energy)
            if absorption < PHI_INV:
                child = self._spawn_child(parent_name, transition.clone())
                if child:
                    self._growth_spawned_this_round += 1

        # New root if no roots resonate at all
        if not any(self.nodes[rn].actualized_this_round
                   for rn in self.root_names if rn in self.nodes):
            if len(self.nodes) < self.max_nodes:
                self._spawn_root(input_tensor.clone())

        # Queue THIS cascade's actualized leaves for next round
        self._growth_queue = [
            n for n in self._get_leaves()
            if n in self.nodes and self.nodes[n].actualized_this_round
        ]

    # ------------------------------------------------------------------
    #  Phase 6: Decay and dissolution
    # ------------------------------------------------------------------

    def _decay_and_dissolve(self):
        """Nodes that haven't actualized recently lose energy.
        Root nodes never dissolve — they're the character vocabulary."""
        dead = []
        root_set = set(self.root_names)

        for name, node in self.nodes.items():
            if node.actualized_this_round:
                continue
            if name in root_set:
                continue  # roots are permanent vocabulary

            # Grace period based on input counter (not events)
            input_age = self.input_counter - node.birth_input
            if input_age < 50:
                continue

            ev = self._get_entropy_var(name)
            instability = min(1.0, ev / XI_SEC)
            eff_gamma = GAMMA * (1.0 + instability * 9.0)

            node.voice *= (1.0 - eff_gamma)
            node.potential *= (1.0 - eff_gamma)

            if float(torch.norm(node.voice)) < XI_SEC:
                dead.append(name)

        # Don't kill everything
        if len(dead) >= len(self.nodes) - len(root_set):
            dead = dead[:-1]

        for name in dead:
            self._dissolve_node(name)

    # ------------------------------------------------------------------
    #  Main cascade entry point
    # ------------------------------------------------------------------

    def seed_with_chars(self, chars: set[str]):
        """Seed tree with one root per unique character."""
        for ch in sorted(chars):
            voice = self.codebook.encode(ch)
            self._spawn_root(voice)
        self._seed_roots = False

    def cascade(self, input_tensor: torch.Tensor, current_char: str, next_char: str):
        """One actualization cascade. The tree walks itself.

        Time is NOT external. Each actualization increments event_counter.
        The cascade IS the clock.
        """
        self.input_counter += 1

        # Phase 0: Evaluate last prediction
        self._evaluate_prediction()

        # Phase 1: Broadcast input to all nodes (top-down filter)
        self._broadcast(input_tensor)

        # Phase 2: Leaf actualization (batch GPU parallel)
        leaf_events = self._actualize_leaves()

        # Phase 3: Cascade upward (level by level)
        cascade_events = self._cascade_upward()

        total_events = leaf_events + cascade_events
        self._events_per_cascade.append(total_events)

        # Phase 4: Root collapse -> prediction
        pred_char, pred_tensor = self._root_collapse_and_predict()
        self._last_pred_tensor = pred_tensor
        self._last_target_char = next_char

        # Phase 5: Growth (deferred)
        self._process_growth(input_tensor)

        # Phase 6: Decay and dissolution
        self._decay_and_dissolve()

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def set_phase(self, name: str):
        self._current_phase = name

    def get_max_depth(self) -> int:
        return self._max_depth

    def phase_acc(self, phase: str) -> float:
        accs = self.phase_accuracy.get(phase, [])
        return sum(accs) / len(accs) if accs else 0.0

    def rolling_acc(self, window: int = 50) -> float:
        recent = self.accuracy_history[-window:]
        return sum(recent) / len(recent) if recent else 0.0


# ==========================================================================
#  Landauer Reinjection
# ==========================================================================

def landauer_reinject(tree: CascadeTree) -> int:
    """Spawn informed children on crystallized leaves + soften entropy."""
    scored = []
    for leaf_name, res_sum in tree.leaf_residuals.items():
        node = tree.nodes.get(leaf_name)
        if node is None:
            continue
        ev = tree._get_entropy_var(leaf_name)
        if ev >= XI_SEC:
            continue  # not crystallized
        count = tree.leaf_residual_counts.get(leaf_name, 1)
        res_energy = float(torch.norm(res_sum))
        scored.append((leaf_name, res_sum, res_energy, count))

    scored.sort(key=lambda x: -x[2])
    spawned = 0
    available = tree.max_nodes - len(tree.nodes)

    for leaf_name, res_sum, res_energy, count in scored:
        if spawned >= available:
            break
        mean_res = res_sum / max(1, count)
        norm = float(torch.norm(mean_res))
        if norm < XI_SEC:
            continue
        child_voice = (mean_res / (norm + 1e-8)) * math.sqrt(norm)
        child = tree._spawn_child(leaf_name, child_voice.clone())
        if child:
            spawned += 1

    # Soften entropy history (phase transition)
    for node in tree.nodes.values():
        if len(node.entropy_history) > 5:
            node.entropy_history = node.entropy_history[-5:]

    # Reset leaf tracking
    tree.leaf_residuals.clear()
    tree.leaf_residual_counts.clear()

    return spawned


# ==========================================================================
#  Corpus
# ==========================================================================

HAMLET = (
    "to be or not to be that is the question "
    "whether tis nobler in the mind to suffer "
    "the slings and arrows of outrageous fortune "
    "or to take arms against a sea of troubles "
    "and by opposing end them to die to sleep "
    "no more and by a sleep to say we end "
    "the heartache and the thousand natural shocks "
    "that flesh is heir to tis a consummation "
    "devoutly to be wished to die to sleep "
    "to sleep perchance to dream ay there is the rub "
    "for in that sleep of death what dreams may come "
    "when we have shuffled off this mortal coil "
    "must give us pause there is the respect "
    "that makes calamity of so long a life "
    "for who would bear the whips and scorns of time "
    "the oppressors wrong the proud mans contumely "
    "the pangs of despised love the laws delay "
    "the insolence of office and the spurns "
    "that patient merit of the unworthy takes "
    "when he himself might his quietus make "
    "with a bare bodkin who would fardels bear "
    "to grunt and sweat under a weary life "
    "but that the dread of something after death "
    "the undiscovered country from whose bourn "
    "no traveller returns puzzles the will "
    "and makes us rather bear those ills we have "
    "than fly to others that we know not of "
    "thus conscience does make cowards of us all "
    "and thus the native hue of resolution "
    "is sicklied over with the pale cast of thought "
    "and enterprises of great pith and moment "
    "with this regard their currents turn awry "
    "and lose the name of action "
)

GENESIS = (
    "in the beginning god created the heaven and the earth "
    "and the earth was without form and void "
    "and darkness was upon the face of the deep "
    "and the spirit of god moved upon the face of the waters "
    "and god said let there be light and there was light "
    "and god saw the light that it was good "
    "and god divided the light from the darkness "
    "and god called the light day and the darkness he called night "
    "and the evening and the morning were the first day "
    "and god said let there be a firmament "
    "in the midst of the waters "
    "and let it divide the waters from the waters "
    "and god made the firmament "
    "and divided the waters which were under the firmament "
    "from the waters which were above the firmament "
    "and it was so and god called the firmament heaven "
    "and the evening and the morning were the second day "
)

PARADISE = (
    "of mans first disobedience and the fruit "
    "of that forbidden tree whose mortal taste "
    "brought death into the world and all our woe "
    "with loss of eden till one greater man "
    "restore us and regain the blissful seat "
    "sing heavenly muse that on the secret top "
    "of oreb or of sinai didst inspire "
    "that shepherd who first taught the chosen seed "
    "in the beginning how the heavens and earth "
    "rose out of chaos or if sion hill "
    "delight thee more and siloas brook that flowed "
    "fast by the oracle of god i thence "
    "invoke thy aid to my adventurous song "
    "that with no middle flight intends to soar "
    "above the aonian mount while it pursues "
    "things unattempted yet in prose or rhyme "
)

CORPUS = HAMLET + GENESIS + PARADISE


# ==========================================================================
#  Main
# ==========================================================================

def main():
    MAX_NODES = 500
    N_EPOCHS = 12

    print("=" * 70)
    print("  SPIKE K CASCADE -- Event-Driven Bottom-Up Actualization")
    print(f"  Device: {DEVICE}")
    print(f"  Corpus: {len(CORPUS)} chars | Epochs: {N_EPOCHS} | Max nodes: {MAX_NODES}")
    print(f"  Time is emergent. The cascade IS the clock.")
    print("=" * 70)

    torch.manual_seed(42)
    codebook = FastCodebook()
    tree = CascadeTree(codebook, max_nodes=MAX_NODES)

    unique = sorted(set(CORPUS))
    tree.seed_with_chars(set(CORPUS))
    print(f"  Unique characters: {len(unique)}")

    print(f"\n  {'Ep':>3s} | {'Acc':>6s} | {'Nodes':>5s} | {'Roots':>3s} | {'Depth':>3s} | "
          f"{'Cryst':>5s} | {'Ev/Cas':>6s} | {'Leaf%':>5s} | {'Time':>6s} | {'ch/s':>6s}")
    print(f"  {'-'*3}-+-{'-'*6}-+-{'-'*5}-+-{'-'*3}-+-{'-'*3}-+-"
          f"{'-'*5}-+-{'-'*6}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}")

    total_start = time.time()
    epoch_accs = []

    for epoch in range(1, N_EPOCHS + 1):
        phase = f"epoch_{epoch}"
        tree.set_phase(phase)
        tree._events_per_cascade.clear()
        tree._cascade_depths.clear()
        tree._leaf_fire_rates.clear()

        t0 = time.time()
        for i in range(len(CORPUS) - 1):
            ch = CORPUS[i]
            nxt = CORPUS[i + 1]
            tensor = codebook.encode(ch)
            tree.cascade(tensor, ch, nxt)

        elapsed = time.time() - t0
        chars_per_sec = len(CORPUS) / elapsed if elapsed > 0 else 0
        acc = tree.phase_acc(phase)
        epoch_accs.append(acc)
        depth = tree.get_max_depth()

        n_cryst = sum(1 for n in tree.nodes if tree._get_entropy_var(n) < XI_SEC)
        cryst_pct = n_cryst / max(1, len(tree.nodes))

        mean_epc = sum(tree._events_per_cascade) / len(tree._events_per_cascade) \
            if tree._events_per_cascade else 0
        mean_lfr = sum(tree._leaf_fire_rates) / len(tree._leaf_fire_rates) \
            if tree._leaf_fire_rates else 0

        # Landauer reinjection
        spawned = landauer_reinject(tree)

        print(
            f"  {epoch:3d} | "
            f"{acc:5.1%} | "
            f"{len(tree.nodes):5d} | "
            f"{len(tree.root_names):3d} | "
            f"{depth:3d} | "
            f"{cryst_pct:4.0%} | "
            f"{mean_epc:6.1f} | "
            f"{mean_lfr:4.0%} | "
            f"{elapsed:5.1f}s | "
            f"{chars_per_sec:5.0f}"
        )

    total_elapsed = time.time() - total_start

    # Analysis
    print(f"\n{'='*70}")
    print(f"  CASCADE RESULTS")
    print(f"{'='*70}")

    half = len(epoch_accs) // 2
    first_half = sum(epoch_accs[:half]) / half
    second_half = sum(epoch_accs[half:]) / (len(epoch_accs) - half)
    peak = max(epoch_accs)
    peak_ep = epoch_accs.index(peak) + 1

    print(f"  First half avg:  {first_half:.1%}")
    print(f"  Second half avg: {second_half:.1%}")
    print(f"  Peak:            {peak:.1%} (epoch {peak_ep})")
    print(f"  Colony learns:   {'YES' if second_half > first_half else 'NO'}")
    print(f"  Total time:      {total_elapsed:.1f}s")
    print(f"  Total events:    {tree.event_counter}")
    print(f"  Total inputs:    {tree.input_counter}")
    print(f"  Events/input:    {tree.event_counter / max(1, tree.input_counter):.1f}")

    print(f"\n  LEARNING CURVE:")
    for i, acc in enumerate(epoch_accs):
        bar = "#" * int(acc * 200)
        print(f"    E{i+1:2d}: {acc:5.1%} | {bar}")

    # Character accuracy
    print(f"\n  CHARACTER ACCURACY (top 15):")
    char_accs = {}
    for ch, accs in tree.char_accuracy.items():
        if len(accs) >= 20:
            char_accs[ch] = sum(accs) / len(accs)
    best = sorted(char_accs.items(), key=lambda x: -x[1])[:15]
    for ch, acc in best:
        n = len(tree.char_accuracy[ch])
        bar = "#" * int(acc * 50)
        print(f"    '{ch}': {acc:5.0%} (n={n:5d}) {bar}")

    # Tree
    print(f"\n  FINAL TREE:")
    print(f"    Nodes: {len(tree.nodes)} / {MAX_NODES}")
    print(f"    Roots: {len(tree.root_names)}")
    print(f"    Max depth: {tree.get_max_depth()}")
    leaves = tree._get_leaves()
    print(f"    Leaves: {len(leaves)}")
    n_cryst = sum(1 for n in tree.nodes if tree._get_entropy_var(n) < XI_SEC)
    print(f"    Crystallized: {n_cryst}/{len(tree.nodes)}")
    print(f"    Device: {DEVICE}")

    # Per-root subtree
    for rname in tree.root_names[:10]:
        if rname not in tree.nodes:
            continue
        count = 0
        max_d = 0
        stack = [(rname, 0)]
        while stack:
            nm, d = stack.pop()
            count += 1
            if d > max_d:
                max_d = d
            node = tree.nodes.get(nm)
            if node:
                for cn in node.child_names:
                    if cn in tree.nodes:
                        stack.append((cn, d + 1))
        if count > 1:
            print(f"    {rname}: {count} nodes, depth {max_d}")


if __name__ == "__main__":
    main()
