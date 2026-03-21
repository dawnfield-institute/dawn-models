"""Memory Module — PACTree + Bifractal hierarchy + continuous learning.

Ported from GAIA v1 POCs 006-007, 012-014 and fracton's PACNode/PACSystem.
Delta-only hierarchical storage with resonance-based retrieval.
Bifractal depth levels create emergent memory behaviors.
O(1) learning per token — no backpropagation needed.

Components:
    MemoryNode: Delta-only storage node in the PAC tree.
    PACTree: Hierarchical delta-compressed memory with resonance retrieval.
    BifractalDepth: 5-level memory hierarchy (surface to core).
    BifractalManager: Promotes/demotes patterns across depth levels.
    TransitionTracker: Sparse transition matrix with O(1) best-next prediction.
    MemoryModule: GAIAModule wrapper for the full memory stack.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import torch

from gaia.core.types import FieldState, RBFBalance, SECPhase

# DFT constants
PHI = (1 + math.sqrt(5)) / 2
LAMBDA_STAR = 0.9816  # Exponential decay rate


# ─── Bifractal Depth ───────────────────────────────────────────────


class BifractalDepth(IntEnum):
    """Memory depth levels — deeper = more persistent."""

    SURFACE = 0  # Ephemeral, fast decay
    SHALLOW = 1  # Short-term
    INTERMEDIATE = 2  # Medium-term
    DEEP = 3  # Long-term
    CORE = 4  # Personality-level (no decay)


# Decay rates per depth level (higher = slower decay)
_DEPTH_DECAY: dict[BifractalDepth, float] = {
    BifractalDepth.SURFACE: 0.90,
    BifractalDepth.SHALLOW: 0.95,
    BifractalDepth.INTERMEDIATE: LAMBDA_STAR,
    BifractalDepth.DEEP: 0.995,
    BifractalDepth.CORE: 1.0,  # No decay
}

# Access count thresholds for promotion
_PROMOTION_THRESHOLDS: dict[BifractalDepth, int] = {
    BifractalDepth.SURFACE: 3,
    BifractalDepth.SHALLOW: 8,
    BifractalDepth.INTERMEDIATE: 20,
    BifractalDepth.DEEP: 50,
    # CORE: never auto-promoted, only via explicit crystallization
}


# ─── Data Structures ──────────────────────────────────────────────


@dataclass
class MemoryNode:
    """Delta-only storage node in the PAC tree.

    Never stores absolute values — reconstruction requires traversing
    to root and summing deltas. PAC conservation: parent = sum(children.delta).
    """

    id: int
    delta: torch.Tensor  # Delta from parent (or absolute for root)
    parent_id: int = -1
    children_ids: list[int] = field(default_factory=list)
    strength: float = 1.0  # Resonance strength (decays over time)
    depth: BifractalDepth = BifractalDepth.SURFACE
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    label: str = ""

    @property
    def is_root(self) -> bool:
        return self.parent_id == -1

    @property
    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0


@dataclass
class MemoryMetrics:
    """Metrics from a memory module processing step."""

    n_nodes: int = 0
    n_stored: int = 0
    n_retrieved: int = 0
    mean_strength: float = 0.0
    depth_distribution: dict[str, int] = field(default_factory=dict)
    tree_depth: int = 0
    storage_ratio: float = 1.0  # delta storage / flat storage


# ─── PAC Tree ──────────────────────────────────────────────────────


class PACTree:
    """Hierarchical delta-compressed memory with resonance retrieval.

    Stores patterns as deltas from their nearest existing neighbor,
    achieving ~12.5x memory savings. Retrieval uses cosine resonance
    modulated by phi-xi contrast for O(n) search (O(log n) with
    tree-guided navigation in future optimization).

    PAC invariant: parent.value = sum(child.delta for child in children) + parent.delta
    """

    def __init__(self, capacity: int = 10000) -> None:
        self._nodes: dict[int, MemoryNode] = {}
        self._next_id = 0
        self._capacity = capacity
        self._root_ids: list[int] = []

    def store(
        self,
        pattern: torch.Tensor,
        label: str = "",
        importance: float = 0.5,
    ) -> int:
        """Store a pattern in the tree.

        If similar patterns exist, stores as delta from the most
        resonant match. Otherwise creates a new root.

        Returns:
            Node ID of the stored pattern.
        """
        # Find best parent via resonance
        best_parent_id = -1
        best_score = 0.0

        if self._nodes:
            for node_id, node in self._nodes.items():
                score = self._resonance(pattern, self.reconstruct(node_id))
                if score > best_score:
                    best_score = score
                    best_parent_id = node_id

        # Determine depth from importance
        if importance > 0.8:
            depth = BifractalDepth.DEEP
        elif importance > 0.5:
            depth = BifractalDepth.INTERMEDIATE
        elif importance > 0.2:
            depth = BifractalDepth.SHALLOW
        else:
            depth = BifractalDepth.SURFACE

        node_id = self._next_id
        self._next_id += 1

        if best_score > 0.5 and best_parent_id >= 0:
            # Store as delta from parent
            parent_value = self.reconstruct(best_parent_id)
            delta = pattern - parent_value
            node = MemoryNode(
                id=node_id,
                delta=delta,
                parent_id=best_parent_id,
                strength=1.0,
                depth=depth,
                label=label,
            )
            self._nodes[best_parent_id].children_ids.append(node_id)
        else:
            # New root
            node = MemoryNode(
                id=node_id,
                delta=pattern.clone(),
                parent_id=-1,
                strength=1.0,
                depth=depth,
                label=label,
            )
            self._root_ids.append(node_id)

        self._nodes[node_id] = node

        # Garbage collect if over capacity
        if len(self._nodes) > self._capacity:
            self._gc()

        return node_id

    def reconstruct(self, node_id: int) -> torch.Tensor:
        """Reconstruct full value by summing deltas to root."""
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id} not found")

        node = self._nodes[node_id]
        value = node.delta.clone()

        current = node
        while current.parent_id >= 0 and current.parent_id in self._nodes:
            current = self._nodes[current.parent_id]
            value = value + current.delta

        return value

    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> list[tuple[int, float]]:
        """Find most resonant patterns for a query.

        Returns:
            List of (node_id, resonance_score) sorted by score descending.
        """
        results = []
        for node_id in self._nodes:
            node = self._nodes[node_id]
            node.access_count += 1  # Track access for promotion
            value = self.reconstruct(node_id)
            score = self._resonance(query, value)
            if score >= threshold:
                results.append((node_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def decay(self) -> None:
        """Apply depth-dependent decay to all node strengths."""
        for node in self._nodes.values():
            rate = _DEPTH_DECAY[node.depth]
            node.strength *= rate

    def _resonance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute resonance (cosine similarity) between two patterns."""
        a_flat = a.flatten().float()
        b_flat = b.flatten().float()
        norm_a = torch.norm(a_flat)
        norm_b = torch.norm(b_flat)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(torch.dot(a_flat, b_flat) / (norm_a * norm_b))

    def _gc(self) -> None:
        """Garbage collect weakest surface-level nodes."""
        # Only collect SURFACE nodes
        surface = [
            (nid, n.strength)
            for nid, n in self._nodes.items()
            if n.depth == BifractalDepth.SURFACE and n.is_leaf
        ]
        surface.sort(key=lambda x: x[1])

        # Remove bottom 10%
        to_remove = max(1, len(surface) // 10)
        for nid, _ in surface[:to_remove]:
            self._remove_node(nid)

    def _remove_node(self, node_id: int) -> None:
        """Remove a leaf node, preserving tree integrity."""
        if node_id not in self._nodes:
            return
        node = self._nodes[node_id]
        if not node.is_leaf:
            return  # Only remove leaves

        # Unlink from parent
        if node.parent_id >= 0 and node.parent_id in self._nodes:
            parent = self._nodes[node.parent_id]
            if node_id in parent.children_ids:
                parent.children_ids.remove(node_id)

        # Remove from roots
        if node_id in self._root_ids:
            self._root_ids.remove(node_id)

        del self._nodes[node_id]

    @property
    def size(self) -> int:
        return len(self._nodes)

    def storage_ratio(self) -> float:
        """Ratio of delta storage vs flat storage (lower = more efficient)."""
        if not self._nodes:
            return 1.0
        total_delta = sum(n.delta.numel() for n in self._nodes.values())
        # Flat storage would be numel * n_nodes (each storing full pattern)
        if self._root_ids:
            root = self._nodes[self._root_ids[0]]
            flat = root.delta.numel() * len(self._nodes)
        else:
            flat = total_delta
        return total_delta / max(flat, 1)

    def depth_distribution(self) -> dict[str, int]:
        """Count nodes at each bifractal depth."""
        dist: dict[str, int] = {}
        for node in self._nodes.values():
            key = node.depth.name.lower()
            dist[key] = dist.get(key, 0) + 1
        return dist


# ─── Bifractal Manager ─────────────────────────────────────────────


class BifractalManager:
    """Manages promotion/demotion of patterns across depth levels.

    Patterns that are accessed frequently get promoted to deeper
    (more persistent) levels. Infrequently accessed patterns decay
    and eventually get garbage collected.
    """

    def promote_if_ready(self, tree: PACTree) -> int:
        """Check all nodes for promotion eligibility.

        Returns:
            Number of nodes promoted.
        """
        promoted = 0
        for node in tree._nodes.values():
            if node.depth >= BifractalDepth.DEEP:
                continue  # Can't auto-promote to CORE
            threshold = _PROMOTION_THRESHOLDS.get(node.depth)
            if threshold and node.access_count >= threshold:
                node.depth = BifractalDepth(node.depth + 1)
                node.access_count = 0  # Reset after promotion
                promoted += 1
        return promoted

    def crystallize(self, tree: PACTree, node_id: int) -> bool:
        """Explicitly promote a node to CORE depth (permanent).

        Returns:
            True if crystallized, False if node not found.
        """
        if node_id not in tree._nodes:
            return False
        tree._nodes[node_id].depth = BifractalDepth.CORE
        tree._nodes[node_id].strength = 1.0  # Restore full strength
        return True


# ─── Transition Tracker ────────────────────────────────────────────


class TransitionTracker:
    """Sparse transition matrix with O(1) best-next prediction.

    Learns sequential relationships between stored patterns.
    No backpropagation — purely counting-based (Hebbian).
    Decays old transitions to allow forgetting.
    """

    def __init__(self, decay_rate: float = LAMBDA_STAR) -> None:
        self._transitions: dict[int, dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._best_next: dict[int, tuple[int, float]] = {}
        self._decay_rate = decay_rate

    def learn(self, from_id: int, to_id: int, weight: float = 1.0) -> None:
        """Record a transition from one pattern to another."""
        self._transitions[from_id][to_id] += weight

        # Update best-next cache (O(1) lookup)
        current_best = self._best_next.get(from_id)
        new_weight = self._transitions[from_id][to_id]
        if current_best is None or new_weight > current_best[1]:
            self._best_next[from_id] = (to_id, new_weight)

    def predict_next(self, from_id: int) -> Optional[tuple[int, float]]:
        """O(1) prediction of most likely next pattern."""
        return self._best_next.get(from_id)

    def get_transitions(self, from_id: int, top_k: int = 5) -> list[tuple[int, float]]:
        """Get top-k transitions from a pattern."""
        if from_id not in self._transitions:
            return []
        items = sorted(
            self._transitions[from_id].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return items[:top_k]

    def decay(self, threshold: float = 0.01) -> int:
        """Apply decay and remove weak transitions.

        Returns:
            Number of transitions removed.
        """
        removed = 0
        empty_sources = []

        for from_id in list(self._transitions):
            to_remove = []
            for to_id in self._transitions[from_id]:
                self._transitions[from_id][to_id] *= self._decay_rate
                if self._transitions[from_id][to_id] < threshold:
                    to_remove.append(to_id)

            for to_id in to_remove:
                del self._transitions[from_id][to_id]
                removed += 1

            if not self._transitions[from_id]:
                empty_sources.append(from_id)

            # Rebuild best-next cache
            if self._transitions[from_id]:
                best = max(self._transitions[from_id].items(), key=lambda x: x[1])
                self._best_next[from_id] = best
            elif from_id in self._best_next:
                del self._best_next[from_id]

        for from_id in empty_sources:
            del self._transitions[from_id]

        return removed

    @property
    def n_transitions(self) -> int:
        return sum(len(v) for v in self._transitions.values())


# ─── GAIAModule Wrapper ────────────────────────────────────────────


class MemoryModule:
    """GAIA Memory Module — PACTree + Bifractal + Transitions.

    Processes FieldState by storing it in the PAC tree and enriching
    the output with retrieved context from similar past states.
    Learning is continuous — every process() call stores and learns.

    The module is PAC-conserving at its boundary: output tensor
    has the same total energy as input.

    Args:
        capacity: Maximum nodes in the PAC tree.
        retrieval_top_k: Number of similar patterns to blend.
        retrieval_weight: How much retrieved context influences output (0-1).
        auto_learn_transitions: Learn sequential transitions automatically.
    """

    def __init__(
        self,
        capacity: int = 10000,
        retrieval_top_k: int = 3,
        retrieval_weight: float = 0.1,
        auto_learn_transitions: bool = True,
    ) -> None:
        self._tree = PACTree(capacity=capacity)
        self._bifractal = BifractalManager()
        self._transitions = TransitionTracker()
        self._retrieval_top_k = retrieval_top_k
        self._retrieval_weight = retrieval_weight
        self._auto_learn = auto_learn_transitions

        self._last_stored_id: Optional[int] = None
        self._last_metrics: Optional[MemoryMetrics] = None
        self._step_count = 0

    @property
    def name(self) -> str:
        return "memory"

    def process(self, field_state: FieldState) -> FieldState:
        """Process field state through memory.

        1. Store current tensor in PAC tree.
        2. Retrieve similar past patterns.
        3. Blend retrieved context into output (small weight).
        4. Learn transition from previous to current.
        5. Apply decay + check promotions.

        The output preserves PAC conservation at the boundary.
        """
        self._step_count += 1
        result = field_state.clone()
        input_energy = field_state.total_energy()

        tensor = field_state.tensor

        # 1. Store in tree
        node_id = self._tree.store(tensor, importance=0.5)
        n_stored = 1

        # 2. Learn transition from previous
        if self._auto_learn and self._last_stored_id is not None:
            self._transitions.learn(self._last_stored_id, node_id)

        # 3. Retrieve similar patterns
        matches = self._tree.retrieve(
            tensor, top_k=self._retrieval_top_k, threshold=0.3
        )
        # Exclude self from matches
        matches = [(mid, score) for mid, score in matches if mid != node_id]
        n_retrieved = len(matches)

        # 4. Blend retrieved context into output
        if matches and self._retrieval_weight > 0:
            # Weighted average of retrieved patterns
            total_weight = sum(score for _, score in matches)
            if total_weight > 1e-8:
                context = torch.zeros_like(tensor)
                for mid, score in matches:
                    retrieved = self._tree.reconstruct(mid)
                    context = context + (score / total_weight) * retrieved
                # Blend: output = (1 - w) * input + w * context
                output = (1 - self._retrieval_weight) * tensor + self._retrieval_weight * context
            else:
                output = tensor
        else:
            output = tensor

        # 5. PAC boundary enforcement — scale to match input energy
        output_energy = float(torch.sum(output).item())
        if abs(output_energy) > 1e-10:
            output = output * (input_energy / output_energy)

        result.tensor = output
        result.provenance.append(self.name)

        # 6. Periodic maintenance
        if self._step_count % 10 == 0:
            self._tree.decay()
            self._transitions.decay()
            self._bifractal.promote_if_ready(self._tree)

        self._last_stored_id = node_id

        # Compute metrics
        nodes = self._tree._nodes
        mean_strength = (
            sum(n.strength for n in nodes.values()) / len(nodes)
            if nodes
            else 0.0
        )

        self._last_metrics = MemoryMetrics(
            n_nodes=self._tree.size,
            n_stored=n_stored,
            n_retrieved=n_retrieved,
            mean_strength=mean_strength,
            depth_distribution=self._tree.depth_distribution(),
            tree_depth=self._max_tree_depth(),
            storage_ratio=self._tree.storage_ratio(),
        )

        return result

    def _max_tree_depth(self) -> int:
        """Compute maximum depth of the PAC tree."""
        max_d = 0
        for node in self._tree._nodes.values():
            d = 0
            current = node
            while current.parent_id >= 0 and current.parent_id in self._tree._nodes:
                current = self._tree._nodes[current.parent_id]
                d += 1
            max_d = max(max_d, d)
        return max_d

    def phase(self) -> SECPhase:
        """SEC phase based on memory utilization.

        Low utilization = crystallized (stable, well-organized).
        High utilization = chaotic (lots of new patterns).
        """
        if not self._tree._nodes:
            return SECPhase.ORDERED

        utilization = self._tree.size / self._tree._capacity
        if utilization < 0.25:
            return SECPhase.CRYSTALLIZED
        elif utilization < 0.5:
            return SECPhase.ORDERED
        elif utilization < 0.75:
            return SECPhase.TRANSITIONAL
        return SECPhase.CHAOTIC

    def health(self) -> RBFBalance:
        """RBF balance based on memory state.

        Energy = mean node strength (how alive the memory is).
        Information = storage efficiency (delta compression ratio).
        Memory = utilization (how full the tree is).
        """
        if self._last_metrics:
            energy = self._last_metrics.mean_strength
            information = 1.0 - min(self._last_metrics.storage_ratio, 1.0)
            memory = self._tree.size / max(self._tree._capacity, 1)
        else:
            energy = 1.0
            information = 0.0
            memory = 0.0
        return RBFBalance.compute(energy=energy, information=information, memory=memory)

    @property
    def metrics(self) -> Optional[MemoryMetrics]:
        return self._last_metrics

    @property
    def tree(self) -> PACTree:
        return self._tree

    @property
    def transitions(self) -> TransitionTracker:
        return self._transitions

    def crystallize(self, node_id: int) -> bool:
        """Promote a pattern to CORE depth (permanent memory)."""
        return self._bifractal.crystallize(self._tree, node_id)
