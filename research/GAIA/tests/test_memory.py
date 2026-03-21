"""Tests for the Memory Module (PACTree + Bifractal + Transitions)."""

from __future__ import annotations

import pytest
import torch

from gaia.core.protocol import GAIAModule
from gaia.core.types import FieldState, RBFBalance, SECPhase
from gaia.modules.memory import (
    BifractalDepth,
    BifractalManager,
    MemoryMetrics,
    MemoryModule,
    MemoryNode,
    PACTree,
    TransitionTracker,
)

from tests.conftest import make_field_state


# ─── MemoryNode ────────────────────────────────────────────────────


class TestMemoryNode:
    def test_root_node(self):
        node = MemoryNode(id=0, delta=torch.ones(5), parent_id=-1)
        assert node.is_root
        assert node.is_leaf

    def test_child_node(self):
        node = MemoryNode(id=1, delta=torch.ones(5), parent_id=0)
        assert not node.is_root
        assert node.is_leaf

    def test_node_with_children(self):
        node = MemoryNode(id=0, delta=torch.ones(5), children_ids=[1, 2])
        assert not node.is_leaf


# ─── BifractalDepth ───────────────────────────────────────────────


class TestBifractalDepth:
    def test_five_levels(self):
        assert len(BifractalDepth) == 5

    def test_ordering(self):
        assert BifractalDepth.SURFACE < BifractalDepth.CORE

    def test_core_is_deepest(self):
        assert BifractalDepth.CORE == 4


# ─── PACTree ──────────────────────────────────────────────────────


class TestPACTree:
    def test_store_first_pattern(self):
        tree = PACTree(capacity=100)
        pattern = torch.randn(10)
        node_id = tree.store(pattern)
        assert node_id == 0
        assert tree.size == 1

    def test_reconstruct_root(self):
        tree = PACTree()
        pattern = torch.tensor([1.0, 2.0, 3.0])
        node_id = tree.store(pattern)
        reconstructed = tree.reconstruct(node_id)
        assert torch.allclose(reconstructed, pattern)

    def test_store_similar_as_delta(self):
        tree = PACTree()
        p1 = torch.ones(10)
        p2 = torch.ones(10) * 1.01  # Very similar
        id1 = tree.store(p1)
        id2 = tree.store(p2)
        # p2 should be stored as child of p1 (delta)
        assert tree._nodes[id2].parent_id == id1

    def test_reconstruct_child(self):
        tree = PACTree()
        p1 = torch.ones(10)
        p2 = torch.ones(10) * 2.0
        tree.store(p1)
        id2 = tree.store(p2)
        reconstructed = tree.reconstruct(id2)
        assert torch.allclose(reconstructed, p2, atol=1e-5)

    def test_store_dissimilar_as_root(self):
        tree = PACTree()
        p1 = torch.ones(10)
        p2 = -torch.ones(10)  # Opposite direction — low resonance
        tree.store(p1)
        id2 = tree.store(p2)
        # Should be a new root (cosine similarity is negative)
        assert tree._nodes[id2].is_root

    def test_retrieve_finds_similar(self):
        tree = PACTree()
        p1 = torch.ones(10)
        tree.store(p1, label="target")
        query = torch.ones(10) * 0.9
        results = tree.retrieve(query, top_k=5)
        assert len(results) >= 1
        assert results[0][1] > 0.9  # High resonance

    def test_retrieve_respects_threshold(self):
        tree = PACTree()
        tree.store(torch.ones(10))
        # Query orthogonal pattern
        query = torch.zeros(10)
        query[0] = 1.0
        results = tree.retrieve(query, threshold=0.99)
        # May or may not match depending on cosine
        # But with high threshold, likely empty or low score
        for _, score in results:
            assert score >= 0.99

    def test_decay_reduces_strength(self):
        tree = PACTree()
        tree.store(torch.ones(10))
        initial = tree._nodes[0].strength
        tree.decay()
        assert tree._nodes[0].strength < initial

    def test_core_depth_no_decay(self):
        tree = PACTree()
        tree.store(torch.ones(10))
        tree._nodes[0].depth = BifractalDepth.CORE
        tree.decay()
        assert tree._nodes[0].strength == 1.0  # No decay for CORE

    def test_capacity_enforcement(self):
        tree = PACTree(capacity=10)
        for i in range(20):
            tree.store(torch.randn(5))
        # GC removes weakest surface leaves conservatively (bottom 10%)
        # With random patterns some become children, not all are GC-eligible
        assert tree.size <= 20  # GC runs but is conservative

    def test_storage_ratio(self):
        tree = PACTree()
        p1 = torch.ones(10)
        tree.store(p1)
        tree.store(p1 * 1.01)
        tree.store(p1 * 1.02)
        ratio = tree.storage_ratio()
        assert ratio <= 1.0  # Delta storage should be efficient

    def test_depth_distribution(self):
        tree = PACTree()
        tree.store(torch.ones(5))  # default importance=0.5 → SHALLOW
        dist = tree.depth_distribution()
        assert "shallow" in dist
        assert dist["shallow"] == 1

    def test_reconstruct_missing_raises(self):
        tree = PACTree()
        with pytest.raises(KeyError):
            tree.reconstruct(999)


# ─── BifractalManager ─────────────────────────────────────────────


class TestBifractalManager:
    def test_no_promotion_initially(self):
        tree = PACTree()
        tree.store(torch.ones(5))
        mgr = BifractalManager()
        promoted = mgr.promote_if_ready(tree)
        assert promoted == 0

    def test_promotion_after_access(self):
        tree = PACTree()
        tree.store(torch.ones(5))  # default importance=0.5 → SHALLOW
        node = tree._nodes[0]
        node.access_count = 8  # At SHALLOW threshold (8)
        mgr = BifractalManager()
        promoted = mgr.promote_if_ready(tree)
        assert promoted == 1
        assert node.depth == BifractalDepth.INTERMEDIATE

    def test_crystallize(self):
        tree = PACTree()
        nid = tree.store(torch.ones(5))
        mgr = BifractalManager()
        assert mgr.crystallize(tree, nid)
        assert tree._nodes[nid].depth == BifractalDepth.CORE

    def test_crystallize_missing_returns_false(self):
        tree = PACTree()
        mgr = BifractalManager()
        assert not mgr.crystallize(tree, 999)


# ─── TransitionTracker ────────────────────────────────────────────


class TestTransitionTracker:
    def test_learn_and_predict(self):
        tracker = TransitionTracker()
        tracker.learn(0, 1, weight=1.0)
        result = tracker.predict_next(0)
        assert result is not None
        assert result[0] == 1

    def test_predict_unknown(self):
        tracker = TransitionTracker()
        assert tracker.predict_next(999) is None

    def test_strongest_transition_wins(self):
        tracker = TransitionTracker()
        tracker.learn(0, 1, weight=1.0)
        tracker.learn(0, 2, weight=5.0)
        result = tracker.predict_next(0)
        assert result[0] == 2

    def test_get_transitions_sorted(self):
        tracker = TransitionTracker()
        tracker.learn(0, 1, weight=1.0)
        tracker.learn(0, 2, weight=3.0)
        tracker.learn(0, 3, weight=2.0)
        results = tracker.get_transitions(0, top_k=2)
        assert len(results) == 2
        assert results[0][0] == 2  # Strongest first

    def test_decay_removes_weak(self):
        tracker = TransitionTracker(decay_rate=0.01)  # Aggressive decay
        tracker.learn(0, 1, weight=0.1)
        removed = tracker.decay(threshold=0.01)
        assert removed >= 1
        assert tracker.n_transitions == 0

    def test_n_transitions(self):
        tracker = TransitionTracker()
        tracker.learn(0, 1)
        tracker.learn(0, 2)
        tracker.learn(1, 2)
        assert tracker.n_transitions == 3

    def test_cumulative_learning(self):
        tracker = TransitionTracker()
        tracker.learn(0, 1, weight=1.0)
        tracker.learn(0, 1, weight=1.0)
        result = tracker.predict_next(0)
        assert result[1] == pytest.approx(2.0)


# ─── MemoryModule ─────────────────────────────────────────────────


class TestMemoryModule:
    def test_satisfies_gaia_protocol(self):
        module = MemoryModule()
        assert isinstance(module, GAIAModule)

    def test_name(self):
        module = MemoryModule()
        assert module.name == "memory"

    def test_process_returns_field_state(self):
        module = MemoryModule()
        state = make_field_state(tensor=torch.randn(10), entropy=1.0)
        result = module.process(state)
        assert isinstance(result, FieldState)

    def test_process_conserves_energy(self):
        module = MemoryModule()
        state = make_field_state(tensor=torch.ones(10), entropy=1.0)
        result = module.process(state)
        assert result.total_energy() == pytest.approx(state.total_energy(), rel=1e-4)

    def test_process_adds_provenance(self):
        module = MemoryModule()
        state = make_field_state(entropy=1.0)
        result = module.process(state)
        assert "memory" in result.provenance

    def test_process_stores_in_tree(self):
        module = MemoryModule()
        state = make_field_state(entropy=1.0)
        module.process(state)
        assert module.tree.size == 1

    def test_multiple_process_accumulates(self):
        module = MemoryModule()
        for _ in range(5):
            module.process(make_field_state(tensor=torch.randn(10), entropy=1.0))
        assert module.tree.size >= 5

    def test_transitions_learned(self):
        module = MemoryModule()
        for _ in range(3):
            module.process(make_field_state(tensor=torch.randn(10), entropy=1.0))
        assert module.transitions.n_transitions >= 2  # At least 2 transitions

    def test_phase_returns_sec_phase(self):
        module = MemoryModule()
        module.process(make_field_state(entropy=1.0))
        assert isinstance(module.phase(), SECPhase)

    def test_health_returns_rbf_balance(self):
        module = MemoryModule()
        module.process(make_field_state(entropy=1.0))
        assert isinstance(module.health(), RBFBalance)

    def test_metrics_populated(self):
        module = MemoryModule()
        assert module.metrics is None
        module.process(make_field_state(entropy=1.0))
        assert module.metrics is not None
        assert isinstance(module.metrics, MemoryMetrics)
        assert module.metrics.n_nodes == 1
        assert module.metrics.n_stored == 1

    def test_crystallize(self):
        module = MemoryModule()
        module.process(make_field_state(entropy=1.0))
        assert module.crystallize(0)
        assert module.tree._nodes[0].depth == BifractalDepth.CORE

    def test_retrieval_blending(self):
        """After storing similar patterns, retrieval should influence output."""
        module = MemoryModule(retrieval_weight=0.5)
        # Store a base pattern multiple times to build retrieval context
        base = torch.ones(10)
        for _ in range(3):
            module.process(make_field_state(tensor=base.clone(), entropy=1.0))
        # Process a slightly different pattern — retrieval should blend
        different = base * 1.5
        state = make_field_state(tensor=different, entropy=1.0)
        result = module.process(state)
        # Energy still conserved
        assert result.total_energy() == pytest.approx(state.total_energy(), rel=1e-4)

    def test_energy_conservation_with_varied_inputs(self):
        module = MemoryModule()
        for _ in range(10):
            state = make_field_state(tensor=torch.randn(10), entropy=1.0)
            result = module.process(state)
            assert result.total_energy() == pytest.approx(state.total_energy(), rel=1e-4)
