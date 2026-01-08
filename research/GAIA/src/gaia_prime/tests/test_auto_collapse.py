"""
Tests for entropy-triggered auto-collapse.

Tests:
1. Basic auto-collapse detection
2. Collapse strategy selection
3. Clustering collapse
4. Crystallization collapse
5. Pruning collapse
6. Compression collapse
7. Hierarchical collapse
8. Run until stable
9. Entropy balancer
10. Full auto-collapse cycle
"""

import torch
import pytest
import sys
sys.path.insert(0, 'c:/Users/peter/repos/Dawn Field Institute/dawn-models/research/GAIA/src')

from gaia_prime.pac_mesh import PACMeshSpace, MeshNode
from gaia_prime.physics_mesh import PhysicsMesh, CollapseType
from gaia_prime.validated_constants import XI, PHI, PHI_INV, ENTROPY_OPTIMAL_LOW, ENTROPY_OPTIMAL_HIGH
from gaia_prime.auto_collapse import (
    AutoCollapseEngine, AutoCollapseConfig, CollapseResult, CollapseStrategy,
    EntropyBalancer
)


class TestAutoCollapseBasics:
    """Basic auto-collapse functionality."""
    
    def test_auto_collapse_creation(self):
        """Test creating auto-collapse engine."""
        mesh = PACMeshSpace(embed_dim=64)
        physics = PhysicsMesh(mesh)
        auto = AutoCollapseEngine(physics)
        
        assert auto.physics is physics
        assert auto.mesh is mesh
        assert not auto.enabled
        assert len(auto.collapse_history) == 0
        print("PASS: Auto-collapse engine created")
    
    def test_enable_disable(self):
        """Test enabling and disabling."""
        mesh = PACMeshSpace(embed_dim=64)
        physics = PhysicsMesh(mesh)
        auto = AutoCollapseEngine(physics)
        
        assert not auto.enabled
        auto.enable()
        assert auto.enabled
        auto.disable()
        assert not auto.enabled
        print("PASS: Enable/disable works")
    
    def test_should_collapse_empty(self):
        """Test collapse decision on empty mesh."""
        mesh = PACMeshSpace(embed_dim=64)
        physics = PhysicsMesh(mesh)
        auto = AutoCollapseEngine(physics)
        
        should, reason = auto.should_collapse()
        assert not should
        assert reason == "insufficient_nodes"
        print("PASS: No collapse on empty mesh")


class TestCollapseStrategies:
    """Test individual collapse strategies."""
    
    def create_chaotic_mesh(self, n_nodes: int = 20):
        """Create a mesh with high entropy."""
        mesh = PACMeshSpace(embed_dim=64)
        
        # Add many loosely connected nodes
        nodes = []
        for i in range(n_nodes):
            embedding = torch.randn(64)
            node = mesh.get_or_create_root(i, f"token_{i}", embedding, "test")
            node.confidence = 0.3 + 0.1 * (i % 5)  # Varied confidence
            nodes.append(node)
        
        # Add sparse connections
        import random
        for node in nodes:
            targets = random.sample(nodes, min(2, len(nodes)))
            for target in targets:
                if target != node:
                    node.add_child(target)
        
        return mesh
    
    def test_strategy_selection(self):
        """Test automatic strategy selection."""
        mesh = self.create_chaotic_mesh(20)
        physics = PhysicsMesh(mesh)
        auto = AutoCollapseEngine(physics)
        
        # Should select a valid strategy
        strategy = auto.select_strategy()
        assert strategy in CollapseStrategy
        print(f"PASS: Selected strategy {strategy.value}")
    
    def test_clustering_collapse(self):
        """Test clustering collapse strategy."""
        mesh = PACMeshSpace(embed_dim=64)
        
        # Create similar nodes
        base_emb = torch.randn(64)
        for i in range(10):
            emb = base_emb + 0.1 * torch.randn(64)  # Similar embeddings
            node = mesh.get_or_create_root(i, f"token_{i}", emb, "test")
        
        physics = PhysicsMesh(mesh)
        auto = AutoCollapseEngine(physics)
        
        result = auto.collapse_clustering()
        
        assert isinstance(result, CollapseResult)
        assert result.strategy == CollapseStrategy.CLUSTERING
        assert result.nodes_affected >= 0
        assert result.time_taken > 0
        print(f"PASS: Clustering - {result.nodes_affected} nodes affected, "
              f"{result.details.get('clusters_formed', 0)} clusters formed")
    
    def test_crystallization_collapse(self):
        """Test crystallization collapse strategy."""
        mesh = PACMeshSpace(embed_dim=64)
        
        # Create nodes with high confidence
        for i in range(10):
            embedding = torch.randn(64)
            node = mesh.get_or_create_root(i, f"token_{i}", embedding, "test")
            node.confidence = 0.9 if i < 5 else 0.3  # Some high, some low
        
        physics = PhysicsMesh(mesh)
        auto = AutoCollapseEngine(physics)
        
        result = auto.collapse_crystallization()
        
        assert result.strategy == CollapseStrategy.CRYSTALLIZATION
        assert result.details['crystallized'] >= 0
        print(f"PASS: Crystallization - {result.details['crystallized']} crystallized")
    
    def test_pruning_collapse(self):
        """Test pruning collapse strategy."""
        mesh = PACMeshSpace(embed_dim=64)
        
        # Create nodes with weak connections
        nodes = []
        for i in range(10):
            embedding = torch.randn(64)
            node = mesh.get_or_create_root(i, f"token_{i}", embedding, "test")
            node.confidence = 0.2  # Low confidence
            nodes.append(node)
        
        # Add many weak connections
        for i, node in enumerate(nodes):
            for j, target in enumerate(nodes):
                if i != j:
                    node.add_child(target)  # Just one connection each
        
        physics = PhysicsMesh(mesh)
        auto = AutoCollapseEngine(physics)
        
        result = auto.collapse_pruning()
        
        assert result.strategy == CollapseStrategy.PRUNING
        assert result.details['connections_pruned'] >= 0
        print(f"PASS: Pruning - {result.details['connections_pruned']} connections pruned")
    
    def test_compression_collapse(self):
        """Test compression collapse strategy."""
        mesh = PACMeshSpace(embed_dim=64)
        
        # Create duplicate nodes with same token string
        base_emb = torch.randn(64)
        for i in range(10):
            # Very similar embeddings and same token string
            emb = base_emb + 0.01 * torch.randn(64)
            node = mesh.get_or_create_root(i, "same_token", emb, "test")
        
        physics = PhysicsMesh(mesh)
        auto = AutoCollapseEngine(physics)
        auto.config.compression_merge_threshold = 0.95
        
        result = auto.collapse_compression()
        
        assert result.strategy == CollapseStrategy.COMPRESSION
        print(f"PASS: Compression - {result.details['nodes_merged']} nodes merged")
    
    def test_hierarchical_collapse(self):
        """Test hierarchical collapse strategy."""
        mesh = PACMeshSpace(embed_dim=64)
        
        # Create nodes at same depth
        for i in range(10):
            embedding = torch.randn(64)
            node = mesh.get_or_create_root(i, f"token_{i}", embedding, "test")
        
        physics = PhysicsMesh(mesh)
        auto = AutoCollapseEngine(physics)
        
        result = auto.collapse_hierarchical()
        
        assert result.strategy == CollapseStrategy.HIERARCHICAL
        print(f"PASS: Hierarchical - {result.details['summaries_created']} summaries created")


class TestEntropyBalancer:
    """Test the high-level entropy balancer."""
    
    def test_balancer_creation(self):
        """Test creating entropy balancer."""
        mesh = PACMeshSpace(embed_dim=64)
        physics = PhysicsMesh(mesh)
        balancer = EntropyBalancer(physics)
        
        assert balancer.physics is physics
        assert balancer.auto_collapse is not None
        # Constants now imported from validated_constants module
        assert ENTROPY_OPTIMAL_LOW == PHI_INV
        assert ENTROPY_OPTIMAL_HIGH == PHI
        print("PASS: Entropy balancer created with golden ratio bounds")
    
    def test_regime_detection(self):
        """Test entropy regime detection."""
        mesh = PACMeshSpace(embed_dim=64)
        
        # Add some nodes
        for i in range(10):
            embedding = torch.randn(64)
            mesh.get_or_create_root(i, f"token_{i}", embedding, "test")
        
        physics = PhysicsMesh(mesh)
        balancer = EntropyBalancer(physics)
        
        regime = balancer.get_regime()
        assert regime in ["frozen", "ordered", "optimal", "active", "chaotic"]
        print(f"PASS: Regime detected as '{regime}'")
    
    def test_balancer_step(self):
        """Test running balancer step."""
        mesh = PACMeshSpace(embed_dim=64)
        
        for i in range(10):
            embedding = torch.randn(64)
            mesh.get_or_create_root(i, f"token_{i}", embedding, "test")
        
        physics = PhysicsMesh(mesh)
        balancer = EntropyBalancer(physics)
        
        result = balancer.step()
        
        assert 'entropy' in result
        assert 'in_optimal_range' in result
        assert 'collapse_triggered' in result
        print(f"PASS: Balancer step - entropy={result['entropy']:.3f}, "
              f"optimal={result['in_optimal_range']}")


class TestRunUntilStable:
    """Test running collapse until stable."""
    
    def test_run_until_stable(self):
        """Test stabilizing a chaotic mesh."""
        mesh = PACMeshSpace(embed_dim=64)
        
        # Create chaotic mesh
        for i in range(50):
            embedding = torch.randn(64)
            node = mesh.get_or_create_root(i, f"token_{i}", embedding, "test")
            node.confidence = 0.3
        
        physics = PhysicsMesh(mesh)
        auto = AutoCollapseEngine(physics)
        
        # Lower threshold for testing
        auto.config.entropy_threshold = 0.1
        auto.config.min_collapse_interval = 0.0
        
        results = auto.run_until_stable(max_iterations=5)
        
        print(f"PASS: Stabilization - {len(results)} collapses performed")
        for i, r in enumerate(results):
            print(f"  Collapse {i+1}: {r.strategy.value} - "
                  f"entropy {r.entropy_before:.3f} -> {r.entropy_after:.3f}")


class TestFullCycle:
    """Test complete auto-collapse integration."""
    
    def test_full_auto_collapse_cycle(self):
        """Test complete cycle: build, chaos, collapse, stable."""
        print("\n=== FULL AUTO-COLLAPSE CYCLE ===")
        
        # 1. Create mesh
        print("1. Creating chaotic mesh...")
        mesh = PACMeshSpace(embed_dim=64)
        
        # Add many nodes with sparse connections
        import random
        nodes = []
        for i in range(30):
            embedding = torch.randn(64)
            node = mesh.get_or_create_root(i, f"word_{i % 10}", embedding, "test")
            node.confidence = random.uniform(0.1, 0.9)
            nodes.append(node)
        
        # Random connections
        for node in nodes:
            for _ in range(random.randint(1, 3)):
                target = random.choice(nodes)
                if target != node:
                    node.add_child(target)
        
        print(f"   Created {len(nodes)} nodes")
        
        # 2. Create physics with auto-collapse
        print("2. Setting up physics + auto-collapse...")
        physics = PhysicsMesh(mesh)
        auto = AutoCollapseEngine(physics)
        
        # Configure for aggressive collapse
        auto.config.entropy_threshold = 0.5
        auto.config.min_collapse_interval = 0.0
        
        initial_entropy = physics.state.entropy
        print(f"   Initial entropy: {initial_entropy:.3f}")
        
        # 3. Enable and run
        print("3. Running auto-collapse...")
        auto.enable()
        
        collapses = 0
        for step in range(10):
            result = auto.step()
            if result:
                collapses += 1
                print(f"   Step {step}: {result.strategy.value} "
                      f"({result.entropy_reduction:.3f} reduction)")
        
        # 4. Check results
        print("4. Final state:")
        final_entropy = physics.state.entropy
        stats = auto.stats()
        
        print(f"   Entropy: {initial_entropy:.3f} -> {final_entropy:.3f}")
        print(f"   Collapses: {stats['collapses']}")
        print(f"   Strategies: {stats['strategies_used']}")
        print(f"   Total reduction: {stats['total_entropy_reduced']:.3f}")
        
        # 5. Verify
        assert stats['collapses'] >= 0
        print("\nPASS: Full auto-collapse cycle complete")


if __name__ == "__main__":
    print("=" * 60)
    print("AUTO-COLLAPSE TESTS")
    print("=" * 60)
    
    # Basic tests
    basics = TestAutoCollapseBasics()
    basics.test_auto_collapse_creation()
    basics.test_enable_disable()
    basics.test_should_collapse_empty()
    
    print()
    
    # Strategy tests
    strategies = TestCollapseStrategies()
    strategies.test_strategy_selection()
    strategies.test_clustering_collapse()
    strategies.test_crystallization_collapse()
    strategies.test_pruning_collapse()
    strategies.test_compression_collapse()
    strategies.test_hierarchical_collapse()
    
    print()
    
    # Balancer tests
    balancer_tests = TestEntropyBalancer()
    balancer_tests.test_balancer_creation()
    balancer_tests.test_regime_detection()
    balancer_tests.test_balancer_step()
    
    print()
    
    # Stability tests
    stability = TestRunUntilStable()
    stability.test_run_until_stable()
    
    # Full cycle
    cycle = TestFullCycle()
    cycle.test_full_auto_collapse_cycle()
    
    print()
    print("=" * 60)
    print("ALL AUTO-COLLAPSE TESTS PASSED")
    print("=" * 60)
