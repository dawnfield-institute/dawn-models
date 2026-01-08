"""
Entropy-Triggered Auto-Collapse.

Automatically forms structure when entropy exceeds thresholds.
This is the "physics engine" that runs continuously, maintaining
the balance between chaos (entropy) and order (structure).

Key insight from Dawn Field Theory:
- High entropy = too much uncertainty = system unstable
- Collapse = phase transition where structure crystallizes
- Auto-collapse keeps system at "edge of chaos" (optimal learning)

Collapse strategies:
1. CLUSTERING: Group similar high-entropy nodes
2. CRYSTALLIZATION: Lock in confident patterns
3. PRUNING: Remove weak connections
4. COMPRESSION: Merge redundant paths
5. HIERARCHICAL: Create new summary nodes
"""

import torch
from typing import List, Dict, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import math
import time
from collections import defaultdict

from .pac_mesh import PACMeshSpace, MeshNode
from .physics_mesh import (
    PhysicsMesh, EntropyMonitor, CollapseEngine, CollapseEvent, CollapseType,
    FieldState
)
# Import validated constants (derived, not fitted)
from .validated_constants import XI, PHI, PHI_INV, LAMBDA_STAR, ENTROPY_OPTIMAL_LOW, ENTROPY_OPTIMAL_HIGH


class CollapseStrategy(Enum):
    """Strategies for automatic collapse."""
    CLUSTERING = "clustering"         # Group similar nodes
    CRYSTALLIZATION = "crystallization"  # Lock confident patterns
    PRUNING = "pruning"              # Remove weak connections
    COMPRESSION = "compression"       # Merge redundant paths
    HIERARCHICAL = "hierarchical"     # Create summary nodes


@dataclass
class CollapseResult:
    """Result of an auto-collapse operation."""
    strategy: CollapseStrategy
    nodes_affected: int
    entropy_before: float
    entropy_after: float
    time_taken: float
    details: Dict[str, any] = field(default_factory=dict)
    
    @property
    def entropy_reduction(self) -> float:
        return self.entropy_before - self.entropy_after


@dataclass
class AutoCollapseConfig:
    """Configuration for auto-collapse behavior."""
    # Thresholds
    entropy_threshold: float = 2.0        # Trigger collapse above this
    entropy_target: float = 1.0           # Target entropy after collapse
    variance_threshold: float = 0.5       # Also trigger on high variance
    
    # Cooldowns (prevent oscillation)
    min_collapse_interval: float = 0.5    # Seconds between collapses
    min_nodes_for_collapse: int = 5       # Need this many nodes
    
    # Strategy weights
    strategy_weights: Dict[CollapseStrategy, float] = field(default_factory=lambda: {
        CollapseStrategy.CLUSTERING: 0.3,
        CollapseStrategy.CRYSTALLIZATION: 0.25,
        CollapseStrategy.PRUNING: 0.2,
        CollapseStrategy.COMPRESSION: 0.15,
        CollapseStrategy.HIERARCHICAL: 0.1,
    })
    
    # Strategy-specific
    cluster_similarity_threshold: float = 0.7
    prune_weight_threshold: float = 0.1
    compression_merge_threshold: float = 0.9
    crystallization_confidence_threshold: float = 0.8


class AutoCollapseEngine:
    """
    Automatically triggers and manages collapse events.
    
    Monitors entropy and applies appropriate collapse strategies
    to maintain system at optimal operating point.
    
    Usage:
        physics = PhysicsMesh(mesh)
        auto = AutoCollapseEngine(physics)
        
        # Enable continuous monitoring
        auto.enable()
        
        # Or manual step
        result = auto.step()
        if result:
            print(f"Collapsed: {result.entropy_reduction:.2f} entropy reduced")
    """
    
    def __init__(self,
                 physics: PhysicsMesh,
                 config: AutoCollapseConfig = None):
        self.physics = physics
        self.mesh = physics.mesh
        self.config = config or AutoCollapseConfig()
        
        # State
        self.enabled = False
        self.last_collapse_time = 0.0
        self.collapse_history: List[CollapseResult] = []
        self.strategies_used: Dict[CollapseStrategy, int] = defaultdict(int)
        
        # Callbacks
        self.on_collapse_callbacks: List[Callable[[CollapseResult], None]] = []
        
    def enable(self) -> None:
        """Enable automatic collapse monitoring."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable automatic collapse monitoring."""
        self.enabled = False
    
    def should_collapse(self) -> Tuple[bool, str]:
        """Check if collapse should be triggered."""
        # Get current state
        state = self.physics.step()
        entropy = state.entropy
        variance = self.physics.entropy_monitor.get_variance()
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_collapse_time < self.config.min_collapse_interval:
            return False, "cooldown"
        
        # Check minimum nodes
        if len(self.mesh.nodes) < self.config.min_nodes_for_collapse:
            return False, "insufficient_nodes"
        
        # Check entropy threshold
        if entropy > self.config.entropy_threshold:
            return True, "entropy_spike"
        
        # Check variance threshold
        if variance > self.config.variance_threshold:
            return True, "entropy_variance"
        
        return False, "stable"
    
    def select_strategy(self) -> CollapseStrategy:
        """
        Select best collapse strategy based on current state.
        
        Uses weighted random selection, but adjusts based on
        what would be most effective.
        """
        weights = dict(self.config.strategy_weights)
        
        # Adjust based on state
        state = self.physics.state
        
        # If many nodes, prefer clustering
        if len(self.mesh.nodes) > 100:
            weights[CollapseStrategy.CLUSTERING] *= 1.5
            
        # If high attractor count, prefer crystallization
        if len(self.physics.attractors) > 10:
            weights[CollapseStrategy.CRYSTALLIZATION] *= 1.3
            
        # If many weak connections (estimate), prefer pruning
        weak_nodes = sum(1 for n in self.mesh.nodes.values() if n.confidence < 0.3)
        if weak_nodes > len(self.mesh.nodes) * 0.3:
            weights[CollapseStrategy.PRUNING] *= 1.4
        
        # Normalize and select
        total = sum(weights.values())
        probs = {s: w/total for s, w in weights.items()}
        
        import random
        r = random.random()
        cumsum = 0.0
        for strategy, prob in probs.items():
            cumsum += prob
            if r < cumsum:
                return strategy
        
        return CollapseStrategy.CLUSTERING  # Default
    
    def collapse_clustering(self) -> CollapseResult:
        """
        Collapse by clustering similar nodes.
        
        High-entropy regions often have many similar but disconnected nodes.
        Clustering links them, reducing path uncertainty.
        """
        start_time = time.time()
        entropy_before = self.physics.state.entropy
        
        affected = 0
        clusters_formed = 0
        
        # Find similar nodes
        nodes = list(self.mesh.nodes.values())
        threshold = self.config.cluster_similarity_threshold
        
        # Simple O(n^2) clustering - could optimize
        clustered = set()
        for i, node1 in enumerate(nodes):
            if node1.node_id in clustered:
                continue
                
            cluster = [node1]
            for node2 in nodes[i+1:]:
                if node2.node_id in clustered:
                    continue
                    
                sim = torch.cosine_similarity(
                    node1.embedding.unsqueeze(0),
                    node2.embedding.unsqueeze(0)
                ).item()
                
                if sim >= threshold:
                    cluster.append(node2)
                    clustered.add(node2.node_id)
            
            if len(cluster) > 1:
                # Link cluster members
                for j, n1 in enumerate(cluster):
                    for n2 in cluster[j+1:]:
                        n1.add_child(n2)
                        n2.add_child(n1)
                        affected += 2
                
                # Mark central node as attractor
                central = cluster[0]
                self.physics.attractors[central.node_id] = 0.5
                clusters_formed += 1
        
        # Update physics
        self.physics.step()
        entropy_after = self.physics.state.entropy
        
        return CollapseResult(
            strategy=CollapseStrategy.CLUSTERING,
            nodes_affected=affected,
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            time_taken=time.time() - start_time,
            details={'clusters_formed': clusters_formed}
        )
    
    def collapse_crystallization(self) -> CollapseResult:
        """
        Collapse by crystallizing confident patterns.
        
        Locks in high-confidence nodes, preventing future decay.
        """
        start_time = time.time()
        entropy_before = self.physics.state.entropy
        
        affected = 0
        threshold = self.config.crystallization_confidence_threshold
        
        for node in self.mesh.nodes.values():
            if node.confidence >= threshold:
                if node.node_id not in self.physics.collapse.crystallized:
                    # Mark as crystallized and set as attractor
                    self.physics.collapse.crystallized.add(node.node_id)
                    node.confidence = 1.0  # Full confidence
                    self.physics.attractors[node.node_id] = node.confidence
                    affected += 1
        
        self.physics.step()
        entropy_after = self.physics.state.entropy
        
        return CollapseResult(
            strategy=CollapseStrategy.CRYSTALLIZATION,
            nodes_affected=affected,
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            time_taken=time.time() - start_time,
            details={'crystallized': affected}
        )
    
    def collapse_pruning(self) -> CollapseResult:
        """
        Collapse by removing weak connections.
        
        Weak connections add entropy without contributing structure.
        """
        start_time = time.time()
        entropy_before = self.physics.state.entropy
        
        affected = 0
        threshold = self.config.prune_weight_threshold
        
        for node in self.mesh.nodes.values():
            # Find weak children to remove
            weak_children = []
            for child_id, (child, count) in node.children.items():
                # Compute weight as fraction of total
                total = sum(c for _, c in node.children.values())
                weight = count / total if total > 0 else 0
                
                if weight < threshold:
                    weak_children.append(child_id)
            
            # Remove weak connections
            for child_id in weak_children:
                del node.children[child_id]
                affected += 1
        
        # Also reduce importance of low-confidence nodes
        for node_id in list(self.physics.importance.keys()):
            if node_id in self.mesh.nodes:
                if self.mesh.nodes[node_id].confidence < 0.3:
                    self.physics.importance[node_id] *= 0.5
        
        self.physics.step()
        entropy_after = self.physics.state.entropy
        
        return CollapseResult(
            strategy=CollapseStrategy.PRUNING,
            nodes_affected=affected,
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            time_taken=time.time() - start_time,
            details={'connections_pruned': affected}
        )
    
    def collapse_compression(self) -> CollapseResult:
        """
        Collapse by merging highly similar nodes.
        
        Redundant paths are compressed into single paths.
        """
        start_time = time.time()
        entropy_before = self.physics.state.entropy
        
        affected = 0
        threshold = self.config.compression_merge_threshold
        
        # Find merge candidates
        nodes = list(self.mesh.nodes.values())
        merged = set()
        
        for i, node1 in enumerate(nodes):
            if node1.node_id in merged:
                continue
            
            for node2 in nodes[i+1:]:
                if node2.node_id in merged:
                    continue
                
                # Check if similar enough to merge
                sim = torch.cosine_similarity(
                    node1.embedding.unsqueeze(0),
                    node2.embedding.unsqueeze(0)
                ).item()
                
                if sim >= threshold and node1.token_str == node2.token_str:
                    # Merge node2 into node1
                    # Transfer children
                    for child_id, (child, count) in node2.children.items():
                        if child_id in node1.children:
                            existing, existing_count = node1.children[child_id]
                            node1.children[child_id] = (existing, existing_count + count)
                        else:
                            node1.children[child_id] = (child, count)
                    
                    # Transfer incoming paths
                    for path, count in node2.incoming_paths.items():
                        node1.incoming_paths[path] = node1.incoming_paths.get(path, 0) + count
                    
                    # Boost confidence
                    node1.confidence = min(1.0, node1.confidence + node2.confidence * 0.5)
                    
                    merged.add(node2.node_id)
                    affected += 1
        
        # Remove merged nodes
        for node_id in merged:
            if node_id in self.mesh.nodes:
                del self.mesh.nodes[node_id]
        
        self.physics.step()
        entropy_after = self.physics.state.entropy
        
        return CollapseResult(
            strategy=CollapseStrategy.COMPRESSION,
            nodes_affected=affected,
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            time_taken=time.time() - start_time,
            details={'nodes_merged': affected}
        )
    
    def collapse_hierarchical(self) -> CollapseResult:
        """
        Collapse by creating hierarchical summary nodes.
        
        Groups of related nodes get a parent summary node.
        """
        start_time = time.time()
        entropy_before = self.physics.state.entropy
        
        affected = 0
        summaries_created = 0
        
        # Group nodes by depth
        depth_groups: Dict[int, List[MeshNode]] = defaultdict(list)
        for node in self.mesh.nodes.values():
            depth_groups[node.depth].append(node)
        
        # Create summaries for large groups
        for depth, nodes in depth_groups.items():
            if len(nodes) < 5:
                continue
            
            # Compute average embedding for summary
            embeddings = torch.stack([n.embedding for n in nodes])
            summary_emb = embeddings.mean(dim=0)
            
            # Create summary node
            summary_id = f"summary_d{depth}_{summaries_created}"
            summary_token_id = hash(summary_id) % 1000000
            
            summary_node = self.mesh.get_or_create_root(
                summary_token_id,
                f"[SUMMARY_D{depth}]",
                summary_emb,
                "auto_collapse"
            )
            
            # Link children to summary
            for node in nodes[:10]:  # Limit connections
                summary_node.add_child(node)
            
            # Make summary an attractor
            self.physics.attractors[summary_node.node_id] = 0.7
            
            summaries_created += 1
            affected += min(10, len(nodes))
        
        self.physics.step()
        entropy_after = self.physics.state.entropy
        
        return CollapseResult(
            strategy=CollapseStrategy.HIERARCHICAL,
            nodes_affected=affected,
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            time_taken=time.time() - start_time,
            details={'summaries_created': summaries_created}
        )
    
    def collapse(self, strategy: Optional[CollapseStrategy] = None) -> CollapseResult:
        """
        Execute collapse with specified or auto-selected strategy.
        """
        if strategy is None:
            strategy = self.select_strategy()
        
        # Execute strategy
        if strategy == CollapseStrategy.CLUSTERING:
            result = self.collapse_clustering()
        elif strategy == CollapseStrategy.CRYSTALLIZATION:
            result = self.collapse_crystallization()
        elif strategy == CollapseStrategy.PRUNING:
            result = self.collapse_pruning()
        elif strategy == CollapseStrategy.COMPRESSION:
            result = self.collapse_compression()
        elif strategy == CollapseStrategy.HIERARCHICAL:
            result = self.collapse_hierarchical()
        else:
            result = self.collapse_clustering()  # Default
        
        # Record
        self.last_collapse_time = time.time()
        self.collapse_history.append(result)
        self.strategies_used[strategy] += 1
        
        # Callbacks
        for callback in self.on_collapse_callbacks:
            callback(result)
        
        return result
    
    def step(self) -> Optional[CollapseResult]:
        """
        Run one step of auto-collapse monitoring.
        
        Checks if collapse is needed and executes if so.
        """
        should, reason = self.should_collapse()
        
        if should:
            return self.collapse()
        
        return None
    
    def run_until_stable(self, max_iterations: int = 10) -> List[CollapseResult]:
        """
        Run collapses until entropy is stable.
        
        Useful for initial stabilization.
        """
        results = []
        
        for _ in range(max_iterations):
            result = self.step()
            if result is None:
                break
            results.append(result)
            
            # Check if we've reached target
            if result.entropy_after < self.config.entropy_target:
                break
        
        return results
    
    def stats(self) -> Dict:
        """Get auto-collapse statistics."""
        total_reduction = sum(r.entropy_reduction for r in self.collapse_history)
        
        return {
            'enabled': self.enabled,
            'collapses': len(self.collapse_history),
            'total_entropy_reduced': total_reduction,
            'strategies_used': dict(self.strategies_used),
            'current_entropy': self.physics.state.entropy,
            'last_collapse_ago': time.time() - self.last_collapse_time if self.last_collapse_time else None
        }


class EntropyBalancer:
    """
    High-level entropy management that combines all components.
    
    Maintains the system at the "edge of chaos" for optimal learning:
    - Too low entropy: System is rigid, can't learn new patterns
    - Too high entropy: System is chaotic, can't maintain patterns
    - Optimal: Dynamic balance where structure forms and adapts
    
    Constants imported from validated_constants.py (derived, not fitted).
    """
    
    def __init__(self, physics: PhysicsMesh):
        self.physics = physics
        self.auto_collapse = AutoCollapseEngine(physics)
        
        # Configure for edge-of-chaos using validated constants
        self.auto_collapse.config.entropy_threshold = ENTROPY_OPTIMAL_HIGH * 1.2
        self.auto_collapse.config.entropy_target = (ENTROPY_OPTIMAL_LOW + ENTROPY_OPTIMAL_HIGH) / 2
        
        # History for analysis
        self.entropy_trajectory: List[float] = []
    
    def step(self) -> Dict:
        """Run one step of entropy balancing."""
        # Record current state
        self.entropy_trajectory.append(self.physics.state.entropy)
        if len(self.entropy_trajectory) > 1000:
            self.entropy_trajectory = self.entropy_trajectory[-500:]
        
        # Run physics
        self.physics.step()
        
        # Check for auto-collapse
        collapse_result = self.auto_collapse.step()
        
        # Analyze state
        entropy = self.physics.state.entropy
        in_optimal_range = ENTROPY_OPTIMAL_LOW <= entropy <= ENTROPY_OPTIMAL_HIGH
        
        return {
            'entropy': entropy,
            'in_optimal_range': in_optimal_range,
            'collapse_triggered': collapse_result is not None,
            'collapse_result': collapse_result,
            'trajectory_length': len(self.entropy_trajectory)
        }
    
    def get_regime(self) -> str:
        """Get current entropy regime."""
        entropy = self.physics.state.entropy
        
        if entropy < ENTROPY_OPTIMAL_LOW * 0.5:
            return "frozen"  # Too rigid
        elif entropy < ENTROPY_OPTIMAL_LOW:
            return "ordered"  # Slightly rigid
        elif entropy <= ENTROPY_OPTIMAL_HIGH:
            return "optimal"  # Edge of chaos
        elif entropy <= ENTROPY_OPTIMAL_HIGH * 1.5:
            return "active"  # Slightly chaotic
        else:
            return "chaotic"  # Too disordered
