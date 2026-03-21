"""
PAC-Lazy Core Primitives
========================

Core Node and PACSystem primitives for the PAC-Lazy Transformer.
Based on validated test.py substrate, extended for GAIA integration.

Laws (Non-Negotiable):
1. PAC: No expansion without consuming potential
2. SEC: Structure symbolic until pressure demands refinement
3. Locality: Nodes only interact through explicit neighbors
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import deque
import math

# Dawn Field Constants
PHI = 1.618033988749895
XI = 0.0618
PHI_XI = 1.710  # Crystallization threshold (φ × ξ empirically validated)
LAMBDA_STAR = 0.9816


@dataclass
class PACNode:
    """Minimal node primitive for PAC-Lazy computation.
    
    Invariants:
      - Stores deltas (Δ) only; no absolute values are canonical
      - Holds potential budget used to gate refinement
      - Defines locality via explicit neighbor edges
      - Defines structure via parent/child references
    """
    nid: str
    delta: torch.Tensor = None  # Local residual (no absolute values)
    potential: float = 0.0      # PAC budget held
    
    # Causal adjacency
    neighbors: List[str] = field(default_factory=list)
    
    # Structural hierarchy
    children: List[str] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)
    
    # SEC state
    expanded: bool = False
    active: bool = False
    
    # Optional: token mapping
    token_id: Optional[int] = None
    
    def __repr__(self) -> str:
        delta_str = f"Δ={self.delta.norm().item():.4f}" if self.delta is not None else "Δ=None"
        return (
            f"PACNode({self.nid}, {delta_str}, "
            f"pot={self.potential:.2f}, "
            f"active={self.active}, expanded={self.expanded})"
        )


class PACLazySystem:
    """PAC+SEC substrate for lazy tensor computation.
    
    This extends the validated PACSystem with:
    - Tensor deltas instead of scalar deltas
    - GPU acceleration
    - Integration with GAIA field encoding
    """
    
    def __init__(self, 
                 field_shape: Tuple[int, ...] = (384,),
                 total_potential: float = 100.0,
                 device: str = 'cuda'):
        
        self.field_shape = field_shape
        self.total_potential = total_potential
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.nodes: Dict[str, PACNode] = {}
        self.active_nodes: Set[str] = set()
        
        # Track potential conservation
        self.allocated_potential = 0.0
        
        # Statistics
        self.stats = {
            'expansions': 0,
            'collapses': 0,
            'fractures': 0,
            'propagations': 0
        }
    
    # ---- Node Management ----
    
    def add_node(self, nid: str, token_id: Optional[int] = None) -> PACNode:
        """Add a new node to the system."""
        node = PACNode(nid=nid, token_id=token_id)
        node.delta = torch.zeros(self.field_shape, device=self.device)
        self.nodes[nid] = node
        return node
    
    def get_node(self, nid: str) -> Optional[PACNode]:
        """Get node by ID."""
        return self.nodes.get(nid)
    
    # ---- Causal Links ----
    
    def link_neighbors(self, from_nid: str, to_nid: str) -> None:
        """Create directed causal adjacency (from -> to)."""
        self.nodes[from_nid].neighbors.append(to_nid)
    
    def link_bidirectional(self, nid_a: str, nid_b: str) -> None:
        """Create bidirectional causal link."""
        self.link_neighbors(nid_a, nid_b)
        self.link_neighbors(nid_b, nid_a)
    
    # ---- Structural Links ----
    
    def link_parent_child(self, parent_nid: str, child_nid: str) -> None:
        """Create structural refinement link."""
        self.nodes[parent_nid].children.append(child_nid)
        self.nodes[child_nid].parents.append(parent_nid)
    
    # ---- PAC Operations ----
    
    def inject_delta(self, nid: str, delta: torch.Tensor, 
                    potential_cost: float = 1.0) -> bool:
        """Inject a delta into a node, consuming potential budget.
        
        Returns True if injection succeeded (budget available).
        """
        if self.allocated_potential + potential_cost > self.total_potential:
            return False  # Budget exhausted
            
        node = self.nodes[nid]
        node.delta = node.delta + delta.to(self.device)
        node.potential += potential_cost
        node.active = True
        
        self.active_nodes.add(nid)
        self.allocated_potential += potential_cost
        
        return True
    
    def propagate_local(self, decay: float = 0.5) -> int:
        """Local causal propagation only.
        
        Nodes only influence neighbors through explicit links.
        No global update.
        
        Returns number of nodes activated.
        """
        self.stats['propagations'] += 1
        
        q = deque(self.active_nodes)
        visited = set(q)
        activated = 0
        
        while q:
            nid = q.popleft()
            node = self.nodes[nid]
            
            for nb_nid in node.neighbors:
                if nb_nid not in visited:
                    nb = self.nodes[nb_nid]
                    
                    # Transfer scaled delta
                    transferred_delta = node.delta * decay
                    transferred_potential = node.potential * decay
                    
                    if transferred_potential > 0.01:  # Threshold for propagation
                        nb.delta = nb.delta + transferred_delta
                        nb.potential += transferred_potential
                        nb.active = True
                        
                        self.active_nodes.add(nb_nid)
                        visited.add(nb_nid)
                        q.append(nb_nid)
                        activated += 1
        
        return activated
    
    # ---- SEC Operations ----
    
    def sec_expand(self, nid: str, 
                   threshold: float = PHI_XI,
                   child_factory: Optional[callable] = None) -> bool:
        """Expand node's children when potential crosses threshold.
        
        SEC = Symbolic Entropy Collapse
        - Before threshold: structure remains symbolic
        - After threshold: children receive potential
        
        Args:
            nid: Node to potentially expand
            threshold: Potential threshold for expansion
            child_factory: Optional function to create children if none exist
            
        Returns True if expansion occurred.
        """
        node = self.nodes[nid]
        
        if node.expanded:
            return False
            
        if node.potential < threshold:
            return False
            
        # Create children if factory provided and no children exist
        if not node.children and child_factory:
            child_ids = child_factory(nid)
            for child_id in child_ids:
                if child_id not in self.nodes:
                    self.add_node(child_id)
                self.link_parent_child(nid, child_id)
        
        if not node.children:
            return False
            
        # Consume potential for expansion
        expansion_cost = threshold
        node.potential -= expansion_cost
        self.allocated_potential -= expansion_cost
        
        # Distribute to children
        share = expansion_cost / len(node.children)
        for child_id in node.children:
            child = self.nodes[child_id]
            child.potential += share
            child.active = True
            self.active_nodes.add(child_id)
            self.allocated_potential += share
        
        node.expanded = True
        self.stats['expansions'] += 1
        
        return True
    
    def sec_collapse(self, nid: str, refund_ratio: float = 0.2) -> bool:
        """Collapse expanded structure back to symbolic.
        
        Returns refunded potential to parent.
        """
        node = self.nodes[nid]
        
        if not node.expanded:
            return False
            
        # Collect potential from children
        collected = 0.0
        for child_id in node.children:
            child = self.nodes[child_id]
            collected += child.potential
            child.potential = 0.0
            child.active = False
            child.delta = torch.zeros(self.field_shape, device=self.device)
            self.active_nodes.discard(child_id)
            self.allocated_potential -= collected
        
        # Refund portion to parent
        refund = collected * refund_ratio
        node.potential += refund
        self.allocated_potential += refund
        
        node.expanded = False
        self.stats['collapses'] += 1
        
        return True
    
    # ---- Structural Mutation ----
    
    def fracture(self, parent_nid: str, child_nid: str,
                stress_threshold: float = PHI_XI) -> bool:
        """Detach child from parent if stress exceeds threshold.
        
        This enables structural mutation for continuous learning.
        """
        parent = self.nodes.get(parent_nid)
        child = self.nodes.get(child_nid)
        
        if not parent or not child:
            return False
            
        if child_nid not in parent.children:
            return False
            
        # Check stress (delta magnitude as stress proxy)
        stress = child.delta.norm().item() if child.delta is not None else 0
        
        if stress < stress_threshold:
            return False
            
        # Perform fracture
        parent.children.remove(child_nid)
        child.parents.remove(parent_nid)
        
        self.stats['fractures'] += 1
        
        return True
    
    def merge(self, node_a: str, node_b: str, 
              merged_id: Optional[str] = None) -> Optional[str]:
        """Merge two nodes into one.
        
        Combines deltas and potentials, inherits all relationships.
        """
        a = self.nodes.get(node_a)
        b = self.nodes.get(node_b)
        
        if not a or not b:
            return None
            
        # Create merged node
        merged_id = merged_id or f"{node_a}+{node_b}"
        merged = self.add_node(merged_id)
        
        # Combine deltas
        merged.delta = a.delta + b.delta
        merged.potential = a.potential + b.potential
        merged.active = a.active or b.active
        
        # Inherit neighbors (union)
        merged.neighbors = list(set(a.neighbors + b.neighbors) - {node_a, node_b})
        
        # Update neighbor references
        for nb_id in merged.neighbors:
            nb = self.nodes[nb_id]
            if node_a in nb.neighbors:
                nb.neighbors.remove(node_a)
                if merged_id not in nb.neighbors:
                    nb.neighbors.append(merged_id)
            if node_b in nb.neighbors:
                nb.neighbors.remove(node_b)
                if merged_id not in nb.neighbors:
                    nb.neighbors.append(merged_id)
        
        # Inherit children (union)
        merged.children = list(set(a.children + b.children))
        for child_id in merged.children:
            child = self.nodes[child_id]
            if node_a in child.parents:
                child.parents.remove(node_a)
            if node_b in child.parents:
                child.parents.remove(node_b)
            if merged_id not in child.parents:
                child.parents.append(merged_id)
        
        # Inherit parents (union)
        merged.parents = list(set(a.parents + b.parents))
        for parent_id in merged.parents:
            parent = self.nodes[parent_id]
            if node_a in parent.children:
                parent.children.remove(node_a)
            if node_b in parent.children:
                parent.children.remove(node_b)
            if merged_id not in parent.children:
                parent.children.append(merged_id)
        
        # Remove original nodes
        del self.nodes[node_a]
        del self.nodes[node_b]
        self.active_nodes.discard(node_a)
        self.active_nodes.discard(node_b)
        
        if merged.active:
            self.active_nodes.add(merged_id)
        
        return merged_id
    
    # ---- Observation / Projection ----
    
    def observe(self, nid: str) -> Optional[torch.Tensor]:
        """Observe a node's current value by composing deltas from root.
        
        This is the only way to get an "absolute" value.
        The value is temporary and not stored.
        """
        node = self.nodes.get(nid)
        if not node:
            return None
            
        # Compose deltas from all ancestors
        value = node.delta.clone()
        
        # Walk up to all parents and accumulate
        visited = {nid}
        queue = deque(node.parents)
        
        while queue:
            parent_id = queue.popleft()
            if parent_id in visited:
                continue
            visited.add(parent_id)
            
            parent = self.nodes[parent_id]
            if parent.delta is not None:
                value = value + parent.delta
            
            queue.extend(parent.parents)
        
        return value
    
    def observe_frontier(self) -> Dict[str, torch.Tensor]:
        """Observe all active nodes."""
        return {nid: self.observe(nid) for nid in self.active_nodes}
    
    # ---- Statistics ----
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        return {
            'total_nodes': len(self.nodes),
            'active_nodes': len(self.active_nodes),
            'total_potential': self.total_potential,
            'allocated_potential': self.allocated_potential,
            'available_potential': self.total_potential - self.allocated_potential,
            'utilization': self.allocated_potential / self.total_potential,
            **self.stats
        }


# ---- Quick Validation ----

def test_pac_lazy_core():
    """Quick validation of core primitives."""
    print("=== PAC-Lazy Core Validation ===\n")
    
    system = PACLazySystem(field_shape=(384,), total_potential=10.0)
    
    # Create nodes
    for nid in ['A', 'B', 'C', 'D']:
        system.add_node(nid)
    
    # Create causal chain: A -> B -> C
    system.link_neighbors('A', 'B')
    system.link_neighbors('B', 'C')
    
    # D is isolated (no causal link)
    
    # Inject at A
    delta_a = torch.randn(384)
    system.inject_delta('A', delta_a, potential_cost=2.0)
    
    # Propagate
    activated = system.propagate_local(decay=0.5)
    print(f"Activated {activated} nodes via propagation")
    
    # Check locality
    print(f"\nNode states:")
    for nid in ['A', 'B', 'C', 'D']:
        node = system.get_node(nid)
        print(f"  {node}")
    
    d_active = system.get_node('D').active
    print(f"\nD isolated (should be False): {not d_active}")
    
    # Test SEC expansion
    print("\n--- SEC Expansion Test ---")
    
    # Create parent with children
    system.add_node('parent')
    system.add_node('child1')
    system.add_node('child2')
    system.link_parent_child('parent', 'child1')
    system.link_parent_child('parent', 'child2')
    
    # Inject below threshold
    system.inject_delta('parent', torch.randn(384), potential_cost=1.0)
    expanded = system.sec_expand('parent', threshold=PHI_XI)
    print(f"Expansion at pot=1.0 (should be False): {expanded}")
    
    # Inject above threshold
    system.inject_delta('parent', torch.randn(384), potential_cost=1.0)
    expanded = system.sec_expand('parent', threshold=PHI_XI)
    print(f"Expansion at pot=2.0 (should be True): {expanded}")
    
    # Check children activated
    c1 = system.get_node('child1')
    c2 = system.get_node('child2')
    print(f"Children activated: child1={c1.active}, child2={c2.active}")
    
    # Test collapse
    collapsed = system.sec_collapse('parent', refund_ratio=0.5)
    print(f"\nCollapse (should be True): {collapsed}")
    print(f"Children after collapse: child1.active={system.get_node('child1').active}")
    
    # Stats
    print(f"\nSystem stats: {system.get_stats()}")
    
    print("\n✅ PAC-Lazy Core validation complete")


if __name__ == '__main__':
    test_pac_lazy_core()
