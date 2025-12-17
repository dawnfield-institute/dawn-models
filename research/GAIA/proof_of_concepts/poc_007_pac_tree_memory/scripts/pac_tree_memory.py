"""
PAC Tree Memory Implementation
==============================

Hierarchical memory using PAC (Persistent Arithmetic Conservation) principles.
Replaces O(n) brute-force search with O(log n) tree navigation.

Key insights from Euclidean Distance Validation experiments:
1. Store deltas between levels (massive compression)
2. Navigate by resonance (no brute force)
3. Transitions guide traversal (learned paths)
4. ξ-modulation creates natural clustering
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field as dataclass_field
from collections import defaultdict
import heapq
import numpy as np

# Dawn Field constants
PHI = 1.618033988749895
XI = 0.0618
PHI_XI = PHI * XI  # 1.710 - branching threshold
LAMBDA_STAR = 0.9816
BRANCHING_FACTOR = 8  # Optimal from depth-width tradeoff experiments


@dataclass
class PACNode:
    """Node in the PAC tree"""
    id: int
    level: int
    parent_id: Optional[int] = None
    children_ids: List[int] = dataclass_field(default_factory=list)
    
    # Storage: leaf nodes have full pattern_field, internal nodes have delta
    pattern_field: Optional[torch.Tensor] = None  # Full field (leaf only)
    delta: Optional[torch.Tensor] = None  # Delta from parent (internal nodes)
    centroid: Optional[torch.Tensor] = None  # Cluster centroid (for navigation)
    
    # Pattern mapping (for leaf nodes)
    pattern_id: Optional[int] = None
    
    # Transitions from this node
    transitions: Dict[int, float] = dataclass_field(default_factory=dict)
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0
    
    @property
    def is_root(self) -> bool:
        return self.parent_id is None


class PACTreeMemory:
    """
    Hierarchical memory using PAC tree structure.
    
    Key features:
    - Delta compression: internal nodes store delta from parent
    - Resonance navigation: find patterns via tree traversal
    - Transition guidance: learned paths improve retrieval
    - O(log n) retrieval instead of O(n)
    """
    
    def __init__(self, field_shape: Tuple[int, int, int] = (32, 32, 32),
                 device: str = 'cuda',
                 branching_factor: int = BRANCHING_FACTOR):
        self.field_shape = field_shape
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.branching_factor = branching_factor
        
        # Tree structure
        self.nodes: Dict[int, PACNode] = {}
        self.next_node_id = 0
        
        # Create root node
        root = self._create_node(level=0, parent_id=None)
        root.delta = torch.zeros(field_shape, device=self.device)
        root.centroid = torch.zeros(field_shape, device=self.device)
        self.root_id = root.id
        
        # Pattern to node mapping
        self.pattern_to_node: Dict[int, int] = {}
        
        # Transition index for O(1) lookup
        self._transition_index: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            'stores': 0,
            'retrievals': 0,
            'nodes_visited': 0,
            'cache_hits': 0
        }
        
    def _create_node(self, level: int, parent_id: Optional[int]) -> PACNode:
        """Create a new node"""
        node_id = self.next_node_id
        self.next_node_id += 1
        
        node = PACNode(id=node_id, level=level, parent_id=parent_id)
        self.nodes[node_id] = node
        
        if parent_id is not None:
            self.nodes[parent_id].children_ids.append(node_id)
            
        return node
    
    def store(self, pattern_id: int, field: torch.Tensor) -> int:
        """
        Store a pattern in the tree.
        
        Returns the node ID where pattern was stored.
        """
        if pattern_id in self.pattern_to_node:
            return self.pattern_to_node[pattern_id]
            
        field = field.to(self.device)
        self.stats['stores'] += 1
        
        # Navigate tree to find best insertion point
        current_id = self.root_id
        path = [current_id]
        
        while True:
            current = self.nodes[current_id]
            
            # If no children, insert here
            if not current.children_ids:
                break
                
            # Find best child by resonance
            best_child_id = None
            best_resonance = -float('inf')
            
            for child_id in current.children_ids:
                child = self.nodes[child_id]
                
                # Compute resonance with child centroid
                if child.centroid is not None:
                    resonance = self._compute_resonance(field, child.centroid)
                elif child.pattern_field is not None:
                    resonance = self._compute_resonance(field, child.pattern_field)
                else:
                    resonance = 0.0
                    
                if resonance > best_resonance:
                    best_resonance = resonance
                    best_child_id = child_id
                    
            # Decide whether to descend or create new branch
            if best_resonance < PHI_XI and len(current.children_ids) < self.branching_factor:
                # Low resonance, create new sibling
                break
            elif best_child_id is None:
                break
            else:
                # Descend into best child
                current_id = best_child_id
                path.append(current_id)
                
        # Create leaf node for this pattern
        leaf = self._create_node(level=len(path), parent_id=current_id)
        leaf.pattern_field = field.clone()
        leaf.pattern_id = pattern_id
        
        # Map pattern to node
        self.pattern_to_node[pattern_id] = leaf.id
        
        # Update centroids along path
        self._update_centroids(path + [leaf.id])
        
        # Check if reorganization needed
        if len(self.nodes[current_id].children_ids) > self.branching_factor:
            self._reorganize_node(current_id)
            
        return leaf.id
    
    def _compute_resonance(self, field1: torch.Tensor, field2: torch.Tensor) -> float:
        """Compute resonance between two fields"""
        # Flatten for cosine similarity
        f1 = field1.flatten()
        f2 = field2.flatten()
        
        # Normalize
        f1_norm = F.normalize(f1.unsqueeze(0), dim=1).squeeze()
        f2_norm = F.normalize(f2.unsqueeze(0), dim=1).squeeze()
        
        # Base cosine similarity
        base_sim = torch.dot(f1_norm, f2_norm)
        
        # ξ-modulation: correlation of contrasts
        c1 = (f1 - f1.mean()).abs()
        c2 = (f2 - f2.mean()).abs()
        c1_norm = F.normalize(c1.unsqueeze(0), dim=1).squeeze()
        c2_norm = F.normalize(c2.unsqueeze(0), dim=1).squeeze()
        xi_factor = torch.dot(c1_norm, c2_norm)
        
        return (base_sim + XI * xi_factor).item()
    
    def _update_centroids(self, path: List[int]):
        """Update centroids along path from leaf to root"""
        for node_id in reversed(path):
            node = self.nodes[node_id]
            
            if node.is_leaf:
                node.centroid = node.pattern_field.clone() if node.pattern_field is not None else None
            elif node.children_ids:
                # Compute centroid from children
                child_centroids = []
                for child_id in node.children_ids:
                    child = self.nodes[child_id]
                    if child.centroid is not None:
                        child_centroids.append(child.centroid)
                    elif child.pattern_field is not None:
                        child_centroids.append(child.pattern_field)
                        
                if child_centroids:
                    node.centroid = torch.stack(child_centroids).mean(dim=0)
                    
                    # Compute delta from parent
                    if node.parent_id is not None:
                        parent = self.nodes[node.parent_id]
                        if parent.centroid is not None:
                            node.delta = node.centroid - parent.centroid
    
    def _reorganize_node(self, node_id: int):
        """Reorganize node that has too many children"""
        node = self.nodes[node_id]
        if len(node.children_ids) <= self.branching_factor:
            return
            
        # Collect child centroids/fields
        child_data = []
        for child_id in node.children_ids:
            child = self.nodes[child_id]
            if child.centroid is not None:
                child_data.append((child_id, child.centroid))
            elif child.pattern_field is not None:
                child_data.append((child_id, child.pattern_field))
                
        if len(child_data) <= self.branching_factor:
            return
            
        # Cluster children
        clusters = self._cluster_by_resonance(child_data, self.branching_factor)
        
        # Create intermediate nodes for clusters
        node.children_ids = []
        
        for cluster in clusters:
            if len(cluster) == 1:
                # Single child, keep as direct child
                child_id = cluster[0][0]
                self.nodes[child_id].parent_id = node_id
                node.children_ids.append(child_id)
            else:
                # Multiple children, create intermediate node
                intermediate = self._create_node(level=node.level + 1, parent_id=node_id)
                
                for child_id, _ in cluster:
                    self.nodes[child_id].parent_id = intermediate.id
                    self.nodes[child_id].level = intermediate.level + 1
                    intermediate.children_ids.append(child_id)
                    
                self._update_centroids([intermediate.id])
    
    def _cluster_by_resonance(self, items: List[Tuple[int, torch.Tensor]], 
                              n_clusters: int) -> List[List[Tuple[int, torch.Tensor]]]:
        """Simple clustering by resonance"""
        if len(items) <= n_clusters:
            return [[item] for item in items]
            
        # Initialize clusters with spaced items
        clusters = [[] for _ in range(n_clusters)]
        indices = np.linspace(0, len(items)-1, n_clusters, dtype=int)
        
        for i, idx in enumerate(indices):
            clusters[i].append(items[idx])
            
        # Assign remaining items
        assigned = set(indices)
        for i, (item_id, field) in enumerate(items):
            if i in assigned:
                continue
                
            # Find best cluster
            best_cluster = 0
            best_resonance = -float('inf')
            
            for j, cluster in enumerate(clusters):
                if cluster:
                    _, centroid = cluster[0]
                    resonance = self._compute_resonance(field, centroid)
                    if resonance > best_resonance:
                        best_resonance = resonance
                        best_cluster = j
                        
            clusters[best_cluster].append((item_id, field))
            
        return [c for c in clusters if c]
    
    def retrieve(self, query: torch.Tensor, 
                context_ids: Optional[List[int]] = None,
                top_k: int = 100,
                exclude: Optional[Set[int]] = None) -> List[Tuple[int, float]]:
        """
        Retrieve candidate patterns using tree navigation.
        
        Uses transitions first (if context provided), then navigates tree.
        """
        self.stats['retrievals'] += 1
        exclude = exclude or set()
        query = query.to(self.device)
        
        candidates = []
        seen_patterns = set()
        
        # Phase 1: Transition-guided candidates
        if context_ids:
            transition_candidates = self._get_transition_candidates(context_ids)
            for pattern_id, strength in transition_candidates:
                if pattern_id not in exclude and pattern_id not in seen_patterns:
                    if pattern_id in self.pattern_to_node:
                        node_id = self.pattern_to_node[pattern_id]
                        node = self.nodes[node_id]
                        if node.pattern_field is not None:
                            resonance = self._compute_resonance(query, node.pattern_field)
                            # Boost by transition strength
                            score = resonance + XI * strength
                            candidates.append((pattern_id, score))
                            seen_patterns.add(pattern_id)
                            
        # Phase 2: Tree navigation for additional candidates
        remaining = top_k - len(candidates)
        if remaining > 0:
            tree_candidates = self._navigate_tree(query, remaining * 2, seen_patterns | exclude)
            candidates.extend(tree_candidates)
            
        # Sort and return top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def _get_transition_candidates(self, context_ids: List[int]) -> List[Tuple[int, float]]:
        """Get candidates based on transitions from context"""
        scores = defaultdict(float)
        
        for pattern_id in context_ids:
            if pattern_id in self.pattern_to_node:
                node_id = self.pattern_to_node[pattern_id]
                
                # Get transitions from this pattern
                for target_pattern_id, strength in self._transition_index.get(pattern_id, []):
                    scores[target_pattern_id] += strength
                    
        # Sort by strength
        candidates = [(pid, score) for pid, score in scores.items()]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def _navigate_tree(self, query: torch.Tensor, max_results: int,
                      exclude: Set[int]) -> List[Tuple[int, float]]:
        """Navigate tree to find resonant patterns"""
        results = []
        
        # Priority queue: (-resonance, node_id)
        queue = [(0.0, self.root_id)]
        visited = set()
        
        while queue and len(results) < max_results:
            neg_resonance, node_id = heapq.heappop(queue)
            
            if node_id in visited:
                continue
            visited.add(node_id)
            self.stats['nodes_visited'] += 1
            
            node = self.nodes[node_id]
            
            # If leaf with pattern, add to results
            if node.is_leaf and node.pattern_id is not None:
                if node.pattern_id not in exclude:
                    results.append((node.pattern_id, -neg_resonance))
                continue
                
            # Explore children
            for child_id in node.children_ids:
                if child_id in visited:
                    continue
                    
                child = self.nodes[child_id]
                
                # Compute resonance for prioritization
                if child.centroid is not None:
                    resonance = self._compute_resonance(query, child.centroid)
                elif child.pattern_field is not None:
                    resonance = self._compute_resonance(query, child.pattern_field)
                else:
                    resonance = 0.0
                    
                heapq.heappush(queue, (-resonance, child_id))
                
        return results
    
    def learn_transition(self, from_pattern_id: int, to_pattern_id: int, 
                        strength: float = 0.1):
        """Learn a transition between patterns"""
        # Update transition index
        transitions = self._transition_index[from_pattern_id]
        
        # Check if transition already exists
        for i, (target, s) in enumerate(transitions):
            if target == to_pattern_id:
                transitions[i] = (target, s + strength)
                return
                
        # Add new transition
        transitions.append((to_pattern_id, strength))
        
        # Also store at node level if nodes exist
        if from_pattern_id in self.pattern_to_node:
            node_id = self.pattern_to_node[from_pattern_id]
            node = self.nodes[node_id]
            if to_pattern_id in node.transitions:
                node.transitions[to_pattern_id] += strength
            else:
                node.transitions[to_pattern_id] = strength
    
    def get_transitions_from(self, pattern_id: int) -> List[Tuple[int, float]]:
        """Get all transitions from a pattern - O(1) lookup"""
        return self._transition_index.get(pattern_id, [])
    
    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics"""
        n_nodes = len(self.nodes)
        n_leaves = sum(1 for n in self.nodes.values() if n.is_leaf)
        n_internal = n_nodes - n_leaves
        
        # Memory calculation
        field_size = np.prod(self.field_shape) * 4  # float32
        leaf_memory = n_leaves * field_size
        internal_memory = n_internal * field_size  # deltas/centroids
        total_memory = leaf_memory + internal_memory
        
        # Flat storage comparison
        n_patterns = len(self.pattern_to_node)
        flat_memory = n_patterns * field_size
        
        return {
            'total_nodes': n_nodes,
            'leaf_nodes': n_leaves,
            'internal_nodes': n_internal,
            'patterns_stored': n_patterns,
            'memory_bytes': total_memory,
            'memory_mb': total_memory / (1024 ** 2),
            'flat_memory_mb': flat_memory / (1024 ** 2),
            'compression_ratio': flat_memory / max(total_memory, 1),
            'max_depth': max((n.level for n in self.nodes.values()), default=0),
            'avg_children': np.mean([len(n.children_ids) for n in self.nodes.values() 
                                    if not n.is_leaf]) if n_internal > 0 else 0,
            'stats': self.stats.copy()
        }
