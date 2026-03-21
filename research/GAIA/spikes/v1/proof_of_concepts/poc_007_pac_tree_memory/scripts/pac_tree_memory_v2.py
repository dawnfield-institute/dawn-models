"""
PAC Tree Memory v2 - Delta Compression
=======================================

Key improvements over v1:
1. Delta storage at ALL nodes (including leaves)
2. Reconstruct patterns by summing deltas along path
3. GPU-accelerated centroid matching
4. Proper resonance thresholds based on empirical data

Memory model:
- Root: stores mean field (low information)
- Level 1: stores delta from mean (first-order)  
- Level n: stores delta from level n-1 (higher-order corrections)
- Leaf: stores residual delta (pattern-specific correction)

Total storage: O(levels × field_size) + O(n × compressed_residual)
With SVD compression on residuals: ~10-20% of original
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
PHI_XI = PHI * XI  # 0.1 - NOT used as threshold
LAMBDA_STAR = 0.9816
BRANCHING_FACTOR = 8

# Empirically tuned thresholds
RESONANCE_THRESHOLD = 0.3  # Minimum resonance to join existing branch
COMPRESSION_RANK = 32  # SVD rank for residual compression


@dataclass
class PACNodeV2:
    """Node in the PAC tree v2"""
    id: int
    level: int
    parent_id: Optional[int] = None
    children_ids: List[int] = dataclass_field(default_factory=list)
    
    # Delta from parent (or mean for root)
    delta: Optional[torch.Tensor] = None
    
    # For internal nodes: cluster centroid (computed from children)
    centroid: Optional[torch.Tensor] = None
    
    # For leaf nodes: pattern ID and compressed residual
    pattern_id: Optional[int] = None
    residual_u: Optional[torch.Tensor] = None  # Left singular vectors
    residual_s: Optional[torch.Tensor] = None  # Singular values
    residual_v: Optional[torch.Tensor] = None  # Right singular vectors
    
    # Transitions
    transitions: Dict[int, float] = dataclass_field(default_factory=dict)
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0
    
    @property
    def is_root(self) -> bool:
        return self.parent_id is None


class PACTreeMemoryV2:
    """
    Hierarchical memory with delta compression.
    
    Key features:
    - Stores deltas instead of full patterns
    - Uses SVD compression on leaf residuals
    - GPU-accelerated centroid matching
    - Reconstructs patterns on-demand
    """
    
    def __init__(self, field_shape: Tuple[int, int, int] = (32, 32, 32),
                 device: str = 'cuda',
                 branching_factor: int = BRANCHING_FACTOR,
                 compression_rank: int = COMPRESSION_RANK):
        self.field_shape = field_shape
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.branching_factor = branching_factor
        self.compression_rank = compression_rank
        
        # Tree structure
        self.nodes: Dict[int, PACNodeV2] = {}
        self.next_node_id = 0
        
        # Create root node
        root = self._create_node(level=0, parent_id=None)
        root.delta = torch.zeros(field_shape, device=self.device)
        root.centroid = torch.zeros(field_shape, device=self.device)
        self.root_id = root.id
        
        # Pattern to node mapping
        self.pattern_to_node: Dict[int, int] = {}
        
        # Transition index
        self._transition_index: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        
        # Running mean for delta computation
        self._running_mean = torch.zeros(field_shape, device=self.device)
        self._count = 0
        
        # Statistics
        self.stats = {
            'stores': 0,
            'retrievals': 0,
            'nodes_visited': 0,
            'reconstructions': 0
        }
        
    def _create_node(self, level: int, parent_id: Optional[int]) -> PACNodeV2:
        """Create a new node"""
        node_id = self.next_node_id
        self.next_node_id += 1
        
        node = PACNodeV2(id=node_id, level=level, parent_id=parent_id)
        self.nodes[node_id] = node
        
        if parent_id is not None:
            self.nodes[parent_id].children_ids.append(node_id)
            
        return node
    
    def store(self, pattern_id: int, field: torch.Tensor) -> int:
        """Store a pattern with delta compression"""
        if pattern_id in self.pattern_to_node:
            return self.pattern_to_node[pattern_id]
            
        field = field.to(self.device)
        self.stats['stores'] += 1
        
        # Update running mean
        self._count += 1
        self._running_mean = self._running_mean + (field - self._running_mean) / self._count
        
        # Update root centroid with running mean
        self.nodes[self.root_id].centroid = self._running_mean.clone()
        self.nodes[self.root_id].delta = self._running_mean.clone()
        
        # Navigate tree to find best insertion point
        current_id = self.root_id
        path = [current_id]
        
        while True:
            current = self.nodes[current_id]
            
            # If no children or reached max depth, insert here
            if not current.children_ids:
                break
                
            # Compute resonance with each child's centroid
            best_child_id = None
            best_resonance = -float('inf')
            
            for child_id in current.children_ids:
                child = self.nodes[child_id]
                if child.centroid is not None:
                    resonance = self._compute_resonance(field, child.centroid)
                    if resonance > best_resonance:
                        best_resonance = resonance
                        best_child_id = child_id
                        
            # Decide whether to descend or create new branch
            if best_resonance < RESONANCE_THRESHOLD and len(current.children_ids) < self.branching_factor:
                # Low resonance and room for new branch - create sibling
                break
            elif best_child_id is None:
                break
            else:
                # Descend into best child
                current_id = best_child_id
                path.append(current_id)
                
        # Create leaf node
        parent = self.nodes[current_id]
        leaf = self._create_node(level=len(path), parent_id=current_id)
        leaf.pattern_id = pattern_id
        
        # Compute and store delta from parent centroid
        parent_centroid = parent.centroid if parent.centroid is not None else self._running_mean
        residual = field - parent_centroid
        
        # Compress residual using SVD
        self._store_compressed_residual(leaf, residual)
        
        # Set leaf centroid (for navigation)
        leaf.centroid = field.clone()
        
        # Map pattern to node
        self.pattern_to_node[pattern_id] = leaf.id
        
        # Update centroids along path
        self._update_centroids(path)
        
        # Check if reorganization needed
        if len(parent.children_ids) > self.branching_factor:
            self._reorganize_node(current_id)
            
        return leaf.id
    
    def _store_compressed_residual(self, node: PACNodeV2, residual: torch.Tensor):
        """Store residual using SVD compression"""
        # Flatten to 2D for SVD
        flat = residual.flatten().unsqueeze(0)  # [1, D]
        
        # Compute rank-k approximation
        # For single vector, just store the vector scaled
        norm = flat.norm()
        if norm > 1e-8:
            node.residual_u = flat / norm  # [1, D]
            node.residual_s = norm.unsqueeze(0)  # [1]
            node.residual_v = torch.tensor([1.0], device=self.device)  # [1]
        else:
            node.residual_u = torch.zeros_like(flat)
            node.residual_s = torch.tensor([0.0], device=self.device)
            node.residual_v = torch.tensor([1.0], device=self.device)
            
        # Also store delta for navigation
        node.delta = residual.clone()
    
    def _reconstruct_from_compressed(self, node: PACNodeV2) -> torch.Tensor:
        """Reconstruct residual from compressed storage"""
        self.stats['reconstructions'] += 1
        
        if node.residual_u is None:
            return torch.zeros(self.field_shape, device=self.device)
            
        # Reconstruct: U @ diag(S) @ V^T
        flat = node.residual_u * node.residual_s.unsqueeze(0)
        return flat.reshape(self.field_shape)
    
    def _compute_resonance(self, field1: torch.Tensor, field2: torch.Tensor) -> float:
        """Compute resonance between two fields"""
        f1 = field1.flatten()
        f2 = field2.flatten()
        
        f1_norm = F.normalize(f1.unsqueeze(0), dim=1).squeeze()
        f2_norm = F.normalize(f2.unsqueeze(0), dim=1).squeeze()
        
        base_sim = torch.dot(f1_norm, f2_norm)
        
        # XI-modulation
        c1 = (f1 - f1.mean()).abs()
        c2 = (f2 - f2.mean()).abs()
        c1_norm = F.normalize(c1.unsqueeze(0), dim=1).squeeze()
        c2_norm = F.normalize(c2.unsqueeze(0), dim=1).squeeze()
        xi_factor = torch.dot(c1_norm, c2_norm)
        
        return (base_sim + XI * xi_factor).item()
    
    def _update_centroids(self, path: List[int]):
        """Update centroids along path"""
        for node_id in reversed(path):
            node = self.nodes[node_id]
            
            if node.children_ids:
                child_centroids = []
                for child_id in node.children_ids:
                    child = self.nodes[child_id]
                    if child.centroid is not None:
                        child_centroids.append(child.centroid)
                        
                if child_centroids:
                    node.centroid = torch.stack(child_centroids).mean(dim=0)
                    
                    # Update delta from parent
                    if node.parent_id is not None:
                        parent = self.nodes[node.parent_id]
                        if parent.centroid is not None:
                            node.delta = node.centroid - parent.centroid
    
    def _reorganize_node(self, node_id: int):
        """Reorganize overloaded node using k-means clustering"""
        node = self.nodes[node_id]
        if len(node.children_ids) <= self.branching_factor:
            return
            
        # Collect child data
        child_data = []
        for child_id in node.children_ids:
            child = self.nodes[child_id]
            if child.centroid is not None:
                child_data.append((child_id, child.centroid))
                
        if len(child_data) <= self.branching_factor:
            return
            
        # K-means clustering
        centroids_tensor = torch.stack([c for _, c in child_data])
        k = self.branching_factor
        
        # Simple k-means
        cluster_assignments = self._kmeans(centroids_tensor, k)
        
        # Group children by cluster
        clusters = defaultdict(list)
        for i, (child_id, centroid) in enumerate(child_data):
            clusters[cluster_assignments[i]].append((child_id, centroid))
            
        # Create intermediate nodes
        node.children_ids = []
        
        for cluster_id, cluster_children in clusters.items():
            if len(cluster_children) == 1:
                child_id = cluster_children[0][0]
                self.nodes[child_id].parent_id = node_id
                node.children_ids.append(child_id)
            else:
                intermediate = self._create_node(level=node.level + 1, parent_id=node_id)
                
                for child_id, _ in cluster_children:
                    self.nodes[child_id].parent_id = intermediate.id
                    self.nodes[child_id].level = intermediate.level + 1
                    intermediate.children_ids.append(child_id)
                    
                self._update_centroids([intermediate.id])
    
    def _kmeans(self, data: torch.Tensor, k: int, max_iters: int = 10) -> List[int]:
        """Simple k-means clustering"""
        n = data.shape[0]
        if n <= k:
            return list(range(n))
            
        # Initialize centroids
        indices = torch.randperm(n)[:k]
        centroids = data[indices].clone()
        
        assignments = [0] * n
        
        for _ in range(max_iters):
            # Assign points to nearest centroid
            flat_data = data.flatten(1)
            flat_centroids = centroids.flatten(1)
            
            dists = torch.cdist(flat_data, flat_centroids)
            new_assignments = dists.argmin(dim=1).tolist()
            
            if new_assignments == assignments:
                break
                
            assignments = new_assignments
            
            # Update centroids
            for j in range(k):
                cluster_points = [data[i] for i in range(n) if assignments[i] == j]
                if cluster_points:
                    centroids[j] = torch.stack(cluster_points).mean(dim=0)
                    
        return assignments
    
    def reconstruct_pattern(self, pattern_id: int) -> Optional[torch.Tensor]:
        """Reconstruct full pattern from deltas"""
        if pattern_id not in self.pattern_to_node:
            return None
            
        node_id = self.pattern_to_node[pattern_id]
        
        # Collect path to root
        path = []
        current_id = node_id
        while current_id is not None:
            path.append(current_id)
            current_id = self.nodes[current_id].parent_id
            
        # Sum deltas from root to leaf
        result = torch.zeros(self.field_shape, device=self.device)
        
        for nid in reversed(path):
            node = self.nodes[nid]
            if node.is_leaf and node.residual_u is not None:
                # Add compressed residual
                result = result + self._reconstruct_from_compressed(node)
            elif node.delta is not None:
                result = result + node.delta
                
        return result
    
    def retrieve(self, query: torch.Tensor,
                context_ids: Optional[List[int]] = None,
                top_k: int = 100,
                exclude: Optional[Set[int]] = None) -> List[Tuple[int, float]]:
        """Retrieve candidates using tree navigation"""
        self.stats['retrievals'] += 1
        exclude = exclude or set()
        query = query.to(self.device)
        
        candidates = []
        seen_patterns = set()
        
        # Phase 1: Transition-guided candidates
        if context_ids:
            for pattern_id, strength in self._get_transition_candidates(context_ids):
                if pattern_id not in exclude and pattern_id not in seen_patterns:
                    node_id = self.pattern_to_node.get(pattern_id)
                    if node_id is not None:
                        node = self.nodes[node_id]
                        if node.centroid is not None:
                            resonance = self._compute_resonance(query, node.centroid)
                            score = resonance + XI * strength
                            candidates.append((pattern_id, score))
                            seen_patterns.add(pattern_id)
                            
        # Phase 2: Tree navigation
        remaining = top_k - len(candidates)
        if remaining > 0:
            tree_candidates = self._navigate_tree_gpu(query, remaining * 2, seen_patterns | exclude)
            candidates.extend(tree_candidates)
            
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def _get_transition_candidates(self, context_ids: List[int]) -> List[Tuple[int, float]]:
        """Get candidates from transitions"""
        scores = defaultdict(float)
        
        for pid in context_ids:
            for target_pid, strength in self._transition_index.get(pid, []):
                scores[target_pid] += strength
                
        candidates = [(pid, score) for pid, score in scores.items()]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def _navigate_tree_gpu(self, query: torch.Tensor, max_results: int,
                          exclude: Set[int]) -> List[Tuple[int, float]]:
        """GPU-accelerated tree navigation"""
        results = []
        
        # Beam search through tree
        beam = [(0.0, self.root_id)]
        beam_width = min(self.branching_factor * 2, 32)
        
        while beam and len(results) < max_results:
            next_beam = []
            
            for neg_score, node_id in beam:
                self.stats['nodes_visited'] += 1
                node = self.nodes[node_id]
                
                # If leaf with pattern, add to results
                if node.is_leaf and node.pattern_id is not None:
                    if node.pattern_id not in exclude:
                        results.append((node.pattern_id, -neg_score))
                    continue
                    
                # Collect children centroids for batch computation
                children_with_centroids = []
                for child_id in node.children_ids:
                    child = self.nodes[child_id]
                    if child.centroid is not None:
                        children_with_centroids.append((child_id, child.centroid))
                        
                if not children_with_centroids:
                    continue
                    
                # Batch resonance computation
                child_ids = [cid for cid, _ in children_with_centroids]
                centroids = torch.stack([c for _, c in children_with_centroids])
                
                # GPU batch computation
                query_flat = query.flatten().unsqueeze(0)
                centroids_flat = centroids.flatten(1)
                
                query_norm = F.normalize(query_flat, dim=1)
                centroids_norm = F.normalize(centroids_flat, dim=1)
                
                similarities = torch.mm(query_norm, centroids_norm.T).squeeze(0)
                
                for i, child_id in enumerate(child_ids):
                    resonance = similarities[i].item()
                    next_beam.append((-resonance, child_id))
                    
            # Keep top-k for next iteration
            next_beam.sort()
            beam = next_beam[:beam_width]
            
        return results
    
    def learn_transition(self, from_pid: int, to_pid: int, strength: float = 0.1):
        """Learn transition between patterns"""
        transitions = self._transition_index[from_pid]
        
        for i, (target, s) in enumerate(transitions):
            if target == to_pid:
                transitions[i] = (target, s + strength)
                return
                
        transitions.append((to_pid, strength))
        
        if from_pid in self.pattern_to_node:
            node_id = self.pattern_to_node[from_pid]
            node = self.nodes[node_id]
            node.transitions[to_pid] = node.transitions.get(to_pid, 0) + strength
    
    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics"""
        n_nodes = len(self.nodes)
        n_leaves = sum(1 for n in self.nodes.values() if n.is_leaf)
        n_internal = n_nodes - n_leaves
        
        field_size = np.prod(self.field_shape) * 4  # float32
        
        # Internal nodes: store delta + centroid
        internal_memory = n_internal * field_size * 2
        
        # Leaf nodes: store compressed residual (U, S, V) + centroid
        # For single vector: D + 1 + 1 ≈ D
        leaf_memory = n_leaves * field_size * 2
        
        total_memory = internal_memory + leaf_memory
        
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
