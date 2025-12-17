"""
Tiered Memory Cache for GAIA
============================

Hybrid memory system that combines GPU caching with PAC tree cold storage.

Architecture:
- Tier 1 (GPU Hot Cache): Full patterns, fast access, limited capacity
- Tier 2 (CPU Warm): Compressed patterns, medium access, larger capacity  
- Tier 3 (PAC Tree): Delta-compressed, slower access, unlimited capacity

Key insights:
1. GPU brute force is fastest for cached patterns
2. PAC tree provides efficient cold pattern storage
3. Transitions guide prefetching for cache warming
4. LRU with transition boost for smart eviction
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
from collections import OrderedDict
from dataclasses import dataclass
import time

from pac_tree_memory_v2 import PACTreeMemoryV2, PHI, XI


@dataclass
class CacheStats:
    """Statistics for cache performance"""
    hits: int = 0
    misses: int = 0
    prefetch_hits: int = 0
    evictions: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class TieredMemoryCache:
    """
    Tiered memory with GPU caching and PAC tree cold storage.
    
    Features:
    - Fast GPU cache for hot patterns
    - PAC tree for cold pattern compression
    - Transition-guided prefetching
    - Smart eviction with access frequency + transitions
    """
    
    def __init__(self, 
                 field_shape: Tuple[int, int, int] = (32, 32, 32),
                 device: str = 'cuda',
                 gpu_cache_size: int = 1000,  # Max patterns in GPU cache
                 prefetch_k: int = 10):  # How many to prefetch
        
        self.field_shape = field_shape
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.gpu_cache_size = gpu_cache_size
        self.prefetch_k = prefetch_k
        
        # Tier 1: GPU Hot Cache (OrderedDict for LRU)
        self._gpu_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._gpu_matrix: Optional[torch.Tensor] = None  # Stacked for batch ops
        self._gpu_matrix_valid = False
        
        # Tier 2: CPU Warm Cache
        self._cpu_cache: Dict[int, torch.Tensor] = {}
        
        # Tier 3: PAC Tree Cold Storage
        self._pac_tree = PACTreeMemoryV2(field_shape, device='cpu')
        
        # Pattern registry
        self._all_patterns: Set[int] = set()
        
        # Transition learning
        self._transitions: Dict[int, Dict[int, float]] = {}
        
        # Statistics
        self.stats = CacheStats()
        
    def store(self, pattern_id: int, field: torch.Tensor):
        """Store a pattern (goes to cold storage first, cached on access)"""
        if pattern_id in self._all_patterns:
            return
            
        self._all_patterns.add(pattern_id)
        
        # Store in PAC tree (cold storage)
        self._pac_tree.store(pattern_id, field.cpu())
        
        # Also cache in GPU if under limit
        if len(self._gpu_cache) < self.gpu_cache_size:
            self._add_to_gpu_cache(pattern_id, field.to(self.device))
            
    def _add_to_gpu_cache(self, pattern_id: int, field: torch.Tensor):
        """Add pattern to GPU cache with eviction if needed"""
        if pattern_id in self._gpu_cache:
            # Move to end (most recently used)
            self._gpu_cache.move_to_end(pattern_id)
            return
            
        # Evict if at capacity
        while len(self._gpu_cache) >= self.gpu_cache_size:
            evicted_id, _ = self._gpu_cache.popitem(last=False)
            self.stats.evictions += 1
            
        # Add to cache
        self._gpu_cache[pattern_id] = field.to(self.device)
        self._gpu_matrix_valid = False
        
    def _rebuild_gpu_matrix(self):
        """Rebuild stacked GPU matrix for batch operations"""
        if not self._gpu_cache:
            self._gpu_matrix = None
            return
            
        patterns = list(self._gpu_cache.values())
        self._gpu_matrix = torch.stack(patterns)
        self._gpu_matrix_valid = True
        
    def _get_from_cold(self, pattern_id: int) -> Optional[torch.Tensor]:
        """Retrieve from PAC tree and promote to cache"""
        field = self._pac_tree.reconstruct_pattern(pattern_id)
        
        if field is not None:
            # Promote to GPU cache
            self._add_to_gpu_cache(pattern_id, field.to(self.device))
            return self._gpu_cache[pattern_id]
            
        return None
    
    def retrieve(self, query: torch.Tensor,
                context_ids: Optional[List[int]] = None,
                top_k: int = 100) -> List[Tuple[int, float]]:
        """
        Retrieve candidates using tiered search.
        
        1. Check GPU cache first (fast)
        2. Use transitions to prefetch likely patterns
        3. Fall back to PAC tree for cold patterns
        """
        query = query.to(self.device)
        results = []
        
        # Prefetch based on context transitions
        if context_ids:
            self._prefetch_from_transitions(context_ids)
            
        # Phase 1: GPU cache search (fast)
        if self._gpu_cache:
            if not self._gpu_matrix_valid:
                self._rebuild_gpu_matrix()
                
            if self._gpu_matrix is not None:
                cache_results = self._search_gpu_cache(query, top_k)
                results.extend(cache_results)
                self.stats.hits += len(cache_results)
                
        # Phase 2: PAC tree for additional candidates
        if len(results) < top_k:
            remaining = top_k - len(results)
            seen = {r[0] for r in results}
            
            cold_results = self._pac_tree.retrieve(
                query.cpu(), 
                context_ids=context_ids,
                top_k=remaining,
                exclude=seen
            )
            
            # Promote accessed patterns to cache
            for pattern_id, score in cold_results:
                self.stats.misses += 1
                self._get_from_cold(pattern_id)  # Promotes to cache
                results.append((pattern_id, score))
                
        # Sort and return
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _search_gpu_cache(self, query: torch.Tensor, top_k: int) -> List[Tuple[int, float]]:
        """Fast GPU cache search"""
        if self._gpu_matrix is None:
            return []
            
        # Batch similarity computation
        query_flat = query.flatten().unsqueeze(0)
        cache_flat = self._gpu_matrix.flatten(1)
        
        query_norm = F.normalize(query_flat, dim=1)
        cache_norm = F.normalize(cache_flat, dim=1)
        
        similarities = torch.mm(query_norm, cache_norm.T).squeeze(0)
        
        # Get top-k
        k = min(top_k, len(self._gpu_cache))
        top_vals, top_idxs = torch.topk(similarities, k)
        
        # Map indices to pattern IDs
        cache_ids = list(self._gpu_cache.keys())
        results = [
            (cache_ids[idx.item()], val.item())
            for idx, val in zip(top_idxs, top_vals)
        ]
        
        # Update LRU for accessed patterns
        for pid, _ in results:
            self._gpu_cache.move_to_end(pid)
            
        return results
    
    def _prefetch_from_transitions(self, context_ids: List[int]):
        """Prefetch likely next patterns based on transitions"""
        candidates = {}
        
        for pid in context_ids:
            if pid in self._transitions:
                for target_pid, strength in self._transitions[pid].items():
                    if target_pid not in self._gpu_cache:
                        candidates[target_pid] = candidates.get(target_pid, 0) + strength
                        
        # Sort by transition strength
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        
        # Prefetch top-k
        for target_pid, _ in sorted_candidates[:self.prefetch_k]:
            if target_pid not in self._gpu_cache:
                field = self._pac_tree.reconstruct_pattern(target_pid)
                if field is not None:
                    self._add_to_gpu_cache(target_pid, field.to(self.device))
                    self.stats.prefetch_hits += 1
    
    def learn_transition(self, from_pid: int, to_pid: int, strength: float = 0.1):
        """Learn transition for prefetching"""
        if from_pid not in self._transitions:
            self._transitions[from_pid] = {}
            
        self._transitions[from_pid][to_pid] = \
            self._transitions[from_pid].get(to_pid, 0) + strength
            
        # Also store in PAC tree
        self._pac_tree.learn_transition(from_pid, to_pid, strength)
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            'total_patterns': len(self._all_patterns),
            'gpu_cached': len(self._gpu_cache),
            'gpu_cache_size': self.gpu_cache_size,
            'cache_utilization': len(self._gpu_cache) / self.gpu_cache_size,
            'hit_rate': self.stats.hit_rate,
            'hits': self.stats.hits,
            'misses': self.stats.misses,
            'prefetch_hits': self.stats.prefetch_hits,
            'evictions': self.stats.evictions,
            'pac_tree_stats': self._pac_tree.get_memory_stats()
        }


def test_tiered_cache():
    """Quick test of tiered cache"""
    print("=== Testing Tiered Memory Cache ===\n")
    
    cache = TieredMemoryCache(
        field_shape=(32, 32, 32),
        gpu_cache_size=100,
        prefetch_k=5
    )
    
    # Store 500 patterns
    print("Storing 500 patterns...")
    for i in range(500):
        cache.store(i, torch.randn(32, 32, 32))
        
    # Learn some transitions
    print("Learning transitions...")
    for i in range(400):
        cache.learn_transition(i, i + 1, 1.0)
        
    # Retrieve with context
    print("\nQuerying with context...")
    for i in range(50):
        query = torch.randn(32, 32, 32)
        context = [max(0, i - 1)]
        results = cache.retrieve(query, context_ids=context, top_k=10)
        
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Total patterns: {stats['total_patterns']}")
    print(f"  GPU cached: {stats['gpu_cached']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Prefetch hits: {stats['prefetch_hits']}")
    print(f"  Evictions: {stats['evictions']}")
    
    return stats


if __name__ == '__main__':
    test_tiered_cache()
