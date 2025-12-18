"""
PAC-Lazy Transformer
====================

A transformer architecture built on PAC-Lazy primitives.

Key innovations:
1. Tokens are nodes with deltas (not absolute embeddings)
2. Attention is causal propagation (only neighbors interact)
3. Context is PAC-bounded (potential limits active frontier)
4. Depth is SEC-adaptive (expand children when needed)
5. Learning is structural (fracture/merge mutations)

This is the "living transformer" that never stops learning.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import math

from pac_lazy_core import PACLazySystem, PACNode, PHI, XI, PHI_XI, LAMBDA_STAR


@dataclass
class PACTransformerConfig:
    """Configuration for PAC-Lazy Transformer."""
    embedding_dim: int = 384
    total_potential: float = 100.0
    propagation_decay: float = 0.5
    expansion_threshold: float = PHI_XI
    fracture_threshold: float = PHI_XI
    max_active_nodes: int = 1000
    device: str = 'cuda'


class PACLazyTransformer:
    """
    PAC-Lazy Transformer: Intelligence through conservation laws.
    
    Unlike traditional transformers:
    - No fixed context window (PAC-bounded)
    - No layer-by-layer depth (SEC-adaptive)
    - No separate training/inference (continuous mutation)
    - No global attention (causal locality)
    """
    
    def __init__(self, config: PACTransformerConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = config.embedding_dim
        
        # Core PAC system
        self.system = PACLazySystem(
            field_shape=(config.embedding_dim,),
            total_potential=config.total_potential,
            device=config.device
        )
        
        # Token to node mapping
        self.token_to_node: Dict[int, str] = {}
        self.next_token_nid = 0
        
        # Sequence tracking for causal links
        self.sequence_history: List[str] = []
        
        # Vocabulary embeddings (learned via mutations)
        self.vocab_deltas: Dict[int, torch.Tensor] = {}
        
    def _get_token_nid(self, token_id: int) -> str:
        """Get or create node ID for token."""
        if token_id not in self.token_to_node:
            nid = f"tok_{self.next_token_nid}"
            self.next_token_nid += 1
            self.token_to_node[token_id] = nid
            
            # Create node
            node = self.system.add_node(nid, token_id=token_id)
            
            # Initialize with vocab delta if exists
            if token_id in self.vocab_deltas:
                node.delta = self.vocab_deltas[token_id].clone()
                
        return self.token_to_node[token_id]
    
    def process_token(self, token_id: int, 
                     embedding: torch.Tensor,
                     learn: bool = True) -> torch.Tensor:
        """Process a token through the PAC-Lazy system.
        
        Args:
            token_id: Token identifier
            embedding: Initial embedding (used as delta source)
            learn: Whether to learn from this interaction
            
        Returns:
            Observed value at this token's node
        """
        nid = self._get_token_nid(token_id)
        node = self.system.get_node(nid)
        
        # Inject embedding as delta
        delta = embedding.to(self.device)
        potential_cost = delta.norm().item() * 0.01  # Scale to reasonable range
        
        # Check if we have potential budget
        if not self.system.inject_delta(nid, delta, potential_cost):
            # Budget exhausted - need to collapse some nodes
            self._collapse_least_active()
            self.system.inject_delta(nid, delta, potential_cost)
        
        # Create causal link from previous token
        if self.sequence_history:
            prev_nid = self.sequence_history[-1]
            self.system.link_neighbors(prev_nid, nid)
            
            # Bidirectional for attention-like behavior
            self.system.link_neighbors(nid, prev_nid)
        
        # Add to sequence
        self.sequence_history.append(nid)
        
        # Propagate causally
        self.system.propagate_local(self.config.propagation_decay)
        
        # Check for SEC expansion
        if node.potential >= self.config.expansion_threshold:
            # Create detail children for this token
            def child_factory(parent_nid):
                return [
                    f"{parent_nid}_detail_{i}" 
                    for i in range(4)  # 4 detail levels
                ]
            self.system.sec_expand(nid, self.config.expansion_threshold, child_factory)
        
        # Learning: update vocab delta
        if learn and token_id not in self.vocab_deltas:
            self.vocab_deltas[token_id] = delta.clone()
        elif learn:
            # Blend with existing
            self.vocab_deltas[token_id] = (
                LAMBDA_STAR * self.vocab_deltas[token_id] +
                (1 - LAMBDA_STAR) * delta
            )
        
        # Return observed value
        return self.system.observe(nid)
    
    def _collapse_least_active(self):
        """Collapse nodes with lowest potential to free budget."""
        if not self.system.active_nodes:
            return
            
        # Find nodes with expanded children
        expanded = [
            nid for nid in self.system.active_nodes
            if self.system.get_node(nid).expanded
        ]
        
        if not expanded:
            return
            
        # Sort by potential (lowest first)
        expanded.sort(key=lambda nid: self.system.get_node(nid).potential)
        
        # Collapse the lowest
        for nid in expanded[:max(1, len(expanded) // 4)]:
            self.system.sec_collapse(nid)
    
    def predict_next(self, top_k: int = 10) -> List[Tuple[int, float]]:
        """Predict next token based on current frontier.
        
        Uses causal propagation to find most resonant tokens.
        """
        if not self.sequence_history:
            return []
            
        # Get the current frontier (active nodes)
        frontier = self.system.observe_frontier()
        
        if not frontier:
            return []
            
        # Compose frontier into query
        query = torch.stack(list(frontier.values())).mean(dim=0)
        query = F.normalize(query.unsqueeze(0), dim=1).squeeze()
        
        # Score against vocab deltas
        candidates = []
        for token_id, delta in self.vocab_deltas.items():
            if delta is not None:
                delta_norm = F.normalize(delta.unsqueeze(0), dim=1).squeeze()
                score = torch.dot(query, delta_norm).item()
                candidates.append((token_id, score))
        
        # Sort and return top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def learn_transition(self, from_token: int, to_token: int):
        """Strengthen causal link between tokens."""
        from_nid = self._get_token_nid(from_token)
        to_nid = self._get_token_nid(to_token)
        
        # Already linked via process_token, but we can strengthen
        # by adjusting potentials
        from_node = self.system.get_node(from_nid)
        to_node = self.system.get_node(to_nid)
        
        if from_node and to_node:
            # Transfer some delta to strengthen relationship
            transfer = from_node.delta * 0.1
            to_node.delta = to_node.delta + transfer
    
    def check_fracture(self):
        """Check for structural mutations (fractures)."""
        fractures = []
        
        for nid in list(self.system.active_nodes):
            node = self.system.get_node(nid)
            if not node:
                continue
                
            for child_id in list(node.children):
                if self.system.fracture(nid, child_id, self.config.fracture_threshold):
                    fractures.append((nid, child_id))
        
        return fractures
    
    def reset_sequence(self):
        """Reset sequence tracking for new input."""
        self.sequence_history = []
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        system_stats = self.system.get_stats()
        return {
            **system_stats,
            'vocab_size': len(self.vocab_deltas),
            'sequence_length': len(self.sequence_history),
        }


def test_pac_lazy_transformer():
    """Test the PAC-Lazy Transformer."""
    print("=== PAC-Lazy Transformer Test ===\n")
    
    config = PACTransformerConfig(
        embedding_dim=384,
        total_potential=50.0,
        device='cuda'
    )
    
    model = PACLazyTransformer(config)
    
    # Simulate processing a sequence
    print("Processing sequence of 10 tokens...")
    
    for i in range(10):
        token_id = i % 5  # Vocabulary of 5 tokens
        embedding = torch.randn(384)
        
        output = model.process_token(token_id, embedding, learn=True)
        print(f"  Token {token_id}: output_norm={output.norm().item():.4f}")
    
    # Check predictions
    print("\nPredicting next token...")
    predictions = model.predict_next(top_k=5)
    print(f"  Top predictions: {predictions}")
    
    # Check for fractures
    fractures = model.check_fracture()
    print(f"\nFractures detected: {len(fractures)}")
    
    # Stats
    stats = model.get_stats()
    print(f"\nSystem stats:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Active nodes: {stats['active_nodes']}")
    print(f"  Potential utilization: {stats['utilization']:.2%}")
    print(f"  Vocab size: {stats['vocab_size']}")
    print(f"  Expansions: {stats['expansions']}")
    print(f"  Collapses: {stats['collapses']}")
    
    print("\n✅ PAC-Lazy Transformer test complete")


if __name__ == '__main__':
    test_pac_lazy_transformer()
