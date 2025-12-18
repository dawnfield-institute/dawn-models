"""
GAIA v4.0 - Transformer Organs

Specialized processing units that attach to the GAIA Cortex.
Each organ handles a specific aspect of cognition.
"""

import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from fracton.core import PACSystem
from fracton.field import evolve, compute_resonance
from fracton.physics import PHI, XI, PHI_XI, LAMBDA_STAR

from .cortex import TransformerOrgan


class LanguageOrgan(TransformerOrgan):
    """
    Specialized for linguistic processing.
    
    Uses transition memory for token prediction.
    Tracks token-to-token transitions and weights.
    """
    
    def __init__(self, substrate: PACSystem = None):
        super().__init__("language", substrate)
        self.transitions: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._last_token: Optional[int] = None
    
    def process(self, 
                field: torch.Tensor,
                resonant: List[Tuple[int, float]] = None) -> torch.Tensor:
        """
        Process field through language organ.
        
        1. Find similar patterns in substrate
        2. Apply transition-based prediction
        3. Blend with original field
        """
        self._activation_count += 1
        
        if not resonant or self.substrate is None:
            return field
        
        # Get top resonant pattern
        top_id, top_score = resonant[0]
        
        # Check for transitions from this pattern
        if top_id in self.transitions:
            # Get weighted average of next patterns
            next_patterns = []
            weights = []
            
            for next_id, weight in self.transitions[top_id].items():
                try:
                    pattern = self.substrate.reconstruct(next_id)
                    next_patterns.append(pattern)
                    weights.append(weight)
                except:
                    continue
            
            if next_patterns:
                # Weighted blend
                total_weight = sum(weights)
                blended = torch.zeros_like(field)
                for pattern, weight in zip(next_patterns, weights):
                    blended = blended + (weight / total_weight) * pattern
                
                # Mix with original field
                mix_ratio = min(top_score, PHI_XI)  # Cap at phase threshold
                return (1 - mix_ratio) * field + mix_ratio * blended
        
        return field
    
    def learn_transition(self, from_id: int, to_id: int, weight: float = 1.0) -> None:
        """Learn a transition between patterns."""
        self.transitions[from_id][to_id] += weight
    
    def should_activate(self, field: torch.Tensor) -> bool:
        """Activate for all fields (language is universal)."""
        return True


class ReasoningOrgan(TransformerOrgan):
    """
    Specialized for logical reasoning.
    
    Uses field evolution for inference chains.
    Longer evolution = deeper reasoning.
    """
    
    def __init__(self, 
                 substrate: PACSystem = None,
                 reasoning_depth: int = 10):
        super().__init__("reasoning", substrate)
        self.reasoning_depth = reasoning_depth
    
    def process(self, 
                field: torch.Tensor,
                resonant: List[Tuple[int, float]] = None) -> torch.Tensor:
        """
        Process field through reasoning organ.
        
        Uses extended field evolution to simulate
        logical inference chains.
        """
        self._activation_count += 1
        
        # Apply extended evolution for deeper reasoning
        evolved = evolve(field, steps=self.reasoning_depth)
        
        # If we have resonant patterns, incorporate them
        if resonant and self.substrate:
            # Find patterns that resonate with evolved field
            for node_id, score in resonant:
                if score > PHI_XI:  # Only strong resonances
                    try:
                        pattern = self.substrate.reconstruct(node_id)
                        # Blend proportionally to resonance
                        evolved = evolved + score * XI * pattern
                    except:
                        continue
        
        # Normalize
        norm = torch.norm(evolved)
        if norm > 1e-10:
            evolved = evolved / norm
        
        return evolved
    
    def should_activate(self, field: torch.Tensor) -> bool:
        """
        Activate for high-energy fields (complex inputs need reasoning).
        """
        energy = torch.sum(field ** 2).item()
        return energy > PHI_XI


class MemoryOrgan(TransformerOrgan):
    """
    Specialized for long-term storage and retrieval.
    
    Manages pattern crystallization and recall.
    """
    
    def __init__(self, 
                 substrate: PACSystem = None,
                 crystallization_threshold: float = 0.8):
        super().__init__("memory", substrate)
        self.crystallization_threshold = crystallization_threshold
        self.crystallized: Dict[int, float] = {}  # node_id -> importance
    
    def process(self, 
                field: torch.Tensor,
                resonant: List[Tuple[int, float]] = None) -> torch.Tensor:
        """
        Process field through memory organ.
        
        1. Check if field should crystallize
        2. Retrieve similar crystallized patterns
        3. Blend memories with input
        """
        self._activation_count += 1
        
        if self.substrate is None:
            return field
        
        # Check for crystallization (strong, novel patterns)
        should_crystallize = self._check_crystallization(field, resonant)
        if should_crystallize:
            node_id = self.substrate.inject(field, label="crystallized")
            self.crystallized[node_id] = torch.sum(field ** 2).item()
        
        # Retrieve and blend memories
        if resonant:
            memories = []
            weights = []
            
            for node_id, score in resonant:
                if node_id in self.crystallized:
                    try:
                        memory = self.substrate.reconstruct(node_id)
                        memories.append(memory)
                        weights.append(score * self.crystallized[node_id])
                    except:
                        continue
            
            if memories:
                total_weight = sum(weights) + 1.0  # +1 for original field
                blended = field.clone()
                for memory, weight in zip(memories, weights):
                    blended = blended + (weight / total_weight) * memory
                return blended / (1 + len(memories) * XI)  # Normalize
        
        return field
    
    def _check_crystallization(self, 
                               field: torch.Tensor,
                               resonant: List[Tuple[int, float]]) -> bool:
        """Check if field should be crystallized."""
        # Field must have significant energy
        energy = torch.sum(field ** 2).item()
        if energy < PHI_XI:
            return False
        
        # Must be novel (not too similar to existing patterns)
        if resonant:
            max_similarity = max(score for _, score in resonant)
            if max_similarity > self.crystallization_threshold:
                return False  # Too similar to existing
        
        return True
    
    def should_activate(self, field: torch.Tensor) -> bool:
        """Always activate memory organ."""
        return True
    
    def forget(self, threshold: float = None) -> int:
        """
        Forget low-importance memories.
        
        Returns number of forgotten patterns.
        """
        if threshold is None:
            threshold = XI
        
        to_forget = [
            node_id for node_id, importance in self.crystallized.items()
            if importance < threshold
        ]
        
        for node_id in to_forget:
            del self.crystallized[node_id]
            # Note: substrate cleanup happens via GC
        
        return len(to_forget)


class AttentionOrgan(TransformerOrgan):
    """
    Specialized for attention-based processing.
    
    Implements field-based attention without explicit matrices.
    """
    
    def __init__(self, 
                 substrate: PACSystem = None,
                 attention_heads: int = 4):
        super().__init__("attention", substrate)
        self.attention_heads = attention_heads
    
    def process(self, 
                field: torch.Tensor,
                resonant: List[Tuple[int, float]] = None) -> torch.Tensor:
        """
        Apply field-based attention.
        
        Uses resonance as attention weights.
        """
        self._activation_count += 1
        
        if not resonant or self.substrate is None:
            return field
        
        # Multi-head attention via field splitting
        dim = field.shape[0]
        head_dim = dim // self.attention_heads
        
        attended = torch.zeros_like(field)
        
        for h in range(self.attention_heads):
            start = h * head_dim
            end = start + head_dim
            
            head_field = field[start:end]
            head_attended = head_field.clone()
            
            # Attend to resonant patterns
            for node_id, score in resonant[:3]:  # Top 3 per head
                try:
                    pattern = self.substrate.reconstruct(node_id)
                    head_pattern = pattern[start:end]
                    
                    # Attention weight from resonance
                    weight = score * PHI  # Amplify strong resonances
                    head_attended = head_attended + weight * head_pattern
                except:
                    continue
            
            # Normalize head
            norm = torch.norm(head_attended)
            if norm > 1e-10:
                head_attended = head_attended / norm
            
            attended[start:end] = head_attended
        
        return attended
    
    def should_activate(self, field: torch.Tensor) -> bool:
        """Activate when we have context to attend to."""
        return self.substrate is not None and len(self.substrate) > 0
