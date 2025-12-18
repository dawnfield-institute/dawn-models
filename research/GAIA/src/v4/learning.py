"""
GAIA v4.0 - Continuous Learning Module

Always-on learning in the style of CIMM (Continuous Inference Memory Model).
Learning happens as a side effect of processing, not as a separate phase.
"""

import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time

from fracton.core import PACSystem
from fracton.physics import PHI, XI, PHI_XI, LAMBDA_STAR


@dataclass
class LearningEvent:
    """Record of a learning event."""
    timestamp: float
    pattern_id: int
    context_ids: List[int]
    learning_rate: float
    crystallized: bool


class ContinuousLearner:
    """
    CIMM-style continuous learning on Fracton substrate.
    
    Key insight: Learning is delta injection, not weight updates.
    Every pattern processed leaves a trace in the substrate.
    
    Learning modes:
    - Passive: Every processed pattern is stored
    - Active: Only high-importance patterns are crystallized
    - Consolidation: Background process strengthens connections
    """
    
    def __init__(self, 
                 substrate: PACSystem,
                 learning_rate: float = XI,
                 crystallization_threshold: float = PHI_XI):
        """
        Initialize continuous learner.
        
        Args:
            substrate: Fracton PACSystem for storage
            learning_rate: Rate of connection strengthening
            crystallization_threshold: Energy threshold for crystallization
        """
        self.substrate = substrate
        self.learning_rate = learning_rate
        self.crystallization_threshold = crystallization_threshold
        
        # Connection strengths (edge weights)
        self.connections: Dict[Tuple[int, int], float] = {}
        
        # Learning history
        self.history: List[LearningEvent] = []
        
        # Context window
        self.context_window: List[int] = []
        self.context_size: int = 10
    
    def learn(self, 
              pattern: torch.Tensor,
              importance: float = 1.0) -> int:
        """
        Learn from a pattern in context.
        
        1. Inject pattern as delta from context
        2. Strengthen transitions to context patterns
        3. Check for crystallization (phase transition)
        
        Args:
            pattern: Pattern tensor to learn
            importance: Importance weight (affects crystallization)
            
        Returns:
            Node ID of stored pattern
        """
        # Find context patterns in substrate
        context_ids = self.context_window[-self.context_size:]
        
        # Determine if this should crystallize
        energy = torch.sum(pattern ** 2).item()
        should_crystallize = (
            energy * importance > self.crystallization_threshold
        )
        
        # Inject pattern
        if context_ids:
            # Store as delta from most recent context
            parent_id = context_ids[-1]
            try:
                node_id = self.substrate.inject(
                    pattern, 
                    parent_id=parent_id,
                    label="learned"
                )
            except:
                # Parent not found, create root
                node_id = self.substrate.inject(pattern, label="learned")
        else:
            node_id = self.substrate.inject(pattern, label="learned")
        
        # Strengthen connections to context
        for ctx_id in context_ids:
            self._strengthen_connection(ctx_id, node_id, importance)
        
        # Record event
        self.history.append(LearningEvent(
            timestamp=time.time(),
            pattern_id=node_id,
            context_ids=context_ids.copy(),
            learning_rate=self.learning_rate,
            crystallized=should_crystallize
        ))
        
        # Update context window
        self.context_window.append(node_id)
        if len(self.context_window) > self.context_size * 2:
            self.context_window = self.context_window[-self.context_size:]
        
        return node_id
    
    def _strengthen_connection(self, 
                               from_id: int, 
                               to_id: int,
                               weight: float = 1.0) -> None:
        """Strengthen connection between two patterns."""
        key = (from_id, to_id)
        current = self.connections.get(key, 0.0)
        
        # Hebbian-style strengthening with decay
        new_weight = current * LAMBDA_STAR + weight * self.learning_rate
        self.connections[key] = new_weight
    
    def get_transitions(self, from_id: int) -> List[Tuple[int, float]]:
        """Get learned transitions from a pattern."""
        transitions = []
        for (f, t), weight in self.connections.items():
            if f == from_id:
                transitions.append((t, weight))
        
        # Sort by weight
        transitions.sort(key=lambda x: x[1], reverse=True)
        return transitions
    
    def predict_next(self, 
                     pattern: torch.Tensor,
                     top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Predict likely next patterns based on learned transitions.
        
        Args:
            pattern: Current pattern
            top_k: Number of predictions
            
        Returns:
            List of (node_id, probability) tuples
        """
        # Find similar patterns in substrate
        similar = self.substrate.find_resonant(pattern, top_k=10)
        
        if not similar:
            return []
        
        # Aggregate transitions from similar patterns
        candidates: Dict[int, float] = {}
        for node_id, similarity in similar:
            transitions = self.get_transitions(node_id)
            for next_id, weight in transitions:
                score = similarity * weight
                candidates[next_id] = candidates.get(next_id, 0.0) + score
        
        # Sort and return top k
        sorted_candidates = sorted(
            candidates.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        # Normalize to probabilities
        total = sum(score for _, score in sorted_candidates)
        if total > 0:
            return [(nid, score / total) for nid, score in sorted_candidates]
        
        return sorted_candidates
    
    def consolidate(self) -> int:
        """
        Background consolidation of learned patterns.
        
        Strengthens connections between frequently co-occurring
        patterns and weakens rarely-used connections.
        
        Returns:
            Number of connections modified
        """
        modified = 0
        
        # Decay weak connections
        to_remove = []
        for key, weight in self.connections.items():
            new_weight = weight * LAMBDA_STAR
            if new_weight < XI / 10:  # Below threshold
                to_remove.append(key)
            else:
                self.connections[key] = new_weight
                modified += 1
        
        for key in to_remove:
            del self.connections[key]
            modified += 1
        
        return modified
    
    def stats(self) -> Dict:
        """Get learning statistics."""
        return {
            "patterns_learned": len(self.history),
            "connections": len(self.connections),
            "crystallizations": sum(1 for e in self.history if e.crystallized),
            "context_size": len(self.context_window),
            "avg_learning_rate": sum(e.learning_rate for e in self.history) / max(1, len(self.history))
        }


class PatternCrystallizer:
    """
    Handles crystallization of patterns into long-term memory.
    
    Crystallized patterns are:
    - Marked as important
    - Protected from garbage collection
    - Used as anchors for new learning
    """
    
    def __init__(self, substrate: PACSystem):
        self.substrate = substrate
        self.crystallized: Dict[int, float] = {}  # node_id -> importance
    
    def crystallize(self, 
                    node_id: int,
                    importance: float = 1.0) -> bool:
        """
        Crystallize a pattern.
        
        Args:
            node_id: Pattern to crystallize
            importance: Initial importance score
            
        Returns:
            True if crystallized, False if already crystallized
        """
        if node_id in self.crystallized:
            # Already crystallized, boost importance
            self.crystallized[node_id] += importance * XI
            return False
        
        self.crystallized[node_id] = importance
        
        # Boost potential to prevent GC
        self.substrate.update_potential(node_id, PHI)
        
        return True
    
    def melt(self, node_id: int) -> bool:
        """
        Un-crystallize a pattern (allow it to be forgotten).
        
        Returns:
            True if melted, False if wasn't crystallized
        """
        if node_id not in self.crystallized:
            return False
        
        del self.crystallized[node_id]
        
        # Reduce potential
        self.substrate.update_potential(node_id, XI)
        
        return True
    
    def decay_importance(self) -> int:
        """
        Decay importance of all crystallized patterns.
        
        Returns:
            Number of patterns that melted
        """
        melted = 0
        to_melt = []
        
        for node_id, importance in self.crystallized.items():
            new_importance = importance * LAMBDA_STAR
            if new_importance < XI:
                to_melt.append(node_id)
            else:
                self.crystallized[node_id] = new_importance
        
        for node_id in to_melt:
            self.melt(node_id)
            melted += 1
        
        return melted
    
    def is_crystallized(self, node_id: int) -> bool:
        """Check if a pattern is crystallized."""
        return node_id in self.crystallized
    
    def get_anchors(self, top_k: int = 10) -> List[Tuple[int, float]]:
        """Get most important crystallized patterns."""
        sorted_patterns = sorted(
            self.crystallized.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_patterns[:top_k]
