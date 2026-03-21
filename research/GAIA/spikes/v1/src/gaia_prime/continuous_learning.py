"""
CIMM-Style Continuous Learning for PhysicsMesh.

Key principles from legacy CIMM:
1. Learning happens DURING inference, not separately
2. Every pattern leaves a trace (delta injection)
3. Connections strengthen through use (Hebbian)
4. Consolidation runs in background
5. Crystallization creates long-term memory

This module adds always-on learning to PhysicsMesh.
"""

import torch
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
import time
import math

from .pac_mesh import PACMeshSpace, MeshNode
from .physics_mesh import (
    PhysicsMesh, XI, PHI, PHI_INV, LAMBDA_STAR, CollapseType
)


@dataclass
class LearningEvent:
    """Record of a learning event."""
    timestamp: float
    node_id: str
    context_ids: List[str]
    importance: float
    learning_type: str  # 'inference', 'feedback', 'consolidation'
    crystallized: bool = False


@dataclass 
class ConnectionWeight:
    """Weighted connection between nodes."""
    from_id: str
    to_id: str
    weight: float
    last_used: float = 0.0
    use_count: int = 0


class ContinuousLearner:
    """
    CIMM-style continuous learning for PhysicsMesh.
    
    Learning modes:
    - INFERENCE: Every processed pattern is stored (passive)
    - FEEDBACK: User corrections strengthen paths (active)
    - CONSOLIDATION: Background strengthening/pruning (maintenance)
    
    Usage:
        physics = PhysicsMesh(mesh)
        learner = ContinuousLearner(physics)
        
        # Learn during inference
        result = learner.process("hello world", embeddings)
        
        # Give feedback
        learner.feedback("good")  # Strengthens recent path
        
        # Background consolidation
        learner.consolidate()
    """
    
    # Constants from legacy
    CRYSTALLIZATION_THRESHOLD = PHI * XI  # ~1.70
    DECAY_RATE = LAMBDA_STAR  # 0.618
    MIN_CONNECTION_WEIGHT = XI / 10  # 0.1057
    CONTEXT_WINDOW_SIZE = 10
    
    def __init__(self,
                 physics: PhysicsMesh,
                 learning_rate: float = XI,
                 passive_learning: bool = True,
                 auto_consolidate: bool = True,
                 consolidation_interval: int = 100):
        """
        Initialize continuous learner.
        
        Args:
            physics: PhysicsMesh for storage and physics
            learning_rate: Rate of connection strengthening
            passive_learning: Learn from all processed patterns
            auto_consolidate: Automatically run consolidation
            consolidation_interval: Steps between auto-consolidation
        """
        self.physics = physics
        self.mesh = physics.mesh
        self.learning_rate = learning_rate
        self.passive_learning = passive_learning
        self.auto_consolidate = auto_consolidate
        self.consolidation_interval = consolidation_interval
        
        # Connection weights (learned transitions)
        self.connections: Dict[Tuple[str, str], ConnectionWeight] = {}
        
        # Context window (recent node ids)
        self.context_window: List[str] = []
        
        # Learning history
        self.history: List[LearningEvent] = []
        
        # Recent path for feedback
        self.recent_path: List[str] = []
        
        # Statistics
        self.step_count = 0
        self.patterns_learned = 0
        self.consolidations = 0
    
    def learn(self,
              node: MeshNode,
              importance: float = 1.0,
              learning_type: str = "inference") -> None:
        """
        Learn from a single pattern in context.
        
        1. Strengthen connections to context
        2. Check for crystallization
        3. Update context window
        """
        node_id = node.node_id
        context_ids = self.context_window[-self.CONTEXT_WINDOW_SIZE:]
        
        # Calculate energy for crystallization check
        energy = torch.sum(node.embedding ** 2).item()
        should_crystallize = (energy * importance > self.CRYSTALLIZATION_THRESHOLD)
        
        # Strengthen connections to context (Hebbian learning)
        for ctx_id in context_ids:
            self._strengthen_connection(ctx_id, node_id, importance)
        
        # Handle crystallization
        if should_crystallize:
            self._crystallize(node, importance)
        
        # Record importance in physics layer
        self.physics.remember(node, importance)
        
        # Record event
        self.history.append(LearningEvent(
            timestamp=time.time(),
            node_id=node_id,
            context_ids=context_ids.copy(),
            importance=importance,
            learning_type=learning_type,
            crystallized=should_crystallize
        ))
        
        # Update context window
        self.context_window.append(node_id)
        if len(self.context_window) > self.CONTEXT_WINDOW_SIZE * 2:
            self.context_window = self.context_window[-self.CONTEXT_WINDOW_SIZE:]
        
        # Track recent path for feedback
        self.recent_path.append(node_id)
        if len(self.recent_path) > 50:
            self.recent_path = self.recent_path[-50:]
        
        self.patterns_learned += 1
        self.step_count += 1
        
        # Auto-consolidation
        if self.auto_consolidate and self.step_count % self.consolidation_interval == 0:
            self.consolidate()
    
    def _strengthen_connection(self,
                               from_id: str,
                               to_id: str,
                               weight: float = 1.0) -> None:
        """Strengthen connection between two nodes (Hebbian)."""
        key = (from_id, to_id)
        
        if key in self.connections:
            conn = self.connections[key]
            # Hebbian strengthening with decay
            conn.weight = conn.weight * self.DECAY_RATE + weight * self.learning_rate
            conn.last_used = time.time()
            conn.use_count += 1
        else:
            self.connections[key] = ConnectionWeight(
                from_id=from_id,
                to_id=to_id,
                weight=weight * self.learning_rate,
                last_used=time.time(),
                use_count=1
            )
    
    def _crystallize(self, node: MeshNode, importance: float) -> None:
        """Crystallize a pattern into long-term memory."""
        # Add to physics attractors
        if node.node_id not in self.physics.attractors:
            self.physics.attractors[node.node_id] = importance
            
        # Mark as high confidence
        node.confidence = min(1.0, node.confidence + 0.2)
    
    def learn_sequence(self,
                       tokens: List[str],
                       embeddings: List[torch.Tensor],
                       source: str = "learning",
                       importance: float = 0.5) -> List[MeshNode]:
        """
        Learn a full sequence of tokens.
        
        Creates nodes and learns connections between them.
        """
        nodes = self.physics.store_pattern(tokens, embeddings, source, importance)
        
        for node in nodes:
            self.learn(node, importance, "inference")
        
        return nodes
    
    def feedback(self, 
                 feedback_type: str,
                 strength: float = 1.0) -> int:
        """
        Give feedback on recent path.
        
        feedback_type:
        - "good": Strengthen recent path
        - "bad": Weaken recent path
        - "repeat": Strongly reinforce recent path
        
        Returns:
            Number of connections modified
        """
        if not self.recent_path:
            return 0
        
        modified = 0
        
        if feedback_type == "good":
            # Strengthen recent connections
            for i in range(len(self.recent_path) - 1):
                self._strengthen_connection(
                    self.recent_path[i],
                    self.recent_path[i + 1],
                    strength * self.learning_rate
                )
                modified += 1
                
        elif feedback_type == "bad":
            # Weaken recent connections
            for i in range(len(self.recent_path) - 1):
                key = (self.recent_path[i], self.recent_path[i + 1])
                if key in self.connections:
                    self.connections[key].weight *= (1 - strength * 0.5)
                    modified += 1
                    
        elif feedback_type == "repeat":
            # Strongly reinforce (crystallize path)
            for i in range(len(self.recent_path) - 1):
                self._strengthen_connection(
                    self.recent_path[i],
                    self.recent_path[i + 1],
                    strength * self.learning_rate * PHI  # Boosted
                )
                modified += 1
            
            # Crystallize nodes in path
            for node_id in self.recent_path:
                if node_id in self.mesh.nodes:
                    node = self.mesh.nodes[node_id]
                    self._crystallize(node, strength)
        
        # Record feedback event
        self.history.append(LearningEvent(
            timestamp=time.time(),
            node_id=self.recent_path[-1] if self.recent_path else "",
            context_ids=self.recent_path[-5:],
            importance=strength,
            learning_type="feedback",
            crystallized=(feedback_type == "repeat")
        ))
        
        return modified
    
    def consolidate(self) -> Dict[str, int]:
        """
        Background consolidation of learned connections.
        
        1. Decay weak connections
        2. Prune very weak connections
        3. Strengthen frequently-used connections
        4. Run physics consolidation
        
        Returns:
            Statistics about consolidation
        """
        stats = {
            "decayed": 0,
            "pruned": 0,
            "strengthened": 0,
            "physics_collapsed": 0
        }
        
        # Decay and prune connections
        to_remove = []
        for key, conn in self.connections.items():
            # Decay based on age
            age = time.time() - conn.last_used
            age_factor = math.exp(-age / 3600)  # Decay over hours
            
            new_weight = conn.weight * self.DECAY_RATE * age_factor
            
            if new_weight < self.MIN_CONNECTION_WEIGHT:
                to_remove.append(key)
                stats["pruned"] += 1
            else:
                conn.weight = new_weight
                stats["decayed"] += 1
                
                # Strengthen high-use connections
                if conn.use_count > 10:
                    conn.weight *= (1 + self.learning_rate * 0.1)
                    stats["strengthened"] += 1
        
        for key in to_remove:
            del self.connections[key]
        
        # Run physics consolidation
        self.physics.step()
        
        # Check for entropy-triggered collapses
        if self.physics.state.entropy > 2.0:
            self.physics.force_collapse(CollapseType.ENTROPY_SPIKE)
            stats["physics_collapsed"] += 1
        
        self.consolidations += 1
        
        return stats
    
    def get_transitions(self, node_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get learned transitions from a node."""
        transitions = []
        for (from_id, to_id), conn in self.connections.items():
            if from_id == node_id:
                transitions.append((to_id, conn.weight))
        
        transitions.sort(key=lambda x: x[1], reverse=True)
        return transitions[:top_k]
    
    def predict_next(self,
                     context: List[MeshNode],
                     top_k: int = 5) -> List[Tuple[MeshNode, float]]:
        """
        Predict next patterns using learned transitions.
        
        Combines:
        1. Physics-based prediction (attractors, resonance)
        2. Learned transitions (connection weights)
        """
        if not context:
            return []
        
        # Get physics predictions
        physics_preds = self.physics.predict_next(context, top_k * 2)
        
        # Get learned transition predictions
        candidates: Dict[str, float] = {}
        
        for node in context[-3:]:  # Recent context
            transitions = self.get_transitions(node.node_id, top_k)
            for to_id, weight in transitions:
                candidates[to_id] = candidates.get(to_id, 0.0) + weight
        
        # Combine physics and learned predictions
        combined: Dict[str, float] = {}
        
        for node, score in physics_preds:
            combined[node.node_id] = combined.get(node.node_id, 0.0) + score * PHI_INV
        
        for node_id, score in candidates.items():
            combined[node_id] = combined.get(node_id, 0.0) + score * PHI_INV
        
        # Sort and return
        sorted_preds = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for node_id, score in sorted_preds:
            if node_id in self.mesh.nodes:
                results.append((self.mesh.nodes[node_id], score))
        
        return results
    
    def stats(self) -> Dict:
        """Get learning statistics."""
        crystallized = sum(1 for e in self.history if e.crystallized)
        feedback_count = sum(1 for e in self.history if e.learning_type == "feedback")
        
        return {
            "patterns_learned": self.patterns_learned,
            "connections": len(self.connections),
            "crystallizations": crystallized,
            "feedback_events": feedback_count,
            "consolidations": self.consolidations,
            "context_size": len(self.context_window),
            "physics_attractors": len(self.physics.attractors),
            "step_count": self.step_count
        }
    
    def export_connections(self) -> Dict:
        """Export connection weights for persistence."""
        return {
            f"{k[0]}|{k[1]}": {
                "weight": v.weight,
                "use_count": v.use_count
            }
            for k, v in self.connections.items()
        }
    
    def import_connections(self, data: Dict) -> int:
        """Import connection weights from saved data."""
        imported = 0
        for key_str, values in data.items():
            from_id, to_id = key_str.split("|")
            key = (from_id, to_id)
            self.connections[key] = ConnectionWeight(
                from_id=from_id,
                to_id=to_id,
                weight=values["weight"],
                use_count=values.get("use_count", 0),
                last_used=time.time()
            )
            imported += 1
        return imported


class AdaptiveGenerator:
    """
    Generator that learns while generating.
    
    Every generation leaves a trace, improving future generations.
    """
    
    def __init__(self,
                 learner: ContinuousLearner,
                 embeddings: 'SimpleEmbeddings',
                 learn_from_output: bool = True,
                 feedback_enabled: bool = True):
        self.learner = learner
        self.physics = learner.physics
        self.mesh = learner.mesh
        self.embeddings = embeddings
        self.learn_from_output = learn_from_output
        self.feedback_enabled = feedback_enabled
        
        # Import PhysicsGenerator for generation
        from .physics_generator import PhysicsGenerator, GenerationConfig
        self.generator = PhysicsGenerator(
            self.physics,
            self.embeddings,
            GenerationConfig(
                learn_from_generation=False  # We handle learning here
            )
        )
    
    def generate(self,
                 prompt: str,
                 max_tokens: int = 50,
                 temperature: float = 0.8) -> 'GenerationResult':
        """
        Generate with learning.
        
        Uses learned transitions in addition to physics predictions.
        """
        from .physics_generator import GenerationConfig, GenerationResult
        
        # Configure generation
        self.generator.config.max_tokens = max_tokens
        self.generator.config.temperature = temperature
        
        # Encode prompt and learn from it
        prompt_nodes = self.generator.encode_prompt(prompt)
        for node in prompt_nodes:
            self.learner.learn(node, importance=0.5, learning_type="inference")
        
        # Generate
        result = self.generator.generate(prompt)
        
        # Learn from generated output
        if self.learn_from_output:
            for node in result.nodes:
                self.learner.learn(node, importance=0.3, learning_type="inference")
        
        return result
    
    def good(self) -> int:
        """Give positive feedback on last generation."""
        if self.feedback_enabled:
            return self.learner.feedback("good")
        return 0
    
    def bad(self) -> int:
        """Give negative feedback on last generation."""
        if self.feedback_enabled:
            return self.learner.feedback("bad")
        return 0
    
    def repeat(self) -> int:
        """Strongly reinforce last generation path."""
        if self.feedback_enabled:
            return self.learner.feedback("repeat")
        return 0
    
    def stats(self) -> Dict:
        """Get combined statistics."""
        learner_stats = self.learner.stats()
        learner_stats["mesh_nodes"] = len(self.mesh.nodes)
        learner_stats["physics_entropy"] = self.physics.state.entropy
        return learner_stats
