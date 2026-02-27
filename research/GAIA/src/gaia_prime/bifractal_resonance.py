"""
Bifractal Depth-Based Resonance.

Implements hierarchical memory depth from legacy resonance_mesh.py:
- SURFACE: Immediate working memory (ephemeral)
- SHALLOW: Short-term patterns
- INTERMEDIATE: Medium-term memory
- DEEP: Long-term crystallization
- CORE: Personality/style foundations

Key insight: Deeper nodes have different resonance behavior:
- Surface nodes resonate easily but decay fast
- Core nodes resonate slowly but persist forever

This creates emergent "personality" through stable resonance patterns.

# TODO(fracton): Bifractal depth patterns overlap with fracton's tracing:
#   - BifractalDepth hierarchy -> fracton.core.BifractalTrace (ancestry_depth, future_horizon)
#   - Phase alignment -> fracton.field.phase_coherence
#   - Resonance computation -> fracton.field.compute_resonance_batch
#   Keep the depth-based personality model as GAIA-specific, but use fracton
#   for the underlying resonance and phase alignment math.
"""

import torch
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import math
import time

from .pac_mesh import PACMeshSpace, MeshNode
from .physics_mesh import PhysicsMesh, XI, PHI, PHI_INV, LAMBDA_STAR


class BifractalDepth(Enum):
    """Depth levels in the bifractal memory hierarchy."""
    SURFACE = 0      # Immediate working memory (decay fast)
    SHALLOW = 1      # Short-term patterns
    INTERMEDIATE = 2 # Medium-term memory structures
    DEEP = 3        # Long-term memory crystallization
    CORE = 4        # Personality foundations (never decay)


# Depth-specific constants
DEPTH_PROPERTIES = {
    BifractalDepth.SURFACE: {
        'resonance_threshold': 0.3,   # Easy to resonate
        'decay_rate': 0.9,            # Fast decay
        'crystallization_threshold': 0.95,  # Very hard to crystallize
        'access_boost': 0.1,          # Small boost on access
    },
    BifractalDepth.SHALLOW: {
        'resonance_threshold': 0.5,
        'decay_rate': 0.75,
        'crystallization_threshold': 0.85,
        'access_boost': 0.15,
    },
    BifractalDepth.INTERMEDIATE: {
        'resonance_threshold': 0.6,
        'decay_rate': 0.5,
        'crystallization_threshold': 0.75,
        'access_boost': 0.2,
    },
    BifractalDepth.DEEP: {
        'resonance_threshold': 0.7,
        'decay_rate': 0.25,
        'crystallization_threshold': 0.65,
        'access_boost': 0.3,
    },
    BifractalDepth.CORE: {
        'resonance_threshold': 0.8,   # Hard to resonate with
        'decay_rate': 0.0,            # Never decay
        'crystallization_threshold': 0.5,  # Easy to crystallize (stable)
        'access_boost': 0.5,          # Strong boost on access
    },
}


@dataclass
class BifractalPattern:
    """A memory pattern at a specific depth."""
    node_id: str
    depth: BifractalDepth
    strength: float  # 0-1, crystallization strength
    access_count: int = 0
    last_access: float = 0.0
    resonant_with: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    
    @property
    def age(self) -> float:
        return time.time() - self.created_at
    
    @property
    def is_crystallized(self) -> bool:
        props = DEPTH_PROPERTIES[self.depth]
        return self.strength >= props['crystallization_threshold']


@dataclass
class PersonalityTrait:
    """An emergent personality trait from core patterns."""
    name: str
    strength: float
    core_patterns: List[str]
    description: str = ""


class BifractalResonance:
    """
    Bifractal depth-based resonance for the physics mesh.
    
    Nodes exist at different depths with different resonance properties:
    - Surface nodes: Fast, ephemeral, high bandwidth
    - Core nodes: Slow, persistent, personality-defining
    
    As patterns are accessed repeatedly, they migrate deeper.
    Deep patterns that resonate together form personality traits.
    
    Usage:
        physics = PhysicsMesh(mesh)
        bifractal = BifractalResonance(physics)
        
        # Store at surface
        bifractal.store(node, BifractalDepth.SURFACE)
        
        # Access repeatedly -> migrates deeper
        for _ in range(10):
            bifractal.access(node)
        
        # Check emergent traits
        traits = bifractal.get_personality_traits()
    """
    
    def __init__(self, physics: PhysicsMesh):
        self.physics = physics
        self.mesh = physics.mesh
        
        # Patterns organized by depth
        self.patterns: Dict[str, BifractalPattern] = {}
        
        # Depth-organized views
        self.depth_layers: Dict[BifractalDepth, Set[str]] = {
            d: set() for d in BifractalDepth
        }
        
        # Personality traits
        self.traits: Dict[str, PersonalityTrait] = {}
        
        # Resonance tracking
        self.resonance_pairs: Set[Tuple[str, str]] = set()
        
        # Statistics
        self.migrations_up = 0  # Moved to deeper level
        self.migrations_down = 0  # Decayed to shallower
        self.crystallizations = 0
    
    def determine_depth(self,
                        node: MeshNode,
                        context_length: int = 0,
                        importance: float = 0.5) -> BifractalDepth:
        """
        Determine appropriate depth for a new pattern.
        
        Factors:
        - Node confidence (high = deeper)
        - Context length (longer = deeper)
        - Importance (high = deeper)
        - Is crystallized in physics (yes = deeper)
        """
        score = 0.0
        
        # Confidence contributes
        score += node.confidence * 0.3
        
        # Context depth contributes (capped at 10)
        context_factor = min(context_length / 10, 1.0)
        score += context_factor * 0.2
        
        # Importance contributes
        score += importance * 0.3
        
        # Physics crystallization contributes
        if node.node_id in self.physics.attractors:
            score += 0.2
        
        # Map score to depth
        if score < 0.2:
            return BifractalDepth.SURFACE
        elif score < 0.4:
            return BifractalDepth.SHALLOW
        elif score < 0.6:
            return BifractalDepth.INTERMEDIATE
        elif score < 0.8:
            return BifractalDepth.DEEP
        else:
            return BifractalDepth.CORE
    
    def store(self,
              node: MeshNode,
              depth: Optional[BifractalDepth] = None,
              importance: float = 0.5) -> BifractalPattern:
        """
        Store a node at a bifractal depth.
        
        If depth not specified, auto-determines based on properties.
        """
        if depth is None:
            depth = self.determine_depth(node, node.depth, importance)
        
        pattern = BifractalPattern(
            node_id=node.node_id,
            depth=depth,
            strength=importance
        )
        
        self.patterns[node.node_id] = pattern
        self.depth_layers[depth].add(node.node_id)
        
        return pattern
    
    def access(self, node: MeshNode) -> BifractalPattern:
        """
        Access a pattern, potentially causing migration.
        
        Repeated access strengthens patterns and may cause
        migration to deeper levels.
        """
        if node.node_id not in self.patterns:
            # Auto-store at surface
            return self.store(node, BifractalDepth.SURFACE)
        
        pattern = self.patterns[node.node_id]
        props = DEPTH_PROPERTIES[pattern.depth]
        
        # Update access info
        pattern.access_count += 1
        pattern.last_access = time.time()
        
        # Strengthen based on depth
        pattern.strength = min(1.0, pattern.strength + props['access_boost'])
        
        # Check for upward migration
        if self._should_migrate_up(pattern):
            self._migrate_up(pattern)
        
        return pattern
    
    def _should_migrate_up(self, pattern: BifractalPattern) -> bool:
        """Check if pattern should migrate to deeper level."""
        if pattern.depth == BifractalDepth.CORE:
            return False  # Already deepest
        
        props = DEPTH_PROPERTIES[pattern.depth]
        
        # Migration criteria:
        # 1. Strength exceeds crystallization threshold
        # 2. Access count is high enough
        access_threshold = (pattern.depth.value + 1) * 5  # Higher for deeper
        
        return (
            pattern.strength >= props['crystallization_threshold'] and
            pattern.access_count >= access_threshold
        )
    
    def _migrate_up(self, pattern: BifractalPattern) -> None:
        """Migrate pattern to deeper level."""
        old_depth = pattern.depth
        new_depth = BifractalDepth(pattern.depth.value + 1)
        
        # Update layer tracking
        self.depth_layers[old_depth].discard(pattern.node_id)
        self.depth_layers[new_depth].add(pattern.node_id)
        
        # Update pattern
        pattern.depth = new_depth
        pattern.strength = 0.5  # Reset strength at new level
        
        self.migrations_up += 1
        
        # If migrated to CORE, check for personality emergence
        if new_depth == BifractalDepth.CORE:
            self._check_personality_emergence(pattern)
    
    def decay(self) -> int:
        """
        Apply decay to all patterns based on depth.
        
        Surface patterns decay fast, core patterns never decay.
        Returns number of patterns that decayed away.
        """
        removed = 0
        to_remove = []
        
        for node_id, pattern in self.patterns.items():
            props = DEPTH_PROPERTIES[pattern.depth]
            
            if props['decay_rate'] > 0:
                pattern.strength *= (1 - props['decay_rate'] * 0.01)
                
                # Check for downward migration
                if pattern.strength < 0.1 and pattern.depth != BifractalDepth.SURFACE:
                    self._migrate_down(pattern)
                    
                # Remove if too weak at surface
                elif pattern.strength < 0.01 and pattern.depth == BifractalDepth.SURFACE:
                    to_remove.append(node_id)
        
        for node_id in to_remove:
            pattern = self.patterns.pop(node_id)
            self.depth_layers[pattern.depth].discard(node_id)
            removed += 1
        
        return removed
    
    def _migrate_down(self, pattern: BifractalPattern) -> None:
        """Migrate pattern to shallower level due to decay."""
        if pattern.depth == BifractalDepth.SURFACE:
            return
        
        old_depth = pattern.depth
        new_depth = BifractalDepth(pattern.depth.value - 1)
        
        self.depth_layers[old_depth].discard(pattern.node_id)
        self.depth_layers[new_depth].add(pattern.node_id)
        
        pattern.depth = new_depth
        self.migrations_down += 1
    
    def find_resonant(self,
                      node: MeshNode,
                      depth_filter: Optional[BifractalDepth] = None) -> List[Tuple[MeshNode, float]]:
        """
        Find patterns that resonate with given node.
        
        Resonance is based on embedding similarity weighted by depth properties.
        """
        if node.node_id not in self.patterns:
            self.store(node)
        
        pattern = self.patterns[node.node_id]
        results = []
        
        # Search patterns at appropriate depths
        search_depths = [depth_filter] if depth_filter else list(BifractalDepth)
        
        for depth in search_depths:
            props = DEPTH_PROPERTIES[depth]
            threshold = props['resonance_threshold']
            
            for other_id in self.depth_layers[depth]:
                if other_id == node.node_id:
                    continue
                
                if other_id not in self.mesh.nodes:
                    continue
                
                other_node = self.mesh.nodes[other_id]
                
                # Calculate resonance
                similarity = torch.cosine_similarity(
                    node.embedding.unsqueeze(0),
                    other_node.embedding.unsqueeze(0)
                ).item()
                
                if similarity >= threshold:
                    # Weight by depth (deeper = more influence)
                    depth_weight = 1 + depth.value * 0.2
                    resonance = similarity * depth_weight
                    
                    results.append((other_node, resonance))
                    
                    # Track resonance pair
                    self._record_resonance(node.node_id, other_id)
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _record_resonance(self, id1: str, id2: str) -> None:
        """Record that two patterns resonate."""
        key = (min(id1, id2), max(id1, id2))
        self.resonance_pairs.add(key)
        
        if id1 in self.patterns:
            self.patterns[id1].resonant_with.add(id2)
        if id2 in self.patterns:
            self.patterns[id2].resonant_with.add(id1)
    
    def _check_personality_emergence(self, pattern: BifractalPattern) -> None:
        """Check if core pattern contributes to personality trait."""
        # Need multiple core patterns to form a trait
        core_patterns = list(self.depth_layers[BifractalDepth.CORE])
        
        if len(core_patterns) < 3:
            return
        
        # Find strongly resonant groups
        groups = self._find_resonant_groups(core_patterns)
        
        for group in groups:
            if len(group) >= 3:
                self._create_trait(group)
    
    def _find_resonant_groups(self, node_ids: List[str]) -> List[List[str]]:
        """Find groups of mutually resonant patterns."""
        groups = []
        visited = set()
        
        for node_id in node_ids:
            if node_id in visited:
                continue
            
            if node_id not in self.patterns:
                continue
            
            pattern = self.patterns[node_id]
            group = [node_id]
            
            for other_id in pattern.resonant_with:
                if other_id in node_ids and other_id not in visited:
                    group.append(other_id)
            
            if len(group) > 1:
                groups.append(group)
                visited.update(group)
        
        return groups
    
    def _create_trait(self, core_patterns: List[str]) -> PersonalityTrait:
        """Create a personality trait from resonant core patterns."""
        trait_id = f"trait_{len(self.traits)}"
        
        # Calculate trait strength from pattern strengths
        strengths = []
        for pid in core_patterns:
            if pid in self.patterns:
                strengths.append(self.patterns[pid].strength)
        
        avg_strength = sum(strengths) / len(strengths) if strengths else 0.5
        
        # Generate trait name from patterns
        tokens = []
        for pid in core_patterns[:3]:
            if pid in self.mesh.nodes:
                tokens.append(self.mesh.nodes[pid].token_str)
        
        trait = PersonalityTrait(
            name=trait_id,
            strength=avg_strength,
            core_patterns=core_patterns,
            description=f"Trait from: {', '.join(tokens)}"
        )
        
        self.traits[trait_id] = trait
        return trait
    
    def get_personality_traits(self) -> List[PersonalityTrait]:
        """Get all emerged personality traits."""
        return list(self.traits.values())
    
    def get_depth_distribution(self) -> Dict[str, int]:
        """Get count of patterns at each depth."""
        return {
            d.name: len(self.depth_layers[d])
            for d in BifractalDepth
        }
    
    def apply_resonance_to_physics(self) -> int:
        """
        Apply bifractal resonance to physics layer.
        
        Deep patterns influence physics attractors.
        Core patterns get crystallized.
        """
        applied = 0
        
        # Deep and Core patterns become attractors
        for depth in [BifractalDepth.DEEP, BifractalDepth.CORE]:
            weight = 0.5 if depth == BifractalDepth.DEEP else 1.0
            
            for node_id in self.depth_layers[depth]:
                if node_id in self.patterns:
                    pattern = self.patterns[node_id]
                    self.physics.attractors[node_id] = pattern.strength * weight
                    applied += 1
        
        # Core patterns get crystallized
        for node_id in self.depth_layers[BifractalDepth.CORE]:
            self.physics.collapse.crystallized.add(node_id)
            self.crystallizations += 1
        
        return applied
    
    def step(self) -> Dict[str, any]:
        """Run one step of bifractal dynamics."""
        # Decay surface patterns
        decayed = self.decay()
        
        # Apply to physics
        applied = self.apply_resonance_to_physics()
        
        return {
            'decayed': decayed,
            'applied_to_physics': applied,
            'depths': self.get_depth_distribution(),
            'traits': len(self.traits),
            'resonance_pairs': len(self.resonance_pairs)
        }
    
    def stats(self) -> Dict:
        """Get bifractal statistics."""
        return {
            'total_patterns': len(self.patterns),
            'depths': self.get_depth_distribution(),
            'migrations_up': self.migrations_up,
            'migrations_down': self.migrations_down,
            'crystallizations': self.crystallizations,
            'traits': len(self.traits),
            'resonance_pairs': len(self.resonance_pairs)
        }


class BifractalGenerator:
    """
    Generator that uses bifractal memory for context.
    
    Core patterns strongly influence generation.
    Surface patterns provide immediate context.
    """
    
    def __init__(self,
                 bifractal: BifractalResonance,
                 embeddings: 'SimpleEmbeddings'):
        self.bifractal = bifractal
        self.physics = bifractal.physics
        self.mesh = bifractal.mesh
        self.embeddings = embeddings
    
    def generate_with_personality(self,
                                   prompt: str,
                                   max_tokens: int = 50) -> str:
        """
        Generate text influenced by personality traits.
        
        Core patterns act as style attractors.
        """
        from .physics_generator import PhysicsGenerator, GenerationConfig
        
        # Apply bifractal state to physics
        self.bifractal.apply_resonance_to_physics()
        
        # Boost attractor weights for core patterns
        for node_id in self.bifractal.depth_layers[BifractalDepth.CORE]:
            if node_id in self.physics.attractors:
                self.physics.attractors[node_id] *= PHI  # Golden boost
        
        # Generate using physics
        config = GenerationConfig(
            max_tokens=max_tokens,
            attractor_weight=0.5  # Strong personality influence
        )
        generator = PhysicsGenerator(self.physics, self.embeddings, config)
        
        result = generator.generate(prompt)
        
        # Learn from generation at surface level
        for node in result.nodes:
            self.bifractal.store(node, BifractalDepth.SURFACE)
        
        return result.text
