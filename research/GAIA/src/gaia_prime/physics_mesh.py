"""
Physics Mesh: Deep Intelligence Layer for PAC Mesh.

Transforms the mesh from a data structure into a physics-governed field.
The interface becomes intelligent substrate through:

1. Entropy-Driven Collapse: High entropy triggers convergence events
2. Xi Conservation: f(parent) = Σf(children) with Xi = 1.0571
3. Phase Alignment: Similar embeddings resonate and reinforce
4. Crystallization: High-confidence patterns freeze into attractors
5. Curvature: Measure mesh geometry to detect embedding tension

Architecture:
    ┌─────────────────────────────────────────┐
    │          PHYSICS ENGINE LAYER           │
    │  EntropyMonitor → ConservationEnforcer  │
    │         ↓                ↓              │
    │     CollapseEngine ← ResonanceField     │
    └─────────────────────────────────────────┘
                      ↓
              Crystallized Knowledge

Based on legacy systems:
- TinyCIMM-Planck: EntropyMonitorLite, quantum_collapse
- TinyCIMM-Euler: UnifiedSymbolicCollapseTracker, SEC
- ConservationEngine: Xi operator, potential/actualization
- CollapseCore: entropy_tension, curvature
- ResonanceMesh: BifractalMemoryPattern, phase alignment
"""

import torch
import torch.nn.functional as F
import math
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time

from .pac_mesh import PACMeshSpace, MeshNode

# Import validated constants (derived, not fitted)
from .validated_constants import (
    XI, PHI, PHI_INV, LAMBDA_STAR,
    ENTROPY_OPTIMAL_LOW, ENTROPY_OPTIMAL_HIGH, COLLAPSE_THRESHOLD,
    ATTRACTION_FRACTION, REPULSION_FRACTION
)


class CollapseType(Enum):
    """Types of collapse events in the mesh."""
    ENTROPY_SPIKE = "entropy_spike"      # Sudden entropy increase
    CONVERGENCE = "convergence"          # Multiple paths meet
    CRYSTALLIZATION = "crystallization"  # Pattern freezes
    RESONANCE = "resonance"              # Phase alignment lock
    TENSION_RELEASE = "tension_release"  # Curvature relaxation


@dataclass
class CollapseEvent:
    """Record of a collapse event in the mesh."""
    collapse_type: CollapseType
    timestamp: float
    node_ids: List[int]
    entropy_before: float
    entropy_after: float
    magnitude: float
    crystallized: bool = False


@dataclass
class FieldState:
    """Current state of the physics field."""
    entropy: float = 0.0
    entropy_momentum: float = 0.0
    curvature: float = 0.0
    resonance_strength: float = 0.0
    conservation_residual: float = 0.0
    crystallization_ratio: float = 0.0
    
    @property
    def is_stable(self) -> bool:
        """Is the field in a stable state?"""
        return (self.entropy < 1.0 and 
                self.curvature < 0.5 and 
                self.conservation_residual < 0.1)
    
    @property
    def collapse_pressure(self) -> float:
        """Pressure toward collapse (high = imminent)."""
        return (self.entropy * 0.4 + 
                self.curvature * 0.3 + 
                (1 - self.resonance_strength) * 0.3)


# =============================================================================
# ENTROPY MONITOR (from TinyCIMM-Planck)
# =============================================================================

class EntropyMonitor:
    """
    Track mesh-wide entropy with momentum.
    
    Entropy measures disorder/uncertainty in the mesh.
    High entropy = many uncertain paths
    Low entropy = crystallized, confident structure
    
    Triggers collapse events when entropy exceeds thresholds.
    """
    
    def __init__(self, 
                 momentum: float = 0.9,
                 window_size: int = 50,
                 collapse_threshold: float = 2.0):
        self.momentum = momentum
        self.window_size = window_size
        self.collapse_threshold = collapse_threshold
        
        # State
        self.current_entropy = 0.0
        self.entropy_momentum = 0.0
        self.entropy_history: deque = deque(maxlen=window_size)
        
        # Collapse detection
        self.collapse_events: List[CollapseEvent] = []
        self.last_collapse_time = 0.0
        self.collapse_cooldown = 0.1  # seconds
    
    def compute_node_entropy(self, node: MeshNode) -> float:
        """Compute entropy of a single node's children distribution."""
        if not node.children:
            return 0.0
        
        # Get child counts
        counts = [count for (child, count) in node.children.values()]
        total = sum(counts)
        
        if total == 0:
            return 0.0
        
        # Shannon entropy of transition distribution
        probs = [c / total for c in counts]
        entropy = -sum(p * math.log(p + 1e-9) for p in probs)
        
        return entropy
    
    def compute_mesh_entropy(self, mesh: PACMeshSpace) -> float:
        """
        Compute total entropy of the mesh.
        
        Combines:
        1. Node distribution entropy (how spread are nodes)
        2. Transition entropy (how uncertain are paths)
        3. Convergence entropy (how much disagreement)
        """
        if not mesh.nodes:
            return 0.0
        
        # 1. Node transition entropy
        node_entropies = []
        for node in mesh.nodes.values():
            h = self.compute_node_entropy(node)
            if h > 0:
                node_entropies.append(h)
        
        avg_transition_entropy = (
            sum(node_entropies) / len(node_entropies) 
            if node_entropies else 0.0
        )
        
        # 2. Depth distribution entropy
        depth_counts: Dict[int, int] = {}
        for node in mesh.nodes.values():
            depth_counts[node.depth] = depth_counts.get(node.depth, 0) + 1
        
        total_nodes = len(mesh.nodes)
        depth_probs = [c / total_nodes for c in depth_counts.values()]
        depth_entropy = -sum(p * math.log(p + 1e-9) for p in depth_probs)
        
        # 3. Convergence factor (inverse - more convergence = less entropy)
        convergent_nodes = [n for n in mesh.nodes.values() if n.is_convergence_point]
        convergence_ratio = len(convergent_nodes) / total_nodes if total_nodes else 0
        convergence_factor = 1.0 - convergence_ratio  # Less convergence = more entropy
        
        # Combine with weights
        total_entropy = (
            avg_transition_entropy * 0.5 +
            depth_entropy * 0.3 +
            convergence_factor * 0.2
        ) * XI  # Scale by balance operator
        
        return total_entropy
    
    def update(self, mesh: PACMeshSpace) -> Tuple[float, Optional[CollapseEvent]]:
        """
        Update entropy state and check for collapse events.
        
        Returns: (current_entropy, collapse_event_or_none)
        """
        new_entropy = self.compute_mesh_entropy(mesh)
        
        # Update with momentum
        prev_entropy = self.current_entropy
        self.current_entropy = (
            self.momentum * self.current_entropy + 
            (1 - self.momentum) * new_entropy
        )
        
        # Track momentum (rate of change)
        self.entropy_momentum = new_entropy - prev_entropy
        
        # Add to history
        self.entropy_history.append(self.current_entropy)
        
        # Check for collapse event
        collapse_event = None
        current_time = time.time()
        
        if (current_time - self.last_collapse_time > self.collapse_cooldown and
            self.current_entropy > self.collapse_threshold):
            
            # Entropy spike detected
            collapse_event = CollapseEvent(
                collapse_type=CollapseType.ENTROPY_SPIKE,
                timestamp=current_time,
                node_ids=[],  # Will be filled by collapse engine
                entropy_before=self.current_entropy,
                entropy_after=0.0,  # Will be updated after collapse
                magnitude=self.current_entropy / self.collapse_threshold
            )
            self.collapse_events.append(collapse_event)
            self.last_collapse_time = current_time
        
        return self.current_entropy, collapse_event
    
    def get_variance(self) -> float:
        """Get variance of entropy over recent history."""
        if len(self.entropy_history) < 2:
            return 0.0
        
        history = list(self.entropy_history)
        mean = sum(history) / len(history)
        variance = sum((h - mean) ** 2 for h in history) / len(history)
        return variance


# =============================================================================
# CONSERVATION ENFORCER (from ConservationEngine)
# =============================================================================

class ConservationEnforcer:
    """
    Enforce PAC conservation: f(parent) = Σf(children).
    
    Uses Xi operator (1.0571) for balance.
    Tracks violations and applies corrections.
    """
    
    def __init__(self, tolerance: float = 0.01):
        self.xi = XI
        self.tolerance = tolerance
        self.violation_history: List[float] = []
    
    def compute_node_conservation(self, node: MeshNode) -> Tuple[float, float]:
        """
        Check conservation for a node.
        
        Returns: (residual, correction_needed)
        
        Conservation law: The "potential" of a parent should equal
        the sum of "actualized" children, scaled by Xi.
        """
        if not node.children:
            return 0.0, 0.0
        
        # Parent potential = confidence * incoming
        parent_potential = node.confidence * (node.total_incoming + 1)
        
        # Children actualization = sum of (child_conf * count)
        children_actual = sum(
            child.confidence * count 
            for (child, count) in node.children.values()
        )
        
        # Conservation: parent = Xi * children (with tolerance)
        expected = parent_potential / self.xi
        residual = abs(children_actual - expected)
        
        if residual > self.tolerance:
            # Correction needed
            correction = (expected - children_actual) / len(node.children)
            return residual, correction
        
        return residual, 0.0
    
    def compute_mesh_conservation(self, mesh: PACMeshSpace) -> float:
        """
        Compute overall conservation residual for the mesh.
        
        Returns value between 0 (perfect conservation) and 1+ (violation).
        """
        if not mesh.nodes:
            return 0.0
        
        total_residual = 0.0
        count = 0
        
        for node in mesh.nodes.values():
            if node.children:  # Only check nodes with children
                residual, _ = self.compute_node_conservation(node)
                total_residual += residual
                count += 1
        
        avg_residual = total_residual / count if count else 0.0
        self.violation_history.append(avg_residual)
        
        return avg_residual
    
    def apply_corrections(self, mesh: PACMeshSpace) -> int:
        """
        Apply conservation corrections to nodes with violations.
        
        Returns number of nodes corrected.
        """
        corrections_made = 0
        
        for node in mesh.nodes.values():
            residual, correction = self.compute_node_conservation(node)
            
            if abs(correction) > 1e-6:
                # Distribute correction across children
                for child, count in node.children.values():
                    # Adjust child confidence to restore conservation
                    child.confidence = max(0.0, min(1.0,
                        child.confidence + correction / count
                    ))
                corrections_made += 1
        
        return corrections_made


# =============================================================================
# RESONANCE FIELD (from ResonanceMesh)
# =============================================================================

class ResonanceField:
    """
    Phase alignment and reinforcement through resonance.
    
    When embeddings are similar (in phase), they reinforce.
    When out of phase, they interfere destructively.
    
    This creates natural clustering of semantically related nodes.
    """
    
    def __init__(self,
                 resonance_threshold: float = 0.8,
                 phase_coupling: float = 0.1):
        self.resonance_threshold = resonance_threshold
        self.phase_coupling = phase_coupling
        
        # Track resonance pairs
        self.resonance_pairs: List[Tuple[int, int, float]] = []
    
    def compute_phase_alignment(self, 
                                 emb1: torch.Tensor, 
                                 emb2: torch.Tensor) -> float:
        """
        Compute phase alignment between two embeddings.
        
        Returns value in [-1, 1]:
        - 1.0 = perfectly in phase (reinforce)
        - 0.0 = orthogonal (no interaction)
        - -1.0 = anti-phase (cancel)
        """
        # Normalize embeddings
        emb1_norm = F.normalize(emb1.flatten().unsqueeze(0), dim=1)
        emb2_norm = F.normalize(emb2.flatten().unsqueeze(0), dim=1)
        
        # Cosine similarity as phase alignment
        phase = torch.mm(emb1_norm, emb2_norm.t()).item()
        
        return phase
    
    def find_resonant_pairs(self, mesh: PACMeshSpace) -> List[Tuple[int, int, float]]:
        """
        Find all pairs of nodes in resonance.
        
        Resonance = phase alignment above threshold.
        """
        self.resonance_pairs = []
        nodes = list(mesh.nodes.values())
        
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                if node1.embedding is None or node2.embedding is None:
                    continue
                
                phase = self.compute_phase_alignment(
                    node1.embedding, node2.embedding
                )
                
                if abs(phase) > self.resonance_threshold:
                    self.resonance_pairs.append((
                        node1.node_id, 
                        node2.node_id, 
                        phase
                    ))
        
        return self.resonance_pairs
    
    def apply_resonance_effects(self, mesh: PACMeshSpace) -> int:
        """
        Apply resonance effects: reinforce in-phase, weaken anti-phase.
        
        Returns number of nodes affected.
        """
        affected = set()
        
        for id1, id2, phase in self.resonance_pairs:
            node1 = mesh.nodes.get(id1)
            node2 = mesh.nodes.get(id2)
            
            if node1 is None or node2 is None:
                continue
            
            # In-phase: both get confidence boost
            if phase > 0:
                boost = phase * self.phase_coupling * PHI_INV
                node1.confidence = min(1.0, node1.confidence + boost)
                node2.confidence = min(1.0, node2.confidence + boost)
            
            # Anti-phase: weaker one loses confidence
            else:
                penalty = abs(phase) * self.phase_coupling * PHI_INV
                if node1.confidence < node2.confidence:
                    node1.confidence = max(0.0, node1.confidence - penalty)
                else:
                    node2.confidence = max(0.0, node2.confidence - penalty)
            
            affected.add(id1)
            affected.add(id2)
        
        return len(affected)
    
    def get_resonance_strength(self, mesh: PACMeshSpace) -> float:
        """
        Overall resonance strength of the mesh.
        
        High = many aligned patterns (stable)
        Low = chaotic, unaligned (unstable)
        """
        if not self.resonance_pairs:
            return 0.0
        
        total_nodes = len(mesh.nodes)
        resonant_nodes = len(set(
            id for (id1, id2, _) in self.resonance_pairs 
            for id in (id1, id2)
        ))
        
        return resonant_nodes / total_nodes if total_nodes else 0.0


# =============================================================================
# COLLAPSE ENGINE (from CollapseCore)
# =============================================================================

class CollapseEngine:
    """
    Handle collapse events and crystallization.
    
    Collapse types:
    1. Entropy collapse: High entropy → convergence to attractor
    2. Tension release: High curvature → geometry relaxation
    3. Crystallization: Stable pattern → frozen attractor
    
    Collapse is CREATIVE, not destructive.
    New structure emerges from collapse.
    """
    
    def __init__(self,
                 crystallization_threshold: float = 0.9,
                 tension_threshold: float = 0.7):
        self.crystallization_threshold = crystallization_threshold
        self.tension_threshold = tension_threshold
        
        # Crystallized nodes (frozen, stable attractors)
        self.crystallized: Set[int] = set()
        
        # Curvature cache
        self.node_curvatures: Dict[int, float] = {}
    
    def compute_curvature(self, node: MeshNode, mesh: PACMeshSpace) -> float:
        """
        Compute local curvature around a node.
        
        Curvature = how much the embedding space "bends" here.
        High curvature = tension, potential for collapse.
        Low curvature = smooth, stable region.
        """
        if node.embedding is None or not node.children:
            return 0.0
        
        # Collect child embeddings
        child_embeddings = []
        for child, _ in node.children.values():
            if child.embedding is not None:
                child_embeddings.append(child.embedding)
        
        if not child_embeddings:
            return 0.0
        
        # Curvature = variance of angles from parent to children
        angles = []
        parent_emb = F.normalize(node.embedding.flatten().unsqueeze(0), dim=1)
        
        for child_emb in child_embeddings:
            child_norm = F.normalize(child_emb.flatten().unsqueeze(0), dim=1)
            cos_angle = torch.mm(parent_emb, child_norm.t()).item()
            angle = math.acos(max(-1, min(1, cos_angle)))  # Clamp for numerical stability
            angles.append(angle)
        
        if len(angles) < 2:
            return 0.0
        
        # Variance of angles = curvature
        mean_angle = sum(angles) / len(angles)
        variance = sum((a - mean_angle) ** 2 for a in angles) / len(angles)
        curvature = math.sqrt(variance)
        
        self.node_curvatures[node.node_id] = curvature
        return curvature
    
    def check_crystallization(self, node: MeshNode) -> bool:
        """
        Check if a node should crystallize.
        
        Crystallization = pattern is stable enough to freeze.
        """
        if node.node_id in self.crystallized:
            return True
        
        # Criteria for crystallization:
        # 1. High confidence
        # 2. Multiple sources agree
        # 3. Is a convergence point (multiple paths led here)
        # 4. Low local curvature
        
        # Default to 0 curvature if not computed (leaf nodes)
        curvature = self.node_curvatures.get(node.node_id, 0.0)
        
        score = (
            node.confidence * 0.4 +
            min(len(node.sources) / 3, 1.0) * 0.2 +
            min(node.convergence_factor / 3, 1.0) * 0.2 +
            (1 - curvature) * 0.2
        )
        
        return score > self.crystallization_threshold
    
    def execute_collapse(self, 
                         mesh: PACMeshSpace,
                         event: CollapseEvent) -> List[int]:
        """
        Execute a collapse event.
        
        Returns list of node IDs affected.
        """
        affected = []
        
        if event.collapse_type == CollapseType.ENTROPY_SPIKE:
            # Find highest-entropy nodes and collapse them
            node_entropies = []
            for node in mesh.nodes.values():
                if node.children:
                    counts = [c for (_, c) in node.children.values()]
                    total = sum(counts)
                    if total > 0:
                        probs = [c/total for c in counts]
                        h = -sum(p * math.log(p + 1e-9) for p in probs)
                        node_entropies.append((node.node_id, h))
            
            # Collapse top-entropy nodes toward their most likely child
            node_entropies.sort(key=lambda x: x[1], reverse=True)
            for node_id, entropy in node_entropies[:5]:  # Top 5 entropy nodes
                node = mesh.nodes.get(node_id)
                if node and node.children:
                    # Boost confidence of most likely child
                    best_child = max(
                        node.children.values(), 
                        key=lambda x: x[1]
                    )[0]
                    best_child.confidence = min(1.0, 
                        best_child.confidence + PHI_INV * event.magnitude
                    )
                    affected.append(node_id)
        
        elif event.collapse_type == CollapseType.CONVERGENCE:
            # Strengthen convergence points
            for node in mesh.nodes.values():
                if node.is_convergence_point:
                    node.confidence = min(1.0,
                        node.confidence + PHI_INV * node.convergence_factor / 10
                    )
                    affected.append(node.node_id)
        
        elif event.collapse_type == CollapseType.CRYSTALLIZATION:
            # Crystallize stable nodes
            for node in mesh.nodes.values():
                if self.check_crystallization(node):
                    self.crystallized.add(node.node_id)
                    node.confidence = 1.0  # Full confidence
                    affected.append(node.node_id)
        
        elif event.collapse_type == CollapseType.TENSION_RELEASE:
            # Reduce curvature by averaging embeddings
            for node_id, curvature in self.node_curvatures.items():
                if curvature > self.tension_threshold:
                    node = mesh.nodes.get(node_id)
                    if node and node.children and node.embedding is not None:
                        # Average with children to reduce tension
                        child_embs = [
                            c.embedding for c, _ in node.children.values()
                            if c.embedding is not None
                        ]
                        if child_embs:
                            avg_child = torch.stack(child_embs).mean(dim=0)
                            node.embedding = (
                                node.embedding * LAMBDA_STAR + 
                                avg_child * (1 - LAMBDA_STAR)
                            )
                            affected.append(node_id)
        
        event.node_ids = affected
        return affected


# =============================================================================
# PHYSICS MESH - Main Integration Class
# =============================================================================

class PhysicsMesh:
    """
    Physics-governed PAC Mesh with deep intelligence layer.
    
    Wraps PACMeshSpace with physics engines:
    - EntropyMonitor: Track disorder, trigger collapse
    - ConservationEnforcer: Maintain f(parent) = Σf(children)
    - ResonanceField: Phase alignment and reinforcement
    - CollapseEngine: Crystallization and structure emergence
    
    KEY INSIGHT: The mesh IS the memory. Physics operations:
    - READ from mesh (query attractors, find resonant patterns)
    - WRITE to mesh (crystallize, merge, create convergence)
    
    Usage:
        mesh = PACMeshSpace()
        physics = PhysicsMesh(mesh)
        
        # Add nodes to mesh...
        mesh.get_or_create_root(...)
        
        # Process physics
        state = physics.step()
        
        # Query memory
        similar = physics.query("Paris")  # Find resonant patterns
        
        # Store important pattern
        physics.remember(node, importance=0.9)
    """
    
    def __init__(self, mesh: PACMeshSpace):
        self.mesh = mesh
        
        # Physics engines
        self.entropy_monitor = EntropyMonitor()
        self.conservation = ConservationEnforcer()
        self.resonance = ResonanceField()
        self.collapse = CollapseEngine()
        
        # Current state
        self.state = FieldState()
        
        # === MEMORY LAYER ===
        # Attractor memory: crystallized patterns that influence new patterns
        self.attractors: Dict[int, float] = {}  # node_id → strength
        
        # Resonance memory: which patterns resonate together
        self.resonance_memory: Dict[int, Set[int]] = {}  # node_id → set of resonant node_ids
        
        # Collapse memory: history of collapse events for learning
        self.collapse_memory: List[CollapseEvent] = []
        
        # Importance scores: how "important" is each node?
        self.importance: Dict[int, float] = {}
        
        # History
        self.state_history: List[FieldState] = []
        self.step_count = 0
    
    # =========================================================================
    # MEMORY OPERATIONS - Use mesh as substrate
    # =========================================================================
    
    def remember(self, node: MeshNode, importance: float = 0.5) -> None:
        """
        Mark a node as important in memory.
        
        High importance nodes:
        - Resist collapse
        - Attract similar patterns
        - Influence confidence of neighbors
        """
        self.importance[node.node_id] = importance
        
        # High importance → boost toward crystallization
        if importance > 0.8:
            node.confidence = min(1.0, node.confidence + importance * 0.2)
    
    def query(self, 
              embedding: torch.Tensor, 
              top_k: int = 5,
              threshold: float = 0.5) -> List[Tuple[MeshNode, float]]:
        """
        Query memory for patterns similar to embedding.
        
        Uses resonance field to find phase-aligned patterns.
        Prioritizes crystallized attractors.
        
        Returns: List of (node, similarity) pairs
        """
        results = []
        
        for node in self.mesh.nodes.values():
            if node.embedding is None:
                continue
            
            # Compute phase alignment (similarity)
            phase = self.resonance.compute_phase_alignment(
                embedding, node.embedding
            )
            
            if phase > threshold:
                # Boost score for crystallized nodes
                boost = 1.0
                if node.node_id in self.collapse.crystallized:
                    boost = PHI  # Golden ratio boost for attractors
                elif node.node_id in self.attractors:
                    boost = 1.0 + self.attractors[node.node_id]
                
                results.append((node, phase * boost))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def query_by_token(self, token_str: str, top_k: int = 5) -> List[Tuple[MeshNode, float]]:
        """
        Query memory for patterns containing a token.
        
        Returns nodes whose token matches, sorted by importance.
        """
        results = []
        
        for node in self.mesh.nodes.values():
            if token_str.lower() in node.token_str.lower():
                score = node.confidence
                
                # Boost for importance
                if node.node_id in self.importance:
                    score *= (1 + self.importance[node.node_id])
                
                # Boost for crystallization
                if node.node_id in self.collapse.crystallized:
                    score *= PHI
                
                # Boost for convergence
                score *= (1 + node.convergence_factor * 0.1)
                
                results.append((node, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_context_memory(self, 
                           context: List[MeshNode], 
                           depth: int = 3) -> List[MeshNode]:
        """
        Get memory relevant to a context sequence.
        
        Finds patterns that resonate with the context,
        prioritizing patterns that follow similar contexts.
        """
        if not context:
            return []
        
        # Build context embedding (average of context node embeddings)
        context_embs = [n.embedding for n in context if n.embedding is not None]
        if not context_embs:
            return []
        
        context_emb = torch.stack(context_embs).mean(dim=0)
        
        # Query for similar patterns
        results = self.query(context_emb, top_k=depth * 2)
        
        # Also check children of context nodes
        continuation_candidates = []
        for node in context[-depth:]:  # Last few context nodes
            for child, count in node.children.values():
                score = child.confidence * count
                if child.node_id in self.collapse.crystallized:
                    score *= PHI
                continuation_candidates.append((child, score))
        
        # Merge and dedupe
        seen = set()
        memory = []
        
        for node, score in sorted(
            results + continuation_candidates, 
            key=lambda x: x[1], 
            reverse=True
        ):
            if node.node_id not in seen:
                memory.append(node)
                seen.add(node.node_id)
                if len(memory) >= depth:
                    break
        
        return memory
    
    def store_pattern(self,
                      tokens: List[str],
                      embeddings: List[torch.Tensor],
                      source: str,
                      importance: float = 0.5) -> List[MeshNode]:
        """
        Store a sequence pattern in the mesh.
        
        Creates nodes if needed, strengthens existing paths,
        and marks with importance.
        """
        if not tokens or len(tokens) != len(embeddings):
            return []
        
        nodes = []
        parent = None
        context = []
        
        for i, (token, emb) in enumerate(zip(tokens, embeddings)):
            token_id = hash(token) % 1000000
            
            if i == 0:
                node = self.mesh.get_or_create_root(
                    token_id, token, emb, source
                )
            else:
                context.append(parent.token_id)
                node = self.mesh.get_or_create_context_node(
                    tuple(context), token_id, token, emb, source
                )
                parent.add_child(node)
            
            # Set importance
            self.remember(node, importance)
            
            nodes.append(node)
            parent = node
        
        return nodes
    
    def merge_resonant_nodes(self, threshold: float = 0.95) -> int:
        """
        Merge nodes that are nearly identical (very high resonance).
        
        This is a collapse operation - reduces redundancy.
        Returns number of merges performed.
        """
        merges = 0
        
        # Find very high resonance pairs
        for id1, id2, phase in self.resonance.resonance_pairs:
            if phase < threshold:
                continue
            
            node1 = self.mesh.nodes.get(id1)
            node2 = self.mesh.nodes.get(id2)
            
            if node1 is None or node2 is None:
                continue
            
            # Keep the more important/crystallized node
            keep, remove = (node1, node2) if (
                node1.confidence > node2.confidence or
                id1 in self.collapse.crystallized
            ) else (node2, node1)
            
            # Transfer properties to kept node
            keep.confidence = max(keep.confidence, remove.confidence)
            keep.sources.update(remove.sources)
            for pid, count in remove.incoming_paths.items():
                keep.incoming_paths[pid] = keep.incoming_paths.get(pid, 0) + count
            
            # Update importance
            if remove.node_id in self.importance:
                self.importance[keep.node_id] = max(
                    self.importance.get(keep.node_id, 0),
                    self.importance[remove.node_id]
                )
            
            # Note: Full removal would require updating all references
            # For now, we just mark as merged
            remove.confidence = 0.0  # Effectively dead
            
            merges += 1
        
        return merges
    
    def step(self) -> FieldState:
        """
        Perform one physics timestep.
        
        1. Update entropy
        2. Check conservation
        3. Find resonances and update resonance memory
        4. Handle collapse events
        5. Update attractors from crystallized nodes
        6. Apply attractor influence on nearby nodes
        
        Returns current FieldState.
        """
        self.step_count += 1
        
        # 1. Entropy update and collapse detection
        entropy, collapse_event = self.entropy_monitor.update(self.mesh)
        self.state.entropy = entropy
        self.state.entropy_momentum = self.entropy_monitor.entropy_momentum
        
        # 2. Conservation check
        residual = self.conservation.compute_mesh_conservation(self.mesh)
        self.state.conservation_residual = residual
        
        # Apply corrections if needed
        if residual > self.conservation.tolerance:
            self.conservation.apply_corrections(self.mesh)
        
        # 3. Resonance detection, effects, and MEMORY UPDATE
        self.resonance.find_resonant_pairs(self.mesh)
        self.resonance.apply_resonance_effects(self.mesh)
        self.state.resonance_strength = self.resonance.get_resonance_strength(self.mesh)
        
        # Build resonance memory
        for id1, id2, phase in self.resonance.resonance_pairs:
            if phase > 0.7:  # Strong resonance
                if id1 not in self.resonance_memory:
                    self.resonance_memory[id1] = set()
                if id2 not in self.resonance_memory:
                    self.resonance_memory[id2] = set()
                self.resonance_memory[id1].add(id2)
                self.resonance_memory[id2].add(id1)
        
        # 4. Curvature calculation
        total_curvature = 0.0
        for node in self.mesh.nodes.values():
            curvature = self.collapse.compute_curvature(node, self.mesh)
            total_curvature += curvature
        
        self.state.curvature = (
            total_curvature / len(self.mesh.nodes) 
            if self.mesh.nodes else 0.0
        )
        
        # 5. Handle collapse events and store in memory
        if collapse_event:
            self.collapse.execute_collapse(self.mesh, collapse_event)
            collapse_event.entropy_after = self.entropy_monitor.current_entropy
            self.collapse_memory.append(collapse_event)
        
        # 6. Update attractors from crystallized nodes
        for node_id in self.collapse.crystallized:
            if node_id not in self.attractors:
                node = self.mesh.nodes.get(node_id)
                if node:
                    # Attractor strength based on convergence and confidence
                    strength = node.confidence * (1 + node.convergence_factor * 0.1)
                    self.attractors[node_id] = strength
        
        # 7. Attractor influence: crystallized nodes pull similar patterns
        self._apply_attractor_influence()
        
        # 8. Update crystallization ratio
        crystallized_count = len(self.collapse.crystallized)
        self.state.crystallization_ratio = (
            crystallized_count / len(self.mesh.nodes)
            if self.mesh.nodes else 0.0
        )
        
        # Record history
        self.state_history.append(FieldState(
            entropy=self.state.entropy,
            entropy_momentum=self.state.entropy_momentum,
            curvature=self.state.curvature,
            resonance_strength=self.state.resonance_strength,
            conservation_residual=self.state.conservation_residual,
            crystallization_ratio=self.state.crystallization_ratio
        ))
        
        return self.state
    
    def _apply_attractor_influence(self) -> None:
        """
        Crystallized attractors influence nearby nodes.
        
        - Similar embeddings get pulled toward attractor
        - Confidence of nearby nodes increases
        - Creates "gravity wells" in embedding space
        """
        if not self.attractors:
            return
        
        for attractor_id, strength in self.attractors.items():
            attractor = self.mesh.nodes.get(attractor_id)
            if attractor is None or attractor.embedding is None:
                continue
            
            # Find nodes that resonate with this attractor
            resonant_set = self.resonance_memory.get(attractor_id, set())
            
            for node_id in resonant_set:
                node = self.mesh.nodes.get(node_id)
                if node is None or node.node_id in self.collapse.crystallized:
                    continue
                
                # Confidence boost from attractor proximity
                boost = strength * PHI_INV * 0.1  # Gentle influence
                node.confidence = min(1.0, node.confidence + boost)
                
                # Embedding drift toward attractor (very subtle)
                if node.embedding is not None:
                    drift = (attractor.embedding - node.embedding) * 0.01 * strength
                    node.embedding = node.embedding + drift
    
    def force_collapse(self, collapse_type: CollapseType) -> CollapseEvent:
        """
        Force a collapse event of the specified type.
        
        Useful for testing or manual intervention.
        """
        event = CollapseEvent(
            collapse_type=collapse_type,
            timestamp=time.time(),
            node_ids=[],
            entropy_before=self.state.entropy,
            entropy_after=0.0,
            magnitude=1.0
        )
        
        self.collapse.execute_collapse(self.mesh, event)
        event.entropy_after = self.entropy_monitor.current_entropy
        
        return event
    
    def crystallize_all_stable(self) -> int:
        """
        Crystallize all nodes that meet stability criteria.
        
        Returns number of newly crystallized nodes.
        """
        new_crystallized = 0
        
        for node in self.mesh.nodes.values():
            if node.node_id not in self.collapse.crystallized:
                if self.collapse.check_crystallization(node):
                    self.collapse.crystallized.add(node.node_id)
                    node.confidence = 1.0
                    new_crystallized += 1
        
        return new_crystallized
    
    def get_attractors(self) -> List[MeshNode]:
        """
        Get all attractor nodes (crystallized + high convergence).
        
        Attractors are stable patterns that "attract" other patterns.
        """
        attractors = []
        
        for node in self.mesh.nodes.values():
            is_crystallized = node.node_id in self.collapse.crystallized
            is_convergent = node.convergence_factor >= 3
            is_confident = node.confidence >= 0.8
            
            if is_crystallized or (is_convergent and is_confident):
                attractors.append(node)
        
        # Sort by importance (convergence * confidence)
        attractors.sort(
            key=lambda n: n.convergence_factor * n.confidence,
            reverse=True
        )
        
        return attractors
    
    def report(self) -> str:
        """Generate human-readable state report."""
        lines = [
            "+=======================================+",
            "|     PHYSICS MESH STATE REPORT         |",
            "+=======================================+",
            f"| Mesh Nodes: {len(self.mesh.nodes):>6}                    |",
            f"| Crystallized: {len(self.collapse.crystallized):>5} ({self.state.crystallization_ratio*100:.1f}%)           |",
            f"| Attractors: {len(self.attractors):>6}                    |",
            f"| Resonance Memory: {len(self.resonance_memory):>4} pairs          |",
            "+---------------------------------------+",
            f"| Entropy:      {self.state.entropy:>7.4f}                |",
            f"| Momentum:     {self.state.entropy_momentum:>7.4f}                |",
            f"| Curvature:    {self.state.curvature:>7.4f}                |",
            f"| Resonance:    {self.state.resonance_strength:>7.4f}                |",
            f"| Conservation: {self.state.conservation_residual:>7.4f}                |",
            "+---------------------------------------+",
            f"| Collapse Pressure: {self.state.collapse_pressure:>5.2f}              |",
            f"| Stable: {'YES' if self.state.is_stable else 'NO ':>3}                            |",
            "+=======================================+"
        ]
        return "\n".join(lines)
    
    # =========================================================================
    # PREDICTION - Use memory to predict next tokens
    # =========================================================================
    
    def predict_next(self, 
                     context: List[MeshNode], 
                     top_k: int = 5) -> List[Tuple[MeshNode, float]]:
        """
        Predict next tokens based on context and memory.
        
        Uses:
        1. Direct children of last context node
        2. Attractor influence
        3. Resonance memory for similar contexts
        
        Returns: List of (node, probability) pairs
        """
        if not context:
            return []
        
        candidates: Dict[int, float] = {}
        last_node = context[-1]
        
        # 1. Direct children (primary signal)
        if last_node.children:
            total_count = sum(c for (_, c) in last_node.children.values())
            for child, count in last_node.children.values():
                prob = count / total_count
                # Boost for crystallized
                if child.node_id in self.collapse.crystallized:
                    prob *= PHI
                candidates[child.node_id] = candidates.get(child.node_id, 0) + prob
        
        # 2. Attractor influence
        for attractor_id, strength in self.attractors.items():
            attractor = self.mesh.nodes.get(attractor_id)
            if attractor is None or attractor.embedding is None:
                continue
            
            # Check if attractor resonates with context
            if last_node.embedding is not None:
                phase = self.resonance.compute_phase_alignment(
                    last_node.embedding, attractor.embedding
                )
                if phase > 0.5:
                    # Add attractor's children as candidates
                    for child, count in attractor.children.values():
                        bonus = phase * strength * count * 0.1
                        candidates[child.node_id] = candidates.get(child.node_id, 0) + bonus
        
        # 3. Resonance memory - what followed similar contexts?
        if last_node.node_id in self.resonance_memory:
            for resonant_id in self.resonance_memory[last_node.node_id]:
                resonant = self.mesh.nodes.get(resonant_id)
                if resonant and resonant.children:
                    for child, count in resonant.children.values():
                        bonus = count * 0.05  # Smaller weight for indirect signal
                        candidates[child.node_id] = candidates.get(child.node_id, 0) + bonus
        
        # Convert to node-probability pairs
        results = []
        for node_id, score in candidates.items():
            node = self.mesh.nodes.get(node_id)
            if node:
                results.append((node, score))
        
        # Normalize scores
        total_score = sum(s for _, s in results) if results else 1.0
        results = [(n, s / total_score) for n, s in results]
        
        # Sort by probability
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def learn_from_sequence(self,
                           tokens: List[str],
                           embeddings: List[torch.Tensor],
                           source: str,
                           confirmed: bool = False) -> None:
        """
        Learn a sequence, using attractors to influence storage.
        
        If sequence passes through existing attractors, those paths
        get reinforced. New patterns may crystallize.
        
        Args:
            tokens: Token strings
            embeddings: Token embeddings
            source: Source identifier
            confirmed: If True, treat as ground truth (high importance)
        """
        if not tokens:
            return
        
        importance = 0.9 if confirmed else 0.5
        
        # Store pattern
        nodes = self.store_pattern(tokens, embeddings, source, importance)
        
        # Check for attractor proximity - if we passed near an attractor,
        # boost both the attractor and our path
        for node in nodes:
            if node.embedding is None:
                continue
            
            # Find nearby attractors
            for attractor_id, strength in self.attractors.items():
                attractor = self.mesh.nodes.get(attractor_id)
                if attractor is None or attractor.embedding is None:
                    continue
                
                phase = self.resonance.compute_phase_alignment(
                    node.embedding, attractor.embedding
                )
                
                if phase > 0.7:
                    # Mutual reinforcement
                    node.confidence = min(1.0, node.confidence + phase * 0.1)
                    self.attractors[attractor_id] = min(2.0, strength + phase * 0.05)
        
        # Run physics to integrate new patterns
        self.step()
        
        # If confirmed, try to crystallize
        if confirmed:
            for node in nodes:
                if node.confidence > 0.8:
                    self.remember(node, 0.95)

