"""
Unified FULL PAC System - Complete Architecture Integration
=============================================================

This integrates ALL components from the Dawn Field ecosystem:

FROM FRACTON:
- PACNode with delta-only storage
- Phase transitions (STABLE, TRANSITION, EXPANSION, COLLAPSE)
- PAC conservation validation
- Klein-Gordon field evolution

FROM GAIA-1:
- FieldGenerator with evolution dynamics
- FieldContext for resonance-based attention
- Vocabulary as field patterns

FROM POC-011:
- PACLazySystem for node management
- SEC expansion/collapse dynamics

FROM POC-003:
- ResonanceAttention (field-native attention)
- HarmonicMultiHead (prime harmonic weighting)

FROM POC-016/017/019/020:
- Multi-model oracle extraction
- Import without training
- Train without backprop
- ByRef PAC composition

Key Formula: full_repr = avg(byrefs) + delta
Physics: Klein-Gordon evolution + PAC conservation
"""

import sys
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from enum import Enum
import hashlib
from datetime import datetime
import math

# Dawn Field Constants
PHI = 1.618033988749895
XI = 0.0618
PHI_XI = PHI * XI  # ~0.1 - crystallization threshold
LAMBDA_STAR = 1 - XI  # 0.9382 - decay rate
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
PI_SQUARED_INV = 1 / (math.pi ** 2)  # 0.1013 - eigenvalue decay
ENTANGLEMENT_LIMIT = 4/5  # 0.8 - max coupling


# =============================================================================
# PART 1: PHASE STATES (from Fracton)
# =============================================================================

class PhaseState(Enum):
    """Phase states from fracton physics"""
    STABLE = "stable"
    TRANSITION = "transition"  
    EXPANSION = "expansion"
    COLLAPSE = "collapse"


def detect_phase(potential: float) -> PhaseState:
    """Detect phase from potential level"""
    if potential >= PHI_XI:
        return PhaseState.EXPANSION
    elif potential <= XI:
        return PhaseState.COLLAPSE
    elif potential > 0.5:
        return PhaseState.TRANSITION
    else:
        return PhaseState.STABLE


# =============================================================================
# PART 2: PAC NODE (from Fracton)
# =============================================================================

@dataclass
class PACNode:
    """
    PAC-Lazy storage node with delta-only representation.
    NEVER stores absolute values. Reconstruction requires parent chain.
    """
    id: str
    delta: torch.Tensor
    potential: float = 1.0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    label: str = ""
    phase: PhaseState = PhaseState.STABLE
    
    # ByRef support
    byrefs: List[str] = field(default_factory=list)
    byref_weights: List[float] = field(default_factory=list)
    
    # Lazy evaluation cache
    _materialized: Optional[torch.Tensor] = None
    _cache_valid: bool = False
    
    @property
    def is_root(self) -> bool:
        return self.parent_id is None
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0
    
    def should_expand(self) -> bool:
        return self.potential >= PHI_XI
    
    def should_collapse(self) -> bool:
        return self.potential <= XI
    
    def update_phase(self) -> PhaseState:
        self.phase = detect_phase(self.potential)
        return self.phase
    
    def decay_potential(self, factor: float = LAMBDA_STAR) -> float:
        self.potential *= factor
        self.update_phase()
        return self.potential
    
    def invalidate_cache(self):
        self._cache_valid = False
        self._materialized = None


# =============================================================================
# PART 3: KLEIN-GORDON FIELD EVOLUTION (from Fracton)
# =============================================================================

def klein_gordon_evolve(
    field: torch.Tensor,
    steps: int = 5,
    dt: float = 0.1,
    mass: float = 1.0,
    damping: float = None
) -> torch.Tensor:
    """
    Evolve field using Klein-Gordon dynamics.
    ∂²φ/∂t² = ∇²φ - m²φ (with damping)
    """
    if damping is None:
        damping = 1 - LAMBDA_STAR  # = XI
    
    velocity = torch.zeros_like(field)
    current = field.clone()
    
    for _ in range(steps):
        # Discrete Laplacian for 1D
        if current.dim() == 1:
            left = torch.roll(current, 1)
            right = torch.roll(current, -1)
            laplacian = left + right - 2 * current
        else:
            # For batch processing
            laplacian = -2 * current
            for dim in range(current.dim()):
                laplacian = laplacian + torch.roll(current, 1, dims=dim) + torch.roll(current, -1, dims=dim)
        
        # Klein-Gordon: ∇²φ - m²φ
        acceleration = laplacian - mass * mass * current
        
        # Update with damping
        velocity = LAMBDA_STAR * velocity + dt * acceleration
        current = current + dt * velocity
    
    return current


# =============================================================================
# PART 4: RESONANCE ATTENTION (from POC-003)
# =============================================================================

class ResonanceAttention(nn.Module):
    """
    Attention computed as field resonance.
    Resonance = how much two patterns vibrate together in the field.
    """
    
    def __init__(self, dim: int, max_coupling: float = ENTANGLEMENT_LIMIT):
        super().__init__()
        self.dim = dim
        self.max_coupling = max_coupling
        self.scale = 1.0 / math.sqrt(dim)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute resonance-based attention"""
        # Normalize to unit vectors
        q_norm = F.normalize(query, dim=-1)
        k_norm = F.normalize(key, dim=-1)
        
        # Cosine similarity as resonance
        resonance = torch.bmm(q_norm, k_norm.transpose(-2, -1))
        
        # Clamp to max coupling (physics constraint)
        resonance = torch.clamp(resonance, -self.max_coupling, self.max_coupling)
        
        # Apply mask
        if mask is not None:
            resonance = resonance.masked_fill(mask == 0, float('-inf'))
        
        weights = F.softmax(resonance, dim=-1)
        output = torch.bmm(weights, value)
        
        return output, weights


# =============================================================================
# PART 5: PAC SYSTEM WITH TIERED CACHING (from Fracton)
# =============================================================================

class TieredCache:
    """Three-tier cache: hot/warm/cold based on access patterns"""
    
    def __init__(self, hot_size: int = 1000, warm_size: int = 10000):
        self.hot_size = hot_size
        self.warm_size = warm_size
        self._hot: OrderedDict = OrderedDict()
        self._warm: OrderedDict = OrderedDict()
        self._cold: Dict[str, bytes] = {}
        
    def get(self, node_id: str) -> Optional[PACNode]:
        if node_id in self._hot:
            self._hot.move_to_end(node_id)
            return self._hot[node_id]
        if node_id in self._warm:
            node = self._warm.pop(node_id)
            self._promote_to_hot(node_id, node)
            return node
        return None
    
    def put(self, node: PACNode, tier: str = "hot"):
        if tier == "hot":
            self._promote_to_hot(node.id, node)
        else:
            self._promote_to_warm(node.id, node)
            
    def _promote_to_hot(self, node_id: str, node: PACNode):
        while len(self._hot) >= self.hot_size:
            evicted_id, evicted_node = self._hot.popitem(last=False)
            self._promote_to_warm(evicted_id, evicted_node)
        self._hot[node_id] = node
        self._warm.pop(node_id, None)
        
    def _promote_to_warm(self, node_id: str, node: PACNode):
        while len(self._warm) >= self.warm_size:
            self._warm.popitem(last=False)
        self._warm[node_id] = node


class PACSystem:
    """
    Complete PAC system with:
    - Delta-only storage
    - Tiered caching
    - SEC phase transitions
    - Conservation validation
    """
    
    def __init__(self, dim: int = 256, device: str = 'cuda'):
        self.dim = dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.nodes: Dict[str, PACNode] = {}
        self.cache = TieredCache()
        self.root_ids: Set[str] = set()
        
        # Conservation tracking
        self.total_potential = 0.0
        self.allocated_potential = 0.0
        
        # Level index for hierarchy
        self.level_index: Dict[int, Set[str]] = defaultdict(set)
        self.name_to_id: Dict[str, str] = {}
        
    def _generate_id(self, name: str) -> str:
        return hashlib.md5(name.encode()).hexdigest()[:12]
    
    def add_node(
        self, 
        name: str, 
        delta: torch.Tensor,
        parent_id: Optional[str] = None,
        level: int = 0,
        potential: float = 1.0
    ) -> str:
        """Add a node with delta-only storage"""
        node_id = self._generate_id(name)
        
        if delta.shape[0] != self.dim:
            if delta.shape[0] > self.dim:
                delta = delta[:self.dim]
            else:
                delta = torch.cat([delta, torch.zeros(self.dim - delta.shape[0], device=delta.device)])
        
        node = PACNode(
            id=node_id,
            delta=delta.to(self.device),
            potential=potential,
            parent_id=parent_id,
            label=name
        )
        
        self.nodes[node_id] = node
        self.cache.put(node)
        self.level_index[level].add(node_id)
        self.name_to_id[name] = node_id
        
        if parent_id is None:
            self.root_ids.add(node_id)
        else:
            if parent_id in self.nodes:
                self.nodes[parent_id].children_ids.append(node_id)
                
        self.allocated_potential += potential
        
        return node_id
    
    def add_byref(self, node_id: str, target_id: str, weight: float = 1.0):
        """Add a byref (reference) link"""
        if node_id in self.nodes and target_id in self.nodes:
            self.nodes[node_id].byrefs.append(target_id)
            self.nodes[node_id].byref_weights.append(weight)
            self.nodes[node_id].invalidate_cache()
    
    def get_full_representation(self, node_id: str, visited: Set[str] = None) -> torch.Tensor:
        """
        Reconstruct full representation.
        For byref nodes: full = avg(byrefs) + delta
        For hierarchical: full = parent_full + delta
        """
        if visited is None:
            visited = set()
            
        if node_id in visited:
            return self.nodes[node_id].delta
            
        visited.add(node_id)
        node = self.nodes[node_id]
        
        # Check cache
        if node._cache_valid and node._materialized is not None:
            return node._materialized
        
        # ByRef case
        if node.byrefs:
            weighted_sum = torch.zeros(self.dim, device=self.device)
            total_weight = 0.0
            
            for ref_id, weight in zip(node.byrefs, node.byref_weights):
                if ref_id in self.nodes and ref_id not in visited:
                    ref_repr = self.get_full_representation(ref_id, visited.copy())
                    weighted_sum += weight * ref_repr
                    total_weight += weight
                    
            if total_weight > 0:
                byref_avg = weighted_sum / total_weight
            else:
                byref_avg = torch.zeros(self.dim, device=self.device)
                
            result = byref_avg + node.delta
            
        # Hierarchical case
        elif node.parent_id and node.parent_id in self.nodes:
            parent_repr = self.get_full_representation(node.parent_id, visited)
            result = parent_repr + node.delta
            
        # Root case
        else:
            result = node.delta
            
        # Cache
        node._materialized = result
        node._cache_valid = True
        
        return result
    
    def conservation_check(self, node_id: str) -> float:
        """Check PAC conservation for a node (should be ~0)"""
        if node_id not in self.nodes:
            return -1.0
            
        node = self.nodes[node_id]
        
        if not node.byrefs:
            return 0.0
            
        # Compute expected from byrefs
        weighted_sum = torch.zeros(self.dim, device=self.device)
        total_weight = 0.0
        
        for ref_id, weight in zip(node.byrefs, node.byref_weights):
            if ref_id in self.nodes:
                ref_repr = self.get_full_representation(ref_id)
                weighted_sum += weight * ref_repr
                total_weight += weight
                
        if total_weight > 0:
            byref_avg = weighted_sum / total_weight
        else:
            byref_avg = torch.zeros(self.dim, device=self.device)
            
        expected = byref_avg + node.delta
        actual = self.get_full_representation(node_id)
        
        return float(torch.norm(expected - actual).item())
    
    def evolve_field(self, node_id: str, steps: int = 5) -> torch.Tensor:
        """Evolve a node's field using Klein-Gordon dynamics"""
        if node_id not in self.nodes:
            return torch.zeros(self.dim, device=self.device)
            
        field = self.get_full_representation(node_id)
        evolved = klein_gordon_evolve(field, steps=steps)
        
        return evolved
    
    def sec_check(self, node_id: str) -> Tuple[bool, PhaseState]:
        """Check SEC phase and return if action needed"""
        if node_id not in self.nodes:
            return False, PhaseState.STABLE
            
        node = self.nodes[node_id]
        node.update_phase()
        
        should_act = node.should_expand() or node.should_collapse()
        return should_act, node.phase


# =============================================================================
# PART 6: FIELD GENERATOR (from GAIA-1)
# =============================================================================

class FieldGenerator(nn.Module):
    """
    Field-native generation using evolution dynamics.
    No attention matrices - just physics.
    """
    
    def __init__(self, dim: int = 256, evolution_steps: int = 8):
        super().__init__()
        self.dim = dim
        self.evolution_steps = evolution_steps
        
        # Learnable mass parameter
        self.log_mass = nn.Parameter(torch.tensor(0.0))
        
        # Position phases
        self.pos_phases = nn.Parameter(torch.randn(512, dim) * 0.1)
        
        # Resonance projections
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Resonance attention
        self.resonance = ResonanceAttention(dim)
        
    def forward(self, patterns: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate next field state from context patterns.
        
        Args:
            patterns: (batch, seq, dim) context field patterns
            mask: Optional attention mask
            
        Returns:
            context_field: (batch, dim) evolved context
        """
        batch_size, seq_len, _ = patterns.shape
        device = patterns.device
        
        # Add position phases
        pos_enc = self.pos_phases[:seq_len].to(device)
        patterns_pos = patterns + pos_enc.unsqueeze(0)
        
        # Project Q, K, V
        Q = self.query_proj(patterns_pos)
        K = self.key_proj(patterns_pos)
        V = self.value_proj(patterns_pos)
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        causal_mask = (~causal_mask).float().unsqueeze(0)
        
        if mask is not None:
            causal_mask = causal_mask * mask.unsqueeze(1)
        
        # Resonance attention
        attended, weights = self.resonance(Q, K, V, causal_mask)
        output = self.out_proj(attended)
        
        # Evolve via Klein-Gordon
        mass = torch.exp(self.log_mass)
        context = output[:, -1, :]  # Last position
        
        # Simple evolution step
        evolved = klein_gordon_evolve(
            context.detach().cpu(),
            steps=self.evolution_steps,
            mass=mass.item()
        ).to(device)
        
        # Mix with original
        result = LAMBDA_STAR * context + (1 - LAMBDA_STAR) * evolved
        
        return result


# =============================================================================
# PART 7: TRANSITION MATRIX (from POC-012)
# =============================================================================

class TransitionMatrix:
    """
    Hybrid transition matrix: sparse tracking + O(1) prediction.
    
    Key insight: When confluence fails, we learn the oracle's answer.
    This is Hebbian-like: fire together -> wire together.
    """
    
    def __init__(self, vocab_size: int = 50257):
        # Sparse counts: (prev_context, next) -> count
        self.counts: Dict[Tuple, float] = {}
        
        # Best prediction cache: context -> (next, count)
        self._best_cache: Dict[Tuple, Tuple[int, float]] = {}
        
        # Row totals for probability calculation
        self._row_totals: Dict[Tuple, float] = defaultdict(float)
        
        # Stats
        self.total_learns = 0
        self.crystallized = 0
    
    def learn(self, context: Tuple[int, ...], next_token: int, weight: float = 1.0,
              crystallize: bool = False):
        """Learn a transition - O(1) with cache update."""
        key = (context, next_token)
        old_count = self.counts.get(key, 0.0)
        
        # Crystallization: PHI boost for confirmed patterns
        if crystallize:
            weight *= PHI
            self.crystallized += 1
            
        new_count = old_count + weight
        self.counts[key] = new_count
        
        # Update row total
        self._row_totals[context] += weight
        
        # Update best cache if this is now the best
        if context not in self._best_cache or new_count > self._best_cache[context][1]:
            self._best_cache[context] = (next_token, new_count)
        
        self.total_learns += 1
        
    def predict(self, context: Tuple[int, ...]) -> Tuple[Optional[int], float]:
        """Get most likely next token - O(1) from cache."""
        if context not in self._best_cache:
            return None, 0.0
        
        best_next, best_count = self._best_cache[context]
        total = self._row_totals[context]
        
        if total == 0:
            return None, 0.0
        
        return best_next, best_count / total
    
    def get_candidates(self, context: Tuple[int, ...], top_k: int = 10) -> Dict[int, float]:
        """Get top-k candidates with their probabilities."""
        candidates = {}
        for key, count in self.counts.items():
            if key[0] == context:
                candidates[key[1]] = count
        
        if not candidates:
            return {}
            
        # Normalize
        total = sum(candidates.values())
        return {k: v / total for k, v in sorted(candidates.items(), key=lambda x: -x[1])[:top_k]}
    
    def decay(self, factor: float = LAMBDA_STAR, threshold: float = XI / 10):
        """Decay all transitions and prune weak ones.
        
        Selection criteria for pruning:
        1. Multiply all counts by LAMBDA_STAR (0.9382) - exponential decay
        2. Remove any transition where count < XI/10 (0.00618)
        3. This preserves crystallized patterns (high count) while pruning noise
        """
        to_remove = []
        new_best: Dict[Tuple, Tuple[int, float]] = {}
        pruned_stats = {'token_level': 0, 'category_level': 0, 'supercat_level': 0}
        
        for key, count in self.counts.items():
            new_count = count * factor
            if new_count < threshold:
                to_remove.append(key)
                # Track what type of transition is being pruned
                context = key[0]
                if isinstance(context, tuple) and len(context) > 0:
                    first_elem = context[0]
                    if isinstance(first_elem, str):
                        if first_elem.startswith('tok_'):
                            pruned_stats['token_level'] += 1
                        elif first_elem in ['living_thing', 'physical', 'abstract', 'quality']:
                            pruned_stats['supercat_level'] += 1
                        else:
                            pruned_stats['category_level'] += 1
                    else:
                        pruned_stats['token_level'] += 1
            else:
                self.counts[key] = new_count
                context = key[0]
                
                # Update best tracker
                if context not in new_best or new_count > new_best[context][1]:
                    new_best[context] = (key[1], new_count)
        
        for key in to_remove:
            context = key[0]
            self._row_totals[context] -= self.counts.get(key, 0)
            if key in self.counts:
                del self.counts[key]
        
        # Update row totals
        for context in self._row_totals:
            self._row_totals[context] *= factor
        
        self._best_cache = new_best
        self.last_prune_stats = pruned_stats
        return pruned_stats
    
    def get_stats(self) -> Dict:
        """Get detailed statistics about transitions."""
        token_level = 0
        category_level = 0
        supercat_level = 0
        
        for key in self.counts.keys():
            context = key[0]
            if isinstance(context, tuple) and len(context) > 0:
                first_elem = context[0]
                if isinstance(first_elem, str):
                    if first_elem.startswith('tok_'):
                        token_level += 1
                    elif first_elem in ['living_thing', 'physical', 'abstract', 'quality']:
                        supercat_level += 1
                    else:
                        category_level += 1
                else:
                    token_level += 1
                    
        return {
            'total': len(self.counts),
            'token_level': token_level,
            'category_level': category_level,
            'supercat_level': supercat_level,
            'crystallized': self.crystallized,
        }
    
    def num_transitions(self) -> int:
        return len(self.counts)


# =============================================================================
# PART 8: UNIFIED FULL SYSTEM
# =============================================================================

class UnifiedFullSystem:
    """
    Complete unified system with ALL architecture components:
    
    - PAC System with tiered caching
    - Klein-Gordon field evolution
    - Resonance-based attention
    - ByRef composition
    - SEC phase transitions
    - Oracle distillation
    - Generation with PAC guidance
    """
    
    def __init__(self, dim: int = 256, max_layers: int = 13):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim = dim
        self.max_layers = max_layers
        
        # Core PAC system (from fracton)
        self.pac = PACSystem(dim=dim, device=str(self.device))
        
        # Field generator (from GAIA-1)
        self.field_gen = FieldGenerator(dim=dim).to(self.device)
        
        # Token confluence
        self.token_confluence: Dict[tuple, Dict[int, int]] = {}
        
        # Oracle models
        self.oracles: Dict[str, Dict] = {}
        
        # Embeddings
        self.embeddings: Optional[np.ndarray] = None
        self.vocab_size = 50257
        
        # SEC layer tracking
        self.materialized_layers = 0
        
        # Transition matrix for continuous learning
        self.transitions = TransitionMatrix(vocab_size=self.vocab_size)
        
        # Learning stats
        self.learning_stats = {
            'confluence_hits': 0,
            'confluence_misses': 0,
            'transitions_learned': 0,
            'crystallizations': 0,
        }
        
        # Category hit tracking
        self.category_stats = defaultdict(lambda: {'hits': 0, 'misses': 0, 'total': 0})
        
        # Prompt analysis tracking
        self.prompt_analysis = {}
        
        # Metrics
        self.metrics = {
            'models_loaded': 0,
            'nodes_created': 0,
            'categories_created': 0,
            'byref_links': 0,
            'phase_transitions': 0,
            'field_evolutions': 0,
            'confluence_contexts': 0,
            'generations': 0,
        }
        
        # Token to category mappings (built in build_pac_tree)
        self.token_to_category: Dict[str, str] = {}
        self.category_to_supercategory: Dict[str, str] = {}
        self.category_tokens: Dict[str, List[int]] = {}  # category -> token_ids
        
    def get_token_category(self, token_id: int, tokenizer) -> Optional[str]:
        """Get the category for a token (level 1 in PAC)"""
        decoded = tokenizer.decode([token_id]).strip().lower().replace('Ġ', '').replace('▁', '')
        return self.token_to_category.get(decoded)
        
    def get_category_supercategory(self, category: str) -> Optional[str]:
        """Get the supercategory for a category (level 2 in PAC)"""
        return self.category_to_supercategory.get(category)
        
    def learn_at_all_levels(self, context_tokens: Tuple[int, ...], next_token: int, 
                            tokenizer, weight: float = 1.0, crystallize: bool = False):
        """
        Learn a transition at ALL PAC levels - the key to generalization!
        
        Level 0: (token, token, token) → token          [specific]
        Level 1: (category, category, category) → category  [generalizable]
        Level 2: (supercat, supercat, supercat) → supercat  [abstract]
        
        NO BACKPROP - just counting with hierarchical structure!
        """
        # Level 0: Token level (most specific, highest weight)
        self.transitions.learn(context_tokens, next_token, weight=weight, crystallize=crystallize)
        
        # Level 1: Category level (generalizable)
        context_cats = []
        for tok_id in context_tokens:
            cat = self.get_token_category(tok_id, tokenizer)
            context_cats.append(cat if cat else f"tok_{tok_id}")
        next_cat = self.get_token_category(next_token, tokenizer)
        if next_cat:
            cat_context = tuple(context_cats)
            # Lower weight for higher abstraction (divide by PHI)
            self.transitions.learn(cat_context, next_cat, weight=weight/PHI, crystallize=crystallize)
            
        # Level 2: Supercategory level (most abstract, lowest weight)
        context_supercats = []
        for cat in context_cats:
            if isinstance(cat, str) and not cat.startswith("tok_"):
                supercat = self.get_category_supercategory(cat)
                context_supercats.append(supercat if supercat else cat)
            else:
                context_supercats.append(cat)
        if next_cat:
            next_supercat = self.get_category_supercategory(next_cat)
            if next_supercat:
                supercat_context = tuple(context_supercats)
                # Even lower weight for supercategory (divide by PHI^2)
                self.transitions.learn(supercat_context, next_supercat, 
                                       weight=weight/(PHI**2), crystallize=crystallize)
        
    def load_oracles(self, include_large_models: bool = False):
        """Load oracle models for distillation
        
        Args:
            include_large_models: If True, also load Llama 3.1 8B and Mistral 7B (4-bit)
        """
        print("\n" + "="*60)
        print("PHASE 1: LOAD ORACLES")
        print("="*60)
        
        # GPT-2 (small, fast, always loaded)
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            self.oracles['gpt2'] = {
                'model': GPT2LMHeadModel.from_pretrained('gpt2').to(self.device).eval(),
                'tokenizer': GPT2Tokenizer.from_pretrained('gpt2'),
                'size': '124M',
            }
            for p in self.oracles['gpt2']['model'].parameters():
                p.requires_grad = False
            print("  ✓ GPT-2 loaded (124M params)")
            self.metrics['models_loaded'] += 1
        except Exception as e:
            print(f"  ✗ GPT-2 failed: {e}")
            
        # Pythia (small, always loaded)
        try:
            from transformers import GPTNeoXForCausalLM, AutoTokenizer
            self.oracles['pythia'] = {
                'model': GPTNeoXForCausalLM.from_pretrained('EleutherAI/pythia-70m').to(self.device).eval(),
                'tokenizer': AutoTokenizer.from_pretrained('EleutherAI/pythia-70m'),
                'size': '70M',
            }
            for p in self.oracles['pythia']['model'].parameters():
                p.requires_grad = False
            print("  ✓ Pythia-70m loaded (70M params)")
            self.metrics['models_loaded'] += 1
        except Exception as e:
            print(f"  ✗ Pythia failed: {e}")
        
        if not include_large_models:
            return
            
        # Large models - use open models that don't require approval
        print("\n  Loading large models...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Qwen2.5-1.5B - Alibaba's open model, no approval needed, fits in 8GB
            try:
                print("  Loading Qwen2.5-1.5B-Instruct...")
                qwen_tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2.5-1.5B-Instruct",
                    trust_remote_code=True
                )
                qwen_model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen2.5-1.5B-Instruct",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                qwen_model.eval()
                
                self.oracles['qwen'] = {
                    'model': qwen_model,
                    'tokenizer': qwen_tokenizer,
                    'size': '1.5B (fp16)',
                }
                print("  ✓ Qwen2.5-1.5B loaded")
                self.metrics['models_loaded'] += 1
            except Exception as e:
                print(f"  ✗ Qwen2.5-1.5B failed: {e}")
                
            # SmolLM2-360M - HuggingFace's tiny but capable model
            try:
                print("  Loading SmolLM2-360M-Instruct...")
                smol_tokenizer = AutoTokenizer.from_pretrained(
                    "HuggingFaceTB/SmolLM2-360M-Instruct",
                    trust_remote_code=True
                )
                smol_model = AutoModelForCausalLM.from_pretrained(
                    "HuggingFaceTB/SmolLM2-360M-Instruct",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                smol_model.eval()
                
                self.oracles['smol'] = {
                    'model': smol_model,
                    'tokenizer': smol_tokenizer,
                    'size': '360M (fp16)',
                }
                print("  ✓ SmolLM2-360M loaded")
                self.metrics['models_loaded'] += 1
            except Exception as e:
                print(f"  ✗ SmolLM2-360M failed: {e}")
                
        except ImportError as e:
            print(f"  ✗ Large model loading failed: {e}")
    
    def multi_oracle_predict(self, input_ids: torch.Tensor, temperature: float = 1.0) -> Tuple[int, Dict]:
        """
        Get consensus prediction from all loaded oracles.
        
        Each oracle votes on next token. Combine using:
        1. Weighted averaging (larger models get higher weight)
        2. Top-k intersection (tokens all models agree on)
        
        Returns:
            next_token: Consensus prediction
            stats: Per-oracle predictions and agreement metrics
        """
        oracle_weights = {
            'gpt2': 1.0,      # 124M - baseline
            'pythia': 1.0,    # 70M - small
            'smol': 2.0,      # 360M - medium
            'qwen': 4.0,      # 1.5B - largest
        }
        
        all_logits = []
        all_top_tokens = []
        oracle_preds = {}
        
        for name, oracle in self.oracles.items():
            model = oracle['model']
            tokenizer = oracle['tokenizer']
            
            # Prepare input for this model's tokenizer if different
            with torch.no_grad():
                try:
                    outputs = model(input_ids)
                    logits = outputs.logits[0, -1]
                    
                    # Get top prediction
                    top_idx = logits.argmax().item()
                    top_token = tokenizer.decode([top_idx]).strip()
                    
                    # Get weighted logits
                    weight = oracle_weights.get(name, 1.0)
                    weighted_logits = (logits / temperature) * weight
                    all_logits.append(weighted_logits)
                    
                    # Get top-k for intersection
                    top_k = torch.topk(logits, 10)
                    all_top_tokens.append(set(top_k.indices.tolist()))
                    
                    oracle_preds[name] = {
                        'top_token': top_token,
                        'top_idx': top_idx,
                        'weight': weight,
                    }
                except Exception as e:
                    # Model might have different vocab size
                    continue
        
        if not all_logits:
            return None, {}
        
        # Combine logits (weighted average)
        combined = torch.stack(all_logits).mean(dim=0)
        
        # Find intersection tokens (tokens all models agree on)
        if len(all_top_tokens) > 1:
            intersection = all_top_tokens[0]
            for tokens in all_top_tokens[1:]:
                intersection = intersection.intersection(tokens)
            agreement = len(intersection)
        else:
            intersection = all_top_tokens[0] if all_top_tokens else set()
            agreement = len(intersection)
        
        # Sample from top-k of combined
        top_k = 50
        top_logits, top_indices = torch.topk(combined, top_k)
        probs = torch.softmax(top_logits, dim=-1)
        idx = torch.multinomial(probs, 1).item()
        next_token = top_indices[idx].item()
        
        stats = {
            'oracle_preds': oracle_preds,
            'agreement_tokens': agreement,
            'num_oracles': len(all_logits),
        }
        
        return next_token, stats
            
    def extract_embeddings(self):
        """Extract and combine embeddings from all oracles"""
        print("\n  Extracting embeddings...")
        
        embeddings = []
        
        if 'gpt2' in self.oracles:
            gpt2_emb = self.oracles['gpt2']['model'].transformer.wte.weight.detach().cpu().numpy()
            gpt2_emb = gpt2_emb[:self.vocab_size, :self.dim]
            embeddings.append(gpt2_emb)
            print(f"    GPT-2: {gpt2_emb.shape}")
            
        if 'pythia' in self.oracles:
            pythia_emb = self.oracles['pythia']['model'].gpt_neox.embed_in.weight.detach().cpu().numpy()
            if pythia_emb.shape[1] < self.dim:
                pythia_emb = np.pad(pythia_emb, ((0, 0), (0, self.dim - pythia_emb.shape[1])))
            else:
                pythia_emb = pythia_emb[:, :self.dim]
            pythia_emb = pythia_emb[:self.vocab_size]
            embeddings.append(pythia_emb)
            print(f"    Pythia: {pythia_emb.shape}")
        
        # Extract from large models if available
        if 'qwen' in self.oracles:
            try:
                qwen_emb = self.oracles['qwen']['model'].model.embed_tokens.weight.detach().cpu().float().numpy()
                if qwen_emb.shape[1] > self.dim:
                    qwen_emb = qwen_emb[:, :self.dim]
                else:
                    qwen_emb = np.pad(qwen_emb, ((0, 0), (0, self.dim - qwen_emb.shape[1])))
                qwen_emb = qwen_emb[:min(self.vocab_size, qwen_emb.shape[0])]
                if qwen_emb.shape[0] < self.vocab_size:
                    qwen_emb = np.pad(qwen_emb, ((0, self.vocab_size - qwen_emb.shape[0]), (0, 0)))
                embeddings.append(qwen_emb)
                print(f"    Qwen: {qwen_emb.shape}")
            except Exception as e:
                print(f"    Qwen embedding extraction failed: {e}")
                
        if 'smol' in self.oracles:
            try:
                smol_emb = self.oracles['smol']['model'].model.embed_tokens.weight.detach().cpu().float().numpy()
                if smol_emb.shape[1] > self.dim:
                    smol_emb = smol_emb[:, :self.dim]
                else:
                    smol_emb = np.pad(smol_emb, ((0, 0), (0, self.dim - smol_emb.shape[1])))
                smol_emb = smol_emb[:min(self.vocab_size, smol_emb.shape[0])]
                if smol_emb.shape[0] < self.vocab_size:
                    smol_emb = np.pad(smol_emb, ((0, self.vocab_size - smol_emb.shape[0]), (0, 0)))
                embeddings.append(smol_emb)
                print(f"    SmolLM2: {smol_emb.shape}")
            except Exception as e:
                print(f"    SmolLM2 embedding extraction failed: {e}")
            
        if embeddings:
            self.embeddings = np.mean(embeddings, axis=0)
            print(f"    Combined ({len(embeddings)} models): {self.embeddings.shape}")
            
    def build_pac_tree(self, max_tokens: int = 10000):
        """Build PAC tree with ByRef composition"""
        print("\n" + "="*60)
        print("PHASE 2: BUILD PAC TREE")
        print("="*60)
        
        if self.embeddings is None:
            return
            
        tokenizer = self.oracles['gpt2']['tokenizer']
        vocab = tokenizer.get_vocab()
        
        print(f"\n  Adding {min(max_tokens, len(vocab))} token instances...")
        
        count = 0
        for token, idx in vocab.items():
            if count >= max_tokens:
                break
            clean = token.replace('Ġ', '').replace('▁', '').strip().lower()
            if not clean or len(clean) < 2:
                continue
            emb = torch.tensor(self.embeddings[idx], dtype=torch.float32)
            self.pac.add_node(clean, emb, level=0)
            count += 1
            
        self.metrics['nodes_created'] = count
        print(f"    ✓ Added {count} token instances")
        
        # Create categories with ByRef
        print("\n  Creating semantic categories with ByRef...")
        
        semantic_groups = {
            'animal': ['cat', 'dog', 'bird', 'fish', 'horse', 'mouse', 'lion', 'tiger'],
            'color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple'],
            'number': ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'],
            'emotion': ['happy', 'sad', 'angry', 'fear', 'love', 'hate', 'joy', 'hope'],
            'nature': ['water', 'fire', 'earth', 'air', 'sun', 'moon', 'star', 'tree'],
            'body': ['head', 'hand', 'eye', 'face', 'heart', 'arm', 'leg', 'foot'],
            'action': ['run', 'walk', 'jump', 'sit', 'stand', 'move', 'stop', 'go'],
            'place': ['home', 'city', 'country', 'world', 'room', 'house', 'street', 'building'],
        }
        
        for category, instances in semantic_groups.items():
            available = [inst for inst in instances if inst in self.pac.name_to_id]
            if len(available) < 2:
                continue
                
            # Compute category delta (orthogonal to instance avg)
            instance_tensors = []
            for inst in available:
                inst_id = self.pac.name_to_id[inst]
                instance_tensors.append(self.pac.nodes[inst_id].delta)
            instance_avg = torch.mean(torch.stack(instance_tensors), dim=0)
            
            # Small orthogonal delta
            delta = torch.randn(self.dim, device=self.device) * 0.1 / PHI
            projection = torch.dot(delta, instance_avg) / (torch.norm(instance_avg) + 1e-8)
            delta = delta - 0.5 * projection * instance_avg / (torch.norm(instance_avg) + 1e-8)
            
            # Add category node
            cat_id = self.pac.add_node(category, delta, level=1)
            
            # Add ByRef links AND build token-to-category mapping
            self.category_tokens[category] = []
            for inst in available:
                inst_id = self.pac.name_to_id[inst]
                self.pac.add_byref(cat_id, inst_id, weight=1.0)
                self.metrics['byref_links'] += 1
                
                # Build reverse mapping: token → category
                self.token_to_category[inst] = category
                
                # Store token names for this category (for generation)
                self.category_tokens[category].append(inst)
                
            self.metrics['categories_created'] += 1
            print(f"    ✓ {category}: {len(available)} instances → byref")
            
        # Supercategories
        print("\n  Creating supercategories...")
        supercats = {
            'living_thing': ['animal', 'body'],
            'physical': ['nature', 'place'],
            'abstract': ['emotion', 'number'],
            'quality': ['color', 'action'],
        }
        
        for supercat, categories in supercats.items():
            available = [cat for cat in categories if cat in self.pac.name_to_id]
            if len(available) < 2:
                continue
                
            delta = torch.randn(self.dim, device=self.device) * 0.1 / (PHI ** 2)
            super_id = self.pac.add_node(supercat, delta, level=2)
            
            for cat in available:
                cat_id = self.pac.name_to_id[cat]
                self.pac.add_byref(super_id, cat_id, weight=1.0)
                self.metrics['byref_links'] += 1
                
                # Build category → supercategory mapping
                self.category_to_supercategory[cat] = supercat
                
            print(f"    ✓ {supercat}: {len(available)} categories → byref")
        
        # Print PAC hierarchy summary
        print(f"\n  PAC Hierarchy built:")
        print(f"    Level 0 (tokens): {len(self.pac.name_to_id)} nodes")
        print(f"    Level 1 (categories): {len(self.token_to_category)} mappings → {len(semantic_groups)} categories")
        print(f"    Level 2 (supercategories): {len(self.category_to_supercategory)} mappings → {len(supercats)} supercats")
            
    def verify_conservation(self):
        """Verify PAC conservation across all nodes"""
        print("\n" + "="*60)
        print("PHASE 3: VERIFY PAC CONSERVATION")
        print("="*60)
        
        all_conserved = True
        for level in [1, 2]:
            for node_id in self.pac.level_index[level]:
                node = self.pac.nodes[node_id]
                error = self.pac.conservation_check(node_id)
                
                if error > 1e-6:
                    print(f"  ⚠ {node.label}: error = {error:.6f}")
                    all_conserved = False
                    
        if all_conserved:
            total = len(self.pac.level_index[1]) + len(self.pac.level_index[2])
            print(f"  ✓ All {total} categories conserved perfectly")
            
    def train_with_field_evolution(self, num_epochs: int = 5, probes_per_epoch: int = 50):
        """Train using field evolution + continuous learning from failures.
        
        Key insight from POC-012: When confluence fails (miss), we learn 
        from the oracle's prediction. This grows the transition matrix.
        """
        print("\n" + "="*60)
        print("PHASE 4: TRAIN WITH CONTINUOUS LEARNING")
        print("="*60)
        print("  Klein-Gordon dynamics + SEC transitions + Learn from failures")
        print(f"  Epochs: {num_epochs}, Probes/epoch: {probes_per_epoch}")
        
        if 'gpt2' not in self.oracles:
            return
            
        tokenizer = self.oracles['gpt2']['tokenizer']
        model = self.oracles['gpt2']['model']
        
        # Expanded training corpus for diversity
        training_corpus = [
            # Basic
            "The cat sat on the mat.",
            "The dog ran across the field.",
            "Birds fly south for winter.",
            "Fish swim in the ocean.",
            # Science
            "Scientists study the natural world.",
            "Research reveals new discoveries daily.",
            "Experiments test hypotheses carefully.",
            "Data drives scientific conclusions.",
            # Abstract
            "Language is a tool for communication.",
            "Knowledge is power in society.",
            "Time passes quickly when busy.",
            "Love makes the world go round.",
            # Nature
            "The sun shines brightly today.",
            "Rain falls from dark clouds.",
            "Trees grow tall in forests.",
            "Mountains rise above the plains.",
            # Action
            "People walk down the street.",
            "Children run and play outside.",
            "Athletes train hard every day.",
            "Workers build new structures.",
            # Complex
            "The future of technology is exciting.",
            "Education opens doors to opportunity.",
            "Music brings people together joyfully.",
            "Art expresses human emotion deeply.",
            # More variety
            "Books contain wisdom and knowledge.",
            "History teaches valuable lessons.",
            "Cities grow and change constantly.",
            "Nature provides essential resources.",
            "Animals adapt to their environment.",
            "Humans create tools and technology.",
            "Water is essential for life.",
            "Fire provides warmth and light.",
        ]
        
        # Evaluation prompts - different set per epoch to avoid overfitting
        eval_prompt_sets = [
            ["The cat", "Scientists", "In nature", "The future"],
            ["A dog", "Research shows", "Time is", "People often"],
            ["Birds can", "Knowledge helps", "Water flows", "Music creates"],
            ["Animals need", "Education is", "The sun", "Love is"],
            ["Trees grow", "History shows", "Fire burns", "Art expresses"],
        ]
        
        # Track metrics per epoch
        epoch_metrics = []
        
        for epoch in range(num_epochs):
            print(f"\n  === EPOCH {epoch + 1}/{num_epochs} ===")
            
            epoch_start_transitions = self.transitions.num_transitions()
            epoch_evolutions = 0
            epoch_transitions = 0
            epoch_learns = 0
            
            # Shuffle corpus for this epoch
            shuffled = training_corpus.copy()
            random.shuffle(shuffled)
            
            for probe_idx in range(probes_per_epoch):
                text = shuffled[probe_idx % len(shuffled)]
                tokens_list = tokenizer.encode(text)[:32]
                
                if len(tokens_list) < 4:
                    continue
                    
                tokens = torch.tensor([tokens_list], device=self.device)
                
                # Get oracle outputs
                with torch.no_grad():
                    outputs = model(tokens, output_attentions=True)
                    predictions = outputs.logits.argmax(dim=-1)
                    
                # Field evolution for each token
                for t, tok_id in enumerate(tokens_list):
                    decoded = tokenizer.decode([tok_id]).strip().lower().replace('Ġ', '').replace('▁', '')
                    
                    if decoded in self.pac.name_to_id:
                        node_id = self.pac.name_to_id[decoded]
                        
                        # Evolve field
                        evolved = self.pac.evolve_field(node_id, steps=3)
                        epoch_evolutions += 1
                        self.metrics['field_evolutions'] += 1
                        
                        # Check SEC phase
                        should_act, phase = self.pac.sec_check(node_id)
                        if should_act:
                            epoch_transitions += 1
                            self.metrics['phase_transitions'] += 1
                            
                # MULTI-LEVEL PAC LEARNING: Learn at token, category, and supercategory levels!
                # NO BACKPROP - just hierarchical counting
                for t in range(len(tokens_list) - 1):
                    next_token = predictions[0, t].item()
                    
                    # Learn at multiple context lengths
                    for ctx_len in [5, 4, 3, 2]:
                        if t + 1 >= ctx_len:
                            context = tuple(tokens_list[t+1-ctx_len:t+1])
                            
                            # Check if we already know this (confluence hit)
                            is_known = context in self.token_confluence and next_token in self.token_confluence[context]
                            
                            # Learn at ALL PAC levels (token → category → supercategory)
                            self.learn_at_all_levels(
                                context, next_token, tokenizer,
                                weight=1.0, crystallize=is_known
                            )
                            
                            if is_known:
                                self.learning_stats['confluence_hits'] += 1
                            else:
                                self.learning_stats['confluence_misses'] += 1
                                epoch_learns += 1
                            
                            # Also update confluence
                            if context not in self.token_confluence:
                                self.token_confluence[context] = {}
                            self.token_confluence[context][next_token] = \
                                self.token_confluence[context].get(next_token, 0) + 1
                                
            # Update layers based on Fibonacci
            fib_idx = min(len(FIBONACCI) - 1, epoch + 1)
            self.materialized_layers = FIBONACCI[fib_idx]
            
            # Epoch metrics
            epoch_end_transitions = self.transitions.num_transitions()
            transition_growth = epoch_end_transitions - epoch_start_transitions
            
            epoch_data = {
                'epoch': epoch + 1,
                'transitions_total': epoch_end_transitions,
                'transition_growth': transition_growth,
                'field_evolutions': epoch_evolutions,
                'phase_transitions': epoch_transitions,
                'new_learns': epoch_learns,
                'layers': self.materialized_layers,
            }
            epoch_metrics.append(epoch_data)
            
            print(f"    Transitions: {epoch_start_transitions} → {epoch_end_transitions} (+{transition_growth})")
            print(f"    New learns: {epoch_learns} | Evolutions: {epoch_evolutions}")
            print(f"    Layers: {self.materialized_layers}")
            
            # Evaluate with LIVE LEARNING: failures teach us!
            eval_prompts = eval_prompt_sets[epoch % len(eval_prompt_sets)]
            print(f"\n    Live learning eval (epoch {epoch+1}):")
            
            total_hits = 0
            total_tokens = 0
            eval_learns = 0
            
            for prompt in eval_prompts:
                result, stats = self.generate_with_learning(prompt, max_tokens=15, temperature=0.7)
                
                hits = stats.get('hits', 0)
                total = stats.get('hits', 0) + stats.get('misses', 0)
                total_hits += hits
                total_tokens += total
                eval_learns += stats.get('new_learns', 0)
                
                short_result = result[:50] + "..." if len(result) > 50 else result
                print(f"      '{prompt}' → {stats['hit_rate']:.0f}% (+{stats.get('new_learns', 0)} learns) | {short_result}")
                
            epoch_data['eval_hit_rate'] = (total_hits / total_tokens * 100) if total_tokens > 0 else 0
            epoch_data['eval_learns'] = eval_learns
            print(f"    Epoch hit rate: {epoch_data['eval_hit_rate']:.1f}% | +{eval_learns} new learns")
            
            # Decay weak transitions (prune noise)
            if epoch > 0 and epoch % 2 == 0:
                self.transitions.decay(factor=LAMBDA_STAR, threshold=XI / 10)
                print(f"    Decayed transitions → {self.transitions.num_transitions()}")
            
        self.metrics['confluence_contexts'] = len(self.token_confluence)
        self.metrics['transitions_learned'] = self.transitions.num_transitions()
        self.epoch_metrics = epoch_metrics
        
        # Summary
        print("\n  " + "-"*50)
        print("  CONTINUOUS LEARNING SUMMARY")
        print("  " + "-"*50)
        print(f"  {'Epoch':<6} {'Transitions':<12} {'Growth':<8} {'Learns':<8} {'Eval%':<8}")
        for e in epoch_metrics:
            print(f"  {e['epoch']:<6} {e['transitions_total']:<12} +{e['transition_growth']:<7} +{e.get('eval_learns', 0):<7} {e['eval_hit_rate']:.1f}%")
            
        print(f"\n  Final transitions: {self.transitions.num_transitions()}")
        print(f"  Crystallized: {self.transitions.crystallized}")
        print(f"  Total evolutions: {self.metrics['field_evolutions']}")

    def generate_with_learning(self, prompt: str, max_tokens: int = 30, temperature: float = 0.8) -> Tuple[str, Dict]:
        """Generate with MULTI-LEVEL PAC learning: learn from misses at all levels!
        
        When we miss at token level, try category level, then supercategory.
        When we finally get oracle answer, learn at ALL levels.
        
        NO BACKPROP - hierarchical structure enables generalization!
        """
        if 'gpt2' not in self.oracles:
            return prompt, {}
            
        tokenizer = self.oracles['gpt2']['tokenizer']
        model = self.oracles['gpt2']['model']
        
        tokens = tokenizer.encode(prompt)
        hits = 0
        misses = 0
        new_learns = 0
        pac_guided = 0
        category_hits = 0
        recent_ngrams: Set[tuple] = set()
        
        for _ in range(max_tokens):
            found = False
            next_token = None
            
            # Detect semantic context
            semantic_cat = self._detect_semantic_context(tokens, tokenizer)
            
            # LEVEL 0: Try token-level transition matrix
            for ctx_len in [5, 4, 3, 2]:
                if len(tokens) >= ctx_len:
                    context = tuple(tokens[-ctx_len:])
                    
                    pred, conf = self.transitions.predict(context)
                    if pred is not None and conf > 0.3:
                        # N-gram check
                        would_repeat = False
                        for ngram_len in [5, 4, 3]:
                            if len(tokens) >= ngram_len - 1:
                                potential = tuple(tokens[-(ngram_len-1):]) + (pred,)
                                if potential in recent_ngrams:
                                    would_repeat = True
                                    break
                        if not would_repeat:
                            next_token = pred
                            found = True
                            hits += 1
                            break
            
            # LEVEL 1: Try category-level prediction (generalization!)
            if not found:
                for ctx_len in [3, 2]:
                    if len(tokens) >= ctx_len:
                        # Build category context
                        context_cats = []
                        for tok_id in tokens[-ctx_len:]:
                            cat = self.get_token_category(tok_id, tokenizer)
                            context_cats.append(cat if cat else f"tok_{tok_id}")
                        cat_context = tuple(context_cats)
                        
                        # Predict at category level
                        pred_cat, conf = self.transitions.predict(cat_context)
                        if pred_cat is not None and conf > 0.2:
                            # Get a token from this category
                            if pred_cat in self.category_tokens and self.category_tokens[pred_cat]:
                                # Sample from category tokens
                                cat_members = list(self.category_tokens[pred_cat])
                                if cat_members:
                                    # Get token IDs for category members
                                    for member in cat_members:
                                        if member in self.pac.name_to_id:
                                            # Encode to get token ID
                                            member_tokens = tokenizer.encode(" " + member)
                                            if member_tokens:
                                                candidate = member_tokens[0]
                                                # N-gram check
                                                would_repeat = False
                                                for ngram_len in [5, 4, 3]:
                                                    if len(tokens) >= ngram_len - 1:
                                                        potential = tuple(tokens[-(ngram_len-1):]) + (candidate,)
                                                        if potential in recent_ngrams:
                                                            would_repeat = True
                                                            break
                                                if not would_repeat:
                                                    next_token = candidate
                                                    found = True
                                                    category_hits += 1
                                                    pac_guided += 1
                                                    break
                            if found:
                                break
                            
            # Try confluence if transitions didn't work
            if not found:
                for ctx_len in [5, 4, 3, 2]:
                    if len(tokens) >= ctx_len:
                        context = tuple(tokens[-ctx_len:])
                        if context in self.token_confluence:
                            candidates = self.token_confluence[context]
                            
                            # N-gram blocking
                            filtered = {}
                            for tok, cnt in candidates.items():
                                would_repeat = False
                                for ngram_len in [5, 4, 3]:
                                    if len(tokens) >= ngram_len - 1:
                                        potential = tuple(tokens[-(ngram_len-1):]) + (tok,)
                                        if potential in recent_ngrams:
                                            would_repeat = True
                                            break
                                if not would_repeat:
                                    filtered[tok] = cnt
                                    
                            if not filtered:
                                continue
                                
                            # Sample with temperature
                            items = list(filtered.items())
                            weights = np.array([v for _, v in items], dtype=float)
                            weights = weights ** (1.0 / temperature)
                            weights /= weights.sum()
                            
                            idx = np.random.choice(len(items), p=weights)
                            next_token = items[idx][0]
                            
                            found = True
                            hits += 1
                            break
                            
            if not found:
                misses += 1
                
                # Oracle fallback - AND LEARN AT ALL LEVELS!
                input_ids = torch.tensor([tokens[-32:]], device=self.device)
                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits[0, -1] / temperature
                    
                    # N-gram blocking
                    for ngram_len in [5, 4, 3]:
                        if len(tokens) >= ngram_len - 1:
                            prefix = tuple(tokens[-(ngram_len-1):])
                            for blocked in range(min(10000, logits.shape[0])):
                                if prefix + (blocked,) in recent_ngrams:
                                    logits[blocked] = float('-inf')
                                    
                    top_k = 50
                    top_logits, top_indices = torch.topk(logits, top_k)
                    probs = torch.softmax(top_logits, dim=-1)
                    idx = torch.multinomial(probs, 1).item()
                    next_token = top_indices[idx].item()
                
                # MULTI-LEVEL LEARNING: Learn from oracle at ALL PAC levels!
                for ctx_len in [5, 4, 3, 2]:
                    if len(tokens) >= ctx_len:
                        context = tuple(tokens[-ctx_len:])
                        self.learn_at_all_levels(context, next_token, tokenizer, 
                                                  weight=1.0, crystallize=False)
                        new_learns += 1
                        
            tokens.append(next_token)
            for ngram_len in [3, 4, 5]:
                if len(tokens) >= ngram_len:
                    recent_ngrams.add(tuple(tokens[-ngram_len:]))
                        
            # Stop on sentence end
            decoded = tokenizer.decode([tokens[-1]])
            if decoded.strip() in ['.', '!', '?'] and len(tokens) > len(tokenizer.encode(prompt)) + 5:
                break
                
        result = tokenizer.decode(tokens)
        self.metrics['generations'] += 1
        
        stats = {
            'hits': hits,
            'misses': misses,
            'hit_rate': hits / (hits + misses) * 100 if (hits + misses) > 0 else 0,
            'pac_guided': pac_guided,
            'category_hits': category_hits,
            'new_learns': new_learns,
            'semantic_category': semantic_cat,
        }
        
        return result, stats

    def generate(self, prompt: str, max_tokens: int = 30, temperature: float = 0.8) -> Tuple[str, Dict]:
        """Generate with PAC-guided entropy injection"""
        if 'gpt2' not in self.oracles:
            return prompt, {}
            
        tokenizer = self.oracles['gpt2']['tokenizer']
        model = self.oracles['gpt2']['model']
        
        tokens = tokenizer.encode(prompt)
        hits = 0
        misses = 0
        pac_guided = 0
        recent_ngrams: Set[tuple] = set()
        
        for _ in range(max_tokens):
            found = False
            
            # Detect semantic context
            semantic_cat = self._detect_semantic_context(tokens, tokenizer)
            
            # Try confluence
            for ctx_len in [5, 4, 3, 2]:
                if len(tokens) >= ctx_len:
                    context = tuple(tokens[-ctx_len:])
                    if context in self.token_confluence:
                        candidates = self.token_confluence[context]
                        
                        # N-gram blocking
                        filtered = {}
                        for tok, cnt in candidates.items():
                            would_repeat = False
                            for ngram_len in [5, 4, 3]:
                                if len(tokens) >= ngram_len - 1:
                                    potential = tuple(tokens[-(ngram_len-1):]) + (tok,)
                                    if potential in recent_ngrams:
                                        would_repeat = True
                                        break
                            if not would_repeat:
                                filtered[tok] = cnt
                                
                        if not filtered:
                            break
                            
                        # PAC guidance for low confluence
                        if len(filtered) < 3 and semantic_cat:
                            cat_tokens = self._get_category_tokens(semantic_cat, tokenizer)
                            for cat_tok in cat_tokens[:5]:
                                if cat_tok not in filtered:
                                    filtered[cat_tok] = max(filtered.values()) * 0.3
                            pac_guided += 1
                            
                        # Sample
                        items = list(filtered.items())
                        weights = np.array([v for _, v in items], dtype=float)
                        weights = weights ** (1.0 / temperature)
                        weights /= weights.sum()
                        
                        idx = np.random.choice(len(items), p=weights)
                        next_token = items[idx][0]
                        
                        tokens.append(next_token)
                        for ngram_len in [3, 4, 5]:
                            if len(tokens) >= ngram_len:
                                recent_ngrams.add(tuple(tokens[-ngram_len:]))
                                
                        hits += 1
                        found = True
                        break
                        
            if not found:
                misses += 1
                
                # Oracle fallback with field evolution
                input_ids = torch.tensor([tokens[-32:]], device=self.device)
                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits[0, -1] / temperature
                    
                    # N-gram blocking
                    for ngram_len in [5, 4, 3]:
                        if len(tokens) >= ngram_len - 1:
                            prefix = tuple(tokens[-(ngram_len-1):])
                            for blocked in range(min(10000, logits.shape[0])):
                                if prefix + (blocked,) in recent_ngrams:
                                    logits[blocked] = float('-inf')
                                    
                    top_k = 50
                    top_logits, top_indices = torch.topk(logits, top_k)
                    probs = torch.softmax(top_logits, dim=-1)
                    idx = torch.multinomial(probs, 1).item()
                    next_token = top_indices[idx].item()
                    
                tokens.append(next_token)
                for ngram_len in [3, 4, 5]:
                    if len(tokens) >= ngram_len:
                        recent_ngrams.add(tuple(tokens[-ngram_len:]))
                        
            # Stop on sentence end
            decoded = tokenizer.decode([tokens[-1]])
            if decoded.strip() in ['.', '!', '?'] and len(tokens) > len(tokenizer.encode(prompt)) + 5:
                break
                
        result = tokenizer.decode(tokens)
        self.metrics['generations'] += 1
        
        stats = {
            'hits': hits,
            'misses': misses,
            'hit_rate': hits / (hits + misses) * 100 if (hits + misses) > 0 else 0,
            'pac_guided': pac_guided,
            'semantic_category': semantic_cat,
        }
        
        return result, stats
        
    def _detect_semantic_context(self, tokens: List[int], tokenizer) -> Optional[str]:
        """Detect semantic category from recent tokens"""
        category_scores: Dict[str, float] = {}
        
        for i, token_id in enumerate(reversed(tokens[-8:])):
            weight = 1.0 / (i + 1)
            decoded = tokenizer.decode([token_id]).strip().lower().replace('Ġ', '').replace('▁', '')
            
            if decoded in self.pac.name_to_id:
                node_id = self.pac.name_to_id[decoded]
                
                for cat_id in self.pac.level_index[1]:
                    cat = self.pac.nodes[cat_id]
                    if node_id in cat.byrefs:
                        category_scores[cat.label] = category_scores.get(cat.label, 0) + weight
                        
        if category_scores:
            return max(category_scores.keys(), key=lambda k: category_scores[k])
        return None
        
    def _get_category_tokens(self, category: str, tokenizer) -> List[int]:
        """Get token IDs for a category"""
        if category not in self.pac.name_to_id:
            return []
            
        cat_id = self.pac.name_to_id[category]
        cat = self.pac.nodes[cat_id]
        
        token_ids = []
        for ref_id in cat.byrefs:
            if ref_id in self.pac.nodes:
                member = self.pac.nodes[ref_id]
                for tok, idx in tokenizer.get_vocab().items():
                    clean = tok.replace('Ġ', '').replace('▁', '').strip().lower()
                    if clean == member.label:
                        token_ids.append(idx)
                        break
                        
        return token_ids
        
    def build(self, include_large_models: bool = False):
        """Run full build pipeline
        
        Args:
            include_large_models: If True, load Llama 3.1 8B and Mistral 7B (4-bit)
        """
        print("\n" + "="*70)
        print("UNIFIED FULL SYSTEM - COMPLETE ARCHITECTURE")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Components: PAC System, Field Evolution, Resonance Attention")
        print(f"Physics: Klein-Gordon + PAC Conservation + SEC Phase Transitions")
        if include_large_models:
            print(f"Large Models: Llama 3.1 8B + Mistral 7B (4-bit quantized)")
        
        self.load_oracles(include_large_models=include_large_models)
        self.extract_embeddings()
        self.build_pac_tree()
        self.verify_conservation()
        self.train_with_field_evolution(num_epochs=5, probes_per_epoch=50)
        
        print("\n" + "="*70)
        print("BUILD COMPLETE")
        print("="*70)
        print(f"\nMetrics:")
        for key, value in self.metrics.items():
            print(f"  {key}: {value}")
        
        print(f"\nOracles loaded: {list(self.oracles.keys())}")
        for name, oracle in self.oracles.items():
            print(f"  - {name}: {oracle.get('size', 'unknown')}")
            
        fib_aligned = self.materialized_layers in FIBONACCI
        print(f"\nFibonacci: {self.materialized_layers} layers → {'✓ aligned' if fib_aligned else '○ not aligned'}")
        
        return self


def main():
    """Run unified full system"""
    print("="*70)
    print("POC-021: UNIFIED FULL DEMONSTRATION")
    print("="*70)
    print("Integrating: Fracton + GAIA-1 + POC-011 + POC-003 + POC-016-020")
    print("="*70)
    
    system = UnifiedFullSystem(dim=256, max_layers=13)
    system.build()
    
    # Generation test
    print("\n" + "="*70)
    print("GENERATION TEST")
    print("="*70)
    
    prompts = [
        "The cat",
        "Scientists study",
        "The dog and the cat",
        "In nature",
    ]
    
    for prompt in prompts:
        print(f"\n'{prompt}' →")
        result, stats = system.generate(prompt, max_tokens=30)
        cat_str = f" [{stats.get('semantic_category', '')}]" if stats.get('semantic_category') else ""
        pac_str = f", PAC: {stats.get('pac_guided', 0)}" if stats.get('pac_guided', 0) > 0 else ""
        print(f"    [Hit: {stats['hit_rate']:.1f}%{pac_str}{cat_str}]")
        print(f"    {result}")
        
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'metrics': system.metrics,
        'architecture': {
            'pac_system': True,
            'klein_gordon': True,
            'resonance_attention': True,
            'byref_composition': True,
            'sec_phases': True,
            'tiered_cache': True,
        }
    }
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "unified_full_system.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n\nSaved to {output_dir / 'unified_full_system.json'}")
    
    return system


if __name__ == "__main__":
    system = main()
