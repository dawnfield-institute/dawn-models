"""
QBE-Integrated PAC Import (Using Existing Components)
======================================================

REUSES existing solved infrastructure:
- POC-002: PhaseTransitionMonitor, FibonacciScheduler
- POC-011: PACLazySystem primitives
- POC-012: TransitionMatrix for prediction
- adaptive_controller.py: QBE core equations
- fracton physics constants

This is the INTEGRATION test, not re-invention.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

# Add paths for existing modules
sys.path.insert(0, str(Path(__file__).parent.parent / "poc_002_resonance_training" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "poc_011_pac_lazy_transformer" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "fracton"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# REUSE existing components
from physics_trainer import (
    PhaseTransitionMonitor,  # Already handles crystallization at PHI_XI
    FibonacciScheduler,      # Already handles Fib learning rates
    PHI, XI, PHI_XI, LAMBDA_STAR, FIBONACCI
)
from pac_lazy_core import PACNode, PACLazySystem

print("✓ Loaded existing POC-002 and POC-011 components")


# =============================================================================
# TransitionMatrix from POC-012 (reused, not re-invented)
# =============================================================================

class TransitionMatrix:
    """
    Hybrid transition matrix: sparse tracking + O(1) prediction.
    From POC-012 - extended with n-gram context.
    """
    
    def __init__(self, vocab_size: int, device: str, context_size: int = 3):
        self.vocab_size = vocab_size
        self.device = device
        self.context_size = context_size  # Use last N tokens as context
        
        # Sparse counts: (context_tuple, next) -> count
        self.counts: Dict[Tuple, float] = {}
        
        # Best prediction cache: context -> (next, count)
        self._best_cache: Dict[Tuple, Tuple[int, float]] = {}
        
        # Row totals for probability calculation
        self._row_totals: Dict[Tuple, float] = defaultdict(float)
        
        # Top-k cache per context
        self._topk_cache: Dict[Tuple, List[Tuple[int, float]]] = {}
        
        self._total_transitions = 0
    
    def _make_context(self, tokens: List[int]) -> Tuple:
        """Create context key from last N tokens."""
        if len(tokens) >= self.context_size:
            return tuple(tokens[-self.context_size:])
        return tuple(tokens)
    
    def add(self, context: List[int], next_token: int, weight: float = 1.0):
        """Add a transition with context."""
        ctx = self._make_context(context)
        key = (ctx, next_token)
        old_count = self.counts.get(key, 0.0)
        new_count = old_count + weight
        self.counts[key] = new_count
        
        # Update row total
        self._row_totals[ctx] += weight
        
        # Update best cache
        if ctx not in self._best_cache or new_count > self._best_cache[ctx][1]:
            self._best_cache[ctx] = (next_token, new_count)
        
        # Invalidate top-k cache
        if ctx in self._topk_cache:
            del self._topk_cache[ctx]
        
        self._total_transitions += 1
    
    def get_top_next(self, context: List[int]) -> Tuple[int, float]:
        """Get most likely next token - O(1) from cache!"""
        ctx = self._make_context(context)
        
        if ctx not in self._best_cache:
            # Try shorter context (backoff)
            for i in range(1, len(ctx)):
                shorter = ctx[i:]
                if shorter in self._best_cache:
                    best_next, best_count = self._best_cache[shorter]
                    total = self._row_totals[shorter]
                    return best_next, (best_count / total) * 0.9  # Slight penalty for backoff
            return 0, 0.0
        
        best_next, best_count = self._best_cache[ctx]
        total = self._row_totals[ctx]
        
        if total == 0:
            return 0, 0.0
        
        return best_next, best_count / total
    
    def get_top_k(self, context: List[int], k: int = 10) -> List[Tuple[int, float]]:
        """Get top-k next tokens with probabilities, with backoff."""
        ctx = self._make_context(context)
        
        # Try exact match first
        if ctx in self._topk_cache:
            return self._topk_cache[ctx][:k]
        
        # Build candidates from exact context
        candidates = []
        for key, count in self.counts.items():
            c, n = key
            if c == ctx:
                total = self._row_totals[ctx]
                prob = count / total if total > 0 else 0
                candidates.append((n, prob))
        
        # Backoff to shorter context if needed
        if not candidates:
            for i in range(1, len(ctx)):
                shorter = ctx[i:]
                for key, count in self.counts.items():
                    c, n = key
                    if c == shorter:
                        total = self._row_totals[shorter]
                        prob = (count / total) * 0.8 if total > 0 else 0  # Backoff penalty
                        candidates.append((n, prob))
                if candidates:
                    break
        
        candidates.sort(key=lambda x: -x[1])
        self._topk_cache[ctx] = candidates[:50]
        
        return candidates[:k]
    
    def learn_sequence(self, token_ids: List[int], weight: float = 1.0):
        """Learn all transitions in a sequence with context."""
        for i in range(len(token_ids) - 1):
            # Use tokens up to position i as context
            context = token_ids[:i+1]
            next_token = token_ids[i + 1]
            self.add(context, next_token, weight)
    
    def num_transitions(self) -> int:
        return len(self.counts)


@dataclass
class GrowthEvent:
    """Record of a growth event."""
    iteration: int
    old_dim: int
    new_dim: int
    trigger: str
    crystals: int = 0


class GrowingPACTransformer(nn.Module):
    """
    Transformer that grows using EXISTING POC infrastructure.
    
    Uses:
    - PhaseTransitionMonitor for crystallization detection (POC-002)
    - FibonacciScheduler for growth amounts (POC-002)
    - PACLazySystem for node management (POC-011)
    - TransitionMatrix for prediction (POC-012)
    """
    
    def __init__(self, vocab_size: int, initial_dim: int = 64, 
                 max_dim: int = 512, device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.vocab_size = vocab_size
        self.current_dim = initial_dim
        self.max_dim = max_dim
        
        # REUSE: Phase transition monitor from POC-002
        self.phase_monitor = PhaseTransitionMonitor(threshold=PHI_XI, device=self.device)
        
        # REUSE: Fibonacci scheduler from POC-002
        self.fib_scheduler = FibonacciScheduler(base_lr=0.1)
        
        # REUSE: PAC-Lazy system from POC-011
        self.pac_system = PACLazySystem(
            field_shape=(initial_dim,),
            total_potential=1000.0,
            device=device
        )
        
        # REUSE: TransitionMatrix from POC-012 for prediction
        self.transitions = TransitionMatrix(vocab_size, device)
        
        # Learnable components (minimal)
        self.embedding = nn.Parameter(
            torch.randn(vocab_size, initial_dim, device=self.device) * 0.02
        )
        self.mlp_up = nn.Linear(initial_dim, initial_dim * 4, device=self.device)
        self.mlp_down = nn.Linear(initial_dim * 4, initial_dim, device=self.device)
        
        # Growth tracking
        self.growth_events: List[GrowthEvent] = []
        self.total_crystals = 0
        self.growth_thresholds: Set[int] = set()
        
        # Vocab embeddings from PAC extraction
        self.vocab_embeddings: torch.Tensor = None
        
    def load_knowledge(self, pac_path: Path):
        """Load extracted PAC knowledge."""
        vocab_data = torch.load(pac_path / "pac_vocab.pt", weights_only=True)
        original = vocab_data['vocab_deltas'].to(self.device)
        
        # Resize to current dim
        if original.shape[1] == self.current_dim:
            self.vocab_embeddings = original
        elif original.shape[1] > self.current_dim:
            self.vocab_embeddings = original[:, :self.current_dim]
        else:
            padding = torch.randn(
                original.shape[0], self.current_dim - original.shape[1],
                device=self.device
            ) * 0.01
            self.vocab_embeddings = torch.cat([original, padding], dim=1)
        
        # Initialize embedding from loaded vocab
        with torch.no_grad():
            self.embedding.data = self.vocab_embeddings[:self.vocab_size]
        
        print(f"  ✓ Loaded embeddings: {self.vocab_embeddings.shape}")
    
    def _grow_if_needed(self) -> Optional[GrowthEvent]:
        """Check if growth is needed based on phase transitions."""
        if self.current_dim >= self.max_dim:
            return None
        
        # Use Fibonacci thresholds for crystal counts
        for i, fib in enumerate(FIBONACCI):
            if fib > 100:  # Don't wait too long
                break
            if self.total_crystals >= fib and fib not in self.growth_thresholds:
                self.growth_thresholds.add(fib)
                
                # Growth amount from Fibonacci scheduler
                growth = FIBONACCI[min(len(self.growth_events) + 1, len(FIBONACCI) - 1)] * 8
                new_dim = min(self.current_dim + growth, self.max_dim)
                
                if new_dim > self.current_dim:
                    event = GrowthEvent(
                        iteration=self.phase_monitor.get_stats()['total_steps'],
                        old_dim=self.current_dim,
                        new_dim=new_dim,
                        trigger=f"crystals_{fib}",
                        crystals=self.total_crystals
                    )
                    self._perform_growth(new_dim, event)
                    return event
        
        return None
    
    def _perform_growth(self, new_dim: int, event: GrowthEvent):
        """Perform the actual growth."""
        print(f"  🌱 GROWTH: {self.current_dim} → {new_dim} dim "
              f"(trigger: {event.trigger}, crystals: {event.crystals})")
        
        # Grow embedding
        with torch.no_grad():
            old_embed = self.embedding.data
            new_embed = torch.zeros(self.vocab_size, new_dim, device=self.device)
            new_embed[:, :self.current_dim] = old_embed
            new_embed[:, self.current_dim:] = torch.randn(
                self.vocab_size, new_dim - self.current_dim, device=self.device
            ) * 0.02
            self.embedding = nn.Parameter(new_embed)
        
        # Grow MLP
        with torch.no_grad():
            # Up projection
            new_up = nn.Linear(new_dim, new_dim * 4, device=self.device)
            nn.init.xavier_uniform_(new_up.weight)
            new_up.weight.data[:self.current_dim * 4, :self.current_dim] = self.mlp_up.weight.data
            new_up.bias.data[:self.current_dim * 4] = self.mlp_up.bias.data
            self.mlp_up = new_up
            
            # Down projection
            new_down = nn.Linear(new_dim * 4, new_dim, device=self.device)
            nn.init.xavier_uniform_(new_down.weight)
            new_down.weight.data[:self.current_dim, :self.current_dim * 4] = self.mlp_down.weight.data
            new_down.bias.data[:self.current_dim] = self.mlp_down.bias.data
            self.mlp_down = new_down
        
        # Grow vocab embeddings
        if self.vocab_embeddings is not None:
            old_vocab = self.vocab_embeddings
            padding = torch.randn(
                old_vocab.shape[0], new_dim - self.current_dim,
                device=self.device
            ) * 0.01
            self.vocab_embeddings = torch.cat([old_vocab, padding], dim=1)
        
        # Update PAC system field shape
        self.pac_system.field_shape = (new_dim,)
        
        self.current_dim = new_dim
        self.growth_events.append(event)
    
    def process_sequence(self, token_ids: List[int], debug: bool = False):
        """Process sequence using PAC-Lazy nodes + phase monitoring."""
        # Add tokens as PAC nodes
        node_ids = []
        for i, tid in enumerate(token_ids):
            nid = f"tok_{i}"
            node = self.pac_system.add_node(nid, token_id=tid)
            
            # Inject delta from vocab embedding
            if tid < len(self.vocab_embeddings):
                delta = self.vocab_embeddings[tid]
                # Resize if needed
                if len(delta) != self.current_dim:
                    delta = delta[:self.current_dim] if len(delta) > self.current_dim else \
                            F.pad(delta, (0, self.current_dim - len(delta)))
                self.pac_system.inject_delta(nid, delta, potential_cost=0.1)
            
            node_ids.append(nid)
        
        # Link neighbors (causal sequence)
        for i in range(len(node_ids) - 1):
            self.pac_system.link_neighbors(node_ids[i], node_ids[i + 1])
        
        # Propagate with LAMBDA_STAR decay
        for _ in range(10):
            self.pac_system.propagate_local(decay=LAMBDA_STAR)
        
        # Check phase transitions using POC-002 monitor
        crystals_this_round = 0
        for nid in node_ids:
            node = self.pac_system.get_node(nid)
            if node.delta is not None:
                is_transition, metric = self.phase_monitor.check_transition(
                    node.delta, 
                    step=self.phase_monitor.get_stats()['total_steps']
                )
                if is_transition:
                    crystals_this_round += 1
                    self.total_crystals += 1
        
        # Check for growth
        self._grow_if_needed()
        
        if debug:
            stats = self.phase_monitor.get_stats()
            print(f"    [PAC] tokens={len(token_ids)}, transitions={stats['transitions']}, "
                  f"total_crystals={self.total_crystals}")
        
        # Clear nodes for next sequence (keep stats)
        self.pac_system.nodes.clear()
        self.pac_system.active_nodes.clear()
        self.pac_system.allocated_potential = 0.0
        
        return {'crystals': crystals_this_round}
    
    def compose_and_predict(self, token_ids: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Predict next token using BOTH:
        1. Transition matrix (learned patterns with context) - from POC-012
        2. Embedding similarity (semantic space) - fallback
        
        Hybrid approach: transition has priority if strong, else use embedding.
        """
        if not token_ids:
            return []
        
        # METHOD 1: Check transition matrix first (learned patterns with context)
        transition_preds = self.transitions.get_top_k(token_ids, k=top_k)
        
        if transition_preds and transition_preds[0][1] > 0.1:
            # Strong transition signal - use it
            return transition_preds
        
        # METHOD 2: Embedding similarity fallback
        weights = []
        deltas = []
        
        for i, tid in enumerate(token_ids):
            if tid < len(self.vocab_embeddings):
                delta = self.vocab_embeddings[tid]
                if len(delta) != self.current_dim:
                    delta = delta[:self.current_dim] if len(delta) > self.current_dim else \
                            F.pad(delta, (0, self.current_dim - len(delta)))
                deltas.append(delta)
                weights.append(np.exp((i - len(token_ids) + 1) / 2.0))
        
        if not deltas:
            return []
        
        weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        weights = weights / weights.sum()
        
        stacked = torch.stack(deltas)
        composed = (stacked * weights.unsqueeze(1)).sum(dim=0)
        
        # MLP transform
        h = F.gelu(self.mlp_up(composed))
        composed = composed + self.mlp_down(h)
        
        # Score against vocab
        composed_norm = F.normalize(composed.unsqueeze(0), dim=1).squeeze()
        vocab_norm = F.normalize(self.vocab_embeddings[:, :self.current_dim], dim=1)
        scores = vocab_norm @ composed_norm
        
        top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
        
        # Combine with transition predictions if any exist
        result = []
        seen = set()
        
        # Add transition predictions first (they're learned patterns)
        for tid, prob in transition_preds:
            result.append((tid, prob + 0.5))  # Boost learned patterns
            seen.add(tid)
        
        # Add embedding predictions
        for idx, score in zip(top_indices, top_scores):
            tid = idx.item()
            if tid not in seen:
                result.append((tid, score.item()))
                seen.add(tid)
        
        # Sort by score and return top-k
        result.sort(key=lambda x: -x[1])
        return result[:top_k]
    
    def generate(self, token_ids: List[int], max_new_tokens: int = 20,
                temperature: float = 0.7) -> List[int]:
        """Generate tokens with continuous learning."""
        generated = list(token_ids)
        
        # Learn initial transitions from prompt
        self.transitions.learn_sequence(token_ids, weight=PHI)  # φ weight for prompt
        
        for i in range(max_new_tokens):
            # Process and get predictions
            self.process_sequence(generated, debug=(i == 0))
            predictions = self.compose_and_predict(generated, top_k=50)
            
            if not predictions:
                break
            
            # Sample
            scores = torch.tensor([p[1] for p in predictions]) / temperature
            probs = F.softmax(scores, dim=0)
            idx = torch.multinomial(probs, 1).item()
            
            next_token = predictions[idx][0]
            generated.append(next_token)
            
            # LEARN: Add this transition with context
            self.transitions.add(generated[:-1], next_token, weight=1.0)
            
            self.fib_scheduler.step()
        
        return generated


def test_integrated():
    """Test the integrated growing transformer."""
    from transformers import GPT2Tokenizer
    
    print("="*70)
    print("INTEGRATED PAC GROWING TRANSFORMER")
    print("="*70)
    print("\nReusing existing components:")
    print("  - PhaseTransitionMonitor (POC-002)")
    print("  - FibonacciScheduler (POC-002)")
    print("  - PACLazySystem (POC-011)")
    print("  - TransitionMatrix (POC-012)")
    
    pac_path = Path(__file__).parent.parent / "poc_016_pac_extraction" / "extracted_v3" / "pythia_70m"
    
    if not pac_path.exists():
        print(f"❌ PAC not found at {pac_path}")
        return
    
    print("\nInitializing with 64-dim...")
    model = GrowingPACTransformer(
        vocab_size=50304,
        initial_dim=64,
        max_dim=512
    )
    model.load_knowledge(pac_path)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # ==========================================================================
    # PHASE 1: Pre-train transitions on sample text
    # ==========================================================================
    print("\n" + "="*70)
    print("PHASE 1: LEARNING TRANSITIONS (no backprop)")
    print("="*70)
    
    training_texts = [
        "The weather today is sunny and warm.",
        "The weather today is cold and rainy.",
        "The weather today is cloudy with a chance of rain.",
        "Once upon a time there was a princess.",
        "Once upon a time there was a dragon.",
        "Once upon a time there was a brave knight.",
        "The meaning of life is to find happiness.",
        "The meaning of life is to help others.",
        "In the beginning there was nothing but darkness.",
        "In the beginning there was nothing but light.",
        "The sun rises in the east and sets in the west.",
        "Water flows downhill and fire burns upward.",
        "The quick brown fox jumps over the lazy dog.",
        "All good things must come to an end.",
        "Actions speak louder than words.",
    ]
    
    for text in training_texts:
        tokens = tokenizer.encode(text)
        model.transitions.learn_sequence(tokens, weight=PHI)  # φ weight
        model.process_sequence(tokens)  # Trigger crystallization
    
    print(f"  Learned {model.transitions.num_transitions()} transitions")
    print(f"  Phase transitions: {model.phase_monitor.get_stats()['transitions']}")
    print(f"  Current dim: {model.current_dim}")
    
    # ==========================================================================
    # PHASE 2: Generate with continuous learning
    # ==========================================================================
    print("\n" + "="*70)
    print("PHASE 2: GENERATION WITH CONTINUOUS LEARNING")
    print("="*70)
    
    prompts = [
        "The weather today is",
        "Once upon a time there was a",
        "The meaning of life is to",
        "In the beginning there was nothing but"
    ]
    
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt)
        
        print(f"\nPrompt: '{prompt}'")
        print(f"  Transitions before: {model.transitions.num_transitions()}")
        
        generated = model.generate(token_ids, max_new_tokens=15, temperature=0.6)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        
        print(f"  Transitions after: {model.transitions.num_transitions()}")
        print(f"  Current dim: {model.current_dim}")
        print(f"  → {text}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    stats = model.phase_monitor.get_stats()
    print(f"""
Integrated Growing Transformer Results:

Starting dimension: 64
Final dimension: {model.current_dim}
Total growth events: {len(model.growth_events)}
Total phase transitions: {stats['transitions']}
Total learned transitions: {model.transitions.num_transitions()}

REUSED Components:
✓ PhaseTransitionMonitor: {stats['total_steps']} steps monitored
✓ FibonacciScheduler: {model.fib_scheduler.step_count} steps
✓ PACLazySystem: {model.pac_system.stats}
✓ TransitionMatrix: {model.transitions.num_transitions()} learned

Key insight: Transitions provide learned patterns,
embeddings provide semantic fallback.
""")


if __name__ == "__main__":
    test_integrated()
