"""
Transition Matrix: GPU-accelerated n-gram learning.

Validated in:
- POC-022: 65% hit rate at 100K vocab, log learning curve R²=0.973
- POC-021: Continuous learning, hit rates 83-93% on patterns
- POC-012: +24.7% accuracy improvement during inference

No backprop - just counting transitions.
"""

import torch
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass 
class TransitionStats:
    """Statistics for the transition matrix."""
    total_transitions: int = 0
    unique_contexts: int = 0
    unique_transitions: int = 0
    sparsity: float = 0.0


class TransitionMatrix:
    """
    GPU-accelerated transition matrix for n-gram prediction.
    
    Stores P(next_token | context) for all observed contexts.
    Learning is pure counting - no gradients.
    
    Memory-efficient: sparse representation for cold contexts,
    dense for hot contexts.
    
    Usage:
        tm = TransitionMatrix(vocab_size=50257)
        tm.learn_batch(sequences)  # (batch, seq_len)
        token_ids, probs = tm.predict(context)
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        max_context_len: int = 5,
        hot_contexts: int = 10000,
        top_k_per_context: int = 100,
        device: str = 'cuda'
    ):
        self.vocab_size = vocab_size
        self.max_context_len = max_context_len
        self.hot_contexts = hot_contexts
        self.top_k_per_context = top_k_per_context
        self.device = device
        
        # Sparse storage: context_hash → {next_token → count}
        self.context_counts: Dict[int, Dict[int, int]] = {}
        self.context_totals: Dict[int, int] = {}
        
        # Hot context cache (GPU, dense)
        # Shape: (hot_contexts, top_k_per_context) for indices
        # Shape: (hot_contexts, top_k_per_context) for probs
        self.hot_indices: Optional[torch.Tensor] = None
        self.hot_probs: Optional[torch.Tensor] = None
        self.hot_context_map: Dict[int, int] = {}  # context_hash → hot_idx
        
        # Statistics
        self.stats = TransitionStats()
    
    def _context_hash(self, context: Tuple[int, ...]) -> int:
        """Hash a context tuple to an integer."""
        # Use Python's built-in tuple hash
        return hash(context)
    
    def learn(self, context: Tuple[int, ...], next_token: int):
        """
        Learn a single transition.
        
        Args:
            context: Tuple of preceding token IDs
            next_token: The token that followed
        """
        ctx_hash = self._context_hash(context)
        
        if ctx_hash not in self.context_counts:
            self.context_counts[ctx_hash] = {}
            self.context_totals[ctx_hash] = 0
            self.stats.unique_contexts += 1
        
        if next_token not in self.context_counts[ctx_hash]:
            self.context_counts[ctx_hash][next_token] = 0
            self.stats.unique_transitions += 1
        
        self.context_counts[ctx_hash][next_token] += 1
        self.context_totals[ctx_hash] += 1
        self.stats.total_transitions += 1
        
        # Invalidate hot cache
        if ctx_hash in self.hot_context_map:
            self.hot_indices = None  # Force rebuild
    
    def learn_batch(self, sequences: torch.Tensor, context_lengths: List[int] = None):
        """
        Learn transitions from batch of sequences.
        
        Args:
            sequences: (batch, seq_len) tensor of token IDs
            context_lengths: List of context lengths to learn (default: [1,2,3])
        """
        if context_lengths is None:
            context_lengths = list(range(1, min(self.max_context_len + 1, sequences.shape[1])))
        
        sequences = sequences.cpu()  # Work on CPU for dict operations
        batch_size, seq_len = sequences.shape
        
        for ctx_len in context_lengths:
            for b in range(batch_size):
                for i in range(seq_len - ctx_len):
                    context = tuple(sequences[b, i:i+ctx_len].tolist())
                    next_token = sequences[b, i+ctx_len].item()
                    self.learn(context, next_token)
    
    def _build_hot_cache(self):
        """
        Build GPU cache for most common contexts.
        
        This enables fast batch prediction.
        """
        # Find most common contexts
        context_freqs = [
            (ctx_hash, total) 
            for ctx_hash, total in self.context_totals.items()
        ]
        context_freqs.sort(key=lambda x: x[1], reverse=True)
        
        hot_contexts = context_freqs[:self.hot_contexts]
        
        # Build dense matrices
        n_hot = len(hot_contexts)
        self.hot_indices = torch.zeros(
            n_hot, self.top_k_per_context, 
            dtype=torch.long, device=self.device
        )
        self.hot_probs = torch.zeros(
            n_hot, self.top_k_per_context, 
            device=self.device
        )
        self.hot_context_map = {}
        
        for hot_idx, (ctx_hash, total) in enumerate(hot_contexts):
            self.hot_context_map[ctx_hash] = hot_idx
            
            # Get top-k transitions for this context
            counts = self.context_counts[ctx_hash]
            sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            
            for k, (token_id, count) in enumerate(sorted_items[:self.top_k_per_context]):
                self.hot_indices[hot_idx, k] = token_id
                self.hot_probs[hot_idx, k] = count / total
    
    def predict(
        self, 
        context: Tuple[int, ...],
        top_k: int = 10,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prediction for a context.
        
        Returns:
            (token_ids, probs) tensors
        """
        ctx_hash = self._context_hash(context)
        
        if ctx_hash not in self.context_counts:
            # Unknown context
            return (
                torch.tensor([], dtype=torch.long, device=self.device),
                torch.tensor([], device=self.device)
            )
        
        counts = self.context_counts[ctx_hash]
        total = self.context_totals[ctx_hash]
        
        # Get top-k
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        token_ids = torch.tensor(
            [t for t, c in sorted_items], 
            dtype=torch.long, device=self.device
        )
        
        # Apply temperature
        logits = torch.tensor(
            [c / total for t, c in sorted_items], 
            device=self.device
        ).log() / temperature
        probs = torch.softmax(logits, dim=0)
        
        return token_ids, probs
    
    def predict_batch(
        self,
        contexts: List[Tuple[int, ...]],
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch prediction using hot cache.
        
        Returns:
            (token_ids, probs) both shape (batch, top_k)
        """
        if self.hot_indices is None:
            self._build_hot_cache()
        
        batch_size = len(contexts)
        predictions = torch.zeros(
            batch_size, top_k, dtype=torch.long, device=self.device
        )
        probs = torch.zeros(batch_size, top_k, device=self.device)
        
        for i, context in enumerate(contexts):
            ctx_hash = self._context_hash(context)
            
            if ctx_hash in self.hot_context_map:
                hot_idx = self.hot_context_map[ctx_hash]
                predictions[i] = self.hot_indices[hot_idx, :top_k]
                probs[i] = self.hot_probs[hot_idx, :top_k]
            else:
                # Fallback to single prediction
                tids, ps = self.predict(context, top_k)
                if len(tids) > 0:
                    n = min(len(tids), top_k)
                    predictions[i, :n] = tids[:n]
                    probs[i, :n] = ps[:n]
        
        return predictions, probs
    
    def get_hit_rate(
        self, 
        sequences: torch.Tensor, 
        context_len: int = 2,
        top_k: int = 10
    ) -> float:
        """
        Compute hit rate: fraction of correct predictions in top-k.
        
        This is our main quality metric (POC-022 achieved 65% at 100K vocab).
        """
        sequences = sequences.cpu()
        batch_size, seq_len = sequences.shape
        
        hits = 0
        total = 0
        
        for b in range(batch_size):
            for i in range(seq_len - context_len):
                context = tuple(sequences[b, i:i+context_len].tolist())
                target = sequences[b, i+context_len].item()
                
                token_ids, probs = self.predict(context, top_k)
                
                if target in token_ids.tolist():
                    hits += 1
                total += 1
        
        return hits / total if total > 0 else 0.0
    
    def save(self, path: str):
        """Save transition matrix to disk."""
        import json
        
        data = {
            'vocab_size': self.vocab_size,
            'max_context_len': self.max_context_len,
            'stats': {
                'total_transitions': self.stats.total_transitions,
                'unique_contexts': self.stats.unique_contexts,
                'unique_transitions': self.stats.unique_transitions,
            },
            'contexts': {
                str(ctx_hash): {
                    'counts': counts,
                    'total': self.context_totals[ctx_hash]
                }
                for ctx_hash, counts in self.context_counts.items()
            }
        }
        
        with open(path, 'w') as f:
            json.dump(data, f)
        
        print(f"Saved {self.stats.unique_contexts} contexts to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cuda') -> 'TransitionMatrix':
        """Load transition matrix from disk."""
        import json
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        tm = cls(
            vocab_size=data['vocab_size'],
            max_context_len=data['max_context_len'],
            device=device
        )
        
        for ctx_hash_str, ctx_data in data['contexts'].items():
            ctx_hash = int(ctx_hash_str)
            tm.context_counts[ctx_hash] = {
                int(k): v for k, v in ctx_data['counts'].items()
            }
            tm.context_totals[ctx_hash] = ctx_data['total']
        
        tm.stats.total_transitions = data['stats']['total_transitions']
        tm.stats.unique_contexts = data['stats']['unique_contexts']
        tm.stats.unique_transitions = data['stats']['unique_transitions']
        
        print(f"Loaded {tm.stats.unique_contexts} contexts from {path}")
        return tm


if __name__ == "__main__":
    # Quick test
    tm = TransitionMatrix(vocab_size=100, device='cpu')
    
    # Learn some transitions
    tm.learn((0, 1), 2)
    tm.learn((0, 1), 2)
    tm.learn((0, 1), 3)
    tm.learn((1, 2), 4)
    
    print(f"Stats: {tm.stats}")
    
    # Predict
    tids, probs = tm.predict((0, 1))
    print(f"Predictions for (0, 1): {list(zip(tids.tolist(), probs.tolist()))}")
