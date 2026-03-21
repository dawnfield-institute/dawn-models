"""
POC-022 Experiment 01: Vocabulary Scaling
Test PAC learning at increasing vocabulary sizes: 50K → 100K → 200K

GPU-optimized implementation using PyTorch throughout.

Falsification: Hit rate < 50% at 100K vocabulary
"""

import json
import time
import gc
from datetime import datetime
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

# Dawn Field Constants
PHI = 1.618033988749895
XI = 0.0618
PHI_XI = PHI * XI

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def generate_embeddings(vocab_size: int, embed_dim: int = 768) -> torch.Tensor:
    """Generate normalized embeddings on GPU."""
    torch.manual_seed(42)
    embeddings = torch.randn(vocab_size, embed_dim, device=DEVICE)
    embeddings = F.normalize(embeddings, dim=1)
    return embeddings


def generate_sequences(vocab_size: int, num_sequences: int = 1000, seq_length: int = 32) -> torch.Tensor:
    """Generate token sequences using Zipf distribution on GPU."""
    torch.manual_seed(42)
    # Zipf-like distribution
    probs = 1.0 / torch.arange(1, vocab_size + 1, device=DEVICE, dtype=torch.float32)
    probs = probs / probs.sum()
    
    # Sample sequences
    sequences = torch.multinomial(probs, num_sequences * seq_length, replacement=True)
    return sequences.view(num_sequences, seq_length)


class GPUTransitionMatrix:
    """GPU-accelerated transition matrix for next-token prediction."""
    
    def __init__(self, vocab_size: int, device: torch.device):
        self.vocab_size = vocab_size
        self.device = device
        # Sparse representation: only store observed transitions
        self.counts = {}  # (from_id, to_id) -> count
        self.from_totals = torch.zeros(vocab_size, device=device)
        
        # For batch prediction, we'll build a dense matrix for hot tokens
        self.hot_size = min(5000, vocab_size)  # Top tokens for dense lookup
        self.hot_matrix = None
        
    def learn_batch(self, sequences: torch.Tensor):
        """Learn transitions from batch of sequences."""
        # sequences: (batch, seq_len)
        from_tokens = sequences[:, :-1].reshape(-1)
        to_tokens = sequences[:, 1:].reshape(-1)
        
        # Count transitions on GPU
        pairs = from_tokens * self.vocab_size + to_tokens
        unique_pairs, counts = pairs.unique(return_counts=True)
        
        # Update sparse counts
        for pair, count in zip(unique_pairs.tolist(), counts.tolist()):
            from_id = pair // self.vocab_size
            to_id = pair % self.vocab_size
            key = (from_id, to_id)
            self.counts[key] = self.counts.get(key, 0) + count
            
        # Update from totals
        from_unique, from_counts = from_tokens.unique(return_counts=True)
        self.from_totals.scatter_add_(0, from_unique, from_counts.float())
        
    def build_hot_matrix(self):
        """Build dense matrix for most common source tokens."""
        # Find most common source tokens
        nonzero_mask = self.from_totals > 0
        if nonzero_mask.sum() == 0:
            return
            
        # Get top tokens by frequency
        top_counts, top_indices = self.from_totals.topk(min(self.hot_size, nonzero_mask.sum().item()))
        
        # Build dense matrix for these
        self.hot_tokens = top_indices
        self.hot_matrix = torch.zeros(len(top_indices), self.vocab_size, device=self.device)
        
        hot_set = set(top_indices.tolist())
        for (from_id, to_id), count in self.counts.items():
            if from_id in hot_set:
                hot_idx = (self.hot_tokens == from_id).nonzero(as_tuple=True)[0]
                if len(hot_idx) > 0:
                    self.hot_matrix[hot_idx[0], to_id] = count
        
        # Normalize rows
        row_sums = self.hot_matrix.sum(dim=1, keepdim=True).clamp(min=1)
        self.hot_matrix = self.hot_matrix / row_sums
        
    def predict_batch(self, from_tokens: torch.Tensor, top_k: int = 10) -> tuple:
        """Predict next tokens for a batch of source tokens."""
        if self.hot_matrix is None:
            self.build_hot_matrix()
            
        batch_size = from_tokens.shape[0]
        
        # Map to hot indices
        hot_mask = (from_tokens.unsqueeze(1) == self.hot_tokens.unsqueeze(0)).any(dim=1)
        
        # Get predictions for hot tokens
        predictions = torch.zeros(batch_size, top_k, device=self.device, dtype=torch.long)
        scores = torch.zeros(batch_size, top_k, device=self.device)
        
        if hot_mask.any():
            hot_indices = from_tokens[hot_mask]
            # Find positions in hot matrix
            pos = (hot_indices.unsqueeze(1) == self.hot_tokens.unsqueeze(0)).float().argmax(dim=1)
            probs = self.hot_matrix[pos]
            top_scores, top_ids = probs.topk(top_k, dim=1)
            
            predictions[hot_mask] = top_ids
            scores[hot_mask] = top_scores
            
        return predictions, scores
    
    def num_transitions(self) -> int:
        return len(self.counts)


def run_scale_test(vocab_size: int, num_sequences: int = 500, epochs: int = 3) -> dict:
    """Run scale test for a single vocabulary size."""
    print(f"\n{'='*60}")
    print(f"Testing vocabulary size: {vocab_size:,}")
    print(f"{'='*60}")
    
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    start_time = time.time()
    memory_before = get_memory_mb()
    
    # Generate data on GPU
    print("  Generating embeddings on GPU...")
    embeddings = generate_embeddings(vocab_size)
    
    print("  Generating sequences on GPU...")
    sequences = generate_sequences(vocab_size, num_sequences)
    
    setup_time = time.time() - start_time
    memory_after = get_memory_mb()
    print(f"  Setup: {setup_time:.1f}s, Memory: {memory_after:.1f}MB")
    
    # Create transition matrix
    trans_matrix = GPUTransitionMatrix(vocab_size, DEVICE)
    
    epoch_results = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Learn transitions
        trans_matrix.learn_batch(sequences)
        trans_matrix.build_hot_matrix()
        
        # Evaluate hit rate
        from_tokens = sequences[:, :-1].reshape(-1)
        target_tokens = sequences[:, 1:].reshape(-1)
        
        predictions, _ = trans_matrix.predict_batch(from_tokens, top_k=10)
        
        # Check if target is in top-k predictions
        hits = (predictions == target_tokens.unsqueeze(1)).any(dim=1)
        hit_rate = hits.float().mean().item()
        
        epoch_time = time.time() - epoch_start
        tokens_per_sec = len(from_tokens) / epoch_time
        
        epoch_results.append({
            "epoch": epoch + 1,
            "hit_rate": hit_rate,
            "time_s": epoch_time,
            "tokens_per_sec": tokens_per_sec,
        })
        
        print(f"  Epoch {epoch+1}: hit_rate={hit_rate:.3f}, {tokens_per_sec:,.0f} tok/s")
    
    return {
        "vocab_size": vocab_size,
        "setup_time_s": setup_time,
        "memory_mb": memory_after - memory_before,
        "final_hit_rate": epoch_results[-1]["hit_rate"],
        "transitions_learned": trans_matrix.num_transitions(),
        "epochs": epoch_results,
    }


def main():
    """Run the vocabulary scaling experiment."""
    vocab_sizes = [10_000, 25_000, 50_000, 100_000]
    
    print("POC-022: Vocabulary Scaling Test (GPU-Optimized)")
    print("=" * 60)
    print(f"Testing vocabulary sizes: {vocab_sizes}")
    print(f"Device: {DEVICE}")
    print(f"Falsification threshold: hit_rate < 50% at 100K")
    
    results = {
        "experiment": "exp_01_vocab_scaling",
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "config": {"vocab_sizes": vocab_sizes},
        "metrics": {},
    }
    
    for vocab_size in vocab_sizes:
        metrics = run_scale_test(vocab_size, num_sequences=500, epochs=3)
        results["metrics"][str(vocab_size)] = metrics
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"exp_01_vocab_scaling_{timestamp}.json"
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for vocab_str, metrics in results["metrics"].items():
        hit_rate = metrics["final_hit_rate"]
        status = "✅" if hit_rate >= 0.5 else "⚠️ FALSIFIED"
        print(f"{status} vocab={int(vocab_str):,}: hit_rate={hit_rate:.1%}, transitions={metrics['transitions_learned']:,}")
    
    print(f"\nResults saved to: {results_path}")
    return results


if __name__ == "__main__":
    main()
