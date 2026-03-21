"""
POC-022 Experiment 01b: WikiText Real Data Validation
Extends exp_01 with real language data instead of synthetic.

Optimized: Pre-downloads and caches tokenized data.
"""

import json
import time
import gc
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = Path(__file__).parent.parent / "data_cache"


def get_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


class GPUTransitionMatrix:
    """GPU-accelerated transition matrix - memory efficient for large vocab."""
    
    def __init__(self, vocab_size: int, device: torch.device):
        self.vocab_size = vocab_size
        self.device = device
        self.counts = {}
        self.from_totals = torch.zeros(vocab_size, device=device)
        # Sparse representation allows larger sizes
        self.hot_size = min(10000, vocab_size)
        self.top_k_per_row = 500  # Store top-k predictions per row
        self.hot_matrix = None
        self.hot_tokens = None
        
    def learn_batch(self, sequences: torch.Tensor):
        from_tokens = sequences[:, :-1].reshape(-1)
        to_tokens = sequences[:, 1:].reshape(-1)
        
        pairs = from_tokens.long() * self.vocab_size + to_tokens.long()
        unique_pairs, counts = pairs.unique(return_counts=True)
        
        for pair, count in zip(unique_pairs.tolist(), counts.tolist()):
            from_id = pair // self.vocab_size
            to_id = pair % self.vocab_size
            self.counts[(from_id, to_id)] = self.counts.get((from_id, to_id), 0) + count
            
        from_unique, from_counts = from_tokens.unique(return_counts=True)
        self.from_totals.scatter_add_(0, from_unique.long(), from_counts.float())
        
    def build_hot_matrix(self):
        nonzero_mask = self.from_totals > 0
        if nonzero_mask.sum() == 0:
            return
            
        top_counts, top_indices = self.from_totals.topk(
            min(self.hot_size, int(nonzero_mask.sum().item()))
        )
        
        self.hot_tokens = top_indices
        # Sparse representation: only store top-k targets per source
        # Shape: (hot_size, top_k_per_row) for indices and values
        self.hot_indices = torch.zeros(len(top_indices), self.top_k_per_row, device=self.device, dtype=torch.long)
        self.hot_probs = torch.zeros(len(top_indices), self.top_k_per_row, device=self.device)
        
        hot_list = top_indices.tolist()
        hot_map = {t: i for i, t in enumerate(hot_list)}
        
        # Group counts by source token
        from_counts = {}
        for (from_id, to_id), count in self.counts.items():
            if from_id in hot_map:
                if from_id not in from_counts:
                    from_counts[from_id] = []
                from_counts[from_id].append((to_id, count))
        
        # Build sparse matrix
        for from_id, targets in from_counts.items():
            idx = hot_map[from_id]
            # Sort by count, take top-k
            targets.sort(key=lambda x: x[1], reverse=True)
            top_targets = targets[:self.top_k_per_row]
            
            total = sum(c for _, c in targets)
            for j, (to_id, count) in enumerate(top_targets):
                self.hot_indices[idx, j] = to_id
                self.hot_probs[idx, j] = count / total
        
    def predict_batch(self, from_tokens: torch.Tensor, top_k: int = 10) -> tuple:
        if self.hot_probs is None:
            self.build_hot_matrix()
            
        batch_size = from_tokens.shape[0]
        predictions = torch.zeros(batch_size, top_k, device=self.device, dtype=torch.long)
        scores = torch.zeros(batch_size, top_k, device=self.device)
        
        if self.hot_tokens is None:
            return predictions, scores
        
        # Find which inputs are in hot set
        # Process in chunks to avoid OOM
        chunk_size = 50000
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            chunk = from_tokens[start:end]
            
            # Find matches
            matches = (chunk.unsqueeze(1) == self.hot_tokens.unsqueeze(0))
            has_match = matches.any(dim=1)
            
            if has_match.any():
                match_indices = matches[has_match].float().argmax(dim=1)
                chunk_preds = self.hot_indices[match_indices, :top_k]
                chunk_scores = self.hot_probs[match_indices, :top_k]
                
                predictions[start:end][has_match] = chunk_preds
                scores[start:end][has_match] = chunk_scores
            
        return predictions, scores
    
    def num_transitions(self) -> int:
        return len(self.counts)


def load_or_cache_wikitext(max_sequences: int = 5000, seq_length: int = 64):
    """Load WikiText-2 (smaller, faster) with caching."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"wikitext2_{max_sequences}_{seq_length}.pt"
    
    if cache_file.exists():
        print(f"Loading cached data from {cache_file.name}...")
        data = torch.load(cache_file)
        return data["sequences"].to(DEVICE), data["vocab_size"]
    
    print("First run - downloading and tokenizing WikiText-2...")
    
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    sequences = []
    current_tokens = []
    
    for item in dataset:
        text = item["text"].strip()
        if not text:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        current_tokens.extend(tokens)
        
        while len(current_tokens) >= seq_length:
            sequences.append(current_tokens[:seq_length])
            current_tokens = current_tokens[seq_length:]
            if len(sequences) >= max_sequences:
                break
        if len(sequences) >= max_sequences:
            break
    
    sequences_tensor = torch.tensor(sequences)
    vocab_size = tokenizer.vocab_size
    
    torch.save({"sequences": sequences_tensor, "vocab_size": vocab_size}, cache_file)
    print(f"Cached to {cache_file.name}")
    
    return sequences_tensor.to(DEVICE), vocab_size


def run_wikitext_test(max_sequences: int = 5000, epochs: int = 3) -> dict:
    """Run test on WikiText-2."""
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    start_time = time.time()
    memory_before = get_memory_mb()
    
    sequences, vocab_size = load_or_cache_wikitext(max_sequences)
    
    load_time = time.time() - start_time
    memory_after = get_memory_mb()
    
    print(f"  Loaded {len(sequences)} seqs, vocab={vocab_size:,}, {load_time:.1f}s, {memory_after:.1f}MB")
    
    trans_matrix = GPUTransitionMatrix(vocab_size, DEVICE)
    epoch_results = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        trans_matrix.learn_batch(sequences)
        trans_matrix.build_hot_matrix()
        
        from_tokens = sequences[:, :-1].reshape(-1)
        target_tokens = sequences[:, 1:].reshape(-1)
        
        predictions, _ = trans_matrix.predict_batch(from_tokens, top_k=10)
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
        "dataset": "wikitext-2",
        "vocab_size": vocab_size,
        "num_sequences": len(sequences),
        "load_time_s": load_time,
        "memory_mb": memory_after - memory_before,
        "final_hit_rate": epoch_results[-1]["hit_rate"],
        "transitions_learned": trans_matrix.num_transitions(),
        "epochs": epoch_results,
    }


def main():
    print("POC-022 Exp 01b: WikiText Real Data Validation")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    results = {
        "experiment": "exp_01b_wikitext_real",
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "metrics": {},
    }
    
    for max_seq in [2000, 5000, 10000]:
        print(f"\n--- {max_seq} sequences ---")
        metrics = run_wikitext_test(max_sequences=max_seq, epochs=3)
        results["metrics"][str(max_seq)] = metrics
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"exp_01b_wikitext_{timestamp}.json"
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    for seq_str, m in results["metrics"].items():
        status = "✅" if m["final_hit_rate"] >= 0.5 else "⚠️"
        print(f"{status} {int(seq_str):,} seqs: {m['final_hit_rate']:.1%} hit rate, {m['transitions_learned']:,} transitions")
    
    print(f"\nSaved: {results_path}")
    return results


if __name__ == "__main__":
    main()
