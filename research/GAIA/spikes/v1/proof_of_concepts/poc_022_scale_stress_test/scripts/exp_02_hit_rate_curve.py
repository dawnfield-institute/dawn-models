"""
POC-022 Experiment 02: Hit Rate Curve
Measure prediction accuracy as a function of training data size.

Key question: How does hit rate scale with more training data?
If PAC is correct, should see logarithmic improvement (diminishing returns).
"""

import json
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from transformers import GPT2Tokenizer
from datasets import load_dataset

# Dawn Field constants
PHI = 1.6180339887
XI = 0.0618
PHI_INV = 1 / PHI  # 0.618...


class GPUTransitionMatrix:
    """GPU-accelerated sparse transition matrix."""
    
    def __init__(self, vocab_size: int, device: torch.device):
        self.vocab_size = vocab_size
        self.device = device
        self.counts = {}
        self.from_totals = torch.zeros(vocab_size, device=device)
        self.hot_size = min(10000, vocab_size)
        self.top_k_per_row = 500
        self.hot_indices = None
        self.hot_probs = None
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
        self.hot_indices = torch.zeros(len(top_indices), self.top_k_per_row, 
                                       device=self.device, dtype=torch.long)
        self.hot_probs = torch.zeros(len(top_indices), self.top_k_per_row, 
                                     device=self.device)
        
        hot_list = top_indices.tolist()
        hot_map = {t: i for i, t in enumerate(hot_list)}
        
        from_counts = {}
        for (from_id, to_id), count in self.counts.items():
            if from_id in hot_map:
                if from_id not in from_counts:
                    from_counts[from_id] = []
                from_counts[from_id].append((to_id, count))
        
        for from_id, targets in from_counts.items():
            idx = hot_map[from_id]
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
        
        chunk_size = 50000
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            chunk = from_tokens[start:end]
            
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


def load_wikitext(device: torch.device) -> tuple:
    """Load full WikiText-2 dataset."""
    print("Loading WikiText-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    dataset = load_dataset('wikitext', 'wikitext-2-v1', split='train', trust_remote_code=True)
    all_text = ' '.join([t for t in dataset['text'] if t.strip()])
    all_tokens = tokenizer.encode(all_text)
    return torch.tensor(all_tokens, device=device), tokenizer.vocab_size


def measure_hit_rate(all_tokens: torch.Tensor, vocab_size: int, 
                     num_train_seqs: int, num_test_seqs: int,
                     seq_len: int, device: torch.device) -> dict:
    """
    Train on num_train_seqs, test on separate num_test_seqs.
    Returns detailed metrics.
    """
    t0 = time.time()
    
    total_seqs_needed = num_train_seqs + num_test_seqs
    max_seqs = len(all_tokens) // seq_len
    
    if total_seqs_needed > max_seqs:
        # Scale down proportionally
        ratio = max_seqs / total_seqs_needed
        num_train_seqs = int(num_train_seqs * ratio)
        num_test_seqs = int(num_test_seqs * ratio)
    
    # Split data
    train_end = num_train_seqs * seq_len
    test_end = train_end + num_test_seqs * seq_len
    
    train_tokens = all_tokens[:train_end].reshape(num_train_seqs, seq_len)
    test_tokens = all_tokens[train_end:test_end].reshape(num_test_seqs, seq_len)
    
    # Train
    matrix = GPUTransitionMatrix(vocab_size, device)
    matrix.learn_batch(train_tokens)
    
    # Test on HELD-OUT data
    from_tokens = test_tokens[:, :-1].reshape(-1)
    to_tokens = test_tokens[:, 1:].reshape(-1)
    
    preds, scores = matrix.predict_batch(from_tokens, top_k=10)
    hits = (preds == to_tokens.unsqueeze(1)).any(dim=1)
    
    hit_rate = hits.float().mean().item()
    elapsed = time.time() - t0
    
    return {
        "hit_rate": hit_rate,
        "train_seqs": num_train_seqs,
        "test_seqs": num_test_seqs,
        "transitions": matrix.num_transitions(),
        "unique_tokens_seen": int((matrix.from_totals > 0).sum().item()),
        "time_s": elapsed,
    }


def fit_curve(x_vals: list, hit_rates: list) -> dict:
    """
    Fit the hit rate curve and check for φ relationships.
    """
    x = np.array(x_vals, dtype=float)
    y = np.array(hit_rates, dtype=float)
    log_x = np.log(x)
    
    # Linear fit
    linear_coeffs = np.polyfit(x, y, 1)
    linear_pred = np.polyval(linear_coeffs, x)
    linear_r2 = 1 - np.sum((y - linear_pred)**2) / np.sum((y - y.mean())**2)
    
    # Log fit: y = a + b*log(x)
    log_coeffs = np.polyfit(log_x, y, 1)
    log_pred = np.polyval(log_coeffs, log_x)
    log_r2 = 1 - np.sum((y - log_pred)**2) / np.sum((y - y.mean())**2)
    
    # Power law: log(y) = log(a) + b*log(x)
    log_y = np.log(y + 1e-10)
    power_coeffs = np.polyfit(log_x, log_y, 1)
    power_pred = np.exp(np.polyval(power_coeffs, log_x))
    power_r2 = 1 - np.sum((y - power_pred)**2) / np.sum((y - y.mean())**2)
    
    # Check for φ relationships
    log_slope = log_coeffs[0]  # How hit rate grows with log(data)
    
    return {
        "linear": {"r2": linear_r2, "slope": linear_coeffs[0], "intercept": linear_coeffs[1]},
        "logarithmic": {"r2": log_r2, "slope": log_coeffs[0], "intercept": log_coeffs[1]},
        "power_law": {"r2": power_r2, "exponent": power_coeffs[0], "scale": np.exp(power_coeffs[1])},
        "phi_analysis": {
            "log_slope": log_slope,
            "slope_over_phi_inv": log_slope / PHI_INV if PHI_INV else 0,
        },
        "best_fit": max([
            ("linear", linear_r2),
            ("logarithmic", log_r2), 
            ("power_law", power_r2)
        ], key=lambda x: x[1])[0]
    }


def main():
    """Run hit rate curve experiment with real data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("POC-022 Exp 02: Hit Rate vs Training Data Size")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"φ = {PHI:.6f}, 1/φ = {PHI_INV:.6f}")
    print()
    
    # Load data once
    all_tokens, vocab_size = load_wikitext(device)
    print(f"Total tokens: {len(all_tokens):,}, vocab: {vocab_size:,}")
    print()
    
    # Vary training data size, keep test size fixed
    seq_len = 64
    test_seqs = 500  # Fixed test set
    train_sizes = [100, 250, 500, 1000, 2000, 4000, 8000, 16000]
    
    results = {
        "experiment": "exp_02_hit_rate_curve",
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "test_seqs": test_seqs,
        "constants": {"PHI": PHI, "PHI_INV": PHI_INV, "XI": XI},
        "data_points": [],
    }
    
    hit_rates = []
    
    for train_seqs in train_sizes:
        print(f"train={train_seqs:>6,} seqs...", end=" ", flush=True)
        
        metrics = measure_hit_rate(all_tokens, vocab_size, train_seqs, test_seqs, 
                                   seq_len, device)
        hit_rates.append(metrics["hit_rate"])
        
        results["data_points"].append({
            "train_sequences": train_seqs,
            **metrics
        })
        
        print(f"hit={metrics['hit_rate']:.3f}, trans={metrics['transitions']:,}, "
              f"tokens_seen={metrics['unique_tokens_seen']:,}")
    
    # Fit curve
    print("\n" + "=" * 60)
    print("CURVE FITTING")
    print("=" * 60)
    
    curve_fit = fit_curve(train_sizes, hit_rates)
    results["curve_fit"] = curve_fit
    
    print(f"\nBest fit: {curve_fit['best_fit']}")
    print(f"\nR² values:")
    print(f"  Linear:      {curve_fit['linear']['r2']:.4f}")
    print(f"  Logarithmic: {curve_fit['logarithmic']['r2']:.4f}")
    print(f"  Power law:   {curve_fit['power_law']['r2']:.4f}")
    
    print(f"\nLogarithmic fit: hit_rate = {curve_fit['logarithmic']['intercept']:.3f} + "
          f"{curve_fit['logarithmic']['slope']:.4f} * log(train_seqs)")
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"exp_02_hit_rate_curve_{timestamp}.json"
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved: {results_path}")
    
    return results


if __name__ == "__main__":
    main()
