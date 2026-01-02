"""
POC-023 Experiment 03: Depth Harmonics
Bridge between Prime Harmonic Manifold and Language PAC Trees.

Key insight: Multiple depths form a "prediction chord" - each depth contributes a note.
When scales harmonize (consonance), predictions are reliable.
When scales clash (dissonance), we're in sparse territory.

This experiment:
1. Build depth transition matrices (like prime gap chords)
2. Compute eigenvalue structure
3. Check for phi or 1/2 signatures
4. Measure chord concentration (how "in tune" are depths?)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import torch

# Dawn Field constants
PHI = 1.6180339887
PHI_INV = 1 / PHI  # 0.618...
XI = 0.0618
HALF = 0.5


class PACNode:
    """PAC node - children shared through parent's children dict."""
    __slots__ = ['token_id', 'counts', 'total', 'children', 'depth']
    
    def __init__(self, token_id: int, depth: int):
        self.token_id = token_id
        self.depth = depth
        self.counts = {}
        self.total = 0
        self.children = {}
    
    def observe(self, target: int):
        self.counts[target] = self.counts.get(target, 0) + 1
        self.total += 1
    
    def get_or_create_child(self, token_id: int) -> 'PACNode':
        if token_id not in self.children:
            self.children[token_id] = PACNode(token_id, self.depth + 1)
        return self.children[token_id]
    
    def predict_top_k(self, k: int = 10) -> list:
        if not self.counts:
            return []
        sorted_preds = sorted(self.counts.items(), key=lambda x: -x[1])
        return [t for t, c in sorted_preds[:k]]


class PACTree:
    """PAC tree for depth harmonic analysis."""
    
    def __init__(self, vocab_size: int, max_depth: int, device: torch.device):
        self.vocab_size = vocab_size
        self.max_depth = max_depth
        self.device = device
        self.roots = {}
        self.node_count = 0
    
    def learn(self, tokens: torch.Tensor):
        tokens_list = tokens.cpu().tolist()
        
        for i in range(1, len(tokens_list)):
            target = tokens_list[i]
            
            for depth in range(1, min(self.max_depth + 1, i + 1)):
                context_start = i - depth
                context = tokens_list[context_start:i]
                
                first_token = context[0]
                if first_token not in self.roots:
                    self.roots[first_token] = PACNode(first_token, 0)
                    self.node_count += 1
                
                current = self.roots[first_token]
                for token in context[1:]:
                    if token not in current.children:
                        self.node_count += 1
                    current = current.get_or_create_child(token)
                current.observe(target)
    
    def predict_at_depth(self, context: list, depth: int, top_k: int = 10) -> list:
        """Get prediction at specific depth."""
        if depth > len(context) or depth < 1:
            return []
        
        ctx = context[-depth:]
        first_token = ctx[0]
        
        if first_token not in self.roots:
            return []
        
        current = self.roots[first_token]
        for token in ctx[1:]:
            if token not in current.children:
                return []
            current = current.children[token]
        
        return current.predict_top_k(top_k)


def load_wikitext() -> tuple:
    """Load WikiText-2 from cache."""
    cache_path = Path(__file__).parent.parent.parent / "poc_022_scale_stress_test" / "data_cache" / "wikitext2_10000_64.pt"
    
    if cache_path.exists():
        print(f"Loading from cache: {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        tokens_2d = data["sequences"]
        vocab_size = data["vocab_size"]
        tokens = tokens_2d.flatten().cpu().tolist()
        print(f"Loaded {len(tokens):,} tokens from cache")
        return tokens, vocab_size
    else:
        raise FileNotFoundError(f"Cache not found: {cache_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("POC-023 Exp 03: Depth Harmonics")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"PHI = {PHI:.6f}, 1/PHI = {PHI_INV:.6f}")
    print()
    print("Bridging Prime Harmonic Manifold to Language PAC Trees")
    print()
    
    # Load data
    all_tokens, vocab_size = load_wikitext()
    print(f"Total tokens: {len(all_tokens):,}, vocab: {vocab_size:,}")
    
    split = int(len(all_tokens) * 0.8)
    train_tokens = all_tokens[:split]
    test_tokens = all_tokens[split:]
    print(f"Train: {len(train_tokens):,}, Test: {len(test_tokens):,}")
    
    max_depth = 5
    
    # Train tree
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    t0 = time.time()
    tree = PACTree(vocab_size, max_depth, device)
    tree.learn(torch.tensor(train_tokens, device=device))
    print(f"Training time: {time.time() - t0:.1f}s")
    print(f"Total nodes: {tree.node_count:,}")
    
    # Build depth transition matrix
    print("\n" + "=" * 60)
    print("DEPTH TRANSITION MATRIX")
    print("=" * 60)
    print("Counting: when depth d1 predicts token X, what does depth d2 predict?")
    
    t1 = time.time()
    
    # Transition matrix: transitions[d1][d2] = {"agree": count, "disagree": count}
    # This measures how often depth d2 agrees with depth d1
    depth_transitions = defaultdict(lambda: defaultdict(lambda: {"agree": 0, "disagree": 0}))
    
    # Also track: chord concentration per prediction
    # How many depths agree on the same top-1?
    chord_concentrations = []
    
    # Track marginal gains with more precision
    depth_hit_rates = {d: {"hits": 0, "total": 0} for d in range(1, max_depth + 1)}
    
    sample_size = min(50000, len(test_tokens) - max_depth - 1)
    
    for i in range(max_depth + 1, max_depth + 1 + sample_size):
        target = test_tokens[i]
        context = test_tokens[:i]
        
        # Get top-1 prediction at each depth
        top1_by_depth = {}
        top10_by_depth = {}
        
        for d in range(1, max_depth + 1):
            preds = tree.predict_at_depth(context, d, top_k=10)
            if preds:
                top1_by_depth[d] = preds[0]
                top10_by_depth[d] = set(preds)
                
                # Track hit rate at this depth
                depth_hit_rates[d]["total"] += 1
                if target in top10_by_depth[d]:
                    depth_hit_rates[d]["hits"] += 1
        
        # Build transition counts between all depth pairs
        for d1 in top1_by_depth:
            for d2 in top1_by_depth:
                if d1 != d2:
                    if top1_by_depth[d1] == top1_by_depth[d2]:
                        depth_transitions[d1][d2]["agree"] += 1
                    else:
                        depth_transitions[d1][d2]["disagree"] += 1
        
        # Chord concentration: how many unique predictions?
        if len(top1_by_depth) >= 2:
            unique_preds = len(set(top1_by_depth.values()))
            total_depths = len(top1_by_depth)
            concentration = 1.0 - (unique_preds - 1) / (total_depths - 1)  # 1 = perfect agreement
            chord_concentrations.append(concentration)
    
    eval_time = time.time() - t1
    print(f"Evaluation time: {eval_time:.1f}s")
    
    # Build and analyze transition matrix
    print("\nDepth Transition Matrix (agreement rate d1 -> d2):")
    
    depths = list(range(1, max_depth + 1))
    matrix = []
    
    print(f"\n{'':>8}", end="")
    for d2 in depths:
        print(f"{d2:>8}", end="")
    print()
    print("-" * (8 + 8 * len(depths)))
    
    for d1 in depths:
        row = []
        print(f"d{d1:>6} |", end="")
        for d2 in depths:
            if d1 == d2:
                rate = 1.0  # Self-agreement
            else:
                stats = depth_transitions[d1][d2]
                total = stats["agree"] + stats["disagree"]
                rate = stats["agree"] / total if total > 0 else 0
            row.append(rate)
            print(f"{rate:>8.3f}", end="")
        matrix.append(row)
        print()
    
    # Compute eigenvalues of transition matrix
    print("\n" + "=" * 60)
    print("EIGENVALUE ANALYSIS")
    print("=" * 60)
    
    matrix_tensor = torch.tensor(matrix, dtype=torch.float64)
    eigenvalues = torch.linalg.eigvals(matrix_tensor)
    eigenvalues_real = eigenvalues.real.sort(descending=True).values
    
    print("\nEigenvalues (real parts, sorted):")
    for i, ev in enumerate(eigenvalues_real):
        # Check proximity to key constants
        diff_phi = abs(ev.item() - PHI_INV)
        diff_half = abs(ev.item() - HALF)
        diff_one = abs(ev.item() - 1.0)
        
        marker = ""
        if diff_one < 0.01:
            marker = " <- 1.0"
        elif diff_phi < 0.02:
            marker = f" <- 1/PHI ({diff_phi:.4f} error)"
        elif diff_half < 0.02:
            marker = f" <- 1/2 ({diff_half:.4f} error)"
        
        print(f"  lambda_{i+1} = {ev.item():.6f}{marker}")
    
    # Key eigenvalue ratios
    if len(eigenvalues_real) >= 2:
        ratio_1_2 = eigenvalues_real[0].item() / eigenvalues_real[1].item()
        print(f"\nEigenvalue ratio lambda_1/lambda_2 = {ratio_1_2:.4f}")
        if abs(ratio_1_2 - PHI) < 0.1:
            print(f"  Close to PHI! (error = {abs(ratio_1_2 - PHI):.4f})")
    
    # Chord concentration analysis
    print("\n" + "=" * 60)
    print("CHORD CONCENTRATION (Harmonic Coherence)")
    print("=" * 60)
    
    if chord_concentrations:
        mean_conc = sum(chord_concentrations) / len(chord_concentrations)
        
        # Bucket by concentration
        high_conc = [c for c in chord_concentrations if c > 0.8]
        med_conc = [c for c in chord_concentrations if 0.4 <= c <= 0.8]
        low_conc = [c for c in chord_concentrations if c < 0.4]
        
        print(f"\nMean chord concentration: {mean_conc:.3f}")
        print(f"  (1.0 = all depths agree, 0.0 = all depths differ)")
        print(f"\nDistribution:")
        print(f"  High (>0.8): {len(high_conc):,} ({100*len(high_conc)/len(chord_concentrations):.1f}%)")
        print(f"  Medium:      {len(med_conc):,} ({100*len(med_conc)/len(chord_concentrations):.1f}%)")
        print(f"  Low (<0.4):  {len(low_conc):,} ({100*len(low_conc)/len(chord_concentrations):.1f}%)")
    
    # Marginal gain ratio analysis
    print("\n" + "=" * 60)
    print("MARGINAL GAIN RATIO (Checking for PHI)")
    print("=" * 60)
    
    hit_rates = []
    for d in range(1, max_depth + 1):
        stats = depth_hit_rates[d]
        if stats["total"] > 0:
            hr = stats["hits"] / stats["total"]
            hit_rates.append((d, hr))
            print(f"  Depth {d}: {hr:.4f} ({stats['total']:,} samples)")
    
    if len(hit_rates) >= 2:
        print("\nMarginal gains and ratios:")
        gains = []
        for i in range(1, len(hit_rates)):
            d_prev, hr_prev = hit_rates[i-1]
            d_curr, hr_curr = hit_rates[i]
            gain = hr_curr - hr_prev
            gains.append(gain)
            print(f"  Depth {d_prev} -> {d_curr}: gain = {gain:+.4f}")
        
        if len(gains) >= 2:
            print("\nGain decay ratios:")
            ratios = []
            for i in range(len(gains) - 1):
                if gains[i+1] != 0:
                    ratio = gains[i] / gains[i+1]
                    ratios.append(ratio)
                    
                    phi_error = abs(ratio - PHI)
                    marker = f" <- PHI! (error {phi_error:.3f})" if phi_error < 0.2 else ""
                    print(f"  gain_{i+1}/gain_{i+2} = {ratio:.3f}{marker}")
            
            if ratios:
                mean_ratio = sum(ratios) / len(ratios)
                print(f"\nMean decay ratio: {mean_ratio:.3f}")
                print(f"PHI = {PHI:.3f}, error = {abs(mean_ratio - PHI):.3f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SYNTHESIS: Prime Harmonic <-> Language PAC Bridge")
    print("=" * 60)
    
    print("""
Prime Harmonic Manifold found:
  - Gap transition matrices with eigenvalue -> 1/2
  - Primes 97 std devs from random (non-trivial structure)
  - phi at SEC criticality threshold

Language PAC Depth Harmonics found:
  - Depth transition matrix eigenvalues (see above)
  - Chord concentration measures harmonic coherence
  - Marginal gain decay ratio (approaching phi?)

The bridge: SCALE HARMONY
  - In primes: consecutive gaps form chords
  - In language: consecutive depths form chords
  - Both measure: when does structure crystallize?
""")
    
    # Save results
    output = {
        "experiment": "exp_03_depth_harmonics",
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "vocab_size": vocab_size,
        "train_tokens": len(train_tokens),
        "test_tokens": len(test_tokens),
        "sample_size": sample_size,
        "max_depth": max_depth,
        "constants": {"PHI": PHI, "PHI_INV": PHI_INV, "HALF": HALF},
        "transition_matrix": matrix,
        "eigenvalues": [ev.item() for ev in eigenvalues_real],
        "mean_chord_concentration": mean_conc if chord_concentrations else None,
        "depth_hit_rates": {d: {"hits": s["hits"], "total": s["total"], 
                                "rate": s["hits"]/s["total"] if s["total"] > 0 else 0}
                           for d, s in depth_hit_rates.items()},
        "node_count": tree.node_count,
        "eval_time_s": eval_time,
    }
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"exp_03_depth_harmonics_{timestamp}.json"
    
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nSaved: {results_path}")
    
    return output


if __name__ == "__main__":
    main()
