"""
exp_02_depth_convergence.py - Does population ratio converge to φ at depth?

POC-024: φ Structure Validation

Key observation from exp_01:
  Depth 3 ratio = 0.6218 ≈ 1/φ exactly!
  Other depths don't match.

Hypothesis: φ emerges asymptotically as depth increases?

Test: Build deeper trees, check ratio convergence.
"""

import json
import random
import math
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import numpy as np

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI


class DeepPACTree:
    """PAC tree with configurable max depth"""
    
    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth
        self.children = defaultdict(lambda: defaultdict(int))
        
    def learn_sequence(self, tokens: list):
        for i in range(len(tokens) - 1):
            child = tokens[i + 1]
            for d in range(min(self.max_depth, i + 1)):
                context = tuple(tokens[max(0, i - d):i + 1])
                self.children[context][child] += 1
    
    def get_depth_populations(self) -> dict:
        by_depth = defaultdict(set)
        for context in self.children.keys():
            depth = len(context) - 1
            by_depth[depth].add(context)
        return {d: len(nodes) for d, nodes in sorted(by_depth.items())}


def generate_corpus(vocab_size: int, length: int, seed: int) -> list:
    random.seed(seed)
    transitions = {}
    for i in range(vocab_size):
        n_successors = random.randint(2, 8)
        successors = random.sample(range(vocab_size), n_successors)
        weights = [1.0 / (j + 1) ** 1.5 for j in range(n_successors)]
        total = sum(weights)
        transitions[i] = [(s, w/total) for s, w in zip(successors, weights)]
    
    tokens = [random.randint(0, vocab_size - 1)]
    for _ in range(length - 1):
        current = tokens[-1]
        successors, probs = zip(*transitions[current])
        next_token = random.choices(successors, weights=probs)[0]
        tokens.append(next_token)
    return tokens


def main():
    print("=" * 70)
    print("POC-024 Exp 02: Depth Convergence to φ")
    print("=" * 70)
    print(f"\n1/φ = {PHI_INV:.6f}")
    
    # Test with deeper trees and multiple seeds
    max_depth = 10
    n_seeds = 5
    corpus_length = 200000
    vocab_size = 500
    
    all_ratios = defaultdict(list)
    
    for seed in range(n_seeds):
        print(f"\n--- Seed {seed} ---")
        corpus = generate_corpus(vocab_size, corpus_length, seed=seed)
        
        tree = DeepPACTree(max_depth=max_depth)
        tree.learn_sequence(corpus)
        
        populations = tree.get_depth_populations()
        depths = sorted(populations.keys())
        
        for i in range(len(depths) - 1):
            d = depths[i]
            ratio = populations[d] / populations[depths[i+1]]
            all_ratios[d].append(ratio)
            print(f"  Depth {d}: N={populations[d]:>6}, ratio={ratio:.4f}")
    
    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE: Mean ratio ± std across seeds")
    print("=" * 70)
    
    print(f"\n  {'Depth':<8} {'Mean Ratio':<12} {'Std':<10} {'vs 1/φ':<12} {'Significance'}")
    print("  " + "-" * 60)
    
    convergence_depths = []
    for d in sorted(all_ratios.keys()):
        ratios = all_ratios[d]
        mean_r = np.mean(ratios)
        std_r = np.std(ratios)
        diff = mean_r - PHI_INV
        
        # Is it significantly close to 1/φ?
        z_score = abs(diff) / (std_r / np.sqrt(len(ratios))) if std_r > 0 else float('inf')
        sig = "✓ MATCHES" if abs(diff) < 0.05 else ""
        
        print(f"  {d:<8} {mean_r:<12.4f} {std_r:<10.4f} {diff:+.4f}      {sig}")
        
        if abs(diff) < 0.05:
            convergence_depths.append(d)
    
    # Check trend
    print("\n" + "=" * 70)
    print("TREND ANALYSIS")
    print("=" * 70)
    
    depths_list = sorted(all_ratios.keys())
    means = [np.mean(all_ratios[d]) for d in depths_list]
    
    print(f"\n  Depth 0 ratio: {means[0]:.4f}")
    print(f"  Last depth ratio: {means[-1]:.4f}")
    print(f"  1/φ target: {PHI_INV:.4f}")
    
    # Does it converge toward 1/φ?
    if len(means) >= 3:
        trend = means[-1] - means[0]
        target_direction = PHI_INV - means[0]
        
        if trend * target_direction > 0:
            print(f"\n  ✅ Trend is toward 1/φ (moving {'up' if trend > 0 else 'down'})")
        else:
            print(f"\n  ❌ Trend is away from 1/φ")
    
    if convergence_depths:
        print(f"\n  Depths matching 1/φ (±0.05): {convergence_depths}")
    else:
        print(f"\n  No depths match 1/φ within tolerance")
    
    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_depth": max_depth,
            "n_seeds": n_seeds,
            "corpus_length": corpus_length,
            "vocab_size": vocab_size
        },
        "ratios_by_depth": {str(d): {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
                           for d, v in all_ratios.items()},
        "phi_inv": PHI_INV,
        "convergence_depths": convergence_depths
    }
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"exp_02_depth_convergence_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
