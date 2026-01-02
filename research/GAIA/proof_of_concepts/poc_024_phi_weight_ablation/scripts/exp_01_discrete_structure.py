"""
exp_01_discrete_structure.py - Test Fibonacci in PAC tree discrete structure

POC-024: φ Structure Validation

From Fibonacci Derivation Breakthrough:
  Conservation:    Parent = Child₁ + Child₂
  Self-similarity: Child₁/Child₂ = Parent/Child₁
  → r² = r + 1
  → r = φ

This applies to DISCRETE COUNTS, not continuous weights.

Tests:
1. Branching factor distribution - how many children per parent?
2. Depth population ratios - N(d) / N(d+1) → φ?
3. Count-weighted splits - when children have different counts, ratio → φ?
"""

import json
import random
import math
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

# Constants
PHI = (1 + math.sqrt(5)) / 2  # 1.618...
PHI_INV = 1 / PHI             # 0.618...

# =============================================================================
# PAC TREE WITH STRUCTURAL ANALYSIS
# =============================================================================

class StructuralPACTree:
    """PAC tree that tracks discrete structural properties"""
    
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        # Per-parent children: context_tuple -> {child_token: count}
        self.children = defaultdict(lambda: defaultdict(int))
        
    def learn_sequence(self, tokens: list):
        """Learn transitions from a token sequence"""
        for i in range(len(tokens) - 1):
            child = tokens[i + 1]
            
            # Learn at each depth level
            for d in range(min(self.max_depth, i + 1)):
                context = tuple(tokens[max(0, i - d):i + 1])
                self.children[context][child] += 1
    
    def get_branching_factors(self) -> dict:
        """Count children per parent at each depth"""
        by_depth = defaultdict(list)
        
        for context, children in self.children.items():
            depth = len(context) - 1  # 0-indexed depth
            n_children = len(children)
            by_depth[depth].append(n_children)
        
        return {d: vals for d, vals in sorted(by_depth.items())}
    
    def get_depth_populations(self) -> dict:
        """Count unique contexts (nodes) at each depth"""
        by_depth = defaultdict(set)
        
        for context in self.children.keys():
            depth = len(context) - 1
            by_depth[depth].add(context)
        
        return {d: len(nodes) for d, nodes in sorted(by_depth.items())}
    
    def get_child_count_ratios(self) -> list:
        """For parents with 2+ children, get ratio of top two counts"""
        ratios = []
        
        for context, children in self.children.items():
            if len(children) >= 2:
                counts = sorted(children.values(), reverse=True)
                # Ratio of largest to second largest
                if counts[1] > 0:
                    ratio = counts[0] / counts[1]
                    ratios.append(ratio)
        
        return ratios
    
    def get_total_vs_unique_ratio(self) -> dict:
        """Ratio of total observations to unique children per depth"""
        by_depth = defaultdict(lambda: {"total": 0, "unique": 0})
        
        for context, children in self.children.items():
            depth = len(context) - 1
            by_depth[depth]["total"] += sum(children.values())
            by_depth[depth]["unique"] += len(children)
        
        return {d: vals["total"] / vals["unique"] if vals["unique"] > 0 else 0 
                for d, vals in sorted(by_depth.items())}


# =============================================================================
# CORPUS GENERATION
# =============================================================================

def generate_corpus(vocab_size: int = 1000, length: int = 100000, seed: int = 42) -> list:
    """Generate corpus with Zipf-like transition structure"""
    random.seed(seed)
    
    # Create transition preferences (Zipf-like)
    transitions = {}
    for i in range(vocab_size):
        n_successors = random.randint(2, 8)  # 2-8 successors
        successors = random.sample(range(vocab_size), n_successors)
        # Zipf weights: first successor much more likely
        weights = [1.0 / (j + 1) ** 1.5 for j in range(n_successors)]
        total = sum(weights)
        transitions[i] = [(s, w/total) for s, w in zip(successors, weights)]
    
    # Generate sequence
    tokens = [random.randint(0, vocab_size - 1)]
    for _ in range(length - 1):
        current = tokens[-1]
        successors, probs = zip(*transitions[current])
        next_token = random.choices(successors, weights=probs)[0]
        tokens.append(next_token)
    
    return tokens


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    print("=" * 70)
    print("POC-024 Exp 01: Discrete Structure Analysis")
    print("=" * 70)
    print(f"\nφ = {PHI:.6f}")
    print(f"1/φ = {PHI_INV:.6f}")
    
    # Generate corpus
    print("\n--- Generating Corpus ---")
    corpus = generate_corpus(vocab_size=1000, length=100000, seed=42)
    print(f"Corpus length: {len(corpus)}")
    print(f"Unique tokens: {len(set(corpus))}")
    
    # Build tree
    print("\n--- Building PAC Tree ---")
    tree = StructuralPACTree(max_depth=5)
    tree.learn_sequence(corpus)
    print(f"Total contexts: {len(tree.children)}")
    
    # Analysis 1: Branching factors
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Branching Factors (children per parent)")
    print("=" * 70)
    
    branching = tree.get_branching_factors()
    for depth, factors in branching.items():
        mean_bf = np.mean(factors)
        std_bf = np.std(factors)
        median_bf = np.median(factors)
        print(f"  Depth {depth}: mean={mean_bf:.2f} ± {std_bf:.2f}, median={median_bf:.1f}, n={len(factors)}")
    
    # Analysis 2: Depth populations
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Depth Populations")
    print("=" * 70)
    
    populations = tree.get_depth_populations()
    print(f"\n  {'Depth':<8} {'Nodes':<10} {'Ratio to Next':<15} {'vs φ':<10}")
    print("  " + "-" * 50)
    
    depths = sorted(populations.keys())
    ratios_to_phi = []
    for i, d in enumerate(depths):
        n = populations[d]
        if i < len(depths) - 1:
            next_n = populations[depths[i+1]]
            ratio = n / next_n if next_n > 0 else float('inf')
            diff_from_phi = abs(ratio - PHI_INV)
            ratios_to_phi.append((ratio, diff_from_phi))
            print(f"  {d:<8} {n:<10} {ratio:<15.4f} {diff_from_phi:+.4f}")
        else:
            print(f"  {d:<8} {n:<10} {'(last)':<15}")
    
    if ratios_to_phi:
        mean_ratio = np.mean([r[0] for r in ratios_to_phi])
        print(f"\n  Mean ratio: {mean_ratio:.4f}")
        print(f"  φ inverse:  {PHI_INV:.4f}")
        print(f"  Difference: {abs(mean_ratio - PHI_INV):.4f}")
    
    # Analysis 3: Child count ratios
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Child Count Ratios (top1/top2 for each parent)")
    print("=" * 70)
    
    count_ratios = tree.get_child_count_ratios()
    if count_ratios:
        mean_ratio = np.mean(count_ratios)
        median_ratio = np.median(count_ratios)
        std_ratio = np.std(count_ratios)
        
        # What fraction are close to φ?
        close_to_phi = sum(1 for r in count_ratios if abs(r - PHI) < 0.2)
        close_to_2 = sum(1 for r in count_ratios if abs(r - 2.0) < 0.2)
        
        print(f"\n  N ratios: {len(count_ratios)}")
        print(f"  Mean:     {mean_ratio:.4f}")
        print(f"  Median:   {median_ratio:.4f}")
        print(f"  Std:      {std_ratio:.4f}")
        print(f"  φ =       {PHI:.4f}")
        print(f"\n  Close to φ (±0.2): {close_to_phi} ({100*close_to_phi/len(count_ratios):.1f}%)")
        print(f"  Close to 2 (±0.2): {close_to_2} ({100*close_to_2/len(count_ratios):.1f}%)")
        
        # Distribution buckets
        buckets = Counter()
        for r in count_ratios:
            if r < 1.2:
                buckets["1.0-1.2"] += 1
            elif r < 1.5:
                buckets["1.2-1.5"] += 1
            elif r < 1.8:
                buckets["1.5-1.8 (φ zone)"] += 1
            elif r < 2.2:
                buckets["1.8-2.2"] += 1
            elif r < 3.0:
                buckets["2.2-3.0"] += 1
            else:
                buckets["3.0+"] += 1
        
        print("\n  Distribution:")
        for bucket, count in sorted(buckets.items()):
            pct = 100 * count / len(count_ratios)
            bar = "█" * int(pct / 2)
            print(f"    {bucket:<15}: {count:>5} ({pct:>5.1f}%) {bar}")
    
    # Analysis 4: Fibonacci sequence check
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Fibonacci Sequence Check")
    print("=" * 70)
    
    # Do depth populations follow Fibonacci?
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    
    print("\n  Checking if populations ~ Fibonacci scaled:")
    if len(populations) >= 2:
        # Scale Fibonacci to match depth 0
        scale = populations[0] / fib[0] if populations[0] > 0 else 1
        
        print(f"  {'Depth':<8} {'Actual':<10} {'Fib×scale':<12} {'Ratio':<10}")
        print("  " + "-" * 45)
        
        for d in sorted(populations.keys()):
            actual = populations[d]
            if d < len(fib):
                expected = fib[d] * scale
                ratio = actual / expected if expected > 0 else 0
                match = "✓" if 0.5 < ratio < 2.0 else "✗"
                print(f"  {d:<8} {actual:<10} {expected:<12.1f} {ratio:<10.3f} {match}")
    
    # Analysis 5: Total/Unique ratio (concentration)
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Total/Unique Ratio by Depth")
    print("=" * 70)
    
    tu_ratios = tree.get_total_vs_unique_ratio()
    print(f"\n  {'Depth':<8} {'Total/Unique':<15}")
    print("  " + "-" * 25)
    for d, ratio in tu_ratios.items():
        print(f"  {d:<8} {ratio:<15.2f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\n  Question: Does φ appear in discrete PAC tree structure?")
    
    if ratios_to_phi:
        mean_pop_ratio = np.mean([r[0] for r in ratios_to_phi])
        if abs(mean_pop_ratio - PHI_INV) < 0.1:
            print(f"\n  ✅ Depth population ratio ({mean_pop_ratio:.3f}) ≈ 1/φ ({PHI_INV:.3f})")
        else:
            print(f"\n  ❌ Depth population ratio ({mean_pop_ratio:.3f}) ≠ 1/φ ({PHI_INV:.3f})")
    
    if count_ratios:
        median_cr = np.median(count_ratios)
        if abs(median_cr - PHI) < 0.2:
            print(f"  ✅ Child count ratio median ({median_cr:.3f}) ≈ φ ({PHI:.3f})")
        else:
            print(f"  ❌ Child count ratio median ({median_cr:.3f}) ≠ φ ({PHI:.3f})")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "corpus": {"length": len(corpus), "vocab": len(set(corpus))},
        "tree": {"contexts": len(tree.children)},
        "branching_factors": {str(d): {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} 
                             for d, v in branching.items()},
        "depth_populations": populations,
        "population_ratios": [(r, d) for r, d in ratios_to_phi] if ratios_to_phi else [],
        "child_count_ratios": {
            "mean": float(np.mean(count_ratios)) if count_ratios else None,
            "median": float(np.median(count_ratios)) if count_ratios else None,
            "std": float(np.std(count_ratios)) if count_ratios else None,
            "n": len(count_ratios)
        },
        "phi": PHI,
        "phi_inv": PHI_INV
    }
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"exp_01_discrete_structure_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")
    
    return output


if __name__ == "__main__":
    main()
