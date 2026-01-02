"""
exp_07_intervention.py

HYPOTHESIS: We can use convergence metrics to improve generation quality.

INTERVENTION STRATEGIES:
1. Reject-and-resample: When metrics drop below threshold, resample
2. Temperature modulation: Increase temp when too crystallized, decrease when drifting
3. Depth-weighted sampling: Weight candidates by agreement across depths

If intervention improves quality, we have actionable convergence maintenance.
"""

import torch
import json
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path

# Dawn constants
PHI = 1.6180339887
PHI_INV = 0.6180339887
XI = 1.0571428

@dataclass
class GenerationResult:
    """Result of a generation run."""
    strategy: str
    tokens: List[int]
    mean_concentration: float
    mean_quality: float
    collapse_count: int
    rare_count: int
    resamples: int = 0

class PACTreeIntervention:
    """PAC tree with intervention-capable generation."""
    
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.children: Dict[int, Dict[Tuple[int, ...], Dict[int, int]]] = {
            d: defaultdict(lambda: defaultdict(int))
            for d in range(1, max_depth + 1)
        }
        self.token_counts = defaultdict(int)
        self.vocab_size = 0
        
    def train(self, tokens: List[int]):
        """Learn from token sequence."""
        for i, token in enumerate(tokens):
            self.token_counts[token] += 1
            self.vocab_size = max(self.vocab_size, token + 1)
            
            for depth in range(1, self.max_depth + 1):
                if i >= depth:
                    context = tuple(tokens[i-depth:i])
                    self.children[depth][context][token] += 1
    
    def get_candidates(self, context: List[int]) -> Dict[int, Dict[str, float]]:
        """Get all candidate tokens with their depth-level support."""
        candidates = {}
        
        for depth in range(1, self.max_depth + 1):
            if depth > len(context):
                continue
            ctx = tuple(context[-depth:])
            children = self.children[depth].get(ctx, {})
            total = sum(children.values()) + 1e-10
            
            for token, count in children.items():
                if token not in candidates:
                    candidates[token] = {'depths': {}, 'total_weight': 0}
                candidates[token]['depths'][depth] = count / total
                candidates[token]['total_weight'] += (count / total) * depth
                
        return candidates
    
    def compute_concentration(self, context: List[int], token: int) -> float:
        """Compute concentration for a specific token choice."""
        agreements = 0
        total = 0
        
        for depth in range(1, self.max_depth + 1):
            if depth > len(context):
                continue
            ctx = tuple(context[-depth:])
            children = self.children[depth].get(ctx, {})
            if children:
                total += 1
                pred = max(children, key=children.get)
                if pred == token:
                    agreements += 1
                    
        return agreements / total if total > 0 else 0.0
    
    def compute_xi_balance(self, context: List[int], token: int) -> float:
        """Compute Xi balance for shallow vs deep agreement."""
        shallow_agree = 0
        shallow_total = 0
        deep_agree = 0
        deep_total = 0
        
        for depth in range(1, self.max_depth + 1):
            if depth > len(context):
                continue
            ctx = tuple(context[-depth:])
            children = self.children[depth].get(ctx, {})
            if children:
                pred = max(children, key=children.get)
                agrees = (pred == token)
                
                if depth <= 2:
                    shallow_total += 1
                    shallow_agree += int(agrees)
                else:
                    deep_total += 1
                    deep_agree += int(agrees)
        
        shallow_rate = shallow_agree / shallow_total if shallow_total > 0 else 0.5
        deep_rate = deep_agree / deep_total if deep_total > 0 else 0.5
        
        return (shallow_rate + 0.1) / (deep_rate + 0.1)
    
    def sample_baseline(self, context: List[int], temperature: float = 1.0) -> Tuple[int, float]:
        """Baseline sampling - no intervention."""
        candidates = self.get_candidates(context)
        
        if not candidates:
            token = max(self.token_counts, key=self.token_counts.get)
            total = sum(self.token_counts.values())
            return token, self.token_counts[token] / total
        
        tokens = list(candidates.keys())
        weights = [candidates[t]['total_weight'] for t in tokens]
        
        if temperature > 0:
            weights = [w ** (1/temperature) for w in weights]
            total = sum(weights) + 1e-10
            weights = [w/total for w in weights]
            idx = torch.multinomial(torch.tensor(weights), 1).item()
            return tokens[idx], weights[idx]
        else:
            idx = weights.index(max(weights))
            return tokens[idx], weights[idx] / sum(weights)
    
    def sample_reject_resample(self, context: List[int], temperature: float = 1.0,
                                min_concentration: float = 0.3, max_attempts: int = 5) -> Tuple[int, float, int]:
        """Reject-and-resample: reject low-concentration samples."""
        attempts = 0
        
        while attempts < max_attempts:
            token, prob = self.sample_baseline(context, temperature)
            concentration = self.compute_concentration(context, token)
            
            if concentration >= min_concentration:
                return token, prob, attempts
            
            attempts += 1
            temperature *= 1.2  # Increase temperature to explore more
        
        # Fall back to greedy if all rejected
        token, prob = self.sample_baseline(context, temperature=0)
        return token, prob, attempts
    
    def sample_temp_modulation(self, context: List[int], base_temp: float = 1.0,
                                prev_concentration: float = 0.5) -> Tuple[int, float, float]:
        """Temperature modulation based on previous concentration."""
        # If crystallized (high conc), add randomness
        # If drifting (low conc), reduce randomness
        if prev_concentration > 0.8:
            temp = base_temp * 1.3  # More random
        elif prev_concentration < 0.3:
            temp = base_temp * 0.5  # More deterministic
        else:
            temp = base_temp
            
        token, prob = self.sample_baseline(context, temp)
        return token, prob, temp
    
    def sample_depth_weighted(self, context: List[int], concentration_weight: float = 2.0) -> Tuple[int, float]:
        """Weight candidates by their concentration (agreement across depths)."""
        candidates = self.get_candidates(context)
        
        if not candidates:
            token = max(self.token_counts, key=self.token_counts.get)
            total = sum(self.token_counts.values())
            return token, self.token_counts[token] / total
        
        # Compute concentration for each candidate
        scored = {}
        for token in candidates:
            conc = self.compute_concentration(context, token)
            base_weight = candidates[token]['total_weight']
            # Boost by concentration
            scored[token] = base_weight * (1 + concentration_weight * conc)
        
        tokens = list(scored.keys())
        weights = [scored[t] for t in tokens]
        total = sum(weights) + 1e-10
        weights = [w/total for w in weights]
        
        idx = torch.multinomial(torch.tensor(weights), 1).item()
        return tokens[idx], weights[idx]
    
    def is_rare(self, token: int) -> bool:
        """Check if token is rare."""
        total = sum(self.token_counts.values())
        return self.token_counts.get(token, 0) / total < 0.001

def run_generation(tree: PACTreeIntervention, strategy: str, 
                   seed_context: List[int], length: int = 50) -> GenerationResult:
    """Run generation with specified strategy."""
    context = seed_context.copy()
    generated = []
    concentrations = []
    qualities = []
    collapse_count = 0
    rare_count = 0
    resample_count = 0
    prev_concentration = 0.5
    
    for step in range(length):
        if strategy == 'baseline':
            token, prob = tree.sample_baseline(context)
            resamples = 0
        elif strategy == 'reject_resample':
            token, prob, resamples = tree.sample_reject_resample(context)
            resample_count += resamples
        elif strategy == 'temp_modulation':
            token, prob, _ = tree.sample_temp_modulation(context, prev_concentration=prev_concentration)
            resamples = 0
        elif strategy == 'depth_weighted':
            token, prob = tree.sample_depth_weighted(context)
            resamples = 0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Compute metrics
        concentration = tree.compute_concentration(context, token)
        
        # Quality: 1.0 minus penalties
        quality = 1.0
        if tree.is_rare(token):
            quality -= 0.3
            rare_count += 1
        if token in context[-5:]:
            quality -= 0.2
        if prob < 0.1:
            quality -= 0.3
        quality = max(0.0, quality)
        
        # Track collapse
        if step > 0 and concentration - prev_concentration < -0.3:
            collapse_count += 1
        
        generated.append(token)
        concentrations.append(concentration)
        qualities.append(quality)
        prev_concentration = concentration
        
        context.append(token)
        context = context[-20:]
    
    return GenerationResult(
        strategy=strategy,
        tokens=generated,
        mean_concentration=sum(concentrations) / len(concentrations),
        mean_quality=sum(qualities) / len(qualities),
        collapse_count=collapse_count,
        rare_count=rare_count,
        resamples=resample_count
    )

def main():
    print("=" * 60)
    print("EXP 07: INTERVENTION TESTING")
    print("Can convergence metrics improve generation quality?")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        text = ' '.join([x for x in ds['text'] if x.strip()])[:200000]
    except Exception as e:
        print(f"Dataset error: {e}, using sample text")
        text = "The cat sat on the mat. " * 5000
    
    # Tokenize
    words = text.lower().split()
    vocab = {w: i for i, w in enumerate(sorted(set(words)))}
    tokens = [vocab[w] for w in words]
    
    print(f"Vocabulary: {len(vocab)}")
    print(f"Tokens: {len(tokens)}")
    
    # Train tree
    tree = PACTreeIntervention(max_depth=5)
    tree.train(tokens)
    print(f"PAC tree trained")
    
    # Run comparison
    strategies = ['baseline', 'reject_resample', 'temp_modulation', 'depth_weighted']
    n_runs = 30
    seq_length = 50
    
    results = {s: [] for s in strategies}
    
    print(f"\nRunning {n_runs} generations per strategy...")
    
    for run_idx in range(n_runs):
        # Same seed context for all strategies
        start_idx = torch.randint(0, len(tokens) - 20, (1,)).item()
        seed_context = tokens[start_idx:start_idx + 5]
        
        for strategy in strategies:
            result = run_generation(tree, strategy, seed_context, seq_length)
            results[strategy].append(result)
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)
    
    print(f"\n{'Strategy':<18} {'Quality':>10} {'Conc':>10} {'Collapse':>10} {'Rare':>8} {'Resample':>10}")
    print("-" * 70)
    
    summary = {}
    for strategy in strategies:
        strat_results = results[strategy]
        mean_quality = sum(r.mean_quality for r in strat_results) / len(strat_results)
        mean_conc = sum(r.mean_concentration for r in strat_results) / len(strat_results)
        mean_collapse = sum(r.collapse_count for r in strat_results) / len(strat_results)
        mean_rare = sum(r.rare_count for r in strat_results) / len(strat_results)
        mean_resample = sum(r.resamples for r in strat_results) / len(strat_results)
        
        print(f"{strategy:<18} {mean_quality:>10.3f} {mean_conc:>10.3f} {mean_collapse:>10.1f} {mean_rare:>8.1f} {mean_resample:>10.1f}")
        
        summary[strategy] = {
            'mean_quality': round(mean_quality, 4),
            'mean_concentration': round(mean_conc, 4),
            'mean_collapse': round(mean_collapse, 2),
            'mean_rare': round(mean_rare, 2),
            'mean_resample': round(mean_resample, 2)
        }
    
    # Compute improvement over baseline
    print("\n" + "-" * 40)
    print("IMPROVEMENT OVER BASELINE")
    print("-" * 40)
    
    baseline_quality = summary['baseline']['mean_quality']
    baseline_collapse = summary['baseline']['mean_collapse']
    
    for strategy in strategies[1:]:
        quality_lift = (summary[strategy]['mean_quality'] - baseline_quality) / baseline_quality * 100
        collapse_reduction = (baseline_collapse - summary[strategy]['mean_collapse']) / baseline_collapse * 100 if baseline_collapse > 0 else 0
        
        print(f"\n{strategy}:")
        print(f"  Quality: {quality_lift:+.1f}%")
        print(f"  Collapse reduction: {collapse_reduction:+.1f}%")
    
    # Best strategy
    print("\n" + "=" * 60)
    print("BEST PERFORMING STRATEGY")
    print("=" * 60)
    
    best = max(strategies, key=lambda s: summary[s]['mean_quality'])
    print(f"\nBest quality: {best}")
    print(f"  Quality: {summary[best]['mean_quality']:.4f}")
    print(f"  Concentration: {summary[best]['mean_concentration']:.4f}")
    print(f"  Collapse rate: {summary[best]['mean_collapse']:.2f}")
    
    if best != 'baseline':
        improvement = (summary[best]['mean_quality'] - baseline_quality) / baseline_quality * 100
        print(f"  Improvement over baseline: {improvement:+.1f}%")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'n_runs': n_runs,
        'seq_length': seq_length,
        'strategies': summary,
        'best_strategy': best,
        'dawn_constants': {'PHI': PHI, 'XI': XI}
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f'exp_07_intervention_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved: {results_file.name}")
    
    # Validation status
    print("\n" + "=" * 60)
    print("VALIDATION STATUS")
    print("=" * 60)
    
    any_improvement = any(summary[s]['mean_quality'] > baseline_quality for s in strategies[1:])
    if any_improvement:
        print("+ At least one intervention strategy improves over baseline")
        print("+ Convergence metrics are ACTIONABLE for quality improvement")
    else:
        print("! No intervention improved over baseline")
        print("! Metrics may be descriptive but not actionable at this scale")

if __name__ == '__main__':
    main()
