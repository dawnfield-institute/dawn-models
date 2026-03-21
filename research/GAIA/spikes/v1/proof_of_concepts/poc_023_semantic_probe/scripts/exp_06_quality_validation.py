"""
exp_06_quality_validation.py

HYPOTHESIS: Convergence metrics predict actual generation quality.
If low concentration/Xi divergence correlates with "bad" generations,
we have a practical quality signal.

APPROACH:
1. Generate many sequences from PAC tree
2. Track convergence metrics per token
3. Compare metrics at "known bad" positions vs "known good"
4. Bad = rare token, repeated token, out-of-distribution
5. Good = common patterns, smooth continuation

This is still PAC tree validation - proving the principle
before applying to GPT-2.
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
class TokenMetrics:
    """Metrics for a single generated token."""
    position: int
    token_id: int
    concentration: float
    velocity: float
    xi_balance: float
    depth_agreements: Dict[int, bool] = field(default_factory=dict)
    
@dataclass
class QualityIndicators:
    """Quality indicators for a token (ground truth)."""
    is_rare: bool = False
    is_repeated: bool = False
    is_low_prob: bool = False
    quality_score: float = 1.0  # 1.0 = good, 0.0 = bad

class PACTreeWithQuality:
    """PAC tree with quality tracking during generation."""
    
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        # Per-parent children for proper byref
        self.children: Dict[int, Dict[Tuple[int, ...], Dict[int, int]]] = {
            d: defaultdict(lambda: defaultdict(int))
            for d in range(1, max_depth + 1)
        }
        self.token_counts = defaultdict(int)
        self.vocab_size = 0
        
    def train(self, tokens: List[int]):
        """Learn from token sequence - delta only per parent context."""
        for i, token in enumerate(tokens):
            self.token_counts[token] += 1
            self.vocab_size = max(self.vocab_size, token + 1)
            
            for depth in range(1, self.max_depth + 1):
                if i >= depth:
                    context = tuple(tokens[i-depth:i])
                    self.children[depth][context][token] += 1
                    
    def predict_at_depth(self, context: List[int], depth: int) -> Optional[int]:
        """Predict next token using specific depth."""
        if depth > len(context):
            return None
        ctx = tuple(context[-depth:])
        children = self.children[depth].get(ctx, {})
        if not children:
            return None
        return max(children, key=children.get)
    
    def get_depth_predictions(self, context: List[int]) -> Dict[int, Optional[int]]:
        """Get predictions from all depths."""
        return {d: self.predict_at_depth(context, d) 
                for d in range(1, self.max_depth + 1)}
    
    def sample_next(self, context: List[int], temperature: float = 1.0) -> Tuple[int, float]:
        """Sample next token, return (token, probability)."""
        # Combine all depth predictions
        candidates = defaultdict(float)
        
        for depth in range(1, self.max_depth + 1):
            if depth > len(context):
                continue
            ctx = tuple(context[-depth:])
            children = self.children[depth].get(ctx, {})
            total = sum(children.values()) + 1e-10
            
            for token, count in children.items():
                # Weight by depth (deeper = more specific)
                candidates[token] += (count / total) * depth
                
        if not candidates:
            # Fallback to unigram
            total = sum(self.token_counts.values())
            token = max(self.token_counts, key=self.token_counts.get)
            return token, self.token_counts[token] / total
            
        # Normalize and sample
        total = sum(candidates.values())
        probs = {t: c/total for t, c in candidates.items()}
        
        if temperature > 0:
            # Temperature-scaled sampling
            tokens = list(probs.keys())
            weights = [probs[t] ** (1/temperature) for t in tokens]
            total_w = sum(weights)
            weights = [w/total_w for w in weights]
            
            idx = torch.multinomial(torch.tensor(weights), 1).item()
            return tokens[idx], probs[tokens[idx]]
        else:
            # Greedy
            token = max(probs, key=probs.get)
            return token, probs[token]
    
    def compute_metrics(self, context: List[int], next_token: int) -> TokenMetrics:
        """Compute full metrics for a generation step."""
        preds = self.get_depth_predictions(context)
        
        # Depth agreements
        agreements = {d: (pred == next_token) for d, pred in preds.items() if pred is not None}
        
        # Concentration
        if agreements:
            concentration = sum(agreements.values()) / len(agreements)
        else:
            concentration = 0.0
            
        # For velocity, need previous - return 0 if not tracking
        velocity = 0.0
        
        # Xi balance: ratio of shallow to deep agreement rates (normalized around 1.0)
        # Shallow = depths 1-2, Deep = depths 3-5
        shallow_agree = sum(1 for d in [1, 2] if agreements.get(d, False))
        shallow_total = sum(1 for d in [1, 2] if d in agreements)
        deep_agree = sum(1 for d in [3, 4, 5] if agreements.get(d, False))
        deep_total = sum(1 for d in [3, 4, 5] if d in agreements)
        
        shallow_rate = shallow_agree / shallow_total if shallow_total > 0 else 0.5
        deep_rate = deep_agree / deep_total if deep_total > 0 else 0.5
        
        # Xi balance: shallow/deep ratio, expected ~1.0 for balanced
        xi_balance = (shallow_rate + 0.1) / (deep_rate + 0.1)  # Smoothed to avoid div/0
            
        return TokenMetrics(
            position=len(context),
            token_id=next_token,
            concentration=concentration,
            velocity=velocity,
            xi_balance=xi_balance,
            depth_agreements=agreements
        )

def assess_quality(token: int, context: List[int], token_counts: Dict[int, int], 
                   prob: float) -> QualityIndicators:
    """Assess quality indicators for a generated token."""
    
    total_count = sum(token_counts.values())
    token_freq = token_counts.get(token, 0) / total_count if total_count > 0 else 0
    
    # Rare: frequency below 0.001
    is_rare = token_freq < 0.001
    
    # Repeated: appears in recent context
    is_repeated = token in context[-5:] if len(context) >= 5 else token in context
    
    # Low prob: below threshold
    is_low_prob = prob < 0.1
    
    # Compute quality score (lower = worse)
    quality = 1.0
    if is_rare:
        quality -= 0.3
    if is_repeated:
        quality -= 0.2
    if is_low_prob:
        quality -= 0.3
        
    return QualityIndicators(
        is_rare=is_rare,
        is_repeated=is_repeated,
        is_low_prob=is_low_prob,
        quality_score=max(0.0, quality)
    )

def correlation(x: List[float], y: List[float]) -> float:
    """Pearson correlation between two lists."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    mx, my = sum(x)/n, sum(y)/n
    
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den_x = sum((xi - mx)**2 for xi in x) ** 0.5
    den_y = sum((yi - my)**2 for yi in y) ** 0.5
    
    if den_x * den_y == 0:
        return 0.0
    return num / (den_x * den_y)

def main():
    print("=" * 60)
    print("EXP 06: QUALITY VALIDATION")
    print("Testing if convergence metrics predict generation quality")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load WikiText-2
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        text = ' '.join([x for x in ds['text'] if x.strip()])[:200000]
    except Exception as e:
        print(f"Dataset error: {e}, using sample text")
        text = "The cat sat on the mat. " * 5000
    
    # Simple tokenization
    words = text.lower().split()
    vocab = {w: i for i, w in enumerate(sorted(set(words)))}
    inv_vocab = {i: w for w, i in vocab.items()}
    tokens = [vocab[w] for w in words]
    
    print(f"Vocabulary: {len(vocab)} words")
    print(f"Tokens: {len(tokens)}")
    
    # Train PAC tree
    tree = PACTreeWithQuality(max_depth=5)
    tree.train(tokens)
    print(f"PAC tree trained")
    
    # Generate sequences and track metrics + quality
    n_sequences = 20
    seq_length = 50
    
    all_metrics: List[TokenMetrics] = []
    all_quality: List[QualityIndicators] = []
    
    print(f"\nGenerating {n_sequences} sequences of {seq_length} tokens...")
    
    for seq_idx in range(n_sequences):
        # Random seed context
        start_idx = torch.randint(0, len(tokens) - 20, (1,)).item()
        context = tokens[start_idx:start_idx + 5]
        
        prev_concentration = None
        
        for step in range(seq_length):
            # Generate next token
            next_token, prob = tree.sample_next(context, temperature=0.8)
            
            # Compute convergence metrics
            metrics = tree.compute_metrics(context, next_token)
            
            # Update velocity if we have previous
            if prev_concentration is not None:
                metrics.velocity = metrics.concentration - prev_concentration
            prev_concentration = metrics.concentration
            
            # Assess quality
            quality = assess_quality(next_token, context, tree.token_counts, prob)
            
            all_metrics.append(metrics)
            all_quality.append(quality)
            
            context.append(next_token)
            context = context[-20:]  # Keep manageable
    
    print(f"Collected {len(all_metrics)} token samples")
    
    # Analyze correlations
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    
    concentrations = [m.concentration for m in all_metrics]
    velocities = [m.velocity for m in all_metrics]
    xi_balances = [m.xi_balance for m in all_metrics]
    qualities = [q.quality_score for q in all_quality]
    
    corr_conc_qual = correlation(concentrations, qualities)
    corr_vel_qual = correlation(velocities, qualities)
    corr_xi_qual = correlation(xi_balances, qualities)
    
    print(f"\nCorrelation with Quality Score:")
    print(f"  Concentration: r = {corr_conc_qual:+.4f}")
    print(f"  Velocity:      r = {corr_vel_qual:+.4f}")
    print(f"  Xi Balance:    r = {corr_xi_qual:+.4f}")
    
    # Binned analysis
    print("\n" + "-" * 40)
    print("BINNED ANALYSIS: Metrics by Quality Tier")
    print("-" * 40)
    
    # Split into quality tiers
    tiers = {'high': [], 'medium': [], 'low': []}
    for m, q in zip(all_metrics, all_quality):
        if q.quality_score >= 0.8:
            tiers['high'].append(m)
        elif q.quality_score >= 0.5:
            tiers['medium'].append(m)
        else:
            tiers['low'].append(m)
    
    print(f"\n{'Tier':<8} {'N':>6} {'Mean Conc':>12} {'Mean Xi':>10} {'Collapse%':>12}")
    print("-" * 50)
    
    for tier_name in ['high', 'medium', 'low']:
        tier_data = tiers[tier_name]
        if not tier_data:
            continue
        mean_conc = sum(m.concentration for m in tier_data) / len(tier_data)
        mean_xi = sum(m.xi_balance for m in tier_data) / len(tier_data)
        collapse_pct = sum(1 for m in tier_data if m.velocity < -0.3) / len(tier_data) * 100
        
        print(f"{tier_name:<8} {len(tier_data):>6} {mean_conc:>12.3f} {mean_xi:>10.3f} {collapse_pct:>11.1f}%")
    
    # Divergence pattern analysis
    print("\n" + "-" * 40)
    print("DIVERGENCE PATTERNS")
    print("-" * 40)
    
    patterns = {
        'stable': 0,       # High conc, Xi near 1.0
        'drifting': 0,     # Falling conc, any Xi
        'overconfident': 0, # Low conc, high Xi
        'fragile': 0       # High conc, low Xi
    }
    
    pattern_quality = {k: [] for k in patterns}
    
    for m, q in zip(all_metrics, all_quality):
        xi_deviation = abs(m.xi_balance - XI)
        
        if m.concentration >= 0.7 and xi_deviation < 0.2:
            pattern = 'stable'
        elif m.velocity < -0.2:
            pattern = 'drifting'
        elif m.concentration < 0.4 and m.xi_balance > 1.2:
            pattern = 'overconfident'
        elif m.concentration >= 0.7 and m.xi_balance < 0.8:
            pattern = 'fragile'
        else:
            pattern = 'stable'  # Default
            
        patterns[pattern] += 1
        pattern_quality[pattern].append(q.quality_score)
    
    print(f"\n{'Pattern':<14} {'Count':>8} {'Mean Quality':>14} {'Rare%':>8} {'Repeated%':>10}")
    print("-" * 60)
    
    for pattern_name, count in patterns.items():
        if count == 0:
            continue
        mean_q = sum(pattern_quality[pattern_name]) / len(pattern_quality[pattern_name])
        
        # Get the corresponding quality indicators
        pattern_indices = []
        for i, (m, q) in enumerate(zip(all_metrics, all_quality)):
            xi_deviation = abs(m.xi_balance - XI)
            if pattern_name == 'stable' and m.concentration >= 0.7 and xi_deviation < 0.2:
                pattern_indices.append(i)
            elif pattern_name == 'drifting' and m.velocity < -0.2:
                pattern_indices.append(i)
            elif pattern_name == 'overconfident' and m.concentration < 0.4 and m.xi_balance > 1.2:
                pattern_indices.append(i)
            elif pattern_name == 'fragile' and m.concentration >= 0.7 and m.xi_balance < 0.8:
                pattern_indices.append(i)
        
        if pattern_indices:
            rare_pct = sum(1 for i in pattern_indices if all_quality[i].is_rare) / len(pattern_indices) * 100
            repeated_pct = sum(1 for i in pattern_indices if all_quality[i].is_repeated) / len(pattern_indices) * 100
        else:
            rare_pct = repeated_pct = 0
            
        print(f"{pattern_name:<14} {count:>8} {mean_q:>14.3f} {rare_pct:>7.1f}% {repeated_pct:>9.1f}%")
    
    # Key insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    
    # Check if concentration predicts quality
    high_conc_quality = [q.quality_score for m, q in zip(all_metrics, all_quality) if m.concentration >= 0.7]
    low_conc_quality = [q.quality_score for m, q in zip(all_metrics, all_quality) if m.concentration < 0.4]
    
    if high_conc_quality and low_conc_quality:
        high_mean = sum(high_conc_quality) / len(high_conc_quality)
        low_mean = sum(low_conc_quality) / len(low_conc_quality)
        lift = high_mean / low_mean if low_mean > 0 else float('inf')
        print(f"\n1. High vs Low Concentration:")
        print(f"   High conc quality: {high_mean:.3f} (n={len(high_conc_quality)})")
        print(f"   Low conc quality:  {low_mean:.3f} (n={len(low_conc_quality)})")
        print(f"   Quality lift: {lift:.2f}x")
    
    # Check if Xi deviation predicts issues
    close_xi = [q.quality_score for m, q in zip(all_metrics, all_quality) 
                if abs(m.xi_balance - XI) < 0.2]
    far_xi = [q.quality_score for m, q in zip(all_metrics, all_quality) 
              if abs(m.xi_balance - XI) >= 0.5]
    
    if close_xi and far_xi:
        close_mean = sum(close_xi) / len(close_xi)
        far_mean = sum(far_xi) / len(far_xi)
        print(f"\n2. Xi Balance Deviation:")
        print(f"   Close to XI ({XI:.3f}): quality={close_mean:.3f} (n={len(close_xi)})")
        print(f"   Far from XI: quality={far_mean:.3f} (n={len(far_xi)})")
    
    # Collapse events vs quality
    collapse_quality = [q.quality_score for m, q in zip(all_metrics, all_quality) if m.velocity < -0.3]
    stable_quality = [q.quality_score for m, q in zip(all_metrics, all_quality) if abs(m.velocity) < 0.1]
    
    if collapse_quality and stable_quality:
        collapse_mean = sum(collapse_quality) / len(collapse_quality)
        stable_mean = sum(stable_quality) / len(stable_quality)
        print(f"\n3. Velocity Stability:")
        print(f"   Collapse events (V<-0.3): quality={collapse_mean:.3f} (n={len(collapse_quality)})")
        print(f"   Stable (|V|<0.1): quality={stable_mean:.3f} (n={len(stable_quality)})")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(all_metrics),
        'correlations': {
            'concentration_quality': round(corr_conc_qual, 4),
            'velocity_quality': round(corr_vel_qual, 4),
            'xi_quality': round(corr_xi_qual, 4)
        },
        'tier_summary': {
            tier: {
                'count': len(data),
                'mean_concentration': round(sum(m.concentration for m in data) / len(data), 4) if data else 0,
                'mean_xi': round(sum(m.xi_balance for m in data) / len(data), 4) if data else 0
            }
            for tier, data in tiers.items()
        },
        'pattern_counts': patterns,
        'pattern_quality': {k: round(sum(v)/len(v), 4) if v else 0 for k, v in pattern_quality.items()},
        'dawn_constants': {
            'PHI': PHI,
            'XI': XI,
            'PHI_INV': PHI_INV
        }
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f'exp_06_quality_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved: {results_file.name}")
    
    print("\n" + "=" * 60)
    print("VALIDATION STATUS")
    print("=" * 60)
    
    # Check if metrics predict quality
    if corr_conc_qual > 0.1:
        print("+ Concentration correlates with quality")
    else:
        print("! Concentration correlation weak")
        
    if high_conc_quality and low_conc_quality:
        if lift > 1.1:
            print(f"+ Quality lift from concentration: {lift:.2f}x")
        else:
            print(f"! Quality lift marginal: {lift:.2f}x")
    
    if patterns['drifting'] > 0 and pattern_quality['drifting']:
        drift_q = sum(pattern_quality['drifting']) / len(pattern_quality['drifting'])
        if drift_q < 0.7:
            print(f"+ Drifting pattern shows lower quality ({drift_q:.2f})")
        else:
            print(f"! Drifting pattern quality acceptable ({drift_q:.2f})")

if __name__ == '__main__':
    main()
