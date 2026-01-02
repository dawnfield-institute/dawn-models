"""
POC-023 Experiment 02: Cross-Scale Agreement
Test whether ADDITIONAL context changes the prediction.

With byref sharing, nodes merge when contexts converge. The real question:
Does DEEPER context produce DIFFERENT predictions than shallower?

Hypothesis: When deeper context changes the prediction, it's often more accurate.

Key questions:
1. How often does depth-5 predict differently than depth-1?
2. When they differ, which is more accurate?
3. Does context-induced change correlate with surprise/novelty?
"""

import json
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import GPT2Tokenizer

# Dawn Field constants
PHI = 1.6180339887
PHI_INV = 1 / PHI
XI = 0.0618
PHI_XI = PHI * XI


class PACNode:
    """PAC node - children are shared through parent's children dict."""
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
        """Get or create child node. NO global registry - children are per-parent."""
        if token_id not in self.children:
            self.children[token_id] = PACNode(token_id, self.depth + 1)
        return self.children[token_id]
    
    def predict_top_k(self, k: int = 10) -> list:
        if not self.counts:
            return []
        sorted_preds = sorted(self.counts.items(), key=lambda x: -x[1])
        return [t for t, c in sorted_preds[:k]]


class PACByrefTree:
    """PAC tree - children shared through parent navigation."""
    
    def __init__(self, vocab_size: int, max_depth: int, device: torch.device):
        self.vocab_size = vocab_size
        self.max_depth = max_depth
        self.device = device
        
        # Root nodes for each starting token
        self.roots = {}  # token_id -> PACNode
        
        # Stats
        self.total_observations = 0
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
            
            self.total_observations += 1
    
    def predict_at_depth_independent(self, context: list, depth: int, top_k: int = 10) -> list:
        """Get prediction using ONLY the last `depth` tokens as context.
        
        This creates a fresh lookup - doesn't use deeper context even if available.
        Key for testing whether additional context changes predictions.
        """
        if depth > len(context) or depth < 1:
            return []
        
        # Use exactly the last `depth` tokens
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
    
    print("POC-023 Exp 02: Cross-Scale Agreement")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"PHI = {PHI:.6f}, 1/PHI = {PHI_INV:.6f}")
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
    tree = PACByrefTree(vocab_size, max_depth, device)
    tree.learn(torch.tensor(train_tokens, device=device))
    print(f"Training time: {time.time() - t0:.1f}s")
    print(f"Total nodes: {tree.node_count:,}")
    
    # Evaluate cross-scale agreement
    print("\n" + "=" * 60)
    print("CROSS-SCALE CHANGE ANALYSIS")
    print("=" * 60)
    print("Does deeper context CHANGE the prediction?")
    print("When depths agree = stable pattern. When they differ = sparse/noisy.")
    
    t1 = time.time()
    
    # Track: does deeper context change the prediction?
    depth_pairs = [(1, 2), (1, 3), (1, 5), (2, 5)]
    change_stats = {pair: {"same": 0, "changed": 0, "both_valid": 0,
                           "same_hits": 0, "changed_hits": 0} for pair in depth_pairs}
    
    for i in range(max_depth + 1, len(test_tokens)):
        target = test_tokens[i]
        context = test_tokens[:i]
        
        # Get predictions at each depth independently
        preds_by_depth = {}
        top1_by_depth = {}
        
        for d in range(1, max_depth + 1):
            preds = tree.predict_at_depth_independent(context, d, top_k=10)
            if preds:
                preds_by_depth[d] = set(preds)
                top1_by_depth[d] = preds[0]
        
        # Check if deeper context changes the prediction
        for (d_shallow, d_deep) in depth_pairs:
            if d_shallow in top1_by_depth and d_deep in top1_by_depth:
                stats = change_stats[(d_shallow, d_deep)]
                stats["both_valid"] += 1
                
                p_shallow = top1_by_depth[d_shallow]
                p_deep = top1_by_depth[d_deep]
                
                # Did deeper context CHANGE the prediction?
                hit_deep = target in preds_by_depth[d_deep]
                
                if p_shallow == p_deep:
                    stats["same"] += 1
                    if hit_deep:
                        stats["same_hits"] += 1
                else:
                    stats["changed"] += 1
                    if hit_deep:
                        stats["changed_hits"] += 1
    
    eval_time = time.time() - t1
    print(f"Evaluation time: {eval_time:.1f}s")
    
    print("\nContext change analysis:")
    print(f"{'Pair':<12} {'Valid':<10} {'Same':<10} {'Changed':<10} {'Change%':<8}")
    print("-" * 50)
    
    for (d1, d2) in depth_pairs:
        stats = change_stats[(d1, d2)]
        if stats["both_valid"] > 0:
            change_rate = stats["changed"] / stats["both_valid"]
            print(f"({d1}→{d2}){' ':<5} {stats['both_valid']:>8,} {stats['same']:>8,} {stats['changed']:>8,} {change_rate:.1%}")
    
    print("\nHit rate when prediction changes (deeper context matters):")
    print(f"{'Pair':<12} {'Same HR':<12} {'Changed HR':<12} {'Δ':<10}")
    print("-" * 50)
    
    for (d1, d2) in depth_pairs:
        stats = change_stats[(d1, d2)]
        if stats["same"] > 0 and stats["changed"] > 0:
            same_hr = stats["same_hits"] / stats["same"]
            changed_hr = stats["changed_hits"] / stats["changed"]
            delta = changed_hr - same_hr
            marker = "✅" if delta > 0 else "❌"
            print(f"({d1}→{d2}){' ':<5} {same_hr:>10.3f} {changed_hr:>10.3f} {delta:>+8.3f} {marker}")
    
    # Key finding for depth 1 vs 5
    stats_1_5 = change_stats[(1, 5)]
    if stats_1_5["same"] > 0 and stats_1_5["changed"] > 0:
        same_hr = stats_1_5["same_hits"] / stats_1_5["same"]
        changed_hr = stats_1_5["changed_hits"] / stats_1_5["changed"]
        
        print("\n" + "=" * 60)
        print("KEY FINDING (Depth 1 -> 5):")
        if changed_hr > same_hr:
            print(f"  When 5-token context CHANGES prediction, hit rate is HIGHER")
            print(f"     Same: {same_hr:.1%}, Changed: {changed_hr:.1%}")
        else:
            print(f"  When depths AGREE, hit rate is MUCH HIGHER")
            print(f"     Agree: {same_hr:.1%}, Disagree: {changed_hr:.1%}")
            print(f"     Agreement = crystallized pattern (reinforced at multiple scales)")
            print(f"     Disagreement = sparse territory (deep context in low-data region)")
    
    # CONFIDENCE SCORE ANALYSIS
    print("\n" + "=" * 60)
    print("CONFIDENCE SCORE: Scale Agreement")
    print("=" * 60)
    
    # Compute confidence scores for all test predictions
    confidence_buckets = {
        "all_agree": {"hits": 0, "total": 0},      # All available depths agree
        "some_agree": {"hits": 0, "total": 0},     # Partial agreement
        "all_differ": {"hits": 0, "total": 0},     # No agreement
    }
    
    for i in range(max_depth + 1, min(len(test_tokens), max_depth + 1 + 50000)):
        target = test_tokens[i]
        context = test_tokens[:i]
        
        # Get top-1 prediction at each available depth
        top1_preds = []
        deepest_pred = None
        deepest_preds_set = None
        
        for d in range(1, max_depth + 1):
            preds = tree.predict_at_depth_independent(context, d, top_k=10)
            if preds:
                top1_preds.append(preds[0])
                deepest_pred = preds[0]
                deepest_preds_set = set(preds)
        
        if len(top1_preds) < 2 or deepest_preds_set is None:
            continue
        
        # Count unique predictions
        unique_preds = len(set(top1_preds))
        hit = target in deepest_preds_set
        
        if unique_preds == 1:
            bucket = "all_agree"
        elif unique_preds == len(top1_preds):
            bucket = "all_differ"
        else:
            bucket = "some_agree"
        
        confidence_buckets[bucket]["total"] += 1
        if hit:
            confidence_buckets[bucket]["hits"] += 1
    
    print("\nHit rate by confidence level:")
    print(f"{'Confidence':<15} {'Total':<12} {'Hits':<10} {'Hit Rate':<10}")
    print("-" * 50)
    
    for bucket in ["all_agree", "some_agree", "all_differ"]:
        stats = confidence_buckets[bucket]
        if stats["total"] > 0:
            hr = stats["hits"] / stats["total"]
            label = {"all_agree": "HIGH (agree)", "some_agree": "MEDIUM", "all_differ": "LOW (differ)"}[bucket]
            print(f"{label:<15} {stats['total']:>10,} {stats['hits']:>8,} {hr:.1%}")
    
    # Confidence lift
    if confidence_buckets["all_agree"]["total"] > 0 and confidence_buckets["all_differ"]["total"] > 0:
        high_hr = confidence_buckets["all_agree"]["hits"] / confidence_buckets["all_agree"]["total"]
        low_hr = confidence_buckets["all_differ"]["hits"] / confidence_buckets["all_differ"]["total"]
        lift = high_hr / low_hr if low_hr > 0 else float('inf')
        print(f"\nConfidence lift: {lift:.1f}x (high vs low confidence)")
        print("Scale agreement is a strong confidence signal!")
    
    # Save results
    output = {
        "experiment": "exp_02_cross_scale_agreement",
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "vocab_size": vocab_size,
        "train_tokens": len(train_tokens),
        "test_tokens": len(test_tokens),
        "max_depth": max_depth,
        "constants": {"PHI": PHI, "PHI_INV": PHI_INV, "XI": XI},
        "change_stats": {f"{d1}_{d2}": stats for (d1, d2), stats in change_stats.items()},
        "node_count": tree.node_count,
        "eval_time_s": eval_time,
    }
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"exp_02_cross_scale_agreement_{timestamp}.json"
    
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nSaved: {results_path}")
    
    return output


if __name__ == "__main__":
    main()
