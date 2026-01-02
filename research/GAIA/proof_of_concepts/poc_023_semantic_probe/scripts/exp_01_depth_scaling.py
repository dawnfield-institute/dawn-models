"""
POC-023 Experiment 01: Depth Scaling with PAC Byref Tree
Test hit rate as a function of n-gram depth using PAC hierarchy.

Key insights:
1. Each depth learns RESIDUALS from previous depth (delta-only)
2. Tree nodes are BYREF - shared when contexts merge
3. "the cat sat" and "a cat sat" share the "cat→sat" node

This is how PAC achieves compression - convergent branches merge.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import GPT2Tokenizer
from datasets import load_dataset

# Dawn Field constants
PHI = 1.6180339887
PHI_INV = 1 / PHI
XI = 0.0618
PHI_XI = PHI * XI


class PACNode:
    """Byref PAC node - shared across branches that converge."""
    __slots__ = ['token_id', 'counts', 'total', 'children', 'depth']
    
    def __init__(self, token_id: int, depth: int):
        self.token_id = token_id
        self.depth = depth
        self.counts = {}  # target_id -> count
        self.total = 0
        self.children = {}  # next_token_id -> PACNode (byref)
    
    def observe(self, target: int):
        """Record observation of target following this context."""
        self.counts[target] = self.counts.get(target, 0) + 1
        self.total += 1
    
    def get_or_create_child(self, token_id: int, node_registry: dict) -> 'PACNode':
        """Get child node, creating if needed. Uses global registry for byref."""
        if token_id in self.children:
            return self.children[token_id]
        
        # Check global registry for existing node at this depth+token
        key = (self.depth + 1, token_id)
        if key in node_registry:
            # BYREF: Reuse existing node
            node = node_registry[key]
        else:
            # Create new node and register
            node = PACNode(token_id, self.depth + 1)
            node_registry[key] = node
        
        self.children[token_id] = node
        return node
    
    def predict_top_k(self, k: int = 10) -> list:
        """Return top-k predictions."""
        if not self.counts:
            return []
        sorted_preds = sorted(self.counts.items(), key=lambda x: -x[1])
        return [t for t, c in sorted_preds[:k]]


class PACByrefTree:
    """PAC tree with byref node sharing.
    
    Nodes are shared when contexts converge:
    - "the cat" and "a cat" share the same "cat" child node
    - This achieves natural compression of the transition space
    """
    
    def __init__(self, vocab_size: int, max_depth: int, device: torch.device):
        self.vocab_size = vocab_size
        self.max_depth = max_depth
        self.device = device
        
        # Global node registry for byref sharing
        # Key: (depth, token_id) -> PACNode
        self.node_registry = {}
        
        # Root nodes for each starting token
        self.roots = {}  # token_id -> PACNode
        
        # Stats
        self.total_observations = 0
    
    def learn(self, tokens: torch.Tensor):
        """Learn from token sequence with byref sharing."""
        tokens_list = tokens.cpu().tolist()
        
        for i in range(1, len(tokens_list)):
            target = tokens_list[i]
            
            # Walk backwards through context depths
            for depth in range(1, min(self.max_depth + 1, i + 1)):
                context_start = i - depth
                context = tokens_list[context_start:i]
                
                # Navigate/create path through tree
                first_token = context[0]
                
                if first_token not in self.roots:
                    key = (0, first_token)
                    if key in self.node_registry:
                        self.roots[first_token] = self.node_registry[key]
                    else:
                        node = PACNode(first_token, 0)
                        self.node_registry[key] = node
                        self.roots[first_token] = node
                
                current = self.roots[first_token]
                
                # Navigate deeper (byref sharing happens in get_or_create_child)
                for j, token in enumerate(context[1:], 1):
                    current = current.get_or_create_child(token, self.node_registry)
                
                # Record observation at this depth
                current.observe(target)
            
            self.total_observations += 1
    
    def predict(self, context: list, top_k: int = 10) -> tuple:
        """Get prediction using deepest available context.
        
        Returns: (predictions, depth_used)
        """
        best_preds = []
        best_depth = 0
        
        # Try increasingly deep contexts
        for depth in range(1, min(self.max_depth + 1, len(context) + 1)):
            ctx = context[-depth:]
            
            first_token = ctx[0]
            if first_token not in self.roots:
                continue
            
            current = self.roots[first_token]
            
            # Navigate to context end
            valid = True
            for token in ctx[1:]:
                if token not in current.children:
                    valid = False
                    break
                current = current.children[token]
            
            if valid and current.total > 0:
                preds = current.predict_top_k(top_k)
                if preds:
                    best_preds = preds
                    best_depth = depth
        
        return best_preds, best_depth
    
    def evaluate(self, tokens: torch.Tensor, top_k: int = 10) -> dict:
        """Evaluate hit rate on test tokens."""
        tokens_list = tokens.cpu().tolist()
        
        hits = 0
        total = 0
        depth_usage = {d: 0 for d in range(self.max_depth + 1)}
        depth_hits = {d: 0 for d in range(self.max_depth + 1)}
        
        for i in range(1, len(tokens_list)):
            target = tokens_list[i]
            context = tokens_list[:i]
            
            preds, depth_used = self.predict(context, top_k)
            
            total += 1
            depth_usage[depth_used] += 1
            
            if target in preds:
                hits += 1
                depth_hits[depth_used] += 1
        
        # Per-depth hit rates
        depth_stats = {}
        for d in range(self.max_depth + 1):
            if depth_usage[d] > 0:
                depth_stats[d] = {
                    "used": depth_usage[d],
                    "hits": depth_hits[d],
                    "hit_rate": depth_hits[d] / depth_usage[d]
                }
        
        return {
            "total": total,
            "hits": hits,
            "hit_rate": hits / total if total > 0 else 0,
            "depth_stats": depth_stats,
            "unique_nodes": len(self.node_registry),
            "root_nodes": len(self.roots),
        }


def load_wikitext() -> list:
    """Load WikiText-2 as token list, using cache if available."""
    print("Loading WikiText-2...")
    
    # Check for cached data from POC-022
    cache_path = Path(__file__).parent.parent.parent / "poc_022_scale_stress_test" / "data_cache" / "wikitext2_10000_64.pt"
    
    if cache_path.exists():
        print(f"Loading from cache: {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        tokens_2d = data["sequences"]  # Shape: (num_seqs, seq_len)
        vocab_size = data["vocab_size"]
        # Flatten to 1D
        tokens = tokens_2d.flatten().cpu().tolist()
        print(f"Loaded {len(tokens):,} tokens from cache")
        return tokens, vocab_size
    
    # Fall back to downloading
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    dataset = load_dataset('wikitext', 'wikitext-2-v1', split='train', trust_remote_code=True)
    all_text = ' '.join([t for t in dataset['text'] if t.strip()])
    tokens = tokenizer.encode(all_text)
    return tokens, tokenizer.vocab_size


def main():
    """Run PAC byref tree depth scaling experiment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("POC-023 Exp 01: PAC Byref Tree Depth Scaling")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"φ = {PHI:.6f}, 1/φ = {PHI_INV:.6f}")
    print()
    print("Key: Nodes are shared (byref) when contexts converge")
    print()
    
    # Load data
    all_tokens, vocab_size = load_wikitext()
    print(f"Total tokens: {len(all_tokens):,}, vocab: {vocab_size:,}")
    
    # Split 80/20 train/test
    split = int(len(all_tokens) * 0.8)
    train_tokens = all_tokens[:split]
    test_tokens = all_tokens[split:]
    print(f"Train: {len(train_tokens):,}, Test: {len(test_tokens):,}")
    
    # Test with max depth 5
    max_depth = 5
    
    print("\n" + "=" * 60)
    print("TRAINING PAC BYREF TREE")
    print("=" * 60)
    
    t0 = time.time()
    tree = PACByrefTree(vocab_size, max_depth, device)
    tree.learn(torch.tensor(train_tokens, device=device))
    train_time = time.time() - t0
    
    print(f"Training time: {train_time:.1f}s")
    print(f"Unique nodes (byref): {len(tree.node_registry):,}")
    print(f"Root nodes: {len(tree.roots):,}")
    
    # Compression ratio
    naive_nodes = sum(vocab_size ** d for d in range(1, max_depth + 1))
    compression = naive_nodes / len(tree.node_registry) if tree.node_registry else 0
    print(f"Compression vs naive: {compression:,.0f}x")
    
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    t1 = time.time()
    metrics = tree.evaluate(torch.tensor(test_tokens, device=device), top_k=10)
    eval_time = time.time() - t1
    print(f"Evaluation time: {eval_time:.1f}s")
    
    print(f"\nOverall hit rate: {metrics['hit_rate']:.3f}")
    
    # Per-depth breakdown
    print("\nPer-depth analysis:")
    print(f"{'Depth':<8} {'Used':<12} {'Hits':<10} {'Hit Rate':<10}")
    print("-" * 45)
    
    for d in sorted(metrics["depth_stats"].keys()):
        stats = metrics["depth_stats"][d]
        print(f"{d:<8} {stats['used']:>10,} {stats['hits']:>8,} {stats['hit_rate']:.3f}")
    
    # Marginal gain analysis
    print("\n" + "=" * 60)
    print("MARGINAL GAIN ANALYSIS")
    print("=" * 60)
    
    depths = sorted(metrics["depth_stats"].keys())
    if len(depths) >= 2:
        hit_rates = [metrics["depth_stats"][d]["hit_rate"] for d in depths]
        print("\nHit rate by depth (when that depth was used):")
        for i, d in enumerate(depths):
            hr = hit_rates[i]
            if i > 0:
                gain = hr - hit_rates[i-1]
                print(f"  Depth {d}: {hr:.3f} (Δ = {gain:+.3f})")
            else:
                print(f"  Depth {d}: {hr:.3f} (base)")
    
    # Save results
    output = {
        "experiment": "exp_01_pac_byref_depth",
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "vocab_size": vocab_size,
        "train_tokens": len(train_tokens),
        "test_tokens": len(test_tokens),
        "max_depth": max_depth,
        "constants": {"PHI": PHI, "PHI_INV": PHI_INV, "XI": XI},
        "metrics": metrics,
        "unique_nodes": len(tree.node_registry),
        "compression_ratio": compression,
        "train_time_s": train_time,
        "eval_time_s": eval_time,
    }
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"exp_01_pac_byref_depth_{timestamp}.json"
    
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nSaved: {results_path}")
    
    return output


if __name__ == "__main__":
    main()
