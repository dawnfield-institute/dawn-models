"""
POC-023 Experiment 04: Hallucination Proxy
Generate from PAC tree and measure chord concentration per token.

Hypothesis: Low chord concentration = sparse territory = "hallucination-like" output
- High concentration tokens: model is confident, pattern is crystallized
- Low concentration tokens: model is guessing, pattern is sparse

Method:
1. Generate sequences from PAC tree
2. Track chord concentration for each generated token
3. Decode and display with concentration annotations
4. Analyze: do low-concentration tokens look "weird"?
"""

import json
import time
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import torch
from transformers import GPT2Tokenizer

# Dawn Field constants
PHI = 1.6180339887
PHI_INV = 1 / PHI
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
        return [(t, c) for t, c in sorted_preds[:k]]
    
    def sample(self, temperature: float = 1.0) -> int:
        """Sample next token based on counts."""
        if not self.counts:
            return None
        
        tokens = list(self.counts.keys())
        counts = list(self.counts.values())
        
        # Apply temperature
        if temperature != 1.0:
            counts = [c ** (1.0 / temperature) for c in counts]
        
        total = sum(counts)
        probs = [c / total for c in counts]
        
        return random.choices(tokens, weights=probs, k=1)[0]


class PACTree:
    """PAC tree for generation with chord concentration tracking."""
    
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
    
    def get_node_at_depth(self, context: list, depth: int) -> PACNode:
        """Get node at specific depth, or None if path doesn't exist."""
        if depth > len(context) or depth < 1:
            return None
        
        ctx = context[-depth:]
        first_token = ctx[0]
        
        if first_token not in self.roots:
            return None
        
        current = self.roots[first_token]
        for token in ctx[1:]:
            if token not in current.children:
                return None
            current = current.children[token]
        
        return current
    
    def compute_chord_concentration(self, context: list) -> tuple:
        """Compute chord concentration for this context.
        
        Returns: (concentration, predictions_by_depth, depths_available)
        """
        top1_by_depth = {}
        
        for d in range(1, self.max_depth + 1):
            node = self.get_node_at_depth(context, d)
            if node and node.total > 0:
                preds = node.predict_top_k(1)
                if preds:
                    top1_by_depth[d] = preds[0][0]  # token_id
        
        if len(top1_by_depth) < 2:
            return None, top1_by_depth, len(top1_by_depth)
        
        # Concentration: 1 = all agree, 0 = all differ
        unique_preds = len(set(top1_by_depth.values()))
        total_depths = len(top1_by_depth)
        concentration = 1.0 - (unique_preds - 1) / (total_depths - 1)
        
        return concentration, top1_by_depth, total_depths
    
    def generate_with_tracking(self, seed_tokens: list, length: int, 
                                temperature: float = 1.0) -> list:
        """Generate tokens and track chord concentration for each.
        
        Returns list of (token_id, concentration, depths_available)
        """
        context = seed_tokens.copy()
        generated = []
        
        for _ in range(length):
            # Compute concentration before generating
            conc, preds_by_depth, depths_avail = self.compute_chord_concentration(context)
            
            # Get deepest available node for sampling
            node = None
            for d in range(self.max_depth, 0, -1):
                node = self.get_node_at_depth(context, d)
                if node and node.total > 0:
                    break
            
            if node is None or node.total == 0:
                break
            
            # Sample next token
            next_token = node.sample(temperature)
            if next_token is None:
                break
            
            generated.append((next_token, conc, depths_avail))
            context.append(next_token)
        
        return generated


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
    
    print("POC-023 Exp 04: Hallucination Proxy")
    print("=" * 60)
    print(f"Device: {device}")
    print()
    print("Hypothesis: Low chord concentration = hallucination territory")
    print()
    
    # Load tokenizer for decoding
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
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
    
    # Generate multiple sequences
    print("\n" + "=" * 60)
    print("GENERATION WITH CONCENTRATION TRACKING")
    print("=" * 60)
    
    num_sequences = 5
    gen_length = 30
    temperature = 0.8
    
    all_concentrations = []
    high_conc_tokens = []
    low_conc_tokens = []
    
    for seq_idx in range(num_sequences):
        # Pick random seed from test set
        seed_start = random.randint(0, len(test_tokens) - max_depth - 1)
        seed_tokens = test_tokens[seed_start:seed_start + max_depth]
        
        # Generate
        generated = tree.generate_with_tracking(seed_tokens, gen_length, temperature)
        
        print(f"\n--- Sequence {seq_idx + 1} ---")
        
        # Decode seed
        seed_text = tokenizer.decode(seed_tokens)
        print(f"Seed: {seed_text}")
        print()
        
        # Decode generated with concentration annotations
        print("Generated (concentration in brackets):")
        output_parts = []
        
        for token_id, conc, depths in generated:
            token_text = tokenizer.decode([token_id])
            
            if conc is not None:
                all_concentrations.append(conc)
                
                # Track high vs low
                if conc > 0.8:
                    high_conc_tokens.append((token_text, conc))
                    marker = "+"  # High confidence
                elif conc < 0.4:
                    low_conc_tokens.append((token_text, conc))
                    marker = "!"  # Low confidence (potential hallucination)
                else:
                    marker = " "
                
                output_parts.append(f"{token_text}[{conc:.1f}{marker}]")
            else:
                output_parts.append(f"{token_text}[?]")
        
        print("".join(output_parts))
    
    # Analysis
    print("\n" + "=" * 60)
    print("CONCENTRATION ANALYSIS")
    print("=" * 60)
    
    if all_concentrations:
        mean_conc = sum(all_concentrations) / len(all_concentrations)
        high_count = sum(1 for c in all_concentrations if c > 0.8)
        low_count = sum(1 for c in all_concentrations if c < 0.4)
        
        print(f"\nTotal generated tokens: {len(all_concentrations)}")
        print(f"Mean concentration: {mean_conc:.3f}")
        print(f"High confidence (>0.8): {high_count} ({100*high_count/len(all_concentrations):.1f}%)")
        print(f"Low confidence (<0.4): {low_count} ({100*low_count/len(all_concentrations):.1f}%)")
        
        print("\n--- HIGH CONFIDENCE TOKENS (crystallized) ---")
        for token, conc in high_conc_tokens[:20]:
            print(f"  '{token}' ({conc:.2f})")
        
        print("\n--- LOW CONFIDENCE TOKENS (potential hallucinations) ---")
        for token, conc in low_conc_tokens[:20]:
            print(f"  '{token}' ({conc:.2f})")
    
    # Compare to real test data concentration
    print("\n" + "=" * 60)
    print("COMPARISON: Generated vs Real Text")
    print("=" * 60)
    
    real_concentrations = []
    sample_size = min(1000, len(test_tokens) - max_depth - 1)
    
    for i in range(max_depth + 1, max_depth + 1 + sample_size):
        context = test_tokens[:i]
        conc, _, depths = tree.compute_chord_concentration(context)
        if conc is not None:
            real_concentrations.append(conc)
    
    if real_concentrations and all_concentrations:
        real_mean = sum(real_concentrations) / len(real_concentrations)
        gen_mean = sum(all_concentrations) / len(all_concentrations)
        
        print(f"\nReal text mean concentration: {real_mean:.3f}")
        print(f"Generated mean concentration: {gen_mean:.3f}")
        print(f"Difference: {gen_mean - real_mean:+.3f}")
        
        if gen_mean < real_mean:
            print("\n! Generated text has LOWER concentration than real text")
            print("  This suggests generation is drifting into sparse territory")
        else:
            print("\nGenerated text has similar or higher concentration")
            print("  Generation is staying in crystallized pattern space")
    
    # Save results
    output = {
        "experiment": "exp_04_hallucination_proxy",
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "vocab_size": vocab_size,
        "train_tokens": len(train_tokens),
        "max_depth": max_depth,
        "gen_length": gen_length,
        "temperature": temperature,
        "num_sequences": num_sequences,
        "generated_concentration_mean": mean_conc if all_concentrations else None,
        "real_concentration_mean": real_mean if real_concentrations else None,
        "high_confidence_count": high_count if all_concentrations else 0,
        "low_confidence_count": low_count if all_concentrations else 0,
        "node_count": tree.node_count,
    }
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"exp_04_hallucination_proxy_{timestamp}.json"
    
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nSaved: {results_path}")
    
    return output


if __name__ == "__main__":
    main()
