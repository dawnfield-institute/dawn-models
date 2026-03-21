"""
Experiment 04: PAC-Lazy + GAIA Transitions
==========================================

Integrates PAC-Lazy structural memory with GAIA's transition-based
prediction for a fair comparison with the 5.91 baseline.

The key insight: GAIA's low perplexity comes from direct transition
probabilities, not contextual embeddings. We combine:
1. PAC-Lazy for structural memory (nodes, neighbors, potential)
2. GAIA transitions for prediction (word-to-word probabilities)
"""

import torch
import torch.nn.functional as F
import json
import math
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# PAC constants
PHI = 1.6180339887
XI = 0.0618
PHI_XI = 1.710
LAMBDA_STAR = 0.9816


class PACTransitionMemory:
    """
    PAC-bounded transition memory.
    
    Like GAIA's transition approach, but with:
    - PAC potential limiting active patterns
    - SEC expansion for new patterns
    - Structural fracture for forgetting
    """
    
    def __init__(self, total_potential: float = 100.0, device: str = 'cuda'):
        self.device = device
        self.total_potential = total_potential
        self.current_potential = total_potential
        
        # Token vocabulary (learned from data)
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # Transition weights: (from_id, to_id) -> weight
        self.transitions: Dict[Tuple[int, int], float] = defaultdict(float)
        
        # Indexed by from_id for fast lookup
        self.from_index: Dict[int, Dict[int, float]] = defaultdict(dict)
        
        # Outgoing transitions for normalization
        self.outgoing_sum: Dict[int, float] = defaultdict(float)
        
        # PAC tracking
        self.expansions = 0
        self.collapses = 0
        
        # Context (last N tokens)
        self.context: List[int] = []
        self.max_context = 5
        
        # Cost scales (very small to allow many patterns)
        self.token_cost = 0.0001  # Cost per new token
        self.transition_cost = 0.00001  # Cost per new transition
        
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)
    
    def add_token(self, token: str) -> int:
        """Add token to vocabulary (costs potential)."""
        if token in self.token_to_id:
            return self.token_to_id[token]
        
        # Check potential budget
        if self.current_potential < self.token_cost:
            # Need to collapse (forget least used transitions)
            self._collapse()
        
        token_id = len(self.token_to_id)
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        self.current_potential -= self.token_cost
        
        return token_id
    
    def learn_transition(self, from_id: int, to_id: int, weight: float = 1.0):
        """Learn transition between tokens (costs potential for new patterns)."""
        key = (from_id, to_id)
        
        is_new = key not in self.transitions
        
        if is_new:
            # Check if we can afford new transition
            if self.current_potential < self.transition_cost:
                self._collapse()
            
            self.current_potential -= self.transition_cost
            self.expansions += 1
        
        # Update transition weight (accumulate counts)
        old_weight = self.transitions[key]
        new_weight = old_weight + weight  # Simple count accumulation
        
        # Update tracking
        self.outgoing_sum[from_id] -= old_weight
        self.transitions[key] = new_weight
        self.from_index[from_id][to_id] = new_weight  # Indexed lookup
        self.outgoing_sum[from_id] += new_weight
    
    def _collapse(self):
        """Collapse least-used transitions (reclaim potential)."""
        if not self.transitions:
            return
        
        # Find weakest transitions
        sorted_trans = sorted(self.transitions.items(), key=lambda x: x[1])
        
        # Remove bottom 10%
        n_remove = max(1, len(sorted_trans) // 10)
        for (from_id, to_id), weight in sorted_trans[:n_remove]:
            self.outgoing_sum[from_id] -= weight
            del self.transitions[(from_id, to_id)]
            if to_id in self.from_index.get(from_id, {}):
                del self.from_index[from_id][to_id]
            self.current_potential += self.transition_cost
        
        self.collapses += 1
    
    def push_context(self, token: str):
        """Add token to context."""
        if token not in self.token_to_id:
            self.add_token(token)
        
        self.context.append(self.token_to_id[token])
        if len(self.context) > self.max_context:
            self.context.pop(0)
    
    def clear_context(self):
        """Clear context."""
        self.context = []
    
    def predict(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Predict next token from context."""
        if not self.context:
            return []
        
        # Get last token
        last_id = self.context[-1]
        
        # Use indexed lookup - O(1) instead of O(n)
        candidates = []
        total = self.outgoing_sum.get(last_id, 0) + 1e-10
        
        for to_id, weight in self.from_index.get(last_id, {}).items():
            prob = weight / total
            token = self.id_to_token.get(to_id, '<unk>')
            candidates.append((token, prob))
        
        # Sort by probability
        candidates.sort(key=lambda x: -x[1])
        
        return candidates[:top_k]
    
    def get_stats(self) -> Dict:
        return {
            'vocab_size': self.vocab_size,
            'n_transitions': len(self.transitions),
            'current_potential': self.current_potential,
            'utilization': 1 - (self.current_potential / self.total_potential),
            'expansions': self.expansions,
            'collapses': self.collapses
        }


def load_wikitext2(split: str, max_samples: int = None) -> Optional[List[str]]:
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        texts = [x['text'] for x in ds if x['text'].strip()]
        return texts[:max_samples] if max_samples else texts
    except ImportError:
        return None


def tokenize(text: str) -> List[str]:
    """Simple word tokenization."""
    import re
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text.split()


def train_on_corpus(memory: PACTransitionMemory, texts: List[str]) -> Dict:
    """Train on corpus."""
    n_tokens = 0
    
    for text in texts:
        tokens = tokenize(text)
        if len(tokens) < 2:
            continue
        
        # Add all tokens
        for token in tokens:
            memory.add_token(token)
        
        # Learn transitions
        for i in range(len(tokens) - 1):
            from_id = memory.token_to_id[tokens[i]]
            to_id = memory.token_to_id[tokens[i + 1]]
            memory.learn_transition(from_id, to_id)
            n_tokens += 1
    
    return {'tokens_processed': n_tokens}


def calculate_perplexity(memory: PACTransitionMemory, texts: List[str]) -> Dict:
    """Calculate perplexity on test set."""
    total_log_prob = 0.0
    total_tokens = 0
    oov_tokens = 0
    correct = 0
    
    for text in texts:
        tokens = tokenize(text)
        if len(tokens) < 3:
            continue
        
        # Filter to known tokens
        known = [t for t in tokens if t in memory.token_to_id]
        oov_tokens += len(tokens) - len(known)
        
        if len(known) < 3:
            continue
        
        memory.clear_context()
        
        for i in range(len(known) - 1):
            memory.push_context(known[i])
            
            if i >= 1:  # Need context
                next_token = known[i + 1]
                next_id = memory.token_to_id.get(next_token)
                last_id = memory.context[-1]
                
                # Direct lookup - O(1)
                total = memory.outgoing_sum.get(last_id, 0) + 1e-10
                weight = memory.from_index.get(last_id, {}).get(next_id, 0)
                prob = max(weight / total, 1e-10)
                
                # Check top prediction
                top_candidates = memory.from_index.get(last_id, {})
                if top_candidates:
                    top_id = max(top_candidates, key=top_candidates.get)
                    if memory.id_to_token.get(top_id) == next_token:
                        correct += 1
                
                total_log_prob += math.log(prob)
                total_tokens += 1
    
    avg_log_prob = total_log_prob / max(total_tokens, 1)
    perplexity = math.exp(-avg_log_prob)
    accuracy = correct / max(total_tokens, 1)
    
    return {
        'perplexity': perplexity,
        'accuracy': accuracy,
        'tokens_evaluated': total_tokens,
        'oov_tokens': oov_tokens
    }


def main():
    print("=" * 60)
    print("POC-011 Exp 04: PAC-Lazy + GAIA Transitions")
    print("=" * 60)
    print("Combining PAC structural memory with transition prediction")
    
    # Load data
    print("\n1. Loading WikiText-2...")
    train_texts = load_wikitext2('train')
    test_texts = load_wikitext2('test')
    
    if train_texts is None:
        print("  datasets not available")
        return
    
    print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")
    
    # Create PAC memory
    print("\n2. Creating PAC Transition Memory...")
    memory = PACTransitionMemory(
        total_potential=100.0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train
    print("\n3. Training on corpus...")
    start = time.perf_counter()
    train_result = train_on_corpus(memory, train_texts)
    train_time = time.perf_counter() - start
    
    stats = memory.get_stats()
    print(f"  Tokens processed: {train_result['tokens_processed']:,}")
    print(f"  Vocabulary size: {stats['vocab_size']:,}")
    print(f"  Transitions: {stats['n_transitions']:,}")
    print(f"  Train time: {train_time:.2f}s")
    print(f"  Potential utilization: {stats['utilization']:.1%}")
    print(f"  Expansions: {stats['expansions']:,}")
    print(f"  Collapses: {stats['collapses']}")
    
    # Evaluate
    print("\n4. Evaluating perplexity...")
    start = time.perf_counter()
    eval_result = calculate_perplexity(memory, test_texts)
    eval_time = time.perf_counter() - start
    
    print(f"  Test perplexity: {eval_result['perplexity']:.2f}")
    print(f"  Accuracy (top-1): {eval_result['accuracy']:.2%}")
    print(f"  Tokens evaluated: {eval_result['tokens_evaluated']:,}")
    print(f"  OOV tokens: {eval_result['oov_tokens']:,}")
    print(f"  Eval time: {eval_time:.2f}s")
    
    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    gaia_baseline = 5.91
    our_ppl = eval_result['perplexity']
    
    print(f"  GAIA unified baseline:    {gaia_baseline:.2f}")
    print(f"  PAC Transition Memory:    {our_ppl:.2f}")
    print(f"  Ratio:                    {our_ppl/gaia_baseline:.2f}x")
    
    if our_ppl < gaia_baseline * 1.5:
        print("  ✅ Within 1.5x of GAIA baseline!")
    elif our_ppl < gaia_baseline * 2:
        print("  ✅ Within 2x of GAIA baseline")
    elif our_ppl < gaia_baseline * 5:
        print("  ⚠️ Within 5x (comparable approach)")
    else:
        print("  ⚠️ Higher than expected")
    
    print("\nNote: Both use transition-based prediction (apples-to-apples)")
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'training': {
            'tokens': train_result['tokens_processed'],
            'time_s': train_time
        },
        'model_stats': stats,
        'evaluation': eval_result,
        'baseline_comparison': {
            'gaia_baseline': gaia_baseline,
            'ratio': our_ppl / gaia_baseline
        }
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = results_dir / f'exp_04_pac_transitions_{timestamp}.json'
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
