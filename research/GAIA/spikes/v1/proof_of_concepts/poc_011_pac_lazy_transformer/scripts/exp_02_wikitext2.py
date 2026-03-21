"""
Experiment 02: PAC-Lazy on WikiText-2
=====================================

Tests the PAC-Lazy Transformer on WikiText-2 to compare
against GAIA unified baseline (perplexity 5.91).

Key questions:
1. Can PAC-Lazy achieve comparable perplexity?
2. How does memory scale vs fixed context?
3. Does adaptive depth help with complex sentences?
"""

import torch
import torch.nn.functional as F
import time
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent))

from pac_lazy_core import PACLazySystem, PHI_XI
from pac_lazy_transformer import PACLazyTransformer, PACTransformerConfig


def load_wikitext2(split: str = 'train', max_samples: int = None):
    """Load WikiText-2 dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        
        texts = [x['text'] for x in ds if x['text'].strip()]
        if max_samples:
            texts = texts[:max_samples]
        return texts
    except ImportError:
        print("  datasets library not available, using synthetic data")
        return None


def build_vocab(texts: List[str], vocab_size: int = 5000) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build vocabulary from texts."""
    from collections import Counter
    
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)
    
    # Most common words
    most_common = word_counts.most_common(vocab_size - 2)
    
    word_to_id = {'<unk>': 0, '<pad>': 1}
    for word, _ in most_common:
        word_to_id[word] = len(word_to_id)
    
    id_to_word = {v: k for k, v in word_to_id.items()}
    
    return word_to_id, id_to_word


def tokenize(text: str, word_to_id: Dict[str, int]) -> List[int]:
    """Tokenize text to token IDs."""
    words = text.lower().split()
    return [word_to_id.get(w, 0) for w in words]  # 0 = <unk>


def get_word_embedding(word_id: int, vocab_size: int, dim: int, device: str) -> torch.Tensor:
    """Generate consistent embedding for word ID."""
    # Use word_id as seed for reproducibility (generate on CPU then move)
    gen = torch.Generator().manual_seed(word_id)
    emb = torch.randn(dim, generator=gen)
    emb = F.normalize(emb, dim=0)
    return emb.to(device)


def compute_perplexity(losses: List[float]) -> float:
    """Compute perplexity from list of losses."""
    if not losses:
        return float('inf')
    avg_loss = sum(losses) / len(losses)
    return math.exp(avg_loss)


def train_epoch(model: PACLazyTransformer, 
                texts: List[str],
                word_to_id: Dict[str, int],
                max_seq_len: int = 100) -> Dict:
    """Train for one epoch."""
    
    total_loss = 0
    total_tokens = 0
    losses = []
    
    for text_idx, text in enumerate(texts):
        tokens = tokenize(text, word_to_id)
        if len(tokens) < 2:
            continue
        
        # Truncate
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
        
        model.reset_sequence()
        
        for i in range(len(tokens) - 1):
            token_id = tokens[i]
            next_token = tokens[i + 1]
            
            embedding = get_word_embedding(
                token_id, 
                len(word_to_id), 
                model.embedding_dim,
                model.device
            )
            
            # Process and learn
            model.process_token(token_id, embedding, learn=True)
            model.learn_transition(token_id, next_token)
            
            # Compute loss via prediction
            predictions = model.predict_next(top_k=10)
            
            if predictions:
                # Find rank of correct next token
                pred_dict = {p[0]: p[1] for p in predictions}
                
                if next_token in pred_dict:
                    # Cross-entropy-like loss
                    score = pred_dict[next_token]
                    loss = -math.log(max(score, 1e-10))
                else:
                    # Not in top-k, assign high loss
                    loss = 5.0
                
                losses.append(loss)
                total_loss += loss
                total_tokens += 1
        
        if (text_idx + 1) % 100 == 0:
            print(f"    Processed {text_idx + 1}/{len(texts)} texts")
    
    return {
        'avg_loss': total_loss / max(total_tokens, 1),
        'total_tokens': total_tokens,
        'perplexity': compute_perplexity(losses)
    }


def evaluate(model: PACLazyTransformer,
            texts: List[str],
            word_to_id: Dict[str, int],
            max_seq_len: int = 100) -> Dict:
    """Evaluate model on texts."""
    
    losses = []
    correct = 0
    total = 0
    
    for text in texts:
        tokens = tokenize(text, word_to_id)
        if len(tokens) < 2:
            continue
        
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
        
        model.reset_sequence()
        
        for i in range(len(tokens) - 1):
            token_id = tokens[i]
            next_token = tokens[i + 1]
            
            embedding = get_word_embedding(
                token_id,
                len(word_to_id),
                model.embedding_dim,
                model.device
            )
            
            # Process without learning
            model.process_token(token_id, embedding, learn=False)
            
            # Predict
            predictions = model.predict_next(top_k=10)
            
            if predictions:
                pred_dict = {p[0]: p[1] for p in predictions}
                
                if next_token in pred_dict:
                    score = pred_dict[next_token]
                    loss = -math.log(max(score, 1e-10))
                else:
                    loss = 5.0
                
                losses.append(loss)
                
                # Check if top-1 is correct
                if predictions[0][0] == next_token:
                    correct += 1
                total += 1
    
    return {
        'perplexity': compute_perplexity(losses),
        'accuracy': correct / max(total, 1),
        'total_tokens': total
    }


def run_synthetic_benchmark():
    """Run benchmark on synthetic data."""
    print("\n=== Running Synthetic Benchmark ===")
    
    # Create synthetic vocabulary
    vocab_size = 1000
    word_to_id = {f'word_{i}': i for i in range(vocab_size)}
    id_to_word = {i: f'word_{i}' for i in range(vocab_size)}
    
    # Create synthetic texts with patterns
    import random
    random.seed(42)
    
    train_texts = []
    for _ in range(500):
        # Create sentences with local patterns
        pattern_start = random.randint(0, 100)
        pattern_len = random.randint(3, 8)
        pattern = [pattern_start + i for i in range(pattern_len)]
        
        # Repeat pattern with noise
        sentence = []
        for _ in range(random.randint(3, 6)):
            sentence.extend(pattern)
            # Add noise word
            sentence.append(random.randint(0, vocab_size - 1))
        
        text = ' '.join(f'word_{w}' for w in sentence[:50])
        train_texts.append(text)
    
    # Test texts with same patterns
    test_texts = []
    for _ in range(100):
        pattern_start = random.randint(0, 100)
        pattern_len = random.randint(3, 8)
        pattern = [pattern_start + i for i in range(pattern_len)]
        
        sentence = []
        for _ in range(random.randint(3, 6)):
            sentence.extend(pattern)
            sentence.append(random.randint(0, vocab_size - 1))
        
        text = ' '.join(f'word_{w}' for w in sentence[:50])
        test_texts.append(text)
    
    return train_texts, test_texts, word_to_id, id_to_word


def main():
    print("=" * 60)
    print("POC-011 Experiment 02: PAC-Lazy on WikiText-2")
    print("=" * 60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Target baseline: GAIA unified perplexity = 5.91")
    
    results = {}
    
    # Try to load WikiText-2
    print("\n1. Loading data...")
    train_texts = load_wikitext2('train', max_samples=1000)
    
    if train_texts is None:
        # Use synthetic data
        train_texts, test_texts, word_to_id, id_to_word = run_synthetic_benchmark()
    else:
        test_texts = load_wikitext2('test', max_samples=200)
        print(f"  Train texts: {len(train_texts)}")
        print(f"  Test texts: {len(test_texts)}")
        
        # Build vocabulary
        print("\n2. Building vocabulary...")
        word_to_id, id_to_word = build_vocab(train_texts, vocab_size=5000)
        print(f"  Vocabulary size: {len(word_to_id)}")
    
    # Create model
    print("\n3. Creating PAC-Lazy Transformer...")
    config = PACTransformerConfig(
        embedding_dim=384,
        total_potential=200.0,  # Larger budget for more tokens
        expansion_threshold=PHI_XI,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    model = PACLazyTransformer(config)
    print(f"  Embedding dim: {config.embedding_dim}")
    print(f"  Total potential: {config.total_potential}")
    
    # Train
    print("\n4. Training...")
    start = time.perf_counter()
    train_result = train_epoch(model, train_texts[:500], word_to_id)  # Subset for speed
    train_time = time.perf_counter() - start
    
    print(f"  Train perplexity: {train_result['perplexity']:.2f}")
    print(f"  Train time: {train_time:.2f}s")
    print(f"  Tokens processed: {train_result['total_tokens']}")
    
    results['train'] = {
        'perplexity': train_result['perplexity'],
        'time_s': train_time,
        'tokens': train_result['total_tokens']
    }
    
    # Evaluate
    print("\n5. Evaluating...")
    start = time.perf_counter()
    eval_result = evaluate(model, test_texts[:100], word_to_id)  # Subset for speed
    eval_time = time.perf_counter() - start
    
    print(f"  Test perplexity: {eval_result['perplexity']:.2f}")
    print(f"  Test accuracy: {eval_result['accuracy']:.2%}")
    print(f"  Eval time: {eval_time:.2f}s")
    
    results['test'] = {
        'perplexity': eval_result['perplexity'],
        'accuracy': eval_result['accuracy'],
        'time_s': eval_time,
        'tokens': eval_result['total_tokens']
    }
    
    # Model stats
    stats = model.get_stats()
    results['model_stats'] = stats
    
    print("\n6. Model Statistics...")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Active nodes: {stats['active_nodes']}")
    print(f"  Vocab size: {stats['vocab_size']}")
    print(f"  Expansions: {stats['expansions']}")
    print(f"  Collapses: {stats['collapses']}")
    print(f"  Utilization: {stats['utilization']:.1%}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    gaia_baseline = 5.91
    test_ppl = eval_result['perplexity']
    
    print(f"  GAIA Unified baseline: {gaia_baseline:.2f}")
    print(f"  PAC-Lazy perplexity:   {test_ppl:.2f}")
    
    if test_ppl < gaia_baseline * 2:
        print("  ✅ PAC-Lazy is within 2x of baseline")
    elif test_ppl < gaia_baseline * 5:
        print("  ⚠️ PAC-Lazy needs optimization (2-5x baseline)")
    else:
        print("  ❌ PAC-Lazy needs significant work (>5x baseline)")
    
    print(f"\n  Memory efficiency:")
    print(f"    Active nodes: {stats['active_nodes']}")
    print(f"    Potential utilization: {stats['utilization']:.1%}")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'baseline_perplexity': gaia_baseline,
        'results': results
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = results_dir / f'exp_02_wikitext2_{timestamp}.json'
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
