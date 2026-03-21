"""
Experiment 01: WikiText-103 Full Scale Training
===============================================

Train GAIA on WikiText-103 (100M+ tokens).
Compare with published baselines.
"""

import torch
import json
import time
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add paths
src_path = Path(__file__).resolve().parents[3] / 'src'
training_path = Path(__file__).resolve().parents[3] / 'training'
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(training_path))

from gaia_unified import GAIAUnified, GAIAConfig


def tokenize(text: str):
    """Simple tokenization."""
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text.split()


def run_wikitext103_training(max_minutes: float = 60.0):
    """Train on WikiText-103 with time limit."""
    
    print("\n" + "="*60)
    print("WikiText-103 Scale Validation")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load dataset
    from datasets import load_dataset
    print("\nLoading WikiText-103...")
    start_load = time.time()
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
    print(f"Loaded in {time.time() - start_load:.1f}s")
    
    train_data = dataset['train']
    val_data = dataset['validation']
    print(f"Train samples: {len(train_data):,}")
    print(f"Val samples: {len(val_data):,}")
    
    # Build vocabulary from training data
    print("\n" + "-"*40)
    print("Building vocabulary...")
    print("-"*40)
    
    word_counts = {}
    sentences_scanned = 0
    
    for item in train_data:
        text = item['text'].strip()
        if text:
            tokens = tokenize(text)
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
            sentences_scanned += 1
            
        if sentences_scanned % 50000 == 0:
            print(f"  Scanned {sentences_scanned:,} samples, {len(word_counts):,} unique words")
            
    # Filter by frequency (min 5 occurrences)
    min_freq = 5
    vocab = [w for w, c in sorted(word_counts.items(), key=lambda x: -x[1]) 
             if c >= min_freq][:30000]  # Cap at 30K
    
    print(f"\nVocabulary: {len(vocab):,} tokens (min freq {min_freq})")
    
    # Initialize model
    print("\n" + "-"*40)
    print("Initializing model...")
    print("-"*40)
    
    config = GAIAConfig(
        field_shape=(32, 32, 32),
        memory_capacity=200000,
        device=device
    )
    
    model = GAIAUnified(config)
    model = model.to(device)
    
    # Add vocabulary
    start_encode = time.time()
    model.add_tokens(vocab)
    print(f"Encoded {model.vocab_size} tokens in {time.time() - start_encode:.1f}s")
    
    # Training
    print("\n" + "-"*40)
    print("Training...")
    print("-"*40)
    
    start_time = time.time()
    max_seconds = max_minutes * 60
    
    sentences_processed = 0
    tokens_processed = 0
    batch_start = time.time()
    
    results = {
        'start_time': datetime.now().isoformat(),
        'device': device,
        'vocab_size': model.vocab_size,
        'checkpoints': []
    }
    
    for item in train_data:
        # Time check
        elapsed = time.time() - start_time
        if elapsed > max_seconds:
            print(f"\nTime limit reached ({max_minutes} minutes)")
            break
            
        text = item['text'].strip()
        if not text:
            continue
            
        tokens = tokenize(text)
        known = [t for t in tokens if t in model.token_to_id]
        
        if len(known) >= 2:
            # Direct transition learning
            ids = [model.token_to_id[t] for t in known]
            for i in range(len(ids) - 1):
                model.memory.learn_transition(ids[i], ids[i + 1])
            tokens_processed += len(known)
            
        sentences_processed += 1
        
        # Progress every 10K
        if sentences_processed % 10000 == 0:
            batch_elapsed = time.time() - batch_start
            rate = 10000 / max(batch_elapsed, 0.001)
            transitions = len(model.memory.transitions)
            print(f"  {sentences_processed:,} sentences, {transitions:,} transitions, {rate:.0f} sent/sec")
            batch_start = time.time()
            
        # Checkpoint every 100K
        if sentences_processed % 100000 == 0:
            checkpoint = {
                'sentences': sentences_processed,
                'tokens': tokens_processed,
                'transitions': len(model.memory.transitions),
                'elapsed_minutes': (time.time() - start_time) / 60
            }
            results['checkpoints'].append(checkpoint)
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    # Final stats
    total_time = time.time() - start_time
    results['total_time_minutes'] = total_time / 60
    results['sentences_processed'] = sentences_processed
    results['tokens_processed'] = tokens_processed
    results['transitions'] = len(model.memory.transitions)
    
    print("\n" + "-"*40)
    print("Training complete")
    print("-"*40)
    print(f"Time: {total_time / 60:.1f} minutes")
    print(f"Sentences: {sentences_processed:,}")
    print(f"Tokens: {tokens_processed:,}")
    print(f"Transitions: {len(model.memory.transitions):,}")
    
    # Evaluation
    print("\n" + "-"*40)
    print("Evaluating on validation set...")
    print("-"*40)
    
    # Rebuild cache before eval
    model.memory._cache_dirty = True
    model.memory._rebuild_cache()
    
    import math
    total_log_prob = 0.0
    total_tokens = 0
    eval_count = 0
    
    with torch.no_grad():
        for item in val_data:
            if eval_count >= 500:  # Limit eval for speed
                break
                
            text = item['text'].strip()
            if not text:
                continue
                
            tokens = tokenize(text)
            known = [t for t in tokens if t in model.token_to_id]
            
            if len(known) >= 3:
                model.clear_context()
                
                for j in range(len(known) - 1):
                    model.push_context(known[j])
                    
                    if j >= 1:
                        preds = model.predict(top_k=100)
                        
                        next_token = known[j + 1]
                        prob = 1e-6  # Smoothing
                        
                        for tok, p in preds:
                            if tok == next_token:
                                prob = max(p, 1e-10)
                                break
                                
                        total_log_prob += math.log(prob)
                        total_tokens += 1
                        
                eval_count += 1
                
    if total_tokens > 0:
        perplexity = math.exp(-total_log_prob / total_tokens)
    else:
        perplexity = float('inf')
        
    results['perplexity'] = perplexity
    results['eval_tokens'] = total_tokens
    
    print(f"\nPerplexity: {perplexity:.2f}")
    print(f"Eval tokens: {total_tokens:,}")
    
    # Baselines comparison
    print("\n" + "="*60)
    print("Comparison with Baselines")
    print("="*60)
    
    baselines = {
        'GPT-2 Large (774M)': 19.93,
        'Transformer-XL Base': 23.6,
        'GPT-2 Medium (355M)': 22.76,
        'XLNet Base': 23.5,
        'BERT Large': 25.8
    }
    
    for name, ppl in sorted(baselines.items(), key=lambda x: x[1]):
        ratio = perplexity / ppl
        symbol = "✅" if ratio < 1 else "⚠️" if ratio < 1.5 else "❌"
        print(f"  {name:25} {ppl:6.2f}  (GAIA is {ratio:.2f}x) {symbol}")
        
    print(f"  {'GAIA (field-native)':25} {perplexity:6.2f}  ← Our result")
    
    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f'exp_01_wikitext103_{timestamp}.json'
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved: {results_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--minutes', type=float, default=30.0)
    args = parser.parse_args()
    
    results = run_wikitext103_training(max_minutes=args.minutes)
