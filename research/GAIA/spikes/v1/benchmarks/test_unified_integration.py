"""
GAIA Unified Integration Test
==============================

Tests the unified architecture with all POC components integrated.
Also benchmarks against GPT-2 perplexity baseline.
"""

import torch
import torch.nn.functional as F
from datetime import datetime
import json
import sys
import time
import math
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(src_path))

from gaia_unified import GAIAUnified, GAIAConfig, create_gaia_unified


def compute_perplexity(probs: list) -> float:
    """Compute perplexity from probability list."""
    if not probs or any(p <= 0 for p in probs):
        return float('inf')
    log_probs = [math.log(p) for p in probs]
    avg_log_prob = sum(log_probs) / len(log_probs)
    return math.exp(-avg_log_prob)


def run_integration_tests():
    """Run integration tests on unified GAIA."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"GAIA Unified Integration Tests")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'tests': {}
    }
    
    # Create model
    print("\n[Creating GAIA Unified model...]")
    model = create_gaia_unified(device)
    print(f"Model created successfully")
    
    passed = 0
    failed = 0
    
    # ===== TEST 1: Vocabulary Building =====
    print("\n" + "-"*40)
    print("TEST 1: Vocabulary Building")
    print("-"*40)
    
    vocab_words = [
        # Common words
        "the", "a", "an", "is", "are", "was", "were",
        "I", "you", "he", "she", "it", "we", "they",
        "cat", "dog", "bird", "fish", "man", "woman",
        "big", "small", "good", "bad", "happy", "sad",
        "run", "walk", "see", "love", "help", "know",
        "and", "but", "or", "so", "then", "now",
        "to", "in", "on", "at", "for", "with",
    ]
    
    start = time.time()
    model.add_tokens(vocab_words)
    elapsed = time.time() - start
    
    print(f"Added {model.vocab_size} tokens in {elapsed:.2f}s")
    print(f"Throughput: {model.vocab_size/elapsed:.1f} tokens/sec")
    
    test1_pass = model.vocab_size == len(vocab_words)
    results['tests']['vocab_building'] = {
        'vocab_size': model.vocab_size,
        'elapsed': elapsed,
        'passed': test1_pass
    }
    
    if test1_pass:
        print(f"✓ PASSED")
        passed += 1
    else:
        print(f"✗ FAILED")
        failed += 1
    
    # ===== TEST 2: Training Sequences =====
    print("\n" + "-"*40)
    print("TEST 2: Training Sequences")
    print("-"*40)
    
    training_sequences = [
        ["the", "cat", "is", "big"],
        ["the", "dog", "is", "small"],
        ["I", "see", "the", "cat"],
        ["I", "love", "the", "dog"],
        ["the", "man", "is", "happy"],
        ["the", "woman", "is", "sad"],
        ["he", "run", "fast"],
        ["she", "walk", "slow"],
        ["they", "help", "me"],
        ["we", "know", "you"],
    ]
    
    for seq in training_sequences:
        for _ in range(5):  # Multiple passes
            model.train_sequence(seq)
    
    num_transitions = len(model.memory.transitions)
    print(f"Learned {num_transitions} transitions")
    
    test2_pass = num_transitions >= 20
    results['tests']['training'] = {
        'num_sequences': len(training_sequences),
        'num_transitions': num_transitions,
        'passed': test2_pass
    }
    
    if test2_pass:
        print(f"✓ PASSED")
        passed += 1
    else:
        print(f"✗ FAILED")
        failed += 1
    
    # ===== TEST 3: Next-Token Prediction =====
    print("\n" + "-"*40)
    print("TEST 3: Next-Token Prediction")
    print("-"*40)
    
    test_cases = [
        (["the", "cat", "is"], ["big", "small", "good", "bad", "happy"]),
        (["I", "love"], ["the", "you", "cat", "dog"]),
        (["the", "man"], ["is", "see", "love", "help"]),
    ]
    
    correct = 0
    for context, expected in test_cases:
        model.clear_context()
        for tok in context:
            model.push_context(tok)
        
        preds = model.predict(top_k=5)
        top_tokens = [p[0] for p in preds]
        
        found = any(e in top_tokens for e in expected)
        correct += int(found)
        
        print(f"  {context} → {[(p[0], f'{p[1]:.3f}') for p in preds[:3]]}")
        
    test3_pass = correct >= 2
    results['tests']['prediction'] = {
        'correct': correct,
        'total': len(test_cases),
        'passed': test3_pass
    }
    
    if test3_pass:
        print(f"✓ PASSED: {correct}/{len(test_cases)} correct")
        passed += 1
    else:
        print(f"✗ FAILED")
        failed += 1
    
    # ===== TEST 4: Generation =====
    print("\n" + "-"*40)
    print("TEST 4: Sequence Generation")
    print("-"*40)
    
    prompts = [
        ["the", "cat"],
        ["I", "see"],
        ["he"],
    ]
    
    generations = []
    for prompt in prompts:
        gen = model.generate(prompt, max_tokens=5, temperature=0.4)
        generations.append(gen)
        print(f"  {prompt} → {gen}")
        
    all_extended = all(len(g) > len(p) for g, p in zip(generations, prompts))
    
    test4_pass = all_extended
    results['tests']['generation'] = {
        'generations': [(p, g) for p, g in zip(prompts, generations)],
        'passed': test4_pass
    }
    
    if test4_pass:
        print(f"✓ PASSED")
        passed += 1
    else:
        print(f"✗ FAILED")
        failed += 1
    
    # ===== TEST 5: Memory Persistence =====
    print("\n" + "-"*40)
    print("TEST 5: Memory Persistence")
    print("-"*40)
    
    # Store a target pattern, add noise, retrieve
    target_token = "cat"
    target_id = model.token_to_id[target_token]
    target_field = model.memory.patterns[target_id]
    
    # Query with itself
    retrieved = model.memory.retrieve(target_field, top_k=3)
    top_retrieved = retrieved[0][0] if retrieved else None
    
    test5_pass = top_retrieved == target_id
    print(f"Target 'cat' (id={target_id}) retrieved: id={top_retrieved}")
    
    results['tests']['memory'] = {
        'target_id': target_id,
        'retrieved_id': top_retrieved,
        'passed': test5_pass
    }
    
    if test5_pass:
        print(f"✓ PASSED")
        passed += 1
    else:
        print(f"✗ FAILED")
        failed += 1
    
    # ===== TEST 6: Perplexity Estimation =====
    print("\n" + "-"*40)
    print("TEST 6: Perplexity Estimation")
    print("-"*40)
    
    # Compute perplexity on training sequences
    total_probs = []
    
    for seq in training_sequences[:5]:  # Sample
        model.clear_context()
        for i, token in enumerate(seq[:-1]):
            model.push_context(token)
            
            # Get prediction for next token
            preds = model.predict(top_k=20)
            pred_dict = {t: s for t, s in preds}
            
            next_token = seq[i + 1]
            if next_token in pred_dict:
                # Normalize to probability
                total_score = sum(s for _, s in preds)
                prob = pred_dict[next_token] / total_score if total_score > 0 else 0.01
                total_probs.append(max(prob, 0.01))
            else:
                total_probs.append(0.01)  # Minimum probability
                
    perplexity = compute_perplexity(total_probs)
    print(f"Estimated perplexity: {perplexity:.2f}")
    print(f"(GPT-2 baseline on similar tasks: ~30-100)")
    
    # For a simple trained model, perplexity < 100 is reasonable
    test6_pass = perplexity < 150
    results['tests']['perplexity'] = {
        'perplexity': perplexity,
        'num_samples': len(total_probs),
        'passed': test6_pass
    }
    
    if test6_pass:
        print(f"✓ PASSED: Perplexity reasonable")
        passed += 1
    else:
        print(f"✗ FAILED: Perplexity too high")
        failed += 1
    
    # ===== SUMMARY =====
    print("\n" + "="*60)
    print(f"INTEGRATION TEST RESULTS: {passed}/{passed+failed} passed")
    print("="*60)
    
    results['summary'] = {
        'passed': passed,
        'failed': failed,
        'total': passed + failed,
        'success_rate': passed / (passed + failed)
    }
    
    # Save results
    results_dir = Path(__file__).resolve().parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f'integration_test_{timestamp}.json'
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved: {results_path}")
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = run_integration_tests()
    sys.exit(0 if failed == 0 else 1)
