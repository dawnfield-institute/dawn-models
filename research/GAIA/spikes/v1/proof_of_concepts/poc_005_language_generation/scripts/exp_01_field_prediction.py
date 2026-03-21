"""
Experiment 01: Field-Based Next-Token Prediction
=================================================

Tests whether field evolution can predict plausible next tokens.

Hypothesis: Field dynamics naturally point toward likely continuations
by evolving the context field and finding nearby vocabulary tokens.

Success Criteria:
- Top-5 predictions include sensible continuations
- Higher-frequency patterns have higher prediction scores
- Context influences predictions appropriately
"""

import torch
import torch.nn.functional as F
from datetime import datetime
import json
import sys
from pathlib import Path

# Add scripts to path
scripts_path = Path(__file__).resolve().parent
sys.path.insert(0, str(scripts_path))

from field_generator import FieldVocabulary, FieldPredictor, FieldGenerator

# Dawn Field Constants
PHI = 1.618033988749895
XI = 0.0618
PHI_XI = PHI * XI  # 1.710...
LAMBDA_STAR = 0.9816

def run_tests():
    """Run all field prediction tests."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"POC-005 Experiment 01: Field-Based Prediction")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'tests': {}
    }
    
    # Initialize vocabulary with common words
    print("\n[Initializing vocabulary...]")
    vocab = FieldVocabulary(device=device, field_shape=(24, 24, 24))
    
    # Core vocabulary for tests
    core_vocab = [
        # Pronouns
        "I", "you", "he", "she", "it", "we", "they",
        # Verbs
        "am", "is", "are", "was", "were", "have", "has", "had",
        "go", "goes", "went", "come", "comes", "came",
        "see", "sees", "saw", "look", "looks", "looked",
        "like", "likes", "liked", "love", "loves", "loved",
        "want", "wants", "wanted", "need", "needs", "needed",
        "think", "thinks", "thought", "know", "knows", "knew",
        "say", "says", "said", "tell", "tells", "told",
        "run", "runs", "ran", "walk", "walks", "walked",
        "eat", "eats", "ate", "drink", "drinks", "drank",
        # Nouns
        "cat", "dog", "bird", "fish", "horse", "cow",
        "house", "home", "car", "tree", "book", "phone",
        "food", "water", "fire", "earth", "sky", "sun", "moon",
        "man", "woman", "child", "boy", "girl", "baby",
        "mother", "father", "friend", "teacher", "doctor",
        # Adjectives
        "big", "small", "good", "bad", "happy", "sad",
        "fast", "slow", "hot", "cold", "new", "old",
        "red", "blue", "green", "white", "black",
        # Common words
        "the", "a", "an", "to", "in", "on", "at", "for",
        "of", "with", "and", "but", "or", "if", "so", "very",
        "this", "that", "here", "there", "now", "then",
    ]
    
    vocab.add_tokens(core_vocab)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create predictor
    predictor = FieldPredictor(vocab)
    
    # Training sequences (common patterns)
    training_sequences = [
        ["I", "am", "happy"],
        ["I", "am", "sad"],
        ["I", "like", "the", "cat"],
        ["I", "like", "the", "dog"],
        ["I", "love", "you"],
        ["the", "cat", "is", "big"],
        ["the", "dog", "is", "small"],
        ["the", "cat", "is", "fast"],
        ["the", "dog", "is", "good"],
        ["I", "see", "the", "cat"],
        ["I", "see", "the", "dog"],
        ["the", "sun", "is", "hot"],
        ["the", "moon", "is", "cold"],
        ["I", "want", "food"],
        ["I", "want", "water"],
        ["the", "child", "is", "happy"],
        ["the", "mother", "is", "good"],
        ["I", "go", "home"],
        ["I", "go", "to", "the", "house"],
        ["you", "are", "good"],
        ["you", "are", "happy"],
        ["he", "is", "fast"],
        ["she", "is", "good"],
    ]
    
    print(f"\n[Training on {len(training_sequences)} sequences...]")
    for seq in training_sequences:
        predictor.train_on_sequence(seq)
    
    passed = 0
    failed = 0
    
    # ===== TEST 1: Basic prediction =====
    print("\n" + "-"*40)
    print("TEST 1: Basic Prediction")
    print("-"*40)
    
    context = ["I", "am"]
    predictions = predictor.predict(context, top_k=5)
    
    print(f"Context: {context}")
    print(f"Top 5 predictions:")
    for token, score in predictions:
        print(f"  {token}: {score:.4f}")
    
    # Check if common continuations are in top 5
    top_tokens = [p[0] for p in predictions]
    expected = ["happy", "sad", "good", "fast"]
    found = any(e in top_tokens for e in expected)
    
    test1_pass = found
    results['tests']['basic_prediction'] = {
        'context': context,
        'predictions': predictions[:5],
        'passed': test1_pass
    }
    
    if test1_pass:
        print("✓ PASSED: Found expected continuation")
        passed += 1
    else:
        print("✗ FAILED: No expected continuation in top 5")
        failed += 1
    
    # ===== TEST 2: Context sensitivity =====
    print("\n" + "-"*40)
    print("TEST 2: Context Sensitivity")
    print("-"*40)
    
    context1 = ["the", "cat", "is"]
    context2 = ["I", "like"]
    
    pred1 = predictor.predict(context1, top_k=5)
    pred2 = predictor.predict(context2, top_k=5)
    
    print(f"Context 1: {context1}")
    print(f"  Predictions: {[(p[0], f'{p[1]:.3f}') for p in pred1[:3]]}")
    print(f"Context 2: {context2}")
    print(f"  Predictions: {[(p[0], f'{p[1]:.3f}') for p in pred2[:3]]}")
    
    # Different contexts should give different predictions
    top1 = set([p[0] for p in pred1[:3]])
    top2 = set([p[0] for p in pred2[:3]])
    
    test2_pass = top1 != top2  # Should be different
    results['tests']['context_sensitivity'] = {
        'context1': context1,
        'context2': context2,
        'predictions1': pred1[:3],
        'predictions2': pred2[:3],
        'different': test2_pass,
        'passed': test2_pass
    }
    
    if test2_pass:
        print("✓ PASSED: Different contexts give different predictions")
        passed += 1
    else:
        print("✗ FAILED: Predictions not context-sensitive")
        failed += 1
    
    # ===== TEST 3: Sequence generation =====
    print("\n" + "-"*40)
    print("TEST 3: Sequence Generation")
    print("-"*40)
    
    generator = FieldGenerator(predictor)
    
    prompt = ["I", "like"]
    generated = generator.generate(prompt, max_tokens=5, temperature=0.5)
    
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
    
    # Check generation produced something
    test3_pass = len(generated) > len(prompt)
    results['tests']['sequence_generation'] = {
        'prompt': prompt,
        'generated': generated,
        'tokens_added': len(generated) - len(prompt),
        'passed': test3_pass
    }
    
    if test3_pass:
        print(f"✓ PASSED: Generated {len(generated) - len(prompt)} tokens")
        passed += 1
    else:
        print("✗ FAILED: No tokens generated")
        failed += 1
    
    # ===== TEST 4: Resonance scores =====
    print("\n" + "-"*40)
    print("TEST 4: Resonance-Based Generation")
    print("-"*40)
    
    prompt = ["the", "cat"]
    generated, scores = generator.generate_with_resonance(prompt, max_tokens=5)
    
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
    print(f"Resonance scores: {[f'{s:.3f}' for s in scores]}")
    
    # Check we got scores and they're reasonable
    test4_pass = len(scores) > 0 and all(0 <= s <= 1 for s in scores)
    results['tests']['resonance_generation'] = {
        'prompt': prompt,
        'generated': generated,
        'scores': scores,
        'passed': test4_pass
    }
    
    if test4_pass:
        print(f"✓ PASSED: Resonance scores in valid range")
        passed += 1
    else:
        print("✗ FAILED: Invalid resonance scores")
        failed += 1
    
    # ===== TEST 5: Transition learning =====
    print("\n" + "-"*40)
    print("TEST 5: Transition Learning Effect")
    print("-"*40)
    
    # Add many examples of a specific pattern
    for _ in range(10):
        predictor.train_on_sequence(["I", "love", "cat"])
    
    # Predict after training
    pred_after = predictor.predict(["I", "love"], top_k=5)
    
    print(f"After reinforcing 'I love cat' 10x:")
    print(f"  Predictions for ['I', 'love']: {[(p[0], f'{p[1]:.3f}') for p in pred_after[:3]]}")
    
    # Cat should be in top predictions
    top_after = [p[0] for p in pred_after[:3]]
    test5_pass = "cat" in top_after
    results['tests']['transition_learning'] = {
        'pattern': ["I", "love", "cat"],
        'reinforcements': 10,
        'predictions': pred_after[:5],
        'cat_in_top3': test5_pass,
        'passed': test5_pass
    }
    
    if test5_pass:
        print("✓ PASSED: 'cat' in top predictions after training")
        passed += 1
    else:
        print("✗ FAILED: Training didn't reinforce pattern")
        failed += 1
    
    # ===== TEST 6: Multiple generation samples =====
    print("\n" + "-"*40)
    print("TEST 6: Generation Diversity")
    print("-"*40)
    
    prompts_and_results = []
    test_prompts = [
        ["I"],
        ["the"],
        ["you", "are"],
        ["I", "see"],
    ]
    
    for prompt in test_prompts:
        gen = generator.generate(prompt, max_tokens=4, temperature=0.5)
        prompts_and_results.append((prompt, gen))
        print(f"  {prompt} → {gen}")
    
    # Check that generations are varied
    all_unique = len(set([tuple(g) for _, g in prompts_and_results])) == len(test_prompts)
    test6_pass = all(len(g) > len(p) for p, g in prompts_and_results)
    
    results['tests']['generation_diversity'] = {
        'samples': [(p, g) for p, g in prompts_and_results],
        'all_unique': all_unique,
        'all_extended': test6_pass,
        'passed': test6_pass
    }
    
    if test6_pass:
        print(f"✓ PASSED: All prompts successfully extended")
        passed += 1
    else:
        print("✗ FAILED: Some prompts not extended")
        failed += 1
    
    # ===== SUMMARY =====
    print("\n" + "="*60)
    print(f"EXPERIMENT 01 RESULTS: {passed}/{passed+failed} passed")
    print("="*60)
    
    results['summary'] = {
        'passed': passed,
        'failed': failed,
        'total': passed + failed,
        'success_rate': passed / (passed + failed)
    }
    
    # Save results
    results_dir = Path(__file__).resolve().parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f'exp_01_field_prediction_{timestamp}.json'
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved: {results_path}")
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = run_tests()
    sys.exit(0 if failed == 0 else 1)
