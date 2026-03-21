"""
Experiment 04: Coherent Multi-Sentence Generation
==================================================

Tests whether field dynamics can generate coherent multi-token
sequences that maintain semantic consistency.

Hypothesis: Field evolution with resonance creates stable
attractors that produce thematically coherent output.

Success Criteria:
- Generate 10+ tokens without degeneration
- Maintain topical consistency within generation
- Show narrative structure (beginning, continuation)
- Conservation metrics stable through generation
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

from field_generator import FieldVocabulary, FieldPredictor, FieldGenerator, PHI, XI, PHI_XI, LAMBDA_STAR


def compute_field_energy(field: torch.Tensor) -> float:
    """Compute field energy (for conservation check)."""
    return (field ** 2).sum().item()


def run_tests():
    """Run coherent generation tests."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"POC-005 Experiment 04: Coherent Generation")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'tests': {}
    }
    
    # Build rich vocabulary
    print("\n[Building vocabulary...]")
    vocab = FieldVocabulary(device=device, field_shape=(24, 24, 24))
    
    # Narrative vocabulary
    words = {
        'subjects': ["cat", "dog", "bird", "man", "woman", "child", "king", "queen"],
        'verbs': ["runs", "walks", "sees", "loves", "helps", "finds", "gives", "takes"],
        'objects': ["food", "water", "book", "house", "gold", "crown", "apple", "bread"],
        'adjectives': ["big", "small", "happy", "sad", "fast", "slow", "good", "bad"],
        'articles': ["the", "a"],
        'prepositions': ["to", "from", "with", "for", "in", "on"],
        'connectors': ["and", "but", "then", "so", "now"],
        'pronouns': ["I", "you", "he", "she", "it", "they", "we"],
    }
    
    all_words = []
    for category in words.values():
        all_words.extend(category)
    vocab.add_tokens(list(set(all_words)))
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create predictor and train
    predictor = FieldPredictor(vocab)
    
    # Training sentences (simple narratives)
    narratives = [
        # Subject + Verb + Object patterns
        ["the", "cat", "finds", "food"],
        ["the", "dog", "finds", "water"],
        ["the", "bird", "finds", "bread"],
        ["the", "man", "takes", "the", "book"],
        ["the", "woman", "gives", "the", "apple"],
        ["the", "child", "loves", "the", "cat"],
        # Adjective patterns
        ["the", "big", "cat", "runs", "fast"],
        ["the", "small", "dog", "walks", "slow"],
        ["the", "happy", "child", "loves", "food"],
        # Connector patterns
        ["the", "cat", "runs", "and", "the", "dog", "walks"],
        ["he", "sees", "the", "cat", "and", "loves", "it"],
        ["she", "finds", "gold", "and", "gives", "it", "to", "him"],
        # Pronoun patterns
        ["I", "see", "the", "cat"],
        ["you", "love", "the", "dog"],
        ["he", "helps", "the", "child"],
        ["she", "takes", "the", "book"],
        ["they", "find", "the", "gold"],
        # Royal narrative
        ["the", "king", "gives", "gold", "to", "the", "queen"],
        ["the", "queen", "loves", "the", "king"],
        ["the", "king", "takes", "the", "crown"],
    ]
    
    print(f"\n[Training on {len(narratives)} narrative patterns...]")
    for narrative in narratives:
        for _ in range(5):
            predictor.train_on_sequence(narrative)
    
    generator = FieldGenerator(predictor)
    
    passed = 0
    failed = 0
    
    # ===== TEST 1: Extended Generation =====
    print("\n" + "-"*40)
    print("TEST 1: Extended Generation (10+ tokens)")
    print("-"*40)
    
    prompt = ["the", "king"]
    generated = generator.generate(prompt, max_tokens=12, temperature=0.4)
    
    print(f"Prompt: {' '.join(prompt)}")
    print(f"Generated: {' '.join(generated)}")
    print(f"Length: {len(generated)} tokens")
    
    test1_pass = len(generated) >= 10
    results['tests']['extended_generation'] = {
        'prompt': prompt,
        'generated': generated,
        'length': len(generated),
        'passed': test1_pass
    }
    
    if test1_pass:
        print(f"✓ PASSED: Generated {len(generated)} tokens")
        passed += 1
    else:
        print(f"✗ FAILED: Only {len(generated)} tokens")
        failed += 1
    
    # ===== TEST 2: No Degeneration =====
    print("\n" + "-"*40)
    print("TEST 2: No Repetitive Degeneration")
    print("-"*40)
    
    prompt = ["I", "see"]
    generated = generator.generate(prompt, max_tokens=10, temperature=0.4)
    
    print(f"Generated: {' '.join(generated)}")
    
    # Check for excessive repetition (no same word 3x in a row)
    has_degeneration = False
    for i in range(len(generated) - 2):
        if generated[i] == generated[i+1] == generated[i+2]:
            has_degeneration = True
            break
    
    test2_pass = not has_degeneration
    results['tests']['no_degeneration'] = {
        'generated': generated,
        'has_repetition': has_degeneration,
        'passed': test2_pass
    }
    
    if test2_pass:
        print(f"✓ PASSED: No repetitive degeneration")
        passed += 1
    else:
        print(f"✗ FAILED: Found repetitive degeneration")
        failed += 1
    
    # ===== TEST 3: Topical Consistency =====
    print("\n" + "-"*40)
    print("TEST 3: Topical Consistency")
    print("-"*40)
    
    # Generate from a topic-specific prompt
    prompt = ["the", "cat", "finds"]
    generated, scores = generator.generate_with_resonance(prompt, max_tokens=6)
    
    print(f"Prompt: {' '.join(prompt)}")
    print(f"Generated: {' '.join(generated)}")
    print(f"Resonance: {[f'{s:.3f}' for s in scores]}")
    
    # Check that resonance doesn't drop too quickly
    if len(scores) >= 3:
        avg_first_half = sum(scores[:len(scores)//2 + 1]) / (len(scores)//2 + 1)
        avg_second_half = sum(scores[len(scores)//2:]) / len(scores[len(scores)//2:])
        consistency = avg_second_half / avg_first_half if avg_first_half > 0 else 0
        print(f"  Resonance consistency: {consistency:.3f}")
    else:
        consistency = 1.0
    
    test3_pass = consistency > 0.5  # Second half at least 50% as strong
    results['tests']['topical_consistency'] = {
        'generated': generated,
        'scores': scores,
        'consistency': consistency,
        'passed': test3_pass
    }
    
    if test3_pass:
        print(f"✓ PASSED: Maintained topical consistency")
        passed += 1
    else:
        print(f"✗ FAILED: Consistency dropped below threshold")
        failed += 1
    
    # ===== TEST 4: Field Energy Conservation =====
    print("\n" + "-"*40)
    print("TEST 4: Field Energy Stability")
    print("-"*40)
    
    # Track field energy through generation
    prompt = ["the", "man"]
    context = prompt.copy()
    energies = []
    
    initial_field = predictor.combine_context(context)
    energies.append(compute_field_energy(initial_field))
    
    for _ in range(5):
        predictions = predictor.predict(context, top_k=3)
        if not predictions:
            break
        next_token = predictions[0][0]
        context.append(next_token)
        field = predictor.combine_context(context)
        energies.append(compute_field_energy(field))
    
    print(f"Generated: {' '.join(context)}")
    print(f"Field energies: {[f'{e:.2f}' for e in energies]}")
    
    # Check energy stability (shouldn't explode or collapse)
    if len(energies) >= 2:
        max_energy = max(energies)
        min_energy = min(energies)
        ratio = max_energy / min_energy if min_energy > 0 else float('inf')
        print(f"  Energy ratio (max/min): {ratio:.2f}")
    else:
        ratio = 1.0
    
    test4_pass = ratio < 5.0  # Energy shouldn't vary by more than 5x
    results['tests']['energy_stability'] = {
        'context': context,
        'energies': energies,
        'ratio': ratio,
        'passed': test4_pass
    }
    
    if test4_pass:
        print(f"✓ PASSED: Field energy stable")
        passed += 1
    else:
        print(f"✗ FAILED: Energy unstable (ratio={ratio:.2f})")
        failed += 1
    
    # ===== TEST 5: Diverse Prompts =====
    print("\n" + "-"*40)
    print("TEST 5: Diverse Prompt Handling")
    print("-"*40)
    
    test_prompts = [
        ["I"],
        ["the", "big"],
        ["she", "loves"],
        ["and", "then"],
    ]
    
    all_generated = []
    all_extended = True
    for prompt in test_prompts:
        gen = generator.generate(prompt, max_tokens=6, temperature=0.4)
        all_generated.append(gen)
        extended = len(gen) > len(prompt)
        if not extended:
            all_extended = False
        print(f"  {' '.join(prompt)} → {' '.join(gen)}")
    
    test5_pass = all_extended
    results['tests']['diverse_prompts'] = {
        'generations': [(p, g) for p, g in zip(test_prompts, all_generated)],
        'all_extended': all_extended,
        'passed': test5_pass
    }
    
    if test5_pass:
        print(f"✓ PASSED: All prompts extended successfully")
        passed += 1
    else:
        print(f"✗ FAILED: Some prompts not extended")
        failed += 1
    
    # ===== TEST 6: Connector Integration =====
    print("\n" + "-"*40)
    print("TEST 6: Connector Integration")
    print("-"*40)
    
    # Test if connectors like "and" create proper continuations
    prompt = ["the", "cat", "runs", "and"]
    predictions = predictor.predict(prompt, top_k=5)
    
    print(f"After '{' '.join(prompt)}':")
    print(f"  Predictions: {[(p[0], f'{p[1]:.3f}') for p in predictions[:5]]}")
    
    # After "and", should get subjects, articles, pronouns, or even verbs (parallel construction)
    expected_categories = set(words['articles'] + words['subjects'] + words['pronouns'] + words['verbs'])
    top_tokens = [p[0] for p in predictions[:5]]
    matches = sum(1 for t in top_tokens if t in expected_categories)
    
    test6_pass = matches >= 2  # At least 2 valid continuations in top 5
    results['tests']['connector_integration'] = {
        'prompt': prompt,
        'predictions': predictions[:5],
        'matches': matches,
        'passed': test6_pass
    }
    
    if test6_pass:
        print(f"✓ PASSED: Connector followed by appropriate continuation")
        passed += 1
    else:
        print(f"✗ FAILED: Connector pattern not learned")
        failed += 1
    
    # ===== SUMMARY =====
    print("\n" + "="*60)
    print(f"EXPERIMENT 04 RESULTS: {passed}/{passed+failed} passed")
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
    results_path = results_dir / f'exp_04_coherent_generation_{timestamp}.json'
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved: {results_path}")
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = run_tests()
    sys.exit(0 if failed == 0 else 1)
