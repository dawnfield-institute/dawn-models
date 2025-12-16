"""
Experiment 02: Character Pattern Encoding
==========================================

POC-001: Test character-level encoding (A-Z, 0-9, punctuation).

Hypothesis:
- Single characters encode into distinct field patterns
- Similar characters (e.g., 'a' vs 'b') have higher similarity than distant ones
- Character patterns survive field evolution

Technical Requirements:
- PyTorch only (no numpy)
- GPU acceleration
"""

import torch
import sys
import string
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    FieldEncoder, FieldEvolver, ExperimentResult,
    measure_pattern_distance, measure_pattern_similarity,
    get_gpu_info, get_results_dir, generate_experiment_id, DEVICE
)


def run_alphabet_distinctiveness_test(encoder: FieldEncoder) -> Dict[str, Any]:
    """
    Test 1: All 26 letters encode distinctly.
    
    Success criteria: Min pairwise distance > 0.01
    """
    print("\n=== Test 1: Alphabet Distinctiveness ===")
    
    letters = list(string.ascii_lowercase)
    encodings = {}
    
    for char in letters:
        result = encoder.encode_character(char)
        encodings[char] = result
    
    print(f"  Encoded {len(letters)} characters")
    
    # Calculate pairwise distances
    distances = []
    min_dist = float('inf')
    min_pair = ('', '')
    
    for c1, c2 in combinations(letters, 2):
        dist = measure_pattern_distance(
            encodings[c1].field_state,
            encodings[c2].field_state
        )
        distances.append(dist)
        if dist < min_dist:
            min_dist = dist
            min_pair = (c1, c2)
    
    avg_dist = sum(distances) / len(distances)
    
    print(f"  Average distance: {avg_dist:.4f}")
    print(f"  Minimum distance: {min_dist:.4f} ('{min_pair[0]}' vs '{min_pair[1]}')")
    
    return {
        'test': 'alphabet_distinctiveness',
        'chars_tested': len(letters),
        'avg_distance': avg_dist,
        'min_distance': min_dist,
        'min_pair': min_pair,
        'success': min_dist > 0.01
    }


def run_adjacent_similarity_test(encoder: FieldEncoder) -> Dict[str, Any]:
    """
    Test 2: Adjacent letters have higher similarity than distant letters.
    
    We expect: sim(a,b) > sim(a,z), sim(m,n) > sim(m,a)
    """
    print("\n=== Test 2: Adjacent Letter Similarity ===")
    
    # Adjacent pairs
    adjacent = [('a', 'b'), ('m', 'n'), ('x', 'y'), ('d', 'e'), ('p', 'q')]
    
    # Distant pairs  
    distant = [('a', 'z'), ('m', 'a'), ('b', 'x'), ('c', 'w'), ('d', 'r')]
    
    adj_sims = []
    for c1, c2 in adjacent:
        e1 = encoder.encode_character(c1)
        e2 = encoder.encode_character(c2)
        sim = measure_pattern_similarity(e1.field_state, e2.field_state)
        adj_sims.append(sim)
        print(f"  Adjacent '{c1}'-'{c2}': {sim:.4f}")
    
    dist_sims = []
    for c1, c2 in distant:
        e1 = encoder.encode_character(c1)
        e2 = encoder.encode_character(c2)
        sim = measure_pattern_similarity(e1.field_state, e2.field_state)
        dist_sims.append(sim)
        print(f"  Distant  '{c1}'-'{c2}': {sim:.4f}")
    
    avg_adj = sum(adj_sims) / len(adj_sims)
    avg_dist = sum(dist_sims) / len(dist_sims)
    
    print(f"\n  Avg adjacent similarity: {avg_adj:.4f}")
    print(f"  Avg distant similarity:  {avg_dist:.4f}")
    print(f"  Adjacent > Distant: {avg_adj > avg_dist}")
    
    return {
        'test': 'adjacent_similarity',
        'avg_adjacent_sim': avg_adj,
        'avg_distant_sim': avg_dist,
        'separation': avg_adj - avg_dist,
        'success': avg_adj > avg_dist
    }


def run_digit_encoding_test(encoder: FieldEncoder) -> Dict[str, Any]:
    """
    Test 3: Digits 0-9 encode distinctly.
    """
    print("\n=== Test 3: Digit Encoding ===")
    
    digits = list(string.digits)
    encodings = {}
    
    for d in digits:
        result = encoder.encode_character(d)
        encodings[d] = result
        print(f"  Digit '{d}': energy={result.field_energy:.4f}")
    
    # Pairwise distances
    distances = []
    for d1, d2 in combinations(digits, 2):
        dist = measure_pattern_distance(
            encodings[d1].field_state,
            encodings[d2].field_state
        )
        distances.append(dist)
    
    avg_dist = sum(distances) / len(distances)
    min_dist = min(distances)
    
    print(f"\n  Avg distance: {avg_dist:.4f}")
    print(f"  Min distance: {min_dist:.4f}")
    
    return {
        'test': 'digit_encoding',
        'digits_tested': len(digits),
        'avg_distance': avg_dist,
        'min_distance': min_dist,
        'success': min_dist > 0.01
    }


def run_letter_vs_digit_separation_test(encoder: FieldEncoder) -> Dict[str, Any]:
    """
    Test 4: Letters and digits form separate clusters.
    
    Within-class similarity should be higher than between-class.
    """
    print("\n=== Test 4: Letter vs Digit Separation ===")
    
    letters = ['a', 'e', 'i', 'o', 'u']  # Sample
    digits = ['0', '1', '2', '3', '4']
    
    # Encode all
    letter_encodings = {c: encoder.encode_character(c) for c in letters}
    digit_encodings = {c: encoder.encode_character(c) for c in digits}
    
    # Within-letter similarities
    letter_sims = []
    for c1, c2 in combinations(letters, 2):
        sim = measure_pattern_similarity(
            letter_encodings[c1].field_state,
            letter_encodings[c2].field_state
        )
        letter_sims.append(sim)
    
    # Within-digit similarities
    digit_sims = []
    for d1, d2 in combinations(digits, 2):
        sim = measure_pattern_similarity(
            digit_encodings[d1].field_state,
            digit_encodings[d2].field_state
        )
        digit_sims.append(sim)
    
    # Cross-class similarities
    cross_sims = []
    for c in letters:
        for d in digits:
            sim = measure_pattern_similarity(
                letter_encodings[c].field_state,
                digit_encodings[d].field_state
            )
            cross_sims.append(sim)
    
    avg_letter = sum(letter_sims) / len(letter_sims)
    avg_digit = sum(digit_sims) / len(digit_sims)
    avg_cross = sum(cross_sims) / len(cross_sims)
    avg_within = (avg_letter + avg_digit) / 2
    
    print(f"  Avg within-letter similarity: {avg_letter:.4f}")
    print(f"  Avg within-digit similarity:  {avg_digit:.4f}")
    print(f"  Avg cross-class similarity:   {avg_cross:.4f}")
    print(f"  Separation (within - cross):  {avg_within - avg_cross:.4f}")
    
    return {
        'test': 'letter_digit_separation',
        'avg_within_letter': avg_letter,
        'avg_within_digit': avg_digit,
        'avg_cross_class': avg_cross,
        'separation': avg_within - avg_cross,
        'success': avg_within > avg_cross
    }


def run_evolution_stability_test(
    encoder: FieldEncoder,
    evolver: FieldEvolver
) -> Dict[str, Any]:
    """
    Test 5: Character patterns survive evolution.
    """
    print("\n=== Test 5: Character Evolution Stability ===")
    
    test_chars = ['a', 'm', 'z', '0', '5']
    steps = 100
    
    results = []
    for char in test_chars:
        encoding = encoder.encode_character(char)
        evolution = evolver.evolve(encoding.field_state, steps)
        
        print(f"  '{char}' after {steps} steps: correlation={evolution.correlation_with_initial:.4f}")
        results.append({
            'char': char,
            'correlation': evolution.correlation_with_initial,
            'survived': evolution.pattern_survived
        })
    
    survival_rate = sum(1 for r in results if r['survived']) / len(results)
    avg_correlation = sum(r['correlation'] for r in results) / len(results)
    
    print(f"\n  Survival rate: {survival_rate*100:.1f}%")
    print(f"  Avg correlation: {avg_correlation:.4f}")
    
    return {
        'test': 'character_evolution',
        'steps': steps,
        'chars_tested': len(test_chars),
        'survival_rate': survival_rate,
        'avg_correlation': avg_correlation,
        'success': survival_rate >= 0.8
    }


def run_encoding_speed_test(encoder: FieldEncoder) -> Dict[str, Any]:
    """
    Test 6: Encoding speed benchmark.
    """
    print("\n=== Test 6: Encoding Speed ===")
    
    # Full character set
    all_chars = string.ascii_lowercase + string.ascii_uppercase + string.digits
    
    import time
    start = time.perf_counter()
    
    for char in all_chars:
        encoder.encode_character(char)
    
    total_time = (time.perf_counter() - start) * 1000
    per_char = total_time / len(all_chars)
    
    print(f"  Encoded {len(all_chars)} characters in {total_time:.2f}ms")
    print(f"  Per character: {per_char:.3f}ms")
    
    return {
        'test': 'encoding_speed',
        'total_chars': len(all_chars),
        'total_time_ms': total_time,
        'per_char_ms': per_char,
        'success': per_char < 10  # <10ms per char is acceptable
    }


def main():
    """Run all character encoding experiments."""
    print("=" * 60)
    print("POC-001 Experiment 02: Character Pattern Encoding")
    print("=" * 60)
    
    gpu_info = get_gpu_info()
    print(f"\nDevice: {DEVICE}")
    if gpu_info['available']:
        print(f"GPU: {gpu_info['device_name']}")
    
    encoder = FieldEncoder(field_dims=(64, 64), device=DEVICE)
    evolver = FieldEvolver(field_dims=(64, 64), device=DEVICE)
    
    all_results = []
    
    all_results.append(run_alphabet_distinctiveness_test(encoder))
    all_results.append(run_adjacent_similarity_test(encoder))
    all_results.append(run_digit_encoding_test(encoder))
    all_results.append(run_letter_vs_digit_separation_test(encoder))
    all_results.append(run_evolution_stability_test(encoder, evolver))
    all_results.append(run_encoding_speed_test(encoder))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in all_results if r['success'])
    total = len(all_results)
    
    for result in all_results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"  {result['test']}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    # Save results
    experiment = ExperimentResult(
        experiment_id=generate_experiment_id("exp02_characters"),
        timestamp=datetime.now().isoformat(),
        device=str(DEVICE),
        parameters={
            'field_dims': (64, 64),
            'xi_target': 1.0571
        },
        encodings=[],
        metrics={
            'tests_passed': passed,
            'tests_total': total,
            'test_results': all_results
        },
        success=(passed == total),
        notes="Character-level encoding experiment"
    )
    
    results_path = experiment.save(get_results_dir())
    print(f"\nResults saved to: {results_path}")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
