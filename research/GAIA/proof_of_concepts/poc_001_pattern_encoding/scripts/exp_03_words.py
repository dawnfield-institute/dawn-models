"""
Experiment 03: Word Pattern Encoding
=====================================

POC-001: Test word-level encoding with semantic relationships.

Hypothesis:
- Words encode into distinct field patterns
- Semantically similar words produce similar patterns
- Word patterns survive field evolution

This is the critical test for field-native language learning.

Technical Requirements:
- PyTorch only (no numpy)
- GPU acceleration
"""

import torch
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    FieldEncoder, FieldEvolver, ExperimentResult,
    measure_pattern_distance, measure_pattern_similarity,
    get_gpu_info, get_results_dir, generate_experiment_id, DEVICE
)


# Semantic groups for testing
SEMANTIC_GROUPS = {
    'animals': ['cat', 'dog', 'bird', 'fish', 'horse'],
    'colors': ['red', 'blue', 'green', 'yellow', 'orange'],
    'actions': ['run', 'walk', 'jump', 'swim', 'fly'],
    'objects': ['car', 'house', 'tree', 'book', 'phone'],
}

# Similar word pairs (semantic similarity)
SIMILAR_PAIRS = [
    ('cat', 'dog'),      # Both animals
    ('run', 'walk'),     # Both movement
    ('red', 'blue'),     # Both colors
    ('car', 'truck'),    # Both vehicles
    ('big', 'large'),    # Synonyms
]

# Dissimilar word pairs
DISSIMILAR_PAIRS = [
    ('cat', 'red'),      # Animal vs color
    ('run', 'house'),    # Action vs object
    ('blue', 'jump'),    # Color vs action
    ('fish', 'book'),    # Animal vs object
    ('green', 'horse'),  # Color vs animal
]


def run_word_distinctiveness_test(encoder: FieldEncoder) -> Dict[str, Any]:
    """
    Test 1: Different words encode distinctly.
    """
    print("\n=== Test 1: Word Distinctiveness ===")
    
    words = ['cat', 'dog', 'run', 'big', 'red', 'car', 'sky', 'day']
    encodings = {}
    
    for word in words:
        result = encoder.encode_word(word)
        encodings[word] = result
        print(f"  '{word}': energy={result.field_energy:.4f}, time={result.encoding_time_ms:.2f}ms")
    
    distances = []
    min_dist = float('inf')
    min_pair = ('', '')
    
    for w1, w2 in combinations(words, 2):
        dist = measure_pattern_distance(
            encodings[w1].field_state,
            encodings[w2].field_state
        )
        distances.append(dist)
        if dist < min_dist:
            min_dist = dist
            min_pair = (w1, w2)
    
    avg_dist = sum(distances) / len(distances)
    
    print(f"\n  Average distance: {avg_dist:.4f}")
    print(f"  Minimum distance: {min_dist:.4f} ('{min_pair[0]}' vs '{min_pair[1]}')")
    
    return {
        'test': 'word_distinctiveness',
        'words_tested': len(words),
        'avg_distance': avg_dist,
        'min_distance': min_dist,
        'min_pair': min_pair,
        'success': min_dist > 0.01
    }


def run_semantic_clustering_test(encoder: FieldEncoder) -> Dict[str, Any]:
    """
    Test 2: Words in same semantic category cluster together.
    
    Within-group similarity should exceed between-group similarity.
    """
    print("\n=== Test 2: Semantic Clustering ===")
    
    # Encode all words in groups
    group_encodings = {}
    for group, words in SEMANTIC_GROUPS.items():
        group_encodings[group] = {
            w: encoder.encode_word(w) for w in words
        }
    
    # Calculate within-group similarities
    within_sims = []
    for group, encodings in group_encodings.items():
        words = list(encodings.keys())
        for w1, w2 in combinations(words, 2):
            sim = measure_pattern_similarity(
                encodings[w1].field_state,
                encodings[w2].field_state
            )
            within_sims.append(sim)
    
    # Calculate between-group similarities
    between_sims = []
    groups = list(SEMANTIC_GROUPS.keys())
    for g1, g2 in combinations(groups, 2):
        for w1 in list(group_encodings[g1].keys())[:2]:  # Sample
            for w2 in list(group_encodings[g2].keys())[:2]:
                sim = measure_pattern_similarity(
                    group_encodings[g1][w1].field_state,
                    group_encodings[g2][w2].field_state
                )
                between_sims.append(sim)
    
    avg_within = sum(within_sims) / len(within_sims)
    avg_between = sum(between_sims) / len(between_sims)
    
    print(f"  Avg within-group similarity:  {avg_within:.4f}")
    print(f"  Avg between-group similarity: {avg_between:.4f}")
    print(f"  Separation: {avg_within - avg_between:.4f}")
    
    # Note: This might not pass yet - semantic similarity needs more work
    return {
        'test': 'semantic_clustering',
        'groups': len(SEMANTIC_GROUPS),
        'avg_within': avg_within,
        'avg_between': avg_between,
        'separation': avg_within - avg_between,
        'success': avg_within > avg_between  # May need refinement
    }


def run_similar_vs_dissimilar_test(encoder: FieldEncoder) -> Dict[str, Any]:
    """
    Test 3: Similar word pairs have higher similarity than dissimilar pairs.
    """
    print("\n=== Test 3: Similar vs Dissimilar Pairs ===")
    
    similar_sims = []
    for w1, w2 in SIMILAR_PAIRS:
        e1 = encoder.encode_word(w1)
        e2 = encoder.encode_word(w2)
        sim = measure_pattern_similarity(e1.field_state, e2.field_state)
        similar_sims.append(sim)
        print(f"  Similar '{w1}'-'{w2}': {sim:.4f}")
    
    dissimilar_sims = []
    for w1, w2 in DISSIMILAR_PAIRS:
        e1 = encoder.encode_word(w1)
        e2 = encoder.encode_word(w2)
        sim = measure_pattern_similarity(e1.field_state, e2.field_state)
        dissimilar_sims.append(sim)
        print(f"  Dissimilar '{w1}'-'{w2}': {sim:.4f}")
    
    avg_similar = sum(similar_sims) / len(similar_sims)
    avg_dissimilar = sum(dissimilar_sims) / len(dissimilar_sims)
    
    print(f"\n  Avg similar pair:    {avg_similar:.4f}")
    print(f"  Avg dissimilar pair: {avg_dissimilar:.4f}")
    
    return {
        'test': 'similar_vs_dissimilar',
        'avg_similar': avg_similar,
        'avg_dissimilar': avg_dissimilar,
        'separation': avg_similar - avg_dissimilar,
        'success': avg_similar > avg_dissimilar
    }


def run_word_length_invariance_test(encoder: FieldEncoder) -> Dict[str, Any]:
    """
    Test 4: Words of different lengths encode comparably.
    
    Short and long words should both produce valid patterns.
    """
    print("\n=== Test 4: Word Length Invariance ===")
    
    short_words = ['a', 'an', 'the', 'be', 'to']
    medium_words = ['hello', 'world', 'python', 'field']
    long_words = ['programming', 'intelligence', 'consciousness', 'transformer']
    
    def analyze_group(words, label):
        encodings = [encoder.encode_word(w) for w in words]
        energies = [e.field_energy for e in encodings]
        times = [e.encoding_time_ms for e in encodings]
        avg_energy = sum(energies) / len(energies)
        avg_time = sum(times) / len(times)
        print(f"  {label}: avg_energy={avg_energy:.4f}, avg_time={avg_time:.2f}ms")
        return avg_energy, avg_time
    
    short_e, short_t = analyze_group(short_words, "Short (1-3)")
    med_e, med_t = analyze_group(medium_words, "Medium (5-6)")
    long_e, long_t = analyze_group(long_words, "Long (10+)")
    
    # Energy should be similar (normalized)
    energy_variance = max(short_e, med_e, long_e) - min(short_e, med_e, long_e)
    
    print(f"\n  Energy variance: {energy_variance:.4f}")
    
    return {
        'test': 'length_invariance',
        'short_energy': short_e,
        'medium_energy': med_e,
        'long_energy': long_e,
        'energy_variance': energy_variance,
        'success': energy_variance < 0.1  # Energies within 10%
    }


def run_word_evolution_stability_test(
    encoder: FieldEncoder,
    evolver: FieldEvolver
) -> Dict[str, Any]:
    """
    Test 5: Word patterns survive evolution.
    """
    print("\n=== Test 5: Word Evolution Stability ===")
    
    test_words = ['cat', 'hello', 'transformer', 'run']
    steps = 100
    
    results = []
    for word in test_words:
        encoding = encoder.encode_word(word)
        evolution = evolver.evolve(encoding.field_state, steps)
        
        print(f"  '{word}' after {steps} steps: correlation={evolution.correlation_with_initial:.4f}")
        results.append({
            'word': word,
            'correlation': evolution.correlation_with_initial,
            'survived': evolution.pattern_survived
        })
    
    survival_rate = sum(1 for r in results if r['survived']) / len(results)
    avg_corr = sum(r['correlation'] for r in results) / len(results)
    
    print(f"\n  Survival rate: {survival_rate*100:.1f}%")
    print(f"  Avg correlation: {avg_corr:.4f}")
    
    return {
        'test': 'word_evolution',
        'steps': steps,
        'survival_rate': survival_rate,
        'avg_correlation': avg_corr,
        'success': survival_rate >= 0.8
    }


def run_prefix_similarity_test(encoder: FieldEncoder) -> Dict[str, Any]:
    """
    Test 6: Words with common prefixes have higher similarity.
    
    e.g., "running" ~ "runner" > "running" ~ "jumping"
    """
    print("\n=== Test 6: Prefix Similarity ===")
    
    # Words with common prefixes
    prefix_pairs = [
        ('run', 'running'),
        ('walk', 'walking'),
        ('play', 'player'),
        ('think', 'thinking'),
    ]
    
    # Words without common prefixes
    no_prefix_pairs = [
        ('run', 'walk'),
        ('play', 'think'),
        ('jump', 'sleep'),
        ('read', 'write'),
    ]
    
    prefix_sims = []
    for w1, w2 in prefix_pairs:
        e1 = encoder.encode_word(w1)
        e2 = encoder.encode_word(w2)
        sim = measure_pattern_similarity(e1.field_state, e2.field_state)
        prefix_sims.append(sim)
        print(f"  Prefix '{w1}'-'{w2}': {sim:.4f}")
    
    no_prefix_sims = []
    for w1, w2 in no_prefix_pairs:
        e1 = encoder.encode_word(w1)
        e2 = encoder.encode_word(w2)
        sim = measure_pattern_similarity(e1.field_state, e2.field_state)
        no_prefix_sims.append(sim)
        print(f"  No prefix '{w1}'-'{w2}': {sim:.4f}")
    
    avg_prefix = sum(prefix_sims) / len(prefix_sims)
    avg_no_prefix = sum(no_prefix_sims) / len(no_prefix_sims)
    
    print(f"\n  Avg with-prefix similarity: {avg_prefix:.4f}")
    print(f"  Avg no-prefix similarity:   {avg_no_prefix:.4f}")
    
    return {
        'test': 'prefix_similarity',
        'avg_with_prefix': avg_prefix,
        'avg_no_prefix': avg_no_prefix,
        'separation': avg_prefix - avg_no_prefix,
        'success': avg_prefix > avg_no_prefix
    }


def main():
    """Run all word encoding experiments."""
    print("=" * 60)
    print("POC-001 Experiment 03: Word Pattern Encoding")
    print("=" * 60)
    
    gpu_info = get_gpu_info()
    print(f"\nDevice: {DEVICE}")
    if gpu_info['available']:
        print(f"GPU: {gpu_info['device_name']}")
    
    encoder = FieldEncoder(field_dims=(64, 64), device=DEVICE)
    evolver = FieldEvolver(field_dims=(64, 64), device=DEVICE)
    
    all_results = []
    
    all_results.append(run_word_distinctiveness_test(encoder))
    all_results.append(run_semantic_clustering_test(encoder))
    all_results.append(run_similar_vs_dissimilar_test(encoder))
    all_results.append(run_word_length_invariance_test(encoder))
    all_results.append(run_word_evolution_stability_test(encoder, evolver))
    all_results.append(run_prefix_similarity_test(encoder))
    
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
    
    # Analysis notes
    if passed < total:
        print("\n⚠️  Some tests may fail because pure character-based encoding")
        print("   doesn't capture semantic meaning. This is expected and")
        print("   highlights why we need training through resonance!")
    
    # Save results
    experiment = ExperimentResult(
        experiment_id=generate_experiment_id("exp03_words"),
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
        success=(passed >= 4),  # 4/6 is acceptable for baseline
        notes="Word-level encoding experiment - baseline without training"
    )
    
    results_path = experiment.save(get_results_dir())
    print(f"\nResults saved to: {results_path}")
    
    return passed >= 4


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
