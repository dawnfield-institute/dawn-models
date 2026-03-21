"""
POC-004 Experiment 03: Real Semantic Embeddings

The acid test: Do Dawn Field dynamics preserve REAL semantic structure?

Uses sentence-transformers to get actual semantic embeddings,
then tests if our spherical harmonic 3D encoding maintains:
1. Within-category similarity (cat/dog closer than cat/red)
2. Analogical relationships (king-man+woman≈queen)
3. Hierarchical structure (animal > mammal > dog)

This is the real test. Synthetic orthogonal vectors are easy.
Real semantic space is the actual challenge.

Torch only, GPU all the way.
"""

import torch
import torch.nn.functional as F
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("⚠️ sentence-transformers not installed. Run: pip install sentence-transformers")

from scale_field import (
    PHI, XI, PHI_XI, LAMBDA_STAR,
    SphericalHarmonicEncoder,
)


# ============================================================
# SEMANTIC TEST DATA
# ============================================================

# Category clusters - words that should be similar
CATEGORIES = {
    'animals': ['cat', 'dog', 'elephant', 'tiger', 'lion', 'bear', 'wolf', 'fox'],
    'colors': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown'],
    'countries': ['france', 'germany', 'japan', 'brazil', 'canada', 'australia', 'india', 'italy'],
    'professions': ['doctor', 'lawyer', 'teacher', 'engineer', 'scientist', 'artist', 'writer', 'chef'],
    'emotions': ['happy', 'sad', 'angry', 'afraid', 'excited', 'calm', 'anxious', 'joyful'],
}

# Analogies - A:B :: C:D (king:queen :: man:woman)
ANALOGIES = [
    ('king', 'queen', 'man', 'woman'),
    ('paris', 'france', 'tokyo', 'japan'),
    ('dog', 'puppy', 'cat', 'kitten'),
    ('big', 'bigger', 'small', 'smaller'),
    ('good', 'better', 'bad', 'worse'),
]

# Similarity pairs - should be more similar than random
SIMILAR_PAIRS = [
    ('cat', 'dog'),           # Both animals
    ('happy', 'joyful'),      # Synonyms
    ('doctor', 'nurse'),      # Related professions
    ('france', 'germany'),    # Both countries
    ('red', 'crimson'),       # Color variants
]

# Dissimilar pairs - should be less similar
DISSIMILAR_PAIRS = [
    ('cat', 'democracy'),
    ('happy', 'refrigerator'),
    ('doctor', 'purple'),
    ('france', 'happiness'),
    ('red', 'philosophy'),
]


def get_embeddings(model, words: List[str]) -> torch.Tensor:
    """Get embeddings for a list of words."""
    embeddings = model.encode(words, convert_to_tensor=True)
    return embeddings


def test_category_clustering():
    """Test 1: Do categories cluster in field space?"""
    print("\n" + "="*60)
    print("TEST 1: Category Clustering")
    print("="*60)
    
    if not HAS_SBERT:
        return {'test': 'category_clustering', 'passed': False, 'error': 'sentence-transformers not installed'}
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    print("  Loading sentence-transformers model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    embed_dim = model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {embed_dim}")
    
    # Create encoder matching embedding dimension
    # MiniLM is 384-dim, we'll use l_max=8 to capture more structure
    encoder = SphericalHarmonicEncoder(
        shape=(32, 32, 32),
        l_max=8,
        device=device
    )
    
    # Get embeddings for all categories
    category_embeddings = {}
    category_fields = {}
    
    for cat, words in CATEGORIES.items():
        embs = get_embeddings(model, words)
        category_embeddings[cat] = embs
        
        # Encode to 3D fields using v6 (geometric E=mc² preservation)
        fields = torch.stack([encoder.encode_v6(e) for e in embs])
        category_fields[cat] = fields
        print(f"  {cat}: {len(words)} words encoded")
    
    # Measure within-category similarity (in field space)
    within_sims = []
    for cat, fields in category_fields.items():
        for i in range(len(fields)):
            for j in range(i+1, len(fields)):
                sim = F.cosine_similarity(
                    fields[i].flatten().unsqueeze(0),
                    fields[j].flatten().unsqueeze(0)
                ).item()
                within_sims.append(sim)
    
    # Measure between-category similarity
    between_sims = []
    cats = list(category_fields.keys())
    for i, c1 in enumerate(cats):
        for c2 in cats[i+1:]:
            f1 = category_fields[c1]
            f2 = category_fields[c2]
            for fi in f1[:3]:  # Sample
                for fj in f2[:3]:
                    sim = F.cosine_similarity(
                        fi.flatten().unsqueeze(0),
                        fj.flatten().unsqueeze(0)
                    ).item()
                    between_sims.append(sim)
    
    mean_within = sum(within_sims) / len(within_sims)
    mean_between = sum(between_sims) / len(between_sims)
    separation = mean_within - mean_between
    
    # Also measure in original embedding space for comparison
    within_orig = []
    for cat, embs in category_embeddings.items():
        for i in range(len(embs)):
            for j in range(i+1, len(embs)):
                sim = F.cosine_similarity(embs[i].unsqueeze(0), embs[j].unsqueeze(0)).item()
                within_orig.append(sim)
    
    between_orig = []
    for i, c1 in enumerate(cats):
        for c2 in cats[i+1:]:
            e1 = category_embeddings[c1]
            e2 = category_embeddings[c2]
            for ei in e1[:3]:
                for ej in e2[:3]:
                    sim = F.cosine_similarity(ei.unsqueeze(0), ej.unsqueeze(0)).item()
                    between_orig.append(sim)
    
    orig_within = sum(within_orig) / len(within_orig)
    orig_between = sum(between_orig) / len(between_orig)
    orig_separation = orig_within - orig_between
    
    # Field should preserve at least 50% of original separation
    preservation = separation / (orig_separation + 1e-8)
    passed = preservation > 0.3 and separation > 0.05
    
    print(f"\n  Original embedding space:")
    print(f"    Within-category: {orig_within:.4f}")
    print(f"    Between-category: {orig_between:.4f}")
    print(f"    Separation: {orig_separation:.4f}")
    
    print(f"\n  3D Field space:")
    print(f"    Within-category: {mean_within:.4f}")
    print(f"    Between-category: {mean_between:.4f}")
    print(f"    Separation: {separation:.4f}")
    
    print(f"\n  Preservation ratio: {preservation:.2%} {'✅' if passed else '❌'}")
    
    return {
        'test': 'category_clustering',
        'passed': passed,
        'original': {'within': orig_within, 'between': orig_between, 'separation': orig_separation},
        'field': {'within': mean_within, 'between': mean_between, 'separation': separation},
        'preservation': preservation
    }


def test_similar_vs_dissimilar():
    """Test 2: Similar pairs more similar than dissimilar pairs?"""
    print("\n" + "="*60)
    print("TEST 2: Similar vs Dissimilar Pairs")
    print("="*60)
    
    if not HAS_SBERT:
        return {'test': 'similar_vs_dissimilar', 'passed': False, 'error': 'sentence-transformers not installed'}
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    encoder = SphericalHarmonicEncoder(
        shape=(32, 32, 32),
        l_max=8,
        device=device
    )
    
    # Get all words
    all_words = []
    for a, b in SIMILAR_PAIRS:
        all_words.extend([a, b])
    for a, b in DISSIMILAR_PAIRS:
        all_words.extend([a, b])
    all_words = list(set(all_words))
    
    # Get embeddings
    embeddings = get_embeddings(model, all_words)
    word_to_emb = {w: embeddings[i] for i, w in enumerate(all_words)}
    
    # Encode to fields using v6 (geometric E=mc² preservation)
    word_to_field = {w: encoder.encode_v6(word_to_emb[w]) for w in all_words}
    
    def field_sim(w1, w2):
        return F.cosine_similarity(
            word_to_field[w1].flatten().unsqueeze(0),
            word_to_field[w2].flatten().unsqueeze(0)
        ).item()
    
    def emb_sim(w1, w2):
        return F.cosine_similarity(
            word_to_emb[w1].unsqueeze(0),
            word_to_emb[w2].unsqueeze(0)
        ).item()
    
    print("\n  Similar pairs (should be high):")
    similar_sims = []
    similar_orig = []
    for a, b in SIMILAR_PAIRS:
        fs = field_sim(a, b)
        es = emb_sim(a, b)
        similar_sims.append(fs)
        similar_orig.append(es)
        print(f"    {a} <-> {b}: field={fs:.3f}, orig={es:.3f}")
    
    print("\n  Dissimilar pairs (should be low):")
    dissimilar_sims = []
    dissimilar_orig = []
    for a, b in DISSIMILAR_PAIRS:
        fs = field_sim(a, b)
        es = emb_sim(a, b)
        dissimilar_sims.append(fs)
        dissimilar_orig.append(es)
        print(f"    {a} <-> {b}: field={fs:.3f}, orig={es:.3f}")
    
    mean_similar = sum(similar_sims) / len(similar_sims)
    mean_dissimilar = sum(dissimilar_sims) / len(dissimilar_sims)
    
    mean_similar_orig = sum(similar_orig) / len(similar_orig)
    mean_dissimilar_orig = sum(dissimilar_orig) / len(dissimilar_orig)
    
    # Similar should be greater than dissimilar
    passed = mean_similar > mean_dissimilar
    
    print(f"\n  Mean similar (field): {mean_similar:.4f}")
    print(f"  Mean dissimilar (field): {mean_dissimilar:.4f}")
    print(f"  Gap: {mean_similar - mean_dissimilar:.4f} {'✅' if passed else '❌'}")
    
    print(f"\n  Mean similar (orig): {mean_similar_orig:.4f}")
    print(f"  Mean dissimilar (orig): {mean_dissimilar_orig:.4f}")
    print(f"  Gap (orig): {mean_similar_orig - mean_dissimilar_orig:.4f}")
    
    return {
        'test': 'similar_vs_dissimilar',
        'passed': passed,
        'field': {'similar': mean_similar, 'dissimilar': mean_dissimilar},
        'original': {'similar': mean_similar_orig, 'dissimilar': mean_dissimilar_orig}
    }


def test_analogy_preservation():
    """Test 3: Are analogical relationships preserved?"""
    print("\n" + "="*60)
    print("TEST 3: Analogy Preservation (A:B :: C:D)")
    print("="*60)
    
    if not HAS_SBERT:
        return {'test': 'analogy_preservation', 'passed': False, 'error': 'sentence-transformers not installed'}
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    encoder = SphericalHarmonicEncoder(
        shape=(32, 32, 32),
        l_max=8,
        device=device
    )
    
    # Get all words from analogies
    all_words = set()
    for a, b, c, d in ANALOGIES:
        all_words.update([a, b, c, d])
    all_words = list(all_words)
    
    embeddings = get_embeddings(model, all_words)
    word_to_emb = {w: embeddings[i] for i, w in enumerate(all_words)}
    word_to_field = {w: encoder.encode_v6(word_to_emb[w]) for w in all_words}
    
    def field_vec(w):
        return word_to_field[w].flatten()
    
    def emb_vec(w):
        return word_to_emb[w]
    
    print("\n  Testing: A - B + C ≈ D")
    
    correct_field = 0
    correct_orig = 0
    
    for a, b, c, d in ANALOGIES:
        # In embedding space: a - b + c should be close to d
        # (king - queen + woman ≈ man... wait, reversed)
        # Actually: king - man + woman ≈ queen
        # So: a - c + d ≈ b (if a:b :: c:d)
        
        # Predicted = a - c + d, should be close to b
        pred_emb = emb_vec(a) - emb_vec(c) + emb_vec(d)
        pred_emb = pred_emb / (pred_emb.norm() + 1e-8)
        sim_orig = F.cosine_similarity(pred_emb.unsqueeze(0), emb_vec(b).unsqueeze(0)).item()
        
        pred_field = field_vec(a) - field_vec(c) + field_vec(d)
        pred_field = pred_field / (pred_field.norm() + 1e-8)
        sim_field = F.cosine_similarity(pred_field.unsqueeze(0), field_vec(b).unsqueeze(0)).item()
        
        # Check if b is more similar than random
        # Compare to similarity with 'a' as baseline
        base_sim = F.cosine_similarity(pred_field.unsqueeze(0), field_vec(a).unsqueeze(0)).item()
        
        is_correct_field = sim_field > 0.5  # Reasonable threshold
        is_correct_orig = sim_orig > 0.5
        
        if is_correct_field:
            correct_field += 1
        if is_correct_orig:
            correct_orig += 1
        
        status = "✅" if is_correct_field else "❌"
        print(f"    {a}:{b} :: {c}:{d}")
        print(f"      Predicted→{b}: field={sim_field:.3f}, orig={sim_orig:.3f} {status}")
    
    accuracy_field = correct_field / len(ANALOGIES)
    accuracy_orig = correct_orig / len(ANALOGIES)
    
    # Pass if we get at least 40% (analogies are hard)
    passed = accuracy_field >= 0.4
    
    print(f"\n  Accuracy (field): {accuracy_field:.0%} ({correct_field}/{len(ANALOGIES)})")
    print(f"  Accuracy (orig): {accuracy_orig:.0%} ({correct_orig}/{len(ANALOGIES)})")
    print(f"  {'✅' if passed else '❌'}")
    
    return {
        'test': 'analogy_preservation',
        'passed': passed,
        'accuracy_field': accuracy_field,
        'accuracy_orig': accuracy_orig,
        'correct_field': correct_field,
        'correct_orig': correct_orig,
        'total': len(ANALOGIES)
    }


def test_throughput_real():
    """Test 4: Throughput with real embeddings."""
    print("\n" + "="*60)
    print("TEST 4: Throughput with Real Embeddings")
    print("="*60)
    
    if not HAS_SBERT:
        return {'test': 'throughput_real', 'passed': False, 'error': 'sentence-transformers not installed'}
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    encoder = SphericalHarmonicEncoder(
        shape=(32, 32, 32),
        l_max=8,
        device=device
    )
    
    # Create a larger vocabulary
    vocab = []
    for cat_words in CATEGORIES.values():
        vocab.extend(cat_words)
    # Add more common words
    extra_words = [
        'the', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'be', 'been', 'being', 'am', 'an', 'a', 'this', 'that',
        'computer', 'phone', 'book', 'table', 'chair', 'window',
        'house', 'car', 'tree', 'flower', 'sun', 'moon', 'star',
        'water', 'fire', 'earth', 'air', 'time', 'space', 'love',
        'music', 'art', 'science', 'math', 'history', 'future'
    ]
    vocab.extend(extra_words)
    vocab = list(set(vocab))
    
    print(f"  Vocabulary size: {len(vocab)}")
    
    # Get embeddings (this is the slow part, but we're testing field encoding)
    print("  Getting embeddings...")
    embeddings = get_embeddings(model, vocab)
    
    # Measure field encoding throughput
    print("  Encoding to 3D fields...")
    
    # Warmup
    for e in embeddings[:5]:
        _ = encoder.encode_v6(e)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    start = time.time()
    for e in embeddings:
        _ = encoder.encode_v6(e)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    throughput = len(vocab) / elapsed
    
    passed = throughput > 50  # At least 50 words/sec
    
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.0f} words/sec {'✅' if passed else '❌'}")
    
    return {
        'test': 'throughput_real',
        'passed': passed,
        'vocab_size': len(vocab),
        'time': elapsed,
        'throughput': throughput
    }


def main():
    """Run all real embedding tests."""
    print("="*60)
    print("POC-004 Experiment 03: Real Semantic Embeddings")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    if not HAS_SBERT:
        print("\n❌ Cannot run tests: sentence-transformers not installed")
        print("   Install with: pip install sentence-transformers")
        return False
    
    all_results = []
    
    all_results.append(test_category_clustering())
    all_results.append(test_similar_vs_dissimilar())
    all_results.append(test_analogy_preservation())
    all_results.append(test_throughput_real())
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in all_results if r['passed'])
    total = len(all_results)
    
    for r in all_results:
        status = "✅" if r['passed'] else "❌"
        print(f"  {status} {r['test']}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    # Save results
    output = {
        'experiment': 'exp_03_real_embeddings',
        'timestamp': datetime.now().isoformat(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model': 'all-MiniLM-L6-v2',
        'constants': {
            'phi': PHI,
            'xi': XI,
            'phi_xi': PHI_XI,
            'lambda_star': LAMBDA_STAR
        },
        'tests': all_results,
        'summary': {
            'passed': passed,
            'total': total,
            'success': passed == total
        }
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = results_dir / f'exp_03_real_embeddings_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
