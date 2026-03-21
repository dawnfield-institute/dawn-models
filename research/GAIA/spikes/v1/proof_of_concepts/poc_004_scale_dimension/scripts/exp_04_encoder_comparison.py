"""
POC-004 Experiment 04: Encoder Version Comparison
Compare v1 (multiplicative angular) vs refined (additive angular)

Key question: Can we get the best of both?
- v1: Good category separation (103%), bad pairwise (inversions)
- Refined: Good pairwise (0.97 for cat/dog), weaker category (35%)
"""

import torch
import json
import sys
from datetime import datetime
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from scale_field import SphericalHarmonicEncoder

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("WARNING: sentence-transformers not available")

def compute_similarity(f1, f2):
    """Field cosine similarity."""
    f1_flat = f1.flatten()
    f2_flat = f2.flatten()
    return torch.dot(f1_flat, f2_flat) / (f1_flat.norm() * f2_flat.norm() + 1e-8)

def main():
    print("=" * 60)
    print("POC-004 Experiment 04: Encoder Comparison")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type.upper()}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "device": device.type,
        "tests": {}
    }
    
    if not HAS_ST:
        print("\nSkipping: sentence-transformers not available")
        return
    
    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create encoder
    encoder = SphericalHarmonicEncoder(
        shape=(32, 32, 32),
        l_max=8,
        device=device
    )
    
    # Test pairs
    similar_pairs = [
        ("cat", "dog"),
        ("happy", "joyful"),
        ("doctor", "nurse"),
        ("france", "germany"),
        ("red", "crimson"),
        ("king", "queen"),
        ("run", "sprint"),
        ("car", "automobile"),
    ]
    
    dissimilar_pairs = [
        ("cat", "democracy"),
        ("happy", "refrigerator"),
        ("doctor", "purple"),
        ("france", "happiness"),
        ("red", "philosophy"),
        ("king", "banana"),
        ("run", "purple"),
        ("car", "sadness"),
    ]
    
    # Collect all words
    all_words = set()
    for a, b in similar_pairs + dissimilar_pairs:
        all_words.add(a)
        all_words.add(b)
    all_words = sorted(all_words)
    
    # Get embeddings
    print(f"\nEmbedding {len(all_words)} words...")
    embeddings = model.encode(all_words, convert_to_tensor=True)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Define correlation function
    def correlation(x, y):
        x = torch.tensor(x)
        y = torch.tensor(y)
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        return (torch.dot(x_centered, y_centered) / 
                (x_centered.norm() * y_centered.norm() + 1e-8)).item()
    
    # Encode with all versions
    versions = ['v1', 'refined', 'v2', 'v5', 'v6']
    encode_methods = {
        'v1': encoder.encode_v1,
        'refined': encoder.encode,
        'v2': encoder.encode_v2,
        'v5': encoder.encode_v5,
        'v6': encoder.encode_v6,
    }
    
    fields = {v: {} for v in versions}
    
    for ver in versions:
        print(f"Encoding with {ver}...")
        method = encode_methods[ver]
        for word in all_words:
            emb = embeddings[word_to_idx[word]]
            fields[ver][word] = method(emb)
    
    print("\n" + "=" * 100)
    print("SIMILAR PAIRS COMPARISON")
    print("=" * 100)
    header = f"{'Pair':<22} {'Orig':>7}"
    for v in versions:
        header += f" {v:>7}"
    print(header)
    print("-" * 100)
    
    similar_results = {v: [] for v in versions}
    similar_orig = []
    
    for a, b in similar_pairs:
        e1 = embeddings[word_to_idx[a]]
        e2 = embeddings[word_to_idx[b]]
        orig_sim = torch.dot(e1, e2) / (e1.norm() * e2.norm() + 1e-8)
        similar_orig.append(orig_sim.item())
        
        row = f"{a}<->{b:<13} {orig_sim.item():>7.3f}"
        for ver in versions:
            sim = compute_similarity(fields[ver][a], fields[ver][b])
            similar_results[ver].append(sim.item())
            flag = "❌" if sim < 0 else ""
            row += f" {sim.item():>7.3f}{flag}"
        print(row)
    
    print("\n" + "=" * 100)
    print("DISSIMILAR PAIRS COMPARISON")
    print("=" * 100)
    header = f"{'Pair':<22} {'Orig':>7}"
    for v in versions:
        header += f" {v:>7}"
    print(header)
    print("-" * 100)
    
    dissimilar_results = {v: [] for v in versions}
    dissimilar_orig = []
    
    for a, b in dissimilar_pairs:
        e1 = embeddings[word_to_idx[a]]
        e2 = embeddings[word_to_idx[b]]
        orig_sim = torch.dot(e1, e2) / (e1.norm() * e2.norm() + 1e-8)
        dissimilar_orig.append(orig_sim.item())
        
        row = f"{a}<->{b:<13} {orig_sim.item():>7.3f}"
        for ver in versions:
            sim = compute_similarity(fields[ver][a], fields[ver][b])
            dissimilar_results[ver].append(sim.item())
            flag = "⚠️" if sim > 0.7 else ""
            row += f" {sim.item():>7.3f}{flag}"
        print(row)
    
    # Summary stats
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    
    metrics = {}
    for ver in versions:
        mean_sim = sum(similar_results[ver]) / len(similar_results[ver])
        mean_dis = sum(dissimilar_results[ver]) / len(dissimilar_results[ver])
        gap = mean_sim - mean_dis
        inversions = sum(1 for s in similar_results[ver] if s < 0)
        
        all_orig = similar_orig + dissimilar_orig
        all_ver = similar_results[ver] + dissimilar_results[ver]
        corr = correlation(all_orig, all_ver)
        
        metrics[ver] = {
            'mean_sim': mean_sim,
            'mean_dis': mean_dis,
            'gap': gap,
            'inversions': inversions,
            'correlation': corr
        }
    
    mean_sim_orig = sum(similar_orig) / len(similar_orig)
    mean_dis_orig = sum(dissimilar_orig) / len(dissimilar_orig)
    gap_orig = mean_sim_orig - mean_dis_orig
    
    header = f"{'Metric':<22} {'Orig':>10}"
    for v in versions:
        header += f" {v:>10}"
    print(f"\n{header}")
    print("-" * 90)
    
    print(f"{'Mean similar':<22} {mean_sim_orig:>10.3f}", end="")
    for v in versions:
        print(f" {metrics[v]['mean_sim']:>10.3f}", end="")
    print()
    
    print(f"{'Mean dissimilar':<22} {mean_dis_orig:>10.3f}", end="")
    for v in versions:
        print(f" {metrics[v]['mean_dis']:>10.3f}", end="")
    print()
    
    print(f"{'Gap (sim-dissim)':<22} {gap_orig:>10.3f}", end="")
    for v in versions:
        print(f" {metrics[v]['gap']:>10.3f}", end="")
    print()
    
    print(f"\n{'Sign inversions':<22} {'N/A':>10}", end="")
    for v in versions:
        print(f" {metrics[v]['inversions']:>10}", end="")
    print()
    
    print(f"{'Correlation w/ orig':<22} {'1.000':>10}", end="")
    for v in versions:
        print(f" {metrics[v]['correlation']:>10.3f}", end="")
    print()
    
    # Winner analysis
    print("\n" + "=" * 100)
    print("ANALYSIS")
    print("=" * 100)
    
    best_gap = max(versions, key=lambda v: metrics[v]['gap'])
    best_inv = min(versions, key=lambda v: metrics[v]['inversions'])
    best_corr = max(versions, key=lambda v: metrics[v]['correlation'])
    
    print(f"✅ Best gap (similar-dissimilar): {best_gap} ({metrics[best_gap]['gap']:.3f})")
    print(f"✅ Fewest sign inversions: {best_inv} ({metrics[best_inv]['inversions']})")
    print(f"✅ Best correlation with original: {best_corr} ({metrics[best_corr]['correlation']:.3f})")
    
    # Overall winner: weighted score
    scores = {}
    for ver in versions:
        score = (metrics[ver]['gap'] * 0.4 
                - metrics[ver]['inversions'] * 0.2 
                + metrics[ver]['correlation'] * 0.4)
        scores[ver] = score
    
    winner = max(scores, key=scores.get)
    print(f"\n🏆 OVERALL WINNER: {winner} (score: {scores[winner]:.3f})")
    
    for ver in versions:
        m = metrics[ver]
        print(f"   {ver}: gap={m['gap']:.3f}, inv={m['inversions']}, corr={m['correlation']:.3f} → score={scores[ver]:.3f}")
    
    # Save results
    results["tests"]["encoder_comparison"] = {
        "similar_pairs": {
            "original": similar_orig,
            **{v: similar_results[v] for v in versions}
        },
        "dissimilar_pairs": {
            "original": dissimilar_orig,
            **{v: dissimilar_results[v] for v in versions}
        },
        "summary": {
            "gap_original": gap_orig,
            **{f"gap_{v}": metrics[v]['gap'] for v in versions},
            **{f"inversions_{v}": metrics[v]['inversions'] for v in versions},
            **{f"correlation_{v}": metrics[v]['correlation'] for v in versions},
            "winner": winner
        }
    }
    
    # Save
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"exp_04_encoder_comparison_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {result_file}")

if __name__ == "__main__":
    main()
