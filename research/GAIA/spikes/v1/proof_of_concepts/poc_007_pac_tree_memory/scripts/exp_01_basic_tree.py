"""
Experiment 01: Basic PAC Tree Operations
=========================================

Validates core PAC tree functionality:
1. Store patterns efficiently
2. Retrieve by resonance
3. Memory compression
4. O(log n) scaling

This establishes baseline for further experiments.
"""

import torch
import time
import json
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from pac_tree_memory import PACTreeMemory, PHI, XI, PHI_XI


def test_basic_store_retrieve():
    """Test basic storage and retrieval"""
    print("\n=== Test: Basic Store/Retrieve ===")
    
    memory = PACTreeMemory(field_shape=(32, 32, 32), device='cuda')
    
    # Store 100 random patterns
    n_patterns = 100
    patterns = {}
    
    for i in range(n_patterns):
        field = torch.randn(32, 32, 32)
        patterns[i] = field
        memory.store(i, field)
        
    print(f"Stored {n_patterns} patterns")
    
    # Retrieve and check
    correct = 0
    for pattern_id, field in patterns.items():
        results = memory.retrieve(field, top_k=10)
        if results and results[0][0] == pattern_id:
            correct += 1
            
    accuracy = correct / n_patterns
    print(f"Self-retrieval accuracy: {accuracy:.2%}")
    
    stats = memory.get_memory_stats()
    print(f"Nodes: {stats['total_nodes']} (leaves: {stats['leaf_nodes']})")
    print(f"Memory: {stats['memory_mb']:.2f} MB")
    
    return {
        'test': 'basic_store_retrieve',
        'n_patterns': n_patterns,
        'accuracy': accuracy,
        'passed': accuracy > 0.9
    }


def test_similar_pattern_retrieval():
    """Test retrieval of similar patterns"""
    print("\n=== Test: Similar Pattern Retrieval ===")
    
    memory = PACTreeMemory(field_shape=(32, 32, 32), device='cuda')
    
    # Create cluster of similar patterns
    base_field = torch.randn(32, 32, 32)
    noise_level = 0.1
    
    # Store base pattern
    memory.store(0, base_field)
    
    # Store variations
    for i in range(1, 20):
        variation = base_field + torch.randn_like(base_field) * noise_level
        memory.store(i, variation)
        
    # Store unrelated patterns
    for i in range(20, 100):
        memory.store(i, torch.randn(32, 32, 32))
        
    # Query with slight variation of base
    query = base_field + torch.randn_like(base_field) * 0.05
    results = memory.retrieve(query, top_k=20)
    
    # Check if similar patterns rank higher
    similar_ids = set(range(20))
    top_10 = set(r[0] for r in results[:10])
    recall = len(top_10 & similar_ids) / 10
    
    print(f"Similar patterns in top-10: {len(top_10 & similar_ids)}")
    print(f"Recall: {recall:.2%}")
    
    return {
        'test': 'similar_pattern_retrieval',
        'recall': recall,
        'passed': recall > 0.7
    }


def test_transition_learning():
    """Test transition learning and retrieval"""
    print("\n=== Test: Transition Learning ===")
    
    memory = PACTreeMemory(field_shape=(32, 32, 32), device='cuda')
    
    # Store patterns
    n_patterns = 50
    patterns = {}
    for i in range(n_patterns):
        patterns[i] = torch.randn(32, 32, 32)
        memory.store(i, patterns[i])
        
    # Create transition chain: 0 -> 1 -> 2 -> 3 -> 4
    for i in range(4):
        memory.learn_transition(i, i+1, strength=1.0)
        
    # Query from pattern 2's context
    results = memory.retrieve(patterns[3], context_ids=[2], top_k=10)
    
    # Pattern 3 should be highly ranked due to transition
    pattern_3_rank = None
    for rank, (pid, score) in enumerate(results):
        if pid == 3:
            pattern_3_rank = rank
            break
            
    print(f"Pattern 3 rank (with context [2]): {pattern_3_rank}")
    
    # Without context
    results_no_context = memory.retrieve(patterns[3], top_k=10)
    pattern_3_rank_no_ctx = None
    for rank, (pid, score) in enumerate(results_no_context):
        if pid == 3:
            pattern_3_rank_no_ctx = rank
            break
            
    print(f"Pattern 3 rank (no context): {pattern_3_rank_no_ctx}")
    
    return {
        'test': 'transition_learning',
        'rank_with_context': pattern_3_rank,
        'rank_without_context': pattern_3_rank_no_ctx,
        'passed': pattern_3_rank is not None and pattern_3_rank < 3
    }


def test_memory_efficiency():
    """Test memory efficiency at different scales"""
    print("\n=== Test: Memory Efficiency ===")
    
    scales = [100, 500, 1000, 2000]
    results = []
    
    for n in scales:
        memory = PACTreeMemory(field_shape=(32, 32, 32), device='cuda')
        
        for i in range(n):
            memory.store(i, torch.randn(32, 32, 32))
            
        stats = memory.get_memory_stats()
        results.append({
            'n_patterns': n,
            'tree_memory_mb': stats['memory_mb'],
            'flat_memory_mb': stats['flat_memory_mb'],
            'compression': stats['compression_ratio'],
            'depth': stats['max_depth']
        })
        
        print(f"n={n}: tree={stats['memory_mb']:.2f}MB, flat={stats['flat_memory_mb']:.2f}MB, "
              f"compression={stats['compression_ratio']:.2f}x, depth={stats['max_depth']}")
        
    return {
        'test': 'memory_efficiency',
        'results': results,
        'passed': all(r['compression'] > 0.8 for r in results)  # Tree should be efficient
    }


def test_retrieval_speed():
    """Test retrieval speed vs brute force"""
    print("\n=== Test: Retrieval Speed ===")
    
    n_patterns = 2000
    memory = PACTreeMemory(field_shape=(32, 32, 32), device='cuda')
    
    patterns = []
    for i in range(n_patterns):
        p = torch.randn(32, 32, 32)
        patterns.append(p)
        memory.store(i, p)
        
    # Time tree retrieval
    n_queries = 100
    queries = [torch.randn(32, 32, 32) for _ in range(n_queries)]
    
    start = time.perf_counter()
    for q in queries:
        memory.retrieve(q, top_k=100)
    tree_time = time.perf_counter() - start
    tree_ms = (tree_time / n_queries) * 1000
    
    # Time brute force
    all_patterns = torch.stack(patterns).cuda()
    
    start = time.perf_counter()
    for q in queries:
        q_flat = q.cuda().flatten().unsqueeze(0)
        q_norm = torch.nn.functional.normalize(q_flat, dim=1)
        p_flat = all_patterns.flatten(1)
        p_norm = torch.nn.functional.normalize(p_flat, dim=1)
        sims = torch.mm(q_norm, p_norm.T)
        top_k = torch.topk(sims, k=100, dim=1)
    brute_time = time.perf_counter() - start
    brute_ms = (brute_time / n_queries) * 1000
    
    speedup = brute_ms / tree_ms
    print(f"Tree retrieval: {tree_ms:.2f}ms per query")
    print(f"Brute force: {brute_ms:.2f}ms per query")
    print(f"Speedup: {speedup:.2f}x")
    
    stats = memory.get_memory_stats()
    print(f"Avg nodes visited: {stats['stats']['nodes_visited'] / stats['stats']['retrievals']:.1f}")
    
    return {
        'test': 'retrieval_speed',
        'tree_ms': tree_ms,
        'brute_ms': brute_ms,
        'speedup': speedup,
        'n_patterns': n_patterns,
        'passed': True  # Speed test is informational
    }


def test_tree_structure():
    """Test that tree structure is balanced"""
    print("\n=== Test: Tree Structure ===")
    
    memory = PACTreeMemory(field_shape=(32, 32, 32), device='cuda')
    
    # Store patterns with various similarity structures
    for i in range(500):
        memory.store(i, torch.randn(32, 32, 32))
        
    stats = memory.get_memory_stats()
    
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Max depth: {stats['max_depth']}")
    print(f"Avg children: {stats['avg_children']:.2f}")
    
    # Check balance
    expected_depth = max(1, int(torch.tensor(500.0).log() / torch.tensor(8.0).log()))
    depth_ratio = stats['max_depth'] / expected_depth
    
    print(f"Expected depth (log8): {expected_depth}")
    print(f"Depth ratio: {depth_ratio:.2f}")
    
    return {
        'test': 'tree_structure',
        'total_nodes': stats['total_nodes'],
        'max_depth': stats['max_depth'],
        'avg_children': stats['avg_children'],
        'depth_ratio': depth_ratio,
        'passed': depth_ratio < 3  # Should be reasonably balanced
    }


def main():
    """Run all experiments"""
    print("=" * 60)
    print("POC-007 Experiment 01: Basic PAC Tree Operations")
    print("=" * 60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Constants: PHI={PHI:.4f}, XI={XI:.4f}, PHI*XI={PHI_XI:.4f}")
    
    results = []
    
    # Run tests
    results.append(test_basic_store_retrieve())
    results.append(test_similar_pattern_retrieval())
    results.append(test_transition_learning())
    results.append(test_memory_efficiency())
    results.append(test_retrieval_speed())
    results.append(test_tree_structure())
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    
    for r in results:
        status = "✅ PASS" if r['passed'] else "❌ FAIL"
        print(f"{status}: {r['test']}")
        
    print(f"\nTotal: {passed}/{total} tests passed")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'constants': {'PHI': PHI, 'XI': XI, 'PHI_XI': PHI_XI},
        'tests': results,
        'summary': {'passed': passed, 'total': total}
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = results_dir / f'exp_01_basic_tree_{timestamp}.json'
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
        
    print(f"\nResults saved to: {output_path}")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
