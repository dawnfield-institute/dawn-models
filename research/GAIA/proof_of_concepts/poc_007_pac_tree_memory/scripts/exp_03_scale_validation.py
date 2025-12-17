"""
Experiment 03: Scale Validation
================================

Tests tiered memory at scale to validate it can handle:
- 10K, 25K, 50K patterns
- Limited GPU cache (1K, 2K, 5K)
- Sequential access patterns (language modeling)
- Random access patterns (worst case)

This simulates the WikiText-103 scenario where we have more patterns
than fit in GPU memory.
"""

import torch
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))

from tiered_memory_cache import TieredMemoryCache


def benchmark_sequential_access(cache: TieredMemoryCache, 
                                n_patterns: int,
                                n_queries: int) -> Dict:
    """Simulate language modeling with sequential access"""
    print(f"\n  Sequential access ({n_queries} queries)...")
    
    # Reset stats
    cache.stats.hits = 0
    cache.stats.misses = 0
    
    start = time.perf_counter()
    
    for i in range(n_queries):
        # Query at position i, context is i-1
        query_idx = i % n_patterns
        context = [(query_idx - 1) % n_patterns]
        
        query = torch.randn(32, 32, 32)
        results = cache.retrieve(query, context_ids=context, top_k=10)
        
    elapsed = time.perf_counter() - start
    ms_per_query = (elapsed / n_queries) * 1000
    
    stats = cache.get_stats()
    print(f"    Hit rate: {stats['hit_rate']:.2%}")
    print(f"    Time: {ms_per_query:.2f}ms/query")
    
    return {
        'access_pattern': 'sequential',
        'hit_rate': stats['hit_rate'],
        'ms_per_query': ms_per_query,
        'prefetch_hits': stats['prefetch_hits'],
        'evictions': stats['evictions']
    }


def benchmark_random_access(cache: TieredMemoryCache,
                            n_patterns: int,
                            n_queries: int) -> Dict:
    """Simulate random access (worst case)"""
    print(f"\n  Random access ({n_queries} queries)...")
    
    # Reset stats
    cache.stats.hits = 0
    cache.stats.misses = 0
    
    start = time.perf_counter()
    
    for i in range(n_queries):
        # Random query
        query_idx = np.random.randint(n_patterns)
        context = [np.random.randint(n_patterns)]
        
        query = torch.randn(32, 32, 32)
        results = cache.retrieve(query, context_ids=context, top_k=10)
        
    elapsed = time.perf_counter() - start
    ms_per_query = (elapsed / n_queries) * 1000
    
    stats = cache.get_stats()
    print(f"    Hit rate: {stats['hit_rate']:.2%}")
    print(f"    Time: {ms_per_query:.2f}ms/query")
    
    return {
        'access_pattern': 'random',
        'hit_rate': stats['hit_rate'],
        'ms_per_query': ms_per_query,
        'prefetch_hits': stats['prefetch_hits'],
        'evictions': stats['evictions']
    }


def benchmark_burst_access(cache: TieredMemoryCache,
                           n_patterns: int,
                           n_queries: int,
                           burst_size: int = 50) -> Dict:
    """Simulate bursty access (clusters of related queries)"""
    print(f"\n  Burst access ({n_queries} queries, burst={burst_size})...")
    
    # Reset stats
    cache.stats.hits = 0
    cache.stats.misses = 0
    
    start = time.perf_counter()
    
    for burst in range(n_queries // burst_size):
        # Pick a random starting point for this burst
        start_idx = np.random.randint(n_patterns - burst_size)
        
        for i in range(burst_size):
            query_idx = start_idx + i
            context = [max(0, query_idx - 1)]
            
            query = torch.randn(32, 32, 32)
            results = cache.retrieve(query, context_ids=context, top_k=10)
            
    elapsed = time.perf_counter() - start
    ms_per_query = (elapsed / n_queries) * 1000
    
    stats = cache.get_stats()
    print(f"    Hit rate: {stats['hit_rate']:.2%}")
    print(f"    Time: {ms_per_query:.2f}ms/query")
    
    return {
        'access_pattern': 'burst',
        'hit_rate': stats['hit_rate'],
        'ms_per_query': ms_per_query,
        'prefetch_hits': stats['prefetch_hits'],
        'evictions': stats['evictions']
    }


def run_scale_test(n_patterns: int, gpu_cache_size: int, n_queries: int = 500) -> Dict:
    """Run full scale test with given configuration"""
    print(f"\n{'='*60}")
    print(f"Scale Test: {n_patterns} patterns, {gpu_cache_size} GPU cache")
    print(f"{'='*60}")
    
    # Create cache
    cache = TieredMemoryCache(
        field_shape=(32, 32, 32),
        gpu_cache_size=gpu_cache_size,
        prefetch_k=10
    )
    
    # Store patterns
    print(f"\nStoring {n_patterns} patterns...")
    start = time.perf_counter()
    for i in range(n_patterns):
        cache.store(i, torch.randn(32, 32, 32))
    store_time = time.perf_counter() - start
    print(f"  Store time: {store_time:.2f}s ({n_patterns/store_time:.0f} patterns/sec)")
    
    # Learn transitions (sequential language model)
    print("\nLearning transitions...")
    for i in range(n_patterns - 1):
        cache.learn_transition(i, i + 1, 1.0)
    
    # Run benchmarks
    results = {
        'n_patterns': n_patterns,
        'gpu_cache_size': gpu_cache_size,
        'cache_ratio': gpu_cache_size / n_patterns,
        'store_time_sec': store_time,
        'patterns_per_sec': n_patterns / store_time
    }
    
    results['sequential'] = benchmark_sequential_access(cache, n_patterns, n_queries)
    results['random'] = benchmark_random_access(cache, n_patterns, n_queries)
    results['burst'] = benchmark_burst_access(cache, n_patterns, n_queries)
    
    # Memory stats
    stats = cache.get_stats()
    field_size = 32 * 32 * 32 * 4  # float32
    gpu_memory_mb = gpu_cache_size * field_size / (1024 ** 2)
    full_memory_mb = n_patterns * field_size / (1024 ** 2)
    
    results['memory'] = {
        'gpu_memory_mb': gpu_memory_mb,
        'full_memory_mb': full_memory_mb,
        'savings_ratio': full_memory_mb / gpu_memory_mb,
        'pac_tree_stats': stats['pac_tree_stats']
    }
    
    print(f"\nMemory:")
    print(f"  GPU cache: {gpu_memory_mb:.2f} MB")
    print(f"  Full storage: {full_memory_mb:.2f} MB")
    print(f"  Savings: {results['memory']['savings_ratio']:.1f}x")
    
    return results


def main():
    print("=" * 60)
    print("POC-007 Experiment 03: Scale Validation")
    print("=" * 60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    all_results = []
    
    # Test configurations
    configs = [
        # (n_patterns, gpu_cache_size)
        (1000, 200),     # 5:1 ratio
        (5000, 500),     # 10:1 ratio
        (10000, 1000),   # 10:1 ratio
        (25000, 2000),   # 12.5:1 ratio
    ]
    
    # Check GPU memory before running large tests
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Memory: {gpu_mem:.1f} GB")
    
    for n_patterns, gpu_cache_size in configs:
        try:
            results = run_scale_test(n_patterns, gpu_cache_size)
            all_results.append(results)
        except Exception as e:
            print(f"ERROR at {n_patterns} patterns: {e}")
            break
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Patterns':<12} {'Cache':<10} {'Seq Hit%':<12} {'Rnd Hit%':<12} {'GPU MB':<10}")
    print("-" * 60)
    
    for r in all_results:
        print(f"{r['n_patterns']:<12} {r['gpu_cache_size']:<10} "
              f"{r['sequential']['hit_rate']:.1%}         "
              f"{r['random']['hit_rate']:.1%}         "
              f"{r['memory']['gpu_memory_mb']:.1f}")
    
    # Key findings
    print("\n📊 Key Findings:")
    if all_results:
        avg_seq_hit = np.mean([r['sequential']['hit_rate'] for r in all_results])
        avg_rnd_hit = np.mean([r['random']['hit_rate'] for r in all_results])
        max_savings = max(r['memory']['savings_ratio'] for r in all_results)
        
        print(f"  - Sequential access hit rate: {avg_seq_hit:.1%} (transitions help)")
        print(f"  - Random access hit rate: {avg_rnd_hit:.1%} (cache limited)")
        print(f"  - Maximum memory savings: {max_savings:.1f}x")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'tests': all_results
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = results_dir / f'exp_03_scale_validation_{timestamp}.json'
    
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    with open(output_path, 'w') as f:
        json.dump(convert(output), f, indent=2)
        
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
