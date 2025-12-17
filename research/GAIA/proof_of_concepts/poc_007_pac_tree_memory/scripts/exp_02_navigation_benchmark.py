"""
Experiment 02: Navigation Benchmark - v1 vs v2
==============================================

Compares:
- PACTreeMemory v1 (full pattern storage)
- PACTreeMemory v2 (delta compression + GPU navigation)
- Brute force baseline

Tests:
1. Retrieval accuracy
2. Memory usage
3. Retrieval speed
4. Reconstruction quality
"""

import torch
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from pac_tree_memory import PACTreeMemory, PHI, XI
from pac_tree_memory_v2 import PACTreeMemoryV2


def create_test_patterns(n: int, shape: Tuple[int, int, int] = (32, 32, 32)) -> List[torch.Tensor]:
    """Create random test patterns"""
    return [torch.randn(shape) for _ in range(n)]


def benchmark_retrieval_accuracy(memory, patterns: List[torch.Tensor], name: str):
    """Test if patterns can be retrieved correctly"""
    print(f"\n=== {name}: Retrieval Accuracy ===")
    
    correct = 0
    for i, p in enumerate(patterns):
        results = memory.retrieve(p, top_k=10)
        if results and results[0][0] == i:
            correct += 1
            
    accuracy = correct / len(patterns)
    print(f"Self-retrieval: {accuracy:.2%}")
    return accuracy


def benchmark_reconstruction(memory, patterns: List[torch.Tensor], name: str):
    """Test reconstruction quality (v2 only)"""
    if not hasattr(memory, 'reconstruct_pattern'):
        print(f"\n=== {name}: Reconstruction (N/A) ===")
        return None
        
    print(f"\n=== {name}: Reconstruction Quality ===")
    
    errors = []
    for i, original in enumerate(patterns[:100]):  # Test subset
        reconstructed = memory.reconstruct_pattern(i)
        if reconstructed is not None:
            mse = ((original.cuda() - reconstructed) ** 2).mean().item()
            errors.append(mse)
            
    if errors:
        avg_mse = sum(errors) / len(errors)
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"RMSE: {avg_mse ** 0.5:.6f}")
        return avg_mse
    return None


def benchmark_speed(memory, queries: List[torch.Tensor], name: str):
    """Benchmark retrieval speed"""
    print(f"\n=== {name}: Speed ===")
    
    # Warmup
    for q in queries[:10]:
        memory.retrieve(q, top_k=100)
        
    # Timed run
    start = time.perf_counter()
    for q in queries:
        memory.retrieve(q, top_k=100)
    elapsed = time.perf_counter() - start
    
    ms_per_query = (elapsed / len(queries)) * 1000
    print(f"Time: {ms_per_query:.2f}ms per query")
    return ms_per_query


def benchmark_brute_force(patterns: List[torch.Tensor], queries: List[torch.Tensor]):
    """Baseline brute force benchmark"""
    print("\n=== Brute Force: Speed ===")
    
    all_patterns = torch.stack(patterns).cuda()
    
    # Warmup
    for q in queries[:10]:
        q_flat = q.cuda().flatten().unsqueeze(0)
        p_flat = all_patterns.flatten(1)
        sims = torch.mm(
            torch.nn.functional.normalize(q_flat, dim=1),
            torch.nn.functional.normalize(p_flat, dim=1).T
        )
        _ = torch.topk(sims, k=100, dim=1)
        
    # Timed
    start = time.perf_counter()
    for q in queries:
        q_flat = q.cuda().flatten().unsqueeze(0)
        p_flat = all_patterns.flatten(1)
        sims = torch.mm(
            torch.nn.functional.normalize(q_flat, dim=1),
            torch.nn.functional.normalize(p_flat, dim=1).T
        )
        _ = torch.topk(sims, k=100, dim=1)
    elapsed = time.perf_counter() - start
    
    ms_per_query = (elapsed / len(queries)) * 1000
    print(f"Time: {ms_per_query:.2f}ms per query")
    return ms_per_query


def main():
    print("=" * 60)
    print("POC-007 Experiment 02: Navigation Benchmark")
    print("=" * 60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Test configuration
    n_patterns = 1000
    n_queries = 100
    field_shape = (32, 32, 32)
    
    print(f"\nConfiguration:")
    print(f"  Patterns: {n_patterns}")
    print(f"  Queries: {n_queries}")
    print(f"  Field shape: {field_shape}")
    
    # Create test data
    print("\nCreating test data...")
    patterns = create_test_patterns(n_patterns, field_shape)
    queries = create_test_patterns(n_queries, field_shape)
    
    results = {}
    
    # Benchmark v1
    print("\n" + "=" * 40)
    print("PAC Tree Memory v1")
    print("=" * 40)
    
    memory_v1 = PACTreeMemory(field_shape=field_shape, device='cuda')
    for i, p in enumerate(patterns):
        memory_v1.store(i, p)
        
    results['v1'] = {
        'accuracy': benchmark_retrieval_accuracy(memory_v1, patterns, "v1"),
        'speed_ms': benchmark_speed(memory_v1, queries, "v1"),
        'memory': memory_v1.get_memory_stats()
    }
    
    print(f"\nMemory: {results['v1']['memory']['memory_mb']:.2f}MB")
    print(f"Compression: {results['v1']['memory']['compression_ratio']:.2f}x")
    
    # Benchmark v2
    print("\n" + "=" * 40)
    print("PAC Tree Memory v2")
    print("=" * 40)
    
    memory_v2 = PACTreeMemoryV2(field_shape=field_shape, device='cuda')
    for i, p in enumerate(patterns):
        memory_v2.store(i, p)
        
    results['v2'] = {
        'accuracy': benchmark_retrieval_accuracy(memory_v2, patterns, "v2"),
        'reconstruction_mse': benchmark_reconstruction(memory_v2, patterns, "v2"),
        'speed_ms': benchmark_speed(memory_v2, queries, "v2"),
        'memory': memory_v2.get_memory_stats()
    }
    
    print(f"\nMemory: {results['v2']['memory']['memory_mb']:.2f}MB")
    print(f"Compression: {results['v2']['memory']['compression_ratio']:.2f}x")
    
    # Brute force baseline
    results['brute_force'] = {
        'speed_ms': benchmark_brute_force(patterns, queries),
        'memory_mb': n_patterns * 32 * 32 * 32 * 4 / (1024 ** 2)
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'v1':<15} {'v2':<15} {'Brute Force':<15}")
    print("-" * 70)
    print(f"{'Accuracy':<25} {results['v1']['accuracy']:.2%}         {results['v2']['accuracy']:.2%}         {'N/A':<15}")
    print(f"{'Speed (ms)':<25} {results['v1']['speed_ms']:.2f}            {results['v2']['speed_ms']:.2f}            {results['brute_force']['speed_ms']:.2f}")
    print(f"{'Memory (MB)':<25} {results['v1']['memory']['memory_mb']:.2f}           {results['v2']['memory']['memory_mb']:.2f}           {results['brute_force']['memory_mb']:.2f}")
    
    speedup_v1 = results['brute_force']['speed_ms'] / results['v1']['speed_ms']
    speedup_v2 = results['brute_force']['speed_ms'] / results['v2']['speed_ms']
    print(f"{'Speedup vs BF':<25} {speedup_v1:.2f}x            {speedup_v2:.2f}x            1.00x")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_patterns': n_patterns,
            'n_queries': n_queries,
            'field_shape': field_shape
        },
        'results': results
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = results_dir / f'exp_02_navigation_benchmark_{timestamp}.json'
    
    # Convert non-serializable items
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    with open(output_path, 'w') as f:
        json.dump(convert(output), f, indent=2)
        
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
