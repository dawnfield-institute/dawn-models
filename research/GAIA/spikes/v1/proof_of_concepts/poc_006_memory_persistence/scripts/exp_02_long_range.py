"""
Experiment 02: Long-Range Retrieval
====================================

Tests pattern retrieval accuracy after 100-1000+ intervening patterns.

Hypothesis: Resonance-based retrieval remains accurate even after
many new patterns, unlike attention which degrades with distance.

Success Criteria:
- >90% retrieval accuracy at depth 100
- >80% retrieval accuracy at depth 500
- >70% retrieval accuracy at depth 1000
- Graceful degradation curve
"""

import torch
import torch.nn.functional as F
from datetime import datetime
import json
import sys
from pathlib import Path

scripts_path = Path(__file__).resolve().parent
sys.path.insert(0, str(scripts_path))

from memory_field import MemoryField, SequentialMemory

# Dawn Field Constants
PHI = 1.618033988749895
XI = 0.0618
PHI_XI = PHI * XI
LAMBDA_STAR = 0.9816


def run_tests():
    """Run long-range retrieval tests."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"POC-006 Experiment 02: Long-Range Retrieval")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'tests': {}
    }
    
    passed = 0
    failed = 0
    
    # ===== TEST 1: 100-Pattern Depth =====
    print("\n" + "-"*40)
    print("TEST 1: Retrieval at Depth 100")
    print("-"*40)
    
    memory = MemoryField(shape=(24, 24, 24), device=device, capacity=2000)
    
    # Store target pattern first
    target = torch.randn(384, device=device)
    target = target / target.norm()
    target_id = memory.store(target)
    
    # Store 100 more random patterns
    for _ in range(100):
        noise = torch.randn(384, device=device)
        noise = noise / noise.norm()
        memory.store(noise)
    
    # Try to retrieve target via resonance
    retrieved = memory.retrieve(target, top_k=5)
    top_id = retrieved[0][0] if retrieved else None
    top_score = retrieved[0][1] if retrieved else 0
    
    print(f"Target ID: {target_id}")
    print(f"Top retrieved: ID={top_id}, score={top_score:.4f}")
    print(f"Top 5: {retrieved}")
    
    test1_pass = top_id == target_id
    results['tests']['depth_100'] = {
        'target_id': target_id,
        'retrieved': retrieved,
        'correct': test1_pass,
        'passed': test1_pass
    }
    
    if test1_pass:
        print(f"✓ PASSED: Target retrieved at depth 100")
        passed += 1
    else:
        print(f"✗ FAILED: Wrong pattern retrieved")
        failed += 1
    
    # ===== TEST 2: 500-Pattern Depth =====
    print("\n" + "-"*40)
    print("TEST 2: Retrieval at Depth 500")
    print("-"*40)
    
    memory2 = MemoryField(shape=(24, 24, 24), device=device, capacity=2000)
    
    # Store target
    target2 = torch.randn(384, device=device)
    target2 = target2 / target2.norm()
    target2_id = memory2.store(target2)
    
    # Store 500 more
    for _ in range(500):
        noise = torch.randn(384, device=device)
        noise = noise / noise.norm()
        memory2.store(noise)
    
    retrieved2 = memory2.retrieve(target2, top_k=5)
    top_id2 = retrieved2[0][0] if retrieved2 else None
    top_score2 = retrieved2[0][1] if retrieved2 else 0
    
    print(f"Target ID: {target2_id}")
    print(f"Top retrieved: ID={top_id2}, score={top_score2:.4f}")
    
    test2_pass = top_id2 == target2_id
    results['tests']['depth_500'] = {
        'target_id': target2_id,
        'top_retrieved': top_id2,
        'top_score': top_score2,
        'correct': test2_pass,
        'passed': test2_pass
    }
    
    if test2_pass:
        print(f"✓ PASSED: Target retrieved at depth 500")
        passed += 1
    else:
        print(f"✗ FAILED: Wrong pattern retrieved")
        failed += 1
    
    # ===== TEST 3: 1000-Pattern Depth =====
    print("\n" + "-"*40)
    print("TEST 3: Retrieval at Depth 1000")
    print("-"*40)
    
    memory3 = MemoryField(shape=(24, 24, 24), device=device, capacity=2000)
    
    # Store target
    target3 = torch.randn(384, device=device)
    target3 = target3 / target3.norm()
    target3_id = memory3.store(target3)
    
    # Store 1000 more
    for _ in range(1000):
        noise = torch.randn(384, device=device)
        noise = noise / noise.norm()
        memory3.store(noise)
    
    retrieved3 = memory3.retrieve(target3, top_k=5)
    top_id3 = retrieved3[0][0] if retrieved3 else None
    top_score3 = retrieved3[0][1] if retrieved3 else 0
    
    print(f"Target ID: {target3_id}")
    print(f"Top retrieved: ID={top_id3}, score={top_score3:.4f}")
    
    test3_pass = top_id3 == target3_id
    results['tests']['depth_1000'] = {
        'target_id': target3_id,
        'top_retrieved': top_id3,
        'top_score': top_score3,
        'correct': test3_pass,
        'passed': test3_pass
    }
    
    if test3_pass:
        print(f"✓ PASSED: Target retrieved at depth 1000")
        passed += 1
    else:
        print(f"✗ FAILED: Wrong pattern retrieved")
        failed += 1
    
    # ===== TEST 4: Multiple Targets =====
    print("\n" + "-"*40)
    print("TEST 4: Multiple Target Retrieval")
    print("-"*40)
    
    memory4 = MemoryField(shape=(24, 24, 24), device=device, capacity=2000)
    
    # Store 10 targets, then 200 noise, measure retrieval
    targets = []
    target_ids = []
    for i in range(10):
        t = torch.randn(384, device=device)
        t = t / t.norm()
        tid = memory4.store(t)
        targets.append(t)
        target_ids.append(tid)
    
    # Add noise
    for _ in range(200):
        noise = torch.randn(384, device=device)
        noise = noise / noise.norm()
        memory4.store(noise)
    
    # Retrieve each target
    correct = 0
    for i, (target, tid) in enumerate(zip(targets, target_ids)):
        retrieved = memory4.retrieve(target, top_k=1)
        if retrieved and retrieved[0][0] == tid:
            correct += 1
    
    accuracy = correct / len(targets)
    print(f"Retrieved {correct}/{len(targets)} targets correctly")
    print(f"Accuracy: {accuracy*100:.1f}%")
    
    test4_pass = accuracy >= 0.9
    results['tests']['multiple_targets'] = {
        'num_targets': len(targets),
        'correct': correct,
        'accuracy': accuracy,
        'passed': test4_pass
    }
    
    if test4_pass:
        print(f"✓ PASSED: {accuracy*100:.1f}% accuracy")
        passed += 1
    else:
        print(f"✗ FAILED: Accuracy below 90%")
        failed += 1
    
    # ===== TEST 5: Similar Patterns Discrimination =====
    print("\n" + "-"*40)
    print("TEST 5: Similar Pattern Discrimination")
    print("-"*40)
    
    memory5 = MemoryField(shape=(24, 24, 24), device=device, capacity=2000)
    
    # Create base pattern
    base = torch.randn(384, device=device)
    base = base / base.norm()
    
    # Create 5 variations (similar but different)
    variations = []
    variation_ids = []
    for i in range(5):
        # Add small noise to base
        noise = torch.randn(384, device=device) * 0.3
        var = base + noise
        var = var / var.norm()
        vid = memory5.store(var)
        variations.append(var)
        variation_ids.append(vid)
    
    # Add 100 random patterns
    for _ in range(100):
        noise = torch.randn(384, device=device)
        noise = noise / noise.norm()
        memory5.store(noise)
    
    # Query each variation - should retrieve itself
    correct = 0
    for var, vid in zip(variations, variation_ids):
        retrieved = memory5.retrieve(var, top_k=1)
        if retrieved and retrieved[0][0] == vid:
            correct += 1
    
    accuracy5 = correct / len(variations)
    print(f"Similar pattern discrimination: {correct}/{len(variations)}")
    
    test5_pass = accuracy5 >= 0.8
    results['tests']['similar_discrimination'] = {
        'num_variations': len(variations),
        'correct': correct,
        'accuracy': accuracy5,
        'passed': test5_pass
    }
    
    if test5_pass:
        print(f"✓ PASSED: Can discriminate similar patterns")
        passed += 1
    else:
        print(f"✗ FAILED: Poor discrimination")
        failed += 1
    
    # ===== TEST 6: Throughput at Scale =====
    print("\n" + "-"*40)
    print("TEST 6: Retrieval Throughput")
    print("-"*40)
    
    import time
    
    memory6 = MemoryField(shape=(24, 24, 24), device=device, capacity=2000)
    
    # Store 500 patterns
    for _ in range(500):
        p = torch.randn(384, device=device)
        p = p / p.norm()
        memory6.store(p)
    
    # Time 100 retrievals
    queries = [torch.randn(384, device=device) for _ in range(100)]
    
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    
    for q in queries:
        memory6.retrieve(q, top_k=5)
        
    torch.cuda.synchronize() if device == 'cuda' else None
    elapsed = time.time() - start
    
    throughput = 100 / elapsed
    print(f"Throughput: {throughput:.1f} retrievals/sec")
    
    test6_pass = throughput > 3  # At least 3 retrievals/sec (500 pattern memory)
    results['tests']['throughput'] = {
        'num_queries': 100,
        'elapsed': elapsed,
        'throughput': throughput,
        'passed': test6_pass
    }
    
    if test6_pass:
        print(f"✓ PASSED: Throughput acceptable")
        passed += 1
    else:
        print(f"✗ FAILED: Too slow")
        failed += 1
    
    # ===== SUMMARY =====
    print("\n" + "="*60)
    print(f"EXPERIMENT 02 RESULTS: {passed}/{passed+failed} passed")
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
    results_path = results_dir / f'exp_02_long_range_{timestamp}.json'
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved: {results_path}")
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = run_tests()
    sys.exit(0 if failed == 0 else 1)
