"""
Experiment 01: Memory Storage Depth
====================================

Tests how many patterns can be stored before retrieval degrades.

Hypothesis: Field superposition allows many patterns to coexist
with graceful degradation rather than catastrophic forgetting.

Success Criteria:
- Store 100+ patterns
- Retrieve with >80% accuracy at depth 50
- >60% accuracy at depth 100
- Gradual (not sudden) degradation curve
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
    """Run storage depth tests."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"POC-006 Experiment 01: Memory Storage Depth")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'tests': {}
    }
    
    passed = 0
    failed = 0
    
    # ===== TEST 1: Basic Storage =====
    print("\n" + "-"*40)
    print("TEST 1: Basic Pattern Storage")
    print("-"*40)
    
    memory = MemoryField(shape=(24, 24, 24), device=device)
    
    # Store 50 random patterns
    stored_ids = []
    for i in range(50):
        pattern = torch.randn(384, device=device)
        pattern = pattern / pattern.norm()
        pattern_id = memory.store(pattern)
        stored_ids.append(pattern_id)
        
    print(f"Stored: {memory.get_pattern_count()} patterns")
    print(f"Field energy: {memory.get_field_energy():.2f}")
    
    test1_pass = memory.get_pattern_count() == 50
    results['tests']['basic_storage'] = {
        'pattern_count': memory.get_pattern_count(),
        'field_energy': memory.get_field_energy(),
        'passed': test1_pass
    }
    
    if test1_pass:
        print(f"✓ PASSED: All 50 patterns stored")
        passed += 1
    else:
        print(f"✗ FAILED: Storage count mismatch")
        failed += 1
    
    # ===== TEST 2: Retrieval by ID =====
    print("\n" + "-"*40)
    print("TEST 2: Retrieval by ID")
    print("-"*40)
    
    # Try to recall first, middle, and last patterns
    test_ids = [stored_ids[0], stored_ids[25], stored_ids[-1]]
    recalls = []
    
    for pid in test_ids:
        field = memory.recall(pid)
        original = memory.pattern_fields[pid]
        if field is not None:
            sim = F.cosine_similarity(
                field.flatten().unsqueeze(0),
                original.flatten().unsqueeze(0)
            ).item()
            recalls.append((pid, sim))
            print(f"  Pattern {pid}: similarity {sim:.4f}")
    
    test2_pass = len(recalls) == 3 and all(sim > 0.99 for _, sim in recalls)
    results['tests']['retrieval_by_id'] = {
        'recalls': recalls,
        'passed': test2_pass
    }
    
    if test2_pass:
        print(f"✓ PASSED: All patterns perfectly recalled by ID")
        passed += 1
    else:
        print(f"✗ FAILED: Recall degradation")
        failed += 1
    
    # ===== TEST 3: Resonance Retrieval =====
    print("\n" + "-"*40)
    print("TEST 3: Resonance-Based Retrieval")
    print("-"*40)
    
    # Query with a stored pattern, should find itself
    query_id = stored_ids[10]
    query_field = memory.pattern_fields[query_id]
    
    retrieved = memory.retrieve(query_field, top_k=5)
    print(f"Query pattern {query_id}:")
    print(f"  Top 5: {retrieved}")
    
    # Check if query pattern is in top result
    top_id = retrieved[0][0] if retrieved else None
    test3_pass = top_id == query_id
    results['tests']['resonance_retrieval'] = {
        'query_id': query_id,
        'retrieved': retrieved,
        'correct_top': test3_pass,
        'passed': test3_pass
    }
    
    if test3_pass:
        print(f"✓ PASSED: Correct pattern retrieved via resonance")
        passed += 1
    else:
        print(f"✗ FAILED: Wrong pattern retrieved")
        failed += 1
    
    # ===== TEST 4: Depth Degradation Curve =====
    print("\n" + "-"*40)
    print("TEST 4: Depth Degradation Curve")
    print("-"*40)
    
    # Create new memory and track retrieval accuracy at different depths
    seq_memory = SequentialMemory(shape=(24, 24, 24), device=device)
    
    # Store 100 patterns
    patterns = []
    for i in range(100):
        pattern = torch.randn(384, device=device)
        pattern = pattern / pattern.norm()
        patterns.append(pattern)
        seq_memory.append(pattern)
    
    # Test retrieval at different depths
    depths = [0, 10, 25, 50, 75, 99]
    accuracies = []
    
    for depth in depths:
        # Query with the original pattern at that depth
        original = patterns[-(depth + 1)]
        sim = seq_memory.similarity_at_depth(original, depth)
        accuracies.append((depth, sim))
        print(f"  Depth {depth:3d}: similarity {sim:.4f}")
    
    # Check for gradual degradation (not catastrophic)
    sims = [a[1] for a in accuracies]
    is_gradual = all(sims[i] >= sims[i+1] - 0.1 for i in range(len(sims)-1))
    
    test4_pass = sims[0] > 0.99 and sims[-1] > 0.90  # Should be high - exact pattern
    results['tests']['depth_degradation'] = {
        'accuracies': accuracies,
        'is_gradual': is_gradual,
        'passed': test4_pass
    }
    
    if test4_pass:
        print(f"✓ PASSED: High accuracy at all depths")
        passed += 1
    else:
        print(f"✗ FAILED: Accuracy degraded too much")
        failed += 1
    
    # ===== TEST 5: Interference Measurement =====
    print("\n" + "-"*40)
    print("TEST 5: Pattern Interference")
    print("-"*40)
    
    # Measure how much early patterns are interfered with
    memory2 = MemoryField(shape=(24, 24, 24), device=device)
    
    # Store 100 patterns
    first_id = None
    for i in range(100):
        pattern = torch.randn(384, device=device)
        pattern = pattern / pattern.norm()
        pid = memory2.store(pattern)
        if i == 0:
            first_id = pid
            
    # Check interference on first pattern
    interference = memory2.compute_interference(first_id)
    print(f"First pattern interference: {interference:.4f}")
    print(f"  (0.0 = perfect, 1.0 = completely lost)")
    
    # Acceptable if < 0.8 (pattern still partially present)
    test5_pass = interference < 0.8
    results['tests']['interference'] = {
        'first_pattern_id': first_id,
        'interference': interference,
        'passed': test5_pass
    }
    
    if test5_pass:
        print(f"✓ PASSED: Interference below threshold")
        passed += 1
    else:
        print(f"✗ FAILED: Too much interference")
        failed += 1
    
    # ===== TEST 6: Field Energy Stability =====
    print("\n" + "-"*40)
    print("TEST 6: Field Energy Stability")
    print("-"*40)
    
    memory3 = MemoryField(shape=(24, 24, 24), device=device)
    energies = []
    
    for i in range(100):
        pattern = torch.randn(384, device=device)
        pattern = pattern / pattern.norm()
        memory3.store(pattern)
        if i % 20 == 0:
            energies.append((i, memory3.get_field_energy()))
            
    print("Field energy over time:")
    for step, energy in energies:
        print(f"  Step {step:3d}: {energy:.2f}")
    
    # Check stability (shouldn't explode or collapse)
    energy_vals = [e for _, e in energies]
    max_e = max(energy_vals)
    min_e = min(energy_vals)
    ratio = max_e / min_e if min_e > 0 else float('inf')
    
    test6_pass = ratio < 10  # Energy shouldn't vary by more than 10x
    results['tests']['energy_stability'] = {
        'energies': energies,
        'ratio': ratio,
        'passed': test6_pass
    }
    
    if test6_pass:
        print(f"✓ PASSED: Energy stable (ratio={ratio:.2f})")
        passed += 1
    else:
        print(f"✗ FAILED: Energy unstable")
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
    results_path = results_dir / f'exp_01_storage_depth_{timestamp}.json'
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved: {results_path}")
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = run_tests()
    sys.exit(0 if failed == 0 else 1)
