"""
Experiment 24: TinyCIMM-Möbius Convergence Maintenance

Tests whether a Möbius network can:
1. Learn a pattern (Phase 1: Training)
2. Continue learning new patterns without forgetting (Phase 2: Continuous)
3. Maintain φ-resonance stability (Phase 3: Validation)

Key hypothesis: The Möbius frequency acts as memory - once a pattern is learned,
the fixed points encode it, and new learning should preserve this structure.

This tests "catastrophic forgetting" resistance through:
1. Baseline (no anchor memory) - expected to fail
2. With PhiAnchorMemory - should maintain convergence

The PhiAnchorMemory is the TinyCIMM-Möbius equivalent of micro_memory from TinyCIMM-Planck.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from tinycimm_mobius import TinyCIMMMobius, PHI, PHI_INV

# Constants
RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def run_convergence_test(model, fibs, results_key):
    """Run the 3-phase convergence test on a model."""
    
    results = {
        'phases': {},
        'success_criteria': {
            'phase1_error': 0.01,
            'phase2_error': 0.1,
            'phase3_retention': 0.05
        }
    }
    
    def fib_stream(n):
        for _ in range(n):
            idx = np.random.randint(5, 25)
            x = np.array([fibs[idx] / fibs[idx-1]])
            y = np.array([fibs[idx+1] / fibs[idx]])
            yield x, y
    
    def polynomial_stream(n):
        for _ in range(n):
            x_val = np.random.uniform(0.5, 2.0)
            x = np.array([x_val])
            y = np.array([0.5 * x_val**2 + 0.3 * x_val])
            yield x, y
    
    # ========== PHASE 1: Train on Fibonacci ==========
    print(f"\n  Phase 1: Fibonacci Training ({results_key})")
    
    # Set task for anchor memory
    if model.anchor_memory:
        model.anchor_memory.set_task('fibonacci')
    
    phase1_history = model.continuous_train(
        fib_stream(500), 
        max_steps=500, 
        log_interval=200
    )
    
    # Evaluate Phase 1
    fib_errors = []
    for idx in range(5, 20):
        x = torch.tensor([[fibs[idx] / fibs[idx-1]]], dtype=torch.float32)
        y_true = fibs[idx+1] / fibs[idx]
        y_pred = model(x).item()
        fib_errors.append(abs(y_pred - y_true))
    
    phase1_error = np.mean(fib_errors)
    phase1_success = phase1_error < results['success_criteria']['phase1_error']
    
    results['phases']['phase1'] = {
        'test_error': float(phase1_error),
        'phi_frequency': model.get_phi_frequency(),
        'success': bool(phase1_success)
    }
    
    print(f"    Error: {phase1_error:.6f} ({'✓' if phase1_success else '✗'})")
    
    # Store Fibonacci weights for later comparison
    fib_weights = model.get_memory_summary()['layer_params']
    
    # ========== PHASE 2: Continue on Different Pattern ==========
    print(f"  Phase 2: Polynomial Learning ({results_key})")
    
    # Set task for anchor memory
    if model.anchor_memory:
        model.anchor_memory.set_task('polynomial')
    
    phase2_history = model.continuous_train(
        polynomial_stream(300),
        max_steps=300,
        log_interval=200
    )
    
    # Evaluate Phase 2
    poly_errors = []
    for _ in range(20):
        x_val = np.random.uniform(0.5, 2.0)
        x = torch.tensor([[x_val]], dtype=torch.float32)
        y_true = 0.5 * x_val**2 + 0.3 * x_val
        y_pred = model(x).item()
        poly_errors.append(abs(y_pred - y_true))
    
    phase2_error = np.mean(poly_errors)
    phase2_success = phase2_error < results['success_criteria']['phase2_error']
    
    results['phases']['phase2'] = {
        'test_error': float(phase2_error),
        'phi_frequency': model.get_phi_frequency(),
        'success': bool(phase2_success)
    }
    
    print(f"    Error: {phase2_error:.6f} ({'✓' if phase2_success else '✗'})")
    
    # ========== PHASE 3: Test Retention ==========
    print(f"  Phase 3: Fibonacci Retention ({results_key})")
    
    # Re-evaluate Fibonacci (without any retraining)
    retention_errors = []
    for idx in range(5, 20):
        x = torch.tensor([[fibs[idx] / fibs[idx-1]]], dtype=torch.float32)
        y_true = fibs[idx+1] / fibs[idx]
        y_pred = model(x).item()
        retention_errors.append(abs(y_pred - y_true))
    
    phase3_error = np.mean(retention_errors)
    retention_ratio = phase3_error / max(phase1_error, 1e-8)
    phase3_success = phase3_error < results['success_criteria']['phase3_retention']
    
    results['phases']['phase3'] = {
        'test_error': float(phase3_error),
        'original_error': float(phase1_error),
        'degradation_ratio': float(retention_ratio),
        'phi_frequency': model.get_phi_frequency(),
        'success': bool(phase3_success)
    }
    
    print(f"    Retention Error: {phase3_error:.6f} (was {phase1_error:.6f})")
    print(f"    Degradation: {retention_ratio:.1f}x ({'✓' if phase3_success else '✗'})")
    
    # Overall
    overall_success = phase1_success and phase2_success and phase3_success
    results['overall_success'] = bool(overall_success)
    results['interpretation'] = (
        'Convergence maintained' if overall_success
        else 'Catastrophic forgetting' if not phase3_success
        else 'Learning failure'
    )
    
    return results


def experiment_convergence_maintenance():
    """
    Compare convergence maintenance: baseline vs anchor memory.
    """
    print("=" * 70)
    print("Experiment 24: Convergence Maintenance in TinyCIMM-Möbius")
    print("=" * 70)
    
    # Generate Fibonacci sequence
    fibs = [1, 1]
    for _ in range(30):
        fibs.append(fibs[-1] + fibs[-2])
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'hypothesis': 'PhiAnchorMemory prevents catastrophic forgetting in Möbius networks',
        'experiments': {}
    }
    
    # Test 1: Baseline (no anchor memory)
    print("\n" + "=" * 50)
    print("TEST 1: Baseline (No Anchor Memory)")
    print("=" * 50)
    
    model_baseline = TinyCIMMMobius(
        input_size=1, 
        hidden_layers=3, 
        output_size=1,
        continuous_lr=0.005,
        init='fibonacci',
        use_anchor_memory=False
    )
    
    results_baseline = run_convergence_test(model_baseline, fibs, 'baseline')
    all_results['experiments']['baseline'] = results_baseline
    
    # Test 2: With Anchor Memory
    print("\n" + "=" * 50)
    print("TEST 2: With PhiAnchorMemory")
    print("=" * 50)
    
    model_anchored = TinyCIMMMobius(
        input_size=1, 
        hidden_layers=3, 
        output_size=1,
        continuous_lr=0.005,
        init='fibonacci',
        use_anchor_memory=True,
        anchor_capacity=5,
        anchor_penalty=0.2  # Stronger regularization
    )
    
    results_anchored = run_convergence_test(model_anchored, fibs, 'anchored')
    all_results['experiments']['anchored'] = results_anchored
    
    # Test 3: Strong Anchor Memory
    print("\n" + "=" * 50)
    print("TEST 3: Strong PhiAnchorMemory (penalty=0.5)")
    print("=" * 50)
    
    model_strong = TinyCIMMMobius(
        input_size=1, 
        hidden_layers=3, 
        output_size=1,
        continuous_lr=0.005,
        init='fibonacci',
        use_anchor_memory=True,
        anchor_capacity=5,
        anchor_penalty=0.5  # Very strong regularization
    )
    
    results_strong = run_convergence_test(model_strong, fibs, 'strong_anchor')
    all_results['experiments']['strong_anchor'] = results_strong
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("SUMMARY: Convergence Maintenance Comparison")
    print("=" * 70)
    
    for name, result in all_results['experiments'].items():
        p1 = '✓' if result['phases']['phase1']['success'] else '✗'
        p2 = '✓' if result['phases']['phase2']['success'] else '✗'
        p3 = '✓' if result['phases']['phase3']['success'] else '✗'
        deg = result['phases']['phase3']['degradation_ratio']
        
        print(f"\n  {name}:")
        print(f"    Phase 1 (Training):  {p1}")
        print(f"    Phase 2 (Continue):  {p2}")
        print(f"    Phase 3 (Retention): {p3}  (degradation: {deg:.1f}x)")
        print(f"    Verdict: {result['interpretation']}")
    
    # Compute improvement
    baseline_deg = all_results['experiments']['baseline']['phases']['phase3']['degradation_ratio']
    anchored_deg = all_results['experiments']['anchored']['phases']['phase3']['degradation_ratio']
    improvement = baseline_deg / max(anchored_deg, 0.01)
    
    all_results['comparison'] = {
        'baseline_degradation': float(baseline_deg),
        'anchored_degradation': float(anchored_deg),
        'improvement_factor': float(improvement),
        'anchor_memory_helps': anchored_deg < baseline_deg
    }
    
    print(f"\n  Improvement with Anchor Memory: {improvement:.1f}x less degradation")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f'convergence_comparison_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return all_results


if __name__ == '__main__':
    results = experiment_convergence_maintenance()
    
    print("\n" + "=" * 70)
    print("Experiment Complete")
    print("=" * 70)
