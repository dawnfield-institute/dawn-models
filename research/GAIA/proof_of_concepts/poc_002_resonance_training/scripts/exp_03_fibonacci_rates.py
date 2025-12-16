"""
Experiment 03: Fibonacci Learning Rates
=======================================

Test whether Fibonacci-governed learning rates (1/F_n)
outperform fixed learning rates for resonance training.

From PAC Confluence Xi:
- Learning rate = 1/F_n based on pattern complexity
- Natural gauge hierarchy for stable convergence
- Fibonacci sequence mirrors growth patterns in fields

Protocol:
1. Compare fixed lr vs Fibonacci lr
2. Test convergence speed and stability
3. Measure final semantic separation quality
"""

import torch
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import time

from physics_trainer import (
    ResonanceTrainer,
    FibonacciScheduler,
    FIBONACCI,
    PHI_XI,
    get_device,
)


def test_fibonacci_scheduler():
    """Test Fibonacci scheduler produces correct rates."""
    print("\n[Test 1: Fibonacci Scheduler]")
    
    scheduler = FibonacciScheduler(base_lr=0.1)
    
    rates = []
    for i in range(50):
        lr = scheduler.get_lr()
        rates.append(lr)
        scheduler.step()
    
    # Check that rates decrease
    unique_rates = sorted(set(rates), reverse=True)
    
    print(f"  Fibonacci sequence: {FIBONACCI[:8]}...")
    print(f"  Learning rates: {unique_rates[:5]}...")
    print(f"  Rate at step 0: {rates[0]:.4f}")
    print(f"  Rate at step 49: {rates[49]:.4f}")
    
    # Rates should decrease over time
    assert rates[0] >= rates[49], "Rates should decrease"
    
    print(f"  ✓ Fibonacci scheduler working")
    return {'rates': unique_rates, 'start': rates[0], 'end': rates[49]}


def test_complexity_based_lr():
    """Test learning rate based on pattern complexity."""
    print("\n[Test 2: Complexity-Based LR]")
    
    scheduler = FibonacciScheduler(base_lr=0.1)
    
    # Test different complexities
    complexities = [1, 3, 5, 8, 10]
    rates = {}
    
    for c in complexities:
        lr = scheduler.get_lr(complexity=c)
        rates[c] = lr
    
    print(f"  Complexity → Learning Rate:")
    for c, lr in rates.items():
        fib = FIBONACCI[min(c, len(FIBONACCI)-1)]
        print(f"    {c} → {lr:.6f} (1/F_{c} = 1/{fib})")
    
    # Higher complexity = lower lr
    assert rates[1] > rates[10], "Higher complexity should have lower lr"
    
    print(f"  ✓ Complexity-based rates working")
    return rates


def test_fixed_vs_fibonacci():
    """Compare training with fixed vs Fibonacci learning rates."""
    print("\n[Test 3: Fixed vs Fibonacci Learning Rates]")
    
    # Same training data for both
    training_data = [
        ("cat", "dog"),
        ("cat", "animal"),
        ("dog", "animal"),
        ("red", "blue"),
        ("red", "color"),
        ("blue", "color"),
    ]
    
    # Fixed learning rate trainer
    trainer_fixed = ResonanceTrainer(use_fibonacci_lr=False, use_phase_monitor=False)
    
    # Fibonacci learning rate trainer
    trainer_fib = ResonanceTrainer(use_fibonacci_lr=True, use_phase_monitor=False)
    
    epochs = 30
    
    # Track similarity over time
    fixed_history = []
    fib_history = []
    
    for epoch in range(epochs):
        # Train one epoch
        trainer_fixed.train_cooccurrence(training_data, epochs=1)
        trainer_fib.train_cooccurrence(training_data, epochs=1)
        
        # Record similarity
        fixed_sim = trainer_fixed.similarity("cat", "dog")
        fib_sim = trainer_fib.similarity("cat", "dog")
        
        fixed_history.append(fixed_sim)
        fib_history.append(fib_sim)
    
    # Analyze
    fixed_final = fixed_history[-1]
    fib_final = fib_history[-1]
    
    fixed_stability = torch.var(torch.tensor(fixed_history[-10:])).item()
    fib_stability = torch.var(torch.tensor(fib_history[-10:])).item()
    
    print(f"  Fixed LR: final_sim={fixed_final:.3f}, variance={fixed_stability:.6f}")
    print(f"  Fibonacci LR: final_sim={fib_final:.3f}, variance={fib_stability:.6f}")
    
    # Calculate convergence speed (epochs to reach 0.8)
    fixed_conv = next((i for i, s in enumerate(fixed_history) if s >= 0.8), epochs)
    fib_conv = next((i for i, s in enumerate(fib_history) if s >= 0.8), epochs)
    
    print(f"  Epochs to 0.8 similarity: fixed={fixed_conv}, fib={fib_conv}")
    
    print(f"  ✓ Comparison complete")
    return {
        'fixed': {'final': fixed_final, 'variance': fixed_stability, 'conv_epochs': fixed_conv},
        'fibonacci': {'final': fib_final, 'variance': fib_stability, 'conv_epochs': fib_conv},
    }


def test_semantic_quality():
    """Test final semantic quality with Fibonacci vs fixed rates."""
    print("\n[Test 4: Semantic Quality]")
    
    # More complex training data
    animals = ["cat", "dog", "bird", "fish", "lion"]
    colors = ["red", "blue", "green", "yellow", "orange"]
    numbers = ["one", "two", "three", "four", "five"]
    
    training_data = []
    for group in [animals, colors, numbers]:
        for i, item1 in enumerate(group):
            for item2 in group[i+1:]:
                training_data.append((item1, item2))
    
    # Train with both methods
    trainer_fixed = ResonanceTrainer(use_fibonacci_lr=False, use_phase_monitor=True)
    trainer_fib = ResonanceTrainer(use_fibonacci_lr=True, use_phase_monitor=True)
    
    trainer_fixed.train_cooccurrence(training_data, epochs=30)
    trainer_fib.train_cooccurrence(training_data, epochs=30)
    
    # Measure within vs between class similarity
    def measure_separation(trainer):
        within = []
        between = []
        
        for item1 in animals[:3]:
            for item2 in animals[:3]:
                if item1 != item2:
                    within.append(trainer.similarity(item1, item2))
            for item2 in colors[:3]:
                between.append(trainer.similarity(item1, item2))
        
        return {
            'within': sum(within) / len(within),
            'between': sum(between) / len(between),
            'separation': sum(within) / len(within) - sum(between) / len(between),
        }
    
    fixed_sep = measure_separation(trainer_fixed)
    fib_sep = measure_separation(trainer_fib)
    
    print(f"  Fixed LR: within={fixed_sep['within']:.3f}, between={fixed_sep['between']:.3f}, sep={fixed_sep['separation']:.3f}")
    print(f"  Fibonacci LR: within={fib_sep['within']:.3f}, between={fib_sep['between']:.3f}, sep={fib_sep['separation']:.3f}")
    
    # Both should achieve good separation
    assert fixed_sep['separation'] > 0.3, "Fixed should achieve separation"
    assert fib_sep['separation'] > 0.3, "Fibonacci should achieve separation"
    
    print(f"  ✓ Both methods achieve good semantic quality")
    return {'fixed': fixed_sep, 'fibonacci': fib_sep}


def test_training_speed():
    """Measure training time with different lr schedules."""
    print("\n[Test 5: Training Speed]")
    
    training_data = [(f"p{i}", f"p{i+1}") for i in range(20)]
    epochs = 50
    
    # Fixed lr timing
    trainer_fixed = ResonanceTrainer(use_fibonacci_lr=False, use_phase_monitor=False)
    start = time.perf_counter()
    trainer_fixed.train_cooccurrence(training_data, epochs=epochs)
    fixed_time = time.perf_counter() - start
    
    # Fibonacci lr timing
    trainer_fib = ResonanceTrainer(use_fibonacci_lr=True, use_phase_monitor=False)
    start = time.perf_counter()
    trainer_fib.train_cooccurrence(training_data, epochs=epochs)
    fib_time = time.perf_counter() - start
    
    print(f"  Fixed LR time: {fixed_time*1000:.2f}ms")
    print(f"  Fibonacci LR time: {fib_time*1000:.2f}ms")
    print(f"  Overhead: {(fib_time - fixed_time) / fixed_time * 100:.1f}%")
    
    # Fibonacci shouldn't be much slower
    overhead = (fib_time - fixed_time) / fixed_time
    assert overhead < 1.0, "Fibonacci overhead should be < 100%"
    
    print(f"  ✓ Fibonacci LR adds minimal overhead")
    return {'fixed_ms': fixed_time * 1000, 'fib_ms': fib_time * 1000, 'overhead_pct': overhead * 100}


def test_conservation_with_fibonacci():
    """Test PAC conservation is maintained with Fibonacci rates."""
    print("\n[Test 6: Conservation with Fibonacci]")
    
    trainer = ResonanceTrainer(use_fibonacci_lr=True, use_phase_monitor=True)
    
    training_data = [(f"x{i}", f"y{i}") for i in range(30)]
    
    trainer.train_cooccurrence(training_data, epochs=100)
    
    residual = trainer.conservation_check()
    
    print(f"  Conservation residual: {residual:.2e}")
    
    assert residual < 1e-3, f"Conservation violated: {residual}"
    
    print(f"  ✓ PAC conservation maintained with Fibonacci LR")
    return {'residual': residual}


def run_all_experiments():
    """Run all Fibonacci lr experiments."""
    print("=" * 60)
    print("POC-002 Experiment 03: Fibonacci Learning Rates")
    print("=" * 60)
    print(f"Device: {get_device()}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Fibonacci: {FIBONACCI[:10]}...")
    
    results = {
        'experiment': 'poc_002_exp_03_fibonacci_rates',
        'timestamp': datetime.now().isoformat(),
        'device': str(get_device()),
        'fibonacci_sequence': FIBONACCI,
        'tests': {},
    }
    
    tests = [
        ('scheduler', test_fibonacci_scheduler),
        ('complexity_lr', test_complexity_based_lr),
        ('fixed_vs_fib', test_fixed_vs_fibonacci),
        ('semantic_quality', test_semantic_quality),
        ('training_speed', test_training_speed),
        ('conservation', test_conservation_with_fibonacci),
    ]
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            results['tests'][name] = result if isinstance(result, dict) else {'result': result}
            results['tests'][name]['passed'] = True
        except AssertionError as e:
            results['tests'][name] = {'passed': False, 'error': str(e)}
        except Exception as e:
            results['tests'][name] = {'passed': False, 'error': str(e)}
    
    # Summary
    passed = sum(1 for t in results['tests'].values() if t.get('passed', False))
    total = len(results['tests'])
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    results['summary'] = {
        'passed': passed,
        'total': total,
        'success': passed == total,
    }
    
    # Save
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    filename = f"exp_03_fibonacci_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, float):
            return round(obj, 6) if abs(obj) < 1e10 else str(obj)
        else:
            return obj
    
    with open(results_dir / filename, 'w') as f:
        json.dump(clean_for_json(results), f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    return results


if __name__ == "__main__":
    run_all_experiments()
