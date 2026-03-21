"""
Experiment 02: SEC Phase Transition Monitoring
==============================================

Test whether the SEC phase transition at φ × ξ = 1.710 
predicts semantic structure formation.

Hypothesis:
When field entropy crosses the crystallization threshold,
semantic bonds strengthen at an accelerated rate.

Protocol:
1. Compare training WITH and WITHOUT phase monitoring
2. Measure if transitions correlate with semantic improvement
3. Test if triggering crystallization manually improves learning
"""

import torch
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from physics_trainer import (
    DawnFieldTrainer,
    ResonanceTrainer,
    PhaseTransitionMonitor,
    PHI_XI,
    PHI,
    LAMBDA_STAR,
    get_device,
)


def test_phase_monitor_detection():
    """Test that phase monitor correctly identifies high-entropy states."""
    print("\n[Test 1: Phase Monitor Detection]")
    
    device = get_device()
    monitor = PhaseTransitionMonitor(device=device)
    
    # Create fields with varying entropy
    low_entropy = torch.zeros(64, device=device)
    low_entropy[0] = 1.0  # All mass in one place
    
    high_entropy = torch.ones(64, device=device) / 64  # Uniform
    
    # Check detections
    is_low, metric_low = monitor.check_transition(low_entropy, 0)
    is_high, metric_high = monitor.check_transition(high_entropy, 1)
    
    print(f"  Low entropy field: metric={metric_low:.4f}, transition={is_low}")
    print(f"  High entropy field: metric={metric_high:.4f}, transition={is_high}")
    print(f"  Threshold (φ×ξ): {PHI_XI:.4f}")
    
    # High entropy should have higher metric
    assert metric_high > metric_low, "High entropy should have higher metric"
    
    print(f"  ✓ Phase detection working correctly")
    return {'low': metric_low, 'high': metric_high, 'threshold': PHI_XI}


def test_training_with_vs_without_phase():
    """Compare training with and without phase transition monitoring."""
    print("\n[Test 2: Training With vs Without Phase Monitoring]")
    
    training_data = [
        ("alpha", "beta"),
        ("alpha", "gamma"),
        ("beta", "gamma"),
        ("one", "two"),
        ("one", "three"),
        ("two", "three"),
    ]
    
    # Train WITHOUT phase monitoring
    trainer_no_phase = ResonanceTrainer(use_phase_monitor=False, use_fibonacci_lr=True)
    
    stats_no_phase = trainer_no_phase.train_cooccurrence(
        training_data, epochs=20, verbose=False
    )
    
    sim_same_no_phase = trainer_no_phase.similarity("alpha", "beta")
    sim_diff_no_phase = trainer_no_phase.similarity("alpha", "one")
    
    # Train WITH phase monitoring
    trainer_with_phase = ResonanceTrainer(use_phase_monitor=True, use_fibonacci_lr=True)
    
    stats_with_phase = trainer_with_phase.train_cooccurrence(
        training_data, epochs=20, verbose=False
    )
    
    sim_same_with_phase = trainer_with_phase.similarity("alpha", "beta")
    sim_diff_with_phase = trainer_with_phase.similarity("alpha", "one")
    
    sep_no_phase = sim_same_no_phase - sim_diff_no_phase
    sep_with_phase = sim_same_with_phase - sim_diff_with_phase
    
    print(f"  WITHOUT phase monitor:")
    print(f"    sim(alpha,beta)={sim_same_no_phase:.3f}, sim(alpha,one)={sim_diff_no_phase:.3f}")
    print(f"    Separation: {sep_no_phase:.3f}")
    
    print(f"  WITH phase monitor:")
    print(f"    sim(alpha,beta)={sim_same_with_phase:.3f}, sim(alpha,one)={sim_diff_with_phase:.3f}")  
    print(f"    Separation: {sep_with_phase:.3f}")
    print(f"    Transitions: {stats_with_phase['transitions']}")
    
    # Both should show good separation
    assert sep_no_phase > 0.3, "No-phase should still learn"
    assert sep_with_phase > 0.3, "With-phase should learn"
    
    improvement = sep_with_phase - sep_no_phase
    print(f"  ✓ Phase monitoring {'improved' if improvement > 0 else 'similar to'} training")
    
    return {
        'no_phase': {'sep': sep_no_phase, 'same': sim_same_no_phase, 'diff': sim_diff_no_phase},
        'with_phase': {'sep': sep_with_phase, 'same': sim_same_with_phase, 'diff': sim_diff_with_phase},
        'improvement': improvement,
    }


def test_crystallization_at_threshold():
    """Test if semantic bonds crystallize at the φ × ξ threshold."""
    print("\n[Test 3: Crystallization at Threshold]")
    
    device = get_device()
    trainer = DawnFieldTrainer(device=device)
    
    # Train and track similarity after each transition
    training_data = [("x", "y"), ("y", "z"), ("x", "z")]
    
    similarity_at_transitions = []
    similarity_between = []
    
    # Custom training to track transitions
    for epoch in range(30):
        trainer.train(training_data, epochs=1)
        
        current_sim = trainer.similarity("x", "y")
        
        phase_stats = trainer.resonance.phase_monitor.get_stats()
        
        if epoch > 0:
            prev_transitions = similarity_at_transitions[-1][1] if similarity_at_transitions else 0
            current_transitions = phase_stats['transitions']
            
            if current_transitions > prev_transitions:
                similarity_at_transitions.append((current_sim, current_transitions))
            else:
                similarity_between.append(current_sim)
    
    # Analyze
    if similarity_at_transitions:
        avg_at_transition = sum(s for s, _ in similarity_at_transitions) / len(similarity_at_transitions)
    else:
        avg_at_transition = 0
        
    if similarity_between:
        avg_between = sum(similarity_between) / len(similarity_between)
    else:
        avg_between = 0
    
    print(f"  Transitions detected: {len(similarity_at_transitions)}")
    print(f"  Avg similarity at transitions: {avg_at_transition:.3f}")
    print(f"  Avg similarity between transitions: {avg_between:.3f}")
    
    # Both should be high due to training
    print(f"  ✓ Crystallization monitoring complete")
    
    return {
        'transitions_count': len(similarity_at_transitions),
        'avg_at_transition': avg_at_transition,
        'avg_between': avg_between,
    }


def test_lambda_decay():
    """Test optimal memory decay at λ* = 0.9816."""
    print("\n[Test 4: Lambda* Memory Decay]")
    
    device = get_device()
    
    # Test different decay rates
    decay_rates = [0.5, 0.8, LAMBDA_STAR, 0.99, 1.0]
    results = {}
    
    for decay in decay_rates:
        # Create field with initial pattern
        field = torch.ones(64, device=device)
        
        # Apply decay over steps
        for step in range(100):
            field = field * decay
            
        final_energy = torch.sum(field ** 2).item()
        
        results[decay] = {
            'final_energy': final_energy,
            'survived': final_energy > 1e-10,
        }
    
    print(f"  Decay rates tested: {decay_rates}")
    print(f"  λ* = {LAMBDA_STAR} (optimal from SEC)")
    
    # λ* should give good balance
    lambda_result = results[LAMBDA_STAR]
    print(f"  At λ*: final_energy={lambda_result['final_energy']:.6f}")
    
    print(f"  ✓ Memory decay tested at λ*")
    return results


def test_golden_ratio_emergence():
    """Test if φ appears in training dynamics."""
    print("\n[Test 5: Golden Ratio Emergence]")
    
    trainer = DawnFieldTrainer()
    
    # Train and track ratio of consecutive similarities
    training_data = [("a", "b"), ("c", "d"), ("a", "c")]
    
    similarities = []
    for epoch in range(20):
        trainer.train(training_data, epochs=1)
        sim = trainer.similarity("a", "b")
        similarities.append(sim)
    
    # Look for φ-like ratios
    ratios = []
    for i in range(1, len(similarities)):
        if similarities[i-1] > 1e-6:
            ratio = similarities[i] / similarities[i-1]
            ratios.append(ratio)
    
    # Filter valid ratios
    valid_ratios = [r for r in ratios if 0.1 < r < 10]
    
    if valid_ratios:
        avg_ratio = sum(valid_ratios) / len(valid_ratios)
        
        # Check how close to φ
        phi_distance = abs(avg_ratio - PHI)
        
        print(f"  Average consecutive ratio: {avg_ratio:.4f}")
        print(f"  Golden ratio φ: {PHI:.4f}")
        print(f"  Distance from φ: {phi_distance:.4f}")
    else:
        print(f"  No valid ratios found")
        avg_ratio = 0
        phi_distance = float('inf')
    
    print(f"  ✓ Ratio analysis complete")
    return {'avg_ratio': avg_ratio, 'phi': PHI, 'distance': phi_distance}


def test_phase_transition_count_scaling():
    """Test how phase transitions scale with training complexity."""
    print("\n[Test 6: Transition Count Scaling]")
    
    results = {}
    
    for num_patterns in [2, 4, 8, 16]:
        trainer = DawnFieldTrainer()
        
        # Generate training data
        training_data = [(f"p{i}", f"p{i+1}") for i in range(num_patterns)]
        
        stats = trainer.train(training_data, epochs=10)
        
        phase_stats = stats.get('phase_stats', {})
        transitions = phase_stats.get('transitions', 0)
        
        results[num_patterns] = {
            'transitions': transitions,
            'steps': phase_stats.get('total_steps', 0),
            'ratio': transitions / phase_stats.get('total_steps', 1) if phase_stats.get('total_steps', 0) > 0 else 0,
        }
    
    print(f"  Pattern count → Transitions:")
    for n, r in results.items():
        print(f"    {n} patterns: {r['transitions']} transitions ({r['ratio']:.1%})")
    
    print(f"  ✓ Scaling analysis complete")
    return results


def run_all_experiments():
    """Run all SEC experiments."""
    print("=" * 60)
    print("POC-002 Experiment 02: SEC Phase Transition")
    print("=" * 60)
    print(f"Device: {get_device()}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"φ × ξ = {PHI_XI:.4f}")
    
    results = {
        'experiment': 'poc_002_exp_02_phase_transition',
        'timestamp': datetime.now().isoformat(),
        'device': str(get_device()),
        'constants': {
            'phi_xi': PHI_XI,
            'phi': PHI,
            'lambda_star': LAMBDA_STAR,
        },
        'tests': {},
    }
    
    tests = [
        ('phase_detection', test_phase_monitor_detection),
        ('with_vs_without', test_training_with_vs_without_phase),
        ('crystallization', test_crystallization_at_threshold),
        ('lambda_decay', test_lambda_decay),
        ('golden_ratio', test_golden_ratio_emergence),
        ('scaling', test_phase_transition_count_scaling),
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
    
    filename = f"exp_02_phase_transition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, float):
            if obj != obj:  # NaN check
                return None
            return round(obj, 6) if abs(obj) < 1e10 else str(obj)
        else:
            return obj
    
    with open(results_dir / filename, 'w') as f:
        json.dump(clean_for_json(results), f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    return results


if __name__ == "__main__":
    run_all_experiments()
