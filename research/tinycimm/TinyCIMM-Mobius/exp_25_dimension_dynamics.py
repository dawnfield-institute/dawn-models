"""
Experiment 25: MobiusStrandField Dimension Dynamics

Tests the core prediction of Dawn Field Theory:
- φ-structured patterns collapse effective dimension (strands lock)
- Non-φ patterns require more dimensions (strands independent)

This validates the idea that "2D Möbius topology generates apparent 3D+1"
through strand phase relationships.

Key insight: Each "strand" is an independent Möbius transformation.
The interactions between strands create emergent dimensionality.
When strands lock into φ-harmonic relationships, dimension collapses.
"""

import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
from mobius_strand_field import MobiusStrandField, TinyCIMMMobiusField, PHI

RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def experiment_dimension_dynamics():
    """
    Compare dimension evolution across different pattern types.
    
    Patterns tested:
    1. Fibonacci ratios (strong φ-structure)
    2. Golden spiral (weak φ-structure)
    3. Polynomial (no φ-structure)
    4. Random noise (maximum entropy)
    
    Prediction:
    - Fibonacci: Maximum collapse
    - Golden spiral: Partial collapse
    - Polynomial: Dimension grows
    - Random: Maximum dimension
    """
    print("=" * 70)
    print("Experiment 25: MobiusStrandField Dimension Dynamics")
    print("=" * 70)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'hypothesis': 'φ-patterns collapse dimension, non-φ patterns grow dimension',
        'experiments': {}
    }
    
    # Generate data streams
    fibs = [1, 1]
    for _ in range(30):
        fibs.append(fibs[-1] + fibs[-2])
    
    def fibonacci_stream(n):
        """Strong φ-structure: consecutive Fibonacci ratios."""
        for _ in range(n):
            idx = np.random.randint(5, 25)
            x = np.array([fibs[idx] / fibs[idx-1]])
            y = np.array([fibs[idx+1] / fibs[idx]])
            yield x, y
    
    def golden_spiral_stream(n):
        """Weak φ-structure: points on golden spiral."""
        for i in range(n):
            theta = i * 0.1
            r = PHI ** (theta / (2 * np.pi))
            x = np.array([r * np.cos(theta)])
            y = np.array([r * np.sin(theta)])
            yield x, y
    
    def polynomial_stream(n):
        """No φ-structure: quadratic function."""
        for _ in range(n):
            x_val = np.random.uniform(0.5, 2.0)
            x = np.array([x_val])
            y = np.array([0.5 * x_val**2 + 0.3 * x_val])
            yield x, y
    
    def random_stream(n):
        """Maximum entropy: pure noise."""
        for _ in range(n):
            x = np.array([np.random.randn()])
            y = np.array([np.random.randn()])
            yield x, y
    
    patterns = [
        ('fibonacci', fibonacci_stream, 'Strong φ-structure'),
        ('golden_spiral', golden_spiral_stream, 'Weak φ-structure'),
        ('polynomial', polynomial_stream, 'No φ-structure'),
        ('random', random_stream, 'Maximum entropy')
    ]
    
    for name, stream_fn, description in patterns:
        print(f"\n{'='*50}")
        print(f"Pattern: {name} ({description})")
        print('='*50)
        
        model = TinyCIMMMobiusField(n_strands=4, init='random')
        
        initial_state = model.field.get_field_state()
        print(f"Initial: dim={initial_state.effective_dimension:.3f}")
        
        history = model.continuous_train(
            stream_fn(400), 
            max_steps=400, 
            log_interval=200
        )
        
        final_state = model.field.get_field_state()
        
        dims = [h['effective_dim'] for h in history]
        dim_change = dims[-1] - dims[0]
        dim_ratio = dims[-1] / max(dims[0], 0.01)
        
        results['experiments'][name] = {
            'description': description,
            'initial_dim': float(dims[0]),
            'final_dim': float(dims[-1]),
            'dim_change': float(dim_change),
            'dim_ratio': float(dim_ratio),
            'coherence': float(final_state.coherence),
            'chord': final_state.chord_type,
            'collapsed': dims[-1] < dims[0] * 0.9,
            'grew': dims[-1] > dims[0] * 1.1
        }
        
        print(f"Final: dim={dims[-1]:.3f} (change: {dim_change:+.3f})")
        print(f"Behavior: {'COLLAPSED' if dim_change < -0.05 else 'GREW' if dim_change > 0.05 else 'STABLE'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Dimension Dynamics by Pattern Type")
    print("=" * 70)
    
    print(f"\n{'Pattern':<20} {'Initial':<10} {'Final':<10} {'Change':<10} {'Behavior':<15}")
    print("-" * 65)
    
    for name, data in results['experiments'].items():
        behavior = 'COLLAPSED' if data['collapsed'] else 'GREW' if data['grew'] else 'STABLE'
        print(f"{name:<20} {data['initial_dim']:<10.3f} {data['final_dim']:<10.3f} "
              f"{data['dim_change']:+<10.3f} {behavior:<15}")
    
    # Validate prediction
    fib_collapsed = results['experiments']['fibonacci']['collapsed']
    poly_grew = results['experiments']['polynomial']['grew'] or not results['experiments']['polynomial']['collapsed']
    
    results['prediction_validated'] = fib_collapsed and poly_grew
    
    if results['prediction_validated']:
        print("\n✓ PREDICTION VALIDATED: φ-patterns collapse, non-φ patterns don't")
    else:
        print("\n✗ PREDICTION NOT VALIDATED")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f'dimension_dynamics_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return results


def experiment_strand_count_scaling():
    """
    How does dimension collapse scale with number of strands?
    
    More strands = more potential dimensions
    But φ-patterns should collapse regardless of strand count
    """
    print("\n" + "=" * 70)
    print("Experiment: Strand Count Scaling")
    print("=" * 70)
    
    fibs = [1, 1]
    for _ in range(30):
        fibs.append(fibs[-1] + fibs[-2])
    
    def fib_stream(n):
        for _ in range(n):
            idx = np.random.randint(5, 25)
            x = np.array([fibs[idx] / fibs[idx-1]])
            y = np.array([fibs[idx+1] / fibs[idx]])
            yield x, y
    
    strand_counts = [2, 4, 8, 16]
    results = {}
    
    for n_strands in strand_counts:
        print(f"\n--- {n_strands} Strands ---")
        
        model = TinyCIMMMobiusField(n_strands=n_strands, init='random')
        
        history = model.continuous_train(
            fib_stream(300),
            max_steps=300,
            log_interval=300  # Only show start/end
        )
        
        dims = [h['effective_dim'] for h in history]
        collapse_ratio = dims[-1] / max(dims[0], 0.01)
        
        results[n_strands] = {
            'initial_dim': dims[0],
            'final_dim': dims[-1],
            'collapse_ratio': collapse_ratio
        }
        
        print(f"  Dimension: {dims[0]:.3f} → {dims[-1]:.3f} (ratio: {collapse_ratio:.2f})")
    
    print("\n--- Scaling Analysis ---")
    for n, data in results.items():
        print(f"  {n} strands: {data['collapse_ratio']:.2f}x collapse")
    
    return results


if __name__ == '__main__':
    # Main experiment
    dim_results = experiment_dimension_dynamics()
    
    # Scaling experiment
    scaling_results = experiment_strand_count_scaling()
    
    print("\n" + "=" * 70)
    print("Experiment Complete")
    print("=" * 70)
