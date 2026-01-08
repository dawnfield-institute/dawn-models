"""
Experiment 27: Prime Gap Continuous Learning

Can a MobiusStrandField lock onto prime gap structure?

Hypothesis:
If prime gaps contain φ-structure (as suggested by SEC Prime Manifold),
then the strand field should:
1. Collapse dimension (find low-dimensional attractor)
2. Lock into φ-resonant fixed points
3. Predict gaps better than random baseline

This is a direct test of whether primes "know about" φ.

Patterns tested:
1. Prime gaps: g_n = p_{n+1} - p_n
2. Gap ratios: g_n / g_{n-1} (like Fibonacci ratios)
3. Normalized gaps: g_n / log(p_n) (prime number theorem baseline)
4. Control: Random gaps (shuffled primes)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

from mobius_strand_field import MobiusStrandField, TinyCIMMMobiusField, PHI

RESULTS_DIR = Path(__file__).parent / 'results'
FIGURES_DIR = Path(__file__).parent / 'figures'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def sieve_of_eratosthenes(limit):
    """Generate primes up to limit."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(limit + 1) if is_prime[i]]


def compute_prime_gaps(primes):
    """Compute gaps between consecutive primes."""
    return [primes[i+1] - primes[i] for i in range(len(primes) - 1)]


def experiment_prime_gap_locking():
    """
    Main experiment: Can strand field lock onto prime gap structure?
    """
    print("=" * 70)
    print("Experiment 27: Prime Gap Continuous Learning")
    print("=" * 70)
    
    # Generate primes
    primes = sieve_of_eratosthenes(100000)
    gaps = compute_prime_gaps(primes)
    
    print(f"Generated {len(primes)} primes, {len(gaps)} gaps")
    print(f"Gap statistics: mean={np.mean(gaps):.2f}, std={np.std(gaps):.2f}")
    print(f"Max gap: {max(gaps)}, Min gap: {min(gaps)}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_primes': len(primes),
        'experiments': {}
    }
    
    # Data streams
    def gap_stream(gaps_list):
        """Stream of (current_gap, next_gap) pairs."""
        n = len(gaps_list)
        while True:
            i = np.random.randint(0, n - 1)
            x = np.array([gaps_list[i] / 10.0])  # Normalize
            y = np.array([gaps_list[i + 1] / 10.0])
            yield x, y
    
    def gap_ratio_stream(gaps_list):
        """Stream of gap ratios (like Fibonacci ratios)."""
        n = len(gaps_list)
        while True:
            i = np.random.randint(1, n - 1)
            if gaps_list[i-1] > 0:
                x = np.array([gaps_list[i] / gaps_list[i-1]])
                y = np.array([gaps_list[i+1] / max(gaps_list[i], 1)])
                yield x, y
    
    def normalized_gap_stream(primes_list, gaps_list):
        """Stream of gaps normalized by log(p) - tests departure from PNT."""
        n = len(gaps_list)
        while True:
            i = np.random.randint(1, n - 1)
            log_p = np.log(primes_list[i])
            x = np.array([gaps_list[i] / log_p])
            y = np.array([gaps_list[i+1] / np.log(primes_list[i+1])])
            yield x, y
    
    def random_gaps_stream(gaps_list):
        """Control: Shuffled gaps (destroys structure)."""
        shuffled = gaps_list.copy()
        np.random.shuffle(shuffled)
        return gap_stream(shuffled)
    
    # Fibonacci stream for comparison
    fibs = [1, 1]
    for _ in range(50):
        fibs.append(fibs[-1] + fibs[-2])
    
    def fib_ratio_stream():
        while True:
            i = np.random.randint(5, 40)
            x = np.array([fibs[i] / fibs[i-1]])
            y = np.array([fibs[i+1] / fibs[i]])
            yield x, y
    
    patterns = [
        ('prime_gaps', lambda: gap_stream(gaps), 'Prime gaps g_n → g_{n+1}'),
        ('gap_ratios', lambda: gap_ratio_stream(gaps), 'Gap ratios (like Fib ratios)'),
        ('normalized_gaps', lambda: normalized_gap_stream(primes, gaps), 'Gaps / log(p)'),
        ('random_gaps', lambda: random_gaps_stream(gaps), 'Shuffled gaps (control)'),
        ('fibonacci', fib_ratio_stream, 'Fibonacci ratios (reference)'),
    ]
    
    n_strands = 6
    n_steps = 500
    
    all_histories = {}
    
    for name, stream_fn, description in patterns:
        print(f"\n{'='*50}")
        print(f"Pattern: {name}")
        print(f"Description: {description}")
        print('='*50)
        
        model = TinyCIMMMobiusField(n_strands=n_strands, init='random')
        
        dim_history = []
        freq_history = []
        loss_history = []
        coherence_history = []
        
        stream = stream_fn()
        
        for step in range(n_steps):
            x, y = next(stream)
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
            
            metrics = model.continuous_step(x, y)
            
            dim_history.append(metrics['effective_dim'])
            freq_history.append(model.field.get_frequency_spectrum().detach().numpy())
            loss_history.append(metrics['loss'])
            coherence_history.append(metrics['coherence'])
            
            if step % 100 == 0:
                print(f"Step {step}: dim={metrics['effective_dim']:.3f}, "
                      f"loss={metrics['loss']:.4f}, chord={metrics['chord']}")
        
        freq_history = np.array(freq_history)
        
        # Analysis
        dim_start = dim_history[0]
        dim_end = dim_history[-1]
        dim_change = dim_end - dim_start
        
        avg_freq = np.mean(freq_history[-100:])
        final_coherence = coherence_history[-1]
        
        # Check for locking (stable dimension + high coherence)
        dim_stable = np.std(dim_history[-100:]) < 0.1
        locked = dim_stable and final_coherence > 0.4
        
        state = model.field.get_field_state()
        
        results['experiments'][name] = {
            'description': description,
            'dim_start': float(dim_start),
            'dim_end': float(dim_end),
            'dim_change': float(dim_change),
            'final_loss': float(loss_history[-1]),
            'avg_loss': float(np.mean(loss_history[-100:])),
            'avg_freq': float(avg_freq),
            'coherence': float(final_coherence),
            'chord': state.chord_type,
            'locked': locked,
            'collapsed': dim_end < dim_start * 0.8
        }
        
        all_histories[name] = {
            'dim': dim_history,
            'freq': freq_history,
            'loss': loss_history,
            'coherence': coherence_history
        }
        
        print(f"\nResults:")
        print(f"  Dimension: {dim_start:.3f} → {dim_end:.3f} ({dim_change:+.3f})")
        print(f"  Avg φ-frequency: {avg_freq:.4f}")
        print(f"  Coherence: {final_coherence:.4f}")
        print(f"  Chord: {state.chord_type}")
        print(f"  Locked: {'YES' if locked else 'NO'}")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: Prime Gap Structure Analysis")
    print("=" * 70)
    
    print(f"\n{'Pattern':<18} {'Dim Δ':<10} {'Freq':<10} {'Coher':<10} {'Chord':<12} {'Locked':<8}")
    print("-" * 68)
    
    for name, data in results['experiments'].items():
        locked = '✓ YES' if data['locked'] else '✗ NO'
        print(f"{name:<18} {data['dim_change']:+.3f}     {data['avg_freq']:.4f}     "
              f"{data['coherence']:.4f}     {data['chord']:<12} {locked}")
    
    # Key comparison: Do primes show φ-structure like Fibonacci?
    prime_freq = results['experiments']['prime_gaps']['avg_freq']
    fib_freq = results['experiments']['fibonacci']['avg_freq']
    random_freq = results['experiments']['random_gaps']['avg_freq']
    
    print(f"\n--- φ-Frequency Comparison ---")
    print(f"  Fibonacci (reference): {fib_freq:.4f}")
    print(f"  Prime gaps:            {prime_freq:.4f}")
    print(f"  Random gaps (control): {random_freq:.4f}")
    
    # Is prime structure closer to Fibonacci or Random?
    fib_distance = abs(prime_freq - fib_freq)
    random_distance = abs(prime_freq - random_freq)
    
    if fib_distance < random_distance:
        print(f"\n✓ Prime gaps show MORE φ-structure than random!")
        print(f"  (Distance to Fib: {fib_distance:.4f} < Distance to Random: {random_distance:.4f})")
        results['conclusion'] = 'primes_show_phi_structure'
    else:
        print(f"\n✗ Prime gaps don't show clear φ-structure advantage")
        results['conclusion'] = 'no_clear_phi_structure'
    
    # Visualize
    visualize_prime_results(all_histories, results)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f'prime_gap_locking_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return results, all_histories


def visualize_prime_results(histories, results):
    """Create visualization of prime gap experiments."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    patterns = ['prime_gaps', 'gap_ratios', 'normalized_gaps', 
                'random_gaps', 'fibonacci']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'gold']
    
    # Top row: Dimension evolution
    ax_dim = axes[0, 0]
    for name, color in zip(patterns, colors):
        ax_dim.plot(histories[name]['dim'], color=color, 
                   label=name, alpha=0.8, linewidth=1.5)
    ax_dim.set_xlabel('Training Step')
    ax_dim.set_ylabel('Effective Dimension')
    ax_dim.set_title('Dimension Evolution by Pattern')
    ax_dim.legend(fontsize=8)
    ax_dim.set_ylim(0, 2)
    
    # Top middle: φ-Frequency comparison (final)
    ax_freq = axes[0, 1]
    final_freqs = [results['experiments'][p]['avg_freq'] for p in patterns]
    bars = ax_freq.bar(range(len(patterns)), final_freqs, color=colors)
    ax_freq.set_xticks(range(len(patterns)))
    ax_freq.set_xticklabels([p.replace('_', '\n') for p in patterns], fontsize=8)
    ax_freq.set_ylabel('Average φ-Frequency')
    ax_freq.set_title('φ-Resonance by Pattern')
    ax_freq.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='threshold')
    
    # Add value labels
    for bar, freq in zip(bars, final_freqs):
        ax_freq.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{freq:.3f}', ha='center', fontsize=8)
    
    # Top right: Coherence comparison
    ax_coh = axes[0, 2]
    coherences = [results['experiments'][p]['coherence'] for p in patterns]
    bars = ax_coh.bar(range(len(patterns)), coherences, color=colors)
    ax_coh.set_xticks(range(len(patterns)))
    ax_coh.set_xticklabels([p.replace('_', '\n') for p in patterns], fontsize=8)
    ax_coh.set_ylabel('Coherence')
    ax_coh.set_title('Strand Coherence by Pattern')
    
    # Bottom left: Prime gaps spectrum heatmap
    ax_prime = axes[1, 0]
    freq_hist = histories['prime_gaps']['freq']
    im = ax_prime.imshow(freq_hist.T, aspect='auto', cmap='viridis',
                        extent=[0, len(freq_hist), 0, freq_hist.shape[1]], 
                        origin='lower')
    ax_prime.set_xlabel('Training Step')
    ax_prime.set_ylabel('Strand')
    ax_prime.set_title('Prime Gaps: Strand Frequency Spectrum')
    plt.colorbar(im, ax=ax_prime, label='φ-freq')
    
    # Bottom middle: Fibonacci spectrum for comparison
    ax_fib = axes[1, 1]
    freq_hist_fib = histories['fibonacci']['freq']
    im2 = ax_fib.imshow(freq_hist_fib.T, aspect='auto', cmap='viridis',
                       extent=[0, len(freq_hist_fib), 0, freq_hist_fib.shape[1]], 
                       origin='lower')
    ax_fib.set_xlabel('Training Step')
    ax_fib.set_ylabel('Strand')
    ax_fib.set_title('Fibonacci: Strand Frequency Spectrum')
    plt.colorbar(im2, ax=ax_fib, label='φ-freq')
    
    # Bottom right: Loss comparison
    ax_loss = axes[1, 2]
    for name, color in zip(patterns, colors):
        losses = histories[name]['loss']
        # Smooth with moving average
        window = 20
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax_loss.semilogy(smoothed, color=color, label=name, alpha=0.8)
    ax_loss.set_xlabel('Training Step')
    ax_loss.set_ylabel('Loss (log scale)')
    ax_loss.set_title('Training Loss')
    ax_loss.legend(fontsize=8)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig_path = FIGURES_DIR / f'prime_gap_analysis_{timestamp}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {fig_path}")
    
    plt.show()


def experiment_gap_prediction():
    """
    Secondary experiment: Can the trained model predict prime gaps?
    
    Train on first N gaps, test on next M gaps.
    Compare to random baseline.
    """
    print("\n" + "=" * 70)
    print("Gap Prediction Experiment")
    print("=" * 70)
    
    primes = sieve_of_eratosthenes(50000)
    gaps = compute_prime_gaps(primes)
    
    # Split
    train_gaps = gaps[:len(gaps)//2]
    test_gaps = gaps[len(gaps)//2:]
    
    print(f"Train gaps: {len(train_gaps)}, Test gaps: {len(test_gaps)}")
    
    # Train model
    model = TinyCIMMMobiusField(n_strands=6, init='random')
    
    print("\nTraining...")
    for step in range(500):
        i = np.random.randint(0, len(train_gaps) - 1)
        x = torch.tensor([[train_gaps[i] / 10.0]], dtype=torch.float32)
        y = torch.tensor([[train_gaps[i + 1] / 10.0]], dtype=torch.float32)
        model.continuous_step(x, y)
    
    state = model.field.get_field_state()
    print(f"Final state: dim={state.effective_dimension:.3f}, chord={state.chord_type}")
    
    # Test prediction
    print("\nTesting prediction...")
    errors = []
    for i in range(len(test_gaps) - 1):
        x = torch.tensor([[test_gaps[i] / 10.0]], dtype=torch.float32)
        y_true = test_gaps[i + 1] / 10.0
        y_pred = model(x).item()
        errors.append(abs(y_pred - y_true))
    
    # Random baseline: predict mean gap
    mean_gap = np.mean(train_gaps) / 10.0
    baseline_errors = [abs(mean_gap - test_gaps[i] / 10.0) for i in range(len(test_gaps) - 1)]
    
    print(f"\nResults:")
    print(f"  Model MAE: {np.mean(errors):.4f}")
    print(f"  Baseline MAE: {np.mean(baseline_errors):.4f}")
    print(f"  Improvement: {(1 - np.mean(errors) / np.mean(baseline_errors)) * 100:.1f}%")
    
    if np.mean(errors) < np.mean(baseline_errors):
        print("✓ Model beats baseline - learned some structure!")
    else:
        print("✗ Model doesn't beat baseline")
    
    return np.mean(errors), np.mean(baseline_errors)


if __name__ == '__main__':
    # Main experiment
    results, histories = experiment_prime_gap_locking()
    
    # Prediction test
    model_err, baseline_err = experiment_gap_prediction()
    
    print("\n" + "=" * 70)
    print("Experiment Complete")
    print("=" * 70)
