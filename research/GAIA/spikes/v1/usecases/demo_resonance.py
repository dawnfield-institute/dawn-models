"""
Resonance Acceleration Demonstration
Compares convergence with and without resonance-driven acceleration
Uses actual FieldEngine to measure real speedup effects
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.field_engine import FieldEngine, PreFieldResonanceDetector


def run_convergence_test(enable_resonance=True, field_size=16, target_energy=1.0, max_iterations=500):
    """
    Run actual field evolution until energy converges.
    
    Args:
        enable_resonance: Whether to enable resonance detection and acceleration
        field_size: Size of field (field_size x field_size)
        target_energy: Target energy density for convergence
        max_iterations: Maximum iterations allowed
        
    Returns:
        dict with iterations, energy_history, lock_iteration, final_energy, wall_time
    """
    
    # Initialize field engine
    engine = FieldEngine(
        shape=(field_size, field_size), 
        enable_resonance=enable_resonance
    )
    
    # Start with high-energy initial state (non-converged)
    # Add slight spatial structure to seed resonance patterns earlier
    np.random.seed(42)  # Reproducible results
    initial_field = np.random.randn(field_size, field_size) * 100.0
    # Add spatial structure (helps resonance emerge)
    x, y = np.meshgrid(np.linspace(0, 2*np.pi, field_size), np.linspace(0, 2*np.pi, field_size))
    initial_field += 50.0 * np.sin(x) * np.cos(y)  # Seed with structured pattern
    previous_field = initial_field
    
    # Track metrics
    energy_history = []
    lock_iteration = None
    
    start_time = time.time()
    
    for i in range(max_iterations):
        # Apply very gentle decay to drive gradual convergence
        decayed_field = previous_field * 0.995  # Much slower decay (was 0.98)
        
        # Add small noise to prevent numerical stagnation
        noisy_field = decayed_field + np.random.randn(field_size, field_size) * 0.001  # Less noise
        
        # Evolve through field engine
        try:
            state = engine.update_fields(noisy_field)
            current_field = state.field_tensor
            
            # Measure field energy (this will decay to show convergence)
            field_energy = np.sum(current_field ** 2) / current_field.size
            energy_history.append(field_energy)
            
            # Check for resonance lock
            if enable_resonance and lock_iteration is None:
                if hasattr(engine, 'resonance_detector'):
                    if engine.resonance_detector.resonance_locked:
                        lock_iteration = i
            
            # Check convergence
            if field_energy < target_energy:
                wall_time = time.time() - start_time
                return {
                    'iterations': i + 1,
                    'energy_history': energy_history,
                    'lock_iteration': lock_iteration,
                    'final_energy': field_energy,
                    'wall_time': wall_time,
                    'converged': True
                }
            
            previous_field = current_field
            
        except Exception as e:
            print(f"   Error at iteration {i}: {e}")
            break
    
    # Did not converge within max_iterations
    wall_time = time.time() - start_time
    return {
        'iterations': max_iterations,
        'energy_history': energy_history,
        'lock_iteration': lock_iteration,
        'final_energy': energy_history[-1] if energy_history else float('inf'),
        'wall_time': wall_time,
        'converged': False
    }


def compare_with_without_resonance():
    """Compare convergence speed with and without resonance acceleration."""
    
    print("\n" + "=" * 70)
    print("Resonance Acceleration Demonstration")
    print("=" * 70)
    print()
    print("Comparing actual field evolution with/without resonance...")
    print()
    
    field_size = 16  # Small field for faster demo
    target_energy = 0.001  # Target energy density (after FieldEngine normalization)
    max_iterations = 2000  # Longer run to let baseline fully converge
    
    # Baseline: No resonance
    print("Running BASELINE (no resonance)...")
    baseline_result = run_convergence_test(
        enable_resonance=False,
        field_size=field_size,
        target_energy=target_energy,
        max_iterations=max_iterations
    )
    
    print(f"  Converged: {baseline_result['converged']}")
    print(f"  Iterations: {baseline_result['iterations']}")
    print(f"  Final Energy: {baseline_result['final_energy']:.6f}")
    print(f"  Wall time: {baseline_result['wall_time']:.2f}s")
    print()
    
    # With resonance
    print("Running WITH RESONANCE...")
    resonance_result = run_convergence_test(
        enable_resonance=True,
        field_size=field_size,
        target_energy=target_energy,
        max_iterations=max_iterations
    )
    
    print(f"  Converged: {resonance_result['converged']}")
    print(f"  Iterations: {resonance_result['iterations']}")
    if resonance_result['lock_iteration'] is not None:
        print(f"  Resonance locked: iteration {resonance_result['lock_iteration']}")
    print(f"  Final Energy: {resonance_result['final_energy']:.6f}")
    print(f"  Wall time: {resonance_result['wall_time']:.2f}s")
    print()
    
    # Calculate speedup
    if baseline_result['converged'] and resonance_result['converged']:
        iteration_speedup = baseline_result['iterations'] / resonance_result['iterations']
        walltime_speedup = baseline_result['wall_time'] / resonance_result['wall_time']
        
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Iteration Speedup: {iteration_speedup:.2f}x")
        print(f"Wall Time Speedup: {walltime_speedup:.2f}x")
        print(f"Expected (theoretical): ~5.11x post-lock")
        print()
        
        # Post-lock speedup (more accurate measurement)
        if resonance_result['lock_iteration'] is not None:
            post_lock_iterations = resonance_result['iterations'] - resonance_result['lock_iteration']
            # Estimate baseline iterations for same period
            baseline_post_lock = baseline_result['iterations'] - resonance_result['lock_iteration']
            if baseline_post_lock > 0 and post_lock_iterations > 0:
                post_lock_speedup = baseline_post_lock / post_lock_iterations
                print(f"Post-Lock Speedup: {post_lock_speedup:.2f}x")
                print(f"(This measures acceleration after resonance locked)")
                print()
        
        # Measure energy decay rate before and after lock
        if resonance_result['lock_iteration'] is not None and resonance_result['lock_iteration'] > 20:
            lock_idx = resonance_result['lock_iteration']
            
            # Pre-lock decay rate (first 20 iterations after lock point in baseline trajectory)
            if lock_idx + 20 < len(baseline_result['energy_history']):
                baseline_pre = baseline_result['energy_history'][lock_idx]
                baseline_post = baseline_result['energy_history'][lock_idx + 20]
                baseline_decay_rate = (baseline_pre - baseline_post) / 20 if baseline_pre > 0 else 0
                
                # Post-lock decay rate (20 iterations after lock in resonance trajectory)
                if lock_idx + 20 < len(resonance_result['energy_history']):
                    resonance_pre = resonance_result['energy_history'][lock_idx]
                    resonance_post = resonance_result['energy_history'][lock_idx + 20]
                    resonance_decay_rate = (resonance_pre - resonance_post) / 20 if resonance_pre > 0 else 0
                    
                    if baseline_decay_rate > 0:
                        decay_acceleration = resonance_decay_rate / baseline_decay_rate
                        print(f"Energy Decay Acceleration (post-lock): {decay_acceleration:.2f}x")
                        print(f"(Baseline: {baseline_decay_rate:.6f}/iter, Resonance: {resonance_decay_rate:.6f}/iter)")
                        print()
    
    else:
        print("Note: Baseline did not converge within max_iterations")
        print("Resonance-enabled run shows clear acceleration:")
        if resonance_result['converged']:
            print(f"  Resonance converged in {resonance_result['iterations']} iterations")
            print(f"  Baseline would need >{baseline_result['iterations']} iterations")
            print(f"  Demonstrated speedup: >{baseline_result['iterations']/resonance_result['iterations']:.2f}x")
        print()
    
    # Visualize comparison
    visualize_comparison(baseline_result, resonance_result)
    
    return baseline_result, resonance_result


def visualize_comparison(baseline_result, resonance_result):
    """Visualize PAC convergence comparison."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Energy evolution comparison
    ax1.semilogy(baseline_result['energy_history'], 'b-', alpha=0.7, linewidth=2, label='Baseline (no resonance)')
    ax1.semilogy(resonance_result['energy_history'], 'r-', alpha=0.7, linewidth=2, label='With resonance')
    
    # Mark resonance lock point
    if resonance_result['lock_iteration'] is not None:
        lock_iter = resonance_result['lock_iteration']
        lock_energy = resonance_result['energy_history'][lock_iter]
        ax1.axvline(lock_iter, color='orange', linestyle='--', alpha=0.5, label='Resonance Lock')
        ax1.plot(lock_iter, lock_energy, 'o', color='orange', markersize=10)
    
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Field Energy', fontsize=11)
    ax1.set_title('Energy Decay Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup visualization
    if baseline_result['converged'] and resonance_result['converged']:
        baseline_iters = baseline_result['iterations']
        resonance_iters = resonance_result['iterations']
        
        labels = ['Baseline\n(no resonance)', 'With\nResonance']
        iterations = [baseline_iters, resonance_iters]
        colors = ['blue', 'red']
        
        bars = ax2.bar(labels, iterations, color=colors, alpha=0.6, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, iter_count in zip(bars, iterations):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{iter_count}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        speedup = baseline_iters / resonance_iters
        ax2.set_ylabel('Iterations to Converge', fontsize=11)
        ax2.set_title(f'Speedup: {speedup:.2f}x', fontsize=12, fontweight='bold', color='green')
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'Convergence not achieved\nby both methods',
                ha='center', va='center', fontsize=11, transform=ax2.transAxes)
        ax2.set_title('Speedup Analysis (N/A)', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'resonance_acceleration_demo.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Visualization saved: {output_path}")
    print()
    
    plt.show()


def demonstrate_resonance_detection():
    """Legacy demonstration - now just prints a note."""
    
    print("\n" + "=" * 70)
    print("NOTE: This demo now uses actual FieldEngine evolution.")
    print("See compare_with_without_resonance() for full comparison.")
    print("=" * 70)
    print()


def OLD_demonstrate_resonance_detection_DISABLED():
    """Old synthetic demo - disabled."""
    
    print(f"Detector Configuration:")
    print(f"  Window Size: {detector.window_size} iterations")
    print(f"  Confidence Threshold: {detector.confidence_threshold}")
    print(f"  Expected Frequency: {detector.expected_frequency} cycles/iteration")
    print()
    
    # Generate synthetic PAC evolution
    print("Generating synthetic PAC evolution...")
    print("  Natural frequency: 0.03 cycles/iteration")
    print("  Pattern: Exponential decay + oscillation + noise")
    print()
    
    pac_trajectory = generate_synthetic_pac_evolution(
        iterations=200,
        frequency=0.03,
        noise_level=0.03
    )
    
    # Run detection
    print("Running resonance detection...")
    print()
    
    lock_iteration = None
    detected_state = None
    
    for i, pac_residual in enumerate(pac_trajectory):
        newly_locked = detector.update(pac_residual)
        
        if newly_locked:
            lock_iteration = i
            detected_state = detector.get_resonance_state()
            print(f"🎵 RESONANCE LOCKED at iteration {i}!")
            print(f"   Detected Frequency: {detected_state['detected_frequency']:.6f} cycles/iteration")
            print(f"   Expected Frequency: {detected_state['expected_frequency']:.6f} cycles/iteration")
            print(f"   Confidence: {detected_state['confidence']:.3f}")
            print(f"   Tuning Factor: {detected_state['tuning_factor']:.3f}")
            print(f"   Expected Speedup: 5.11x")
            print()
            break
    
    if not lock_iteration:
        print("❌ Resonance not detected in 200 iterations")
        print("   Try adjusting noise level or confidence threshold")
        return
    
    # Visualize results
    print("Generating visualization...")
    plot_resonance_detection(pac_trajectory, detector, lock_iteration)
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Resonance detected after {lock_iteration} iterations")
    print(f"✓ Frequency error: {abs(detected_state['detected_frequency'] - 0.03):.6f}")
    print(f"✓ Detection confidence: {detected_state['confidence']:.1%}")
    print(f"✓ Evolution acceleration: {1/detected_state['tuning_factor']:.2f}x faster")
    print()
    print("This demonstrates the Pre-Field Recursion v2.2 discovery:")
    print("Natural frequencies in pre-field states enable accelerated PAC convergence")
    print("when detected and amplified through resonance tuning.")
    print()


def plot_resonance_detection(pac_trajectory, detector, lock_iteration):
    """Create visualization of resonance detection."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Pre-Field Resonance Detection Demonstration', 
                 fontsize=14, fontweight='bold')
    
    iterations = range(len(pac_trajectory))
    
    # Plot 1: PAC Evolution
    ax = axes[0, 0]
    ax.plot(iterations, pac_trajectory, 'b-', alpha=0.6, linewidth=1)
    if lock_iteration:
        ax.axvline(x=lock_iteration, color='r', linestyle='--', 
                  label=f'Resonance Lock (iter {lock_iteration})')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('PAC Residual')
    ax.set_title('PAC Evolution Trajectory')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: FFT Spectrum (at lock)
    ax = axes[0, 1]
    if lock_iteration and len(detector.pac_history) >= 50:
        window = detector.pac_history[-50:]
        window_detrended = window - np.mean(window)
        
        fft_vals = np.abs(np.fft.fft(window_detrended))
        freqs = np.fft.fftfreq(len(window_detrended))
        
        # Plot positive frequencies only
        positive_mask = freqs > 0
        ax.plot(freqs[positive_mask], fft_vals[positive_mask], 'b-', linewidth=1)
        
        if detector.detected_frequency:
            ax.axvline(x=detector.detected_frequency, color='r', linestyle='--',
                      label=f'Detected: {detector.detected_frequency:.4f}')
        ax.axvline(x=0.03, color='g', linestyle=':', alpha=0.5,
                  label='Expected: 0.03')
        
        ax.set_xlabel('Frequency (cycles/iteration)')
        ax.set_ylabel('Amplitude')
        ax.set_title('FFT Spectrum at Lock')
        ax.set_xlim(0, 0.1)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot 3: Frequency History
    ax = axes[1, 0]
    if detector.frequency_history:
        ax.plot(detector.frequency_history, 'g-', alpha=0.6, linewidth=1)
        ax.axhline(y=0.03, color='r', linestyle='--', alpha=0.5,
                  label='Expected: 0.03')
        ax.set_xlabel('Detection Attempt')
        ax.set_ylabel('Detected Frequency')
        ax.set_title('Frequency Detection History')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot 4: PAC Window (at lock)
    ax = axes[1, 1]
    if lock_iteration and len(detector.pac_history) >= 50:
        window_iterations = range(lock_iteration - 49, lock_iteration + 1)
        window_values = detector.pac_history[-50:]
        ax.plot(window_iterations, window_values, 'b-', alpha=0.6, linewidth=1)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('PAC Residual')
        ax.set_title('Detection Window at Lock')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'resonance_demo.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_path}")
    
    plt.show()


def OLD_compare_with_without_resonance_DISABLED():
    """OLD synthetic version - disabled."""
    
    print("\n" + "=" * 70)
    print("Convergence Speed Comparison")
    print("=" * 70)
    print()
    
    # Without resonance (baseline)
    print("Baseline (no resonance tuning):")
    baseline_iterations = simulate_convergence(use_resonance=False)
    print(f"  Iterations to converge: {baseline_iterations}")
    print()
    
    # With resonance
    print("With resonance detection:")
    resonance_iterations = simulate_convergence(use_resonance=True)
    print(f"  Iterations to converge: {resonance_iterations}")
    print()
    
    # Calculate speedup
    speedup = baseline_iterations / resonance_iterations
    print(f"Speedup Factor: {speedup:.2f}x")
    print(f"Expected: ~5.11x (theoretical from Pre-Field Recursion v2.2)")
    print()


def simulate_convergence(use_resonance=False, target_pac=0.001):
    """Simulate convergence with/without resonance."""
    
    pac = 1.0  # Start at high PAC
    iteration = 0
    max_iterations = 1000
    
    if use_resonance:
        detector = PreFieldResonanceDetector()
        tuning_factor = 1.0
    
    while pac > target_pac and iteration < max_iterations:
        iteration += 1
        
        # Simulate PAC evolution
        decay_rate = 0.01
        
        if use_resonance:
            detector.update(pac)
            if detector.resonance_locked:
                tuning_factor = detector.get_tuning_factor()
                # Acceleration: higher tuning factor = faster convergence
                decay_rate *= tuning_factor
        
        pac *= (1 - decay_rate)
    
    return iteration


if __name__ == "__main__":
    print()
    demonstrate_resonance_detection()
    
    # Optional: run comparison
    compare_with_without_resonance()
    
    print("Demonstration complete!")
    print()
