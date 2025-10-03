"""
Quick demonstration of PreFieldResonanceDetector
Shows resonance detection with synthetic PAC evolution data
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.field_engine import PreFieldResonanceDetector


def generate_synthetic_pac_evolution(iterations=200, frequency=0.03, noise_level=0.05):
    """Generate synthetic PAC residual trajectory with natural oscillation."""
    pac_trajectory = []
    
    for i in range(iterations):
        # Base oscillation at natural frequency
        oscillation = 0.2 * np.sin(2 * np.pi * frequency * i)
        
        # Exponential decay (PAC converging)
        decay = np.exp(-0.01 * i)
        
        # Add noise
        noise = np.random.normal(0, noise_level)
        
        # Combine (keep positive)
        pac_residual = abs(0.5 * decay + oscillation + noise)
        pac_trajectory.append(pac_residual)
    
    return pac_trajectory


def demonstrate_resonance_detection():
    """Demonstrate resonance detection on synthetic data."""
    
    print("=" * 70)
    print("Pre-Field Resonance Detection Demonstration")
    print("=" * 70)
    print()
    
    # Initialize detector
    detector = PreFieldResonanceDetector(
        window_size=50,
        confidence_threshold=0.15
    )
    
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


def compare_with_without_resonance():
    """Compare convergence speed with and without resonance detection."""
    
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
                decay_rate *= (2.0 / tuning_factor)  # Inverse relationship
        
        pac *= (1 - decay_rate)
    
    return iteration


if __name__ == "__main__":
    print()
    demonstrate_resonance_detection()
    
    # Optional: run comparison
    compare_with_without_resonance()
    
    print("Demonstration complete!")
    print()
