"""
Experiment 26: Visualizing MobiusStrandField Dynamics

Visualizations:
1. Strand frequency spectrum over time (waterfall plot)
2. Phase coupling matrix evolution (heatmap animation)
3. Dimension collapse trajectory (with pattern annotations)
4. Strand fixed points in complex plane (geometric view)
5. Emergent "chord" as audio-style spectrogram
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
import json

from mobius_strand_field import MobiusStrandField, TinyCIMMMobiusField, PHI

RESULTS_DIR = Path(__file__).parent / 'results'
FIGURES_DIR = Path(__file__).parent / 'figures'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def visualize_strand_evolution():
    """
    Create a comprehensive visualization of strand field evolution.
    
    Shows 4 subplots:
    1. Frequency spectrum (each strand's φ-resonance over time)
    2. Phase relationships (how strands couple)
    3. Dimension trajectory (effective dim over time)
    4. Fixed points in complex plane
    """
    print("=" * 70)
    print("Visualizing MobiusStrandField Evolution")
    print("=" * 70)
    
    # Setup
    n_strands = 6
    n_steps = 300
    
    # Data streams
    fibs = [1, 1]
    for _ in range(30):
        fibs.append(fibs[-1] + fibs[-2])
    
    def fib_stream():
        while True:
            idx = np.random.randint(5, 25)
            x = np.array([fibs[idx] / fibs[idx-1]])
            y = np.array([fibs[idx+1] / fibs[idx]])
            yield x, y
    
    # Create model
    model = TinyCIMMMobiusField(n_strands=n_strands, init='random')
    
    # Collect evolution data
    freq_history = []
    phase_history = []
    dim_history = []
    fixed_points_history = []
    coherence_history = []
    loss_history = []
    
    stream = fib_stream()
    
    print("Training and collecting data...")
    for step in range(n_steps):
        x, y = next(stream)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        
        metrics = model.continuous_step(x, y)
        
        # Collect strand data
        freqs = model.field.get_frequency_spectrum().detach().numpy()
        phases = torch.stack([s.phase() for s in model.field.strands]).detach().numpy()
        
        # Collect fixed points
        fps = []
        for s in model.field.strands:
            z1, z2 = s.fixed_points()
            fps.append((z1.item(), z2.item()))
        
        freq_history.append(freqs)
        phase_history.append(phases)
        dim_history.append(metrics['effective_dim'])
        fixed_points_history.append(fps)
        coherence_history.append(metrics['coherence'])
        loss_history.append(metrics['loss'])
        
        if step % 100 == 0:
            print(f"Step {step}: dim={metrics['effective_dim']:.3f}, loss={metrics['loss']:.4f}")
    
    freq_history = np.array(freq_history)
    phase_history = np.array(phase_history)
    dim_history = np.array(dim_history)
    coherence_history = np.array(coherence_history)
    loss_history = np.array(loss_history)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
    
    # 1. Frequency Spectrum (Waterfall)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(freq_history.T, aspect='auto', cmap='viridis',
                      extent=[0, n_steps, 0, n_strands], origin='lower')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Strand')
    ax1.set_title('φ-Frequency Spectrum (Strand Resonance)')
    ax1.axhline(y=0.5, color='gold', linestyle='--', alpha=0.5, label='φ threshold')
    plt.colorbar(im1, ax=ax1, label='φ-resonance')
    
    # Add horizontal line for φ = 0.618 threshold (high resonance)
    for i in range(n_strands):
        high_res = freq_history[:, i] > 0.6
        ax1.scatter(np.where(high_res)[0], np.ones(np.sum(high_res)) * (i + 0.5),
                   c='gold', s=1, alpha=0.3)
    
    # 2. Phase Coupling Evolution
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Show phase differences at end vs start
    phase_start = phase_history[0]
    phase_end = phase_history[-1]
    
    phase_diff_start = np.abs(phase_start[:, np.newaxis] - phase_start[np.newaxis, :])
    phase_diff_end = np.abs(phase_end[:, np.newaxis] - phase_end[np.newaxis, :])
    
    # Correlation from phase (1 = aligned, 0 = orthogonal)
    corr_end = np.cos(phase_diff_end)
    
    im2 = ax2.imshow(corr_end, cmap='RdBu_r', vmin=-1, vmax=1)
    ax2.set_xlabel('Strand')
    ax2.set_ylabel('Strand')
    ax2.set_title('Phase Coupling (Final State)')
    ax2.set_xticks(range(n_strands))
    ax2.set_yticks(range(n_strands))
    plt.colorbar(im2, ax=ax2, label='cos(phase diff)')
    
    # 3. Dimension & Coherence Trajectory
    ax3 = fig.add_subplot(gs[1, 0])
    
    ax3.plot(dim_history, 'b-', linewidth=2, label='Effective Dimension')
    ax3.axhline(y=dim_history[0], color='b', linestyle=':', alpha=0.5)
    ax3.fill_between(range(n_steps), 0, dim_history, alpha=0.2)
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(coherence_history, 'r-', linewidth=1, alpha=0.7, label='Coherence')
    
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Effective Dimension', color='b')
    ax3_twin.set_ylabel('Coherence', color='r')
    ax3.set_title('Dimension Collapse & Coherence')
    
    # Mark dimension collapse
    if dim_history[-1] < dim_history[0] * 0.9:
        ax3.annotate('COLLAPSED', xy=(n_steps*0.8, dim_history[-1]),
                    fontsize=12, color='green', fontweight='bold')
    
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    
    # 4. Fixed Points in Complex Plane (Final State)
    ax4 = fig.add_subplot(gs[1, 1])
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_strands))
    
    # Plot φ and -1/φ as reference points
    ax4.scatter([PHI], [0], s=200, c='gold', marker='*', edgecolors='black',
               label=f'φ = {PHI:.4f}', zorder=10)
    ax4.scatter([-1/PHI], [0], s=200, c='orange', marker='*', edgecolors='black',
               label=f'-1/φ = {-1/PHI:.4f}', zorder=10)
    
    # Plot each strand's fixed points
    final_fps = fixed_points_history[-1]
    for i, (z1, z2) in enumerate(final_fps):
        ax4.scatter([z1], [0], s=80, c=[colors[i]], marker='o', 
                   label=f'Strand {i}' if i < 3 else None)
        ax4.scatter([z2], [0], s=80, c=[colors[i]], marker='s', alpha=0.5)
        # Draw line between fixed points
        ax4.plot([z1, z2], [0, 0], c=colors[i], alpha=0.3, linewidth=2)
    
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax4.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Real')
    ax4.set_ylabel('Imaginary')
    ax4.set_title('Fixed Points (Final) - Circles=z₁, Squares=z₂')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.set_xlim(-3, 3)
    ax4.set_ylim(-0.5, 0.5)
    ax4.set_aspect('equal')
    
    # 5. Loss curve
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.semilogy(loss_history, 'k-', linewidth=1)
    ax5.set_xlabel('Training Step')
    ax5.set_ylabel('Loss (log scale)')
    ax5.set_title('Training Loss')
    ax5.grid(True, alpha=0.3)
    
    # 6. Strand Frequency Evolution (individual traces)
    ax6 = fig.add_subplot(gs[2, 1])
    for i in range(n_strands):
        ax6.plot(freq_history[:, i], c=colors[i], alpha=0.7, 
                label=f'Strand {i}')
    ax6.axhline(y=1/(1 + 2/PHI), color='gold', linestyle='--', 
               label='φ-optimal', alpha=0.5)
    ax6.set_xlabel('Training Step')
    ax6.set_ylabel('φ-Frequency')
    ax6.set_title('Individual Strand Frequencies')
    ax6.legend(loc='center left', fontsize=8)
    ax6.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig_path = FIGURES_DIR / f'strand_evolution_{timestamp}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_path}")
    
    plt.show()
    
    return freq_history, dim_history


def visualize_dimension_comparison():
    """
    Side-by-side comparison: Fibonacci vs Random pattern dimension evolution.
    """
    print("\n" + "=" * 70)
    print("Dimension Comparison: Fibonacci vs Random")
    print("=" * 70)
    
    n_strands = 6
    n_steps = 400
    
    # Fibonacci stream
    fibs = [1, 1]
    for _ in range(30):
        fibs.append(fibs[-1] + fibs[-2])
    
    def fib_stream():
        while True:
            idx = np.random.randint(5, 25)
            yield np.array([fibs[idx] / fibs[idx-1]]), np.array([fibs[idx+1] / fibs[idx]])
    
    def random_stream():
        while True:
            yield np.array([np.random.randn()]), np.array([np.random.randn()])
    
    def poly_stream():
        while True:
            x = np.random.uniform(0.5, 2.0)
            yield np.array([x]), np.array([0.5 * x**2 + 0.3 * x])
    
    patterns = [
        ('Fibonacci (φ-structure)', fib_stream, 'tab:blue'),
        ('Random (entropy)', random_stream, 'tab:red'),
        ('Polynomial', poly_stream, 'tab:green')
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for idx, (name, stream_fn, color) in enumerate(patterns):
        print(f"\nTraining: {name}")
        
        model = TinyCIMMMobiusField(n_strands=n_strands, init='random')
        
        dim_history = []
        freq_history = []
        
        stream = stream_fn()
        for step in range(n_steps):
            x, y = next(stream)
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
            
            metrics = model.continuous_step(x, y)
            dim_history.append(metrics['effective_dim'])
            freq_history.append(model.field.get_frequency_spectrum().detach().numpy())
        
        freq_history = np.array(freq_history)
        
        # Top row: Dimension evolution
        ax_dim = axes[0, idx]
        ax_dim.plot(dim_history, color=color, linewidth=2)
        ax_dim.fill_between(range(n_steps), 0, dim_history, alpha=0.2, color=color)
        ax_dim.axhline(y=dim_history[0], linestyle=':', color=color, alpha=0.5)
        ax_dim.set_title(f'{name}\nDimension: {dim_history[0]:.2f} → {dim_history[-1]:.2f}')
        ax_dim.set_xlabel('Step')
        ax_dim.set_ylabel('Effective Dimension')
        ax_dim.set_ylim(0, 2)
        
        # Annotate
        change = dim_history[-1] - dim_history[0]
        if change < -0.1:
            ax_dim.annotate('↓ COLLAPSED', xy=(n_steps*0.7, dim_history[-1]+0.1),
                          fontsize=10, color='green', fontweight='bold')
        elif change > 0.1:
            ax_dim.annotate('↑ GREW', xy=(n_steps*0.7, dim_history[-1]-0.2),
                          fontsize=10, color='red', fontweight='bold')
        
        # Bottom row: Frequency spectrum heatmap
        ax_freq = axes[1, idx]
        im = ax_freq.imshow(freq_history.T, aspect='auto', cmap='viridis',
                           extent=[0, n_steps, 0, n_strands], origin='lower',
                           vmin=0, vmax=1)
        ax_freq.set_xlabel('Step')
        ax_freq.set_ylabel('Strand')
        ax_freq.set_title('φ-Frequency Spectrum')
    
    plt.colorbar(im, ax=axes[1, :], label='φ-resonance', shrink=0.8)
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig_path = FIGURES_DIR / f'dimension_comparison_{timestamp}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_path}")
    
    plt.show()


def visualize_strand_geometry():
    """
    Visualize strands as a geometric object in 3D.
    
    Each strand's fixed points define a "direction" in space.
    The coupling between strands creates edges.
    This shows the emergent geometry.
    """
    print("\n" + "=" * 70)
    print("Strand Geometry Visualization")
    print("=" * 70)
    
    n_strands = 8
    
    # Create and train model
    model = TinyCIMMMobiusField(n_strands=n_strands, init='harmonic')
    
    fibs = [1, 1]
    for _ in range(30):
        fibs.append(fibs[-1] + fibs[-2])
    
    # Train briefly
    for _ in range(200):
        idx = np.random.randint(5, 25)
        x = torch.tensor([[fibs[idx] / fibs[idx-1]]], dtype=torch.float32)
        y = torch.tensor([[fibs[idx+1] / fibs[idx]]], dtype=torch.float32)
        model.continuous_step(x, y)
    
    # Extract strand properties for visualization
    freqs = model.field.get_frequency_spectrum().detach().numpy()
    phases = torch.stack([s.phase() for s in model.field.strands]).detach().numpy()
    amps = torch.stack([s.amplitude() for s in model.field.strands]).detach().numpy()
    
    # Create 3D visualization
    fig = plt.figure(figsize=(14, 6))
    
    # Left: 3D strand positions (using freq, phase, amp as coordinates)
    ax1 = fig.add_subplot(121, projection='3d')
    
    colors = plt.cm.viridis(freqs)
    scatter = ax1.scatter(freqs, phases, amps, 
                         c=freqs, cmap='viridis', s=200, 
                         edgecolors='black', linewidth=1)
    
    # Draw coupling lines between strongly coupled strands
    coupling = model.field.coupling.detach().numpy()
    threshold = np.percentile(np.abs(coupling), 80)
    
    for i in range(n_strands):
        for j in range(i+1, n_strands):
            if np.abs(coupling[i, j]) > threshold:
                ax1.plot([freqs[i], freqs[j]], 
                        [phases[i], phases[j]], 
                        [amps[i], amps[j]], 
                        'k-', alpha=0.3, linewidth=np.abs(coupling[i, j]))
    
    # Mark φ reference
    ax1.scatter([1], [0], [1], c='gold', s=300, marker='*', 
               edgecolors='black', label='φ-optimal')
    
    ax1.set_xlabel('φ-Frequency')
    ax1.set_ylabel('Phase')
    ax1.set_zlabel('Amplitude')
    ax1.set_title('Strand Positions in Feature Space')
    
    # Add strand labels
    for i in range(n_strands):
        ax1.text(freqs[i], phases[i], amps[i], f'  {i}', fontsize=8)
    
    plt.colorbar(scatter, ax=ax1, label='φ-resonance', shrink=0.6)
    
    # Right: Circular arrangement showing phase relationships
    ax2 = fig.add_subplot(122, projection='polar')
    
    # Plot strands as points on a circle, radius = frequency
    theta = phases + np.pi  # Shift to positive angles
    r = freqs
    
    scatter2 = ax2.scatter(theta, r, c=amps, cmap='plasma', s=200,
                          edgecolors='black', linewidth=1)
    
    # Connect strands with coupling strength
    for i in range(n_strands):
        for j in range(i+1, n_strands):
            if np.abs(coupling[i, j]) > threshold:
                ax2.plot([theta[i], theta[j]], [r[i], r[j]], 
                        'k-', alpha=0.3, linewidth=np.abs(coupling[i, j]) * 2)
    
    # Add strand labels
    for i in range(n_strands):
        ax2.annotate(f'{i}', (theta[i], r[i]), fontsize=10, 
                    ha='center', va='bottom')
    
    ax2.set_title('Phase-Frequency Polar View\n(color = amplitude)')
    plt.colorbar(scatter2, ax=ax2, label='Amplitude', shrink=0.6)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig_path = FIGURES_DIR / f'strand_geometry_{timestamp}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_path}")
    
    # Print summary
    state = model.field.get_field_state()
    print(f"\nField State:")
    print(f"  Effective Dimension: {state.effective_dimension:.3f}")
    print(f"  Coherence: {state.coherence:.3f}")
    print(f"  Chord Type: {state.chord_type}")
    
    plt.show()


def visualize_fixed_point_evolution():
    """
    Animate how fixed points move during training.
    """
    print("\n" + "=" * 70)
    print("Fixed Point Evolution Animation")
    print("=" * 70)
    
    n_strands = 4
    n_steps = 200
    
    model = TinyCIMMMobiusField(n_strands=n_strands, init='random')
    
    fibs = [1, 1]
    for _ in range(30):
        fibs.append(fibs[-1] + fibs[-2])
    
    # Collect fixed points during training
    fps_history = []
    dim_history = []
    
    for step in range(n_steps):
        idx = np.random.randint(5, 25)
        x = torch.tensor([[fibs[idx] / fibs[idx-1]]], dtype=torch.float32)
        y = torch.tensor([[fibs[idx+1] / fibs[idx]]], dtype=torch.float32)
        
        metrics = model.continuous_step(x, y)
        
        fps = []
        for s in model.field.strands:
            z1, z2 = s.fixed_points()
            fps.append((z1.item(), z2.item()))
        fps_history.append(fps)
        dim_history.append(metrics['effective_dim'])
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_strands))
    
    # Left: Fixed point trajectories
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    # Plot φ and -1/φ targets
    ax1.scatter([PHI], [0], s=300, c='gold', marker='*', 
               edgecolors='black', zorder=100, label=f'φ = {PHI:.3f}')
    ax1.scatter([-1/PHI], [0], s=300, c='orange', marker='*', 
               edgecolors='black', zorder=100, label=f'-1/φ = {-1/PHI:.3f}')
    
    # Draw trajectories for each strand's z1 fixed point
    for i in range(n_strands):
        z1_traj = [fps[i][0] for fps in fps_history]
        z2_traj = [fps[i][1] for fps in fps_history]
        
        # Color by time (light to dark)
        t = np.linspace(0, 1, len(z1_traj))
        
        # Plot z1 trajectory
        for j in range(len(z1_traj)-1):
            ax1.plot([z1_traj[j], z1_traj[j+1]], 
                    [0.05*i, 0.05*i], 
                    c=colors[i], alpha=t[j], linewidth=2)
        
        # Mark start and end
        ax1.scatter([z1_traj[0]], [0.05*i], c=[colors[i]], marker='o', 
                   s=50, edgecolors='black', alpha=0.5)
        ax1.scatter([z1_traj[-1]], [0.05*i], c=[colors[i]], marker='s', 
                   s=100, edgecolors='black', label=f'Strand {i} end')
    
    ax1.set_xlabel('Fixed Point Value (Real)')
    ax1.set_ylabel('Strand (offset for visibility)')
    ax1.set_title('Fixed Point z₁ Migration During Training\n(light→dark = time)')
    ax1.set_xlim(-3, 3)
    ax1.legend(loc='upper right', fontsize=8)
    
    # Right: Dimension over time
    ax2.plot(dim_history, 'b-', linewidth=2)
    ax2.fill_between(range(n_steps), 0, dim_history, alpha=0.2)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Effective Dimension')
    ax2.set_title('Dimension Collapse')
    ax2.axhline(y=dim_history[0], linestyle=':', color='gray', alpha=0.5)
    
    # Annotate collapse
    ax2.annotate(f'Start: {dim_history[0]:.2f}', 
                xy=(10, dim_history[0]), fontsize=10)
    ax2.annotate(f'End: {dim_history[-1]:.2f}', 
                xy=(n_steps-50, dim_history[-1]), fontsize=10)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig_path = FIGURES_DIR / f'fixed_point_evolution_{timestamp}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_path}")
    
    plt.show()


if __name__ == '__main__':
    # Run all visualizations
    visualize_strand_evolution()
    visualize_dimension_comparison()
    visualize_strand_geometry()
    visualize_fixed_point_evolution()
    
    print("\n" + "=" * 70)
    print("All visualizations complete!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 70)
