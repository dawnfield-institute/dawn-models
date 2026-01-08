"""
Experiment 28: Prime Gap Continuous Learning with Möbius Strand Field

Use the MobiusStrandField architecture for continuous learning on prime gaps.
Key hypothesis: The φ-resonance in the Möbius strands should "lock on" to 
prime gap structure, allowing increasingly informed predictions.

GPU-OPTIMIZED: Batched processing, minimal CPU-GPU transfers.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Tuple, List, Dict
import matplotlib.pyplot as plt
from collections import Counter

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

def setup_gpu() -> torch.device:
    """Configure GPU with optimal settings."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🚀 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        torch.cuda.empty_cache()
        return device
    else:
        print("⚠️  CPU mode")
        return torch.device('cpu')


# ============================================================================
# PRIME GENERATION
# ============================================================================

def sieve_of_eratosthenes(limit: int) -> np.ndarray:
    """Fast prime sieve."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]


# ============================================================================
# MÖBIUS STRAND FIELD (FULLY VECTORIZED FOR GPU)
# ============================================================================

PHI = 1.618033988749895
PHI_INV = 0.618033988749895


class MobiusStrandField(nn.Module):
    """Möbius Strand Field - FULLY VECTORIZED for GPU."""
    
    def __init__(self, n_strands: int = 8, input_size: int = 10, output_size: int = 1,
                 coupling_strength: float = 0.1, device='cpu'):
        super().__init__()
        self.n_strands = n_strands
        self.device = device
        self.coupling_strength = coupling_strength
        
        # Vectorized Möbius parameters [n_strands, 4] for (a,b,c,d)
        # Initialize with harmonic phases
        phases = torch.tensor([i * np.pi / 4 for i in range(n_strands)], device=device)
        self.mobius_params = nn.Parameter(torch.stack([
            torch.cos(phases),           # a
            torch.sin(phases),           # b  
            -torch.sin(phases) * 0.5,    # c
            torch.cos(phases)            # d
        ], dim=1))  # [n_strands, 4]
        
        self.input_proj = nn.Linear(input_size, n_strands, device=device)
        self.coupling = nn.Parameter(
            torch.eye(n_strands, device=device) + torch.randn(n_strands, n_strands, device=device) * 0.1
        )
        self.output_proj = nn.Linear(n_strands, output_size, device=device)
        self.output_scale = nn.Parameter(torch.tensor(1.0, device=device))
        self.output_bias = nn.Parameter(torch.tensor(0.0, device=device))
        
        # Cache for metrics (computed less frequently)
        self._cached_freq = 0.5
        self._cached_dim = 1.0
        self._cache_step = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fully vectorized forward pass."""
        # x: [batch, input_size]
        strand_inputs = self.input_proj(x)  # [batch, n_strands]
        
        # Vectorized Möbius: M(z) = (a*z + b) / (c*z + d)
        a = self.mobius_params[:, 0]  # [n_strands]
        b = self.mobius_params[:, 1]
        c = self.mobius_params[:, 2]
        d = self.mobius_params[:, 3]
        
        # [batch, n_strands] operations
        numerator = a * strand_inputs + b
        denominator = c * strand_inputs + d + 1e-8
        strand_outputs = numerator / denominator
        
        # Phase coupling
        coupled = torch.matmul(strand_outputs, self.coupling) * self.coupling_strength
        strand_outputs = strand_outputs + coupled
        
        output = self.output_proj(strand_outputs)
        return self.output_scale * output + self.output_bias
    
    def compute_metrics(self, force: bool = False) -> Tuple[float, float, str]:
        """Compute φ-frequency, dimension, chord. Cached for speed."""
        with torch.no_grad():
            a = self.mobius_params[:, 0]
            b = self.mobius_params[:, 1]
            c = self.mobius_params[:, 2]
            d = self.mobius_params[:, 3]
            
            # Vectorized fixed points
            discriminant = (d - a)**2 + 4 * c * b
            sqrt_disc = torch.sqrt(torch.abs(discriminant) + 1e-8)
            
            # Avoid division by zero
            c_safe = torch.where(torch.abs(c) < 1e-8, torch.ones_like(c) * 1e-8, c)
            z1 = (-(d - a) + sqrt_disc) / (2 * c_safe)
            z2 = (-(d - a) - sqrt_disc) / (2 * c_safe)
            
            # φ-frequency for each strand
            PHI = 1.618033988749895
            PHI_INV = 0.618033988749895
            dist_phi = torch.minimum(torch.abs(z1 - PHI), torch.abs(z2 - PHI))
            dist_neg = torch.minimum(torch.abs(z1 + PHI_INV), torch.abs(z2 + PHI_INV))
            freqs = 1.0 / (1.0 + dist_phi + dist_neg)
            
            mean_freq = freqs.mean().item()
            freq_std = freqs.std().item()
            
            # Effective dimension from phases
            phases = torch.atan2(z1 - z2, torch.ones_like(z1))
            phase_diff = phases.unsqueeze(0) - phases.unsqueeze(1)
            correlation = torch.cos(phase_diff)
            eigenvalues = torch.linalg.eigvalsh(correlation)
            eigenvalues = torch.abs(eigenvalues)
            eigenvalues = eigenvalues / (eigenvalues.sum() + 1e-8)
            participation_ratio = 1.0 / (torch.sum(eigenvalues**2) + 1e-8)
            eff_dim = max(0, participation_ratio.item() - 1)
            
            # Chord type
            if mean_freq > 0.8 and freq_std < 0.1:
                chord = "unison"
            elif mean_freq > 0.6 and freq_std < 0.2:
                chord = "phi_chord"
            elif freq_std < 0.15:
                chord = "locked"
            else:
                chord = "polyphonic"
            
            self._cached_freq = mean_freq
            self._cached_dim = eff_dim
            
            return mean_freq, eff_dim, chord


class PrimeGapPredictor(nn.Module):
    """GPU-optimized continuous learning for prime gaps with BATCHED updates."""
    
    def __init__(self, window_size: int = 10, n_strands: int = 8, lr: float = 0.001, 
                 batch_size: int = 64, device='cpu'):
        super().__init__()
        self.window_size = window_size
        self.device = device
        self.batch_size = batch_size
        
        self.field = MobiusStrandField(n_strands, window_size, 1, 0.1, device)
        self.to(device)
        
        # Pre-computed stats (set from data)
        self.register_buffer('gap_mean', torch.tensor(0.0, device=device))
        self.register_buffer('gap_std', torch.tensor(1.0, device=device))
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Tracking (store on CPU to save GPU memory)
        self.step_count = 0
        self.all_predictions = []
        self.all_true_gaps = []
        self.loss_history = []
        self.freq_history = []
        self.dim_history = []
        self.chord_history = []
    
    def set_stats(self, mean: float, std: float):
        """Set normalization stats from data."""
        self.gap_mean = torch.tensor(mean, device=self.device)
        self.gap_std = torch.tensor(std, device=self.device)
    
    def train_batch(self, windows: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Train on a batch - MUCH faster than single samples."""
        self.step_count += len(targets)
        
        # Normalize
        windows_norm = (windows - self.gap_mean) / (self.gap_std + 1e-8)
        targets_norm = (targets - self.gap_mean) / (self.gap_std + 1e-8)
        
        # Forward
        preds_norm = self.field(windows_norm).squeeze(-1)
        loss = nn.functional.mse_loss(preds_norm, targets_norm)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        # Denormalize predictions for tracking
        with torch.no_grad():
            preds = preds_norm * (self.gap_std + 1e-8) + self.gap_mean
            errors = torch.abs(preds - targets)
            mae = errors.mean().item()
            
            # Store predictions (move to CPU)
            self.all_predictions.extend(preds.cpu().tolist())
            self.all_true_gaps.extend(targets.cpu().tolist())
            self.loss_history.append(loss.item())
        
        return {'loss': loss.item(), 'mae': mae, 'batch_size': len(targets)}
    
    def compute_and_log_metrics(self):
        """Compute expensive metrics (call periodically, not every batch)."""
        freq, dim, chord = self.field.compute_metrics()
        self.freq_history.append(freq)
        self.dim_history.append(dim)
        self.chord_history.append(chord)
        return freq, dim, chord


def plot_results(predictor: PrimeGapPredictor, gaps: np.ndarray, save_path: Path):
    """Create comprehensive visualization."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    predictions = np.array(predictor.all_predictions)
    true_gaps = np.array(predictor.all_true_gaps)
    n = len(predictions)
    
    if n == 0:
        print("No predictions to plot!")
        return
    
    errors = np.abs(predictions - true_gaps)
    
    # 1. Predictions vs True Gaps (scatter)
    ax = axes[0, 0]
    ax.scatter(true_gaps, predictions, alpha=0.3, s=1, c='blue')
    max_val = max(np.percentile(true_gaps, 99), np.percentile(predictions, 99))
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect')
    ax.set_xlabel('True Gap')
    ax.set_ylabel('Predicted Gap')
    ax.set_title('Predictions vs True Gaps')
    ax.legend()
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    
    # 2. Time series comparison (last 500)
    ax = axes[0, 1]
    show_n = min(500, n)
    ax.plot(range(show_n), true_gaps[-show_n:], 'b-', alpha=0.7, label='True', lw=0.8)
    ax.plot(range(show_n), predictions[-show_n:], 'r-', alpha=0.7, label='Predicted', lw=0.8)
    ax.set_xlabel('Step (last 500)')
    ax.set_ylabel('Gap')
    ax.set_title('Predictions vs True (Recent)')
    ax.legend()
    
    # 3. Error over time (smoothed)
    ax = axes[1, 0]
    window = min(100, n // 10) if n > 10 else 1
    if window > 1:
        smoothed_error = np.convolve(errors, np.ones(window)/window, mode='valid')
        ax.plot(smoothed_error, 'b-', lw=1)
    else:
        ax.plot(errors, 'b-', lw=1)
    ax.axhline(y=np.std(gaps), color='r', linestyle='--', label=f'Baseline MAE={np.std(gaps):.2f}')
    ax.set_xlabel('Step')
    ax.set_ylabel('MAE (smoothed)')
    ax.set_title('Prediction Error Over Time')
    ax.legend()
    
    # 4. φ-Frequency evolution
    ax = axes[1, 1]
    freqs = np.array(predictor.freq_history)
    if len(freqs) > 0:
        ax.plot(freqs, 'g-', lw=1)
        ax.axhline(y=0.618, color='gold', linestyle='--', label='1/φ ≈ 0.618')
    ax.set_xlabel('Metric Checkpoint')
    ax.set_ylabel('φ-Frequency')
    ax.set_title('φ-Frequency Evolution (Locking Indicator)')
    ax.legend()
    
    # 5. Effective Dimension evolution
    ax = axes[2, 0]
    dims = np.array(predictor.dim_history)
    if len(dims) > 0:
        ax.plot(dims, 'm-', lw=1)
    ax.set_xlabel('Metric Checkpoint')
    ax.set_ylabel('Effective Dimension')
    ax.set_title('Dimension Collapse (Structure Discovery)')
    
    # 6. Prediction error histogram
    ax = axes[2, 1]
    ax.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=np.mean(errors), color='r', linestyle='-', label=f'Mean={np.mean(errors):.2f}')
    ax.axvline(x=np.median(errors), color='g', linestyle='--', label=f'Median={np.median(errors):.2f}')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Saved visualization to: {save_path}")


def create_batched_data(gaps: np.ndarray, window_size: int, device: torch.device):
    """Create all windows and targets as GPU tensors upfront."""
    n = len(gaps) - window_size
    
    # Create windows [n, window_size]
    windows = np.array([gaps[i:i+window_size] for i in range(n)])
    targets = gaps[window_size:window_size + n]
    
    # Move to GPU
    windows_tensor = torch.tensor(windows, dtype=torch.float32, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
    
    return windows_tensor, targets_tensor


def run_experiment(n_primes: int = 10_000, window_size: int = 10, n_strands: int = 8,
                   lr: float = 0.01, batch_size: int = 256, n_epochs: int = 10, log_interval: int = 20):
    """Run prime gap continuous learning - GPU BATCHED with epochs."""
    
    print("=" * 60)
    print("EXP 28: Prime Gap Continuous Learning (Möbius Strand Field)")
    print("=" * 60)
    
    device = setup_gpu()
    
    # Generate primes
    limit = int(n_primes * (np.log(n_primes) + np.log(np.log(n_primes + 10)) + 2))
    primes = sieve_of_eratosthenes(limit)[:n_primes]
    gaps = np.diff(primes).astype(np.float32)
    
    print(f"\nDataset: {len(primes):,} primes, {len(gaps):,} gaps")
    print(f"Gap stats: mean={gaps.mean():.2f}, std={gaps.std():.2f}, range=[{gaps.min():.0f}, {gaps.max():.0f}]")
    
    # Create batched data on GPU
    print(f"\nPreparing GPU tensors...")
    windows, targets = create_batched_data(gaps, window_size, device)
    n_samples = len(targets)
    n_batches = (n_samples + batch_size - 1) // batch_size
    print(f"   {n_samples:,} samples, {n_batches} batches of {batch_size}, {n_epochs} epochs")
    
    # Create model
    predictor = PrimeGapPredictor(window_size, n_strands, lr, batch_size, device)
    predictor.set_stats(float(gaps.mean()), float(gaps.std()))
    
    total_params = sum(p.numel() for p in predictor.parameters())
    print(f"Model: {n_strands} strands, {total_params} params, lr={lr}")
    if device.type == 'cuda':
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e6:.1f} MB")
    
    # Training with epochs
    print(f"\n{'Epoch':>6} {'Batch':>6} | {'Loss':>8} | {'MAE':>8} | {'φ-Freq':>8} | {'Dim':>6} | {'Chord':>10}")
    print("-" * 75)
    
    start_time = time.time()
    total_batches_done = 0
    
    for epoch in range(n_epochs):
        # Shuffle each epoch for better learning
        perm = torch.randperm(n_samples, device=device)
        windows_shuffled = windows[perm]
        targets_shuffled = targets[perm]
        
        epoch_losses = []
        epoch_maes = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            batch_windows = windows_shuffled[start_idx:end_idx]
            batch_targets = targets_shuffled[start_idx:end_idx]
            
            result = predictor.train_batch(batch_windows, batch_targets)
            epoch_losses.append(result['loss'])
            epoch_maes.append(result['mae'])
            total_batches_done += 1
            
            # Log progress
            if (batch_idx + 1) % log_interval == 0 or batch_idx == n_batches - 1:
                freq, dim, chord = predictor.compute_and_log_metrics()
                
                recent_loss = np.mean(epoch_losses[-log_interval:])
                recent_mae = np.mean(epoch_maes[-log_interval:])
                
                print(f"{epoch+1:>6} {batch_idx+1:>6} | {recent_loss:>8.2f} | {recent_mae:>8.2f} | {freq:>8.4f} | {dim:>6.2f} | {chord:>10}")
        
        # Epoch summary
        epoch_loss = np.mean(epoch_losses)
        epoch_mae = np.mean(epoch_maes)
        print(f"  [Epoch {epoch+1} summary: avg_loss={epoch_loss:.2f}, avg_mae={epoch_mae:.2f}]")
    
    elapsed = time.time() - start_time
    
    # Final evaluation (forward pass only, no shuffle)
    print("\nFinal evaluation...")
    predictor.all_predictions = []
    predictor.all_true_gaps = []
    
    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            batch_windows = windows[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]
            
            # Forward only
            windows_norm = (batch_windows - predictor.gap_mean) / (predictor.gap_std + 1e-8)
            preds_norm = predictor.field(windows_norm).squeeze(-1)
            preds = preds_norm * (predictor.gap_std + 1e-8) + predictor.gap_mean
            
            predictor.all_predictions.extend(preds.cpu().tolist())
            predictor.all_true_gaps.extend(batch_targets.cpu().tolist())
    
    # Final stats
    print("-" * 75)
    
    all_preds = np.array(predictor.all_predictions)
    all_true = np.array(predictor.all_true_gaps)
    final_mae = np.mean(np.abs(all_preds - all_true))
    baseline_mae = gaps.std()
    improvement = (baseline_mae - final_mae) / baseline_mae * 100
    
    total_samples = n_samples * n_epochs
    print(f"\nFINAL RESULTS ({elapsed:.1f}s, {total_samples/elapsed:.0f} samples/sec)")
    print(f"  Final MAE: {final_mae:.2f}")
    print(f"  Baseline MAE: {baseline_mae:.2f}")
    print(f"  Improvement: {improvement:.2f}%")
    
    # φ-locking analysis
    if len(predictor.freq_history) >= 2:
        n_metrics = len(predictor.freq_history)
        early_freq = np.mean(predictor.freq_history[:n_metrics//5])
        late_freq = np.mean(predictor.freq_history[-n_metrics//5:])
        early_dim = np.mean(predictor.dim_history[:n_metrics//5])
        late_dim = np.mean(predictor.dim_history[-n_metrics//5:])
        
        print(f"\nφ-LOCKING:")
        if late_freq > early_freq:
            print(f"  ✓ φ-Freq INCREASED: {early_freq:.4f} → {late_freq:.4f}")
        else:
            print(f"  ✗ φ-Freq decreased: {early_freq:.4f} → {late_freq:.4f}")
        
        if late_dim < early_dim:
            print(f"  ✓ Dimension COLLAPSED: {early_dim:.2f} → {late_dim:.2f}")
        else:
            print(f"  ✗ Dimension increased: {early_dim:.2f} → {late_dim:.2f}")
    else:
        early_freq = late_freq = early_dim = late_dim = 0
    
    # Visualization
    fig_path = Path(__file__).parent / 'figures' / f'exp_28_predictions_{n_primes}.png'
    fig_path.parent.mkdir(exist_ok=True)
    plot_results(predictor, gaps, fig_path)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'experiment': 'exp_28_prime_gap_continuous',
        'timestamp': timestamp,
        'config': {'n_primes': n_primes, 'window_size': window_size, 'n_strands': n_strands, 
                   'lr': lr, 'batch_size': batch_size, 'n_epochs': n_epochs},
        'final_mae': float(final_mae),
        'baseline_mae': float(baseline_mae),
        'improvement_pct': float(improvement),
        'early_freq': float(early_freq),
        'late_freq': float(late_freq),
        'freq_increased': bool(late_freq > early_freq) if early_freq else False,
        'early_dim': float(early_dim),
        'late_dim': float(late_dim),
        'dim_collapsed': bool(late_dim < early_dim) if early_dim else False,
        'elapsed_seconds': elapsed,
        'samples_per_second': total_samples / elapsed
    }
    
    save_path = Path(__file__).parent / 'results' / f'exp_28_{n_primes}_{timestamp}.json'
    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {save_path}")
    
    return predictor, results


if __name__ == "__main__":
    # 100k primes with multiple epochs for proper learning
    predictor, results = run_experiment(
        n_primes=100_000,
        window_size=10,
        n_strands=8,
        lr=0.01,         # Higher LR for faster convergence
        batch_size=512,
        n_epochs=20,     # Multiple passes for proper learning
        log_interval=50
    )
