"""
Experiment 30: PAC Elastic Loss

Key insight: PAC isn't a hard constraint - it's an elastic tension.

Instead of enforcing conservation:
  - PAC deviation adds to loss (elastic energy)
  - Split ratios are LEARNED, not hardcoded to φ
  - We watch: does the network converge to φ on its own?

The "elastic on a stick" model:
  - Stick = initial PAC total (anchor point)
  - Elastic = deviation cost
  - Prediction = the thing being pulled
  - Tension = PAC loss term

If φ-conservation is truly optimal, the network will discover it.
If not, we learn something important about what's actually fundamental.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# ============================================================================
# CONSTANTS (for reference, NOT enforced)
# ============================================================================

PHI = 1.618033988749895
PHI_INV = 0.618033988749895

# ============================================================================
# GPU SETUP
# ============================================================================

def setup_gpu() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        return device
    return torch.device('cpu')


# ============================================================================
# PRIME GENERATION
# ============================================================================

def sieve_of_eratosthenes(limit: int) -> np.ndarray:
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]


# ============================================================================
# PAC ELASTIC MÖBIUS FIELD
# ============================================================================

class PACElasticMobiusField(nn.Module):
    """
    Möbius field where PAC conservation is an elastic tension, not a constraint.
    
    Key differences from exp_29:
    1. PAC deviation is part of loss (elastic_strength controls tension)
    2. Split ratios are LEARNED parameters (not hardcoded to φ)
    3. We track if learned ratios converge to φ naturally
    4. Continuous soft-splitting via differentiable amplitude redistribution
    """
    
    def __init__(
        self,
        n_strands: int = 8,
        input_size: int = 10,
        output_size: int = 1,
        elastic_strength: float = 1.0,  # How hard the elastic pulls
        device: torch.device = None,
    ):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.n_strands = n_strands
        self.elastic_strength = elastic_strength
        
        # Möbius parameters [n_strands, 4] for (a,b,c,d)
        phases = torch.tensor([i * np.pi / n_strands for i in range(n_strands)], device=device)
        initial_params = torch.stack([
            torch.cos(phases),           # a
            torch.sin(phases),           # b  
            -torch.sin(phases) * 0.5,    # c
            torch.cos(phases)            # d
        ], dim=1)
        self.mobius_params = nn.Parameter(initial_params)
        
        # Amplitude weights - these determine PAC total
        self.amplitudes = nn.Parameter(torch.ones(n_strands, device=device) / n_strands)
        
        # LEARNED split ratio - initialize FAR from φ to test discovery
        # sigmoid(-2) ≈ 0.12, sigmoid(2) ≈ 0.88
        # φ⁻¹ ≈ 0.618, so we start at 0.3 (sigmoid(-0.847) ≈ 0.3)
        # If network truly discovers φ, it should move from 0.3 → 0.618
        init_val = -0.847  # sigmoid(-0.847) ≈ 0.3, far from φ
        self.split_ratio = nn.Parameter(torch.tensor(init_val, device=device))
        
        # Input/output projections
        self.input_proj = nn.Linear(input_size, n_strands, device=device)
        self.output_proj = nn.Linear(n_strands, output_size, device=device)
        self.output_scale = nn.Parameter(torch.tensor(1.0, device=device))
        self.output_bias = nn.Parameter(torch.tensor(0.0, device=device))
        
        # Phase coupling
        self.coupling = nn.Parameter(
            torch.eye(n_strands, device=device) * 0.9 + torch.randn(n_strands, n_strands, device=device) * 0.1
        )
        self.coupling_strength = 0.1
        
        # PAC anchor (set on first forward)
        self.register_buffer('pac_anchor', torch.tensor(0.0, device=device))
        self.pac_initialized = False
        
        # Tracking
        self.pac_totals: List[float] = []
        self.split_ratios: List[float] = []
        self.elastic_losses: List[float] = []
    
    def compute_frequencies(self) -> torch.Tensor:
        """Compute φ-frequency for each strand."""
        a = self.mobius_params[:, 0]
        b = self.mobius_params[:, 1]
        c = self.mobius_params[:, 2]
        d = self.mobius_params[:, 3]
        
        discriminant = (d - a)**2 + 4 * c * b
        sqrt_disc = torch.sqrt(torch.abs(discriminant) + 1e-8)
        
        c_safe = torch.where(torch.abs(c) < 1e-8, torch.ones_like(c) * 1e-8, c)
        z1 = (-(d - a) + sqrt_disc) / (2 * c_safe)
        z2 = (-(d - a) - sqrt_disc) / (2 * c_safe)
        
        dist_phi = torch.minimum(torch.abs(z1 - PHI), torch.abs(z2 - PHI))
        dist_neg = torch.minimum(torch.abs(z1 + PHI_INV), torch.abs(z2 + PHI_INV))
        
        return 1.0 / (1.0 + dist_phi + dist_neg)
    
    def compute_pac_total(self) -> torch.Tensor:
        """
        Compute total PAC quantity: Σ(freq × amp).
        
        The split_ratio modulates how amplitudes are distributed,
        creating gradient flow through this parameter.
        """
        freqs = self.compute_frequencies()
        amps = torch.softmax(self.amplitudes, dim=0)
        
        # Apply split ratio as a modulation factor
        # This creates coupling between split_ratio and the PAC total
        ratio = torch.sigmoid(self.split_ratio)
        
        # Modulate: even strands get ratio, odd strands get (1-ratio)
        # This makes split_ratio influence the effective amplitude distribution
        modulation = torch.zeros(self.n_strands, device=self.device)
        modulation[0::2] = ratio
        modulation[1::2] = 1 - ratio
        modulation = modulation / modulation.sum()  # Normalize
        
        # Blend original amps with modulated distribution
        effective_amps = 0.5 * amps + 0.5 * modulation
        
        return torch.sum(freqs * effective_amps)
    
    def compute_elastic_loss(self) -> torch.Tensor:
        """
        The elastic tension - how far are we from the PAC anchor?
        
        This is the "rubber band" pulling us back to conservation.
        """
        current = self.compute_pac_total()
        deviation = current - self.pac_anchor
        
        # Quadratic elastic potential (Hooke's law)
        elastic_energy = self.elastic_strength * deviation ** 2
        
        return elastic_energy
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (prediction, elastic_loss).
        
        The elastic loss should be added to prediction loss externally.
        """
        # Initialize PAC anchor on first forward
        if not self.pac_initialized:
            self.pac_anchor = self.compute_pac_total().detach()
            self.pac_initialized = True
        
        # Project to strand space
        strand_inputs = self.input_proj(x)
        
        # Vectorized Möbius transform
        a = self.mobius_params[:, 0]
        b = self.mobius_params[:, 1]
        c = self.mobius_params[:, 2]
        d = self.mobius_params[:, 3]
        
        numerator = a * strand_inputs + b
        denominator = c * strand_inputs + d + 1e-8
        strand_outputs = numerator / denominator
        
        # Weight by amplitudes
        amps = torch.softmax(self.amplitudes, dim=0)
        strand_outputs = strand_outputs * amps
        
        # Phase coupling
        if strand_outputs.dim() == 1:
            strand_outputs = strand_outputs.unsqueeze(0)
        coupled = torch.matmul(strand_outputs, self.coupling) * self.coupling_strength
        strand_outputs = strand_outputs + coupled
        
        # Output
        output = self.output_proj(strand_outputs)
        prediction = self.output_scale * output + self.output_bias
        
        # Compute elastic loss
        elastic_loss = self.compute_elastic_loss()
        
        # Track for analysis
        if self.training:
            self.pac_totals.append(self.compute_pac_total().item())
            self.split_ratios.append(torch.sigmoid(self.split_ratio).item())  # Keep in [0,1]
            self.elastic_losses.append(elastic_loss.item())
        
        return prediction, elastic_loss
    
    def get_learned_ratio(self) -> float:
        """Get the learned split ratio (should converge to φ⁻¹ ≈ 0.618)."""
        return torch.sigmoid(self.split_ratio).item()
    
    def get_state(self) -> Dict:
        """Get current field state."""
        freqs = self.compute_frequencies()
        amps = torch.softmax(self.amplitudes, dim=0)
        
        return {
            'n_strands': self.n_strands,
            'mean_freq': freqs.mean().item(),
            'pac_total': self.compute_pac_total().item(),
            'pac_anchor': self.pac_anchor.item(),
            'pac_deviation': abs(self.compute_pac_total().item() - self.pac_anchor.item()),
            'learned_ratio': self.get_learned_ratio(),
            'ratio_vs_phi': abs(self.get_learned_ratio() - PHI_INV),
            'elastic_strength': self.elastic_strength,
        }


# ============================================================================
# PRIME GAP PREDICTOR WITH PAC ELASTIC
# ============================================================================

class PACElasticPredictor(nn.Module):
    """Prime gap predictor with PAC elastic tension."""
    
    def __init__(self, window_size: int = 10, n_strands: int = 8,
                 elastic_strength: float = 1.0, lr: float = 0.01, device=None):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.window_size = window_size
        
        self.field = PACElasticMobiusField(
            n_strands=n_strands,
            input_size=window_size,
            output_size=1,
            elastic_strength=elastic_strength,
            device=self.device
        )
        
        self.register_buffer('gap_mean', torch.tensor(0.0, device=device))
        self.register_buffer('gap_std', torch.tensor(1.0, device=device))
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Tracking
        self.pred_losses: List[float] = []
        self.total_losses: List[float] = []
    
    def set_stats(self, mean: float, std: float):
        self.gap_mean = torch.tensor(mean, device=self.device)
        self.gap_std = torch.tensor(std, device=self.device)
    
    def train_batch(self, windows: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Train one batch, returning losses."""
        windows_norm = (windows - self.gap_mean) / (self.gap_std + 1e-8)
        targets_norm = (targets - self.gap_mean) / (self.gap_std + 1e-8)
        
        preds_norm, elastic_loss = self.field(windows_norm)
        preds_norm = preds_norm.squeeze(-1)
        
        # Prediction loss
        pred_loss = nn.functional.mse_loss(preds_norm, targets_norm)
        
        # Total loss = prediction + elastic tension
        total_loss = pred_loss + elastic_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        with torch.no_grad():
            preds = preds_norm * (self.gap_std + 1e-8) + self.gap_mean
            mae = torch.abs(preds - targets).mean().item()
        
        self.pred_losses.append(pred_loss.item())
        self.total_losses.append(total_loss.item())
        
        return {
            'pred_loss': pred_loss.item(),
            'elastic_loss': elastic_loss.item(),
            'total_loss': total_loss.item(),
            'mae': mae,
            'learned_ratio': self.field.get_learned_ratio()
        }


# ============================================================================
# EXPERIMENT
# ============================================================================

def create_batched_data(gaps: np.ndarray, window_size: int, device: torch.device):
    n = len(gaps) - window_size
    windows = np.array([gaps[i:i+window_size] for i in range(n)])
    targets = gaps[window_size:window_size + n]
    return (torch.tensor(windows, dtype=torch.float32, device=device),
            torch.tensor(targets, dtype=torch.float32, device=device))


def run_experiment(n_primes: int = 50_000, window_size: int = 10, 
                   n_strands: int = 8, elastic_strength: float = 1.0,
                   lr: float = 0.01, batch_size: int = 256, n_epochs: int = 30):
    
    print("=" * 70)
    print("EXP 30: PAC Elastic Loss - Does the Network Discover φ?")
    print("=" * 70)
    
    device = setup_gpu()
    
    # Generate data
    limit = int(n_primes * (np.log(n_primes) + np.log(np.log(n_primes + 10)) + 2))
    primes = sieve_of_eratosthenes(limit)[:n_primes]
    gaps = np.diff(primes).astype(np.float32)
    
    print(f"\nDataset: {len(primes):,} primes, {len(gaps):,} gaps")
    print(f"Gap stats: mean={gaps.mean():.2f}, std={gaps.std():.2f}")
    
    windows, targets = create_batched_data(gaps, window_size, device)
    n_samples = len(targets)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Create model
    predictor = PACElasticPredictor(window_size, n_strands, elastic_strength, lr, device)
    predictor.set_stats(float(gaps.mean()), float(gaps.std()))
    
    print(f"Model: {n_strands} strands, elastic_strength={elastic_strength}")
    print(f"Training: {n_samples:,} samples, {n_epochs} epochs")
    print(f"\n🎯 KEY TEST: Does learned split ratio → 1/φ ≈ {PHI_INV:.4f}?")
    print()
    
    print(f"{'Epoch':>5} | {'PredLoss':>8} | {'Elastic':>8} | {'MAE':>6} | {'Ratio':>6} | {'vs φ⁻¹':>7}")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples, device=device)
        windows_shuffled = windows[perm]
        targets_shuffled = targets[perm]
        
        epoch_pred_loss = []
        epoch_elastic = []
        epoch_mae = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            result = predictor.train_batch(
                windows_shuffled[start_idx:end_idx],
                targets_shuffled[start_idx:end_idx]
            )
            epoch_pred_loss.append(result['pred_loss'])
            epoch_elastic.append(result['elastic_loss'])
            epoch_mae.append(result['mae'])
        
        state = predictor.field.get_state()
        ratio = state['learned_ratio']
        ratio_diff = abs(ratio - PHI_INV)
        
        # Highlight when ratio is close to φ⁻¹
        marker = "✓" if ratio_diff < 0.05 else " "
        
        print(f"{epoch+1:>5} | {np.mean(epoch_pred_loss):>8.4f} | {np.mean(epoch_elastic):>8.5f} | "
              f"{np.mean(epoch_mae):>6.2f} | {ratio:>6.4f} | {ratio_diff:>6.4f} {marker}")
    
    elapsed = time.time() - start_time
    
    # Final evaluation
    print("\n" + "=" * 60)
    predictor.eval()
    
    with torch.no_grad():
        windows_norm = (windows - predictor.gap_mean) / (predictor.gap_std + 1e-8)
        preds_norm, _ = predictor.field(windows_norm)
        preds = preds_norm.squeeze(-1) * (predictor.gap_std + 1e-8) + predictor.gap_mean
        
        final_mae = torch.abs(preds - targets).mean().item()
    
    baseline_mae = gaps.std()
    improvement = (baseline_mae - final_mae) / baseline_mae * 100
    
    final_state = predictor.field.get_state()
    
    print(f"\nFINAL RESULTS ({elapsed:.1f}s)")
    print(f"  Final MAE: {final_mae:.2f}")
    print(f"  Baseline MAE: {baseline_mae:.2f}")
    print(f"  Improvement: {improvement:.2f}%")
    
    print(f"\n🎯 PHI DISCOVERY TEST:")
    print(f"  Target ratio (1/φ): {PHI_INV:.6f}")
    print(f"  Learned ratio:      {final_state['learned_ratio']:.6f}")
    print(f"  Difference:         {final_state['ratio_vs_phi']:.6f}")
    
    if final_state['ratio_vs_phi'] < 0.02:
        print(f"  ✓ CONVERGED TO φ! The network discovered the golden ratio.")
    elif final_state['ratio_vs_phi'] < 0.05:
        print(f"  ~ CLOSE to φ. Suggestive but not definitive.")
    else:
        print(f"  ✗ DID NOT converge to φ. Learned different optimal ratio.")
    
    print(f"\nPAC ELASTIC STATE:")
    print(f"  PAC anchor: {final_state['pac_anchor']:.4f}")
    print(f"  PAC current: {final_state['pac_total']:.4f}")
    print(f"  PAC deviation: {final_state['pac_deviation']:.6f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Learned ratio over training
    ax = axes[0, 0]
    ax.plot(predictor.field.split_ratios, 'b-', lw=1, alpha=0.7)
    ax.axhline(y=PHI_INV, color='gold', linestyle='--', linewidth=2, label=f'1/φ = {PHI_INV:.4f}')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Initial (0.5)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Learned Split Ratio')
    ax.set_title('Does the Network Discover φ?')
    ax.legend()
    ax.set_ylim(0.4, 0.7)
    
    # 2. PAC total over training
    ax = axes[0, 1]
    ax.plot(predictor.field.pac_totals, 'g-', lw=1, alpha=0.7)
    ax.axhline(y=final_state['pac_anchor'], color='red', linestyle='--', label='PAC Anchor')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('PAC Total')
    ax.set_title('PAC Conservation (Elastic Tension)')
    ax.legend()
    
    # 3. Elastic loss over training
    ax = axes[1, 0]
    ax.plot(predictor.field.elastic_losses, 'r-', lw=1, alpha=0.7)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Elastic Loss')
    ax.set_title('Elastic Tension Over Time')
    ax.set_yscale('log')
    
    # 4. Prediction loss over training
    ax = axes[1, 1]
    window = min(50, len(predictor.pred_losses) // 10)
    if window > 1:
        smoothed = np.convolve(predictor.pred_losses, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, 'b-', lw=1)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Prediction Loss')
    ax.set_title('Training Progress')
    
    plt.tight_layout()
    fig_path = Path(__file__).parent / 'figures' / f'exp_30_pac_elastic_{n_primes}.png'
    fig_path.parent.mkdir(exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n📊 Saved: {fig_path}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'experiment': 'exp_30_pac_elastic',
        'timestamp': timestamp,
        'config': {
            'n_primes': n_primes, 'n_strands': n_strands,
            'elastic_strength': elastic_strength, 'n_epochs': n_epochs
        },
        'final_mae': float(final_mae),
        'baseline_mae': float(baseline_mae),
        'improvement_pct': float(improvement),
        'phi_discovery': {
            'target_ratio': float(PHI_INV),
            'learned_ratio': float(final_state['learned_ratio']),
            'difference': float(final_state['ratio_vs_phi']),
            'converged_to_phi': final_state['ratio_vs_phi'] < 0.02
        },
        'pac_state': {
            'anchor': float(final_state['pac_anchor']),
            'final': float(final_state['pac_total']),
            'deviation': float(final_state['pac_deviation'])
        },
        'elapsed_seconds': elapsed
    }
    
    save_path = Path(__file__).parent / 'results' / f'exp_30_{n_primes}_{timestamp}.json'
    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {save_path}")
    
    return predictor, results


def run_elastic_strength_sweep():
    """Test different elastic strengths to find optimal tension."""
    print("\n" + "=" * 70)
    print("ELASTIC STRENGTH SWEEP: Finding Optimal PAC Tension")
    print("=" * 70 + "\n")
    
    strengths = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    results = []
    
    for strength in strengths:
        print(f"\n--- Elastic Strength: {strength} ---")
        _, result = run_experiment(
            n_primes=30_000,
            elastic_strength=strength,
            n_epochs=20
        )
        results.append({
            'strength': strength,
            'mae': result['final_mae'],
            'improvement': result['improvement_pct'],
            'learned_ratio': result['phi_discovery']['learned_ratio'],
            'ratio_diff': result['phi_discovery']['difference']
        })
    
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'Strength':>10} | {'MAE':>6} | {'Improve':>8} | {'Ratio':>6} | {'vs φ⁻¹':>7}")
    print("-" * 50)
    for r in results:
        marker = "✓" if r['ratio_diff'] < 0.05 else " "
        print(f"{r['strength']:>10.1f} | {r['mae']:>6.2f} | {r['improvement']:>7.1f}% | "
              f"{r['learned_ratio']:>6.4f} | {r['ratio_diff']:>6.4f} {marker}")
    
    return results


if __name__ == "__main__":
    # Single experiment with detailed tracking
    predictor, results = run_experiment(
        n_primes=50_000,
        window_size=10,
        n_strands=8,
        elastic_strength=1.0,
        lr=0.01,
        batch_size=256,
        n_epochs=30
    )
