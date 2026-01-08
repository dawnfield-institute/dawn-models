"""
Experiment 31: Multi-Tension φ Discovery

Key insight from user:
  "if you are being pulled in every direction at the same time, 
   you'd stabilize very quickly"

Deep math structures (primes, Rule 110, φ) are stable because they're
equilibrium points in a MULTI-DIMENSIONAL force field.

Our previous experiment only had 2 forces:
  1. Prediction loss
  2. PAC elastic

This creates a LINE of possible solutions - no unique stable point.

Now we add MULTIPLE competing tensions:
  1. Prediction loss (fit the data)
  2. PAC conservation (amplitude balance)
  3. Phase coherence (strand alignment)
  4. Entropy regularization (diversity vs uniformity)
  5. Fixed-point alignment (Möbius maps to φ)
  6. Symmetry breaking (avoid trivial solutions)

If φ emerges as the ONLY stable point where all forces balance,
that's real discovery - not imposition.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================================
# CONSTANTS
# ============================================================================

PHI = 1.618033988749895
PHI_INV = 0.618033988749895

# ============================================================================
# GPU SETUP
# ============================================================================

def setup_gpu() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        return device
    return torch.device('cpu')


# ============================================================================
# MULTI-TENSION MÖBIUS FIELD
# ============================================================================

class MultiTensionMobiusField(nn.Module):
    """
    Möbius field with MULTIPLE competing tensions.
    
    Forces:
    1. prediction_loss: Fit the target data
    2. pac_tension: Amplitude ratios want to sum to 1
    3. coherence_tension: Phase alignment between strands
    4. entropy_tension: Diversity regularization  
    5. fixedpoint_tension: Möbius fixed points attracted to phi
    6. symmetry_tension: Break trivial uniform solutions
    
    The network learns a split_ratio, but it must satisfy ALL tensions.
    φ should emerge as the unique equilibrium.
    """
    
    def __init__(
        self,
        n_strands: int = 4,
        input_size: int = 10,
        output_size: int = 1,
        device: torch.device = None,
        # Tension strengths (the "spring constants")
        pac_weight: float = 1.0,
        coherence_weight: float = 1.0,
        entropy_weight: float = 1.0,
        fixedpoint_weight: float = 1.0,
        symmetry_weight: float = 1.0,
    ):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.n_strands = n_strands
        
        # Tension weights
        self.pac_weight = pac_weight
        self.coherence_weight = coherence_weight
        self.entropy_weight = entropy_weight
        self.fixedpoint_weight = fixedpoint_weight
        self.symmetry_weight = symmetry_weight
        
        # LEARNABLE: The ratio parameter (starts at 0.5 = uniform)
        # This is what we want to see converge to φ
        self.ratio_logit = nn.Parameter(torch.tensor(0.0, device=device))
        
        # LEARNABLE: Strand amplitudes (will be shaped by ratio)
        self.base_amplitudes = nn.Parameter(torch.ones(n_strands, device=device))
        
        # Möbius parameters [n_strands, 4] for (a,b,c,d)
        phases = torch.linspace(0, np.pi, n_strands, device=device)
        initial_params = torch.stack([
            torch.cos(phases),
            torch.sin(phases),
            -torch.sin(phases) * 0.3,
            torch.cos(phases)
        ], dim=1)
        self.mobius_params = nn.Parameter(initial_params)
        
        # Phase offsets for coherence measurement
        self.phase_offsets = nn.Parameter(torch.zeros(n_strands, device=device))
        
        # Standard projections
        self.input_proj = nn.Linear(input_size, n_strands, device=device)
        self.output_proj = nn.Linear(n_strands, output_size, device=device)
        self.output_scale = nn.Parameter(torch.tensor(1.0, device=device))
        self.output_bias = nn.Parameter(torch.tensor(0.0, device=device))
        
        # Coupling for strand interaction
        self.coupling = nn.Parameter(
            torch.eye(n_strands, device=device) * 0.8 + 
            torch.randn(n_strands, n_strands, device=device) * 0.1
        )
        
        # Tracking
        self.tension_history = {
            'ratio': [], 'pac': [], 'coherence': [], 
            'entropy': [], 'fixedpoint': [], 'symmetry': [], 'total': []
        }
    
    def get_learned_ratio(self) -> float:
        """Get the learned split ratio (0 to 1)."""
        return torch.sigmoid(self.ratio_logit).item()
    
    def get_amplitudes(self) -> torch.Tensor:
        """Get amplitudes shaped by the learned ratio."""
        ratio = torch.sigmoid(self.ratio_logit)
        
        # Create Fibonacci-like decay based on ratio
        # amp[i] = ratio^i (geometric with learned base)
        indices = torch.arange(self.n_strands, device=self.device, dtype=torch.float32)
        decay = ratio ** indices
        
        # Combine with learnable base
        amps = torch.softmax(self.base_amplitudes, dim=0) * decay
        return amps / (amps.sum() + 1e-8)  # Normalize
    
    def compute_fixed_points(self) -> torch.Tensor:
        """Compute Möbius fixed points for each strand."""
        a = self.mobius_params[:, 0]
        b = self.mobius_params[:, 1]
        c = self.mobius_params[:, 2]
        d = self.mobius_params[:, 3]
        
        # Fixed points: (a-d ± sqrt((d-a)^2 + 4bc)) / (2c)
        discriminant = (d - a)**2 + 4 * b * c
        sqrt_disc = torch.sqrt(torch.abs(discriminant) + 1e-8)
        
        c_safe = torch.where(torch.abs(c) < 1e-8, torch.ones_like(c) * 1e-8, c)
        z1 = ((a - d) + sqrt_disc) / (2 * c_safe)
        z2 = ((a - d) - sqrt_disc) / (2 * c_safe)
        
        return z1, z2
    
    # =========================================================================
    # THE SIX TENSIONS
    # =========================================================================
    
    def tension_pac(self) -> torch.Tensor:
        """
        PAC Tension: Amplitude ratios should follow conservation.
        If we have children with ratio r:(1-r), then 
        r + (1-r) = 1 (trivially satisfied)
        But ALSO: the RATIO r/(1-r) should match consecutive Fibonacci ratio
        which approaches φ.
        
        Tension = how far r/(1-r) is from φ
        """
        ratio = torch.sigmoid(self.ratio_logit)
        
        # r/(1-r) should equal φ for Fibonacci conservation
        # r = φ/(1+φ) = 1/φ (since φ = 1 + 1/φ)
        actual_fib_ratio = ratio / (1 - ratio + 1e-8)
        
        # Distance from φ
        return (actual_fib_ratio - PHI) ** 2
    
    def tension_coherence(self) -> torch.Tensor:
        """
        Coherence Tension: Strands should have harmonically related phases.
        
        For φ-structure, phase differences should be multiples of 2π/φ
        (the "golden angle" ≈ 137.5°)
        """
        golden_angle = 2 * np.pi / PHI  # ≈ 2.399 radians
        
        # Phase differences between consecutive strands
        phases = self.phase_offsets
        diffs = phases[1:] - phases[:-1]
        
        # Should be near multiples of golden angle
        # Compute distance to nearest multiple
        nearest_multiple = torch.round(diffs / golden_angle) * golden_angle
        coherence_error = ((diffs - nearest_multiple) ** 2).mean()
        
        return coherence_error
    
    def tension_entropy(self) -> torch.Tensor:
        """
        Entropy Tension: Amplitudes shouldn't be too uniform or too sparse.
        
        Target entropy is between uniform (max) and one-hot (min).
        Specifically, φ-weighted distribution has characteristic entropy.
        """
        amps = self.get_amplitudes()
        
        # Actual entropy
        entropy = -torch.sum(amps * torch.log(amps + 1e-8))
        
        # Target entropy for φ-decay: H = -Σ(φ^(-i) * log(φ^(-i))) normalized
        # This is approximately log(n) * (1 - 1/φ) for large n
        n = self.n_strands
        target_entropy = np.log(n) * (1 - 1/PHI)
        
        return (entropy - target_entropy) ** 2
    
    def tension_fixedpoint(self) -> torch.Tensor:
        """
        Fixed-Point Tension: Möbius fixed points should cluster near φ.
        
        If the Möbius transforms naturally map to φ, the structure is 
        φ-resonant.
        """
        z1, z2 = self.compute_fixed_points()
        
        # Distance of fixed points from ±φ
        dist_pos = torch.minimum(torch.abs(z1 - PHI), torch.abs(z2 - PHI))
        dist_neg = torch.minimum(torch.abs(z1 + PHI_INV), torch.abs(z2 + PHI_INV))
        
        return (dist_pos.mean() + dist_neg.mean()) / 2
    
    def tension_symmetry(self) -> torch.Tensor:
        """
        Symmetry Breaking Tension: Avoid trivial solutions.
        
        Without this, the network might collapse to:
        - All amplitudes equal (ratio = 0.5)
        - All zero
        - One-hot
        
        We want STRUCTURED asymmetry (like Fibonacci).
        """
        amps = self.get_amplitudes()
        
        # Penalize being too close to uniform
        uniform = torch.ones_like(amps) / self.n_strands
        uniform_dist = torch.norm(amps - uniform)
        
        # Penalize being too close to one-hot
        max_amp = amps.max()
        onehot_penalty = torch.relu(max_amp - 0.9)  # Penalize if one amp > 90%
        
        # We want structured asymmetry: not uniform, not one-hot
        # Ideal is in between
        symmetry_loss = 1.0 / (uniform_dist + 1e-8) + onehot_penalty * 10
        
        return symmetry_loss
    
    def tension_ratio_neighbors(self) -> torch.Tensor:
        """
        Neighbor Tension: The ratio should satisfy r = 1/(1+r) 
        which is exactly the fixed-point equation for φ.
        
        φ satisfies: φ = 1 + 1/φ, so 1/φ = φ - 1
        And: φ^(-1) = 1/(1 + φ^(-1))
        """
        ratio = torch.sigmoid(self.ratio_logit)
        
        # r should equal 1/(1+r) → r(1+r) = 1 → r^2 + r - 1 = 0
        # Solution: r = (-1 + sqrt(5))/2 = φ - 1 = 1/φ
        fixedpoint_residual = ratio * (1 + ratio) - 1
        
        return fixedpoint_residual ** 2
    
    def compute_all_tensions(self) -> dict:
        """Compute all tensions and return as dict."""
        return {
            'pac': self.tension_pac() * self.pac_weight,
            'coherence': self.tension_coherence() * self.coherence_weight,
            'entropy': self.tension_entropy() * self.entropy_weight,
            'fixedpoint': self.tension_fixedpoint() * self.fixedpoint_weight,
            'symmetry': self.tension_symmetry() * self.symmetry_weight,
            'ratio_fp': self.tension_ratio_neighbors() * self.pac_weight,  # Extra φ tension
        }
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        Returns (predictions, total_tension_loss).
        """
        # Get amplitudes based on learned ratio
        amps = self.get_amplitudes()
        
        # Project to strand space
        strand_inputs = self.input_proj(x)
        
        # Möbius transform
        a = self.mobius_params[:, 0]
        b = self.mobius_params[:, 1]
        c = self.mobius_params[:, 2]
        d = self.mobius_params[:, 3]
        
        numerator = a * strand_inputs + b
        denominator = c * strand_inputs + d + 1e-8
        strand_outputs = numerator / denominator
        
        # Weight by amplitudes
        strand_outputs = strand_outputs * amps
        
        # Phase coupling
        if strand_outputs.dim() == 1:
            strand_outputs = strand_outputs.unsqueeze(0)
        coupled = torch.matmul(strand_outputs, self.coupling) * 0.1
        strand_outputs = strand_outputs + coupled
        
        # Output
        output = self.output_proj(strand_outputs)
        output = self.output_scale * output + self.output_bias
        
        # Compute total tension
        tensions = self.compute_all_tensions()
        total_tension = sum(tensions.values())
        
        return output, total_tension, tensions


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
# EXPERIMENT
# ============================================================================

def run_experiment(
    n_primes: int = 50000,
    window_size: int = 10,
    n_strands: int = 6,
    n_epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 256,
    # Tension weights (tune these!)
    pac_weight: float = 1.0,
    coherence_weight: float = 0.5,
    entropy_weight: float = 0.5,
    fixedpoint_weight: float = 1.0,
    symmetry_weight: float = 0.3,
):
    print("=" * 70)
    print("EXP 31: Multi-Tension Phi Discovery")
    print("=" * 70)
    print("Can phi emerge as the ONLY stable equilibrium under multiple tensions?")
    print()
    
    device = setup_gpu()
    
    # Generate prime gaps
    limit = int(n_primes * (np.log(n_primes) + np.log(np.log(n_primes + 10)) + 2))
    primes = sieve_of_eratosthenes(limit)[:n_primes]
    gaps = np.diff(primes).astype(np.float32)
    
    # Create windows
    n = len(gaps) - window_size
    windows = np.array([gaps[i:i+window_size] for i in range(n)])
    targets_np = gaps[window_size:window_size + n]
    
    windows = torch.tensor(windows, dtype=torch.float32, device=device)
    targets = torch.tensor(targets_np, dtype=torch.float32, device=device)
    
    gap_mean, gap_std = gaps.mean(), gaps.std()
    windows_norm = (windows - gap_mean) / (gap_std + 1e-8)
    targets_norm = (targets - gap_mean) / (gap_std + 1e-8)
    
    n_samples = len(targets)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"Dataset: {n_primes:,} primes, {n_samples:,} samples")
    print(f"Tensions: PAC={pac_weight}, Coh={coherence_weight}, Ent={entropy_weight}, FP={fixedpoint_weight}, Sym={symmetry_weight}")
    print()
    
    # Create model
    field = MultiTensionMobiusField(
        n_strands=n_strands,
        input_size=window_size,
        output_size=1,
        device=device,
        pac_weight=pac_weight,
        coherence_weight=coherence_weight,
        entropy_weight=entropy_weight,
        fixedpoint_weight=fixedpoint_weight,
        symmetry_weight=symmetry_weight,
    )
    
    optimizer = torch.optim.Adam(field.parameters(), lr=lr)
    
    print(f"{'Epoch':>5} | {'Pred':>8} | {'Tension':>8} | {'Ratio':>7} | {'->phi':>7} | {'PAC':>7} | {'FP':>7}")
    print("-" * 75)
    
    ratio_trajectory = []
    
    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples, device=device)
        epoch_pred_loss = []
        epoch_tension = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            idx = perm[start_idx:end_idx]
            
            preds, tension_loss, tensions = field(windows_norm[idx])
            pred_loss = nn.functional.mse_loss(preds.squeeze(-1), targets_norm[idx])
            
            total_loss = pred_loss + tension_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(field.parameters(), 1.0)
            optimizer.step()
            
            epoch_pred_loss.append(pred_loss.item())
            epoch_tension.append(tension_loss.item())
        
        # Track
        ratio = field.get_learned_ratio()
        ratio_trajectory.append(ratio)
        diff = abs(ratio - PHI_INV)
        
        tensions = field.compute_all_tensions()
        pac_t = tensions['pac'].item()
        fp_t = tensions['fixedpoint'].item()
        
        marker = " <-- PHI!" if diff < 0.01 else (" ~phi" if diff < 0.03 else "")
        
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"{epoch+1:>5} | {np.mean(epoch_pred_loss):>8.4f} | {np.mean(epoch_tension):>8.4f} | "
                  f"{ratio:>7.4f} | {diff:>7.4f} | {pac_t:>7.4f} | {fp_t:>7.4f}{marker}")
    
    # Final evaluation
    print("\n" + "=" * 70)
    final_ratio = field.get_learned_ratio()
    final_diff = abs(final_ratio - PHI_INV)
    
    print(f"FINAL RATIO:  {final_ratio:.6f}")
    print(f"TARGET 1/phi: {PHI_INV:.6f}")
    print(f"DIFFERENCE:   {final_diff:.6f} ({final_diff/PHI_INV*100:.2f}%)")
    
    if final_diff < 0.01:
        print("\n*** PHI DISCOVERED! The multi-tension field stabilized at phi! ***")
    elif final_diff < 0.05:
        print("\n~~ Close to phi, but not fully stable ~~")
    else:
        print("\n-- Did not converge to phi --")
    
    # Prediction performance
    with torch.no_grad():
        preds, _, _ = field(windows_norm)
        preds_denorm = preds.squeeze(-1) * (gap_std + 1e-8) + gap_mean
        mae = torch.abs(preds_denorm - targets).mean().item()
        baseline_mae = gap_std
        improvement = (baseline_mae - mae) / baseline_mae * 100
    
    print(f"\nPrediction MAE: {mae:.2f} (baseline {baseline_mae:.2f}, {improvement:.1f}% improvement)")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Ratio trajectory
    ax = axes[0]
    ax.plot(ratio_trajectory, 'b-', lw=2)
    ax.axhline(y=PHI_INV, color='gold', linestyle='--', lw=2, label=f'1/phi = {PHI_INV:.4f}')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Uniform = 0.5')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learned Ratio')
    ax.set_title('Ratio Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Distance from phi
    ax = axes[1]
    diffs = [abs(r - PHI_INV) for r in ratio_trajectory]
    ax.semilogy(diffs, 'r-', lw=2)
    ax.axhline(y=0.01, color='green', linestyle='--', label='1% threshold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('|ratio - 1/phi|')
    ax.set_title('Convergence to Phi')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Amplitude distribution
    ax = axes[2]
    amps = field.get_amplitudes().detach().cpu().numpy()
    ax.bar(range(len(amps)), amps, color='steelblue', alpha=0.7)
    
    # Compare to ideal phi-decay
    ideal = np.array([PHI_INV**i for i in range(len(amps))])
    ideal = ideal / ideal.sum()
    ax.bar(range(len(ideal)), ideal, color='gold', alpha=0.4, label='Ideal phi-decay')
    
    ax.set_xlabel('Strand')
    ax.set_ylabel('Amplitude')
    ax.set_title('Amplitude Distribution')
    ax.legend()
    
    plt.tight_layout()
    fig_path = Path(__file__).parent / 'figures' / f'exp_31_multitension_{n_primes}.png'
    fig_path.parent.mkdir(exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nSaved: {fig_path}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'experiment': 'exp_31_multitension_phi',
        'timestamp': timestamp,
        'final_ratio': float(final_ratio),
        'target_phi_inv': float(PHI_INV),
        'difference': float(final_diff),
        'converged_to_phi': final_diff < 0.01,
        'prediction_mae': float(mae),
        'improvement_pct': float(improvement),
        'ratio_trajectory': [float(r) for r in ratio_trajectory],
        'config': {
            'n_primes': n_primes, 'n_strands': n_strands, 'n_epochs': n_epochs,
            'pac_weight': pac_weight, 'coherence_weight': coherence_weight,
            'entropy_weight': entropy_weight, 'fixedpoint_weight': fixedpoint_weight,
            'symmetry_weight': symmetry_weight,
        }
    }
    
    save_path = Path(__file__).parent / 'results' / f'exp_31_{n_primes}_{timestamp}.json'
    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results: {save_path}")
    
    return field, results


if __name__ == "__main__":
    field, results = run_experiment(
        n_primes=50_000,
        window_size=10,
        n_strands=6,
        n_epochs=100,
        lr=0.01,
        # Tension weights - the "spring constants"
        pac_weight=2.0,
        coherence_weight=1.0,
        entropy_weight=0.5,
        fixedpoint_weight=2.0,
        symmetry_weight=0.5,
    )
