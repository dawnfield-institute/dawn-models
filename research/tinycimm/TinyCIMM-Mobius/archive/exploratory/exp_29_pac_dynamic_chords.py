"""
Experiment 29: PAC-Conserved Dynamic Chords

Key insight from Dawn Field Theory:
  PAC: f(Parent) = Σ f(Children)

Applied to Möbius strands:
- Total φ-frequency is CONSERVED across chord transitions
- Strands can SPLIT (1 → 2) when exploring
- Strands can MERGE (2 → 1) when locking
- Split/merge ratios naturally tend toward φ (Fibonacci)

This creates an adaptive architecture:
- High entropy → strands differentiate (more dimensions)
- Low entropy → strands integrate (fewer dimensions)
- PAC ensures no φ-energy is created or destroyed

The field "breathes" - expanding to explore, contracting to lock.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from collections import Counter
from dataclasses import dataclass

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
# PAC CHORD EVENT TRACKING
# ============================================================================

@dataclass
class ChordEvent:
    """Record of a PAC-conserved chord transition."""
    step: int
    event_type: str  # 'split', 'merge', 'stable'
    before_freq: float
    after_freq: float
    conservation_error: float  # Should be ~0 for PAC
    trigger: str  # What caused the transition


class PACChordHistory:
    """Track chord dynamics and PAC conservation."""
    
    def __init__(self):
        self.events: List[ChordEvent] = []
        self.freq_timeline: List[float] = []
        self.n_strands_timeline: List[int] = []
        self.conservation_errors: List[float] = []
    
    def record(self, event: ChordEvent):
        self.events.append(event)
        self.conservation_errors.append(event.conservation_error)
    
    def get_summary(self) -> Dict:
        if not self.events:
            return {'n_events': 0}
        
        splits = [e for e in self.events if e.event_type == 'split']
        merges = [e for e in self.events if e.event_type == 'merge']
        
        return {
            'n_events': len(self.events),
            'n_splits': len(splits),
            'n_merges': len(merges),
            'mean_conservation_error': float(np.mean(self.conservation_errors)),
            'max_conservation_error': float(np.max(self.conservation_errors)) if self.conservation_errors else 0
        }


# ============================================================================
# PAC-CONSERVED MÖBIUS STRAND FIELD
# ============================================================================

class PACMobiusStrandField(nn.Module):
    """
    Möbius Strand Field with PAC-conserved dynamic chords.
    
    Key dynamics:
    1. Each strand has (a,b,c,d) Möbius params + amplitude weight
    2. Total φ-frequency × amplitude is conserved
    3. Strands can split/merge based on entropy
    4. Split ratio tends toward φ (Fibonacci emergence)
    
    Conservation law:
        Σ(freq_i × amp_i) = constant
    """
    
    def __init__(
        self,
        initial_strands: int = 4,
        max_strands: int = 16,
        min_strands: int = 2,
        input_size: int = 10,
        output_size: int = 1,
        device: torch.device = None,
        split_threshold: float = 0.5,  # Entropy threshold for split (higher = less splits)
        merge_threshold: float = 0.7,  # Phase coherence threshold for merge (0.3 required)
    ):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.max_strands = max_strands
        self.min_strands = min_strands
        self.split_threshold = split_threshold
        self.merge_threshold = merge_threshold
        
        # Dynamic strand parameters [n_strands, 4] for (a,b,c,d)
        n = initial_strands
        phases = torch.tensor([i * np.pi / 4 for i in range(n)], device=device)
        
        initial_params = torch.stack([
            torch.cos(phases),           # a
            torch.sin(phases),           # b  
            -torch.sin(phases) * 0.5,    # c
            torch.cos(phases)            # d
        ], dim=1)
        
        self.mobius_params = nn.Parameter(initial_params)
        
        # Amplitude weights (for PAC conservation)
        self.amplitudes = nn.Parameter(torch.ones(n, device=device) / n)
        
        # Input/output projections (will be resized dynamically)
        self.input_proj = nn.Linear(input_size, n, device=device)
        self.output_proj = nn.Linear(n, output_size, device=device)
        self.output_scale = nn.Parameter(torch.tensor(1.0, device=device))
        self.output_bias = nn.Parameter(torch.tensor(0.0, device=device))
        
        # Coupling matrix
        self.coupling = nn.Parameter(
            torch.eye(n, device=device) * 0.9 + torch.randn(n, n, device=device) * 0.1
        )
        self.coupling_strength = 0.1
        
        # PAC conservation target (set on first forward)
        self.register_buffer('pac_target', torch.tensor(0.0, device=device))
        self.pac_initialized = False
        
        # History
        self.history = PACChordHistory()
        self.step_count = 0
        
        # Recent entropy for split/merge decisions
        self.entropy_buffer: List[float] = []
        self.entropy_window = 20
    
    @property
    def n_strands(self) -> int:
        return self.mobius_params.shape[0]
    
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
        """Compute total PAC-conserved quantity: Σ(freq × amp)."""
        freqs = self.compute_frequencies()
        amps = torch.softmax(self.amplitudes, dim=0)  # Normalize amplitudes
        return torch.sum(freqs * amps)
    
    def compute_entropy(self, activations: torch.Tensor) -> float:
        """Compute activation entropy (high = exploring, low = locked)."""
        with torch.no_grad():
            probs = torch.softmax(torch.abs(activations.mean(dim=0)), dim=0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            return entropy.item()
    
    def compute_phase_coherence(self) -> float:
        """How aligned are strand phases? (high = can merge)."""
        with torch.no_grad():
            a = self.mobius_params[:, 0]
            b = self.mobius_params[:, 1]
            c = self.mobius_params[:, 2]
            d = self.mobius_params[:, 3]
            
            discriminant = (d - a)**2 + 4 * c * b
            sqrt_disc = torch.sqrt(torch.abs(discriminant) + 1e-8)
            c_safe = torch.where(torch.abs(c) < 1e-8, torch.ones_like(c) * 1e-8, c)
            z1 = (-(d - a) + sqrt_disc) / (2 * c_safe)
            z2 = (-(d - a) - sqrt_disc) / (2 * c_safe)
            
            phases = torch.atan2(z1 - z2, torch.ones_like(z1))
            phase_std = torch.std(phases).item()
            
            return 1.0 / (1.0 + phase_std)
    
    def split_strand(self, strand_idx: int):
        """Split one strand into two (PAC-conserved).
        
        NOTE: Due to optimizer state issues with dynamic resizing during training,
        we only allow splits between epochs (tracked via deferred_split).
        """
        if self.n_strands >= self.max_strands:
            return False
        
        # Track the split request - will be applied between epochs
        if not hasattr(self, 'pending_splits'):
            self.pending_splits = []
        self.pending_splits.append(strand_idx)
        
        return True  # Request accepted
    
    def apply_pending_splits(self):
        """Apply all pending splits (call between epochs)."""
        if not hasattr(self, 'pending_splits') or not self.pending_splits:
            return False
        
        splits_applied = 0
        with torch.no_grad():
            for strand_idx in sorted(set(self.pending_splits), reverse=True):
                if self.n_strands >= self.max_strands:
                    break
                
                # Get parent strand
                parent_params = self.mobius_params[strand_idx].clone()
                parent_amp = self.amplitudes[strand_idx].clone()
                parent_freq = self.compute_frequencies()[strand_idx]
                
                # PAC split: parent amplitude → two children with φ ratio
                child1_amp = parent_amp * PHI_INV  # ~0.618
                child2_amp = parent_amp * (1 - PHI_INV)  # ~0.382
                
                # Slightly perturb parameters for differentiation
                child1_params = parent_params + torch.randn_like(parent_params) * 0.1
                child2_params = parent_params + torch.randn_like(parent_params) * 0.1
                
                # Build new parameter tensors
                new_params = torch.cat([
                    self.mobius_params[:strand_idx],
                    child1_params.unsqueeze(0),
                    child2_params.unsqueeze(0),
                    self.mobius_params[strand_idx+1:]
                ], dim=0)
                
                new_amps = torch.cat([
                    self.amplitudes[:strand_idx],
                    child1_amp.unsqueeze(0),
                    child2_amp.unsqueeze(0),
                    self.amplitudes[strand_idx+1:]
                ])
                
                # Update parameters
                self.mobius_params = nn.Parameter(new_params)
                self.amplitudes = nn.Parameter(new_amps)
                
                # Record event
                after_total = self.compute_pac_total()
                conservation_error = abs(after_total.item() - self.pac_target.item())
                
                self.history.record(ChordEvent(
                    step=self.step_count,
                    event_type='split',
                    before_freq=parent_freq.item(),
                    after_freq=after_total.item(),
                    conservation_error=conservation_error,
                    trigger='high_entropy'
                ))
                splits_applied += 1
        
        self.pending_splits.clear()
        if splits_applied > 0:
            self._resize_projections()
        return splits_applied > 0
    
    def merge_strands(self, idx1: int, idx2: int):
        """Merge two strands into one (PAC-conserved).
        
        NOTE: Due to optimizer state issues with dynamic resizing during training,
        we only allow merges between epochs (tracked via deferred_merge).
        """
        if self.n_strands <= self.min_strands:
            return False
        
        if not hasattr(self, 'pending_merges'):
            self.pending_merges = []
        self.pending_merges.append((min(idx1, idx2), max(idx1, idx2)))
        
        return True  # Request accepted
    
    def apply_pending_merges(self):
        """Apply all pending merges (call between epochs)."""
        if not hasattr(self, 'pending_merges') or not self.pending_merges:
            return False
        
        merges_applied = 0
        with torch.no_grad():
            # Only apply one merge per epoch to keep things stable
            idx1, idx2 = self.pending_merges[0]
            
            if self.n_strands > self.min_strands:
                # Get parent strands
                params1 = self.mobius_params[idx1]
                params2 = self.mobius_params[idx2]
                amp1 = self.amplitudes[idx1]
                amp2 = self.amplitudes[idx2]
                
                before_total = self.compute_pac_total()
                
                # PAC merge: weighted average of parameters
                total_amp = amp1 + amp2
                merged_params = (params1 * amp1 + params2 * amp2) / (total_amp + 1e-8)
                merged_amp = total_amp
                
                # Build new tensors (remove both, add merged)
                keep_mask = torch.ones(self.n_strands, dtype=torch.bool, device=self.device)
                keep_mask[idx1] = False
                keep_mask[idx2] = False
                
                new_params = torch.cat([
                    self.mobius_params[keep_mask],
                    merged_params.unsqueeze(0)
                ], dim=0)
                
                new_amps = torch.cat([
                    self.amplitudes[keep_mask],
                    merged_amp.unsqueeze(0)
                ])
                
                # Update
                self.mobius_params = nn.Parameter(new_params)
                self.amplitudes = nn.Parameter(new_amps)
                
                # Record
                after_total = self.compute_pac_total()
                conservation_error = abs(after_total.item() - before_total.item())
                
                self.history.record(ChordEvent(
                    step=self.step_count,
                    event_type='merge',
                    before_freq=before_total.item(),
                    after_freq=after_total.item(),
                    conservation_error=conservation_error,
                    trigger='high_coherence'
                ))
                merges_applied += 1
        
        self.pending_merges.clear()
        if merges_applied > 0:
            self._resize_projections()
        return merges_applied > 0
    
    def _resize_projections(self):
        """Resize input/output projections after split/merge."""
        n = self.n_strands
        in_features = self.input_proj.in_features
        out_features = self.output_proj.out_features
        
        # Create new projections with correct size
        new_input = nn.Linear(in_features, n, device=self.device)
        new_output = nn.Linear(n, out_features, device=self.device)
        
        # Initialize reasonably
        nn.init.xavier_uniform_(new_input.weight)
        nn.init.xavier_uniform_(new_output.weight)
        nn.init.zeros_(new_input.bias)
        nn.init.zeros_(new_output.bias)
        
        self.input_proj = new_input
        self.output_proj = new_output
        
        # Resize coupling matrix
        new_coupling = torch.eye(n, device=self.device) * 0.9 + torch.randn(n, n, device=self.device) * 0.1
        self.coupling = nn.Parameter(new_coupling)
    
    def maybe_adapt_chords(self, activations: torch.Tensor):
        """Check if we should split or merge based on entropy/coherence.
        
        NOTE: This only REQUESTS splits/merges. Actual application happens
        between epochs via apply_pending_splits/merges.
        """
        entropy = self.compute_entropy(activations)
        self.entropy_buffer.append(entropy)
        if len(self.entropy_buffer) > self.entropy_window:
            self.entropy_buffer.pop(0)
        
        if len(self.entropy_buffer) < self.entropy_window:
            return
        
        avg_entropy = np.mean(self.entropy_buffer)
        coherence = self.compute_phase_coherence()
        
        # High entropy + strands available → request split of weakest strand
        if avg_entropy > self.split_threshold and self.n_strands < self.max_strands:
            with torch.no_grad():
                weakest = torch.argmin(self.amplitudes).item()
                self.split_strand(weakest)
                return
        
        # High coherence + strands available → request merge of most similar
        if coherence > (1 - self.merge_threshold) and self.n_strands > self.min_strands:
            with torch.no_grad():
                n = self.n_strands
                min_dist = float('inf')
                merge_pair = (0, 1)
                
                for i in range(n):
                    for j in range(i+1, n):
                        dist = torch.norm(self.mobius_params[i] - self.mobius_params[j]).item()
                        if dist < min_dist:
                            min_dist = dist
                            merge_pair = (i, j)
                
                self.merge_strands(*merge_pair)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.step_count += 1
        
        # Initialize PAC target on first forward
        if not self.pac_initialized:
            self.pac_target = self.compute_pac_total().detach()
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
        
        # Check for chord adaptation (only during training)
        if self.training:
            self.maybe_adapt_chords(strand_outputs)
        
        # Output
        output = self.output_proj(strand_outputs)
        return self.output_scale * output + self.output_bias
    
    def get_state(self) -> Dict:
        """Get current chord state."""
        freqs = self.compute_frequencies()
        amps = torch.softmax(self.amplitudes, dim=0)
        
        return {
            'n_strands': self.n_strands,
            'mean_freq': freqs.mean().item(),
            'freq_std': freqs.std().item(),
            'pac_total': self.compute_pac_total().item(),
            'pac_target': self.pac_target.item(),
            'pac_error': abs(self.compute_pac_total().item() - self.pac_target.item()),
            'coherence': self.compute_phase_coherence(),
            'amp_entropy': -(amps * torch.log(amps + 1e-8)).sum().item()
        }


# ============================================================================
# PAC PRIME GAP PREDICTOR
# ============================================================================

class PACPrimeGapPredictor(nn.Module):
    """Prime gap predictor with PAC-conserved dynamic chords."""
    
    def __init__(self, window_size: int = 10, initial_strands: int = 4, 
                 max_strands: int = 16, lr: float = 0.01, device=None):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.window_size = window_size
        
        self.field = PACMobiusStrandField(
            initial_strands=initial_strands,
            max_strands=max_strands,
            input_size=window_size,
            output_size=1,
            device=self.device
        )
        
        self.register_buffer('gap_mean', torch.tensor(0.0, device=device))
        self.register_buffer('gap_std', torch.tensor(1.0, device=device))
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Tracking
        self.loss_history = []
        self.strand_history = []
        self.freq_history = []
        self.pac_error_history = []
    
    def set_stats(self, mean: float, std: float):
        self.gap_mean = torch.tensor(mean, device=self.device)
        self.gap_std = torch.tensor(std, device=self.device)
    
    def train_batch(self, windows: torch.Tensor, targets: torch.Tensor) -> Dict:
        windows_norm = (windows - self.gap_mean) / (self.gap_std + 1e-8)
        targets_norm = (targets - self.gap_mean) / (self.gap_std + 1e-8)
        
        preds_norm = self.field(windows_norm).squeeze(-1)
        loss = nn.functional.mse_loss(preds_norm, targets_norm)
        
        # Add PAC conservation regularization
        pac_error = abs(self.field.compute_pac_total() - self.field.pac_target)
        total_loss = loss + 0.1 * pac_error
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        with torch.no_grad():
            preds = preds_norm * (self.gap_std + 1e-8) + self.gap_mean
            mae = torch.abs(preds - targets).mean().item()
        
        self.loss_history.append(loss.item())
        self.strand_history.append(self.field.n_strands)
        self.freq_history.append(self.field.compute_frequencies().mean().item())
        self.pac_error_history.append(pac_error.item())
        
        return {'loss': loss.item(), 'mae': mae, 'n_strands': self.field.n_strands}


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
                   initial_strands: int = 4, max_strands: int = 16,
                   lr: float = 0.01, batch_size: int = 256, n_epochs: int = 20):
    
    print("=" * 70)
    print("EXP 29: PAC-Conserved Dynamic Chords for Prime Gap Prediction")
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
    predictor = PACPrimeGapPredictor(window_size, initial_strands, max_strands, lr, device)
    predictor.set_stats(float(gaps.mean()), float(gaps.std()))
    
    print(f"Model: {initial_strands}→{max_strands} strands, PAC-conserved")
    print(f"Training: {n_samples:,} samples, {n_epochs} epochs\n")
    
    print(f"{'Epoch':>5} | {'Loss':>8} | {'MAE':>6} | {'Strands':>7} | {'φ-Freq':>7} | {'Coherence':>9}")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # Apply any pending chord changes from previous epoch
        changed = False
        if predictor.field.apply_pending_splits():
            print(f"  → Split applied! Now {predictor.field.n_strands} strands")
            changed = True
        if predictor.field.apply_pending_merges():
            print(f"  → Merge applied! Now {predictor.field.n_strands} strands")
            changed = True
        
        # Recreate optimizer if architecture changed
        if changed:
            predictor.optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
        
        perm = torch.randperm(n_samples, device=device)
        windows_shuffled = windows[perm]
        targets_shuffled = targets[perm]
        
        epoch_losses = []
        epoch_maes = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            result = predictor.train_batch(
                windows_shuffled[start_idx:end_idx],
                targets_shuffled[start_idx:end_idx]
            )
            epoch_losses.append(result['loss'])
            epoch_maes.append(result['mae'])
        
        # Epoch summary with coherence
        state = predictor.field.get_state()
        coherence = predictor.field.compute_phase_coherence()
        print(f"{epoch+1:>5} | {np.mean(epoch_losses):>8.4f} | {np.mean(epoch_maes):>6.2f} | "
              f"{state['n_strands']:>7} | {state['mean_freq']:>7.4f} | {coherence:>7.4f}")
    
    elapsed = time.time() - start_time
    
    # Final evaluation
    print("\n" + "=" * 60)
    predictor.eval()
    
    with torch.no_grad():
        windows_norm = (windows - predictor.gap_mean) / (predictor.gap_std + 1e-8)
        preds_norm = predictor.field(windows_norm).squeeze(-1)
        preds = preds_norm * (predictor.gap_std + 1e-8) + predictor.gap_mean
        
        final_mae = torch.abs(preds - targets).mean().item()
    
    baseline_mae = gaps.std()
    improvement = (baseline_mae - final_mae) / baseline_mae * 100
    
    print(f"\nFINAL RESULTS ({elapsed:.1f}s)")
    print(f"  Final MAE: {final_mae:.2f}")
    print(f"  Baseline MAE: {baseline_mae:.2f}")
    print(f"  Improvement: {improvement:.2f}%")
    
    # Chord dynamics summary
    history = predictor.field.history.get_summary()
    print(f"\nCHORD DYNAMICS:")
    print(f"  Total events: {history.get('n_events', 0)}")
    print(f"  Splits: {history.get('n_splits', 0)}")
    print(f"  Merges: {history.get('n_merges', 0)}")
    print(f"  Mean PAC error: {history.get('mean_conservation_error', 0):.6f}")
    print(f"  Final strands: {predictor.field.n_strands}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Strand count over training
    ax = axes[0, 0]
    ax.plot(predictor.strand_history, 'b-', lw=1)
    ax.set_xlabel('Batch')
    ax.set_ylabel('Number of Strands')
    ax.set_title('Chord Dynamics (Strand Count)')
    ax.axhline(y=initial_strands, color='gray', linestyle='--', alpha=0.5)
    
    # 2. φ-Frequency evolution
    ax = axes[0, 1]
    ax.plot(predictor.freq_history, 'g-', lw=1)
    ax.axhline(y=PHI_INV, color='gold', linestyle='--', label=f'1/φ={PHI_INV:.3f}')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Mean φ-Frequency')
    ax.set_title('φ-Frequency Evolution')
    ax.legend()
    
    # 3. PAC Conservation Error
    ax = axes[1, 0]
    ax.plot(predictor.pac_error_history, 'r-', lw=1)
    ax.set_xlabel('Batch')
    ax.set_ylabel('PAC Error')
    ax.set_title('PAC Conservation (should stay low)')
    ax.set_yscale('log')
    
    # 4. Loss over training
    ax = axes[1, 1]
    window = min(50, len(predictor.loss_history) // 10)
    if window > 1:
        smoothed = np.convolve(predictor.loss_history, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, 'b-', lw=1)
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss (smoothed)')
    ax.set_title('Training Loss')
    
    plt.tight_layout()
    fig_path = Path(__file__).parent / 'figures' / f'exp_29_pac_chords_{n_primes}.png'
    fig_path.parent.mkdir(exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n📊 Saved: {fig_path}")
    
    # Second figure: Chord events timeline
    split_events = [e for e in predictor.field.history.events if e.event_type == 'split']
    merge_events = [e for e in predictor.field.history.events if e.event_type == 'merge']
    
    if split_events or merge_events:
        fig2, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.scatter([e.step for e in split_events], [e.before_freq for e in split_events], 
                   c='green', marker='^', s=80, label=f'Split ({len(split_events)})', alpha=0.7)
        ax.scatter([e.step for e in merge_events], [e.before_freq for e in merge_events],
                   c='red', marker='v', s=80, label=f'Merge ({len(merge_events)})', alpha=0.7)
        ax.axhline(y=PHI_INV, color='gold', linestyle='--', linewidth=2, label=f'1/φ={PHI_INV:.3f}')
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('φ-Frequency at Event', fontsize=12)
        ax.set_title(f'PAC Breathing: {len(split_events)} Splits ↑ / {len(merge_events)} Merges ↓', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        fig2_path = Path(__file__).parent / 'figures' / f'exp_29_pac_events_{n_primes}.png'
        plt.savefig(fig2_path, dpi=150)
        plt.close()
        print(f"📊 Events plot: {fig2_path}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'experiment': 'exp_29_pac_dynamic_chords',
        'timestamp': timestamp,
        'config': {
            'n_primes': n_primes, 'initial_strands': initial_strands,
            'max_strands': max_strands, 'n_epochs': n_epochs
        },
        'final_mae': float(final_mae),
        'baseline_mae': float(baseline_mae),
        'improvement_pct': float(improvement),
        'chord_dynamics': history,
        'final_strands': predictor.field.n_strands,
        'elapsed_seconds': elapsed
    }
    
    save_path = Path(__file__).parent / 'results' / f'exp_29_{n_primes}_{timestamp}.json'
    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {save_path}")
    
    return predictor, results


if __name__ == "__main__":
    # Direct comparison with exp_28 (100k primes, same conditions)
    print("\n" + "="*70)
    print("COMPARISON: PAC Dynamic Chords vs Static (exp_28)")
    print("="*70)
    
    predictor, results = run_experiment(
        n_primes=100_000,  # Same as exp_28
        window_size=10,
        initial_strands=8,  # Same starting point as exp_28
        max_strands=16,
        lr=0.01,
        batch_size=512,  # Same as exp_28
        n_epochs=20  # Same as exp_28
    )
    
    print("\n" + "="*70)
    print("COMPARISON WITH EXP_28 (Static 8 strands):")
    print("  exp_28 Static:  MAE=7.88, Improvement=25.6%")
    print(f"  exp_29 PAC:     MAE={results['final_mae']:.2f}, Improvement={results['improvement_pct']:.1f}%")
    print("="*70)
