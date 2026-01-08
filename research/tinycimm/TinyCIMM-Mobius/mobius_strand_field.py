"""
MobiusStrandField: Multi-Strand Möbius with Emergent Dimensionality

Key Insight:
Reality consists of strands - each observable interaction is a strand with its own
frequency. When strands interact (phase-couple), dimensions emerge from their
relationships. This is exactly what Dawn Field Theory predicts:

  "2D Möbius topology generates apparent 3D+1 spacetime"

Implementation:
- N parallel Möbius strands, each with (a,b,c,d) parameters
- Each strand has its own φ-frequency (resonance with golden fixed points)
- Strands interact through phase coupling (fixed point alignment)
- Emergent dimensions = eigenmodes of the strand interaction matrix

The "dimension" of the field is not fixed - it emerges from how many
independent frequency modes exist in the strand ensemble.

This creates a natural analogy:
- 1 strand = 0D point (single frequency)
- 2 strands = 1D line (phase difference)
- 3 strands = 2D surface (phase triangle)
- N strands = (N-1)D manifold (phase polytope)

But the magic is: the EFFECTIVE dimension can be lower than N-1 if strands
lock into harmonic relationships. A "chord" of 4 strands in perfect φ-ratio
has effective dimension ~1 (all frequencies derived from one).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import math

# Golden ratio constants
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = PHI - 1


@dataclass
class StrandState:
    """State of a single Möbius strand."""
    frequency: float          # φ-resonance strength
    phase: float              # Phase angle (from fixed points)
    amplitude: float          # Energy (determinant magnitude)
    fixed_points: Tuple[float, float]  # The two fixed points


@dataclass
class FieldState:
    """Emergent state of the strand field."""
    effective_dimension: float  # How many independent modes
    coherence: float           # How phase-locked are the strands
    total_energy: float        # Sum of strand amplitudes
    dominant_frequency: float  # Strongest mode
    chord_type: str            # Harmonic classification


class MobiusStrand(nn.Module):
    """
    A single Möbius strand with learnable parameters.
    
    Each strand is a Möbius transformation M(z) = (az+b)/(cz+d)
    with its own frequency (determined by how close its fixed points
    are to φ and -1/φ).
    """
    
    def __init__(self, strand_id: int, init: str = 'random', device='cpu'):
        super().__init__()
        self.strand_id = strand_id
        self.device = device
        
        if init == 'fibonacci':
            # Start near Fibonacci configuration
            self.a = nn.Parameter(torch.tensor(1.0 + np.random.randn() * 0.1, device=device))
            self.b = nn.Parameter(torch.tensor(1.0 + np.random.randn() * 0.1, device=device))
            self.c = nn.Parameter(torch.tensor(1.0 + np.random.randn() * 0.1, device=device))
            self.d = nn.Parameter(torch.tensor(0.0 + np.random.randn() * 0.1, device=device))
        elif init == 'identity':
            self.a = nn.Parameter(torch.tensor(1.0, device=device))
            self.b = nn.Parameter(torch.tensor(0.0, device=device))
            self.c = nn.Parameter(torch.tensor(0.0, device=device))
            self.d = nn.Parameter(torch.tensor(1.0, device=device))
        elif init == 'harmonic':
            # Initialize with harmonic offset based on strand_id
            phase = strand_id * np.pi / 4  # Each strand offset by π/4
            self.a = nn.Parameter(torch.tensor(np.cos(phase), device=device))
            self.b = nn.Parameter(torch.tensor(np.sin(phase), device=device))
            self.c = nn.Parameter(torch.tensor(-np.sin(phase) * 0.5, device=device))
            self.d = nn.Parameter(torch.tensor(np.cos(phase), device=device))
        else:  # random
            self.a = nn.Parameter(torch.randn(1, device=device).squeeze() * 0.5 + 0.5)
            self.b = nn.Parameter(torch.randn(1, device=device).squeeze() * 0.5)
            self.c = nn.Parameter(torch.randn(1, device=device).squeeze() * 0.5)
            self.d = nn.Parameter(torch.randn(1, device=device).squeeze() * 0.5 + 0.5)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply Möbius transformation."""
        return (self.a * z + self.b) / (self.c * z + self.d + 1e-8)
    
    def fixed_points(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the two fixed points."""
        discriminant = (self.d - self.a)**2 + 4 * self.c * self.b
        sqrt_disc = torch.sqrt(torch.abs(discriminant) + 1e-8)
        
        if self.c.abs() < 1e-8:
            z1 = self.b / (self.a - self.d + 1e-8)
            z2 = z1
        else:
            z1 = (-(self.d - self.a) + sqrt_disc) / (2 * self.c + 1e-8)
            z2 = (-(self.d - self.a) - sqrt_disc) / (2 * self.c + 1e-8)
        
        return z1, z2
    
    def frequency(self) -> torch.Tensor:
        """φ-resonance frequency."""
        z1, z2 = self.fixed_points()
        dist_to_phi = torch.min(torch.abs(z1 - PHI), torch.abs(z2 - PHI))
        dist_to_neg_phi_inv = torch.min(torch.abs(z1 + PHI_INV), torch.abs(z2 + PHI_INV))
        freq = 1.0 / (1.0 + dist_to_phi + dist_to_neg_phi_inv)
        return freq
    
    def phase(self) -> torch.Tensor:
        """Phase angle from fixed point geometry."""
        z1, z2 = self.fixed_points()
        return torch.atan2(z1 - z2, torch.tensor(1.0, device=self.device))
    
    def amplitude(self) -> torch.Tensor:
        """Energy = determinant magnitude."""
        return torch.abs(self.a * self.d - self.b * self.c)
    
    def get_state(self) -> StrandState:
        """Get current strand state."""
        z1, z2 = self.fixed_points()
        return StrandState(
            frequency=self.frequency().item(),
            phase=self.phase().item(),
            amplitude=self.amplitude().item(),
            fixed_points=(z1.item(), z2.item())
        )


class MobiusStrandField(nn.Module):
    """
    A field of Möbius strands with emergent dimensionality.
    
    Key properties:
    - N strands, each with independent (a,b,c,d) parameters
    - Strands interact through phase coupling matrix
    - Effective dimension emerges from eigenspectrum of coupling
    - Coherence measures how "locked" the strands are
    
    The field processes inputs by:
    1. Projecting input to each strand (parallel processing)
    2. Each strand applies its Möbius transform
    3. Strands couple through phase interaction
    4. Output is coherent superposition of strand outputs
    """
    
    def __init__(
        self,
        n_strands: int = 4,
        input_size: int = 1,
        output_size: int = 1,
        init: str = 'harmonic',
        coupling_strength: float = 0.1,
        device: str = 'cpu'
    ):
        super().__init__()
        self.n_strands = n_strands
        self.device = device
        self.coupling_strength = coupling_strength
        
        # Create strands
        self.strands = nn.ModuleList([
            MobiusStrand(strand_id=i, init=init, device=device)
            for i in range(n_strands)
        ])
        
        # Input projection: maps input to each strand
        self.input_proj = nn.Linear(input_size, n_strands, device=device)
        
        # Learnable coupling matrix (how strands interact)
        self.coupling = nn.Parameter(
            torch.eye(n_strands, device=device) + 
            torch.randn(n_strands, n_strands, device=device) * 0.1
        )
        
        # Output projection: combines strand outputs
        self.output_proj = nn.Linear(n_strands, output_size, device=device)
        self.output_scale = nn.Parameter(torch.tensor(1.0, device=device))
        self.output_bias = nn.Parameter(torch.tensor(0.0, device=device))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the strand field.
        
        1. Project input to strand space
        2. Each strand processes in parallel
        3. Apply phase coupling
        4. Project to output
        """
        # Project input to strand space
        strand_inputs = self.input_proj(x)  # [batch, n_strands]
        
        # Process through each strand
        strand_outputs = []
        for i, strand in enumerate(self.strands):
            strand_out = strand(strand_inputs[:, i])
            strand_outputs.append(strand_out)
        
        strand_outputs = torch.stack(strand_outputs, dim=-1)  # [batch, n_strands]
        
        # Apply phase coupling (strands influence each other)
        # This creates emergent dimensionality
        coupled = torch.matmul(strand_outputs, self.coupling) * self.coupling_strength
        strand_outputs = strand_outputs + coupled
        
        # Project to output
        output = self.output_proj(strand_outputs)
        output = self.output_scale * output + self.output_bias
        
        return output
    
    def get_phase_matrix(self) -> torch.Tensor:
        """
        Compute phase relationship matrix between strands.
        
        Entry (i,j) = phase difference between strand i and strand j.
        This matrix encodes the emergent geometry.
        """
        phases = torch.stack([s.phase() for s in self.strands])
        phase_diff = phases.unsqueeze(0) - phases.unsqueeze(1)
        return phase_diff
    
    def get_frequency_spectrum(self) -> torch.Tensor:
        """Get frequency of each strand."""
        return torch.stack([s.frequency() for s in self.strands])
    
    def compute_effective_dimension(self) -> float:
        """
        Compute effective dimensionality of the strand field.
        
        Uses eigenspectrum of the phase coupling matrix.
        If all strands are independent: dim ≈ n_strands - 1
        If all strands are locked: dim ≈ 0
        
        This is the key emergent property!
        """
        # Get phase matrix
        phase_matrix = self.get_phase_matrix()
        
        # Compute correlation from phase differences
        # cos(phase_diff) = 1 when phases are aligned, -1 when opposite
        correlation = torch.cos(phase_matrix)
        
        # Eigenvalues of correlation matrix
        eigenvalues = torch.linalg.eigvalsh(correlation)
        eigenvalues = torch.abs(eigenvalues)
        eigenvalues = eigenvalues / (eigenvalues.sum() + 1e-8)
        
        # Participation ratio: effective number of dimensions
        # If one eigenvalue dominates: PR ≈ 1
        # If all equal: PR ≈ n_strands
        participation_ratio = 1.0 / (torch.sum(eigenvalues**2) + 1e-8)
        
        # Effective dimension is PR - 1 (subtract the "ground state")
        return max(0, participation_ratio.item() - 1)
    
    def compute_coherence(self) -> float:
        """
        Compute coherence: how phase-locked are the strands?
        
        High coherence = strands in φ-harmonic relationship
        Low coherence = strands independent (more dimensions)
        """
        freqs = self.get_frequency_spectrum()
        phases = torch.stack([s.phase() for s in self.strands])
        
        # Coherence from phase variance
        phase_spread = torch.std(phases).item()
        coherence = 1.0 / (1.0 + phase_spread)
        
        return coherence
    
    def get_field_state(self) -> FieldState:
        """Get emergent field state."""
        freqs = self.get_frequency_spectrum()
        
        # Classify chord type based on frequency relationships
        avg_freq = freqs.mean().item()
        freq_spread = freqs.std().item()
        
        if avg_freq > 0.8 and freq_spread < 0.1:
            chord_type = "unison"  # All at φ
        elif avg_freq > 0.6 and freq_spread < 0.2:
            chord_type = "phi_chord"  # φ harmonic
        elif freq_spread < 0.15:
            chord_type = "locked"  # Same frequency, not φ
        else:
            chord_type = "polyphonic"  # Multiple independent frequencies
        
        return FieldState(
            effective_dimension=self.compute_effective_dimension(),
            coherence=self.compute_coherence(),
            total_energy=sum(s.amplitude().item() for s in self.strands),
            dominant_frequency=freqs.max().item(),
            chord_type=chord_type
        )
    
    def get_strand_summary(self) -> List[Dict]:
        """Get summary of all strands."""
        return [
            {
                'id': s.strand_id,
                'frequency': s.frequency().item(),
                'phase': s.phase().item(),
                'amplitude': s.amplitude().item(),
                'params': {
                    'a': s.a.item(), 'b': s.b.item(),
                    'c': s.c.item(), 'd': s.d.item()
                }
            }
            for s in self.strands
        ]


class TinyCIMMMobiusField(nn.Module):
    """
    TinyCIMM with MobiusStrandField: Continuous learning with emergent dimensions.
    
    This combines:
    - Multiple Möbius strands (each with own frequency)
    - Phase coupling between strands (emergent geometry)
    - Entropy-based adaptation (TinyCIMM pattern)
    - Dimension tracking (how many modes are active)
    
    The key hypothesis: patterns that have natural φ-structure will
    collapse the field to low effective dimension (few modes).
    Patterns without φ-structure require more dimensions (more modes).
    """
    
    def __init__(
        self,
        n_strands: int = 4,
        input_size: int = 1,
        output_size: int = 1,
        init: str = 'harmonic',
        coupling_strength: float = 0.1,
        continuous_lr: float = 0.01,
        device: str = 'cpu'
    ):
        super().__init__()
        self.device = device
        
        # The strand field
        self.field = MobiusStrandField(
            n_strands=n_strands,
            input_size=input_size,
            output_size=output_size,
            init=init,
            coupling_strength=coupling_strength,
            device=device
        )
        
        # Continuous learning state
        self.step_count = 0
        self.dimension_history = []
        self.coherence_history = []
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=continuous_lr)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.field(x)
    
    def continuous_step(self, x: torch.Tensor, y_true: torch.Tensor) -> Dict:
        """Single step of continuous learning with dimension tracking."""
        self.step_count += 1
        
        # Forward
        y_pred = self.forward(x)
        loss = nn.functional.mse_loss(y_pred, y_true)
        
        # Get field state before update
        field_state = self.field.get_field_state()
        
        # Track history
        self.dimension_history.append(field_state.effective_dimension)
        self.coherence_history.append(field_state.coherence)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'effective_dim': field_state.effective_dimension,
            'coherence': field_state.coherence,
            'chord': field_state.chord_type,
            'dominant_freq': field_state.dominant_frequency,
            'step': self.step_count
        }
    
    def continuous_train(self, data_stream, max_steps: int = 1000, log_interval: int = 100):
        """Train on data stream with dimension monitoring."""
        history = []
        
        for step, (x, y) in enumerate(data_stream):
            if step >= max_steps:
                break
            
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
            if not torch.is_tensor(y):
                y = torch.tensor(y, dtype=torch.float32, device=self.device)
            
            if x.dim() == 1:
                x = x.unsqueeze(0)
            if y.dim() == 1:
                y = y.unsqueeze(0)
            
            metrics = self.continuous_step(x, y)
            history.append(metrics)
            
            if step % log_interval == 0:
                print(f"Step {step}: loss={metrics['loss']:.4f}, "
                      f"dim={metrics['effective_dim']:.2f}, "
                      f"coherence={metrics['coherence']:.3f}, "
                      f"chord={metrics['chord']}")
        
        return history


def demo_strand_field():
    """Demonstrate the strand field properties."""
    print("=" * 70)
    print("MobiusStrandField: Emergent Dimensionality Demo")
    print("=" * 70)
    
    # Create field with 4 strands
    field = MobiusStrandField(n_strands=4, init='harmonic')
    
    print("\n--- Initial Strand States ---")
    for strand_info in field.get_strand_summary():
        print(f"  Strand {strand_info['id']}: "
              f"freq={strand_info['frequency']:.4f}, "
              f"phase={strand_info['phase']:.4f}")
    
    state = field.get_field_state()
    print(f"\n--- Initial Field State ---")
    print(f"  Effective Dimension: {state.effective_dimension:.2f}")
    print(f"  Coherence: {state.coherence:.3f}")
    print(f"  Chord Type: {state.chord_type}")
    
    # Test forward pass
    x = torch.tensor([[1.0]])
    y = field(x)
    print(f"\n--- Forward Pass ---")
    print(f"  Input: {x.item():.4f}")
    print(f"  Output: {y.item():.4f}")
    
    return field


def experiment_fibonacci_dimension_collapse():
    """
    Key experiment: Does learning Fibonacci collapse the dimension?
    
    Hypothesis: φ-structured patterns should reduce effective dimension
    because strands lock into harmonic relationships.
    """
    print("\n" + "=" * 70)
    print("Experiment: Fibonacci Dimension Collapse")
    print("=" * 70)
    
    # Generate Fibonacci data
    fibs = [1, 1]
    for _ in range(30):
        fibs.append(fibs[-1] + fibs[-2])
    
    def fib_stream(n):
        for _ in range(n):
            idx = np.random.randint(5, 25)
            x = np.array([fibs[idx] / fibs[idx-1]])
            y = np.array([fibs[idx+1] / fibs[idx]])
            yield x, y
    
    # Train
    model = TinyCIMMMobiusField(n_strands=4, init='random')
    
    print("\nInitial state:")
    state = model.field.get_field_state()
    print(f"  Effective Dimension: {state.effective_dimension:.2f}")
    
    history = model.continuous_train(fib_stream(500), max_steps=500, log_interval=100)
    
    print("\nFinal state:")
    state = model.field.get_field_state()
    print(f"  Effective Dimension: {state.effective_dimension:.2f}")
    print(f"  Coherence: {state.coherence:.3f}")
    print(f"  Chord: {state.chord_type}")
    
    # Analyze dimension evolution
    dims = [h['effective_dim'] for h in history]
    print(f"\nDimension Evolution:")
    print(f"  Start: {dims[0]:.2f}")
    print(f"  End: {dims[-1]:.2f}")
    print(f"  Collapsed: {dims[-1] < dims[0]}")
    
    return model, history


def experiment_polynomial_dimension():
    """
    Counter-experiment: Do non-φ patterns keep dimensions high?
    
    Hypothesis: Polynomial patterns (no φ-structure) should NOT
    collapse dimension - strands remain independent.
    """
    print("\n" + "=" * 70)
    print("Experiment: Polynomial Dimension (Control)")
    print("=" * 70)
    
    def poly_stream(n):
        for _ in range(n):
            x_val = np.random.uniform(0.5, 2.0)
            x = np.array([x_val])
            y = np.array([0.5 * x_val**2 + 0.3 * x_val])
            yield x, y
    
    model = TinyCIMMMobiusField(n_strands=4, init='random')
    
    print("\nInitial state:")
    state = model.field.get_field_state()
    print(f"  Effective Dimension: {state.effective_dimension:.2f}")
    
    history = model.continuous_train(poly_stream(500), max_steps=500, log_interval=100)
    
    print("\nFinal state:")
    state = model.field.get_field_state()
    print(f"  Effective Dimension: {state.effective_dimension:.2f}")
    print(f"  Coherence: {state.coherence:.3f}")
    print(f"  Chord: {state.chord_type}")
    
    dims = [h['effective_dim'] for h in history]
    print(f"\nDimension Evolution:")
    print(f"  Start: {dims[0]:.2f}")
    print(f"  End: {dims[-1]:.2f}")
    print(f"  Maintained/Grew: {dims[-1] >= dims[0] * 0.9}")
    
    return model, history


if __name__ == '__main__':
    # Demo
    field = demo_strand_field()
    
    # Key experiments
    fib_model, fib_history = experiment_fibonacci_dimension_collapse()
    poly_model, poly_history = experiment_polynomial_dimension()
    
    print("\n" + "=" * 70)
    print("SUMMARY: Dimension Collapse Comparison")
    print("=" * 70)
    
    fib_dims = [h['effective_dim'] for h in fib_history]
    poly_dims = [h['effective_dim'] for h in poly_history]
    
    print(f"\nFibonacci (φ-structured):")
    print(f"  Dimension: {fib_dims[0]:.2f} → {fib_dims[-1]:.2f}")
    print(f"  Collapsed: {'YES' if fib_dims[-1] < fib_dims[0] * 0.8 else 'NO'}")
    
    print(f"\nPolynomial (no φ-structure):")
    print(f"  Dimension: {poly_dims[0]:.2f} → {poly_dims[-1]:.2f}")
    print(f"  Maintained: {'YES' if poly_dims[-1] >= poly_dims[0] * 0.8 else 'NO'}")
    
    # The key prediction
    if fib_dims[-1] < poly_dims[-1]:
        print("\n✓ CONFIRMED: φ-patterns collapse dimension, non-φ patterns don't")
    else:
        print("\n✗ NOT CONFIRMED: Need more investigation")
