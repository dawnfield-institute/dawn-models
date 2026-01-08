"""
TinyCIMM-Möbius: Continuous Learning with Möbius Frequency Memory

A minimal continuous learning architecture where:
- Memory = Möbius transformation parameters (a,b,c,d)
- Frequency = resonance with φ-fixed points  
- Harmonics = emergent chords from stacked Möbius layers
- Stability = cross-ratio preservation during learning

Key Insight:
The Möbius transformation M(z) = (az+b)/(cz+d) has two fixed points.
For Fibonacci matrices, these are φ and -1/φ. During continuous learning,
the network should maintain proximity to these attractors.

The "frequency" is how strongly the network resonates with these fixed points.
High frequency = stable pattern recognized
Low frequency = exploring new patterns

This creates a natural harmonic structure:
- Fundamental: Single Möbius layer at Fibonacci configuration
- First harmonic: Two layers with complementary phases
- nth harmonic: n layers forming a "chord" of φ-resonances
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import math

# Golden ratio constants
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = PHI - 1  # = 1/φ


@dataclass
class MobiusHarmonic:
    """A single harmonic in the Möbius frequency spectrum."""
    frequency: float  # Resonance strength with φ-fixed points
    phase: float      # Phase offset (related to which fixed point)
    amplitude: float  # Activation strength
    order: int        # Harmonic order (1 = fundamental, 2 = first overtone, etc.)


@dataclass  
class MobiusMemoryState:
    """Complete memory state of a TinyCIMM-Möbius network."""
    harmonics: List[MobiusHarmonic] = field(default_factory=list)
    entropy: float = 0.0
    stability: float = 1.0
    cross_ratio_drift: float = 0.0
    steps_since_collapse: int = 0


class MobiusNeuron(nn.Module):
    """
    Single Möbius neuron with learnable parameters.
    
    M(z) = (a*z + b) / (c*z + d)
    
    The neuron has built-in nonlinearity (no activation needed).
    Memory is encoded in (a,b,c,d) parameters.
    """
    
    def __init__(self, init: str = 'fibonacci', device='cpu'):
        super().__init__()
        self.device = device
        
        if init == 'fibonacci':
            # Start near Fibonacci: M(z) = (z+1)/(z+0) = 1 + 1/z
            self.a = nn.Parameter(torch.tensor(1.0, device=device))
            self.b = nn.Parameter(torch.tensor(1.0, device=device))
            self.c = nn.Parameter(torch.tensor(1.0, device=device))
            self.d = nn.Parameter(torch.tensor(0.01, device=device))  # Small to avoid exact 0
        elif init == 'identity':
            # Start at identity: M(z) = z
            self.a = nn.Parameter(torch.tensor(1.0, device=device))
            self.b = nn.Parameter(torch.tensor(0.0, device=device))
            self.c = nn.Parameter(torch.tensor(0.0, device=device))
            self.d = nn.Parameter(torch.tensor(1.0, device=device))
        else:
            # Random init
            self.a = nn.Parameter(torch.randn(1, device=device).squeeze() * 0.5)
            self.b = nn.Parameter(torch.randn(1, device=device).squeeze() * 0.5)
            self.c = nn.Parameter(torch.randn(1, device=device).squeeze() * 0.5)
            self.d = nn.Parameter(torch.randn(1, device=device).squeeze() * 0.5)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply Möbius transformation."""
        return (self.a * z + self.b) / (self.c * z + self.d + 1e-8)
    
    def fixed_points(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the two fixed points of this Möbius transformation."""
        # M(z) = z => (a*z + b) / (c*z + d) = z
        # => a*z + b = c*z^2 + d*z
        # => c*z^2 + (d-a)*z - b = 0
        # z = [-(d-a) ± sqrt((d-a)^2 + 4*c*b)] / (2*c)
        
        discriminant = (self.d - self.a)**2 + 4 * self.c * self.b
        sqrt_disc = torch.sqrt(torch.abs(discriminant) + 1e-8)
        
        if self.c.abs() < 1e-8:
            # Linear case: (a-d)*z = b => z = b/(a-d)
            z1 = self.b / (self.a - self.d + 1e-8)
            z2 = z1
        else:
            z1 = (-(self.d - self.a) + sqrt_disc) / (2 * self.c + 1e-8)
            z2 = (-(self.d - self.a) - sqrt_disc) / (2 * self.c + 1e-8)
        
        return z1, z2
    
    def phi_frequency(self) -> torch.Tensor:
        """
        Compute resonance frequency with φ-fixed points.
        
        High frequency = close to Fibonacci configuration
        Low frequency = far from Fibonacci
        """
        z1, z2 = self.fixed_points()
        
        # Distance from ideal Fibonacci fixed points (φ and -1/φ)
        dist_to_phi = torch.min(torch.abs(z1 - PHI), torch.abs(z2 - PHI))
        dist_to_neg_phi_inv = torch.min(torch.abs(z1 + PHI_INV), torch.abs(z2 + PHI_INV))
        
        # Frequency inversely related to distance
        freq = 1.0 / (1.0 + dist_to_phi + dist_to_neg_phi_inv)
        return freq
    
    def determinant(self) -> torch.Tensor:
        """Compute ad - bc (should be 1 for normalized Möbius)."""
        return self.a * self.d - self.b * self.c


class MobiusEntropyMonitor:
    """Monitor entropy and stability of Möbius learning."""
    
    def __init__(self, momentum: float = 0.9, window_size: int = 50):
        self.momentum = momentum
        self.window_size = window_size
        self.entropy = 0.0
        self.entropy_history: List[float] = []
        self.frequency_history: List[float] = []
        self.stability_history: List[float] = []
        
    def update(self, prediction: torch.Tensor, frequency: torch.Tensor) -> float:
        """Update entropy estimate based on prediction spread."""
        # Entropy from prediction variance
        if prediction.numel() > 1:
            pred_entropy = torch.var(prediction).item()
        else:
            pred_entropy = 0.0
        
        # Combine with frequency stability
        freq_val = frequency.item() if torch.is_tensor(frequency) else frequency
        
        # Smooth update
        self.entropy = self.momentum * self.entropy + (1 - self.momentum) * pred_entropy
        self.entropy_history.append(self.entropy)
        self.frequency_history.append(freq_val)
        
        # Trim history
        if len(self.entropy_history) > self.window_size:
            self.entropy_history.pop(0)
            self.frequency_history.pop(0)
        
        return self.entropy
    
    def get_stability(self) -> float:
        """Compute stability from frequency consistency."""
        if len(self.frequency_history) < 2:
            return 1.0
        freq_tensor = torch.tensor(self.frequency_history)
        stability = 1.0 / (1.0 + torch.var(freq_tensor).item())
        return stability
    
    def detect_collapse(self, threshold: float = 0.1) -> bool:
        """Detect if entropy has collapsed (pattern locked)."""
        if len(self.entropy_history) < 10:
            return False
        recent = self.entropy_history[-10:]
        return np.std(recent) < threshold and np.mean(recent) < threshold


class MobiusHarmonicAnalyzer:
    """Analyze harmonic structure of stacked Möbius layers."""
    
    def __init__(self):
        self.harmonic_history: List[List[MobiusHarmonic]] = []
    
    def analyze(self, neurons: List[MobiusNeuron]) -> List[MobiusHarmonic]:
        """Extract harmonic spectrum from Möbius neuron stack."""
        harmonics = []
        
        for i, neuron in enumerate(neurons):
            freq = neuron.phi_frequency().item()
            z1, z2 = neuron.fixed_points()
            
            # Phase = angle between fixed points
            phase = torch.atan2(z1 - z2, torch.tensor(1.0)).item()
            
            # Amplitude = determinant magnitude (energy)
            det = neuron.determinant()
            amplitude = torch.abs(det).item()
            
            harmonic = MobiusHarmonic(
                frequency=freq,
                phase=phase,
                amplitude=amplitude,
                order=i + 1
            )
            harmonics.append(harmonic)
        
        self.harmonic_history.append(harmonics)
        return harmonics
    
    def get_chord(self, harmonics: List[MobiusHarmonic]) -> str:
        """Identify the harmonic chord type."""
        if not harmonics:
            return "silence"
        
        avg_freq = np.mean([h.frequency for h in harmonics])
        freq_spread = np.std([h.frequency for h in harmonics])
        
        if avg_freq > 0.8:
            if freq_spread < 0.1:
                return "pure_phi"  # All layers at φ resonance
            else:
                return "phi_chord"  # Mixed φ harmonics
        elif avg_freq > 0.5:
            return "transitional"  # Moving toward φ
        else:
            return "exploratory"  # Far from φ, still learning


class PhiAnchorMemory:
    """
    Memory system that preserves learned φ-resonances.
    
    The key insight: when a Möbius network has learned a pattern well
    (high φ-frequency, low entropy), we snapshot the parameters.
    During future learning, we add a regularization loss to prevent
    drifting too far from this anchor.
    
    This is the TinyCIMM-Möbius equivalent of "micro_memory" from TinyCIMM-Planck.
    Instead of storing activations, we store parameter configurations that
    achieved high φ-resonance.
    """
    
    def __init__(self, capacity: int = 5, drift_penalty: float = 0.1):
        self.capacity = capacity
        self.drift_penalty = drift_penalty
        self.anchors: List[Dict] = []  # List of {params, freq, chord, task}
        self.current_task = 'default'
    
    def snapshot(self, neurons: List[MobiusNeuron], freq: float, chord: str):
        """Take a snapshot if this is a high-quality configuration."""
        # Only snapshot if high frequency and pure chord
        if freq < 0.7 or chord not in ['pure_phi', 'phi_chord']:
            return False
        
        # Check if we already have a similar anchor
        for anchor in self.anchors:
            if anchor['task'] == self.current_task:
                # Update if this is better
                if freq > anchor['freq']:
                    anchor['params'] = self._extract_params(neurons)
                    anchor['freq'] = freq
                    anchor['chord'] = chord
                return True
        
        # Add new anchor
        if len(self.anchors) >= self.capacity:
            # Remove lowest frequency anchor
            self.anchors.sort(key=lambda x: x['freq'], reverse=True)
            self.anchors.pop()
        
        self.anchors.append({
            'task': self.current_task,
            'params': self._extract_params(neurons),
            'freq': freq,
            'chord': chord
        })
        return True
    
    def _extract_params(self, neurons: List[MobiusNeuron]) -> List[Dict]:
        """Extract current parameters from neurons."""
        return [
            {'a': n.a.detach().clone(), 'b': n.b.detach().clone(),
             'c': n.c.detach().clone(), 'd': n.d.detach().clone()}
            for n in neurons
        ]
    
    def compute_anchor_loss(self, neurons: List[MobiusNeuron]) -> torch.Tensor:
        """Compute regularization loss to stay near anchors."""
        if not self.anchors:
            return torch.tensor(0.0)
        
        total_loss = torch.tensor(0.0)
        
        for anchor in self.anchors:
            for i, (neuron, anchor_params) in enumerate(zip(neurons, anchor['params'])):
                # L2 distance from anchor params
                drift = (
                    (neuron.a - anchor_params['a'])**2 +
                    (neuron.b - anchor_params['b'])**2 +
                    (neuron.c - anchor_params['c'])**2 +
                    (neuron.d - anchor_params['d'])**2
                )
                total_loss = total_loss + drift * self.drift_penalty * anchor['freq']
        
        return total_loss / len(self.anchors)
    
    def set_task(self, task_name: str):
        """Switch to a new task (enables task-specific memory)."""
        self.current_task = task_name
    
    def get_summary(self) -> Dict:
        """Get summary of stored anchors."""
        return {
            'n_anchors': len(self.anchors),
            'anchors': [
                {'task': a['task'], 'freq': a['freq'], 'chord': a['chord']}
                for a in self.anchors
            ]
        }


class TinyCIMMMobius(nn.Module):
    """
    TinyCIMM-Möbius: Continuous Learning with Möbius Frequency Memory
    
    Architecture:
    - Input projection to complex plane
    - Stack of MobiusNeuron layers (each with 4 learnable params)
    - Output projection back to real
    - Harmonic analyzer for interpretability
    - Entropy-based adaptation during continuous learning
    - PhiAnchorMemory for preventing catastrophic forgetting
    
    Memory Model:
    - Short-term: Current (a,b,c,d) parameters
    - Long-term: Harmonic spectrum history + φ-anchor snapshots
    - Stability: Cross-ratio preservation across updates
    """
    
    def __init__(
        self, 
        input_size: int,
        hidden_layers: int = 3,
        output_size: int = 1,
        device: str = 'cpu',
        init: str = 'fibonacci',
        continuous_lr: float = 0.01,
        use_anchor_memory: bool = True,
        anchor_capacity: int = 5,
        anchor_penalty: float = 0.1
    ):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.continuous_lr = continuous_lr
        self.use_anchor_memory = use_anchor_memory
        
        # Input projection: R^n -> C (complex plane)
        self.input_proj = nn.Linear(input_size, 2, device=device)  # [real, imag]
        
        # Stack of Möbius neurons
        self.mobius_layers = nn.ModuleList([
            MobiusNeuron(init=init, device=device) 
            for _ in range(hidden_layers)
        ])
        
        # Output projection: C -> R^m  
        self.output_scale = nn.Parameter(torch.tensor(1.0, device=device))
        self.output_bias = nn.Parameter(torch.tensor(0.0, device=device))
        self.output_proj = nn.Linear(2, output_size, device=device)
        
        # Monitoring and analysis
        self.entropy_monitor = MobiusEntropyMonitor()
        self.harmonic_analyzer = MobiusHarmonicAnalyzer()
        self.memory_state = MobiusMemoryState()
        
        # Anchor memory for catastrophic forgetting prevention
        self.anchor_memory = PhiAnchorMemory(
            capacity=anchor_capacity, 
            drift_penalty=anchor_penalty
        ) if use_anchor_memory else None
        
        # Continuous learning state
        self.step_count = 0
        self.last_loss = None
        self.adaptation_cooldown = 0
        
        # Optimizer for continuous learning
        self.optimizer = torch.optim.Adam(self.parameters(), lr=continuous_lr)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Möbius stack."""
        # Project to complex plane
        proj = self.input_proj(x)  # [batch, 2]
        z = proj[:, 0] + 1j * proj[:, 1]  # Complex tensor
        
        # Apply Möbius stack
        for layer in self.mobius_layers:
            z = layer(z.real)  # Use real part for real Möbius
        
        # Project back to output
        out_features = torch.stack([z.real, z.imag], dim=-1) if torch.is_complex(z) else torch.stack([z, torch.zeros_like(z)], dim=-1)
        out = self.output_proj(out_features)
        out = self.output_scale * out + self.output_bias
        
        return out
    
    def get_phi_frequency(self) -> float:
        """Get aggregate φ-frequency across all layers."""
        freqs = [layer.phi_frequency().item() for layer in self.mobius_layers]
        return float(np.mean(freqs))
    
    def get_harmonics(self) -> List[MobiusHarmonic]:
        """Get current harmonic spectrum."""
        return self.harmonic_analyzer.analyze(list(self.mobius_layers))
    
    def get_chord(self) -> str:
        """Get current harmonic chord type."""
        harmonics = self.get_harmonics()
        return self.harmonic_analyzer.get_chord(harmonics)
    
    def continuous_step(self, x: torch.Tensor, y_true: torch.Tensor) -> Dict:
        """
        Single step of continuous learning.
        
        Returns metrics about the learning step.
        """
        self.step_count += 1
        
        # Forward pass
        y_pred = self.forward(x)
        
        # Compute loss
        task_loss = nn.functional.mse_loss(y_pred, y_true)
        
        # Add anchor regularization if enabled
        anchor_loss = torch.tensor(0.0)
        if self.use_anchor_memory and self.anchor_memory:
            anchor_loss = self.anchor_memory.compute_anchor_loss(list(self.mobius_layers))
        
        loss = task_loss + anchor_loss
        
        # Get current frequency
        phi_freq = self.get_phi_frequency()
        
        # Update entropy monitor
        entropy = self.entropy_monitor.update(y_pred, torch.tensor(phi_freq))
        stability = self.entropy_monitor.get_stability()
        
        # Snapshot to anchor memory if high quality
        if self.use_anchor_memory and self.anchor_memory:
            self.anchor_memory.snapshot(list(self.mobius_layers), phi_freq, self.get_chord())
        
        # Detect if we should adapt
        should_adapt = True
        if self.adaptation_cooldown > 0:
            self.adaptation_cooldown -= 1
            should_adapt = False
        
        # Entropy-based learning rate modulation
        # High entropy = explore more (higher LR)
        # Low entropy = exploit (lower LR)
        effective_lr = self.continuous_lr * (1.0 + entropy)
        
        # Stability-based gradient clipping
        # Low stability = clip more aggressively
        max_grad_norm = 1.0 * stability + 0.1
        
        if should_adapt:
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping based on stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            
            # Adaptive learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = effective_lr
            
            self.optimizer.step()
        
        # Check for collapse
        collapsed = self.entropy_monitor.detect_collapse()
        if collapsed:
            self.memory_state.steps_since_collapse = 0
        else:
            self.memory_state.steps_since_collapse += 1
        
        # Update memory state
        harmonics = self.get_harmonics()
        self.memory_state.harmonics = harmonics
        self.memory_state.entropy = entropy
        self.memory_state.stability = stability
        
        # Compute cross-ratio drift (memory stability metric)
        if self.last_loss is not None:
            self.memory_state.cross_ratio_drift = abs(task_loss.item() - self.last_loss)
        self.last_loss = task_loss.item()
        
        return {
            'loss': task_loss.item(),
            'anchor_loss': anchor_loss.item() if torch.is_tensor(anchor_loss) else anchor_loss,
            'phi_frequency': phi_freq,
            'entropy': entropy,
            'stability': stability,
            'chord': self.get_chord(),
            'collapsed': collapsed,
            'effective_lr': effective_lr,
            'step': self.step_count
        }
    
    def continuous_train(
        self, 
        data_stream,  # Iterator of (x, y) pairs
        max_steps: int = 1000,
        log_interval: int = 100,
        convergence_threshold: float = 0.001
    ) -> List[Dict]:
        """
        Continuous training on a data stream.
        
        This is the key difference from batch training:
        - Each sample is seen once (or in a stream)
        - Network adapts continuously
        - Memory (Möbius params) evolves over time
        """
        history = []
        converged = False
        
        for step, (x, y) in enumerate(data_stream):
            if step >= max_steps:
                break
            
            # Convert to tensors if needed
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
            if not torch.is_tensor(y):
                y = torch.tensor(y, dtype=torch.float32, device=self.device)
            
            # Ensure proper shape
            if x.dim() == 1:
                x = x.unsqueeze(0)
            if y.dim() == 1:
                y = y.unsqueeze(0)
            
            # Continuous learning step
            metrics = self.continuous_step(x, y)
            history.append(metrics)
            
            # Logging
            if step % log_interval == 0:
                chord = metrics['chord']
                print(f"Step {step}: loss={metrics['loss']:.4f}, "
                      f"φ-freq={metrics['phi_frequency']:.3f}, "
                      f"entropy={metrics['entropy']:.4f}, "
                      f"chord={chord}")
            
            # Check convergence
            if len(history) > 50:
                recent_losses = [h['loss'] for h in history[-50:]]
                if np.std(recent_losses) < convergence_threshold:
                    converged = True
                    print(f"Converged at step {step}")
                    break
        
        return history
    
    def get_memory_summary(self) -> Dict:
        """Get summary of current memory state."""
        harmonics = self.get_harmonics()
        
        return {
            'n_layers': len(self.mobius_layers),
            'phi_frequency': self.get_phi_frequency(),
            'chord': self.get_chord(),
            'entropy': self.memory_state.entropy,
            'stability': self.memory_state.stability,
            'collapsed': self.entropy_monitor.detect_collapse(),
            'steps_since_collapse': self.memory_state.steps_since_collapse,
            'harmonics': [
                {'order': h.order, 'freq': h.frequency, 'phase': h.phase, 'amp': h.amplitude}
                for h in harmonics
            ],
            'layer_params': [
                {'a': l.a.item(), 'b': l.b.item(), 'c': l.c.item(), 'd': l.d.item()}
                for l in self.mobius_layers
            ]
        }


def create_fibonacci_data_stream(n_samples: int = 1000):
    """Create a data stream of Fibonacci ratios for testing."""
    fibs = [1, 1]
    for _ in range(50):
        fibs.append(fibs[-1] + fibs[-2])
    
    for _ in range(n_samples):
        # Random index
        idx = np.random.randint(2, len(fibs) - 1)
        x = np.array([fibs[idx] / fibs[idx-1]])  # Ratio F_n/F_{n-1}
        y = np.array([fibs[idx+1] / fibs[idx]])  # Next ratio F_{n+1}/F_n
        yield x, y


def create_mobius_transform_stream(n_samples: int = 1000):
    """Create a data stream of Möbius transformation values."""
    # Target: M(z) = (2z + 1) / (z + 1)
    for _ in range(n_samples):
        z = np.random.uniform(-0.5, 2.0)
        x = np.array([z])
        y = np.array([(2*z + 1) / (z + 1)])
        yield x, y


if __name__ == '__main__':
    print("=" * 70)
    print("TinyCIMM-Möbius: Continuous Learning with Möbius Frequency Memory")
    print("=" * 70)
    
    # Test 1: Fibonacci ratio prediction
    print("\n--- Test 1: Fibonacci Ratio Stream ---")
    model = TinyCIMMMobius(input_size=1, hidden_layers=2, output_size=1)
    
    stream = create_fibonacci_data_stream(500)
    history = model.continuous_train(stream, max_steps=500, log_interval=50)
    
    summary = model.get_memory_summary()
    print(f"\nFinal State:")
    print(f"  φ-frequency: {summary['phi_frequency']:.4f}")
    print(f"  Chord: {summary['chord']}")
    print(f"  Entropy: {summary['entropy']:.4f}")
    print(f"  Collapsed: {summary['collapsed']}")
    
    # Test 2: Möbius transform learning
    print("\n--- Test 2: Möbius Transform Stream ---")
    model2 = TinyCIMMMobius(input_size=1, hidden_layers=1, output_size=1, init='identity')
    
    stream2 = create_mobius_transform_stream(500)
    history2 = model2.continuous_train(stream2, max_steps=500, log_interval=50)
    
    summary2 = model2.get_memory_summary()
    print(f"\nFinal State:")
    print(f"  φ-frequency: {summary2['phi_frequency']:.4f}")
    print(f"  Chord: {summary2['chord']}")
    print(f"  Layer params: {summary2['layer_params']}")
    
    # Test actual prediction
    print("\n--- Verification ---")
    test_z = torch.tensor([[0.5]], dtype=torch.float32)
    pred = model2(test_z)
    true_val = (2 * 0.5 + 1) / (0.5 + 1)
    print(f"M(0.5) predicted: {pred.item():.4f}")
    print(f"M(0.5) true: {true_val:.4f}")
    print(f"Error: {abs(pred.item() - true_val):.4f}")
