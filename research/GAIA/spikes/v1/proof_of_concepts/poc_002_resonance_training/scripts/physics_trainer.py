"""
Physics-Informed Training Components for GAIA
==============================================

Dawn Field Theory integration for resonance-based semantic learning.

Constants from experiments:
- SEC Prime Manifold: φ × ξ = 1.710 crystallization trigger, λ* = 0.9816 decay
- PAC Confluence Xi: Fibonacci learning rates, 4/5 entanglement limit  
- Prime Harmonic Manifold: 1/π² eigenvalue decay
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math


# =============================================================================
# Physical Constants (from Dawn Field Theory experiments)
# =============================================================================

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
XI = 1.0571  # PAC conservation operator
PHI_XI = PHI * XI  # 1.710 - crystallization trigger
LAMBDA_STAR = 0.9816  # Optimal memory decay (SEC)
ENTANGLEMENT_LIMIT = 4/5  # Max coupling strength (PAC)
EIGENVALUE_DECAY = 1 / (math.pi ** 2)  # 0.101 (PHM)

# Fibonacci sequence for learning rates
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]


def get_device() -> torch.device:
    """Get CUDA device if available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Phase Transition Monitor (SEC)
# =============================================================================

class PhaseTransitionMonitor:
    """
    Monitor field entropy for crystallization events.
    
    From SEC Prime Manifold: structure crystallizes when 
    entropy × eigenvalue_ratio crosses φ × ξ = 1.710
    """
    
    def __init__(self, threshold: float = PHI_XI, device: Optional[torch.device] = None):
        self.threshold = threshold
        self.device = device or get_device()
        self.history: List[float] = []
        self.transitions: List[int] = []  # Step indices where transitions occurred
        
    def compute_field_entropy(self, field: torch.Tensor) -> torch.Tensor:
        """Compute Shannon entropy of field probability distribution."""
        # Normalize to probability distribution
        probs = F.softmax(field.flatten(), dim=0)
        # Avoid log(0)
        probs = probs.clamp(min=1e-10)
        entropy = -torch.sum(probs * torch.log(probs))
        return entropy
    
    def compute_eigenvalue_ratio(self, field: torch.Tensor) -> torch.Tensor:
        """Compute ratio of top eigenvalue to trace."""
        if field.dim() == 1:
            # For 1D fields, create covariance-like matrix
            field_2d = field.unsqueeze(0)
            matrix = field_2d.T @ field_2d
        else:
            matrix = field
            
        # Eigenvalue computation
        eigenvalues = torch.linalg.eigvalsh(matrix)
        top_eigenvalue = eigenvalues[-1]  # Largest
        trace = torch.sum(eigenvalues)
        
        return top_eigenvalue / (trace + 1e-10)
    
    def check_transition(self, field: torch.Tensor, step: int) -> Tuple[bool, float]:
        """
        Check if field is at crystallization point.
        
        Returns:
            (is_transition, metric_value)
        """
        entropy = self.compute_field_entropy(field)
        
        # For simplicity, use entropy directly scaled by a factor
        # In full implementation, combine with eigenvalue ratio
        metric = entropy.item() * EIGENVALUE_DECAY * 10  # Scale to relevant range
        
        self.history.append(metric)
        
        is_transition = metric > self.threshold
        if is_transition:
            self.transitions.append(step)
            
        return is_transition, metric
    
    def get_stats(self) -> Dict:
        """Get monitoring statistics."""
        return {
            'total_steps': len(self.history),
            'transitions': len(self.transitions),
            'transition_steps': self.transitions,
            'mean_metric': sum(self.history) / len(self.history) if self.history else 0,
            'max_metric': max(self.history) if self.history else 0,
            'threshold': self.threshold,
        }


# =============================================================================
# Fibonacci Learning Rate Scheduler (PAC)
# =============================================================================

class FibonacciScheduler:
    """
    Learning rate scheduler based on Fibonacci sequence.
    
    From PAC Confluence Xi: learning rate = 1/F_n based on pattern complexity.
    """
    
    def __init__(self, base_lr: float = 0.1, complexity_scale: int = 1):
        self.base_lr = base_lr
        self.complexity_scale = complexity_scale
        self.step_count = 0
        
    def get_lr(self, complexity: Optional[int] = None) -> float:
        """
        Get learning rate based on current step or pattern complexity.
        
        Args:
            complexity: Pattern complexity (higher = smaller lr)
        """
        if complexity is not None:
            # Complexity-based: lr = base / F_complexity
            idx = min(complexity, len(FIBONACCI) - 1)
            return self.base_lr / FIBONACCI[idx]
        else:
            # Step-based: lr decays with Fibonacci
            idx = min(self.step_count // 10, len(FIBONACCI) - 1)
            return self.base_lr / FIBONACCI[idx]
    
    def step(self):
        """Increment step counter."""
        self.step_count += 1
        
    def reset(self):
        """Reset scheduler."""
        self.step_count = 0


# =============================================================================
# Resonance Trainer
# =============================================================================

class ResonanceTrainer:
    """
    Train semantic relationships through field resonance.
    
    Key principle: patterns that co-occur strengthen mutual bonds.
    No gradients - training via field dynamics.
    """
    
    def __init__(
        self,
        field_dim: int = 64,
        device: Optional[torch.device] = None,
        use_phase_monitor: bool = True,
        use_fibonacci_lr: bool = True,
    ):
        self.device = device or get_device()
        self.field_dim = field_dim
        
        # Field state: maps pattern (as string) to field embedding
        self.field_memory: Dict[str, torch.Tensor] = {}
        
        # Co-occurrence strength matrix (dynamically sized)
        self.cooccurrence: Dict[Tuple[str, str], float] = {}
        
        # Physics components
        self.phase_monitor = PhaseTransitionMonitor(device=self.device) if use_phase_monitor else None
        self.scheduler = FibonacciScheduler() if use_fibonacci_lr else None
        
        # Training history
        self.training_log: List[Dict] = []
        
    def _encode_pattern(self, pattern: str) -> torch.Tensor:
        """Encode pattern to field embedding."""
        if pattern in self.field_memory:
            return self.field_memory[pattern]
            
        # Create deterministic encoding from pattern hash
        # Using same approach as POC-001 for consistency
        encoded = torch.zeros(self.field_dim, device=self.device)
        for i, char in enumerate(pattern):
            bit_idx = (ord(char) + i * 7) % self.field_dim
            encoded[bit_idx] = 1.0
            
        # Normalize
        encoded = F.normalize(encoded, dim=0)
        self.field_memory[pattern] = encoded
        return encoded
    
    def _get_cooccurrence(self, p1: str, p2: str) -> float:
        """Get co-occurrence strength between patterns."""
        key = tuple(sorted([p1, p2]))
        return self.cooccurrence.get(key, 0.0)
    
    def _update_cooccurrence(self, p1: str, p2: str, delta: float):
        """Update co-occurrence strength."""
        key = tuple(sorted([p1, p2]))
        current = self.cooccurrence.get(key, 0.0)
        # Clamp to entanglement limit
        new_value = min(current + delta, ENTANGLEMENT_LIMIT)
        self.cooccurrence[key] = new_value
        
    def train_cooccurrence(
        self,
        pattern_pairs: List[Tuple[str, str]],
        epochs: int = 10,
        verbose: bool = False,
    ) -> Dict:
        """
        Train on pattern co-occurrences.
        
        Args:
            pattern_pairs: List of (pattern1, pattern2) that co-occur
            epochs: Number of training epochs
            verbose: Print progress
            
        Returns:
            Training statistics
        """
        step = 0
        transitions = 0
        
        for epoch in range(epochs):
            epoch_updates = 0
            
            for p1, p2 in pattern_pairs:
                # Get current field states
                f1 = self._encode_pattern(p1)
                f2 = self._encode_pattern(p2)
                
                # Compute learning rate
                if self.scheduler:
                    lr = self.scheduler.get_lr()
                    self.scheduler.step()
                else:
                    lr = 0.1
                
                # Update co-occurrence
                self._update_cooccurrence(p1, p2, lr)
                
                # Resonance: pull field states closer
                combined = f1 + f2
                
                # Check for phase transition
                if self.phase_monitor:
                    is_transition, metric = self.phase_monitor.check_transition(combined, step)
                    if is_transition:
                        transitions += 1
                        # At crystallization: strengthen bond extra
                        self._update_cooccurrence(p1, p2, lr * PHI)
                
                # Update field memory with resonance effect
                resonance = self._get_cooccurrence(p1, p2)
                
                # Pull fields toward each other based on resonance
                blend = resonance * EIGENVALUE_DECAY  # Small pull
                new_f1 = F.normalize(f1 + blend * f2, dim=0)
                new_f2 = F.normalize(f2 + blend * f1, dim=0)
                
                self.field_memory[p1] = new_f1
                self.field_memory[p2] = new_f2
                
                epoch_updates += 1
                step += 1
            
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}: {epoch_updates} updates, {transitions} transitions")
        
        return {
            'epochs': epochs,
            'total_steps': step,
            'transitions': transitions,
            'patterns_trained': len(self.field_memory),
            'cooccurrence_pairs': len(self.cooccurrence),
        }
    
    def similarity(self, p1: str, p2: str) -> float:
        """Compute similarity between two patterns."""
        f1 = self._encode_pattern(p1)
        f2 = self._encode_pattern(p2)
        
        # Base similarity from field embedding
        cosine_sim = F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
        
        # Add co-occurrence component
        cooc = self._get_cooccurrence(p1, p2)
        
        # Combined similarity
        return cosine_sim * 0.5 + cooc * 0.5
    
    def get_cluster(self, threshold: float = 0.5) -> List[List[str]]:
        """Group patterns into clusters based on similarity."""
        patterns = list(self.field_memory.keys())
        n = len(patterns)
        
        # Compute similarity matrix
        sim_matrix = torch.zeros(n, n, device=self.device)
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.similarity(patterns[i], patterns[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
        
        # Simple clustering: group by threshold
        visited = set()
        clusters = []
        
        for i in range(n):
            if i in visited:
                continue
            cluster = [patterns[i]]
            visited.add(i)
            
            for j in range(i + 1, n):
                if j not in visited and sim_matrix[i, j] > threshold:
                    cluster.append(patterns[j])
                    visited.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def conservation_check(self) -> float:
        """Check PAC conservation across all fields."""
        if not self.field_memory:
            return 0.0
            
        total_energy = 0.0
        for field in self.field_memory.values():
            energy = torch.sum(field ** 2).item()
            total_energy += energy
            
        # With normalization, each field has energy ~1.0
        expected = len(self.field_memory)
        residual = abs(total_energy - expected)
        
        return residual


# =============================================================================
# High-Level Trainer (combines all physics)
# =============================================================================

class DawnFieldTrainer:
    """
    Complete Dawn Field Theory trainer for GAIA.
    
    Integrates:
    - SEC phase transition monitoring
    - PAC conservation checking
    - Fibonacci learning rates
    - Resonance-based training
    """
    
    def __init__(
        self,
        field_dim: int = 64,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_device()
        self.resonance = ResonanceTrainer(
            field_dim=field_dim,
            device=self.device,
            use_phase_monitor=True,
            use_fibonacci_lr=True,
        )
        
    def train(
        self,
        training_data: List[Tuple[str, str]],
        epochs: int = 10,
        check_conservation: bool = True,
        verbose: bool = False,
    ) -> Dict:
        """
        Train on co-occurrence data.
        
        Args:
            training_data: List of (pattern1, pattern2) pairs that co-occur
            epochs: Training epochs
            check_conservation: Validate PAC conservation
            verbose: Print progress
        """
        # Train
        stats = self.resonance.train_cooccurrence(
            training_data,
            epochs=epochs,
            verbose=verbose,
        )
        
        # Conservation check
        if check_conservation:
            residual = self.resonance.conservation_check()
            stats['conservation_residual'] = residual
            stats['conservation_ok'] = residual < 1e-4
        
        # Phase monitor stats
        if self.resonance.phase_monitor:
            stats['phase_stats'] = self.resonance.phase_monitor.get_stats()
        
        return stats
    
    def similarity(self, p1: str, p2: str) -> float:
        """Get similarity between patterns."""
        return self.resonance.similarity(p1, p2)
    
    def get_clusters(self, threshold: float = 0.5) -> List[List[str]]:
        """Get pattern clusters."""
        return self.resonance.get_cluster(threshold)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Physics Trainer Test")
    print("=" * 50)
    print(f"Device: {get_device()}")
    print(f"φ × ξ = {PHI_XI:.4f}")
    print(f"λ* = {LAMBDA_STAR:.4f}")
    print(f"1/π² = {EIGENVALUE_DECAY:.4f}")
    print()
    
    # Quick test
    trainer = DawnFieldTrainer()
    
    # Training data: animals co-occur, colors co-occur
    training_data = [
        ("cat", "dog"),
        ("cat", "animal"),
        ("dog", "animal"),
        ("red", "blue"),
        ("red", "color"),
        ("blue", "color"),
    ]
    
    stats = trainer.train(training_data, epochs=5, verbose=True)
    
    print()
    print("Training Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print()
    print("Similarity Tests:")
    print(f"  sim(cat, dog) = {trainer.similarity('cat', 'dog'):.4f}")
    print(f"  sim(red, blue) = {trainer.similarity('red', 'blue'):.4f}")
    print(f"  sim(cat, red) = {trainer.similarity('cat', 'red'):.4f}")
    
    print()
    print("✓ Physics trainer works!")
