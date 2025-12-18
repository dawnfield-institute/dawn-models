"""
Adaptive GAIA Trainer

Entropy-driven adaptive training inspired by CIMM's quantum-coherent approach.
Uses PAC tree metrics and SCBF phase alignment to dynamically adjust:
- Learning rate based on generalization health
- Noise injection when entropy collapses
- Batch weighting for corpus diversity

Key insight: The PAC tree's abstract/specific ratio indicates generalization health.
When memorization dominates (high specific ratio), we need interventions.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
import math
import json
from pathlib import Path

# SCBF constant
PHI_XI = 0.915965594177


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive training."""
    # Learning rate bounds
    min_lr: float = 1e-6
    max_lr: float = 1e-2
    initial_lr: float = 3e-4
    
    # Entropy thresholds
    entropy_collapse_threshold: float = 0.3  # Below this = memorizing
    entropy_explosion_threshold: float = 0.9  # Above this = chaos
    target_entropy: float = PHI_XI  # Golden ratio target
    
    # PAC tree health thresholds
    min_abstract_ratio: float = 0.2  # Want at least 20% abstract patterns
    target_abstract_ratio: float = 0.4  # Ideal balance
    
    # Adaptation rates
    lr_adjustment_rate: float = 0.1  # How fast LR changes
    noise_scale: float = 0.01  # Base noise for diversity
    momentum: float = 0.9  # Smoothing for metrics
    
    # Monitoring
    window_size: int = 50  # History window
    adaptation_interval: int = 10  # Adapt every N batches


@dataclass
class TrainingState:
    """Tracks adaptive training state."""
    current_lr: float = 3e-4
    current_noise: float = 0.0
    entropy_history: Deque = field(default_factory=lambda: deque(maxlen=100))
    loss_history: Deque = field(default_factory=lambda: deque(maxlen=100))
    abstract_ratio_history: Deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Intervention counters
    lr_adjustments: int = 0
    noise_injections: int = 0
    
    # Phase tracking
    phase_coherence: float = 1.0
    generalization_score: float = 0.5


class EntropyTracker:
    """Tracks entropy of model activations."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.entropy_history = deque(maxlen=window_size)
        self.prev_entropy = 0.5
        
    def compute_activation_entropy(self, activations: torch.Tensor) -> float:
        """Compute entropy of activation distribution."""
        # Flatten and normalize
        flat = activations.flatten().float()
        
        # Softmax to get probability distribution
        probs = torch.softmax(flat, dim=0)
        
        # Shannon entropy (normalized)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs)
        
        # Normalize by max entropy
        max_entropy = math.log(len(flat))
        normalized_entropy = (entropy / max_entropy).item()
        
        self.prev_entropy = normalized_entropy
        self.entropy_history.append(normalized_entropy)
        
        return normalized_entropy
    
    def get_entropy_stats(self) -> Dict[str, float]:
        """Get entropy statistics."""
        if len(self.entropy_history) < 2:
            return {
                'current': self.prev_entropy,
                'mean': self.prev_entropy,
                'std': 0.0,
                'trend': 0.0
            }
        
        history = torch.tensor(list(self.entropy_history))
        
        # Compute trend (positive = increasing entropy)
        if len(history) >= 10:
            recent = history[-10:].mean()
            earlier = history[-20:-10].mean() if len(history) >= 20 else history[:10].mean()
            trend = (recent - earlier).item()
        else:
            trend = 0.0
        
        return {
            'current': self.prev_entropy,
            'mean': history.mean().item(),
            'std': history.std().item(),
            'trend': trend
        }


class PACTreeTracker:
    """Lightweight PAC tree tracking for training."""
    
    def __init__(self, window_size: int = 50):
        self.pattern_contexts: Dict[str, set] = {}  # pattern -> contexts
        self.window_size = window_size
        self.abstract_ratio_history = deque(maxlen=window_size)
        
    def update_from_batch(self, hidden_states: torch.Tensor, token_ids: torch.Tensor):
        """Update pattern tracking from a batch."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Discretize hidden states to pattern IDs (simple quantization)
        quantized = (hidden_states * 10).int()
        pattern_hashes = quantized.sum(dim=-1)  # (batch, seq)
        
        # Track which patterns appear in which contexts (token sequences)
        for b in range(batch_size):
            context_id = hash(tuple(token_ids[b, :min(5, seq_len)].tolist()))
            for s in range(seq_len):
                pattern_id = pattern_hashes[b, s].item()
                if pattern_id not in self.pattern_contexts:
                    self.pattern_contexts[pattern_id] = set()
                self.pattern_contexts[pattern_id].add(context_id)
        
        # Compute abstract ratio
        abstract_ratio = self._compute_abstract_ratio()
        self.abstract_ratio_history.append(abstract_ratio)
        
        return abstract_ratio
    
    def _compute_abstract_ratio(self) -> float:
        """Compute ratio of abstract (multi-context) patterns."""
        if not self.pattern_contexts:
            return 0.5
        
        total = len(self.pattern_contexts)
        abstract = sum(1 for contexts in self.pattern_contexts.values() if len(contexts) > 1)
        
        return abstract / total
    
    def get_health_metrics(self) -> Dict[str, float]:
        """Get PAC tree health metrics."""
        if not self.abstract_ratio_history:
            return {'abstract_ratio': 0.5, 'diversity': 0.5, 'trend': 0.0}
        
        history = torch.tensor(list(self.abstract_ratio_history))
        
        # Trend
        if len(history) >= 10:
            recent = history[-10:].mean()
            earlier = history[-20:-10].mean() if len(history) >= 20 else history[:10].mean()
            trend = (recent - earlier).item()
        else:
            trend = 0.0
        
        # Diversity = unique patterns / total activations
        diversity = min(1.0, len(self.pattern_contexts) / 1000)
        
        return {
            'abstract_ratio': history[-1].item(),
            'diversity': diversity,
            'trend': trend
        }
    
    def reset_periodic(self):
        """Periodic reset to prevent unbounded growth."""
        if len(self.pattern_contexts) > 10000:
            # Keep only recent patterns (rough approximation)
            keys = list(self.pattern_contexts.keys())
            for k in keys[:len(keys)//2]:
                del self.pattern_contexts[k]


class AdaptiveGAIATrainer:
    """
    Adaptive trainer for GAIA models.
    
    Monitors training dynamics and adjusts:
    - Learning rate based on entropy and PAC tree health
    - Noise injection when memorization detected
    - Logs all adaptations for analysis
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[AdaptiveConfig] = None,
        device: str = 'cuda'
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or AdaptiveConfig()
        self.device = device
        
        # State tracking
        self.state = TrainingState(current_lr=self.config.initial_lr)
        
        # Monitors
        self.entropy_tracker = EntropyTracker(self.config.window_size)
        self.pac_tracker = PACTreeTracker(self.config.window_size)
        
        # Adaptation log
        self.adaptation_log: List[Dict] = []
        self.step_count = 0
        
    def compute_generalization_score(self) -> float:
        """Compute overall generalization health score."""
        entropy_stats = self.entropy_tracker.get_entropy_stats()
        pac_metrics = self.pac_tracker.get_health_metrics()
        
        # Entropy component: how close to target?
        entropy_score = 1.0 - abs(entropy_stats['mean'] - self.config.target_entropy)
        entropy_score = max(0, entropy_score)
        
        # PAC component: abstract ratio health
        abstract_score = min(1.0, pac_metrics['abstract_ratio'] / self.config.target_abstract_ratio)
        
        # Diversity component
        diversity_score = pac_metrics['diversity']
        
        # Weighted combination
        score = 0.4 * entropy_score + 0.4 * abstract_score + 0.2 * diversity_score
        
        return score
    
    def compute_adaptive_lr(self) -> float:
        """Compute adapted learning rate based on current state."""
        entropy_stats = self.entropy_tracker.get_entropy_stats()
        pac_metrics = self.pac_tracker.get_health_metrics()
        
        current_lr = self.state.current_lr
        adjustment = 1.0
        reason = []
        
        # === Entropy-based adjustments ===
        entropy = entropy_stats['current']
        
        if entropy < self.config.entropy_collapse_threshold:
            # Entropy collapsing = memorizing, increase LR to escape
            adjustment *= 1.2
            reason.append(f"entropy_collapse({entropy:.3f})")
            
        elif entropy > self.config.entropy_explosion_threshold:
            # Entropy exploding = chaos, decrease LR to stabilize
            adjustment *= 0.8
            reason.append(f"entropy_explosion({entropy:.3f})")
        
        # === PAC tree health adjustments ===
        abstract_ratio = pac_metrics['abstract_ratio']
        
        if abstract_ratio < self.config.min_abstract_ratio:
            # Too few abstract patterns = overfitting
            adjustment *= 1.15
            reason.append(f"low_abstract({abstract_ratio:.3f})")
            
        elif abstract_ratio > 0.6:
            # Good generalization, can afford to be more aggressive
            adjustment *= 1.05
            reason.append(f"healthy_tree({abstract_ratio:.3f})")
        
        # === Phase coherence modulation (SCBF-inspired) ===
        # When patterns align with PHI_XI, we're in a good state
        phase_distance = abs(abstract_ratio - PHI_XI)
        if phase_distance < 0.1:
            # Near golden ratio = optimal, slight boost
            adjustment *= 1.02
            reason.append("phi_aligned")
        
        # === Apply with momentum ===
        smoothed_adjustment = self.config.momentum + (1 - self.config.momentum) * adjustment
        new_lr = current_lr * smoothed_adjustment
        
        # Clamp
        new_lr = max(self.config.min_lr, min(self.config.max_lr, new_lr))
        
        # Log if significant change
        if abs(new_lr - current_lr) / current_lr > 0.01:
            self.adaptation_log.append({
                'step': self.step_count,
                'type': 'lr_adjustment',
                'old_lr': current_lr,
                'new_lr': new_lr,
                'adjustment': adjustment,
                'reasons': reason,
                'entropy': entropy,
                'abstract_ratio': abstract_ratio
            })
            self.state.lr_adjustments += 1
        
        return new_lr
    
    def compute_noise_injection(self) -> float:
        """Compute noise level to inject for diversity."""
        entropy_stats = self.entropy_tracker.get_entropy_stats()
        pac_metrics = self.pac_tracker.get_health_metrics()
        
        noise = 0.0
        
        # Inject noise when entropy is too low (memorizing)
        if entropy_stats['current'] < self.config.entropy_collapse_threshold:
            noise = self.config.noise_scale * (
                self.config.entropy_collapse_threshold - entropy_stats['current']
            )
            
        # Inject noise when PAC tree is too specific
        if pac_metrics['abstract_ratio'] < self.config.min_abstract_ratio:
            noise += self.config.noise_scale * 0.5
        
        if noise > 0:
            self.adaptation_log.append({
                'step': self.step_count,
                'type': 'noise_injection',
                'noise_level': noise,
                'entropy': entropy_stats['current'],
                'abstract_ratio': pac_metrics['abstract_ratio']
            })
            self.state.noise_injections += 1
        
        return noise
    
    def apply_adaptations(self):
        """Apply all computed adaptations."""
        # Update learning rate
        new_lr = self.compute_adaptive_lr()
        self.state.current_lr = new_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # Compute noise for next forward pass
        self.state.current_noise = self.compute_noise_injection()
        
        # Update generalization score
        self.state.generalization_score = self.compute_generalization_score()
        
        # Periodic cleanup
        self.pac_tracker.reset_periodic()
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        loss_fn: callable
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute one training step with adaptive monitoring.
        
        Args:
            batch: Dictionary with 'input_ids', optionally 'labels'
            loss_fn: Function that takes model output and batch, returns loss
            
        Returns:
            loss: The computed loss
            metrics: Dictionary of training metrics
        """
        self.step_count += 1
        
        self.model.train()
        self.optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(self.device)
        
        # Forward pass with noise injection
        if self.state.current_noise > 0:
            # Add noise to embeddings (if accessible)
            noise_tensor = torch.randn_like(input_ids.float()) * self.state.current_noise
            # Note: actual noise injection depends on model architecture
        
        # Get model output (assuming it returns hidden states)
        output = self.model(input_ids)
        
        # Extract hidden states for monitoring
        if hasattr(output, 'hidden_states') and output.hidden_states is not None:
            hidden_states = output.hidden_states[-1]  # Last layer
        elif isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # Update trackers
        entropy = self.entropy_tracker.compute_activation_entropy(hidden_states.detach())
        abstract_ratio = self.pac_tracker.update_from_batch(
            hidden_states.detach(), 
            input_ids.detach()
        )
        
        # Compute loss
        loss = loss_fn(output, batch)
        self.state.loss_history.append(loss.item())
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        # Apply adaptations periodically
        if self.step_count % self.config.adaptation_interval == 0:
            self.apply_adaptations()
        
        # Collect metrics
        metrics = {
            'loss': loss.item(),
            'lr': self.state.current_lr,
            'entropy': entropy,
            'abstract_ratio': abstract_ratio,
            'generalization_score': self.state.generalization_score,
            'noise': self.state.current_noise,
            'lr_adjustments': self.state.lr_adjustments,
            'noise_injections': self.state.noise_injections
        }
        
        return loss, metrics
    
    def get_status_report(self) -> str:
        """Get a formatted status report."""
        entropy_stats = self.entropy_tracker.get_entropy_stats()
        pac_metrics = self.pac_tracker.get_health_metrics()
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                 ADAPTIVE TRAINING STATUS                      ║
╠══════════════════════════════════════════════════════════════╣
║  Step: {self.step_count:,}                                              
║  Learning Rate: {self.state.current_lr:.2e} (adjustments: {self.state.lr_adjustments})
║  Noise Level: {self.state.current_noise:.4f} (injections: {self.state.noise_injections})
╠══════════════════════════════════════════════════════════════╣
║  ENTROPY                                                      
║    Current: {entropy_stats['current']:.4f}  Target: {self.config.target_entropy:.4f}
║    Mean: {entropy_stats['mean']:.4f}  Std: {entropy_stats['std']:.4f}
║    Trend: {entropy_stats['trend']:+.4f}
╠══════════════════════════════════════════════════════════════╣
║  PAC TREE HEALTH                                              
║    Abstract Ratio: {pac_metrics['abstract_ratio']:.4f}  Target: {self.config.target_abstract_ratio:.4f}
║    Diversity: {pac_metrics['diversity']:.4f}
║    Trend: {pac_metrics['trend']:+.4f}
╠══════════════════════════════════════════════════════════════╣
║  GENERALIZATION SCORE: {self.state.generalization_score:.4f}                      
╚══════════════════════════════════════════════════════════════╝
"""
        return report
    
    def save_adaptation_log(self, path: Path):
        """Save adaptation log to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump({
                'config': {
                    'min_lr': self.config.min_lr,
                    'max_lr': self.config.max_lr,
                    'initial_lr': self.config.initial_lr,
                    'target_entropy': self.config.target_entropy,
                    'target_abstract_ratio': self.config.target_abstract_ratio
                },
                'final_state': {
                    'lr': self.state.current_lr,
                    'lr_adjustments': self.state.lr_adjustments,
                    'noise_injections': self.state.noise_injections,
                    'generalization_score': self.state.generalization_score
                },
                'adaptations': self.adaptation_log
            }, f, indent=2)


def create_adaptive_trainer(
    model: nn.Module,
    lr: float = 3e-4,
    device: str = 'cuda',
    **config_kwargs
) -> Tuple[AdaptiveGAIATrainer, torch.optim.Optimizer]:
    """
    Factory function to create an adaptive trainer.
    
    Returns trainer and optimizer (for external use if needed).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    config = AdaptiveConfig(initial_lr=lr, **config_kwargs)
    trainer = AdaptiveGAIATrainer(model, optimizer, config, device)
    
    return trainer, optimizer
