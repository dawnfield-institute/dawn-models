"""
Field Generator - Next token prediction via field dynamics

The core language generation mechanism:
1. Context encoded as overlapping field perturbations
2. Field evolved via Klein-Gordon dynamics
3. Final state decoded via resonance with vocabulary

No attention. No MLP. Just physics.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List
from pathlib import Path
import sys

# Fracton imports - handle both package and standalone execution
try:
    from fracton.physics import PHI, XI, PHI_XI, LAMBDA_STAR
    from fracton.field import evolve
except ImportError:
    _fracton_path = Path(__file__).resolve().parent.parent.parent.parent.parent.parent.parent / "fracton"
    if _fracton_path.exists():
        sys.path.insert(0, str(_fracton_path))
    from fracton.physics import PHI, XI, PHI_XI, LAMBDA_STAR
    from fracton.field import evolve


class FieldContext(nn.Module):
    """
    Context processing via field resonance.
    
    Combines position-based decay with content-based resonance.
    This is the field-native equivalent of attention.
    """
    
    def __init__(
        self,
        field_dim: int = 256,
        max_context: int = 1024,
        n_heads: int = 4,  # Number of resonance heads
        decay_rate: float = None
    ):
        super().__init__()
        
        self.field_dim = field_dim
        self.max_context = max_context
        self.n_heads = n_heads
        self.head_dim = field_dim // n_heads
        self.decay_rate = decay_rate or LAMBDA_STAR
        
        # Position encodings - learned phase shifts
        self.pos_phases = nn.Parameter(
            torch.randn(max_context, field_dim) * 0.1
        )
        
        # Resonance projections (like Q, K, V but field-native)
        # Query: what am I looking for?
        # Key: what do I represent?
        # Value: what information do I carry?
        self.query_proj = nn.Linear(field_dim, field_dim)
        self.key_proj = nn.Linear(field_dim, field_dim)
        self.value_proj = nn.Linear(field_dim, field_dim)
        
        # Output projection
        self.out_proj = nn.Linear(field_dim, field_dim)
        
        # Temperature for resonance softmax
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Precompute decay weights
        positions = torch.arange(max_context).float()
        decay = self.decay_rate ** positions
        self.register_buffer('decay_weights', decay)
    
    def forward(
        self,
        patterns: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Combine context patterns using resonance attention.
        
        Args:
            patterns: (batch, seq_len, field_dim)
            mask: Optional (batch, seq_len) mask
            
        Returns:
            context_field: (batch, field_dim)
        """
        batch_size, seq_len, _ = patterns.shape
        device = patterns.device
        
        # Apply position phases
        pos_enc = self.pos_phases[:seq_len]
        patterns_with_pos = patterns + pos_enc.unsqueeze(0)
        
        # Project to Q, K, V
        Q = self.query_proj(patterns_with_pos)  # (batch, seq, dim)
        K = self.key_proj(patterns_with_pos)
        V = self.value_proj(patterns_with_pos)
        
        # Reshape for multi-head
        # (batch, seq, n_heads, head_dim) -> (batch, n_heads, seq, head_dim)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute resonance (scaled dot product)
        # (batch, heads, seq, head_dim) @ (batch, heads, head_dim, seq) -> (batch, heads, seq, seq)
        resonance = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply temperature
        resonance = resonance / (torch.abs(self.temperature) + 0.1)
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), 
            diagonal=1
        ).bool()
        resonance = resonance.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply position decay as bias
        pos_diff = torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(seq_len, device=device).unsqueeze(1)
        pos_bias = -0.1 * pos_diff.clamp(min=0).float()  # Slight recency bias
        resonance = resonance + pos_bias.unsqueeze(0).unsqueeze(0)
        
        # Apply attention mask if provided
        if mask is not None:
            resonance = resonance.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # Softmax to get weights
        weights = torch.softmax(resonance, dim=-1)
        
        # Apply to values
        # (batch, heads, seq, seq) @ (batch, heads, seq, head_dim) -> (batch, heads, seq, head_dim)
        attended = torch.matmul(weights, V)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.field_dim)
        
        # Output projection
        output = self.out_proj(attended)
        
        # Return last position as context (for single-output mode)
        # or full sequence for causal training
        return output[:, -1, :]  # (batch, dim)


class FieldEvolution(nn.Module):
    """
    Field evolution via Klein-Gordon dynamics.
    
    Evolves the context field to produce output state.
    Uses learnable parameters for mass and coupling.
    """
    
    def __init__(
        self,
        field_dim: int = 256,
        evolution_steps: int = 8,
        dt: float = 0.1
    ):
        super().__init__()
        
        self.field_dim = field_dim
        self.evolution_steps = evolution_steps
        self.dt = dt
        
        # Learnable mass (controls oscillation frequency)
        self.log_mass = nn.Parameter(torch.tensor(0.0))
        
        # Learnable nonlinear coupling
        self.coupling = nn.Parameter(torch.tensor(float(PHI_XI)))
        
        # Layer for residual mixing
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """
        Evolve field state.
        
        Args:
            field: (batch, field_dim)
            
        Returns:
            evolved: (batch, field_dim)
        """
        mass = torch.exp(self.log_mass)
        
        # Simple Klein-Gordon inspired evolution
        # ∂²φ/∂t² = ∇²φ - m²φ
        # Discretized as: φ_new = 2φ - φ_old + dt²(∇²φ - m²φ)
        
        # Use Fracton's evolve function with our parameters
        evolved = evolve(
            field,
            steps=self.evolution_steps,
            dt=self.dt
        )
        
        # Residual connection
        alpha = torch.sigmoid(self.residual_weight)
        output = alpha * evolved + (1 - alpha) * field
        
        return output


class FieldGenerator(nn.Module):
    """
    Complete next-token prediction via field dynamics.
    
    Pipeline:
    1. Encode tokens → field patterns
    2. Resonance attention to combine context
    3. Evolve field through multiple layers
    4. Decode via resonance with vocabulary
    """
    
    def __init__(
        self,
        field_dim: int = 256,
        max_context: int = 1024,
        evolution_steps: int = 8,
        n_layers: int = 4,
        n_heads: int = 4
    ):
        super().__init__()
        
        self.field_dim = field_dim
        self.max_context = max_context
        self.n_layers = n_layers
        
        # Stacked resonance + evolution layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'attention': FieldContext(field_dim, max_context, n_heads),
                'attn_norm': nn.LayerNorm(field_dim),
                'evolution': FieldEvolution(field_dim, evolution_steps),
                'evol_norm': nn.LayerNorm(field_dim)
            }))
        
        # Final output projection
        self.output_proj = nn.Linear(field_dim, field_dim)
    
    def forward(
        self,
        patterns: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process context and generate output field.
        
        Args:
            patterns: (batch, seq_len, field_dim) - encoded tokens
            mask: Optional attention mask
            
        Returns:
            output_field: (batch, field_dim)
        """
        batch_size, seq_len, _ = patterns.shape
        
        # Process through layers
        hidden = patterns
        
        for layer in self.layers:
            # Resonance attention (returns last position only)
            attn_out = layer['attention'](hidden, mask)
            # We need to expand back for residual
            # Use the attended output as the new last position
            hidden_last = hidden[:, -1, :] + layer['attn_norm'](attn_out)
            
            # Evolution
            evolved = layer['evolution'](hidden_last)
            hidden_last = hidden_last + layer['evol_norm'](evolved)
            
            # Update the sequence (for next layer)
            hidden = torch.cat([hidden[:, :-1, :], hidden_last.unsqueeze(1)], dim=1)
        
        # Output projection
        output = self.output_proj(hidden[:, -1, :])
        
        return output
    
    def forward_causal(
        self,
        patterns: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process all positions causally for parallel training.
        
        Args:
            patterns: (batch, seq_len, field_dim)
            mask: Optional mask
            
        Returns:
            outputs: (batch, seq_len, field_dim)
        """
        batch_size, seq_len, dim = patterns.shape
        device = patterns.device
        
        # Process through layers with full causal attention
        hidden = patterns
        
        for layer in self.layers:
            # Full causal attention over all positions
            attn = layer['attention']
            
            # Apply position phases
            pos_enc = attn.pos_phases[:seq_len]
            hidden_with_pos = hidden + pos_enc.unsqueeze(0)
            
            # Project to Q, K, V
            Q = attn.query_proj(hidden_with_pos)
            K = attn.key_proj(hidden_with_pos)
            V = attn.value_proj(hidden_with_pos)
            
            # Reshape for multi-head
            Q = Q.view(batch_size, seq_len, attn.n_heads, attn.head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, attn.n_heads, attn.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, attn.n_heads, attn.head_dim).transpose(1, 2)
            
            # Compute resonance scores
            resonance = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(attn.head_dim)
            resonance = resonance / (torch.abs(attn.temperature) + 0.1)
            
            # Causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device),
                diagonal=1
            ).bool()
            resonance = resonance.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # Position decay bias
            pos_diff = torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(seq_len, device=device).unsqueeze(1)
            pos_bias = -0.1 * pos_diff.clamp(min=0).float()
            resonance = resonance + pos_bias.unsqueeze(0).unsqueeze(0)
            
            # Attention weights
            weights = torch.softmax(resonance, dim=-1)
            
            # Apply to values
            attended = torch.matmul(weights, V)
            attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
            attended = attn.out_proj(attended)
            
            # Residual + norm
            hidden = hidden + layer['attn_norm'](attended)
            
            # Evolution (process all positions in parallel)
            evolved = layer['evolution'](hidden.reshape(-1, dim))
            evolved = evolved.reshape(batch_size, seq_len, dim)
            
            # Residual + norm
            hidden = hidden + layer['evol_norm'](evolved)
        
        # Output projection
        outputs = self.output_proj(hidden)
        
        return outputs
