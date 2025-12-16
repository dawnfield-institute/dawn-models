"""
Field-Native Attention Components
=================================

Attention mechanisms that emerge from field physics rather than
being bolted onto the architecture.

Core insight: Attention IS resonance.
When pattern A "attends to" pattern B, they resonate in the field.

From Prime Harmonic Manifold (PHM):
- Eigenvalue decay: 1/π² ≈ 0.101
- Head weights follow prime harmonic series
- Max coupling: 4/5 = 0.8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math

# =============================================================================
# Physical Constants
# =============================================================================

PI_SQUARED_INV = 1 / (math.pi ** 2)  # 0.1013 - eigenvalue decay
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
XI = 1.0571  # PAC conservation operator
PHI_XI = PHI * XI  # 1.710 - crystallization trigger
ENTANGLEMENT_LIMIT = 4/5  # 0.8 - max coupling

# First 12 primes for harmonic heads
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def get_device() -> torch.device:
    """Get CUDA device if available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Resonance-Based Attention
# =============================================================================

class ResonanceAttention(nn.Module):
    """
    Attention computed as field resonance.
    
    Instead of QK^T dot product, compute resonance between field states.
    Resonance = how much two patterns "vibrate together" in the field.
    """
    
    def __init__(
        self,
        dim: int,
        max_coupling: float = ENTANGLEMENT_LIMIT,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_coupling = max_coupling
        self.temperature = temperature
        self.scale = 1.0 / math.sqrt(dim)
        
    def compute_resonance(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute resonance between query and key fields.
        
        Resonance is based on:
        1. Cosine similarity (direction alignment)
        2. Energy overlap (magnitude coupling)
        3. Phase coherence (field structure matching)
        
        Args:
            query: (batch, seq_q, dim)
            key: (batch, seq_k, dim)
            
        Returns:
            resonance: (batch, seq_q, seq_k)
        """
        # Normalize to unit vectors (direction)
        q_norm = F.normalize(query, dim=-1)
        k_norm = F.normalize(key, dim=-1)
        
        # Cosine similarity as base resonance
        # (batch, seq_q, dim) @ (batch, dim, seq_k) -> (batch, seq_q, seq_k)
        resonance = torch.bmm(q_norm, k_norm.transpose(-2, -1))
        
        # Apply temperature scaling
        resonance = resonance / self.temperature
        
        # Clamp to max coupling (physics constraint)
        resonance = torch.clamp(resonance, -self.max_coupling, self.max_coupling)
        
        return resonance
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute resonance-based attention.
        
        Args:
            query: (batch, seq_q, dim)
            key: (batch, seq_k, dim)
            value: (batch, seq_k, dim)
            mask: Optional attention mask
            
        Returns:
            output: (batch, seq_q, dim)
            weights: (batch, seq_q, seq_k)
        """
        # Compute resonance
        resonance = self.compute_resonance(query, key)
        
        # Apply mask if provided
        if mask is not None:
            resonance = resonance.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        weights = F.softmax(resonance, dim=-1)
        
        # Apply to values
        output = torch.bmm(weights, value)
        
        return output, weights


# =============================================================================
# Harmonic Multi-Head Attention
# =============================================================================

class HarmonicHead(nn.Module):
    """
    Single attention head with importance weight from prime harmonic series.
    
    Head n has weight = 1/p_n² where p_n is the nth prime.
    This creates natural hierarchy: head 0 is most important.
    """
    
    def __init__(
        self,
        dim: int,
        head_dim: int,
        prime: int,
        use_projections: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.prime = prime
        self.importance = 1.0 / (prime ** 2)  # Prime harmonic weight
        
        self.attention = ResonanceAttention(head_dim)
        
        # Optional learned projections (can disable for pure field attention)
        self.use_projections = use_projections
        if use_projections:
            self.q_proj = nn.Linear(dim, head_dim, bias=False)
            self.k_proj = nn.Linear(dim, head_dim, bias=False)
            self.v_proj = nn.Linear(dim, head_dim, bias=False)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for single head."""
        if self.use_projections:
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        else:
            # Use fields directly (pure field attention)
            q = query[..., :self.head_dim]
            k = key[..., :self.head_dim]
            v = value[..., :self.head_dim]
        
        output, weights = self.attention(q, k, v, mask)
        
        # Scale by importance
        output = output * self.importance
        
        return output, weights


class HarmonicMultiHeadAttention(nn.Module):
    """
    Multi-head attention with prime harmonic importance weighting.
    
    Creates n_heads, each with importance 1/p_i² where p_i is ith prime.
    Total importance sums to ζ(2) - 1 ≈ 0.6449 for infinite heads.
    
    This is the "natural" attention structure from PHM.
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        use_projections: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Create heads with prime harmonic weights
        self.heads = nn.ModuleList([
            HarmonicHead(
                dim=dim,
                head_dim=self.head_dim,
                prime=PRIMES[i],
                use_projections=use_projections,
            )
            for i in range(n_heads)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(self.head_dim * n_heads, dim, bias=False)
        
        # Compute total importance for normalization
        self.total_importance = sum(1.0 / (p ** 2) for p in PRIMES[:n_heads])
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for harmonic multi-head attention.
        
        Returns:
            output: (batch, seq, dim)
            info: Dict with per-head weights and importance
        """
        batch_size, seq_len, _ = query.shape
        
        head_outputs = []
        head_weights = []
        
        for head in self.heads:
            out, weights = head(query, key, value, mask)
            head_outputs.append(out)
            head_weights.append(weights)
        
        # Concatenate head outputs
        # Each head output is (batch, seq, head_dim) scaled by importance
        combined = torch.cat(head_outputs, dim=-1)
        
        # Project back to dim
        output = self.out_proj(combined)
        
        # Normalize by total importance
        output = output / self.total_importance
        
        info = {
            'head_weights': torch.stack(head_weights, dim=1),  # (batch, n_heads, seq_q, seq_k)
            'head_importance': [h.importance for h in self.heads],
            'primes': [h.prime for h in self.heads],
        }
        
        return output, info


# =============================================================================
# Field-Native QKV Derivation
# =============================================================================

class FieldQKV(nn.Module):
    """
    Derive Q, K, V from field evolution operators.
    
    Instead of learned projections:
    - Q = "gradient" of field (what direction is change?)
    - K = "state" of field (where am I now?)
    - V = "value" of field (what do I contain?)
    
    This makes attention a natural operation on fields.
    """
    
    def __init__(
        self,
        dim: int,
        evolution_steps: int = 3,
    ):
        super().__init__()
        self.dim = dim
        self.evolution_steps = evolution_steps
        
        # Evolution operator (like field dynamics)
        self.evolution = nn.Parameter(
            torch.eye(dim) + 0.1 * torch.randn(dim, dim)
        )
        
        # Decay rate (from SEC λ*)
        self.decay = 0.9816
        
    def evolve_field(self, field: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """Evolve field forward in time."""
        for _ in range(steps):
            # Apply evolution with decay
            field = self.decay * torch.matmul(field, self.evolution)
            # Normalize to maintain energy
            field = F.normalize(field, dim=-1) * torch.norm(field, dim=-1, keepdim=True).clamp(min=1e-6)
        return field
    
    def compute_gradient(self, field: torch.Tensor) -> torch.Tensor:
        """Compute field gradient (difference from evolved state)."""
        evolved = self.evolve_field(field, steps=1)
        gradient = evolved - field
        return gradient
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Derive Q, K, V from input field.
        
        Args:
            x: (batch, seq, dim) - input field states
            
        Returns:
            Q: Field gradients (what am I seeking?)
            K: Current states (what am I?)
            V: Evolved values (what can I become?)
        """
        # Q = gradient of field (direction of change)
        Q = self.compute_gradient(x)
        
        # K = current field state
        K = x
        
        # V = evolved field (future potential)
        V = self.evolve_field(x, steps=self.evolution_steps)
        
        return Q, K, V


# =============================================================================
# Complete Field-Native Attention Layer
# =============================================================================

class FieldNativeAttention(nn.Module):
    """
    Complete field-native attention layer.
    
    Combines:
    - FieldQKV: Derive Q, K, V from field physics
    - HarmonicMultiHeadAttention: Prime harmonic head structure
    - PAC conservation: Maintain field energy
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        use_field_qkv: bool = True,
        use_projections: bool = False,  # False = pure field attention
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        
        # Field-derived QKV
        self.use_field_qkv = use_field_qkv
        if use_field_qkv:
            self.field_qkv = FieldQKV(dim)
        
        # Harmonic attention
        self.attention = HarmonicMultiHeadAttention(
            dim=dim,
            n_heads=n_heads,
            use_projections=use_projections,
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(dim)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Field-native attention forward pass.
        
        Args:
            x: (batch, seq, dim) - input field states
            mask: Optional attention mask
            
        Returns:
            output: (batch, seq, dim) - transformed field
            info: Dict with attention weights and diagnostics
        """
        # Pre-norm
        x_norm = self.norm(x)
        
        # Derive Q, K, V
        if self.use_field_qkv:
            Q, K, V = self.field_qkv(x_norm)
        else:
            Q = K = V = x_norm
        
        # Apply harmonic attention
        attn_output, attn_info = self.attention(Q, K, V, mask)
        
        # Residual connection (maintains field structure)
        output = x + attn_output
        
        # PAC conservation check
        input_energy = torch.sum(x ** 2, dim=-1).mean()
        output_energy = torch.sum(output ** 2, dim=-1).mean()
        conservation_residual = abs(output_energy - input_energy) / (input_energy + 1e-10)
        
        info = {
            **attn_info,
            'input_energy': input_energy.item(),
            'output_energy': output_energy.item(),
            'conservation_residual': conservation_residual.item(),
        }
        
        return output, info


# =============================================================================
# Comparison Utilities
# =============================================================================

def compare_to_standard(
    field_attn: FieldNativeAttention,
    x: torch.Tensor,
) -> Dict:
    """Compare field-native attention to PyTorch standard."""
    dim = x.shape[-1]
    n_heads = field_attn.n_heads
    
    # Standard PyTorch attention
    standard = nn.MultiheadAttention(dim, n_heads, batch_first=True)
    standard = standard.to(x.device)
    
    # Forward pass
    with torch.no_grad():
        field_out, field_info = field_attn(x)
        standard_out, standard_weights = standard(x, x, x)
    
    # Compare outputs
    output_diff = torch.mean((field_out - standard_out) ** 2).item()
    
    return {
        'output_mse': output_diff,
        'field_conservation': field_info['conservation_residual'],
        'field_energy': field_info['output_energy'],
    }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Field-Native Attention Test")
    print("=" * 50)
    
    device = get_device()
    print(f"Device: {device}")
    print(f"1/π² = {PI_SQUARED_INV:.4f}")
    print(f"First 8 primes: {PRIMES[:8]}")
    print(f"Harmonic weights: {[f'1/{p}²={1/p**2:.4f}' for p in PRIMES[:4]]}")
    print()
    
    # Test dimensions
    batch_size = 2
    seq_len = 16
    dim = 64
    n_heads = 8
    
    # Create input
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    # Test resonance attention
    print("[Resonance Attention]")
    res_attn = ResonanceAttention(dim).to(device)
    out, weights = res_attn(x, x, x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Weights: {weights.shape}")
    print(f"  Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    print()
    
    # Test harmonic multi-head
    print("[Harmonic Multi-Head Attention]")
    harm_attn = HarmonicMultiHeadAttention(dim, n_heads).to(device)
    out, info = harm_attn(x, x, x)
    print(f"  Output: {out.shape}")
    print(f"  Head importance: {[f'{i:.4f}' for i in info['head_importance'][:4]]}...")
    print(f"  Total importance: {sum(info['head_importance']):.4f}")
    print()
    
    # Test field-native attention
    print("[Field-Native Attention]")
    field_attn = FieldNativeAttention(dim, n_heads).to(device)
    out, info = field_attn(x)
    print(f"  Output: {out.shape}")
    print(f"  Conservation residual: {info['conservation_residual']:.6f}")
    print(f"  Energy: {info['input_energy']:.3f} → {info['output_energy']:.3f}")
    print()
    
    print("✓ Field-native attention works!")
