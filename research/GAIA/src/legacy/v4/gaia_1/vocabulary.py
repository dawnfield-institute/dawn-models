"""
Field Vocabulary - Token embeddings as field patterns

Each token is a unique perturbation pattern in the field.
Similarity between tokens = resonance between patterns.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List, Tuple
from pathlib import Path
import sys

# Fracton imports - handle both package and standalone execution
try:
    from fracton.physics import PHI, XI, PHI_XI
except ImportError:
    # Add fracton to path
    _fracton_path = Path(__file__).resolve().parent.parent.parent.parent.parent.parent.parent / "fracton"
    if _fracton_path.exists():
        sys.path.insert(0, str(_fracton_path))
    from fracton.physics import PHI, XI, PHI_XI


class FieldVocabulary(nn.Module):
    """
    Vocabulary as field patterns.
    
    Each token gets a learnable field pattern. The patterns are
    initialized with structure (spherical harmonics basis) and
    learned during training.
    
    Key insight: We don't need separate embeddings and output projections.
    Input encoding and output decoding use the SAME patterns via resonance.
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        field_dim: int = 256,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.field_dim = field_dim
        self.device = device
        
        # Token patterns - learnable field perturbations
        # Initialized with structured noise (not random!)
        self.patterns = nn.Parameter(
            self._init_patterns(vocab_size, field_dim)
        )
        
        # Pattern norms for efficient resonance computation
        self.register_buffer('_pattern_norms', torch.zeros(vocab_size))
        self._update_norms()
    
    def _init_patterns(self, vocab_size: int, field_dim: int) -> torch.Tensor:
        """
        Initialize patterns with physics-inspired structure.
        
        Uses a combination of:
        - Random basis (diversity)
        - Scaled by position (hierarchy)
        - Normalized to unit energy
        """
        # Start with structured random
        patterns = torch.randn(vocab_size, field_dim)
        
        # Apply frequency-based scaling (like spherical harmonics)
        freqs = torch.arange(field_dim).float() + 1
        freq_scale = 1.0 / torch.sqrt(freqs)  # Higher freqs get smaller amplitude
        patterns = patterns * freq_scale.unsqueeze(0)
        
        # Normalize to unit energy
        norms = torch.norm(patterns, dim=1, keepdim=True)
        patterns = patterns / (norms + 1e-8)
        
        # Scale by PHI for field compatibility
        patterns = patterns * math.sqrt(PHI)
        
        return patterns
    
    def _update_norms(self):
        """Update cached pattern norms."""
        with torch.no_grad():
            self._pattern_norms = torch.norm(self.patterns, dim=1)
    
    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode token IDs to field patterns.
        
        Args:
            token_ids: (batch, seq_len) or (seq_len,)
            
        Returns:
            Field patterns: (batch, seq_len, field_dim) or (seq_len, field_dim)
        """
        return self.patterns[token_ids]
    
    def decode_resonance(
        self, 
        field: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode field state to token probabilities via resonance.
        
        Computes resonance (cosine similarity) between field state
        and all vocabulary patterns.
        
        Args:
            field: (batch, field_dim) or (field_dim,)
            temperature: Softmax temperature
            top_k: Optional top-k filtering
            
        Returns:
            (probs, logits): Token probabilities and raw logits
        """
        # Ensure 2D
        single = field.dim() == 1
        if single:
            field = field.unsqueeze(0)
        
        # Compute resonance (cosine similarity)
        # field: (batch, dim), patterns: (vocab, dim)
        field_norm = torch.norm(field, dim=1, keepdim=True)
        logits = torch.mm(field, self.patterns.T)  # (batch, vocab)
        logits = logits / (field_norm * self._pattern_norms.unsqueeze(0) + 1e-8)
        
        # Scale logits for better gradients
        logits = logits * math.sqrt(self.field_dim)
        
        # Apply temperature
        logits = logits / temperature
        
        # Optional top-k filtering
        if top_k is not None:
            values, indices = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
            # Use where instead of multiplication to avoid NaN from 0 * -inf
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, indices, True)
            logits = torch.where(mask, logits, torch.tensor(float('-inf'), device=logits.device))
        
        probs = torch.softmax(logits, dim=-1)
        
        if single:
            probs = probs.squeeze(0)
            logits = logits.squeeze(0)
        
        return probs, logits
    
    def sample(
        self,
        field: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Sample token from field state.
        
        Args:
            field: (batch, field_dim) or (field_dim,)
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            
        Returns:
            Sampled token IDs
        """
        probs, _ = self.decode_resonance(field, temperature, top_k)
        
        # Handle NaN/Inf
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            # Fall back to uniform
            probs = torch.ones_like(probs) / probs.shape[-1]
        
        # Ensure no negative values
        probs = torch.clamp(probs, min=1e-10)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Optional nucleus sampling
        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs = sorted_probs.clone()
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-10)
            
            # Sample from filtered distribution
            if probs.dim() == 1:
                idx = torch.multinomial(sorted_probs, 1)
                token = sorted_idx[idx]
            else:
                tokens = []
                for i in range(probs.shape[0]):
                    idx = torch.multinomial(sorted_probs[i], 1)
                    tokens.append(sorted_idx[i, idx])
                token = torch.stack(tokens)
        else:
            # Regular sampling
            if probs.dim() == 1:
                token = torch.multinomial(probs, 1)
            else:
                token = torch.multinomial(probs, 1).squeeze(-1)
        
        return token
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode tokens to field patterns."""
        return self.encode(token_ids)
