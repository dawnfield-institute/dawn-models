"""
GAIA-1: First Talkable Field-Native Language Model

The complete model combining:
- FieldVocabulary: Token ↔ field pattern encoding
- FieldGenerator: Context processing via evolution
- Training: Cross-entropy on resonance logits
- Generation: Autoregressive sampling

This is a pure Dawn Field Theory implementation.
No transformers. No attention matrices. Just physics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import sys

# Fracton imports - handle both package and standalone execution
try:
    from fracton.physics import PHI, XI, PHI_XI
except ImportError:
    _fracton_path = Path(__file__).resolve().parent.parent.parent.parent.parent.parent.parent / "fracton"
    if _fracton_path.exists():
        sys.path.insert(0, str(_fracton_path))
    from fracton.physics import PHI, XI, PHI_XI

# Local imports - handle both package and standalone
try:
    from .vocabulary import FieldVocabulary
    from .generator import FieldGenerator
except ImportError:
    from vocabulary import FieldVocabulary
    from generator import FieldGenerator


@dataclass
class GAIA1Config:
    """Configuration for GAIA-1 model."""
    
    # Vocabulary
    vocab_size: int = 50257  # GPT-2 vocabulary
    
    # Field dimensions
    field_dim: int = 256
    
    # Context
    max_context: int = 512
    
    # Evolution
    evolution_steps: int = 8
    n_layers: int = 4
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Generation
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __post_init__(self):
        # Validate
        assert self.field_dim > 0
        assert self.vocab_size > 0


class GAIA1(nn.Module):
    """
    GAIA-1: First Talkable Field-Native Model
    
    A pure physics-based language model using Dawn Field Theory.
    
    Architecture:
    - Vocabulary patterns (learnable field perturbations)
    - Context superposition (weighted by recency)
    - Field evolution (Klein-Gordon dynamics)
    - Resonance decoding (cosine similarity to vocab)
    
    Usage:
        model = GAIA1(GAIA1Config())
        
        # Training
        loss = model.compute_loss(input_ids, target_ids)
        
        # Generation
        output = model.generate("Hello", max_tokens=50)
    """
    
    def __init__(self, config: GAIA1Config):
        super().__init__()
        
        self.config = config
        
        # Vocabulary as field patterns
        self.vocab = FieldVocabulary(
            vocab_size=config.vocab_size,
            field_dim=config.field_dim,
            device=config.device
        )
        
        # Field generator
        self.generator = FieldGenerator(
            field_dim=config.field_dim,
            max_context=config.max_context,
            evolution_steps=config.evolution_steps,
            n_layers=config.n_layers
        )
        
        # Move to device
        self.to(config.device)
        
        # Tokenizer (lazy loaded)
        self._tokenizer = None
    
    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            try:
                from transformers import GPT2Tokenizer
                self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                self._tokenizer.pad_token = self._tokenizer.eos_token
            except ImportError:
                raise ImportError("Install transformers: pip install transformers")
        return self._tokenizer
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        tokens = self.tokenizer.encode(text, return_tensors='pt')
        return tokens.to(self.config.device)
    
    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        if token_ids.dim() > 1:
            token_ids = token_ids.squeeze(0)
        return self.tokenizer.decode(token_ids.tolist())
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: Optional (batch, seq_len)
            
        Returns:
            (logits, hidden_states)
        """
        # Encode to field patterns
        patterns = self.vocab.encode(input_ids)  # (batch, seq, dim)
        
        # Process causally for each position
        # This is the slow path - for training we want parallel
        hidden = self.generator.forward_causal(patterns, attention_mask)
        
        # Decode each position to logits
        batch_size, seq_len, _ = hidden.shape
        
        # Reshape for batch decoding
        hidden_flat = hidden.view(-1, self.config.field_dim)
        
        # Get logits via resonance
        _, logits = self.vocab.decode_resonance(hidden_flat, temperature=1.0)
        
        logits = logits.view(batch_size, seq_len, -1)
        
        return logits, hidden
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute training loss.
        
        For language modeling, labels are input_ids shifted by 1.
        
        Args:
            input_ids: (batch, seq_len)
            labels: Optional (batch, seq_len) - if None, uses shifted input_ids
            attention_mask: Optional mask
            
        Returns:
            loss: Scalar tensor
        """
        if labels is None:
            # Shift for next-token prediction
            labels = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            if attention_mask is not None:
                attention_mask = attention_mask[:, :-1].contiguous()
        
        logits, _ = self.forward(input_ids, attention_mask)
        
        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id if self._tokenizer else -100
        )
        
        return loss
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        stop_tokens: List[str] = None
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling
            stop_tokens: Stop generation on these tokens
            
        Returns:
            Generated text
        """
        self.eval()
        
        temperature = temperature or self.config.temperature
        top_k = top_k or self.config.top_k
        top_p = top_p or self.config.top_p
        
        # Encode prompt
        input_ids = self.encode_text(prompt)
        generated = input_ids.squeeze(0).tolist()
        
        # Stop token IDs
        stop_ids = set()
        if stop_tokens:
            for tok in stop_tokens:
                stop_ids.update(self.tokenizer.encode(tok))
        stop_ids.add(self.tokenizer.eos_token_id)
        
        # Generate tokens
        for _ in range(max_tokens):
            # Get context (limit to max_context)
            context = torch.tensor(generated[-self.config.max_context:]).unsqueeze(0)
            context = context.to(self.config.device)
            
            # Encode and process using causal forward (same as training)
            patterns = self.vocab.encode(context)
            hidden = self.generator.forward_causal(patterns)  # (batch, seq, dim)
            
            # Take the last position's output (predicts the next token)
            last_hidden = hidden[:, -1, :]  # (batch, dim)
            
            # Sample next token
            next_token = self.vocab.sample(
                last_hidden.squeeze(0),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            token_id = next_token.item()
            generated.append(token_id)
            
            # Check stop condition
            if token_id in stop_ids:
                break
        
        # Decode
        output_text = self.decode_tokens(torch.tensor(generated))
        
        return output_text
    
    def get_parameter_count(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path: Path):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'config': self.config,
            'state_dict': self.state_dict()
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load(cls, path: Path, device: str = None) -> 'GAIA1':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        
        config = checkpoint['config']
        if device:
            config.device = device
        
        model = cls(config)
        model.load_state_dict(checkpoint['state_dict'])
        
        return model
    
    def __repr__(self) -> str:
        params = self.get_parameter_count()
        return (f"GAIA1(vocab={self.config.vocab_size}, "
                f"dim={self.config.field_dim}, "
                f"layers={self.config.n_layers}, "
                f"params={params:,})")
