"""
GAIA-PAC: Main model class orchestrating all components.

This is GAIA v1.0 - built ONLY from POC-validated mechanisms:
- Grafted embeddings (POC-016, 017, 020): 100% success
- PAC tree with delta storage (POC-007, 020): 12.5x memory
- Transition matrix (POC-021, 022): 65% hit rate
- Concentration monitoring (POC-023): λ≈0.5, +3.6% quality
- φ thresholds (POC-024): Critical transition at depth 4

NO BACKPROP. NO NEURAL NETWORKS. PURE PAC CONSERVATION.
"""

import torch
from typing import Optional, Dict, List, Union
from pathlib import Path
import json
import time

from .pac_tree import PACTree
from .embeddings import GraftedEmbeddings
from .transitions import TransitionMatrix
from .concentration import ConcentrationMonitor, PHI_INV, LAMBDA_HALF
from .generator import PACGenerator, GenerationResult


class GAIA_Prime:
    """
    GAIA Prime: Generative AI via Information Architecture.
    
    A language model built entirely from PAC/SEC principles:
    - Conservation: Information is conserved, not created
    - Hierarchical: Parent-child relationships with delta encoding
    - Statistical: Learn transition probabilities, not weights
    - Quality-gated: Reject low-concentration outputs
    
    Architecture:
    
        ┌─────────────────────────────────────────┐
        │            GAIA Prime Model             │
        ├─────────────────────────────────────────┤
        │                                         │
        │  ┌─────────────────────────────────┐   │
        │  │     GraftedEmbeddings           │   │ ← LEARNED from GPT-2/Pythia
        │  │     (frozen, 50257 × 768)       │   │    (POC-016, 017, 020)
        │  └─────────────────────────────────┘   │
        │              ↓                         │
        │  ┌─────────────────────────────────┐   │
        │  │         PACTree                 │   │ ← Delta-only storage
        │  │   (byref nodes, conservation)   │   │    (POC-007, 020)
        │  └─────────────────────────────────┘   │
        │              ↓                         │
        │  ┌─────────────────────────────────┐   │
        │  │     TransitionMatrix            │   │ ← N-gram counting
        │  │   (GPU-accelerated, sparse)     │   │    (POC-021, 022)
        │  └─────────────────────────────────┘   │
        │              ↓                         │
        │  ┌─────────────────────────────────┐   │
        │  │   ConcentrationMonitor          │   │ ← λ≈0.5 quality gate
        │  │   (reject-resample if low)      │   │    (POC-023)
        │  └─────────────────────────────────┘   │
        │              ↓                         │
        │  ┌─────────────────────────────────┐   │
        │  │       PACGenerator              │   │ ← Text output
        │  │   (greedy/sample/beam)          │   │
        │  └─────────────────────────────────┘   │
        │                                         │
        └─────────────────────────────────────────┘
    
    Usage:
        # Create model
        model = GAIA_Prime.from_gpt2()  # or from_pythia()
        
        # Train on text
        model.learn("This is some training text...")
        
        # Generate
        output = model.generate("Once upon a time")
        print(output.text)
    
    Validated performance:
        - Hit rate: 65% at 100K vocab (POC-022)
        - Quality: +3.6% with reject-resample (POC-023)
        - Memory: 12.5x savings with delta storage (POC-020)
        - Learning: O(1) per token, no gradients (POC-019)
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        embeddings: GraftedEmbeddings,
        context_size: int = 5,
        concentration_threshold: float = PHI_INV,
        device: str = 'cuda',
        use_reject_resample: bool = True,
    ):
        """
        Initialize GAIA-PAC model.
        
        Args:
            embeddings: Pre-extracted embeddings (from GPT-2, Pythia, etc.)
            context_size: N-gram context length (default 5)
            concentration_threshold: Quality threshold (default φ⁻¹ ≈ 0.618)
            device: 'cuda' or 'cpu'
            use_reject_resample: Enable reject-resample (POC-023)
        """
        self.embeddings = embeddings
        self.device = device
        
        # Initialize PAC tree with grafted embeddings
        self.tree = PACTree(embed_dim=embeddings.embed_dim, device=device)
        self.tree.graft_embeddings(embeddings.embeddings)
        
        # Initialize transition matrix
        self.transitions = TransitionMatrix(
            vocab_size=embeddings.vocab_size,
            max_context_len=context_size,
            device=device
        )
        
        # Initialize concentration monitor
        self.monitor = ConcentrationMonitor(
            threshold=concentration_threshold,
            device=device
        )
        
        # Initialize generator
        self.generator = PACGenerator(
            pac_tree=self.tree,
            transition_matrix=self.transitions,
            concentration_monitor=self.monitor,
            device=device,
            use_reject_resample=use_reject_resample
        )
        
        # Model metadata
        self.metadata = {
            'version': self.VERSION,
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source_model': embeddings.model_name,
            'vocab_size': embeddings.vocab_size,
            'embed_dim': embeddings.embed_dim,
            'context_size': context_size,
            'tokens_learned': 0,
        }
    
    @classmethod
    def from_gpt2(
        cls,
        model_name: str = 'gpt2',
        context_size: int = 5,
        device: str = 'cuda',
        **kwargs
    ) -> 'GAIA_Prime':
        """
        Create GAIA-PAC by grafting GPT-2 embeddings.
        
        Args:
            model_name: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
            context_size: N-gram context length
            device: 'cuda' or 'cpu'
        
        Returns:
            GAIA_Prime instance
        """
        embeddings = GraftedEmbeddings.from_gpt2(model_name, device=device)
        return cls(
            embeddings=embeddings,
            context_size=context_size,
            device=device,
            **kwargs
        )
    
    @classmethod
    def from_pythia(
        cls,
        model_name: str = 'EleutherAI/pythia-70m',
        context_size: int = 5,
        device: str = 'cuda',
        **kwargs
    ) -> 'GAIA_Prime':
        """
        Create GAIA-PAC by grafting Pythia embeddings.
        
        Args:
            model_name: Pythia variant (70m, 160m, 410m, etc.)
            context_size: N-gram context length
            device: 'cuda' or 'cpu'
        
        Returns:
            GAIA_Prime instance
        """
        embeddings = GraftedEmbeddings.from_pythia(model_name, device=device)
        return cls(
            embeddings=embeddings,
            context_size=context_size,
            device=device,
            **kwargs
        )
    
    def learn(
        self,
        text: Optional[str] = None,
        tokens: Optional[torch.Tensor] = None,
        batch_size: int = 1024,
    ) -> Dict:
        """
        Learn from text or tokens.
        
        This updates transition probabilities - NO BACKPROP.
        O(1) per token, pure counting.
        
        Args:
            text: Input text to learn from
            tokens: Pre-tokenized input [batch, seq] or [seq]
            batch_size: Batch size for processing
        
        Returns:
            Learning statistics
        """
        if tokens is None:
            if text is None:
                raise ValueError("Provide either text or tokens")
            tokens = self.embeddings.encode(text)
        
        # Ensure 2D
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        
        # Move to device
        tokens = tokens.to(self.device)
        
        # Learn transitions
        stats = self.transitions.learn_batch(tokens)
        
        # Update tree structure (sparse update with context windows)
        context_size = self.transitions.max_context_len
        for seq_idx in range(tokens.shape[0]):
            seq = tokens[seq_idx]
            for i in range(context_size, len(seq)):
                # Build context tuple from previous tokens
                context = tuple(seq[i - context_size:i].tolist())
                next_token = seq[i].item()
                self.tree.learn_transition(context, next_token)
        
        # Update metadata
        self.metadata['tokens_learned'] += tokens.numel()
        
        return {
            'tokens_processed': tokens.numel(),
            'sequences': tokens.shape[0],
            'transition_stats': stats,
        }
    
    def generate(
        self,
        prompt: Optional[str] = None,
        prompt_tokens: Optional[torch.Tensor] = None,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        decode: bool = True,
    ) -> GenerationResult:
        """
        Generate text from prompt.
        
        Args:
            prompt: Text prompt
            prompt_tokens: Pre-tokenized prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            decode: Whether to decode tokens to text
        
        Returns:
            GenerationResult with tokens, text, and stats
        """
        # Get prompt tokens
        if prompt_tokens is None:
            if prompt is None:
                raise ValueError("Provide either prompt or prompt_tokens")
            prompt_tokens = self.embeddings.encode(prompt)
        
        # Ensure 2D
        if prompt_tokens.dim() == 1:
            prompt_tokens = prompt_tokens.unsqueeze(0)
        
        # Move to device
        prompt_tokens = prompt_tokens.to(self.device)
        
        # Update generator params
        self.generator.temperature = temperature
        self.generator.top_k = top_k
        self.generator.top_p = top_p
        
        # Generate
        output_tokens, stats = self.generator.generate(
            prompt_tokens,
            max_tokens=max_tokens
        )
        
        # Decode if requested
        text = ""
        if decode:
            text = self.embeddings.decode(output_tokens.squeeze(0))
        
        return GenerationResult(
            tokens=output_tokens,
            text=text,
            stats=stats
        )
    
    def get_perplexity(
        self,
        text: Optional[str] = None,
        tokens: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute perplexity on text.
        
        Uses transition matrix probabilities.
        Lower is better.
        """
        if tokens is None:
            if text is None:
                raise ValueError("Provide either text or tokens")
            tokens = self.embeddings.encode(text)
        
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        
        tokens = tokens.to(self.device)
        
        # Compute log probabilities
        total_log_prob = 0.0
        count = 0
        
        for seq_idx in range(tokens.shape[0]):
            seq = tokens[seq_idx]
            for i in range(self.transitions.context_size, len(seq)):
                context = seq[i - self.transitions.context_size:i].unsqueeze(0)
                target = seq[i].item()
                
                # Get prediction
                pred_tokens, probs = self.transitions.predict(context)
                
                # Find probability of actual token
                if target in pred_tokens.tolist():
                    idx = pred_tokens.tolist().index(target)
                    prob = probs[idx].item()
                else:
                    prob = 1e-10  # Smoothing
                
                total_log_prob += torch.log(torch.tensor(prob + 1e-10))
                count += 1
        
        # Perplexity = exp(-mean(log_probs))
        if count > 0:
            return torch.exp(-total_log_prob / count).item()
        return float('inf')
    
    def get_statistics(self) -> Dict:
        """Get comprehensive model statistics."""
        return {
            'metadata': self.metadata,
            'tree': self.tree.stats,
            'transitions': {
                'total_transitions': self.transitions.stats.total_transitions,
                'unique_contexts': self.transitions.stats.unique_contexts,
                'unique_transitions': self.transitions.stats.unique_transitions,
            },
            'concentration': self.monitor.get_statistics(),
            'generator': self.generator._get_stats(),
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to directory.
        
        Creates:
            path/
            ├── metadata.json
            ├── embeddings.pt
            ├── tree.pt
            └── transitions.pt
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        with open(path / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save embeddings
        self.embeddings.save(path / 'embeddings.pt')
        
        # Save tree
        self.tree.save(path / 'tree.pt')
        
        # Save transitions
        self.transitions.save(path / 'transitions.pt')
        
        print(f"Saved GAIA-PAC model to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], device: str = 'cuda') -> 'GAIA_Prime':
        """
        Load model from directory.
        
        Args:
            path: Directory containing saved model
            device: Device to load to
        
        Returns:
            GAIA_Prime instance
        """
        path = Path(path)
        
        # Load metadata
        with open(path / 'metadata.json') as f:
            metadata = json.load(f)
        
        # Load embeddings
        embeddings = GraftedEmbeddings.load(path / 'embeddings.pt')
        
        # Create instance
        model = cls(
            embeddings=embeddings,
            context_size=metadata.get('context_size', 5),
            device=device
        )
        
        # Load tree
        model.tree = PACTree.load(path / 'tree.pt', device=device)
        
        # Load transitions
        model.transitions = TransitionMatrix.load(path / 'transitions.pt', device=device)
        
        # Restore metadata
        model.metadata = metadata
        
        print(f"Loaded GAIA-PAC model from {path}")
        return model
    
    def __repr__(self) -> str:
        return (
            f"GAIA_Prime(\n"
            f"  version={self.VERSION},\n"
            f"  source={self.metadata['source_model']},\n"
            f"  vocab_size={self.metadata['vocab_size']},\n"
            f"  tokens_learned={self.metadata['tokens_learned']:,},\n"
            f"  device={self.device}\n"
            f")"
        )


# Convenience aliases
GAIA = GAIA_Prime
GaiaModel = GAIA_Prime


if __name__ == "__main__":
    print("GAIA Prime v2.0.0")
    print("=" * 50)
    print("Usage:")
    print("  from gaia_prime import GAIA_Prime")
    print("  model = GAIA_Prime.from_gpt2()")
    print("  model.learn('Training text...')")
    print("  result = model.generate('Once upon')")
    print("  print(result.text)")
