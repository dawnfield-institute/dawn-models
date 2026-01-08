"""
Grafted Embeddings: Learn and graft from pretrained models.

Validated in:
- POC-016: PAC extraction from Pythia
- POC-017: PAC import without training
- POC-020: 100% graft success, dimension-agnostic

We LEARN:
- Token embeddings (semantic space)
- Token↔ID mapping (vocabulary)

We BUILD on top with PAC learning.
"""

import torch
from pathlib import Path
from typing import Tuple, Optional
import json


class GraftedEmbeddings:
    """
    Extract and manage embeddings grafted from pretrained models.
    
    Usage:
        emb = GraftedEmbeddings.from_gpt2()
        emb = GraftedEmbeddings.from_pythia('70m')
        
        # Get embedding
        vec = emb.get_embedding(token_id)
        
        # Encode/decode text
        ids = emb.encode("hello world")
        text = emb.decode([15496, 995])
    """
    
    def __init__(
        self,
        embeddings: torch.Tensor,
        vocab_size: int,
        model_name: str,
        tokenizer,
        device: str = 'cuda'
    ):
        self.embeddings = embeddings.to(device)
        self.vocab_size = vocab_size
        self.embed_dim = embeddings.shape[1]
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = device
        
        print(f"GraftedEmbeddings: {vocab_size} tokens, {self.embed_dim}D from {model_name}")
    
    @classmethod
    def from_gpt2(cls, model_name: str = 'gpt2', device: str = 'cuda') -> 'GraftedEmbeddings':
        """
        Extract embeddings from GPT-2.
        
        Args:
            model_name: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
            device: 'cuda' or 'cpu'
        
        Returns:
            GraftedEmbeddings instance
        """
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        print(f"Loading {model_name}...")
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Extract embedding weights
        embeddings = model.transformer.wte.weight.detach().clone()
        vocab_size = embeddings.shape[0]
        
        # Free the model
        del model
        torch.cuda.empty_cache()
        
        return cls(
            embeddings=embeddings,
            vocab_size=vocab_size,
            model_name=model_name,
            tokenizer=tokenizer,
            device=device
        )
    
    @classmethod
    def from_pythia(cls, model_name: str = 'EleutherAI/pythia-70m', device: str = 'cuda') -> 'GraftedEmbeddings':
        """
        Extract embeddings from Pythia.
        
        Args:
            model_name: Full model name like 'EleutherAI/pythia-70m', 
                       'EleutherAI/pythia-160m', 'EleutherAI/pythia-410m'
            device: 'cuda' or 'cpu'
        
        Returns:
            GraftedEmbeddings instance
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading {model_name}...")
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Extract embedding weights
        embeddings = model.gpt_neox.embed_in.weight.detach().clone()
        vocab_size = embeddings.shape[0]
        
        # Free the model
        del model
        torch.cuda.empty_cache()
        
        return cls(
            embeddings=embeddings,
            vocab_size=vocab_size,
            model_name=model_name,
            tokenizer=tokenizer,
            device=device
        )
    
    def get_embedding(self, token_id: int) -> torch.Tensor:
        """Get embedding for a token."""
        return self.embeddings[token_id]
    
    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for multiple tokens."""
        return self.embeddings[token_ids]
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs tensor."""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(token_ids, dtype=torch.long)
    
    def decode(self, token_ids) -> str:
        """Decode token IDs to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids)
    
    def save(self, path: str):
        """Save embeddings to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        torch.save(self.embeddings.cpu(), path / 'embeddings.pt')
        
        # Save metadata
        metadata = {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'model_name': self.model_name,
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(path / 'tokenizer'))
        
        print(f"Saved embeddings to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cuda') -> 'GraftedEmbeddings':
        """Load embeddings from disk."""
        from transformers import AutoTokenizer
        
        path = Path(path)
        
        # Load embeddings
        embeddings = torch.load(path / 'embeddings.pt')
        
        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(path / 'tokenizer'))
        
        return cls(
            embeddings=embeddings,
            vocab_size=metadata['vocab_size'],
            model_name=metadata['model_name'],
            tokenizer=tokenizer,
            device=device
        )


class SimpleEmbeddings:
    """
    Simple hash-based embeddings for testing.
    
    No external dependencies - just hash tokens to vectors.
    Useful for physics mesh tests without loading models.
    """
    
    def __init__(self, dim: int = 64, device: str = 'cpu'):
        self.dim = dim
        self.device = device
        self._cache = {}
        
        # Seed for reproducibility
        torch.manual_seed(42)
    
    def embed(self, text: str) -> torch.Tensor:
        """Get embedding for text (cached)."""
        if text in self._cache:
            return self._cache[text]
        
        # Hash-based deterministic embedding
        torch.manual_seed(hash(text) % (2**32))
        emb = torch.randn(self.dim, device=self.device)
        emb = emb / (emb.norm() + 1e-9)  # Normalize
        
        self._cache[text] = emb
        return emb
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between texts."""
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return torch.dot(emb1, emb2).item()


if __name__ == "__main__":
    # Test extraction
    print("Testing GPT-2 extraction...")
    gpt2_emb = GraftedEmbeddings.from_gpt2(device='cpu')
    
    # Test encode/decode
    text = "Hello, world!"
    ids = gpt2_emb.encode(text)
    decoded = gpt2_emb.decode(ids)
    print(f"'{text}' → {ids} → '{decoded}'")
    
    # Test embedding lookup
    emb = gpt2_emb.get_embedding(ids[0])
    print(f"Embedding shape: {emb.shape}")
    
    # Test SimpleEmbeddings
    print("\nTesting SimpleEmbeddings...")
    simple = SimpleEmbeddings(dim=64)
    emb1 = simple.embed("hello")
    emb2 = simple.embed("world")
    print(f"hello embedding: {emb1[:5]}...")
    print(f"Similarity(hello, hello): {simple.similarity('hello', 'hello'):.4f}")
    print(f"Similarity(hello, world): {simple.similarity('hello', 'world'):.4f}")
