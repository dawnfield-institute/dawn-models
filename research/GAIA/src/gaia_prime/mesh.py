"""
Multi-Model PAC Mesh: Learn from multiple pretrained models.

Key insight: Different models have learned different patterns.
By combining them, we fill gaps and reinforce common knowledge.

Where models agree → strong byref nodes (reinforced)
Where models differ → mesh fills gaps (coverage)

This is LEARNING, not stealing. We're building knowledge
from multiple teachers.
"""

import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

from .pac_tree import PACTree, PACNode
from .embeddings import GraftedEmbeddings
from .transitions import TransitionMatrix
from .concentration import ConcentrationMonitor, PHI_INV


@dataclass
class ModelSource:
    """Metadata about a source model we learned from."""
    name: str
    vocab_size: int
    embed_dim: int
    contribution: float = 0.0  # How much this model contributed


@dataclass
class MeshStatistics:
    """Statistics about the PAC mesh."""
    total_nodes: int = 0
    reinforced_nodes: int = 0  # Nodes with multiple model agreement
    gap_filled_nodes: int = 0  # Nodes from single model (gap filling)
    coverage: float = 0.0  # Fraction of possible transitions covered
    reinforcement_ratio: float = 0.0  # How much agreement across models


class PACMesh:
    """
    Multi-model PAC mesh that learns from multiple sources.
    
    The mesh combines knowledge from multiple pretrained models:
    - GPT-2 family (gpt2, gpt2-medium, gpt2-large)
    - Pythia family (70m, 160m, 410m)
    - Any other transformer with accessible embeddings
    
    Where models agree, nodes are REINFORCED (byref optimization).
    Where they differ, gaps are FILLED (coverage expansion).
    
    Usage:
        mesh = PACMesh()
        mesh.learn_from_gpt2('gpt2')
        mesh.learn_from_gpt2('gpt2-medium')
        mesh.learn_from_pythia('EleutherAI/pythia-70m')
        
        # Now mesh has combined knowledge
        model = mesh.build_model()
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        # Sources we've learned from
        self.sources: List[ModelSource] = []
        
        # Unified vocabulary (union of all model vocabs)
        self.unified_vocab: Dict[str, int] = {}
        self.unified_id_to_token: Dict[int, str] = {}
        self.next_unified_id = 0
        
        # Embedding accumulator: token_id → list of embeddings from different models
        self.embedding_sources: Dict[int, List[Tuple[str, torch.Tensor]]] = {}
        
        # Transition accumulator: context → {next_token → count_by_model}
        self.transition_sources: Dict[Tuple[int, ...], Dict[int, Dict[str, int]]] = {}
        
        # Final merged structures
        self.merged_embeddings: Optional[torch.Tensor] = None
        self.merged_tree: Optional[PACTree] = None
        self.merged_transitions: Optional[TransitionMatrix] = None
        
        # Statistics
        self.stats = MeshStatistics()
    
    def learn_from_gpt2(self, model_name: str = 'gpt2') -> Dict:
        """
        Learn embeddings and structure from a GPT-2 variant.
        
        Args:
            model_name: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
        
        Returns:
            Learning statistics
        """
        print(f"Learning from {model_name}...")
        
        # Extract embeddings
        embeddings = GraftedEmbeddings.from_gpt2(model_name, device='cpu')
        
        # Add to mesh
        stats = self._integrate_embeddings(embeddings, model_name)
        
        # Record source
        self.sources.append(ModelSource(
            name=model_name,
            vocab_size=embeddings.vocab_size,
            embed_dim=embeddings.embed_dim,
            contribution=stats['tokens_integrated']
        ))
        
        print(f"  Integrated {stats['tokens_integrated']} tokens, "
              f"{stats['new_tokens']} new, {stats['reinforced']} reinforced")
        
        return stats
    
    def learn_from_pythia(self, model_name: str = 'EleutherAI/pythia-70m') -> Dict:
        """
        Learn embeddings and structure from a Pythia variant.
        
        Args:
            model_name: Full model name like 'EleutherAI/pythia-70m'
        
        Returns:
            Learning statistics
        """
        print(f"Learning from {model_name}...")
        
        # Extract embeddings
        embeddings = GraftedEmbeddings.from_pythia(model_name, device='cpu')
        
        # Add to mesh
        stats = self._integrate_embeddings(embeddings, model_name)
        
        # Record source
        self.sources.append(ModelSource(
            name=model_name,
            vocab_size=embeddings.vocab_size,
            embed_dim=embeddings.embed_dim,
            contribution=stats['tokens_integrated']
        ))
        
        print(f"  Integrated {stats['tokens_integrated']} tokens, "
              f"{stats['new_tokens']} new, {stats['reinforced']} reinforced")
        
        return stats
    
    def _integrate_embeddings(
        self, 
        embeddings: GraftedEmbeddings,
        source_name: str
    ) -> Dict:
        """
        Integrate embeddings from a model into the mesh.
        
        For each token:
        - If new: add to unified vocab and store embedding
        - If existing: add embedding to sources (for later averaging/reinforcement)
        """
        new_tokens = 0
        reinforced = 0
        
        # Iterate through vocabulary
        for token_str, token_id in embeddings.tokenizer.get_vocab().items():
            # Get embedding for this token
            emb = embeddings.embeddings[token_id].clone()
            
            # Check if token already in unified vocab
            if token_str in self.unified_vocab:
                # Token exists - reinforce
                unified_id = self.unified_vocab[token_str]
                self.embedding_sources[unified_id].append((source_name, emb))
                reinforced += 1
            else:
                # New token - add to vocab
                unified_id = self.next_unified_id
                self.unified_vocab[token_str] = unified_id
                self.unified_id_to_token[unified_id] = token_str
                self.embedding_sources[unified_id] = [(source_name, emb)]
                self.next_unified_id += 1
                new_tokens += 1
        
        return {
            'tokens_integrated': embeddings.vocab_size,
            'new_tokens': new_tokens,
            'reinforced': reinforced,
            'source': source_name,
        }
    
    def learn_transitions_from_text(
        self, 
        text: str,
        tokenizer_source: str = None
    ) -> Dict:
        """
        Learn transition patterns from text.
        
        Uses the first available tokenizer to encode text,
        then learns transitions in the unified space.
        """
        if not self.sources:
            raise ValueError("Must learn from at least one model first")
        
        # Use specified or first tokenizer
        source = tokenizer_source or self.sources[0].name
        
        # Get tokenizer from embeddings
        embeddings = GraftedEmbeddings.from_gpt2(source, device='cpu')
        tokens = embeddings.encode(text)
        
        # Learn transitions
        context_size = 5
        transitions_learned = 0
        
        for i in range(context_size, len(tokens)):
            context = tuple(tokens[i-context_size:i].tolist())
            next_token = tokens[i].item()
            
            if context not in self.transition_sources:
                self.transition_sources[context] = {}
            
            if next_token not in self.transition_sources[context]:
                self.transition_sources[context][next_token] = {}
            
            model_counts = self.transition_sources[context][next_token]
            model_counts[source] = model_counts.get(source, 0) + 1
            transitions_learned += 1
        
        return {
            'transitions_learned': transitions_learned,
            'unique_contexts': len(self.transition_sources),
        }
    
    def merge(self) -> 'PACMesh':
        """
        Merge all learned embeddings into unified structures.
        
        Embedding merge strategy:
        - Single source: use directly
        - Multiple sources: average (reinforced node)
        
        This creates the final PAC tree with byref optimization.
        """
        print(f"\nMerging {len(self.sources)} model sources...")
        
        if not self.sources:
            raise ValueError("No models learned from yet")
        
        # Determine embedding dimension (use max for compatibility)
        embed_dims = [s.embed_dim for s in self.sources]
        target_dim = max(embed_dims)
        print(f"  Target embedding dimension: {target_dim}")
        
        # Create merged embedding tensor
        vocab_size = len(self.unified_vocab)
        self.merged_embeddings = torch.zeros(vocab_size, target_dim)
        
        reinforced_count = 0
        gap_filled_count = 0
        
        for unified_id, sources in self.embedding_sources.items():
            if len(sources) == 1:
                # Single source - gap filling
                source_name, emb = sources[0]
                # Pad if needed
                if emb.shape[0] < target_dim:
                    padded = torch.zeros(target_dim)
                    padded[:emb.shape[0]] = emb
                    emb = padded
                self.merged_embeddings[unified_id] = emb
                gap_filled_count += 1
            else:
                # Multiple sources - reinforcement via averaging
                padded_embs = []
                for source_name, emb in sources:
                    if emb.shape[0] < target_dim:
                        padded = torch.zeros(target_dim)
                        padded[:emb.shape[0]] = emb
                        emb = padded
                    padded_embs.append(emb)
                
                # Average embeddings (reinforcement)
                self.merged_embeddings[unified_id] = torch.stack(padded_embs).mean(dim=0)
                reinforced_count += 1
        
        # Update statistics
        self.stats.total_nodes = vocab_size
        self.stats.reinforced_nodes = reinforced_count
        self.stats.gap_filled_nodes = gap_filled_count
        self.stats.reinforcement_ratio = reinforced_count / vocab_size if vocab_size > 0 else 0
        
        print(f"  Unified vocabulary: {vocab_size} tokens")
        print(f"  Reinforced (multi-model): {reinforced_count}")
        print(f"  Gap-filled (single-model): {gap_filled_count}")
        print(f"  Reinforcement ratio: {self.stats.reinforcement_ratio:.1%}")
        
        # Create PAC tree with merged embeddings
        self.merged_tree = PACTree(embed_dim=target_dim, device=self.device)
        self.merged_tree.graft_embeddings(self.merged_embeddings.to(self.device))
        
        # Create transition matrix
        self.merged_transitions = TransitionMatrix(
            vocab_size=vocab_size,
            max_context_len=5,
            device=self.device
        )
        
        # Transfer learned transitions
        for context, next_tokens in self.transition_sources.items():
            for next_token, model_counts in next_tokens.items():
                total_count = sum(model_counts.values())
                for _ in range(total_count):
                    self.merged_transitions.learn(context, next_token)
        
        return self
    
    def build_model(self) -> 'gaia_prime':
        """
        Build a gaia_prime model from the merged mesh.
        
        Returns:
            Ready-to-use gaia_prime instance
        """
        from .model import gaia_prime
        
        if self.merged_embeddings is None:
            self.merge()
        
        # Create custom embeddings wrapper
        class MeshEmbeddings:
            def __init__(mesh_self, mesh: PACMesh):
                mesh_self.embeddings = mesh.merged_embeddings.to(mesh.device)
                mesh_self.vocab_size = len(mesh.unified_vocab)
                mesh_self.embed_dim = mesh.merged_embeddings.shape[1]
                mesh_self.model_name = f"mesh[{'+'.join(s.name for s in mesh.sources)}]"
                mesh_self.unified_vocab = mesh.unified_vocab
                mesh_self.id_to_token = mesh.unified_id_to_token
                
                # Use first source's tokenizer for encoding
                # (Could be improved with unified tokenizer)
                first_source = mesh.sources[0].name
                if 'pythia' in first_source.lower():
                    from transformers import AutoTokenizer
                    mesh_self.tokenizer = AutoTokenizer.from_pretrained(first_source)
                else:
                    from transformers import GPT2Tokenizer
                    mesh_self.tokenizer = GPT2Tokenizer.from_pretrained(first_source)
            
            def encode(mesh_self, text: str) -> torch.Tensor:
                token_ids = mesh_self.tokenizer.encode(text, add_special_tokens=False)
                return torch.tensor(token_ids, dtype=torch.long)
            
            def decode(mesh_self, token_ids) -> str:
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.tolist()
                return mesh_self.tokenizer.decode(token_ids)
        
        # Create model with mesh embeddings
        mesh_embeddings = MeshEmbeddings(self)
        
        model = gaia_prime(
            embeddings=mesh_embeddings,
            context_size=5,
            device=self.device
        )
        
        # Replace tree and transitions with merged versions
        model.tree = self.merged_tree
        model.transitions = self.merged_transitions
        
        # Update metadata
        model.metadata['source_model'] = mesh_embeddings.model_name
        model.metadata['mesh_sources'] = [s.name for s in self.sources]
        model.metadata['reinforced_nodes'] = self.stats.reinforced_nodes
        model.metadata['gap_filled_nodes'] = self.stats.gap_filled_nodes
        
        return model
    
    def get_statistics(self) -> Dict:
        """Get mesh statistics."""
        return {
            'sources': [
                {
                    'name': s.name,
                    'vocab_size': s.vocab_size,
                    'embed_dim': s.embed_dim,
                    'contribution': s.contribution,
                }
                for s in self.sources
            ],
            'unified_vocab_size': len(self.unified_vocab),
            'total_nodes': self.stats.total_nodes,
            'reinforced_nodes': self.stats.reinforced_nodes,
            'gap_filled_nodes': self.stats.gap_filled_nodes,
            'reinforcement_ratio': self.stats.reinforcement_ratio,
            'transitions_learned': len(self.transition_sources),
        }


def demo_multi_model():
    """Demonstrate multi-model learning."""
    print("=" * 60)
    print("PAC Mesh: Multi-Model Learning Demo")
    print("=" * 60)
    
    # Create mesh
    mesh = PACMesh(device='cpu')
    
    # Learn from multiple models
    mesh.learn_from_gpt2('gpt2')
    mesh.learn_from_gpt2('gpt2-medium')
    
    # Merge
    mesh.merge()
    
    # Build model
    model = mesh.build_model()
    print(f"\nBuilt model: {model}")
    
    # Learn from text
    text = "The quick brown fox jumps over the lazy dog. " * 10
    model.learn(text)
    
    # Generate
    result = model.generate("The quick", max_tokens=10)
    print(f"\nGenerated: {result.text}")
    
    # Statistics
    print(f"\nMesh statistics: {mesh.get_statistics()}")


if __name__ == "__main__":
    demo_multi_model()
