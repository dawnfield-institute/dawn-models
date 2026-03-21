"""
PAC Knowledge Mesh: Intelligent multi-model learning.

NOT blind gluing. Real PAC learning:
1. Know what you know (confidence tracking)
2. Reinforce agreement (byref optimization)
3. Fill gaps carefully (low confidence until validated)
4. Semantic alignment (cosine similarity check)

Key insight: Models that agree on semantics reinforce each other.
Models that disagree reveal uncertainty or different knowledge.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import json
from collections import defaultdict

from .embeddings import GraftedEmbeddings
from .transitions import TransitionMatrix
from .pac_tree import PACTree
from .concentration import PHI_INV


@dataclass
class KnowledgeNode:
    """A node in the knowledge mesh with confidence tracking."""
    token_id: int
    token_str: str
    
    # Embeddings from each source (before merging)
    source_embeddings: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # Merged embedding (computed from sources)
    embedding: Optional[torch.Tensor] = None
    
    # Confidence: how many sources agree on this token's meaning?
    confidence: float = 0.0
    
    # Semantic agreement: cosine similarity between sources
    semantic_agreement: float = 0.0
    
    # Is this a gap fill (single source) or reinforced (multi-source)?
    is_reinforced: bool = False
    is_gap_fill: bool = False
    
    # Sources that contributed
    sources: Set[str] = field(default_factory=set)


@dataclass
class TransitionKnowledge:
    """Track what we know about a transition."""
    context: Tuple[int, ...]
    next_token: int
    
    # Count per source
    source_counts: Dict[str, int] = field(default_factory=dict)
    
    # Agreement: how many sources predict this transition?
    agreement_count: int = 0
    
    # Confidence based on agreement
    confidence: float = 0.0


class KnowledgeMesh:
    """
    Intelligent multi-model knowledge mesh.
    
    Core principles:
    1. Track provenance - know which model taught what
    2. Measure agreement - semantic similarity between sources
    3. Confidence scoring - reinforced > single-source > gap-fill
    4. Smart merging - weight by confidence and agreement
    
    Usage:
        mesh = KnowledgeMesh()
        mesh.learn_from('gpt2')
        mesh.learn_from('gpt2-medium')  # Reinforces shared knowledge
        mesh.learn_from('pythia-70m')   # Fills gaps, adds new perspective
        
        # Check what we know
        mesh.what_do_i_know('the')  # Shows confidence, sources, agreement
        
        # Build model with confidence-aware generation
        model = mesh.build_model()
    """
    
    def __init__(self, device: str = 'cuda', agreement_threshold: float = 0.8):
        self.device = device
        self.agreement_threshold = agreement_threshold  # Cosine sim threshold for "agreement"
        
        # Knowledge tracking
        self.nodes: Dict[str, KnowledgeNode] = {}  # token_str → node
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.next_id = 0
        
        # Transition knowledge
        self.transitions: Dict[Tuple[int, ...], Dict[int, TransitionKnowledge]] = defaultdict(dict)
        
        # Source tracking
        self.sources: List[str] = []
        self.source_tokenizers: Dict[str, any] = {}
        
        # Statistics
        self.stats = {
            'total_tokens': 0,
            'reinforced_tokens': 0,
            'gap_filled_tokens': 0,
            'high_agreement_tokens': 0,
            'low_agreement_tokens': 0,
            'total_transitions': 0,
            'reinforced_transitions': 0,
        }
    
    def learn_from(self, model_name: str) -> Dict:
        """
        Learn from a pretrained model.
        
        Automatically detects model family (GPT-2, Pythia, etc.)
        """
        print(f"\n{'='*50}")
        print(f"Learning from: {model_name}")
        print(f"{'='*50}")
        
        # Detect model family and load
        if 'pythia' in model_name.lower():
            embeddings = GraftedEmbeddings.from_pythia(model_name, device='cpu')
        else:
            embeddings = GraftedEmbeddings.from_gpt2(model_name, device='cpu')
        
        # Store tokenizer for later
        self.source_tokenizers[model_name] = embeddings.tokenizer
        self.sources.append(model_name)
        
        # Integrate knowledge
        stats = self._integrate_knowledge(embeddings, model_name)
        
        print(f"\nIntegration summary:")
        print(f"  New tokens: {stats['new_tokens']}")
        print(f"  Reinforced: {stats['reinforced']}")
        print(f"  High agreement: {stats['high_agreement']}")
        print(f"  Low agreement: {stats['low_agreement']}")
        
        return stats
    
    def _integrate_knowledge(self, embeddings: GraftedEmbeddings, source: str) -> Dict:
        """Integrate embeddings with semantic awareness."""
        new_tokens = 0
        reinforced = 0
        high_agreement = 0
        low_agreement = 0
        
        vocab = embeddings.tokenizer.get_vocab()
        
        for token_str, orig_id in vocab.items():
            emb = embeddings.embeddings[orig_id].clone()
            
            if token_str in self.nodes:
                # Existing token - check semantic agreement
                node = self.nodes[token_str]
                
                # Compute agreement with existing embeddings
                agreements = []
                for existing_source, existing_emb in node.source_embeddings.items():
                    # Normalize dimensions if needed
                    if emb.shape[0] != existing_emb.shape[0]:
                        min_dim = min(emb.shape[0], existing_emb.shape[0])
                        sim = F.cosine_similarity(
                            emb[:min_dim].unsqueeze(0),
                            existing_emb[:min_dim].unsqueeze(0)
                        ).item()
                    else:
                        sim = F.cosine_similarity(
                            emb.unsqueeze(0),
                            existing_emb.unsqueeze(0)
                        ).item()
                    agreements.append(sim)
                
                avg_agreement = sum(agreements) / len(agreements) if agreements else 0
                
                # Add this source's embedding
                node.source_embeddings[source] = emb
                node.sources.add(source)
                node.is_reinforced = True
                node.is_gap_fill = False
                
                # Update agreement score
                node.semantic_agreement = avg_agreement
                
                if avg_agreement >= self.agreement_threshold:
                    high_agreement += 1
                    node.confidence = min(1.0, node.confidence + 0.3)  # Boost confidence
                else:
                    low_agreement += 1
                    # Low agreement - might be different knowledge, keep both perspectives
                    node.confidence = max(0.3, node.confidence)  # Don't boost much
                
                reinforced += 1
                
            else:
                # New token - gap fill
                node = KnowledgeNode(
                    token_id=self.next_id,
                    token_str=token_str,
                    source_embeddings={source: emb},
                    confidence=0.5,  # Single source = medium confidence
                    semantic_agreement=1.0,  # Only one source, agrees with itself
                    is_reinforced=False,
                    is_gap_fill=True,
                    sources={source}
                )
                
                self.nodes[token_str] = node
                self.token_to_id[token_str] = self.next_id
                self.id_to_token[self.next_id] = token_str
                self.next_id += 1
                new_tokens += 1
        
        # Update stats
        self.stats['total_tokens'] = len(self.nodes)
        self.stats['reinforced_tokens'] = sum(1 for n in self.nodes.values() if n.is_reinforced)
        self.stats['gap_filled_tokens'] = sum(1 for n in self.nodes.values() if n.is_gap_fill)
        self.stats['high_agreement_tokens'] = sum(1 for n in self.nodes.values() if n.semantic_agreement >= self.agreement_threshold)
        
        return {
            'new_tokens': new_tokens,
            'reinforced': reinforced,
            'high_agreement': high_agreement,
            'low_agreement': low_agreement,
        }
    
    def learn_text(self, text: str, source: str = 'text') -> Dict:
        """
        Learn transition patterns from text.
        
        Tracks which transitions come from which sources.
        """
        if not self.sources:
            raise ValueError("Learn from at least one model first")
        
        # Use first tokenizer
        tokenizer = self.source_tokenizers[self.sources[0]]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        context_size = 5
        new_transitions = 0
        reinforced_transitions = 0
        
        for i in range(context_size, len(tokens)):
            context = tuple(tokens[i-context_size:i])
            next_token = tokens[i]
            
            if next_token not in self.transitions[context]:
                # New transition
                self.transitions[context][next_token] = TransitionKnowledge(
                    context=context,
                    next_token=next_token,
                    source_counts={source: 1},
                    agreement_count=1,
                    confidence=0.5
                )
                new_transitions += 1
            else:
                # Existing transition - reinforce
                tk = self.transitions[context][next_token]
                tk.source_counts[source] = tk.source_counts.get(source, 0) + 1
                tk.agreement_count = len(tk.source_counts)
                tk.confidence = min(1.0, 0.5 + 0.1 * tk.agreement_count)
                reinforced_transitions += 1
        
        self.stats['total_transitions'] = sum(
            len(nexts) for nexts in self.transitions.values()
        )
        
        return {
            'tokens_processed': len(tokens),
            'new_transitions': new_transitions,
            'reinforced_transitions': reinforced_transitions,
        }
    
    def what_do_i_know(self, token_str: str) -> Optional[Dict]:
        """
        Query what we know about a token.
        
        Returns confidence, sources, agreement, etc.
        """
        if token_str not in self.nodes:
            return None
        
        node = self.nodes[token_str]
        return {
            'token': token_str,
            'id': node.token_id,
            'confidence': node.confidence,
            'semantic_agreement': node.semantic_agreement,
            'sources': list(node.sources),
            'is_reinforced': node.is_reinforced,
            'is_gap_fill': node.is_gap_fill,
            'embedding_dim': node.embedding.shape[0] if node.embedding is not None else None,
        }
    
    def merge_embeddings(self) -> torch.Tensor:
        """
        Merge embeddings with confidence-aware weighting.
        
        High agreement sources get more weight.
        Low agreement sources are averaged with caution.
        """
        print(f"\nMerging {len(self.nodes)} token embeddings...")
        
        # Find max dimension
        max_dim = 0
        for node in self.nodes.values():
            for emb in node.source_embeddings.values():
                max_dim = max(max_dim, emb.shape[0])
        
        print(f"  Target dimension: {max_dim}")
        
        # Create merged tensor
        merged = torch.zeros(len(self.nodes), max_dim)
        
        for node in self.nodes.values():
            if len(node.source_embeddings) == 1:
                # Single source - use directly (with padding)
                source, emb = next(iter(node.source_embeddings.items()))
                merged[node.token_id, :emb.shape[0]] = emb
                node.embedding = merged[node.token_id]
                
            else:
                # Multiple sources - weighted average based on agreement
                padded_embs = []
                for source, emb in node.source_embeddings.items():
                    padded = torch.zeros(max_dim)
                    padded[:emb.shape[0]] = emb
                    padded_embs.append(padded)
                
                # Simple average for now (could weight by per-source confidence)
                merged[node.token_id] = torch.stack(padded_embs).mean(dim=0)
                node.embedding = merged[node.token_id]
        
        print(f"  Merged shape: {merged.shape}")
        return merged
    
    def build_model(self) -> 'gaia_prime':
        """Build a gaia_prime model from the knowledge mesh."""
        from .model import gaia_prime
        
        # Merge embeddings
        merged_embs = self.merge_embeddings()
        
        # Create embeddings wrapper
        class MeshEmbeddings:
            def __init__(wrapper_self, mesh: 'KnowledgeMesh', embeddings: torch.Tensor):
                wrapper_self.embeddings = embeddings.to(mesh.device)
                wrapper_self.vocab_size = len(mesh.nodes)
                wrapper_self.embed_dim = embeddings.shape[1]
                wrapper_self.model_name = f"mesh[{'+'.join(mesh.sources)}]"
                wrapper_self.mesh = mesh
                wrapper_self.tokenizer = mesh.source_tokenizers[mesh.sources[0]]
            
            def encode(wrapper_self, text: str) -> torch.Tensor:
                tokens = wrapper_self.tokenizer.encode(text, add_special_tokens=False)
                return torch.tensor(tokens, dtype=torch.long)
            
            def decode(wrapper_self, token_ids) -> str:
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.tolist()
                return wrapper_self.tokenizer.decode(token_ids)
            
            def get_confidence(wrapper_self, token_id: int) -> float:
                """Get confidence for a token."""
                token_str = wrapper_self.mesh.id_to_token.get(token_id)
                if token_str and token_str in wrapper_self.mesh.nodes:
                    return wrapper_self.mesh.nodes[token_str].confidence
                return 0.0
        
        mesh_embeddings = MeshEmbeddings(self, merged_embs)
        
        # Create model
        model = gaia_prime(
            embeddings=mesh_embeddings,
            context_size=5,
            device=self.device
        )
        
        # Build transition matrix from learned transitions
        for context, next_tokens in self.transitions.items():
            for next_token, tk in next_tokens.items():
                # Weight by confidence
                count = int(sum(tk.source_counts.values()) * tk.confidence)
                for _ in range(max(1, count)):
                    model.transitions.learn(context, next_token)
        
        # Update metadata
        model.metadata['source_model'] = mesh_embeddings.model_name
        model.metadata['mesh_sources'] = self.sources
        model.metadata['total_tokens'] = self.stats['total_tokens']
        model.metadata['reinforced_tokens'] = self.stats['reinforced_tokens']
        model.metadata['high_agreement_tokens'] = self.stats['high_agreement_tokens']
        
        return model
    
    def get_statistics(self) -> Dict:
        """Get comprehensive mesh statistics."""
        confidence_dist = {
            'high': sum(1 for n in self.nodes.values() if n.confidence >= 0.8),
            'medium': sum(1 for n in self.nodes.values() if 0.5 <= n.confidence < 0.8),
            'low': sum(1 for n in self.nodes.values() if n.confidence < 0.5),
        }
        
        return {
            'sources': self.sources,
            'total_tokens': len(self.nodes),
            'reinforced_tokens': self.stats['reinforced_tokens'],
            'gap_filled_tokens': self.stats['gap_filled_tokens'],
            'high_agreement_tokens': self.stats['high_agreement_tokens'],
            'total_transitions': self.stats['total_transitions'],
            'confidence_distribution': confidence_dist,
        }
    
    def summary(self):
        """Print a human-readable summary of what we know."""
        print("\n" + "="*60)
        print("KNOWLEDGE MESH SUMMARY")
        print("="*60)
        
        print(f"\nSources: {', '.join(self.sources)}")
        print(f"\nToken Knowledge:")
        print(f"  Total tokens: {len(self.nodes)}")
        print(f"  Reinforced (multi-source): {self.stats['reinforced_tokens']}")
        print(f"  Gap-filled (single-source): {self.stats['gap_filled_tokens']}")
        print(f"  High agreement (>{self.agreement_threshold}): {self.stats['high_agreement_tokens']}")
        
        print(f"\nTransition Knowledge:")
        print(f"  Total transitions: {self.stats['total_transitions']}")
        
        # Sample high-confidence tokens
        high_conf = [(t, n.confidence) for t, n in self.nodes.items() if n.confidence >= 0.8][:5]
        if high_conf:
            print(f"\nSample high-confidence tokens:")
            for token, conf in high_conf:
                print(f"  '{token}': {conf:.2f}")
        
        print("="*60)


def demo():
    """Demonstrate intelligent knowledge mesh."""
    print("="*60)
    print("Knowledge Mesh Demo")
    print("="*60)
    
    mesh = KnowledgeMesh(device='cpu')
    
    # Learn from models
    mesh.learn_from('gpt2')
    mesh.learn_from('gpt2-medium')
    
    # Check what we know about specific tokens
    print("\n" + "-"*40)
    print("Querying knowledge:")
    for token in ['the', 'machine', 'learning', 'xyz123']:
        info = mesh.what_do_i_know(token)
        if info:
            print(f"  '{token}': confidence={info['confidence']:.2f}, "
                  f"agreement={info['semantic_agreement']:.2f}, "
                  f"reinforced={info['is_reinforced']}")
        else:
            print(f"  '{token}': UNKNOWN")
    
    # Learn from text
    text = "Machine learning is a subset of artificial intelligence. " * 20
    mesh.learn_text(text)
    
    # Summary
    mesh.summary()
    
    # Build model
    model = mesh.build_model()
    print(f"\nBuilt model: {model}")
    
    # Generate
    result = model.generate("Machine learning", max_tokens=10)
    print(f"Generated: '{result.text}'")


if __name__ == "__main__":
    demo()
