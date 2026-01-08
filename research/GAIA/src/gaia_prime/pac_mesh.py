"""
PAC Mesh: Multi-dimensional convergent tree structure.

Core insight: Each node is the ROOT of its own PAC subtree.
Trees CONVERGE when different paths lead to the same node.
Convergence is tracked via BYREF - shared references.

Structure:
    "the" ──→ "quick" ──→ "brown" ──→ "fox"
                              ↑
    "a"   ──→ "dark"  ──→ "brown" ──→ "fox"
                              │
                         CONVERGENCE (byref)

When models agree on convergence → reinforced path
When models disagree → multiple branches (uncertainty)

Dimensions:
1. Embedding space (semantic position)
2. Context depth (sequence position)
3. Convergence topology (shared nodes)
4. Confidence (model agreement)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from weakref import ref, WeakValueDictionary
import hashlib


@dataclass
class MeshNode:
    """
    A node in the PAC mesh.
    
    Each node is simultaneously:
    - A token in vocabulary
    - The ROOT of its own subtree (what comes after)
    - A potential CONVERGENCE point (multiple paths lead here)
    """
    
    # Identity
    node_id: int
    token_id: int
    token_str: str
    
    # Embedding (semantic position in space)
    embedding: torch.Tensor
    
    # Convergence tracking: which paths lead TO this node?
    # Key: parent_node_id, Value: count
    incoming_paths: Dict[int, int] = field(default_factory=dict)
    
    # Subtree: which nodes does this lead TO?
    # Key: child_token_id, Value: (child_node reference, count)
    # Using actual references for byref behavior
    children: Dict[int, Tuple['MeshNode', int]] = field(default_factory=dict)
    
    # Confidence from model agreement
    confidence: float = 0.5
    
    # Sources that know about this node
    sources: Set[str] = field(default_factory=set)
    
    # Depth in context (0 = root token, 1+ = context continuation)
    depth: int = 0
    
    @property
    def is_convergence_point(self) -> bool:
        """Does this node have multiple incoming paths?"""
        return len(self.incoming_paths) > 1
    
    @property
    def convergence_factor(self) -> int:
        """How many paths converge here?"""
        return len(self.incoming_paths)
    
    @property
    def total_incoming(self) -> int:
        """Total traversals through this node."""
        return sum(self.incoming_paths.values())
    
    def add_child(self, child: 'MeshNode', count: int = 1):
        """Add or update a child connection (creates byref link)."""
        if child.token_id in self.children:
            existing_child, existing_count = self.children[child.token_id]
            # Verify it's the same node (byref check)
            assert existing_child is child, "Convergence violation: different nodes for same token"
            self.children[child.token_id] = (child, existing_count + count)
        else:
            self.children[child.token_id] = (child, count)
        
        # Update child's incoming paths
        child.incoming_paths[self.node_id] = child.incoming_paths.get(self.node_id, 0) + count


class PACMeshSpace:
    """
    Multi-dimensional PAC mesh in embedding space.
    
    The mesh is organized as:
    - Level 0: Token embeddings (roots of subtrees)
    - Level 1+: Context continuations (branches that may converge)
    
    Key operations:
    - learn_sequence(): Build/reinforce paths through the mesh
    - find_convergences(): Identify shared structure
    - navigate(): Walk the mesh from any starting point
    """
    
    def __init__(self, embed_dim: int, device: str = 'cuda'):
        self.embed_dim = embed_dim
        self.device = device
        
        # Node storage (byref via Python object references)
        self.nodes: Dict[int, MeshNode] = {}  # node_id → MeshNode
        self.next_node_id = 0
        
        # Token → root node mapping (Level 0)
        self.token_roots: Dict[int, MeshNode] = {}  # token_id → root MeshNode
        
        # Convergence index: context_hash → MeshNode
        # This is how we find existing nodes to converge with
        self.context_index: Dict[int, MeshNode] = {}
        
        # Source tracking
        self.sources: List[str] = []
        
        # Statistics
        self.stats = {
            'total_nodes': 0,
            'convergence_points': 0,
            'max_convergence': 0,
            'total_paths': 0,
            'unique_paths': 0,
        }
    
    def _context_hash(self, token_ids: Tuple[int, ...]) -> int:
        """Hash a context tuple for convergence lookup."""
        return hash(token_ids)
    
    def get_or_create_root(
        self, 
        token_id: int, 
        token_str: str,
        embedding: torch.Tensor,
        source: str
    ) -> MeshNode:
        """Get or create a root node for a token."""
        if token_id in self.token_roots:
            node = self.token_roots[token_id]
            node.sources.add(source)
            # Reinforce embedding via averaging
            if source not in node.sources:
                node.embedding = (node.embedding + embedding) / 2
            return node
        
        # Create new root
        node = MeshNode(
            node_id=self.next_node_id,
            token_id=token_id,
            token_str=token_str,
            embedding=embedding.to(self.device),
            depth=0,
            sources={source}
        )
        
        self.nodes[self.next_node_id] = node
        self.token_roots[token_id] = node
        self.next_node_id += 1
        self.stats['total_nodes'] += 1
        
        return node
    
    def get_or_create_context_node(
        self,
        context: Tuple[int, ...],
        final_token_id: int,
        final_token_str: str,
        embedding: torch.Tensor,
        source: str
    ) -> MeshNode:
        """
        Get or create a node for a specific context.
        
        This is where CONVERGENCE happens:
        - If context already exists → return existing node (byref)
        - If new context → create new node
        """
        ctx_hash = self._context_hash(context + (final_token_id,))
        
        if ctx_hash in self.context_index:
            # CONVERGENCE: This context already exists
            node = self.context_index[ctx_hash]
            node.sources.add(source)
            node.confidence = min(1.0, node.confidence + 0.1)  # Boost for agreement
            return node
        
        # New context node
        node = MeshNode(
            node_id=self.next_node_id,
            token_id=final_token_id,
            token_str=final_token_str,
            embedding=embedding.to(self.device),
            depth=len(context),
            sources={source}
        )
        
        self.nodes[self.next_node_id] = node
        self.context_index[ctx_hash] = node
        self.next_node_id += 1
        self.stats['total_nodes'] += 1
        
        return node
    
    def learn_sequence(
        self,
        token_ids: List[int],
        token_strs: List[str],
        embeddings: torch.Tensor,
        source: str,
        context_size: int = 5
    ) -> Dict:
        """
        Learn a sequence, building/reinforcing mesh paths.
        
        For each position, we:
        1. Get/create the node for that context
        2. Link it to the previous node (add_child creates byref)
        3. Track convergence when paths meet
        """
        if source not in self.sources:
            self.sources.append(source)
        
        paths_created = 0
        convergences_found = 0
        
        # Process sequence with sliding context window
        for i in range(len(token_ids)):
            token_id = token_ids[i]
            token_str = token_strs[i]
            embedding = embeddings[i]
            
            if i == 0:
                # First token: create/get root
                current_node = self.get_or_create_root(
                    token_id, token_str, embedding, source
                )
            else:
                # Build context
                start = max(0, i - context_size)
                context = tuple(token_ids[start:i])
                
                # Get or create node (convergence check happens here)
                was_new = self._context_hash(context + (token_id,)) not in self.context_index
                
                new_node = self.get_or_create_context_node(
                    context, token_id, token_str, embedding, source
                )
                
                if not was_new:
                    convergences_found += 1
                
                # Link from previous node (byref connection)
                current_node.add_child(new_node)
                paths_created += 1
                
                current_node = new_node
        
        self.stats['total_paths'] += paths_created
        
        # Update convergence stats
        self.stats['convergence_points'] = sum(
            1 for n in self.nodes.values() if n.is_convergence_point
        )
        self.stats['max_convergence'] = max(
            (n.convergence_factor for n in self.nodes.values()),
            default=0
        )
        
        return {
            'paths_created': paths_created,
            'convergences_found': convergences_found,
            'total_nodes': self.stats['total_nodes'],
        }
    
    def navigate(
        self,
        start_token_id: int,
        max_depth: int = 10
    ) -> List[List[MeshNode]]:
        """
        Navigate from a token through all possible paths.
        
        Returns all paths up to max_depth.
        At convergence points, paths may merge or branch.
        """
        if start_token_id not in self.token_roots:
            return []
        
        root = self.token_roots[start_token_id]
        paths = [[root]]
        
        for depth in range(max_depth):
            new_paths = []
            for path in paths:
                current = path[-1]
                if not current.children:
                    new_paths.append(path)  # Terminal path
                else:
                    for child_token_id, (child_node, count) in current.children.items():
                        new_paths.append(path + [child_node])
            paths = new_paths
            if not paths:
                break
        
        return paths
    
    def get_next_probabilities(
        self,
        context: Tuple[int, ...]
    ) -> Dict[int, float]:
        """
        Get next token probabilities from current context.
        
        Walks the mesh to find the context node, then returns
        child probabilities weighted by count and convergence.
        """
        if not context:
            return {}
        
        # Find the context node
        ctx_hash = self._context_hash(context)
        
        # Try to find matching context
        current_node = None
        
        # Start from root of first token
        if context[0] in self.token_roots:
            current_node = self.token_roots[context[0]]
            
            # Walk through context
            for i, token_id in enumerate(context[1:], 1):
                if token_id in [tid for tid, _ in current_node.children.keys()] if isinstance(list(current_node.children.keys())[0] if current_node.children else 0, tuple) else current_node.children:
                    child_node, _ = current_node.children[token_id]
                    current_node = child_node
                else:
                    current_node = None
                    break
        
        if current_node is None or not current_node.children:
            return {}
        
        # Calculate probabilities
        total = sum(count for _, count in current_node.children.values())
        probs = {}
        
        for token_id, (child_node, count) in current_node.children.items():
            # Weight by count and convergence factor
            weight = count * (1 + 0.1 * child_node.convergence_factor)
            probs[token_id] = weight / total
        
        # Normalize
        total_prob = sum(probs.values())
        return {k: v / total_prob for k, v in probs.items()}
    
    def find_convergences(self, min_factor: int = 2) -> List[MeshNode]:
        """Find all convergence points with at least min_factor incoming paths."""
        return [
            node for node in self.nodes.values()
            if node.convergence_factor >= min_factor
        ]
    
    def get_statistics(self) -> Dict:
        """Get mesh statistics."""
        convergence_dist = defaultdict(int)
        for node in self.nodes.values():
            convergence_dist[node.convergence_factor] += 1
        
        return {
            'total_nodes': self.stats['total_nodes'],
            'root_nodes': len(self.token_roots),
            'context_nodes': len(self.context_index),
            'convergence_points': self.stats['convergence_points'],
            'max_convergence': self.stats['max_convergence'],
            'total_paths': self.stats['total_paths'],
            'convergence_distribution': dict(convergence_dist),
            'sources': self.sources,
        }
    
    def summary(self):
        """Print mesh summary."""
        stats = self.get_statistics()
        print("\n" + "="*60)
        print("PAC MESH SPACE SUMMARY")
        print("="*60)
        print(f"Sources: {', '.join(stats['sources'])}")
        print(f"\nNodes:")
        print(f"  Total: {stats['total_nodes']}")
        print(f"  Root (L0): {stats['root_nodes']}")
        print(f"  Context (L1+): {stats['context_nodes']}")
        print(f"\nConvergence:")
        print(f"  Convergence points: {stats['convergence_points']}")
        print(f"  Max convergence factor: {stats['max_convergence']}")
        print(f"  Distribution: {stats['convergence_distribution']}")
        print(f"\nPaths: {stats['total_paths']}")
        print("="*60)


class MultiModelMesh:
    """
    Learn from multiple models into a unified PAC mesh.
    
    Each model contributes:
    - Embeddings (semantic positions)
    - Implicit structure (from tokenizer vocabulary)
    
    Text learning contributes:
    - Explicit paths (sequences)
    - Convergence patterns (shared continuations)
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.mesh: Optional[PACMeshSpace] = None
        self.tokenizer = None
        self.vocab: Dict[str, int] = {}
        self.id_to_str: Dict[int, str] = {}
        self.embeddings: Optional[torch.Tensor] = None
        
    def learn_from_model(self, model_name: str) -> Dict:
        """Learn embeddings and vocabulary from a pretrained model."""
        from .embeddings import GraftedEmbeddings
        
        print(f"\nLearning from model: {model_name}")
        
        # Load embeddings
        if 'pythia' in model_name.lower():
            graft = GraftedEmbeddings.from_pythia(model_name, device='cpu')
        else:
            graft = GraftedEmbeddings.from_gpt2(model_name, device='cpu')
        
        # Initialize mesh if needed
        if self.mesh is None:
            self.mesh = PACMeshSpace(embed_dim=graft.embed_dim, device=self.device)
            self.tokenizer = graft.tokenizer
            self.embeddings = graft.embeddings.to(self.device)
            self.vocab = graft.tokenizer.get_vocab()
            self.id_to_str = {v: k for k, v in self.vocab.items()}
        
        # Add all tokens as root nodes
        new_roots = 0
        reinforced = 0
        
        for token_str, token_id in graft.tokenizer.get_vocab().items():
            embedding = graft.embeddings[token_id]
            
            was_new = token_id not in self.mesh.token_roots
            self.mesh.get_or_create_root(token_id, token_str, embedding, model_name)
            
            if was_new:
                new_roots += 1
            else:
                reinforced += 1
        
        print(f"  New roots: {new_roots}, Reinforced: {reinforced}")
        
        return {
            'new_roots': new_roots,
            'reinforced': reinforced,
            'total_nodes': self.mesh.stats['total_nodes'],
        }
    
    def learn_from_text(self, text: str, source: str = 'text') -> Dict:
        """Learn sequence patterns from text."""
        if self.mesh is None:
            raise ValueError("Learn from a model first")
        
        # Tokenize
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        token_strs = [self.id_to_str.get(tid, f'[{tid}]') for tid in token_ids]
        embeddings = self.embeddings[token_ids]
        
        # Learn sequence
        stats = self.mesh.learn_sequence(
            token_ids=token_ids,
            token_strs=token_strs,
            embeddings=embeddings,
            source=source
        )
        
        return stats
    
    def build_model(self) -> 'gaia_prime':
        """Build a gaia_prime model from the mesh."""
        from .model import gaia_prime
        from .transitions import TransitionMatrix
        
        if self.mesh is None:
            raise ValueError("Learn from a model first")
        
        # Create embeddings wrapper
        class MeshEmbeddings:
            def __init__(wrapper_self, mesh: 'MultiModelMesh'):
                wrapper_self.embeddings = mesh.embeddings
                wrapper_self.vocab_size = len(mesh.vocab)
                wrapper_self.embed_dim = mesh.embeddings.shape[1]
                wrapper_self.model_name = f"mesh[{'+'.join(mesh.mesh.sources)}]"
                wrapper_self.tokenizer = mesh.tokenizer
                wrapper_self._mesh = mesh
            
            def encode(wrapper_self, text: str) -> torch.Tensor:
                tokens = wrapper_self.tokenizer.encode(text, add_special_tokens=False)
                return torch.tensor(tokens, dtype=torch.long)
            
            def decode(wrapper_self, token_ids) -> str:
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.tolist()
                return wrapper_self.tokenizer.decode(token_ids)
        
        mesh_emb = MeshEmbeddings(self)
        
        # Create model
        model = gaia_prime(
            embeddings=mesh_emb,
            context_size=5,
            device=self.device
        )
        
        # Build transitions from mesh paths
        for node in self.mesh.nodes.values():
            for child_token_id, (child_node, count) in node.children.items():
                # Create context from path to this node
                # For now, use simple token→token transitions
                context = (node.token_id,)
                for _ in range(count):
                    model.transitions.learn(context, child_token_id)
        
        model.metadata['mesh_sources'] = self.mesh.sources
        model.metadata['convergence_points'] = self.mesh.stats['convergence_points']
        
        return model
    
    def summary(self):
        """Print mesh summary."""
        if self.mesh:
            self.mesh.summary()


class ModelKnowledgeExtractor:
    """
    Extract learned knowledge from pretrained models.
    
    Goes beyond just embeddings - we can query the model's
    actual predictions to build our transition matrix.
    
    This is like asking the model: "What have you learned?"
    and capturing that knowledge in our own structure.
    """
    
    def __init__(self, model_name: str = 'gpt2', device: str = 'cpu'):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def load(self):
        """Load the model for querying."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"Loading {self.model_name} for knowledge extraction...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"  Loaded: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
    def extract_transitions(
        self,
        contexts: List[str],
        top_k: int = 10
    ) -> Dict[Tuple[int, ...], Dict[int, float]]:
        """
        Query the model for next-token predictions.
        
        For each context, get the model's predicted distribution
        over next tokens. This captures WHAT THE MODEL LEARNED
        without us having to train anything.
        
        Returns:
            {context_tuple: {next_token_id: probability}}
        """
        import torch
        
        if self.model is None:
            self.load()
        
        transitions = {}
        
        with torch.no_grad():
            for context in contexts:
                # Tokenize
                inputs = self.tokenizer(context, return_tensors='pt').to(self.device)
                
                # Get logits for next token
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]  # Last position
                
                # Get top-k predictions
                probs = torch.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs, k=top_k)
                
                # Store as context tuple → predictions
                context_ids = tuple(inputs.input_ids[0].tolist())
                transitions[context_ids] = {
                    int(idx): float(prob) 
                    for idx, prob in zip(top_indices, top_probs)
                }
        
        return transitions
    
    def extract_continuation_style(
        self,
        prompts: List[str],
        max_tokens: int = 20,
        temperature: float = 0.7
    ) -> List[Tuple[str, str]]:
        """
        Generate continuations to learn the model's "style".
        
        Returns list of (prompt, continuation) pairs that can
        be used to teach our model the style patterns.
        """
        import torch
        
        if self.model is None:
            self.load()
        
        results = []
        
        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
                
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                continuation = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                
                results.append((prompt, continuation))
        
        return results
    
    def teach_mesh(
        self,
        mesh: 'MultiModelMesh',
        sample_texts: List[str],
        top_k: int = 5
    ) -> Dict:
        """
        Extract knowledge from this model and teach it to a mesh.
        
        This is the key function: we query the model's predictions
        and add them as transitions in our mesh structure.
        """
        if self.model is None:
            self.load()
        
        transitions_added = 0
        contexts_queried = 0
        
        # For each text, slide through with context windows
        for text in sample_texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            for i in range(1, len(tokens)):
                # Get context
                start = max(0, i - 5)
                context = tokens[start:i]
                context_text = self.tokenizer.decode(context)
                
                # Query model for predictions
                extracted = self.extract_transitions([context_text], top_k=top_k)
                contexts_queried += 1
                
                # Add to mesh's knowledge
                for ctx_ids, predictions in extracted.items():
                    for next_token, prob in predictions.items():
                        if prob > 0.01:  # Only significant predictions
                            # This teaches the mesh what the model "knows"
                            token_str = self.tokenizer.decode([next_token])
                            if next_token < len(mesh.embeddings):
                                embedding = mesh.embeddings[next_token]
                                mesh.mesh.get_or_create_context_node(
                                    context=ctx_ids,
                                    final_token_id=next_token,
                                    final_token_str=token_str,
                                    embedding=embedding,
                                    source=f"extracted:{self.model_name}"
                                )
                                transitions_added += 1
        
        return {
            'contexts_queried': contexts_queried,
            'transitions_added': transitions_added,
            'source': self.model_name
        }


def demo():
    """Demonstrate the PAC mesh."""
    print("="*60)
    print("PAC Mesh Space Demo")
    print("="*60)
    
    mesh = MultiModelMesh(device='cpu')
    
    # Learn from models
    mesh.learn_from_model('gpt2')
    
    # Learn from text
    text = """
    The quick brown fox jumps over the lazy dog.
    A quick brown fox leaps over the lazy dog.
    The fast brown fox jumps over a lazy dog.
    """ * 10
    
    stats = mesh.learn_from_text(text)
    print(f"\nText learning: {stats}")
    
    # Show mesh
    mesh.summary()
    
    # Find convergences
    convergences = mesh.mesh.find_convergences(min_factor=2)
    print(f"\nConvergence points (≥2 incoming):")
    for node in convergences[:10]:
        print(f"  '{node.token_str}': {node.convergence_factor} paths, "
              f"confidence={node.confidence:.2f}")
    
    # Build and test model
    model = mesh.build_model()
    print(f"\nBuilt model: {model}")
    
    result = model.generate("The quick", max_tokens=10)
    print(f"Generated: '{result.text}'")


if __name__ == "__main__":
    demo()
