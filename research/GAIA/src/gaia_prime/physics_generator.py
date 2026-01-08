"""
Physics Generator: Text generation using PhysicsMesh as memory.

Generates text by:
1. Encoding prompt into mesh context
2. Using predict_next() to get candidates
3. Sampling from candidates with physics-aware weighting
4. Optionally learning from generated sequences (CIMM-style)

Key insight: Generation IS memory traversal + attractor influence.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import random
import math

from .pac_mesh import PACMeshSpace, MeshNode
from .physics_mesh import PhysicsMesh, XI, PHI, PHI_INV, LAMBDA_STAR
from .embeddings import SimpleEmbeddings


@dataclass
class GenerationConfig:
    """Configuration for physics-based generation."""
    
    # Length control
    max_tokens: int = 50
    min_tokens: int = 5
    
    # Sampling
    temperature: float = 0.8
    top_k: int = 10
    top_p: float = 0.9  # Nucleus sampling
    
    # Physics influence
    attractor_weight: float = 0.3  # How much attractors influence sampling
    resonance_weight: float = 0.2  # How much resonance affects choices
    entropy_threshold: float = 2.0  # Trigger collapse if entropy exceeds
    
    # Continuous learning
    learn_from_generation: bool = False  # CIMM-style learning
    learning_importance: float = 0.3  # Importance for self-generated patterns
    
    # Stop conditions
    stop_tokens: List[str] = None
    repetition_penalty: float = 1.2
    
    def __post_init__(self):
        if self.stop_tokens is None:
            self.stop_tokens = ['.', '!', '?', '\n']


@dataclass 
class GenerationResult:
    """Result from physics-based generation."""
    
    text: str
    tokens: List[str]
    nodes: List[MeshNode]
    
    # Physics metrics
    avg_confidence: float
    crystallized_count: int
    attractor_influences: int
    entropy_at_end: float
    
    # Path info
    convergence_points: int
    unique_paths: int


class PhysicsGenerator:
    """
    Generate text using PhysicsMesh as intelligent memory.
    
    Generation process:
    1. Encode prompt → find/create nodes in mesh
    2. Build context from prompt nodes
    3. Loop:
       a. predict_next() → get candidates from memory
       b. Apply physics weighting (attractors, resonance)
       c. Sample next token
       d. Add to context
       e. Optionally learn from generated token
       f. Run physics step (entropy/collapse)
    4. Return generated sequence
    
    Usage:
        mesh = PACMeshSpace(embed_dim=64, device='cpu')
        physics = PhysicsMesh(mesh)
        embeddings = SimpleEmbeddings(dim=64)
        
        generator = PhysicsGenerator(physics, embeddings)
        result = generator.generate("The capital of France is")
        print(result.text)
    """
    
    def __init__(self,
                 physics: PhysicsMesh,
                 embeddings: SimpleEmbeddings,
                 config: GenerationConfig = None):
        self.physics = physics
        self.mesh = physics.mesh
        self.embeddings = embeddings
        self.config = config or GenerationConfig()
        
        # Generation state
        self.recent_tokens: List[str] = []  # For repetition penalty
        self.attractor_hits = 0
        
    def encode_prompt(self, prompt: str) -> List[MeshNode]:
        """
        Encode prompt into mesh nodes.
        
        Finds existing nodes or creates new ones.
        Returns context as list of nodes.
        """
        # Simple tokenization (split on spaces)
        tokens = prompt.split()
        
        context_nodes = []
        parent = None
        context = []
        
        for i, token in enumerate(tokens):
            emb = self.embeddings.embed(token)
            token_id = hash(token) % 1000000
            
            if i == 0:
                node = self.mesh.get_or_create_root(
                    token_id, token, emb, "prompt"
                )
            else:
                context.append(parent.token_id)
                node = self.mesh.get_or_create_context_node(
                    tuple(context), token_id, token, emb, "prompt"
                )
                parent.add_child(node)
            
            context_nodes.append(node)
            parent = node
        
        return context_nodes
    
    def sample_next(self,
                    candidates: List[Tuple[MeshNode, float]],
                    context: List[MeshNode]) -> Optional[MeshNode]:
        """
        Sample next token from candidates using physics-aware weighting.
        
        Applies:
        - Temperature scaling
        - Top-k / top-p filtering
        - Repetition penalty
        - Attractor bonus
        - Resonance bonus
        """
        if not candidates:
            return None
        
        # Extract scores
        nodes = [n for n, _ in candidates]
        scores = [s for _, s in candidates]
        
        # Apply repetition penalty
        for i, node in enumerate(nodes):
            if node.token_str in self.recent_tokens:
                count = self.recent_tokens.count(node.token_str)
                scores[i] /= (self.config.repetition_penalty ** count)
        
        # Apply attractor bonus
        for i, node in enumerate(nodes):
            if node.node_id in self.physics.attractors:
                scores[i] *= (1 + self.config.attractor_weight)
                self.attractor_hits += 1
            
            # Resonance bonus
            if node.node_id in self.physics.resonance_memory:
                resonant_count = len(self.physics.resonance_memory[node.node_id])
                scores[i] *= (1 + self.config.resonance_weight * resonant_count * 0.1)
        
        # Temperature scaling
        if self.config.temperature != 1.0:
            scores = [s ** (1.0 / self.config.temperature) for s in scores]
        
        # Normalize
        total = sum(scores)
        if total <= 0:
            return nodes[0] if nodes else None
        
        probs = [s / total for s in scores]
        
        # Top-k filtering
        if self.config.top_k < len(probs):
            # Keep only top-k
            indexed = list(enumerate(probs))
            indexed.sort(key=lambda x: x[1], reverse=True)
            kept_indices = [idx for idx, _ in indexed[:self.config.top_k]]
            
            # Zero out others
            for i in range(len(probs)):
                if i not in kept_indices:
                    probs[i] = 0.0
        
        # Top-p (nucleus) filtering
        if self.config.top_p < 1.0:
            sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
            cumsum = 0.0
            cutoff_idx = len(probs)
            for i, (idx, p) in enumerate(sorted_probs):
                cumsum += p
                if cumsum > self.config.top_p:
                    cutoff_idx = i + 1
                    break
            
            kept = set(idx for idx, _ in sorted_probs[:cutoff_idx])
            for i in range(len(probs)):
                if i not in kept:
                    probs[i] = 0.0
        
        # Re-normalize
        total = sum(probs)
        if total <= 0:
            return nodes[0]
        probs = [p / total for p in probs]
        
        # Sample
        r = random.random()
        cumsum = 0.0
        for node, prob in zip(nodes, probs):
            cumsum += prob
            if r < cumsum:
                return node
        
        return nodes[-1]  # Fallback
    
    def generate(self, 
                 prompt: str,
                 config: GenerationConfig = None) -> GenerationResult:
        """
        Generate text continuation from prompt.
        
        Uses physics mesh as memory for prediction.
        """
        config = config or self.config
        
        # Reset state
        self.recent_tokens = []
        self.attractor_hits = 0
        
        # Encode prompt
        context = self.encode_prompt(prompt)
        generated_tokens = []
        generated_nodes = []
        
        # Run initial physics to establish state
        self.physics.step()
        
        # Generation loop
        for step in range(config.max_tokens):
            # Get predictions from memory
            candidates = self.physics.predict_next(context, top_k=config.top_k * 2)
            
            if not candidates:
                # No predictions - try to find any continuation
                last_node = context[-1] if context else None
                if last_node and last_node.children:
                    candidates = [
                        (child, count / sum(c for _, c in last_node.children.values()))
                        for child, count in last_node.children.values()
                    ]
                else:
                    break  # No way forward
            
            # Sample next token
            next_node = self.sample_next(candidates, context)
            
            if next_node is None:
                break
            
            # Add to generated sequence
            generated_tokens.append(next_node.token_str)
            generated_nodes.append(next_node)
            self.recent_tokens.append(next_node.token_str)
            
            # Update context
            context.append(next_node)
            if len(context) > 10:  # Keep context window reasonable
                context = context[-10:]
            
            # Check stop conditions
            if any(stop in next_node.token_str for stop in config.stop_tokens):
                if len(generated_tokens) >= config.min_tokens:
                    break
            
            # CIMM-style continuous learning
            if config.learn_from_generation:
                self.physics.remember(next_node, config.learning_importance)
            
            # Run physics step
            state = self.physics.step()
            
            # Entropy-triggered collapse
            if state.entropy > config.entropy_threshold:
                self.physics.force_collapse(
                    __import__('gaia_prime.physics_mesh', fromlist=['CollapseType']).CollapseType.ENTROPY_SPIKE
                )
        
        # Compute metrics
        avg_confidence = (
            sum(n.confidence for n in generated_nodes) / len(generated_nodes)
            if generated_nodes else 0.0
        )
        
        crystallized_count = sum(
            1 for n in generated_nodes 
            if n.node_id in self.physics.collapse.crystallized
        )
        
        convergence_points = sum(
            1 for n in generated_nodes 
            if n.is_convergence_point
        )
        
        unique_paths = len(set(n.node_id for n in generated_nodes))
        
        return GenerationResult(
            text=prompt + " " + " ".join(generated_tokens),
            tokens=generated_tokens,
            nodes=generated_nodes,
            avg_confidence=avg_confidence,
            crystallized_count=crystallized_count,
            attractor_influences=self.attractor_hits,
            entropy_at_end=self.physics.state.entropy,
            convergence_points=convergence_points,
            unique_paths=unique_paths
        )
    
    def generate_with_guidance(self,
                               prompt: str,
                               guidance_tokens: List[str],
                               guidance_weight: float = 0.5) -> GenerationResult:
        """
        Generate with soft guidance toward certain tokens.
        
        Useful for steering generation without hard constraints.
        """
        # Store original config
        original_attractor_weight = self.config.attractor_weight
        
        # Temporarily boost nodes matching guidance tokens
        guidance_embs = [self.embeddings.embed(t) for t in guidance_tokens]
        
        # Find/create guidance nodes and mark as temporary attractors
        temp_attractors = []
        for token, emb in zip(guidance_tokens, guidance_embs):
            results = self.physics.query(emb, top_k=1, threshold=0.7)
            if results:
                node, _ = results[0]
                if node.node_id not in self.physics.attractors:
                    self.physics.attractors[node.node_id] = guidance_weight
                    temp_attractors.append(node.node_id)
        
        # Increase attractor influence
        self.config.attractor_weight = original_attractor_weight + guidance_weight
        
        # Generate
        result = self.generate(prompt)
        
        # Restore
        self.config.attractor_weight = original_attractor_weight
        for node_id in temp_attractors:
            if node_id in self.physics.attractors:
                del self.physics.attractors[node_id]
        
        return result


class PhysicsChat:
    """
    Interactive chat interface using PhysicsGenerator.
    
    Maintains conversation history as mesh paths.
    """
    
    def __init__(self,
                 physics: PhysicsMesh,
                 embeddings: SimpleEmbeddings,
                 config: GenerationConfig = None):
        self.generator = PhysicsGenerator(physics, embeddings, config)
        self.physics = physics
        self.conversation_nodes: List[MeshNode] = []
        self.turn_count = 0
    
    def respond(self, user_input: str) -> str:
        """
        Generate response to user input.
        
        Uses conversation history as context.
        """
        self.turn_count += 1
        
        # Encode user input
        user_nodes = self.generator.encode_prompt(user_input)
        
        # Mark user input as important
        for node in user_nodes:
            self.physics.remember(node, importance=0.7)
        
        # Add to conversation
        self.conversation_nodes.extend(user_nodes)
        
        # Build prompt with context
        if self.conversation_nodes:
            # Use recent conversation as context
            context_tokens = [n.token_str for n in self.conversation_nodes[-20:]]
            prompt = " ".join(context_tokens)
        else:
            prompt = user_input
        
        # Generate response
        result = self.generator.generate(prompt)
        
        # Add response to conversation
        self.conversation_nodes.extend(result.nodes)
        
        # Run physics to integrate
        self.physics.step()
        
        return " ".join(result.tokens)
    
    def reset(self):
        """Reset conversation history."""
        self.conversation_nodes = []
        self.turn_count = 0


# Convenience function
def create_generator(device: str = 'cpu', embed_dim: int = 64) -> PhysicsGenerator:
    """Create a ready-to-use physics generator."""
    mesh = PACMeshSpace(embed_dim=embed_dim, device=device)
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=embed_dim)
    return PhysicsGenerator(physics, embeddings)
