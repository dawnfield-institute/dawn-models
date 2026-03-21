"""
Field Generator Core
====================

Core components for language generation through field dynamics.
Reuses v6 encoder from POC-004 and attention from POC-003.

Key components:
- FieldVocabulary: Token-to-field mapping with real embeddings
- FieldPredictor: Next-token prediction via field evolution
- FieldGenerator: Sequence generation with resonance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

# Add POC-004 to path for scale_field
poc_004_path = Path(__file__).resolve().parents[1].parent / 'poc_004_scale_dimension' / 'scripts'
sys.path.insert(0, str(poc_004_path))

from scale_field import (
    SphericalHarmonicEncoder,
    PHI, XI, PHI_XI, LAMBDA_STAR
)

# Try to import sentence-transformers for real embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("WARNING: sentence-transformers not available, using synthetic embeddings")


class FieldVocabulary:
    """
    Vocabulary with field-encoded tokens.
    
    Uses sentence-transformers for real semantic embeddings,
    then encodes to 3D fields using v6 encoder.
    """
    
    def __init__(self, device='cuda', field_shape=(32, 32, 32)):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.field_shape = field_shape
        
        # Encoder
        self.encoder = SphericalHarmonicEncoder(
            shape=field_shape,
            l_max=8,
            device=self.device
        )
        
        # Embedding model
        if HAS_SBERT:
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        else:
            self.embed_model = None
            
        # Vocabulary storage
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.token_embeddings: Dict[int, torch.Tensor] = {}  # Raw embeddings
        self.token_fields: Dict[int, torch.Tensor] = {}      # 3D fields
        self.next_id = 0
        
    def add_token(self, token: str) -> int:
        """Add token to vocabulary, return ID."""
        if token in self.token_to_id:
            return self.token_to_id[token]
            
        token_id = self.next_id
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        self.next_id += 1
        
        # Get embedding
        if self.embed_model:
            embedding = self.embed_model.encode([token], convert_to_tensor=True)[0]
        else:
            # Synthetic: hash-based embedding
            torch.manual_seed(hash(token) % 2**32)
            embedding = torch.randn(384, device=self.device)
            embedding = embedding / embedding.norm()
            
        self.token_embeddings[token_id] = embedding
        
        # Encode to field using v6
        field = self.encoder.encode_v6(embedding)
        self.token_fields[token_id] = field
        
        return token_id
        
    def add_tokens(self, tokens: List[str]) -> List[int]:
        """Add multiple tokens, return IDs."""
        return [self.add_token(t) for t in tokens]
        
    def get_field(self, token: str) -> torch.Tensor:
        """Get field for token (adding if needed)."""
        token_id = self.add_token(token)
        return self.token_fields[token_id]
        
    def get_embedding(self, token: str) -> torch.Tensor:
        """Get raw embedding for token."""
        token_id = self.add_token(token)
        return self.token_embeddings[token_id]
        
    def find_nearest(self, field: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find nearest tokens to a field."""
        scores = []
        
        for token_id, cached_field in self.token_fields.items():
            sim = F.cosine_similarity(
                field.flatten().unsqueeze(0),
                cached_field.flatten().unsqueeze(0)
            ).item()
            scores.append((self.id_to_token[token_id], sim))
            
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
        
    def __len__(self):
        return len(self.token_to_id)


class FieldPredictor:
    """
    Predict next tokens through field evolution.
    
    Uses Klein-Gordon dynamics to evolve context field,
    then finds nearest vocabulary token.
    """
    
    def __init__(self, vocab: FieldVocabulary):
        self.vocab = vocab
        self.device = vocab.device
        
        # Evolution parameters
        self.dt = 0.01
        self.evolution_steps = 10
        
        # Transition statistics
        self.transitions: Dict[Tuple[int, int], float] = {}
        
    def combine_context(self, tokens: List[str], 
                       weights: Optional[List[float]] = None) -> torch.Tensor:
        """Combine token fields with position weighting."""
        fields = [self.vocab.get_field(t) for t in tokens]
        
        if weights is None:
            # Recency weighting (more recent = higher weight)
            n = len(fields)
            weights = [LAMBDA_STAR ** (n - i - 1) for i in range(n)]
            
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
        
        # Weighted combination
        combined = torch.zeros_like(fields[0])
        for field, weight in zip(fields, weights):
            combined += weight * field
            
        return combined
        
    def evolve_field(self, field: torch.Tensor) -> torch.Tensor:
        """Evolve field forward using Klein-Gordon dynamics."""
        evolved = field.clone()
        
        # 3D Laplacian kernel
        kernel = torch.zeros(3, 3, 3, device=self.device)
        kernel[1, 1, 1] = -6
        kernel[0, 1, 1] = kernel[2, 1, 1] = 1
        kernel[1, 0, 1] = kernel[1, 2, 1] = 1
        kernel[1, 1, 0] = kernel[1, 1, 2] = 1
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        for _ in range(self.evolution_steps):
            # Pad for convolution
            padded = F.pad(evolved.unsqueeze(0).unsqueeze(0), 
                          (1, 1, 1, 1, 1, 1), mode='replicate')
            
            # Laplacian
            laplacian = F.conv3d(padded, kernel).squeeze()
            
            # Klein-Gordon evolution: ∂²φ/∂t² = ∇²φ - m²φ
            # Simplified as first-order: φ' = φ + dt*(∇²φ - λ*φ)
            evolved = evolved + self.dt * (laplacian - LAMBDA_STAR * evolved)
            
        return evolved
        
    def predict(self, context: List[str], top_k: int = 5, 
                exclude_context: bool = True) -> List[Tuple[str, float]]:
        """Predict next token from context."""
        # Combine context fields
        context_field = self.combine_context(context)
        
        # Evolve forward
        evolved_field = self.evolve_field(context_field)
        
        # Apply transition biases if available (stronger effect)
        if len(context) > 0:
            last_token_id = self.vocab.token_to_id.get(context[-1])
            if last_token_id is not None:
                for (from_id, to_id), strength in self.transitions.items():
                    if from_id == last_token_id and to_id in self.vocab.token_fields:
                        # Strong boost toward learned transitions
                        target_field = self.vocab.token_fields[to_id]
                        evolved_field = evolved_field + 0.5 * strength * target_field
        
        # Find nearest tokens, excluding context tokens to avoid repetition
        scores = []
        context_set = set(context) if exclude_context else set()
        
        for token_id, cached_field in self.vocab.token_fields.items():
            token = self.vocab.id_to_token[token_id]
            if token in context_set:
                continue  # Skip tokens already in context
            sim = F.cosine_similarity(
                evolved_field.flatten().unsqueeze(0),
                cached_field.flatten().unsqueeze(0)
            ).item()
            scores.append((token, sim))
            
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
        
    def learn_transition(self, from_token: str, to_token: str, strength: float = 1.0):
        """Learn a transition between tokens through resonance."""
        from_id = self.vocab.add_token(from_token)
        to_id = self.vocab.add_token(to_token)
        
        key = (from_id, to_id)
        if key in self.transitions:
            # Strengthen existing transition
            self.transitions[key] = min(self.transitions[key] + strength * 0.1, 2.0)
        else:
            self.transitions[key] = strength
            
    def train_on_sequence(self, tokens: List[str]):
        """Train on a sequence by learning transitions."""
        # Add all tokens to vocab
        self.vocab.add_tokens(tokens)
        
        # Learn transitions
        for i in range(len(tokens) - 1):
            self.learn_transition(tokens[i], tokens[i + 1])
            

class FieldGenerator:
    """
    Generate sequences through field dynamics.
    
    Combines prediction with resonance and phase transitions.
    """
    
    def __init__(self, predictor: FieldPredictor):
        self.predictor = predictor
        self.device = predictor.device
        
    def generate(self, prompt: List[str], max_tokens: int = 10,
                temperature: float = 1.0) -> List[str]:
        """Generate continuation from prompt."""
        sequence = prompt.copy()
        
        for _ in range(max_tokens):
            # Predict next
            predictions = self.predictor.predict(sequence, top_k=10)
            
            if not predictions:
                break
                
            # Apply temperature
            if temperature == 0:
                # Greedy
                next_token = predictions[0][0]
            else:
                # Sample with temperature
                scores = torch.tensor([p[1] for p in predictions], device=self.device)
                probs = F.softmax(scores / temperature, dim=0)
                idx = torch.multinomial(probs, 1).item()
                next_token = predictions[idx][0]
                
            # Check for repetition - avoid same token twice in a row
            if len(sequence) >= 1 and next_token == sequence[-1]:
                # Take second choice if available
                if len(predictions) > 1:
                    next_token = predictions[1][0]
                else:
                    break
                    
            sequence.append(next_token)
            
        return sequence
        
    def generate_with_resonance(self, prompt: List[str], 
                                max_tokens: int = 10) -> Tuple[List[str], List[float]]:
        """Generate with resonance scores for each token."""
        sequence = prompt.copy()
        scores = []
        
        for _ in range(max_tokens):
            predictions = self.predictor.predict(sequence, top_k=5)
            
            if not predictions:
                break
            
            # Pick best prediction that isn't the last token
            next_token, score = predictions[0]
            if len(sequence) > 0 and next_token == sequence[-1] and len(predictions) > 1:
                next_token, score = predictions[1]
                
            sequence.append(next_token)
            scores.append(score)
            
            # Stop if confidence drops too low
            if score < 0.3:
                break
                
        return sequence, scores
