"""
GAIA POC Unified Architecture
==============================

Unified architecture combining all POC components into a single model.
This is the validated POC-based implementation ready for benchmarking.

Components integrated:
- POC-001: Pattern encoding (v6 encoder with ξ-modulation)
- POC-002: Resonance training (no backprop)
- POC-003: Field attention (resonance-based)
- POC-004: 3D scaling (spherical harmonics)
- POC-005: Language generation (Klein-Gordon evolution)
- POC-006: Memory persistence (field superposition)

Total POC validation: 113+ tests passed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

# Dawn Field Constants
PHI = 1.618033988749895
XI = 0.0618
PHI_XI = PHI * XI  # 1.710 - crystallization threshold
LAMBDA_STAR = 0.9816  # Optimal decay

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False


@dataclass
class GAIAConfig:
    """Configuration for GAIA POC model."""
    field_shape: Tuple[int, int, int] = (24, 24, 24)
    embedding_dim: int = 384
    l_max: int = 8
    memory_capacity: int = 1000
    context_depth: int = 10
    evolution_steps: int = 10
    dt: float = 0.01
    device: str = 'cuda'


class SphericalEncoderV6(nn.Module):
    """
    v6 Encoder: Geometric E=mc² Preservation
    
    Key innovations (from POC-004):
    - ξ-modulation: weight by local contrast
    - DCT orthogonal bases
    - 0.977 correlation with original embeddings
    """
    
    def __init__(self, config: GAIAConfig):
        super().__init__()
        self.config = config
        self.device = config.device if torch.cuda.is_available() else 'cpu'
        self.shape = config.field_shape
        
        # Pre-compute coordinate grids
        x = torch.linspace(-1, 1, self.shape[0])
        y = torch.linspace(-1, 1, self.shape[1])
        z = torch.linspace(-1, 1, self.shape[2])
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        self.register_buffer('X', X)
        self.register_buffer('Y', Y)
        self.register_buffer('Z', Z)
        
        # Pre-compute DCT bases
        self._init_bases(config.embedding_dim)
        
    def _init_bases(self, n_bases: int):
        """Initialize DCT orthogonal bases."""
        bases_list = []
        
        # Generate frequency combinations
        freqs = []
        for fx in range(8):
            for fy in range(8):
                for fz in range(8):
                    if fx + fy + fz > 0:
                        freqs.append((fx, fy, fz))
        
        freqs.sort(key=lambda f: f[0]**2 + f[1]**2 + f[2]**2)
        freqs = freqs[:n_bases]
        
        for fx, fy, fz in freqs:
            basis = torch.cos(fx * math.pi * self.X) * \
                    torch.cos(fy * math.pi * self.Y) * \
                    torch.cos(fz * math.pi * self.Z)
            basis = basis / (basis.norm() + 1e-8)
            bases_list.append(basis)
            
        self.register_buffer('bases', torch.stack(bases_list))
        
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Encode embedding to 3D field."""
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
            
        batch_size = embedding.shape[0]
        n_bases = min(embedding.shape[1], self.bases.shape[0])
        
        # ξ-modulation
        local_mean = embedding.mean(dim=-1, keepdim=True)
        contrast = (embedding - local_mean).abs()
        max_contrast = contrast.max(dim=-1, keepdim=True)[0] + 1e-8
        xi_weight = 1.0 + XI * contrast / max_contrast
        
        weighted = embedding[:, :n_bases] * xi_weight[:, :n_bases]
        
        # Project onto bases
        fields = torch.einsum('bn,nxyz->bxyz', weighted, self.bases[:n_bases])
        
        return fields.squeeze(0) if batch_size == 1 else fields


class FieldMemory(nn.Module):
    """
    Field-based memory with resonance retrieval (POC-006).
    """
    
    def __init__(self, config: GAIAConfig):
        super().__init__()
        self.config = config
        self.device = config.device if torch.cuda.is_available() else 'cpu'
        
        self.patterns: Dict[int, torch.Tensor] = {}
        self.next_id = 0
        self.transitions: Dict[Tuple[int, int], float] = {}
        
    def store(self, field: torch.Tensor, token_id: Optional[int] = None) -> int:
        """Store pattern."""
        if token_id is None:
            token_id = self.next_id
        self.next_id = max(self.next_id, token_id + 1)
        
        self.patterns[token_id] = field.clone()
        
        if len(self.patterns) > self.config.memory_capacity:
            oldest = min(self.patterns.keys())
            del self.patterns[oldest]
            
        return token_id
        
    def retrieve(self, query: torch.Tensor, top_k: int = 5,
                exclude: Optional[set] = None) -> List[Tuple[int, float]]:
        """Retrieve by resonance."""
        exclude = exclude or set()
        scores = []
        
        for pid, field in self.patterns.items():
            if pid in exclude:
                continue
            sim = F.cosine_similarity(
                query.flatten().unsqueeze(0),
                field.flatten().unsqueeze(0)
            ).item()
            scores.append((pid, sim))
            
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
        
    def learn_transition(self, from_id: int, to_id: int):
        """Learn transition."""
        key = (from_id, to_id)
        self.transitions[key] = self.transitions.get(key, 0) + 0.1


class KleinGordonEvolution(nn.Module):
    """
    Klein-Gordon field evolution (POC-005).
    """
    
    def __init__(self, config: GAIAConfig):
        super().__init__()
        self.config = config
        
        kernel = torch.zeros(3, 3, 3)
        kernel[1, 1, 1] = -6
        kernel[0, 1, 1] = kernel[2, 1, 1] = 1
        kernel[1, 0, 1] = kernel[1, 2, 1] = 1
        kernel[1, 1, 0] = kernel[1, 1, 2] = 1
        self.register_buffer('kernel', kernel.unsqueeze(0).unsqueeze(0))
        
    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """Evolve field."""
        evolved = field.clone()
        
        for _ in range(self.config.evolution_steps):
            padded = F.pad(evolved.unsqueeze(0).unsqueeze(0),
                          (1, 1, 1, 1, 1, 1), mode='replicate')
            laplacian = F.conv3d(padded, self.kernel).squeeze()
            evolved = evolved + self.config.dt * (laplacian - LAMBDA_STAR * evolved)
            
        return evolved


class ResonanceAttention(nn.Module):
    """
    Resonance-based attention (POC-003).
    No QKV projections - pure field physics.
    """
    
    def forward(self, query: torch.Tensor, 
                keys: List[torch.Tensor]) -> torch.Tensor:
        """Compute attention via resonance."""
        if not keys:
            return torch.zeros(0, device=query.device)
            
        weights = []
        for key in keys:
            sim = F.cosine_similarity(
                query.flatten().unsqueeze(0),
                key.flatten().unsqueeze(0)
            ).squeeze()  # Ensure scalar
            weights.append(sim)
            
        weights = torch.stack(weights)
        
        # Recency bias
        n = len(weights)
        recency = torch.tensor([LAMBDA_STAR ** (n - i - 1) for i in range(n)],
                              device=query.device, dtype=weights.dtype)
        weights = weights * recency
        
        return F.softmax(weights / PHI_XI, dim=0)


class GAIAUnified(nn.Module):
    """
    GAIA Unified POC Architecture
    
    Combines all POC findings into a single model:
    - v6 spherical encoding
    - Field-based memory
    - Klein-Gordon evolution
    - Resonance attention
    - Transition learning
    """
    
    def __init__(self, config: Optional[GAIAConfig] = None):
        super().__init__()
        self.config = config or GAIAConfig()
        self.device = self.config.device if torch.cuda.is_available() else 'cpu'
        
        # Components
        self.encoder = SphericalEncoderV6(self.config)
        self.memory = FieldMemory(self.config)
        self.evolution = KleinGordonEvolution(self.config)
        self.attention = ResonanceAttention()
        
        # Vocabulary
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # Embedder
        if HAS_SBERT:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        else:
            self.embedder = None
            
        # Context
        self.context: List[torch.Tensor] = []
        self.context_ids: List[int] = []
        
    def to(self, device):
        """Move to device."""
        self.device = device
        return super().to(device)
        
    def add_token(self, token: str) -> int:
        """Add token to vocabulary."""
        if token in self.token_to_id:
            return self.token_to_id[token]
            
        if self.embedder:
            embedding = self.embedder.encode([token], convert_to_tensor=True)[0]
        else:
            torch.manual_seed(hash(token) % 2**32)
            embedding = torch.randn(self.config.embedding_dim, device=self.device)
            embedding = embedding / embedding.norm()
            
        field = self.encoder(embedding.to(self.device))
        
        token_id = len(self.token_to_id)
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        self.memory.store(field, token_id)
        
        return token_id
        
    def add_tokens(self, tokens: List[str]) -> List[int]:
        """Add multiple tokens."""
        return [self.add_token(t) for t in tokens]
        
    def train_sequence(self, tokens: List[str]):
        """Train via transition learning."""
        ids = self.add_tokens(tokens)
        for i in range(len(ids) - 1):
            self.memory.learn_transition(ids[i], ids[i + 1])
            
    def push_context(self, token: str):
        """Add token to context."""
        token_id = self.add_token(token)
        field = self.memory.patterns.get(token_id)
        
        if field is not None:
            self.context.append(field)
            self.context_ids.append(token_id)
            
            if len(self.context) > self.config.context_depth:
                self.context.pop(0)
                self.context_ids.pop(0)
                
    def get_context_field(self) -> torch.Tensor:
        """Get attention-weighted context."""
        if not self.context:
            return torch.zeros(*self.config.field_shape, device=self.device)
            
        query = self.context[-1]
        weights = self.attention(query, self.context)
        
        combined = torch.zeros(*self.config.field_shape, device=self.device)
        for i, field in enumerate(self.context):
            w = weights[i].item()  # Extract scalar
            combined += w * field
            
        return combined
        
    def predict(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict next token."""
        if not self.context:
            return []
            
        context_field = self.get_context_field()
        evolved = self.evolution(context_field)
        
        # Transition boost
        if self.context_ids:
            last_id = self.context_ids[-1]
            for (from_id, to_id), strength in self.memory.transitions.items():
                if from_id == last_id and to_id in self.memory.patterns:
                    evolved = evolved + 0.5 * strength * self.memory.patterns[to_id]
                    
        exclude = set(self.context_ids[-3:])
        retrieved = self.memory.retrieve(evolved, top_k, exclude)
        
        return [(self.id_to_token[tid], score) for tid, score in retrieved
                if tid in self.id_to_token]
        
    def generate(self, prompt: List[str], max_tokens: int = 10,
                temperature: float = 0.5) -> List[str]:
        """Generate from prompt."""
        self.context = []
        self.context_ids = []
        
        for token in prompt:
            self.push_context(token)
            
        sequence = prompt.copy()
        
        for _ in range(max_tokens):
            preds = self.predict(top_k=10)
            if not preds:
                break
                
            if temperature == 0:
                next_token = preds[0][0]
            else:
                scores = torch.tensor([p[1] for p in preds], device=self.device)
                probs = F.softmax(scores / temperature, dim=0)
                idx = torch.multinomial(probs, 1).item()
                next_token = preds[idx][0]
                
            sequence.append(next_token)
            self.push_context(next_token)
            
        return sequence
        
    def clear_context(self):
        """Clear context."""
        self.context = []
        self.context_ids = []
        
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)


def create_gaia_unified(device: str = 'cuda') -> GAIAUnified:
    """Create GAIA unified model."""
    config = GAIAConfig(device=device)
    model = GAIAUnified(config)
    return model.to(device if torch.cuda.is_available() else 'cpu')
