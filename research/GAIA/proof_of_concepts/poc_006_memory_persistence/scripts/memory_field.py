"""
Memory Field Core
=================

Field-based memory system with storage and retrieval.
Uses resonance for recall and conservation for stability.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

# Add POC-004 for encoder
poc_004_path = Path(__file__).resolve().parents[2] / 'poc_004_scale_dimension' / 'scripts'
sys.path.insert(0, str(poc_004_path))

from scale_field import SphericalHarmonicEncoder, PHI, XI, PHI_XI, LAMBDA_STAR


class MemoryField:
    """
    Field-based memory with persistent storage.
    
    Uses superposition to store multiple patterns,
    resonance for retrieval, and conservation for stability.
    """
    
    def __init__(self, shape=(32, 32, 32), device='cuda', capacity=1000):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.shape = shape
        self.capacity = capacity
        
        # Main memory field (superposition of all patterns)
        self.field = torch.zeros(*shape, device=self.device)
        
        # Pattern storage for retrieval validation
        self.patterns: Dict[int, torch.Tensor] = {}
        self.pattern_fields: Dict[int, torch.Tensor] = {}
        self.pattern_keys: Dict[int, torch.Tensor] = {}  # For resonance lookup
        self.next_id = 0
        
        # Encoder
        self.encoder = SphericalHarmonicEncoder(
            shape=shape,
            l_max=8,
            device=self.device
        )
        
        # Memory parameters
        self.decay = LAMBDA_STAR  # How much old patterns fade
        self.consolidation_strength = PHI_XI  # Crystallization threshold
        
    def store(self, pattern: torch.Tensor, key: Optional[torch.Tensor] = None) -> int:
        """Store a pattern in memory, return ID."""
        pattern_id = self.next_id
        self.next_id += 1
        
        # Encode pattern to field
        if pattern.dim() == 1:
            pattern_field = self.encoder.encode_v6(pattern)
        else:
            pattern_field = pattern
            
        # Store original for validation
        self.patterns[pattern_id] = pattern.clone() if pattern.dim() == 1 else None
        self.pattern_fields[pattern_id] = pattern_field.clone()
        
        # Create key (for retrieval) - use pattern mean as signature
        if key is None:
            key = pattern_field.mean(dim=(0, 1))  # Z-axis projection
        self.pattern_keys[pattern_id] = key
        
        # Add to superposition field with decay of existing
        self.field = self.decay * self.field + pattern_field
        
        # Normalize to prevent explosion
        max_val = self.field.abs().max()
        if max_val > 10.0:
            self.field = self.field / max_val * 10.0
            
        # Manage capacity
        if len(self.patterns) > self.capacity:
            oldest_id = min(self.patterns.keys())
            del self.patterns[oldest_id]
            del self.pattern_fields[oldest_id]
            del self.pattern_keys[oldest_id]
            
        return pattern_id
        
    def retrieve(self, query: torch.Tensor, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Retrieve patterns by resonance with query.
        Returns list of (pattern_id, similarity) tuples.
        """
        if query.dim() == 1:
            query_field = self.encoder.encode_v6(query)
        else:
            query_field = query
            
        # Find resonating patterns
        scores = []
        for pattern_id, stored_field in self.pattern_fields.items():
            # Resonance = field correlation
            sim = F.cosine_similarity(
                query_field.flatten().unsqueeze(0),
                stored_field.flatten().unsqueeze(0)
            ).item()
            scores.append((pattern_id, sim))
            
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
        
    def recall(self, pattern_id: int) -> Optional[torch.Tensor]:
        """Recall a specific pattern by ID."""
        if pattern_id not in self.pattern_fields:
            return None
        return self.pattern_fields[pattern_id]
        
    def compute_interference(self, pattern_id: int) -> float:
        """
        Compute how much a pattern has been interfered with.
        Returns 0.0 (no interference) to 1.0 (completely overwritten).
        """
        if pattern_id not in self.pattern_fields:
            return 1.0
            
        original = self.pattern_fields[pattern_id]
        
        # Project field onto original pattern direction
        projection = F.cosine_similarity(
            self.field.flatten().unsqueeze(0),
            original.flatten().unsqueeze(0)
        ).item()
        
        # Higher projection = less interference
        return 1.0 - abs(projection)
        
    def get_pattern_count(self) -> int:
        """Number of patterns stored."""
        return len(self.patterns)
        
    def get_field_energy(self) -> float:
        """Total field energy (for conservation check)."""
        return (self.field ** 2).sum().item()
        
    def get_field_entropy(self) -> float:
        """Field entropy (measure of information content)."""
        # Normalize field to probability-like distribution
        field_abs = self.field.abs()
        if field_abs.sum() < 1e-10:
            return 0.0
        p = field_abs / field_abs.sum()
        # Shannon entropy
        log_p = torch.log2(p + 1e-10)
        return -(p * log_p).sum().item()
        
    def consolidate(self):
        """
        Consolidate memory by crystallizing strong patterns.
        Weak patterns fade, strong patterns sharpen.
        """
        # Find local maxima (crystallization points)
        kernel = torch.ones(3, 3, 3, device=self.device) / 27
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        padded = F.pad(self.field.unsqueeze(0).unsqueeze(0),
                      (1, 1, 1, 1, 1, 1), mode='replicate')
        local_mean = F.conv3d(padded, kernel).squeeze()
        
        # Sharpen: enhance deviations from local mean
        deviation = self.field - local_mean
        self.field = local_mean + self.consolidation_strength * deviation


class SequentialMemory:
    """
    Sequential memory for ordered pattern storage.
    Maintains temporal order for sequence recall.
    """
    
    def __init__(self, shape=(32, 32, 32), device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.shape = shape
        
        # Ordered storage
        self.sequence: List[torch.Tensor] = []
        self.sequence_fields: List[torch.Tensor] = []
        
        # Encoder
        self.encoder = SphericalHarmonicEncoder(
            shape=shape,
            l_max=8,
            device=self.device
        )
        
    def append(self, pattern: torch.Tensor):
        """Add pattern to sequence."""
        if pattern.dim() == 1:
            field = self.encoder.encode_v6(pattern)
        else:
            field = pattern
            
        self.sequence.append(pattern.clone() if pattern.dim() == 1 else None)
        self.sequence_fields.append(field)
        
    def get_context(self, depth: int = 10) -> torch.Tensor:
        """
        Get context field from last N patterns.
        Uses exponential decay for recency weighting.
        """
        if not self.sequence_fields:
            return torch.zeros(*self.shape, device=self.device)
            
        n = min(depth, len(self.sequence_fields))
        fields = self.sequence_fields[-n:]
        
        # Recency weighting
        weights = [LAMBDA_STAR ** (n - i - 1) for i in range(n)]
        total = sum(weights)
        weights = [w / total for w in weights]
        
        context = torch.zeros(*self.shape, device=self.device)
        for field, weight in zip(fields, weights):
            context += weight * field
            
        return context
        
    def retrieve_at(self, index: int) -> Optional[torch.Tensor]:
        """Retrieve pattern at specific position."""
        if 0 <= index < len(self.sequence_fields):
            return self.sequence_fields[index]
        return None
        
    def similarity_at_depth(self, query: torch.Tensor, depth: int) -> float:
        """
        Compute similarity to pattern at given depth from end.
        depth=0 is most recent, depth=N is N patterns ago.
        """
        if depth >= len(self.sequence_fields):
            return 0.0
            
        target_field = self.sequence_fields[-(depth + 1)]
        
        if query.dim() == 1:
            query_field = self.encoder.encode_v6(query)
        else:
            query_field = query
            
        return F.cosine_similarity(
            query_field.flatten().unsqueeze(0),
            target_field.flatten().unsqueeze(0)
        ).item()
        
    def __len__(self):
        return len(self.sequence)
