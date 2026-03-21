"""
GAIA v4.0 - Cortex Module

Central integration layer for GAIA cognition.
Built on Fracton PAC-Lazy substrate.

The Cortex routes information between organs, maintains global
field coherence, and integrates organ outputs into unified responses.
"""

import torch
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time

# Import Fracton substrate
from fracton.core import PACSystem
from fracton.field import spherical_encode_batch, evolve, compute_resonance
from fracton.physics import PHI, XI, PHI_XI, LAMBDA_STAR
from fracton.storage import KronosBackend


@dataclass
class GAIAConfig:
    """Configuration for GAIA v4.0."""
    
    # Field dimensions
    field_dim: int = 64
    
    # Vocabulary size (for encoding)
    vocab_size: int = 50257
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Evolution parameters
    evolution_steps: int = 5
    evolution_dt: float = 0.1
    
    # Substrate parameters (passed to Fracton)
    substrate_capacity: int = 100000
    hot_cache_size: int = 10000
    warm_cache_size: int = 100000
    
    # Resonance thresholds
    resonance_threshold: float = 0.5
    top_k_resonant: int = 10
    
    # Organ configuration
    default_organs: List[str] = field(default_factory=lambda: [
        'language', 'reasoning', 'memory'
    ])
    
    # Kronos persistence (optional)
    kronos_path: Optional[str] = None  # Path for persistent storage
    kronos_namespace: str = "gaia"     # Namespace for FDO docs
    auto_persist: bool = True          # Auto-save important patterns
    persist_threshold: float = 0.5     # Importance threshold


@dataclass
class GAIAResponse:
    """Response from GAIA processing."""
    
    # Evolved field state
    field_state: torch.Tensor
    
    # Substrate node ID where pattern stored
    node_id: int
    
    # Resonant patterns found
    resonant_patterns: List[Tuple[int, float]]
    
    # Contributions from each organ
    organ_contributions: Dict[str, torch.Tensor]
    
    # Physics metrics (from Fracton)
    conservation_residual: float = 0.0
    field_energy: float = 0.0
    
    # Timing
    processing_time: float = 0.0


class GAIACortex:
    """
    Central integration layer for GAIA cognition.
    
    Responsibilities:
    - Route information between organs
    - Maintain global field coherence
    - Integrate organ outputs into unified response
    - Monitor field energy and trigger phase transitions
    
    Usage:
        config = GAIAConfig(device='cuda')
        cortex = GAIACortex(config)
        
        response = cortex.process("hello world")
        # or
        response = cortex.process([token_ids])
    """
    
    def __init__(self, config: GAIAConfig = None):
        self.config = config or GAIAConfig()
        
        # Initialize Kronos backend if path provided
        self._kronos_backend = None
        if self.config.kronos_path:
            self._kronos_backend = KronosBackend(
                Path(self.config.kronos_path),
                self.config.kronos_namespace
            )
        
        # Fracton substrate for all field operations
        self.substrate = PACSystem(
            device=self.config.device,
            hot_cache_size=self.config.hot_cache_size,
            warm_cache_size=self.config.warm_cache_size,
            kronos_backend=self._kronos_backend,
            auto_persist=self.config.auto_persist,
            persist_threshold=self.config.persist_threshold
        )
        
        # Attached organs (specialized transformers)
        self.organs: Dict[str, 'TransformerOrgan'] = {}
        
        # Global field state (consciousness field)
        self.consciousness_field = torch.zeros(
            self.config.field_dim, 
            device=self.config.device
        )
        
        # Pre-computed reference fields for decoding
        self._reference_fields: Optional[torch.Tensor] = None
        
        # Statistics
        self._process_count = 0
        self._total_time = 0.0
    
    def attach_organ(self, organ: 'TransformerOrgan') -> None:
        """
        Attach a transformer organ to the cortex.
        
        Args:
            organ: TransformerOrgan instance to attach
        """
        organ.substrate = self.substrate  # Share substrate
        self.organs[organ.name] = organ
    
    def process(self, input_data: Any) -> GAIAResponse:
        """
        Main cognitive processing loop.
        
        1. Encode input via Fracton
        2. Inject into substrate
        3. Route to relevant organs
        4. Integrate organ responses
        5. Evolve consciousness field
        6. Return unified response
        
        Args:
            input_data: Can be:
                - str: Text to encode
                - int: Single token ID
                - List[int]: Token ID sequence
                - torch.Tensor: Pre-encoded field
                
        Returns:
            GAIAResponse with processing results
        """
        start_time = time.time()
        self._process_count += 1
        
        # Encode input
        field = self._encode_input(input_data)
        
        # Inject into PAC-Lazy substrate
        node_id = self.substrate.inject(field, label="input")
        
        # Find resonant patterns (lazy expansion)
        resonant = self.substrate.find_resonant(
            field, 
            top_k=self.config.top_k_resonant,
            threshold=self.config.resonance_threshold
        )
        
        # Route to organs based on field characteristics
        organ_responses = self._route_to_organs(field, resonant)
        
        # Integrate responses
        integrated = self._integrate_responses(organ_responses)
        
        # Evolve consciousness field
        combined = self.consciousness_field + integrated
        self.consciousness_field = evolve(
            combined,
            steps=self.config.evolution_steps,
            dt=self.config.evolution_dt
        )
        
        # Compute physics metrics
        conservation_residual = 0.0
        if node_id in [n.id for n in self.substrate._roots.values()]:
            valid, residual = self.substrate.validate_conservation(node_id)
            conservation_residual = residual
        
        field_energy = torch.sum(self.consciousness_field ** 2).item()
        
        processing_time = time.time() - start_time
        self._total_time += processing_time
        
        return GAIAResponse(
            field_state=self.consciousness_field.clone(),
            node_id=node_id,
            resonant_patterns=resonant,
            organ_contributions=organ_responses,
            conservation_residual=conservation_residual,
            field_energy=field_energy,
            processing_time=processing_time
        )
    
    def _encode_input(self, input_data: Any) -> torch.Tensor:
        """Encode input data to field representation."""
        
        if isinstance(input_data, torch.Tensor):
            # Already a field
            if input_data.shape[-1] == self.config.field_dim:
                return input_data.to(self.config.device)
            # Token IDs - encode
            return spherical_encode_batch(
                input_data.to(self.config.device),
                vocab_size=self.config.vocab_size,
                dim=self.config.field_dim
            )
        
        if isinstance(input_data, int):
            # Single token ID
            token_ids = torch.tensor([input_data], device=self.config.device)
            return spherical_encode_batch(
                token_ids,
                vocab_size=self.config.vocab_size,
                dim=self.config.field_dim
            )[0]
        
        if isinstance(input_data, list):
            # List of token IDs
            token_ids = torch.tensor(input_data, device=self.config.device)
            encoded = spherical_encode_batch(
                token_ids,
                vocab_size=self.config.vocab_size,
                dim=self.config.field_dim
            )
            # Average for sequence
            return encoded.mean(dim=0)
        
        if isinstance(input_data, str):
            # Text - need tokenizer (use simple hash for now)
            # In production, use transformers tokenizer
            token_ids = [hash(c) % self.config.vocab_size for c in input_data]
            token_tensor = torch.tensor(token_ids, device=self.config.device)
            encoded = spherical_encode_batch(
                token_tensor,
                vocab_size=self.config.vocab_size,
                dim=self.config.field_dim
            )
            return encoded.mean(dim=0)
        
        raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def _route_to_organs(self, 
                         field: torch.Tensor,
                         resonant: List[Tuple[int, float]]) -> Dict[str, torch.Tensor]:
        """Route field to appropriate organs for processing."""
        responses = {}
        
        for name, organ in self.organs.items():
            if organ.should_activate(field):
                responses[name] = organ.process(field, resonant)
        
        # If no organs activated, return identity
        if not responses:
            responses['passthrough'] = field
        
        return responses
    
    def _integrate_responses(self, 
                             responses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Integrate organ responses into unified field."""
        if not responses:
            return torch.zeros(self.config.field_dim, device=self.config.device)
        
        # Weighted average based on field energy
        total = torch.zeros(self.config.field_dim, device=self.config.device)
        weight_sum = 0.0
        
        for name, response in responses.items():
            energy = torch.sum(response ** 2).item()
            weight = max(energy, XI)  # Minimum weight
            total = total + weight * response
            weight_sum += weight
        
        if weight_sum > 0:
            total = total / weight_sum
        
        return total
    
    def predict_next(self, 
                     context: torch.Tensor,
                     top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Predict next token given context.
        
        Uses resonance matching against reference fields.
        
        Args:
            context: Context field
            top_k: Number of predictions
            
        Returns:
            List of (token_id, probability) tuples
        """
        # Ensure reference fields exist
        if self._reference_fields is None:
            from fracton.field import create_reference_fields
            self._reference_fields = create_reference_fields(
                vocab_size=self.config.vocab_size,
                dim=self.config.field_dim,
                device=self.config.device
            )
        
        # Process context through cortex
        response = self.process(context)
        
        # Find most resonant reference fields
        from fracton.field import compute_resonance_batch
        scores = compute_resonance_batch(
            response.field_state, 
            self._reference_fields
        )
        
        # Get top k
        top_scores, top_indices = torch.topk(scores, top_k)
        
        # Convert to probabilities via softmax
        probs = torch.softmax(top_scores / XI, dim=0)
        
        return [
            (idx.item(), prob.item())
            for idx, prob in zip(top_indices, probs)
        ]
    
    def reset(self) -> None:
        """Reset cortex state."""
        self.consciousness_field = torch.zeros(
            self.config.field_dim,
            device=self.config.device
        )
    
    def stats(self) -> Dict[str, Any]:
        """Get cortex statistics."""
        return {
            "process_count": self._process_count,
            "total_time": self._total_time,
            "avg_time": self._total_time / max(1, self._process_count),
            "substrate": self.substrate.stats(),
            "consciousness_energy": torch.sum(self.consciousness_field ** 2).item(),
            "organs": list(self.organs.keys()),
            "kronos_enabled": self._kronos_backend is not None
        }
    
    # === Persistence Methods ===
    
    def save_consciousness(self, name: str = None) -> str:
        """
        Save complete consciousness state to Kronos.
        
        Saves:
        - All substrate patterns
        - Consciousness field
        - Processing state
        
        Args:
            name: Optional episode name
            
        Returns:
            episode_id
        """
        if self._kronos_backend is None:
            raise RuntimeError("Kronos not configured. Set kronos_path in GAIAConfig.")
        
        # Save consciousness field as a pattern too
        consciousness_id = self.substrate.inject(
            self.consciousness_field,
            label="__consciousness_field__",
            importance=1.0,  # Always persist
            persist=True
        )
        
        # Save full state
        episode_id = self.substrate.save_state(
            name=name or "consciousness_snapshot",
            metadata={
                "process_count": self._process_count,
                "total_time": self._total_time,
                "consciousness_node_id": consciousness_id,
                "organs": list(self.organs.keys())
            }
        )
        
        return episode_id
    
    def restore_consciousness(self, episode_id: str) -> None:
        """
        Restore consciousness from Kronos episode.
        
        Args:
            episode_id: Episode to restore
        """
        if self._kronos_backend is None:
            raise RuntimeError("Kronos not configured. Set kronos_path in GAIAConfig.")
        
        # Restore substrate state
        self.substrate.restore_state(episode_id)
        
        # Find and restore consciousness field
        for node in self.substrate.cache._hot.values():
            if node.label == "__consciousness_field__":
                # This is our consciousness field
                self.consciousness_field = self.substrate.reconstruct(node.id)
                break
    
    def list_saved_states(self) -> List[Dict[str, Any]]:
        """
        List available saved consciousness states.
        
        Returns:
            List of episode metadata
        """
        if self._kronos_backend is None:
            return []
        
        return self._kronos_backend.list_episodes_detailed()
    
    def __repr__(self) -> str:
        kronos_info = " +kronos" if self._kronos_backend else ""
        return (f"GAIACortex(device={self.config.device}, "
                f"organs={list(self.organs.keys())}, "
                f"substrate_size={len(self.substrate)}{kronos_info})")


class TransformerOrgan:
    """
    Base class for GAIA transformer organs.
    
    Organs are specialized processing units that:
    - Receive field input from Cortex
    - Apply domain-specific transformation
    - Return transformed field
    
    All computation uses Fracton primitives.
    """
    
    def __init__(self, name: str, substrate: PACSystem = None):
        self.name = name
        self.substrate = substrate  # Set by Cortex when attached
        self.local_memory: Dict[int, int] = {}  # Pattern hash -> node_id
        self._activation_count = 0
    
    def process(self, 
                field: torch.Tensor,
                resonant: List[Tuple[int, float]] = None) -> torch.Tensor:
        """
        Transform field according to organ specialization.
        
        Args:
            field: Input field
            resonant: Resonant patterns from substrate
            
        Returns:
            Transformed field
        """
        raise NotImplementedError
    
    def should_activate(self, field: torch.Tensor) -> bool:
        """
        Check if organ should process this field.
        
        Override in subclasses for specialized activation logic.
        """
        return True  # Default: always activate
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
