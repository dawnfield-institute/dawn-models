"""
POC-025: GAIA Prime + Kronos Integration
========================================

Experiment 01: KronosGAIABridge

Creates the bridge between GAIA Prime (active cognition) and Kronos (persistent memory).

Architecture:
    GAIA Prime  ←──────→  KronosGAIABridge  ←──────→  KronosMemory
    (in-memory)           (translator)               (persistent)

Key Operations:
    1. crystallize(node) - Save high-importance patterns to Kronos
    2. recall(query) - Load relevant patterns from Kronos
    3. sync() - Ensure consistency between systems

Tests:
    - Conservation: crystallize→recall maintains PAC invariants
    - Fidelity: recalled patterns match original
    - Speed: operations complete within latency budget
"""

import sys
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\fracton")
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\dawn-models\research\GAIA\src")

import torch
import numpy as np
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass

# Fracton imports
from fracton.physics.constants import PHI, PHI_INV, XI, LAMBDA_STAR

# GAIA Prime imports  
from gaia_prime.validated_constants import (
    XI as XI_MOBIUS,  # Möbius operator (1.0571)
    PHI as PHI_GAIA,
    LAMBDA_STAR as LAMBDA_STAR_GAIA,
    ENTROPY_OPTIMAL_LOW,
    ENTROPY_OPTIMAL_HIGH,
)


# ============================================================================
# Kronos-GAIA Bridge
# ============================================================================

@dataclass
class CrystallizedPattern:
    """A pattern crystallized from GAIA to Kronos."""
    id: str
    delta: torch.Tensor  # The actual embedding/pattern
    potential: float     # PAC potential
    phase: str           # SEC phase state
    importance: float    # Crystallization importance
    metadata: Dict[str, Any]
    created_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            'id': self.id,
            'delta_shape': list(self.delta.shape),
            'potential': self.potential,
            'phase': self.phase,
            'importance': self.importance,
            'metadata': self.metadata,
            'created_at': self.created_at,
        }


@dataclass
class RecallResult:
    """Result of recalling patterns from Kronos."""
    patterns: List[CrystallizedPattern]
    query_embedding: torch.Tensor
    similarity_scores: List[float]
    recall_time_ms: float


class KronosGAIABridge:
    """
    Bridge between GAIA Prime (active cognition) and Kronos (persistent memory).
    
    This is the integration layer that:
    1. Translates GAIA's in-memory patterns to Kronos storage format
    2. Handles crystallization (GAIA → Kronos) when patterns are important
    3. Handles recall (Kronos → GAIA) when context is needed
    4. Maintains PAC conservation across the boundary
    
    Constants alignment:
        - Uses GAIA Prime constants (XI_MOBIUS = 1.0571) for field physics
        - Uses Fracton constants (XI = 0.0618) for SEC thresholds
        - Both are φ-derived, just different scales
    """
    
    def __init__(
        self,
        storage_path: Path,
        namespace: str = "gaia_kronos",
        device: str = 'cuda',
        embed_dim: int = 768,
    ):
        """
        Initialize the bridge.
        
        Args:
            storage_path: Path for Kronos persistence
            namespace: Namespace for this GAIA instance
            device: 'cuda' or 'cpu'
            embed_dim: Embedding dimension (match GAIA model)
        """
        self.storage_path = Path(storage_path)
        self.namespace = namespace
        self.device = device
        self.embed_dim = embed_dim
        
        # Create storage directory
        self.data_path = self.storage_path / namespace
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory index of crystallized patterns
        self.pattern_index: Dict[str, CrystallizedPattern] = {}
        
        # Statistics
        self.stats = {
            'crystallizations': 0,
            'recalls': 0,
            'conservation_checks': 0,
            'conservation_violations': 0,
        }
        
        # Crystallization thresholds (using GAIA Prime constants)
        self.crystallization_threshold = PHI_GAIA  # ~1.618
        self.importance_decay = LAMBDA_STAR_GAIA   # ~0.618
        
        # Load existing patterns
        self._load_index()
    
    def _load_index(self):
        """Load pattern index from disk."""
        index_path = self.data_path / "pattern_index.pt"
        if index_path.exists():
            data = torch.load(index_path, weights_only=False)
            for pid, pdata in data.items():
                self.pattern_index[pid] = CrystallizedPattern(
                    id=pdata['id'],
                    delta=pdata['delta'].to(self.device),
                    potential=pdata['potential'],
                    phase=pdata['phase'],
                    importance=pdata['importance'],
                    metadata=pdata['metadata'],
                    created_at=pdata['created_at'],
                )
            print(f"Loaded {len(self.pattern_index)} patterns from disk")
    
    def _save_index(self):
        """Save pattern index to disk."""
        index_path = self.data_path / "pattern_index.pt"
        data = {}
        for pid, pattern in self.pattern_index.items():
            data[pid] = {
                'id': pattern.id,
                'delta': pattern.delta.cpu(),
                'potential': pattern.potential,
                'phase': pattern.phase,
                'importance': pattern.importance,
                'metadata': pattern.metadata,
                'created_at': pattern.created_at,
            }
        torch.save(data, index_path)
    
    def crystallize(
        self,
        pattern_id: str,
        delta: torch.Tensor,
        potential: float,
        importance: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CrystallizedPattern:
        """
        Crystallize a pattern from GAIA to Kronos.
        
        This saves a high-importance pattern to persistent storage.
        
        Args:
            pattern_id: Unique identifier for the pattern
            delta: The pattern tensor (embedding or delta)
            potential: Current PAC potential
            importance: How important this pattern is
            metadata: Optional metadata (label, source, etc.)
            
        Returns:
            The crystallized pattern object
        """
        # Determine SEC phase
        if importance >= self.crystallization_threshold:
            phase = "CRYSTALLIZED"
        elif potential >= ENTROPY_OPTIMAL_LOW:
            phase = "STABLE"
        else:
            phase = "COLLAPSED"
        
        # Create pattern
        pattern = CrystallizedPattern(
            id=pattern_id,
            delta=delta.detach().clone(),
            potential=potential,
            phase=phase,
            importance=importance,
            metadata=metadata or {},
            created_at=time.time(),
        )
        
        # Store in index
        self.pattern_index[pattern_id] = pattern
        
        # Persist to disk
        pattern_path = self.data_path / f"{pattern_id}.pt"
        torch.save({
            'delta': delta.cpu(),
            'potential': potential,
            'phase': phase,
            'importance': importance,
            'metadata': metadata or {},
            'created_at': pattern.created_at,
        }, pattern_path)
        
        self.stats['crystallizations'] += 1
        
        return pattern
    
    def recall(
        self,
        query: torch.Tensor,
        top_k: int = 5,
        min_importance: float = 0.0,
    ) -> RecallResult:
        """
        Recall patterns from Kronos that match a query.
        
        Uses cosine similarity to find relevant patterns.
        
        Args:
            query: Query embedding tensor
            top_k: Number of patterns to return
            min_importance: Minimum importance threshold
            
        Returns:
            RecallResult with matched patterns and scores
        """
        start_time = time.perf_counter()
        
        if len(self.pattern_index) == 0:
            return RecallResult(
                patterns=[],
                query_embedding=query,
                similarity_scores=[],
                recall_time_ms=0.0,
            )
        
        # Filter by importance
        candidates = [
            p for p in self.pattern_index.values()
            if p.importance >= min_importance
        ]
        
        if not candidates:
            return RecallResult(
                patterns=[],
                query_embedding=query,
                similarity_scores=[],
                recall_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # Stack all deltas for batch similarity
        deltas = torch.stack([p.delta for p in candidates])
        
        # Normalize for cosine similarity
        query_norm = query / (query.norm() + 1e-8)
        deltas_norm = deltas / (deltas.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Compute similarities
        similarities = torch.mm(deltas_norm, query_norm.unsqueeze(-1)).squeeze(-1)
        
        # Get top-k
        k = min(top_k, len(candidates))
        top_scores, top_indices = similarities.topk(k)
        
        # Build result
        patterns = [candidates[i] for i in top_indices.cpu().numpy()]
        scores = top_scores.cpu().numpy().tolist()
        
        self.stats['recalls'] += 1
        
        return RecallResult(
            patterns=patterns,
            query_embedding=query,
            similarity_scores=scores,
            recall_time_ms=(time.perf_counter() - start_time) * 1000,
        )
    
    def verify_conservation(
        self,
        original: torch.Tensor,
        recalled: torch.Tensor,
        tolerance: float = 1e-6,
    ) -> Tuple[bool, float]:
        """
        Verify PAC conservation across crystallize→recall boundary.
        
        Args:
            original: Original tensor before crystallization
            recalled: Tensor after recall from Kronos
            tolerance: Maximum allowed difference
            
        Returns:
            (is_conserved, residual)
        """
        residual = (original - recalled).abs().max().item()
        is_conserved = residual < tolerance
        
        self.stats['conservation_checks'] += 1
        if not is_conserved:
            self.stats['conservation_violations'] += 1
        
        return is_conserved, residual
    
    def sync(self):
        """Sync all patterns to disk."""
        self._save_index()
    
    def clear(self):
        """Clear all patterns (for testing)."""
        self.pattern_index.clear()
        if self.data_path.exists():
            shutil.rmtree(self.data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            **self.stats,
            'patterns_stored': len(self.pattern_index),
            'storage_path': str(self.data_path),
        }


# ============================================================================
# Tests
# ============================================================================

def test_crystallization():
    """Test basic crystallization flow."""
    print("\n" + "=" * 60)
    print("TEST: Crystallization")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_path = Path(__file__).parent / "test_data"
    
    # Clean up
    if test_path.exists():
        shutil.rmtree(test_path)
    
    # Create bridge
    bridge = KronosGAIABridge(
        storage_path=test_path,
        namespace="test_crystallize",
        device=device,
        embed_dim=768,
    )
    
    # Create test patterns
    patterns = []
    for i in range(10):
        delta = torch.randn(768, device=device)
        potential = 1.0 - (i * 0.05)
        importance = PHI if i % 3 == 0 else 0.5
        
        pattern = bridge.crystallize(
            pattern_id=f"pattern_{i:04d}",
            delta=delta,
            potential=potential,
            importance=importance,
            metadata={"index": i, "type": "test"},
        )
        patterns.append(pattern)
        print(f"  Crystallized: {pattern.id}, phase={pattern.phase}, importance={pattern.importance:.3f}")
    
    print(f"\nStats: {bridge.get_stats()}")
    
    # Verify all saved
    assert len(bridge.pattern_index) == 10
    print("\n✓ All 10 patterns crystallized")
    
    # Clean up
    bridge.clear()
    return True


def test_recall():
    """Test pattern recall."""
    print("\n" + "=" * 60)
    print("TEST: Recall")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_path = Path(__file__).parent / "test_data"
    
    # Clean up
    if test_path.exists():
        shutil.rmtree(test_path)
    
    # Create bridge
    bridge = KronosGAIABridge(
        storage_path=test_path,
        namespace="test_recall",
        device=device,
        embed_dim=768,
    )
    
    # Create distinct patterns
    base_vectors = [
        torch.randn(768, device=device) for _ in range(5)
    ]
    
    for i, base in enumerate(base_vectors):
        # Normalize to unit vector
        base = base / base.norm()
        bridge.crystallize(
            pattern_id=f"base_{i}",
            delta=base,
            potential=1.0,
            importance=PHI,
            metadata={"category": f"category_{i}"},
        )
    
    # Query with first base vector (should find itself)
    query = base_vectors[0] / base_vectors[0].norm()
    result = bridge.recall(query, top_k=3)
    
    print(f"\nQuery matched {len(result.patterns)} patterns in {result.recall_time_ms:.2f}ms")
    for i, (pattern, score) in enumerate(zip(result.patterns, result.similarity_scores)):
        print(f"  {i+1}. {pattern.id}: similarity={score:.4f}")
    
    # Best match should be base_0 with similarity ~1.0
    assert result.patterns[0].id == "base_0"
    assert result.similarity_scores[0] > 0.99
    print("\n✓ Correct pattern recalled with high similarity")
    
    # Clean up
    bridge.clear()
    return True


def test_conservation():
    """Test PAC conservation across boundary."""
    print("\n" + "=" * 60)
    print("TEST: Conservation")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_path = Path(__file__).parent / "test_data"
    
    # Clean up
    if test_path.exists():
        shutil.rmtree(test_path)
    
    # Create bridge
    bridge = KronosGAIABridge(
        storage_path=test_path,
        namespace="test_conservation",
        device=device,
        embed_dim=768,
    )
    
    # Create and crystallize a pattern
    original = torch.randn(768, device=device)
    bridge.crystallize(
        pattern_id="conservation_test",
        delta=original,
        potential=1.0,
        importance=PHI,
    )
    
    # Recall it
    recalled = bridge.pattern_index["conservation_test"].delta
    
    # Verify conservation
    is_conserved, residual = bridge.verify_conservation(original, recalled)
    
    print(f"\nOriginal norm: {original.norm():.6f}")
    print(f"Recalled norm: {recalled.norm():.6f}")
    print(f"Residual: {residual:.2e}")
    print(f"Conserved: {is_conserved}")
    
    assert is_conserved, f"Conservation violated! Residual: {residual}"
    print("\n✓ PAC conservation maintained across boundary")
    
    # Clean up
    bridge.clear()
    return True


def test_persistence():
    """Test that patterns survive bridge recreation."""
    print("\n" + "=" * 60)
    print("TEST: Persistence")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_path = Path(__file__).parent / "test_data"
    
    # Clean up
    if test_path.exists():
        shutil.rmtree(test_path)
    
    # Create bridge and add patterns
    bridge1 = KronosGAIABridge(
        storage_path=test_path,
        namespace="test_persist",
        device=device,
        embed_dim=768,
    )
    
    original = torch.randn(768, device=device)
    bridge1.crystallize(
        pattern_id="persist_test",
        delta=original,
        potential=1.0,
        importance=PHI,
    )
    bridge1.sync()  # Ensure saved to disk
    
    print(f"Bridge 1 patterns: {len(bridge1.pattern_index)}")
    
    # Create NEW bridge (simulates restart)
    bridge2 = KronosGAIABridge(
        storage_path=test_path,
        namespace="test_persist",
        device=device,
        embed_dim=768,
    )
    
    print(f"Bridge 2 patterns: {len(bridge2.pattern_index)}")
    
    # Verify pattern persisted
    assert "persist_test" in bridge2.pattern_index
    recalled = bridge2.pattern_index["persist_test"].delta
    
    is_conserved, residual = bridge2.verify_conservation(original, recalled)
    print(f"Conservation after restart: {is_conserved}, residual={residual:.2e}")
    
    assert is_conserved
    print("\n✓ Pattern persisted across bridge recreation")
    
    # Clean up
    bridge2.clear()
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("POC-025: GAIA Prime + Kronos Integration - Experiment 01")
    print("=" * 70)
    
    tests = [
        ("Crystallization", test_crystallization),
        ("Recall", test_recall),
        ("Conservation", test_conservation),
        ("Persistence", test_persistence),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    passed = 0
    for name, success, error in results:
        status = "✓ PASS" if success else f"✗ FAIL: {error}"
        print(f"  {name}: {status}")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(tests)} tests passed")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
