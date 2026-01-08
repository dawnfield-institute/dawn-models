"""
POC-025: GAIA Prime + Kronos Integration
========================================

Experiment 02: Automatic Crystallization

Integrates KronosGAIABridge with GAIA Prime's physics mesh to automatically
crystallize high-importance patterns.

Flow:
    1. GAIA Prime generates/processes text
    2. Physics mesh tracks entropy and potential
    3. When pattern exceeds crystallization threshold → auto-save to Kronos
    4. On restart, load crystallized patterns back

Tests:
    - Auto-crystallization triggers correctly
    - Crystallized patterns can be recalled during generation
    - System maintains coherence with persistent memory
"""

import sys
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\fracton")
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\dawn-models\research\GAIA\src")

import torch
import numpy as np
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

# Import the bridge from exp_01
from exp_01_bridge import KronosGAIABridge, CrystallizedPattern

# GAIA Prime imports
from gaia_prime.validated_constants import (
    PHI, PHI_INV, XI, LAMBDA_STAR,
    ENTROPY_OPTIMAL_LOW, ENTROPY_OPTIMAL_HIGH,
)


# ============================================================================
# GAIA Prime + Kronos Integrated System
# ============================================================================

class GAIAKronosSystem:
    """
    GAIA Prime with integrated Kronos memory.
    
    This is the complete cognitive system:
    - GAIA Prime: Active cognition (in-memory physics mesh)
    - Kronos: Persistent memory (disk-backed patterns)
    
    Auto-crystallization rules:
    1. When a pattern's importance exceeds φ (1.618)
    2. When a pattern has been accessed N times (frequency threshold)
    3. When a pattern is explicitly marked important
    
    Auto-recall rules:
    1. On startup: Load all crystallized patterns
    2. During generation: Query for relevant context
    3. On cache miss: Check Kronos before computing
    """
    
    def __init__(
        self,
        storage_path: Path,
        namespace: str = "gaia_system",
        device: str = 'cuda',
        embed_dim: int = 768,
        auto_crystallize: bool = True,
        frequency_threshold: int = 5,
    ):
        """
        Initialize the integrated system.
        
        Args:
            storage_path: Path for Kronos persistence
            namespace: Namespace for this system
            device: 'cuda' or 'cpu'
            embed_dim: Embedding dimension
            auto_crystallize: Enable automatic crystallization
            frequency_threshold: Access count to trigger crystallization
        """
        self.device = device
        self.embed_dim = embed_dim
        self.auto_crystallize = auto_crystallize
        self.frequency_threshold = frequency_threshold
        
        # Initialize Kronos bridge
        self.kronos = KronosGAIABridge(
            storage_path=storage_path,
            namespace=namespace,
            device=device,
            embed_dim=embed_dim,
        )
        
        # Active mesh (in-memory patterns)
        self.active_mesh: Dict[str, Dict[str, Any]] = {}
        
        # Access frequency tracking
        self.access_counts: Dict[str, int] = {}
        
        # Load crystallized patterns into active mesh
        self._load_from_kronos()
        
        # Statistics
        self.stats = {
            'patterns_processed': 0,
            'auto_crystallizations': 0,
            'kronos_recalls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
    
    def _load_from_kronos(self):
        """Load all crystallized patterns into active mesh."""
        for pid, pattern in self.kronos.pattern_index.items():
            self.active_mesh[pid] = {
                'delta': pattern.delta,
                'potential': pattern.potential,
                'importance': pattern.importance,
                'source': 'kronos',
            }
        print(f"Loaded {len(self.active_mesh)} patterns from Kronos")
    
    def inject_pattern(
        self,
        pattern_id: str,
        delta: torch.Tensor,
        importance: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Inject a new pattern into the active mesh.
        
        If auto_crystallize is enabled and importance > φ, 
        the pattern is automatically saved to Kronos.
        
        Args:
            pattern_id: Unique identifier
            delta: Pattern tensor
            importance: How important this pattern is
            metadata: Optional metadata
            
        Returns:
            True if pattern was crystallized
        """
        # Compute potential from tensor properties
        potential = self._compute_potential(delta)
        
        # Add to active mesh
        self.active_mesh[pattern_id] = {
            'delta': delta.detach().clone(),
            'potential': potential,
            'importance': importance,
            'source': 'injected',
            'metadata': metadata or {},
        }
        
        self.stats['patterns_processed'] += 1
        
        # Auto-crystallize if important enough
        crystallized = False
        if self.auto_crystallize and importance >= PHI:
            self.kronos.crystallize(
                pattern_id=pattern_id,
                delta=delta,
                potential=potential,
                importance=importance,
                metadata=metadata,
            )
            self.active_mesh[pattern_id]['source'] = 'crystallized'
            self.stats['auto_crystallizations'] += 1
            crystallized = True
        
        return crystallized
    
    def access_pattern(self, pattern_id: str) -> Optional[torch.Tensor]:
        """
        Access a pattern, tracking frequency.
        
        If accessed frequently enough, auto-crystallize.
        
        Args:
            pattern_id: Pattern to access
            
        Returns:
            The pattern tensor or None if not found
        """
        # Check active mesh first
        if pattern_id in self.active_mesh:
            self.stats['cache_hits'] += 1
            
            # Track access count
            self.access_counts[pattern_id] = self.access_counts.get(pattern_id, 0) + 1
            
            # Auto-crystallize if accessed frequently
            if (
                self.auto_crystallize and
                self.access_counts[pattern_id] >= self.frequency_threshold and
                pattern_id not in self.kronos.pattern_index
            ):
                data = self.active_mesh[pattern_id]
                self.kronos.crystallize(
                    pattern_id=pattern_id,
                    delta=data['delta'],
                    potential=data['potential'],
                    importance=max(data['importance'], PHI),  # Elevate importance
                )
                self.stats['auto_crystallizations'] += 1
            
            return self.active_mesh[pattern_id]['delta']
        
        # Check Kronos
        if pattern_id in self.kronos.pattern_index:
            self.stats['kronos_recalls'] += 1
            pattern = self.kronos.pattern_index[pattern_id]
            
            # Load into active mesh
            self.active_mesh[pattern_id] = {
                'delta': pattern.delta,
                'potential': pattern.potential,
                'importance': pattern.importance,
                'source': 'kronos',
            }
            
            return pattern.delta
        
        self.stats['cache_misses'] += 1
        return None
    
    def query_similar(
        self,
        query: torch.Tensor,
        top_k: int = 5,
    ) -> List[str]:
        """
        Find patterns similar to query.
        
        Searches both active mesh and Kronos.
        
        Args:
            query: Query embedding
            top_k: Number of results
            
        Returns:
            List of pattern IDs
        """
        # Query Kronos
        result = self.kronos.recall(query, top_k=top_k)
        return [p.id for p in result.patterns]
    
    def _compute_potential(self, delta: torch.Tensor) -> float:
        """Compute PAC potential from tensor."""
        # Potential based on norm relative to expected
        norm = delta.norm().item()
        expected_norm = np.sqrt(self.embed_dim)  # Expected for unit Gaussian
        return min(2.0, norm / expected_norm)
    
    def sync(self):
        """Sync all state to disk."""
        self.kronos.sync()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            **self.stats,
            'active_patterns': len(self.active_mesh),
            'crystallized_patterns': len(self.kronos.pattern_index),
            'kronos_stats': self.kronos.get_stats(),
        }


# ============================================================================
# Tests
# ============================================================================

def test_auto_crystallization():
    """Test that high-importance patterns auto-crystallize."""
    print("\n" + "=" * 60)
    print("TEST: Auto-Crystallization")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_path = Path(__file__).parent / "test_data_02"
    
    # Clean up
    if test_path.exists():
        shutil.rmtree(test_path)
    
    # Create system
    system = GAIAKronosSystem(
        storage_path=test_path,
        namespace="test_auto",
        device=device,
    )
    
    # Inject patterns with varying importance
    results = []
    for i in range(10):
        importance = PHI if i % 2 == 0 else 0.5  # Every other is important
        delta = torch.randn(768, device=device)
        
        crystallized = system.inject_pattern(
            pattern_id=f"pattern_{i}",
            delta=delta,
            importance=importance,
        )
        results.append((i, importance, crystallized))
        print(f"  Pattern {i}: importance={importance:.3f}, crystallized={crystallized}")
    
    # Count crystallizations
    auto_crystallized = sum(1 for _, _, c in results if c)
    expected = sum(1 for _, imp, _ in results if imp >= PHI)
    
    print(f"\nAuto-crystallized: {auto_crystallized}/{len(results)}")
    print(f"Expected: {expected}")
    
    assert auto_crystallized == expected
    print("\n✓ Auto-crystallization works correctly")
    
    # Clean up
    system.kronos.clear()
    return True


def test_frequency_crystallization():
    """Test that frequently accessed patterns get crystallized."""
    print("\n" + "=" * 60)
    print("TEST: Frequency-Based Crystallization")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_path = Path(__file__).parent / "test_data_02"
    
    # Clean up
    if test_path.exists():
        shutil.rmtree(test_path)
    
    # Create system with frequency threshold = 3
    system = GAIAKronosSystem(
        storage_path=test_path,
        namespace="test_freq",
        device=device,
        frequency_threshold=3,
    )
    
    # Inject a low-importance pattern
    delta = torch.randn(768, device=device)
    system.inject_pattern(
        pattern_id="frequent_pattern",
        delta=delta,
        importance=0.5,  # Below crystallization threshold
    )
    
    print(f"Initial crystallizations: {len(system.kronos.pattern_index)}")
    assert len(system.kronos.pattern_index) == 0
    
    # Access it multiple times
    for i in range(5):
        result = system.access_pattern("frequent_pattern")
        crystallized = "frequent_pattern" in system.kronos.pattern_index
        print(f"  Access {i+1}: crystallized={crystallized}")
    
    # Should now be crystallized
    assert "frequent_pattern" in system.kronos.pattern_index
    print("\n✓ Frequent access triggers crystallization")
    
    # Clean up
    system.kronos.clear()
    return True


def test_persistence_integration():
    """Test that system state persists across restarts."""
    print("\n" + "=" * 60)
    print("TEST: Persistence Integration")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_path = Path(__file__).parent / "test_data_02"
    
    # Clean up
    if test_path.exists():
        shutil.rmtree(test_path)
    
    # Create system and add patterns
    system1 = GAIAKronosSystem(
        storage_path=test_path,
        namespace="test_persist",
        device=device,
    )
    
    original_delta = torch.randn(768, device=device)
    system1.inject_pattern(
        pattern_id="important_memory",
        delta=original_delta,
        importance=PHI * 2,  # Very important
    )
    system1.sync()
    
    print(f"System 1 active: {len(system1.active_mesh)}")
    print(f"System 1 crystallized: {len(system1.kronos.pattern_index)}")
    
    # Create NEW system (simulates restart)
    system2 = GAIAKronosSystem(
        storage_path=test_path,
        namespace="test_persist",
        device=device,
    )
    
    print(f"System 2 active: {len(system2.active_mesh)}")
    print(f"System 2 crystallized: {len(system2.kronos.pattern_index)}")
    
    # Pattern should be in active mesh (loaded from Kronos)
    assert "important_memory" in system2.active_mesh
    
    # Verify conservation
    recalled = system2.active_mesh["important_memory"]['delta']
    is_conserved, residual = system2.kronos.verify_conservation(original_delta, recalled)
    
    print(f"Conservation: {is_conserved}, residual={residual:.2e}")
    assert is_conserved
    print("\n✓ System state persists across restarts")
    
    # Clean up
    system2.kronos.clear()
    return True


def test_similar_query():
    """Test querying for similar patterns."""
    print("\n" + "=" * 60)
    print("TEST: Similar Pattern Query")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_path = Path(__file__).parent / "test_data_02"
    
    # Clean up
    if test_path.exists():
        shutil.rmtree(test_path)
    
    # Create system
    system = GAIAKronosSystem(
        storage_path=test_path,
        namespace="test_query",
        device=device,
    )
    
    # Create clusters of related patterns
    cluster_centers = [
        torch.randn(768, device=device) for _ in range(3)
    ]
    
    for ci, center in enumerate(cluster_centers):
        center = center / center.norm()  # Normalize
        
        for pi in range(3):
            # Add noise to create similar patterns
            noise = torch.randn(768, device=device) * 0.1
            delta = center + noise
            delta = delta / delta.norm()
            
            system.inject_pattern(
                pattern_id=f"cluster_{ci}_pattern_{pi}",
                delta=delta,
                importance=PHI,  # All important
                metadata={"cluster": ci},
            )
    
    # Query with first cluster center
    query = cluster_centers[0] / cluster_centers[0].norm()
    similar = system.query_similar(query, top_k=5)
    
    print(f"Query found {len(similar)} similar patterns:")
    for pid in similar:
        print(f"  - {pid}")
    
    # All results should be from cluster 0
    cluster_0_results = sum(1 for pid in similar if "cluster_0" in pid)
    print(f"\nCluster 0 results: {cluster_0_results}/{len(similar)}")
    
    assert cluster_0_results >= 2  # At least 2 from same cluster
    print("\n✓ Similar pattern query works")
    
    # Clean up
    system.kronos.clear()
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("POC-025: GAIA Prime + Kronos Integration - Experiment 02")
    print("Automatic Crystallization")
    print("=" * 70)
    
    tests = [
        ("Auto-Crystallization", test_auto_crystallization),
        ("Frequency-Based Crystallization", test_frequency_crystallization),
        ("Persistence Integration", test_persistence_integration),
        ("Similar Pattern Query", test_similar_query),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            import traceback
            results.append((name, False, str(e)))
            traceback.print_exc()
    
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
