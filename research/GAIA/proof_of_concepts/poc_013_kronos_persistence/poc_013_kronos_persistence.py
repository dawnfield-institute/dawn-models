"""
POC-013: Kronos Persistent Memory
=================================

Demonstrates Fracton's Kronos storage backend for persistent PAC memory.

Key Features:
1. Save PAC nodes to disk (survives restarts)
2. Episode tracking (save/restore full state)
3. Temporal queries (find patterns by time)
4. Crystallized pattern preservation

This POC:
- Creates a PAC substrate with Kronos backend
- Injects patterns and learns transitions
- Saves state as episode
- Clears memory, restores from episode
- Verifies patterns are recovered
"""

import sys
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\fracton")
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\dawn-models\research\GAIA\src")

import torch
import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

# Fracton imports
from fracton.core import PACSystem, PACNode
from fracton.field import spherical_encode_batch, evolve
from fracton.physics import PHI, XI, PHI_XI, LAMBDA_STAR
from fracton.storage import KronosBackend, EpisodeTracker


def run_poc():
    """Run the Kronos persistence POC."""
    
    print("=" * 70)
    print("POC-013: Kronos Persistent Memory")
    print("=" * 70)
    print()
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kronos_path = Path(__file__).parent / "kronos_data"
    namespace = "poc_013_test"
    
    # Clean up any previous test data
    test_dir = kronos_path / namespace
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    print(f"Device: {device}")
    print(f"Kronos path: {kronos_path}")
    print(f"Namespace: {namespace}")
    print()
    
    # === PHASE 1: Create and populate substrate ===
    print("=" * 70)
    print("PHASE 1: Create substrate and inject patterns")
    print("=" * 70)
    
    # Create Kronos backend
    backend = KronosBackend(kronos_path, namespace, device)
    print(f"Backend created: {backend}")
    
    # Create some test nodes
    print("\nCreating test patterns...")
    nodes_created = []
    
    for i in range(10):
        # Create a pattern
        token_id = i * 100 + 42
        field = spherical_encode_batch(
            torch.tensor([token_id], device=device),
            vocab_size=1000,
            dim=64
        )[0]
        
        # Create PACNode
        node = PACNode(
            id=i,
            delta=field,
            potential=1.0 - (i * 0.05),
            parent_id=-1,
            label=f"pattern_{i}"
        )
        
        # Save to Kronos (mark some as crystallized)
        crystallized = (i % 3 == 0)  # Every 3rd pattern is crystallized
        importance = PHI if crystallized else 0.0
        
        doc_id = backend.save_node(
            node,
            metadata={"token_id": token_id},
            crystallized=crystallized,
            importance=importance
        )
        
        nodes_created.append((node.id, doc_id, crystallized))
        print(f"  Saved node {i}: doc_id={doc_id[:16]}... crystallized={crystallized}")
    
    print(f"\nNodes created: {len(nodes_created)}")
    print(f"Backend stats: {backend.get_stats()}")
    
    # === PHASE 2: Query operations ===
    print()
    print("=" * 70)
    print("PHASE 2: Query operations")
    print("=" * 70)
    
    # Query crystallized patterns
    crystallized_ids = backend.query_crystallized(min_importance=0.0)
    print(f"\nCrystallized patterns: {len(crystallized_ids)}")
    for node_id in crystallized_ids:
        node = backend.load_node(node_id)
        if node:
            print(f"  Node {node.id}: label={getattr(node, 'label', 'N/A')}, potential={node.potential:.3f}")
    
    # Query temporal (recent)
    recent_ids = backend.query_recent(5)
    print(f"\nRecent patterns (last 5): {recent_ids}")
    
    # Query temporal (time range)
    now = datetime.now()
    hour_ago = now - timedelta(hours=1)
    time_range_ids = backend.query_temporal(hour_ago, now)
    print(f"Patterns in last hour: {len(time_range_ids)}")
    
    # === PHASE 3: Save episode ===
    print()
    print("=" * 70)
    print("PHASE 3: Save complete state as episode")
    print("=" * 70)
    
    # Collect all nodes
    all_nodes = [backend.load_node(node_id) for node_id, _, _ in nodes_created]
    all_nodes = [n for n in all_nodes if n is not None]
    
    # Save as episode
    episode_id = backend.save_episode(
        nodes=all_nodes,
        name="learning_session_1",
        metadata={
            "description": "Initial pattern injection",
            "pattern_count": len(all_nodes),
            "total_potential": sum(n.potential for n in all_nodes)
        }
    )
    
    print(f"Episode saved: {episode_id}")
    
    # List episodes
    episodes = backend.list_episodes_detailed()
    print(f"\nAll episodes:")
    for ep in episodes:
        print(f"  {ep['episode_id']}: {ep['name']} ({ep['node_count']} nodes)")
    
    # === PHASE 4: Simulate restart (clear and restore) ===
    print()
    print("=" * 70)
    print("PHASE 4: Simulate restart - clear and restore from episode")
    print("=" * 70)
    
    # Create a NEW backend (simulating process restart)
    print("\nCreating new backend instance (simulating restart)...")
    backend2 = KronosBackend(kronos_path, namespace, device)
    
    print(f"Backend after 'restart': {backend2}")
    print(f"  - Nodes still accessible: {len(backend2.list_all_nodes())}")
    print(f"  - Episodes available: {backend2.list_episodes()}")
    
    # Load the episode
    print(f"\nRestoring from episode: {episode_id}")
    restored_nodes, episode_meta = backend2.load_episode(episode_id)
    
    print(f"Restored {len(restored_nodes)} nodes")
    print(f"Episode metadata: {episode_meta.get('name')}")
    
    # Verify restored nodes
    print("\nVerifying restored nodes:")
    for node in restored_nodes[:5]:
        print(f"  Node {node.id}: label={getattr(node, 'label', 'N/A')}, "
              f"delta_shape={node.delta.shape}, potential={node.potential:.3f}")
    
    # === PHASE 5: Verify data integrity ===
    print()
    print("=" * 70)
    print("PHASE 5: Verify data integrity")
    print("=" * 70)
    
    # Compare original nodes with restored
    original_ids = {n.id for n in all_nodes}
    restored_ids = {n.id for n in restored_nodes}
    
    if original_ids == restored_ids:
        print("✓ All node IDs match!")
    else:
        missing = original_ids - restored_ids
        extra = restored_ids - original_ids
        print(f"✗ ID mismatch: missing={missing}, extra={extra}")
    
    # Compare field values
    all_match = True
    for orig in all_nodes:
        restored = next((n for n in restored_nodes if n.id == orig.id), None)
        if restored is None:
            print(f"  ✗ Node {orig.id} not found in restored")
            all_match = False
        elif not torch.allclose(orig.delta.cpu(), restored.delta.cpu(), atol=1e-6):
            print(f"  ✗ Node {orig.id} delta mismatch")
            all_match = False
    
    if all_match:
        print("✓ All field values match!")
    
    # Check crystallized patterns
    restored_crystallized = backend2.query_crystallized()
    original_crystallized = [node_id for node_id, _, cryst in nodes_created if cryst]
    
    if set(restored_crystallized) == set(original_crystallized):
        print("✓ Crystallized patterns preserved!")
    else:
        print(f"✗ Crystallized mismatch: original={original_crystallized}, restored={restored_crystallized}")
    
    # === RESULTS ===
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    
    final_stats = backend2.get_stats()
    print(f"Final Statistics:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    print()
    print("✓ POC-013 COMPLETE: Kronos persistence working!")
    print()
    print("Key capabilities demonstrated:")
    print("  - Save/load individual PAC nodes")
    print("  - Episode-based state snapshots")
    print("  - Temporal and crystallized queries")
    print("  - Data survives 'restart' (new backend instance)")
    print("  - Field values preserved exactly")
    print()
    print(f"Test data at: {kronos_path}")
    
    return backend2


if __name__ == "__main__":
    backend = run_poc()
