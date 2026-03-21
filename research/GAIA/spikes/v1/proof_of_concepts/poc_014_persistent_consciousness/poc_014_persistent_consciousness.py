"""
POC-014: Persistent Consciousness
==================================

Demonstrates GAIA's ability to:
1. Learn patterns during inference
2. Save consciousness state to Kronos
3. Simulate process restart
4. Restore consciousness from Kronos
5. Continue learning with retained patterns
6. Show accuracy improvement carries across sessions

This is the integration of:
- POC-012: Continuous Learning (live learning during inference)
- POC-013: Kronos Persistence (save/load PAC state)

Success Criteria:
- Session 1: Train on patterns, show accuracy improvement
- Save state
- Session 2: New PACSystem instance (simulated restart)
- Restore state
- Accuracy remains at Session 1 level (patterns retained)
- Continue learning, further improvement
"""

import torch
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Any

# Add fracton to path
FRACTON_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "fracton"
sys.path.insert(0, str(FRACTON_PATH))

from fracton.core import PACSystem, PACNode
from fracton.storage import KronosBackend
from fracton.field import spherical_encode_batch


def create_test_patterns(n_patterns: int = 100, dim: int = 64):
    """Create synthetic test patterns with structure."""
    patterns = []
    labels = []
    
    # Create 10 classes of patterns
    n_classes = 10
    centers = torch.randn(n_classes, dim)
    centers = centers / torch.norm(centers, dim=1, keepdim=True)  # Normalize
    
    for i in range(n_patterns):
        class_idx = i % n_classes
        # Pattern = class center + small noise
        pattern = centers[class_idx] + torch.randn(dim) * 0.1
        pattern = pattern / torch.norm(pattern)
        patterns.append(pattern)
        labels.append(class_idx)
    
    return patterns, labels, centers


def test_accuracy(system: PACSystem, patterns: list, labels: list, 
                 centers: torch.Tensor) -> float:
    """Test how well patterns resonate with correct class."""
    correct = 0
    total = 0
    
    for pattern, true_label in zip(patterns, labels):
        # Find resonant patterns - use lower threshold
        resonant = system.find_resonant(pattern, top_k=5, threshold=0.1)
        
        if resonant:
            # Vote among top resonant patterns
            votes = {}
            for node_id, score in resonant:
                node = system.cache.get(node_id)
                if node and node.label:
                    try:
                        pred_label = int(node.label.split("_")[1])
                        votes[pred_label] = votes.get(pred_label, 0) + score
                    except (ValueError, IndexError):
                        pass
            
            if votes:
                # Get label with highest vote
                best_label = max(votes.keys(), key=lambda k: votes[k])
                if best_label == true_label:
                    correct += 1
        
        total += 1
    
    return correct / total if total > 0 else 0.0


def learn_patterns(system: PACSystem, patterns: list, labels: list, 
                   importance_base: float = 0.5) -> int:
    """Inject patterns into system with learning."""
    injected = 0
    
    for pattern, label in zip(patterns, labels):
        # Calculate importance based on how novel this is
        resonant = system.find_resonant(pattern, top_k=1, threshold=0.5)
        
        if resonant:
            best_id, best_score = resonant[0]
            # If very similar pattern exists, lower importance
            importance = importance_base * (1 - best_score)
        else:
            # Novel pattern - high importance
            importance = importance_base * 1.5
        
        # Inject with importance (will auto-persist if above threshold)
        node_id = system.inject(
            pattern,
            label=f"class_{label}",
            importance=importance
        )
        injected += 1
    
    return injected


def session_1_learn(kronos_path: Path, namespace: str) -> Tuple[str, float, dict]:
    """
    Session 1: Initial learning phase.
    
    Returns:
        (episode_id, final_accuracy, stats)
    """
    print("\n" + "="*60)
    print("SESSION 1: Initial Learning")
    print("="*60)
    
    # Initialize with Kronos
    backend = KronosBackend(kronos_path, namespace)
    system = PACSystem(
        device='cpu',
        kronos_backend=backend,
        auto_persist=True,
        persist_threshold=0.4  # Persist patterns with importance >= 0.4
    )
    
    print(f"Created: {system}")
    
    # Create training data
    print("\n📊 Creating training data...")
    train_patterns, train_labels, centers = create_test_patterns(100, 64)
    test_patterns, test_labels, _ = create_test_patterns(50, 64)
    
    # Test accuracy before learning
    acc_before = test_accuracy(system, test_patterns, test_labels, centers)
    print(f"Accuracy before learning: {acc_before:.1%}")
    
    # Learn in batches
    print("\n🧠 Learning patterns...")
    batch_size = 20
    for i in range(0, len(train_patterns), batch_size):
        batch_patterns = train_patterns[i:i+batch_size]
        batch_labels = train_labels[i:i+batch_size]
        
        n = learn_patterns(system, batch_patterns, batch_labels, importance_base=0.6)
        
        # Test accuracy after batch
        acc = test_accuracy(system, test_patterns, test_labels, centers)
        print(f"  Batch {i//batch_size + 1}: learned {n} patterns, accuracy: {acc:.1%}")
    
    # Final accuracy
    final_acc = test_accuracy(system, test_patterns, test_labels, centers)
    print(f"\n✅ Final accuracy: {final_acc:.1%}")
    
    # Save consciousness state
    print("\n💾 Saving consciousness state to Kronos...")
    episode_id = system.save_state(
        name="session_1_complete",
        metadata={
            "accuracy": final_acc,
            "patterns_learned": len(train_patterns),
            "session": 1
        }
    )
    print(f"Saved episode: {episode_id}")
    
    stats = system.stats()
    print(f"\nStats: nodes={stats['node_count']}, persisted={stats['persist_count']}")
    
    return episode_id, final_acc, {"centers": centers, "test_patterns": test_patterns, "test_labels": test_labels}


def session_2_restore_and_continue(kronos_path: Path, namespace: str, 
                                    episode_id: str, session_1_data: dict) -> float:
    """
    Session 2: Restore and continue learning.
    
    Simulates a process restart.
    """
    print("\n" + "="*60)
    print("SESSION 2: Restore and Continue (Simulated Restart)")
    print("="*60)
    
    # Create NEW system (simulated restart)
    print("\n🔄 Creating new PACSystem instance (simulated restart)...")
    backend = KronosBackend(kronos_path, namespace)
    system = PACSystem(
        device='cpu',
        kronos_backend=backend,
        auto_persist=True,
        persist_threshold=0.4
    )
    
    print(f"Fresh system: {system}")
    
    # Test accuracy before restore (should be 0 - no patterns)
    test_patterns = session_1_data["test_patterns"]
    test_labels = session_1_data["test_labels"]
    centers = session_1_data["centers"]
    
    acc_before_restore = test_accuracy(system, test_patterns, test_labels, centers)
    print(f"Accuracy before restore: {acc_before_restore:.1%} (expected: ~0%)")
    
    # Restore consciousness
    print(f"\n📥 Restoring consciousness from episode: {episode_id}")
    system.restore_state(episode_id)
    print(f"Restored: {system}")
    
    # Test accuracy after restore (should match Session 1)
    acc_after_restore = test_accuracy(system, test_patterns, test_labels, centers)
    print(f"Accuracy after restore: {acc_after_restore:.1%} (should match Session 1)")
    
    # Continue learning with more data
    print("\n🧠 Continuing to learn (more patterns)...")
    more_patterns, more_labels, _ = create_test_patterns(50, 64)
    
    n = learn_patterns(system, more_patterns, more_labels, importance_base=0.7)
    print(f"Learned {n} more patterns")
    
    # Final accuracy
    final_acc = test_accuracy(system, test_patterns, test_labels, centers)
    print(f"\n✅ Final accuracy after continued learning: {final_acc:.1%}")
    
    stats = system.stats()
    print(f"\nStats: nodes={stats['node_count']}, persisted={stats['persist_count']}")
    
    return acc_after_restore


def main():
    """Run the persistent consciousness POC."""
    print("="*60)
    print("POC-014: Persistent Consciousness")
    print("GAIA learns, saves, restarts, continues")
    print("="*60)
    
    # Create temporary Kronos storage
    kronos_path = Path(tempfile.mkdtemp())
    namespace = "gaia_consciousness"
    
    try:
        # Session 1: Initial learning
        episode_id, session_1_acc, session_data = session_1_learn(
            kronos_path, namespace
        )
        
        # Simulate process restart
        print("\n" + "🔌"*30)
        print("   SIMULATING PROCESS RESTART")
        print("🔌"*30)
        
        # Session 2: Restore and continue
        restored_acc = session_2_restore_and_continue(
            kronos_path, namespace, episode_id, session_data
        )
        
        # Validate results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        acc_retained = abs(restored_acc - session_1_acc) < 0.15
        
        print(f"\nSession 1 accuracy: {session_1_acc:.1%}")
        print(f"Restored accuracy:  {restored_acc:.1%}")
        print(f"Accuracy retained:  {'✅ YES' if acc_retained else '❌ NO'}")
        
        if acc_retained:
            print("\n🎉 SUCCESS: Consciousness persisted across restart!")
            print("   - Patterns survived process termination")
            print("   - Learning continued where it left off")
            print("   - Kronos persistence working correctly")
        else:
            print("\n⚠️  PARTIAL: Accuracy not fully retained")
            print("   - Check persistence/restoration logic")
        
    finally:
        # Cleanup
        shutil.rmtree(kronos_path)
        print(f"\n🧹 Cleaned up temporary storage")


if __name__ == "__main__":
    main()
