"""
POC-025: GAIA Prime + Kronos Integration
========================================

Experiment 03: Session Management

Tests full session save/restore with GAIA Prime state.

Features:
    1. Save complete GAIA session as Kronos episode
    2. Restore session from episode
    3. Verify learning persists across restart
    4. Benchmark restoration speed

This is the "continuous consciousness" capability - the model
remembers what it learned even after being shut down.
"""

import sys
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\fracton")
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\dawn-models\research\GAIA\src")

import torch
import numpy as np
import time
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Import from previous experiments
from exp_01_bridge import KronosGAIABridge
from exp_02_crystallization import GAIAKronosSystem

# GAIA Prime imports
from gaia_prime.validated_constants import PHI, PHI_INV, XI


# ============================================================================
# Session Manager
# ============================================================================

@dataclass
class SessionMetadata:
    """Metadata for a saved session."""
    session_id: str
    created_at: str
    gaia_version: str
    patterns_count: int
    transitions_count: int
    learning_steps: int
    description: str


class GAIASessionManager:
    """
    Manages GAIA Prime sessions with Kronos persistence.
    
    Sessions capture:
    1. All crystallized patterns
    2. Transition statistics
    3. Learning progress
    4. Configuration
    
    This enables:
    - Checkpointing during training
    - Resuming from specific point
    - Sharing trained states
    - Continuous consciousness
    """
    
    def __init__(
        self,
        storage_path: Path,
        device: str = 'cuda',
        embed_dim: int = 768,
    ):
        """
        Initialize session manager.
        
        Args:
            storage_path: Path for session storage
            device: 'cuda' or 'cpu'
            embed_dim: Embedding dimension
        """
        self.storage_path = Path(storage_path)
        self.device = device
        self.embed_dim = embed_dim
        
        # Sessions directory
        self.sessions_path = self.storage_path / "sessions"
        self.sessions_path.mkdir(parents=True, exist_ok=True)
        
        # Current session
        self.current_session_id: Optional[str] = None
        self.system: Optional[GAIAKronosSystem] = None
        
        # Simulated learning state (in real GAIA, this would be TransitionMatrix)
        self.transitions: Dict[str, Dict[str, int]] = {}
        self.learning_steps = 0
    
    def new_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new session.
        
        Args:
            session_id: Optional custom ID (default: timestamp)
            
        Returns:
            The session ID
        """
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.current_session_id = session_id
        session_path = self.sessions_path / session_id
        
        # Create session-specific system
        self.system = GAIAKronosSystem(
            storage_path=session_path,
            namespace="active",
            device=self.device,
            embed_dim=self.embed_dim,
        )
        
        # Reset learning state
        self.transitions = {}
        self.learning_steps = 0
        
        print(f"Created new session: {session_id}")
        return session_id
    
    def learn(self, sequence: List[str]) -> int:
        """
        Learn from a sequence (simplified N-gram learning).
        
        In real GAIA, this would update TransitionMatrix.
        Here we just track N-gram counts for testing.
        
        Args:
            sequence: List of tokens to learn from
            
        Returns:
            Number of transitions learned
        """
        if self.system is None:
            raise RuntimeError("No active session. Call new_session() first.")
        
        transitions_learned = 0
        for i in range(len(sequence) - 1):
            prev = sequence[i]
            next_tok = sequence[i + 1]
            
            if prev not in self.transitions:
                self.transitions[prev] = {}
            
            self.transitions[prev][next_tok] = self.transitions[prev].get(next_tok, 0) + 1
            transitions_learned += 1
            self.learning_steps += 1
        
        return transitions_learned
    
    def inject_embedding(
        self,
        token: str,
        importance: float = 1.0,
    ) -> bool:
        """
        Inject a token embedding into the session.
        
        Args:
            token: Token string
            importance: How important this token is
            
        Returns:
            True if crystallized
        """
        if self.system is None:
            raise RuntimeError("No active session.")
        
        # Create a deterministic embedding from token (for testing)
        # In real GAIA, this would come from GraftedEmbeddings
        torch.manual_seed(hash(token) % (2**32))
        delta = torch.randn(self.embed_dim, device=self.device)
        
        return self.system.inject_pattern(
            pattern_id=f"token_{token}",
            delta=delta,
            importance=importance,
            metadata={"token": token},
        )
    
    def save_session(self, description: str = "") -> SessionMetadata:
        """
        Save the current session.
        
        Args:
            description: Human-readable description
            
        Returns:
            Session metadata
        """
        if self.system is None or self.current_session_id is None:
            raise RuntimeError("No active session.")
        
        session_path = self.sessions_path / self.current_session_id
        
        # Sync Kronos
        self.system.sync()
        
        # Save transitions
        transitions_path = session_path / "transitions.json"
        with open(transitions_path, 'w') as f:
            json.dump(self.transitions, f)
        
        # Save metadata
        metadata = SessionMetadata(
            session_id=self.current_session_id,
            created_at=datetime.now().isoformat(),
            gaia_version="2.0.0",
            patterns_count=len(self.system.kronos.pattern_index),
            transitions_count=sum(
                sum(counts.values()) 
                for counts in self.transitions.values()
            ),
            learning_steps=self.learning_steps,
            description=description,
        )
        
        metadata_path = session_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        print(f"Saved session {self.current_session_id}:")
        print(f"  Patterns: {metadata.patterns_count}")
        print(f"  Transitions: {metadata.transitions_count}")
        print(f"  Learning steps: {metadata.learning_steps}")
        
        return metadata
    
    def load_session(self, session_id: str) -> SessionMetadata:
        """
        Load a saved session.
        
        Args:
            session_id: ID of session to load
            
        Returns:
            Session metadata
        """
        session_path = self.sessions_path / session_id
        
        if not session_path.exists():
            raise ValueError(f"Session not found: {session_id}")
        
        # Load metadata
        metadata_path = session_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        metadata = SessionMetadata(**metadata_dict)
        
        # Load system (Kronos patterns)
        self.current_session_id = session_id
        self.system = GAIAKronosSystem(
            storage_path=session_path,
            namespace="active",
            device=self.device,
            embed_dim=self.embed_dim,
        )
        
        # Load transitions
        transitions_path = session_path / "transitions.json"
        if transitions_path.exists():
            with open(transitions_path, 'r') as f:
                self.transitions = json.load(f)
        else:
            self.transitions = {}
        
        self.learning_steps = metadata.learning_steps
        
        print(f"Loaded session {session_id}:")
        print(f"  Patterns: {len(self.system.kronos.pattern_index)}")
        print(f"  Transitions: {metadata.transitions_count}")
        
        return metadata
    
    def list_sessions(self) -> List[str]:
        """List all saved sessions."""
        return [
            p.name for p in self.sessions_path.iterdir()
            if p.is_dir() and (p / "metadata.json").exists()
        ]
    
    def predict_next(self, token: str) -> Optional[str]:
        """
        Predict next token (simplified, for testing).
        
        Args:
            token: Current token
            
        Returns:
            Most likely next token or None
        """
        if token not in self.transitions:
            return None
        
        next_counts = self.transitions[token]
        if not next_counts:
            return None
        
        return max(next_counts.items(), key=lambda x: x[1])[0]


# ============================================================================
# Tests
# ============================================================================

def test_session_save_restore():
    """Test saving and restoring a session."""
    print("\n" + "=" * 60)
    print("TEST: Session Save/Restore")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_path = Path(__file__).parent / "test_sessions"
    
    # Clean up
    if test_path.exists():
        shutil.rmtree(test_path)
    
    # Create manager and new session
    manager = GAIASessionManager(
        storage_path=test_path,
        device=device,
    )
    
    session_id = manager.new_session("test_session_001")
    
    # Learn some transitions
    text = "the cat sat on the mat the cat ran fast".split()
    manager.learn(text)
    
    # Inject some embeddings
    for word in set(text):
        manager.inject_embedding(word, importance=PHI)
    
    # Save session
    metadata = manager.save_session("Test learning session")
    
    print(f"\nSaved session metadata:")
    print(f"  Patterns: {metadata.patterns_count}")
    print(f"  Transitions: {metadata.transitions_count}")
    
    # Verify prediction before "restart"
    next_after_the = manager.predict_next("the")
    print(f"\nPrediction before restart: 'the' -> '{next_after_the}'")
    
    # Create NEW manager (simulates restart)
    manager2 = GAIASessionManager(
        storage_path=test_path,
        device=device,
    )
    
    # List available sessions
    sessions = manager2.list_sessions()
    print(f"\nAvailable sessions: {sessions}")
    
    # Load the session
    loaded_metadata = manager2.load_session("test_session_001")
    
    # Verify prediction after "restart"
    next_after_the_2 = manager2.predict_next("the")
    print(f"Prediction after restart: 'the' -> '{next_after_the_2}'")
    
    assert next_after_the == next_after_the_2
    assert loaded_metadata.patterns_count == metadata.patterns_count
    print("\n✓ Session saved and restored correctly")
    
    # Clean up
    shutil.rmtree(test_path)
    return True


def test_learning_persistence():
    """Test that learning persists across sessions."""
    print("\n" + "=" * 60)
    print("TEST: Learning Persistence")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_path = Path(__file__).parent / "test_sessions"
    
    # Clean up
    if test_path.exists():
        shutil.rmtree(test_path)
    
    # Session 1: Initial learning
    manager1 = GAIASessionManager(storage_path=test_path, device=device)
    manager1.new_session("learning_001")
    
    # Learn from corpus
    corpus = [
        "the quick brown fox jumps over the lazy dog".split(),
        "the lazy dog sleeps all day long".split(),
        "the quick fox runs very fast".split(),
    ]
    
    for sentence in corpus:
        manager1.learn(sentence)
    
    steps_1 = manager1.learning_steps
    print(f"Session 1 learning steps: {steps_1}")
    
    manager1.save_session("Initial training")
    
    # Session 2: Continue learning
    manager2 = GAIASessionManager(storage_path=test_path, device=device)
    manager2.load_session("learning_001")
    
    # Verify previous learning loaded
    assert manager2.learning_steps == steps_1
    
    # Continue learning
    more_corpus = [
        "the clever fox outsmarts everyone".split(),
        "the dog chases the cat".split(),
    ]
    
    for sentence in more_corpus:
        manager2.learn(sentence)
    
    steps_2 = manager2.learning_steps
    print(f"Session 2 learning steps: {steps_2}")
    
    assert steps_2 > steps_1
    
    manager2.save_session("Continued training")
    
    # Session 3: Verify accumulated learning
    manager3 = GAIASessionManager(storage_path=test_path, device=device)
    manager3.load_session("learning_001")
    
    steps_3 = manager3.learning_steps
    print(f"Session 3 learning steps: {steps_3}")
    
    assert steps_3 == steps_2
    print("\n✓ Learning accumulates across sessions")
    
    # Clean up
    shutil.rmtree(test_path)
    return True


def test_restoration_speed():
    """Benchmark session restoration speed."""
    print("\n" + "=" * 60)
    print("TEST: Restoration Speed")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_path = Path(__file__).parent / "test_sessions"
    
    # Clean up
    if test_path.exists():
        shutil.rmtree(test_path)
    
    # Create a larger session
    manager = GAIASessionManager(storage_path=test_path, device=device)
    manager.new_session("speed_test")
    
    # Add many patterns
    n_patterns = 100
    for i in range(n_patterns):
        manager.inject_embedding(f"token_{i:04d}", importance=PHI)
    
    # Add many transitions
    tokens = [f"token_{i:04d}" for i in range(n_patterns)]
    for _ in range(10):
        np.random.shuffle(tokens)
        manager.learn(tokens)
    
    manager.save_session("Speed test session")
    
    print(f"Created session with {n_patterns} patterns")
    
    # Benchmark restoration
    times = []
    for trial in range(5):
        manager2 = GAIASessionManager(storage_path=test_path, device=device)
        
        start = time.perf_counter()
        manager2.load_session("speed_test")
        elapsed = (time.perf_counter() - start) * 1000
        
        times.append(elapsed)
        print(f"  Trial {trial+1}: {elapsed:.2f}ms")
    
    avg_time = np.mean(times)
    print(f"\nAverage restoration time: {avg_time:.2f}ms")
    
    # Should be well under 1 second for 100 patterns
    assert avg_time < 1000
    print("✓ Restoration time acceptable")
    
    # Clean up
    shutil.rmtree(test_path)
    return True


def test_pattern_recall_after_restore():
    """Test that recalled patterns are usable after restore."""
    print("\n" + "=" * 60)
    print("TEST: Pattern Recall After Restore")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_path = Path(__file__).parent / "test_sessions"
    
    # Clean up
    if test_path.exists():
        shutil.rmtree(test_path)
    
    # Create session with patterns
    manager1 = GAIASessionManager(storage_path=test_path, device=device)
    manager1.new_session("recall_test")
    
    # Inject patterns
    manager1.inject_embedding("apple", importance=PHI)
    manager1.inject_embedding("banana", importance=PHI)
    manager1.inject_embedding("cherry", importance=PHI)
    
    manager1.save_session("Fruit patterns")
    
    # Get the apple embedding for comparison
    torch.manual_seed(hash("apple") % (2**32))
    expected_apple = torch.randn(768, device=device)
    
    # Restore session
    manager2 = GAIASessionManager(storage_path=test_path, device=device)
    manager2.load_session("recall_test")
    
    # Query for patterns similar to "apple"
    torch.manual_seed(hash("apple") % (2**32))
    query = torch.randn(768, device=device)
    
    similar = manager2.system.query_similar(query, top_k=3)
    print(f"Patterns similar to 'apple': {similar}")
    
    assert "token_apple" in similar
    print("\n✓ Patterns recallable after restore")
    
    # Clean up
    shutil.rmtree(test_path)
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("POC-025: GAIA Prime + Kronos Integration - Experiment 03")
    print("Session Management")
    print("=" * 70)
    
    tests = [
        ("Session Save/Restore", test_session_save_restore),
        ("Learning Persistence", test_learning_persistence),
        ("Restoration Speed", test_restoration_speed),
        ("Pattern Recall After Restore", test_pattern_recall_after_restore),
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
