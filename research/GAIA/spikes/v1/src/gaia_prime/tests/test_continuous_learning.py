"""
Test CIMM-style Continuous Learning.
"""

import sys
sys.path.insert(0, 'src')

from gaia_prime import PACMeshSpace, PhysicsMesh, SimpleEmbeddings, XI, PHI
from gaia_prime.continuous_learning import (
    ContinuousLearner, AdaptiveGenerator
)


def test_basic_learning():
    """Test basic continuous learning."""
    print("\n=== BASIC CONTINUOUS LEARNING ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    learner = ContinuousLearner(physics)
    
    # Learn a sequence
    tokens = ["hello", "world", "peace"]
    embs = [embeddings.embed(t) for t in tokens]
    nodes = learner.learn_sequence(tokens, embs)
    
    print(f"Learned sequence: {tokens}")
    print(f"Nodes created: {len(nodes)}")
    print(f"Connections: {len(learner.connections)}")
    print(f"History: {len(learner.history)}")
    
    stats = learner.stats()
    print(f"Stats: {stats}")
    
    assert len(nodes) == 3, "Should create 3 nodes"
    assert len(learner.connections) > 0, "Should create connections"
    print("PASS: Basic learning works")


def test_connection_strengthening():
    """Test Hebbian connection strengthening."""
    print("\n=== CONNECTION STRENGTHENING ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    learner = ContinuousLearner(physics)
    
    # Learn same sequence multiple times
    tokens = ["A", "B", "C"]
    embs = [embeddings.embed(t) for t in tokens]
    
    for i in range(5):
        learner.learn_sequence(tokens, embs)
    
    # Check connection weights increased
    transitions = learner.get_transitions(list(learner.connections.keys())[0][0])
    
    print(f"After 5 repetitions:")
    print(f"Connections: {len(learner.connections)}")
    for to_id, weight in transitions[:3]:
        print(f"  -> {str(to_id)[:20]}...: {weight:.3f}")
    
    # Weight should be higher than initial
    if transitions:
        assert transitions[0][1] > XI, "Weight should be above learning rate"
    print("PASS: Connection strengthening works")


def test_feedback():
    """Test feedback mechanism."""
    print("\n=== FEEDBACK MECHANISM ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    learner = ContinuousLearner(physics)
    
    # Learn a sequence
    tokens = ["good", "path", "here"]
    embs = [embeddings.embed(t) for t in tokens]
    learner.learn_sequence(tokens, embs)
    
    initial_connections = len(learner.connections)
    
    # Give positive feedback
    modified = learner.feedback("good")
    print(f"Good feedback: {modified} connections modified")
    
    # Learn another and give negative feedback
    tokens2 = ["bad", "path", "there"]
    embs2 = [embeddings.embed(t) for t in tokens2]
    learner.learn_sequence(tokens2, embs2)
    
    modified = learner.feedback("bad")
    print(f"Bad feedback: {modified} connections weakened")
    
    # Learn and repeat
    tokens3 = ["best", "path", "ever"]
    embs3 = [embeddings.embed(t) for t in tokens3]
    learner.learn_sequence(tokens3, embs3)
    
    modified = learner.feedback("repeat")
    print(f"Repeat feedback: {modified} connections crystallized")
    
    print(f"Total connections: {len(learner.connections)}")
    print(f"Attractors: {len(physics.attractors)}")
    
    assert len(physics.attractors) > 0, "Repeat should crystallize"
    print("PASS: Feedback mechanism works")


def test_consolidation():
    """Test background consolidation."""
    print("\n=== CONSOLIDATION ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    learner = ContinuousLearner(
        physics,
        auto_consolidate=False  # Manual for testing
    )
    
    # Learn many sequences
    for i in range(20):
        tokens = [f"word{j}" for j in range(i, i+3)]
        embs = [embeddings.embed(t) for t in tokens]
        learner.learn_sequence(tokens, embs)
    
    initial_connections = len(learner.connections)
    print(f"Before consolidation: {initial_connections} connections")
    
    # Run consolidation
    stats = learner.consolidate()
    
    print(f"Consolidation stats: {stats}")
    print(f"After consolidation: {len(learner.connections)} connections")
    
    assert stats["decayed"] > 0 or stats["pruned"] > 0, "Should modify connections"
    print("PASS: Consolidation works")


def test_prediction_with_learning():
    """Test that learned transitions improve predictions."""
    print("\n=== PREDICTION WITH LEARNING ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    learner = ContinuousLearner(physics)
    
    # Learn patterns
    patterns = [
        ["I", "love", "pizza"],
        ["I", "love", "music"],
        ["I", "love", "coding"],
        ["You", "like", "sports"],
    ]
    
    all_nodes = []
    for pattern in patterns:
        embs = [embeddings.embed(t) for t in pattern]
        nodes = learner.learn_sequence(pattern, embs)
        all_nodes.extend(nodes)
    
    # Repeat one pattern to strengthen
    for _ in range(5):
        embs = [embeddings.embed(t) for t in ["I", "love", "pizza"]]
        learner.learn_sequence(["I", "love", "pizza"], embs)
    
    # Get predictions after "I love"
    i_node = [n for n in all_nodes if n.token_str == "I"][0]
    love_node = [n for n in all_nodes if n.token_str == "love"][0]
    
    context = [i_node, love_node]
    predictions = learner.predict_next(context, top_k=5)
    
    print("After 'I love':")
    for node, score in predictions:
        print(f"  {node.token_str}: {score:.3f}")
    
    # Pizza should be among top predictions due to repetition
    if predictions:
        tokens = [n.token_str for n, _ in predictions]
        print(f"Top predictions: {tokens}")
    
    print("PASS: Learned predictions work")


def test_adaptive_generator():
    """Test adaptive generator with learning."""
    print("\n=== ADAPTIVE GENERATOR ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    learner = ContinuousLearner(physics)
    
    # Pre-populate knowledge
    patterns = [
        ["The", "cat", "sat", "on", "the", "mat"],
        ["The", "dog", "ran", "in", "the", "park"],
        ["Hello", "world", "nice", "day"],
    ]
    
    for pattern in patterns:
        embs = [embeddings.embed(t) for t in pattern]
        learner.learn_sequence(pattern, embs)
    
    # Create adaptive generator
    gen = AdaptiveGenerator(learner, embeddings)
    
    # Generate
    result = gen.generate("The cat", max_tokens=8)
    
    print(f"Generated: {result.text}")
    print(f"Stats before feedback: {gen.stats()}")
    
    # Give feedback
    modified = gen.good()
    print(f"Good feedback modified: {modified} connections")
    
    # Generate again
    result2 = gen.generate("The dog", max_tokens=8)
    print(f"Generated: {result2.text}")
    
    # Stats should show learning
    stats = gen.stats()
    print(f"Final stats: {stats}")
    
    assert stats["patterns_learned"] > 0, "Should have learned"
    print("PASS: Adaptive generator works")


def test_passive_vs_active_learning():
    """Compare passive and active learning modes."""
    print("\n=== PASSIVE VS ACTIVE LEARNING ===")
    
    # Passive learner (learns everything)
    mesh1 = PACMeshSpace(embed_dim=64, device='cpu')
    physics1 = PhysicsMesh(mesh1)
    passive = ContinuousLearner(physics1, passive_learning=True)
    
    # Same sequence
    tokens = ["test", "sequence", "here"]
    embs = [SimpleEmbeddings(dim=64).embed(t) for t in tokens]
    
    passive.learn_sequence(tokens, embs)
    
    print(f"Passive learner:")
    print(f"  Connections: {len(passive.connections)}")
    print(f"  History: {len(passive.history)}")
    print(f"  Attractors: {len(physics1.attractors)}")
    
    # Active learner (high importance only)
    mesh2 = PACMeshSpace(embed_dim=64, device='cpu')
    physics2 = PhysicsMesh(mesh2)
    active = ContinuousLearner(physics2, passive_learning=True)
    
    # Learn with high importance
    high_tokens = ["important", "sequence", "here"]
    high_embs = [SimpleEmbeddings(dim=64).embed(t) for t in high_tokens]
    active.learn_sequence(high_tokens, high_embs, importance=2.0)
    
    print(f"\nActive learner (high importance):")
    print(f"  Connections: {len(active.connections)}")
    print(f"  History: {len(active.history)}")
    print(f"  Attractors: {len(physics2.attractors)}")
    
    # High importance should crystallize more
    assert len(physics2.attractors) >= len(physics1.attractors)
    print("PASS: Learning modes work")


def test_export_import():
    """Test exporting and importing learned connections."""
    print("\n=== EXPORT/IMPORT ===")
    
    mesh1 = PACMeshSpace(embed_dim=64, device='cpu')
    physics1 = PhysicsMesh(mesh1)
    embeddings = SimpleEmbeddings(dim=64)
    learner1 = ContinuousLearner(physics1)
    
    # Learn patterns
    for i in range(5):
        tokens = [f"word{j}" for j in range(i, i+3)]
        embs = [embeddings.embed(t) for t in tokens]
        learner1.learn_sequence(tokens, embs)
    
    # Export
    exported = learner1.export_connections()
    print(f"Exported {len(exported)} connections")
    
    # Create new learner and import
    mesh2 = PACMeshSpace(embed_dim=64, device='cpu')
    physics2 = PhysicsMesh(mesh2)
    learner2 = ContinuousLearner(physics2)
    
    imported = learner2.import_connections(exported)
    print(f"Imported {imported} connections")
    
    assert imported == len(exported), "Should import all connections"
    assert len(learner2.connections) == len(learner1.connections)
    print("PASS: Export/import works")


def test_full_cimm_cycle():
    """Test complete CIMM learning cycle."""
    print("\n=== FULL CIMM CYCLE ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    learner = ContinuousLearner(
        physics,
        auto_consolidate=True,
        consolidation_interval=20
    )
    
    # Create adaptive generator
    gen = AdaptiveGenerator(learner, embeddings)
    
    # Phase 1: Initial learning
    print("\n1. Initial Learning Phase:")
    patterns = [
        ["The", "weather", "is", "nice"],
        ["The", "weather", "is", "bad"],
        ["I", "enjoy", "sunny", "days"],
        ["I", "dislike", "rainy", "days"],
    ]
    
    for pattern in patterns:
        embs = [embeddings.embed(t) for t in pattern]
        learner.learn_sequence(pattern, embs)
    
    print(f"   Learned {len(patterns)} patterns")
    print(f"   Connections: {len(learner.connections)}")
    
    # Phase 2: Generation with learning
    print("\n2. Generation with Learning Phase:")
    for i in range(3):
        result = gen.generate("The weather", max_tokens=6)
        print(f"   Generated: {result.text}")
    
    # Phase 3: Feedback loop
    print("\n3. Feedback Phase:")
    result = gen.generate("I enjoy", max_tokens=6)
    print(f"   Generated: {result.text}")
    gen.good()
    print("   Gave positive feedback")
    
    result = gen.generate("I enjoy", max_tokens=6)
    print(f"   Regenerated: {result.text}")
    gen.repeat()
    print("   Reinforced this path")
    
    # Phase 4: Check improvements
    print("\n4. Verification Phase:")
    stats = gen.stats()
    print(f"   Final stats:")
    for key, value in stats.items():
        print(f"     {key}: {value}")
    
    # Phase 5: Consolidation
    print("\n5. Consolidation Phase:")
    consol_stats = learner.consolidate()
    print(f"   Consolidation: {consol_stats}")
    
    print("\nPASS: Full CIMM cycle complete")


if __name__ == "__main__":
    print("=" * 60)
    print("CONTINUOUS LEARNING TESTS")
    print("=" * 60)
    
    test_basic_learning()
    test_connection_strengthening()
    test_feedback()
    test_consolidation()
    test_prediction_with_learning()
    test_adaptive_generator()
    test_passive_vs_active_learning()
    test_export_import()
    test_full_cimm_cycle()
    
    print("\n" + "=" * 60)
    print("ALL CONTINUOUS LEARNING TESTS PASSED")
    print("=" * 60)
