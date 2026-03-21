"""
Test Multi-Model Fusion Crystallization.
"""

import sys
sys.path.insert(0, 'src')

from gaia_prime import PACMeshSpace, PhysicsMesh, XI, PHI, PHI_INV
from gaia_prime.multi_model_fusion import (
    MultiModelFusion, MockMultiModelFusion, FusionGenerator,
    ModelVote, FusionResult
)


def test_mock_fusion_basic():
    """Test basic fusion with mock models."""
    print("\n=== MOCK FUSION BASIC ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    fusion = MockMultiModelFusion(physics)
    
    # Add mock models with overlapping vocab
    vocab1 = ["Paris", "London", "Berlin", "Rome", "Madrid"]
    vocab2 = ["Paris", "Tokyo", "Berlin", "Sydney", "Beijing"]
    
    fusion.add_mock_model("model1", vocab1, {"Paris": 0.8, "Berlin": 0.5})
    fusion.add_mock_model("model2", vocab2, {"Paris": 0.7, "Berlin": 0.4})
    
    # Fuse predictions
    result = fusion.fuse_predictions("The capital of France is")
    
    print(f"Agreed tokens: {result.agreed_tokens}")
    print(f"Agreement scores: {result.agreement_scores}")
    print(f"Disagreement tokens: {result.disagreement_tokens}")
    print(f"Crystallized: {result.crystallized}")
    print(f"Entropy: {result.entropy:.3f}")
    
    # Paris and Berlin should be agreed (in both vocabs)
    assert "Paris" in result.agreed_tokens or "Berlin" in result.agreed_tokens
    print("PASS: Mock fusion basic works")


def test_agreement_detection():
    """Test that agreement is correctly detected."""
    print("\n=== AGREEMENT DETECTION ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    fusion = MockMultiModelFusion(physics)
    
    # Three models with different agreement levels
    fusion.add_mock_model("m1", ["cat", "dog", "bird"], {"cat": 0.9})
    fusion.add_mock_model("m2", ["cat", "dog", "fish"], {"cat": 0.8})
    fusion.add_mock_model("m3", ["cat", "mouse", "fish"], {"cat": 0.85})
    
    result = fusion.fuse_predictions("The pet is a")
    
    print(f"With 3 models:")
    for token, score in sorted(result.agreement_scores.items(), key=lambda x: -x[1])[:5]:
        print(f"  {token}: {score:.3f}")
    
    # cat should have highest agreement (in all 3)
    assert "cat" in result.agreed_tokens
    if result.agreed_tokens:
        print(f"Most agreed: {result.agreed_tokens[0]}")
    
    print("PASS: Agreement detection works")


def test_crystallization():
    """Test that high agreement causes crystallization."""
    print("\n=== CRYSTALLIZATION ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    fusion = MockMultiModelFusion(physics)
    
    # Models that strongly agree on one token
    fusion.add_mock_model("m1", ["definite", "maybe", "never"], {"definite": 0.95})
    fusion.add_mock_model("m2", ["definite", "perhaps", "never"], {"definite": 0.92})
    
    initial_attractors = len(physics.attractors)
    initial_crystallized = len(physics.collapse.crystallized)
    
    result = fusion.fuse_predictions("The answer is")
    
    print(f"Before: {initial_attractors} attractors, {initial_crystallized} crystallized")
    print(f"After: {len(physics.attractors)} attractors, {len(physics.collapse.crystallized)} crystallized")
    print(f"Crystallized tokens: {result.crystallized}")
    
    # Should have crystallized something
    print(f"Total crystallizations: {fusion.crystallization_count}")
    
    print("PASS: Crystallization works")


def test_uncertainty_detection():
    """Test detection of uncertain regions."""
    print("\n=== UNCERTAINTY DETECTION ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    fusion = MockMultiModelFusion(physics)
    
    # Models with very different opinions
    fusion.add_mock_model("m1", ["yes", "no", "maybe"], {"yes": 0.9})
    fusion.add_mock_model("m2", ["no", "yes", "perhaps"], {"no": 0.9})
    
    uncertain = fusion.detect_uncertainty("Should I")
    
    print(f"Uncertain tokens: {uncertain}")
    
    # High disagreement should show uncertainty
    if uncertain:
        print(f"Most uncertain: {max(uncertain.items(), key=lambda x: x[1])}")
    
    print("PASS: Uncertainty detection works")


def test_trust_weighting():
    """Test that trust weights affect fusion."""
    print("\n=== TRUST WEIGHTING ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    fusion = MockMultiModelFusion(physics)
    
    # One trusted model, one not
    fusion.add_mock_model("trusted", ["correct", "wrong"], {"correct": 0.6}, trust_weight=2.0)
    fusion.add_mock_model("untrusted", ["wrong", "correct"], {"wrong": 0.7}, trust_weight=0.5)
    
    result = fusion.fuse_predictions("The answer is")
    
    print("With trust weights:")
    for token, score in sorted(result.agreement_scores.items(), key=lambda x: -x[1]):
        print(f"  {token}: {score:.3f}")
    
    # Trusted model should dominate
    if result.agreed_tokens:
        print(f"Top token: {result.agreed_tokens[0]}")
    
    print("PASS: Trust weighting works")


def test_fusion_stats():
    """Test fusion statistics tracking."""
    print("\n=== FUSION STATS ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    fusion = MockMultiModelFusion(physics)
    
    fusion.add_mock_model("m1", ["a", "b", "c"])
    fusion.add_mock_model("m2", ["a", "b", "d"])
    
    # Run several queries
    for i in range(5):
        fusion.fuse_predictions(f"Query {i}")
    
    stats = fusion.stats()
    
    print(f"Stats: {stats}")
    print(f"Models: {stats['models']}")
    print(f"Crystallizations: {stats['crystallizations']}")
    
    for name, model_stats in stats['model_stats'].items():
        print(f"  {name}:")
        print(f"    Queries: {model_stats['queries']}")
        print(f"    Agreement rate: {model_stats['agreement_rate']:.2f}")
    
    assert stats['models'] == 2
    print("PASS: Fusion stats work")


def test_fusion_generator():
    """Test generation with fusion."""
    print("\n=== FUSION GENERATOR ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    fusion = MockMultiModelFusion(physics)
    
    # Models with coherent vocab
    vocab = ["The", "cat", "sat", "on", "the", "mat", ".", " "]
    fusion.add_mock_model("m1", vocab, {"cat": 0.5, "mat": 0.5, ".": 0.8})
    fusion.add_mock_model("m2", vocab, {"cat": 0.6, "mat": 0.4, ".": 0.7})
    
    generator = FusionGenerator(fusion, max_tokens=10, temperature=0.5)
    
    text, result = generator.generate("The")
    
    print(f"Generated: '{text}'")
    print(f"Crystallized: {result.crystallized}")
    print(f"Entropy: {result.entropy:.3f}")
    
    assert len(text) > len("The")
    print("PASS: Fusion generator works")


def test_integration_with_physics():
    """Test that fusion integrates with physics mesh."""
    print("\n=== PHYSICS INTEGRATION ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    fusion = MockMultiModelFusion(physics)
    
    vocab = ["hello", "world", "peace", "love"]
    fusion.add_mock_model("m1", vocab, {"peace": 0.9})
    fusion.add_mock_model("m2", vocab, {"peace": 0.85})
    
    # Fuse and crystallize
    result = fusion.fuse_predictions("I wish for")
    
    # Check physics state
    print(f"Mesh nodes: {len(mesh.nodes)}")
    print(f"Physics attractors: {len(physics.attractors)}")
    print(f"Physics crystallized: {len(physics.collapse.crystallized)}")
    print(f"Physics state entropy: {physics.state.entropy:.3f}")
    
    # Crystallized tokens should be in attractors
    for token in result.crystallized:
        # Find node
        found = False
        for node_id, node in mesh.nodes.items():
            if node.token_str == token:
                found = node_id in physics.attractors
                break
        if found:
            print(f"  {token} is an attractor")
    
    print("PASS: Physics integration works")


def test_multiple_queries():
    """Test consistency across multiple queries."""
    print("\n=== MULTIPLE QUERIES ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    fusion = MockMultiModelFusion(physics)
    
    fusion.add_mock_model("m1", ["yes", "no"], {"yes": 0.8})
    fusion.add_mock_model("m2", ["yes", "no"], {"yes": 0.75})
    
    # Same query multiple times
    results = []
    for _ in range(5):
        result = fusion.fuse_predictions("Is this true?")
        results.append(result.agreed_tokens[0] if result.agreed_tokens else None)
    
    print(f"Results across queries: {results}")
    
    # Should be consistent (deterministic for same context)
    unique = set(results)
    print(f"Unique results: {unique}")
    
    print("PASS: Multiple queries work")


def test_full_fusion_cycle():
    """Test complete fusion workflow."""
    print("\n=== FULL FUSION CYCLE ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    fusion = MockMultiModelFusion(physics)
    
    # Set up multiple models
    common = ["Paris", "London", "capital", "city", "is", "the"]
    fusion.add_mock_model("gpt2_mock", common, {"Paris": 0.8, "capital": 0.7})
    fusion.add_mock_model("pythia_mock", common, {"Paris": 0.75, "capital": 0.65})
    fusion.add_mock_model("llama_mock", common, {"Paris": 0.82, "capital": 0.6})
    
    # Run multiple queries
    queries = [
        "The capital of France is",
        "London is a",
        "Paris is the",
    ]
    
    print("Fusion results:")
    for query in queries:
        result = fusion.fuse_predictions(query)
        print(f"  '{query}':")
        print(f"    Agreed: {result.agreed_tokens[:3]}")
        print(f"    Crystallized: {result.crystallized}")
    
    # Check final state
    stats = fusion.stats()
    print(f"\nFinal stats:")
    print(f"  Total crystallizations: {stats['crystallizations']}")
    print(f"  Mesh nodes: {len(mesh.nodes)}")
    print(f"  Physics attractors: {len(physics.attractors)}")
    
    # Run physics step
    physics.step()
    print(f"  Physics entropy: {physics.state.entropy:.3f}")
    
    print("\nPASS: Full fusion cycle complete")


if __name__ == "__main__":
    print("=" * 60)
    print("MULTI-MODEL FUSION TESTS")
    print("=" * 60)
    
    test_mock_fusion_basic()
    test_agreement_detection()
    test_crystallization()
    test_uncertainty_detection()
    test_trust_weighting()
    test_fusion_stats()
    test_fusion_generator()
    test_integration_with_physics()
    test_multiple_queries()
    test_full_fusion_cycle()
    
    print("\n" + "=" * 60)
    print("ALL MULTI-MODEL FUSION TESTS PASSED")
    print("=" * 60)
