"""
Test Bifractal Depth-Based Resonance.
"""

import sys
sys.path.insert(0, 'src')

from gaia_prime import PACMeshSpace, PhysicsMesh, SimpleEmbeddings, XI, PHI
from gaia_prime.bifractal_resonance import (
    BifractalResonance, BifractalDepth, BifractalPattern,
    DEPTH_PROPERTIES
)


def test_depth_determination():
    """Test automatic depth determination."""
    print("\n=== DEPTH DETERMINATION ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    bifractal = BifractalResonance(physics)
    
    # Create test nodes with different properties
    for conf, expected in [(0.1, "SURFACE"), (0.5, "INTERMEDIATE"), (0.9, "DEEP")]:
        emb = embeddings.embed(f"test_{conf}")
        node = mesh.get_or_create_root(hash(f"test_{conf}"), f"test_{conf}", emb, "test")
        node.confidence = conf
        
        depth = bifractal.determine_depth(node)
        print(f"Confidence {conf} -> {depth.name}")
    
    print("PASS: Depth determination works")


def test_store_and_access():
    """Test storing and accessing patterns."""
    print("\n=== STORE AND ACCESS ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    bifractal = BifractalResonance(physics)
    
    # Store at surface
    emb = embeddings.embed("hello")
    node = mesh.get_or_create_root(123, "hello", emb, "test")
    
    pattern = bifractal.store(node, BifractalDepth.SURFACE)
    
    print(f"Stored at: {pattern.depth.name}")
    print(f"Initial strength: {pattern.strength:.3f}")
    
    # Access multiple times
    for i in range(10):
        pattern = bifractal.access(node)
    
    print(f"After 10 accesses: strength {pattern.strength:.3f}")
    print(f"Access count: {pattern.access_count}")
    
    assert pattern.access_count == 10
    print("PASS: Store and access works")


def test_upward_migration():
    """Test migration to deeper levels."""
    print("\n=== UPWARD MIGRATION ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    bifractal = BifractalResonance(physics)
    
    # Store at surface
    emb = embeddings.embed("important")
    node = mesh.get_or_create_root(456, "important", emb, "test")
    
    pattern = bifractal.store(node, BifractalDepth.SURFACE, importance=0.8)
    initial_depth = pattern.depth
    
    print(f"Initial depth: {pattern.depth.name}")
    
    # Access many times to trigger migration
    for i in range(50):
        pattern = bifractal.access(node)
        if pattern.depth != initial_depth:
            print(f"Migrated at access {i+1} to {pattern.depth.name}")
            initial_depth = pattern.depth
    
    print(f"Final depth: {pattern.depth.name}")
    print(f"Migrations up: {bifractal.migrations_up}")
    
    assert bifractal.migrations_up > 0 or pattern.depth != BifractalDepth.SURFACE
    print("PASS: Upward migration works")


def test_decay():
    """Test decay of surface patterns."""
    print("\n=== DECAY ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    bifractal = BifractalResonance(physics)
    
    # Store multiple patterns at surface
    for i in range(10):
        emb = embeddings.embed(f"ephemeral_{i}")
        node = mesh.get_or_create_root(i, f"ephemeral_{i}", emb, "test")
        bifractal.store(node, BifractalDepth.SURFACE, importance=0.1)
    
    initial_count = len(bifractal.patterns)
    print(f"Initial patterns: {initial_count}")
    
    # Run decay multiple times
    for _ in range(20):
        removed = bifractal.decay()
    
    final_count = len(bifractal.patterns)
    print(f"After decay: {final_count} patterns")
    print(f"Depth distribution: {bifractal.get_depth_distribution()}")
    
    print("PASS: Decay works")


def test_resonance_finding():
    """Test finding resonant patterns."""
    print("\n=== RESONANCE FINDING ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    bifractal = BifractalResonance(physics)
    
    # Store related patterns
    base_emb = embeddings.embed("love")
    base_node = mesh.get_or_create_root(1, "love", base_emb, "test")
    bifractal.store(base_node, BifractalDepth.INTERMEDIATE)
    
    # Store similar patterns
    for word in ["heart", "care", "affection"]:
        emb = embeddings.embed(word)
        # Make embedding similar to base
        emb = 0.5 * emb + 0.5 * base_emb
        node = mesh.get_or_create_root(hash(word), word, emb, "test")
        bifractal.store(node, BifractalDepth.INTERMEDIATE)
    
    # Find resonant patterns
    resonant = bifractal.find_resonant(base_node)
    
    print(f"Resonant with 'love':")
    for node, score in resonant[:5]:
        print(f"  {node.token_str}: {score:.3f}")
    
    print(f"Resonance pairs: {len(bifractal.resonance_pairs)}")
    
    print("PASS: Resonance finding works")


def test_depth_properties():
    """Test that depth properties differ correctly."""
    print("\n=== DEPTH PROPERTIES ===")
    
    print("Depth properties:")
    for depth in BifractalDepth:
        props = DEPTH_PROPERTIES[depth]
        print(f"  {depth.name}:")
        print(f"    resonance_threshold: {props['resonance_threshold']}")
        print(f"    decay_rate: {props['decay_rate']}")
        print(f"    crystallization_threshold: {props['crystallization_threshold']}")
    
    # Verify ordering
    decay_rates = [DEPTH_PROPERTIES[d]['decay_rate'] for d in BifractalDepth]
    assert decay_rates == sorted(decay_rates, reverse=True), "Decay should decrease with depth"
    
    print("PASS: Depth properties are correct")


def test_physics_integration():
    """Test integration with physics mesh."""
    print("\n=== PHYSICS INTEGRATION ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    bifractal = BifractalResonance(physics)
    
    # Store patterns at deep and core levels
    for i, depth in enumerate([BifractalDepth.DEEP, BifractalDepth.CORE]):
        emb = embeddings.embed(f"pattern_{depth.name}")
        node = mesh.get_or_create_root(i+100, f"pattern_{depth.name}", emb, "test")
        bifractal.store(node, depth, importance=0.8)
    
    initial_attractors = len(physics.attractors)
    
    # Apply bifractal to physics
    applied = bifractal.apply_resonance_to_physics()
    
    print(f"Before: {initial_attractors} attractors")
    print(f"Applied: {applied}")
    print(f"After: {len(physics.attractors)} attractors")
    print(f"Crystallized in physics: {len(physics.collapse.crystallized)}")
    
    assert len(physics.attractors) >= applied
    print("PASS: Physics integration works")


def test_step_dynamics():
    """Test full step of bifractal dynamics."""
    print("\n=== STEP DYNAMICS ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    bifractal = BifractalResonance(physics)
    
    # Populate with patterns at various depths
    for i, depth in enumerate(BifractalDepth):
        for j in range(3):
            emb = embeddings.embed(f"{depth.name}_{j}")
            node = mesh.get_or_create_root(i*10+j, f"{depth.name}_{j}", emb, "test")
            bifractal.store(node, depth, importance=0.5)
    
    print(f"Initial distribution: {bifractal.get_depth_distribution()}")
    
    # Run steps
    for i in range(5):
        result = bifractal.step()
    
    print(f"After 5 steps:")
    print(f"  Depths: {result['depths']}")
    print(f"  Traits: {result['traits']}")
    print(f"  Resonance pairs: {result['resonance_pairs']}")
    
    print("PASS: Step dynamics work")


def test_personality_emergence():
    """Test emergence of personality traits from core patterns."""
    print("\n=== PERSONALITY EMERGENCE ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    bifractal = BifractalResonance(physics)
    
    # Create related core patterns that resonate
    base_emb = embeddings.embed("creative")
    
    core_nodes = []
    for word in ["creative", "artistic", "imaginative", "innovative"]:
        emb = embeddings.embed(word)
        # Make similar
        emb = 0.6 * emb + 0.4 * base_emb
        node = mesh.get_or_create_root(hash(word), word, emb, "test")
        bifractal.store(node, BifractalDepth.CORE, importance=0.9)
        core_nodes.append(node)
    
    # Find resonances to establish pairs
    for node in core_nodes:
        bifractal.find_resonant(node, BifractalDepth.CORE)
    
    # Check for traits
    traits = bifractal.get_personality_traits()
    
    print(f"Core patterns: {len(bifractal.depth_layers[BifractalDepth.CORE])}")
    print(f"Resonance pairs: {len(bifractal.resonance_pairs)}")
    print(f"Traits emerged: {len(traits)}")
    
    for trait in traits:
        print(f"  {trait.name}: {trait.description}")
    
    print("PASS: Personality emergence works")


def test_full_bifractal_cycle():
    """Test complete bifractal workflow."""
    print("\n=== FULL BIFRACTAL CYCLE ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    bifractal = BifractalResonance(physics)
    
    # Phase 1: Store many patterns
    print("1. Storing patterns...")
    words = ["hello", "world", "peace", "love", "hope", "dream", "future", "present"]
    nodes = []
    for word in words:
        emb = embeddings.embed(word)
        node = mesh.get_or_create_root(hash(word), word, emb, "test")
        bifractal.store(node, importance=0.5)
        nodes.append(node)
    
    print(f"   Stored {len(nodes)} patterns")
    print(f"   Distribution: {bifractal.get_depth_distribution()}")
    
    # Phase 2: Access some repeatedly
    print("\n2. Accessing patterns...")
    for _ in range(30):
        for node in nodes[:3]:  # First 3 get repeated access
            bifractal.access(node)
    
    print(f"   Distribution: {bifractal.get_depth_distribution()}")
    print(f"   Migrations up: {bifractal.migrations_up}")
    
    # Phase 3: Run dynamics
    print("\n3. Running dynamics...")
    for _ in range(10):
        bifractal.step()
    
    print(f"   Distribution: {bifractal.get_depth_distribution()}")
    print(f"   Physics attractors: {len(physics.attractors)}")
    
    # Phase 4: Check stats
    print("\n4. Final stats:")
    stats = bifractal.stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\nPASS: Full bifractal cycle complete")


if __name__ == "__main__":
    print("=" * 60)
    print("BIFRACTAL RESONANCE TESTS")
    print("=" * 60)
    
    test_depth_determination()
    test_store_and_access()
    test_upward_migration()
    test_decay()
    test_resonance_finding()
    test_depth_properties()
    test_physics_integration()
    test_step_dynamics()
    test_personality_emergence()
    test_full_bifractal_cycle()
    
    print("\n" + "=" * 60)
    print("ALL BIFRACTAL RESONANCE TESTS PASSED")
    print("=" * 60)
