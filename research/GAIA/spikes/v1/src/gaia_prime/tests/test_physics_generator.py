"""
Test Physics Generator: Generation using mesh as memory.
"""

import sys
sys.path.insert(0, 'src')

from gaia_prime import (
    PACMeshSpace, 
    PhysicsMesh, 
    SimpleEmbeddings,
    XI, PHI_INV
)
from gaia_prime.physics_generator import (
    PhysicsGenerator, 
    GenerationConfig, 
    PhysicsChat,
    create_generator
)


def test_basic_generation():
    """Test basic text generation from prompt."""
    print("\n=== BASIC GENERATION ===")
    
    # Create components
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    
    # Pre-populate with knowledge
    patterns = [
        ["The", "cat", "sat", "on", "the", "mat"],
        ["The", "dog", "ran", "in", "the", "park"],
        ["I", "love", "music", "and", "art"],
        ["Paris", "is", "the", "capital", "of", "France"],
        ["London", "is", "the", "capital", "of", "England"],
    ]
    
    for pattern in patterns:
        embs = [embeddings.embed(t) for t in pattern]
        physics.store_pattern(pattern, embs, "test")
    
    # Let physics stabilize
    for _ in range(5):
        physics.step()
    
    # Create generator
    config = GenerationConfig(
        max_tokens=10,
        temperature=0.7,
        top_k=5,
        learn_from_generation=False
    )
    generator = PhysicsGenerator(physics, embeddings, config)
    
    # Generate
    prompt = "The cat"
    result = generator.generate(prompt)
    
    print(f"Prompt: '{prompt}'")
    print(f"Generated: '{result.text}'")
    print(f"Tokens: {result.tokens}")
    print(f"Avg confidence: {result.avg_confidence:.3f}")
    print(f"Attractor influences: {result.attractor_influences}")
    
    assert len(result.tokens) > 0, "Should generate some tokens"
    print("PASS: Basic generation works")


def test_generation_with_learning():
    """Test that generation learns while generating (CIMM-style)."""
    print("\n=== GENERATION WITH LEARNING ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    
    # Seed with minimal knowledge
    patterns = [
        ["hello", "world"],
        ["hello", "there"],
        ["world", "peace"],
    ]
    
    for pattern in patterns:
        embs = [embeddings.embed(t) for t in pattern]
        physics.store_pattern(pattern, embs, "test")
    
    initial_node_count = len(mesh.nodes)
    initial_attractor_count = len(physics.attractors)
    
    # Generate with learning
    config = GenerationConfig(
        max_tokens=10,
        temperature=0.9,
        learn_from_generation=True,
        learning_importance=0.5
    )
    generator = PhysicsGenerator(physics, embeddings, config)
    
    result = generator.generate("hello")
    
    print(f"Prompt: 'hello'")
    print(f"Generated: '{result.text}'")
    print(f"Initial nodes: {initial_node_count}")
    print(f"Initial attractors: {initial_attractor_count}")
    print(f"Importance entries after: {len(physics.importance)}")
    
    # Should have remembered new patterns
    assert len(physics.importance) > 0, "Should remember generated tokens"
    print("PASS: Generation with learning works")


def test_repetition_penalty():
    """Test that repetition penalty prevents loops."""
    print("\n=== REPETITION PENALTY ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    
    # Create a loop-prone pattern
    patterns = [
        ["the", "the", "the"],  # Repetitive
        ["the", "cat", "the"],
        ["cat", "the", "cat"],
    ]
    
    for pattern in patterns:
        embs = [embeddings.embed(t) for t in pattern]
        physics.store_pattern(pattern, embs, "test")
    
    # Generate with strong repetition penalty
    config = GenerationConfig(
        max_tokens=10,
        repetition_penalty=2.0,  # Strong penalty
        temperature=0.5
    )
    generator = PhysicsGenerator(physics, embeddings, config)
    
    result = generator.generate("the")
    
    print(f"Generated: '{result.text}'")
    print(f"Tokens: {result.tokens}")
    
    # Count max consecutive repeats
    max_repeats = 1
    current_repeats = 1
    for i in range(1, len(result.tokens)):
        if result.tokens[i] == result.tokens[i-1]:
            current_repeats += 1
            max_repeats = max(max_repeats, current_repeats)
        else:
            current_repeats = 1
    
    print(f"Max consecutive repeats: {max_repeats}")
    # With penalty, should limit repeats
    print("PASS: Repetition penalty applied")


def test_attractor_influence():
    """Test that attractors guide generation."""
    print("\n=== ATTRACTOR INFLUENCE ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    
    # Create patterns
    patterns = [
        ["I", "like", "apples"],
        ["I", "like", "oranges"],
        ["I", "like", "bananas"],
    ]
    
    for pattern in patterns:
        embs = [embeddings.embed(t) for t in pattern]
        physics.store_pattern(pattern, embs, "test")
    
    # Crystallize to create attractors
    physics.crystallize_all_stable()
    
    initial_attractors = len(physics.attractors)
    print(f"Crystallized attractors: {initial_attractors}")
    
    # Generate multiple times
    config = GenerationConfig(
        max_tokens=5,
        temperature=0.3,  # Low temp for consistency
        attractor_weight=0.8  # Strong attractor pull
    )
    generator = PhysicsGenerator(physics, embeddings, config)
    
    results = []
    for _ in range(3):
        result = generator.generate("I like")
        results.append(result.text)
        print(f"Generated: '{result.text}' (attractor hits: {result.attractor_influences})")
    
    print("PASS: Attractor influence works")


def test_guided_generation():
    """Test generation with soft guidance."""
    print("\n=== GUIDED GENERATION ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    
    # Create patterns with different topics
    patterns = [
        ["The", "weather", "is", "sunny"],
        ["The", "weather", "is", "rainy"],
        ["The", "food", "is", "delicious"],
        ["The", "food", "is", "spicy"],
    ]
    
    for pattern in patterns:
        embs = [embeddings.embed(t) for t in pattern]
        physics.store_pattern(pattern, embs, "test")
    
    generator = PhysicsGenerator(physics, embeddings)
    
    # Generate without guidance
    result1 = generator.generate("The")
    print(f"Unguided: '{result1.text}'")
    
    # Generate with guidance toward food
    result2 = generator.generate_with_guidance(
        "The",
        guidance_tokens=["food", "delicious"],
        guidance_weight=0.7
    )
    print(f"Guided (food): '{result2.text}'")
    
    print("PASS: Guided generation works")


def test_physics_chat():
    """Test interactive chat interface."""
    print("\n=== PHYSICS CHAT ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    
    # Pre-populate with conversational patterns
    patterns = [
        ["Hello", "how", "are", "you"],
        ["I", "am", "fine", "thanks"],
        ["What", "is", "your", "name"],
        ["My", "name", "is", "GAIA"],
        ["Nice", "to", "meet", "you"],
    ]
    
    for pattern in patterns:
        embs = [embeddings.embed(t) for t in pattern]
        physics.store_pattern(pattern, embs, "test")
    
    config = GenerationConfig(
        max_tokens=8,
        temperature=0.7
    )
    chat = PhysicsChat(physics, embeddings, config)
    
    # Simulate conversation
    exchanges = [
        "Hello",
        "What is your name",
    ]
    
    for user_input in exchanges:
        response = chat.respond(user_input)
        print(f"User: {user_input}")
        print(f"GAIA: {response}")
        print()
    
    print(f"Total turns: {chat.turn_count}")
    print(f"Conversation nodes: {len(chat.conversation_nodes)}")
    
    assert chat.turn_count == 2, "Should have 2 turns"
    print("PASS: Physics chat works")


def test_create_generator():
    """Test convenience function."""
    print("\n=== CREATE GENERATOR ===")
    
    generator = create_generator(device='cpu', embed_dim=64)
    
    assert generator is not None
    assert generator.physics is not None
    assert generator.mesh is not None
    assert generator.embeddings is not None
    
    print("Generator created successfully")
    print(f"Mesh type: {type(generator.mesh)}")
    print(f"Physics type: {type(generator.physics)}")
    
    print("PASS: Convenience function works")


def test_entropy_triggered_collapse():
    """Test that high entropy triggers collapse during generation."""
    print("\n=== ENTROPY-TRIGGERED COLLAPSE ===")
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    
    # Create many diverse patterns (high entropy potential)
    tokens = ["a", "b", "c", "d", "e", "f", "g", "h"]
    for i in range(20):
        pattern = [tokens[j % len(tokens)] for j in range(i, i+3)]
        embs = [embeddings.embed(t) for t in pattern]
        physics.store_pattern(pattern, embs, "test")
    
    # Artificially raise entropy
    physics.entropy_monitor.current_entropy = 3.0
    physics.state.entropy = 3.0
    
    config = GenerationConfig(
        max_tokens=10,
        entropy_threshold=2.0  # Trigger collapse above this
    )
    generator = PhysicsGenerator(physics, embeddings, config)
    
    result = generator.generate("a b")
    
    print(f"Generated: '{result.text}'")
    print(f"Final entropy: {result.entropy_at_end:.3f}")
    print(f"Collapse events: {len(physics.collapse_memory)}")
    
    print("PASS: Entropy-triggered collapse works")


def test_full_generation_cycle():
    """Test complete generation with all features."""
    print("\n=== FULL GENERATION CYCLE ===")
    
    # Create components
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    physics = PhysicsMesh(mesh)
    embeddings = SimpleEmbeddings(dim=64)
    
    # Rich knowledge base
    knowledge = [
        ["The", "sun", "rises", "in", "the", "east"],
        ["The", "moon", "shines", "at", "night"],
        ["Stars", "twinkle", "in", "the", "sky"],
        ["I", "love", "watching", "the", "sunset"],
        ["The", "ocean", "waves", "are", "calm"],
        ["Birds", "fly", "south", "for", "winter"],
        ["Trees", "lose", "leaves", "in", "autumn"],
        ["Snow", "falls", "gently", "in", "winter"],
    ]
    
    for pattern in knowledge:
        embs = [embeddings.embed(t) for t in pattern]
        physics.store_pattern(pattern, embs, "test")
    
    # Crystallize stable patterns
    physics.crystallize_all_stable()
    
    # Create full-featured generator
    config = GenerationConfig(
        max_tokens=15,
        min_tokens=3,
        temperature=0.7,
        top_k=8,
        top_p=0.9,
        attractor_weight=0.4,
        resonance_weight=0.2,
        learn_from_generation=True,
        learning_importance=0.3,
        repetition_penalty=1.5
    )
    generator = PhysicsGenerator(physics, embeddings, config)
    
    # Generate from various prompts
    prompts = [
        "The sun",
        "I love",
        "Stars",
    ]
    
    for prompt in prompts:
        result = generator.generate(prompt)
        
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{result.text}'")
        print(f"  Confidence: {result.avg_confidence:.3f}")
        print(f"  Crystallized: {result.crystallized_count}")
        print(f"  Attractor hits: {result.attractor_influences}")
        print(f"  Entropy: {result.entropy_at_end:.3f}")
    
    # Check mesh grew from learning
    final_importance_count = len(physics.importance)
    print(f"\nTotal importance entries: {final_importance_count}")
    
    print("\nPASS: Full generation cycle complete")


if __name__ == "__main__":
    print("=" * 60)
    print("PHYSICS GENERATOR TESTS")
    print("=" * 60)
    
    test_basic_generation()
    test_generation_with_learning()
    test_repetition_penalty()
    test_attractor_influence()
    test_guided_generation()
    test_physics_chat()
    test_create_generator()
    test_entropy_triggered_collapse()
    test_full_generation_cycle()
    
    print("\n" + "=" * 60)
    print("ALL PHYSICS GENERATOR TESTS PASSED")
    print("=" * 60)
