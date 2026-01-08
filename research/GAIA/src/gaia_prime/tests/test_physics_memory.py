"""
Test Physics Memory: Deep Intelligence with PAC Tree as Memory

Demonstrates:
1. Physics layer READING from mesh (query, get_context_memory)
2. Physics layer WRITING to mesh (remember, store_pattern)
3. Attractors influencing new patterns
4. Prediction using memory
5. Learning with attractor reinforcement
"""

import torch
import sys
sys.path.insert(0, '.')

from gaia_prime.pac_mesh import PACMeshSpace
from gaia_prime.physics_mesh import (
    PhysicsMesh, CollapseType, XI, PHI, PHI_INV
)
from gaia_prime.embeddings import SimpleEmbeddings


def test_memory_store_and_query():
    """Test storing patterns and querying memory."""
    print("\n" + "="*60)
    print("MEMORY: STORE AND QUERY")
    print("="*60)
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    embeddings = SimpleEmbeddings(dim=64)
    physics = PhysicsMesh(mesh)
    
    # Store some patterns
    patterns = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "sat", "on", "the", "rug"],
        ["a", "bird", "flew", "over", "the", "house"],
    ]
    
    print("\nStoring patterns...")
    for pattern in patterns:
        embs = [embeddings.embed(t) for t in pattern]
        nodes = physics.store_pattern(pattern, embs, "test", importance=0.7)
        print(f"  Stored: '{' '.join(pattern)}' ({len(nodes)} nodes)")
    
    print(f"\nMesh now has {len(mesh.nodes)} nodes")
    
    # Query by embedding
    print("\nQuerying by embedding...")
    query_emb = embeddings.embed("cat")
    results = physics.query(query_emb, top_k=3)
    print(f"  Query for 'cat' embedding:")
    for node, score in results:
        print(f"    '{node.token_str}': {score:.3f}")
    
    # Query by token
    print("\nQuerying by token...")
    results = physics.query_by_token("sat", top_k=3)
    print(f"  Query for 'sat' token:")
    for node, score in results:
        print(f"    '{node.token_str}': {score:.3f}")
    
    print("\n[OK] Memory store and query working")


def test_context_memory():
    """Test getting memory relevant to context."""
    print("\n" + "="*60)
    print("MEMORY: CONTEXT-AWARE RETRIEVAL")
    print("="*60)
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    embeddings = SimpleEmbeddings(dim=64)
    physics = PhysicsMesh(mesh)
    
    # Store patterns with shared context
    patterns = [
        ["the", "quick", "brown", "fox"],
        ["the", "quick", "red", "car"],
        ["the", "slow", "brown", "bear"],
    ]
    
    for pattern in patterns:
        embs = [embeddings.embed(t) for t in pattern]
        physics.store_pattern(pattern, embs, "test", importance=0.7)
    
    # Run physics to build resonance memory
    for _ in range(5):
        physics.step()
    
    print(f"\nStored {len(patterns)} patterns")
    print(f"Resonance memory size: {len(physics.resonance_memory)}")
    
    # Build context
    context_tokens = ["the", "quick"]
    context_nodes = []
    for token in context_tokens:
        results = physics.query_by_token(token, top_k=1)
        if results:
            context_nodes.append(results[0][0])
    
    print(f"\nContext: {context_tokens}")
    
    # Get context memory
    memory = physics.get_context_memory(context_nodes, depth=3)
    print(f"Context memory ({len(memory)} nodes):")
    for node in memory:
        print(f"  '{node.token_str}' (conf={node.confidence:.2f})")
    
    print("\n[OK] Context memory working")


def test_attractor_influence():
    """Test how crystallized attractors influence new patterns."""
    print("\n" + "="*60)
    print("ATTRACTORS: INFLUENCE ON NEW PATTERNS")
    print("="*60)
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    embeddings = SimpleEmbeddings(dim=64)
    physics = PhysicsMesh(mesh)
    
    # Create a strong attractor pattern
    attractor_tokens = ["Paris", "is", "beautiful"]
    attractor_embs = [embeddings.embed(t) for t in attractor_tokens]
    attractor_nodes = physics.store_pattern(
        attractor_tokens, attractor_embs, "confirmed", importance=0.95
    )
    
    # Manually crystallize
    for node in attractor_nodes:
        node.confidence = 0.99
        for s in ["gpt2", "wiki", "atlas", "user"]:
            node.sources.add(s)
        node.incoming_paths[100] = 5
    
    # Run physics to crystallize
    for _ in range(3):
        physics.step()
    
    physics.crystallize_all_stable()
    
    print(f"Crystallized {len(physics.collapse.crystallized)} nodes")
    print(f"Attractors: {len(physics.attractors)}")
    
    # Now add a similar pattern
    similar_tokens = ["Rome", "is", "beautiful"]
    similar_embs = [embeddings.embed(t) for t in similar_tokens]
    
    print(f"\nAdding similar pattern: '{' '.join(similar_tokens)}'")
    print("Before:")
    
    similar_nodes = physics.store_pattern(
        similar_tokens, similar_embs, "new", importance=0.5
    )
    
    for node in similar_nodes:
        print(f"  '{node.token_str}': conf={node.confidence:.3f}")
    
    # Run physics - attractors should influence
    print("\nAfter physics (attractor influence):")
    for _ in range(5):
        physics.step()
    
    for node in similar_nodes:
        print(f"  '{node.token_str}': conf={node.confidence:.3f}")
    
    print("\n[OK] Attractor influence working")


def test_prediction():
    """Test prediction using memory."""
    print("\n" + "="*60)
    print("PREDICTION: MEMORY-BASED")
    print("="*60)
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    embeddings = SimpleEmbeddings(dim=64)
    physics = PhysicsMesh(mesh)
    
    # Store training patterns
    patterns = [
        ["I", "love", "cats"],
        ["I", "love", "dogs"],
        ["I", "hate", "bugs"],
        ["you", "love", "music"],
    ]
    
    for pattern in patterns:
        embs = [embeddings.embed(t) for t in pattern]
        physics.store_pattern(pattern, embs, "train", importance=0.7)
    
    # Run physics
    for _ in range(5):
        physics.step()
    
    print(f"Trained on {len(patterns)} patterns")
    
    # Build context for prediction
    context_tokens = ["I", "love"]
    context_nodes = []
    
    # Find nodes for context
    for token in context_tokens:
        results = physics.query_by_token(token, top_k=1)
        if results:
            context_nodes.append(results[0][0])
    
    print(f"\nContext: '{' '.join(context_tokens)}'")
    
    # Predict next
    predictions = physics.predict_next(context_nodes, top_k=5)
    print("Predictions:")
    for node, prob in predictions:
        print(f"  '{node.token_str}': {prob*100:.1f}%")
    
    print("\n[OK] Prediction working")


def test_learning_with_attractors():
    """Test learning that uses attractor reinforcement."""
    print("\n" + "="*60)
    print("LEARNING: ATTRACTOR REINFORCEMENT")
    print("="*60)
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    embeddings = SimpleEmbeddings(dim=64)
    physics = PhysicsMesh(mesh)
    
    # Create base knowledge (will become attractor)
    base_patterns = [
        ["capital", "of", "France", "is", "Paris"],
        ["capital", "of", "Germany", "is", "Berlin"],
    ]
    
    print("Creating base knowledge...")
    for pattern in base_patterns:
        embs = [embeddings.embed(t) for t in pattern]
        nodes = physics.store_pattern(pattern, embs, "wiki", importance=0.9)
        # Boost to crystallization
        for node in nodes:
            node.confidence = 0.95
            for s in ["wiki", "atlas", "textbook"]:
                node.sources.add(s)
            node.incoming_paths[hash(pattern[0]) % 1000] = 3
    
    # Run physics to crystallize
    for _ in range(5):
        physics.step()
    
    physics.crystallize_all_stable()
    print(f"Crystallized: {len(physics.collapse.crystallized)} attractors")
    
    # Now learn new pattern that's similar
    new_pattern = ["capital", "of", "Italy", "is", "Rome"]
    new_embs = [embeddings.embed(t) for t in new_pattern]
    
    print(f"\nLearning: '{' '.join(new_pattern)}'")
    print("  (confirmed=True for high importance)")
    
    physics.learn_from_sequence(
        new_pattern, new_embs, "user", confirmed=True
    )
    
    # Check what happened
    print(f"\nAfter learning:")
    print(f"  Total nodes: {len(mesh.nodes)}")
    print(f"  Attractors: {len(physics.attractors)}")
    print(f"  Crystallized: {len(physics.collapse.crystallized)}")
    
    # Query for capitals
    print("\nQuerying 'capital':")
    results = physics.query_by_token("capital", top_k=3)
    for node, score in results:
        crystallized = "FROZEN" if node.node_id in physics.collapse.crystallized else ""
        print(f"  '{node.token_str}': {score:.3f} {crystallized}")
    
    print("\n[OK] Attractor-reinforced learning working")


def test_full_intelligence_cycle():
    """Test complete intelligence cycle with GPT-2."""
    print("\n" + "="*60)
    print("FULL INTELLIGENCE CYCLE")
    print("="*60)
    
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        print("Loading GPT-2...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.eval()
        
        mesh = PACMeshSpace(embed_dim=64, device='cpu')
        embeddings = SimpleEmbeddings(dim=64)
        physics = PhysicsMesh(mesh)
        
        # Phase 1: Learn from GPT-2 predictions
        print("\nPhase 1: Learning from GPT-2...")
        prompts = [
            "The capital of France is",
            "The capital of Germany is",
            "The capital of Spain is",
        ]
        
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=0)
            
            # Get top prediction
            top_prob, top_idx = probs.topk(1)
            pred_token = tokenizer.decode([top_idx.item()])
            
            # Build full sequence
            tokens = tokenizer.tokenize(prompt) + [pred_token]
            embs = [embeddings.embed(t) for t in tokens]
            
            # Learn with importance based on probability
            importance = top_prob.item() * 0.8 + 0.2
            physics.store_pattern(tokens, embs, "gpt2", importance=importance)
            
            print(f"  '{prompt}' -> '{pred_token}' (p={top_prob.item():.3f})")
        
        # Run physics to integrate
        for _ in range(10):
            physics.step()
        
        print(f"\n  Mesh: {len(mesh.nodes)} nodes")
        print(f"  Resonance memory: {len(physics.resonance_memory)} pairs")
        
        # Phase 2: Crystallize strong patterns
        print("\nPhase 2: Crystallizing...")
        
        # Boost 'capital' patterns
        for node in mesh.nodes.values():
            if 'capital' in node.token_str.lower():
                node.confidence = 0.9
                for s in ["gpt2", "confirmed", "geography"]:
                    node.sources.add(s)
                node.incoming_paths[999] = 5
        
        for _ in range(5):
            physics.step()
        
        crystals = physics.crystallize_all_stable()
        print(f"  Crystallized: {crystals} nodes")
        print(f"  Attractors: {len(physics.attractors)}")
        
        # Phase 3: Query memory
        print("\nPhase 3: Querying memory...")
        
        results = physics.query_by_token("Paris", top_k=3)
        print(f"  Query 'Paris':")
        for node, score in results:
            status = "ATTRACTOR" if node.node_id in physics.attractors else ""
            print(f"    '{node.token_str}': {score:.3f} {status}")
        
        # Phase 4: Predict using memory
        print("\nPhase 4: Prediction from memory...")
        
        # Build context
        context = ["The", "capital", "of"]
        context_nodes = []
        for t in context:
            res = physics.query_by_token(t, top_k=1)
            if res:
                context_nodes.append(res[0][0])
        
        if context_nodes:
            predictions = physics.predict_next(context_nodes, top_k=5)
            print(f"  Context: '{' '.join(context)}'")
            print(f"  Predictions:")
            for node, prob in predictions:
                print(f"    '{node.token_str}': {prob*100:.1f}%")
        
        # Final report
        print("\n" + physics.report())
        
        print("\n[OK] Full intelligence cycle complete!")
        
    except ImportError:
        print("[SKIP] transformers not available")


if __name__ == "__main__":
    test_memory_store_and_query()
    test_context_memory()
    test_attractor_influence()
    test_prediction()
    test_learning_with_attractors()
    test_full_intelligence_cycle()
    
    print("\n" + "="*60)
    print("ALL MEMORY TESTS COMPLETE")
    print("="*60)
    print("""
The physics layer now uses the PAC mesh as MEMORY:

1. QUERY OPERATIONS (READ)
   - query(embedding) - find similar patterns
   - query_by_token(token) - find by token string
   - get_context_memory(context) - get relevant continuations

2. STORE OPERATIONS (WRITE)
   - remember(node, importance) - mark as important
   - store_pattern(tokens, embs) - add sequence to mesh
   - learn_from_sequence() - learn with attractor reinforcement

3. ATTRACTOR DYNAMICS
   - Crystallized nodes become attractors
   - Attractors influence nearby patterns
   - Similar patterns drift toward attractors
   - Mutual reinforcement between related patterns

4. PREDICTION
   - predict_next(context) - uses children + attractors + resonance
   - Memory-augmented generation

5. LEARNING
   - New patterns interact with existing attractors
   - Confirmed patterns crystallize faster
   - Attractor proximity boosts confidence
""")
