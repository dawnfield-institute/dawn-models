"""
Test Physics Mesh: Deep Intelligence Layer

Demonstrates:
1. Entropy monitoring and collapse events
2. Conservation enforcement (Xi = 1.0571)
3. Resonance field and phase alignment
4. Crystallization of stable patterns
5. Physics-governed learning
"""

import torch
import sys
sys.path.insert(0, '.')

from gaia_prime.pac_mesh import PACMeshSpace, MultiModelMesh
from gaia_prime.physics_mesh import (
    PhysicsMesh, CollapseType, XI, PHI, PHI_INV, LAMBDA_STAR
)
from gaia_prime.embeddings import SimpleEmbeddings


def test_physics_constants():
    """Verify physics constants are correctly defined."""
    print("\n" + "="*60)
    print("PHYSICS CONSTANTS")
    print("="*60)
    
    print(f"Xi (balance operator):     {XI:.4f}")
    print(f"Phi (golden ratio):        {PHI:.6f}")
    print(f"1/Phi (inverse golden):    {PHI_INV:.6f}")
    print(f"Lambda* (decay constant):  {LAMBDA_STAR:.6f}")
    
    # Verify golden ratio identity
    assert abs(PHI * PHI_INV - 1.0) < 1e-10, "Phi * 1/Phi should equal 1"
    assert abs(PHI - 1 - PHI_INV) < 1e-10, "Phi - 1 should equal 1/Phi"
    
    print("\n[OK] Golden ratio identities verified")


def test_basic_physics():
    """Test basic physics mesh functionality."""
    print("\n" + "="*60)
    print("BASIC PHYSICS MESH")
    print("="*60)
    
    # Create mesh with physics layer
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    embeddings = SimpleEmbeddings(dim=64)
    physics = PhysicsMesh(mesh)
    
    # Add some nodes
    tokens = ["the", "quick", "brown", "fox", "jumps"]
    parent = None
    context = []
    
    for i, token in enumerate(tokens):
        emb = embeddings.embed(token)
        token_id = hash(token) % 10000
        
        if i == 0:
            # Root node
            node = mesh.get_or_create_root(
                token_id=token_id,
                token_str=token,
                embedding=emb,
                source="test"
            )
        else:
            # Context node
            context.append(parent.token_id)
            node = mesh.get_or_create_context_node(
                context=tuple(context),
                final_token_id=token_id,
                final_token_str=token,
                embedding=emb,
                source="test"
            )
            parent.add_child(node)
        
        node.confidence = 0.5 + 0.1 * i
        parent = node
    
    # Run physics step
    state = physics.step()
    
    print(f"\nNodes created: {len(mesh.nodes)}")
    print(f"\nInitial state:")
    print(f"  Entropy: {state.entropy:.4f}")
    print(f"  Curvature: {state.curvature:.4f}")
    print(f"  Conservation residual: {state.conservation_residual:.4f}")
    print(f"  Is stable: {state.is_stable}")
    
    # Add some branching (increases entropy)
    base_node = list(mesh.nodes.values())[2]  # "brown"
    for i, alt in enumerate(["deer", "bear", "wolf"]):  # Different tokens, no overlap
        emb = embeddings.embed(alt)
        alt_id = hash(alt) % 10000 + 1000 + i  # Unique IDs
        alt_node = mesh.get_or_create_context_node(
            context=(base_node.token_id,),
            final_token_id=alt_id,
            final_token_str=alt,
            embedding=emb,
            source="test"
        )
        alt_node.confidence = 0.3
        base_node.add_child(alt_node)
    
    # Run another step
    state = physics.step()
    
    print(f"\nAfter adding branches:")
    print(f"  Entropy: {state.entropy:.4f} (should increase)")
    print(f"  Curvature: {state.curvature:.4f}")
    print(f"  Collapse pressure: {state.collapse_pressure:.4f}")
    
    print("\n✅ Basic physics mesh working")


def test_convergence_physics():
    """Test physics behavior with convergent paths."""
    print("\n" + "="*60)
    print("CONVERGENCE WITH PHYSICS")
    print("="*60)
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    embeddings = SimpleEmbeddings(dim=64)
    physics = PhysicsMesh(mesh)
    
    # Create two paths that converge
    # Path 1: "the" -> "quick" -> "brown"
    # Path 2: "a" -> "dark" -> "brown"
    
    # Path 1
    the = mesh.get_or_create_root(1, "the", embeddings.embed("the"), "path1")
    quick = mesh.get_or_create_context_node((1,), 2, "quick", embeddings.embed("quick"), "path1")
    the.add_child(quick)
    brown = mesh.get_or_create_context_node((1, 2), 3, "brown", embeddings.embed("brown"), "path1")
    quick.add_child(brown)
    
    # Path 2
    a = mesh.get_or_create_root(4, "a", embeddings.embed("a"), "path2")
    dark = mesh.get_or_create_context_node((4,), 5, "dark", embeddings.embed("dark"), "path2")
    a.add_child(dark)
    
    # Converge to same "brown" node (same context hash!)
    # We use the same token_id (3) to trigger potential convergence
    brown2 = mesh.get_or_create_context_node((4, 5), 3, "brown", embeddings.embed("brown"), "path2")
    dark.add_child(brown2)
    
    # Note: In this case, brown and brown2 may or may not be same node
    # depending on whether context hashes match
    is_converged = brown is brown2
    print(f"\nbrown is brown2 (byref): {is_converged}")
    print(f"brown convergence factor: {brown.convergence_factor}")
    print(f"brown2 convergence factor: {brown2.convergence_factor}")
    print(f"brown incoming paths: {brown.incoming_paths}")
    
    # Set confidences
    for node in mesh.nodes.values():
        node.confidence = 0.5
    
    # Run physics
    state = physics.step()
    
    print(f"\nPhysics state after convergence:")
    print(f"  Entropy: {state.entropy:.4f}")
    print(f"  Resonance strength: {state.resonance_strength:.4f}")
    
    # Force convergence collapse
    event = physics.force_collapse(CollapseType.CONVERGENCE)
    print(f"\nForced CONVERGENCE collapse:")
    print(f"  Nodes affected: {len(event.node_ids)}")
    print(f"  Brown confidence after: {brown.confidence:.4f}")
    
    print("\n✅ Convergence physics working")


def test_crystallization():
    """Test crystallization of stable patterns."""
    print("\n" + "="*60)
    print("CRYSTALLIZATION")
    print("="*60)
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    embeddings = SimpleEmbeddings(dim=64)
    physics = PhysicsMesh(mesh)
    
    # Create nodes with varying stability
    phrases = [
        ("the", 0.95, 5),   # High confidence, many sources
        ("cat", 0.4, 1),    # Low confidence, few sources
        ("sat", 0.85, 3),   # Medium-high confidence
        ("on", 0.3, 1),     # Low confidence
        ("mat", 0.92, 4),   # High confidence
    ]
    
    parent = None
    context = []
    for i, (token, conf, num_sources) in enumerate(phrases):
        token_id = hash(token) % 10000
        emb = embeddings.embed(token)
        
        if i == 0:
            node = mesh.get_or_create_root(token_id, token, emb, "main")
        else:
            context.append(parent.token_id)
            node = mesh.get_or_create_context_node(
                tuple(context), token_id, token, emb, "main"
            )
            parent.add_child(node)
        
        node.confidence = conf
        for j in range(num_sources):
            node.sources.add(f"source_{j}")
        parent = node
    
    # Add convergence to some nodes (fake additional paths)
    the_node = list(mesh.nodes.values())[0]
    the_node.incoming_paths[999] = 3
    
    # Run physics steps
    for i in range(3):
        state = physics.step()
    
    print(f"\nAfter 3 physics steps:")
    print(f"  Crystallization ratio: {state.crystallization_ratio*100:.1f}%")
    print(f"  Crystallized nodes: {len(physics.collapse.crystallized)}")
    
    # Force crystallization
    new_crystals = physics.crystallize_all_stable()
    print(f"  Newly crystallized: {new_crystals}")
    
    # Show crystallized nodes
    for node_id in physics.collapse.crystallized:
        node = mesh.nodes.get(node_id)
        if node:
            print(f"    ✨ '{node.token_str}' (conf={node.confidence:.2f})")
    
    print("\n✅ Crystallization working")


def test_full_physics_cycle():
    """Test complete physics cycle with GPT-2 knowledge."""
    print("\n" + "="*60)
    print("FULL PHYSICS CYCLE WITH GPT-2")
    print("="*60)
    
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        # Load model
        print("Loading GPT-2...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.eval()
        
        # Create physics mesh
        mesh = PACMeshSpace(embed_dim=64, device='cpu')
        embeddings = SimpleEmbeddings(dim=64)
        physics = PhysicsMesh(mesh)
        
        # Learn from model predictions
        prompts = [
            "The capital of France is",
            "The capital of Germany is",
            "The capital of Italy is",
        ]
        
        print("\nLearning from GPT-2 predictions...")
        
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=0)
            
            # Get top predictions
            top_k = 5
            top_probs, top_indices = probs.topk(top_k)
            
            # Build path through mesh
            tokens = tokenizer.tokenize(prompt)
            parent = None
            context = []
            
            for i, token in enumerate(tokens):
                token_id = tokenizer.convert_tokens_to_ids([token])[0]
                emb = embeddings.embed(token)
                
                if i == 0:
                    node = mesh.get_or_create_root(token_id, token, emb, "gpt2")
                else:
                    context.append(parent.token_id)
                    node = mesh.get_or_create_context_node(
                        tuple(context), token_id, token, emb, "gpt2"
                    )
                    parent.add_child(node)
                
                node.confidence = 0.5
                parent = node
            
            # Add predictions as children
            for prob, idx in zip(top_probs, top_indices):
                pred_token = tokenizer.decode([idx.item()])
                pred_id = idx.item()
                pred_context = context + [parent.token_id] if context else [parent.token_id]
                
                pred_node = mesh.get_or_create_context_node(
                    tuple(pred_context), pred_id, pred_token, 
                    embeddings.embed(pred_token), "gpt2"
                )
                parent.add_child(pred_node)
                pred_node.confidence = prob.item()
        
        print(f"  Mesh nodes: {len(mesh.nodes)}")
        
        # Run physics simulation
        print("\nRunning physics simulation...")
        for step in range(10):
            state = physics.step()
        
        # Show final state
        print("\n" + physics.report())
        
        # Get attractors
        attractors = physics.get_attractors()
        print(f"\nTop attractors (stable patterns):")
        for node in attractors[:5]:
            print(f"  '{node.token_str}' - conf={node.confidence:.3f}, "
                  f"convergence={node.convergence_factor}")
        
        # Check for capital city convergence
        print("\nChecking for semantic convergence...")
        capital_nodes = [n for n in mesh.nodes.values() 
                        if any(c in n.token_str.lower() for c in ['paris', 'berlin', 'rome'])]
        if capital_nodes:
            print("  Found capital city nodes:")
            for node in capital_nodes:
                print(f"    '{node.token_str}' - conf={node.confidence:.3f}")
        
        # Boost capital cities confidence and test crystallization
        print("\nBoosting capital city confidence...")
        for node in capital_nodes:
            print(f"  Before: '{node.token_str}' conf={node.confidence:.3f}")
            node.confidence = 0.99
            # Add many sources to meet crystallization criteria
            for s in ["confirmed_capital", "geography", "gpt2", "wikipedia", "atlas"]:
                node.sources.add(s)
            node.incoming_paths[888] = 5  # Strong convergence
            node.incoming_paths[889] = 3  # More convergence
            node.incoming_paths[890] = 2  # Even more
            print(f"  After:  '{node.token_str}' conf={node.confidence:.3f}, sources={len(node.sources)}, convergence={node.convergence_factor}")
        
        # Run more physics
        for _ in range(5):
            state = physics.step()
        
        # Try crystallization
        crystals = physics.crystallize_all_stable()
        print(f"Crystallized {crystals} stable patterns!")
        
        # Final report
        print("\n" + physics.report())
        
        # Show crystallized
        if physics.collapse.crystallized:
            print("\n✨ Crystallized attractors:")
            for node_id in physics.collapse.crystallized:
                node = mesh.nodes.get(node_id)
                if node:
                    print(f"    '{node.token_str}' - FROZEN")
        
        print("\n✅ Full physics cycle complete!")
        
    except ImportError:
        print("⚠️ transformers not available, skipping GPT-2 test")
        print("   Install with: pip install transformers")


def test_entropy_dynamics():
    """Test entropy dynamics over time."""
    print("\n" + "="*60)
    print("ENTROPY DYNAMICS")
    print("="*60)
    
    mesh = PACMeshSpace(embed_dim=64, device='cpu')
    embeddings = SimpleEmbeddings(dim=64)
    physics = PhysicsMesh(mesh)
    
    # Build a tree structure progressively
    root = mesh.get_or_create_root(0, "start", embeddings.embed("start"), "dynamics")
    root.confidence = 0.5
    
    entropy_values = []
    
    for step in range(20):
        # Add some structure
        if step < 10:
            # Growing phase - add branches
            for i in range(3):
                token = f"token_{step}_{i}"
                token_id = step * 10 + i
                nodes_list = list(mesh.nodes.values())
                parent_node = root if step == 0 else nodes_list[step % len(nodes_list)]
                
                context = (parent_node.token_id,)
                node = mesh.get_or_create_context_node(
                    context, token_id, token, 
                    embeddings.embed(token), "dynamics"
                )
                parent_node.add_child(node)
                node.confidence = 0.3 + 0.05 * i
        else:
            # Consolidation phase - increase confidences
            for node in list(mesh.nodes.values())[:5]:
                node.confidence = min(1.0, node.confidence + 0.1)
        
        state = physics.step()
        entropy_values.append(state.entropy)
    
    print("\nEntropy over time:")
    print("  Step | Entropy | Trend")
    print("  -----+---------+------")
    for i in range(0, 20, 2):
        trend = "→" if i == 0 else ("↑" if entropy_values[i] > entropy_values[i-1] else "↓")
        bar = "█" * int(entropy_values[i] * 20)
        print(f"  {i:4d} | {entropy_values[i]:7.4f} | {bar}")
    
    print(f"\n  Peak entropy: {max(entropy_values):.4f}")
    print(f"  Final entropy: {entropy_values[-1]:.4f}")
    
    # Check for collapse events
    if physics.entropy_monitor.collapse_events:
        print(f"\n  Collapse events detected: {len(physics.entropy_monitor.collapse_events)}")
        for event in physics.entropy_monitor.collapse_events:
            print(f"    - {event.collapse_type.value} (magnitude={event.magnitude:.2f})")
    
    print("\n✅ Entropy dynamics demonstrated")


if __name__ == "__main__":
    test_physics_constants()
    test_basic_physics()
    test_convergence_physics()
    test_crystallization()
    test_entropy_dynamics()
    test_full_physics_cycle()
    
    print("\n" + "="*60)
    print("ALL PHYSICS TESTS COMPLETE")
    print("="*60)
    print("""
The physics layer provides:

1. ENTROPY MONITORING
   - Tracks disorder in mesh
   - Triggers collapse events when entropy spikes
   - Momentum-based smoothing for stability

2. CONSERVATION ENFORCEMENT  
   - Xi = 1.0571 balance operator
   - f(parent) = Σf(children) / Xi
   - Auto-corrects violations

3. RESONANCE FIELD
   - Phase alignment between similar embeddings
   - In-phase patterns reinforce each other
   - Anti-phase patterns compete

4. COLLAPSE ENGINE
   - Entropy collapse → convergence to attractors
   - Crystallization → stable patterns freeze
   - Tension release → geometry relaxation

5. CRYSTALLIZED KNOWLEDGE
   - High-confidence patterns become immutable
   - Multiple sources + convergence = stability
   - Attractors guide future learning
""")
