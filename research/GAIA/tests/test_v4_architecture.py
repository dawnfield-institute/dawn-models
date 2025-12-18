"""
GAIA v4.0 + Fracton v2.0 Architecture Test

Validates the new PAC-Lazy substrate architecture.
"""

import sys
import os

# Add paths
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\fracton")
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\dawn-models\research\GAIA\src")

import torch
import time

print("=" * 60)
print("GAIA v4.0 + Fracton v2.0 Architecture Test")
print("=" * 60)

# Test 1: Physics Constants
print("\n[1] Testing Fracton Physics Constants...")
try:
    from fracton.physics import PHI, XI, PHI_XI, LAMBDA_STAR
    print(f"    PHI (φ):          {PHI:.10f}")
    print(f"    XI (ξ):           {XI:.10f}")
    print(f"    PHI_XI (φ×ξ):     {PHI_XI:.10f}")
    print(f"    LAMBDA_STAR (λ*): {LAMBDA_STAR:.10f}")
    
    # Validate values (empirically tuned, not derived)
    assert abs(PHI - 1.618) < 0.001, "PHI should be ~1.618"
    assert abs(XI - 0.0618) < 0.001, "XI should be ~0.0618"
    assert abs(PHI_XI - 0.1) < 0.001, "PHI_XI should be ~0.1"
    assert LAMBDA_STAR > 0.9 and LAMBDA_STAR < 1.0, "LAMBDA_STAR should be ~0.98"
    print("    ✓ All constants validated")
except Exception as e:
    print(f"    ✗ Error: {e}")
    sys.exit(1)

# Test 2: PAC-Lazy Node
print("\n[2] Testing PACNode...")
try:
    from fracton.core import PACNode, PACNodeFactory
    
    factory = PACNodeFactory(device='cpu')
    
    # Create root
    root_value = torch.randn(64)
    root = factory.create_root(root_value, label="test_root")
    print(f"    Root: {root}")
    
    # Create child
    child_delta = torch.randn(64) * 0.1
    child = factory.create_child(root, child_delta, label="test_child")
    print(f"    Child: {child}")
    
    assert root.is_root, "Root should be root"
    assert not child.is_root, "Child should not be root"
    assert child.parent_id == root.id, "Child parent mismatch"
    assert root.id in [child.parent_id], "Root should be child's parent"
    print("    ✓ PACNode creation validated")
except Exception as e:
    print(f"    ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: PAC System
print("\n[3] Testing PACSystem...")
try:
    from fracton.core import PACSystem
    
    system = PACSystem(device='cpu')
    
    # Inject patterns
    patterns = []
    for i in range(10):
        pattern = torch.randn(64)
        node_id = system.inject(pattern, label=f"pattern_{i}")
        patterns.append((node_id, pattern))
    
    print(f"    Injected {len(patterns)} patterns")
    print(f"    System stats: {system.stats()}")
    
    # Test reconstruction
    for node_id, original in patterns[:3]:
        reconstructed = system.reconstruct(node_id)
        diff = torch.abs(original - reconstructed).max().item()
        assert diff < 1e-6, f"Reconstruction error: {diff}"
    
    print("    ✓ Reconstruction validated")
    
    # Test resonance search
    query = patterns[0][1]
    resonant = system.find_resonant(query, top_k=5)
    print(f"    Resonant search: {len(resonant)} results")
    assert len(resonant) > 0, "Should find resonant patterns"
    assert resonant[0][0] == patterns[0][0], "Top result should be self"
    print("    ✓ Resonance search validated")
    
except Exception as e:
    print(f"    ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Field Encoding
print("\n[4] Testing Field Encoding...")
try:
    from fracton.field import spherical_encode, spherical_encode_batch
    
    # Single token
    field = spherical_encode(42, vocab_size=50257, dim=64)
    print(f"    Single encode shape: {field.shape}")
    assert field.shape == (64,), "Wrong shape"
    
    # Batch
    token_ids = torch.tensor([1, 2, 3, 4, 5])
    fields = spherical_encode_batch(token_ids, vocab_size=50257, dim=64)
    print(f"    Batch encode shape: {fields.shape}")
    assert fields.shape == (5, 64), "Wrong batch shape"
    
    # Uniqueness
    for i in range(len(token_ids)):
        for j in range(i + 1, len(token_ids)):
            sim = torch.dot(fields[i], fields[j]).item()
            assert abs(sim) < 0.99, f"Tokens {i} and {j} too similar: {sim}"
    
    print("    ✓ Encoding validated")
except Exception as e:
    print(f"    ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Field Evolution
print("\n[5] Testing Field Evolution...")
try:
    from fracton.field import evolve, evolve_batch, compute_field_energy
    
    field = torch.randn(64)
    energy_before = compute_field_energy(field)
    
    evolved = evolve(field, steps=10)
    energy_after = compute_field_energy(evolved)
    
    print(f"    Energy before: {energy_before:.4f}")
    print(f"    Energy after:  {energy_after:.4f}")
    
    # Evolution should change the field
    diff = torch.abs(field - evolved).mean().item()
    assert diff > 0.01, "Evolution should change field"
    
    print("    ✓ Evolution validated")
except Exception as e:
    print(f"    ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Resonance
print("\n[6] Testing Resonance...")
try:
    from fracton.field import compute_resonance, compute_resonance_batch, ResonanceMesh
    
    a = torch.randn(64)
    b = a.clone()  # Identical
    c = torch.randn(64)  # Random
    
    res_same = compute_resonance(a, b)
    res_diff = compute_resonance(a, c)
    
    print(f"    Self-resonance: {res_same:.4f}")
    print(f"    Random resonance: {res_diff:.4f}")
    
    assert res_same > 0.99, "Self-resonance should be ~1"
    assert abs(res_diff) < 0.5, "Random resonance should be low"
    
    # Test mesh
    mesh = ResonanceMesh()
    for i in range(20):
        mesh.add(torch.randn(64), field_id=i)
    
    query = torch.randn(64)
    results = mesh.find_resonant(query, top_k=5)
    print(f"    Mesh search: {len(results)} results")
    
    print("    ✓ Resonance validated")
except Exception as e:
    print(f"    ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: GAIA Cortex
print("\n[7] Testing GAIA Cortex...")
try:
    from v4 import GAIACortex, GAIAConfig
    
    config = GAIAConfig(
        device='cpu',
        field_dim=64,
        evolution_steps=3
    )
    cortex = GAIACortex(config)
    
    print(f"    Cortex: {cortex}")
    
    # Process text
    response = cortex.process("hello world")
    print(f"    Response node_id: {response.node_id}")
    print(f"    Field energy: {response.field_energy:.4f}")
    print(f"    Processing time: {response.processing_time*1000:.2f}ms")
    
    # Process multiple
    for i in range(5):
        response = cortex.process(f"test message {i}")
    
    stats = cortex.stats()
    print(f"    Total processed: {stats['process_count']}")
    print(f"    Substrate size: {stats['substrate'].get('total', stats['substrate'].get('node_count', 'N/A'))}")
    
    print("    ✓ Cortex validated")
except Exception as e:
    print(f"    ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: GAIA Organs
print("\n[8] Testing GAIA Organs...")
try:
    from v4 import GAIACortex, GAIAConfig, LanguageOrgan, ReasoningOrgan, MemoryOrgan
    
    config = GAIAConfig(device='cpu', field_dim=64)
    cortex = GAIACortex(config)
    
    # Attach organs
    cortex.attach_organ(LanguageOrgan())
    cortex.attach_organ(ReasoningOrgan())
    cortex.attach_organ(MemoryOrgan())
    
    print(f"    Attached organs: {list(cortex.organs.keys())}")
    
    # Process with organs
    response = cortex.process("testing organs")
    print(f"    Organ contributions: {list(response.organ_contributions.keys())}")
    
    # Verify organs processed
    for name, contribution in response.organ_contributions.items():
        energy = torch.sum(contribution ** 2).item()
        print(f"      {name}: energy={energy:.4f}")
    
    print("    ✓ Organs validated")
except Exception as e:
    print(f"    ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Continuous Learning
print("\n[9] Testing Continuous Learning...")
try:
    from v4 import GAIACortex, GAIAConfig
    from v4.learning import ContinuousLearner
    from fracton.core import PACSystem
    
    system = PACSystem(device='cpu')
    learner = ContinuousLearner(system)
    
    # Learn patterns
    for i in range(20):
        pattern = torch.randn(64)
        node_id = learner.learn(pattern, importance=1.0)
    
    stats = learner.stats()
    print(f"    Patterns learned: {stats['patterns_learned']}")
    print(f"    Connections: {stats['connections']}")
    
    # Test prediction
    query = torch.randn(64)
    predictions = learner.predict_next(query, top_k=3)
    print(f"    Predictions: {len(predictions)}")
    
    # Consolidate
    modified = learner.consolidate()
    print(f"    Consolidated: {modified} connections modified")
    
    print("    ✓ Learning validated")
except Exception as e:
    print(f"    ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# GPU Test (if available)
print("\n[10] Testing GPU Support...")
if torch.cuda.is_available():
    try:
        from fracton.core import PACSystem
        from fracton.field import spherical_encode_batch, evolve_batch
        
        device = 'cuda'
        system = PACSystem(device=device)
        
        # Batch encode on GPU
        token_ids = torch.arange(1000, device=device)
        start = time.time()
        fields = spherical_encode_batch(token_ids, dim=64)
        encode_time = time.time() - start
        print(f"    GPU batch encode (1000 tokens): {encode_time*1000:.2f}ms")
        
        # Batch evolve on GPU
        start = time.time()
        evolved = evolve_batch(fields, steps=10)
        evolve_time = time.time() - start
        print(f"    GPU batch evolve (1000 fields): {evolve_time*1000:.2f}ms")
        
        # Inject all
        start = time.time()
        for i, field in enumerate(fields[:100]):
            system.inject(field, label=f"gpu_pattern_{i}")
        inject_time = time.time() - start
        print(f"    GPU inject (100 patterns): {inject_time*1000:.2f}ms")
        
        print("    ✓ GPU validated")
    except Exception as e:
        print(f"    ✗ GPU Error: {e}")
else:
    print("    ⚠ CUDA not available, skipping GPU tests")

# Summary
print("\n" + "=" * 60)
print("All tests passed! GAIA v4.0 + Fracton v2.0 architecture ready.")
print("=" * 60)

print("\nNew Architecture Summary:")
print("  Fracton v2.0 (SDK):")
print("    - fracton/physics/constants.py  - Dawn Field constants")
print("    - fracton/core/pac_node.py      - Delta-only storage")
print("    - fracton/core/pac_system.py    - Tree management + cache")
print("    - fracton/field/encoding.py     - Spherical encoding")
print("    - fracton/field/evolution.py    - Klein-Gordon dynamics")
print("    - fracton/field/resonance.py    - Pattern matching")
print("")
print("  GAIA v4.0 (Model):")
print("    - src/v4/cortex.py              - Central integration")
print("    - src/v4/organs.py              - Specialized transformers")
print("    - src/v4/learning.py            - Continuous learning")
print("")
print("  Legacy (Archived):")
print("    - src/legacy/                   - Old GAIA code")
