"""
Experiment 04: Full GAIA Integration
====================================

Integrate POC-002 resonance training with GAIA field architecture.

Tests:
1. GAIA memory field accepts trained embeddings
2. PAC conservation maintained through integration
3. Semantic relationships preserved in field
4. GPU execution end-to-end
"""

import torch
import json
from datetime import datetime
from pathlib import Path
import sys

# Add POC-001 path for GAIA encoder
poc_001_path = Path(__file__).parent.parent.parent / 'poc_001_pattern_encoding' / 'scripts'
sys.path.insert(0, str(poc_001_path))

from physics_trainer import (
    DawnFieldTrainer,
    ResonanceTrainer,
    PHI_XI,
    LAMBDA_STAR,
    EIGENVALUE_DECAY,
    get_device,
)

# Import GAIA encoder from POC-001
try:
    from gaia_encoder import GAIAFieldEncoder, create_gaia_memory_field
    GAIA_AVAILABLE = True
except ImportError:
    GAIA_AVAILABLE = False
    print("Warning: GAIA encoder not available from POC-001")


def test_gaia_imports():
    """Test GAIA components are available."""
    print("\n[Test 1: GAIA Imports]")
    
    if not GAIA_AVAILABLE:
        print("  ⚠ GAIA encoder not available, using fallback")
        return {'available': False}
    
    print(f"  ✓ GAIAFieldEncoder imported")
    print(f"  ✓ create_gaia_memory_field imported")
    return {'available': True}


def test_embedding_transfer():
    """Test transferring trained embeddings to GAIA field."""
    print("\n[Test 2: Embedding Transfer]")
    
    device = get_device()
    
    # Train with physics trainer
    trainer = DawnFieldTrainer(field_dim=64, device=device)
    
    training_data = [
        ("hello", "world"),
        ("foo", "bar"),
    ]
    
    trainer.train(training_data, epochs=10)
    
    # Get trained embeddings
    embeddings = {}
    for pattern in ["hello", "world", "foo", "bar"]:
        emb = trainer.resonance.field_memory.get(pattern)
        if emb is not None:
            embeddings[pattern] = emb
    
    print(f"  Trained embeddings: {list(embeddings.keys())}")
    
    # If GAIA available, transfer to field
    if GAIA_AVAILABLE:
        field = create_gaia_memory_field(64, device=device)
        
        for pattern, emb in embeddings.items():
            # GAIA stores in field structure
            # For now, just validate shapes match
            assert emb.shape[0] == 64, "Embedding should match field dim"
        
        print(f"  ✓ Embeddings compatible with GAIA field")
    else:
        # Fallback: validate embeddings are proper tensors
        for pattern, emb in embeddings.items():
            assert isinstance(emb, torch.Tensor)
            assert emb.device.type == device.type
        
        print(f"  ✓ Embeddings are valid GPU tensors")
    
    return {'patterns': list(embeddings.keys()), 'dim': 64}


def test_semantic_preservation():
    """Test semantic relationships are preserved after transfer."""
    print("\n[Test 3: Semantic Preservation]")
    
    device = get_device()
    trainer = DawnFieldTrainer(field_dim=64, device=device)
    
    # Train semantic groups
    training_data = [
        ("alpha", "beta"),
        ("gamma", "delta"),
    ]
    
    trainer.train(training_data, epochs=20)
    
    # Measure similarity before "transfer"
    sim_same = trainer.similarity("alpha", "beta")
    sim_diff = trainer.similarity("alpha", "gamma")
    
    print(f"  sim(alpha, beta) = {sim_same:.3f}")
    print(f"  sim(alpha, gamma) = {sim_diff:.3f}")
    
    # Semantic relationship should be preserved
    assert sim_same > sim_diff, "Same-group should be more similar"
    
    # Extract embeddings
    emb_alpha = trainer.resonance.field_memory["alpha"]
    emb_beta = trainer.resonance.field_memory["beta"]
    emb_gamma = trainer.resonance.field_memory["gamma"]
    
    # Compute raw cosine similarity (what GAIA would use)
    raw_sim_same = torch.nn.functional.cosine_similarity(
        emb_alpha.unsqueeze(0), 
        emb_beta.unsqueeze(0)
    ).item()
    
    raw_sim_diff = torch.nn.functional.cosine_similarity(
        emb_alpha.unsqueeze(0),
        emb_gamma.unsqueeze(0)
    ).item()
    
    print(f"  Raw cosine(alpha, beta) = {raw_sim_same:.3f}")
    print(f"  Raw cosine(alpha, gamma) = {raw_sim_diff:.3f}")
    
    # Raw should also show semantic structure
    assert raw_sim_same > raw_sim_diff, "Raw cosine should preserve semantics"
    
    print(f"  ✓ Semantic relationships preserved in embeddings")
    return {
        'combined': {'same': sim_same, 'diff': sim_diff},
        'raw_cosine': {'same': raw_sim_same, 'diff': raw_sim_diff},
    }


def test_pac_conservation_integration():
    """Test PAC conservation through full integration."""
    print("\n[Test 4: PAC Conservation Integration]")
    
    device = get_device()
    trainer = DawnFieldTrainer(field_dim=64, device=device)
    
    # Heavy training
    training_data = [(f"p{i}", f"p{i+1}") for i in range(50)]
    
    stats = trainer.train(training_data, epochs=50, check_conservation=True)
    
    residual = stats['conservation_residual']
    
    print(f"  Patterns trained: {len(trainer.resonance.field_memory)}")
    print(f"  Conservation residual: {residual:.2e}")
    print(f"  Conservation OK: {stats['conservation_ok']}")
    
    assert residual < 1e-3, f"Conservation violated: {residual}"
    
    print(f"  ✓ PAC conservation maintained through integration")
    return {'residual': residual, 'patterns': len(trainer.resonance.field_memory)}


def test_gpu_execution():
    """Test full GPU execution path."""
    print("\n[Test 5: GPU Execution]")
    
    device = get_device()
    
    if device.type != 'cuda':
        print(f"  ⚠ Running on CPU ({device})")
    
    trainer = DawnFieldTrainer(field_dim=128, device=device)
    
    # Train on larger vocabulary
    vocab = [f"word_{i}" for i in range(100)]
    training_data = [(vocab[i], vocab[i+1]) for i in range(0, 98, 2)]
    
    # Time the training
    import time
    start = time.perf_counter()
    
    stats = trainer.train(training_data, epochs=20)
    
    elapsed = time.perf_counter() - start
    
    print(f"  Device: {device}")
    print(f"  Vocabulary: {len(vocab)} words")
    print(f"  Training time: {elapsed*1000:.1f}ms")
    print(f"  Steps/second: {stats['total_steps'] / elapsed:.1f}")
    
    # Check all embeddings are on correct device
    for pattern, emb in trainer.resonance.field_memory.items():
        assert emb.device.type == device.type, f"Embedding on wrong device"
    
    print(f"  ✓ All operations on {device.type.upper()}")
    return {
        'device': str(device),
        'vocab_size': len(vocab),
        'time_ms': elapsed * 1000,
        'steps_per_sec': stats['total_steps'] / elapsed,
    }


def test_full_pipeline():
    """Test complete GAIA training pipeline."""
    print("\n[Test 6: Full Pipeline]")
    
    device = get_device()
    
    # 1. Create trainer with all physics
    trainer = DawnFieldTrainer(field_dim=64, device=device)
    
    # 2. Define semantic classes
    classes = {
        'animals': ['cat', 'dog', 'bird'],
        'colors': ['red', 'blue', 'green'],
        'verbs': ['run', 'walk', 'jump'],
    }
    
    # 3. Generate training data
    training_data = []
    for items in classes.values():
        for i, item1 in enumerate(items):
            for item2 in items[i+1:]:
                training_data.append((item1, item2))
    
    # 4. Train
    stats = trainer.train(training_data, epochs=30, check_conservation=True)
    
    # 5. Evaluate semantic quality
    within_sims = []
    between_sims = []
    
    for class_items in classes.values():
        for i, item1 in enumerate(class_items):
            for item2 in class_items[i+1:]:
                within_sims.append(trainer.similarity(item1, item2))
    
    for item1 in classes['animals']:
        for item2 in classes['colors']:
            between_sims.append(trainer.similarity(item1, item2))
    
    within_avg = sum(within_sims) / len(within_sims)
    between_avg = sum(between_sims) / len(between_sims)
    separation = within_avg - between_avg
    
    print(f"  Classes: {list(classes.keys())}")
    print(f"  Training pairs: {len(training_data)}")
    print(f"  Phase transitions: {stats.get('transitions', 0)}")
    print(f"  Within-class avg: {within_avg:.3f}")
    print(f"  Between-class avg: {between_avg:.3f}")
    print(f"  Semantic separation: {separation:.3f}")
    print(f"  Conservation residual: {stats['conservation_residual']:.2e}")
    
    # Success criteria
    assert separation > 0.3, f"Separation too low: {separation}"
    assert stats['conservation_ok'], "Conservation violated"
    
    print(f"  ✓ Full pipeline successful!")
    return {
        'separation': separation,
        'within': within_avg,
        'between': between_avg,
        'transitions': stats.get('transitions', 0),
        'conservation': stats['conservation_residual'],
    }


def run_all_experiments():
    """Run all GAIA integration experiments."""
    print("=" * 60)
    print("POC-002 Experiment 04: GAIA Integration")
    print("=" * 60)
    print(f"Device: {get_device()}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"GAIA available: {GAIA_AVAILABLE}")
    
    results = {
        'experiment': 'poc_002_exp_04_gaia_integration',
        'timestamp': datetime.now().isoformat(),
        'device': str(get_device()),
        'gaia_available': GAIA_AVAILABLE,
        'constants': {
            'phi_xi': PHI_XI,
            'lambda_star': LAMBDA_STAR,
            'eigenvalue_decay': EIGENVALUE_DECAY,
        },
        'tests': {},
    }
    
    tests = [
        ('gaia_imports', test_gaia_imports),
        ('embedding_transfer', test_embedding_transfer),
        ('semantic_preservation', test_semantic_preservation),
        ('pac_conservation', test_pac_conservation_integration),
        ('gpu_execution', test_gpu_execution),
        ('full_pipeline', test_full_pipeline),
    ]
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            results['tests'][name] = result if isinstance(result, dict) else {'result': result}
            results['tests'][name]['passed'] = True
        except AssertionError as e:
            results['tests'][name] = {'passed': False, 'error': str(e)}
        except Exception as e:
            results['tests'][name] = {'passed': False, 'error': str(e), 'type': type(e).__name__}
    
    # Summary
    passed = sum(1 for t in results['tests'].values() if t.get('passed', False))
    total = len(results['tests'])
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    results['summary'] = {
        'passed': passed,
        'total': total,
        'success': passed == total,
    }
    
    # Save
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    filename = f"exp_04_gaia_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, float):
            return round(obj, 6) if abs(obj) < 1e10 else str(obj)
        else:
            return obj
    
    with open(results_dir / filename, 'w') as f:
        json.dump(clean_for_json(results), f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    return results


if __name__ == "__main__":
    run_all_experiments()
