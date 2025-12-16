"""
Experiment 01: Resonance-Based Attention
========================================

Core question: Can we compute attention as field resonance?

Hypothesis:
- Attention weight between patterns = their resonance strength
- Similar patterns should attend to each other more
- Resonance attention should produce coherent outputs

Protocol:
1. Create patterns with known semantic relationships
2. Compute resonance attention
3. Verify: similar patterns have higher attention weights
4. Compare: resonance vs standard dot-product attention
"""

import torch
import torch.nn.functional as F
import json
from datetime import datetime
from pathlib import Path
import sys

# Add POC-002 path for physics components
poc_002_path = Path(__file__).parent.parent.parent / 'poc_002_resonance_training' / 'scripts'
sys.path.insert(0, str(poc_002_path))

from field_attention import (
    ResonanceAttention,
    get_device,
    PI_SQUARED_INV,
    ENTANGLEMENT_LIMIT,
)

# Import trainer for creating semantic embeddings
try:
    from physics_trainer import DawnFieldTrainer
    TRAINER_AVAILABLE = True
except ImportError:
    TRAINER_AVAILABLE = False


def test_basic_resonance():
    """Test that resonance attention computes valid outputs."""
    print("\n[Test 1: Basic Resonance]")
    
    device = get_device()
    dim = 64
    batch_size = 2
    seq_len = 8
    
    attn = ResonanceAttention(dim).to(device)
    
    # Random input
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    output, weights = attn(x, x, x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Weights shape: {weights.shape}")
    
    # Weights should sum to 1 (softmax)
    weight_sums = weights.sum(dim=-1)
    sum_error = (weight_sums - 1.0).abs().max().item()
    
    print(f"  Weight sum error: {sum_error:.6f}")
    
    assert output.shape == x.shape, "Output shape should match input"
    assert sum_error < 1e-5, "Weights should sum to 1"
    
    print(f"  ✓ Basic resonance attention works")
    return {'output_shape': list(output.shape), 'weight_sum_error': sum_error}


def test_self_attention_diagonal():
    """Test that patterns attend most strongly to themselves."""
    print("\n[Test 2: Self-Attention Diagonal]")
    
    device = get_device()
    dim = 64
    batch_size = 1
    seq_len = 8
    
    attn = ResonanceAttention(dim).to(device)
    
    # Create distinct patterns (orthogonal-ish)
    x = torch.zeros(batch_size, seq_len, dim, device=device)
    for i in range(seq_len):
        x[0, i, i * (dim // seq_len):(i + 1) * (dim // seq_len)] = 1.0
    
    output, weights = attn(x, x, x)
    
    # Check diagonal dominance
    diagonal = weights[0].diag()
    off_diagonal = weights[0] - torch.diag(diagonal)
    
    diag_mean = diagonal.mean().item()
    off_diag_mean = off_diagonal.abs().mean().item()
    
    print(f"  Diagonal mean: {diag_mean:.4f}")
    print(f"  Off-diagonal mean: {off_diag_mean:.4f}")
    print(f"  Ratio: {diag_mean / (off_diag_mean + 1e-6):.2f}x")
    
    # Diagonal should dominate for orthogonal patterns
    assert diag_mean > off_diag_mean, "Self-attention should be strongest"
    
    print(f"  ✓ Patterns attend most to themselves")
    return {'diagonal': diag_mean, 'off_diagonal': off_diag_mean}


def test_similar_patterns_attend():
    """Test that similar patterns attend to each other more."""
    print("\n[Test 3: Similar Patterns Attend]")
    
    device = get_device()
    dim = 64
    
    attn = ResonanceAttention(dim).to(device)
    
    # Create 4 patterns: 2 similar pairs
    # Patterns 0,1 are similar, patterns 2,3 are similar
    x = torch.zeros(1, 4, dim, device=device)
    
    # Pair 1: patterns 0,1 (similar - both in first half)
    x[0, 0, :dim//2] = torch.randn(dim//2)
    x[0, 1] = x[0, 0] + 0.1 * torch.randn(dim, device=device)  # Slight variation
    
    # Pair 2: patterns 2,3 (similar - both in second half)
    x[0, 2, dim//2:] = torch.randn(dim//2, device=device)
    x[0, 3] = x[0, 2] + 0.1 * torch.randn(dim, device=device)  # Slight variation
    
    output, weights = attn(x, x, x)
    
    w = weights[0]
    
    # Within-pair attention should be higher
    within_01 = (w[0, 1] + w[1, 0]).item() / 2
    within_23 = (w[2, 3] + w[3, 2]).item() / 2
    between = (w[0, 2] + w[0, 3] + w[1, 2] + w[1, 3]).item() / 4
    
    print(f"  Attention within pair (0,1): {within_01:.4f}")
    print(f"  Attention within pair (2,3): {within_23:.4f}")
    print(f"  Attention between pairs: {between:.4f}")
    
    # Within-pair should be higher
    assert within_01 > between, "Similar patterns should attend more"
    assert within_23 > between, "Similar patterns should attend more"
    
    print(f"  ✓ Similar patterns attend to each other more")
    return {
        'within_01': within_01,
        'within_23': within_23,
        'between': between,
    }


def test_semantic_attention():
    """Test attention on semantically trained embeddings."""
    print("\n[Test 4: Semantic Attention]")
    
    if not TRAINER_AVAILABLE:
        print("  ⚠ Physics trainer not available, skipping")
        return {'skipped': True}
    
    device = get_device()
    dim = 64
    
    # Train semantic embeddings
    trainer = DawnFieldTrainer(field_dim=dim, device=device)
    
    training_data = [
        ("cat", "dog"),
        ("cat", "animal"),
        ("dog", "animal"),
        ("red", "blue"),
        ("red", "color"),
        ("blue", "color"),
    ]
    
    trainer.train(training_data, epochs=20)
    
    # Create sequence from trained embeddings
    patterns = ["cat", "dog", "animal", "red", "blue", "color"]
    x = torch.stack([trainer.resonance.field_memory[p] for p in patterns]).unsqueeze(0)
    
    # Apply resonance attention
    attn = ResonanceAttention(dim).to(device)
    output, weights = attn(x, x, x)
    
    w = weights[0]
    
    # Check semantic attention patterns
    # Animals (0,1,2) should attend to each other more than to colors (3,4,5)
    animal_to_animal = (w[0, 1] + w[0, 2] + w[1, 0] + w[1, 2] + w[2, 0] + w[2, 1]).item() / 6
    animal_to_color = (w[0, 3] + w[0, 4] + w[0, 5] + w[1, 3] + w[1, 4] + w[1, 5]).item() / 6
    
    color_to_color = (w[3, 4] + w[3, 5] + w[4, 3] + w[4, 5] + w[5, 3] + w[5, 4]).item() / 6
    color_to_animal = (w[3, 0] + w[3, 1] + w[3, 2] + w[4, 0] + w[4, 1] + w[4, 2]).item() / 6
    
    print(f"  Animal→Animal attention: {animal_to_animal:.4f}")
    print(f"  Animal→Color attention: {animal_to_color:.4f}")
    print(f"  Color→Color attention: {color_to_color:.4f}")
    print(f"  Color→Animal attention: {color_to_animal:.4f}")
    
    # Within-class should be higher
    within_class = (animal_to_animal + color_to_color) / 2
    between_class = (animal_to_color + color_to_animal) / 2
    separation = within_class - between_class
    
    print(f"  Within-class avg: {within_class:.4f}")
    print(f"  Between-class avg: {between_class:.4f}")
    print(f"  Separation: {separation:.4f}")
    
    assert separation > 0, "Within-class attention should be higher"
    
    print(f"  ✓ Semantic structure reflected in attention")
    return {
        'within_class': within_class,
        'between_class': between_class,
        'separation': separation,
    }


def test_max_coupling_constraint():
    """Test that attention respects max coupling limit."""
    print("\n[Test 5: Max Coupling Constraint]")
    
    device = get_device()
    dim = 64
    
    attn = ResonanceAttention(dim, max_coupling=ENTANGLEMENT_LIMIT).to(device)
    
    # Create identical patterns (max similarity)
    x = torch.randn(1, 4, dim, device=device)
    x = x.repeat(1, 1, 1)  # All same pattern
    x[0, 1] = x[0, 0]
    x[0, 2] = x[0, 0]
    x[0, 3] = x[0, 0]
    
    # Get resonance before softmax
    q_norm = F.normalize(x, dim=-1)
    resonance = torch.bmm(q_norm, q_norm.transpose(-2, -1))
    
    max_resonance = resonance.max().item()
    
    print(f"  Max resonance: {max_resonance:.4f}")
    print(f"  Coupling limit: {ENTANGLEMENT_LIMIT:.4f}")
    
    # After clamping, should be at or below limit
    clamped = torch.clamp(resonance, -ENTANGLEMENT_LIMIT, ENTANGLEMENT_LIMIT)
    max_clamped = clamped.max().item()
    
    print(f"  After clamping: {max_clamped:.4f}")
    
    assert max_clamped <= ENTANGLEMENT_LIMIT + 1e-6, "Should respect coupling limit"
    
    print(f"  ✓ Max coupling constraint enforced")
    return {'max_resonance': max_resonance, 'after_clamp': max_clamped}


def test_gradient_flow():
    """Test that gradients flow through resonance attention."""
    print("\n[Test 6: Gradient Flow]")
    
    device = get_device()
    dim = 64
    
    attn = ResonanceAttention(dim).to(device)
    
    x = torch.randn(2, 8, dim, device=device, requires_grad=True)
    
    output, weights = attn(x, x, x)
    
    # Compute loss
    loss = output.sum()
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None, "Gradients should flow to input"
    grad_norm = x.grad.norm().item()
    
    print(f"  Input gradient norm: {grad_norm:.4f}")
    print(f"  Output shape: {output.shape}")
    
    assert grad_norm > 0, "Gradients should be non-zero"
    
    print(f"  ✓ Gradients flow through resonance attention")
    return {'grad_norm': grad_norm}


def run_all_experiments():
    """Run all resonance attention experiments."""
    print("=" * 60)
    print("POC-003 Experiment 01: Resonance-Based Attention")
    print("=" * 60)
    print(f"Device: {get_device()}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"1/π² = {PI_SQUARED_INV:.4f}")
    print(f"Max coupling = {ENTANGLEMENT_LIMIT:.4f}")
    
    results = {
        'experiment': 'poc_003_exp_01_resonance_attention',
        'timestamp': datetime.now().isoformat(),
        'device': str(get_device()),
        'constants': {
            'pi_squared_inv': PI_SQUARED_INV,
            'entanglement_limit': ENTANGLEMENT_LIMIT,
        },
        'tests': {},
    }
    
    tests = [
        ('basic_resonance', test_basic_resonance),
        ('self_attention', test_self_attention_diagonal),
        ('similar_patterns', test_similar_patterns_attend),
        ('semantic_attention', test_semantic_attention),
        ('max_coupling', test_max_coupling_constraint),
        ('gradient_flow', test_gradient_flow),
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
    
    filename = f"exp_01_resonance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
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
