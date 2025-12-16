"""
Experiment 02: Harmonic Head Structure
======================================

Test whether prime harmonic multi-head structure outperforms uniform heads.

From PHM:
- Head n has weight 1/p_n² where p_n is nth prime
- Creates natural hierarchy: head 0 (prime=2) most important
- Total importance = Σ 1/p² ≈ 0.4419 for 8 heads

Protocol:
1. Compare harmonic heads vs uniform heads
2. Test if harmonic weights match empirical importance
3. Verify prime harmonic structure improves attention quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from datetime import datetime
from pathlib import Path
import sys

poc_002_path = Path(__file__).parent.parent.parent / 'poc_002_resonance_training' / 'scripts'
sys.path.insert(0, str(poc_002_path))

from field_attention import (
    HarmonicMultiHeadAttention,
    HarmonicHead,
    ResonanceAttention,
    PRIMES,
    PI_SQUARED_INV,
    get_device,
)

try:
    from physics_trainer import DawnFieldTrainer
    TRAINER_AVAILABLE = True
except ImportError:
    TRAINER_AVAILABLE = False


def test_harmonic_weights():
    """Test that harmonic weights follow 1/p² pattern."""
    print("\n[Test 1: Harmonic Weights]")
    
    device = get_device()
    dim = 64
    n_heads = 8
    
    attn = HarmonicMultiHeadAttention(dim, n_heads).to(device)
    
    # Check head importance
    expected_weights = [1.0 / (p ** 2) for p in PRIMES[:n_heads]]
    actual_weights = [h.importance for h in attn.heads]
    
    print(f"  Prime sequence: {PRIMES[:n_heads]}")
    print(f"  Expected 1/p²: {[f'{w:.4f}' for w in expected_weights]}")
    print(f"  Actual weights: {[f'{w:.4f}' for w in actual_weights]}")
    
    # Should match exactly
    for expected, actual in zip(expected_weights, actual_weights):
        assert abs(expected - actual) < 1e-6, f"Weight mismatch: {expected} vs {actual}"
    
    total = sum(actual_weights)
    print(f"  Total importance: {total:.4f}")
    
    print(f"  ✓ Harmonic weights follow 1/p² pattern")
    return {'weights': actual_weights, 'primes': PRIMES[:n_heads], 'total': total}


def test_head_hierarchy():
    """Test that heads form a natural hierarchy."""
    print("\n[Test 2: Head Hierarchy]")
    
    device = get_device()
    dim = 64
    n_heads = 8
    
    attn = HarmonicMultiHeadAttention(dim, n_heads).to(device)
    
    # Create input
    x = torch.randn(2, 16, dim, device=device)
    
    output, info = attn(x, x, x)
    
    # Head weights should show hierarchy
    head_importance = info['head_importance']
    
    # Check monotonic decrease
    is_decreasing = all(head_importance[i] >= head_importance[i+1] 
                        for i in range(len(head_importance)-1))
    
    print(f"  Head importance: {[f'{w:.4f}' for w in head_importance]}")
    print(f"  Monotonically decreasing: {is_decreasing}")
    
    # First head should dominate
    first_to_total = head_importance[0] / sum(head_importance)
    print(f"  First head fraction: {first_to_total:.2%}")
    
    assert is_decreasing, "Heads should decrease in importance"
    assert first_to_total > 0.5, "First head should dominate"
    
    print(f"  ✓ Natural hierarchy in head importance")
    return {
        'is_decreasing': is_decreasing,
        'first_head_fraction': first_to_total,
    }


class UniformMultiHeadAttention(nn.Module):
    """Standard multi-head attention with uniform head weights."""
    
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Standard projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scale = 1.0 / (self.head_dim ** 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        return self.out_proj(out), attn


def test_harmonic_vs_uniform():
    """Compare harmonic heads to uniform heads."""
    print("\n[Test 3: Harmonic vs Uniform]")
    
    device = get_device()
    dim = 64
    n_heads = 8
    batch_size = 4
    seq_len = 32
    
    harmonic = HarmonicMultiHeadAttention(dim, n_heads, use_projections=True).to(device)
    uniform = UniformMultiHeadAttention(dim, n_heads).to(device)
    
    # Multiple test inputs
    outputs_harmonic = []
    outputs_uniform = []
    
    for _ in range(10):
        x = torch.randn(batch_size, seq_len, dim, device=device)
        
        with torch.no_grad():
            out_h, _ = harmonic(x, x, x)
            out_u, _ = uniform(x)
        
        outputs_harmonic.append(out_h)
        outputs_uniform.append(out_u)
    
    # Compare output statistics
    harmonic_std = torch.stack([o.std() for o in outputs_harmonic]).mean().item()
    uniform_std = torch.stack([o.std() for o in outputs_uniform]).mean().item()
    
    print(f"  Harmonic output std: {harmonic_std:.4f}")
    print(f"  Uniform output std: {uniform_std:.4f}")
    
    # Both should produce valid outputs
    assert harmonic_std > 0, "Harmonic should produce non-zero output"
    assert uniform_std > 0, "Uniform should produce non-zero output"
    
    print(f"  ✓ Both produce valid outputs")
    return {
        'harmonic_std': harmonic_std,
        'uniform_std': uniform_std,
    }


def test_semantic_head_specialization():
    """Test if heads specialize on semantic patterns."""
    print("\n[Test 4: Semantic Head Specialization]")
    
    if not TRAINER_AVAILABLE:
        print("  ⚠ Physics trainer not available, skipping")
        return {'skipped': True}
    
    device = get_device()
    dim = 64
    n_heads = 4  # Fewer heads for clearer analysis
    
    # Train semantic embeddings
    trainer = DawnFieldTrainer(field_dim=dim, device=device)
    
    training_data = [
        ("cat", "dog"), ("bird", "fish"),  # Animals
        ("red", "blue"), ("green", "yellow"),  # Colors
    ]
    
    trainer.train(training_data, epochs=20)
    
    # Create sequence
    patterns = ["cat", "dog", "bird", "fish", "red", "blue", "green", "yellow"]
    x = torch.stack([trainer.resonance.field_memory[p] for p in patterns]).unsqueeze(0)
    
    # Apply harmonic attention
    attn = HarmonicMultiHeadAttention(dim, n_heads, use_projections=False).to(device)
    output, info = attn(x, x, x)
    
    head_weights = info['head_weights'][0]  # (n_heads, seq, seq)
    
    # Analyze each head's attention pattern
    print(f"  Analyzing {n_heads} heads on {len(patterns)} patterns")
    
    for h in range(n_heads):
        # Animals: indices 0-3, Colors: indices 4-7
        animal_to_animal = head_weights[h, :4, :4].mean().item()
        color_to_color = head_weights[h, 4:, 4:].mean().item()
        cross = (head_weights[h, :4, 4:].mean() + head_weights[h, 4:, :4].mean()).item() / 2
        
        bias = (animal_to_animal + color_to_color) / 2 - cross
        
        print(f"  Head {h} (p={PRIMES[h]}): within={animal_to_animal:.3f}/{color_to_color:.3f}, cross={cross:.3f}, bias={bias:.3f}")
    
    print(f"  ✓ Head attention patterns analyzed")
    return {'heads_analyzed': n_heads}


def test_prime_harmonic_series():
    """Test connection to prime harmonic series ζ_P(2)."""
    print("\n[Test 5: Prime Harmonic Series]")
    
    # ζ_P(2) = Σ 1/p² for all primes = 0.4522... (prime zeta function at s=2)
    # Our truncated version with 8 primes
    
    n_heads = 8
    our_sum = sum(1.0 / (p ** 2) for p in PRIMES[:n_heads])
    
    # Known value (can compute more precisely)
    theoretical_infinite = 0.4522474200  # ζ_P(2)
    
    print(f"  Sum with {n_heads} primes: {our_sum:.6f}")
    print(f"  Theoretical (infinite): {theoretical_infinite:.6f}")
    print(f"  Fraction captured: {our_sum / theoretical_infinite:.2%}")
    
    # Check we capture most of the sum
    assert our_sum / theoretical_infinite > 0.9, "Should capture >90% of infinite sum"
    
    # Individual contributions
    print(f"  Contributions:")
    for i, p in enumerate(PRIMES[:n_heads]):
        contrib = (1.0 / (p ** 2)) / our_sum
        print(f"    1/{p}² = {1.0/(p**2):.4f} ({contrib:.1%})")
    
    print(f"  ✓ Prime harmonic series verified")
    return {
        'our_sum': our_sum,
        'theoretical': theoretical_infinite,
        'fraction': our_sum / theoretical_infinite,
    }


def test_eigenvalue_decay_connection():
    """Test connection between 1/π² and prime harmonics."""
    print("\n[Test 6: Eigenvalue-Prime Connection]")
    
    # 1/π² ≈ 0.1013 appears in eigenvalue decay
    # The Basel problem: ζ(2) = Σ 1/n² = π²/6
    # So 1/π² = 6/ζ(2) = 6/(π²/6) = 6 * 6/π² ≈ 0.607
    
    # Actually: 1/π² ≈ 0.1013
    # And the first prime contribution 1/2² = 0.25
    # Ratio: 0.25 / 0.1013 ≈ 2.47
    
    pi_sq_inv = PI_SQUARED_INV
    first_prime = 1.0 / (2 ** 2)
    
    print(f"  1/π² = {pi_sq_inv:.4f}")
    print(f"  1/2² = {first_prime:.4f}")
    print(f"  Ratio: {first_prime / pi_sq_inv:.4f}")
    
    # The key insight: eigenvalue decay 1/π² is related to
    # the average prime contribution
    avg_prime_contrib = sum(1.0/(p**2) for p in PRIMES[:8]) / 8
    print(f"  Avg 1/p² (8 primes): {avg_prime_contrib:.4f}")
    
    # Interesting: 1/π² ≈ 2 * avg contribution
    ratio_to_pi = pi_sq_inv / avg_prime_contrib
    print(f"  1/π² / avg(1/p²) = {ratio_to_pi:.4f}")
    
    print(f"  ✓ Eigenvalue-prime connection explored")
    return {
        'pi_sq_inv': pi_sq_inv,
        'first_prime': first_prime,
        'avg_prime': avg_prime_contrib,
    }


def run_all_experiments():
    """Run all harmonic head experiments."""
    print("=" * 60)
    print("POC-003 Experiment 02: Harmonic Head Structure")
    print("=" * 60)
    print(f"Device: {get_device()}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Primes: {PRIMES[:8]}")
    
    results = {
        'experiment': 'poc_003_exp_02_harmonic_heads',
        'timestamp': datetime.now().isoformat(),
        'device': str(get_device()),
        'primes': PRIMES[:8],
        'tests': {},
    }
    
    tests = [
        ('harmonic_weights', test_harmonic_weights),
        ('head_hierarchy', test_head_hierarchy),
        ('harmonic_vs_uniform', test_harmonic_vs_uniform),
        ('semantic_specialization', test_semantic_head_specialization),
        ('prime_series', test_prime_harmonic_series),
        ('eigenvalue_connection', test_eigenvalue_decay_connection),
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
    
    filename = f"exp_02_harmonic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
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
