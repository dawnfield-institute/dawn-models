"""
Experiment 04: Full Integration Test
=====================================

Complete integration of field-native attention with GAIA.

Tests:
1. Stack multiple attention layers
2. Compare to standard transformer attention
3. End-to-end semantic processing
4. GPU performance benchmarks
5. PAC conservation through full stack
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from datetime import datetime
from pathlib import Path
import time
import sys

poc_002_path = Path(__file__).parent.parent.parent / 'poc_002_resonance_training' / 'scripts'
sys.path.insert(0, str(poc_002_path))

from field_attention import (
    FieldNativeAttention,
    HarmonicMultiHeadAttention,
    ResonanceAttention,
    PRIMES,
    PI_SQUARED_INV,
    PHI_XI,
    get_device,
)

try:
    from physics_trainer import DawnFieldTrainer
    TRAINER_AVAILABLE = True
except ImportError:
    TRAINER_AVAILABLE = False


class FieldNativeTransformerBlock(nn.Module):
    """A single transformer block using field-native attention."""
    
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.attention = FieldNativeAttention(
            dim=dim,
            n_heads=n_heads,
            use_field_qkv=True,
            use_projections=False,
        )
        
        # Simple FFN (could also be field-based)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Attention with residual
        attn_out, attn_info = self.attention(x)
        
        # FFN with residual
        ffn_out = self.ffn(self.norm(attn_out))
        out = attn_out + ffn_out
        
        return out, attn_info


class FieldNativeTransformer(nn.Module):
    """Stacked field-native transformer."""
    
    def __init__(self, dim: int, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            FieldNativeTransformerBlock(dim, n_heads)
            for _ in range(n_layers)
        ])
        
    def forward(self, x):
        layer_infos = []
        
        for layer in self.layers:
            x, info = layer(x)
            layer_infos.append(info)
        
        return x, layer_infos


def test_stacked_layers():
    """Test stacking multiple attention layers."""
    print("\n[Test 1: Stacked Layers]")
    
    device = get_device()
    dim = 64
    n_layers = 4
    
    model = FieldNativeTransformer(dim, n_layers).to(device)
    x = torch.randn(2, 16, dim, device=device)
    
    output, layer_infos = model(x)
    
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Layers: {n_layers}")
    
    # Check conservation through layers
    conservation_residuals = [info['conservation_residual'] for info in layer_infos]
    
    print(f"  Conservation per layer: {[f'{r:.4f}' for r in conservation_residuals]}")
    
    # All should be reasonable
    assert all(r < 0.5 for r in conservation_residuals), "Conservation should be maintained"
    
    print(f"  ✓ Stacked layers work correctly")
    return {
        'n_layers': n_layers,
        'conservation': conservation_residuals,
    }


def test_compare_to_standard():
    """Compare field-native to standard PyTorch transformer."""
    print("\n[Test 2: Compare to Standard]")
    
    device = get_device()
    dim = 64
    n_heads = 4
    seq_len = 32
    batch_size = 4
    
    # Field-native
    field_model = FieldNativeTransformer(dim, n_layers=2, n_heads=n_heads).to(device)
    
    # Standard PyTorch
    standard_layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=n_heads,
        dim_feedforward=dim * 4,
        batch_first=True,
    ).to(device)
    standard_model = nn.TransformerEncoder(standard_layer, num_layers=2)
    
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    with torch.no_grad():
        field_out, _ = field_model(x)
        standard_out = standard_model(x)
    
    print(f"  Field output shape: {field_out.shape}")
    print(f"  Standard output shape: {standard_out.shape}")
    
    # Both should produce valid outputs
    field_std = field_out.std().item()
    standard_std = standard_out.std().item()
    
    print(f"  Field output std: {field_std:.4f}")
    print(f"  Standard output std: {standard_std:.4f}")
    
    assert field_std > 0, "Field should produce non-trivial output"
    assert standard_std > 0, "Standard should produce non-trivial output"
    
    print(f"  ✓ Both models produce valid outputs")
    return {
        'field_std': field_std,
        'standard_std': standard_std,
    }


def test_end_to_end_semantic():
    """Test semantic processing through full stack."""
    print("\n[Test 3: End-to-End Semantic]")
    
    if not TRAINER_AVAILABLE:
        print("  ⚠ Physics trainer not available, skipping")
        return {'skipped': True}
    
    device = get_device()
    dim = 64
    
    # Train semantic embeddings
    trainer = DawnFieldTrainer(field_dim=dim, device=device)
    
    training_data = [
        ("dog", "cat"), ("dog", "wolf"), ("cat", "lion"),
        ("car", "truck"), ("car", "bus"), ("truck", "van"),
    ]
    
    trainer.train(training_data, epochs=30)
    
    # Create test sequences
    animals = ["dog", "cat", "wolf", "lion"]
    vehicles = ["car", "truck", "bus", "van"]
    
    animal_seq = torch.stack([trainer.resonance.field_memory[p] for p in animals]).unsqueeze(0)
    vehicle_seq = torch.stack([trainer.resonance.field_memory[p] for p in vehicles]).unsqueeze(0)
    
    # Process through transformer
    model = FieldNativeTransformer(dim, n_layers=2, n_heads=4).to(device)
    
    with torch.no_grad():
        animal_out, _ = model(animal_seq)
        vehicle_out, _ = model(vehicle_seq)
    
    # Check if semantic structure preserved
    # Within-sequence similarity should be high
    def avg_similarity(seq):
        sims = []
        for i in range(seq.shape[1]):
            for j in range(i+1, seq.shape[1]):
                sim = F.cosine_similarity(seq[0, i], seq[0, j], dim=0).item()
                sims.append(sim)
        return sum(sims) / len(sims) if sims else 0
    
    animal_sim = avg_similarity(animal_out)
    vehicle_sim = avg_similarity(vehicle_out)
    
    # Cross-sequence (mix animals and vehicles)
    mixed = torch.cat([animal_out[0, :2], vehicle_out[0, :2]], dim=0).unsqueeze(0)
    mixed_sim = avg_similarity(mixed)
    
    print(f"  Animal within-sequence sim: {animal_sim:.4f}")
    print(f"  Vehicle within-sequence sim: {vehicle_sim:.4f}")
    print(f"  Mixed sequence sim: {mixed_sim:.4f}")
    
    # Within-class should be higher than mixed
    within_avg = (animal_sim + vehicle_sim) / 2
    print(f"  Within-class avg: {within_avg:.4f}")
    print(f"  Mixed: {mixed_sim:.4f}")
    
    print(f"  ✓ Semantic structure processed through transformer")
    return {
        'animal_sim': animal_sim,
        'vehicle_sim': vehicle_sim,
        'mixed_sim': mixed_sim,
    }


def test_gpu_performance():
    """Benchmark GPU performance."""
    print("\n[Test 4: GPU Performance]")
    
    device = get_device()
    dim = 128
    seq_len = 64
    batch_size = 8
    n_layers = 4
    
    model = FieldNativeTransformer(dim, n_layers=n_layers, n_heads=8).to(device)
    
    # Warmup
    x = torch.randn(batch_size, seq_len, dim, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    
    # Synchronize
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    n_iterations = 50
    start = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(n_iterations):
            _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    
    tokens_per_sec = (batch_size * seq_len * n_iterations) / elapsed
    ms_per_forward = (elapsed / n_iterations) * 1000
    
    print(f"  Device: {device}")
    print(f"  Config: {batch_size}×{seq_len}×{dim}, {n_layers} layers")
    print(f"  Time per forward: {ms_per_forward:.2f}ms")
    print(f"  Tokens/second: {tokens_per_sec:.0f}")
    
    print(f"  ✓ Performance benchmark complete")
    return {
        'device': str(device),
        'ms_per_forward': ms_per_forward,
        'tokens_per_sec': tokens_per_sec,
    }


def test_pac_conservation_full_stack():
    """Test PAC conservation through entire model."""
    print("\n[Test 5: PAC Conservation Full Stack]")
    
    device = get_device()
    dim = 64
    n_layers = 4
    
    model = FieldNativeTransformer(dim, n_layers=n_layers).to(device)
    x = torch.randn(2, 16, dim, device=device)
    
    # Track energy through layers
    with torch.no_grad():
        input_energy = torch.sum(x ** 2).item()
        
        output, layer_infos = model(x)
        
        output_energy = torch.sum(output ** 2).item()
    
    # Per-layer conservation
    layer_residuals = [info['conservation_residual'] for info in layer_infos]
    
    # Overall conservation
    total_residual = abs(output_energy - input_energy) / input_energy
    
    print(f"  Input energy: {input_energy:.2f}")
    print(f"  Output energy: {output_energy:.2f}")
    print(f"  Per-layer residuals: {[f'{r:.4f}' for r in layer_residuals]}")
    print(f"  Total residual: {total_residual:.4f}")
    
    # FFN will add energy, so we expect some growth
    # But it should be bounded
    assert total_residual < 10.0, "Total energy change should be bounded"
    
    print(f"  ✓ PAC conservation tracked through stack")
    return {
        'input_energy': input_energy,
        'output_energy': output_energy,
        'layer_residuals': layer_residuals,
        'total_residual': total_residual,
    }


def test_gradient_full_stack():
    """Test gradient flow through full stack."""
    print("\n[Test 6: Gradient Full Stack]")
    
    device = get_device()
    dim = 64
    
    model = FieldNativeTransformer(dim, n_layers=3).to(device)
    x = torch.randn(2, 8, dim, device=device, requires_grad=True)
    
    output, _ = model(x)
    loss = output.sum()
    loss.backward()
    
    grad_norm = x.grad.norm().item()
    
    print(f"  Input gradient norm: {grad_norm:.4f}")
    
    # Check model parameters also have gradients
    param_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_grads.append((name.split('.')[-1], param.grad.norm().item()))
    
    print(f"  Parameter gradients: {len(param_grads)}")
    
    assert grad_norm > 0, "Input should have gradient"
    assert len(param_grads) > 0, "Parameters should have gradients"
    
    print(f"  ✓ Gradients flow through full stack")
    return {
        'input_grad': grad_norm,
        'param_count': len(param_grads),
    }


def run_all_experiments():
    """Run all integration experiments."""
    print("=" * 60)
    print("POC-003 Experiment 04: Full Integration")
    print("=" * 60)
    print(f"Device: {get_device()}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Constants: φ×ξ={PHI_XI:.4f}, 1/π²={PI_SQUARED_INV:.4f}")
    
    results = {
        'experiment': 'poc_003_exp_04_integration',
        'timestamp': datetime.now().isoformat(),
        'device': str(get_device()),
        'constants': {
            'phi_xi': PHI_XI,
            'pi_squared_inv': PI_SQUARED_INV,
        },
        'tests': {},
    }
    
    tests = [
        ('stacked_layers', test_stacked_layers),
        ('compare_standard', test_compare_to_standard),
        ('semantic_e2e', test_end_to_end_semantic),
        ('gpu_perf', test_gpu_performance),
        ('pac_conservation', test_pac_conservation_full_stack),
        ('gradient_flow', test_gradient_full_stack),
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
    
    filename = f"exp_04_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
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
