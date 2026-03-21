"""
Experiment 04: GAIA Integration Test
=====================================

POC-001: Test GPU-native encoding integrated with actual GAIA.

This experiment:
1. Creates a real PAC_GAIA instance
2. Patches it with our GPU encoder
3. Runs cognitive processing with GPU acceleration
4. Validates PAC conservation is maintained

Technical Requirements:
- PyTorch only for encoding (no numpy)
- GPU acceleration
- Real GAIA integration
"""

import torch
import sys
import time
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add paths - use absolute references
SCRIPT_DIR = Path(__file__).resolve().parent
POC_DIR = SCRIPT_DIR.parent  # poc_001_pattern_encoding
POC_ROOT = POC_DIR.parent  # proof_of_concepts  
GAIA_ROOT = POC_ROOT.parent  # gaia
GAIA_SRC = GAIA_ROOT / 'src'
FRACTON_DIR = GAIA_ROOT.parent.parent / 'fracton'

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(GAIA_SRC))
sys.path.insert(0, str(FRACTON_DIR))

print(f"GAIA_SRC: {GAIA_SRC}")
print(f"FRACTON: {FRACTON_DIR}")

from utils import (
    ExperimentResult, get_gpu_info, get_results_dir, 
    generate_experiment_id, DEVICE
)
from gaia_encoder import GAIAFieldEncoder, create_gaia_encoder, patch_gaia_encoding


def test_encoder_standalone() -> Dict[str, Any]:
    """Test 1: Encoder works standalone."""
    print("\n=== Test 1: Standalone Encoder ===")
    
    encoder = create_gaia_encoder()
    
    tests_passed = 0
    
    # Test string encoding
    field = encoder.encode("Hello GAIA!")
    if field.device.type in ['cuda', 'cpu'] and field.shape == (64, 64):
        print(f"  String encoding: ✓ shape={field.shape}, device={field.device}")
        tests_passed += 1
    else:
        print(f"  String encoding: ✗ shape={field.shape}, device={field.device}")
    
    # Test energy
    energy = encoder.get_field_energy(field)
    if energy > 0:
        print(f"  Energy calculation: ✓ energy={energy:.4f}")
        tests_passed += 1
    else:
        print(f"  Energy calculation: ✗ energy={energy}")
    
    # Test conservation
    residual = encoder.check_conservation(field)
    if residual < 0.01:
        print(f"  Conservation check: ✓ residual={residual:.2e}")
        tests_passed += 1
    else:
        print(f"  Conservation check: ✗ residual={residual}")
    
    return {
        'test': 'encoder_standalone',
        'tests_passed': tests_passed,
        'tests_total': 3,
        'success': tests_passed == 3
    }


def test_gaia_import() -> Dict[str, Any]:
    """Test 2: GAIA can be imported."""
    print("\n=== Test 2: GAIA Import ===")
    
    try:
        from gaia import PAC_GAIA, PAC_GAIAConfig
        print(f"  Import PAC_GAIA: ✓")
        print(f"  Import PAC_GAIAConfig: ✓")
        return {
            'test': 'gaia_import',
            'success': True,
            'error': None
        }
    except Exception as e:
        print(f"  Import failed: {e}")
        return {
            'test': 'gaia_import', 
            'success': False,
            'error': str(e)
        }


def test_gaia_creation() -> Dict[str, Any]:
    """Test 3: GAIA instance can be created."""
    print("\n=== Test 3: GAIA Creation ===")
    
    try:
        from gaia import PAC_GAIA, PAC_GAIAConfig
        
        config = PAC_GAIAConfig(
            memory_coherence=1.0,
            symbolic_structures=10,
            active_signals=5,
            cognitive_integrity=0.95,
            processing_cycles=0,
            total_collapses=0,
            resonance_patterns=3,
            field_dimensions=(64, 64)
        )
        
        gaia = PAC_GAIA(config)
        print(f"  Created PAC_GAIA instance: ✓")
        
        return {
            'test': 'gaia_creation',
            'success': True,
            'config': {
                'field_dimensions': config.field_dimensions,
                'xi_target': config.xi_target
            }
        }
    except Exception as e:
        print(f"  Creation failed: {e}")
        return {
            'test': 'gaia_creation',
            'success': False,
            'error': str(e)
        }


def test_gaia_patching() -> Dict[str, Any]:
    """Test 4: GAIA can be patched with GPU encoder."""
    print("\n=== Test 4: GAIA GPU Patching ===")
    
    try:
        from gaia import PAC_GAIA, PAC_GAIAConfig
        
        config = PAC_GAIAConfig(
            memory_coherence=1.0,
            symbolic_structures=10,
            active_signals=5,
            cognitive_integrity=0.95,
            processing_cycles=0,
            total_collapses=0,
            resonance_patterns=3,
            field_dimensions=(64, 64)
        )
        
        gaia = PAC_GAIA(config)
        
        # Patch with GPU encoder
        patch_gaia_encoding(gaia)
        
        # Check encoder is attached
        if hasattr(gaia, '_gpu_encoder'):
            print(f"  GPU encoder attached: ✓")
            print(f"  Encoder device: {gaia._gpu_encoder.device}")
            return {
                'test': 'gaia_patching',
                'success': True,
                'device': str(gaia._gpu_encoder.device)
            }
        else:
            print(f"  GPU encoder missing: ✗")
            return {
                'test': 'gaia_patching',
                'success': False,
                'error': 'Encoder not attached'
            }
            
    except Exception as e:
        print(f"  Patching failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'test': 'gaia_patching',
            'success': False,
            'error': str(e)
        }


def test_gaia_processing() -> Dict[str, Any]:
    """Test 5: GAIA can process with GPU encoding."""
    print("\n=== Test 5: GAIA Processing ===")
    
    try:
        from gaia import PAC_GAIA, PAC_GAIAConfig
        
        config = PAC_GAIAConfig(
            memory_coherence=1.0,
            symbolic_structures=10,
            active_signals=5,
            cognitive_integrity=0.95,
            processing_cycles=0,
            total_collapses=0,
            resonance_patterns=3,
            field_dimensions=(64, 64)
        )
        
        gaia = PAC_GAIA(config)
        patch_gaia_encoding(gaia)
        
        # Process some inputs
        test_inputs = [
            "Hello GAIA!",
            "Pattern encoding test",
            "Physics-governed cognition"
        ]
        
        results = []
        for input_text in test_inputs:
            start = time.perf_counter()
            response = gaia.process_cognition(input_text)
            elapsed = (time.perf_counter() - start) * 1000
            
            results.append({
                'input': input_text,
                'processing_time_ms': elapsed,
                'confidence': response.confidence,
                'conservation_residual': response.conservation_residual,
                'xi_value': response.xi_operator_value
            })
            
            print(f"  '{input_text[:20]}...': "
                  f"confidence={response.confidence:.3f}, "
                  f"xi={response.xi_operator_value:.4f}, "
                  f"time={elapsed:.1f}ms")
        
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        return {
            'test': 'gaia_processing',
            'success': True,
            'results': results,
            'avg_confidence': avg_confidence
        }
        
    except Exception as e:
        print(f"  Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'test': 'gaia_processing',
            'success': False,
            'error': str(e)
        }


def test_encoding_speed_comparison() -> Dict[str, Any]:
    """Test 6: Compare GPU vs original encoding speed."""
    print("\n=== Test 6: Encoding Speed Comparison ===")
    
    try:
        from gaia import PAC_GAIA, PAC_GAIAConfig
        import numpy as np
        import hashlib
        
        config = PAC_GAIAConfig(
            memory_coherence=1.0,
            symbolic_structures=10,
            active_signals=5,
            cognitive_integrity=0.95,
            processing_cycles=0,
            total_collapses=0,
            resonance_patterns=3,
            field_dimensions=(64, 64)
        )
        
        gaia = PAC_GAIA(config)
        
        # Original numpy encoding (copy from gaia.py)
        def original_encode(input_data, field_dims=(64, 64)):
            hash_bytes = hashlib.sha256(input_data.encode()).digest()
            field_size = int(np.prod(field_dims))
            hash_ints = np.frombuffer(hash_bytes[:field_size*4], dtype=np.float32)
            if len(hash_ints) >= field_size:
                field = hash_ints[:field_size].reshape(field_dims)
            else:
                padded = np.zeros(field_size)
                padded[:len(hash_ints)] = hash_ints
                field = padded.reshape(field_dims)
            return (field - np.mean(field)) / (np.std(field) + 1e-8)
        
        # GPU encoder
        gpu_encoder = create_gaia_encoder()
        
        # Warm up GPU
        for _ in range(10):
            gpu_encoder.encode("warmup")
        
        # Benchmark with larger field (256x256)
        large_encoder = create_gaia_encoder(field_dims=(256, 256))
        for _ in range(10):
            large_encoder.encode("warmup")
        
        test_strings = [f"Test string number {i}" for i in range(100)]
        
        # Original 64x64
        start = time.perf_counter()
        for s in test_strings:
            original_encode(s)
        original_time = (time.perf_counter() - start) * 1000
        
        # GPU 64x64 (after warmup)
        start = time.perf_counter()
        for s in test_strings:
            gpu_encoder.encode(s)
        torch.cuda.synchronize()  # Ensure all GPU ops complete
        gpu_time = (time.perf_counter() - start) * 1000
        
        # GPU 256x256 (where GPU shines)
        def original_encode_large(input_data, field_dims=(256, 256)):
            hash_bytes = hashlib.sha256(input_data.encode()).digest()
            field_size = int(np.prod(field_dims))
            padded = np.zeros(field_size)
            hash_ints = np.frombuffer(hash_bytes, dtype=np.float32)
            padded[:len(hash_ints)] = hash_ints
            field = padded.reshape(field_dims)
            return (field - np.mean(field)) / (np.std(field) + 1e-8)
        
        start = time.perf_counter()
        for s in test_strings:
            original_encode_large(s)
        original_large_time = (time.perf_counter() - start) * 1000
        
        start = time.perf_counter()
        for s in test_strings:
            large_encoder.encode(s)
        torch.cuda.synchronize()
        gpu_large_time = (time.perf_counter() - start) * 1000
        
        speedup_64 = original_time / gpu_time if gpu_time > 0 else 0
        speedup_256 = original_large_time / gpu_large_time if gpu_large_time > 0 else 0
        
        print(f"  64x64 field:")
        print(f"    Numpy: {original_time:.2f}ms, GPU: {gpu_time:.2f}ms, Speedup: {speedup_64:.2f}x")
        print(f"  256x256 field:")
        print(f"    Numpy: {original_large_time:.2f}ms, GPU: {gpu_large_time:.2f}ms, Speedup: {speedup_256:.2f}x")
        
        # Note: Small ops are CPU-bound, GPU wins on larger fields
        print(f"\n  Note: GPU benefits increase with field size and batch processing")
        
        return {
            'test': 'speed_comparison',
            'success': True,
            'speedup_64': speedup_64,
            'speedup_256': speedup_256,
            'original_64_ms': original_time,
            'gpu_64_ms': gpu_time,
            'original_256_ms': original_large_time,
            'gpu_256_ms': gpu_large_time
        }
        
    except Exception as e:
        print(f"  Speed test failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'test': 'speed_comparison',
            'success': False,
            'error': str(e)
        }


def main():
    """Run GAIA integration tests."""
    print("=" * 60)
    print("POC-001 Experiment 04: GAIA Integration")
    print("=" * 60)
    
    gpu_info = get_gpu_info()
    print(f"\nDevice: {DEVICE}")
    if gpu_info['available']:
        print(f"GPU: {gpu_info['device_name']}")
    
    all_results = []
    
    # Run tests in order of dependency
    all_results.append(test_encoder_standalone())
    all_results.append(test_gaia_import())
    
    # Only continue if GAIA imports
    if all_results[-1]['success']:
        all_results.append(test_gaia_creation())
        
        if all_results[-1]['success']:
            all_results.append(test_gaia_patching())
            
            if all_results[-1]['success']:
                all_results.append(test_gaia_processing())
                all_results.append(test_encoding_speed_comparison())
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in all_results if r['success'])
    total = len(all_results)
    
    for result in all_results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"  {result['test']}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    # Save results
    experiment = ExperimentResult(
        experiment_id=generate_experiment_id("exp04_gaia_integration"),
        timestamp=datetime.now().isoformat(),
        device=str(DEVICE),
        parameters={
            'field_dims': (64, 64),
            'xi_target': 1.0571
        },
        encodings=[],
        metrics={
            'tests_passed': passed,
            'tests_total': total,
            'test_results': all_results
        },
        success=(passed >= 4),
        notes="GAIA integration test with GPU encoding"
    )
    
    results_path = experiment.save(get_results_dir())
    print(f"\nResults saved to: {results_path}")
    
    return passed >= 4


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
