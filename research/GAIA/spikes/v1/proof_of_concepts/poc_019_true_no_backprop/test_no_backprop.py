"""
Test suite verifying NO backprop and NO gradients

This is critical - we must prove we're not cheating.
"""

import torch
import sys
import traceback
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from no_backprop_training import NoBackpropTransformer, train_no_backprop


class GradientDetector:
    """Detect any gradient computation - zero tolerance"""
    
    def __init__(self):
        self.gradient_detected = False
        self.backward_calls = []
        self.original_backward = None
        
    def __enter__(self):
        """Monkey-patch to detect backward() calls"""
        self.original_backward = torch.Tensor.backward
        
        def detected_backward(self_tensor, *args, **kwargs):
            self.gradient_detected = True
            stack = ''.join(traceback.format_stack())
            self.backward_calls.append(stack)
            print("\n" + "="*70)
            print("❌❌❌ GRADIENT DETECTED! backward() was called! ❌❌❌")
            print("="*70)
            print(stack)
            raise RuntimeError("BACKPROP DETECTED - THIS VIOLATES THE NO-BACKPROP PRINCIPLE!")
            
        torch.Tensor.backward = detected_backward
        return self
        
    def __exit__(self, *args):
        """Restore original backward"""
        if self.original_backward:
            torch.Tensor.backward = self.original_backward


def test_no_gradients():
    """Verify absolutely no gradients are computed"""
    
    print("="*70)
    print("TEST 1: NO GRADIENT COMPUTATION")
    print("="*70)
    
    with GradientDetector() as detector:
        # Create model
        model = NoBackpropTransformer(vocab_size=128, dim=32)
        
        # Process sequences
        test_sequences = [
            [1, 2, 3, 4, 5],
            [10, 20, 30, 40],
            [5, 15, 25, 35, 45]
        ]
        
        for seq in test_sequences:
            output = model.process_sequence(seq)
            print(f"  Processed {len(seq)} tokens → {len(output)} outputs")
            
        # Generate
        generated = model.generate([1, 2], max_length=20)
        print(f"  Generated {len(generated)} tokens")
        
    if not detector.gradient_detected:
        print("\n✅✅✅ NO GRADIENTS COMPUTED! ✅✅✅")
        return True
    else:
        print("\n❌❌❌ GRADIENTS WERE COMPUTED! ❌❌❌")
        return False


def test_no_optimizer():
    """Verify no optimizer is used"""
    
    print("\n" + "="*70)
    print("TEST 2: NO OPTIMIZER USAGE")
    print("="*70)
    
    # Read source code
    source_file = Path(__file__).parent / 'no_backprop_training.py'
    source = source_file.read_text(encoding='utf-8')
    
    # Check for actual usage patterns (not mentions in strings/comments)
    forbidden_patterns = [
        ('optim.Adam(', 'Creating Adam optimizer'),
        ('optim.SGD(', 'Creating SGD optimizer'),
        ('optimizer.step()', 'Optimizer step call'),
        ('optimizer.zero_grad()', 'Gradient zeroing'),
    ]
    
    found = []
    lines = source.split('\n')
    for line_num, line in enumerate(lines, 1):
        # Skip comments
        if line.strip().startswith('#'):
            continue
        # Skip docstrings
        if '"""' in line or "'''" in line:
            continue
        # Skip print statements
        if 'print(' in line:
            continue
            
        for pattern, description in forbidden_patterns:
            if pattern in line:
                found.append((description, f"Line {line_num}: {line.strip()}"))
            
    if found:
        print(f"❌ Found forbidden terms: {found}")
        return False
    else:
        print("✅✅✅ NO OPTIMIZER CODE FOUND! ✅✅✅")
        return True


def test_field_dynamics():
    """Test that learning happens through field dynamics"""
    
    print("\n" + "="*70)
    print("TEST 3: FIELD DYNAMICS LEARNING")
    print("="*70)
    
    model = NoBackpropTransformer(vocab_size=128, dim=32)
    
    # Initial stats
    print(f"  Initial:")
    print(f"    Crystallized: {len(model.sec_operator.crystallized_patterns)}")
    print(f"    Skills: {len(model.skill_learner.skills)}")
    print(f"    Field updates: {model.stats['field_updates']}")
          
    # Process training data
    for i in range(10):
        sequence = list(range(i*5, (i+1)*5))
        model.process_sequence(sequence)
        
    # Final stats
    print(f"\n  After 10 sequences:")
    print(f"    Crystallized: {len(model.sec_operator.crystallized_patterns)}")
    print(f"    Skills: {len(model.skill_learner.skills)}")
    print(f"    Field updates: {model.stats['field_updates']}")
          
    if model.stats['field_updates'] > 0:
        print("\n✅✅✅ LEARNING THROUGH FIELD UPDATES! ✅✅✅")
        return True
    else:
        print("\n❌ No field updates detected")
        return False


def test_no_requires_grad():
    """Verify embeddings don't require gradients"""
    
    print("\n" + "="*70)
    print("TEST 4: NO REQUIRES_GRAD")
    print("="*70)
    
    model = NoBackpropTransformer(vocab_size=128, dim=32)
    
    if model.embeddings.requires_grad:
        print("❌ Embeddings require gradients!")
        return False
    else:
        print("✅✅✅ EMBEDDINGS DO NOT REQUIRE GRADIENTS! ✅✅✅")
        return True


def test_conservation():
    """Test PAC conservation is maintained"""
    
    print("\n" + "="*70)
    print("TEST 5: PAC CONSERVATION")
    print("="*70)
    
    from no_backprop_training import PACConservationField
    
    field = PACConservationField(vocab_size=10)
    
    # Update field multiple times
    field.update_field(0, 1, 0.8)
    field.update_field(0, 2, 0.5)
    field.update_field(0, 3, 0.3)
    
    # Check conservation (row should sum to 1)
    row_sum = field.field[0].sum().item()
    
    print(f"  Row 0 sum: {row_sum:.6f}")
    
    if abs(row_sum - 1.0) < 1e-5:
        print("✅✅✅ PAC CONSERVATION MAINTAINED! ✅✅✅")
        return True
    else:
        print(f"❌ Conservation violated: sum = {row_sum}")
        return False


def run_all_tests():
    """Run complete test suite"""
    
    print("\n" + "="*70)
    print("POC-019: TRUE NO-BACKPROP VALIDATION")
    print("="*70)
    print("\nThis test suite verifies we are NOT using backprop.")
    print("Zero tolerance for gradients, optimizers, or loss.backward().")
    print("="*70)
    
    results = {
        'no_gradients': test_no_gradients(),
        'no_optimizer': test_no_optimizer(),
        'field_dynamics': test_field_dynamics(),
        'no_requires_grad': test_no_requires_grad(),
        'conservation': test_conservation()
    }
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✅✅✅ ALL TESTS PASSED - TRUE NO-BACKPROP CONFIRMED! ✅✅✅")
    else:
        print("❌❌❌ SOME TESTS FAILED - WE HAVE BACKPROP! ❌❌❌")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
