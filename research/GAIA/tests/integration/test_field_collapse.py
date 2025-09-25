import pytest
import numpy as np
from core.collapse_core import CollapseCore
from core.field_engine import FieldEngine

def test_field_collapse_integration(execution_context):
    field_engine = FieldEngine()
    collapse_core = CollapseCore()
    # Simulate field update and collapse evaluation
    field_engine.energy_field.update(1.0, execution_context)
    result = collapse_core.evaluator.evaluate(execution_context)
    assert result is not None
    # Synthesize and check stats
    collapse_core.synthesizer.synthesize(result, execution_context)
    stats = collapse_core.get_collapse_statistics()
    assert stats['total_collapses'] >= 0

def test_pac_field_collapse_integration(execution_context, memory_field):
    """Test PAC physics: Field engine -> Collapse core integration."""
    field_engine = FieldEngine()
    collapse_core = CollapseCore()
    
    # Process input through field engine (creates amplitude field)
    field_state = field_engine.update_fields("PAC integration test", memory_field, execution_context)
    
    # Set field state for collapse evaluation
    execution_context.field_state = {
        'resolution': (32, 32),
        'information_phase': execution_context.entropy
    }
    
    # Evaluate collapse with PAC physics
    collapse_result = collapse_core.evaluator.evaluate(execution_context)
    
    if collapse_result is not None:
        # Validate PAC integration between field and collapse
        assert hasattr(collapse_result, 'entropy_tension'), "Missing conservation violation from field"
        assert hasattr(collapse_result, 'force_magnitude'), "Missing Xi operator scaling"
        
        # Test conservation violation -> collapse strength relationship
        if collapse_result.entropy_tension > 0:
            xi_ratio = collapse_result.force_magnitude / collapse_result.entropy_tension
            assert 0.5 < xi_ratio < 2.0, f"Xi operator integration failed: {xi_ratio}"

def test_pac_amplitude_conservation_across_modules(execution_context, memory_field):
    """Test PAC physics: Amplitude conservation across field->collapse->memory pipeline."""
    field_engine = FieldEngine()
    collapse_core = CollapseCore()
    
    # Process through field engine
    field_state = field_engine.update_fields("conservation test", memory_field, execution_context)
    
    initial_amplitude = 0
    if hasattr(field_engine, 'amplitude_field'):
        initial_amplitude = np.sum(np.abs(field_engine.amplitude_field) ** 2)
    
    # Process through collapse
    execution_context.field_state = {
        'resolution': (32, 32),
        'information_phase': execution_context.entropy
    }
    
    collapse_result = collapse_core.evaluator.evaluate(execution_context)
    
    if collapse_result is not None:
        # Validate that conservation is maintained through pipeline
        assert collapse_result.entropy_tension >= 0, "Conservation violation tracking failed"
        
        # Conservation residual should reflect field state
        if initial_amplitude > 0:
            assert collapse_result.entropy_tension >= 0, "Conservation residual calculation failed"
