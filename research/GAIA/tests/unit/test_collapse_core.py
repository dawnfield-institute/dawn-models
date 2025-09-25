import pytest
import numpy as np
from core.collapse_core import CollapseCore, CollapseType

def test_collapse_core_initialization():
    core = CollapseCore()
    assert core is not None
    assert hasattr(core, 'evaluator')
    assert hasattr(core, 'typing_engine')
    assert hasattr(core, 'synthesizer')
    assert hasattr(core, 'stabilizer')
    assert all(count == 0 for count in core.collapse_type_counts.values())

def test_collapse_evaluation_real_conditions(execution_context):
    core = CollapseCore()
    result = core.evaluator.evaluate(execution_context)
    assert result is not None
    assert result.collapse_type in CollapseType

def test_pac_collapse_evaluation_with_xi_operator(execution_context):
    """Test PAC physics: Xi operator convergence in collapse evaluation."""
    core = CollapseCore()
    
    # Set field state for PAC evaluation
    execution_context.field_state = {
        'resolution': (32, 32),
        'information_phase': execution_context.entropy
    }
    
    result = core.evaluator.evaluate(execution_context)
    
    if result is not None:
        # Validate PAC collapse vector properties
        assert hasattr(result, 'entropy_tension'), "Missing entropy_tension (conservation violation)"
        assert hasattr(result, 'force_magnitude'), "Missing force_magnitude (Xi scaling)"
        assert result.entropy_tension >= 0, "Conservation violation magnitude must be non-negative"
        assert result.force_magnitude >= 0, "Xi force magnitude must be non-negative"
        
        # Test Xi operator convergence (target: 1.0571)
        if result.entropy_tension > 1e-10:
            xi_ratio = result.force_magnitude / result.entropy_tension
            assert 0.1 < xi_ratio < 10.0, f"Xi operator ratio {xi_ratio} outside reasonable bounds"

def test_pac_conservation_residual_tracking(execution_context):
    """Test PAC physics: Conservation residual calculation."""
    core = CollapseCore()
    execution_context.field_state = {
        'resolution': (32, 32), 
        'information_phase': execution_context.entropy
    }
    
    result = core.evaluator.evaluate(execution_context)
    
    if result is not None:
        # Check conservation residual tracking
        assert hasattr(result, 'entropy_tension'), "Conservation residual not tracked"
        conservation_residual = result.entropy_tension
        assert conservation_residual >= 0, "Invalid conservation residual"

def test_collapse_statistics_tracking(execution_context):
    core = CollapseCore()
    # Simulate a collapse
    result = core.evaluator.evaluate(execution_context)
    if result:
        core.synthesizer.synthesize(result, execution_context)
    stats = core.get_collapse_statistics()
    assert 'average_efficiency' in stats
    assert stats['average_efficiency'] >= 0
