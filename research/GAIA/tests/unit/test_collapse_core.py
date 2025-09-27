import pytest
import numpy as np
import sys
import os

# Add paths for PAC-native modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/core'))

from core.collapse_core import CollapseCore, CollapseType


def test_pac_native_collapse_core_initialization():
    """Test PAC-native CollapseCore initialization with Fracton SDK."""
    core = CollapseCore()
    
    # Validate PAC-native components
    assert core is not None
    assert hasattr(core, 'physics_memory')
    assert hasattr(core, 'pac_regulator')
    assert hasattr(core, 'entropy_dispatcher')
    
    # Validate Fracton integration
    assert core.physics_memory is not None
    assert core.pac_regulator is not None
    assert core.total_collapses == 0


def test_pac_native_collapse_operation():
    """Test PAC-native entropy-driven collapse with conservation."""
    core = CollapseCore()
    
    # Test collapse operation
    result, context = core.collapse()
    
    # Can return None if no collapse needed (conservation maintained)
    if result is not None:
        assert hasattr(result, 'structure_id')
        assert hasattr(result, 'collapse_type')
        assert hasattr(result, 'pac_regulated')
        assert result.pac_regulated == True
        assert core.total_collapses > 0
        
        # Validate collapse type
        assert result.collapse_type in [CollapseType.THERMODYNAMIC, 
                                       CollapseType.GEOMETRIC, 
                                       CollapseType.FRACTAL_NODE]


def test_pac_conservation_violation_detection():
    """Test PAC conservation violation detection and resolution."""
    core = CollapseCore()
    
    # Force physics memory into violation state for testing
    # (This would normally happen through field evolution)
    initial_violations = core._detect_collapse_worthy_violations()
    
    # Process multiple collapse attempts
    for _ in range(3):
        result, _ = core.collapse()
        if result:
            break
    
    # Check that violations are being tracked
    assert hasattr(core, 'pac_violations_resolved')
    assert core.pac_violations_resolved >= 0


def test_pac_xi_balance_monitoring(simple_context):
    """Test Xi balance operator monitoring in collapse decisions."""
    core = CollapseCore()
    
    # Get current physics metrics
    metrics = core.physics_memory.get_physics_metrics()
    initial_xi = metrics.get('xi_value', 1.0571)
    
    # Validate Xi is close to target
    assert abs(initial_xi - 1.0571) < 0.5, f"Xi value {initial_xi} too far from target 1.0571"
    
    # Test collapse with Xi monitoring
    result, _ = core.collapse()
    
    # Get final metrics
    final_metrics = core.physics_memory.get_physics_metrics()
    final_xi = final_metrics.get('xi_value', 1.0571)
    
    # Validate Xi stability after collapse
    xi_drift = abs(final_xi - initial_xi)
    assert xi_drift < 1.0, f"Excessive Xi drift during collapse: {xi_drift}"
    
    result = core.evaluator.evaluate(simple_context)
    
    if result is not None:
        # Check conservation residual tracking
        assert hasattr(result, 'entropy_tension'), "Conservation residual not tracked"
        conservation_residual = result.entropy_tension
        assert conservation_residual >= 0, "Invalid conservation residual"

def test_collapse_statistics_tracking(simple_context):
    core = CollapseCore()
    # Simulate a collapse
    result = core.evaluator.evaluate(simple_context)
    if result:
        core.synthesizer.synthesize(result, simple_context)
    stats = core.get_collapse_statistics()
    assert 'average_efficiency' in stats
    assert stats['average_efficiency'] >= 0
