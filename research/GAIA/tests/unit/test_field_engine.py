import pytest
import numpy as np
import sys
import os

# Add paths for PAC-native modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/core'))

from core.field_engine import FieldEngine


def test_pac_native_field_engine_initialization():
    """Test PAC-native FieldEngine initialization with Fracton SDK."""
    engine = FieldEngine(shape=(16, 16))
    
    # Validate PAC-native components
    assert engine is not None
    assert hasattr(engine, 'physics_memory')
    assert hasattr(engine, 'pac_regulator')
    assert hasattr(engine, 'physics_executor')
    
    # Validate Fracton integration
    assert engine.physics_memory is not None
    assert engine.pac_regulator is not None
    assert engine.shape == (16, 16)


def test_pac_native_field_update():
    """Test PAC-native field update with conservation enforcement."""
    engine = FieldEngine(shape=(8, 8))
    
    # Process input with PAC regulation
    test_input = "PAC conservation test input"
    field_state = engine.update_fields(test_input)
    
    # Validate PAC-native field state
    assert field_state is not None
    assert field_state.pac_regulated == True
    assert hasattr(field_state, 'conservation_residual')
    assert hasattr(field_state, 'xi_balance')
    assert hasattr(field_state, 'field_tensor')
    
    # Validate conservation metrics
    assert abs(field_state.xi_balance - 1.0571) < 0.1  # Balance operator close to target
    assert abs(field_state.conservation_residual) < 1.0  # Conservation residual reasonable


def test_pac_conservation_enforcement():
    """Test PAC conservation enforcement across multiple updates."""
    engine = FieldEngine(shape=(8, 8))
    
    # Process multiple inputs
    inputs = ["input 1", "input 2", "input 3"]
    conservation_residuals = []
    xi_values = []
    
    for inp in inputs:
        field_state = engine.update_fields(inp)
        conservation_residuals.append(abs(field_state.conservation_residual))
        xi_values.append(field_state.xi_balance)
    
    # Validate conservation maintained
    max_residual = max(conservation_residuals)
    assert max_residual < 1.0, f"Conservation violation too large: {max_residual}"
    
    # Validate Xi balance operator stability
    xi_variance = np.std(xi_values)
    assert xi_variance < 0.2, f"Xi balance operator unstable: variance {xi_variance}"
    
    # Validate trend toward target
    mean_xi = np.mean(xi_values)
    assert abs(mean_xi - 1.0571) < 0.2, f"Xi drift from target: {mean_xi} vs 1.0571"

def test_pac_phase_singularity_detection(simple_context, memory_field):
    """Test PAC physics: Phase singularity detection."""
    engine = FieldEngine()
    
    field_state = engine.update_fields("phase singularity test", memory_field, simple_context)
    
    if hasattr(engine, 'amplitude_field'):
        amplitude_field = engine.amplitude_field
        
        # Test for complex structure (phase singularities would be vortices)
        phase_field = np.angle(amplitude_field)
        phase_gradients = np.gradient(phase_field)
        
        # Check that phase structure exists (non-uniform phases)
        phase_variation = np.std(phase_field)
        assert phase_variation > 0, "No phase structure detected"

def test_field_update_and_pressure(simple_context):
    engine = FieldEngine()
    
    # Get initial field state 
    initial_field = engine.physics_memory.get('field_data')
    if initial_field is None:
        initial_field = np.zeros(np.prod(engine.shape))
    
    # Simulate a field update
    field_state = engine.update_fields("test update", context=simple_context)
    
    # Get updated field state
    updated_field = engine.physics_memory.get('field_data')
    
    # Verify field changed
    assert updated_field is not None
    assert not np.array_equal(initial_field, updated_field)
    
    # Verify field state properties
    assert hasattr(field_state, 'energy_density')
    assert hasattr(field_state, 'xi_balance')
    assert field_state.pac_regulated is True
    
    # Get physics metrics
    metrics = engine.physics_memory.get_physics_metrics()
    assert 'field_energy' in metrics
    assert metrics['field_energy'] >= 0
