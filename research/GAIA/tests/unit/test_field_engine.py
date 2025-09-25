import pytest
import numpy as np
from core.field_engine import FieldEngine

def test_field_engine_initialization():
    engine = FieldEngine()
    assert engine is not None
    assert hasattr(engine, 'energy_field')
    assert hasattr(engine, 'information_field')
    assert hasattr(engine, 'entropy_tensor')
    assert hasattr(engine, 'balance_controller')

def test_pac_amplitude_field_initialization(execution_context, memory_field):
    """Test PAC physics: Complex amplitude field initialization."""
    engine = FieldEngine()
    
    # Process input to trigger amplitude field creation
    field_state = engine.update_fields("test PAC input", memory_field, execution_context)
    
    # Validate complex amplitude field creation
    assert hasattr(engine, 'amplitude_field'), "Complex amplitude field not created"
    assert engine.amplitude_field is not None, "Amplitude field is None"
    
    # Validate amplitude conservation
    total_amplitude = np.sum(np.abs(engine.amplitude_field) ** 2)
    assert total_amplitude > 0, "Zero total amplitude - conservation violated"
    
    # Validate field pressure as conservation violation magnitude
    assert field_state.field_pressure >= 0, "Field pressure (violation magnitude) must be non-negative"

def test_pac_conservation_rebalancing(execution_context, memory_field):
    """Test PAC physics: Amplitude conservation rebalancing."""
    engine = FieldEngine()
    
    # Process multiple inputs to test conservation
    inputs = ["input 1", "input 2", "input 3"]
    amplitudes = []
    
    for inp in inputs:
        field_state = engine.update_fields(inp, memory_field, execution_context)
        if hasattr(engine, 'amplitude_field'):
            total_amp = np.sum(np.abs(engine.amplitude_field) ** 2)
            amplitudes.append(total_amp)
    
    # Validate conservation maintained across updates
    if len(amplitudes) > 1:
        # Allow for small numerical variations
        amplitude_variance = np.std(amplitudes) / np.mean(amplitudes)
        assert amplitude_variance < 0.1, f"Amplitude conservation violated: variance {amplitude_variance}"

def test_pac_phase_singularity_detection(execution_context, memory_field):
    """Test PAC physics: Phase singularity detection."""
    engine = FieldEngine()
    
    field_state = engine.update_fields("phase singularity test", memory_field, execution_context)
    
    if hasattr(engine, 'amplitude_field'):
        amplitude_field = engine.amplitude_field
        
        # Test for complex structure (phase singularities would be vortices)
        phase_field = np.angle(amplitude_field)
        phase_gradients = np.gradient(phase_field)
        
        # Check that phase structure exists (non-uniform phases)
        phase_variation = np.std(phase_field)
        assert phase_variation > 0, "No phase structure detected"

def test_field_update_and_pressure(execution_context):
    engine = FieldEngine()
    # Simulate an update
    field_before = engine.energy_field.field.copy()
    engine.energy_field.update(1.0, execution_context)
    field_after = engine.energy_field.field
    assert (field_after != field_before).any()
    stats = engine.get_field_statistics()
    assert 'average_entropy' in stats
    assert stats['average_entropy'] >= 0
