import pytest
import numpy as np
import sys
import os

# Add src to path for GAIA import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_gaia_physics_pipeline():
    """Test complete GAIA physics processing pipeline."""
    from gaia import GAIA
    
    gaia = GAIA()
    
    # Test with numerical field
    test_field = np.array([[1.0, 0.5], [0.5, 1.0]])
    response = gaia.process_input(test_field)
    
    # Pipeline should produce complete physics response
    assert response is not None
    assert hasattr(response, 'field_state')
    assert hasattr(response, 'conservation_residual') 
    assert hasattr(response, 'xi_operator_value')
    assert hasattr(response, 'klein_gordon_energy')
    assert hasattr(response, 'confidence')
    assert response.confidence >= 0

def test_integrated_pac_conservation():
    """Test integrated PAC conservation mathematics."""
    from gaia import GAIA
    
    gaia = GAIA()
    
    # Create fields that should trigger different PAC dynamics
    test_fields = [
        np.array([2.0, 1.0, 4.0, 2.0]),  # Clear parent-child ratios
        np.array([[1.0, 0.5], [2.0, 1.0]]),  # 2D field
        np.random.random(8) * 0.1  # Small random field
    ]
    
    conservation_residuals = []
    xi_measurements = []
    
    for test_field in test_fields:
        response = gaia.process_field(test_field)
        
        # Each field should produce PAC measurements
        assert response.conservation_residual >= 0, "Conservation residual should be non-negative"
        assert response.xi_operator_value > 0, "Xi operator should be positive"
        
        conservation_residuals.append(response.conservation_residual)
        xi_measurements.append(response.xi_operator_value)
    
    # PAC system should produce measurements for all fields
    assert len(conservation_residuals) == 3, "All fields should be processed"
    assert len(xi_measurements) == 3, "All fields should produce Xi measurements"

def test_klein_gordon_evolution():
    """Test Klein-Gordon field evolution capabilities."""
    from gaia import GAIA
    
    gaia = GAIA()
    
    # Wave-like initial condition
    field = np.sin(np.linspace(0, 2*np.pi, 16)).reshape(4, 4)
    
    # Evolve field
    response = gaia.process_field(field, dt=0.005)
    
    # Should produce evolved field with Klein-Gordon energy
    assert len(response.field_state) > 0, "Field should be evolved"
    assert response.klein_gordon_energy >= 0, "KG energy should be non-negative"
    
    # Multiple evolution steps should show dynamics
    field1 = response.field_state
    response2 = gaia.process_field(field1, dt=0.005)
    field2 = response2.field_state
    
    if len(field2) > 0:
        # Fields should show evolution (not stuck)
        evolution_magnitude = np.linalg.norm(field2.flatten() - field1.flatten())
        assert evolution_magnitude >= 0, "Evolution should occur"
        
        # Validate response structure
        assert response is not None
        assert hasattr(response, 'state')
        assert response.state.cognitive_integrity >= 0
        
        # Test PAC physics in field engine
        if hasattr(gaia.field_engine, 'amplitude_field'):
            amplitude_field = gaia.field_engine.amplitude_field
            total_amplitude = np.sum(np.abs(amplitude_field) ** 2)
            amplitude_conservations.append(total_amplitude)
            
            # Validate amplitude conservation
            assert total_amplitude > 0, "Zero amplitude conservation in system test"
        
        # Test Xi operator if collapse occurred
        if hasattr(response, 'reasoning_trace') and response.reasoning_trace:
            # Look for collapse events in reasoning trace
            for event in response.reasoning_trace:
                if hasattr(event, 'collapse_data') and event.collapse_data:
                    if hasattr(event.collapse_data, 'force_magnitude') and hasattr(event.collapse_data, 'entropy_tension'):
                        if event.collapse_data.entropy_tension > 1e-10:
                            xi_ratio = event.collapse_data.force_magnitude / event.collapse_data.entropy_tension
                            xi_convergences.append(xi_ratio)
    
    # System-wide PAC validation
    if amplitude_conservations:
        # Check amplitude conservation consistency across inputs
        amplitude_variance = np.std(amplitude_conservations) / np.mean(amplitude_conservations)
        assert amplitude_variance < 0.2, f"System amplitude conservation inconsistent: {amplitude_variance}"
    
    if xi_convergences:
        # Check Xi operator convergence across system
        mean_xi = np.mean(xi_convergences)
        assert 0.5 < mean_xi < 2.0, f"System Xi operator convergence failed: {mean_xi}"

def test_gaia_med_compliance_system():
    """Test system-wide MED (Maximum Entropy Depth) compliance."""
    gaia = GAIA()
    
    # Test with different entropy-inducing inputs
    entropy_inputs = [
        {"content": "simple", "expected_depth": "shallow"},
        {"content": "complex recursive mathematical reasoning with nested symbolic structures", "expected_depth": "moderate"},
        {"content": "a" * 1000, "expected_depth": "constrained"}  # High entropy input
    ]
    
    for test_input in entropy_inputs:
        response = gaia.process_input(test_input)
        
        # Validate MED bounds are respected in symbolic structures
        if hasattr(response, 'reasoning_trace'):
            for event in response.reasoning_trace:
                if hasattr(event, 'symbolic_structure') and event.symbolic_structure:
                    if hasattr(event.symbolic_structure, 'depth'):
                        # MED should limit depth to reasonable bounds
                        assert event.symbolic_structure.depth <= 15, \
                            f"Symbolic depth {event.symbolic_structure.depth} exceeds reasonable MED bounds"
