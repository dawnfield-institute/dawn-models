import pytest
import numpy as np
import sys
import os

# Add src to path for GAIA import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_field_based_pattern_recognition():
    """Test GAIA recognizes different field patterns through physics."""
    from gaia import GAIA
    
    gaia = GAIA()
    
    # Structured field (low entropy pattern)
    structured_field = np.array([
        [1.0, 2.0, 4.0],
        [2.0, 4.0, 8.0],
        [4.0, 8.0, 16.0]
    ])  # Clear geometric progression
    
    # Random field (high entropy pattern) 
    np.random.seed(42)
    random_field = np.random.random((3, 3))
    
    resp1 = gaia.process_field(structured_field, dt=0.01)
    resp2 = gaia.process_field(random_field, dt=0.01)
    
    # Different patterns should produce different physics responses
    conservation_diff = abs(resp1.conservation_residual - resp2.conservation_residual)
    energy_diff = abs(resp1.klein_gordon_energy - resp2.klein_gordon_energy)
    
    # At least one physics metric should distinguish the patterns
    assert conservation_diff > 1e-12 or energy_diff > 1e-12, "Different patterns should produce different physics"
    
    # Both should produce valid physics measurements
    assert resp1.conservation_residual >= 0, "Structured field should have valid conservation"
    assert resp2.conservation_residual >= 0, "Random field should have valid conservation"

def test_field_evolution_adaptation():
    """Test field evolution adapts to different initial conditions."""
    try:
        from gaia import GAIA
        gaia = GAIA()
    except RuntimeError as e:
        if "PAC Engine" in str(e):
            pytest.skip("PAC Engine not available - skipping physics tests")
        else:
            raise
    
    # Low amplitude field
    low_amplitude = np.array([[0.1, 0.1], [0.1, 0.1]])
    
    # High amplitude field  
    high_amplitude = np.array([[1.0, 1.0], [1.0, 1.0]])
    
    resp_low = gaia.process_field(low_amplitude, dt=0.01)
    resp_high = gaia.process_field(high_amplitude, dt=0.01)
    
    # Different amplitudes should produce different Klein-Gordon energies
    energy_diff = abs(resp_low.klein_gordon_energy - resp_high.klein_gordon_energy)
    assert energy_diff > 1e-10, "Different amplitudes should produce different energies"
    
    # Field states should be different after evolution
    assert not np.allclose(resp_low.field_state, resp_high.field_state, atol=1e-10), "Different initial conditions should produce different evolution"
    assert response_high.state.processing_cycles >= response_low.state.processing_cycles
    assert response_high.structures_created >= response_low.structures_created

def test_pac_memory_conservation_behavior():
    """Test PAC physics: Memory conservation behavioral validation."""
    gaia = GAIA()
    
    # Test memory conservation with similar input patterns
    test_patterns = [
        {"pattern": "AB AB AB CD CD", "type": "repetitive"},
        {"pattern": "123 456 123 456", "type": "numeric_repetitive"},
        {"pattern": "hello world hello cosmos", "type": "semantic_repetitive"}
    ]
    
    memory_stabilities = []
    vortex_counts = []
    
    for pattern in test_patterns:
        response = gaia.process_input(pattern)
        
        # Check if PAC memory system is active
        if hasattr(gaia.superfluid_memory, 'total_vortices_detected'):
            vortex_count = gaia.superfluid_memory.total_vortices_detected
            vortex_counts.append(vortex_count)
            
            # Validate vortex detection (phase singularities)
            assert vortex_count >= 0, "Invalid vortex count in behavioral test"
        
        # Check memory amplitude field conservation
        if hasattr(gaia.superfluid_memory, 'memory_amplitude_field'):
            memory_field = gaia.superfluid_memory.memory_amplitude_field
            if memory_field is not None:
                total_memory_amplitude = np.sum(np.abs(memory_field) ** 2)
                memory_stabilities.append(total_memory_amplitude)
    
    # Behavioral validation: Similar patterns should show similar conservation
    if len(memory_stabilities) > 1:
        stability_variance = np.std(memory_stabilities) / np.mean(memory_stabilities)
        assert stability_variance < 0.5, f"Memory conservation behavior inconsistent: {stability_variance}"

def test_pac_xi_operator_behavioral_consistency():
    """Test PAC physics: Xi operator behavioral consistency across similar inputs."""
    gaia = GAIA()
    
    # Test with mathematically structured inputs
    math_patterns = [
        {"sequence": "2 4 8 16 32", "type": "geometric"},
        {"sequence": "1 3 5 7 9", "type": "arithmetic"},
        {"sequence": "1 1 2 3 5", "type": "fibonacci"}
    ]
    
    xi_ratios = []
    conservation_residuals = []
    
    for pattern in math_patterns:
        response = gaia.process_input(pattern)
        
        # Look for collapse events with Xi operator data
        if hasattr(response, 'reasoning_trace'):
            for event in response.reasoning_trace:
                if hasattr(event, 'collapse_data') and event.collapse_data:
                    if (hasattr(event.collapse_data, 'force_magnitude') and 
                        hasattr(event.collapse_data, 'entropy_tension')):
                        
                        force = event.collapse_data.force_magnitude
                        tension = event.collapse_data.entropy_tension
                        
                        conservation_residuals.append(tension)
                        
                        if tension > 1e-10:
                            xi_ratio = force / tension
                            xi_ratios.append(xi_ratio)
    
    # Behavioral validation: Mathematical patterns should trigger consistent Xi behavior
    if xi_ratios:
        mean_xi = np.mean(xi_ratios)
        xi_std = np.std(xi_ratios)
        
        # Xi operator should be reasonably stable for mathematical inputs
        assert 0.5 < mean_xi < 2.0, f"Xi operator behavior inconsistent: mean={mean_xi}"
        assert xi_std < 1.0, f"Xi operator too variable: std={xi_std}"
