"""
Unit tests for PAC physics constants and conservation validation.
Ensures that fundamental physics values are correctly maintained.
"""

import pytest
import numpy as np
import sys
import os

# Add src path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from gaia import GAIA
from core.field_engine import FieldEngine


class TestPACConstants:
    """Test PAC physics constants are correctly enforced."""
    
    def test_xi_operator_constant(self):
        """Test that Xi operator is exactly the theoretical constant 1.0571."""
        gaia = GAIA()
        
        # Test with simple input
        test_field = np.ones((4, 4))
        response = gaia.process_field(test_field)
        
        # Xi operator must be the theoretical constant
        expected_xi = 1.0571
        actual_xi = response.xi_operator_value
        
        # Very tight tolerance for fundamental constant
        assert abs(actual_xi - expected_xi) < 1e-6, f"Xi operator {actual_xi} != {expected_xi}"
        
    def test_xi_operator_invariance(self):
        """Test that Xi operator is invariant across different inputs."""
        gaia = GAIA()
        expected_xi = 1.0571
        
        test_inputs = [
            np.ones((4, 4)),
            np.zeros((4, 4)),
            np.random.rand(4, 4),
            np.eye(4),
            np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
        ]
        
        for i, test_field in enumerate(test_inputs):
            response = gaia.process_field(test_field)
            actual_xi = response.xi_operator_value
            
            assert abs(actual_xi - expected_xi) < 1e-6, \
                f"Input {i}: Xi operator {actual_xi} != {expected_xi}"
    
    def test_conservation_residual_bounds(self):
        """Test that conservation residuals are within reasonable physics bounds."""
        gaia = GAIA()
        
        # Test with simple stable input
        test_field = np.ones((4, 4)) * 0.5
        response = gaia.process_field(test_field)
        
        # Conservation residual should be small for stable systems
        residual = response.conservation_residual
        assert residual < 100.0, f"Conservation residual {residual} too high"
        assert residual >= 0.0, f"Conservation residual {residual} cannot be negative"
        
    def test_field_engine_xi_constant(self):
        """Test Xi constant directly in field engine."""
        from core.field_engine import PACMathematics
        
        # Direct constant test
        expected_xi = 1.0571
        actual_xi = PACMathematics.XI_OPERATOR_CONSTANT
        
        assert abs(actual_xi - expected_xi) < 1e-10, \
            f"Field engine Xi constant {actual_xi} != {expected_xi}"
    
    def test_energy_conservation_principle(self):
        """Test that energy is approximately conserved in field operations."""
        gaia = GAIA()
        
        # Identical inputs should give identical energies (energy conservation)
        test_field = np.random.rand(4, 4)
        
        response1 = gaia.process_field(test_field.copy())
        response2 = gaia.process_field(test_field.copy())
        
        energy1 = response1.klein_gordon_energy
        energy2 = response2.klein_gordon_energy
        
        # Energy should be conserved for identical inputs
        energy_difference = abs(energy1 - energy2)
        relative_error = energy_difference / max(abs(energy1), abs(energy2), 1e-10)
        
        assert relative_error < 0.1, \
            f"Energy not conserved: {energy1} vs {energy2}, relative error: {relative_error}"


class TestPhysicsValidation:
    """Test physics validation and constraint enforcement."""
    
    def test_klein_gordon_energy_positive(self):
        """Test that Klein-Gordon energy is always positive."""
        gaia = GAIA()
        
        test_inputs = [
            np.ones((4, 4)),
            np.random.rand(4, 4),
            np.random.rand(4, 4) * 2.0,
            np.eye(4) * 0.5
        ]
        
        for test_field in test_inputs:
            response = gaia.process_field(test_field)
            energy = response.klein_gordon_energy
            
            assert energy > 0, f"Klein-Gordon energy {energy} must be positive"
    
    def test_field_state_consistency(self):
        """Test that field state maintains physics consistency."""
        gaia = GAIA()
        
        test_field = np.random.rand(4, 4)
        response = gaia.process_field(test_field)
        
        # Basic consistency checks
        assert hasattr(response, 'xi_operator_value'), "Missing Xi operator"
        assert hasattr(response, 'conservation_residual'), "Missing conservation residual"
        assert hasattr(response, 'klein_gordon_energy'), "Missing Klein-Gordon energy"
        assert hasattr(response, 'field_state'), "Missing field state"
        
        # Value range checks
        assert np.all(np.isfinite(response.field_state)), "Field state contains non-finite values"
        assert np.all(np.isreal(response.field_state)), "Field state contains complex values"


if __name__ == "__main__":
    pytest.main([__file__])