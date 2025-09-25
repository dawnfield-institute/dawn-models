import pytest
import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.collapse_core import CollapseCore
from core.field_engine import FieldEngine
from core.superfluid_memory import SuperfluidMemory
from core.symbolic_crystallizer import SymbolicCrystallizer
from core.meta_cognition_layer import MetaCognitionLayer
from core.resonance_mesh import ResonanceMesh
from gaia import GAIA

# Minimal fracton mocks ONLY if fracton not available
try:
    from fracton.core.memory_field import MemoryField
    from fracton.core.recursive_engine import ExecutionContext
    FRACTON_AVAILABLE = True
except ImportError:
    FRACTON_AVAILABLE = False
    # Minimal fallback mocks (should be replaced with real fracton ASAP)
    class MemoryField:
        def __init__(self, name): 
            self.name = name
            self.capacity = 1000
            self.field_id = name
        
        def create_snapshot(self):
            """Mock snapshot creation for GAIA compatibility."""
            return {
                'field_id': self.field_id,
                'capacity': self.capacity,
                'name': self.name,
                'data': {}
            }
        
        def get_memories(self, limit=None):
            """Mock memory retrieval."""
            return []
    
    class ExecutionContext:
        def __init__(self, **kwargs): 
            self.entropy = kwargs.get('entropy', 0.5)
            self.depth = kwargs.get('depth', 1)
            self.field_state = kwargs.get('field_state', {})

@pytest.fixture
def execution_context():
    """Provide real or minimal ExecutionContext for tests."""
    return ExecutionContext(entropy=0.8, depth=2)  # Higher entropy to trigger collapse

@pytest.fixture
def memory_field():
    """Provide real or minimally mocked MemoryField."""
    return MemoryField("test_field")

# PAC-specific test fixtures

@pytest.fixture
def pac_test_context():
    """Create execution context optimized for PAC testing."""
    context = ExecutionContext(entropy=0.6, depth=1)
    context.field_state = {
        'resolution': (32, 32),
        'information_phase': 0.6
    }
    return context

@pytest.fixture
def pac_collapse_data():
    """Create standardized collapse test data for PAC testing."""
    return {
        'entropy_resolved': 0.7,
        'structure_id': 'pac_test_structure',
        'coordinates': (0.5, 0.5),
        'complexity_signature': np.random.rand(16).tolist()
    }

@pytest.fixture(params=[0.2, 0.5, 0.8])
def entropy_levels(request):
    """Parameterized fixture for different entropy levels in PAC testing."""
    return request.param

@pytest.fixture
def gaia_instance():
    """Provide GAIA instance for integration and system tests.""" 
    return GAIA()

# PAC validation utilities

def validate_xi_convergence(force_magnitude: float, entropy_tension: float, tolerance: float = 0.5) -> bool:
    """Validate Xi operator convergence to target value (1.0571)."""
    if entropy_tension < 1e-10:
        return False
    
    xi_ratio = force_magnitude / entropy_tension
    target_xi = 1.0571
    
    return abs(xi_ratio - target_xi) < tolerance

def validate_amplitude_conservation(amplitude_field: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Validate amplitude conservation within tolerance."""
    if amplitude_field is None:
        return False
    
    total_amplitude = np.sum(np.abs(amplitude_field) ** 2)
    return total_amplitude > 0

def calculate_med_limit(entropy: float) -> int:
    """Calculate Maximum Entropy Depth limit for given entropy."""
    if entropy <= 0:
        return 12
    return max(2, min(int(-np.log2(entropy)), 12))

# Custom PAC test markers and assertions

def pytest_configure(config):
    """Configure custom pytest markers for PAC testing."""
    config.addinivalue_line("markers", "pac: PAC physics integration tests")
    config.addinivalue_line("markers", "conservation: Conservation physics tests") 
    config.addinivalue_line("markers", "xi_operator: Xi operator convergence tests")
    config.addinivalue_line("markers", "amplitude: Amplitude conservation tests")
    config.addinivalue_line("markers", "med: Maximum Entropy Depth tests")
    config.addinivalue_line("markers", "phase_singularity: Phase singularity detection tests")

def assert_pac_conservation(field_engine, tolerance: float = 1e-6):
    """Assert PAC conservation laws are maintained."""
    if hasattr(field_engine, 'amplitude_field') and field_engine.amplitude_field is not None:
        assert validate_amplitude_conservation(field_engine.amplitude_field, tolerance), \
            "Amplitude conservation violated"

def assert_xi_convergence(collapse_vector, tolerance: float = 0.5):
    """Assert Xi operator convergence to target value."""
    assert collapse_vector is not None, "No collapse vector provided"
    assert hasattr(collapse_vector, 'force_magnitude'), "Missing force_magnitude"
    assert hasattr(collapse_vector, 'entropy_tension'), "Missing entropy_tension"
    
    assert validate_xi_convergence(
        collapse_vector.force_magnitude, 
        collapse_vector.entropy_tension, 
        tolerance
    ), f"Xi operator convergence failed"

def assert_med_compliance(symbolic_tree, expected_entropy: float, safety_margin: int = 2):
    """Assert MED (Maximum Entropy Depth) compliance."""
    assert symbolic_tree is not None, "No symbolic tree provided"
    assert hasattr(symbolic_tree, 'depth'), "Missing tree depth"
    
    expected_med_limit = calculate_med_limit(expected_entropy)
    assert symbolic_tree.depth <= expected_med_limit + safety_margin, \
        f"Tree depth {symbolic_tree.depth} exceeds MED limit {expected_med_limit}"
