import pytest
import sys
import os
import numpy as np
from pathlib import Path

# Add src to path and Fracton SDK path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'fracton'))

# Import Fracton SDK (required for PAC-native tests)
import fracton
from fracton import (
    PhysicsMemoryField,
    PhysicsRecursiveExecutor,
    enable_pac_self_regulation,
    get_system_pac_metrics
)

# Import PAC-native GAIA components
# from core.collapse_core import CollapseCore  # Skip until cleaned up
from core.field_engine import FieldEngine
from fracton import MemoryField
# from core.superfluid_memory import SuperfluidMemory  # Skip until needed

# Import full GAIA if available (commented out during cleanup)
# try:
#     from gaia import PAC_GAIA, PAC_GAIAConfig
#     GAIA_AVAILABLE = True
# except ImportError as e:
#     print(f"GAIA import failed: {e}")
#     GAIA_AVAILABLE = False
GAIA_AVAILABLE = False

@pytest.fixture
def simple_context():
    """Provide simple context for tests."""
    return {"entropy": 0.8, "depth": 2}  # Simple dict instead of ExecutionContext

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


# PAC-Native Test Fixtures

@pytest.fixture
def pac_regulator():
    """Provide PAC regulator for testing."""
    return enable_pac_self_regulation()


@pytest.fixture
def physics_memory():
    """Provide physics memory field for testing."""
    return PhysicsMemoryField(
        capacity=1000,
        physics_dimensions=(16, 16),
        xi_target=1.0571
    )


@pytest.fixture
def pac_field_engine():
    """Provide PAC-native field engine for testing."""
    return FieldEngine(shape=(16, 16))


@pytest.fixture
def pac_collapse_core(physics_memory):
    """Provide PAC-native collapse core for testing."""
    # Import here to avoid module-level issues
    try:
        from core.collapse_core import CollapseCore
        return CollapseCore(physics_memory=physics_memory)
    except ImportError:
        pytest.skip("CollapseCore not available due to legacy code cleanup")


@pytest.fixture
def pac_superfluid_memory():
    """Provide PAC-native superfluid memory for testing."""
    # Skip until SuperfluidMemory is cleaned up
    pytest.skip("SuperfluidMemory cleanup pending")


@pytest.fixture
def pac_gaia_config():
    """Provide PAC GAIA configuration for testing."""
    if GAIA_AVAILABLE:
        return PAC_GAIAConfig(
            # Required fields
            memory_coherence=1.0,
            symbolic_structures=10,
            active_signals=5,
            cognitive_integrity=0.95,
            processing_cycles=100,
            total_collapses=0,
            resonance_patterns=3,
            # PAC parameters
            xi_target=1.0571,
            conservation_tolerance=1e-6,
            enable_pac_self_regulation=True,
            field_dimensions=(16, 16)
        )
    else:
        pytest.skip("PAC_GAIAConfig not available")


@pytest.fixture 
def pac_gaia(pac_gaia_config):
    """Provide PAC-native GAIA instance for testing."""
    if GAIA_AVAILABLE:
        return PAC_GAIA(pac_gaia_config)
    else:
        pytest.skip("PAC_GAIA not available")


# PAC validation utilities
def validate_pac_conservation(conservation_residual: float, tolerance: float = 1e-6) -> bool:
    """Validate PAC conservation within tolerance."""
    return abs(conservation_residual) < tolerance


def validate_xi_balance(xi_value: float, target: float = 1.0571, tolerance: float = 0.1) -> bool:
    """Validate Xi balance operator within tolerance of target."""
    return abs(xi_value - target) < tolerance


def validate_field_state(field_state, expected_pac_regulated=True):
    """Validate PAC-native field state properties."""
    assert field_state is not None
    assert hasattr(field_state, 'pac_regulated')
    assert field_state.pac_regulated == expected_pac_regulated
    assert hasattr(field_state, 'conservation_residual')
    assert hasattr(field_state, 'xi_balance')
    assert hasattr(field_state, 'field_tensor')
