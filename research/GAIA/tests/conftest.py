import pytest
import sys
import os
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
        def __init__(self, name): self.name = name
    class ExecutionContext:
        def __init__(self, **kwargs): pass

@pytest.fixture
def execution_context():
    """Provide real or minimal ExecutionContext for tests."""
    return ExecutionContext(entropy=0.8, depth=2)  # Higher entropy to trigger collapse

@pytest.fixture
def memory_field():
    """Provide real or minimally mocked MemoryField."""
    return MemoryField("test_field")
