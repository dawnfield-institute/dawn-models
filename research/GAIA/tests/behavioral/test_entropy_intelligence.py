import pytest
import numpy as np
from gaia import GAIA

def test_entropy_pattern_detection():
    gaia = GAIA()
    structured_data = [1, 2, 3, 4, 5, 6, 7, 8]
    random_data = [7, 2, 9, 1, 5, 3, 8, 4]
    resp1 = gaia.process_input(structured_data)
    resp2 = gaia.process_input(random_data)
    
    # GAIA should detect different patterns through:
    # 1. Different confidence levels
    # 2. Different coherence patterns  
    # 3. Different cognitive load
    # 4. Different field pressure
    assert resp1.confidence != resp2.confidence
    assert resp1.cognitive_load != resp2.cognitive_load
    assert resp1.state.field_pressure != resp2.state.field_pressure

def test_collapse_threshold_adaptation():
    gaia = GAIA(collapse_threshold=0.7)
    low_entropy_input = {"pattern": "simple", "complexity": 0.2}
    response_low = gaia.process_input(low_entropy_input)
    high_entropy_input = {
        "nested": {"deeply": {"complex": {"chaotic": {"data": np.random.rand(100)}}}},
        "multiple": {"branches": {"competing": {"signals": list(range(50))}}},
        "entropy_source": "maximum_complexity"
    }
    response_high = gaia.process_input(high_entropy_input)
    assert response_high.state.processing_cycles >= response_low.state.processing_cycles
    assert response_high.structures_created >= response_low.structures_created
