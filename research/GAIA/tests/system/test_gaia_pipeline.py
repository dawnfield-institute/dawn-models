import pytest
from gaia import GAIA

def test_gaia_pipeline_runs():
    gaia = GAIA()
    response = gaia.process_input({"test": "pipeline"})
    assert response is not None
    assert hasattr(response, 'response_text')
    assert hasattr(response, 'state')
    assert hasattr(response, 'reasoning_trace')
    assert response.state.cognitive_integrity >= 0
