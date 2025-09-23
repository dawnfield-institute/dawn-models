import pytest
from core.collapse_core import CollapseCore
from core.field_engine import FieldEngine

def test_field_collapse_integration(execution_context):
    field_engine = FieldEngine()
    collapse_core = CollapseCore()
    # Simulate field update and collapse evaluation
    field_engine.energy_field.update(1.0, execution_context)
    result = collapse_core.evaluator.evaluate(execution_context)
    assert result is not None
    # Synthesize and check stats
    collapse_core.synthesizer.synthesize(result, execution_context)
    stats = collapse_core.get_collapse_statistics()
    assert stats['total_collapses'] >= 0
