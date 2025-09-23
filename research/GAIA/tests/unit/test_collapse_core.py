import pytest
from core.collapse_core import CollapseCore, CollapseType

def test_collapse_core_initialization():
    core = CollapseCore()
    assert core is not None
    assert hasattr(core, 'evaluator')
    assert hasattr(core, 'typing_engine')
    assert hasattr(core, 'synthesizer')
    assert hasattr(core, 'stabilizer')
    assert all(count == 0 for count in core.collapse_type_counts.values())

def test_collapse_evaluation_real_conditions(execution_context):
    core = CollapseCore()
    result = core.evaluator.evaluate(execution_context)
    assert result is not None
    assert result.collapse_type in CollapseType

def test_collapse_statistics_tracking(execution_context):
    core = CollapseCore()
    # Simulate a collapse
    result = core.evaluator.evaluate(execution_context)
    if result:
        core.synthesizer.synthesize(result, execution_context)
    stats = core.get_collapse_statistics()
    assert 'average_efficiency' in stats
    assert stats['average_efficiency'] >= 0
