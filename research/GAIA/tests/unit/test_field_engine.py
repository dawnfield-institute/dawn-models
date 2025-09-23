import pytest
from core.field_engine import FieldEngine

def test_field_engine_initialization():
    engine = FieldEngine()
    assert engine is not None
    assert hasattr(engine, 'energy_field')
    assert hasattr(engine, 'information_field')
    assert hasattr(engine, 'entropy_tensor')
    assert hasattr(engine, 'balance_controller')

def test_field_update_and_pressure(execution_context):
    engine = FieldEngine()
    # Simulate an update
    field_before = engine.energy_field.field.copy()
    engine.energy_field.update(1.0, execution_context)
    field_after = engine.energy_field.field
    assert (field_after != field_before).any()
    stats = engine.get_field_statistics()
    assert 'average_entropy' in stats
    assert stats['average_entropy'] >= 0
