"""Tests for RecursiveEntity — the recursion enabler.

Proves that a CoupledFieldsBus wrapped as a GAIAModule can be
registered in another bus, enabling fractal nesting.
"""

from __future__ import annotations

import time

import pytest
import torch

from gaia.core.coupled_fields_bus import CoupledFieldsBus
from gaia.core.protocol import GAIAModule
from gaia.core.types import FieldState, RBFBalance, SECPhase
from gaia.network.recursive_entity import RecursiveEntity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StubModule:
    """Deterministic test module with configurable transform."""

    def __init__(self, name: str, transform=None):
        self._name = name
        self._transform = transform or (lambda t: t.clone())

    @property
    def name(self) -> str:
        return self._name

    def process(self, field_state: FieldState) -> FieldState:
        result = field_state.clone()
        result.tensor = self._transform(result.tensor)
        result.provenance.append(self._name)
        return result

    def phase(self) -> SECPhase:
        return SECPhase.ORDERED

    def health(self) -> RBFBalance:
        return RBFBalance.compute(energy=1.0, information=1.0, memory=0.0)


def _make_field(dim: int = 10, value: float = 1.0) -> FieldState:
    return FieldState(
        tensor=torch.full((dim,), value),
        entropy=1.0,
        phase=SECPhase.ORDERED,
        conservation_budget=0.0,
        provenance=[],
        timestamp=time.time(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    """RecursiveEntity satisfies GAIAModule protocol."""

    def test_isinstance_check(self):
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("mod_a"))
        entity = RecursiveEntity("entity_1", bus)
        assert isinstance(entity, GAIAModule)

    def test_has_name(self):
        bus = CoupledFieldsBus(enforcement="soft")
        entity = RecursiveEntity("my_entity", bus)
        assert entity.name == "my_entity"

    def test_has_process(self):
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("mod_a"))
        entity = RecursiveEntity("e", bus)
        result = entity.process(_make_field())
        assert isinstance(result, FieldState)

    def test_has_phase(self):
        bus = CoupledFieldsBus(enforcement="soft")
        entity = RecursiveEntity("e", bus)
        assert isinstance(entity.phase(), SECPhase)

    def test_has_health(self):
        bus = CoupledFieldsBus(enforcement="soft")
        entity = RecursiveEntity("e", bus)
        assert isinstance(entity.health(), RBFBalance)


class TestPACConservation:
    """PAC holds across recursion levels."""

    def test_single_level_pac(self):
        """Energy in ≈ energy out through a wrapped bus."""
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("identity_a"))
        bus.register_module(StubModule("identity_b"))
        entity = RecursiveEntity("e", bus)

        field = _make_field(dim=10, value=2.0)
        input_energy = field.total_energy()
        output = entity.process(field)
        output_energy = output.total_energy()

        assert abs(input_energy - output_energy) < 0.01

    def test_nested_pac(self):
        """PAC holds when a RecursiveEntity is inside another bus."""
        # Inner bus with two modules
        inner_bus = CoupledFieldsBus(enforcement="soft")
        inner_bus.register_module(StubModule("inner_a"))
        inner_bus.register_module(StubModule("inner_b"))
        inner_entity = RecursiveEntity("inner", inner_bus)

        # Outer bus containing the inner entity + another module
        outer_bus = CoupledFieldsBus(enforcement="soft")
        outer_bus.register_module(inner_entity)
        outer_bus.register_module(StubModule("outer_c"))

        field = _make_field(dim=10, value=3.0)
        input_energy = field.total_energy()
        output = outer_bus.process(field)
        output_energy = output.total_energy()

        assert abs(input_energy - output_energy) < 0.01


class TestRecursion:
    """RecursiveEntity can be nested inside another bus."""

    def test_entity_in_bus(self):
        """A RecursiveEntity works as a module in a CoupledFieldsBus."""
        inner = CoupledFieldsBus(enforcement="soft")
        inner.register_module(StubModule("mod_a"))
        entity = RecursiveEntity("wrapped", inner)

        outer = CoupledFieldsBus(enforcement="soft")
        outer.register_module(entity)

        field = _make_field()
        result = outer.process(field)
        assert result is not None
        assert isinstance(result.tensor, torch.Tensor)

    def test_two_entities_in_bus(self):
        """Two RecursiveEntities can coexist in the same bus."""
        bus_a = CoupledFieldsBus(enforcement="soft")
        bus_a.register_module(StubModule("a_mod"))
        entity_a = RecursiveEntity("entity_a", bus_a)

        bus_b = CoupledFieldsBus(enforcement="soft")
        bus_b.register_module(StubModule("b_mod"))
        entity_b = RecursiveEntity("entity_b", bus_b)

        network = CoupledFieldsBus(enforcement="soft")
        network.register_module(entity_a)
        network.register_module(entity_b)

        field = _make_field()
        result = network.process(field)
        assert result is not None

    def test_three_level_nesting(self):
        """Three levels of recursion: module → entity → entity → bus."""
        # Level 0: a module
        mod = StubModule("base")

        # Level 1: bus wrapping the module
        bus_1 = CoupledFieldsBus(enforcement="soft")
        bus_1.register_module(mod)
        entity_1 = RecursiveEntity("level_1", bus_1)

        # Level 2: bus wrapping level 1
        bus_2 = CoupledFieldsBus(enforcement="soft")
        bus_2.register_module(entity_1)
        entity_2 = RecursiveEntity("level_2", bus_2)

        # Level 3: top-level bus
        bus_3 = CoupledFieldsBus(enforcement="soft")
        bus_3.register_module(entity_2)

        field = _make_field()
        result = bus_3.process(field)
        assert result is not None
        assert abs(field.total_energy() - result.total_energy()) < 0.01


class TestPhaseAndHealth:
    """Phase and health aggregation works correctly."""

    def test_phase_before_process(self):
        """Phase defaults to ORDERED before any processing."""
        bus = CoupledFieldsBus(enforcement="soft")
        entity = RecursiveEntity("e", bus)
        assert entity.phase() == SECPhase.ORDERED

    def test_phase_after_process(self):
        """Phase reflects last output."""
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("mod"))
        entity = RecursiveEntity("e", bus)
        entity.process(_make_field())
        # Phase should be a valid SECPhase
        assert entity.phase() in list(SECPhase)

    def test_health_aggregates_modules(self):
        """Health averages across internal modules."""
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("a"))
        bus.register_module(StubModule("b"))
        entity = RecursiveEntity("e", bus)
        health = entity.health()
        # StubModules have balanced health (energy=info=1.0)
        assert abs(health.balance) < 0.01

    def test_health_empty_bus(self):
        """Empty bus returns balanced health."""
        bus = CoupledFieldsBus(enforcement="soft")
        entity = RecursiveEntity("e", bus)
        health = entity.health()
        assert abs(health.balance) < 0.01

    def test_tick_count(self):
        """Tick count increments on process."""
        bus = CoupledFieldsBus(enforcement="soft")
        bus.register_module(StubModule("mod"))
        entity = RecursiveEntity("e", bus)
        assert entity.tick_count == 0
        entity.process(_make_field())
        assert entity.tick_count == 1
        entity.process(_make_field())
        assert entity.tick_count == 2
