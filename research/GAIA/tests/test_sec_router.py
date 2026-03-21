"""Tests for SEC phase router."""

import pytest

from gaia.core.sec_router import SECRouter
from gaia.core.types import SECPhase
from tests.conftest import IdentityModule, make_field_state


class TestSECClassification:
    def setup_method(self):
        self.router = SECRouter()

    def test_crystallized(self):
        assert self.router.classify(0.1) == SECPhase.CRYSTALLIZED

    def test_crystallized_zero(self):
        assert self.router.classify(0.0) == SECPhase.CRYSTALLIZED

    def test_crystallized_negative(self):
        assert self.router.classify(-1.0) == SECPhase.CRYSTALLIZED

    def test_ordered(self):
        assert self.router.classify(1.0) == SECPhase.ORDERED

    def test_transitional(self):
        assert self.router.classify(3.0) == SECPhase.TRANSITIONAL

    def test_chaotic(self):
        assert self.router.classify(5.0) == SECPhase.CHAOTIC

    # Boundary tests (lower-inclusive)
    def test_boundary_0_5_is_ordered(self):
        assert self.router.classify(0.5) == SECPhase.ORDERED

    def test_boundary_just_below_0_5(self):
        assert self.router.classify(0.4999) == SECPhase.CRYSTALLIZED

    def test_boundary_2_0_is_transitional(self):
        assert self.router.classify(2.0) == SECPhase.TRANSITIONAL

    def test_boundary_just_below_2_0(self):
        assert self.router.classify(1.9999) == SECPhase.ORDERED

    def test_boundary_4_0_is_chaotic(self):
        assert self.router.classify(4.0) == SECPhase.CHAOTIC

    def test_boundary_just_below_4_0(self):
        assert self.router.classify(3.9999) == SECPhase.TRANSITIONAL


class TestSECRouting:
    def setup_method(self):
        self.router = SECRouter()
        self.module = IdentityModule()

    def test_route_no_modules_returns_empty(self):
        fs = make_field_state(entropy=1.0)
        assert self.router.route(fs) == []

    def test_route_returns_registered_module(self):
        self.router.register(self.module, [SECPhase.ORDERED])
        fs = make_field_state(entropy=1.0)
        modules = self.router.route(fs)
        assert len(modules) == 1
        assert modules[0].name == "identity"

    def test_route_wrong_phase_returns_empty(self):
        self.router.register(self.module, [SECPhase.CHAOTIC])
        fs = make_field_state(entropy=1.0)  # ORDERED
        assert self.router.route(fs) == []

    def test_register_all_phases(self):
        self.router.register(self.module, None)
        for entropy in [0.1, 1.0, 3.0, 5.0]:
            fs = make_field_state(entropy=entropy)
            modules = self.router.route(fs)
            assert len(modules) == 1

    def test_no_duplicate_registration(self):
        self.router.register(self.module, [SECPhase.ORDERED])
        self.router.register(self.module, [SECPhase.ORDERED])
        fs = make_field_state(entropy=1.0)
        assert len(self.router.route(fs)) == 1

    def test_multiple_modules_same_phase(self):
        module2 = IdentityModule()
        self.router.register(self.module, [SECPhase.ORDERED])
        self.router.register(module2, [SECPhase.ORDERED])
        fs = make_field_state(entropy=1.0)
        assert len(self.router.route(fs)) == 2
