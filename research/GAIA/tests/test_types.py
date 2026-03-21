"""Tests for GAIA v2 core data types."""

import torch

from gaia.core.types import ConservationResult, FieldState, RBFBalance, SECPhase
from tests.conftest import make_field_state


class TestSECPhase:
    def test_enum_values(self):
        assert SECPhase.CRYSTALLIZED.value == "crystallized"
        assert SECPhase.ORDERED.value == "ordered"
        assert SECPhase.TRANSITIONAL.value == "transitional"
        assert SECPhase.CHAOTIC.value == "chaotic"

    def test_four_phases(self):
        assert len(SECPhase) == 4


class TestFieldState:
    def test_creation(self):
        fs = make_field_state()
        assert fs.tensor is not None
        assert fs.entropy == 1.0
        assert fs.phase == SECPhase.ORDERED

    def test_total_energy(self):
        fs = make_field_state(tensor=torch.tensor([1.0, 2.0, 3.0]))
        assert fs.total_energy() == 6.0

    def test_total_energy_negative(self):
        fs = make_field_state(tensor=torch.tensor([-1.0, 2.0, -3.0]))
        assert fs.total_energy() == -2.0

    def test_clone_independence(self):
        fs = make_field_state(tensor=torch.tensor([1.0, 2.0]))
        clone = fs.clone()
        clone.tensor[0] = 99.0
        clone.provenance.append("mutated")
        assert fs.tensor[0] == 1.0
        assert "mutated" not in fs.provenance

    def test_provenance_tracking(self):
        fs = make_field_state(provenance=["module_a"])
        fs.provenance.append("module_b")
        assert fs.provenance == ["module_a", "module_b"]

    def test_default_provenance_empty(self):
        fs = make_field_state()
        assert fs.provenance == []


class TestRBFBalance:
    def test_compute_balanced(self):
        rb = RBFBalance.compute(energy=1.0, information=1.0, memory=0.0)
        assert rb.balance == 0.0

    def test_compute_overactive(self):
        rb = RBFBalance.compute(energy=5.0, information=1.0, memory=0.0)
        assert rb.balance > 0

    def test_compute_underactive(self):
        rb = RBFBalance.compute(energy=0.1, information=5.0, memory=1.0)
        assert rb.balance < 0

    def test_compute_formula(self):
        rb = RBFBalance.compute(
            energy=3.0, information=1.0, memory=2.0,
            rbf_lambda=2.0, rbf_alpha=0.5,
        )
        expected = 2.0 * (3.0 - 1.0) / (1.0 + 0.5 * 2.0)
        assert abs(rb.balance - expected) < 1e-10


class TestConservationResult:
    def test_creation(self):
        cr = ConservationResult(
            conserved=True,
            input_energy=10.0,
            output_energy=10.0,
            residual=0.0,
            module_name="test",
        )
        assert cr.conserved
        assert cr.violation_type is None
