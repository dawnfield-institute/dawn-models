"""Integration tests — full pipeline through the conservation bus."""

from __future__ import annotations

import pytest
import torch

from gaia.core.bus import ConservationBus
from gaia.core.exceptions import ConservationViolation
from gaia.core.protocol import GAIAModule
from gaia.core.types import FieldState, RBFBalance, SECPhase

from tests.conftest import IdentityModule, ScalingModule, UnhealthyModule, make_field_state


class TestFullPipeline:
    """End-to-end tests exercising the full bus pipeline."""

    def test_identity_roundtrip(self):
        """Identity module through full pipeline preserves energy exactly."""
        bus = ConservationBus(enforcement="hard")
        bus.register_module(IdentityModule())
        state = make_field_state(tensor=torch.tensor([1.0, 2.0, 3.0]), entropy=1.0)
        result = bus.process(state)
        assert result.total_energy() == pytest.approx(6.0)
        assert result.provenance == ["identity"]
        assert result.phase == SECPhase.ORDERED

    def test_violation_rejection(self):
        """Scaling module detected and rejected in hard mode."""
        bus = ConservationBus(enforcement="hard")
        bus.register_module(ScalingModule(factor=3.0))
        state = make_field_state(tensor=torch.tensor([1.0, 1.0, 1.0]), entropy=1.0)
        with pytest.raises(ConservationViolation) as exc_info:
            bus.process(state)
        result = exc_info.value.result
        assert result.input_energy == pytest.approx(3.0)
        assert result.output_energy == pytest.approx(9.0)
        assert result.residual > 0

    def test_multi_phase_routing(self):
        """Modules registered for specific phases only activate for matching entropy."""

        class CrystallizedModule:
            @property
            def name(self) -> str:
                return "crystal"

            def process(self, field_state: FieldState) -> FieldState:
                result = field_state.clone()
                result.provenance.append(self.name)
                return result

            def phase(self) -> SECPhase:
                return SECPhase.CRYSTALLIZED

            def health(self) -> RBFBalance:
                return RBFBalance.compute(1.0, 1.0, 0.0)

        class ChaoticModule:
            @property
            def name(self) -> str:
                return "chaos"

            def process(self, field_state: FieldState) -> FieldState:
                result = field_state.clone()
                result.provenance.append(self.name)
                return result

            def phase(self) -> SECPhase:
                return SECPhase.CHAOTIC

            def health(self) -> RBFBalance:
                return RBFBalance.compute(1.0, 1.0, 0.0)

        bus = ConservationBus(enforcement="hard")
        bus.register_module(CrystallizedModule(), phases=[SECPhase.CRYSTALLIZED])
        bus.register_module(ChaoticModule(), phases=[SECPhase.CHAOTIC])

        # Low entropy → CRYSTALLIZED
        low_state = make_field_state(entropy=0.2)
        result_low = bus.process(low_state)
        assert "crystal" in result_low.provenance
        assert "chaos" not in result_low.provenance

        # High entropy → CHAOTIC
        high_state = make_field_state(entropy=5.0)
        result_high = bus.process(high_state)
        assert "chaos" in result_high.provenance
        assert "crystal" not in result_high.provenance

    def test_rbf_suppression_in_pipeline(self):
        """Unhealthy modules are suppressed even when registered for correct phase."""
        bus = ConservationBus(enforcement="hard", rbf_suppression_threshold=0.0)
        bus.register_module(IdentityModule())
        bus.register_module(UnhealthyModule())
        state = make_field_state(entropy=1.0)
        result = bus.process(state)
        assert "identity" in result.provenance
        assert "unhealthy" not in result.provenance

    def test_conservation_budget_accounting(self):
        """Module that declares conservation budget is not flagged as violation."""

        class BudgetModule:
            @property
            def name(self) -> str:
                return "budget"

            def process(self, field_state: FieldState) -> FieldState:
                result = field_state.clone()
                # Add energy but declare it in the budget
                result.tensor = result.tensor + 1.0  # adds 10.0 total energy
                result.conservation_budget = field_state.conservation_budget + 10.0
                result.provenance.append(self.name)
                return result

            def phase(self) -> SECPhase:
                return SECPhase.ORDERED

            def health(self) -> RBFBalance:
                return RBFBalance.compute(1.0, 1.0, 0.0)

        bus = ConservationBus(enforcement="hard", tolerance=1e-6)
        bus.register_module(BudgetModule())
        state = make_field_state(entropy=1.0)
        result = bus.process(state)
        assert result.total_energy() == pytest.approx(20.0)  # 10 + 10 added
        assert "budget" in result.provenance
        assert len(bus.violation_log) == 0


class TestProtocolCompliance:
    """Verify that test modules satisfy the GAIAModule protocol."""

    def test_identity_is_gaia_module(self):
        assert isinstance(IdentityModule(), GAIAModule)

    def test_scaling_is_gaia_module(self):
        assert isinstance(ScalingModule(), GAIAModule)

    def test_unhealthy_is_gaia_module(self):
        assert isinstance(UnhealthyModule(), GAIAModule)
