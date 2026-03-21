"""Shared test fixtures for GAIA v2 tests."""

from __future__ import annotations

import time

import pytest
import torch

from gaia.core.types import FieldState, RBFBalance, SECPhase


def make_field_state(
    tensor: torch.Tensor | None = None,
    entropy: float = 1.0,
    phase: SECPhase = SECPhase.ORDERED,
    conservation_budget: float = 0.0,
    provenance: list[str] | None = None,
) -> FieldState:
    """Factory for creating FieldState instances in tests."""
    if tensor is None:
        tensor = torch.ones(10)
    return FieldState(
        tensor=tensor,
        entropy=entropy,
        phase=phase,
        conservation_budget=conservation_budget,
        provenance=provenance or [],
        timestamp=time.time(),
    )


class IdentityModule:
    """Test module that returns input unchanged (trivially PAC-conserving)."""

    @property
    def name(self) -> str:
        return "identity"

    def process(self, field_state: FieldState) -> FieldState:
        result = field_state.clone()
        result.provenance.append(self.name)
        return result

    def phase(self) -> SECPhase:
        return SECPhase.ORDERED

    def health(self) -> RBFBalance:
        return RBFBalance.compute(energy=1.0, information=1.0, memory=0.0)


class ScalingModule:
    """Test module that scales tensor by a factor (violates PAC unless factor=1)."""

    def __init__(self, factor: float = 2.0) -> None:
        self._factor = factor

    @property
    def name(self) -> str:
        return f"scaler({self._factor})"

    def process(self, field_state: FieldState) -> FieldState:
        result = field_state.clone()
        result.tensor = result.tensor * self._factor
        result.provenance.append(self.name)
        return result

    def phase(self) -> SECPhase:
        return SECPhase.ORDERED

    def health(self) -> RBFBalance:
        return RBFBalance.compute(energy=1.0, information=1.0, memory=0.0)


class UnhealthyModule:
    """Test module with negative RBF balance (should be suppressed)."""

    @property
    def name(self) -> str:
        return "unhealthy"

    def process(self, field_state: FieldState) -> FieldState:
        result = field_state.clone()
        result.provenance.append(self.name)
        return result

    def phase(self) -> SECPhase:
        return SECPhase.ORDERED

    def health(self) -> RBFBalance:
        return RBFBalance.compute(energy=0.1, information=5.0, memory=1.0)


@pytest.fixture
def identity_module():
    return IdentityModule()


@pytest.fixture
def scaling_module():
    return ScalingModule(factor=2.0)


@pytest.fixture
def unhealthy_module():
    return UnhealthyModule()
