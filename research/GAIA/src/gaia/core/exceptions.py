"""Exceptions for the GAIA v2 conservation bus."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import ConservationResult


class ConservationViolation(Exception):
    """Raised when PAC conservation fails in hard enforcement mode."""

    def __init__(self, result: ConservationResult, module_name: str) -> None:
        self.result = result
        self.module_name = module_name
        super().__init__(
            f"PAC conservation violated by module '{module_name}': "
            f"residual={result.residual:.6e} "
            f"(input={result.input_energy:.6f}, output={result.output_energy:.6f})"
        )


class ModuleRegistrationError(Exception):
    """Raised when a module doesn't satisfy the GAIAModule protocol."""


class InvalidFieldState(Exception):
    """Raised when a FieldState is malformed."""
