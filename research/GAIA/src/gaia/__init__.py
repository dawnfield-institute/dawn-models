"""
GAIA v2 — Modular Intelligence Architecture

Specialized modules composing via a PAC conservation bus,
with Fracton as the physics substrate. Each module conserves
entropy at its boundary; the bus enforces global conservation.

Requires: fracton >= 2.1
"""

__version__ = "2.0.0-dev"

from .core import (
    ConservationBus,
    ConservationResult,
    ConservationViolation,
    FieldState,
    GAIAModule,
    InvalidFieldState,
    ModuleRegistrationError,
    RBFBalance,
    SECPhase,
    SECRouter,
)

__all__ = [
    "ConservationBus",
    "ConservationResult",
    "ConservationViolation",
    "FieldState",
    "GAIAModule",
    "InvalidFieldState",
    "ModuleRegistrationError",
    "RBFBalance",
    "SECPhase",
    "SECRouter",
]
