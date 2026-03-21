"""
GAIA Core — Conservation bus, module protocol, SEC routing.

The core package defines the contracts that all GAIA modules must satisfy
and the bus that enforces PAC conservation across module boundaries.
"""

from .bus import ConservationBus
from .exceptions import ConservationViolation, InvalidFieldState, ModuleRegistrationError
from .protocol import GAIAModule
from .sec_router import SECRouter
from .types import ConservationResult, FieldState, RBFBalance, SECPhase

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
