"""
GAIA Core — Conservation bus, module protocol, SEC routing.

The core package defines the contracts that all GAIA modules must satisfy
and the bus that enforces PAC conservation across module boundaries.
"""

from .bus import ConservationBus
from .exceptions import ConservationViolation, InvalidFieldState, ModuleRegistrationError
from .protocol import GAIAModule
from .coupled_fields_bus import CoupledFieldsBus, CoupledFieldState, CoupledWeight
from .resonance_bus import ResonanceBus, ResonanceWeight
from .sec_router import SECRouter
from .types import ConservationResult, FieldState, RBFBalance, SECPhase

__all__ = [
    "ConservationBus",
    "ConservationResult",
    "ConservationViolation",
    "CoupledFieldsBus",
    "CoupledFieldState",
    "CoupledWeight",
    "FieldState",
    "GAIAModule",
    "InvalidFieldState",
    "ModuleRegistrationError",
    "RBFBalance",
    "ResonanceBus",
    "ResonanceWeight",
    "SECPhase",
    "SECRouter",
]
