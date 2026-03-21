"""
GAIA Modules — Pluggable intelligence components.

Each module implements the GAIAModule protocol:
  - process(field_state) -> field_state  (PAC-conserving transformation)
  - phase() -> SECPhase                  (current entropy phase)
  - health() -> RBFBalance               (energy-information balance)

Available:
  - safety: Boltzmann conservation layer, hallucination detection

Planned:
  - language: Token prediction, embedding grafting, generation
  - reasoning: Mobius neurons for recursive dynamics
  - observability: SCBF metrics, QBE equilibrium monitoring
  - memory: Bifractal hierarchy, PACTree, continuous learning
"""

from .safety import (
    BoltzmannHead,
    BoltzmannLayer,
    BoltzmannMonitor,
    ConservationProjector,
    SafetyMetrics,
    SafetyModule,
)

__all__ = [
    "BoltzmannHead",
    "BoltzmannLayer",
    "BoltzmannMonitor",
    "ConservationProjector",
    "SafetyMetrics",
    "SafetyModule",
]
