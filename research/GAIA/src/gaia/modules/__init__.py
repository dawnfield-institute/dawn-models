"""
GAIA Modules — Pluggable intelligence components.

Each module implements the GAIAModule protocol:
  - process(field_state) -> field_state  (PAC-conserving transformation)
  - phase() -> SECPhase                  (current entropy phase)
  - health() -> RBFBalance               (energy-information balance)

Available:
  - safety: Boltzmann conservation layer, hallucination detection
  - reasoning: Mobius neurons for recursive dynamics

Planned:
  - language: Token prediction, embedding grafting, generation
  - observability: SCBF metrics, QBE equilibrium monitoring
  - memory: Bifractal hierarchy, PACTree, continuous learning
"""

from .reasoning import (
    MobiusHarmonicAnalyzer,
    MobiusLayer,
    MobiusNeuron,
    PhiAnchorMemory,
    ReasoningMetrics,
    ReasoningModule,
)
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
    "MobiusHarmonicAnalyzer",
    "MobiusLayer",
    "MobiusNeuron",
    "PhiAnchorMemory",
    "ReasoningMetrics",
    "ReasoningModule",
    "SafetyMetrics",
    "SafetyModule",
]
