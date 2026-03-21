"""
GAIA Modules — Pluggable intelligence components.

Each module implements the GAIAModule protocol:
  - process(field_state) -> field_state  (PAC-conserving transformation)
  - phase() -> SECPhase                  (current entropy phase)
  - health() -> RBFBalance               (energy-information balance)

Available:
  - safety: Boltzmann conservation layer, hallucination detection
  - reasoning: Mobius neurons for recursive dynamics
  - memory: Bifractal hierarchy, PACTree, continuous learning

Planned:
  - language: Token prediction, embedding grafting, generation
  - observability: SCBF metrics, QBE equilibrium monitoring
"""

from .memory import (
    BifractalDepth,
    BifractalManager,
    MemoryMetrics,
    MemoryModule,
    MemoryNode,
    PACTree,
    TransitionTracker,
)
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
    "BifractalDepth",
    "BifractalManager",
    "BoltzmannHead",
    "BoltzmannLayer",
    "BoltzmannMonitor",
    "ConservationProjector",
    "MemoryMetrics",
    "MemoryModule",
    "MemoryNode",
    "MobiusHarmonicAnalyzer",
    "MobiusLayer",
    "MobiusNeuron",
    "PACTree",
    "PhiAnchorMemory",
    "ReasoningMetrics",
    "ReasoningModule",
    "SafetyMetrics",
    "SafetyModule",
    "TransitionTracker",
]
