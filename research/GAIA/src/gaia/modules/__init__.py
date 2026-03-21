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
  - observability: SCBF metrics, QBE equilibrium monitoring

Planned:
  - language: Token prediction, embedding grafting, generation
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
from .observability import (
    CollapseEvent,
    ObservabilityMetrics,
    ObservabilityModule,
    QBEController,
    SCBFMetrics,
    SCBFTracker,
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
    "CollapseEvent",
    "ObservabilityMetrics",
    "ObservabilityModule",
    "MobiusHarmonicAnalyzer",
    "MobiusLayer",
    "MobiusNeuron",
    "PACTree",
    "QBEController",
    "PhiAnchorMemory",
    "ReasoningMetrics",
    "ReasoningModule",
    "SCBFMetrics",
    "SCBFTracker",
    "SafetyMetrics",
    "SafetyModule",
    "TransitionTracker",
]
