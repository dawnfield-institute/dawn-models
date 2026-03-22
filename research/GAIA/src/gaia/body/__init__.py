"""Mock body for GAIA v2 — behavioral harness for brain-like evaluation.

Provides sensory channels, motor decoders, environments, and a closed-loop
orchestrator (BodyLoop) that wraps the ConservationBus. Evaluation shifts
from prediction accuracy to behavioral coherence.
"""

from .environment import Action, GridWorld, Observation
from .loop import BodyLoop, TickRecord, Trajectory
from .metrics import BehavioralScores, TrajectoryAnalyzer
from .motor import GridMotorDecoder
from .senses import ProprioceptiveChannel, VisualChannel

__all__ = [
    "Action",
    "BehavioralScores",
    "BodyLoop",
    "GridMotorDecoder",
    "GridWorld",
    "Observation",
    "ProprioceptiveChannel",
    "TickRecord",
    "Trajectory",
    "TrajectoryAnalyzer",
    "VisualChannel",
]
