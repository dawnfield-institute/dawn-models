"""Mock body for GAIA v2 — behavioral harness for brain-like evaluation.

Provides sensory channels, motor decoders, environments, and a closed-loop
orchestrator (BodyLoop) that wraps the ConservationBus. Evaluation shifts
from prediction accuracy to behavioral coherence.
"""

from .ablation import (
    AblationReport,
    AblationResult,
    ModuleConfig,
    make_bus,
    make_resonance_bus,
    run_ablation_config,
    run_full_ablation,
    standard_configs,
)
from .environment import Action, GridWorld, Observation
from .loop import BodyLoop, TickRecord, Trajectory
from .metrics import BehavioralScores, TrajectoryAnalyzer
from .motor import GridMotorDecoder
from .probes import ProbeResult, run_instrumented
from .senses import ProprioceptiveChannel, VisualChannel

__all__ = [
    "AblationReport",
    "AblationResult",
    "Action",
    "BehavioralScores",
    "BodyLoop",
    "GridMotorDecoder",
    "GridWorld",
    "ModuleConfig",
    "Observation",
    "make_resonance_bus",
    "ProbeResult",
    "ProprioceptiveChannel",
    "TickRecord",
    "Trajectory",
    "TrajectoryAnalyzer",
    "VisualChannel",
    "make_bus",
    "run_ablation_config",
    "run_full_ablation",
    "run_instrumented",
    "standard_configs",
]
