"""
GAIA v4.0 Package

Built on Fracton PAC-Lazy substrate.
"""

from .cortex import (
    GAIACortex,
    GAIAConfig,
    GAIAResponse,
    TransformerOrgan
)

from .organs import (
    LanguageOrgan,
    ReasoningOrgan,
    MemoryOrgan,
    AttentionOrgan
)

from .learning import (
    ContinuousLearner,
    PatternCrystallizer,
    LearningEvent
)

__version__ = "4.0.0"

__all__ = [
    # Core
    "GAIACortex",
    "GAIAConfig",
    "GAIAResponse",
    "TransformerOrgan",
    
    # Organs
    "LanguageOrgan",
    "ReasoningOrgan",
    "MemoryOrgan",
    "AttentionOrgan",
    
    # Learning
    "ContinuousLearner",
    "PatternCrystallizer",
    "LearningEvent"
]
