"""
GAIA Prime: Generative AI via Information Architecture.

The canonical implementation of PAC-native language modeling.
Built on validated Dawn Field Theory principles with derived constants.

Core Architecture:
- PAC Mesh: Multi-model embedding space with conservation
- Physics Layer: Entropy, resonance, collapse dynamics
- Continuous Learning: Hebbian strengthening during inference
- Multi-Model Fusion: Agreement crystallization across LLMs
- Bifractal Resonance: Hierarchical depth-based memory
- Auto-Collapse: Entropy-triggered structure formation

Constants (derived, not fitted):
- XI = 1 + π/F₁₀ ≈ 1.0571 (balance operator)
- PHI = (1+√5)/2 ≈ 1.618 (golden ratio)
- LAMBDA_STAR = 0.618432 (SEC partition threshold)

Usage:
    from gaia_prime import GAIA_Prime, PhysicsMesh, EntropyBalancer
    
    # Create physics-governed mesh
    mesh = PACMeshSpace(embed_dim=768)
    physics = PhysicsMesh(mesh)
    
    # Or use the full model
    model = GAIA_Prime.from_gpt2()
    model.learn("Training text...")
    result = model.generate("Once upon a time")

Legacy implementations archived in src/legacy/
"""

__version__ = "2.0.0"  # Major version bump for prime

# Import order matters for dependencies
from .pac_tree import PACTree, PACNode
from .embeddings import GraftedEmbeddings, SimpleEmbeddings
from .transitions import TransitionMatrix, TransitionStats
from .concentration import ConcentrationMonitor, ConcentrationResult
from .generator import PACGenerator, GenerationResult
from .model import GAIA_Prime, GAIA, GaiaModel
from .mesh import PACMesh, ModelSource, MeshStatistics

# Physics-based intelligence layer
from .pac_mesh import PACMeshSpace, MeshNode, MultiModelMesh
from .physics_mesh import (
    PhysicsMesh, FieldState, CollapseEvent, CollapseType,
    EntropyMonitor, ConservationEnforcer, ResonanceField, CollapseEngine,
)
# Import validated constants from canonical source
from .validated_constants import (
    XI, PHI, PHI_INV, LAMBDA_STAR, LAMBDA_HALF,
    ENTROPY_OPTIMAL_LOW, ENTROPY_OPTIMAL_HIGH, COLLAPSE_THRESHOLD,
    ATTRACTION_FRACTION, REPULSION_FRACTION, KOIDE_Q,
    FIBONACCI, validate_constants
)
from .physics_generator import (
    PhysicsGenerator, GenerationConfig, GenerationResult as PhysicsGenerationResult,
    PhysicsChat, create_generator
)
from .continuous_learning import (
    ContinuousLearner, AdaptiveGenerator, LearningEvent
)
from .multi_model_fusion import (
    MultiModelFusion, FusionResult, FusionGenerator,
    ModelVote, ModelSource as FusionModelSource
)
from .bifractal_resonance import (
    BifractalResonance, BifractalDepth, BifractalPattern,
    PersonalityTrait, BifractalGenerator
)
from .auto_collapse import (
    AutoCollapseEngine, AutoCollapseConfig, CollapseResult, CollapseStrategy,
    EntropyBalancer
)

__all__ = [
    # Version
    '__version__',
    # Core components
    'PACTree',
    'PACNode',
    'GraftedEmbeddings',
    'SimpleEmbeddings',
    'TransitionMatrix',
    'TransitionStats',
    'ConcentrationMonitor',
    'ConcentrationResult',
    'PACGenerator',
    'GenerationResult',
    # Main model
    'GAIA_Prime',
    'GAIA',
    'GaiaModel',
    # Multi-model mesh
    'PACMesh',
    'ModelSource',
    'MeshStatistics',
    # Physics layer
    'PACMeshSpace',
    'MeshNode',
    'MultiModelMesh',
    'PhysicsMesh',
    'FieldState',
    'CollapseEvent',
    'CollapseType',
    'EntropyMonitor',
    'ConservationEnforcer',
    'ResonanceField',
    'CollapseEngine',
    # Generation
    'PhysicsGenerator',
    'GenerationConfig',
    'PhysicsGenerationResult',
    'PhysicsChat',
    'create_generator',
    # Continuous Learning
    'ContinuousLearner',
    'AdaptiveGenerator',
    'LearningEvent',
    # Multi-model Fusion
    'MultiModelFusion',
    'FusionResult',
    'FusionGenerator',
    'ModelVote',
    'FusionModelSource',
    # Bifractal Resonance
    'BifractalResonance',
    'BifractalDepth',
    'BifractalPattern',
    'PersonalityTrait',
    'BifractalGenerator',
    # Auto-Collapse
    'AutoCollapseEngine',
    'AutoCollapseConfig',
    'CollapseResult',
    'CollapseStrategy',
    'EntropyBalancer',
    # Validated Constants (derived, not fitted)
    'XI',
    'PHI',
    'PHI_INV',
    'LAMBDA_STAR',
    'LAMBDA_HALF',
    'ENTROPY_OPTIMAL_LOW',
    'ENTROPY_OPTIMAL_HIGH',
    'COLLAPSE_THRESHOLD',
    'ATTRACTION_FRACTION',
    'REPULSION_FRACTION',
    'KOIDE_Q',
    'FIBONACCI',
    'validate_constants',
]
