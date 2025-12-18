"""
GAIA-1: First Talkable Field-Native Model

Pure Dawn Field Theory implementation.
No transformers. No attention layers. Just physics.

Components:
- FieldVocabulary: Token embeddings as field patterns
- FieldContext: Context processing via evolution
- FieldGenerator: Next-token prediction via resonance
- GAIA1: Complete model with training and inference
"""

from .model import GAIA1, GAIA1Config
from .vocabulary import FieldVocabulary
from .generator import FieldGenerator

__all__ = ['GAIA1', 'GAIA1Config', 'FieldVocabulary', 'FieldGenerator']
