"""
Token-Level PAC Tree Core Module
=================================

Data structures and metrics for observing PAC conservation
in LLM token prediction (logit collapse dynamics).
"""

from .pac_tree import (
    PACNode,
    TokenPACTree,
    PACForest,
    build_pac_tree_from_logits,
)
from .collapse_metrics import (
    compute_collapse_signature,
    compute_pac_ratio,
    compute_conservation_budget,
    classify_sec_phase,
    SECPhase,
)

__all__ = [
    'PACNode',
    'TokenPACTree',
    'PACForest',
    'build_pac_tree_from_logits',
    'compute_collapse_signature',
    'compute_pac_ratio',
    'compute_conservation_budget',
    'classify_sec_phase',
    'SECPhase',
]
