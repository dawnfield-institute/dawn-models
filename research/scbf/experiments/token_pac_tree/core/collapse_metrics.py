"""
SEC Phase Classification and PAC Conservation Metrics
=====================================================

Classifies each token-level collapse event into SEC phases
and computes PAC conservation diagnostics.

SEC Phase Model (from Landauer erasure structure research):
  - CRYSTALLIZED: H < 0.5  — model is highly confident, routine actualization
  - ORDERED:      0.5 < H < 2.0  — healthy branching, normal PAC tree
  - TRANSITIONAL: 2.0 < H < 4.0  — model exploring, broad distribution
  - CHAOTIC:      H > 4.0  — model genuinely uncertain, wide potential

Collapse Anomaly Detection:
  - FORCED_COLLAPSE: high H but top-1 probability is disproportionately large
    → possible hallucination (model fabricates confident answer despite uncertainty)
  - CONSERVATION_VIOLATION: children don't sum correctly
    → possible internal inconsistency
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

# Dawn Field Theory constants
PHI = (1 + math.sqrt(5)) / 2
INV_PHI = 1 / PHI
XI = 1 + math.pi / 55


class SECPhase(Enum):
    """SEC phase classification for a token prediction."""
    CRYSTALLIZED = 'crystallized'     # H < 0.5 — confident, low branching
    ORDERED = 'ordered'               # 0.5 < H < 2.0 — normal prediction
    TRANSITIONAL = 'transitional'     # 2.0 < H < 4.0 — exploring
    CHAOTIC = 'chaotic'               # H > 4.0 — maximum uncertainty


def classify_sec_phase(
    entropy: float,
    thresholds: Tuple[float, float, float] = (0.5, 2.0, 4.0),
) -> SECPhase:
    """Classify a token prediction's entropy into an SEC phase.

    Args:
        entropy: Shannon entropy of the softmax distribution (nats)
        thresholds: (crystallized_upper, ordered_upper, transitional_upper)

    Returns:
        SECPhase enum value
    """
    if entropy < thresholds[0]:
        return SECPhase.CRYSTALLIZED
    elif entropy < thresholds[1]:
        return SECPhase.ORDERED
    elif entropy < thresholds[2]:
        return SECPhase.TRANSITIONAL
    else:
        return SECPhase.CHAOTIC


@dataclass
class CollapseSignature:
    """Full diagnostic signature for a single token collapse event."""
    # SEC phase
    sec_phase: SECPhase
    entropy: float

    # PAC ratios and their distances from DFT constants
    pac_ratio_1_2: Optional[float]
    phi_distance: Optional[float]          # |pac_ratio - phi|
    inv_phi_distance: Optional[float]      # |pac_ratio - 1/phi|

    # Conservation
    prob_conservation_error: float         # |sum(p_children) + p_tail - 1|
    entropy_conservation_error: float      # |sum(H_children) + H_tail - H_total| / H_total

    # Collapse character
    collapse_magnitude: float             # H_total (all entropy is destroyed)
    concentration: float                  # p1 — how much mass in top token
    effective_k: float                    # exp(H) — effective number of choices

    # Anomaly flags
    is_forced_collapse: bool              # high H but high p1 → suspicious
    is_phi_aligned: bool                  # pac_ratio within 5% of phi or 1/phi
    is_xi_aligned: bool                   # some ratio near xi


def compute_pac_ratio(p1: float, p2: float) -> Optional[float]:
    """Compute PAC ratio p1/p2, guarding division by zero."""
    if p2 < 1e-15:
        return None
    return p1 / p2


def compute_conservation_budget(
    children_prob_sum: float,
    tail_prob: float,
    children_entropy_sum: float,
    tail_entropy: float,
    total_entropy: float,
) -> Tuple[float, float]:
    """Check PAC conservation.

    Returns:
        (prob_error, entropy_relative_error)
    """
    prob_error = abs((children_prob_sum + tail_prob) - 1.0)
    if total_entropy > 0:
        entropy_error = abs(
            (children_entropy_sum + tail_entropy) - total_entropy
        ) / total_entropy
    else:
        entropy_error = 0.0
    return prob_error, entropy_error


def compute_collapse_signature(
    entropy: float,
    pac_ratio_1_2: Optional[float],
    prob_conservation_error: float,
    entropy_conservation_error: float,
    top1_prob: float,
    phi_tolerance: float = 0.05,
    forced_collapse_entropy_threshold: float = 2.0,
    forced_collapse_prob_threshold: float = 0.5,
) -> CollapseSignature:
    """Compute full collapse signature for a token prediction.

    Args:
        entropy: H(softmax(logits))
        pac_ratio_1_2: p1/p2 ratio (None if p2 ~ 0)
        prob_conservation_error: from conservation budget
        entropy_conservation_error: from conservation budget
        top1_prob: probability of the top token
        phi_tolerance: relative tolerance for phi alignment (default 5%)
        forced_collapse_entropy_threshold: H above which high p1 is suspicious
        forced_collapse_prob_threshold: p1 above which forced collapse is flagged

    Returns:
        CollapseSignature with all diagnostics
    """
    sec_phase = classify_sec_phase(entropy)

    # Effective number of choices
    effective_k = math.exp(entropy) if entropy < 20 else float('inf')

    # Phi alignment
    phi_distance = None
    inv_phi_distance = None
    is_phi_aligned = False
    is_xi_aligned = False

    if pac_ratio_1_2 is not None:
        phi_distance = abs(pac_ratio_1_2 - PHI)
        inv_phi_distance = abs(pac_ratio_1_2 - INV_PHI)
        # Check if within tolerance of phi or 1/phi
        is_phi_aligned = (
            phi_distance / PHI < phi_tolerance
            or inv_phi_distance / INV_PHI < phi_tolerance
        )
        # Check xi alignment
        xi_distance = abs(pac_ratio_1_2 - XI)
        is_xi_aligned = xi_distance / XI < phi_tolerance

    # Forced collapse detection
    is_forced_collapse = (
        entropy > forced_collapse_entropy_threshold
        and top1_prob > forced_collapse_prob_threshold
    )

    return CollapseSignature(
        sec_phase=sec_phase,
        entropy=entropy,
        pac_ratio_1_2=pac_ratio_1_2,
        phi_distance=phi_distance,
        inv_phi_distance=inv_phi_distance,
        prob_conservation_error=prob_conservation_error,
        entropy_conservation_error=entropy_conservation_error,
        collapse_magnitude=entropy,
        concentration=top1_prob,
        effective_k=effective_k,
        is_forced_collapse=is_forced_collapse,
        is_phi_aligned=is_phi_aligned,
        is_xi_aligned=is_xi_aligned,
    )
