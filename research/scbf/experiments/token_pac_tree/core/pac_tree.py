"""
PAC Tree Data Structures for Token-Level Collapse Analysis
===========================================================

Each token prediction is modelled as a PAC collapse event:
  - PARENT: the full logit distribution (potential)
  - CHILDREN: the top-k candidate tokens with softmax probs (actualization)
  - COLLAPSE: the moment the model commits to one token

PAC conservation check:
  f(parent) = sum(f(children))
  where f = probability mass, entropy contribution, or information content.

The analogy to quantum mechanics is structural, not metaphorical:
  - logits -> wavefunction amplitudes
  - softmax -> Born rule (probability from amplitude)
  - argmax/sampling -> measurement (collapse)
  - entropy of distribution -> superposition breadth
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Dawn Field Theory constants
PHI = (1 + math.sqrt(5)) / 2       # 1.618033988749895
INV_PHI = 1 / PHI                   # 0.6180339887498949
XI = 1 + math.pi / 55               # 1.05712...
TWO_THIRDS = 2 / 3


@dataclass
class PACNode:
    """A single node in the token-level PAC tree.

    For the parent node:
        - token_id is None (represents the full distribution)
        - probability is 1.0 (total potential)
        - entropy is H(softmax(logits))
        - logit is the max logit (for reference)

    For child nodes:
        - token_id is the vocabulary index
        - probability is the softmax probability
        - entropy is -p*log(p) (contribution to parent entropy)
        - logit is the raw logit value
    """
    token_id: Optional[int] = None
    token_str: Optional[str] = None
    probability: float = 0.0
    logit: float = 0.0
    entropy_contribution: float = 0.0     # -p*log(p) for this node
    rank: int = -1                         # 0 = top prediction
    is_parent: bool = False
    children: List['PACNode'] = field(default_factory=list)

    @property
    def information_content(self) -> float:
        """Surprisal: -log2(p). Infinite for p=0."""
        if self.probability <= 0:
            return float('inf')
        return -math.log2(self.probability)


@dataclass
class TokenPACTree:
    """PAC tree for a single token position in a sequence.

    Represents the collapse from potential (full distribution)
    to actualization (selected token).
    """
    position: int                          # sequence position
    parent: PACNode = field(default_factory=PACNode)
    children: List[PACNode] = field(default_factory=list)

    # Pre-collapse distribution stats
    total_entropy: float = 0.0             # H(softmax(logits))
    top_k: int = 0                         # how many children
    vocab_size: int = 0

    # Conservation accounting
    children_prob_sum: float = 0.0         # should = 1.0 if k = |V|
    children_entropy_sum: float = 0.0      # should = total_entropy
    tail_prob: float = 0.0                 # 1 - sum(top_k probs)
    tail_entropy: float = 0.0             # entropy from tokens outside top-k

    # PAC ratios
    pac_ratio_1_2: Optional[float] = None  # p1/p2 — compare to phi
    pac_ratio_2_3: Optional[float] = None  # p2/p3 — compare to phi
    pac_ratio_1_rest: Optional[float] = None  # p1 / (1-p1)

    # Collapse info
    selected_token_id: Optional[int] = None
    selected_token_str: Optional[str] = None
    selected_rank: int = -1
    is_correct: Optional[bool] = None      # if ground truth available
    ground_truth_id: Optional[int] = None
    ground_truth_rank: Optional[int] = None

    def conservation_error(self) -> float:
        """How far the children's probability mass is from 1.0.
        (Should be ~0 if tail is accounted for.)
        """
        return abs((self.children_prob_sum + self.tail_prob) - 1.0)

    def entropy_conservation_error(self) -> float:
        """How far children's entropy contributions are from total entropy."""
        reconstructed = self.children_entropy_sum + self.tail_entropy
        if self.total_entropy == 0:
            return 0.0
        return abs(reconstructed - self.total_entropy) / self.total_entropy

    def collapse_magnitude(self) -> float:
        """Entropy drop from pre-collapse to post-collapse (single token = 0 entropy)."""
        return self.total_entropy

    def to_dict(self) -> dict:
        """Serialisable dictionary."""
        d = {
            'position': self.position,
            'total_entropy': self.total_entropy,
            'top_k': self.top_k,
            'vocab_size': self.vocab_size,
            'children_prob_sum': self.children_prob_sum,
            'children_entropy_sum': self.children_entropy_sum,
            'tail_prob': self.tail_prob,
            'tail_entropy': self.tail_entropy,
            'pac_ratio_1_2': self.pac_ratio_1_2,
            'pac_ratio_2_3': self.pac_ratio_2_3,
            'pac_ratio_1_rest': self.pac_ratio_1_rest,
            'conservation_error': self.conservation_error(),
            'entropy_conservation_error': self.entropy_conservation_error(),
            'collapse_magnitude': self.collapse_magnitude(),
            'selected_token_id': self.selected_token_id,
            'selected_token_str': self.selected_token_str,
            'selected_rank': self.selected_rank,
            'is_correct': self.is_correct,
            'ground_truth_id': self.ground_truth_id,
            'ground_truth_rank': self.ground_truth_rank,
            'children': [
                {
                    'token_id': c.token_id,
                    'token_str': c.token_str,
                    'probability': c.probability,
                    'logit': c.logit,
                    'entropy_contribution': c.entropy_contribution,
                    'rank': c.rank,
                    'information_content': c.information_content,
                }
                for c in self.children
            ],
        }
        return d


class PACForest:
    """Collection of PAC trees across a full sequence.

    Tracks temporal patterns: how collapse dynamics evolve
    token-by-token through a generation.
    """

    def __init__(self):
        self.trees: List[TokenPACTree] = []
        self.prompt_text: str = ''
        self.model_name: str = ''

    def add_tree(self, tree: TokenPACTree):
        self.trees.append(tree)

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, idx) -> TokenPACTree:
        return self.trees[idx]

    # ── Aggregate statistics ──────────────────────────────────

    def entropy_trajectory(self) -> np.ndarray:
        """Entropy at each token position."""
        return np.array([t.total_entropy for t in self.trees])

    def pac_ratio_trajectory(self) -> np.ndarray:
        """p1/p2 ratio at each token position (NaN where undefined)."""
        return np.array([
            t.pac_ratio_1_2 if t.pac_ratio_1_2 is not None else float('nan')
            for t in self.trees
        ])

    def collapse_magnitude_trajectory(self) -> np.ndarray:
        return np.array([t.collapse_magnitude() for t in self.trees])

    def phi_distance_trajectory(self) -> np.ndarray:
        """Distance of p1/p2 from phi at each position."""
        ratios = self.pac_ratio_trajectory()
        return np.abs(ratios - PHI)

    def inv_phi_distance_trajectory(self) -> np.ndarray:
        """Distance of p1/p2 from 1/phi at each position."""
        ratios = self.pac_ratio_trajectory()
        return np.abs(ratios - INV_PHI)

    def accuracy(self) -> Optional[float]:
        """Fraction of positions where selected == ground truth."""
        scored = [t for t in self.trees if t.is_correct is not None]
        if not scored:
            return None
        return sum(1 for t in scored if t.is_correct) / len(scored)

    def mean_ground_truth_rank(self) -> Optional[float]:
        """Mean rank of the ground truth token in the top-k."""
        ranks = [t.ground_truth_rank for t in self.trees
                 if t.ground_truth_rank is not None]
        if not ranks:
            return None
        return float(np.mean(ranks))

    def summary(self) -> dict:
        """Aggregate statistics for the full sequence."""
        entropies = self.entropy_trajectory()
        pac_ratios = self.pac_ratio_trajectory()
        valid_ratios = pac_ratios[~np.isnan(pac_ratios)]
        phi_dists = np.abs(valid_ratios - PHI) if len(valid_ratios) > 0 else np.array([])
        inv_phi_dists = np.abs(valid_ratios - INV_PHI) if len(valid_ratios) > 0 else np.array([])

        return {
            'model': self.model_name,
            'prompt': self.prompt_text[:200],
            'n_tokens': len(self.trees),
            'entropy_mean': float(np.mean(entropies)) if len(entropies) > 0 else None,
            'entropy_std': float(np.std(entropies)) if len(entropies) > 0 else None,
            'entropy_min': float(np.min(entropies)) if len(entropies) > 0 else None,
            'entropy_max': float(np.max(entropies)) if len(entropies) > 0 else None,
            'pac_ratio_mean': float(np.mean(valid_ratios)) if len(valid_ratios) > 0 else None,
            'pac_ratio_std': float(np.std(valid_ratios)) if len(valid_ratios) > 0 else None,
            'pac_ratio_median': float(np.median(valid_ratios)) if len(valid_ratios) > 0 else None,
            'phi_distance_mean': float(np.mean(phi_dists)) if len(phi_dists) > 0 else None,
            'inv_phi_distance_mean': float(np.mean(inv_phi_dists)) if len(inv_phi_dists) > 0 else None,
            'accuracy': self.accuracy(),
            'mean_ground_truth_rank': self.mean_ground_truth_rank(),
        }

    def to_dict(self) -> dict:
        """Full serialisable representation."""
        return {
            'summary': self.summary(),
            'trees': [t.to_dict() for t in self.trees],
        }


# ── Builder ───────────────────────────────────────────────────────────


def build_pac_tree_from_logits(
    logits: torch.Tensor,
    position: int,
    tokenizer=None,
    top_k: int = 10,
    selected_token_id: Optional[int] = None,
    ground_truth_id: Optional[int] = None,
) -> TokenPACTree:
    """Build a PAC tree from a single position's logit vector.

    Args:
        logits: shape [vocab_size] — raw logits for this position
        position: token index in the sequence
        tokenizer: optional, for decoding token strings
        top_k: number of children (top candidates)
        selected_token_id: which token was actually selected (argmax or sampled)
        ground_truth_id: ground truth token id (for accuracy tracking)

    Returns:
        TokenPACTree with parent + top-k children + conservation accounting
    """
    logits = logits.detach().float()
    vocab_size = logits.shape[0]

    # ── Full distribution ────────────────────────────────────
    probs = F.softmax(logits, dim=0)
    log_probs = F.log_softmax(logits, dim=0)

    # Total entropy: H = -sum(p * log(p))
    total_entropy = -(probs * log_probs).sum().item()

    # ── Top-k children ───────────────────────────────────────
    topk_vals, topk_ids = torch.topk(probs, k=min(top_k, vocab_size))
    topk_logits = logits[topk_ids]
    topk_log_probs = log_probs[topk_ids]

    children = []
    children_prob_sum = 0.0
    children_entropy_sum = 0.0

    for rank_i in range(len(topk_ids)):
        tid = topk_ids[rank_i].item()
        p = topk_vals[rank_i].item()
        lp = topk_log_probs[rank_i].item()
        raw_logit = topk_logits[rank_i].item()
        ent_contrib = -p * lp  # contribution to total entropy

        token_str = None
        if tokenizer is not None:
            try:
                token_str = tokenizer.decode([tid])
            except Exception:
                token_str = f'<{tid}>'

        node = PACNode(
            token_id=tid,
            token_str=token_str,
            probability=p,
            logit=raw_logit,
            entropy_contribution=ent_contrib,
            rank=rank_i,
            is_parent=False,
        )
        children.append(node)
        children_prob_sum += p
        children_entropy_sum += ent_contrib

    # ── Tail (everything outside top-k) ──────────────────────
    tail_prob = 1.0 - children_prob_sum
    tail_entropy = total_entropy - children_entropy_sum

    # ── PAC ratios ───────────────────────────────────────────
    pac_ratio_1_2 = None
    pac_ratio_2_3 = None
    pac_ratio_1_rest = None

    if len(children) >= 2 and children[1].probability > 1e-12:
        pac_ratio_1_2 = children[0].probability / children[1].probability

    if len(children) >= 3 and children[2].probability > 1e-12:
        pac_ratio_2_3 = children[1].probability / children[2].probability

    if len(children) >= 1 and children[0].probability < 1.0 - 1e-12:
        pac_ratio_1_rest = children[0].probability / (1.0 - children[0].probability)

    # ── Parent node ──────────────────────────────────────────
    parent = PACNode(
        token_id=None,
        token_str='[DISTRIBUTION]',
        probability=1.0,
        logit=logits.max().item(),
        entropy_contribution=total_entropy,
        rank=-1,
        is_parent=True,
        children=children,
    )

    # ── Selected / ground truth tracking ─────────────────────
    if selected_token_id is None:
        selected_token_id = topk_ids[0].item()  # default: greedy

    selected_rank = -1
    for c in children:
        if c.token_id == selected_token_id:
            selected_rank = c.rank
            break

    selected_str = None
    if tokenizer is not None:
        try:
            selected_str = tokenizer.decode([selected_token_id])
        except Exception:
            selected_str = f'<{selected_token_id}>'

    gt_rank = None
    is_correct = None
    if ground_truth_id is not None:
        is_correct = (selected_token_id == ground_truth_id)
        for c in children:
            if c.token_id == ground_truth_id:
                gt_rank = c.rank
                break
        if gt_rank is None:
            # Ground truth not in top-k — find its actual rank
            all_sorted = torch.argsort(probs, descending=True)
            for r, tid in enumerate(all_sorted):
                if tid.item() == ground_truth_id:
                    gt_rank = r
                    break

    # ── Assemble tree ────────────────────────────────────────
    tree = TokenPACTree(
        position=position,
        parent=parent,
        children=children,
        total_entropy=total_entropy,
        top_k=len(children),
        vocab_size=vocab_size,
        children_prob_sum=children_prob_sum,
        children_entropy_sum=children_entropy_sum,
        tail_prob=tail_prob,
        tail_entropy=tail_entropy,
        pac_ratio_1_2=pac_ratio_1_2,
        pac_ratio_2_3=pac_ratio_2_3,
        pac_ratio_1_rest=pac_ratio_1_rest,
        selected_token_id=selected_token_id,
        selected_token_str=selected_str,
        selected_rank=selected_rank,
        is_correct=is_correct,
        ground_truth_id=ground_truth_id,
        ground_truth_rank=gt_rank,
    )

    return tree
