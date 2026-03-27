"""Language Module — Embedding grafting + n-gram transitions + concentration gating.

Ported from GAIA v1 gaia_prime (2,116 lines across 6 files → ~430 lines).
Cross-model embedding grafting, counting-based transition learning,
multi-depth concentration quality gating. O(1) learning per token.

Source material:
    embeddings.py (POC-016, 017, 020): 100% cross-model graft success
    transitions.py (POC-021, 022): 65% hit rate, log learning R²=0.973
    concentration.py (POC-023, 024): φ⁻¹ threshold, +3.6% reject-resample

Components:
    EmbeddingStore: Frozen token embeddings, nearest-neighbor lookup.
    TransitionCounter: N-gram counting with multi-context prediction.
    ConcentrationGate: Multi-depth agreement quality gate (φ⁻¹ threshold).
    LanguageModule: GAIAModule wrapper for the full language stack.

Dropped from v1:
    pac_tree.py — v2 Memory module's PACTree covers this.
    generator.py — text generation is an interface concern (M7/M8).
    model.py — the ConservationBus IS the orchestrator in v2.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import torch

from gaia.core.types import FieldState, RBFBalance, SECPhase

# DFT constants — canonical source: fracton.constants
try:
    from fracton.constants import PHI, PHI_INV
except ImportError:
    PHI = (1 + math.sqrt(5)) / 2
    PHI_INV = 1.0 / PHI  # 0.618... — critical threshold from POC-024
N_BINS_DEFAULT = 256  # Quantization bins for embedding-free mode


# ─── Data Structures ─────────────────────────────────────────────


@dataclass
class TransitionStats:
    """Statistics for the transition counter."""

    total_transitions: int = 0
    unique_contexts: int = 0
    unique_transitions: int = 0


@dataclass
class ConcentrationResult:
    """Result of multi-depth concentration analysis."""

    concentration: float  # Fraction of depths agreeing with majority
    predicted_token: int  # Token with highest agreement
    depth_votes: dict[int, int]  # depth → predicted token
    confidence: float  # Margin over second place
    is_high_quality: bool  # concentration >= φ⁻¹ threshold


@dataclass
class LanguageMetrics:
    """Module-level metrics exposed after each process() call."""

    total_transitions: int = 0
    unique_contexts: int = 0
    concentration: float = 0.0
    is_high_quality: bool = True
    prediction_entropy: float = 0.0
    blend_weight: float = 0.0
    has_embeddings: bool = False
    vocab_size: int = 0
    step_count: int = 0


# ─── EmbeddingStore ──────────────────────────────────────────────


class EmbeddingStore:
    """Frozen token embeddings extracted from pretrained models.

    Accepts raw torch.Tensor — no mandatory HuggingFace dependency.
    Optional factory methods (from_pretrained) lazily import transformers.

    v1 origin: gaia_prime/embeddings.py (GraftedEmbeddings)
    Validated: POC-016/017/020 — 100% cross-model graft success.
    """

    def __init__(self, embeddings: torch.Tensor) -> None:
        if embeddings.dim() != 2:
            raise ValueError(f"Expected 2D tensor (vocab, embed_dim), got {embeddings.dim()}D")
        self.embeddings = embeddings.detach()
        self.vocab_size = embeddings.shape[0]
        self.embed_dim = embeddings.shape[1]
        # Pre-compute norms for cosine similarity
        self._norms = torch.norm(embeddings, dim=1, keepdim=True).clamp(min=1e-8)

    @staticmethod
    def from_tensor(embeddings: torch.Tensor) -> EmbeddingStore:
        """Create from a raw embedding matrix."""
        return EmbeddingStore(embeddings)

    @staticmethod
    def from_pretrained(model_name: str) -> EmbeddingStore:
        """Create from a HuggingFace model (lazy import).

        Supports GPT-2, Pythia, and any AutoModel with embedding weights.
        """
        try:
            from transformers import AutoModel  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "transformers package required for from_pretrained(). "
                "Install with: pip install transformers"
            ) from e

        model = AutoModel.from_pretrained(model_name)
        # Most models expose embeddings via get_input_embeddings()
        embed_layer = model.get_input_embeddings()
        embeddings = embed_layer.weight.detach().clone()
        del model
        return EmbeddingStore(embeddings)

    def get_embedding(self, token_id: int) -> torch.Tensor:
        """Get embedding vector for a single token ID."""
        return self.embeddings[token_id]

    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for multiple token IDs."""
        return self.embeddings[token_ids]

    def nearest_token(
        self, vector: torch.Tensor, top_k: int = 5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find nearest tokens by cosine similarity.

        Returns:
            (token_ids, similarities) — both shape (top_k,)
        """
        vec_norm = torch.norm(vector).clamp(min=1e-8)
        similarities = (self.embeddings @ vector) / (self._norms.squeeze() * vec_norm)
        k = min(top_k, self.vocab_size)
        top_sims, top_ids = torch.topk(similarities, k)
        return top_ids, top_sims


# ─── TransitionCounter ──────────────────────────────────────────


class TransitionCounter:
    """N-gram transition counting with multi-context prediction.

    Learning is pure counting — no gradients, no backprop.
    Stores P(next_token | context) for all observed contexts as dicts.

    v1 origin: gaia_prime/transitions.py (TransitionMatrix)
    Validated: POC-022 — 65% hit rate at 100K vocab, log R²=0.973
    """

    def __init__(self, max_context_len: int = 3) -> None:
        self.max_context_len = max_context_len
        # context_hash → {next_token → count}
        self._counts: dict[int, dict[int, int]] = {}
        self._totals: dict[int, int] = {}
        self.stats = TransitionStats()

    def _hash(self, context: tuple[int, ...]) -> int:
        """Hash a context tuple."""
        return hash(context)

    def learn(self, context: tuple[int, ...], next_token: int) -> None:
        """Learn a single context → next_token transition."""
        h = self._hash(context)

        if h not in self._counts:
            self._counts[h] = {}
            self._totals[h] = 0
            self.stats.unique_contexts += 1

        if next_token not in self._counts[h]:
            self._counts[h][next_token] = 0
            self.stats.unique_transitions += 1

        self._counts[h][next_token] += 1
        self._totals[h] += 1
        self.stats.total_transitions += 1

    def learn_from_sequence(
        self, token_ids: list[int], context_lengths: list[int] | None = None
    ) -> None:
        """Learn all n-gram transitions from a token sequence.

        Args:
            token_ids: List of integer token IDs.
            context_lengths: Which context lengths to learn (default: 1..max_context_len).
        """
        if context_lengths is None:
            context_lengths = list(range(1, self.max_context_len + 1))

        seq_len = len(token_ids)
        for ctx_len in context_lengths:
            for i in range(seq_len - ctx_len):
                context = tuple(token_ids[i : i + ctx_len])
                next_token = token_ids[i + ctx_len]
                self.learn(context, next_token)

    def predict(
        self, context: tuple[int, ...], top_k: int = 5, temperature: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next token given context.

        Returns:
            (token_ids, probs) — both 1D tensors. Empty if context unknown.
        """
        h = self._hash(context)

        if h not in self._counts:
            return torch.tensor([], dtype=torch.long), torch.tensor([])

        counts = self._counts[h]
        total = self._totals[h]

        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]

        token_ids = torch.tensor([t for t, _ in sorted_items], dtype=torch.long)
        raw_probs = torch.tensor([c / total for _, c in sorted_items])

        # Apply temperature
        if temperature != 1.0 and len(raw_probs) > 0:
            logits = raw_probs.clamp(min=1e-10).log() / temperature
            probs = torch.softmax(logits, dim=0)
        else:
            probs = raw_probs

        return token_ids, probs

    def hit_rate_estimate(self) -> float:
        """Fraction of contexts with more than one observation."""
        if not self._totals:
            return 0.0
        multi_obs = sum(1 for t in self._totals.values() if t > 1)
        return multi_obs / len(self._totals)


# ─── ConcentrationGate ───────────────────────────────────────────


class ConcentrationGate:
    """Multi-depth prediction agreement quality gate.

    When predictions from different n-gram depths agree, output is reliable.
    When they disagree, we're in uncertain territory (hallucination risk).

    v1 origin: gaia_prime/concentration.py (ConcentrationMonitor)
    Validated: POC-023 — λ₃≈0.5 universal, +3.6% quality with reject-resample.
    Threshold: POC-024 — φ⁻¹ = 0.618 at critical transition.
    """

    def __init__(
        self, threshold: float = PHI_INV, max_depth: int = 5, min_depths: int = 2
    ) -> None:
        self.threshold = threshold
        self.max_depth = max_depth
        self.min_depths = min_depths
        self._total_analyzed = 0
        self._high_quality_count = 0
        self._concentration_sum = 0.0

    def evaluate(
        self, depth_predictions: dict[int, tuple[torch.Tensor, torch.Tensor]]
    ) -> ConcentrationResult:
        """Evaluate concentration across multiple prediction depths.

        Args:
            depth_predictions: depth → (token_ids, probs) from different n-gram levels.

        Returns:
            ConcentrationResult with concentration score and quality flag.
        """
        self._total_analyzed += 1

        if not depth_predictions:
            return ConcentrationResult(
                concentration=0.0,
                predicted_token=-1,
                depth_votes={},
                confidence=0.0,
                is_high_quality=False,
            )

        # Get top prediction from each depth
        depth_votes: dict[int, int] = {}
        for depth, (token_ids, probs) in depth_predictions.items():
            if len(token_ids) > 0:
                top_idx = int(probs.argmax().item())
                depth_votes[depth] = int(token_ids[top_idx].item())

        if not depth_votes:
            return ConcentrationResult(
                concentration=0.0,
                predicted_token=-1,
                depth_votes={},
                confidence=0.0,
                is_high_quality=False,
            )

        # Count agreement
        vote_counts: dict[int, int] = {}
        for token in depth_votes.values():
            vote_counts[token] = vote_counts.get(token, 0) + 1

        majority_token = max(vote_counts, key=vote_counts.get)  # type: ignore[arg-type]
        majority_count = vote_counts[majority_token]
        total_votes = len(depth_votes)

        concentration = majority_count / total_votes

        # Confidence = margin over second place
        sorted_counts = sorted(vote_counts.values(), reverse=True)
        second_count = sorted_counts[1] if len(sorted_counts) > 1 else 0
        confidence = (majority_count - second_count) / total_votes

        # POC-023: concentration only meaningful with multiple depths
        if total_votes < self.min_depths:
            concentration = min(concentration, self.threshold * 0.5)

        is_high_quality = concentration >= self.threshold

        # Update stats
        self._concentration_sum += concentration
        if is_high_quality:
            self._high_quality_count += 1

        return ConcentrationResult(
            concentration=concentration,
            predicted_token=majority_token,
            depth_votes=depth_votes,
            confidence=confidence,
            is_high_quality=is_high_quality,
        )

    def should_reject(self, result: ConcentrationResult) -> bool:
        """Whether to reject and resample (POC-023: +3.6% quality)."""
        return not result.is_high_quality

    @property
    def stats(self) -> dict:
        """Monitoring statistics."""
        total = self._total_analyzed
        return {
            "total_analyzed": total,
            "high_quality_rate": self._high_quality_count / total if total > 0 else 0.0,
            "mean_concentration": self._concentration_sum / total if total > 0 else 0.0,
        }


# ─── LanguageModule ──────────────────────────────────────────────


class LanguageModule:
    """Language processing module for the GAIA conservation bus.

    Discretizes FieldState tensors → learns n-gram transitions →
    predicts at multiple depths → gates via concentration →
    blends prediction into output. PAC-conserving.

    Works with or without an EmbeddingStore:
    - With: nearest-neighbor token lookup against frozen embeddings.
    - Without: adaptive quantization (torch.bucketize) into discrete bins.

    v1 origin: gaia_prime/model.py (GAIA_Prime orchestrator, 473 lines)
    """

    def __init__(
        self,
        embeddings: EmbeddingStore | None = None,
        max_context_len: int = 3,
        concentration_threshold: float = PHI_INV,
        prediction_blend: float = 0.1,
        max_depth: int = 5,
        n_bins: int = N_BINS_DEFAULT,
    ) -> None:
        self._embeddings = embeddings
        self._counter = TransitionCounter(max_context_len=max_context_len)
        self._gate = ConcentrationGate(
            threshold=concentration_threshold, max_depth=max_depth
        )
        self._prediction_blend = prediction_blend
        self._max_depth = max_depth
        self._n_bins = n_bins
        self._step_count = 0
        self._last_metrics: LanguageMetrics | None = None

        # Adaptive bin boundaries (computed from first tensor seen)
        self._bin_boundaries: torch.Tensor | None = None
        self._integer_mode: bool = False

    @property
    def name(self) -> str:
        return "language"

    @property
    def metrics(self) -> LanguageMetrics | None:
        return self._last_metrics

    @property
    def counter(self) -> TransitionCounter:
        return self._counter

    @property
    def gate(self) -> ConcentrationGate:
        return self._gate

    # ── Discretization ───────────────────────────────────────────

    @staticmethod
    def _is_integer_tensor(flat: torch.Tensor) -> bool:
        """Check if tensor values are already integer token IDs."""
        if flat.numel() == 0:
            return False
        rounded = flat.round()
        if not torch.allclose(flat, rounded, atol=1e-6):
            return False
        if float(flat.min()) < -0.5:
            return False
        return True

    def _discretize(self, tensor: torch.Tensor) -> list[int]:
        """Convert continuous tensor to discrete token sequence.

        With EmbeddingStore: each element mapped to nearest vocabulary token.
        Without embeddings, two paths:
        - Integer passthrough: values already in [0, n_bins) pass through as-is.
        - Quantization fallback: adaptive binning via torch.bucketize.
        """
        flat = tensor.flatten()

        if self._embeddings is not None:
            # Each scalar element → nearest embedding (treat as 1D vector)
            token_ids = []
            for val in flat:
                vec = val.unsqueeze(0).expand(self._embeddings.embed_dim)
                ids, _ = self._embeddings.nearest_token(vec, top_k=1)
                token_ids.append(int(ids[0].item()))
            return token_ids

        # Integer passthrough: values are already token IDs
        if self._is_integer_tensor(flat) and float(flat.max()) < self._n_bins:
            self._integer_mode = True
            return flat.round().long().tolist()

        # Quantization fallback: adaptive bins
        self._integer_mode = False
        if self._bin_boundaries is None or len(self._bin_boundaries) == 0:
            # Initialize bin boundaries from this tensor's value range
            vmin, vmax = float(flat.min()), float(flat.max())
            if vmax - vmin < 1e-10:
                vmax = vmin + 1.0
            self._bin_boundaries = torch.linspace(vmin, vmax, self._n_bins + 1)[1:-1]

        token_ids = torch.bucketize(flat, self._bin_boundaries).tolist()
        return token_ids

    def _token_to_value(self, token_id: int) -> float:
        """Convert a discrete token back to a continuous value (bin center)."""
        if self._embeddings is not None:
            # Use mean of embedding vector as scalar value
            return float(self._embeddings.get_embedding(token_id).mean().item())

        # Integer mode: token ID is the value
        if self._integer_mode:
            return float(token_id)

        # Bin center from boundaries
        if self._bin_boundaries is not None and len(self._bin_boundaries) > 0:
            bounds = self._bin_boundaries
            if token_id <= 0:
                return float(bounds[0])
            elif token_id >= len(bounds):
                return float(bounds[-1])
            else:
                return float((bounds[token_id - 1] + bounds[token_id]) / 2.0)
        return 0.0

    # ── Core Protocol ────────────────────────────────────────────

    def process(self, field_state: FieldState) -> FieldState:
        """Process field state through language module.

        1. Discretize tensor → token sequence.
        2. Learn n-gram transitions.
        3. Predict at multiple depths.
        4. Gate via concentration.
        5. Blend prediction (if high quality).
        6. PAC boundary enforcement.
        """
        self._step_count += 1
        result = field_state.clone()
        input_energy = field_state.total_energy()

        # 1. Discretize
        tokens = self._discretize(field_state.tensor)

        # 2. Learn transitions
        if len(tokens) > 1:
            self._counter.learn_from_sequence(tokens)

        # 3. Predict at multiple depths
        depth_predictions: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if len(tokens) >= 1:
            for depth in range(1, min(self._max_depth + 1, len(tokens) + 1)):
                context = tuple(tokens[-depth:])
                pred_ids, pred_probs = self._counter.predict(context, top_k=5)
                if len(pred_ids) > 0:
                    depth_predictions[depth] = (pred_ids, pred_probs)

        # 4. Concentration gate
        gate_result = self._gate.evaluate(depth_predictions)

        # 5. Blend prediction into output (if high quality)
        actual_blend = 0.0
        prediction_entropy = 0.0
        output = field_state.tensor.clone()

        if gate_result.is_high_quality and gate_result.predicted_token >= 0:
            # Convert predicted token to a continuous signal
            pred_value = self._token_to_value(gate_result.predicted_token)

            # Build prediction tensor: last element replaced with prediction
            prediction = output.clone()
            flat_pred = prediction.flatten()
            if len(flat_pred) > 0:
                flat_pred[-1] = pred_value
                prediction = flat_pred.view_as(output)

            # Blend
            actual_blend = self._prediction_blend
            output = (1 - actual_blend) * output + actual_blend * prediction

        # Compute prediction entropy from deepest available depth
        if depth_predictions:
            deepest = max(depth_predictions.keys())
            _, probs = depth_predictions[deepest]
            log_probs = probs.clamp(min=1e-10).log()
            prediction_entropy = float(-(probs * log_probs).sum().item())

        # 6. PAC boundary enforcement — scale to match input energy
        output_energy = float(torch.sum(output).item())
        if abs(output_energy) > 1e-10:
            output = output * (input_energy / output_energy)

        result.tensor = output
        result.provenance.append(self.name)

        # Populate metrics
        self._last_metrics = LanguageMetrics(
            total_transitions=self._counter.stats.total_transitions,
            unique_contexts=self._counter.stats.unique_contexts,
            concentration=gate_result.concentration,
            is_high_quality=gate_result.is_high_quality,
            prediction_entropy=prediction_entropy,
            blend_weight=actual_blend,
            has_embeddings=self._embeddings is not None,
            vocab_size=self._embeddings.vocab_size if self._embeddings else self._n_bins,
            step_count=self._step_count,
        )

        return result

    def phase(self) -> SECPhase:
        """SEC phase from concentration (prediction agreement).

        High concentration = crystallized knowledge (strong agreement).
        Low concentration = chaotic territory (disagreement).
        """
        if self._last_metrics is None:
            return SECPhase.ORDERED

        c = self._last_metrics.concentration
        if c >= 0.8:
            return SECPhase.CRYSTALLIZED
        elif c >= PHI_INV:
            return SECPhase.ORDERED
        elif c >= 0.3:
            return SECPhase.TRANSITIONAL
        return SECPhase.CHAOTIC

    def health(self) -> RBFBalance:
        """RBF balance from language state.

        Energy = concentration (prediction quality).
        Information = normalized prediction entropy.
        Memory = normalized context count (knowledge load).

        Module self-suppresses when concentration is low — correct behavior,
        poor predictions should not modify the tensor.
        """
        if self._last_metrics:
            energy = self._last_metrics.concentration
            information = min(self._last_metrics.prediction_entropy, 5.0) / 5.0
            memory = min(self._last_metrics.unique_contexts, 10000) / 10000.0
        else:
            energy = 0.5
            information = 0.5
            memory = 0.0
        return RBFBalance.compute(energy=energy, information=information, memory=memory)
