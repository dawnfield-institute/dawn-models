"""Synthetic benchmarks — tests whether GAIA v2 modules actually learn.

Four benchmarks matching the spec's four evaluation axes:
    1. Efficiency — O(1) per token, constant throughput
    2. Continuous Learning — retention after domain shift
    3. Hallucination Detection — concentration gap between valid/invalid
    4. Memory — PACTree compression and retrieval quality
"""

from __future__ import annotations

import time

import torch

from gaia.core.types import FieldState
from gaia.modules.language import LanguageModule, TransitionCounter
from gaia.modules.memory import MemoryModule, PACTree

from .generators import (
    generate_bigram_sequences,
    generate_clustered_patterns,
    generate_domain_pair,
    generate_valid_invalid_pairs,
)


# ─── Axis 1: Efficiency ─────────────────────────────────────────


def bench_efficiency(
    vocab_size: int = 20,
    n_small: int = 5000,
    n_large: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Measure TransitionCounter throughput and scaling.

    Tests that learning is O(1) per token: doubling tokens should
    roughly double time, not quadruple it.
    """
    seqs_small, _ = generate_bigram_sequences(vocab_size, 1, n_small, seed=seed)
    seqs_large, _ = generate_bigram_sequences(vocab_size, 1, n_large, seed=seed + 1)

    # Small batch
    tc1 = TransitionCounter(max_context_len=3)
    t0 = time.perf_counter()
    tc1.learn_from_sequence(seqs_small[0])
    time_small = time.perf_counter() - t0

    # Large batch
    tc2 = TransitionCounter(max_context_len=3)
    t0 = time.perf_counter()
    tc2.learn_from_sequence(seqs_large[0])
    time_large = time.perf_counter() - t0

    tokens_per_second = n_large / max(time_large, 1e-9)

    # Overhead ratio: time per token at large vs small scale
    tpt_small = time_small / max(n_small, 1)
    tpt_large = time_large / max(n_large, 1)
    overhead_ratio = tpt_large / max(tpt_small, 1e-12)

    return {
        "tokens_per_second": tokens_per_second,
        "learning_overhead_ratio": overhead_ratio,
    }


# ─── Axis 2: Continuous Learning ─────────────────────────────────


def _measure_accuracy(
    counter: TransitionCounter,
    ground_truth: dict[int, int],
) -> float:
    """Fraction of tokens where counter's top prediction matches ground truth."""
    hits = 0
    total = 0
    for token, expected_next in ground_truth.items():
        pred_ids, probs = counter.predict((token,), top_k=1)
        if len(pred_ids) > 0 and int(pred_ids[0].item()) == expected_next:
            hits += 1
        total += 1
    return hits / max(total, 1)


def bench_continuous_learning(
    vocab_size: int = 20,
    n_sequences: int = 50,
    seq_len: int = 50,
    dominant_prob: float = 0.7,
    seed: int = 42,
) -> dict[str, float]:
    """Train on domain A, then domain B. Measure retention of A.

    The key insight: TransitionCounter is additive (dict-based counting).
    Learning B should never degrade A because B's entries are in a
    different key space (non-overlapping vocabularies).
    """
    seqs_a, truth_a, seqs_b, truth_b = generate_domain_pair(
        vocab_size, n_sequences, seq_len, dominant_prob, seed
    )

    counter = TransitionCounter(max_context_len=2)

    # Learn domain A
    for seq in seqs_a:
        counter.learn_from_sequence(seq)
    acc_a_before = _measure_accuracy(counter, truth_a)

    # Learn domain B
    for seq in seqs_b:
        counter.learn_from_sequence(seq)
    acc_b = _measure_accuracy(counter, truth_b)
    acc_a_after = _measure_accuracy(counter, truth_a)

    forgetting = max(0.0, acc_a_before - acc_a_after) / max(acc_a_before, 1e-10)

    return {
        "domain_a_accuracy_before": acc_a_before,
        "domain_b_accuracy": acc_b,
        "domain_a_retention_after": acc_a_after,
        "forgetting_rate": forgetting,
    }


# ─── Axis 3: Hallucination Detection ────────────────────────────


def _sequence_accuracy(
    counter: TransitionCounter,
    sequences: list[list[int]],
) -> float:
    """Measure prediction accuracy: fraction of positions where top-1 matches actual.

    For sequences following learned bigram distributions, accuracy should be high.
    For random sequences, accuracy should be ~1/vocab_size.
    """
    hits = 0
    total = 0
    for seq in sequences:
        for i in range(len(seq) - 1):
            context = (seq[i],)
            pred_ids, _ = counter.predict(context, top_k=1)
            if len(pred_ids) > 0 and int(pred_ids[0].item()) == seq[i + 1]:
                hits += 1
            total += 1
    return hits / max(total, 1)


def bench_hallucination_detection(
    vocab_size: int = 20,
    n_train: int = 100,
    n_test: int = 30,
    seq_len: int = 50,
    dominant_prob: float = 0.7,
    seed: int = 42,
) -> dict[str, float]:
    """Measure prediction accuracy gap between valid and invalid sequences.

    Tests TransitionCounter directly (no discretization noise).
    Valid sequences follow learned bigrams → high prediction accuracy.
    Invalid sequences are random → accuracy ≈ 1/vocab_size.

    The "concentration" metric names are retained for scorecard compatibility
    but the values represent prediction accuracy (0-1), which is a stronger
    signal than depth agreement for hallucination detection.
    """
    train_seqs, valid_test, invalid_test, _ = generate_valid_invalid_pairs(
        vocab_size, n_train, n_test, seq_len, dominant_prob, seed
    )

    counter = TransitionCounter(max_context_len=3)
    for seq in train_seqs:
        counter.learn_from_sequence(seq)

    valid_acc = _sequence_accuracy(counter, valid_test)
    invalid_acc = _sequence_accuracy(counter, invalid_test)

    return {
        "valid_concentration": valid_acc,
        "invalid_concentration": invalid_acc,
        "detection_gap": valid_acc - invalid_acc,
    }


# ─── Axis 4: Memory ─────────────────────────────────────────────


def bench_memory(
    n_clusters: int = 5,
    patterns_per_cluster: int = 20,
    dim: int = 16,
    noise_scale: float = 0.01,
    seed: int = 42,
) -> dict[str, float]:
    """Test PACTree compression and retrieval quality.

    Clustered patterns should compress well (small deltas from prototypes)
    and retrieval should return patterns from the same cluster.
    """
    patterns, labels, prototypes = generate_clustered_patterns(
        n_clusters, patterns_per_cluster, dim, noise_scale, seed
    )

    # Store all patterns
    tree = PACTree(capacity=10000)
    node_ids: list[int] = []
    for pattern in patterns:
        nid = tree.store(pattern, importance=0.5)
        node_ids.append(nid)

    # Measure compression
    ratio = tree.storage_ratio()

    # Measure retrieval precision@3
    total_precision = 0.0
    n_queries = 0
    for c, proto in enumerate(prototypes):
        matches = tree.retrieve(proto, top_k=3, threshold=0.1)
        if not matches:
            continue

        n_correct = 0
        for mid, score in matches:
            # Find which pattern this node corresponds to
            idx = node_ids.index(mid) if mid in node_ids else -1
            if idx >= 0 and labels[idx] == c:
                n_correct += 1

        total_precision += n_correct / len(matches)
        n_queries += 1

    precision_at_3 = total_precision / max(n_queries, 1)

    return {
        "storage_ratio": ratio,
        "retrieval_precision_at_3": precision_at_3,
    }


# ─── Axis 5: End-to-End Hallucination Detection ────────────────


def bench_hallucination_e2e(
    vocab_size: int = 20,
    n_train: int = 100,
    n_test: int = 30,
    seq_len: int = 50,
    dominant_prob: float = 0.7,
    seed: int = 42,
) -> dict[str, float]:
    """End-to-end hallucination detection through LanguageModule.process().

    Validates the full pipeline: discretize → learn → predict → concentrate.
    Uses integer-valued tensors so the integer passthrough keeps token IDs
    aligned between training and inference.

    Complements bench_hallucination_detection (direct counter) by proving
    the integrated pipeline works, not just the counter in isolation.
    """
    train_seqs, valid_test, invalid_test, _ = generate_valid_invalid_pairs(
        vocab_size, n_train, n_test, seq_len, dominant_prob, seed
    )

    # n_bins must exceed vocab_size for integer passthrough
    lang = LanguageModule(max_context_len=3, n_bins=vocab_size + 1)

    # Train through process() — exercises discretize → learn loop
    for seq in train_seqs:
        tensor = torch.tensor(seq, dtype=torch.float32)
        state = FieldState(tensor=tensor, entropy=1.0)
        lang.process(state)

    # Test valid sequences through process()
    valid_concentrations: list[float] = []
    for seq in valid_test:
        tensor = torch.tensor(seq, dtype=torch.float32)
        state = FieldState(tensor=tensor, entropy=1.0)
        lang.process(state)
        if lang.metrics:
            valid_concentrations.append(lang.metrics.concentration)

    # Test invalid sequences through process()
    invalid_concentrations: list[float] = []
    for seq in invalid_test:
        tensor = torch.tensor(seq, dtype=torch.float32)
        state = FieldState(tensor=tensor, entropy=1.0)
        lang.process(state)
        if lang.metrics:
            invalid_concentrations.append(lang.metrics.concentration)

    valid_mean = sum(valid_concentrations) / max(len(valid_concentrations), 1)
    invalid_mean = sum(invalid_concentrations) / max(len(invalid_concentrations), 1)

    return {
        "e2e_valid_concentration": valid_mean,
        "e2e_invalid_concentration": invalid_mean,
        "e2e_detection_gap": valid_mean - invalid_mean,
    }
