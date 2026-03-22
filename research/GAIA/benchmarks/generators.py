"""Synthetic data generators with known ground-truth structure.

Current tests use torch.randn() — no learnable structure. These generators
produce sequences with deterministic bigram distributions so we can measure
whether TransitionCounter, ConcentrationGate, and PACTree actually learn.
"""

from __future__ import annotations

import random

import torch


def generate_bigram_sequences(
    vocab_size: int = 20,
    n_sequences: int = 100,
    seq_len: int = 50,
    dominant_prob: float = 0.7,
    seed: int = 42,
) -> tuple[list[list[int]], dict[int, int]]:
    """Generate sequences from a known bigram distribution.

    Each token i has a dominant successor: (i * 7 + 3) % vocab_size.
    P(dominant) = dominant_prob, rest uniform.

    Returns:
        (sequences, ground_truth) where ground_truth maps token → dominant successor.
    """
    rng = random.Random(seed)

    # Build ground truth bigram table
    ground_truth: dict[int, int] = {}
    for i in range(vocab_size):
        ground_truth[i] = (i * 7 + 3) % vocab_size

    sequences: list[list[int]] = []
    for _ in range(n_sequences):
        seq = [rng.randint(0, vocab_size - 1)]  # random start
        for _ in range(seq_len - 1):
            prev = seq[-1]
            if rng.random() < dominant_prob:
                seq.append(ground_truth[prev])
            else:
                seq.append(rng.randint(0, vocab_size - 1))
        sequences.append(seq)

    return sequences, ground_truth


def generate_domain_pair(
    vocab_size: int = 20,
    n_sequences: int = 50,
    seq_len: int = 50,
    dominant_prob: float = 0.7,
    seed: int = 42,
) -> tuple[list[list[int]], dict[int, int], list[list[int]], dict[int, int]]:
    """Generate two domains with non-overlapping vocabularies.

    Domain A: tokens [0, vocab_size/2)
    Domain B: tokens [vocab_size/2, vocab_size)

    Returns:
        (seqs_a, truth_a, seqs_b, truth_b)
    """
    half = vocab_size // 2
    rng = random.Random(seed)

    # Domain A: tokens 0..half-1
    truth_a: dict[int, int] = {}
    for i in range(half):
        truth_a[i] = (i * 3 + 1) % half

    seqs_a: list[list[int]] = []
    for _ in range(n_sequences):
        seq = [rng.randint(0, half - 1)]
        for _ in range(seq_len - 1):
            prev = seq[-1]
            if rng.random() < dominant_prob:
                seq.append(truth_a[prev])
            else:
                seq.append(rng.randint(0, half - 1))
        seqs_a.append(seq)

    # Domain B: tokens half..vocab_size-1
    truth_b: dict[int, int] = {}
    for i in range(half, vocab_size):
        truth_b[i] = half + ((i - half) * 5 + 2) % half

    seqs_b: list[list[int]] = []
    for _ in range(n_sequences):
        seq = [rng.randint(half, vocab_size - 1)]
        for _ in range(seq_len - 1):
            prev = seq[-1]
            if rng.random() < dominant_prob:
                seq.append(truth_b[prev])
            else:
                seq.append(rng.randint(half, vocab_size - 1))
        seqs_b.append(seq)

    return seqs_a, truth_a, seqs_b, truth_b


def generate_valid_invalid_pairs(
    vocab_size: int = 20,
    n_train: int = 100,
    n_test: int = 30,
    seq_len: int = 50,
    dominant_prob: float = 0.7,
    seed: int = 42,
) -> tuple[list[list[int]], list[list[int]], list[list[int]], dict[int, int]]:
    """Generate training sequences + matched valid/invalid test pairs.

    Valid test sequences: drawn from the same bigram distribution.
    Invalid test sequences: tokens from same vocab but random order (shuffled).

    Returns:
        (train_seqs, valid_test, invalid_test, ground_truth)
    """
    train_seqs, ground_truth = generate_bigram_sequences(
        vocab_size, n_train, seq_len, dominant_prob, seed
    )

    # Valid test: new sequences from same distribution
    valid_test, _ = generate_bigram_sequences(
        vocab_size, n_test, seq_len, dominant_prob, seed + 1000
    )

    # Invalid test: shuffle each sequence (breaks bigram structure)
    rng = random.Random(seed + 2000)
    invalid_test: list[list[int]] = []
    for _ in range(n_test):
        seq = [rng.randint(0, vocab_size - 1) for _ in range(seq_len)]
        invalid_test.append(seq)

    return train_seqs, valid_test, invalid_test, ground_truth


def generate_clustered_patterns(
    n_clusters: int = 5,
    patterns_per_cluster: int = 20,
    dim: int = 16,
    noise_scale: float = 0.01,
    seed: int = 42,
) -> tuple[list[torch.Tensor], list[int], list[torch.Tensor]]:
    """Generate patterns clustered around prototypes.

    Returns:
        (patterns, cluster_labels, prototypes)
    """
    gen = torch.Generator().manual_seed(seed)
    prototypes: list[torch.Tensor] = []
    patterns: list[torch.Tensor] = []
    labels: list[int] = []

    for c in range(n_clusters):
        proto = torch.randn(dim, generator=gen).abs() + 0.5
        prototypes.append(proto)
        for _ in range(patterns_per_cluster):
            noise = torch.randn(dim, generator=gen) * noise_scale
            patterns.append(proto + noise)
            labels.append(c)

    return patterns, labels, prototypes
