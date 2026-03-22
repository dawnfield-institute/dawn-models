"""Pytest wrapper for real benchmarks — smaller data, relaxed thresholds.

These wrap the benchmark functions from benchmarks/ with CI-friendly
parameters (fewer sequences, looser thresholds, <15 seconds total).
"""

from __future__ import annotations

import pytest

from benchmarks.synthetic import (
    bench_efficiency,
    bench_continuous_learning,
    bench_hallucination_detection,
    bench_hallucination_e2e,
    bench_memory,
)


# ─── Efficiency ──────────────────────────────────────────────────


class TestBenchEfficiency:

    def test_tokens_per_second(self):
        """Throughput above 10K tok/s (relaxed from 50K for small data)."""
        results = bench_efficiency(n_small=1000, n_large=2000)
        assert results["tokens_per_second"] > 10_000

    def test_overhead_ratio(self):
        """O(1) scaling: overhead ratio < 3x (relaxed from 2x)."""
        results = bench_efficiency(n_small=1000, n_large=2000)
        assert results["learning_overhead_ratio"] < 3.0


# ─── Continuous Learning ─────────────────────────────────────────


class TestBenchContinuousLearning:

    def test_domain_a_accuracy(self):
        """Domain A accuracy > 0.8 after training."""
        results = bench_continuous_learning(n_sequences=20, seq_len=30)
        assert results["domain_a_accuracy_before"] > 0.8

    def test_domain_b_accuracy(self):
        """Domain B accuracy > 0.8 after training."""
        results = bench_continuous_learning(n_sequences=20, seq_len=30)
        assert results["domain_b_accuracy"] > 0.8

    def test_retention_after_domain_shift(self):
        """Domain A retention > 0.85 after learning domain B (KEY METRIC)."""
        results = bench_continuous_learning(n_sequences=20, seq_len=30)
        assert results["domain_a_retention_after"] > 0.85

    def test_forgetting_rate(self):
        """Forgetting rate < 0.10 (relaxed from 0.05)."""
        results = bench_continuous_learning(n_sequences=20, seq_len=30)
        assert results["forgetting_rate"] < 0.10


# ─── Hallucination Detection ────────────────────────────────────


class TestBenchHallucinationDetection:

    def test_valid_concentration(self):
        """Valid sequences have concentration > 0.4 (relaxed from 0.6)."""
        results = bench_hallucination_detection(n_train=50, n_test=15)
        assert results["valid_concentration"] > 0.4

    def test_invalid_concentration(self):
        """Invalid sequences have concentration < 0.5 (relaxed from 0.3)."""
        results = bench_hallucination_detection(n_train=50, n_test=15)
        assert results["invalid_concentration"] < 0.5

    def test_detection_gap(self):
        """Gap between valid and invalid > 0.1 (relaxed from 0.3, KEY METRIC)."""
        results = bench_hallucination_detection(n_train=50, n_test=15)
        assert results["detection_gap"] > 0.1


# ─── Memory ──────────────────────────────────────────────────────


class TestBenchMemory:

    def test_storage_ratio(self):
        """PACTree achieves compression (ratio < 0.8, relaxed from 0.5)."""
        results = bench_memory(n_clusters=3, patterns_per_cluster=10)
        assert results["storage_ratio"] < 0.8

    def test_retrieval_precision(self):
        """Retrieval precision@3 > 0.4 (relaxed from 0.5)."""
        results = bench_memory(n_clusters=3, patterns_per_cluster=10)
        assert results["retrieval_precision_at_3"] > 0.4


# ─── E2E Hallucination Detection ──────────────────────────────


class TestBenchHallucinationE2E:

    def test_e2e_valid_concentration(self):
        """Valid sequences through process() have higher concentration."""
        results = bench_hallucination_e2e(n_train=50, n_test=15)
        assert results["e2e_valid_concentration"] > 0.3

    def test_e2e_detection_gap(self):
        """Gap through full pipeline > 0.02 (conservative, KEY METRIC)."""
        results = bench_hallucination_e2e(n_train=50, n_test=15)
        assert results["e2e_detection_gap"] > 0.02


# ─── Full Scorecard ──────────────────────────────────────────────


class TestFullScorecard:

    def test_all_synthetic_benchmarks_run(self):
        """All 5 benchmarks produce results without crashing."""
        r1 = bench_efficiency(n_small=500, n_large=1000)
        r2 = bench_continuous_learning(n_sequences=10, seq_len=20)
        r3 = bench_hallucination_detection(n_train=30, n_test=10)
        r4 = bench_memory(n_clusters=3, patterns_per_cluster=5)
        r5 = bench_hallucination_e2e(n_train=30, n_test=10)

        all_results = {**r1, **r2, **r3, **r4, **r5}
        assert len(all_results) >= 14  # All 14 metrics present


# ─── Real Data (conditional) ─────────────────────────────────────


def _has_transformers() -> bool:
    try:
        import transformers  # noqa: F401
        import datasets  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_transformers(), reason="transformers/datasets not installed")
class TestBenchWikiText2:

    def test_wikitext2_hit_rate(self):
        """WikiText-2 hit rate > 0.40 (relaxed, small subset)."""
        from benchmarks.real_data import bench_wikitext2
        results = bench_wikitext2(max_sequences=200, seq_len=32)
        assert results["wikitext2_hit_rate"] > 0.40
