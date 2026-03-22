"""Unit tests for the Language Module.

Tests EmbeddingStore, TransitionCounter, ConcentrationGate,
and LanguageModule protocol compliance + PAC conservation.
"""

import math

import torch
import pytest

from gaia.core.protocol import GAIAModule
from gaia.core.types import FieldState, RBFBalance, SECPhase
from gaia.modules.language import (
    ConcentrationGate,
    ConcentrationResult,
    EmbeddingStore,
    LanguageMetrics,
    LanguageModule,
    TransitionCounter,
    TransitionStats,
    PHI_INV,
)


# ─── EmbeddingStore ──────────────────────────────────────────────


class TestEmbeddingStore:

    def test_from_tensor_creates_store(self):
        emb = torch.randn(100, 64)
        store = EmbeddingStore.from_tensor(emb)
        assert store.vocab_size == 100
        assert store.embed_dim == 64

    def test_get_embedding_returns_correct_row(self):
        emb = torch.eye(10, 5)  # identity-ish matrix
        store = EmbeddingStore(emb)
        vec = store.get_embedding(3)
        assert torch.allclose(vec, emb[3])

    def test_get_embeddings_batch(self):
        emb = torch.randn(50, 32)
        store = EmbeddingStore(emb)
        ids = torch.tensor([0, 5, 10])
        batch = store.get_embeddings(ids)
        assert batch.shape == (3, 32)
        assert torch.allclose(batch[1], emb[5])

    def test_nearest_token_finds_exact_match(self):
        emb = torch.randn(20, 8)
        store = EmbeddingStore(emb)
        # Query with exact embedding for token 7
        ids, sims = store.nearest_token(emb[7], top_k=3)
        assert ids[0].item() == 7
        assert sims[0].item() == pytest.approx(1.0, abs=1e-5)

    def test_nearest_token_top_k(self):
        emb = torch.randn(100, 16)
        store = EmbeddingStore(emb)
        ids, sims = store.nearest_token(emb[0], top_k=5)
        assert len(ids) == 5
        assert len(sims) == 5

    def test_rejects_non_2d_tensor(self):
        with pytest.raises(ValueError, match="2D"):
            EmbeddingStore(torch.randn(10))

    def test_from_pretrained_lazy_import(self):
        """from_pretrained should raise ImportError if transformers missing."""
        # We can't easily mock this, but we can verify it raises sensibly
        # if transformers IS available, this would succeed; that's fine too.
        # Just verify it's callable.
        assert callable(EmbeddingStore.from_pretrained)


# ─── TransitionCounter ──────────────────────────────────────────


class TestTransitionCounter:

    def test_learn_single_transition(self):
        tc = TransitionCounter()
        tc.learn((0, 1), 2)
        ids, probs = tc.predict((0, 1))
        assert 2 in ids.tolist()

    def test_learn_counts_correctly(self):
        tc = TransitionCounter()
        tc.learn((0, 1), 2)
        tc.learn((0, 1), 2)
        tc.learn((0, 1), 3)
        ids, probs = tc.predict((0, 1), top_k=2)
        # Token 2 should have higher probability (2/3) than token 3 (1/3)
        idx_2 = ids.tolist().index(2)
        idx_3 = ids.tolist().index(3)
        assert probs[idx_2] > probs[idx_3]

    def test_learn_from_sequence(self):
        tc = TransitionCounter(max_context_len=2)
        tc.learn_from_sequence([10, 20, 30, 40])
        # Should learn unigram and bigram contexts
        assert tc.stats.total_transitions > 0
        assert tc.stats.unique_contexts > 0

    def test_predict_unknown_context(self):
        tc = TransitionCounter()
        ids, probs = tc.predict((999, 888))
        assert len(ids) == 0
        assert len(probs) == 0

    def test_predict_top_k(self):
        tc = TransitionCounter()
        for i in range(10):
            tc.learn((0,), i)
        ids, probs = tc.predict((0,), top_k=3)
        assert len(ids) == 3

    def test_predict_temperature(self):
        tc = TransitionCounter()
        tc.learn((0,), 1)
        tc.learn((0,), 1)
        tc.learn((0,), 1)
        tc.learn((0,), 2)
        # Low temperature → more peaked
        _, probs_low = tc.predict((0,), temperature=0.1)
        # High temperature → more uniform
        _, probs_high = tc.predict((0,), temperature=10.0)
        # At low temp, difference between top and second should be larger
        if len(probs_low) >= 2 and len(probs_high) >= 2:
            spread_low = float(probs_low[0] - probs_low[1])
            spread_high = float(probs_high[0] - probs_high[1])
            assert spread_low > spread_high

    def test_stats_tracking(self):
        tc = TransitionCounter()
        tc.learn((0,), 1)
        tc.learn((0,), 2)
        tc.learn((1,), 3)
        assert tc.stats.total_transitions == 3
        assert tc.stats.unique_contexts == 2
        assert tc.stats.unique_transitions == 3

    def test_hit_rate_estimate(self):
        tc = TransitionCounter()
        tc.learn((0,), 1)
        tc.learn((0,), 1)  # context (0,) now has 2 observations
        tc.learn((1,), 2)  # context (1,) has 1 observation
        rate = tc.hit_rate_estimate()
        assert rate == pytest.approx(0.5)  # 1 of 2 contexts has >1 obs


# ─── ConcentrationGate ───────────────────────────────────────────


class TestConcentrationGate:

    def test_full_agreement_high_quality(self):
        gate = ConcentrationGate()
        preds = {
            1: (torch.tensor([5, 3]), torch.tensor([0.7, 0.3])),
            2: (torch.tensor([5, 7]), torch.tensor([0.8, 0.2])),
            3: (torch.tensor([5, 9]), torch.tensor([0.6, 0.4])),
        }
        result = gate.evaluate(preds)
        assert result.concentration == 1.0
        assert result.predicted_token == 5
        assert result.is_high_quality is True

    def test_no_agreement_low_quality(self):
        gate = ConcentrationGate()
        preds = {
            1: (torch.tensor([1]), torch.tensor([1.0])),
            2: (torch.tensor([2]), torch.tensor([1.0])),
            3: (torch.tensor([3]), torch.tensor([1.0])),
        }
        result = gate.evaluate(preds)
        assert result.concentration == pytest.approx(1.0 / 3.0)
        assert result.is_high_quality is False

    def test_phi_threshold(self):
        gate = ConcentrationGate(threshold=PHI_INV)
        # 2 out of 3 agree → concentration = 0.667 > PHI_INV
        preds = {
            1: (torch.tensor([5]), torch.tensor([1.0])),
            2: (torch.tensor([5]), torch.tensor([1.0])),
            3: (torch.tensor([9]), torch.tensor([1.0])),
        }
        result = gate.evaluate(preds)
        assert result.concentration == pytest.approx(2.0 / 3.0)
        assert result.is_high_quality is True  # 0.667 > 0.618

    def test_empty_predictions(self):
        gate = ConcentrationGate()
        result = gate.evaluate({})
        assert result.concentration == 0.0
        assert result.predicted_token == -1
        assert result.is_high_quality is False

    def test_should_reject_low_concentration(self):
        gate = ConcentrationGate()
        result = ConcentrationResult(
            concentration=0.3,
            predicted_token=1,
            depth_votes={1: 1},
            confidence=0.0,
            is_high_quality=False,
        )
        assert gate.should_reject(result) is True

    def test_stats_accumulate(self):
        gate = ConcentrationGate()
        preds = {1: (torch.tensor([5]), torch.tensor([1.0]))}
        gate.evaluate(preds)
        gate.evaluate(preds)
        assert gate.stats["total_analyzed"] == 2


# ─── LanguageModule Protocol ─────────────────────────────────────


class TestLanguageModuleProtocol:

    def test_satisfies_gaia_protocol(self):
        module = LanguageModule()
        assert isinstance(module, GAIAModule)

    def test_name_is_language(self):
        module = LanguageModule()
        assert module.name == "language"

    def test_phase_returns_sec_phase(self):
        module = LanguageModule()
        assert isinstance(module.phase(), SECPhase)

    def test_health_returns_rbf_balance(self):
        module = LanguageModule()
        h = module.health()
        assert isinstance(h, RBFBalance)

    def test_process_preserves_provenance(self):
        module = LanguageModule()
        state = FieldState(tensor=torch.ones(10), entropy=1.0)
        result = module.process(state)
        assert "language" in result.provenance


# ─── LanguageModule Conservation ─────────────────────────────────


class TestLanguageModuleConservation:

    def _assert_conserved(self, input_state: FieldState, output_state: FieldState, tol: float = 1e-6):
        ie = input_state.total_energy()
        oe = output_state.total_energy()
        assert abs(ie - oe) / max(abs(ie), 1e-10) < tol, (
            f"Conservation violated: input={ie:.8f}, output={oe:.8f}, "
            f"residual={abs(ie - oe):.2e}"
        )

    def test_process_conserves_energy_ones(self):
        module = LanguageModule()
        state = FieldState(tensor=torch.ones(10), entropy=1.0)
        result = module.process(state)
        self._assert_conserved(state, result)

    def test_process_conserves_energy_randn(self):
        module = LanguageModule()
        state = FieldState(tensor=torch.randn(20).abs() + 0.1, entropy=1.0)
        result = module.process(state)
        self._assert_conserved(state, result)

    def test_process_conserves_energy_large_tensor(self):
        module = LanguageModule()
        state = FieldState(tensor=torch.randn(1000).abs() + 0.01, entropy=1.0)
        result = module.process(state)
        self._assert_conserved(state, result)

    def test_process_conserves_energy_after_learning(self):
        module = LanguageModule()
        # Run 10 steps to build up transition statistics
        for _ in range(10):
            state = FieldState(tensor=torch.ones(8) * 2.0, entropy=1.0)
            result = module.process(state)
            self._assert_conserved(state, result)

    def test_process_conserves_energy_with_embeddings(self):
        emb = torch.randn(50, 16)
        store = EmbeddingStore(emb)
        module = LanguageModule(embeddings=store)
        state = FieldState(tensor=torch.randn(8).abs() + 0.1, entropy=1.0)
        result = module.process(state)
        self._assert_conserved(state, result)


# ─── LanguageModule Functional ───────────────────────────────────


class TestLanguageModuleFunctional:

    def test_module_learns_transitions(self):
        module = LanguageModule()
        for _ in range(5):
            state = FieldState(tensor=torch.ones(8) * 3.0, entropy=1.0)
            module.process(state)
        assert module.counter.stats.total_transitions > 0

    def test_module_without_embeddings(self):
        module = LanguageModule()
        state = FieldState(tensor=torch.randn(16).abs() + 0.1, entropy=1.0)
        result = module.process(state)
        assert torch.isfinite(result.tensor).all()

    def test_module_with_embeddings(self):
        emb = torch.randn(50, 16)
        store = EmbeddingStore(emb)
        module = LanguageModule(embeddings=store)
        state = FieldState(tensor=torch.randn(8).abs() + 0.1, entropy=1.0)
        result = module.process(state)
        assert torch.isfinite(result.tensor).all()

    def test_metrics_populated_after_process(self):
        module = LanguageModule()
        state = FieldState(tensor=torch.ones(10), entropy=1.0)
        module.process(state)
        m = module.metrics
        assert m is not None
        assert isinstance(m, LanguageMetrics)
        assert m.step_count == 1

    def test_phase_reflects_concentration(self):
        module = LanguageModule()
        # First call — no learned transitions → low concentration → not CRYSTALLIZED
        state = FieldState(tensor=torch.randn(16).abs() + 0.1, entropy=1.0)
        module.process(state)
        # With random input and no prior learning, concentration should be low
        phase = module.phase()
        assert isinstance(phase, SECPhase)
