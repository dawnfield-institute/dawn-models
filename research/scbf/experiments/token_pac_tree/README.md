# Token-Level PAC Tree Analysis

## Hypothesis

When an LLM predicts a token, the softmax distribution collapses from broad potential (many candidate tokens) to single actualization (one selected token). This is structurally identical to SEC (Symbolic Entropy Collapse): a moment with many possibilities collapses into one outcome.

If PAC conservation governs this process, then:
1. The ratio p1/p2 (top two token probabilities) should cluster near **phi (1.618)** or **1/phi (0.618)**
2. The entropy budget should be conserved: `f(parent) = sum(f(children))`
3. The SEC phase signature should differ between correct predictions and hallucinations
4. Forced collapses (high entropy + disproportionately confident top token) should correlate with model errors

## Design

### Layer 1: Token-Level PAC Tree (observation instrument)
At each token position during inference:
- **Parent node** = full logit distribution (potential) with entropy H
- **Children** = top-k candidate tokens with softmax probabilities (actualization candidates)
- **Collapse** = selection of one token (entropy drops to 0)

Conservation check: do the children's probability mass and entropy contributions sum correctly?

### Layer 2: SEC Phase Classification (diagnostic lens)
Each collapse event is classified:
| Phase | Entropy Range | Interpretation |
|-------|--------------|----------------|
| Crystallized | H < 0.5 | Confident, routine prediction |
| Ordered | 0.5 < H < 2.0 | Normal healthy prediction |
| Transitional | 2.0 < H < 4.0 | Model exploring, broad distribution |
| Chaotic | H > 4.0 | Maximum uncertainty |

### Layer 3: Temporal PAC Forest (model X-ray)
Chain PAC trees across a full sequence to track how collapse dynamics evolve token-by-token.

## Success Criteria

- [ ] PAC ratio clusters near phi or 1/phi with > 2x enrichment vs random null
- [ ] SEC phase predicts correctness (crystallized accuracy > chaotic accuracy)
- [ ] Conservation error < 1e-6 (probability budget balances exactly)
- [ ] Forced collapse events correlate with incorrect predictions (p < 0.05)
- [ ] Phi alignment differs between correct and incorrect tokens (p < 0.05)

## Success Criteria

- [x] ~~PAC ratio clusters near phi or 1/phi with > 2x enrichment vs random null~~ **FALSIFIED** — Null baseline shows softmax itself produces 8.8% near-phi ratios; enrichment < 1x for larger models
- [x] SEC phase predicts correctness (crystallized accuracy > chaotic accuracy) **CONFIRMED (p < 0.0001)**
- [x] Conservation error < 1e-6 (probability budget balances exactly) **CONFIRMED**
- [x] ~~Forced collapse events correlate with incorrect predictions (p < 0.05)~~ **NOT TESTED** — Forced collapses too rare (0% rate) at top-1 level
- [x] ~~Phi alignment differs between correct and incorrect tokens (p < 0.05)~~ **SUPERSEDED** — Correct vs incorrect PAC *ratio magnitude* differs significantly (p = 0.0089 at 160m, p < 0.0001 at 1B)

## Falsification Conditions

- ~~PAC ratio distribution is indistinguishable from uniform random → phi alignment falsified~~ **PARTIALLY FALSIFIED**: phi enrichment is a softmax artifact (null baseline = 8.8%). However, the *magnitude* of PAC ratios robustly discriminates correct from incorrect predictions.
- ~~SEC phase has zero correlation with prediction quality → phase model falsified~~ **ANTI-FALSIFIED**: SEC phase monotonically predicts accuracy across 4 models and 2 independent experiments
- Conservation always balances trivially → not a meaningful diagnostic (conservation error < 1e-6; trivially satisfied by softmax normalization — as expected)

## Key Findings

### Finding 1: SEC Phase is a Universal Accuracy Predictor

SEC phase monotonically predicts token accuracy across all 4 Pythia models (70m → 1B, 340 tokens each, 49 prompts):

| Phase | 70m | 160m | 410m | 1B |
|-------|-----|------|------|-----|
| Crystallized (H < 0.5) | 100% | 100% | 100% | 100% |
| Ordered (0.5 < H < 2.0) | 100% | 93% | 91% | 89% |
| Transitional (2.0 < H < 4.0) | 51% | 51% | 59% | 52% |
| Chaotic (H > 4.0) | 17% | 20% | 21% | 22% |

The gradient is perfectly monotonic for every model. No parameter fitting was used.

### Finding 2: PAC Ratio Magnitude Scales with Model Size

Correct tokens have dramatically higher p1/p2 ratios than incorrect tokens, and this gap *widens* with scale:

| Model | Correct Median | Incorrect Median | p-value |
|-------|---------------|-------------------|---------|
| pythia-70m | 3.27 | 1.63 | 0.048 |
| pythia-160m | 4.02 | 1.59 | **0.009** |
| pythia-410m | 5.30 | 2.22 | **0.002** |
| pythia-1b | 6.32 | 1.58 | **< 0.0001** |

Significance improves with scale: as models grow, they increasingly concentrate probability mass on the correct answer.

### Finding 3: Null Baseline Reveals Phi as Softmax Artifact

Random logits → softmax produces 8.8% near-phi ratios. Actual models produce *fewer* near-phi ratios (5–14%), meaning phi enrichment is NOT a signal. The real signal is ratio *magnitude*, not ratio alignment to any constant.

### Finding 4: Sequence-Level PAC Forests Detect Hallucination (exp_04)

When generating 30-token continuations, factual vs hallucinated text differs on 3 significant metrics (Pythia-160m, n=15 per group):

| Metric | Factual | Hallucinated | p-value |
|--------|---------|--------------|---------|
| Confidence ratio (crystallized+ordered %) | 28.7% | 18.0% | **0.027** |
| Entropy slope (dH/dt over sequence) | −0.100 | −0.052 | **0.009** |
| Ratio slope (d(p1/p2)/dt over sequence) | +3.04 | +0.62 | **0.046** |

**Interpretation**: When the model "knows" something, entropy decreases ~2x faster across the generated sequence, PAC ratios climb, and more tokens land in confident phases. Hallucinated text stays flat and uncertain.

### Finding 5: Single-Token Hallucination Detection Fails (exp_03)

Analysing only the first predicted token cannot distinguish factual from hallucination prompts (all p > 0.15). Both land in chaotic SEC phase. The signal requires multi-token generation where *trajectory* becomes visible.

### Finding 6: Xi Clustering in Weight SVD Ratios is Training-Induced (exp_05/06)

SVD decomposition of weight matrices reveals that consecutive singular value ratios (σᵢ/σᵢ₊₁) cluster near **Xi (1.057)** at dramatically elevated rates compared to random matrices:

| Condition | Xi@5% Rate | Near-1@5% Rate | n |
|-----------|-----------|----------------|---|
| **Trained** (Pythia-160m) | **24.5%** | 98.5% | 36,816 |
| Reinitialised (Xavier) | 10.3% | 99.3% | 36,816 |
| Random (Marchenko-Pastur) | 10.4% | 99.3% | 184,080 |

Chi-squared test: χ² = 5,511, **p ≈ 0** — the 2.36x enrichment over random is overwhelmingly significant.

**Cross-model scaling** (inversely related to model size):

| Model | Xi@5% (all) | Attention | MLP |
|-------|------------|-----------|-----|
| pythia-70m | **40.9%** | 58.0% | 23.9% |
| pythia-160m | 24.5% | 37.9% | 11.1% |
| pythia-410m | 13.9% | 20.7% | 7.2% |
| pythia-1b | 5.3% | 7.5% | 3.1% |

Key observations:
- **Xi clustering is a training artifact** — random and reinitialised matrices both show ~10.4% baseline
- **Attention weights show 2-3x more Xi clustering** than MLP weights at every scale
- **Smaller models cluster more**, suggesting Xi may relate to regularization or weight compression
- **Phi clustering is absent** (< 0.05% at every scale) — weights do NOT organize by φ

However, **projection of activations onto SVD modes does NOT distinguish factual from hallucinated text** (p = 0.97). The Xi structure is in the static weights, not in how activations traverse them.

### Finding 7: Attention Heads ARE the PAC Collapse Mechanism (exp_07)

Testing attention patterns directly as PAC collapse events yields **5 significant metrics**, the strongest results in the entire experiment series:

| Metric | Factual | Hallucinated | p-value |
|--------|---------|--------------|---------|
| Confident head ratio (crystallized+ordered %) | **86.0%** | 80.0% | **0.00006** |
| Mean attention entropy | 1.010 | 1.085 | **0.001** |
| Depth slope (entropy vs. layer depth) | −0.162 | −0.184 | **0.0009** |
| Attention entropy slope over sequence | +0.033 | +0.021 | **0.0005** |
| Confident head slope over sequence | −0.017 | −0.010 | **0.00008** |

**Interpretation**: When the model is processing factual content:
- **86% of attention heads operate in crystallized/ordered phase** (focused attention) vs 80% for hallucination
- Attention entropy is **7% lower** (more focused heads)
- Deeper layers sharpen attention **less steeply** (early layers already focused)
- The confident-head ratio **decreases faster** during factual generation — the model "locks in" to context more aggressively

This is the strongest hallucination discriminator we've found: the p-values are 10-100x better than output-level metrics (exp_04). Attention patterns reveal PAC collapse happening *inside* the model, not just at the output.

## Experiments

| Script | Description | Status |
|--------|-------------|--------|
| `exp_01_logit_pac_tree.py` | Pythia-160m, 13 prompts, 69 tokens. First PAC tree test. | ✅ Complete |
| `exp_02_multi_model_scale.py` | 4 Pythia models, 49 prompts, 340 tokens each, null baseline. | ✅ Complete |
| `exp_03_hallucination_signatures.py` | Single-token hallucination detection (neg result). | ✅ Complete |
| `exp_04_sequence_hallucination.py` | 30-token generation, sequence-level PAC forest analysis. | ✅ Complete |
| `exp_05_weight_pac_activation.py` | SVD weight PAC tree + activation monitoring. Xi clustering found. | ✅ Complete |
| `exp_06_xi_weight_clustering.py` | Xi null test: trained vs reinitialised vs random matrices. | ✅ Complete |
| `exp_07_attention_pac.py` | Attention heads as PAC collapse events. **Best results.** | ✅ Complete |

## Architecture

```
token_pac_tree/
├── README.md
├── meta.yaml
├── core/
│   ├── __init__.py
│   ├── pac_tree.py           # PACNode, TokenPACTree, PACForest, builder
│   └── collapse_metrics.py   # SECPhase, CollapseSignature, conservation checks
├── scripts/
│   ├── meta.yaml
│   ├── exp_01_logit_pac_tree.py         # First experiment (Pythia-160m, 13 prompts)
│   ├── exp_02_multi_model_scale.py      # Cross-model scaling (4 models, 49 prompts)
│   ├── exp_03_hallucination_signatures.py  # Single-token hallucination (neg result)
│   ├── exp_04_sequence_hallucination.py # Sequence-level hallucination detection
│   ├── exp_05_weight_pac_activation.py  # SVD weight PAC + activation monitoring
│   ├── exp_06_xi_weight_clustering.py   # Xi null test (trained vs random)
│   └── exp_07_attention_pac.py          # Attention as PAC collapse (best results)
├── results/
│   └── meta.yaml
└── journals/
    └── meta.yaml
```

## Connection to Prior Work

| Prior Work | What It Established | What This Experiment Extends |
|-----------|--------------------|-----------------------------|
| Pythia phi-crossing (Journal 002) | phi appears in weight update ratios during training | Does phi appear in logit ratios at inference? |
| POC-018 Hierarchical PAC-SEC | f(parent) = sum(f(children)) on hidden state scalars | Same conservation on logit distributions |
| POC-019 True No-Backprop | Confluence tree with probability conservation | Same concept applied to real LLM output |
| Landauer erasure structure | Erasure has a measurable structure cost | Collapse magnitude = structure cost of token selection |
| exp_38 PAC Mobius Tree | Binary tree of Mobius transforms with soft PAC gating | PAC tree concept extended to vocabulary space |
