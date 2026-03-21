# POC-007: Scale Validation

## Hypothesis
GAIA's field-native architecture maintains performance advantages at production scale (100M+ tokens) while preserving the physics-based learning principles.

## Key Questions
1. Does perplexity remain competitive at WikiText-103 scale?
2. Can we maintain sub-hour training times for large corpora?
3. Do the physics constants (φ×ξ, λ*, 1/p²) remain optimal at scale?
4. How does memory consumption scale with vocabulary size?

## Success Criteria
- [ ] Train on WikiText-103 (100M tokens) in < 1 hour
- [ ] Achieve perplexity < 20 on WikiText-103
- [ ] Memory usage < 32GB for 50K vocabulary
- [ ] Maintain 100% retrieval accuracy
- [ ] Support checkpoint/resume for long training

## Experiments

### exp_01: WikiText-103 Full Training
- Train on complete WikiText-103
- Compare with published baselines
- Profile memory and compute

### exp_02: Vocabulary Scaling
- Test with 10K, 25K, 50K vocabularies
- Measure memory and speed impact
- Verify retrieval accuracy

## Baselines (WikiText-103 Perplexity)
| Model | Perplexity |
|-------|------------|
| GPT-2 Large | 19.93 |
| Transformer-XL Base | 23.6 |
| GPT-2 Medium | 22.76 |
| XLNet Base | 23.5 |
| BERT Large | 25.8 |

## Current Status
WikiText-2 results:
- GAIA: **5.91** perplexity
- GPT-2 Small: 29.41
- Improvement: **5x better**
