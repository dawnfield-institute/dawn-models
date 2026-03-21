# POC-022: Scale Stress Test

## Status: 📋 Planned

**Date Created:** 2026-01-01  
**Challenge:** Scale validation at production vocabulary sizes

## Hypothesis

If MED bounds emergence complexity (depth ≤ 2, nodes ≤ 3), then PAC learning should maintain:
- Comparable hit rates at 100K+ vocabulary
- Sub-linear memory growth via tiered caching
- Convergent learning curves (not divergent)

## Motivation

Current validation caps at ~25K patterns (POC-007). The theory claims:

$$\text{Emergence complexity is bounded regardless of substrate size}$$

This needs direct falsification testing at production scale.

## Design

### Dataset
- **WikiText-103**: 103M tokens, ~267K unique tokens
- Compare to WikiText-2 baseline from POC-007

### Experiments

| Exp | Name | Goal |
|-----|------|------|
| 01 | Vocabulary Scaling | Test 50K → 100K → 200K vocab sizes |
| 02 | Hit Rate Curve | Measure accuracy vs vocabulary size |
| 03 | Memory Profile | GPU cache vs PAC cold storage efficiency |
| 04 | Learning Convergence | Epochs to convergence vs scale |
| 05 | Transition Density | Does transition matrix become sparse? |

### Metrics

```python
metrics = {
    "hit_rate": float,           # % correct predictions
    "memory_mb": float,          # Total memory usage
    "gpu_cache_hit": float,      # % served from hot cache
    "transitions_learned": int,  # Unique transitions in matrix
    "epochs_to_converge": int,   # Epochs until hit rate stabilizes
    "tokens_per_second": float,  # Throughput
}
```

## Success Criteria

- [ ] Hit rate ≥ 60% at 100K vocabulary (WikiText-103)
- [ ] Memory ≤ 2GB for full vocabulary (with tiered caching)
- [ ] Convergence within 10 epochs
- [ ] No exponential blowup in transition matrix

## Falsification Conditions

This POC is designed to **falsify** the scaling claims:

| Condition | Implication |
|-----------|-------------|
| Hit rate < 50% at 100K | PAC learning doesn't scale |
| Memory > 4GB linear | Tiered caching insufficient |
| No convergence at 20 epochs | Learning dynamics break at scale |
| Transition matrix explodes | MED bound violated |

## Theoretical Connection

From [foundational/arithmetic/README.md](../../../../../../dawn-field-theory/foundational/arithmetic/README.md):

> MED: All complex flows converge to symbolic patterns with depth ≤ 2 and nodes ≤ 3.

If true, vocabulary size shouldn't affect the fundamental learning dynamics—just the number of patterns, not their structure.

## Files

```
scripts/
├── exp_01_vocab_scaling.py
├── exp_02_hit_rate_curve.py
├── exp_03_memory_profile.py
├── exp_04_learning_convergence.py
└── exp_05_transition_density.py

results/
└── *.json

journals/
└── YYYY-MM-DD_slug.md
```

## Dependencies

- POC-007: Tiered memory cache architecture
- POC-021: Unified PAC learning system
- WikiText-103 dataset

## Next Steps

1. Set up WikiText-103 data loading
2. Baseline with current system at 25K vocab
3. Incrementally scale to 100K, 200K
4. Document failure modes if they occur
