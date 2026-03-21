# POC-003: Field-Native Attention

> **Can attention mechanisms emerge from field dynamics?**

---

## Status

🔄 **In Progress** - Starting experiments

---

## Research Question

Traditional transformers compute attention as:
```
Attention(Q,K,V) = softmax(QK^T / √d) × V
```

This is a mathematical operation bolted onto the architecture.

**Can we make attention GROW from field physics instead?**

---

## Hypothesis

From Prime Harmonic Manifold (PHM) experiments:

1. **Eigenvalue Decay**: Attention head weights follow 1/π² ≈ 0.101 decay
2. **Harmonic Structure**: Head importance follows prime harmonic series
3. **Field Resonance**: Attention is really "what patterns resonate with what"

The key insight: **Attention IS resonance.**

When pattern A "attends to" pattern B, they're resonating in the field.
The attention weight is the resonance strength.

---

## Physics Foundation

| Principle | Constant | Application |
|-----------|----------|-------------|
| Eigenvalue decay | 1/π² = 0.101 | Head importance scaling |
| Prime harmonics | 1/p² series | Multi-head structure |
| Resonance coupling | 4/5 = 0.8 | Max attention weight |
| Crystallization | φ × ξ = 1.710 | When patterns "lock on" |

---

## Key Insight: Attention as Field Coupling

In standard attention:
- Q, K, V are learned projections
- Attention weights computed via dot product
- Information flows through weighted sum

In field-native attention:
- Patterns exist as field perturbations
- Attention = resonance between perturbations
- Weights emerge from field coupling strength
- No learned projections needed initially

---

## Experimental Design

### Experiment 01: Resonance-Based Attention
- Compute "attention" as resonance between field states
- Compare to standard dot-product attention
- Validate: similar patterns should attend to each other

### Experiment 02: Harmonic Head Structure
- Create multi-head attention using prime harmonic decay
- Head n has weight proportional to 1/p_n²
- Test: does this match learned head importance?

### Experiment 03: Field-Native QKV
- Derive Q, K, V from field evolution operators
- Q = "what am I looking for" (query field)
- K = "what do I contain" (key field)  
- V = "what can I give" (value field)

### Experiment 04: Integration Test
- Full field-native attention layer
- Compare to PyTorch MultiheadAttention
- Validate PAC conservation through attention

---

## Success Criteria

### Must Have
- [ ] Resonance attention produces sensible weights
- [ ] Similar patterns attend to each other more
- [ ] PAC conservation through attention operation
- [ ] GPU execution end-to-end

### Should Have
- [ ] 1/π² decay matches empirical head importance
- [ ] Prime harmonic structure improves over uniform
- [ ] Competitive with standard attention quality

### Nice to Have
- [ ] Faster than standard attention
- [ ] Emergent head specialization
- [ ] Interpretable attention patterns

---

## Connection to POC-002

POC-002 proved: resonance training creates semantic clusters.
POC-003 tests: can resonance ALSO compute attention?

If yes, we have:
- Encoding: field perturbations (POC-001)
- Learning: resonance training (POC-002)  
- Attention: field coupling (POC-003)

The core transformer emerges from physics.
