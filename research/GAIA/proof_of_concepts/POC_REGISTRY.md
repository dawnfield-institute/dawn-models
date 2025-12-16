# GAIA Proof of Concepts Registry

> **Index of all POC experiments for Phase 4 Field-Native Transformers**

---

## Status Legend

| Status | Meaning |
|--------|---------|
| 📋 Planned | Defined but not started |
| 🔄 In Progress | Currently being worked on |
| ✅ Complete | Finished with conclusions |
| ❌ Blocked | Waiting on dependencies |
| 🔁 Iterating | Multiple rounds of experiments |

---

## POC Index

### Symbol Grounding (Priority 1)

| POC | Name | Status | Challenge | Key Question |
|-----|------|--------|-----------|--------------|
| 001 | Pattern Encoding | ✅ Complete | 2.1 | Can we encode meaning in field perturbations? |
| 002 | Resonance Training | ✅ Complete | 2.1 | Can field resonance learn semantic similarity? |
| 003 | Field-Native Attention | ✅ Complete | 2.1 | Can attention emerge from field physics? |
| 004 | Scale & Dimension | ✅ Complete | 2.5 | Do constants work at 10K patterns in 3D? |
| 005 | Language Generation | ✅ Complete | 2.1 | Can field evolution generate coherent language? |

**POC-001 Finding:** Syntactic encoding works (21/23 tests). Semantic needs training.
**POC-002 Finding:** Resonance training achieves 0.83 semantic separation! 24/24 tests passed.
**POC-003 Finding:** Field-native attention works! 25/25 tests. Semantic amplification to 0.999.
**POC-004 Finding:** 3D scaling works! 18/18 tests. v6 encoder achieves 0.977 correlation with original embeddings.
**POC-005 Finding:** Field generation works! 24/24 tests. Grammar emerges from dynamics alone.

Physics validated:
- φ × ξ = 1.710 → crystallization trigger (2D and 3D)
- (φ × ξ)^(3/2) = 2.237 → 3D critical density
- λ* = 0.9816 → optimal decay (verified at scale)
- Fibonacci lr = 1/F_n → stable convergence  
- 1/p² = 1/l² → prime harmonic / spherical harmonic hierarchy
- Attention = resonance → no QKV projections needed
- 29,799 tokens/sec on GPU (2D), 790 patterns/sec (3D)
- Conservation violation < 1e-7 at 64³
- ξ-modulation preserves embedding geometry (0.977 correlation)
- Klein-Gordon evolution enables next-token prediction
- Grammatical categories cluster by field similarity

### Training Efficiency (Priority 2)

| POC | Name | Status | Challenge | Key Question |
|-----|------|--------|-----------|--------------|
| 004 | Crystallization Rate | 📋 Planned | 2.2 | How fast do patterns stabilize? |
| 005 | Pre-conditioning | 📋 Planned | 2.2 | Does field prep accelerate learning? |
| 006 | Field Size Scaling | 📋 Planned | 2.2 | Bigger field = faster learning? |

### Memory Reliability (Priority 3)

| POC | Name | Status | Challenge | Key Question |
|-----|------|--------|-----------|--------------|
| 007 | Pattern Persistence | 📋 Planned | 2.3 | How long do patterns survive? |
| 008 | Memory Interference | 📋 Planned | 2.3 | Do new patterns destroy old ones? |
| 009 | Recall Accuracy | 📋 Planned | 2.3 | Can we reliably retrieve stored patterns? |

### Evaluation Metrics (Priority 4)

| POC | Name | Status | Challenge | Key Question |
|-----|------|--------|-----------|--------------|
| 010 | Metric Correlation | 📋 Planned | 2.4 | Which field metrics predict performance? |
| 011 | Early Indicators | 📋 Planned | 2.4 | Can we detect learning before completion? |
| 012 | Stopping Criteria | 📋 Planned | 2.4 | When is training "done"? |

### Scaling Behavior (Priority 5)

| POC | Name | Status | Challenge | Key Question |
|-----|------|--------|-----------|--------------|
| 013 | Data Size Curves | 📋 Planned | 2.5 | Learning vs data relationship? |
| 014 | Field Capacity | 📋 Planned | 2.5 | Maximum patterns per field size? |
| 015 | Long Training | 📋 Planned | 2.5 | Stability over extended training? |

### Bootstrap (Priority 6)

| POC | Name | Status | Challenge | Key Question |
|-----|------|--------|-----------|--------------|
| 016 | Minimal Seed | 📋 Planned | 2.6 | What's minimum viable bootstrap? |
| 017 | Time to First Symbol | 📋 Planned | 2.6 | How long until first stable pattern? |
| 018 | Failure Modes | 📋 Planned | 2.6 | What causes learning to fail? |

---

## Starting Point: POC-001

We begin with the most fundamental question:

**Can we encode meaningful patterns into GAIA's field at all?**

If we can't bridge text → field → meaning, nothing else matters.

See: [poc_001_pattern_encoding/](./poc_001_pattern_encoding/)

---

## Go/No-Go Criteria

After POC-001 through POC-003:

**Continue if:**
- Distinct patterns are visibly different in field
- Similar patterns show measurable similarity
- Patterns survive >100 evolution steps

**Reconsider if:**
- All patterns look the same
- Patterns decay immediately
- No structure emerges

---

## References

- [Phase 4 Challenges](./../.spec/phase4-challenges.md)
- [Phase 4 Spec](./../.spec/phase4-transformers.spec.md)
- [GAIA Core Spec](./../.spec/gaia.spec.md)
