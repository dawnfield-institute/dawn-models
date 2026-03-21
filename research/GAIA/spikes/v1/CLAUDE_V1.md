# GAIA — Claude Code Context

## Identity
GAIA (Generative AI via Information Architecture) is a language model built entirely from PAC/SEC principles — no backpropagation, no gradient descent. Learning is pure counting (O(1) per token). Information is conserved, not created. Every architectural decision is backed by a validated proof-of-concept experiment (25 PoCs, all passing).

## Current Version
`src/gaia_prime/` — v2.0.0 production. `src/legacy/` is archived reference (v1-v4), do not modify.

## Architecture

```
                    GAIA Prime v2.0.0
 ──────────────────────────────────────────────
  L0  GraftedEmbeddings          embeddings.py
      Frozen from GPT-2/Pythia   (PoC-016,017,020)
                 |
  L1  PACTree                    pac_tree.py
      Delta-only, f(p)=Ef(c)    (PoC-007,020: 12.5x memory)
                 |
  L2  TransitionMatrix           transitions.py
      Sparse N-gram counting     (PoC-021,022: 65% hit rate)
                 |
  L3  ConcentrationMonitor       concentration.py
      Reject-resample at L~0.5   (PoC-023: +3.6% quality)
                 |
  L4  PhysicsMesh                physics_mesh.py
      Entropy + conservation     auto_collapse.py
      + collapse + resonance     bifractal_resonance.py
                 |
  L5  PACGenerator               generator.py
      + ContinuousLearner        continuous_learning.py
      + MultiModelFusion         multi_model_fusion.py
      + PhysicsGenerator         physics_generator.py
 ──────────────────────────────────────────────
  Constants: XI=1.0571, PHI=1.618, LAMBDA_STAR=0.618
  All derived from topology, not fitted.
```

## Module Inventory

### Core Pipeline (L0-L3, L5) — PoC-validated, stable
| Module | Purpose | PoC | Fracton? |
|--------|---------|-----|----------|
| `embeddings.py` | Frozen embeddings from pretrained LLMs | 016,017,020 | No |
| `pac_tree.py` | Delta-only hierarchical storage with conservation | 007,020 | Future |
| `transitions.py` | GPU-accelerated sparse N-gram counting | 021,022 | No |
| `concentration.py` | Quality gating with phi-derived thresholds | 023 | No |
| `generator.py` | Greedy/sample/beam decoding | 023 | No |
| `model.py` | GAIA_Prime orchestrator class | All | No |

### Physics Layer (L4) — Manual implementations, fracton migration planned
| Module | Purpose | Fracton Target |
|--------|---------|---------------|
| `physics_mesh.py` | EntropyMonitor, ConservationEnforcer, CollapseEngine, ResonanceField | `fracton.field`, `fracton.core`, `fracton.physics` |
| `auto_collapse.py` | 5 collapse strategies (clustering, crystallization, pruning, compression, hierarchical) | `fracton.field`, `fracton.core.EntropyDispatcher` |
| `bifractal_resonance.py` | Depth-based bifractal patterns, personality emergence | `fracton.core.BifractalTrace` |

### Extended Components (L1+, L5) — Working, some fracton overlap
| Module | Purpose | Fracton? |
|--------|---------|----------|
| `pac_mesh.py` | Multi-model PAC mesh space | Future (`fracton.core.PACSystem`) |
| `mesh.py` | Multi-model source management | Partial |
| `physics_generator.py` | Physics-governed text generation | Partial (physics state) |
| `continuous_learning.py` | Hebbian O(1) learning during inference | No |
| `multi_model_fusion.py` | Agreement crystallization across LLMs | No |
| `knowledge_mesh.py` | Extracted knowledge representation | No |
| `validated_constants.py` | XI, PHI, LAMBDA_STAR (derived, not fitted) | Yes (`fracton.physics.constants`) |

## PoC Validation Chain

Every gaia_prime component traces back to validated experiments:

| PoC | Validates | Key Result |
|-----|----------|-----------|
| 001-006 | Pattern encoding, resonance, attention, scale, generation, persistence | Foundation layer proven |
| 007 | PAC tree memory | 12.5x savings, 100% hit rate |
| 011 | Fracton 2.0 integration | GPU-native substrate |
| 012 | Continuous learning | +24.7% accuracy, 50-90K steps/sec |
| 013-014 | Kronos persistence | 100% accuracy retention across restart |
| 016-017 | PAC extraction/import | Cross-model grafting |
| 019 | True no-backprop | Pure conservation proven |
| 020 | Multi-model PAC | 100% cross-model success |
| 021-022 | Scale/stress test | 65% hit rate at 100K vocab |
| 023 | Semantic probe | Lambda ~ 0.5 quality threshold |
| 024 | Phi-weight ablation | Golden ratio critical at depth 4 |
| 025 | GAIA+Kronos integration | Complete cognitive system validated |

## How to Work With GAIA

```python
# Quick usage
from gaia_prime import GAIA_Prime
model = GAIA_Prime.from_gpt2()
model.learn("Training text...")
result = model.generate("Once upon a time")

# With physics layer
from gaia_prime import PhysicsMesh, PACMeshSpace
mesh = PACMeshSpace(embed_dim=768)
physics = PhysicsMesh(mesh)
```

Tests: `python -m pytest tests/` from this directory.

## Build-Out Roadmap

1. **Fracton Integration** — Replace manual physics (Layer 4) with fracton SDK. Files marked with `# TODO(fracton)`.
2. **Unified Entry Point** — CLI: `gaia learn`, `gaia generate`, `gaia serve`. Config system.
3. **Kronos Integration** — Wire in persistent memory (PoC-025 proved it works).
4. **Test Hardening** — Comprehensive unit + integration + regression tests.

## Guardrails

- Do NOT modify `src/legacy/` — archived reference only
- Do NOT break the PoC validation chain — every feature must have a backing experiment
- Do NOT add fracton as a hard dependency yet — gaia_prime must work standalone
- Keep constants derived, not fitted — if a constant appears, it must have a mathematical derivation
- PAC conservation invariant: f(parent) = sum(f(children)) must hold everywhere
