# GAIA v2 Roadmap — Modular Intelligence Architecture

**Status**: Design phase
**Started**: March 2026
**Spec**: `research/GAIA/.spec/gaia-v2.spec.md`
**Vault**: `proj-gaia-v2`, `gaia-v2-architecture`, `adr-gaia-modular-architecture`

---

## Vision

GAIA v2 transforms from a monolithic language model prototype into a **modular intelligence architecture**. Specialized modules compose via a PAC conservation bus built on Fracton 2.1. The architecture supports language understanding, physics-aware reasoning, hallucination detection, continuous learning, and eventually hardware control — all governed by thermodynamic conservation laws.

Long-term, GAIA becomes the intelligence core for GRIM — enhancing the AI companion with continuous learning, physics-grounded reasoning, and composable capability expansion.

---

## Milestones

### M0: Foundation — Conservation Bus + Module Protocol

**Goal**: Define and implement the core infrastructure that everything else plugs into.

**Deliverables**:
- `GAIAModule` protocol (process/phase/health interface)
- `FieldState` data type (entropy-encoded state transport)
- `ConservationBus` (PAC validation at boundaries, SEC routing, RBF regulation)
- `SECRouter` (zero-parameter phase-based dispatch)
- Integration tests proving bus correctly validates/rejects conservation violations

**Dependencies**: `fracton >= 2.1` (PACRegulator, EntropyDispatcher, SECFieldEvolver)

**Key decisions**:
- Hard vs soft conservation enforcement (recommendation: configurable, default hard)
- SEC phase thresholds (use DFT-derived: 0.5, 2.0, 4.0)
- FieldState tensor shape conventions

---

### M1: Safety Module — Hallucination Detection

**Goal**: Port TinyCIMM-Boltzmann as the first reference module, proving the module protocol works.

**Source**: `research/tinycimm/TinyCIMM-Boltzmann/tinycimm_boltzmann.py` (829 lines)

**Deliverables**:
- `SafetyModule` implementing `GAIAModule`
- `BoltzmannMonitor` for real-time PAC violation detection
- `ConservationProjector` (soft + hard enforcement modes)
- Tests: module passes bus conservation validation
- Benchmark: reproduce +9.6% hallucination entropy finding on GPT-2

**Why first**: Most immediately useful. Can wrap any existing transformer for monitoring. Doesn't require the other modules to work.

---

### M2: Memory Module — PACTree + Bifractal Hierarchy

**Goal**: Port validated memory systems to work as a GAIA module via Fracton primitives.

**Source**: GAIA v1 POCs 006-007, 012-014; `spikes/v1/src/gaia_prime/pac_tree.py`, `bifractal_resonance.py`

**Deliverables**:
- `MemoryModule` implementing `GAIAModule`
- PACTree via `fracton.core.PACNode` (not reimplemented)
- Bifractal hierarchy via `fracton.BifractalTrace`
- Continuous learning pipeline (O(1) per token)
- Tests: 12.5x memory savings, 100% retrieval at depth 1000, 100% restart retention

**Dependencies**: M0, fracton

---

### M3: Language Module — Embeddings + Token Prediction

**Goal**: Port GAIA Prime's language capabilities as a module.

**Source**: `spikes/v1/src/gaia_prime/` — embeddings.py, transitions.py, generator.py, concentration.py

**Deliverables**:
- `LanguageModule` implementing `GAIAModule`
- Embedding grafting (from_gpt2, from_pythia, from_any)
- TransitionMatrix (sparse N-gram, GPU-accelerated)
- PACGenerator (reject-resample generation)
- ConcentrationMonitor (quality gating)
- Tests: 100% cross-model graft, reproduce 65% hit rate

**Dependencies**: M0, M2 (memory for persistent learning)

---

### M4: Reasoning Module — Mobius Neurons

**Goal**: Port TinyCIMM-Mobius as the reasoning module.

**Source**: `research/tinycimm/TinyCIMM-Mobius/tinycimm_mobius.py` (664 lines)

**Deliverables**:
- `ReasoningModule` implementing `GAIAModule`
- `MobiusNeuron` (M(z) = (az+b)/(cz+d)) as core computation unit
- `PhiAnchorMemory` for anti-forgetting
- `MobiusHarmonic` for frequency tracking
- Tests: reproduce 12,000x MLP advantage, 0.003% phi error
- Investigation: does advantage hold for non-iterative tasks?

**Dependencies**: M0

---

### M5: Integration — Multi-Module Composition

**Goal**: All modules working together through the conservation bus.

**Deliverables**:
- Multi-module routing via SEC phases
- RBF-regulated module activation
- Cross-module PAC conservation validation
- End-to-end inference: input -> bus -> module(s) -> output
- Conservation accounting across full pipeline

**Dependencies**: M1-M4

---

### M6: Benchmarks — Honest Evaluation

**Goal**: Establish baseline numbers on standard and custom benchmarks.

**Deliverables**:
- Standard LM eval harness integration (WikiText-2, LAMBADA, HellaSwag)
- Custom eval axes:
  - Tokens learned per FLOP (vs fine-tuning)
  - Accuracy after N domains without forgetting (vs EWC)
  - PAC violation correlation with hallucination (TruthfulQA)
  - Memory footprint comparison
- Honest reporting: where GAIA wins, where it doesn't, and why

**Dependencies**: M5

---

### M7: GRIM Integration

**Goal**: GAIA as GRIM's intelligence substrate.

**Deliverables**:
- GAIA as MCP server (tools for reasoning, hallucination check, learning)
- Integration with Kronos vault for knowledge persistence
- GRIM pool agents enhanced with GAIA continuous learning
- Proof: GRIM demonstrably improves over time through GAIA

**Dependencies**: M5

---

### M8: Spinal Column — Hardware + Agent Interfaces

**Goal**: GAIA controlling external systems through the module protocol.

**Deliverables**:
- `SpinalInterface` base class (GAIAModule wrapping hardware)
- Reference implementation for at least one peripheral
- MCP tool interface for agent orchestration
- Documentation: how to add a new peripheral as a GAIA module

**Dependencies**: M5

---

## Observability Module (Ongoing)

Instrumentation from TinyCIMM-Euler runs alongside all milestones, not as a separate milestone:
- SCBF 6-metric tracker
- QBE equilibrium monitoring
- Module health dashboards
- Conservation violation logging

---

## Principles

1. **One module at a time**: Each milestone delivers a working, tested module. No big-bang integration.
2. **Fracton is the physics**: If GAIA needs a primitive, add it to Fracton (upstream), not to GAIA.
3. **Honest metrics from day one**: No "perplexity" that's actually cosine similarity. Measure what you claim.
4. **v1 is reference, not legacy**: The 25 POCs are validated findings. Port the learning, not the code.
5. **Conservation is the API**: The module protocol IS the interface contract. Physics does the routing.

---

## Related Documents

- **Spec**: `research/GAIA/.spec/gaia-v2.spec.md`
- **v1 archive**: `research/GAIA/spikes/v1/`
- **TinyCIMM ancestors**: `research/tinycimm/`
- **Fracton SDK**: `fracton/` (workspace root)
- **Vault FDOs**: `gaia-v2-architecture`, `proj-gaia-v2`, `adr-gaia-modular-architecture`
