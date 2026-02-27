# Dawn Models — Claude Code Context

## Identity
Dawn Models is the model repository for the Dawn Field Institute. The primary model is **GAIA** (Generative AI via Information Architecture) — a language model built entirely from PAC/SEC principles with no backpropagation. Also includes TinyCIMM domain-specific variants, SCBF interpretability framework, and CIMM Legacy.

## Architecture

```
dawn-models/
├── research/                      # AGPL-3.0 — experimental
│   ├── GAIA/                      # PRIMARY — validated PAC-native language model
│   │   ├── src/gaia_prime/        # v2.0.0 production code (15 modules, ~8.4K lines)
│   │   ├── src/legacy/            # Archived v1-v4 reference (do not modify)
│   │   ├── proof_of_concepts/     # 25 validated PoCs backing every decision
│   │   ├── tests/                 # Test suite
│   │   ├── benchmarks/            # Performance benchmarks
│   │   ├── docs/                  # Theory & architecture docs (93 files)
│   │   ├── training/              # Training pipelines
│   │   └── usecases/              # Application examples
│   ├── scbf/                      # Symbolic Collapse Bifractal Framework
│   └── tinycimm/                  # 5 domain-specific variants
│       ├── TinyCIMM-Euler/        # Number theory
│       ├── TinyCIMM-Navier/       # Fluid dynamics
│       ├── TinyCIMM-Planck/       # Foundational minimal
│       ├── TinyCIMM-Boltzmann/    # Statistical mechanics
│       └── TinyCIMM-Mobius/       # Topology
├── stable/                        # Apache-2.0 — production
│   └── cimm-legacy/               # Mature CIMM with agents/mesh runtime
└── roadmaps/                      # Development plans
```

## GAIA Prime — The Main Model

GAIA Prime (v2.0.0) is a language model that learns through PAC conservation, not gradient descent. Every architectural decision is backed by a validated proof-of-concept experiment.

### Architecture (5 layers)

```
L0: GraftedEmbeddings       <- Frozen from GPT-2/Pythia (PoC-016,017,020)
L1: PACTree                 <- Delta-only hierarchical storage (PoC-007,020: 12.5x memory savings)
L2: TransitionMatrix        <- Sparse N-gram counting, GPU-accelerated (PoC-021,022: 65% hit rate)
L3: ConcentrationMonitor    <- Quality gating, reject-resample (PoC-023: +3.6% quality)
L4: PhysicsMesh             <- Entropy, conservation, collapse, resonance
L5: PACGenerator            <- Greedy/sample/beam decoding + continuous learning
```

### Key Validated Results
- No backpropagation — learning is pure counting, O(1) per token
- 12.5x memory savings — delta-only PAC tree storage
- 65% hit rate at 100K vocabulary
- 100% cross-model grafting — extract and transplant knowledge between models
- 100% accuracy retention across session restart (Kronos persistence)
- 25/25 PoCs passing — complete system validated

### Fracton Integration Status
gaia_prime currently reimplements physics manually. These files should migrate to fracton:

| File | What it reimplements | Fracton equivalent |
|------|---------------------|-------------------|
| `physics_mesh.py` (48KB) | Entropy, conservation, collapse, resonance | `fracton.field`, `fracton.core`, `fracton.physics` |
| `auto_collapse.py` (22KB) | 5 collapse strategies, entropy tracking | `fracton.field`, `fracton.core.EntropyDispatcher` |
| `bifractal_resonance.py` (20KB) | Depth-based bifractal patterns | `fracton.core.BifractalTrace` |
| `pac_mesh.py` (27KB) | PAC mesh space, node hierarchy | `fracton.core.PACSystem`, `fracton.core.PACNode` |
| `validated_constants.py` (6KB) | XI, PHI, LAMBDA_STAR | `fracton.physics.constants` |

Files marked with `# TODO(fracton)` comments in the source.

## Secondary Models

- **SCBF**: Interpretability framework — measures symbolic collapse and bifractal patterns in neural network weight evolution. Standalone analysis tool, not a model.
- **TinyCIMM**: 5 domain-specific lightweight models. Self-contained, no cross-variant dependencies.
- **CIMM Legacy** (stable/): Mature production implementation with agent mesh runtime.

## Conventions

- Dual licensing: research/ = AGPL-3.0, stable/ = Apache-2.0
- GAIA development happens in `research/GAIA/src/gaia_prime/`
- Legacy code in `src/legacy/` is reference only — do not modify
- New experiments go in `proof_of_concepts/` following POC_REGISTRY.md format
- Each model variant has its own requirements.txt
- Tests: `cd research/GAIA && python -m pytest tests/`

## Related Repos

- `fracton` — Infodynamics SDK (GAIA should consume fracton for physics layer)
- `dawn-field-theory` — core theoretical foundation (experiments validate the math)
- `reality-engine` — physics simulation (imports fracton for PAC/Mobius)
- `kronos-vault` — knowledge graph (FDOs reference GAIA PoCs)
- `GRIM` — MCP server + skills (Kronos integration layer)

## Guardrails

- Do NOT mix AGPL and Apache code across research/stable boundary
- Do NOT modify src/legacy/ — it's archived reference
- Do NOT break gaia_prime's independence — it must work without fracton until migration
- Respect the PoC validation chain — every gaia_prime feature has a backing PoC
- Each TinyCIMM variant is self-contained
