# Dawn Models — Claude Code Context

## Identity
Dawn Models is the model repository for the Dawn Field Institute. The primary model is **GAIA v2** (Generative AI via Information Architecture) — a modular intelligence architecture where specialized modules compose via a PAC conservation bus, built on Fracton SDK 2.1. Also includes TinyCIMM domain-specific variants (ancestors of GAIA v2 modules), SCBF interpretability framework, and CIMM Legacy.

## Architecture

```
dawn-models/
├── research/                      # AGPL-3.0 — experimental
│   ├── GAIA/                      # PRIMARY — modular intelligence architecture
│   │   ├── .spec/                 # v2 spec + v1 spec (reference)
│   │   ├── src/gaia/              # v2 source code (modular architecture)
│   │   │   ├── core/              # Conservation bus, module protocol
│   │   │   ├── modules/           # Pluggable modules (language, reasoning, safety, memory)
│   │   │   └── interfaces/        # Spinal column, MCP, GRIM integration
│   │   ├── tests/                 # v2 test suite
│   │   └── spikes/v1/             # Archived v1 code (8.4K lines + 25 POCs + docs)
│   ├── scbf/                      # Symbolic Collapse Bifractal Framework
│   └── tinycimm/                  # 5 domain-specific variants (GAIA v2 ancestors)
│       ├── TinyCIMM-Euler/        # Number theory (SCBF reference → observability module)
│       ├── TinyCIMM-Navier/       # Fluid dynamics
│       ├── TinyCIMM-Planck/       # Foundational minimal
│       ├── TinyCIMM-Boltzmann/    # Hallucination detection (→ safety module)
│       └── TinyCIMM-Mobius/       # Continuous learning (→ reasoning module)
├── stable/                        # Apache-2.0 — production
│   └── cimm-legacy/               # Mature CIMM with agents/mesh runtime
└── roadmaps/                      # Development plans
```

## GAIA v2 — Modular Intelligence Architecture

GAIA v2 is a ground-up redesign. Specialized modules (language, reasoning, safety, memory, observability) compose via a **conservation bus** that enforces PAC conservation at every module boundary. Uses Fracton SDK 2.1 as the physics substrate — zero reimplementation of physics.

### Module Protocol

Every module implements three methods:
- `process(field_state) -> field_state` — PAC-conserving transformation
- `phase() -> SECPhase` — current entropy phase (for zero-parameter routing)
- `health() -> RBFBalance` — energy-information balance (for self-regulation)

### Planned Modules

| Module | Source | Key Innovation |
|--------|--------|---------------|
| **Safety** | TinyCIMM-Boltzmann | Hallucination = PAC violation (+9.6%), conservation enforcement |
| **Memory** | GAIA v1 POCs | PACTree (12.5x savings), bifractal hierarchy, O(1) continuous learning |
| **Language** | GAIA v1 gaia_prime | Embedding grafting (100% cross-model), TransitionMatrix, generation |
| **Reasoning** | TinyCIMM-Mobius | Mobius neurons (12,000x MLP advantage), PhiAnchorMemory |
| **Observability** | TinyCIMM-Euler | 6-metric SCBF tracker, QBE equilibrium monitoring |

### Interfaces (Future)

- **Spinal Column**: Hardware control (GPIO, CAN bus, robotics)
- **MCP Server**: Agent orchestration tools
- **GRIM Integration**: Intelligence substrate for the AI companion
- **Kronos**: Knowledge persistence via vault FDOs

### v1 Archive

All v1 code lives in `research/GAIA/spikes/v1/`:
- `src/gaia_prime/` — 20 modules, ~8.4K lines (monolithic architecture)
- `src/legacy/` — v1-v4 reference code
- `proof_of_concepts/` — 25 validated POCs backing v2 design decisions
- `docs/` — 93 theory & architecture docs
- `tests/`, `benchmarks/`, `training/`, `usecases/`

### Key Validated Results (Carrying Forward)
- No backpropagation — learning is pure counting, O(1) per token
- 12.5x memory savings — delta-only PAC tree storage
- 100% cross-model grafting — extract and transplant knowledge between models
- Hallucination = PAC violation (+9.6% uncompensated entropy in GPT-2)
- 12,000x MLP advantage with Mobius neurons on iterated dynamics
- SEC predicts crystallization with zero parameters

### Fracton Dependency
GAIA v2 depends on `fracton >= 2.1` for all physics. No reimplemented constants, conservation checks, or field operations. If GAIA needs a primitive Fracton doesn't have, it goes upstream into Fracton.

## Secondary Models

- **SCBF**: Interpretability framework — measures symbolic collapse and bifractal patterns in neural network weight evolution. Standalone analysis tool, not a model.
- **TinyCIMM**: 5 domain-specific lightweight models. Self-contained, no cross-variant dependencies. Boltzmann and Mobius are ancestors of GAIA v2's safety and reasoning modules respectively.
- **CIMM Legacy** (stable/): Mature production implementation with agent mesh runtime. Apache-2.0 licensed.

## Conventions

- Dual licensing: research/ = AGPL-3.0, stable/ = Apache-2.0
- GAIA v2 development happens in `research/GAIA/src/gaia/`
- v1 code in `spikes/v1/` is archived reference — do not modify
- Spec: `research/GAIA/.spec/gaia-v2.spec.md`
- Roadmap: `roadmaps/gaia-v2-roadmap.md`
- Each model variant has its own requirements.txt
- Tests: `cd research/GAIA && python -m pytest tests/`

## Related Repos

- `fracton` — Infodynamics SDK (GAIA's physics substrate, hard dependency)
- `dawn-field-theory` — core theoretical foundation (experiments validate the math)
- `reality-engine` — physics simulation (validates DFT before it reaches GAIA)
- `kronos-vault` — knowledge graph (FDOs track GAIA architecture and progress)
- `GRIM` — AI companion (future GAIA integration target)

## Guardrails

- Do NOT mix AGPL and Apache code across research/stable boundary
- Do NOT modify `spikes/v1/` — it's archived reference
- Do NOT reimplement Fracton physics in GAIA — use `fracton` imports
- Respect the PoC validation chain — v1 POCs back v2 design decisions
- Each TinyCIMM variant is self-contained
- Conservation bus is the API contract — all modules must satisfy it
