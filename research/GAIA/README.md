# GAIA v2 — Modular Intelligence Architecture

GAIA (Generative AI via Information Architecture) is a modular intelligence system where specialized components compose via a **PAC conservation bus**. Each module is a self-contained intelligence unit — language, reasoning, safety, memory — that communicates through entropy-encoded field states. The bus enforces physics-derived conservation at every boundary.

Built on [Fracton SDK 2.1](https://github.com/dawnfield-institute/fracton) as the physics substrate.

## Architecture

```
Modules:     Language | Reasoning | Safety | Memory | Observability | ...
                 |          |          |        |           |
Bus:        [  PAC conservation  |  SEC routing  |  RBF regulation  ]
                 |          |          |        |           |
Substrate:  [              Fracton SDK 2.1                          ]
                 |          |          |        |           |
Interfaces: [ Spinal Column | MCP/Agents | GRIM | Kronos Vault     ]
```

Each module implements a three-method protocol:
- `process(field_state) -> field_state` — PAC-conserving transformation
- `phase() -> SECPhase` — current entropy phase (for routing)
- `health() -> RBFBalance` — energy-information balance (for regulation)

## Key Properties

- **Conservation as contract**: PAC violation at any module boundary = detectable error
- **Zero-parameter routing**: SEC phase of the input determines which module handles it
- **Self-regulating**: RBF naturally suppresses modules with poor energy-to-information ratio
- **Continuous learning**: O(1) per token, no retraining, no catastrophic forgetting
- **Hallucination = thermodynamic violation**: Detected by the safety module, not heuristics

## Validated Foundations (from v1 + TinyCIMM)

| Finding | Source | Confidence |
|---------|--------|-----------|
| PAC conservation holds (residual = 0) | 25 POCs | High |
| O(1) learning, zero gradients | POC-012, 019 | High |
| 12.5x memory savings (PACTree) | POC-007 | High |
| 100% cross-model embedding graft | POC-016, 017, 020 | High |
| Hallucination = +9.6% PAC violation | TinyCIMM-Boltzmann | Medium-High |
| 12,000x MLP advantage (Mobius neurons) | TinyCIMM-Mobius | Medium-High |
| SEC predicts crystallization (zero params) | Computational validation | Medium-High |

## Directory Structure

```
research/GAIA/
├── .spec/
│   ├── gaia-v2.spec.md       # v2 architecture spec
│   └── gaia.spec.md          # v1 spec (reference)
├── src/gaia/                 # v2 source code
│   ├── core/                 # Conservation bus, module protocol
│   ├── modules/              # Pluggable module implementations
│   └── interfaces/           # Spinal column, MCP, GRIM integration
├── tests/                    # v2 test suite
├── spikes/v1/                # All v1 code (preserved)
│   ├── src/                  # gaia_prime/ (8.4K lines) + legacy/
│   ├── proof_of_concepts/    # 25 validated POCs
│   ├── tests/                # v1 tests
│   ├── benchmarks/           # v1 benchmarks
│   ├── docs/                 # 93 documentation files
│   ├── training/             # v1 training pipelines
│   └── usecases/             # v1 application examples
├── meta.yaml
└── README.md                 # This file
```

## Status

**Phase**: Design (March 2026)

See `roadmaps/gaia-v2-roadmap.md` for the full milestone plan (M0-M8).

## Related

- [Fracton SDK](https://github.com/dawnfield-institute/fracton) — physics substrate
- [Dawn Field Theory](https://github.com/dawnfield-institute/dawn-field-theory) — theoretical foundation
- [TinyCIMM variants](../tinycimm/) — ancestor domain-specialized models
- [CIMM Legacy](../../stable/cimm-legacy/) — original production engine
