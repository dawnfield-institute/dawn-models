# Dawn Models: Post-Symbolic AI Architectures

[![License: Dual](https://img.shields.io/badge/License-AGPL%2FApache-blue.svg)](./LICENSING.md)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Development Status](https://img.shields.io/badge/status-active-green.svg)](https://github.com/dawnfield-institute/dawn-models)

---

## Overview

Dawn Models implements post-symbolic AI architectures based on Dawn Field Theory principles. The primary model is **GAIA v2** — a modular intelligence architecture where specialized modules compose via a PAC conservation bus. Also includes TinyCIMM domain-specialized variants, SCBF interpretability framework, and CIMM Legacy.

**This is part of the [Dawn Field Theory](https://github.com/dawnfield-institute/dawn-field-theory) ecosystem.**

## GAIA v2 — Modular Intelligence Architecture

GAIA v2 treats intelligence as a composition of specialized modules — language, reasoning, safety, memory — connected by a **conservation bus** that enforces PAC conservation at every boundary. The bus uses SEC entropy phases for zero-parameter routing and RBF balance for self-regulation. Built on [Fracton SDK 2.1](https://github.com/dawnfield-institute/fracton).

```
Modules:     Language | Reasoning | Safety | Memory | Observability
                 |          |          |        |           |
Bus:        [  PAC conservation  |  SEC routing  |  RBF regulation  ]
                 |          |          |        |           |
Substrate:  [              Fracton SDK 2.1                          ]
                 |          |          |        |           |
Interfaces: [ Spinal Column | MCP/Agents | GRIM | Kronos Vault     ]
```

Key properties:
- **Conservation as contract** — PAC violation at any boundary = detectable hallucination
- **Zero-parameter routing** — SEC phase of input determines which module handles it
- **Continuous learning** — O(1) per token, no retraining, no catastrophic forgetting
- **Modular composition** — swap, add, remove modules without retraining

### Validated Foundations

| Finding | Confidence |
|---------|-----------|
| PAC conservation holds (residual = 0) across all experiments | High |
| O(1) learning per token, zero gradient descent | High |
| 12.5x memory savings via delta-only PACTree | High |
| 100% cross-model embedding graft (GPT-2 to Pythia) | High |
| Hallucination = +9.6% PAC violation in GPT-2 | Medium-High |
| 12,000x MLP advantage with Mobius neurons | Medium-High |

## TinyCIMM — Domain-Specialized Variants

Five lightweight models implementing PAC/SEC/MED principles for different domains. Self-contained, no cross-dependencies. Boltzmann and Mobius are ancestors of GAIA v2's safety and reasoning modules.

- **TinyCIMM-Euler**: Mathematical pattern recognition with 6-metric SCBF instrumentation
- **TinyCIMM-Navier**: Fluid dynamics with turbulent breakthrough detection (4/4)
- **TinyCIMM-Planck**: Quantum-inspired adaptive architecture with grow/prune
- **TinyCIMM-Mobius**: Continuous learning via Mobius transformations (12,000x MLP advantage)
- **TinyCIMM-Boltzmann**: Hallucination detection as PAC violation (+9.6%)

## SCBF — Interpretability Framework

Symbolic Collapse Bifractal Framework — measures symbolic collapse and bifractal patterns in neural network weight evolution. Standalone analysis tool.

## CIMM Legacy — Production Engine

Mature entropy-based intelligence engine with Bayesian optimization, multi-agent consensus, and superfluid dynamics. Apache-2.0 licensed for production use.

## Repository Structure

```
dawn-models/
├── research/                 # AGPL-3.0 — Experimental
│   ├── GAIA/                 # Modular intelligence architecture
│   │   ├── src/gaia/         # v2 source (core, modules, interfaces)
│   │   ├── tests/            # v2 tests
│   │   ├── spikes/v1/        # Archived v1 (25 POCs, 8.4K lines production code)
│   │   └── .spec/            # v2 spec + v1 spec
│   ├── scbf/                 # Interpretability framework
│   └── tinycimm/             # 5 domain-specialized models
├── stable/                   # Apache-2.0 — Production
│   └── cimm-legacy/          # Production CIMM engine
├── roadmaps/                 # Development plans
│   └── gaia-v2-roadmap.md    # GAIA v2 milestone plan (M0-M8)
└── docs/                     # CONTRIBUTING.md, LICENSING.md
```

## Getting Started

```bash
# Install Fracton (required for GAIA v2)
cd ../fracton && pip install -e .

# GAIA v2 (in development)
cd research/GAIA
pip install -e .

# TinyCIMM variants (self-contained)
cd research/tinycimm/TinyCIMM-Mobius
pip install -r requirements.txt

# CIMM Legacy (Apache-2.0, production)
cd stable/cimm-legacy
pip install -r requirements.txt
```

## Licensing

| Use Case | Location | License |
|----------|----------|---------|
| Academic Research | `/research` | AGPL-3.0 |
| Open Source Project | Either | Respective |
| Commercial Product | `/stable` | Apache-2.0 |
| Specialized Commercial | `/research` | Contact us |

See [LICENSING.md](./LICENSING.md) for complete licensing strategy.

## Dawn Field Theory Ecosystem

- **[dawn-field-theory](https://github.com/dawnfield-institute/dawn-field-theory)** — Core theoretical foundation
- **[fracton](https://github.com/dawnfield-institute/fracton)** — Computing SDK (GAIA's physics substrate)
- **[reality-engine](https://github.com/dawnfield-institute/reality-engine)** — Physics simulation and validation
- **[dawn-models](https://github.com/dawnfield-institute/dawn-models)** — AI architectures and implementations
- **[dawn-infrastructure](https://github.com/dawnfield-institute/dawn-infrastructure)** — Deployment and ops

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## Contact

- **General Inquiries**: info@dawnfield.ca
- **Research Collaboration**: info@dawnfield.ca
- **Commercial Licensing**: info@dawnfield.ca

See [LICENSING.md](./LICENSING.md) for complete licensing strategy.
