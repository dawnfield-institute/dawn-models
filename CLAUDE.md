# Dawn Models -- Claude Code Context

## Identity
Dawn Models provides post-symbolic AI architectures based on Dawn Field Theory principles. The repository contains experimental research models (AGPL-3.0) and production-ready implementations (Apache-2.0) using entropy-driven learning, symbolic collapse dynamics, and bifractal computation patterns. Part of the Dawn Field Theory ecosystem.

## Architecture

```
dawn-models/
├── research/                   # AGPL-3.0 -- Experimental models
│   ├── GAIA/                   # Generative AI Intelligence Architecture
│   │   ├── src/                # Core GAIA source
│   │   ├── training/           # Training pipelines
│   │   ├── benchmarks/         # Performance benchmarks
│   │   ├── usecases/           # Application examples
│   │   └── tests/              # Test suite
│   ├── scbf/                   # Symbolic Collapse Bifractal Framework
│   │   ├── scbf_runner.py      # Main runner
│   │   ├── scbf_experiments/   # Experiment scripts
│   │   ├── metrics/            # Collapse metrics
│   │   ├── visualization/      # Bifractal visualization
│   │   └── loggers/            # Experiment logging
│   └── tinycimm/               # TinyCIMM architecture variants
│       ├── TinyCIMM-Euler/     # Number theory, sequence prediction
│       ├── TinyCIMM-Navier/    # Fluid dynamics, turbulence
│       ├── TinyCIMM-Planck/    # Minimal foundational model
│       ├── TinyCIMM-Boltzmann/ # Statistical mechanics
│       └── TinyCIMM-Mobius/    # Mobius topology variant
├── stable/                     # Apache-2.0 -- Production models
│   └── cimm-legacy/            # Stable CIMM implementation
├── roadmaps/                   # Development plans
└── docs/                       # CONTRIBUTING.md, LICENSING.md
```

## Key Models

**CIMM (Cosmic Information Mining Model)**: Post-symbolic AI framework with entropy-based learning and multi-agent mesh runtime
**TinyCIMM Variants**: Domain-specific lightweight models (Euler=math, Navier=fluids, Planck=foundational, Boltzmann=stat-mech, Mobius=topology)
**SCBF**: Interpretability framework for measuring symbolic collapse in neural networks -- weight evolution and bifractal analysis
**GAIA**: Autonomous agent framework for emergent intelligence with multi-agent distributed cognition

## Conventions

- Dual licensing: research/ = AGPL-3.0, stable/ = Apache-2.0
- Each model variant has its own requirements.txt
- Entropy-driven learning is the core paradigm -- models adapt based on information entropy
- Install per-model: `cd research/tinycimm && pip install -r requirements.txt`
- SCBF runner: `python research/scbf/scbf_runner.py`

## Related Repos

- `dawn-field-theory` -- core theoretical foundation
- `fracton` -- Infodynamics SDK (computational substrate)
- `reality-engine` -- physics simulation framework
- `cip-core` -- Cognition Index Protocol
- `dawn-devkit` -- development tools and templates

## Current State

- Active development, multiple model variants in research stage
- CIMM Legacy is production-ready in stable/
- TinyCIMM has 5 domain-specific variants under research
- SCBF interpretability framework operational
- GAIA early-stage autonomous agent framework

## Guardrails

- Do NOT mix AGPL and Apache code across the research/stable boundary
- Do NOT modify stable/cimm-legacy without ensuring backward compatibility
- Respect dual licensing -- research contributions are AGPL-3.0, stable are Apache-2.0
- Each TinyCIMM variant is self-contained -- do not create cross-variant dependencies
- Check licensing implications before moving code between research/ and stable/
