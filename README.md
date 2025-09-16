# Dawn Models: Post-Symbolic AI Architectures

[![License: Dual](https://img.shields.io/badge/License-AGPL%2FApache-blue.svg)](./LICENSING.md)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Development Status](https://img.shields.io/badge/status-active-green.svg)](https://github.com/dawnfield-institute/dawn-models)

## Overview

Dawn Models implements post-symbolic AI architectures based on Dawn Field Theory principles. The repository provides experimental research models and production-ready implementations using entropy-driven learning, symbolic collapse dynamics, and bifractal computation patterns.

**This is part of the [Dawn Field Theory](https://github.com/dawnfield-institute/dawn-field-theory) ecosystem, providing AI models that transcend traditional symbolic computation.**

## Installation

```bash
# For research use (AGPL-3.0)
cd research/tinycimm
pip install -r requirements.txt

# For production use (Apache-2.0)
cd stable/cimm-legacy
pip install -r requirements.txt
```

## Quick Start

```python
# Using stable CIMM model (Apache-2.0)
import sys
sys.path.append('stable/cimm-legacy')
from cimm_core.cimm import CIMM
from agents.base_agent import BaseAgent

# Initialize CIMM agent with entropy-driven learning
agent = BaseAgent(entropy_threshold=0.1)
result = agent.process(data)

# Using research TinyCIMM variant (AGPL-3.0)
import sys
sys.path.append('research/tinycimm/TinyCIMM-Euler')
from tinycimm_euler import TinyCIMMEuler

# Mathematical reasoning with symbolic collapse
model = TinyCIMMEuler(sequence_length=1000)
prediction = model.predict_sequence(input_sequence)
```

## Core Philosophy

- **Dual Licensing Strategy**: Research models (AGPL-3.0) for transparency, stable models (Apache-2.0) for adoption
- **Entropy-Informed Learning**: Models that adapt based on information entropy and collapse dynamics
- **Symbolic Transcendence**: Post-symbolic architectures that operate beyond traditional token processing
- **Field-Aware Intelligence**: Models that understand and respond to contextual field states
- **Recursive Balance**: Sustainable development through multiple value streams

## Model Architectures

## Model Architectures

### Research Models (AGPL-3.0)
*Experimental implementations for specialized research*

#### **CIMM (Cosmic Information Mining Model)**
- Post-symbolic AI framework using entropy-based learning
- Multi-agent agentic mesh runtime for distributed cognition
- Symbolic collapse dynamics for adaptive pattern recognition

#### **TinyCIMM Variants**
- **TinyCIMM-Euler**: Number theory and mathematical sequence prediction
- **TinyCIMM-Navier**: Fluid dynamics and turbulence analysis  
- **TinyCIMM-Planck**: Minimal foundational implementations

#### **SCBF (Symbolic Collapse Bifractal Framework)**
- Interpretability framework for measuring symbolic collapse in neural networks
- Weight evolution analysis and entropy-based pattern detection
- Bifractal analysis for understanding model behavior

#### **GAIA (Generative AI Intelligence Architecture)**
- Early autonomous agent framework for emergent intelligence
- Multi-agent architectures with distributed cognition patterns
- Foundation for self-modifying and recursive intelligence systems

### Stable Models (Apache-2.0)
*Production-ready implementations for any use*

#### **CIMM Legacy**
- Foundational post-symbolic AI framework
- Complete with testing suite and documentation
- Ready for integration into production systems

## Repository Structure

```
dawn-models/
├── research/                 # AGPL-3.0 - Experimental variants
│   ├── GAIA/                # Generative AI Intelligence Architecture
│   ├── scbf/                # Symbolic Collapse Bifractal Framework
│   └── tinycimm/            # TinyCIMM architecture experiments
├── stable/                  # Apache-2.0 - Production models
│   └── cimm-legacy/         # Stable CIMM implementation
├── roadmaps/                # Development roadmaps and planning
└── docs/                    # Documentation (CONTRIBUTING.md, LICENSING.md)
```

## Getting Started

### Basic CIMM Implementation
```python
import sys
sys.path.append('stable/cimm-legacy')
from cimm_core.cimm import CIMM
from agents.base_agent import BaseAgent

# Initialize agent with entropy-driven learning
agent = BaseAgent(
    entropy_threshold=0.1,
    field_awareness=True,
    symbolic_transcendence=True
)

# Process data with post-symbolic intelligence
result = agent.process(input_data)

# Analyze entropy collapse patterns
patterns = agent.get_collapse_dynamics()
```

### Research Model Example
```python
# SCBF interpretability analysis
import sys
sys.path.append('research/scbf')
from scbf_runner import SCBFRunner

# Analyze model symbolic collapse
scbf = SCBFRunner(enable_visualization=True)
analysis = scbf.analyze_model(model, input_data)
bifractal_trace = analysis.get_bifractal_patterns()
```

## Licensing Quick Reference

| Use Case | Location | License | Notes |
|----------|----------|---------|-------|
| Academic Research | `/research` | AGPL-3.0 | Open research, copyleft |
| Open Source Project | Either | Respective | Follow license terms |
| Commercial Product | `/stable` | Apache-2.0 | Free commercial use |
| Specialized Commercial | `/research` | Contact us | Commercial licensing available |

*See [LICENSING.md](./LICENSING.md) for complete licensing strategy.*

## Contributing

We welcome contributions to both research and stable models! See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidelines on:
- Research model contributions (AGPL-3.0)
- Stable model improvements (Apache-2.0)  
- Documentation and infrastructure

## Dawn Field Theory Ecosystem

Dawn Models is part of the larger Dawn Field Theory ecosystem:

- **[dawn-field-theory](https://github.com/dawnfield-institute/dawn-field-theory)** - Core theoretical foundation
- **[dawn-models](https://github.com/dawnfield-institute/dawn-models)** - AI architectures and implementations ⭐
- **[cip-core](https://github.com/dawnfield-institute/cip-core)** - Cognition Index Protocol
- **[fracton](https://github.com/dawnfield-institute/fracton)** - Computational modeling language
- **[dawn-devkit](https://github.com/dawnfield-institute/dawn-devkit)** - Development tools and templates

## Documentation

- **Licensing Strategy**: [LICENSING.md](./LICENSING.md)
- **Contributing Guide**: [CONTRIBUTING.md](./CONTRIBUTING.md)
- **Development Roadmaps**: [roadmaps/](./roadmaps/)
- **Research Models**: [research/](./research/)
- **Stable Models**: [stable/](./stable/)

## Development

- **Model Roadmaps**: [roadmaps/](./roadmaps/) - GAIA, SCBF, and symbolic entropy development plans
- **Research Status**: See individual model directories for current development status
- **Contributing**: [CONTRIBUTING.md](./CONTRIBUTING.md) - Contribution guidelines and processes
- **Issues**: [GitHub Issues](https://github.com/dawnfield-institute/dawn-models/issues)

## Contact & Support

- **General Inquiries**: info@dawnfield.ca
- **Research Collaboration**: info@dawnfield.ca  
- **Commercial Licensing**: info@dawnfield.ca
- **Enterprise Support**: info@dawnfield.ca

*For detailed support tiers and commercial licensing options, see [LICENSING.md](./LICENSING.md).*

## License

Dual License:
- Research models: AGPL-3.0 (see [research/LICENSE](./research/LICENSE))
- Stable models: Apache-2.0 (see [stable/LICENSE](./stable/LICENSE))

See [LICENSING.md](./LICENSING.md) for complete licensing strategy.
