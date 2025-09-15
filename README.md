# Dawn Models

**Version:** 1.0.0  
**Status:** Private Development Repository  
**License:** Dual (AGPL-3.0 Research / Apache-2.0 Stable)

Dawn Models is the official model repository for the Dawn Field Institute, implementing a dual-licensing strategy that balances open scientific research with practical adoption.

---

## 🎯 Quick Start

### 🔬 For Researchers
```bash
# Explore cutting-edge experimental variants
cd research/
# All research models are AGPL-3.0 licensed
```

### 🚀 For Production
```bash
# Use production-ready, Apache-licensed models
cd stable/
# All stable models are Apache-2.0 licensed
```

### 🏢 For Commercial Use
- **Apache models** (`/stable`) → Use freely in any application
- **AGPL variants** (`/research`) → Commercial licensing available
- **Enterprise support** → Contact info@dawnfield.ca

---

## 📁 Repository Structure

```
dawn-models/
├── research/                 # AGPL-3.0 - Experimental variants
│   ├── gaia/                # Generative AI Intelligence Architecture
│   ├── scbf/                # Symbolic Collapse Bifractal Framework
│   ├── tinycimm/            # TinyCIMM architecture experiments
│   │   ├── TinyCIMM-Euler/  # Number theory and mathematical reasoning
│   │   ├── TinyCIMM-Navier/ # Fluid dynamics and turbulence analysis
│   │   └── TinyCIMM-Planck/ # Minimal foundational implementations
│   └── LICENSE             # AGPL-3.0
│
├── stable/                  # Apache-2.0 - Production models
│   ├── cimm-legacy/        # Stable CIMM implementation
│   └── LICENSE             # Apache-2.0
│
├── README.md               # This file
├── LICENSING.md            # Dual-licensing strategy
└── CONTRIBUTING.md         # Contribution guidelines
```

---

## 🔄 The Model Lifecycle

### Research Phase (`/research` - AGPL-3.0)
- Experimental variants and specialized implementations
- Complete research history and development archive
- Academic use and open-source projects encouraged
- Commercial use requires either AGPL compliance or commercial license

### Stable Release (`/stable` - Apache-2.0)
- Production-ready, general-purpose implementations
- Clean, documented, maintained code
- Free for any use including commercial
- Enterprise support available

### Key Principle
**Research variants never graduate** - they remain permanently in `/research` as a complete historical record and for specialized use cases.

---

## 🎨 Available Models

### Research Variants (AGPL-3.0)
*Experimental models for research and specialized use*

#### **SCBF (Symbolic Collapse Bifractal Framework)**
- Interpretability framework for measuring symbolic collapse
- Neural network weight evolution analysis
- Entropy-based pattern detection and bifractal analysis
- Integration with TinyCIMM for experimental validation

#### **GAIA (Generative AI Intelligence Architecture)**
- Early generative intelligence framework for autonomous agent systems
- Foundation for multi-agent architectures and intelligent automation
- Distributed cognition and emergent intelligence patterns
- Experimental codebase from early Dawn Field Institute research

#### **TinyCIMM Variants**
- **TinyCIMM-Euler** - Number theory and mathematical sequence prediction
- **TinyCIMM-Navier** - Fluid dynamics and turbulence analysis  
- **TinyCIMM-Planck** - Minimal foundational implementations

### Stable Models (Apache-2.0)
*Production-ready for any use*

#### **CIMM Legacy**
- Foundational post-symbolic AI framework
- Entropy-based learning and collapse dynamics
- Multi-agent agentic mesh runtime
- Complete with testing suite and documentation

---

## 🤝 Getting Started

### Installation
```bash
# Clone the repository
git clone https://github.com/dawnfield-institute/dawn-models.git
cd dawn-models

# For research use
cd research/tinycimm
pip install -r requirements.txt

# For production use
cd stable/cimm-legacy
pip install -r requirements.txt
```

### Usage Examples
```python
# Using stable CIMM model (Apache-2.0)
import sys
sys.path.append('stable/cimm-legacy')
from cimm_core.cimm import CIMM
from agents.base_agent import BaseAgent

# Initialize CIMM agent
agent = BaseAgent(entropy_threshold=0.1)
result = agent.process(data)

# Using research TinyCIMM variant (AGPL-3.0)
import sys
sys.path.append('research/tinycimm/TinyCIMM-Euler')
from tinycimm_euler import TinyCIMMEuler

# Initialize for mathematical reasoning
model = TinyCIMMEuler(sequence_length=1000)
prediction = model.predict_sequence(input_sequence)

# Using SCBF for interpretability (AGPL-3.0)
import sys
sys.path.append('research/scbf')
from scbf_runner import SCBFRunner

# Analyze model interpretability
scbf = SCBFRunner(enable_visualization=True)
analysis = scbf.analyze_model(model, input_data)
```

---

## 📊 Licensing Quick Reference

| Use Case | Recommended Path | License | Support |
|----------|-----------------|---------|---------|
| Academic Research | `/research` | AGPL-3.0 | Community |
| Open Source Project | Either | Respective | Community |
| Commercial Product | `/stable` | Apache-2.0 | Available |
| Specialized Commercial | Contact us | Commercial | Direct |

---

## 🔬 Research Philosophy

Dawn Models embodies the **Dawn Field Theory** principle of recursive balance:

- **Transparency** - All research is openly available
- **Sustainability** - Multiple value streams support continued development
- **Innovation** - Specialized knowledge creates natural advantages
- **Collaboration** - Clear paths for community contribution

---

## 📈 Roadmap

### Current (Q4 2025)
- [x] Repository structure established
- [x] Dual-licensing framework implemented
- [ ] CIMM legacy model migration
- [ ] TinyCIMM research variants migration

### Next (Q1 2026)
- [ ] First TinyCIMM stable release
- [ ] Community contribution guidelines
- [ ] Commercial licensing automation
- [ ] Enterprise support tier

### Future (2026+)
- [ ] GAIA model family releases
- [ ] Advanced research collaboration tools
- [ ] Partner certification program

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Paths
- **Bug fixes** in stable models → Direct PRs welcome
- **New research variants** → Research collaboration process
- **Documentation improvements** → Always appreciated
- **Enterprise use cases** → Let's discuss!

---

## 📞 Contact & Support

### Quick Links
- **Commercial Licensing:** [dawnfield.ca/licensing](https://dawnfield.ca/licensing) (coming soon, not yet available)
- **Enterprise Support:** info@dawnfield.ca
- **Research Collaboration:** info@dawnfield.ca
- **General Inquiries:** info@dawnfield.ca

### Support Tiers
1. **Community** (Free) - GitHub issues, discussions
2. **Professional** ($) - Email support, priority fixes
3. **Enterprise** ($$) - SLA, custom development
4. **Research Partner** (Collaboration) - Joint development

---

## 📚 Related Resources

- **[Dawn Field Theory](https://github.com/dawnfield-institute/dawn-field-theory)** - Theoretical foundation
- **[CIP Core](https://github.com/dawnfield-institute/cip-core)** - Cognition Index Protocol
- **[Fracton](https://github.com/dawnfield-institute/fracton)** - Compression framework
- **[Dawn Field Institute](https://dawnfield.ca)** - Main website

---

*Dawn Models: Where cutting-edge research meets practical application through sustainable open science.*
