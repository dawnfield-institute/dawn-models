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
│   ├── tinycimm/            # TinyCIMM architecture experiments
│   ├── theoretical/         # Pure theory implementations
│   ├── experiments/         # Research notebooks and explorations
│   └── LICENSE             # AGPL-3.0
│
├── stable/                  # Apache-2.0 - Production models
│   ├── cimm-legacy/        # First stable release
│   ├── tinycimm-v1/        # General purpose (future)
│   ├── gaia-v1/            # Flagship model (future)
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

- **TinyCIMM Variants** - Various architectural experiments
- **Theoretical Models** - Pure Dawn Field Theory implementations
- **SCBF Framework** - Symbolic Collapse Bifractal experiments

### Stable Models (Apache-2.0)
*Production-ready for any use*

- **CIMM Legacy** - Original stable implementation
- *More coming soon...*

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
# Using a stable model (Apache-2.0)
from stable.cimm_legacy import CIMMModel
model = CIMMModel()
result = model.process(data)

# Using a research variant (AGPL-3.0)
from research.tinycimm.attention_v2 import TinyCIMMAttention
model = TinyCIMMAttention()
result = model.process(data)
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
