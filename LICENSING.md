# Dawn Models Licensing Strategy

**Version:** 1.0.0  
**Date:** September 2025  
**Status:** Implementation Document  
**Author:** Dawn Field Institute

---

## Executive Summary

The Dawn Models repository employs a **dual-licensing strategy** that balances open scientific research with sustainable development. Experimental models remain under AGPL-3.0 in perpetuity, while production-ready generalizations are released under Apache-2.0. This creates a natural incentive structure for collaboration, innovation, and institutional sustainability.

---

## 🎯 Strategic Goals

1. **Maintain scientific transparency** - All research and experimental work visible
2. **Enable practical adoption** - Production-ready models for real-world use
3. **Protect institutional IP** - Specialized variants remain under copyleft
4. **Create sustainability paths** - Multiple monetization options without compromising openness
5. **Foster innovation** - Community can build on our research with clear terms

---

## 📁 Repository Structure

```
dawn-models/
├── research/                 # AGPL-3.0 - Permanent research archive
│   ├── tinycimm/            # All experimental variants
│   │   ├── attention-v1/    # Specific architectural experiments
│   │   ├── attention-v2/    # Iterative improvements
│   │   ├── memory-exp/      # Memory field variants
│   │   ├── recursive/       # Recursive execution tests
│   │   └── bifractal/       # Bifractal tracing variants
│   ├── theoretical/         # Pure theory implementations
│   ├── experiments/         # Notebooks and explorations
│   └── LICENSE             # AGPL-3.0
│
├── stable/                  # Apache-2.0 - Production models
│   ├── cimm-legacy/        # Legacy model (released)
│   ├── tinycimm-v1/        # General purpose (future)
│   ├── gaia-v1/            # Flagship model (future)
│   └── LICENSE             # Apache-2.0
│
├── README.md               # Repository overview
├── LICENSING.md            # This document
└── CONTRIBUTING.md         # Contribution guidelines
```

---

## 🔄 The Model Lifecycle

### Phase 1: Research & Development
- **Location:** `/research`
- **License:** AGPL-3.0
- **Duration:** Indefinite (permanent archive)
- **Purpose:** Experimental variants, theoretical explorations, ablation studies

### Phase 2: Generalization
- **Process:** Best ideas from multiple variants combined
- **Output:** Clean, documented, general-purpose implementation
- **Review:** Community feedback period

### Phase 3: Production Release
- **Location:** `/stable`
- **License:** Apache-2.0
- **Support:** Semantic versioning, backwards compatibility
- **Purpose:** Production deployment, commercial use

### Important: Research Never Graduates
Research variants remain in `/research` permanently, creating a complete historical record of development and allowing specialized use cases.

---

## 🎨 The Innovation Framework

```
Specialized Variants (AGPL)          General Release (Apache)
├── tinycimm-attention-v1       ──┐
├── tinycimm-attention-v2         │
├── tinycimm-memory-experimental  │──→ tinycimm-general-v1.0
├── tinycimm-recursive-test       │    (best practices only)
├── tinycimm-bifractal-variant  ──┘
└── [dozens of experiments]

Key: Specialized knowledge remains AGPL, general patterns become Apache
```

---

## 💡 Strategic Advantages

### For the Institute

1. **Knowledge Asymmetry**: We understand which variants work for specific use cases
2. **Consultation Opportunities**: Organizations need help selecting/customizing variants
3. **Attribution Preservation**: All derivatives maintain connection to Dawn Field Theory
4. **Sustainable Revenue**: Multiple paths without compromising open science

### For the Community

1. **Full Transparency**: Complete research history available
2. **Choice of Engagement**:
   - Use general Apache models freely
   - Explore AGPL variants for research
   - Collaborate on improvements
3. **No Vendor Lock-in**: Clear licensing, no surprises
4. **Innovation Platform**: Build on cutting-edge research

### For Commercial Users

1. **Clear Options**:
   - Apache models for standard use cases
   - AGPL variants for competitive advantage (with obligations)
   - Commercial licensing for proprietary needs
2. **Risk Management**: Production models are stable and supported
3. **Expert Support**: Direct access to Dawn Field Institute expertise

---

## 📊 Licensing Decision Matrix

| Use Case | Recommended Path | License | Support Level |
|----------|-----------------|---------|---------------|
| Academic Research | `/research` variants | AGPL-3.0 | Community |
| Prototyping | `/stable` models | Apache-2.0 | Community |
| Open Source Project | Either path | Respective license | Community |
| Commercial Product | `/stable` models | Apache-2.0 | Available |
| Specialized Commercial | Contact for `/research` variant | Commercial | Direct |
| Enterprise Deployment | `/stable` + support contract | Apache-2.0 | Enterprise |

---

## 🤝 Revenue Models

### What's Always Free
- ✅ All `/stable` Apache models for any use
- ✅ All `/research` AGPL models for open-source projects
- ✅ Academic and research use of everything

### When Commercial Licensing Applies

#### Tier 1: Community (Free Forever)
- Apache models for any use (including commercial)
- AGPL models for open-source projects
- Community support forums

#### Tier 2: Proprietary Use of AGPL Variants ($-$)
- Want to use `/research` variants in closed-source products
- <$500k revenue: Nominal fee or exemption
- >$500k revenue: Percentage-based licensing
- Email support included

#### Tier 3: Enterprise Support ($-$$)
- For ANY models (Apache or AGPL)
- SLA guarantees
- Priority bug fixes
- Direct engineering support
- Training and workshops

#### Tier 4: Custom Development ($$+)
- Bespoke model variants
- Joint research projects
- Exclusive features
- Co-publication opportunities

---

## 🔒 Intellectual Property Strategy

### What We Protect
- **Specialized implementations** (AGPL variants)
- **Theoretical insights** (research archive)
- **Institutional knowledge** (which variants for which use cases)
- **Model names** (trademark protection)

### What We Share
- **General-purpose implementations** (Apache stable)
- **Research methodology** (full transparency)
- **Development history** (complete git history)
- **Best practices** (documentation)

---

## 📈 Success Metrics

1. **Research Impact**: Citations of AGPL variants in papers
2. **Adoption**: Downloads/forks of Apache models
3. **Collaboration**: External contributions to research
4. **Sustainability**: Revenue from commercial licensing/support
5. **Innovation**: New use cases discovered by community

---

## 🚀 Implementation Roadmap

### Immediate (Now)
- [x] Define strategy (this document)
- [x] Create directory structure
- [ ] Move existing models appropriately
- [x] Add clear LICENSE files

### Short-term (Q4 2025)
- [ ] Release legacy CIMM to `/stable`
- [ ] Establish contribution guidelines
- [ ] Document variant selection guide
- [ ] Set up commercial inquiry process

### Medium-term (Q1 2026)
- [ ] Graduate first TinyCIMM general version
- [ ] Launch professional support tier
- [ ] Create variant comparison matrix
- [ ] Publish case studies

### Long-term (2026+)
- [ ] GAIA model stable release
- [ ] Enterprise support program
- [ ] Partner certification program
- [ ] Research collaboration framework

---

## ❓ FAQ

### Why not everything Apache/MIT?
We believe specialized research variants represent institutional IP that should benefit the commons while sustaining continued research.

### Why not everything AGPL?
General-purpose models should be freely usable in production without legal complexity.

### Can I use research variants commercially?
Yes, under AGPL terms (source code disclosure) or via commercial license.

### How do models graduate to stable?
Through generalization - combining best practices from variants into clean, documented implementations.

### Do research variants ever get deleted?
No, they remain permanently as a research archive and for specialized use cases.

---

## 📝 Summary

This strategy creates a sustainable open science model where:

1. **All research is transparent** (AGPL `/research`)
2. **Production use is simple** (Apache `/stable`)
3. **Innovation is rewarded** (specialized knowledge has value)
4. **Collaboration is natural** (clear contribution paths)
5. **Sustainability is built-in** (multiple support tiers)

The dual-licensing approach respects both academic principles and practical needs, creating a thriving ecosystem around Dawn Field Theory.

---

## 📞 Contact

**Quick Start:** [dawnfield.ca/licensing](https://dawnfield.ca/licensing) *(commercial inquiry form)*  
**Commercial Licensing:** info@dawnfield.ca  
**Research Collaboration:** research@dawnfield.ca  
**General Inquiries:** info@dawnfield.ca | See [`MISSION.md`](../MISSION.md)

### For Commercial Interest
1. **Using Apache models?** No action needed - build freely!
2. **Revenue <$500k?** You're explicitly exempt from licensing fees
3. **Need specialized variants?** Visit [dawnfield.ca/licensing](https://dawnfield.ca/licensing)
4. **Want professional support?** Email info@dawnfield.ca
5. **Research collaboration?** Email research@dawnfield.ca

---

*This document describes the Dawn Field Institute's approach to sustainable open science through strategic licensing. We welcome feedback and discussion.*
