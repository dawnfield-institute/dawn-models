# Contributing to Dawn Models

Welcome to the Dawn Models project! We're excited about your interest in contributing to our dual-licensed model repository that implements Dawn Field Theory principles.

## 📋 General Contribution Guidelines

**For comprehensive contribution guidelines, please see the main Dawn Field Theory repository:**

**👉 [Dawn Field Theory Contributing Guide](https://github.com/dawnfield-institute/dawn-field-theory/blob/main/CONTRIBUTION.md)**

This covers:
- Registration process (info@dawnfield.ca)
- Engagement philosophy and boundaries
- Citation and attribution guidelines
- Community standards and code of conduct
- Publishing boundaries for research work

---

## 🎯 Dawn Models Specific Guidelines

### Dual-License Contribution Paths

#### 🔬 Research Contributions (AGPL-3.0)
*Contributions to `/research` folder*
- New experimental model variants
- Theoretical implementations
- Research notebooks and explorations
- Ablation studies and architectural experiments

#### 🚀 Stable Model Contributions (Apache-2.0)
*Contributions to `/stable` folder*
- Bug fixes in production models
- Performance improvements
- Documentation enhancements
- General-purpose implementations

#### 📚 Documentation & Infrastructure
*Any folder*
- README improvements
- Tutorial development
- Testing frameworks
- CI/CD enhancements

---

## 📋 Before You Start

### Prerequisites
- Understanding of Dawn Field Theory principles (see [dawn-field-theory](https://github.com/dawnfield-institute/dawn-field-theory))
- Familiarity with our dual-licensing strategy (see [LICENSING.md](LICENSING.md))
- Python development environment
- Git workflow knowledge

### Required Reading
1. **[Dawn Field Theory Overview](https://github.com/dawnfield-institute/dawn-field-theory/blob/main/README.md)**
2. **[Licensing Strategy](LICENSING.md)** - Understand which license applies to your contribution
3. **[Code of Conduct](CODE_OF_CONDUCT.md)** - Our community standards

---

## 🔄 Contribution Workflow

### 1. Fork & Clone
```bash
# Fork the repository on GitHub
git clone https://github.com/YOUR_USERNAME/dawn-models.git
cd dawn-models
git remote add upstream https://github.com/dawnfield-institute/dawn-models.git
```

## 🚀 Getting Started with Dawn Models

### 1. Follow General Guidelines
First, read and follow the main [Dawn Field Theory Contributing Guide](https://github.com/dawnfield-institute/dawn-field-theory/blob/main/CONTRIBUTION.md), including:
- Contributor registration (info@dawnfield.ca)
- Understanding project boundaries
- Citation guidelines for substantial contributions

### 2. Choose Your License Path
Dawn Models uses a dual-licensing strategy:
- **Research models** (`/research`) → AGPL-3.0
- **Stable models** (`/stable`) → Apache-2.0

### 3. Understand the Model Lifecycle
- **Research variants never graduate** - they remain permanently in `/research`
- **Stable models** are new, general-purpose implementations in `/stable`
- Both paths accept contributions with different quality standards

---
```bash
# For research contributions
git checkout -b research/your-feature-name

# For stable contributions
git checkout -b stable/your-feature-name

# For documentation
git checkout -b docs/your-improvement
```

### 3. Make Your Changes
Follow the specific guidelines for your contribution type below.

### 4. Test Your Changes
```bash
# Run tests for the area you're modifying
cd research/tinycimm  # or stable/cimm-legacy
python -m pytest tests/

# Validate imports
python validate_imports.py
```

### 5. Submit a Pull Request
- Clear title describing the change
- Detailed description of what and why
- Reference any related issues
- Include test results

---

## 🔬 Research Contributions

### Experimental Model Variants
Research contributions go to `/research` and are licensed under AGPL-3.0.

#### Structure for New Variants
```
research/
├── your-model-family/
│   ├── variant-name/
│   │   ├── __init__.py
│   │   ├── model.py          # Main implementation
│   │   ├── experiments.py    # Experimental code
│   │   ├── requirements.txt  # Dependencies
│   │   └── README.md        # Variant documentation
│   └── meta.yaml           # Family metadata
```

#### Naming Conventions
- **Model families**: `tinycimm`, `theoretical`, `gaia-experiments`
- **Variants**: `attention-v3`, `memory-recursive`, `bifractal-enhanced`
- **Files**: Descriptive, lowercase with underscores

#### Required Documentation
Each new variant must include:
```markdown
# Variant Name

## Overview
Brief description of the experimental approach

## Dawn Field Theory Connection
How this variant implements or tests DFT principles

## Key Innovations
- Novel architectural features
- Theoretical contributions
- Performance characteristics

## Usage
```python
from research.family.variant import Model
model = Model()
result = model.process(data)
```

## Experiments
Description of experiments conducted

## Results
Key findings and performance metrics

## Future Work
Potential improvements and research directions
```

### Research Standards
1. **Theoretical Grounding**: Connect to Dawn Field Theory principles
2. **Experimental Rigor**: Include proper controls and baselines
3. **Reproducibility**: Provide clear setup and execution instructions
4. **Documentation**: Explain both what and why

---

## 🚀 Stable Model Contributions

### Production-Ready Models
Stable contributions go to `/stable` and are licensed under Apache-2.0.

#### Quality Standards
- **Stability**: Thoroughly tested, no experimental features
- **Documentation**: Complete API documentation
- **Performance**: Benchmarked and optimized
- **Compatibility**: Semantic versioning, backwards compatibility

#### Contribution Types
1. **Bug Fixes**: Critical for maintaining production quality
2. **Performance Improvements**: Optimization without breaking changes
3. **Feature Additions**: Must maintain API stability
4. **Documentation**: Always welcome

#### Testing Requirements
```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/

# Performance benchmarks
python benchmarks/run_benchmarks.py

# Memory profiling
python -m memory_profiler examples/basic_usage.py
```

---

## 📊 Code Quality Standards

### Python Style
- **PEP 8** compliance with 88-character line length
- **Type hints** for all public APIs
- **Docstrings** in Google style for all classes and functions
- **Error handling** with informative error messages

### Example Function
```python
def process_symbolic_collapse(
    input_data: torch.Tensor,
    collapse_threshold: float = 0.1,
    preserve_entropy: bool = True
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Process input through symbolic entropy collapse.
    
    Implements the Dawn Field Theory principle of controlled
    information collapse while preserving essential structure.
    
    Args:
        input_data: Input tensor to process
        collapse_threshold: Minimum entropy for collapse trigger
        preserve_entropy: Whether to maintain entropy conservation
        
    Returns:
        Tuple of (processed_tensor, metrics_dict)
        
    Raises:
        ValueError: If input_data is not 2D tensor
        RuntimeError: If collapse fails to converge
    """
    if input_data.dim() != 2:
        raise ValueError("Input must be 2D tensor")
        
    # Implementation here...
    
    return processed_tensor, {"entropy_preserved": 0.95}
```

### Dependencies
- **Minimize dependencies**: Only add what's essential
- **Pin versions**: Specify exact versions in requirements.txt
- **Document rationale**: Explain why each dependency is needed

---

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── unit/                    # Fast, isolated tests
│   ├── test_model_core.py
│   └── test_utils.py
├── integration/             # Full workflow tests
│   ├── test_training.py
│   └── test_inference.py
├── performance/            # Benchmark tests
│   └── test_speed.py
└── fixtures/               # Test data
    └── sample_data.json
```

### Test Quality
- **Coverage**: Aim for >90% code coverage
- **Edge cases**: Test boundary conditions
- **Error conditions**: Verify proper error handling
- **Performance**: Include regression tests for critical paths

---

## 📚 Documentation Standards

### README Requirements
Every module/variant needs a clear README with:
- **Purpose**: What does this do?
- **Theory**: How does it connect to Dawn Field Theory?
- **Usage**: Clear examples with expected outputs
- **Dependencies**: What's required to run it?
- **Performance**: Benchmarks and computational requirements

### Code Documentation
- **Docstrings**: Google style for all public APIs
- **Comments**: Explain the "why" not the "what"
- **Type hints**: Full typing for better IDE support
- **Examples**: Include usage examples in docstrings

---

## 🔄 Review Process

### Research Contributions
1. **Technical Review**: Code quality and experimental rigor
2. **Theoretical Review**: Connection to Dawn Field Theory
3. **Documentation Review**: Clarity and completeness
4. **Community Feedback**: Public review period

### Stable Contributions
1. **Compatibility Review**: Backwards compatibility check
2. **Performance Review**: No regressions
3. **Security Review**: Vulnerability assessment
4. **API Review**: Public interface stability

### Timeline
- **Initial Review**: Within 48 hours
- **Technical Review**: 3-5 days for research, 1-2 days for stable
- **Community Feedback**: 1 week for major changes
- **Final Decision**: Within 2 weeks

---

## 🎖️ Recognition

### Contributor Credits
- All contributors listed in project documentation
- Significant contributors invited to co-author papers
- Research contributions eligible for academic citation

### Types of Recognition
1. **Code Contributors**: Listed in README and releases
2. **Research Contributors**: Co-authorship opportunities
3. **Documentation Contributors**: Special recognition section
4. **Community Leaders**: Invited to advisory positions

---

## 📞 Getting Help

### Where to Ask Questions
1. **GitHub Issues**: Bug reports and feature requests
2. **GitHub Discussions**: General questions and ideas
3. **Email**: research@dawnfield.ca for research collaboration
4. **Discord**: [Link] for real-time community chat

### Mentorship Program
New contributors can request mentorship for:
- Understanding Dawn Field Theory
- Navigating the codebase
- Research methodology
- Publication opportunities

---

## 🚀 First Contribution Ideas

### Good First Issues
- **Documentation improvements**: Fix typos, clarify examples
- **Test additions**: Increase coverage for existing code
- **Performance benchmarks**: Add timing tests
- **Example notebooks**: Create tutorial content

### Research Projects
- **Ablation studies**: Test specific architectural components
- **Parameter sensitivity**: Explore hyperparameter spaces
- **Theoretical implementations**: Convert math to code
- **Visualization tools**: Create interpretability aids

---

## 📋 Checklist Before Submitting

### All Contributions
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No sensitive information included

### Research Contributions
- [ ] Theoretical foundation documented
- [ ] Experimental methodology described
- [ ] Results properly analyzed
- [ ] Connection to Dawn Field Theory explained
- [ ] AGPL-3.0 license implications understood

### Stable Contributions
- [ ] Backwards compatibility maintained
- [ ] Performance regressions tested
- [ ] API documentation complete
- [ ] Apache-2.0 license implications understood

---

## 📜 License Agreement

By contributing to Dawn Models, you agree that your contributions will be licensed according to the dual-licensing strategy outlined in [LICENSING.md](./LICENSING.md):

1. **Research contributions** (`/research`) will be licensed under AGPL-3.0
2. **Stable contributions** (`/stable`) will be licensed under Apache-2.0
3. You have the right to submit the work under these licenses
4. You understand the implications of the dual-licensing strategy

For detailed contribution policies, citation guidelines, and community standards, see the main [Dawn Field Theory Contributing Guide](https://github.com/dawnfield-institute/dawn-field-theory/blob/main/CONTRIBUTION.md).

---

## 🙏 Thank You

We appreciate your interest in advancing Dawn Field Theory through practical model implementations. Your contributions help bridge the gap between cutting-edge research and real-world applications.

---

*For questions about this contribution guide, please open an issue or contact info@dawnfield.ca*
