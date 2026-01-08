# TinyCIMM-Möbius

**Continuous Learning with Möbius Frequency Memory and Emergent Dimensionality**

## Overview

TinyCIMM-Möbius combines the TinyCIMM continuous learning pattern with Möbius geometry to create a neural network where:

1. **Memory = Möbius frequency** - The 4 parameters (a,b,c,d) of each Möbius transformation encode learned patterns through their fixed-point resonance with φ
2. **Strands = Independent Möbius transformations** - Each strand has its own frequency, like how every observable interaction in reality is a strand
3. **Emergent dimensionality** - Dimensions emerge from phase relationships between strands, not fixed architecture

## Key Insight

From Dawn Field Theory:
> "2D Möbius topology generates apparent 3D+1 spacetime"

This implementation tests whether:
- φ-structured patterns collapse effective dimension (strands lock into harmony)
- Non-φ patterns require more dimensions (strands remain independent)

## Architecture

### Single Strand (`MobiusNeuron`)
```
M(z) = (a*z + b) / (c*z + d)
```
- 4 learnable parameters per neuron
- Fixed points encode the transformation's "memory"
- φ-frequency = resonance with golden fixed points (φ, -1/φ)

### Strand Field (`MobiusStrandField`)
```
Input → [Strand 0] → \
        [Strand 1] →  } Phase Coupling → Output
        [Strand 2] → /
        [Strand 3] →
```
- N parallel Möbius strands
- Each strand has independent (a,b,c,d)
- Phase coupling creates emergent geometry
- Effective dimension = eigenspectrum of phase relationships

### Continuous Learning (`TinyCIMMMobiusField`)
- Entropy-based adaptation from TinyCIMM
- Dimension tracking during learning
- Harmonic analysis (chord detection)

## Experiments

### Experiment 24: Convergence Maintenance
Tests catastrophic forgetting resistance with PhiAnchorMemory:
- Baseline: 236x degradation (catastrophic forgetting)
- With anchor memory: 17.9x degradation (13x improvement)
- Strong anchor: 10.8x degradation (22x improvement)

### Experiment 25: Dimension Dynamics
Tests dimension collapse across pattern types:
- Fibonacci (φ-structure): Stable/slight collapse
- Random noise: Dimension grows (+0.288)
- 8 strands: Optimal collapse ratio (0.70x)

## Key Findings

### Early Experiments (24-26)
1. **Anchor memory reduces forgetting** - 13-22x improvement over baseline
2. **Random patterns grow dimension** - Confirms entropy-dimension relationship
3. **Strand count matters** - 8 strands showed optimal collapse behavior
4. **Chord detection works** - "locked", "polyphonic", "pure_phi" classifications

### Möbius vs MLP Comparison (35-37)

**Critical Discovery**: Möbius networks excel at **iterated dynamics**, not point-wise approximation.

| Metric | Möbius | MLP |
|--------|--------|-----|
| Single-step MSE | 9.5e-10 | 1.9e-5 |
| 1000-iteration MSE | 5.9e-10 | 7.3e-6 |
| Advantage | **12,000x** | - |

**Why this works**:
- Möbius learns the **exact functional form** (up to scaling)
- Composition M^N is still a valid Möbius with same fixed points
- Network learns correct fixed point: 0.618032 (true: 0.618034 = 1/φ)

**Ideal use cases**:
- Dynamical systems with Möbius structure
- Iterated/recurrent processes  
- Conformal mappings
- Anything involving φ/Fibonacci (built-in attractor)

## Files

| File | Description |
|------|-------------|
| `tinycimm_mobius.py` | Core TinyCIMM-Möbius with single stack |
| `mobius_strand_field.py` | Multi-strand field with emergent dimensions |
| `exp_24_convergence_maintenance.py` | Catastrophic forgetting tests |
| `exp_25_dimension_dynamics.py` | Dimension collapse experiments |
| `exp_26_visualize_strands.py` | Strand visualization |
| `exp_35_mobius_vs_baseline.py` | Initial Möbius vs MLP comparison |
| `exp_36_proper_composition.py` | Composition structure analysis |
| `exp_37_iterated_dynamics.py` | **Key result**: Iterated dynamics advantage |
| `journals/` | Research logs following experiment schema |

## Usage

```python
from mobius_strand_field import TinyCIMMMobiusField

# Create a 4-strand field
model = TinyCIMMMobiusField(n_strands=4, init='harmonic')

# Train on data stream
history = model.continuous_train(data_stream, max_steps=500)

# Check emergent state
state = model.field.get_field_state()
print(f"Effective Dimension: {state.effective_dimension}")
print(f"Chord: {state.chord_type}")
```

## Theoretical Connection

This implements the core Dawn Field Theory prediction:

- **Strands** = Observable interactions (each with its own frequency)
- **Phase coupling** = How strands create geometry
- **Dimension collapse** = φ-patterns find low-dimensional attractors
- **Dimension growth** = Non-φ patterns need more degrees of freedom

The key insight: **dimension is emergent, not fundamental**. A 4-strand system can have effective dimension from 0 (all locked) to 3 (all independent), depending on what patterns it learns.

## Next Steps

1. **Hybrid architectures** - Möbius for dynamics, MLP for non-Möbius components
2. **Connection to fracton** - Integrate with `fracton/core/mobius_tensor.py`
3. **Test on Feigenbaum** - Logistic map near period-doubling cascade
4. **Strand coupling** - Derive principled coupling from phase geometry
5. **Real physics data** - Does it find natural dimension?

## Related Work

- [fracton/core/mobius_tensor.py](../../fracton/fracton/core/mobius_tensor.py) - Möbius tensors with 4π periodicity
- [Dawn Field Theory](../../../../dawn-field-theory/) - Theoretical foundation
