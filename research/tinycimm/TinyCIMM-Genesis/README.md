# TinyCIMM-Genesis: Self-Organizing Dynamical System

NOT a neural network. A self-referential dynamical system with measurement instruments. No data, no loss function, no target. Only structural constraints: symmetric W + anti-Hebbian eigenvalue modulation. Initialize randomly, run, measure what constants emerge from dynamics alone.

## Core Question

M10 (Symmetry Self-Application) showed that a specific toy model — the SelfApplicator — has viability boundaries at phi^(-1/N) and spectral radius gamma/ln(phi). But is this because of the specific initialization, or because of the STRUCTURE?

Genesis answers: **it's the structure.** Random symmetric matrices with the same dynamics show identical boundaries.

## Architecture

```python
GenesisSystem(N, seed):
    W = random symmetric (N×N), initial sr = 1.2
    state = randn(N) * 0.5

step():
    state = tanh(W @ state)
    eigvals, eigvecs = eigh(W)
    activities = (eigvecs.T @ state)^2, normalized
    modulation: active > 2×mean → ×weak_factor
                inactive < 0.5×mean → ×strong_factor
    W = eigvecs @ diag(eigvals * modulation) @ eigvecs.T
    renormalize sr to target
```

## Block A Results (20/36)

### Generalization Tests (exp_01-05): 11/20

| Exp | Score | Finding |
|-----|-------|---------|
| 01: Viability boundary | **4/4** | Boundary at phi^(-1/N), mean error 1.05%. N=32: 0.03% |
| 02: Spectral radius | **4/4** | Critical sr = gamma/ln(phi), mean error 2.60%. N=32: 0.09% |
| 03: Phi in ratios | **0/4** | Does NOT generalize — SR normalization equalizes spectrum |
| 04: Cascade depth | **2/4** | Eigenvalue spacing scales ~1/N^2 (Planck thread) |
| 05: Xi from dynamics | **1/4** | Mode transitions occur but nearly costless |

### Failure Investigation (exp_06-09): 9/16

| Exp | Score | Finding |
|-----|-------|---------|
| 06: Self-consistency attractor | **1/4** | Anti-Hebbian is an equalizer, not a phi-generator |
| 07: RMT spacing comparison | **3/4** | New universality class: N^(-2.45) vs GOE N^(-1.51) |
| 08: Phi basin of attraction | **4/4** | Phi-structured W persists >3000 steps; others decay to 0 |
| 09: Metastability depth | **1/4** | Decay tau ~1300 steps, no plateau. PR(phi) = sqrt(5) exact. |

### Headline Results

1. **phi^(-1/N) viability boundary** (exp_01): The alive/dead transition for anti-Hebbian modulation in ANY random symmetric system occurs at weak_factor = phi^(-1/N). Sharp first-order transition (width 0.005). Finite-size correction vanishes at large N.

2. **gamma/ln(phi) critical sr** (exp_02): The minimum spectral radius for viability at the phi^(-1/N) modulation rate is gamma/ln(phi) = 1.1995. Two-parameter consistency confirmed: both directions agree.

3. **New universality class** (exp_07): Anti-Hebbian modulation produces eigenvalue spacing that scales as N^(-2.45), distinct from GOE random matrix theory at N^(-1.51). R^2 > 0.99. Self-organizing symmetric systems form their own universality class.

4. **Phi is metastable, not an attractor** (exp_08, exp_09): Phi-structured eigenvalue ratios persist longer than other geometric structures (e, 2.0, random) under anti-Hebbian dynamics, but eventually decay to zero (tau ~1300 steps). The SelfApplicator's phi structure comes from the self-application construction, not from dynamics converging to phi.

5. **PR(phi) = sqrt(5)** (exp_09): The participation ratio of a phi-geometric eigenvalue spectrum is exactly sqrt(5). Algebraically beautiful but dynamically irrelevant — anti-Hebbian erases the geometric structure.

### Three-Level Classification

| Level | What | Examples | Status |
|-------|------|----------|--------|
| 1. Topological (universal) | Phase boundaries | phi^(-1/N), gamma/ln(phi) | Confirmed |
| 2. Metastable (finite lifetime) | Equilibrium ratios | Phi in eigenvalue ratios (tau ~1300) | Confirmed metastable |
| 3. Construction-specific | Fixed-point properties | Xi per transition, cascade depth gap | Not generalizable |

## Files

| File | Description |
|------|-------------|
| `genesis_system.py` | GenesisSystem class with meta-modulation |
| `spectral_utils.py` | Eigenvalue analysis, DFT constants, hierarchy measures |
| `genesis_measures.py` | Measurement instruments for phi, Xi, cascade depth |
| `experiments/block_a/exp_01_*.py` | Viability boundary scan |
| `experiments/block_a/exp_02_*.py` | Spectral radius viability |
| `experiments/block_a/exp_03_*.py` | Phi in eigenvalue ratios |
| `experiments/block_a/exp_04_*.py` | Cascade depth floor |
| `experiments/block_a/exp_05_*.py` | Xi from dynamics |
| `experiments/block_a/exp_06_*.py` | Self-consistency attractor (Frobenius vs SR) |
| `experiments/block_a/exp_07_*.py` | RMT spacing comparison |
| `experiments/block_a/exp_08_*.py` | Phi basin of attraction |
| `experiments/block_a/exp_09_*.py` | Metastability depth (50K steps) |

## Dependencies

- Python 3.8+
- NumPy, SciPy

No deep learning frameworks. Pure Python + NumPy.
