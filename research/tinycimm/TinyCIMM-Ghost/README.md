# TinyCIMM-Ghost: Constrained Learner

Encoder-Core-Decoder architecture where the core is a symmetric recurrent system with M10 spectral confinement. The encoder/decoder learn via PAC-scaled gradient updates. The core is FROZEN during training — it provides a fixed, spectrally-structured transformation.

## Core Question

Noether's exp_04 fails: SGD beats PAC descent on physics-structured data. Can M10's spectral confinement fix this?

Answer: **partially.** Ghost beats Noether on power-law data (Fibonacci exponents) but loses on Fibonacci cascade (core's tanh bottleneck at sr=1.2 hurts). Ghost has 18x lower PAC violations than SGD.

## Architecture

```
Input x [batch, input_dim]
  → ENCODER: W_enc [core_dim, input_dim], tanh (PAC-scaled gradient)
  → CORE: K=1 step of tanh(W_core @ h), W_core symmetric, sr=1.1995
          Eigenvectors FIXED (drift < 1e-16). Core is FROZEN during training.
  → DECODER: W_dec [output_dim, core_dim], linear (gradient)
  → Output y [batch, output_dim]
```

**Key parameters:**
- `core_dim` = 13 (Fibonacci number)
- `sr` = gamma/ln(phi) = 1.1995 (from M10/Genesis)
- `weak_factor` = phi^(-1/core_dim) (from M10/Genesis)
- Core is frozen during training — only encoder/decoder learn

## Block A Results

| Exp | Score | Finding |
|-----|-------|---------|
| 01: Spectral confinement | **4/4** | Eigenvector drift < 1e-16, SR stable at 1.1995 |
| 03: Ghost vs Noether vs SGD | **2/4** | Ghost beats Noether on power-law, fails on Fibonacci cascade |

### Headline Results

1. **Spectral confinement is EXACT** (exp_01): Eigenvector drift < 1e-16 throughout training. SR stays at gamma/ln(phi) = 1.1995 exactly. Asymmetric negative control shows 0.745 drift.

2. **Ghost beats Noether on power-law** (exp_03 T1): MSE 0.268 vs 0.273 — the spectral core helps when data has Fibonacci-exponent structure. Both beat SGD (0.352).

3. **PAC violations 18x lower** (exp_03 T4): Ghost 0.243 vs SGD 3.107. Conservation enforcement works.

### Failures

- **Fibonacci cascade** (exp_03 T2): Ghost MSE 1.14 vs Noether 0.53. The core's tanh(sr=1.2 @ h) amplifies and saturates, losing information. The core provides spectral structure but at the cost of an information bottleneck.

### Key Insight

M10's spectral confinement works as a GEOMETRIC constraint (eigenvectors fixed, PAC values conserved) but the associated dynamics (anti-Hebbian modulation, sr=1.2) are designed for self-organization, not data processing. For learning, the core should be a frozen structured transformation, not an active self-organizer.

## Files

| File | Description |
|------|-------------|
| `ghost_core.py` | SymmetricRecurrentCore with spectral confinement |
| `ghost_network.py` | GhostNetwork + SGD baseline |
| `spectral_utils.py` | Eigenvalue analysis, DFT constants |
| `experiments/block_a/exp_01_*.py` | Spectral confinement during learning |
| `experiments/block_a/exp_03_*.py` | Ghost vs Noether vs SGD comparison |

## Dependencies

- Python 3.8+
- NumPy

No deep learning frameworks. Pure Python + NumPy.
