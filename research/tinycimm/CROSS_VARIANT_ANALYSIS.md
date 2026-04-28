# TinyCIMM Cross-Variant Analysis: Genesis + Ghost

## Summary

| Variant | Role | Score | Headlines |
|---------|------|-------|-----------|
| **Genesis** | Self-organizer | 20/36 (56%) | phi^(-1/N) boundary confirmed, spacing NOT RMT, phi metastable (tau~1300) |
| **Ghost** | Constrained learner | 6/8 (75%) | Spectral confinement exact, beats Noether on power-law |

## What Generalizes (from M10 to Random Systems)

### phi^(-1/N) as Viability Boundary
- **M10**: SelfApplicator with specific initialization -> boundary at phi^(-1/N)
- **Genesis**: Random symmetric W -> **same boundary**, mean error 1.05%
- **N=32**: boundary = 0.9853, predicted = 0.9851, error = 0.03%
- **Conclusion**: This is a STRUCTURAL prediction. It comes from the constraints (symmetry + anti-Hebbian + tanh), not the initialization.

### gamma/ln(phi) = 1.1995 as Critical Spectral Radius
- **M10**: SelfApplicator sr fixed at 1.2 -> matches gamma/ln(phi) to 0.04%
- **Genesis**: Random W, scan sr -> critical sr = 1.1984 (N=32), error = 0.09%
- **Ghost**: Core sr holds at 1.1995 exactly throughout training
- **Two-parameter consistency**: At (sr=1.2, weak=phi^(-1/N)) and (weak=phi^(-1/N), sr~1.2), both boundaries agree
- **Conclusion**: gamma/ln(phi) is the minimum spectral radius for viability in anti-Hebbian symmetric systems.

### Spectral Confinement (PAC)
- **M10**: Eigenvector drift < 2.4e-15
- **Ghost**: Eigenvector drift < 1e-16 during training
- **Conclusion**: Symmetric W with eigenvalue-only modulation preserves eigenvectors EXACTLY. This is mathematically guaranteed (W = V D V^T reconstruction), not an empirical finding.

## What Does NOT Generalize

### Phi Enrichment in Eigenvalue Ratios
- **M10**: >15% phi enrichment in SelfApplicator spectrum
- **Genesis exp_03**: <1% phi enrichment with random W + SR normalization
- **Genesis exp_06**: Frobenius normalization preserves initial ratio structure but does not CREATE phi ratios. Random W enrichment is below GOE baseline.
- **Why**: Anti-Hebbian modulation is an EQUALIZER. It pushes all eigenvalue ratios toward 1.0. SR normalization accelerates this; Frobenius normalization slows it.
- **Conclusion**: Anti-Hebbian dynamics do not generate phi structure from random initial conditions. See below for the metastability finding.

### Xi as Transition Cost
- **M10**: Xi = gamma + ln(phi) = 1.058 per mode transition
- **Genesis**: Transition cost ~ 0.0002 nats (nearly zero)
- **Why**: With SR normalization, the eigenvalue spectrum changes minimally per step, so mode transitions are nearly costless.
- **Conclusion**: Xi may be specific to systems without SR normalization, or requires a different definition of "transition."

## The Key Discovery: Phi as Metastable State (Exp 08)

Genesis exp_08 tested whether phi-structured eigenvalue ratios are a STABLE FIXED POINT, an UNSTABLE POINT, or a METASTABLE STATE. Answer: **metastable**.

### Evidence

| Initial Ratio | Final Phi Enrichment | Mean Ratio at End | Interpretation |
|---------------|---------------------|-------------------|---------------|
| phi = 1.618 | **16.9%** | 1.103 | Slow decay, still phi-enriched at 3000 steps |
| e = 2.718 | 1.7% | 1.911 | Phi enrichment near zero |
| 2.0 | ~0% | 1.402 | No phi enrichment |
| random | ~0% | 1.027 | Converges to uniform |

Perturbation scan (phi-structured W, varying noise level):

| Perturbation | Final Enrichment |
|-------------|-----------------|
| 0% | 17.3% |
| 1% | 17.3% |
| 5% | 15.6% |
| 10% | 12.3% |
| 20% | 6.6% |
| 50% | 6.2% |

Monotonic degradation, but phi-structured W **never reaches zero enrichment** even at 50% perturbation.

### Interpretation (Updated by Exp 09)

1. **Truly metastable, NOT a weak attractor**: Exp_09 ran for 50,000 steps. Enrichment decays exponentially with tau ~1300 steps (half-life ~900). No plateau — the apparent stability at 17% in exp_08 (3000 steps) was just the slow tail of an exponential. By step 5000, enrichment reaches zero.

2. **Phi is specifically selected**: Among geometric ratios, phi decays slowest. e-structured and 2.0-structured controls reach zero much faster. But even phi reaches zero eventually.

3. **Mechanism is NOT activity equalization**: Exp_09 T3 disproved the proposed mechanism. All initializations converge to the same modulation rate (~15.5/16 modes per step). Phi-structured W doesn't produce fewer modulations — it just takes longer for the ratios to be pushed out of the phi window.

4. **PR(phi) = sqrt(5) is exact but irrelevant**: The participation ratio of a phi-geometric spectrum is exactly sqrt(5) — a beautiful algebraic identity. But anti-Hebbian dynamics erase the geometric structure, so the dynamic PR converges to ~1.3 regardless of initialization.

### Physical Analogy

Phi ratios are a **short-lived excited state**, not a metastable state with a high barrier. The decay timescale (~1300 steps) is finite and not particularly long. The SelfApplicator maintains phi structure because it IS the fixed point (W = f(W)), not because the dynamics preserve it. The universe doesn't evolve toward phi — it must BE the self-application fixed point.

### Implication for DFT

The universe may be at a phi-structured metastable state of self-referential dynamics. The question is not "why phi?" but "what selects the metastable state over the ground state?" Candidates:
- **MED (Maximum Entropy Design)**: phi-structured hierarchies maximize information capacity
- **Cosmological initial conditions**: the self-application fixed point was selected at the origin
- **Anthropic selection**: only phi-structured systems have enough hierarchical complexity for observers

## Planck Thread (Updated with Exp 07)

### Spacing is NOT Random Matrix Theory

Genesis exp_07 compared eigenvalue spacing floor to GOE (Gaussian Orthogonal Ensemble) predictions:

| System | Scaling Exponent | R^2 |
|--------|-----------------|-----|
| Raw GOE (no dynamics) | N^(-1.51) | 0.999 |
| Anti-Hebbian + SR normalization | N^(-2.45) | 0.999 |
| Anti-Hebbian + Frobenius normalization | N^(-1.91) | 0.994 |

Anti-Hebbian modulation creates a **new universality class** with scaling exponent ~2.45, distinct from the GOE exponent of 1.51. The spacing floor is compressed by an order of magnitude:

| N | GOE min spacing | SR-mod min spacing | Ratio (GOE/SR-mod) |
|---|----------------|-------------------|-------------------|
| 8 | 0.114 | 0.022 | 5.2x |
| 16 | 0.039 | 0.004 | 10.6x |
| 32 | 0.014 | 0.001 | 16.2x |

The compression ratio itself grows with N, which is why the exponent differs. Anti-Hebbian equalization pushes eigenvalues closer together, reducing minimum spacing much faster than random fluctuations.

### Why NOT RMT

GOE eigenvalue repulsion gives a universal spacing distribution (Wigner surmise). Anti-Hebbian dynamics override this:
- GOE: eigenvalues repel (spacing maintained by quantum-like repulsion)
- Anti-Hebbian: eigenvalues attracted toward equal magnitude (spacing compressed by equalization)
- Net effect: equalization wins, creating denser packing than GOE predicts

### Phi in Spacing?

No direct phi signature in spacing ratios (T4 FAIL). The min/mean spacing ratio is a small number (~0.01 to 0.07) that decreases with N, not near any phi-derived quantity.

### Connection to QG Proposal

The eigenvalue spacing floor scales as N^(-2.45) in self-organizing systems. If N modes correspond to degrees of freedom, the minimum resolution of eigenvalue structure is:

  delta_lambda_min ~ N^(-2.45)

For a system with N = 10^60 degrees of freedom (particle horizon), this gives an eigenvalue resolution of ~10^(-147), which is in the ballpark of the Planck scale / cosmological scale ratio. The exponent 2.45 is close to 5/2, which could connect to the 5-dimensional structure in M10's derivation chain. Speculative but worth tracking.

## Ghost's Contribution

### What Spectral Confinement Gives a Learning System
1. **Eigenvector fixity** -- guaranteed exact, no drift
2. **PAC conservation** -- 18x lower violations than SGD
3. **Structured latent space** -- fixed geometric transformation
4. **Power-law data advantage** -- Ghost MSE 0.268 vs Noether 0.273 vs SGD 0.352

### What It Costs
1. **Information bottleneck** -- tanh(W@h) at sr=1.2 saturates signal
2. **Fibonacci cascade failure** -- Ghost 2.2x worse than SGD (barely)
3. **Frozen core** -- core must be pre-organized, not learned jointly

### Design Lesson
The M10 constraints work for SELF-ORGANIZATION (Genesis) but create a bottleneck for DATA PROCESSING (Ghost). The solution is dual-mode: self-organize first (Genesis-style), then freeze and use as a structured transformation (Ghost-style).

## Three-Level Classification of DFT Predictions

The Genesis/Ghost investigation reveals three levels of DFT prediction robustness:

### Level 1: Topological (Universal)
- **What**: phi^(-1/N) viability boundary, gamma/ln(phi) critical sr
- **Why robust**: Depend on the TOPOLOGY of phase space (symmetry + modulation + nonlinearity), not on the trajectory
- **Analogy**: Critical exponents in stat mech — same for all systems in the universality class
- **Status**: Confirmed across random initializations and system sizes

### Level 2: Metastable (Specifically Selected)
- **What**: Phi-structured eigenvalue ratios
- **Why metastable**: Phi-spaced modes equalize more slowly than others under anti-Hebbian dynamics, creating a slow manifold
- **Analogy**: Supercooled water — not the ground state, but long-lived
- **Status**: Confirmed as metastable (exp_08), NOT an attractor (exp_06)
- **Open question**: What selects the metastable state? MED? Initial conditions? Anthropic?

### Level 3: Construction-Specific (Non-Universal)
- **What**: Xi per transition, cascade depth first-order gap
- **Why specific**: Depend on details of the SelfApplicator construction, not shared by generic symmetric systems
- **Analogy**: Crystal structure — depends on specific atoms and bonding, not just symmetry
- **Status**: Do not generalize (exp_03, exp_05). May require the full self-application fixed-point construction.

## Updated Score Card

| Exp | Score | Category |
|-----|-------|----------|
| **Genesis 01**: Viability boundary | 4/4 | Level 1 |
| **Genesis 02**: Spectral radius | 4/4 | Level 1 |
| **Genesis 03**: Phi in ratios | 0/4 | Level 2/3 |
| **Genesis 04**: Cascade depth | 2/4 | Level 3 |
| **Genesis 05**: Xi from dynamics | 1/4 | Level 3 |
| **Genesis 06**: Self-consistency attractor | 1/4 | Level 2 (null result: not attractor) |
| **Genesis 07**: RMT comparison | 3/4 | New universality class |
| **Genesis 08**: Phi basin of attraction | 4/4 | Level 2 (metastable) |
| **Genesis 09**: Metastability depth | 1/4 | Level 2 (decisive: no plateau) |
| **Ghost 01**: Spectral confinement | 4/4 | Level 1 |
| **Ghost 03**: vs Noether vs SGD | 2/4 | Applied |

**Total: 26/44 (59%)**

## Next Steps

1. **Reality Engine**: Take Level 1 predictions (boundaries) to the RE pipeline. Test whether Mobius operators show the same viability boundary.

2. **Ghost Phase B**: Replace tanh core with Mobius neurons. The 12,000x MLP advantage should eliminate the information bottleneck.

3. **Exponent 2.45**: Investigate whether the anti-Hebbian spacing exponent connects to dimensional structure in DFT. The closeness to 5/2 is suggestive but could be coincidence.

4. **MED as selection principle**: The metastability finding (exp_09) means phi is NOT dynamically selected. But it might be INFORMATIONALLY selected — if phi-structured fixed points maximize information capacity, MED selects them. This is a mathematical question, not a numerical one.
