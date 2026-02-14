# TinyCIMM-Boltzmann

**PAC-Conserved Multi-Head Learning: Hallucination as Conservation Violation**

## Overview

TinyCIMM-Boltzmann is the first neural architecture where **PAC conservation is an architectural constraint**, not just an observation. The model enforces that the total entropy budget across attention heads remains constant during learning.

Named after **Ludwig Boltzmann** — father of statistical mechanics and the entropy equation $S = k \ln W$ — because the core finding is literally about entropy conservation in information-processing systems.

## Motivation

From `exp_12_pac_conservation` (token_pac_tree series):

| Model | PAC Violation | Compensation Ratio |
|-------|---------------|-------------------|
| pythia-160m | +9.9% | 0.028 |
| pythia-410m | +4.2% | 0.174 |
| gpt2 | +10.0% | **0.000** |
| gpt2-medium | +14.4% | **0.000** |

**Finding**: During hallucination, LLMs create +9.6% uncompensated entropy across all heads. GPT-2 models show literally zero compensation — every layer gains entropy simultaneously. Nothing counterbalances.

**Question**: If we make PAC violation architecturally impossible, does the model naturally avoid hallucination-analogue behavior?

## Architecture

### BoltzmannHead
Single processing head with tracked activation entropy — analogous to transformer attention heads but with entropy monitoring built in.

### BoltzmannLayer
N parallel heads with a shared entropy budget. The `ConservationProjector` ensures that when one head increases entropy (explores), another must decrease (crystallize).

### ConservationProjector
Two modes:
- **Soft**: Adds a conservation loss term (penalty for budget violation)
- **Hard**: Explicitly normalizes head outputs to enforce the budget

### TinyCIMMBoltzmann
Full model with continuous learning, real-time conservation tracking, and SEC phase classification across all heads.

```
Input → BoltzmannLayer 1 [H1|H2|H3|H4] → BoltzmannLayer 2 [H1|H2|H3|H4] → Output
              ↕ ConservationProjector            ↕ ConservationProjector
         (entropy budget = constant)         (entropy budget = constant)
```

## Key Concepts

### PAC Conservation as Constraint
In standard transformers, PAC doesn't hold — compensation ratio ≈ 0 during hallucination. TinyCIMM-Boltzmann forces compensation ratio → 1.0 by penalizing any net change in total entropy.

### SEC Phase Tracking
Each head is classified into SEC phases (zero-parameter, theory-derived):
- **Crystallized** (H < 0.5): Locked pattern
- **Ordered** (0.5 ≤ H < 2.0): Structured processing
- **Transitional** (2.0 ≤ H < 4.0): Phase boundary
- **Chaotic** (H ≥ 4.0): Exploratory

### Conservation Monitor
Real-time tracking of:
- Total entropy budget (should be constant)
- Per-head phase distribution
- Compensation dynamics (cross-head entropy flow)
- Violation trend (growing or self-correcting?)

## Connection to Dawn Field Theory

| DFT Concept | TinyCIMM-Boltzmann Implementation |
|-------------|----------------------------------|
| PAC: f(Parent) = Σf(Children) | Total head entropy = constant (budget) |
| SEC: ∂S/∂t = α∇I - β∇H | Phase classification per head |
| Hallucination = PAC violation | Conservation loss detects and prevents it |
| Compensation ratio | Cross-head entropy flow tracking |

## Usage

```python
from tinycimm_boltzmann import TinyCIMMBoltzmann, create_mixed_stream

model = TinyCIMMBoltzmann(
    input_size=1, hidden_size=32, output_size=1,
    n_heads=4, n_layers=2,
    conservation_mode='soft',      # 'soft', 'hard', or 'none'
    conservation_strength=1.0,     # How strongly to enforce PAC
)

stream = create_mixed_stream(n_factual=300, n_halluc=200)
history = model.continuous_train(stream, max_steps=500, log_interval=50)

summary = model.get_conservation_summary()
print(f"Violation: {summary['total_violation_pct']:+.1f}%")
print(f"Stability: {summary['budget_stability']:.3f}")
```

## Experiments

| Experiment | Status | Description |
|-----------|--------|-------------|
| exp_01_conservation_vs_free | ✅ | Conserved vs unconstrained under factual/noise — 3/4 significant |

## Key Questions

1. Does conservation enforcement reduce hallucination-analogue behavior?
2. Does the constrained model learn factual patterns as well as the unconstrained?
3. Is there a conservation_strength sweet spot (too little = violation, too much = can't learn)?
4. Does the compensation ratio during noise distinguish conserved from free models?
