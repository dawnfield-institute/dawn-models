# SCBF: Symbolic Collapse Bifractal Framework

Dawn Field Theory's interpretability framework for measuring symbolic collapse and
bifractal patterns in neural network learning dynamics.

## Purpose

- Interpretability framework measuring symbolic collapse and bifractal patterns
- Deep mathematical structure analysis during neural network learning
- Entropy-based insights into emergent symbolic representations
- Integration with Dawn Field Theory's core models (TinyCIMM, GAIA, CIMM)
- Visual and quantitative analysis tools; protocol-driven, reproducible experiments

## Contents

### v1 (snapshot-scale instrumentation — stable)
- `metrics/` – symbolic collapse, entropy, ancestry, lineage, attractor metrics
- `loggers/` – experiment tracking (collapse events, lineage, entropy)
- `visualization/` – collapse heatmaps, PCA/t-SNE overlays, dashboards
- `scbf_runner.py` – experiment registration and runner
- `scbf_experiments/`, `experiments/` – protocols and concrete experiments
- `tinycimm_scbf_experiment.py` – TinyCIMM integration
- `test_scbf.py`, `utils/`, `vcpu/`, `docs/`

### v2 (continuous interpretability — 2026-07, `v2/`)

Interpretability for models that never stop changing, built from the validated
instrument set of the Ember III program (10 pre-registered rounds, 2026-07-18/20;
record in `../ember3_scale160m/` and
`../tinycimm/TinyCIMM-Euler/experiments/ember3_drift/`). Spec:
`.spec/scbf-v2.spec.md`. v1 is untouched; import `scbf.v2` explicitly.

- `v2/hookspine.py` – **HookSpine**: gradient-hook instrumentation core. Fused
  per-parameter updates inside the backward pass (7B MoE continuous learning in
  15.3GB VRAM), placement masks (= lesion configurations), per-group and
  stacked-tensor per-slice update-mass telemetry. Bounded memory; loud NaN failure.
- `v2/probes.py` – CKA drift (fixed probes, dimension-agnostic), held-out CE
  splits, **ClozeKnowledgeProbe** (generation-free fact scoring), plasticity probe.
- `v2/battery.py` – lm-eval parsing + the locked 2×SE reading rule.
- `v2/lesion.py` – **gradient-pressure lesioning**: constrain where update pressure
  lands, measure what dies → causal capability localization. First results: MoE
  factual knowledge localized to expert FFNs (−10.4 EM under expert-confined
  pressure, invisible to CE); refined by its own control to the **compensation
  law** — uncompensated updates damage, compensated updates don't (identical
  update mass, opposite outcomes; the protection is co-adaptation freedom).
- `v2/gates.py` – update-admission policies (err / excess / band / rand) with
  measured standings from the Ember III gate rounds.
- `v2/telemetry.py` – RunLog CSV/JSON schemas.
- `v2/tests/` – CPU-only unit tests (11).

Quick start (v2):

```python
from scbf.v2 import HookSpine, olmoe_experts_only, olmoe_expert_stacked_fn

spine = HookSpine(model,
                  trainable=olmoe_experts_only(),     # placement mask / lesion
                  lr=5e-4, fused=True,                # update during backprop
                  stacked_fn=olmoe_expert_stacked_fn) # per-expert mass telemetry
out = model(x, labels=x)
out.loss.backward()   # updates applied + telemetry accumulated + grads freed
print(spine.telemetry()["stacked_mass"])
```

Start here to run, extend, or analyze symbolic collapse and continuous-learning
interpretability experiments.
