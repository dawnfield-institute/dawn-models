# SCBF v2 — Continuous Interpretability (spec)

**Status:** v0 draft, 2026-07-20. Companion to SCBF v1 (untouched). Origin: Ember III
rounds 1–9 (`dawn-models/research/ember3_scale160m/`, `.../ember3_drift/`).

## Purpose

Interpretability for models that never stop changing. v1 measures snapshots of small
models via explicit tracker calls; v2 instruments *live training* of real LMs (7B
demonstrated) through a gradient-hook spine, probe batteries, and placement-
constrained update protocols (gradient-pressure lesioning).

## Contract

### HookSpine (`scbf.v2.hookspine`)
- Attaches post-accumulate-grad hooks to a torch model's trainable parameters.
- **Placement mask**: `trainable(name) -> bool` decides which parameters may update
  — a lesion configuration is just a mask.
- **Fused mode** (`fused=True`): per-parameter clip → update (SGD, pluggable) →
  `grad = None` immediately. Peak gradient memory ≈ one layer. This is the mode that
  ran a 7B MoE continuous learner in 15.3GB VRAM (round 8).
- **Observe mode** (`fused=False`): accumulate statistics, leave grads for an
  external optimizer.
- **Group telemetry**: `group_fn(name) -> key|None` (scalar update-mass per key) and
  `stacked_fn(name) -> (key, axis)|None` (per-slice mass along an axis — the
  stacked-expert-tensor fix from round 8's instrument bug).
- **Bounded memory guarantee**: all accumulators are fixed-size (floats / fixed
  arrays); no per-step lists (v1 port-audit lesson).
- **Loud failure**: non-finite gradient norms raise immediately (NaN silently
  corrupting downstream logic is a documented program failure mode).

### Probes (`scbf.v2.probes`)
Side-effect-free measurement callables returning flat dicts (v1 metric convention):
- `linear_cka(H1, H2)`; `CKADriftProbe` — fixed probe batch, frozen reference,
  dimension-agnostic; single-run drift documented as jitter-sensitive at scale.
- `HeldoutCEProbe` — pure CE on fixed batches (domain / off-domain forgetting split).
- `ClozeKnowledgeProbe` — generation-free knowledge scoring: per-fact answer
  log-likelihood + first-token top-1 under teacher forcing. Separates
  knowledge-erosion from output-format shift (round 9 follow-up).
- `PlasticityProbe` — fixed-segment clone adaptation (small models only; deepcopy).

### Battery (`scbf.v2.battery`)
- Parse lm-eval output dirs; apply the locked reading rule
  (no-change iff |Δ| ≤ 2·√(se₁²+se₂²)); emit verdict tables. Pre-registration of
  the rule BEFORE runs is protocol, not code.

### Lesion (`scbf.v2.lesion`)
- Gradient-pressure lesioning: {placement masks} × stream × {battery/probes} →
  differential capability map. Causal localization by constrained update pressure
  (discovered round 9: expert-only masks destroy MoE factual knowledge, −10.4 EM,
  invisible to CE). Mask builders for HF MoE layouts provided.

### Gates (`scbf.v2.gates`)
- Update-admission policies from rounds 5–6: `ErrGate` (≥ rolling median),
  `ExcessGate` (realized − frozen-reference ≥ rolling median; doubly exogenous),
  `BandGate` (p25–p75), `RandGate`. All ~50%-rate by construction.

### Telemetry (`scbf.v2.telemetry`)
- `RunLog`: timeseries/snapshots rows + meta.json, CSV flush; schemas identical to
  rounds 5–9 so existing analysis ports.

## Non-goals (this version)
v1 migration or modification; visualization; lm-eval config management; distributed
training; optimizers beyond SGD-in-hook.

## Known limits
PlasticityProbe infeasible ≥ ~2B (deepcopy). Task batteries are the capability
heartbeat — CE-family metrics alone are demonstrably blind to knowledge erosion
(round 9). Single-run drift values untrustworthy at scale (round 6).
