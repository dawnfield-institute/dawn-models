# CIMM-legacy comparison: the shakedown rediscovered why CIMM was built that way

**Date:** 2026-07-19
**Trigger:** Peter's pointer to the original CIMM (Cosmic Information Mining Model,
`dawn-models/stable/cimm-legacy/`, March 2025 — whose first substrate was prime gaps).
Question: does the original have continuous-learning machinery the Tiny line dropped
that addresses our measured failure modes?

**Answer: yes, comprehensively.** The shakedown's baseline collapse is, in hindsight, a
controlled demonstration of what happens when CIMM's regulatory stack is removed.

## CIMM mechanisms mapped to our three measured failure modes

| Our failure mode | CIMM's countermeasure (file:line) |
|---|---|
| ReLU death = absorbing state | Gravity pullback toward a tracked entropy baseline: `pred += -0.01*(entropy - entropy_baseline)` (cimm.py:501) — a standing escape gradient; plus bounded neuron expansion with fresh small-init weights (pruning.py:44-56, factor ∈ [1.0, 1.15]) |
| Hot learning rate (TinyCIMM: `lr = 0.001 + 0.005*momentum`, rises with momentum) | Exponential damping on entropy variance: `lr *= exp(-entropy_variance*5)` (adaptive_controller.py:168), superfluid-coherence scaling, hard bounds [1e-6, 0.1], QFI-adaptive gradient clipping `0.1*(1+log(QFI))` (entropy_monitor.py:105-111) |
| Unconditional per-token updates | **Update gating**: weight updates every 10 predictions, pre-validated, with **rollback if post-update entropy moves away from target** (cimm.py:650, entropy_monitor.py:411-432) |

## Two findings that reshape the Ember III round plan

1. **CIMM already contains the update-gating architecture** that the arms-verdict
   journal proposed as round 2. Its gate is endogenous (entropy-target validation +
   rollback). Round 2 (A′/B′/C′) is therefore properly framed as: *CIMM gated updates
   endogenously — does an exogenous gate do better, and is the gate's signal origin
   load-bearing at all?* The program's newest question was latent in Peter's earliest
   model.
2. **CIMM's XGBoost "quantum memory" is a primitive bilevel gate-trainer**
   (quantum_memory.py:22-145): a *separate model class*, trained on a *different
   objective* (which correction magnitudes historically stabilized), predicts the
   refinement applied to updates. That is a concrete early answer to the open
   question "who trains the gate?" (review point: a gate trained by the same
   optimizer on the same stream is capturable) — CIMM's answer is a second learner
   on a slower loop.

## Honest caveats

- **CIMM's stabilization signals are almost entirely endogenous** (own activation
  entropy, own prediction history, own collapse deviations) with one residual-class
  input (prediction error vs realized target via `give_feedback`). By Ember III's
  taxonomy, CIMM is an elaborate *endogenous control system that works* — at least on
  the streams it was run on. It never faced the authorability test. Its stability and
  the exogeneity thesis are orthogonal claims; our harness can now dissect which
  ingredient carries the stability.
- **~7 interacting hand-tuned mechanisms** (damping 0.985, EMA cascades 0.85/0.87,
  refinement bounds ±0.03, QPL bounds [0.05, 2.0], …). Which are load-bearing is
  unknown. Perfect ablation target for the exact harness built this week.

## Revised round-2 plan (supersedes the arms-verdict "next" section in part)

1. **CIMM stabilizer ablation** on the ember3_drift harness: port the minimal set as
   switchable flags on `StabilizedTinyCIMMEuler` — (a) gravity/entropy-baseline
   pullback, (b) variance-damped lr, (c) QFI-adaptive clipping, (d) update-frequency
   gating with entropy-validated rollback. Measure which flags extend the early edge /
   slow erosion / preserve plasticity. (~90s per run; a single afternoon covers the
   2^4 grid at 1 seed + winners at 3 seeds.)
2. **Gate-origin arms (A′/B′/C′)** on the best stabilized config: endogenous
   (CIMM-style entropy validation) vs exogenous (realized-error percentile) vs
   rate-matched random gate. Plasticity-metric fix (fixed probe segment) goes in first.

## Sources

- `cimm_core/cimm.py` (update loop, wave collapse, gravity), `entropy/entropy_monitor.py`
  (QBE-validated updates, adaptive clipping), `entropy/quantum_potential_layer.py`
  (non-Markovian damped QPL), `optimization/adaptive_controller.py` (variance-damped lr),
  `learning/quantum_memory.py` (XGBoost gate-trainer), `learning/superfluid_dynamics.py`
  (curvature-based coherence), `optimization/pruning.py` (Landauer-cost prune + bounded
  expansion). Lore: `cimm-legacy-production`, `legacy-cim-qbe-origins`.
