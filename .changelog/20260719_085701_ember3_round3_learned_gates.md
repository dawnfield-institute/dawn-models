# Ember III round 3 — learned gates (authorability), gravity refutation, erosion source

**Date:** 2026-07-19
**Location:** `research/tinycimm/TinyCIMM-Euler/experiments/ember3_drift/`

## What was added

- `scripts/run_arms.py` (harness 2026-07-19.5): learned gates `--gate-arm
  learn_self|learn_outer` — 3-feature logistic gate, one-step-delayed REINFORCE with
  symmetric credit timing, rate-regularized to ~50%; arms differ ONLY in reward
  channel (self-measured training loss vs leak-free realized error).
- `scripts/analyze_gates.py` extended; `results/gravity_summary.csv`.

## Findings

1. **Authorability (headline)**: L-self is significantly worse than a rate-matched
   random gate (late edge −5.0% vs −2.9%; held-out MAE 8.59 = program worst) — the
   gate trained on the model's own bookkeeping *learned something actively harmful*
   to realized performance; one seed rediscovered the ossification profile via
   reward maximization. L-outer beats L-self on held-out MAE 3/3 paired seeds but
   does not recover the fixed err gate — fixed "update on surprise" remains champion.
2. **Gravity refuted**: the ablation's 1-seed drift-halving does not replicate at
   3 seeds x 50k (drift ranges 0.2–0.8 swamp mechanism effects). Metric lesson:
   drift AUC, not final snapshot. Pre-registered confirmation gate did its job.
3. **Erosion source**: P(t) flat over 50k → no plasticity loss; the edge deficit is
   a one-time transition to a stable non-stationarity tracking cost. Reframe: gate
   signals select equilibrium quality; propose co-moving-frame (excess) drift.

Open debts: CIMM rollback vs random-rollback control; gate-weight logging;
higher-capacity learned gates. Journal: `2026-07-19_round3_learned_gates.md`.
Not committed.
