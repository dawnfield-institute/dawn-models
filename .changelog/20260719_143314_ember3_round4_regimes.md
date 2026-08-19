# Ember III round 4 — temporal vs spatial specialization (regime-switching)

**Date:** 2026-07-19
**Location:** `research/tinycimm/TinyCIMM-Euler/experiments/ember3_drift/`

## What was added

- `scripts/run_regimes.py` — pre-registered regime-switching experiment: alternating
  prime-gap / harmonic segments (per-regime z-scoring, regimes continue across
  visits), dwell sweep {250, 1000, 4000}, arms ADAPT / ADAPT_EG / ROUTED (best-
  snapshot frozen specialists + oracle router) / FROZGEN. `safe_deepcopy` extracted
  into `instrumentation.py`.
- Results: `results/runs_regime/`, `results/regime_summary.csv`.

## Findings

1. **ADAPT beats the oracle-routed frozen pair at every dwell — no crossover down to
   D=250** (0.65–0.69 vs 0.82). Per-regime check passed: the margin is genuine
   structure-tracking on the harmonic regime (0.07–0.14 vs frozen-best 0.44), not
   frozen-specialist decay on the drifting prime regime (parity there).
2. **Mechanism: adaptation substitutes for representation** — the adaptive model
   tracks the function rather than internalizing it. Architecture-level instance of
   the encoding-ladder claim (specialist cache = artifacts; adapter = generator).
3. **Switch economics measured**: recovery ≤50–111 tokens median; retention cost
   dose-dependent (+0.03 → +0.35 as dwell grows) — forgetting is real, small once
   amortized, and now a measured quantity for the PAC/conservation budget question.
4. **Exogenous gate breaks under regime mixing without local calibration**: at D=250
   the global rolling-median threshold leaves the gate open 68% on primes but 16% on
   harmonic (update starvation → frozen-level performance). Routed architectures
   provide regime-local gate calibration for free → the plastic-experts hybrid earns
   its keep twice.
5. Pre-flight caught and fixed a strawman: frozen specialists now selected at their
   best rolling-MAE pretrain state (0.75 → 0.41 on harmonic).

Journal: `2026-07-19_round4_regimes.md`. Not committed.
