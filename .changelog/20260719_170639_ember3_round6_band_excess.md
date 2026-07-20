# Ember III round 6 — band + excess gates at 160M: aleatoric hypothesis directionally confirmed; random unbeaten

**Date:** 2026-07-19
**Location:** `research/ember3_scale160m/`

## What was added

- `scripts/run_scale160m.py`: two round-6 gates — `band` (update iff loss in
  [p25, p75) of rolling window: skips trivial + irreducible tail) and `excess`
  (update iff loss − frozen-model loss on the same chunk ≥ rolling median: a
  doubly-exogenous learnability estimate using the frozen reference as difficulty
  prior). 4 runs (x2 jitter repeats each), 33 GPU-min.

## Findings

1. **Directional confirmation of the round-5 aleatoric hypothesis**: band and excess
   beat the difficulty gates decisively (edge +0.7–1.1pp over err, outside jitter;
   held-out and plasticity recover to random-gate levels).
2. **Random remains unbeaten**: ladder rand ≥ band > excess > err ≈ ent. Reading:
   random's active ingredient is representative *coverage*, which no per-chunk
   scoring rule provides. Next candidates: coverage-aware selection, or accept
   "selection second-order at 50% rate" as the rung-1 bound.
3. **excess = stability champion**: lowest drift AUC of every arm (0.017–0.019,
   tight) at 97% of rand's edge — best stability-per-update profile measured.
4. **Instrument finding**: drift AUC is jitter-bistable under some gates (band
   repeats: 0.023 vs 0.355 at indistinguishable competence) → single-run drift
   values untrustworthy at 160M; **round-5's "ent churns 5–7x" claim downgraded**
   (was a single run). Deeper point: at 160M, wildly different representational
   paths carry identical behavior — drift measures the path, not the function.

Journal: `journals/2026-07-19_round6_band_excess.md`. Not committed.
