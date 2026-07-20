# Round 6: band + excess gates — the aleatoric hypothesis is directionally right, and random still wins

**Date:** 2026-07-19
**Runs:** band x2, excess x2 (jitter repeats), vs round-5 comparators. 33 GPU-min.
Results: `results/scale160m_summary.csv`.

## Results (stream edge last 10% / drift AUC / held-out final / plasticity last)

| arm | edge | drift AUC | held-out | plast |
|---|---|---|---|---|
| rand | **0.364** [.361–.368] | 0.027 | **3.268** | 0.026 |
| band (p25–p75) | 0.358 [.3573–.3586] | 0.023 / **0.355** (!) | 3.272 | 0.027 |
| excess (learnability) | 0.354 [.3534–.3541] | **0.017–0.019** (lowest of all) | 3.284 | 0.026 |
| err | 0.347 [.343–.351] | 0.032 | 3.304 | 0.018 |
| ent | 0.343 | 0.178 | 3.305 | 0.018 |
| none (rate 1.0) | 0.405 | 0.052 | 3.276 | 0.026 |

## Verdict

1. **The aleatoric hypothesis gets directional support**: both tail-aware rules beat
   the difficulty gates decisively (band +1.1pp, excess +0.7pp edge over err, both
   outside the jitter floor; held-out and plasticity recover to rand/none levels).
   Whatever was hurting informed selection in round 5, discounting the irreducible
   tail removes most of it.
2. **But the ambitious prediction failed: nothing informed beats random.** The
   selection-rule ladder now reads rand ≥ band > excess > err ≈ ent. Representative
   sampling remains unbeaten at this scale and rate. The residual gap suggests the
   active ingredient in random's win is **coverage** — a per-chunk scoring rule,
   however well-calibrated, systematically narrows the update distribution, and no
   surprise statistic fixes that. (Next-round candidate: stratified/coverage-aware
   selection; or accept "at 50% rate, selection is second-order, coverage is
   first-order" as the rung-1 conclusion.)
3. **excess is the stability champion**: lowest drift AUC of every arm (0.017–0.019,
   tight across repeats) with healthy competence and plasticity — 97% of rand's edge
   at the least representational movement. If the objective is stability-per-update
   rather than raw edge, the doubly-exogenous learnability gate is the best profile
   measured so far.

## Instrument finding — and a round-5 claim downgraded

band's two repeats: near-identical competence (edge .3573/.3586, held 3.274/3.270)
but **drift AUC 0.355 vs 0.023 — a 15x difference from CUDA jitter alone.** The
representational path is knife-edge sensitive under some gates even when behavior
is not. Consequences: (a) single-run drift values at 160M are not trustworthy;
(b) **round 5's "ent churns at 5–7x" claim is hereby downgraded** — it was a single
run, and band's repeat shows jitter can produce that magnitude. Ent's churn needs
repeats before it's a claim. (c) The deeper observation stands on its own: at 160M,
wildly different representational trajectories can carry indistinguishable
behavior — the function is multiply realized, and drift measures the path, not the
function. This sharpens the co-moving-frame agenda from round 3: raw CKA drift is
even less of a health metric at scale than the toy rounds suggested; the
competence+plasticity pairing is doing the real work.

## Program state after round 6

- Rung-1 practical answer: **representative sampling (or ungated, if stream
  priority) is the champion**; among informed rules, learnability (excess) is the
  best-behaved and cheapest-drifting.
- The exogeneity thesis's scale-story, stated honestly: exogenous *signals* remain
  necessary for anything informed (endogenous = worst throughout), but at this
  rung no informed rule has beaten representative coverage. The thesis's live
  battleground moves to: coverage-aware exogenous selection, and regimes where
  coverage and selection trade off (non-stationary/regime-switching streams at
  scale — round 4's territory, one rung up).
- Ember-v1 vs pythia under this harness remains queued — unchanged by today's
  results and still the first measurable role for the physics at scale.
