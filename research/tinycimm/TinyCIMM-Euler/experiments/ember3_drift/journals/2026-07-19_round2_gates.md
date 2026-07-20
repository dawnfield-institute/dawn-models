# Round 2: CIMM stabilizer ablation + gate-origin arms — first positive separation

**Date:** 2026-07-19
**Runs:** `results/runs_abl_*` (2^4 grid, 20k tokens, seed 0);
`results/runs_gate_{none,ent,err,rand}` (50k tokens; ent/err/rand x 3 seeds, none x 1);
harness 2026-07-19.4 (CIMM stabilizers + origin gates + fixed plasticity segment).
Tables: `results/ablation_summary.csv`, `results/gate_origin_summary.csv`.

## 2a — CIMM stabilizer ablation (gravity / vlr / qclip / gate)

- **Gravity (entropy-baseline pullback, CIMM cimm.py:501) is the standout mechanism**:
  lower drift in every paired comparison, gravity+qclip halves final drift (0.177 vs
  0.32–0.48), best held-ahead MAEs (7.92–7.94). Flagged for 3-seed confirmation as its
  own claim.
- **CIMM's every-10 gate costs early learning** (edge +4.3–5.0% vs +5.5–6.4% ungated)
  and its entropy-validated rollback rejected ~50% of attempted updates — a coin-flip-
  like rate that warrants suspicion (an uninformative validator would reject ~half).
  The two least-eroding configs late were vlr+gate, but they shed plasticity.
- **The pre-stated winner rule selected `abl_0000` (all off)** — plasticity-retention
  and drift-slope criteria eliminated the low-erosion configs; within the eligible pool
  the edge differences were noise at 1 seed. Rule honored; the origin arms therefore ran
  on the plain repaired substrate — which is also the most sensitive choice (no CIMM
  machinery muting drift dynamics).

## 2b — Gate-origin arms: err > rand > ent, non-overlapping

Per-token update gate at matched ~50% rate; the ONLY difference between arms is the
gate signal's origin. Endogenous = previous-step activation entropy >= rolling median;
exogenous = realized prequential error >= rolling median; random = fair coin.

Means over 3 seeds (ranges in brackets); ungated reference 1 seed:

| | ent (endogenous) | err (exogenous) | rand | none |
|---|---|---|---|---|
| late edge vs rolling median | **−4.7%** [−6.2, −3.5] | **−2.3%** [−2.4, −2.2] | −2.9% [−3.0, −2.8] | −2.6% |
| plasticity, final probe | **−0.123** [−0.16, −0.09] | **−0.050** [−0.06, −0.03] | −0.069 | −0.076 |
| final drift | 0.41 | 0.57 | 0.57 | 0.56 |
| early edge | +1.9% | +1.9% | +1.8% | +1.7% |
| update rate | 0.502 | 0.501 | 0.498 | 1.0 |

**Findings (3 seeds, ranges non-overlapping on both key metrics):**

1. **The exogenous gate beats the rate-matched random gate** on late competence
   (−2.3% vs −2.9%, no seed overlap) and plasticity retention (−0.050 vs −0.069).
   Selecting updates by realized surprise does real information work — the first
   positive result for the gating-level exogeneity question.
2. **The endogenous gate is worse than random** — by a wide margin (−4.7% late edge,
   worst plasticity). The model's own entropy signal, used as a gate, selects the
   WRONG updates.
3. **The endogenous arm drifts least (0.41) while performing worst** — it walked
   directly into the ossification corner the round-1 metric design pre-defined
   ("bounded drift with collapsing plasticity is NOT success"). The paired-metric
   design caught exactly the failure mode it was built to catch: the endogenous gate
   buys representational stability by refusing the updates that track the moving
   stream. Self-consistency over world-tracking — the idea doc's §2 degeneration
   argument, observed empirically at micro scale.

## Honest scope

- This is evidence about signal **alignment** at *fixed* (non-learned) gates — the
  exogenous signal points at the right updates; the endogenous one doesn't. It is NOT
  yet the authorability/gaming claim (an endogenous gate being *learned to* evade
  correction) — that requires learned gates, a later round.
- Effect sizes are small in absolute terms (0.6–2.4pp of comparator MAE) but
  consistent and non-overlapping across seeds. One substrate, one stream, 3 seeds.
- No circularity between the err gate and the edge metric: prequential error is
  measured pre-update each token; the gate only shapes which updates occur, which is
  the causal path under test.

## Program state after round 2

- Round 1: drift localized in the per-token update channel (capacity arms null).
- Round 2: within that channel, **gate-signal origin matters, ordering exogenous >
  uninformative > endogenous** — the exogeneity thesis's first empirical support,
  landing at the level the doc's own nociception framing (§2.2) predicted.
- Next candidates: gravity 3-seed confirmation; learned-gate round (authorability
  proper — who trains the gate, CIMM's XGBoost bilevel pattern as one arm); erosion
  source disambiguation via the now-comparable P(t) (flat-ish P with eroding edge
  leans toward stream non-stationarity as a component).
