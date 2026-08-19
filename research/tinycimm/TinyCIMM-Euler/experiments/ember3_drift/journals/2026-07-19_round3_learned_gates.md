# Round 3: learned gates (authorability), gravity refutation, erosion source

**Date:** 2026-07-19
**Runs:** `runs_gate_learn_{self,outer}` (3 seeds x 50k), `runs_grav3` / `runs_gravq3` /
`runs_gate_none` seeds 1-2 (gravity confirmation), harness 2026-07-19.5.
Tables: `gate_origin_summary.csv` (extended), `gravity_summary.csv`.

## 3a — Learned gates: the self-channel gate learns something actively harmful

Design: 3-feature logistic gate (z-scored realized error, short error trend,
previous-step entropy), trained by one-step-delayed REINFORCE with symmetric credit
timing (a_{t-1} rewarded with token-t signals, which reflect a_{t-1}'s consequence),
rate-regularized to ~50% (realized 0.496–0.503 in all runs). **Only the reward channel
differs**: L-self = improvement in the model's own self-measured training-path loss
(authorable: the model's bookkeeping, target-leak included); L-outer = improvement in
leak-free realized prequential error (exogenous: the fixed stream's verdict).

| arm | late edge (mean [range]) | held-out MAE (mean [range]) |
|---|---|---|
| fixed err gate (round 2b) | **−2.3%** [−2.4, −2.2] | 7.93 |
| rand gate | −2.9% | 8.00 |
| ungated (3 seeds now) | −3.0% [−3.3, −2.6] | **7.88** |
| L-outer | −3.4% [−3.9, −2.9] | 8.16 |
| ent gate | −4.7% | 8.08 |
| **L-self** | **−5.0%** [−7.1, −3.1] | **8.59** [8.03, 9.00] |

Findings, in decreasing confidence:

1. **L-self is significantly worse than the rate-matched random gate** (−5.0% vs
   −2.9%; worst held-out MAE in the program). A gate that had learned nothing would
   match `rand`; L-self *learned something that damages realized performance*.
   Optimizing the authorable channel diverged from optimizing reality — the idea
   doc's §2.1 claim, in learned form. Seed 0 is the showpiece: lowest drift (0.28)
   with the worst competence (edge −7.1%, held 9.0) — the ossification profile
   *discovered by reward maximization*.
2. **Channel comparison at matched learner capacity favors the exogenous reward**:
   L-outer beats L-self on held-out MAE in 3/3 paired seeds, on late edge in 2/3.
   Directionally consistent with round 2b; not non-overlapping — stated at that
   strength and no more.
3. **Learning the gate costs more than it earns at this capacity**: L-outer (−3.4%)
   does not recover the fixed err gate (−2.3%). Champion remains the fixed rule
   "update on surprise." The REINFORCE exploration cost exceeds what 3 logistic
   weights can express.

Asymmetry note (pre-stated last session): the positive reading (self-channel harm) is
strong; any null would have been weak (3 weights may lack capacity for clever
exploitation). We got the positive.

Instrumentation debt: gate weight trajectories were not logged — we cannot yet say
*what* L-self learned. One-line fix next round.

## 3b — Gravity confirmation: REFUTED

The ablation's standout (1 seed, 20k) does not replicate at 3 seeds x 50k:

| cfg | late edge | drift final (range) | held final |
|---|---|---|---|
| plain | −3.0% | 0.58 [0.50, 0.67] | 7.88 |
| gravity | −2.7% | 0.61 [0.26, 0.81] | 7.90 |
| gravity+qclip | −3.3% | 0.44 [0.21, 0.65] | 8.00 |

No drift reduction for gravity alone; gravity+qclip's lower mean is inside seed
noise; nothing beats plain on held-out MAE. **Final-snapshot drift is too
seed-noisy to rank mechanisms** (ranges of 0.2–0.8 recur across all configs) —
metric lesson: use time-averaged drift (AUC) in future rounds. The pre-registered
confirmation requirement did its job, and honoring the ablation's winner rule
(rather than hand-picking gravity post-hoc) kept round 2b clean.

## 3c — Erosion source: non-stationarity, not plasticity loss

With the fixed-segment plasticity metric: P(t) is flat over 50k tokens (fit slope
±0.0003; first-half mean ≈ second-half mean) while the competence edge drops once
between the first and second 10k window and then plateaus (−2.3..−3% stable for
30k). **No progressive plasticity loss.** The "erosion" is a one-time transition
into a stable steady-state tracking deficit against a comparator that re-centers
itself as the gap scale grows. Reframe adopted: at this rung the drift problem is a
**tracking-cost / equilibrium-selection problem**, not a degradation problem — gate
signals select the equilibrium's quality. Suggested doc §5.2 revision: measure
drift in the co-moving frame of the stream (excess drift = model motion minus world
motion); lab-frame stillness is ossification, not stability.

## Program state after round 3

- Exogeneity thesis: supported twice — fixed gates (err > rand > ent,
  non-overlapping) and learned gates (self-channel reward actively harmful vs
  matched random; outer-channel better in paired comparison).
- CIMM mechanisms: gating principle validated (as the round-2/3 architecture);
  gravity refuted at confirmation; rollback validator still untested vs
  random-rollback control — **open debt**.
- Metric upgrades queued: drift AUC (not final snapshot), gate-weight logging,
  co-moving-frame drift.
- Open next rounds: rollback control; higher-capacity learned gates (does more
  capacity make self-channel capture worse — the doc's asymptote direction?);
  160M rung once harness lessons consolidate.
