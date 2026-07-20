# Shakedown baseline: as-built substrate collapses to the constant predictor

**Date:** 2026-07-18
**Runs:** 3 arms x 3 seeds x 50k tokens, as-built substrate (`runs/`), harness 2026-07-18.2.
Deterministic: the matrix was run twice (before/after adding rep-health diagnostics) and
reproduced per-run numbers exactly.

## Result

**Every run — all arms, all seeds — collapses to (approximately) the best-constant
predictor.** This is the "both drift/collapse" diagnostic row of the pre-registered
interpretation table, resolved by the diagnostics as a **substrate failure**, not a
metric failure.

Numbers (means over 3 seeds; floors from `meta.json` baselines):

| | Arm A (endogenous) | Arm B (residual) | Arm C (surrogate) |
|---|---|---|---|
| held-ahead MAE final | 7.919 | 7.918 | 8.907 |
| held-ahead MAE best-ever | 7.906 | 7.905 | 7.895 |
| **best-constant floor** | **7.91** | **7.91** | **7.91** |
| stream edge vs rolling-median, first 10% | −2.4% | −2.2% | −1.9% |
| stream edge, last 10% | −3.5% | −3.6% | −3.4% |
| plasticity first probe | −0.02 | +0.06 | −0.03 |
| plasticity last probe | **−0.19** | **−0.22** | −0.08 |
| grow / prune events | 28.7 / 18.0 | 25.7 / 2.0 | 27.3 / 4.0 |

Readings:
- No arm ever beats the constant floor on held-ahead data, at any snapshot.
- No arm ever beats the causal rolling-median surface reader in-stream, even early.
- **Plasticity goes negative**: late in the run, a clone adapting on fresh unseen data
  gets *worse* while training. The dying substrate anti-learns.
- Drift (1−CKA) is intermittently undefined: at those snapshots `const_frac = 1.0` and
  `h_std = 0.0` — every hidden unit has zero variance across the probe set. Even at
  defined snapshots, `const_frac ≈ 0.97–0.98`.

## Mechanism

Prime gaps are strictly positive inputs. Once a unit's weights go negative-dominant its
pre-activation is negative for the whole input distribution — ReLU death is an
**absorbing state**. Dead hidden layer ⇒ only the output bias receives gradient ⇒
constant predictor. Growth injects live neurons (small random weights) but the update
dynamics kill them faster than growth replaces them: a birth/death process whose death
rate wins. The arms *do* differ in event structure (A prunes 18x/run, B/C 2–4x), but the
substrate collapses through the per-token weight channel before structure policy can
matter.

No historical TinyCIMM-Euler run exceeded ~10k steps (online runs: 500). First full
representation death appears at t≈2.5–6.5k. The failure was always present; nobody had
run long enough. This is the port audit's "bounded-run assumption" class, in the
learning dynamics themselves.

## Design principle extracted

**The update dynamics of a continuous learner must have no absorbing states** — every
degenerate configuration needs an escape gradient. The snapshot regime never needed
this rule (you stop before falling in); continuous learning guarantees any absorbing
state is eventually visited.

## Repair ladder (probes at 10k tokens, Arm A, seed 0)

- **R1 — normalization only** (z-score vs causal 1k-gap calibration window): partial.
  No NaN snapshots, but `const_frac` still 0.66–0.78.
- **R2 — R1 + leaky activation (slope 0.01,** `StabilizedTinyCIMMEuler` **subclass,
  original file untouched)**: accepted. `const_frac = 0.0` at every snapshot, drift
  finite everywhere, and a genuine early competence edge: **+4.5% over the causal
  rolling-median in the first 2.5k tokens, eroding to −1..−2% by 10k.**

R2's erosion curve replaces catastrophic collapse with graded competence loss — the
regime the arms experiment actually needs. Whether that erosion is model plasticity
loss or stream non-stationarity (mean gap grows ~ln p) is disambiguated by the
plasticity probe: declining P(t) → the former; flat P(t) with eroding edge → the latter.

## Artifacts

- Baseline record: `results/runs/`, plot `results/summary.png`, stats
  `results/summary_stats.json` (includes per-arm rolling-median edges).
- Repair probes: `results/runs_r1/`, `results/runs_r2/`.
- Repaired-substrate arms matrix: `results/runs_stab/` (in progress at time of writing;
  readout in the next journal entry).
