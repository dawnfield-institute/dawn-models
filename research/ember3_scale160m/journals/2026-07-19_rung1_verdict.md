# Rung 1 verdict: the toy gate ordering does NOT replicate at 160M — a sharper structure appears

**Date:** 2026-07-19
**Runs:** 8 (frozen_ref, none, err x2, ent, rand x3), pythia-160m fp32, full-rank Adam
1e-5, 10k x 512-token WikiText-103 chunks, RTX 3090 (CT103). ~8 min/run.
Results: `results/`, `results/scale160m_summary.csv`.

## Sanity anchors (all passed)

- Frozen pythia-160m wikitext loss 3.53–3.62 (ppl ~34–37) — published ballpark.
- Adaptation is real: stream edge over frozen +0.34..+0.40 nats; held-out improves
  3.59 → 3.26–3.31 in all adaptive arms.
- Gate rates 0.495–0.501; CUDA-jitter repeat (err x2): edge ±0.004, drift AUC ±0.012.
- Pre-flight caught fp16-checkpoint loading → instant NaN → **NaN silently closed the
  err gate permanently** (NaN >= median is False). Fixed with explicit fp32 + a hard
  crash on non-finite loss. Another exhibit in the silent-corruption catalogue.

## Results (per-run; ranges where multiple runs)

| arm | stream edge, last 10% (nats) | drift AUC | held-out final | plast last | rate |
|---|---|---|---|---|---|
| none (ungated) | **0.405** | 0.052 | 3.276 | 0.026 | 1.00 |
| rand | 0.364 [0.361–0.368] | **0.027** | **3.268** [3.258–3.279] | 0.026 | 0.50 |
| err (exogenous) | 0.347 [0.343–0.351] | 0.032 | 3.304 | 0.018 | 0.50 |
| ent (endogenous) | 0.343 | **0.178** | 3.305 | 0.018 | 0.50 |

## Verdict against the pre-registered table

The predicted replication row (err > rand > ent) did **not** fire. What fired is a
combination the table didn't anticipate:

1. **Both informed gates lose to the random gate** — on stream edge (rand 0.361–0.368
   vs err 0.343–0.351, non-overlapping incl. the jitter repeat) AND held-out
   (3.27 vs 3.30). Informed selection-by-difficulty is *worse* than representative
   sampling at matched rate.
2. **Signal origin does not separate competence here**: err ≈ ent on every competence
   metric.
3. **Origin separates DRIFT dramatically**: ent's drift AUC is 0.178 — 5–7x every
   other arm, far outside jitter — with no competence divergence. The endogenous
   gate's selection churns the representation without changing performance. (Note
   the inversion vs toy scale, where ent = lowest drift: the ossification signature
   did not transfer; a churn signature appeared instead.)
4. Update thinning at 50% costs stream adaptation (0.36 vs 0.41) but not held-out
   generality (rand ≈ none) — half the updates, same generality: update efficiency
   doubles under random thinning.

## Mechanistic reading (hypothesis, stated as such)

The toy substrate's surprise was mostly **epistemic** — high prequential error marked
unmodeled-but-learnable structure, so "update on surprise" selected well. WikiText's
high-loss chunks are heavily **aleatoric** — rare names, tables, irreducible tail —
so a median-threshold surprise gate preferentially spends updates on the least
transferable text while skipping the typical distribution. Random sampling is
representative; that wins. err ≈ ent follows: both rules select the same
difficulty tail from opposite directions.

**Refinement this points to (next round candidate): gate on *learnability*, not
difficulty** — realized error that history says is reducible (e.g., a percentile
BAND: skip both the trivially-easy and the irreducible tail; or a learned-outer gate
rewarded on realized improvement, which round 3's L-outer already prototyped). The
exogeneity thesis survives in refined form: the *signal* must still be exogenous —
but the *selection rule* must separate reducible from irreducible surprise, a
distinction the toy substrate never forced.

## What rung 1 bought

Exactly what the ladder discipline is for: a toy-validated rule met a real substrate
and broke in an informative direction, cheaply (68 GPU-minutes), with the instruments
(edge/drift/plasticity + jitter floor) still working perfectly at 160M. The claim
"update on surprise" is now bounded: it holds where surprise is epistemic; it fails
where surprise is aleatoric-contaminated. That is a better theory than the one we
brought up the ladder.

## Infrastructure state

`/data/ember3` on CT103 holds venv + model + data + results (~4.5GB) — kept for
future rungs; removable with one rm -rf. GPU returned to idle.
