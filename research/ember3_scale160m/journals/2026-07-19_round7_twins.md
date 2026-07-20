# Round 7: the physics twins — conservation is stable under continuous learning; gate results are architecture-robust

**Date:** 2026-07-19
**Runs:** 8 — {frozen_ref, none, rand, excess} x {ember_v1_mixed, vanilla_mixed},
SEQ 256, 10k WT103 chunks (GPT-2 tokenization), native training objectives
(ember: CE + PAC conservation; vanilla: CE), pure-CE measurement for both.
~55 GPU-min. Results: `results/results_twins/`.

## Results (single seed per cell — stated up front)

| | ember none/rand/excess | vanilla none/rand/excess |
|---|---|---|
| frozen stream CE (first→last 10%) | 3.607 → 3.636 | 4.041 → 4.115 |
| stream edge last10 (vs own frozen) | 0.241 / 0.221 / 0.214 | 0.301 / 0.286 / 0.270 |
| held-out gain (start→final) | 0.017 / 0.048 / 0.034 | 0.105 / 0.143 / 0.115 |
| fresh-segment response (plast metric) | −0.13 (stable all run) | −0.29 (stable all run) |
| drift AUC | 0.0035–0.0059 | 0.0023–0.0032 |
| conservation loss (start→final) | 0.12–0.16 → **0.07–0.13** | — |

## Findings

1. **The PAC conservation budget is stable — in fact tightening — under open-ended
   continuous learning.** Across all three ember arms, conservation loss *decreases*
   30–40% over 10k updates on out-of-pretraining-mix text. The constraint neither
   blows up nor fights the optimizer; the model adapts while moving toward its
   budget. Nobody had run a conservation-constrained model under continuous
   updating before — this is the first stability datapoint, and it is positive.
   (Side detail: fewer updates → less settling; rand keeps conservation higher
   than ungated.)
2. **Gate behavior is architecture-robust**: within both twins, the ordering is
   none > rand > excess on stream edge and rand ≥ excess ≥ none on held-out gain —
   the same qualitative structure as pythia (rounds 5–6). The physics package does
   not change how update-gating behaves. The rung-1 gate conclusions generalize
   across architectures.
3. **The headroom confound dominates the raw adaptation comparison and is
   acknowledged as such**: ember arrives ~0.43 nats better on the stream (its
   pretraining lineage) and stays better absolutely throughout; vanilla, starting
   worse, gains more from adaptation (edge 0.30 vs 0.24; held-out gain 0.11 vs
   0.02). Adaptation-magnitude differences between twins CANNOT be attributed to
   physics — a worse model has more to learn. Matched-headroom designs (compare at
   equal frozen CE, or race from matched intermediate checkpoints) are the
   follow-up if this axis matters.
4. **One twin-attributable behavioral difference survives the confound**: on the
   identical fixed fresh segment, vanilla's within-segment CE trend is 2.2x worse
   than ember's (−0.29 vs −0.13), constant across the entire run for both. Caveat:
   the within-segment first-vs-last measure conflates adaptation dynamics with the
   segment's own difficulty profile, so this is "response to identical fresh hard
   text," not clean plasticity — but the twins see the same text, so the *gap* is
   real and architecture-attributable. Mechanism unresolved.
5. Drift AUC is tiny for both twins (≤0.006; an order below pythia's) — ember
   marginally higher; not interpreted per the round-6 jitter caveat.

## Verdict on the round question

"Does the physics change continuous-learning behavior?" — **Partially, and where it
does, favorably or neutrally**: the conservation constraint is compatible with (and
settles under) continuous adaptation; gate phenomenology is unchanged
(architecture-robust results are good news for the program's generality); the one
behavioral gap (fresh-text response) favors the physics twin but needs a cleaner
instrument and matched headroom before it becomes a claim.

## Scope

Single seed per cell (CUDA-jitter floors from rounds 5–6 suggest edge differences
of this size are real, but repeats are cheap and should precede any strong claim);
one stream; the twins' absolute capability is TinyStories-era. The round's
strongest product is the conservation-stability datapoint plus the
architecture-robustness of the gate results.
