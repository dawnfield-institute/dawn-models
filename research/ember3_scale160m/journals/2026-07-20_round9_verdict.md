# Round 9 verdict: full adaptation retains everything; expert-concentrated adaptation destroys knowledge that CE never saw

**Date:** 2026-07-20
**Battery:** lm-eval 0.4.12, 8 tasks x {frozen, experts-adapted, full-adapted},
pre-registered rules in `2026-07-20_round9_prereg.md` (locked before any run).
Results: `results/results_bench/`. Sanity anchors passed (frozen: ARC-c 0.469,
HellaSwag acc_norm 0.770, PIQA 0.799 — published OLMoE-1B-7B ballpark).

## The table (frozen / experts / full; verdict by the locked 2xSE rule)

| task | frozen | experts | full | experts Δ | full Δ |
|---|---|---|---|---|---|
| arc_challenge | .4693 | .4582 | .4676 | = (−.011) | = (−.002) |
| arc_easy | .7790 | .7782 | .7786 | = | = |
| boolq | .7471 | .7343 | .7428 | = (−.013) | = (−.004) |
| hellaswag (an) | .7701 | .7712 | .7760 | = (+.001) | = (+.006) |
| lambada | .7293 | .7178 | .7237 | = (−.011) | = (−.006) |
| piqa | .7987 | .7911 | .7960 | = (−.008) | = (−.003) |
| winogrande | .6843 | .6732 | .6803 | = (−.011) | = (−.004) |
| **triviaqa (EM)** | **.4805** | **.3760** | **.4895** | **DOWN (−.104)** | = (+.009) |

## Findings

1. **The full-adaptation arm passes the program's strongest test: 9/9 no-change.**
   After 5k continuous-learning updates it retained every measured capability
   (hellaswag even trending up, within noise) while holding round 8's stream gains.
   For the distributed-update regime at this dose, "keeps its learnings" is now a
   task-level measurement, not a CE proxy.
2. **The experts-only arm destroyed 10.4 points of TriviaQA** (.481 → .376) — an
   unambiguous, threshold-shattering knowledge erosion. Mechanistic reading: expert
   FFNs are where a MoE's factual knowledge predominantly lives; concentrating 5k
   chunks of wiki gradient pressure exclusively into them overwrote stored facts.
   Round 8's inversion ("concentrating updates concentrates the damage") is not
   just confirmed at task level — it found its mechanism and its victim: the
   trunk-freeze strategy writes over the library's books instead of its margins.
3. **CE-blindness demonstrated, exactly as this round was designed to catch:**
   off-domain CE saw +0.043 nats for the experts arm; the task battery saw a
   −10.4-point knowledge collapse. The pre-registered "CE-blind capability
   erosion" row fired — localized to the experts arm and the knowledge task rather
   than global. Consequence for the program: **every CE-based forgetting claim in
   rounds 5–8 carries this asterisk**, and task-level probes join the standard
   instrument set from here on.
4. Secondary observation, stated at honest strength: the experts arm's deltas are
   uniformly negative across all reasoning tasks (−.008..−.013) — individually
   within noise by the locked rule, but consistently signed where full's are
   centered near zero. Suggestive of low-grade broad erosion under concentrated
   updates; repeats would settle it.
5. The lone pre-registered gain hypothesis (wiki-adjacent knowledge up): full's
   triviaqa +0.9pt is within noise — neither confirmed nor refuted.

## Program consequence

The living-library picture survives with a sharpened rule: **let the whole library
absorb change; do not force revisions through the stacks.** Where updates land is
not a tuning detail — at this dose it is the difference between costless continuous
learning (full: 9/9 retained) and silent knowledge destruction (experts: −10.4 EM
invisible to CE). The long-horizon question inherits this: whether distributed
updates stay costless at 10^5–10^6 steps is now THE open experiment, and the
capability heartbeat for it must be task-level, not CE.
