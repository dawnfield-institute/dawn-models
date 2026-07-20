# Round 10 verdict: knowledge erosion confirmed; the mechanism is compensation, not pressure

**Date:** 2026-07-20
**Instruments:** SCBF v2 (first use) — ClozeKnowledgeProbe + HookSpine stacked-tensor
telemetry. Pre-registration: `2026-07-20_round10_prereg.md` (locked before runs).
Results: `results/results_cloze/`, `results/results_expertmass/`.

## Experiment A — cloze probe: the deflation is dead

Generation-free scoring on the same 2,000 TriviaQA facts (answer log-likelihood +
first-token top-1; format shift impossible by construction):

| model | mean answer LL | top-1 |
|---|---|---|
| frozen | −1.724 | .438 |
| experts-only | **−1.954** | **.373** |
| full | **−1.664** | **.458** |

Paired vs frozen: experts **ΔLL −0.230 (2×SE 0.023 — 10× beyond threshold)**,
top-1 −6.6pt → the pre-registered "knowledge erosion confirmed at full strength"
row fired. And the bonus row too: **full is significantly POSITIVE (ΔLL +0.060,
top-1 +2.0)** — distributed continuous wiki-reading *added* fact knowledge while
destroying none. The continuous-learning claim in one line: it read the
encyclopedia and now knows more trivia.

## Experiment B — expert-mass telemetry: the hypothesized mechanism is REFUTED, and the null is the discovery

Pre-registered scoring: (1) mass-tracks-usage: per-layer Spearman mean **0.756**
(≥0.5 supportive; short of the 0.8 "unambiguous" bar) ✓. (2) amplification: mass
top-26% share 43.4% vs usage 42.9% — **tracks, does not amplify**. (3) contrast:
**NULL — the full arm's expert update-mass field is statistically identical to the
experts arm's** (total 464 vs 470, ratio 0.987; per-(layer,expert) correlation
0.981; routing itself barely moved in 5k steps, selection correlation 0.979).

**Consequence: the "frozen router aims the firehose" story is dead, and so is
round 9's "concentrating updates concentrates the damage" in its literal form.**
Both arms hammered the same experts with the same force. The only difference was
whether the ~5% of parameters outside the experts (attention, embeddings, norms,
router) could co-adapt. Same perturbation: destructive when the surrounding
system is pinned, benign — beneficial, even — when it is free to move.

**Refined mechanism (the round-10 finding): uncompensated updates damage;
compensated updates don't.** Stored knowledge in a network is a joint property of
expert storage and trunk readout. Round 9's lesion localized the *vulnerability*
(storage whose readout cannot follow), not storage per se — the classic subtlety
of lesion studies in neuroscience (local function vs network compensation),
reproduced by our gradient-pressure instrument in its second use. The instrument
is sharper than its first interpretation; the record now carries both.

This lands squarely in the institute's home territory: it is a compensation
result — the same theme as Ember v0's measured head-compensation (72% of heads
compensating under the PAC budget) and the conservation framing generally. The
stability of a continuously-learning system is not about shielding its storage;
it is about preserving the system's freedom to redistribute around change.

## Program state

- Continuous learning verdict (rounds 8–10, one paragraph): a pretrained 7B MoE
  under distributed continuous learning gains stream competence AND fact
  knowledge with zero measured capability loss (9/9 benchmarks, cloze positive)
  at the 5k-update dose; constraining adaptation to the knowledge-bearing
  submodule destroys facts (−10.4 EM / −0.23 LL) invisibly to CE metrics; the
  destructive ingredient is broken co-adaptation, not update pressure.
- SCBF v2 validated in first contact: both instruments produced decisive,
  pre-registered verdicts; the stacked-tensor telemetry (round-8 bug fix) worked
  (masses nonzero, unit-tested against manual norms, VRAM 16.3GB).
- Open: dose-response of the compensation result (does full stay clean at 50k
  updates? — folds into the long-horizon resident experiment); partial-trunk
  lesions (which trunk component carries the compensation — router? attention?
  norms?) — a one-afternoon lesion sweep now that the instrument exists;
  round-9/10 asterisk propagation to any external claims.
