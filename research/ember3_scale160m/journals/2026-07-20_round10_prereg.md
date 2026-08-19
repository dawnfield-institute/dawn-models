# Round 10 pre-registration: cloze deflation-closure + expert-mass mechanism test
(locked before either experiment runs)

**Date:** 2026-07-20. SCBF v2 is built (spec `scbf/.spec/scbf-v2.spec.md`, 11/11
unit tests green); these two experiments are its first users and its validation.

## Experiment A — cloze knowledge probe (`exp_cloze.py`)

Question: was round 9's experts-arm TriviaQA collapse (−10.4 EM) knowledge erosion
or output-format shift? Instrument: generation-free per-fact answer log-likelihood
(+ first-token top-1) on the SAME TriviaQA validation facts (first 2000), for
{frozen, experts, full}. Paired per-fact deltas vs frozen; threshold 2x paired SE.

| Outcome | Reading |
|---|---|
| experts ΔLL markedly negative (beyond 2xSE) AND full ΔLL ≈ 0 | **Knowledge erosion confirmed at full strength** — the lesion reading stands; format-shift excluded |
| experts ΔLL ≈ 0 (or ≈ full's) | Round 9's EM drop was **format shift**; lesion claim withdrawn to "output-behavior erosion"; the gradient-pressure-lesioning instrument needs LL-based batteries, not EM |
| both ΔLL negative similarly | Wiki adaptation costs fact-LL generally; round 9's EM gap then reflects format robustness differences — mixed reading, both claims weakened |

## Experiment B — per-expert update-mass telemetry (`exp_expertmass.py`)

Mechanism under test: **frozen-router × usage-skew** — with routing pinned, update
pressure lands proportionally on the highest-usage experts (the knowledge stores).
Instrument: HookSpine stacked-tensor row-norm masses (the round-8 fix; unit-tested
against manual norms) + router selection counts, re-running round-8's arms
(experts primary, full contrast; 5k chunks each, deterministic settings).

Pre-registered predictions:
1. Experts arm: per-layer Spearman(update mass, selection count) strongly positive
   (mean ρ ≥ 0.8 would be unambiguous; ≥ 0.5 supportive).
2. Mass concentration ≥ usage concentration (top-26% share of mass ≥ 42.7%-ish
   usage share) — pressure at least tracks, plausibly amplifies, the skew.
3. Contrast (exploratory, not binding): full arm shows equal-or-lower mass
   concentration / correlation to initial usage — a plastic router redistributes.

Sanity: selection counts must reproduce round-8's saved sel_counts pattern
(same stream, same seed); HookSpine VRAM ≈ round-8's 15.3GB.
