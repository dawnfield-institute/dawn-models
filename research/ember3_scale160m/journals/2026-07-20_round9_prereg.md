# Round 9 pre-registration: task-level retention benchmark (locked before any battery run)

**Date:** 2026-07-20, written while the save re-runs execute and BEFORE any benchmark
is run.

## Why this round exists

Every competence number in rounds 5–8 is CE-family. CE and capability can
dissociate: zero-shot task performance depends on tail behaviors that average CE
smooths over. Round 8's "keeps its learnings" verdict (off-domain CE +0.03) is
therefore unverified at the task level. This round can OVERTURN round 8 — that is
its purpose.

## Design

- **Checkpoints:** frozen OLMoE-1B-7B-0924 (control) vs experts-adapted vs
  full-adapted — deterministic re-runs of round 8's arms (5k WT103 chunks,
  fused-backward SGD lr 5e-4) saved in HF format. (Round 8 measured trajectories
  and discarded weights; re-run jitter is bounded by rounds 5–6 measurements.)
- **Battery** (lm-eval-harness, identical invocations for all three models):
  arc_easy, arc_challenge, hellaswag, piqa, winogrande, boolq, lambada_openai
  (full sets); triviaqa with --limit 2000 (runtime; same limit all models →
  internally comparable). MMLU excluded: near-chance at this model scale, large
  runtime, low information.
- **Reading rule (locked):** per task, Δ = adapted − frozen. No-change iff
  |Δ| ≤ 2·sqrt(se_frozen² + se_adapted²). Pattern verdicts:
  | Outcome | Reading |
  |---|---|
  | All reasoning tasks no-change (either adapted model) | Task-level retention confirmed — round 8's verdict survives its strongest test |
  | Consistent signed drops across ≥3 reasoning tasks | **CE-blind capability erosion — round 8's verdict overturned**; CE-family forgetting metrics insufficient (major program finding) |
  | triviaqa / lambada up, reasoning flat | The lone pre-registered gain hypothesis (wiki-adjacent knowledge) confirmed |
  | experts retains where full drops | Task level partially rehabilitates the trunk-freeze intuition round 8 inverted |
- **Not claimed in advance:** any expectation that adaptation improves reasoning
  benchmarks. The claim under test is asymmetric: stream gains (already measured)
  at zero task-level cost (measured here).

## Anchors

Frozen-model scores must land in the published OLMoE-1B-7B ballpark (sanity that
the harness is wired right); published base-model reference points: ARC-c ~ low
40s, HellaSwag ~ high 70s (acc_norm), PIQA ~ 80. Gross deviation = harness bug,
halt and fix before reading any deltas.
