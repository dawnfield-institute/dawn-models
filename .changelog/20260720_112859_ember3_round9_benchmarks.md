# Ember III round 9 — task-level retention battery: full retains 9/9; experts arm loses 10.4pt TriviaQA that CE never saw

**Date:** 2026-07-20
**Location:** `research/ember3_scale160m/` (scripts `bench.sh`, run_moe.py `--save`)

## What was added

- Round-9 benchmark battery: lm-eval 0.4.12, 8 tasks (arc_e/c, hellaswag, piqa,
  winogrande, boolq, lambada, triviaqa@2000) x {frozen, experts-adapted,
  full-adapted} OLMoE-1B-7B. Pre-registered rules locked before any run
  (`journals/2026-07-20_round9_prereg.md`). Adapted checkpoints = deterministic
  round-8 re-runs saved in HF format (reproduced to 3rd decimal).

## Findings (journal: journals/2026-07-20_round9_verdict.md)

1. **Full-adaptation arm: 9/9 no-change by the locked 2xSE rule** — task-level
   confirmation that the MoE continuous learner keeps its learnings at this dose.
2. **Experts-only arm: TriviaQA −10.4pt (.481→.376)** — expert FFNs hold the
   MoE's factual knowledge; concentrating all update pressure into them overwrote
   it. Round 8's "concentrating updates concentrates the damage" confirmed with
   mechanism and victim. Trunk-freezing is harmful, not just suboptimal, here.
3. **CE-blindness demonstrated**: the same arm showed only +0.043 nats off-domain
   CE. All CE-based forgetting claims (rounds 5-8) now carry this asterisk;
   task-level probes join the standard instrument set.
4. Sanity anchors passed (frozen scores at published ballparks); experts arm shows
   a uniformly-negative sub-threshold trend across reasoning tasks (noted, not
   claimed).

Infra stumbles logged: lm_eval missing `accelerate` dep; nested-quoting variable
loss (fix: bench.sh script file, CRLF-stripped). ~2.5h GPU. Not committed.
