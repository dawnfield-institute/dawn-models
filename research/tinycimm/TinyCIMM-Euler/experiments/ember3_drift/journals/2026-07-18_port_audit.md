# Ember III Port Audit — TinyCIMM-Euler + SCBF bounded-run assumptions

**Date:** 2026-07-18
**Scope:** `tinycimm_euler.py`, `run_online_experiment.py`, `scbf/` (per Ember III idea doc §5.3 / §8.1)
**Regime shift being audited for:** fixed-parameter, bounded-run (500–10k step) assumptions that break under continuous update with structural change.

All line references are to `dawn-models/research/tinycimm/TinyCIMM-Euler/tinycimm_euler.py` unless noted.

## Findings

### Unbounded accumulation (memory leaks under continuous run)

1. **`MathematicalStructureController.decide()`** appends to `complexity_hist` / `performance_hist` / `structure_hist` every call (L845–847) and never trims. The balance histories *are* trimmed to 50 (L796–799) — the leak is only in the decide-path lists. Harmless at 10k steps; unbounded at stream scale.
2. **`UnifiedSymbolicCollapseTracker`** — `raw_entropies` (L157) and `smoothed_entropies` (L167) append per step with no trim, despite a nominal `memory_window=30`. Same pattern for collapse-event lists.
3. **`HigherOrderEntropyMonitor`** — `past_metrics` and `order_history` unbounded (L654–655); worse, `get_variance()` (L658) computes over the *entire* history, so its cost grows linearly with run length and its semantics silently shift from "recent variance" to "lifetime variance."
4. **`SCBFExperimentLogger.logs`** (scbf/loggers/experiment_logger.py) grows unbounded.

### Measurement state destroyed by structural change

5. **`math_memory` / `micro_memory` are cleared on every grow/prune** (L1141–1142, L1150–1151). Any tracker that reads them loses continuity *exactly at the events Ember III cares about*. Consequence adopted as a design constraint: drift measurement must live **outside the model** (probe-set CKA), never inside its memory buffers.
6. **Prune re-indexes neurons** via topk `keep_indices` (L1203–1209). Any external per-neuron bookkeeping breaks silently after the first prune. Second design constraint: the drift metric must be **dimension-agnostic** (CKA over probe Gram matrices, not neuron-indexed similarity).

### Signal integrity

7. **Target leakage in the training-path prediction.** `forward(x, y_true)` applies `higher_order_transform` (L1017–1018, defined L663–669), nudging the prediction ~5% toward the target. `online_adaptation_step` computes its MSE on this *corrected* prediction (L1344–1347). So the loop's own "error" understates true prequential error. The harness computes leak-free prequential error externally (direct `relu(xWᵀ+b)Vᵀ+c`, no correction) for all logging and for the Arm B signal.
8. **`forward()` has side effects** — appends to `micro_memory`/`math_memory`, advances `complexity_factor` (L998–1026). Probing through `forward()` would perturb the system being measured. Harness probes bypass `forward()` entirely (direct matmul, `no_grad`).
9. **`forward_with_qbe` returns `last_h` as "activations"** (L1409–1410) — the hidden state of the most recent forward, whatever it was. Fine in a strict one-token loop; a footgun the moment anything else calls `forward()` in between (e.g. probes through the front door — see 8).

### Update-rule drift

10. **Optimizer party changes after the first structural event.** Init optimizer covers `[W, b, V, c]` only (L944–949); the rebuilt optimizer after grow/prune *adds* `higher_order_processor` params (L1186–1188, L1217–1219). The effective learning rule silently changes partway through a run — a bounded-run assumption nobody hit because runs were short. Accepted as-built for the shakedown (identical across arms, so it cannot confound the A/B/C comparison), but must be fixed before any ladder rung.
11. **`torch.compile(self)` in `__init__` is a no-op** (L984 assigns to the local `self`). The "successfully compiled" message is misleading; harmless.

### Dead config / logging hygiene

12. **`run_online_experiment.py` builds a `MathematicalStructureController` with tuned thresholds (L178–183) that is never used** — `online_adaptation_step` reads `model.structure_controller` (L1383–1384), the default-constructed one. The online experiment's threshold overrides never took effect. (Ember III harness sets `model.structure_controller` directly, which is the real hook.)
13. **Per-token `print` in `decide()`** (L880–892) — console I/O every step; at stream scale this is a log firehose. Harness redirects stdout around model calls rather than editing the substrate.

## Verdict

Code ports; measurement regime does not (as the idea doc predicted, §5.3). The trackers are the gap. None of the above blocks the shakedown: findings 1–4 are tolerable at 50k tokens (memory stays in the tens of MB), 5–9 are routed around by external probe-based instrumentation, 10 is arm-invariant, 12–13 are handled in the harness. All must be addressed in a real port before rung 1 (160M).
