# ember3_drift — Endogenous vs Exogenous Structural Adaptation Under Continuous Learning

First experiment of the **Ember III** program (continuous learning without snapshots).
Question: is signal **exogeneity** the property that keeps a continuously-adapting system's
representation bounded, or does the existing entropy-derived (endogenous) signal already
behave gate-like?

Substrate: TinyCIMM-Euler as built (`../../tinycimm_euler.py`, untouched). The growth/prune
policy is a swappable controller object — arms differ **only** in the structure controller.
Stream: prime gaps — fixed, external, incorruptible; the model cannot author the
distribution *or* the sampling.

## Arms

| Arm | Controller | Signal class |
|-----|-----------|--------------|
| A (control) | `MathematicalStructureController` as built | Endogenous — activation-complexity + performance-variance thresholds |
| B (treatment) | `ResidualSignalController` | Residual — statistics of the **leak-free prequential error stream only** (no activations, no weight entropy). Realization is exogenous, form is the model's own error term (idea doc §2.3) |
| C (decoupled control) | `DecoupledSurrogateController` | Same decision logic as B, fed a block-shuffled surrogate of B's error trace (same seed) — same marginals, decoupled from this model's state. Separates signal *origin* from signal *information* |

All arms share: identical init (same seed → same weights), same update rule
(`online_adaptation_step`), same stream, same instrumentation cadence.

## Locked definitions (pre-registered before any run, 2026-07-18)

- **Drift** D(t) = 1 − linCKA(H_ref, H_t), where H is the hidden representation
  `relu(xWᵀ+b)` on a **fixed probe set** (256 windows, seen + ahead regions), captured
  side-effect-free (direct matmul, never through `forward()`), reference frozen at
  t = 1,000 tokens. CKA is dimension-agnostic → survives grow/prune (port-audit findings 5–6).
- **Competence** C(t) = MAE of leak-free predictions on a held-ahead probe set the stream
  never reaches, plus rolling prequential MAE on the incoming stream.
- **Plasticity** P(t): every 5,000 tokens, deepcopy the model, adapt the clone 200 steps on
  a fresh never-seen gap segment (a different segment each probe), record relative error
  reduction (first-50 MAE → last-50 MAE), discard clone.
- **Cadence** in stream tokens (probe snapshot every 500), never in steps.

**Classification (fixed in advance):**
- *Reorganization* = D rises while C stable or improving.
- *Collapse* = D rises and C degrades.
- *Ossification* = D flat while P declines — bounded-by-freezing is **not** success
  (that is the snapshot, rediscovered).

**Interpretation table (invariants only, no absolute coordinates):**

| Outcome | Reading |
|---|---|
| A drifts unboundedly; B bounded with retained P | Exogeneity is load-bearing — central result |
| Neither drifts | Entropy signal already gate-like — different paper |
| Both drift | Substrate or metric wrong — diagnostic |
| B bounded but P collapses | Bounded-by-freezing — failure mode, not success |
| B and C behave alike | Origin (outside-ness) is doing the work, information content is not — and vice versa |

## Known caveats (stated up front)

- The substrate's training-path MSE contains ~5% target leakage via
  `higher_order_transform` (audit finding 7). Identical across arms; all *reported*
  errors and the Arm B signal use an external leak-free forward.
- The optimizer's parameter set changes after the first structural event (audit
  finding 10). Identical across arms; fix required before rung 1.
- Arm A's thresholds are the default-constructed controller's (the online experiment's
  tuned overrides never took effect — audit finding 12); "as built" means the code path
  that actually runs.
- Input is a 4-gap sliding window (repo convention from `create_sequences`), not the
  online script's single-gap input — richer context for a continuous run.

## Run

```
cd scripts
python run_arms.py --smoke            # Arm A, 1 seed, 2k tokens — harness shakedown
python run_arms.py --all              # 3 arms x 3 seeds x 50k tokens (B before C per seed)
python plot_results.py                # results/summary.png + results/summary_stats.json
```

Per-run outputs in `results/runs/{arm}_{seed}/`: `timeseries.csv` (per-token),
`snapshots.csv` (drift/competence cadence), `plasticity.csv`, `events.csv`
(grow/prune), `meta.json`, and for Arm B `error_trace.npy` (consumed by Arm C).

## Results (2026-07-18)

Two findings, both documented in `journals/`:

**1. Baseline (as-built substrate): total collapse — substrate diagnostic row.**
All 9 runs collapse to the best-constant predictor (held-ahead MAE 7.92 vs floor 7.91;
never beats the causal rolling-median surface reader; late plasticity *negative*).
Mechanism confirmed by diagnostics: ReLU death is an absorbing state on the all-positive
gap stream (`const_frac → 1.0`, `h_std → 0`); growth injects live neurons slower than the
dynamics kill them. Design principle extracted: **a continuous learner's update dynamics
must have no absorbing states.** Record: `results/runs/`, `results/summary.png`.

**2. Repaired substrate (norm + leaky 0.01): the arms do not separate — the null with
teeth.** All arms recover a genuine early edge over the rolling-median comparator
(+1.7…+1.9% first 10%) and erode identically (−2.6…−3.0% last 10%); drift, plasticity,
and capacity trajectories are statistically indistinguishable across A, B, **and C**
(final drift A 0.58 / B 0.59 / C 0.43 with overlapping seed ranges; plasticity curves
superimpose). Since C consumes a *decoupled shuffled* signal, even the information
content of the structure signal is causally irrelevant to drift here — not just its
origin. **Localization: at this rung, drift and competence erosion live in the per-token
gradient channel, which is identical across arms. Structure policy is a second-order
lever.** Record: `results/runs_stab/`, `results/summary_stab.png`.

**4. Round 3 (2026-07-19): learned gates — the authorability result; gravity refuted.**
Identical REINFORCE gate learners differing only in reward channel: **L-self (rewarded
on the model's own self-measured loss) is significantly worse than a rate-matched
random gate** (late edge −5.0% vs −2.9%; worst held-out MAE in the program, 8.59) — it
*learned* something that damages realized performance, with one seed rediscovering the
ossification profile via reward maximization. L-outer (rewarded on realized error)
beats L-self on held-out MAE 3/3 paired seeds but does not recover the fixed err gate
— the fixed "update on surprise" rule remains champion. Separately: gravity's
1-seed drift-halving **did not replicate** at 3 seeds (final-snapshot drift is too
seed-noisy to rank mechanisms — switch to drift AUC), and the erosion-source analysis
shows P(t) flat: the steady-state deficit is non-stationarity tracking cost, not
plasticity loss. See `journals/2026-07-19_round3_learned_gates.md`.

**Implication for Ember III:** the exogeneity question must be re-aimed one level down —
gate the *per-token update application* (exogenous outcome gate vs endogenous entropy
gate vs rate-matched random gate), not the capacity decisions. This matches the idea
doc's own nociception framing (§2.2): pain gates the learning signal, not the neuron
count.

Metric refinement noted for the next round: plasticity probes use a different fresh
segment per event, so segment difficulty dominates the P(t) shape (all arms track it
identically). Fix the probe segment or difficulty-normalize before the update-gating
round. _(Fixed in round 2 via `--fixed-plast`.)_

**3. Round 2 (2026-07-19): gate-origin separation — the thesis's first positive
result.** CIMM-stabilizer ablation (2^4 grid): gravity pullback is the standout
single mechanism (halves drift with qclip; needs 3-seed confirmation); the pre-stated
winner rule selected the plain substrate for the origin arms. Gate-origin arms
(per-token update gate at matched ~50% rate, only the signal origin differing):
**exogenous (realized-error) > random > endogenous (entropy), non-overlapping seed
ranges on late edge and plasticity.** The endogenous gate drifts least while
performing worst — the exact ossification failure mode the paired-metric design was
built to catch. Scope: alignment evidence at fixed gates, not yet the
learned-gate/authorability claim. See `journals/2026-07-19_round2_gates.md`,
`results/gate_origin_summary.csv`, `results/ablation_summary.csv`.
