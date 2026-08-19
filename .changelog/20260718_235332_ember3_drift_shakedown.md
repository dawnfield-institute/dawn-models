# Ember III drift shakedown — first experiment (ember3_drift)

**Date:** 2026-07-18
**Location:** `research/tinycimm/TinyCIMM-Euler/experiments/ember3_drift/`
**Program:** Ember III (continuous learning without snapshots) — first empirical rung.

## What was added

- New experiment `ember3_drift` (standard structure: meta.yaml, README, scripts/,
  results/, journals/) testing endogenous vs exogenous vs decoupled-surrogate
  structural-adaptation signals under continuous online learning on the prime-gap
  stream. Drift definition (1−CKA on fixed probes + competence + plasticity) locked
  and pre-registered before any run.
- `scripts/`: swappable structure controllers (ResidualSignalController,
  DecoupledSurrogateController), side-effect-free instrumentation (linear CKA, probe
  harness, plasticity probe, rep-health diagnostics), deterministic runner, plotting.
- `scripts/stabilized.py`: StabilizedTinyCIMMEuler subclass (leaky activation) —
  **tinycimm_euler.py untouched**; normalization is harness-level.

## Findings (journals have full detail)

1. **Port audit** (13 findings): unbounded tracker/controller accumulation, target
   leakage via higher_order_transform, measurement state wiped on structural change,
   optimizer parameter-set change after first grow/prune, dead controller config in
   run_online_experiment.py.
2. **Baseline collapse**: the as-built substrate collapses to the best-constant
   predictor in all 9 runs — ReLU death is an absorbing state on the all-positive gap
   stream; late plasticity goes negative (anti-learning). Never observed historically
   because no run exceeded ~10k steps. Design principle: continuous learners' update
   dynamics must have no absorbing states.
3. **Arms verdict on repaired substrate (norm + leaky)**: real early competence edge
   (+1.7–1.9% over causal rolling-median), identical erosion and drift across all
   three arms, **B ≈ C** — structure-signal origin AND content causally irrelevant to
   drift at this rung. Localization: drift lives in the per-token gradient channel.
   Next round re-aims exogeneity at per-token update gating.

## Records

- Baseline: `results/runs/`, `results/summary.png`, `results/summary_stats.json`
- Repair probes: `results/runs_r1/`, `results/runs_r2/`
- Repaired matrix: `results/runs_stab/`, `results/summary_stab.png`,
  `results/summary_stats_stab.json`

No changes to existing dawn-models code. Not committed (per agent policy).
