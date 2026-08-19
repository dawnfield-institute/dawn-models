# Round 4: temporal vs spatial specialization — adaptation beats the oracle-routed frozen pair at every dwell

**Date:** 2026-07-19
**Runs:** `results/runs_regime/` (4 arms x dwell {250, 1000, 4000} x 3 seeds, 24k-token
alternating streams), `results/regime_summary.csv`. Design pre-registered in
`scripts/run_regimes.py` docstring before any full run.

## Setup

Two structurally distinct regimes at matched scale (per-regime causal z-scoring):
A = prime gaps (noisy, heavy-tailed, drifting), B = harmonic mix (smooth, stationary).
Regimes continue across visits (no replays). Arms on the champion substrate:
ADAPT (one network, continuous, ungated), ADAPT_EG (+ exogenous err-gate),
ROUTED (per-regime specialists pre-trained 10k tokens, frozen at their BEST rolling-MAE
state, oracle-routed — the upper bound for any learned router), FROZGEN (frozen mixed
generalist).

Pre-flight finding that reshaped the design: the substrate's online loop never
settles, so freezing "wherever it happened to be" is a strawman specialist —
best-snapshot selection added (moved frozen harmonic MAE 0.75 → 0.41).

## Results (normalized MAE, mean of 3 seeds; frozen arms dwell-invariant by construction)

| arm | D=250 | D=1000 | D=4000 | regime A | regime B (250/1000/4000) |
|---|---|---|---|---|---|
| **ADAPT** | **0.693** | **0.681** | **0.652** | 1.24 | **0.139 / 0.119 / 0.069** |
| ADAPT_EG | 0.843 | 0.723 | 0.673 | 1.23 | 0.460 / 0.216 / 0.119 |
| ROUTED (oracle) | 0.824 | 0.824 | 0.824 | 1.20 | 0.444 |
| FROZGEN | 0.874 | 0.874 | 0.874 | 1.28 | 0.471 |

**ADAPT recovery after a switch:** median ≤50 tokens (D=250; at instrument floor),
103 (D=1000), 111 (D=4000). **Retention cost on re-entry** (first-100 MAE minus
previous steady): +0.03 / +0.23 / +0.35 — forgetting is real and dose-dependent
(deeper specialization = more interference), but small against tracking gains once
amortized.

## Findings

1. **The pre-registered "ADAPT wins everywhere" branch fired — no crossover found
   down to D=250.** A single continuously-adapting network beats the oracle-routed
   frozen pair at every tested dwell. If a crossover dwell exists on this substrate,
   it is below 250 tokens.
2. **The per-regime check clears the honest hazard**: the win is NOT frozen-specialist
   decay on the drifting prime regime (regime A is at parity; ROUTED slightly best
   there). The entire margin is regime B: live local re-fit (0.07–0.14) vs the best
   frozen snapshot (0.44).
3. **Mechanism: adaptation substitutes for representation.** The adaptive model does
   not internalize the waveform; it tracks it — every prediction leans on corrections
   from the immediately preceding tokens (surfing the phase). The frozen model must
   BE the function; the adaptive model only has to STAY NEAR it. This is the
   architecture-level instance of the encoding-ladder escape: the cache of
   specialists (MoE) stores artifacts; the adapter is a generator of specialization.
4. **Exogenous gating breaks under regime mixing without local calibration** —
   instrumented probe: at D=250 the gate is open 68% during primes but **16% during
   harmonic** (the 200-token rolling median straddles regimes; large prime errors
   drown the harmonic regime's surprise scale → update starvation → B-MAE as bad as
   frozen). At D=4000: 51%/44%, healthy. Lesson: **surprise thresholds must be
   regime-local** — and a routed architecture provides exactly that for free, so the
   plastic-experts hybrid earns its keep twice (interference containment + local gate
   calibration).
5. Frozen arms' dwell-invariance (identical MAE across dwells) is a designed
   consistency check — each regime's pointer serves identical data regardless of
   interleaving — and it passed exactly.

## Scope and caveats

- Toy substrate; regime B is near-deterministic — the easiest case for tracking.
  Real regimes with rich internal structure may reward representation more than
  local re-fit; the claim established here is the *mechanism and its economics*
  (recovery ~10² tokens, dose-dependent forgetting, no crossover ≥250), not
  MoE-is-unnecessary at scale.
- ROUTED's specialists inherit the substrate's so-so converged quality even with
  best-snapshot selection; a stronger training recipe would raise the frozen bar.
  Oracle routing, however, means the routing side cannot be improved.
- Regime A is noise-dominated (all arms near floor) — a two-structured-regime
  version would sharpen the specialization contrast.

## Program state

Peter's conjecture — "it can become whatever it needs to be for a given moment,
so you may not need a mixture of experts, while keeping the architecture's useful
pieces" — now has its first quantitative footing: at this scale, temporal
specialization dominates spatial specialization at every tested switching rate,
with measured switch costs. The synthesis target it points at: **few plastic
experts, router as regime detector, per-regime exogenous gate thresholds** —
routing for interference containment and gate calibration, adaptation for
currency. Next natural steps: two-structured-regimes variant; regime-local gate
windows for EG; the PAC question (conservation as the budget governing how much
shared core a re-specialization may spend) now has a concrete measured quantity —
the dose-dependent retention cost — to bound.
