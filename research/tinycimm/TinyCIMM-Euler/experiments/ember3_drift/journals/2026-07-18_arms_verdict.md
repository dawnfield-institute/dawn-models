# Arms verdict on the repaired substrate: the null with teeth

**Date:** 2026-07-18
**Runs:** `results/runs_stab/` — 3 arms x 3 seeds x 50k tokens, StabilizedTinyCIMMEuler
(norm + leaky 0.01), harness 2026-07-18.3. Plot: `results/summary_stab.png`.

## Numbers (means over 3 seeds)

| | Arm A (endogenous) | Arm B (residual) | Arm C (decoupled surrogate) |
|---|---|---|---|
| final drift (1−CKA) | 0.575 [0.50–0.67] | 0.592 [0.57–0.63] | 0.427 [0.31–0.53] |
| late drift slope /1k tokens | +0.002 | +0.007 | +0.005 |
| stream edge vs rolling-median, first 10% | **+1.7%** | **+1.8%** | **+1.9%** |
| stream edge, last 10% | −3.0% | −2.6% | −2.7% |
| plasticity first → last | −0.05 → −0.13 | −0.03 → −0.11 | −0.02 → −0.09 |
| held-ahead MAE best / final (floor 7.91) | 7.85 / 7.88 | 7.86 / 7.91 | 7.86 / 7.90 |
| grow / prune events | 21.7 / 0 | 27.0 / 3.7 | 29.0 / 6.0 |
| final neurons | 134–136 | 129–139 | 128–138 |

## Reading against the pre-registered table

The outcome is the **"both drift" row**, but with a decisive addition the original table
didn't anticipate: **B ≈ C**. The residual-signal arm and the decoupled-surrogate arm —
same decision logic, one fed the model's real error stream, one fed a block-shuffled
copy — produce indistinguishable drift, erosion, and plasticity. So at this rung, not
only does the *origin* of the structure signal not matter; its *information content*
doesn't either. The structure controller is causally irrelevant to drift here.

Positive control that the experiment was live: all arms show a genuine early edge over
the causal rolling-median comparator (the repaired substrate really learns), and all
arms erode it on the same trajectory. The drift/erosion is carried by the per-token
gradient channel — identical across arms by design — not by the dozens of grow/prune
events. The plasticity panel makes this visually exact: three superimposed curves.

**Localization result: at TinyCIMM scale on this stream, the locus of drift is the
weights, not the architecture.** Capacity policy is a second-order lever.

## Secondary observations

- Arm A stopped pruning entirely on the repaired substrate (18 prunes/run at baseline →
  0). The endogenous controller's policy is not scale-invariant — normalization moved
  its operating point. B/C's CV-based logic is scale-invariant by construction. Any
  future endogenous-vs-exogenous comparison must control for scale sensitivity of the
  policy itself.
- C's lower mean drift (0.43 vs ~0.58) is within overlapping seed ranges; if real at
  all, the candidate mechanism is its higher prune rate (pruning discards low-importance
  directions and re-anchors the representation). Not claimed; noted for the record.
- Held-ahead MAE dips marginally below the best-constant floor at best (7.85 vs 7.91) —
  the repaired substrate transfers *slightly* ahead, transiently. The stored structure
  is mostly scale-local (artifact, not generator, in the encoding-ladder sense).
- Plasticity P(t) oscillates with probe index identically across arms: each probe uses a
  different fresh segment, so segment difficulty dominates. **Metric fix before the next
  round: fixed probe segment (or difficulty-normalized P).**

## What this buys the program

The shakedown's contribution is a *localization*: the degeneracy that matters lives in
the per-token update channel. The exogeneity hypothesis is therefore untested — not
refuted — because the arms pulled a lever that turns out not to be connected to the
failure mode. Next experiment, same three-arm logic moved one level down:

- **Gate the per-token update application** (skip/apply, or scale) —
  Arm A′: endogenous gate (entropy/confidence-derived);
  Arm B′: exogenous gate (realized-error percentile);
  Arm C′: random gate, rate-matched to B′.
- Same substrate, same instrumentation, plus the plasticity-metric fix above.
- This is the nociception architecture from the idea doc §2.2 taken literally: the gate
  acts on the learning signal itself. The doc's gating question ("who trains the gate")
  stays out of scope for the first round — fixed gates first.

At ~90s per run on this machine, the update-gating round is an afternoon, not a rung.
