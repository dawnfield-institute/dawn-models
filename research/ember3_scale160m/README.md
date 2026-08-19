# ember3_scale160m — Gate-Origin Arms at 160M (Ember III Rung 1)

Round 5 of the Ember III program; first rung of the idea doc's scaling ladder (§6).
Question: **does the gate-signal origin ordering (exogenous > random > endogenous),
established at TinyCIMM scale in `../tinycimm/TinyCIMM-Euler/experiments/ember3_drift/`
rounds 2–3, survive a real 160M transformer under continuous next-token learning on
real text?**

Substrate: `EleutherAI/pythia-160m` (Ember-v1 deliberately deferred to the follow-up
round — "does the PAC-native architecture change drift behavior?" — so this rung has a
standard, comparable baseline). Hardware: node1's RTX 3090, inside CT103 (gpu-lab),
all artifacts confined to `/data/ember3` (root disk is nearly full — do not touch it).

## Pre-registered design (locked 2026-07-19, before any full run)

- **Stream**: WikiText-103 train split in document order (fixed, non-authorable
  distribution AND sampling), pre-tokenized to uint16, served as 512-token chunks;
  10k chunks (~5M tokens) per run.
- **Update rule**: full-rank Adam (lr 1e-5) — no LoRA (POC 4: low-rank confounds).
  Prequential chunk loss measured before the update; one update per accepted chunk.
- **Arms**, per-chunk gate at ~50% rate by construction:
  `none` (ungated) · `err` (prequential loss ≥ rolling median-200 — exogenous) ·
  `ent` (mean predictive entropy ≥ rolling median-200 — endogenous) · `rand` (coin).
- **Seeds**: 0, 1 pre-registered; extend to 2 if per-run ≤ 20 min.
- **Drift**: linear CKA vs post-warmup reference (chunk 250), fixed 64-sequence probe
  batch, layer-6 mean-pooled hiddens, every 100 chunks.
- **Competence**: prequential edge vs the frozen pretrained model on identical chunks
  (single reference pass), plus held-out perplexity on a fixed validation batch.
- **Plasticity**: every 2.5k chunks, clone adapts 100 chunks on one fixed far-ahead
  unseen segment (round-2 metric fix), loss-improvement rate, clone discarded.

**Interpretation table (invariants only):**

| Outcome | Reading |
|---|---|
| err > rand > ent on late prequential edge | Round-2b ordering replicates at 160M — the exogeneity result generalizes across ~4 orders of magnitude |
| All gates ≈ each other, ≠ ungated | Update thinning is the lever at this scale, not signal origin |
| Nothing separates | Gate effects are toy-scale artifacts — bounds the program honestly |
| ent lowest drift + worst competence | The ossification signature transfers |

## Run

```
scripts/remote.ps1 push        # scp scripts to CT103:/data/ember3
scripts/remote.ps1 prep        # one-time: download + tokenize WT103, build manifest
scripts/remote.ps1 preflight   # 1 arm x 200 chunks sanity
scripts/remote.ps1 matrix      # 4 arms x seeds, sequential, nohup
scripts/remote.ps1 pull        # fetch results/ CSVs back
python scripts/analyze_scale.py
```

## Results (2026-07-19)

**The toy ordering did not replicate — a sharper structure appeared.** Both informed
gates (err, ent) lose to the rate-matched random gate on stream edge AND held-out
(non-overlapping incl. the CUDA-jitter repeat); err ≈ ent on competence; but the
endogenous gate produces 5–7x the representational drift of every other arm with no
competence divergence (churn, not ossification — the toy signature inverted).
Random 50% thinning matches ungated held-out generality with half the updates.

Mechanistic hypothesis (stated as such): toy surprise was epistemic (learnable);
WikiText's high-loss tail is heavily aleatoric, so difficulty-threshold gates spend
updates on the least transferable text. **Refined thesis: the gate signal must be
exogenous AND the selection rule must separate reducible from irreducible surprise
— gate on learnability, not difficulty.** Full readout:
`journals/2026-07-19_rung1_verdict.md`, `results/scale160m_summary.csv`.

**Rounds 7–10 (2026-07-19/20): the physics twins, the MoE continuous learner, the
retention verdict, and the compensation law.** Round 7: PAC conservation is stable —
tightening 30–40% — under 10k continuous updates (physics twins; gate phenomenology
architecture-robust). Round 8: OLMoE-1B-7B becomes a continuous learner in 15.3GB
VRAM (fused-backward SGD); full adaptation beats experts-only on both axes. Round 9
(pre-registered battery): full retains 9/9 tasks; experts-only destroys 10.4pt of
TriviaQA invisible to CE — **CE-blindness demonstrated**. Round 10 (SCBF v2 first
use): erosion confirmed generation-free (ΔLL −0.230, 10× threshold), the full arm
GAINS fact knowledge (+0.060), and the mechanism control refutes the concentration
story — identical expert update mass in both arms (ratio 0.987) → **the compensation
law: uncompensated updates damage; compensated updates don't.**

Consolidated map: `journals/2026-07-20_program_synthesis.md`. Founding doc +
empirical addendum: `cradle/docs/ember3_continuous_learning.md` +
`cradle/docs/ember3_addendum_2026-07-20.md`. Instruments:
`../scbf/v2/` (spec: `../scbf/.spec/scbf-v2.spec.md`).

**Round 6 (band + excess gates): directionally confirmed, random still unbeaten.**
Both tail-aware rules beat the difficulty gates decisively (band +1.1pp, excess
+0.7pp edge over err; held-out and plasticity recover to rand levels) — but neither
crosses random. Ladder: rand ≥ band > excess > err ≈ ent. The `excess` gate
(doubly-exogenous learnability: current loss minus frozen-model loss on the same
chunk) is the **stability champion** — lowest drift AUC of all arms at 97% of rand's
edge. Residual reading: random's active ingredient is *coverage*, which no per-chunk
scoring rule provides. Instrument finding: drift AUC is jitter-bistable under some
gates (band repeats: 0.023 vs 0.355 at identical competence) → single-run drift
values untrustworthy at 160M; round-5's "ent churns 5–7x" claim downgraded
accordingly. See `journals/2026-07-19_round6_band_excess.md`.
