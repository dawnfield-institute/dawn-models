# Ember III — Program Synthesis, Rounds 1–10 (2026-07-18 → 07-20)

The consolidated map of the first ten rounds. Doc-level analysis lives in
`cradle/docs/ember3_addendum_2026-07-20.md`; per-round detail in the journals
referenced below. This file is the index.

## Round table

| # | Substrate | Question | Verdict | Journal |
|---|---|---|---|---|
| 1 | TinyCIMM-Euler | Shakedown + capacity-signal arms | Substrate collapses (ReLU death = absorbing state); repaired (norm+leaky); capacity arms null incl. decoupled surrogate → drift lives in per-token updates | `ember3_drift/journals/2026-07-18_shakedown_baseline.md`, `..._arms_verdict.md`, port audit |
| 2 | TinyCIMM-Euler | Fixed update-gate signal origin | err −2.3% > rand −2.9% > ent −4.7% late edge, non-overlapping; ent = ossification corner | `..._round2_gates.md` (2a ablation: gravity "standout" flagged) |
| 3 | TinyCIMM-Euler | Learned gates (authorability) + gravity confirm + erosion source | L-self < random (learned harm); L-outer > L-self 3/3; gravity REFUTED at 3 seeds; P(t) flat → erosion = tracking cost, not plasticity loss | `..._round3_learned_gates.md` |
| 4 | TinyCIMM-Euler | Temporal vs spatial specialization | ADAPT beats oracle-routed frozen pair at every dwell (0.65–0.69 vs 0.82); recovery ~10² tokens; retention cost dose-dependent; gates need regime-local thresholds | `..._round4_regimes.md` |
| 5 | pythia-160m | Does the gate ordering replicate? | NO: rand beats err AND ent; surprise splits epistemic/aleatoric | `ember3_scale160m/journals/2026-07-19_rung1_verdict.md` |
| 6 | pythia-160m | Band + excess gates | Directional confirm; rand unbeaten (coverage > curation); excess = stability champion; drift jitter-bistable (ent-churn claim downgraded) | `..._round6_band_excess.md` |
| 7 | ember_v1_mixed vs vanilla_mixed | Does the physics change CL behavior? | Conservation loss FALLS 30–40% under 10k updates (first datapoint); gate phenomenology architecture-robust; headroom confound acknowledged; 2.2× fresh-text response gap (unresolved) | `..._round7_twins.md` |
| 8 | OLMoE-1B-7B | Pretrained MoE as continuous learner | Works (dom −0.14..−0.18 CE vs off +0.03..+0.04); full beats experts-only BOTH axes; fused-backward SGD = 7B in 15.3GB | `..._round8_moe.md` |
| 9 | OLMoE (benchmarks) | Task-level retention (pre-registered battery) | full 9/9 no-change; experts TriviaQA −10.4pt; **CE-blindness demonstrated** | `..._round9_prereg.md`, `..._round9_verdict.md` |
| 10 | OLMoE (SCBF v2) | Deflation closure + mechanism | Erosion real (cloze ΔLL −0.230, 10× thr); full arm GAINS (+0.060); mechanism refuted by control → **compensation law** (identical mass, ratio 0.987 — trunk co-adaptation is the protection) | `..._round10_prereg.md`, `..._round10_verdict.md` |

## The five keeper results

1. **The compensation law** — uncompensated updates damage; compensated don't
   (r10). Third sighting of "uncompensated change" as the damage term
   (w/ Boltzmann hallucination + POC-1 load-bearing layers).
2. **CE-blindness** — loss metrics cannot see knowledge erosion (r9);
   task-level heartbeats are mandatory sentries.
3. **Continuous learning works at the 5k dose** — a pretrained 7B MoE gains
   stream + fact knowledge with 9/9 retention, unoptimized (r8–10). Floor, not
   ceiling.
4. **Adaptation substitutes for representation** (r4) — why MoE exists; when it
   doesn't need to.
5. **Exogeneity holds where surprise is epistemic; coverage rules where it
   isn't** (r2/3 vs r5/6) — signal origin + learnability selection + representative
   coverage as the gate design space.

Retractions on record: gravity drift-halving (r3), ent-churn magnitude (r6),
"concentration causes damage" literal mechanism (r10), toy gate ordering at
scale (r5). Four claims killed by pre-registered rules — the reason the five
above are credible.

## Instruments (now SCBF v2 — `research/scbf/v2/`, spec `scbf/.spec/scbf-v2.spec.md`)

HookSpine (fused-backward updates, placement masks = lesions, per-slice mass
telemetry) · CKA drift (co-moving-frame caveat) · held-out CE splits · cloze
knowledge probe · plasticity probe · lm-eval battery + 2×SE rule ·
gradient-pressure lesioning · gates (err/excess/band/rand) · RunLog. 11 CPU
tests; validated in first contact (round 10).

## Assets (node1/CT103)

`/data/ember3/`: venv (torch 2.6.0+cu124, transformers 5.14.1, lm_eval 0.4.12,
scipy, accelerate), scbf_v2, three WT103 tokenizations + TinyStories val, all
runners + logs. `/data/models/ember3-olmoe/`: OLMoE base + adapted checkpoints
(experts/full, HF format) + caches. Local: all results CSVs under
`ember3_scale160m/results/`, `ember3_drift/results/`.

## The open queue (priority order)

1. **Replication batch** (~40 GPU-min): full-arm cloze ×2 repeats — the
   knowledge-GAIN result is single-seed and now load-bearing.
2. **Partial-trunk lesion sweep** (~1 afternoon): which component carries
   compensation (router / attention / norms)?
3. **Compensation-prediction test**: PAC compensation metrics on the frozen
   model → predict the lesion map. The physics calls its shot.
4. **Long-horizon resident**: 10⁵–10⁶ updates on node1, task-level heartbeat.
   The program's decisive experiment.
5. GLM routing test (doc §4.4, still open); matched-headroom twins; 410M rung;
   MED/recursive-gravity co-moving anchor.
