"""
Round 4 — temporal vs spatial specialization: can one continuously-adapting
network replace a routed pair of frozen specialists on a regime-switching stream?

PRE-REGISTERED DESIGN (locked before any run, 2026-07-19):

Stream: alternating segments of two structurally distinct regimes, each z-scored
by its own causal 1k calibration window so they differ in STRUCTURE, not scale:
  A: prime gaps (noisy, heavy-tailed, slowly drifting)   — the program substrate
  B: harmonic mix sin(4πt/1000)+0.5sin(8πt/1000)+0.25sin(12πt/1000) (smooth,
     stationary, near-deterministic)
Each regime CONTINUES from where it left off when it returns (fresh data, no
replays). Dwell sweep D ∈ {250, 1000, 4000} tokens; total 24k tokens per
condition; 3 seeds.

Arms (all share the round-2 champion substrate: StabilizedTinyCIMMEuler,
norm + leaky 0.01, as-built capacity controller):
  ADAPT      one network, continuous online adaptation, ungated
  ADAPT_EG   same + exogenous err-gate (realized error >= rolling median)
  ROUTED     two specialists, each pre-trained 10k tokens on its pure regime,
             then FROZEN (predictions via side-effect-free clean_predict only);
             routed by an ORACLE on the true regime label — the upper bound for
             any learned router
  FROZGEN    one generalist pre-trained 10k tokens on a 50/50 interleaved
             (D=500) mix, then frozen — controls for "is adaptation doing
             anything at eval time at all"

Metrics (normalized units; both regimes ~unit scale):
  - overall + per-regime prequential MAE (pre-update, leak-free)
  - RECOVERY: tokens after each switch until rolling-50 MAE re-enters 1.2x the
    final rolling-50 MAE of the PREVIOUS visit to that regime (adaptive arms)
  - RETENTION: first-100-token MAE on re-entering a regime minus last-100-token
    MAE of the previous visit to it (forgetting cost; adaptive arms)
  - comparator: per-regime causal rolling-median (window 500, within-regime)

Pre-registered interpretation:
  - ROUTED wins at small D, ADAPT wins at large D, crossover threshold τ in
    between → the claim quantified: temporal specialization substitutes for
    spatial above τ.
  - ROUTED wins everywhere → adaptation too slow/forgetful at this scale.
  - ADAPT wins everywhere → frozen specialists lose to within-regime drift
    (prime gaps drift; check the per-regime breakdown before crediting
    adaptation).
  - ADAPT_EG vs ADAPT: does surprise-gating speed or slow post-switch recovery?
"""

import json
import os
import sys
import time
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instrumentation import prime_gaps, clean_predict, safe_deepcopy  # noqa: E402
from stabilized import StabilizedTinyCIMMEuler  # noqa: E402
from tinycimm_euler import HigherOrderEntropyMonitor  # noqa: E402

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
W = 4
DWELLS = [250, 1000, 4000]
TOTAL = 24_000
PRETRAIN = 10_000
SEEDS = [0, 1, 2]


def build_regime_series():
    gaps = prime_gaps(3_200_000).astype(np.float64)
    mu_a, sd_a = gaps[:1000].mean(), gaps[:1000].std()
    a = (gaps - mu_a) / sd_a

    t = np.arange(200_000, dtype=np.float64)
    harm = (np.sin(4 * np.pi * t / 1000) + 0.5 * np.sin(8 * np.pi * t / 1000)
            + 0.25 * np.sin(12 * np.pi * t / 1000))
    mu_b, sd_b = harm[:1000].mean(), harm[:1000].std()
    b = (harm - mu_b) / sd_b
    return a.astype(np.float32), b.astype(np.float32)


class RegimePointer:
    """Serves windows from one regime's series, continuing across segments."""

    def __init__(self, series, start):
        self.s = series
        self.i = start

    def next(self):
        x = torch.tensor(self.s[self.i: self.i + W], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor([[self.s[self.i + W]]], dtype=torch.float32)
        self.i += 1
        return x, y


def make_model(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = StabilizedTinyCIMMEuler(input_size=W, hidden_size=16, output_size=1,
                                device=torch.device("cpu"), leaky_slope=0.01)
    m.set_complexity_monitor(HigherOrderEntropyMonitor(momentum=0.85))
    return m


def adapt_on(model, ptr, n, devnull, select_best=False, tail=3000, every=100):
    """Online pretraining. With select_best, checkpoint every `every` tokens over
    the final `tail` and return the state with the best rolling-100 prequential
    MAE — a fair proxy for a converged, model-selected frozen expert (the raw
    online loop never settles, so freezing 'wherever it happened to be' is a
    strawman — pre-flight finding, 2026-07-19)."""
    errs = []
    best = (float("inf"), None)
    for i in range(n):
        x, y = ptr.next()
        errs.append(float(torch.abs(clean_predict(model, x) - y).mean()))
        with redirect_stdout(devnull):
            model.online_adaptation_step(x, y)
        if select_best and i >= n - tail and (i % every == 0) and len(errs) >= 100:
            roll = float(np.mean(errs[-100:]))
            if roll < best[0]:
                best = (roll, safe_deepcopy(model))
    if select_best and best[1] is not None:
        return best[1]
    return model


def schedule(total, dwell):
    """Regime label per token: A,B,A,B,... segments of length dwell."""
    lab = np.zeros(total, dtype=int)
    for s in range(0, total, dwell):
        lab[s: s + dwell] = (s // dwell) % 2
    return lab


def run_condition(arm, dwell, seed, a, b, devnull):
    """Returns per-token dataframe for one (arm, dwell, seed)."""
    # Fresh pointers per condition; pretraining consumes [0, PRETRAIN), the
    # eval stream continues from PRETRAIN so frozen and adaptive arms see the
    # same eval data.
    pre_a, pre_b = RegimePointer(a, 0), RegimePointer(b, 0)
    ev_a, ev_b = RegimePointer(a, PRETRAIN), RegimePointer(b, PRETRAIN)
    lab = schedule(TOTAL, dwell)

    if arm == "ROUTED":
        spec_a = adapt_on(make_model(seed), pre_a, PRETRAIN, devnull, select_best=True)
        spec_b = adapt_on(make_model(seed + 100), pre_b, PRETRAIN, devnull, select_best=True)
        models = {0: spec_a, 1: spec_b}
    elif arm == "FROZGEN":
        gen = make_model(seed)
        mix_lab = schedule(PRETRAIN, 500)
        errs = []
        best = (float("inf"), None)
        for i, r in enumerate(mix_lab):
            x, y = (pre_a if r == 0 else pre_b).next()
            errs.append(float(torch.abs(clean_predict(gen, x) - y).mean()))
            with redirect_stdout(devnull):
                gen.online_adaptation_step(x, y)
            if i >= PRETRAIN - 3000 and i % 100 == 0 and len(errs) >= 100:
                roll = float(np.mean(errs[-100:]))
                if roll < best[0]:
                    best = (roll, safe_deepcopy(gen))
        if best[1] is not None:
            gen = best[1]
        models = {0: gen, 1: gen}
    else:
        model = make_model(seed)
        err_win = []

    rows = []
    for t in range(TOTAL):
        r = int(lab[t])
        x, y = (ev_a if r == 0 else ev_b).next()
        if arm in ("ROUTED", "FROZGEN"):
            err = float(torch.abs(clean_predict(models[r], x) - y).mean())
        else:
            err = float(torch.abs(clean_predict(model, x) - y).mean())
            gate_open = True
            if arm == "ADAPT_EG":
                gate_open = len(err_win) < 50 or err >= float(np.median(err_win))
                err_win.append(err)
                err_win = err_win[-200:]
                model.update_gate_open = gate_open
            with redirect_stdout(devnull):
                model.online_adaptation_step(x, y)
            if arm == "ADAPT_EG":
                model.update_gate_open = True
        row = {"t": t, "regime": r, "err": err}
        if arm == "ADAPT_EG":
            row["gate_open"] = int(gate_open)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["arm"], df["dwell"], df["seed"] = arm, dwell, seed
    return df


def main():
    os.makedirs(os.path.join(RESULTS, "runs_regime"), exist_ok=True)
    a, b = build_regime_series()
    devnull = open(os.devnull, "w")
    smoke = "--smoke" in sys.argv

    arms = ["ADAPT", "ADAPT_EG", "ROUTED", "FROZGEN"]
    dwells = [1000] if smoke else DWELLS
    seeds = [0] if smoke else SEEDS

    all_rows = []
    for seed in seeds:
        for dwell in dwells:
            for arm in arms:
                t0 = time.time()
                df = run_condition(arm, dwell, seed, a, b, devnull)
                df.to_csv(os.path.join(RESULTS, "runs_regime",
                                       f"{arm}_{dwell}_{seed}.csv"), index=False)
                m_all = df["err"].mean()
                m_a = df[df.regime == 0]["err"].mean()
                m_b = df[df.regime == 1]["err"].mean()
                print(f"[done] {arm} dwell={dwell} seed={seed}: MAE {m_all:.4f} "
                      f"(A {m_a:.4f} / B {m_b:.4f}) {time.time()-t0:.0f}s", flush=True)
                all_rows.append({"arm": arm, "dwell": dwell, "seed": seed,
                                 "mae": m_all, "mae_A": m_a, "mae_B": m_b})
    devnull.close()
    summary = pd.DataFrame(all_rows)
    summary.to_csv(os.path.join(RESULTS, "regime_summary.csv"), index=False)
    print()
    print(summary.groupby(["arm", "dwell"])["mae"].agg(["mean", "min", "max"]).round(4).to_string())


if __name__ == "__main__":
    main()
