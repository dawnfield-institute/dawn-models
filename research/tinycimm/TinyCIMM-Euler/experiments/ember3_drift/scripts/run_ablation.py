"""
Round 2a — CIMM-stabilizer ablation: 2^4 grid over (gravity, vlr, qclip, gate),
Arm A (as-built capacity policy), seed 0, 20k tokens, repaired substrate
(norm + leaky 0.01), fixed plasticity segment.

Winner criteria (pre-stated): among configs with
  (i)  plasticity retained: P_last >= P_first - 0.05
  (ii) drift bounded: late drift slope <= +0.005 per 1k tokens
maximize the last-10% prequential edge vs the causal rolling-median comparator;
tiebreak on final held-ahead MAE.

Outputs: results/ablation_summary.csv + printed table + winner line.
"""

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_arms import CFG, run_one  # noqa: E402
from instrumentation import prime_gaps  # noqa: E402

FLAGS = ["gravity", "vlr", "qclip", "gate"]
TOKENS = 20_000


def main():
    results_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_root, exist_ok=True)

    print(f"Sieving primes to {CFG['sieve_limit']} ...", flush=True)
    gaps = prime_gaps(CFG["sieve_limit"])
    W = CFG["window"]
    med = pd.Series(gaps[: W + TOKENS]).rolling(500).median().shift(1).values[W: W + TOKENS]
    comp = np.abs(gaps[W: W + TOKENS] - med)

    rows = []
    for bits in itertools.product([0, 1], repeat=4):
        name = "abl_" + "".join(map(str, bits))
        cfg = dict(CFG)
        cfg.update({"norm": True, "leaky": 0.01, "tag": name, "gate_arm": "none",
                    "fixed_plast": True, "stab": dict(zip(FLAGS, map(bool, bits)))})
        print(f"=== {name} {cfg['stab']} ===", flush=True)
        meta = run_one("A", 0, TOKENS, "cpu", results_root, gaps, cfg)

        run_dir = os.path.join(results_root, f"runs_{name}", "A_0")
        ts = pd.read_csv(os.path.join(run_dir, "timeseries.csv"))
        sn = pd.read_csv(os.path.join(run_dir, "snapshots.csv"))
        pl = pd.read_csv(os.path.join(run_dir, "plasticity.csv"))

        m = ts["prequential_err"].values
        n10 = max(1, len(m) // 10)
        d = sn.dropna(subset=["drift"])
        tail = d.iloc[-max(3, len(d) // 5):]
        slope = float(np.polyfit(tail["t"].values / 1000.0, tail["drift"].values, 1)[0])

        rows.append({
            "config": name,
            **{f: bool(b) for f, b in zip(FLAGS, bits)},
            "edge_first10": float(1 - np.nanmean(m[:n10]) / np.nanmean(comp[:n10])),
            "edge_last10": float(1 - np.nanmean(m[-n10:]) / np.nanmean(comp[-n10:])),
            "drift_final": float(d["drift"].iloc[-1]) if len(d) else float("nan"),
            "drift_slope": slope,
            "plast_first": float(pl["plasticity"].iloc[0]),
            "plast_last": float(pl["plasticity"].iloc[-1]),
            "held_final": float(sn["held_ahead_mae"].iloc[-1]),
            "updates": int(meta["updates_applied"]),
            "rollbacks": int(meta["rollbacks"]),
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(results_root, "ablation_summary.csv")
    df.to_csv(out_csv, index=False)

    ok = df[(df["plast_last"] >= df["plast_first"] - 0.05) & (df["drift_slope"] <= 0.005)]
    pool = ok if len(ok) else df
    winner = pool.sort_values(["edge_last10", "held_final"], ascending=[False, True]).iloc[0]

    print("\n" + df.to_string(index=False), flush=True)
    print(f"\nWINNER: {winner['config']} "
          f"({ {f: bool(winner[f]) for f in FLAGS} }) "
          f"edge_last10={winner['edge_last10']:+.3f} held={winner['held_final']:.3f} "
          f"[eligible pool: {len(ok)}/{len(df)}]", flush=True)
    with open(os.path.join(results_root, "ablation_winner.json"), "w") as f:
        json.dump({"config": winner["config"],
                   "stab": {f: bool(winner[f]) for f in FLAGS}}, f, indent=2)


if __name__ == "__main__":
    main()
