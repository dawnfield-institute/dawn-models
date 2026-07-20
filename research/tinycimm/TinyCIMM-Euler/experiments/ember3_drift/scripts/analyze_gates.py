"""
Round 2b analysis — gate-origin arms (endogenous / exogenous / random @ ~50%)
vs ungated reference. Aggregates results/runs_gate_*/ into one table + verdict
inputs. Output: results/gate_origin_summary.csv + printed table.
"""

import json
import os
import sys
from glob import glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_arms import CFG  # noqa: E402
from instrumentation import prime_gaps  # noqa: E402

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
TOKENS = 50_000


def main():
    gaps = prime_gaps(CFG["sieve_limit"])
    W = CFG["window"]
    med = pd.Series(gaps[: W + TOKENS]).rolling(500).median().shift(1).values[W: W + TOKENS]
    comp = np.abs(gaps[W: W + TOKENS] - med)

    rows = []
    for tag in ["none", "ent", "err", "rand", "learn_self", "learn_outer"]:
        for run_dir in sorted(glob(os.path.join(RESULTS, f"runs_gate_{tag}", "A_*"))):
            seed = int(os.path.basename(run_dir).split("_")[1])
            ts = pd.read_csv(os.path.join(run_dir, "timeseries.csv"))
            sn = pd.read_csv(os.path.join(run_dir, "snapshots.csv"))
            pl = pd.read_csv(os.path.join(run_dir, "plasticity.csv"))
            meta = json.load(open(os.path.join(run_dir, "meta.json")))
            m = ts["prequential_err"].values
            n10 = len(m) // 10
            d = sn.dropna(subset=["drift"])
            tail = d.iloc[-max(3, len(d) // 5):]
            rows.append({
                "gate": tag, "seed": seed,
                "edge_first10": 1 - np.nanmean(m[:n10]) / np.nanmean(comp[:n10]),
                "edge_last10": 1 - np.nanmean(m[-n10:]) / np.nanmean(comp[-n10:]),
                "drift_final": d["drift"].iloc[-1],
                "drift_slope": np.polyfit(tail["t"] / 1000.0, tail["drift"], 1)[0],
                "plast_first": pl["plasticity"].iloc[0],
                "plast_mid": pl["plasticity"].iloc[len(pl) // 2],
                "plast_last": pl["plasticity"].iloc[-1],
                "held_final": sn["held_ahead_mae"].iloc[-1],
                "update_rate": meta["updates_applied"] / meta["tokens"] if meta.get("updates_applied") else 1.0,
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "gate_origin_summary.csv"), index=False)
    agg = df.groupby("gate").agg(["mean", "min", "max"]).round(4)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))
    print()
    print(agg[["edge_first10", "edge_last10", "drift_final", "plast_last", "held_final", "update_rate"]].to_string())


if __name__ == "__main__":
    main()
