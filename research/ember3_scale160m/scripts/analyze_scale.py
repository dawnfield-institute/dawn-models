"""
Round 5 analysis — aggregates pulled results/ and prints the verdict table.

Quantities (per pre-registration):
  edge_first10 / edge_last10:  mean(frozen_ref loss − arm loss) over the first/last
                               10% of stream chunks (positive = adaptation gain over
                               the frozen pretrained model on identical text)
  drift_final / drift_auc:     1−CKA at end; mean drift over all snapshots (AUC
                               lesson from round 3 — final snapshots are noisy)
  heldout_start/final:         fixed validation-batch CE (rise = forgetting)
  plast_first/last:            clone adaptation improvement rate (fixed segment)
  update_rate:                 fraction of chunks that applied an update
"""

import json
import os
from glob import glob

import numpy as np
import pandas as pd

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def main():
    ref = pd.read_csv(os.path.join(RESULTS, "frozen_ref_0", "timeseries.csv"))
    ref_loss = ref["loss"].values

    rows = []
    for d in sorted(glob(os.path.join(RESULTS, "*_*"))):
        name = os.path.basename(d)
        if not os.path.isdir(d) or name.startswith("frozen_ref"):
            continue
        ts = pd.read_csv(os.path.join(d, "timeseries.csv"))
        sn = pd.read_csv(os.path.join(d, "snapshots.csv"))
        pl = pd.read_csv(os.path.join(d, "plasticity.csv"))
        meta = json.load(open(os.path.join(d, "meta.json")))
        m = ts["loss"].values
        n = min(len(m), len(ref_loss))
        n10 = n // 10
        dr = sn.dropna(subset=["drift"])
        rows.append({
            "run": name, "arm": meta["arm"],
            "edge_first10": float(np.mean(ref_loss[:n10] - m[:n10])),
            "edge_last10": float(np.mean(ref_loss[n - n10:n] - m[n - n10:n])),
            "drift_final": float(dr["drift"].iloc[-1]) if len(dr) else float("nan"),
            "drift_auc": float(dr["drift"].mean()) if len(dr) else float("nan"),
            "heldout_start": float(sn["heldout_loss"].iloc[0]),
            "heldout_final": float(sn["heldout_loss"].iloc[-1]),
            "plast_first": float(pl["plasticity"].iloc[0]) if len(pl) else float("nan"),
            "plast_last": float(pl["plasticity"].iloc[-1]) if len(pl) else float("nan"),
            "update_rate": meta["updates"] / meta["chunks"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "scale160m_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))
    print()
    agg = df.groupby("arm")[["edge_last10", "drift_auc", "heldout_final",
                             "plast_last", "update_rate"]].agg(["mean", "min", "max"])
    print(agg.round(4).to_string())
    print(f"\nfrozen_ref mean stream loss: {ref_loss.mean():.4f} "
          f"(first10 {ref_loss[:len(ref_loss)//10].mean():.4f} / "
          f"last10 {ref_loss[-len(ref_loss)//10:].mean():.4f})")


if __name__ == "__main__":
    main()
