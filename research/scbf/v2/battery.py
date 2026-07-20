"""
SCBF v2 task-battery analysis — the capability heartbeat.

Parses lm-eval-harness output directories and applies the locked reading rule from
Ember III round 9: per task, delta = candidate − reference is NO-CHANGE iff
|delta| <= 2 * sqrt(se_ref^2 + se_cand^2). Pre-registering the rule BEFORE running
the battery is protocol; this module only applies it mechanically.

Why batteries are mandatory: CE-family metrics are demonstrably blind to knowledge
erosion (round 9: TriviaQA −10.4pt at +0.043 nats off-domain CE).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, Tuple


def load_lmeval_results(output_dir: str) -> Dict[str, Tuple[float, float]]:
    """Read every results*.json under an lm-eval --output_path dir.
    Returns {task_key: (score, stderr)} using acc (or exact_match), plus
    acc_norm for hellaswag-style tasks as 'task/an'."""
    out: Dict[str, Tuple[float, float]] = {}
    for f in glob.glob(os.path.join(output_dir, "**", "results*.json"), recursive=True):
        j = json.load(open(f, encoding="utf-8"))
        for task, res in j.get("results", {}).items():
            for metric in ("acc", "exact_match", "acc_norm"):
                v = res.get(f"{metric},none")
                if not isinstance(v, (int, float)):
                    continue
                se = res.get(f"{metric}_stderr,none", 0) or 0
                key = task + ("/an" if metric == "acc_norm" else "")
                if metric == "acc_norm" and task != "hellaswag":
                    continue
                if metric in ("acc", "exact_match") or key.endswith("/an"):
                    out[key] = (float(v), float(se))
    return out


def verdict(reference: Dict[str, Tuple[float, float]],
            candidate: Dict[str, Tuple[float, float]], k: float = 2.0) -> dict:
    """Apply the locked rule per shared task. Returns rows + counts."""
    rows = []
    for task in sorted(set(reference) & set(candidate)):
        r, se_r = reference[task]
        c, se_c = candidate[task]
        thr = k * (se_r ** 2 + se_c ** 2) ** 0.5
        d = c - r
        call = "=" if abs(d) <= thr else ("UP" if d > 0 else "DOWN")
        rows.append({"task": task, "reference": r, "candidate": c,
                     "delta": d, "threshold": thr, "verdict": call})
    return {"rows": rows,
            "n_no_change": sum(1 for x in rows if x["verdict"] == "="),
            "n_up": sum(1 for x in rows if x["verdict"] == "UP"),
            "n_down": sum(1 for x in rows if x["verdict"] == "DOWN")}


def format_verdict(v: dict) -> str:
    lines = [f"{'task':<22}{'ref':>9}{'cand':>9}{'delta':>9}   verdict"]
    for r in v["rows"]:
        lines.append(f"{r['task']:<22}{r['reference']:>9.4f}{r['candidate']:>9.4f}"
                     f"{r['delta']:>+9.4f}   {r['verdict']}")
    lines.append(f"no-change {v['n_no_change']} / up {v['n_up']} / down {v['n_down']}")
    return "\n".join(lines)
