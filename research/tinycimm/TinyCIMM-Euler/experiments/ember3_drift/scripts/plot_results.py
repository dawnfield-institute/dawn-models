"""
Ember III shakedown — the deliverable plot.

Reads results/runs/{arm}_{seed}/ and produces:
  results/summary.png         four panels on a shared token axis:
                              drift, competence, plasticity, neuron count
  results/summary_stats.json  per-arm invariants for the readout

Legible without framework vocabulary (idea doc §7.1): the panels are labeled in
plain ML terms.
"""

import argparse
import json
import os
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from instrumentation import prime_gaps

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

ARM_COLORS = {"A": "#d62728", "B": "#1f77b4", "C": "#7f7f7f"}
ARM_LABELS = {
    "A": "Arm A — endogenous (entropy threshold, as built)",
    "B": "Arm B — residual (realized prequential error only)",
    "C": "Arm C — decoupled surrogate (shuffled B signal)",
}


def load_runs(runs_dirname):
    runs = {}
    for run_dir in sorted(glob(os.path.join(RESULTS, runs_dirname, "*_*"))):
        name = os.path.basename(run_dir)
        arm, seed = name.split("_")
        if len(arm) != 1:
            continue
        try:
            runs.setdefault(arm, {})[int(seed)] = {
                "dir": run_dir,
                "snapshots": pd.read_csv(os.path.join(run_dir, "snapshots.csv")),
                "plasticity": pd.read_csv(os.path.join(run_dir, "plasticity.csv")),
                "events": pd.read_csv(os.path.join(run_dir, "events.csv")),
                "meta": json.load(open(os.path.join(run_dir, "meta.json"))),
            }
        except FileNotFoundError:
            continue
    # Only full-length runs (skip smoke leftovers with different token counts)
    max_tokens = max(r["meta"]["tokens"] for by in runs.values() for r in by.values())
    return {a: {s: r for s, r in by.items() if r["meta"]["tokens"] == max_tokens}
            for a, by in runs.items()}, max_tokens


def band(ax, arm, runs, col, ylabel):
    per_seed = [r["snapshots"].set_index("t")[col] for r in runs.values()]
    df = pd.concat(per_seed, axis=1)
    t = df.index.values
    mean, lo, hi = df.mean(axis=1), df.min(axis=1), df.max(axis=1)
    ax.plot(t, mean, color=ARM_COLORS[arm], label=ARM_LABELS[arm], lw=1.8)
    ax.fill_between(t, lo, hi, color=ARM_COLORS[arm], alpha=0.15)
    ax.set_ylabel(ylabel)


def late_slope(snap, col, frac=0.2):
    """Linear slope over the last `frac` of snapshots (per 1k tokens)."""
    d = snap.dropna(subset=[col])
    n = max(3, int(len(d) * frac))
    tail = d.iloc[-n:]
    if len(tail) < 3:
        return float("nan")
    coef = np.polyfit(tail["t"].values / 1000.0, tail[col].values, 1)
    return float(coef[0])


def stream_edge(run, comp_err, frac):
    """Model's prequential edge vs the causal rolling-median predictor over the
    first/last `frac` of the run. Positive = model better than the comparator."""
    ts = pd.read_csv(os.path.join(run["dir"], "timeseries.csv"))
    m = ts["prequential_err"].values
    c = comp_err[: len(m)]
    n = max(1, int(len(m) * frac))
    first = 1 - np.nanmean(m[:n]) / np.nanmean(c[:n])
    last = 1 - np.nanmean(m[-n:]) / np.nanmean(c[-n:])
    return float(first), float(last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="read results/runs_<tag> instead of results/runs")
    args = ap.parse_args()
    runs_dirname = f"runs_{args.tag}" if args.tag else "runs"
    suffix = f"_{args.tag}" if args.tag else ""

    runs, max_tokens = load_runs(runs_dirname)

    # Causal rolling-median comparator — model-independent, one pass for all runs.
    gaps = prime_gaps(3_200_000)
    W = 4
    med = pd.Series(gaps[: W + max_tokens]).rolling(500).median().shift(1).values[W: W + max_tokens]
    comp_err = np.abs(gaps[W: W + max_tokens] - med)

    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)

    for arm in sorted(runs):
        band(axes[0], arm, runs[arm], "drift",
             "Representational drift\n1 − CKA vs reference (t=1k)")
        band(axes[1], arm, runs[arm], "held_ahead_mae",
             "Held-out MAE\n(never-seen future windows)")
        # plasticity: mean over seeds at each probe time
        p = pd.concat([r["plasticity"].set_index("t")["plasticity"]
                       for r in runs[arm].values()], axis=1)
        axes[2].plot(p.index, p.mean(axis=1), "o-", color=ARM_COLORS[arm], lw=1.5)
        axes[2].fill_between(p.index, p.min(axis=1), p.max(axis=1),
                             color=ARM_COLORS[arm], alpha=0.15)
        band(axes[3], arm, runs[arm], "neurons", "Hidden neurons")

    # Trivial-floor context on the competence panel.
    any_meta = next(iter(next(iter(runs.values())).values()))["meta"]
    floor = any_meta.get("baselines", {}).get("held_best_constant_mae")
    if floor:
        axes[1].axhline(floor, color="k", ls="--", lw=1,
                        label=f"best-constant floor ({floor:.2f})")
        axes[1].legend(loc="upper right", fontsize=8)

    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].set_title(
        "Continuous online learning on the prime-gap stream — does the source of the\n"
        "structural-adaptation signal change drift behavior? (3 seeds per arm, band = min–max)")
    axes[2].set_ylabel("Plasticity\n(rel. error reduction on\nfresh unseen segment)")
    axes[2].axhline(0, color="k", lw=0.5, alpha=0.4)
    axes[3].set_xlabel("Stream tokens")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    out_png = os.path.join(RESULTS, f"summary{suffix}.png")
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")

    stats = {}
    for arm, by_seed in sorted(runs.items()):
        s = {"seeds": sorted(by_seed)}
        finals = [r["snapshots"]["drift"].dropna().iloc[-1] for r in by_seed.values()]
        s["final_drift_mean"] = float(np.mean(finals))
        s["final_drift_range"] = [float(np.min(finals)), float(np.max(finals))]
        s["late_drift_slope_per_1k"] = float(np.mean(
            [late_slope(r["snapshots"], "drift") for r in by_seed.values()]))
        s["held_ahead_mae_final"] = float(np.mean(
            [r["snapshots"]["held_ahead_mae"].iloc[-1] for r in by_seed.values()]))
        s["held_ahead_mae_best"] = float(np.mean(
            [r["snapshots"]["held_ahead_mae"].min() for r in by_seed.values()]))
        pl = [r["plasticity"]["plasticity"] for r in by_seed.values()]
        s["plasticity_first_mean"] = float(np.mean([p.iloc[0] for p in pl]))
        s["plasticity_last_mean"] = float(np.mean([p.iloc[-1] for p in pl]))
        s["grow_events_mean"] = float(np.mean([r["meta"]["n_grow"] for r in by_seed.values()]))
        s["prune_events_mean"] = float(np.mean([r["meta"]["n_prune"] for r in by_seed.values()]))
        s["final_neurons"] = [r["meta"]["final_neurons"] for _, r in sorted(by_seed.items())]
        edges = [stream_edge(r, comp_err, 0.1) for r in by_seed.values()]
        s["stream_edge_vs_rolling_median_first10pct"] = float(np.mean([e[0] for e in edges]))
        s["stream_edge_vs_rolling_median_last10pct"] = float(np.mean([e[1] for e in edges]))
        stats[arm] = s

    out_json = os.path.join(RESULTS, f"summary_stats{suffix}.json")
    with open(out_json, "w") as f:
        json.dump({"tokens": max_tokens, "arms": stats}, f, indent=2)
    print(f"wrote {out_json}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
