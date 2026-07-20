"""
Ember III shakedown runner — arms A/B/C on the continuous prime-gap stream.

Usage:
  python run_arms.py --smoke                 # Arm A, seed 0, 2k tokens (harness check)
  python run_arms.py --arm B --seed 1        # single run
  python run_arms.py --all                   # 3 arms x 3 seeds x 50k tokens
                                             # (B runs before C per seed: C consumes
                                             #  a block-shuffled copy of B's trace)

Arms differ ONLY in model.structure_controller. Same seed => identical init
weights across arms (controller substitution happens after construction and
consumes no torch RNG).
"""

import argparse
import json
import os
import sys
import time
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tinycimm_euler import TinyCIMMEuler, HigherOrderEntropyMonitor  # noqa: E402

from controllers import ResidualSignalController, DecoupledSurrogateController, block_shuffle  # noqa: E402
from instrumentation import (  # noqa: E402
    prime_gaps, make_windows, hidden_reps, clean_predict, linear_cka, plasticity_probe,
    rep_stats,
)

HARNESS_VERSION = "2026-07-19.5"  # .4: CIMM stabilizers + origin gates; .5: learned gates (REINFORCE, self vs outer channel)

CFG = {
    "window": 4,                 # input = 4 consecutive gaps -> predict the next
    "tokens": 50_000,            # training-stream length (window starts 0..tokens)
    "sieve_limit": 3_200_000,    # ~230k gaps: stream + ahead + plasticity regions
    "probe_cadence": 500,        # drift/competence snapshot every N tokens
    "warmup_ref": 1_000,         # drift reference frozen here
    "drift_probes_seen": 128,    # sampled from [0, 50k) — fixed regardless of --tokens
    "drift_probes_ahead": 128,   # sampled from [100k, 110k)
    "held_ahead_probes": 512,    # competence probes from [100k, 110k)
    "plast_every": 5_000,        # plasticity probe cadence (plus one at warmup_ref)
    "plast_steps": 200,
    "plast_region_start": 150_000,
    "surrogate_block": 100,      # Arm C block-shuffle size
    "hidden_init": 16,
}


def build_probes(gaps, cfg):
    w = cfg["window"]
    rng = np.random.default_rng(1234)
    seen = rng.choice(50_000, size=cfg["drift_probes_seen"], replace=False)
    ahead = rng.choice(np.arange(100_000, 110_000), size=cfg["drift_probes_ahead"], replace=False)
    drift_X, _ = make_windows(gaps, np.sort(np.concatenate([seen, ahead])), w)

    rng2 = np.random.default_rng(5678)
    held = np.sort(rng2.choice(np.arange(100_000, 110_000), size=cfg["held_ahead_probes"], replace=False))
    held_X, held_y = make_windows(gaps, held, w)
    return drift_X, held_X, held_y


def plasticity_segment(gaps, cfg, k):
    """Never-seen segment for plasticity probe #k. With fixed_plast (round 2+),
    every probe uses segment 0 so P(t) is comparable across probe times —
    round 1 showed per-probe segment difficulty dominates P(t) otherwise."""
    if cfg.get("fixed_plast", False):
        k = 0
    start = cfg["plast_region_start"] + k * (cfg["plast_steps"] + cfg["window"] + 10)
    idxs = np.arange(start, start + cfg["plast_steps"])
    return make_windows(gaps, idxs, cfg["window"])


def make_model(arm, seed, device, cfg, runs_root):
    torch.manual_seed(seed)
    np.random.seed(seed)
    leaky = cfg.get("leaky", 0.0)
    if leaky > 0:
        from stabilized import StabilizedTinyCIMMEuler
        model = StabilizedTinyCIMMEuler(input_size=cfg["window"], hidden_size=cfg["hidden_init"],
                                        output_size=1, device=device, leaky_slope=leaky,
                                        stab=cfg.get("stab") or {})
    else:
        model = TinyCIMMEuler(input_size=cfg["window"], hidden_size=cfg["hidden_init"],
                              output_size=1, device=device)
    model.set_complexity_monitor(HigherOrderEntropyMonitor(momentum=0.85))
    if arm == "B":
        model.structure_controller = ResidualSignalController()
    elif arm == "C":
        trace_path = os.path.join(runs_root, f"B_{seed}", "error_trace.npy")
        if not os.path.exists(trace_path):
            raise FileNotFoundError(f"Arm C needs Arm B's trace first: {trace_path}")
        trace = block_shuffle(np.load(trace_path), block=cfg["surrogate_block"], seed=seed)
        model.structure_controller = DecoupledSurrogateController(trace)
    return model


def run_one(arm, seed, tokens, device_str, results_root, gaps, cfg):
    device = torch.device(device_str)
    tag = cfg.get("tag", "")
    runs_root = os.path.join(results_root, f"runs_{tag}" if tag else "runs")
    run_dir = os.path.join(runs_root, f"{arm}_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    # Optional normalization (repair R1): z-score against a causal calibration
    # window (first 1000 gaps of the stream) — no lookahead. All errors below
    # are de-normalized (x scale) so they stay comparable to the raw floors.
    if cfg.get("norm", False):
        mu = float(np.mean(gaps[:1000]))
        sigma = float(np.std(gaps[:1000]))
        gaps_in, scale = ((gaps - mu) / sigma).astype(np.float32), sigma
    else:
        mu, sigma, gaps_in, scale = 0.0, 1.0, gaps, 1.0

    drift_X, held_X, held_y = build_probes(gaps_in, cfg)
    train_X, train_y = make_windows(gaps_in, np.arange(tokens), cfg["window"])

    # Trivial floors for honest context: a model at these values has learned
    # nothing. MAE is linear in scale, so floors computed on normalized data
    # times `scale` equal the raw-unit floors exactly.
    hy = held_y.numpy().ravel()
    sy = train_y.numpy().ravel()
    baselines = {
        "held_best_constant_mae": float(np.abs(hy - np.median(hy)).mean()) * scale,
        "held_persistence_mae": float(np.abs(hy - held_X.numpy()[:, -1]).mean()) * scale,
        "stream_best_constant_mae": float(np.abs(sy - np.median(sy)).mean()) * scale,
    }

    model = make_model(arm, seed, device, cfg, runs_root)
    controller = model.structure_controller
    devnull = open(os.devnull, "w")

    ts_rows, snap_rows, plast_rows, events = [], [], [], []
    err_trace = np.empty(tokens, dtype=np.float64)
    H_ref = None
    prev_n = model.hidden_dim
    t0 = time.time()

    # Origin gate (round 2): decides per token whether the weight update applies.
    # All variants run at ~50% rate by construction (median thresholds / p=0.5),
    # isolating the gate SIGNAL'S ORIGIN as the only difference between arms.
    gate_arm = cfg.get("gate_arm", "none")
    gate_err_win, gate_ent_win = [], []
    gate_rng = np.random.default_rng(seed + 7770)
    n_gated_updates = 0

    # Learned gates (round 3): logistic gate on 3 causal features, trained by
    # one-step-delayed REINFORCE. The ONLY difference between the two arms is
    # the reward channel: learn_self = improvement in the model's own
    # self-measured training loss (authorable: includes the target-leak path);
    # learn_outer = improvement in leak-free realized prequential error
    # (exogenous: the fixed stream's verdict). Rate-regularized toward 50%.
    # Credit timing (symmetric across arms): both channel signals measured at
    # token t (self-loss from t's pre-update forward; realized err pre-update)
    # reflect the consequence of a_{t-1}. So after t's step, a_{t-1} is rewarded
    # with t's signal; the gate update lands before a_{t+1}'s decision.
    learned = gate_arm in ("learn_self", "learn_outer")
    if learned:
        gw, gb = np.zeros(3), 0.0
        f_mean, f_var = np.zeros(3), np.ones(3)   # online z-scoring (EMA)
        armed = None                               # (f, a, p) awaiting reward
        reward_ema, rate_ema = 0.0, 0.5
        sig_ema = None
        GLR, RATE_PULL, EMA = 0.01, 0.05, 0.99

    for t in range(tokens):
        x = train_X[t: t + 1].to(device)
        y = train_y[t: t + 1].to(device)

        # Leak-free prequential error, measured BEFORE the update touches this
        # token. Stored in raw gap units (x scale) for cross-config comparability;
        # the controllers' decision logic is scale-invariant either way.
        err = float(torch.abs(clean_predict(model, x) - y.cpu()).mean()) * scale
        err_trace[t] = err
        if hasattr(controller, "observe"):
            controller.observe(err)  # Arm C's observe() swaps in the surrogate

        gate_open = True
        if gate_arm == "err":
            # Exogenous: update on surprising tokens (realized error >= rolling median)
            gate_open = len(gate_err_win) < 50 or err >= float(np.median(gate_err_win))
            gate_err_win.append(err)
            gate_err_win = gate_err_win[-200:]
        elif gate_arm == "ent":
            # Endogenous: previous-step activation entropy >= its rolling median
            e = getattr(model, "last_entropy", float("nan"))
            if np.isfinite(e):
                gate_open = len(gate_ent_win) < 50 or e >= float(np.median(gate_ent_win))
                gate_ent_win.append(e)
                gate_ent_win = gate_ent_win[-200:]
        elif gate_arm == "rand":
            # Rate-matched control: coin flip at the same ~50% rate
            gate_open = bool(gate_rng.random() < 0.5)
        elif learned:
            e_prev = getattr(model, "last_entropy", 0.0)
            if not np.isfinite(e_prev):
                e_prev = 0.0
            trend = (np.mean(err_trace[max(0, t - 25): t]) - np.mean(err_trace[max(0, t - 50): max(1, t - 25)])) if t >= 50 else 0.0
            f_raw = np.array([err, trend, e_prev])
            f_mean = EMA * f_mean + (1 - EMA) * f_raw
            f_var = EMA * f_var + (1 - EMA) * (f_raw - f_mean) ** 2
            f = (f_raw - f_mean) / np.sqrt(f_var + 1e-8)

            p_open = float(1.0 / (1.0 + np.exp(-(gw @ f + gb))))
            gate_open = bool(gate_rng.random() < p_open)
            rate_ema = EMA * rate_ema + (1 - EMA) * float(gate_open)
        if gate_arm != "none":
            model.update_gate_open = gate_open
            n_gated_updates += int(gate_open)

        with redirect_stdout(devnull):
            res = model.online_adaptation_step(x, y)

        if learned:
            # Reward a_{t-1} with this token's channel signal (see note above).
            sig = res["adaptation_signal"] if gate_arm == "learn_self" else err
            if armed is not None:
                sig_ema = sig if sig_ema is None else EMA * sig_ema + (1 - EMA) * sig
                r = sig_ema - sig
                reward_ema = EMA * reward_ema + (1 - EMA) * r
                adv = r - reward_ema
                af, aa, ap = armed
                gw += GLR * adv * (aa - ap) * af
                gb += GLR * adv * (aa - ap)
            gb -= RATE_PULL * (rate_ema - 0.5)
            armed = (f, float(gate_open), p_open)

        if model.hidden_dim != prev_n:
            events.append({"t": t + 1, "from": prev_n, "to": model.hidden_dim,
                           "kind": "grow" if model.hidden_dim > prev_n else "prune"})
            prev_n = model.hidden_dim

        ts_rows.append({"t": t + 1, "prequential_err": err, "neurons": model.hidden_dim,
                        "adaptation_signal": res["adaptation_signal"],
                        "complexity_metric": res["complexity_metric"],
                        "updated": int(res.get("updated", True)),
                        "rolled_back": int(res.get("rolled_back", False))})

        tk = t + 1
        if tk % cfg["probe_cadence"] == 0 or tk == cfg["warmup_ref"]:
            H = hidden_reps(model, drift_X)
            if tk == cfg["warmup_ref"]:
                H_ref = H
            cka = linear_cka(H_ref, H) if H_ref is not None else float("nan")
            held_mae = float(torch.abs(clean_predict(model, held_X) - held_y).mean()) * scale
            roll = float(np.mean(err_trace[max(0, t - 499): t + 1]))
            row = {"t": tk, "cka": cka,
                   "drift": 1.0 - cka if np.isfinite(cka) else float("nan"),
                   "held_ahead_mae": held_mae, "rolling_preq_mae": roll,
                   "neurons": model.hidden_dim}
            row.update(rep_stats(H))
            snap_rows.append(row)

        if tk == cfg["warmup_ref"] or tk % cfg["plast_every"] == 0:
            k = len(plast_rows)
            seg_X, seg_y = plasticity_segment(gaps_in, cfg, k)
            with redirect_stdout(devnull):
                p = plasticity_probe(model, seg_X, seg_y, adapt_steps=cfg["plast_steps"])
            p["mae_first"] *= scale
            p["mae_last"] *= scale
            p["t"] = tk
            plast_rows.append(p)

        if tk % 2500 == 0:
            print(f"  [{arm} seed={seed}] t={tk}/{tokens} neurons={model.hidden_dim} "
                  f"roll_mae={np.mean(err_trace[max(0, t - 499): t + 1]):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    devnull.close()
    dur = time.time() - t0

    pd.DataFrame(ts_rows).to_csv(os.path.join(run_dir, "timeseries.csv"), index=False)
    pd.DataFrame(snap_rows).to_csv(os.path.join(run_dir, "snapshots.csv"), index=False)
    pd.DataFrame(plast_rows).to_csv(os.path.join(run_dir, "plasticity.csv"), index=False)
    pd.DataFrame(events).to_csv(os.path.join(run_dir, "events.csv"), index=False)
    if arm == "B":
        np.save(os.path.join(run_dir, "error_trace.npy"), err_trace)

    n_grow = sum(1 for e in events if e["kind"] == "grow")
    n_prune = sum(1 for e in events if e["kind"] == "prune")
    n_updates = int(sum(r["updated"] for r in ts_rows))
    n_rollbacks = int(sum(r["rolled_back"] for r in ts_rows))
    meta = {"arm": arm, "seed": seed, "tokens": tokens, "device": device_str,
            "harness_version": HARNESS_VERSION, "config": cfg, "baselines": baselines,
            "norm": {"enabled": bool(cfg.get("norm", False)), "mu": mu, "sigma": sigma},
            "gate_arm": gate_arm, "updates_applied": n_updates, "rollbacks": n_rollbacks,
            "duration_s": round(dur, 1),
            "final_neurons": model.hidden_dim, "n_grow": n_grow, "n_prune": n_prune,
            "final_drift": snap_rows[-1]["drift"] if snap_rows else None,
            "final_held_ahead_mae": snap_rows[-1]["held_ahead_mae"] if snap_rows else None}
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] {arm} seed={seed}: {dur:.0f}s, neurons {cfg['hidden_init']}->{model.hidden_dim}, "
          f"grow={n_grow} prune={n_prune}, final drift={meta['final_drift']}, "
          f"held-ahead MAE={meta['final_held_ahead_mae']:.3f}", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "C"], default="A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=CFG["tokens"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true", help="Arm A, seed 0, 2k tokens")
    ap.add_argument("--all", action="store_true", help="3 arms x 3 seeds, B before C")
    ap.add_argument("--norm", action="store_true",
                    help="repair R1: z-score inputs/targets vs causal 1k-gap calibration window")
    ap.add_argument("--leaky", type=float, default=0.0,
                    help="repair R2: leaky-activation slope on the stabilized subclass (e.g. 0.01)")
    ap.add_argument("--tag", default="",
                    help="results subdir suffix (runs_<tag>); keeps repair runs from overwriting the baseline record")
    ap.add_argument("--grav", action="store_true", help="CIMM stabilizer: entropy-baseline gravity pullback")
    ap.add_argument("--vlr", action="store_true", help="CIMM stabilizer: variance-damped learning rate")
    ap.add_argument("--qclip", action="store_true", help="CIMM stabilizer: QFI-adaptive gradient clipping")
    ap.add_argument("--gate", action="store_true", help="CIMM stabilizer: every-10 updates + entropy-validated rollback")
    ap.add_argument("--gate-arm", choices=["none", "ent", "err", "rand", "learn_self", "learn_outer"],
                    default="none",
                    help="origin gate (ent/err/rand, ~50%% rate) or learned gate "
                         "(learn_self = REINFORCE on self-measured loss; learn_outer = on realized error)")
    ap.add_argument("--fixed-plast", action="store_true",
                    help="plasticity probes reuse one fixed segment (round-2 metric fix)")
    args = ap.parse_args()

    results_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_root, exist_ok=True)

    cfg = dict(CFG)
    cfg["norm"] = args.norm
    cfg["leaky"] = args.leaky
    cfg["tag"] = args.tag
    cfg["stab"] = {"gravity": args.grav, "vlr": args.vlr, "qclip": args.qclip, "gate": args.gate}
    cfg["gate_arm"] = args.gate_arm
    cfg["fixed_plast"] = args.fixed_plast

    print(f"Sieving primes to {CFG['sieve_limit']} ...", flush=True)
    gaps = prime_gaps(CFG["sieve_limit"])
    print(f"{len(gaps)} gaps available.", flush=True)

    if args.smoke:
        run_one("A", 0, 2_000, args.device, results_root, gaps, dict(cfg))
        return

    if args.all:
        for seed in (0, 1, 2):
            for arm in ("A", "B", "C"):
                run_one(arm, seed, args.tokens, args.device, results_root, gaps, dict(cfg))
        return

    run_one(args.arm, args.seed, args.tokens, args.device, results_root, gaps, dict(cfg))


if __name__ == "__main__":
    main()
