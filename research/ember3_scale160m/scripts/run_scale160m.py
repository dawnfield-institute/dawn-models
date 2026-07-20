"""
Ember III rung 1 runner — gate-origin arms on pythia-160m, continuous next-token
learning on the WikiText-103 stream. Runs on CT103 (/data/ember3), one (arm, seed)
per invocation.

PRE-REGISTERED (2026-07-19, locked before any full run — see also README.md):
  arms: none | err (prequential loss >= rolling median-200, exogenous)
             | ent (mean predictive entropy >= rolling median-200, endogenous)
             | rand (fair coin) | frozen_ref (no updates; reference pass)
  update: full-rank Adam lr 1e-5, clip 1.0, one update per accepted 512-token chunk;
          prequential loss measured on the chunk BEFORE its update
  drift: linear CKA vs reference snapshot at chunk 300, fixed 64-seq probe batch
         (validation), layer-6 mean-pooled hiddens, cadence 100 chunks
  competence: prequential edge vs frozen_ref on identical chunks; held-out CE on a
         fixed 128-seq validation batch
  plasticity: every 2500 chunks, deepcopy + fresh Adam, adapt 100 chunks on ONE
         fixed far-ahead segment (chunks 30_000-30_100), first-20 vs last-20 loss
         improvement, clone discarded

Seeding note (documented deviation from the "2 seeds per arm" plan line): pythia has
no dropout and the stream is fixed, so none/err/ent are deterministic up to CUDA
nondeterminism — seeds only randomize the `rand` gate. Protocol: none/ent x1,
err x2 (second run sizes the CUDA-jitter noise floor), rand x3 seeds, frozen_ref x1.
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = "/data/ember3"
os.environ.setdefault("HF_HOME", f"{ROOT}/hf-cache")

from transformers import AutoModelForCausalLM  # noqa: E402

SEQ = 512
CADENCE = 100
REF_AT = 300
PLAST_EVERY = 2500
PLAST_STEPS = 100
GATE_WIN = 200
LR = 1e-5


def get_chunk(train, i):
    a = train[i * SEQ:(i + 1) * SEQ].astype(np.int64)
    return torch.tensor(a, dtype=torch.long).unsqueeze(0)


def build_eval_batches(valid):
    probe = torch.tensor(
        valid[: 64 * SEQ].astype(np.int64).reshape(64, SEQ), dtype=torch.long)
    held = torch.tensor(
        valid[64 * SEQ: (64 + 128) * SEQ].astype(np.int64).reshape(128, SEQ),
        dtype=torch.long)
    return probe, held


def linear_cka(H1, H2):
    H1 = H1 - H1.mean(axis=0, keepdims=True)
    H2 = H2 - H2.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(H1.T @ H2, ord="fro") ** 2
    denom = np.linalg.norm(H1.T @ H1, ord="fro") * np.linalg.norm(H2.T @ H2, ord="fro")
    return float(cross / denom) if denom > 1e-12 else float("nan")


@torch.no_grad()
def probe_hiddens(model, probe, device, layer=6, bs=16):
    model.eval()
    outs = []
    for i in range(0, probe.shape[0], bs):
        b = probe[i: i + bs].to(device)
        h = model(b, output_hidden_states=True).hidden_states[layer]
        outs.append(h.mean(dim=1).double().cpu().numpy())
    model.train()
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def heldout_loss(model, held, device, bs=16):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, held.shape[0], bs):
        b = held[i: i + bs].to(device)
        out = model(b, labels=b)
        tot += float(out.loss) * b.shape[0]
        n += b.shape[0]
    model.train()
    return tot / n


@torch.no_grad()
def mean_pred_entropy(logits):
    # mean over positions of token-distribution entropy; chunked over positions
    lp = F.log_softmax(logits[0].float(), dim=-1)
    return float(-(lp.exp() * lp).sum(-1).mean())


def plasticity_probe(model, train, device):
    clone = copy.deepcopy(model)
    opt = torch.optim.Adam(clone.parameters(), lr=LR)
    losses = []
    for k in range(PLAST_STEPS):
        x = get_chunk(train, 30_000 + k).to(device)
        out = clone(x, labels=x)
        losses.append(float(out.loss))
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(clone.parameters(), 1.0)
        opt.step()
    first, last = float(np.mean(losses[:20])), float(np.mean(losses[-20:]))
    del clone, opt
    torch.cuda.empty_cache()
    return {"loss_first20": first, "loss_last20": last,
            "plasticity": (first - last) / (first + 1e-9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["none", "err", "ent", "rand", "frozen_ref",
                             "band", "excess"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunks", type=int, default=10_000)
    ap.add_argument("--rep", type=int, default=0, help="repeat index (CUDA-jitter sizing)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + 7770)
    device = torch.device("cuda")

    train = np.load(f"{ROOT}/wt103_train.npy")
    valid = np.load(f"{ROOT}/wt103_valid.npy")
    probe, held = build_eval_batches(valid)

    # Explicit fp32: the checkpoint's stored dtype is fp16, and pure-fp16 full
    # fine-tuning without loss scaling NaNs within the first steps (pre-flight
    # finding — the NaN closed the err gate permanently at exactly 50 updates).
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m", torch_dtype=torch.float32).to(device)
    model.train()

    # Round-6 gates (pre-registered 2026-07-19, direct test of the round-5
    # aleatoric hypothesis):
    #   band:   update iff loss in [p25, p75) of the rolling window — skips the
    #           trivially-easy AND the irreducible tail; 50% rate by construction.
    #   excess: update iff (loss - frozen_ref loss on the SAME chunk) >= rolling
    #           median of excess — the frozen pretrained model is a fixed,
    #           exogenous difficulty prior, so excess estimates REDUCIBLE
    #           (epistemic) surprise. "Gate on learnability, not difficulty."
    ref_loss = None
    if args.arm == "excess":
        ref_ts = pd.read_csv(f"{ROOT}/results/frozen_ref_0/timeseries.csv")
        ref_loss = ref_ts["loss"].values
        assert len(ref_loss) >= args.chunks, "frozen_ref pass shorter than stream"
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    frozen = args.arm == "frozen_ref"

    run_name = f"{args.arm}_{args.seed}" + (f"_r{args.rep}" if args.rep else "")
    out_dir = f"{ROOT}/results/{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    ts_rows, snap_rows, plast_rows = [], [], []
    gate_win = []
    H_ref = None
    t0 = time.time()

    for t in range(args.chunks):
        x = get_chunk(train, t).to(device)
        out = model(x, labels=x)
        loss = float(out.loss)
        if not np.isfinite(loss):
            raise RuntimeError(f"non-finite loss at chunk {t} — refusing to continue "
                               f"(silent NaN would corrupt gate logic and metrics)")
        ent = mean_pred_entropy(out.logits.detach())

        gate_open = True
        if args.arm == "err":
            gate_open = len(gate_win) < 50 or loss >= float(np.median(gate_win))
            gate_win.append(loss)
        elif args.arm == "ent":
            gate_open = len(gate_win) < 50 or ent >= float(np.median(gate_win))
            gate_win.append(ent)
        elif args.arm == "rand":
            gate_open = bool(rng.random() < 0.5)
        elif args.arm == "band":
            if len(gate_win) < 50:
                gate_open = True
            else:
                p25, p75 = np.percentile(gate_win, [25, 75])
                gate_open = bool(p25 <= loss < p75)
            gate_win.append(loss)
        elif args.arm == "excess":
            ex = loss - float(ref_loss[t])
            gate_open = len(gate_win) < 50 or ex >= float(np.median(gate_win))
            gate_win.append(ex)
        gate_win = gate_win[-GATE_WIN:]

        updated = False
        if not frozen and gate_open:
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            updated = True
        else:
            opt.zero_grad(set_to_none=True)

        ts_rows.append({"t": t + 1, "loss": loss, "entropy": ent,
                        "gate_open": int(gate_open), "updated": int(updated)})

        tk = t + 1
        if tk % CADENCE == 0:
            H = probe_hiddens(model, probe, device)
            if tk == REF_AT:
                H_ref = H
            cka = linear_cka(H_ref, H) if H_ref is not None else float("nan")
            hl = heldout_loss(model, held, device)
            snap_rows.append({"t": tk, "cka": cka,
                              "drift": 1.0 - cka if np.isfinite(cka) else float("nan"),
                              "heldout_loss": hl})
        if not frozen and (tk == REF_AT or tk % PLAST_EVERY == 0):
            p = plasticity_probe(model, train, device)
            p["t"] = tk
            plast_rows.append(p)
        if tk % 500 == 0:
            rate = np.mean([r["updated"] for r in ts_rows[-500:]])
            print(f"[{run_name}] t={tk}/{args.chunks} loss={loss:.3f} "
                  f"upd_rate={rate:.2f} vram={torch.cuda.max_memory_allocated()/2**30:.1f}G "
                  f"({time.time()-t0:.0f}s)", flush=True)

    pd.DataFrame(ts_rows).to_csv(f"{out_dir}/timeseries.csv", index=False)
    pd.DataFrame(snap_rows).to_csv(f"{out_dir}/snapshots.csv", index=False)
    pd.DataFrame(plast_rows).to_csv(f"{out_dir}/plasticity.csv", index=False)
    meta = {"arm": args.arm, "seed": args.seed, "rep": args.rep,
            "chunks": args.chunks, "lr": LR, "seq": SEQ,
            "updates": int(sum(r["updated"] for r in ts_rows)),
            "duration_s": round(time.time() - t0, 1),
            "final_heldout_loss": snap_rows[-1]["heldout_loss"] if snap_rows else None,
            "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {run_name}: {meta['duration_s']}s, updates {meta['updates']}, "
          f"final heldout {meta['final_heldout_loss']:.4f}", flush=True)


if __name__ == "__main__":
    main()
