"""
Round 7 — the physics twins: ember_v1_mixed (SECAttention + XiNorm + RBFResidual +
global PAC conservation) vs vanilla_mixed (same 12L/12H/768d LLaMA-style arch, same
mixed corpus, same 50k pretraining steps, no physics) under identical continuous
learning on the WikiText-103 stream. First measurable role for the PAC-native
architecture at scale.

PRE-REGISTERED (2026-07-19, locked before any full run):
  matched pair: purpose-built twins from cradle's telemetry campaign — identical
    data/steps/dims; ONLY the physics layers + conservation objective differ.
    WT103 exposure during pretraining is EQUAL (both mixed corpora included it).
  stream: WT103 (GPT-2 tokenization), SEQ=256 (the twins' training context),
    10k chunks (~2.6M tokens), document order.
  update: each twin trains on ITS OWN native objective (ember: CE + conservation;
    vanilla: CE) — the physics is architecture + constraint, tested as designed.
    ALL measurements (prequential, gates, held-out) use pure CE from logits,
    identically computed for both.
  arms per twin: frozen_ref | none | rand | excess (excess vs the twin's own
    frozen reference — the doubly-exogenous learnability gate, round-6 stability
    champion). lr 1e-5, clip 1.0.
  metrics: as rounds 5-6 (CKA drift on layer-6 hiddens w/ known jitter caveat —
    competence+plasticity carry the verdict; plasticity on the fixed far segment).

  Interpretation (invariants only): the claim under test is NOT "ember wins" —
  it is "does the physics change continuous-learning behavior?" Any consistent
  difference in edge trajectory, forgetting (held-out rise), plasticity retention,
  or gate response between twins is attributable to the physics package.
  Ember's conservation_loss is logged as physics telemetry throughout.
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = "/data/ember3"
os.environ.setdefault("HF_HOME", f"{ROOT}/hf-cache")
sys.path.insert(0, ROOT)  # ember/ package ships to /data/ember3/ember

from ember.config import EmberConfig  # noqa: E402
from ember.v1_model import EmberV1ForCausalLM  # noqa: E402
from ember.vanilla import VanillaForCausalLM  # noqa: E402

SEQ = 256
CADENCE = 100
REF_AT = 300
PLAST_EVERY = 2500
PLAST_STEPS = 100
GATE_WIN = 200
LR = 1e-5


def load_twin(which, device):
    ckpt = f"{ROOT}/twins/{'ember_v1_mixed' if which == 'ember' else 'vanilla_mixed'}"
    config = EmberConfig.from_pretrained(ckpt)
    model = (EmberV1ForCausalLM if which == "ember" else VanillaForCausalLM)(config)
    sd = torch.load(f"{ckpt}/final.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(sd)  # strict — the twin must load exactly
    del sd
    return model.to(device).float()


def get_chunk(train, i):
    a = train[i * SEQ:(i + 1) * SEQ].astype(np.int64)
    return torch.tensor(a, dtype=torch.long).unsqueeze(0)


def ce_from_logits(logits, x):
    """Pure CE, identical for both twins — the measurement channel."""
    sl = logits[..., :-1, :].contiguous()
    st = x[..., 1:].contiguous()
    return float(F.cross_entropy(sl.view(-1, sl.shape[-1]), st.view(-1)))


def linear_cka(H1, H2):
    H1 = H1 - H1.mean(axis=0, keepdims=True)
    H2 = H2 - H2.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(H1.T @ H2, ord="fro") ** 2
    denom = np.linalg.norm(H1.T @ H1, ord="fro") * np.linalg.norm(H2.T @ H2, ord="fro")
    return float(cross / denom) if denom > 1e-12 else float("nan")


@torch.no_grad()
def probe_hiddens(model, probe, device, bs=16):
    model.eval()
    outs = []
    for i in range(0, probe.shape[0], bs):
        b = probe[i: i + bs].to(device)
        hs = model(b, output_hidden_states=True).hidden_states
        h = hs[min(6, len(hs) - 1)]
        outs.append(h.mean(dim=1).double().cpu().numpy())
    model.train()
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def heldout_ce(model, held, device, bs=16):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, held.shape[0], bs):
        b = held[i: i + bs].to(device)
        out = model(b, labels=b)
        tot += ce_from_logits(out.logits, b) * b.shape[0]
        n += b.shape[0]
    model.train()
    return tot / n


def plasticity_probe(model, train, device):
    clone = copy.deepcopy(model)
    opt = torch.optim.Adam(clone.parameters(), lr=LR)
    ces = []
    for k in range(PLAST_STEPS):
        x = get_chunk(train, 30_000 + k).to(device)
        out = clone(x, labels=x)
        ces.append(ce_from_logits(out.logits.detach(), x))
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(clone.parameters(), 1.0)
        opt.step()
    first, last = float(np.mean(ces[:20])), float(np.mean(ces[-20:]))
    del clone, opt
    torch.cuda.empty_cache()
    return {"ce_first20": first, "ce_last20": last,
            "plasticity": (first - last) / (first + 1e-9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["ember", "vanilla"])
    ap.add_argument("--arm", required=True, choices=["frozen_ref", "none", "rand", "excess"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunks", type=int, default=10_000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + 7770)
    device = torch.device("cuda")

    train = np.load(f"{ROOT}/wt103gpt2_train.npy")
    valid = np.load(f"{ROOT}/wt103gpt2_valid.npy")
    probe = torch.tensor(valid[: 64 * SEQ].astype(np.int64).reshape(64, SEQ), dtype=torch.long)
    held = torch.tensor(valid[64 * SEQ: (64 + 128) * SEQ].astype(np.int64).reshape(128, SEQ),
                        dtype=torch.long)

    model = load_twin(args.model, device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    frozen = args.arm == "frozen_ref"

    ref_ce = None
    if args.arm == "excess":
        ref_ts = pd.read_csv(f"{ROOT}/results_twins/{args.model}_frozen_ref_0/timeseries.csv")
        ref_ce = ref_ts["ce"].values
        assert len(ref_ce) >= args.chunks

    run_name = f"{args.model}_{args.arm}_{args.seed}"
    out_dir = f"{ROOT}/results_twins/{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    ts_rows, snap_rows, plast_rows = [], [], []
    gate_win = []
    H_ref = None
    t0 = time.time()

    for t in range(args.chunks):
        x = get_chunk(train, t).to(device)
        out = model(x, labels=x)
        ce = ce_from_logits(out.logits.detach(), x)
        if not np.isfinite(ce):
            raise RuntimeError(f"non-finite CE at chunk {t}")
        cons = float(getattr(out, "conservation_loss", 0.0) or 0.0)

        gate_open = True
        if args.arm == "rand":
            gate_open = bool(rng.random() < 0.5)
        elif args.arm == "excess":
            ex = ce - float(ref_ce[t])
            gate_open = len(gate_win) < 50 or ex >= float(np.median(gate_win))
            gate_win.append(ex)
            gate_win = gate_win[-GATE_WIN:]

        updated = False
        if not frozen and gate_open:
            opt.zero_grad()
            out.loss.backward()  # native objective: ember trains WITH conservation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            updated = True
        else:
            opt.zero_grad(set_to_none=True)

        ts_rows.append({"t": t + 1, "ce": ce, "conservation": cons,
                        "gate_open": int(gate_open), "updated": int(updated)})

        tk = t + 1
        if tk % CADENCE == 0:
            H = probe_hiddens(model, probe, device)
            if tk == REF_AT:
                H_ref = H
            cka = linear_cka(H_ref, H) if H_ref is not None else float("nan")
            snap_rows.append({"t": tk, "cka": cka,
                              "drift": 1.0 - cka if np.isfinite(cka) else float("nan"),
                              "heldout_ce": heldout_ce(model, held, device)})
        if not frozen and (tk == REF_AT or tk % PLAST_EVERY == 0):
            p = plasticity_probe(model, train, device)
            p["t"] = tk
            plast_rows.append(p)
        if tk % 1000 == 0:
            print(f"[{run_name}] t={tk}/{args.chunks} ce={ce:.3f} cons={cons:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    pd.DataFrame(ts_rows).to_csv(f"{out_dir}/timeseries.csv", index=False)
    pd.DataFrame(snap_rows).to_csv(f"{out_dir}/snapshots.csv", index=False)
    pd.DataFrame(plast_rows).to_csv(f"{out_dir}/plasticity.csv", index=False)
    meta = {"model": args.model, "arm": args.arm, "seed": args.seed,
            "chunks": args.chunks, "lr": LR, "seq": SEQ,
            "updates": int(sum(r["updated"] for r in ts_rows)),
            "duration_s": round(time.time() - t0, 1),
            "final_heldout_ce": snap_rows[-1]["heldout_ce"] if snap_rows else None,
            "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {run_name}: {meta['duration_s']}s, updates {meta['updates']}, "
          f"final heldout CE {meta['final_heldout_ce']:.4f}", flush=True)


if __name__ == "__main__":
    main()
