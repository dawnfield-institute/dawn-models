"""
Round 8 — the MoE continuous learner: OLMoE-1B-7B (base 0924, bf16) under
continuous next-token learning on the WT103 stream, RTX 3090 (CT103).

PRE-REGISTERED (2026-07-19/20, locked before any full run — see plan + README):
  arms:
    frozen_ref  no updates (reference pass, no_grad)
    experts     trunk + router frozen; only expert FFNs (`mlp.experts.` params) train
    full        everything trains
  update rule: **fused-backward SGD** — per-parameter update applied inside a
    post-accumulate-grad hook and the gradient freed immediately (peak grad memory
    ~one layer; the model updates WHILE back-propagating). lr 5e-4, per-param clip
    (grad scaled to max-norm 1.0 per tensor). Same rule for all arms.
    NOTE: optimizer differs from rounds 5-7 (Adam) — within-round arms are
    comparable; cross-round only qualitatively.
  loss: model's native objective (CE + router aux as configured); ALL measurements
    use pure CE from logits.
  metrics: prequential CE edge vs frozen_ref; forgetting split — held-out CE on
    WT103-valid (stream domain) AND TinyStories-valid (off-domain, the
    "keeps its learnings" probe); CKA drift on layer-8 hiddens (jitter caveat);
    expert telemetry — per-(layer,expert) router selection counts and cumulative
    |update| mass from the hooks. No plasticity probe (7B deepcopy infeasible).
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = "/data/ember3"
os.environ.setdefault("HF_HOME", "/data/models/ember3-olmoe/hf-cache")

from transformers import OlmoeForCausalLM  # noqa: E402

MODEL = "allenai/OLMoE-1B-7B-0924"
SEQ = 512
CADENCE = 100
REF_AT = 300
GATE_LR = 5e-4
CLIP = 1.0


def ce_from_logits(logits, x):
    sl = logits[..., :-1, :].contiguous().float()
    st = x[..., 1:].contiguous()
    return float(F.cross_entropy(sl.view(-1, sl.shape[-1]), st.view(-1)))


def linear_cka(H1, H2):
    H1 = H1 - H1.mean(axis=0, keepdims=True)
    H2 = H2 - H2.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(H1.T @ H2, ord="fro") ** 2
    denom = np.linalg.norm(H1.T @ H1, ord="fro") * np.linalg.norm(H2.T @ H2, ord="fro")
    return float(cross / denom) if denom > 1e-12 else float("nan")


def get_chunk(train, i):
    a = train[i * SEQ:(i + 1) * SEQ].astype(np.int64)
    return torch.tensor(a, dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def probe_hiddens(model, probe, device, layer=8, bs=4):
    model.eval()
    outs = []
    for i in range(0, probe.shape[0], bs):
        b = probe[i: i + bs].to(device)
        h = model(b, output_hidden_states=True).hidden_states[layer]
        outs.append(h.mean(dim=1).float().cpu().double().numpy())
    model.train()
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def heldout_ce(model, held, device, bs=4):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, held.shape[0], bs):
        b = held[i: i + bs].to(device)
        tot += ce_from_logits(model(b).logits, b) * b.shape[0]
        n += b.shape[0]
    model.train()
    return tot / n


def expert_id_of(name):
    """'model.layers.L.mlp.experts.E....' -> (L, E) or None."""
    parts = name.split(".")
    try:
        li = parts.index("layers")
        if "experts" in parts:
            return int(parts[li + 1]), int(parts[parts.index("experts") + 1])
    except (ValueError, IndexError):
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["frozen_ref", "experts", "full"])
    ap.add_argument("--chunks", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true",
                    help="save adapted model (HF format) for benchmarking — round 9")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    train = np.load(f"{ROOT}/wt103olmoe_train.npy")
    wt_valid = np.load(f"{ROOT}/wt103olmoe_valid.npy")
    ts_valid = np.load(f"{ROOT}/tinystories_olmoe_valid.npy")
    probe = torch.tensor(wt_valid[: 64 * SEQ].astype(np.int64).reshape(64, SEQ), dtype=torch.long)
    held_dom = torch.tensor(wt_valid[64 * SEQ: (64 + 64) * SEQ].astype(np.int64).reshape(64, SEQ),
                            dtype=torch.long)
    held_off = torch.tensor(ts_valid[: 64 * SEQ].astype(np.int64).reshape(64, SEQ),
                            dtype=torch.long)

    model = OlmoeForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    model.train()
    frozen = args.arm == "frozen_ref"

    n_layers = model.config.num_hidden_layers
    n_experts = model.config.num_experts
    sel_counts = np.zeros((n_layers, n_experts), dtype=np.int64)
    upd_mass = np.zeros((n_layers, n_experts), dtype=np.float64)
    hook_fired = {"n": 0}

    if not frozen:
        for name, p in model.named_parameters():
            trainable = True if args.arm == "full" else ("mlp.experts." in name)
            p.requires_grad = trainable
            if trainable:
                eid = expert_id_of(name)

                def make_hook(eid):
                    def hook(param):
                        g = param.grad
                        if g is None:
                            return
                        gn = g.norm()
                        if torch.isfinite(gn) and gn > CLIP:
                            g = g * (CLIP / gn)
                        param.data.add_(g, alpha=-GATE_LR)  # fused SGD step
                        if eid is not None:
                            upd_mass[eid[0], eid[1]] += float(gn) * GATE_LR
                        hook_fired["n"] += 1
                        param.grad = None  # free immediately — peak grad mem ~1 layer
                    return hook

                p.register_post_accumulate_grad_hook(make_hook(eid))
    else:
        for p in model.parameters():
            p.requires_grad = False

    run_name = f"{args.arm}_{args.seed}"
    out_dir = f"{ROOT}/results_moe/{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    ts_rows, snap_rows = [], []
    H_ref = None
    t0 = time.time()

    for t in range(args.chunks):
        x = get_chunk(train, t).to(device)
        if frozen:
            with torch.no_grad():
                out = model(x, output_router_logits=True)
            ce = ce_from_logits(out.logits, x)
        else:
            out = model(x, labels=x, output_router_logits=True)
            ce = ce_from_logits(out.logits.detach(), x)
            if not np.isfinite(ce):
                raise RuntimeError(f"non-finite CE at chunk {t}")
            out.loss.backward()  # updates happen inside the hooks

        with torch.no_grad():
            for li, rl in enumerate(out.router_logits):
                topk = torch.topk(rl.float(), model.config.num_experts_per_tok, dim=-1).indices
                idx, cnt = torch.unique(topk, return_counts=True)
                sel_counts[li, idx.cpu().numpy()] += cnt.cpu().numpy()
        del out

        ts_rows.append({"t": t + 1, "ce": ce})

        tk = t + 1
        if tk % CADENCE == 0:
            H = probe_hiddens(model, probe, device)
            if tk == REF_AT:
                H_ref = H
            cka = linear_cka(H_ref, H) if H_ref is not None else float("nan")
            snap_rows.append({"t": tk, "cka": cka,
                              "drift": 1.0 - cka if np.isfinite(cka) else float("nan"),
                              "heldout_dom": heldout_ce(model, held_dom, device),
                              "heldout_off": heldout_ce(model, held_off, device)})
        if tk % 250 == 0:
            np.save(f"{out_dir}/upd_mass.npy", upd_mass)
            np.save(f"{out_dir}/sel_counts.npy", sel_counts)
        if tk % 500 == 0:
            print(f"[{run_name}] t={tk}/{args.chunks} ce={ce:.3f} "
                  f"vram={torch.cuda.max_memory_allocated()/2**30:.1f}G "
                  f"hooks={hook_fired['n']} ({time.time()-t0:.0f}s)", flush=True)

    pd.DataFrame(ts_rows).to_csv(f"{out_dir}/timeseries.csv", index=False)
    pd.DataFrame(snap_rows).to_csv(f"{out_dir}/snapshots.csv", index=False)
    np.save(f"{out_dir}/upd_mass.npy", upd_mass)
    np.save(f"{out_dir}/sel_counts.npy", sel_counts)
    meta = {"arm": args.arm, "seed": args.seed, "chunks": args.chunks,
            "lr": GATE_LR, "seq": SEQ, "model": MODEL,
            "duration_s": round(time.time() - t0, 1),
            "hook_updates": hook_fired["n"],
            "final_heldout_dom": snap_rows[-1]["heldout_dom"] if snap_rows else None,
            "final_heldout_off": snap_rows[-1]["heldout_off"] if snap_rows else None,
            "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    if args.save and not frozen:
        save_dir = f"/data/models/ember3-olmoe/adapted/{args.arm}"
        model.save_pretrained(save_dir, safe_serialization=True)
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(MODEL).save_pretrained(save_dir)
        print(f"[saved] {save_dir}", flush=True)

    print(f"[done] {run_name}: {meta['duration_s']}s "
          f"heldout dom {meta['final_heldout_dom']} off {meta['final_heldout_off']} "
          f"vram {meta['vram_peak_gb']}G", flush=True)


if __name__ == "__main__":
    main()
