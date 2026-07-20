"""
Round-10 Experiment B — per-expert update-mass telemetry (SCBF v2 HookSpine,
stacked-tensor fix for round 8's nulled instrument).

Mechanism under test: frozen-router x usage-skew — with routing pinned, update
pressure lands proportionally on the highest-usage experts (which round 9 showed
are the knowledge stores). Re-runs the round-8 arms (5k WT103 chunks, fused SGD
5e-4) with per-(layer, expert) update-mass accumulation + router selection counts.
--arm experts (frozen router; primary) and --arm full (plastic router; contrast).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = "/data/ember3"
os.environ.setdefault("HF_HOME", "/data/models/ember3-olmoe/hf-cache")
sys.path.insert(0, ROOT)

from transformers import OlmoeForCausalLM  # noqa: E402
from scbf_v2 import HookSpine, mask_all, olmoe_experts_only, olmoe_expert_stacked_fn  # noqa: E402

SEQ = 512
LR = 5e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["experts", "full"])
    ap.add_argument("--chunks", type=int, default=5_000)
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)

    train = np.load(f"{ROOT}/wt103olmoe_train.npy")
    model = OlmoeForCausalLM.from_pretrained(
        "allenai/OLMoE-1B-7B-0924", torch_dtype=torch.bfloat16).to(device)
    model.train()

    spine = HookSpine(
        model,
        trainable=(olmoe_experts_only() if args.arm == "experts" else mask_all),
        lr=LR, clip=1.0, fused=True,
        stacked_fn=olmoe_expert_stacked_fn)

    L = model.config.num_hidden_layers
    E = model.config.num_experts
    sel = np.zeros((L, E), dtype=np.int64)
    t0 = time.time()

    for t in range(args.chunks):
        a = train[t * SEQ:(t + 1) * SEQ].astype(np.int64)
        x = torch.tensor(a, dtype=torch.long).unsqueeze(0).to(device)
        out = model(x, labels=x, output_router_logits=True)
        if not np.isfinite(float(out.loss)):
            raise RuntimeError(f"non-finite loss at {t}")
        out.loss.backward()  # HookSpine applies updates + accumulates masses
        with torch.no_grad():
            for li, rl in enumerate(out.router_logits):
                topk = torch.topk(rl.float(), model.config.num_experts_per_tok, -1).indices
                idx, cnt = torch.unique(topk, return_counts=True)
                sel[li, idx.cpu().numpy()] += cnt.cpu().numpy()
        del out
        if (t + 1) % 1000 == 0:
            print(f"[{args.arm}] {t+1}/{args.chunks} "
                  f"({time.time()-t0:.0f}s, vram {torch.cuda.max_memory_allocated()/2**30:.1f}G)",
                  flush=True)

    # aggregate per-(layer, expert) over the projection keys
    mass = np.zeros((L, E), dtype=np.float64)
    for (layer, _proj), arr in spine.stacked_mass.items():
        mass[layer] += np.asarray(arr)

    out_dir = f"{ROOT}/results_expertmass/{args.arm}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/mass.npy", mass)
    np.save(f"{out_dir}/sel.npy", sel)

    # per-layer Spearman rank correlation mass vs selections
    from scipy.stats import spearmanr
    rhos = [float(spearmanr(mass[li], sel[li]).statistic) for li in range(L)]
    conc = np.sort(mass.flatten())[::-1]
    conc = conc.cumsum() / conc.sum()
    top26 = float(conc[int(0.26 * mass.size)])
    usage = np.sort(sel.flatten())[::-1].astype(float)
    usage = usage.cumsum() / usage.sum()
    utop26 = float(usage[int(0.26 * sel.size)])
    meta = {"arm": args.arm, "chunks": args.chunks,
            "spearman_by_layer": rhos, "spearman_mean": float(np.mean(rhos)),
            "mass_top26_share": top26, "usage_top26_share": utop26,
            "updates": spine.updates, "clip_events": spine.clip_events,
            "duration_s": round(time.time() - t0, 1),
            "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {args.arm}: spearman mean {meta['spearman_mean']:.3f}, "
          f"mass top26 {top26:.3f} vs usage top26 {utop26:.3f}", flush=True)


if __name__ == "__main__":
    main()
