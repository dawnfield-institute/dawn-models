"""
Round-10 Experiment A — cloze knowledge probe (SCBF v2's first user).
Closes round 9's deflation question: was the experts arm's TriviaQA −10.4pt
knowledge erosion, or output-format shift?

Generation-free scoring (ClozeKnowledgeProbe): per-fact answer log-likelihood +
first-token top-1 under teacher forcing, on the SAME TriviaQA validation facts
(first 2000, matching round 9's --limit slice), for {frozen, experts, full}.
Per-fact results saved for paired analysis. Runs on CT103.
"""

import json
import os
import sys

import numpy as np
import torch

ROOT = "/data/ember3"
os.environ.setdefault("HF_HOME", "/data/models/ember3-olmoe/hf-cache")
sys.path.insert(0, ROOT)

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer, OlmoeForCausalLM  # noqa: E402
from scbf_v2 import ClozeKnowledgeProbe  # noqa: E402

MODELS = {
    "frozen": "allenai/OLMoE-1B-7B-0924",
    "experts": "/data/models/ember3-olmoe/adapted/experts",
    "full": "/data/models/ember3-olmoe/adapted/full",
}
N_FACTS = 2000


def main():
    tok = AutoTokenizer.from_pretrained(MODELS["frozen"])
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    facts = []
    for rec in ds:
        q = rec["question"].strip()
        a = rec["answer"]["value"].strip()
        if q and a:
            facts.append((f"Question: {q}\nAnswer:", a))
        if len(facts) >= N_FACTS:
            break
    print(f"{len(facts)} facts", flush=True)
    probe = ClozeKnowledgeProbe(facts, tok)
    print(f"{len(probe.items)} usable after tokenization", flush=True)

    out = {}
    for name, path in MODELS.items():
        model = OlmoeForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16).to("cuda")
        res = probe(model, batch_size=8)
        out[name] = res
        print(f"[{name}] mean_ll {res['cloze_mean_ll']:.4f} "
              f"top1 {res['cloze_top1']:.4f} n {int(res['cloze_n'])}", flush=True)
        del model
        torch.cuda.empty_cache()

    os.makedirs(f"{ROOT}/results_cloze", exist_ok=True)
    with open(f"{ROOT}/results_cloze/cloze_results.json", "w") as f:
        json.dump(out, f)

    # paired deltas vs frozen
    f_ll = np.array(out["frozen"]["per_fact_ll"])
    for name in ("experts", "full"):
        d = np.array(out[name]["per_fact_ll"]) - f_ll
        se = d.std(ddof=1) / np.sqrt(len(d))
        t1 = out[name]["cloze_top1"] - out["frozen"]["cloze_top1"]
        print(f"PAIRED {name}-frozen: mean dLL {d.mean():+.4f} (SE {se:.4f}, "
              f"2xSE {2*se:.4f}) | top1 delta {t1:+.4f}", flush=True)
    print("[done] exp_cloze", flush=True)


if __name__ == "__main__":
    main()
