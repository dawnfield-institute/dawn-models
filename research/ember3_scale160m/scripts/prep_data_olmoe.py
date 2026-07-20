"""
Round-8 data prep: WT103 (stream + valid) and TinyStories validation slice
(off-domain forgetting probe), tokenized with the OLMoE tokenizer.
Outputs: wt103olmoe_{train,valid}.npy, tinystories_olmoe_valid.npy.
"""

import os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = "/data/ember3"
os.environ.setdefault("HF_HOME", "/data/models/ember3-olmoe/hf-cache")

MODEL = "allenai/OLMoE-1B-7B-0924"


def tok_texts(texts, tok, path, cap_tokens=None):
    ids = []
    for t in texts:
        if t.strip():
            ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        if cap_tokens and len(ids) >= cap_tokens:
            break
    arr = np.array(ids[:cap_tokens] if cap_tokens else ids, dtype=np.uint16)
    np.save(path, arr)
    print(f"{path}: {len(arr):,} tokens", flush=True)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    assert tok.vocab_size < 65536, f"vocab {tok.vocab_size} exceeds uint16"

    wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    # Stream needs 5k chunks x 512 = 2.56M tokens + margin; cap train at 40M for speed.
    tok_texts((r["text"] for r in wt["train"]), tok,
              f"{ROOT}/wt103olmoe_train.npy", cap_tokens=40_000_000)
    tok_texts((r["text"] for r in wt["validation"]), tok, f"{ROOT}/wt103olmoe_valid.npy")

    ts = load_dataset("roneneldan/TinyStories", split="validation")
    tok_texts((r["text"] for r in ts), tok,
              f"{ROOT}/tinystories_olmoe_valid.npy", cap_tokens=300_000)
    print("done", flush=True)


if __name__ == "__main__":
    main()
