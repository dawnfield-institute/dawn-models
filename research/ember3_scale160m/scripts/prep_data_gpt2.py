"""
Round-7 data prep: WikiText-103 tokenized with the GPT-2 tokenizer (the twins'
tokenizer). Same split design as prep_data.py; outputs wt103gpt2_{train,valid}.npy.
Runs on CT103 under /data/ember3 (WT103 raw is already in the HF cache).
"""

import os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = "/data/ember3"
os.environ.setdefault("HF_HOME", f"{ROOT}/hf-cache")


def tokenize_split(ds, tok, name):
    ids = []
    for rec in ds:
        t = rec["text"]
        if t.strip():
            ids.extend(tok(t, add_special_tokens=False)["input_ids"])
    arr = np.array(ids, dtype=np.uint16)
    np.save(f"{ROOT}/wt103gpt2_{name}.npy", arr)
    print(f"{name}: {len(arr):,} tokens", flush=True)


def main():
    tok = AutoTokenizer.from_pretrained("gpt2")
    assert tok.vocab_size < 65536
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    tokenize_split(ds["train"], tok, "train")
    tokenize_split(ds["validation"], tok, "valid")
    print("done", flush=True)


if __name__ == "__main__":
    main()
