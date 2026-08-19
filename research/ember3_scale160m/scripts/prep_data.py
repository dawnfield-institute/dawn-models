"""
One-time data prep for ember3_scale160m (runs on CT103, everything under /data/ember3).

Downloads WikiText-103-raw + the pythia-160m tokenizer, tokenizes the train split in
document order to a uint16 token array, and writes the PRE-REGISTERED split manifest:

  stream:      chunks [0, 10_000)                (512 tokens each, document order)
  probe:       64 sequences from validation      (drift CKA probe batch, fixed)
  heldout:     128 sequences from validation     (perplexity batch, fixed, disjoint)
  plasticity:  chunks [30_000, 30_100)           (one fixed far-ahead unseen segment)
  reference:   the frozen pretrained model's prequential loss on the stream chunks
               is computed once by run_scale160m.py --arm frozen_ref

Outputs: /data/ember3/wt103_train.npy, wt103_valid.npy, data_manifest.json
"""

import json
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
    path = f"{ROOT}/wt103_{name}.npy"
    np.save(path, arr)
    print(f"{name}: {len(arr):,} tokens -> {path}", flush=True)
    return arr


def main():
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    assert tok.vocab_size < 65536, "uint16 storage requires vocab < 64k"
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    train = tokenize_split(ds["train"], tok, "train")
    valid = tokenize_split(ds["validation"], tok, "valid")

    seq = 512
    n_stream = 10_000
    manifest = {
        "seq_len": seq,
        "stream_chunks": [0, n_stream],
        "plasticity_chunks": [30_000, 30_100],
        "probe_seqs": 64,
        "heldout_seqs": 128,
        "train_tokens": int(len(train)),
        "valid_tokens": int(len(valid)),
        "tokenizer": "EleutherAI/pythia-160m",
    }
    need = (30_100 + 1) * seq
    assert len(train) > need, f"train too short: {len(train)} < {need}"
    assert len(valid) > (64 + 128) * seq, "valid too short for probe+heldout"
    with open(f"{ROOT}/data_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("manifest written", flush=True)


if __name__ == "__main__":
    main()
