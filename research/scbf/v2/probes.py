"""
SCBF v2 probes — side-effect-free measurement on live models.

Conventions follow v1 metrics: callables return flat dicts. All probes measure in
eval/no-grad and restore train mode; none mutate model state.

Instrument caveats carried from Ember III (cite when reporting):
- Single-run CKA drift is jitter-sensitive at LM scale (round 6: 0.023 vs 0.355
  across CUDA-jitter repeats at identical competence). Pair with competence probes.
- CE-family metrics are blind to knowledge erosion (round 9: −10.4pt TriviaQA at
  +0.043 nats off-domain CE). Pair with ClozeKnowledgeProbe / task batteries.
"""

from __future__ import annotations

import copy
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def linear_cka(H1: np.ndarray, H2: np.ndarray) -> float:
    """Linear CKA between (n_probes, d1) and (n_probes, d2). Dimension-agnostic —
    survives grow/prune and architecture changes. 1 = identical geometry."""
    H1 = H1 - H1.mean(axis=0, keepdims=True)
    H2 = H2 - H2.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(H1.T @ H2, ord="fro") ** 2
    denom = np.linalg.norm(H1.T @ H1, ord="fro") * np.linalg.norm(H2.T @ H2, ord="fro")
    return float(cross / denom) if denom > 1e-12 else float("nan")


def hf_hidden_fn(layer: int, batch_size: int = 8) -> Callable:
    """Factory: mean-pooled hidden states at `layer` for HF causal LMs."""
    @torch.no_grad()
    def fn(model, batch: torch.Tensor) -> np.ndarray:
        device = next(model.parameters()).device
        outs = []
        for i in range(0, batch.shape[0], batch_size):
            b = batch[i: i + batch_size].to(device)
            hs = model(b, output_hidden_states=True).hidden_states
            h = hs[min(layer, len(hs) - 1)]
            outs.append(h.mean(dim=1).float().cpu().double().numpy())
        return np.concatenate(outs, axis=0)
    return fn


class CKADriftProbe:
    """Fixed probe batch + frozen reference; drift(t) = 1 − CKA(H_ref, H_t)."""

    def __init__(self, probe_batch: torch.Tensor, hidden_fn: Callable):
        self.probe = probe_batch
        self.hidden_fn = hidden_fn
        self.H_ref: Optional[np.ndarray] = None

    def __call__(self, model) -> dict:
        was_training = model.training
        model.eval()
        H = self.hidden_fn(model, self.probe)
        if was_training:
            model.train()
        if self.H_ref is None:
            self.H_ref = H
            return {"cka": 1.0, "drift": 0.0, "is_reference": 1.0}
        cka = linear_cka(self.H_ref, H)
        return {"cka": cka, "drift": 1.0 - cka if np.isfinite(cka) else float("nan"),
                "is_reference": 0.0}


def ce_from_logits(logits: torch.Tensor, x: torch.Tensor) -> float:
    """Pure shifted CE — the measurement channel, identical across architectures."""
    sl = logits[..., :-1, :].contiguous().float()
    st = x[..., 1:].contiguous()
    return float(F.cross_entropy(sl.view(-1, sl.shape[-1]), st.view(-1)))


class HeldoutCEProbe:
    """Pure CE on a fixed batch (forgetting split = one probe per domain)."""

    def __init__(self, batch: torch.Tensor, batch_size: int = 8, name: str = "heldout"):
        self.batch = batch
        self.bs = batch_size
        self.name = name

    @torch.no_grad()
    def __call__(self, model) -> dict:
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        tot, n = 0.0, 0
        for i in range(0, self.batch.shape[0], self.bs):
            b = self.batch[i: i + self.bs].to(device)
            tot += ce_from_logits(model(b).logits, b) * b.shape[0]
            n += b.shape[0]
        if was_training:
            model.train()
        return {f"{self.name}_ce": tot / n}


class ClozeKnowledgeProbe:
    """Generation-free knowledge scoring (round-9 deflation instrument).

    Facts are (prompt, answer) strings. Scores, per fact, under teacher forcing:
      - mean log-likelihood of the answer tokens given the prompt
      - top-1: does argmax at the first answer position equal the first answer token
    Immune to output-format shift by construction (no sampling, no exact-match on
    generated text) — separates knowledge-gone from style-changed.
    """

    def __init__(self, facts: Sequence[Tuple[str, str]], tokenizer, max_len: int = 256):
        self.items = []
        for prompt, answer in facts:
            p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            a = tokenizer(" " + answer.strip(), add_special_tokens=False)["input_ids"]
            if not a or len(p) + len(a) > max_len or len(p) == 0:
                continue
            self.items.append((p, a))

    @torch.no_grad()
    def __call__(self, model, batch_size: int = 8) -> dict:
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        lls: List[float] = []
        top1: List[float] = []
        for i in range(0, len(self.items), batch_size):
            chunk = self.items[i: i + batch_size]
            maxlen = max(len(p) + len(a) for p, a in chunk)
            ids = torch.zeros(len(chunk), maxlen, dtype=torch.long)
            for j, (p, a) in enumerate(chunk):
                ids[j, : len(p) + len(a)] = torch.tensor(p + a)
            logits = model(ids.to(device)).logits.float().cpu()
            logp = F.log_softmax(logits, dim=-1)
            for j, (p, a) in enumerate(chunk):
                pos = range(len(p) - 1, len(p) - 1 + len(a))  # predicts a[0..]
                tok_lls = [float(logp[j, t, a[k]]) for k, t in enumerate(pos)]
                lls.append(float(np.mean(tok_lls)))
                top1.append(float(int(torch.argmax(logits[j, len(p) - 1]) == a[0])))
        if was_training:
            model.train()
        return {"cloze_mean_ll": float(np.mean(lls)), "cloze_median_ll": float(np.median(lls)),
                "cloze_top1": float(np.mean(top1)), "cloze_n": float(len(lls)),
                "per_fact_ll": lls, "per_fact_top1": top1}


class PlasticityProbe:
    """Fixed-segment clone adaptation (small models only — deepcopy). Measures
    'can it still learn fresh structure' — the anti-ossification instrument."""

    def __init__(self, seg_x: torch.Tensor, seg_y: Optional[torch.Tensor] = None,
                 steps: int = 100, lr: float = 1e-5, edge: int = 20):
        self.seg_x, self.seg_y, self.steps, self.lr, self.edge = seg_x, seg_y, steps, lr, edge

    def __call__(self, model) -> dict:
        clone = copy.deepcopy(model)
        opt = torch.optim.Adam(clone.parameters(), lr=self.lr)
        device = next(clone.parameters()).device
        ces = []
        for k in range(min(self.steps, self.seg_x.shape[0])):
            x = self.seg_x[k: k + 1].to(device)
            out = clone(x, labels=x)
            ces.append(ce_from_logits(out.logits.detach(), x))
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(clone.parameters(), 1.0)
            opt.step()
        first, last = float(np.mean(ces[: self.edge])), float(np.mean(ces[-self.edge:]))
        del clone, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"plast_first": first, "plast_last": last,
                "plasticity": (first - last) / (first + 1e-9)}
