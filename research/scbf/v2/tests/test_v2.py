"""SCBF v2 unit tests — CPU-only, synthetic models, plain pytest functions
(v1 test convention). Run: python -m pytest research/scbf/v2/tests/ -q"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # research/ on path
from scbf.v2 import (  # noqa: E402
    HookSpine, mask_pattern, linear_cka, ErrGate, ExcessGate, BandGate, RandGate,
    verdict, LesionMap, RunLog, ClozeKnowledgeProbe,
)


class ToyMoE(nn.Module):
    """Tiny model with a stacked 'experts' tensor (8 experts, axis 0) + a trunk."""

    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(4, 4)
        self.experts_weight = nn.Parameter(torch.randn(8, 4, 4) * 0.1)

    def forward(self, x):
        h = self.trunk(x)
        # every expert contributes so all rows get gradient
        return torch.einsum("bd,edk->bek", h, self.experts_weight).sum(dim=1)


def _run_step(model, spine=None):
    x = torch.randn(3, 4)
    loss = (model(x) ** 2).mean()
    loss.backward()
    return loss


def test_placement_mask_freezes_params():
    m = ToyMoE()
    HookSpine(m, trainable=mask_pattern("experts"), lr=1e-2)
    assert m.experts_weight.requires_grad
    assert not m.trunk.weight.requires_grad


def test_fused_update_moves_params_and_frees_grads():
    m = ToyMoE()
    spine = HookSpine(m, trainable=mask_pattern("experts"), lr=1e-2, fused=True)
    before = m.experts_weight.detach().clone()
    _run_step(m)
    assert not torch.equal(before, m.experts_weight.detach())
    assert m.experts_weight.grad is None  # freed inside the hook
    assert spine.updates >= 1


def test_observe_mode_leaves_grads():
    m = ToyMoE()
    HookSpine(m, trainable=mask_pattern("experts"), fused=False,
              group_fn=lambda n: "experts" if "experts" in n else None)
    before = m.experts_weight.detach().clone()
    _run_step(m)
    assert torch.equal(before, m.experts_weight.detach())  # no update
    assert m.experts_weight.grad is not None               # grads left


def test_stacked_mass_per_slice_matches_manual():
    """The exact instrument the round-8 bug nulled: per-slice mass on a stacked
    tensor must be nonzero and equal the manual per-row grad norms * lr."""
    m = ToyMoE()
    lr = 1e-2
    spine = HookSpine(
        m, trainable=mask_pattern("experts"), lr=lr, clip=1e9, fused=False,
        stacked_fn=lambda n: (("experts", 0), 0) if "experts" in n else None)
    _run_step(m)
    g = m.experts_weight.grad
    manual = torch.linalg.vector_norm(g.reshape(8, -1), dim=1).numpy() * lr
    got = spine.stacked_mass[("experts", 0)]
    assert got.shape == (8,)
    assert (got > 0).all()
    np.testing.assert_allclose(got, manual, rtol=1e-5)


def test_nonfinite_grad_raises():
    m = ToyMoE()
    HookSpine(m, trainable=mask_pattern("experts"), fused=True)
    m.experts_weight.grad = None
    x = torch.randn(3, 4)
    loss = (m(x) ** 2).mean() * float("nan")
    try:
        loss.backward()
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_cka_properties():
    H = np.random.randn(64, 16)
    assert abs(linear_cka(H, H) - 1.0) < 1e-9
    Q, _ = np.linalg.qr(np.random.randn(16, 16))
    assert abs(linear_cka(H, H @ Q) - 1.0) < 1e-6      # rotation-invariant
    H2 = np.random.randn(64, 33)                        # dimension-agnostic
    v = linear_cka(H, H2)
    assert 0.0 <= v <= 1.0


def test_gate_rates_near_half():
    rng = np.random.default_rng(0)
    stream = rng.exponential(1.0, 5000)
    for gate in (ErrGate(), BandGate(), RandGate(seed=1),
                 ExcessGate(ref_losses=np.zeros(5000))):
        opens = sum(gate.decide(float(v)) for v in stream)
        assert 0.40 < opens / len(stream) < 0.62, type(gate).__name__


def test_battery_verdict_rule():
    ref = {"taskA": (0.50, 0.01), "taskB": (0.50, 0.01)}
    cand = {"taskA": (0.505, 0.01), "taskB": (0.40, 0.01)}
    v = verdict(ref, cand)
    calls = {r["task"]: r["verdict"] for r in v["rows"]}
    assert calls["taskA"] == "=" and calls["taskB"] == "DOWN"


def test_lesion_map_differentials():
    lm = LesionMap(reference_name="frozen")
    lm.record("frozen", {"triviaqa": 0.48})
    lm.record("experts", {"triviaqa": 0.376})
    rows = lm.differentials()
    assert len(rows) == 1 and abs(rows[0]["delta"] + 0.104) < 1e-9


def test_runlog_files(tmp_path):
    log = RunLog(str(tmp_path / "run"), meta={"arm": "test"})
    log.step(t=1, ce=1.0)
    log.snapshot(t=1, drift=0.0)
    out = log.finalize(final=1)
    assert (Path(out) / "timeseries.csv").exists()
    assert (Path(out) / "snapshots.csv").exists()
    assert (Path(out) / "meta.json").exists()


class _ToyTok:
    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [ord(c) % 97 + 1 for c in s]}


def test_cloze_probe_shapes():
    torch.manual_seed(0)

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(128, 16)
            self.head = nn.Linear(16, 128)

        def forward(self, x, labels=None):
            logits = self.head(self.emb(x))
            return type("O", (), {"logits": logits})()

    probe = ClozeKnowledgeProbe([("abc def", "ghi"), ("xy z", "qq")], _ToyTok())
    res = probe(LM(), batch_size=2)
    assert res["cloze_n"] == 2 and len(res["per_fact_ll"]) == 2
    assert 0.0 <= res["cloze_top1"] <= 1.0
