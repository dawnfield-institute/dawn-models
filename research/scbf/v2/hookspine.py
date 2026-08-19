"""
HookSpine — the gradient-hook instrumentation core of SCBF v2.

Proven in Ember III rounds 8-9 (OLMoE-1B-7B continuous learning in 15.3GB VRAM):
per-parameter post-accumulate-grad hooks that clip, optionally apply a fused update,
accumulate group telemetry, and free the gradient the moment autograd produces it.
Peak gradient memory ~ one layer; the model updates WHILE back-propagating.

Placement masks double as lesion configurations (see scbf.v2.lesion): a mask is a
causal intervention on WHERE update pressure may land.

Bounded-memory guarantee: accumulators are scalars or fixed-size arrays keyed by
group — never per-step lists (SCBF v1 port-audit lesson, Ember III round 1).
Loud-failure guarantee: non-finite gradient norms raise immediately (silent NaN
corrupted a gate for 9,950 chunks in round 5's pre-flight — never again).
"""

from __future__ import annotations

from typing import Callable, Dict, Hashable, Optional, Tuple

import numpy as np
import torch


class HookSpine:
    """Attach gradient instrumentation (and optionally fused SGD updates) to a model.

    Args:
        model: torch.nn.Module whose parameters are instrumented.
        trainable: predicate name -> bool; parameters failing it get
            requires_grad=False (the placement mask / lesion config).
            None = all parameters trainable.
        lr: fused-SGD learning rate (used when fused=True).
        clip: per-tensor gradient max-norm (scaled down, never up).
        fused: True  -> clip + SGD update + grad=None inside the hook (training).
               False -> accumulate telemetry only; grads left for an external
                        optimizer (observation).
        group_fn: name -> hashable key | None. Scalar update-mass accumulation:
            mass[key] += ||g_clipped|| * lr.
        stacked_fn: name -> (key, axis) | None. Per-slice accumulation along
            `axis` of the (stacked) tensor: mass[key][i] += ||g_i|| * lr.
            This is how per-expert telemetry works on transformers' stacked
            MoE expert tensors (e.g. experts.gate_up_proj: (n_experts, d, d)).
    """

    def __init__(self, model: torch.nn.Module,
                 trainable: Optional[Callable[[str], bool]] = None,
                 lr: float = 5e-4, clip: float = 1.0, fused: bool = True,
                 group_fn: Optional[Callable[[str], Optional[Hashable]]] = None,
                 stacked_fn: Optional[Callable[[str], Optional[Tuple[Hashable, int]]]] = None):
        self.model = model
        self.lr = float(lr)
        self.clip = float(clip)
        self.fused = bool(fused)
        self.mass: Dict[Hashable, float] = {}
        self.stacked_mass: Dict[Hashable, np.ndarray] = {}
        self.updates = 0
        self.clip_events = 0
        self._handles = []

        for name, p in model.named_parameters():
            is_trainable = True if trainable is None else bool(trainable(name))
            p.requires_grad = is_trainable
            if not is_trainable:
                continue
            gkey = group_fn(name) if group_fn else None
            skey = stacked_fn(name) if stacked_fn else None
            if skey is not None:
                key, axis = skey
                n = p.shape[axis]
                if key not in self.stacked_mass:
                    self.stacked_mass[key] = np.zeros(n, dtype=np.float64)
            self._handles.append(
                p.register_post_accumulate_grad_hook(self._make_hook(name, gkey, skey)))

    def _make_hook(self, name, gkey, skey):
        def hook(param: torch.nn.Parameter):
            g = param.grad
            if g is None:
                return
            gn = torch.linalg.vector_norm(g)
            if not torch.isfinite(gn):
                raise RuntimeError(f"non-finite gradient norm on {name} — refusing "
                                   f"to continue (silent NaN is a known corruptor)")
            if gn > self.clip:
                g = g * (self.clip / gn)
                self.clip_events += 1
                gn = torch.tensor(self.clip)
            if skey is not None:
                key, axis = skey
                gm = g.movedim(axis, 0)
                slice_norms = torch.linalg.vector_norm(
                    gm.reshape(gm.shape[0], -1).float(), dim=1)
                self.stacked_mass[key] += slice_norms.cpu().numpy() * self.lr
            if gkey is not None:
                self.mass[gkey] = self.mass.get(gkey, 0.0) + float(gn) * self.lr
            if self.fused:
                param.data.add_(g, alpha=-self.lr)
                param.grad = None
            self.updates += 1
        return hook

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def telemetry(self) -> dict:
        """Flat summary (JSON-safe); stacked arrays reported as lists."""
        return {
            "updates": self.updates,
            "clip_events": self.clip_events,
            "mass": {str(k): v for k, v in self.mass.items()},
            "stacked_mass": {str(k): v.tolist() for k, v in self.stacked_mass.items()},
        }


# ---------------------------------------------------------------- mask builders

def mask_all(name: str) -> bool:
    return True


def mask_pattern(*substrings: str) -> Callable[[str], bool]:
    """Trainable iff the parameter name contains ANY of the substrings."""
    def pred(name: str) -> bool:
        return any(s in name for s in substrings)
    return pred


def olmoe_experts_only() -> Callable[[str], bool]:
    """The round-8/9 'experts' lesion arm: expert FFNs train, trunk+router frozen.
    WARNING (round 9): at 5k-update dose this mask destroyed 10.4pt of TriviaQA —
    expert FFNs hold the MoE's factual knowledge. Use as a lesion, not a default."""
    return mask_pattern("mlp.experts.")


def olmoe_expert_stacked_fn(name: str):
    """stacked_fn for transformers' OLMoE: per-expert (axis 0) telemetry on the
    stacked expert tensors. Key = (layer_idx, proj_name)."""
    if "mlp.experts." not in name:
        return None
    parts = name.split(".")
    try:
        layer = int(parts[parts.index("layers") + 1])
    except (ValueError, IndexError):
        return None
    return ((layer, parts[-1]), 0)
