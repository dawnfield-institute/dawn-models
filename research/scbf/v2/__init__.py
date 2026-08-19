"""
SCBF v2 — Continuous Interpretability
=====================================

Interpretability for models that never stop changing. Built from the validated
instrument set of Ember III rounds 1-9; v1 untouched. Spec:
`scbf/.spec/scbf-v2.spec.md`.

Main components:
- hookspine:  gradient-hook instrumentation core (fused updates, placement masks,
              per-group/per-slice update-mass telemetry) — 7B-proven at 15.3GB VRAM
- probes:     CKA drift, held-out CE, cloze knowledge (generation-free), plasticity
- battery:    lm-eval parsing + the locked 2xSE reading rule
- lesion:     gradient-pressure lesioning (causal capability localization)
- gates:      update-admission policies (err / excess / band / rand)
- telemetry:  RunLog CSV/JSON schemas
"""

from .hookspine import (
    HookSpine,
    mask_all,
    mask_pattern,
    olmoe_experts_only,
    olmoe_expert_stacked_fn,
)
from .probes import (
    linear_cka,
    hf_hidden_fn,
    ce_from_logits,
    CKADriftProbe,
    HeldoutCEProbe,
    ClozeKnowledgeProbe,
    PlasticityProbe,
)
from .battery import load_lmeval_results, verdict, format_verdict
from .lesion import LesionArm, LesionMap
from .gates import ErrGate, ExcessGate, BandGate, RandGate
from .telemetry import RunLog

__version__ = "2.0.0-dev"

__all__ = [
    "HookSpine", "mask_all", "mask_pattern", "olmoe_experts_only",
    "olmoe_expert_stacked_fn",
    "linear_cka", "hf_hidden_fn", "ce_from_logits",
    "CKADriftProbe", "HeldoutCEProbe", "ClozeKnowledgeProbe", "PlasticityProbe",
    "load_lmeval_results", "verdict", "format_verdict",
    "LesionArm", "LesionMap",
    "ErrGate", "ExcessGate", "BandGate", "RandGate",
    "RunLog",
]
