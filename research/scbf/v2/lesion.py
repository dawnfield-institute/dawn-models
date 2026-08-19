"""
Gradient-pressure lesioning — causal capability localization (SCBF v2's new
instrument, discovered Ember III round 9).

Protocol: constrain WHERE update pressure may land (HookSpine placement masks),
run identical continuous-learning streams through each arm, and measure what dies
with capability probes/batteries. Differential capability loss between arms
causally localizes where a capability is stored — a lesion study with gradient
flow instead of a scalpel, on a live learner.

First result (round 9): the experts-only mask on OLMoE destroyed 10.4pt of
TriviaQA (invisible to CE, −10.4 EM at +0.043 nats), while the unmasked arm
retained 9/9 tasks — causally locating MoE factual knowledge in expert FFNs.
Refined mechanism hypothesis under test (round 10 exp B): frozen routing x usage
skew aims pressure at the highest-usage (knowledge-densest) experts.

This module holds the protocol scaffolding; the heavy lifting is HookSpine masks +
the probe/battery modules. A lesioning experiment is:

    arms = {"control": mask_all, "experts_only": olmoe_experts_only(), ...}
    for name, mask in arms.items():
        model = load_fresh()
        spine = HookSpine(model, trainable=mask, lr=..., fused=True, ...)
        run_stream(model, stream)            # identical stream per arm
        save_checkpoint(model, name)
    # then: battery/probes on each checkpoint vs the frozen reference;
    # differential deltas between arms = the localization signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class LesionArm:
    name: str
    trainable: Callable[[str], bool]
    notes: str = ""


@dataclass
class LesionMap:
    """Accumulates per-arm probe/battery results; renders differential views."""
    reference_name: str = "frozen"
    results: Dict[str, dict] = field(default_factory=dict)

    def record(self, arm: str, probe_results: dict):
        self.results.setdefault(arm, {}).update(probe_results)

    def differentials(self, metric_keys: Optional[List[str]] = None) -> List[dict]:
        ref = self.results.get(self.reference_name, {})
        rows = []
        for arm, res in self.results.items():
            if arm == self.reference_name:
                continue
            for k, v in res.items():
                if metric_keys and k not in metric_keys:
                    continue
                if isinstance(v, (int, float)) and isinstance(ref.get(k), (int, float)):
                    rows.append({"arm": arm, "metric": k,
                                 "reference": ref[k], "value": v,
                                 "delta": v - ref[k]})
        return rows
