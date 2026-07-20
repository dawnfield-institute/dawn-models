"""
Repair-ladder + CIMM-stabilizer substrate for Ember III — subclass only,
tinycimm_euler.py untouched.

Round 1 (shakedown) repairs:
  R1  input/target normalization (harness-level, run_arms.py --norm)
  R2  R1 + leaky activation (--leaky 0.01) so ReLU death is not absorbing

Round 2 adds four mechanisms resurrected from the original CIMM
(dawn-models/stable/cimm-legacy/), each individually switchable for ablation:

  gravity  prediction pullback toward a rolling entropy baseline
           (CIMM cimm.py:501 — gravity_force = -0.01*(entropy - baseline))
  vlr      learning rate damped exponentially on entropy variance
           (CIMM adaptive_controller.py:168 — lr *= exp(-5*variance), clamped)
  qclip    QFI-adaptive gradient clipping instead of static max_norm=1.0
           (CIMM entropy_monitor.py:105 — clip = 0.1*(1+log(QFI)))
  gate     update-frequency gating (every K tokens) with entropy-validated
           rollback (CIMM cimm.py:650 + entropy_monitor.py:411 — undo the step
           if post-update entropy moved away from baseline)

Plus the round-2 origin-gate hook: the harness may set `update_gate_open`
per token (False = skip this token's weight update entirely). Used by the
gate-origin arms (endogenous / exogenous / random at matched rate); orthogonal
to the CIMM `gate` flag.

forward() is reproduced verbatim from tinycimm_euler.py:993 with torch.relu
replaced by _act; online_adaptation_step() is reproduced from
tinycimm_euler.py:1338 with the flagged mechanisms spliced in at their CIMM
hook points. With all flags off and gate_open always True, behavior is
identical to the round-1 stabilized substrate.
"""

import os
import sys
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tinycimm_euler import (  # noqa: E402
    TinyCIMMEuler, HigherOrderEntropyMonitor, higher_order_transform, safe_item,
)


class StabilizedTinyCIMMEuler(TinyCIMMEuler):
    def __init__(self, *args, leaky_slope=0.01, stab=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.leaky_slope = leaky_slope
        s = stab or {}
        self.f_gravity = bool(s.get("gravity", False))
        self.f_vlr = bool(s.get("vlr", False))
        self.f_qclip = bool(s.get("qclip", False))
        self.f_gate = bool(s.get("gate", False))
        self.gate_every = int(s.get("gate_every", 10))
        self._ent_hist = deque(maxlen=50)
        self._step_i = 0
        self.update_gate_open = True   # origin-gate hook, harness-driven
        self.last_entropy = float("nan")

    def _act(self, z):
        return F.leaky_relu(z, negative_slope=self.leaky_slope)

    @torch.no_grad()
    def _h_entropy(self, h):
        """Shannon entropy of the normalized |activation| distribution — the
        raw-entropy component of the substrate's SEC entropy, kept pure (no
        tracker side effects) for the stabilizer mechanisms."""
        a = h.abs().flatten() + 1e-9
        p = a / a.sum()
        return float(-(p * p.log()).sum())

    def forward(self, x, y_true=None):
        h = self._act(x @ self.W.T + self.b)

        h_mean = torch.mean(h, dim=0, keepdim=True)
        self.micro_memory.append(h_mean.detach().clone())
        if len(self.micro_memory) > self.micro_memory_size:
            self.micro_memory.pop(0)

        higher_order_signal = self.higher_order_processor(x)  # noqa: F841 — parity with substrate (unused there too)

        self.math_memory.append(h.detach().cpu() * self.pattern_decay)
        if len(self.math_memory) > self.math_memory_size:
            self.math_memory.pop(0)

        y = h @ self.V.T + self.c

        if y_true is not None:
            y = higher_order_transform(y, y_true, self.complexity_factor)

        self.last_h = h
        self.last_x = x
        self.last_prediction = y.detach()
        self.complexity_factor = (self.complexity_factor + 0.01) % 1.0
        return y

    def online_adaptation_step(self, x_input, y_target, recent_predictions=None):
        prediction, hidden_state, activations = self.forward_with_qbe(x_input, y_target)

        ent_now = self._h_entropy(hidden_state)
        self.last_entropy = ent_now
        hist = np.asarray(self._ent_hist, dtype=np.float64) if len(self._ent_hist) else None
        warm = hist is not None and len(hist) >= 10
        baseline = float(hist.mean()) if warm else None

        if self.f_gravity and warm:
            # CIMM cimm.py:501 — pull the prediction toward the entropy baseline
            prediction = prediction + (-0.01) * (ent_now - baseline)

        mse_loss = torch.nn.functional.mse_loss(prediction, y_target)
        loss_components = {
            "total_loss": mse_loss, "mse_loss": mse_loss,
            "qbe_loss": torch.tensor(0.0), "entropy_loss": torch.tensor(0.0),
            "coherence_loss": torch.tensor(0.0),
        }
        adaptation_signal = loss_components["total_loss"]

        lr = 0.001 + 0.005 * self.qbe_controller.momentum
        if self.f_vlr and warm:
            # CIMM adaptive_controller.py:168 — Feynman damping on entropy variance
            var = float(hist[-20:].var())
            lr = float(np.clip(lr * np.exp(-5.0 * var), 1e-5, 0.05))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        do_update = bool(self.update_gate_open)
        if self.f_gate:
            # CIMM cimm.py:650 — weight updates every K predictions
            do_update = do_update and (self._step_i % self.gate_every == 0)

        rolled_back = False
        if do_update and adaptation_signal.requires_grad:
            saved = None
            if self.f_gate and warm:
                saved = {k: v.detach().clone() for k, v in self.state_dict().items()}
            self.optimizer.zero_grad()
            adaptation_signal.backward()
            clip = 1.0
            if self.f_qclip and hist is not None and len(hist) >= 11:
                # CIMM entropy_monitor.py:105 — QFI-adaptive clip (log1p-guarded)
                qfi = float(np.var(np.diff(hist)))
                clip = float(np.clip(0.1 * (1.0 + np.log1p(qfi)), 0.01, 1.0))
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            self.optimizer.step()
            if saved is not None:
                # CIMM entropy_monitor.py:411 — validated update: roll back if the
                # step moved activation entropy AWAY from the baseline
                with torch.no_grad():
                    h_after = self._act(x_input @ self.W.T + self.b)
                ent_after = self._h_entropy(h_after)
                if abs(ent_after - baseline) > abs(ent_now - baseline):
                    self.load_state_dict(saved)
                    rolled_back = True

        field_performance = self.calculate_field_performance(prediction, y_target)

        if self.complexity_monitor is None:
            self.complexity_monitor = HigherOrderEntropyMonitor()
        complexity_metric = self.complexity_monitor.update(activations)
        self.mathematical_structure_adaptation(
            complexity_metric, field_performance["quantum_field_performance"],
            safe_item(adaptation_signal), self.structure_controller)

        error = torch.abs(y_target - prediction).mean().item()
        entropy = self.scbf_tracker.compute_symbolic_entropy_collapse(activations)
        self.qbe_controller.update(error, entropy)

        self._ent_hist.append(ent_now)
        self._step_i += 1

        return {
            "prediction": prediction,
            "adaptation_signal": safe_item(adaptation_signal),
            "complexity_metric": complexity_metric,
            "field_performance": field_performance,
            "cimm_components": loss_components,
            "learning_rate": lr,
            "qbe_status": self.qbe_controller.get_status(),
            "updated": bool(do_update and not rolled_back),
            "rolled_back": rolled_back,
        }
