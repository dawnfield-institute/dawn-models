"""
Ember III structure controllers — Arms B and C.

Both expose the same interface as MathematicalStructureController.decide()
(tinycimm_euler.py:843) so they drop into online_adaptation_step unchanged via
`model.structure_controller = <controller>`. Arm A uses the substrate's own
controller, untouched.

Design constraint (idea doc §2.1): Arm B's controller may consume ONLY the
realized prequential error stream — no activations, no weight statistics, no
complexity metrics. The endogenous arguments decide() receives are ignored.

Effective capacity range is matched to Arm A's operative range [8, 128]
(A gates on max_neurons=128; the model clamps the floor to 8) so capacity
bounds cannot confound the arm comparison.
"""

from collections import deque

import numpy as np


class ResidualSignalController:
    """Arm B: growth/prune driven by statistics of the realized error stream only.

    Policy (locked pre-run):
      stalled  := relative improvement of recent-half vs old-half error < stall_thresh
      grow     if stalled AND recent CV > long CV * grow_cv_ratio   (error volatile
               relative to its own baseline -> unmodeled structure remains)
      prune    if stalled AND recent CV < long CV * prune_cv_ratio  (stable at floor
               -> excess capacity unneeded)
    All statistics are computed from the observed error stream; the model cannot
    author the stream's distribution or sampling (fixed prime-gap sequence).
    """

    def __init__(self, window=200, recent=50, stall_thresh=0.02,
                 grow_cv_ratio=1.1, prune_cv_ratio=0.6,
                 min_neurons=8, max_neurons=128,
                 grow_quantum=0.10, prune_quantum=0.08, cooldown_period=10):
        self.errors = deque(maxlen=window)  # bounded by design (port-audit lesson)
        self.window = window
        self.recent = recent
        self.stall_thresh = stall_thresh
        self.grow_cv_ratio = grow_cv_ratio
        self.prune_cv_ratio = prune_cv_ratio
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons
        self.grow_quantum = grow_quantum
        self.prune_quantum = prune_quantum
        self.cooldown_period = cooldown_period
        self.cooldown_counter = 0
        self.decisions = 0

    def observe(self, prequential_error):
        """Harness calls this once per token, BEFORE the update step, with the
        leak-free |prediction - realized| error."""
        self.errors.append(float(prequential_error))

    def decide(self, complexity_metric, performance, adaptation_signal, num_neurons):
        # Endogenous arguments intentionally ignored.
        self.decisions += 1
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return "none", 0
        if len(self.errors) < self.window:
            return "none", 0

        e = np.asarray(self.errors, dtype=np.float64)
        old_half = e[: self.window - self.recent]
        recent = e[-self.recent:]
        eps = 1e-9

        improvement = (old_half.mean() - recent.mean()) / (old_half.mean() + eps)
        stalled = improvement < self.stall_thresh
        cv_recent = recent.std() / (recent.mean() + eps)
        cv_long = e.std() / (e.mean() + eps)

        if stalled and cv_recent > cv_long * self.grow_cv_ratio and num_neurons < self.max_neurons:
            self.cooldown_counter = self.cooldown_period
            return "grow", max(2, int(num_neurons * self.grow_quantum))
        if stalled and cv_recent < cv_long * self.prune_cv_ratio and num_neurons > self.min_neurons:
            self.cooldown_counter = self.cooldown_period
            return "prune", max(1, int(num_neurons * self.prune_quantum))
        return "none", 0


class DecoupledSurrogateController(ResidualSignalController):
    """Arm C: identical decision logic to Arm B, but fed a pre-recorded,
    block-shuffled surrogate error trace instead of the live error stream.

    Same marginal statistics as B's signal, decoupled from this model's state —
    separates signal ORIGIN (outside the control loop) from signal INFORMATION
    (about this model's actual performance)."""

    def __init__(self, surrogate_trace, **kwargs):
        super().__init__(**kwargs)
        self.surrogate = np.asarray(surrogate_trace, dtype=np.float64)
        self._idx = 0

    def observe(self, prequential_error):
        # Live error deliberately discarded; consume the surrogate instead.
        self.errors.append(float(self.surrogate[self._idx % len(self.surrogate)]))
        self._idx += 1


def block_shuffle(trace, block=100, seed=0):
    """Shuffle a trace in contiguous blocks: preserves marginals and short-range
    texture, destroys alignment with the run that produced it."""
    trace = np.asarray(trace, dtype=np.float64)
    n_blocks = len(trace) // block
    blocks = [trace[i * block:(i + 1) * block] for i in range(n_blocks)]
    tail = trace[n_blocks * block:]
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)
    out = np.concatenate(blocks + ([tail] if len(tail) else []))
    return out
