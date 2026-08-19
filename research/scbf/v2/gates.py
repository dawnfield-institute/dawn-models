"""
SCBF v2 update-admission gates (Ember III rounds 5-6).

A gate decides per-step whether the incoming update is applied. All policies here
run at ~50% admission by construction (median thresholds / fair coin), so signal
CONTENT is the only difference between them — the property that made the Ember III
arm comparisons clean.

Measured standings at 160M on WT103 (rounds 5-6), for calibration when choosing:
  rand >= band > excess > err ~ ent   (late stream edge; coverage beats curation)
  excess = lowest drift of all arms at ~97% of rand's edge (stability champion;
  its signal is doubly exogenous: realized loss minus a frozen model's difficulty
  prior — nothing the learner can author).
Caveat (round 4): global rolling thresholds starve low-noise regimes under regime
mixing — thresholds must be regime-local where regimes exist.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class _RollingGate:
    def __init__(self, window: int = 200, warmup: int = 50):
        self.win: list = []
        self.window = window
        self.warmup = warmup

    def _push(self, v: float):
        self.win.append(float(v))
        self.win = self.win[-self.window:]


class ErrGate(_RollingGate):
    """Admit when realized loss >= rolling median ('update on surprise').
    Round-5 lesson: difficulty-thresholding wastes updates on the aleatoric tail."""

    def decide(self, loss: float) -> bool:
        open_ = len(self.win) < self.warmup or loss >= float(np.median(self.win))
        self._push(loss)
        return bool(open_)


class ExcessGate(_RollingGate):
    """Admit when (realized loss − frozen-reference loss) >= rolling median of the
    excess — learnability, not difficulty. Doubly exogenous."""

    def __init__(self, ref_losses: Sequence[float], window: int = 200, warmup: int = 50):
        super().__init__(window, warmup)
        self.ref = np.asarray(ref_losses, dtype=np.float64)
        self.t = 0

    def decide(self, loss: float) -> bool:
        ex = loss - float(self.ref[self.t])
        self.t += 1
        open_ = len(self.win) < self.warmup or ex >= float(np.median(self.win))
        self._push(ex)
        return bool(open_)


class BandGate(_RollingGate):
    """Admit when loss in [p25, p75) of the rolling window — skips the trivial AND
    the irreducible tail. ~50% rate by construction."""

    def decide(self, loss: float) -> bool:
        if len(self.win) < self.warmup:
            open_ = True
        else:
            p25, p75 = np.percentile(self.win, [25, 75])
            open_ = bool(p25 <= loss < p75)
        self._push(loss)
        return open_


class RandGate:
    """Fair-coin control at matched rate. Round 5-6: unbeaten at 160M — the
    representative-coverage baseline every informed gate must clear."""

    def __init__(self, seed: int = 0, p: float = 0.5):
        self.rng = np.random.default_rng(seed)
        self.p = p

    def decide(self, loss: Optional[float] = None) -> bool:
        return bool(self.rng.random() < self.p)
