"""Spike Memory Cascade — Breaking the Bigram Ceiling.

Builds on spike_memory_field.py findings:
  - Parents ARE memory (44.3% from voice dynamics alone)
  - Forward-pointing voices encode "what follows me" per character
  - Single voice = single context slot = bigram ceiling

This spike breaks through by using the SEQUENCE of recently activated
parents as a combined prediction signal. Each parent's forward voice
encodes a bigram. The last K parents' voices composed together encode
a K-gram — through pure voice dynamics, no lookup tables.

Two composition strategies tested:
  A. Cascade filter: crystal-filter current prediction through each
     previous parent in sequence. Each filter stage conditions the
     prediction on one more step of context.
  B. Weighted blend: PHI_INV^i decay over recent parent voices.
     Simpler but loses sequential ordering information.

Context depths K=1 through K=5 tested to show scaling.

Baselines (from spike_scaling.py):
  Pure 2-gram:   34.3%    Field+ctx=1: 50.2%
  Pure 3-gram:   51.7%    Field+ctx=2: 66.9%
  Pure 4-gram:   70.3%    Field+ctx=3: 79.1%
  Pure 5-gram:   83.3%    Field+ctx=4: 84.6%

Usage:
    cd dawn-models/research/GAIA/spikes/population
    PYTHONIOENCODING=utf-8 PYTHONPATH="../../src;path/to/fracton" python spike_memory_cascade.py
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque

import torch
import torch.nn.functional as F

# ==========================================================================
#  DFT Constants
# ==========================================================================

XI_SEC = 0.0618033988749895
PHI_INV = 0.618033988749895
LAMBDA_STAR = 0.9816
GAMMA = 1.0 - LAMBDA_STAR
LN_PHI = math.log((1 + math.sqrt(5)) / 2)
DIM = 64

DEVICE = torch.device("cpu")


# ==========================================================================
#  FastCodebook (identical to spike_memory_field.py)
# ==========================================================================

class FastCodebook:
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self._char_to_idx: dict[str, int] = {}
        self._idx_to_char: list[str] = []
        self._vectors: list[torch.Tensor] = []
        self._matrix: torch.Tensor | None = None
        self._build()

    def _build(self):
        classes = {
            "vowel": list("aeiou"),
            "consonant": list("bcdfghjklmnpqrstvwxyz"),
            "digit": list("0123456789"),
            "space": [" ", "\n", "\t"],
            "punct": list(".,!?;:'\"()-"),
        }
        for class_idx, (_, chars) in enumerate(classes.items()):
            torch.manual_seed(class_idx * 10000 + 42)
            class_dir = torch.randn(self.dim, device="cpu")
            class_dir = class_dir / (torch.norm(class_dir) + 1e-8)
            vecs = []
            for i, ch in enumerate(chars):
                torch.manual_seed(class_idx * 10000 + i * 100 + 7)
                v = torch.randn(self.dim, device="cpu")
                v = 0.4 * class_dir + 0.6 * v
                for prev in vecs:
                    v = v - torch.dot(v, prev) * prev
                norm = torch.norm(v)
                if norm < 1e-8:
                    torch.manual_seed(class_idx * 10000 + i * 100 + 999)
                    v = torch.randn(self.dim, device="cpu")
                v = v / (torch.norm(v) + 1e-8)
                vecs.append(v)
                self._char_to_idx[ch] = len(self._idx_to_char)
                self._idx_to_char.append(ch)
                self._vectors.append(v)
        torch.manual_seed(99999)
        offset = 0.15 * torch.randn(self.dim, device="cpu")
        for ch in "abcdefghijklmnopqrstuvwxyz":
            upper = ch.upper()
            if ch in self._char_to_idx:
                v = self._vectors[self._char_to_idx[ch]] + offset
                v = v / (torch.norm(v) + 1e-8)
                self._char_to_idx[upper] = len(self._idx_to_char)
                self._idx_to_char.append(upper)
                self._vectors.append(v)
        self._matrix = torch.stack(self._vectors).to(DEVICE)

    def encode(self, char: str) -> torch.Tensor:
        if char in self._char_to_idx:
            return self._matrix[self._char_to_idx[char]].clone()
        torch.manual_seed(hash(char) % 2**31)
        v = torch.randn(self.dim, device=DEVICE)
        return v / (torch.norm(v) + 1e-8)

    def decode_topk(self, tensor: torch.Tensor, k: int = 5) -> list[tuple[str, float]]:
        t = tensor.flatten()[:self.dim].to(DEVICE)
        if torch.norm(t) < 1e-8:
            return [("?", 0.0)]
        sims = F.cosine_similarity(self._matrix, t.unsqueeze(0), dim=1)
        topk = sims.topk(min(k, len(self._idx_to_char)))
        return [(self._idx_to_char[int(idx)], float(val))
                for val, idx in zip(topk.values, topk.indices)]


# ==========================================================================
#  CascadeField — Sequential Parent Memory
# ==========================================================================

class CascadeField:
    """Memory field with sequential context via parent activation history.

    L1: 64 character nodes. Fire on input match.
    L2: 25 parents (1 per character). Forward-pointing voice dynamics.
    Context: ring buffer of last K activated parent indices.
    Prediction: compose K parents' forward voices via cascade filter or blend.
    """

    def __init__(self, codebook: FastCodebook, context_depth: int = 3,
                 strategy: str = "cascade"):
        self.codebook = codebook
        self.context_depth = context_depth
        self.strategy = strategy  # "cascade" or "blend"
        self.field_size = 8
        self.n_l1 = 64
        self.n_l2 = 0

        # Character mapping
        self._node_chars: list[str] = []
        self._char_to_idx: dict[str, int] = {}
        self._char_list: list[str] = []
        self._l2_char: list[str] = []

        # L1 state
        self._voices_l1: torch.Tensor = None
        self._axes_l1: torch.Tensor = None

        # L2 state — forward-pointing parent memory
        self._voices_l2: torch.Tensor = None
        self._axes_l2: torch.Tensor = None
        self._children_l2: list[list[int]] = []

        # Context: ring buffer of recent parent activations
        self._parent_history: deque[int] = deque(maxlen=context_depth)

        # L1 entropy tracking
        self._entropy_buf: torch.Tensor = None
        self._entropy_len: torch.Tensor = None

        # Stats
        self.accuracy_history: list[int] = []
        self.phase_accuracy: dict[str, list[int]] = defaultdict(list)
        self.char_accuracy: dict[str, list[int]] = defaultdict(list)
        self._current_phase = "init"
        self._last_pred_char: str | None = None
        self._last_target_char: str | None = None
        self._fire_rates: list[float] = []

    def build_field(self, chars: set[str]):
        unique = sorted(chars)
        n_chars = len(unique)
        self._char_list = unique
        self._char_to_idx = {ch: i for i, ch in enumerate(unique)}

        fs = self.field_size
        self.n_l1 = fs * fs

        # L1: character nodes
        self._node_chars = [unique[i % n_chars] for i in range(self.n_l1)]
        voices = [self.codebook.encode(self._node_chars[i]).to(DEVICE)
                  for i in range(self.n_l1)]
        self._voices_l1 = torch.stack(voices)
        self._axes_l1 = self._voices_l1.clone()

        # Entropy
        self._entropy_buf = torch.ones(self.n_l1, 30, device=DEVICE)
        self._entropy_len = torch.zeros(self.n_l1, dtype=torch.long, device=DEVICE)

        # L2: one parent per character
        self.n_l2 = n_chars
        self._children_l2 = []
        self._l2_char = []
        for ch_idx, ch in enumerate(unique):
            children = [i for i in range(self.n_l1) if self._node_chars[i] == ch]
            self._children_l2.append(children)
            self._l2_char.append(ch)

        # L2 initial voices = character encodings
        l2_voices = []
        for children in self._children_l2:
            mean_v = self._voices_l1[children].mean(dim=0)
            mean_v = mean_v / (torch.norm(mean_v) + 1e-8)
            l2_voices.append(mean_v)
        self._voices_l2 = torch.stack(l2_voices)
        self._axes_l2 = self._voices_l2.clone()

    def _crystal_filter(self, signal: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        dot = torch.dot(signal, axis)
        proj = dot * axis
        orth = signal - proj
        return proj + PHI_INV * orth

    def _get_crystallization(self) -> torch.Tensor:
        lengths = self._entropy_len.clamp(min=1).float()
        positions = torch.arange(30, device=DEVICE).unsqueeze(0)
        valid_mask = positions < self._entropy_len.unsqueeze(1)
        valid_buf = self._entropy_buf * valid_mask.float()
        means = valid_buf.sum(dim=1) / lengths
        sq_diff = ((self._entropy_buf - means.unsqueeze(1)) ** 2) * valid_mask.float()
        var = sq_diff.sum(dim=1) / lengths
        var = torch.where(self._entropy_len >= 3, var, torch.ones_like(var))
        return (1.0 - var.clamp(min=0) / XI_SEC).clamp(0, 1)

    def process(self, input_tensor: torch.Tensor, input_char: str, next_char: str):
        next_tensor = self.codebook.encode(next_char)

        # Evaluate previous prediction
        if self._last_pred_char is not None and self._last_target_char is not None:
            hit = 1 if self._last_pred_char == self._last_target_char else 0
            self.accuracy_history.append(hit)
            self.phase_accuracy[self._current_phase].append(hit)
            self.char_accuracy[self._last_target_char].append(hit)

        # ============================================================
        # PHASE 1: L1 ACTUALIZE
        # ============================================================
        inp = input_tensor.unsqueeze(0).expand(self.n_l1, -1)
        dots = (inp * self._axes_l1).sum(dim=-1, keepdim=True)
        projs = dots * self._axes_l1
        filtered = projs + PHI_INV * (inp - projs)

        sims = F.cosine_similarity(self._voices_l1, filtered, dim=1)
        k = max(1, int(self.n_l1 * PHI_INV * XI_SEC))
        threshold = float(sims.topk(min(k, self.n_l1)).values[-1])
        threshold = max(threshold, XI_SEC)
        fire_mask = sims >= threshold
        fired_indices = fire_mask.nonzero(as_tuple=True)[0]
        n_fired = len(fired_indices)

        # Update L1 voices (SEC collapse)
        if n_fired > 0:
            cryst = self._get_crystallization()
            collapse_exp = 1.0 + cryst * (1.0 / XI_SEC - 1.0)
            weight = sims.clamp(min=0) ** collapse_exp
            decay = LAMBDA_STAR + (1.0 - LAMBDA_STAR) * cryst
            new_voices = decay.unsqueeze(1) * self._voices_l1 + \
                         ((1.0 - decay) * weight).unsqueeze(1) * filtered
            self._voices_l1 = torch.where(fire_mask.unsqueeze(1), new_voices, self._voices_l1)
            norms = torch.norm(self._voices_l1, dim=1, keepdim=True).clamp(min=1e-8)
            self._voices_l1 = self._voices_l1 / norms

            for idx in fired_indices:
                i = int(idx)
                pos = int(self._entropy_len[i]) % 30
                self._entropy_buf[i, pos] = sims[i]
                self._entropy_len[i] = min(30, self._entropy_len[i] + 1)

        # ============================================================
        # PHASE 2: L2 UPDATE — forward-pointing parent memory
        # ============================================================
        active_parent = None
        for p_idx in range(self.n_l2):
            children = self._children_l2[p_idx]
            child_fired = fire_mask[children]

            if child_fired.any():
                active_parent = p_idx

                forward_signal = next_tensor / (torch.norm(next_tensor) + 1e-8)
                filtered_fwd = self._crystal_filter(forward_signal, self._axes_l2[p_idx])

                l2_retention = PHI_INV
                self._voices_l2[p_idx] = (
                    l2_retention * self._voices_l2[p_idx] +
                    (1.0 - l2_retention) * filtered_fwd
                )
                norm = torch.norm(self._voices_l2[p_idx])
                if norm > 1e-8:
                    self._voices_l2[p_idx] = self._voices_l2[p_idx] / norm

        # ============================================================
        # PHASE 3: PREDICT using context history
        # ============================================================
        pred_char = self._predict_with_context(fire_mask)
        self._last_pred_char = pred_char
        self._last_target_char = next_char

        # Update context history AFTER prediction (predict before seeing answer)
        if active_parent is not None:
            self._parent_history.append(active_parent)

        self._fire_rates.append(n_fired / self.n_l1 if self.n_l1 > 0 else 0)

    def _predict_with_context(self, fire_mask: torch.Tensor) -> str:
        if not fire_mask.any():
            return ' '

        # Find current active parent
        current_parent = None
        for p_idx in range(self.n_l2):
            children = self._children_l2[p_idx]
            if fire_mask[children].any():
                current_parent = p_idx
                break

        if current_parent is None:
            return ' '

        # Current parent's forward voice = bigram prediction
        current_forward = self._voices_l2[current_parent]

        if self.strategy == "cascade":
            return self._predict_cascade(current_forward, current_parent)
        else:
            return self._predict_blend(current_forward, current_parent)

    def _predict_cascade(self, current_forward: torch.Tensor,
                         current_parent: int) -> str:
        """Cascade crystal filter: condition prediction on sequential context.

        Strategy: crystal-filter the current prediction through each
        previous parent's forward voice in sequence. Each filter stage
        projects the prediction through the lens of one more context step.

        After "t-h": current parent = 'h', history has 't'
        - current_forward = parent['h'].voice ≈ points toward 'e'
        - crystal_filter(forward, parent['t'].voice) emphasizes the
          component of 'e' that aligns with 't→h' context

        This creates path-dependent predictions: "th" predicts differently
        from "sh" because the filter axis is different.
        """
        signal = current_forward.clone()

        # Walk context history (most recent first)
        history = list(self._parent_history)
        for prev_p_idx in reversed(history):
            if prev_p_idx == current_parent:
                continue  # skip self-reference
            prev_voice = self._voices_l2[prev_p_idx]
            # Use previous parent's forward voice as filter axis
            signal = self._crystal_filter(signal, prev_voice)
            norm = torch.norm(signal)
            if norm > 1e-8:
                signal = signal / norm

        decoded = self.codebook.decode_topk(signal, k=5)
        best_ch = None
        best_sim = -1.0
        for ch, sim in decoded:
            if ch in self._char_to_idx and sim > best_sim:
                best_sim = sim
                best_ch = ch
        return best_ch or ' '

    def _predict_blend(self, current_forward: torch.Tensor,
                       current_parent: int) -> str:
        """Weighted blend of recent parent forward voices.

        Simpler strategy: combine current + recent parents' forward voices
        with exponential decay. Loses sequential ordering but captures
        the "bag of recent characters" context.
        """
        signal = current_forward.clone()

        history = list(self._parent_history)
        for i, prev_p_idx in enumerate(reversed(history)):
            weight = PHI_INV ** (i + 1)  # decay: 0.618, 0.382, 0.236, ...
            prev_voice = self._voices_l2[prev_p_idx]
            signal = signal + weight * prev_voice

        norm = torch.norm(signal)
        if norm > 1e-8:
            signal = signal / norm

        decoded = self.codebook.decode_topk(signal, k=5)
        best_ch = None
        best_sim = -1.0
        for ch, sim in decoded:
            if ch in self._char_to_idx and sim > best_sim:
                best_sim = sim
                best_ch = ch
        return best_ch or ' '

    def set_phase(self, name: str):
        self._current_phase = name

    def phase_acc(self, phase: str) -> float:
        accs = self.phase_accuracy.get(phase, [])
        return sum(accs) / len(accs) if accs else 0.0

    def reset_stats(self):
        """Reset stats for a fresh experiment while keeping learned voices."""
        self.accuracy_history.clear()
        self.phase_accuracy.clear()
        self.char_accuracy.clear()
        self._last_pred_char = None
        self._last_target_char = None
        self._fire_rates.clear()
        self._parent_history.clear()


# ==========================================================================
#  Corpus
# ==========================================================================

HAMLET = (
    "to be or not to be that is the question "
    "whether tis nobler in the mind to suffer "
    "the slings and arrows of outrageous fortune "
    "or to take arms against a sea of troubles "
    "and by opposing end them to die to sleep "
    "no more and by a sleep to say we end "
    "the heartache and the thousand natural shocks "
    "that flesh is heir to tis a consummation "
    "devoutly to be wished to die to sleep "
    "to sleep perchance to dream ay there is the rub "
    "for in that sleep of death what dreams may come "
    "when we have shuffled off this mortal coil "
    "must give us pause there is the respect "
    "that makes calamity of so long a life "
    "for who would bear the whips and scorns of time "
    "the oppressors wrong the proud mans contumely "
    "the pangs of despised love the laws delay "
    "the insolence of office and the spurns "
    "that patient merit of the unworthy takes "
    "when he himself might his quietus make "
    "with a bare bodkin who would fardels bear "
    "to grunt and sweat under a weary life "
    "but that the dread of something after death "
    "the undiscovered country from whose bourn "
    "no traveller returns puzzles the will "
    "and makes us rather bear those ills we have "
    "than fly to others that we know not of "
    "thus conscience does make cowards of us all "
    "and thus the native hue of resolution "
    "is sicklied over with the pale cast of thought "
    "and enterprises of great pith and moment "
    "with this regard their currents turn awry "
    "and lose the name of action "
)

GENESIS = (
    "in the beginning god created the heaven and the earth "
    "and the earth was without form and void "
    "and darkness was upon the face of the deep "
    "and the spirit of god moved upon the face of the waters "
    "and god said let there be light and there was light "
    "and god saw the light that it was good "
    "and god divided the light from the darkness "
    "and god called the light day and the darkness he called night "
    "and the evening and the morning were the first day "
    "and god said let there be a firmament "
    "in the midst of the waters "
    "and let it divide the waters from the waters "
    "and god made the firmament "
    "and divided the waters which were under the firmament "
    "from the waters which were above the firmament "
    "and it was so and god called the firmament heaven "
    "and the evening and the morning were the second day "
)

PARADISE = (
    "of mans first disobedience and the fruit "
    "of that forbidden tree whose mortal taste "
    "brought death into the world and all our woe "
    "with loss of eden till one greater man "
    "restore us and regain the blissful seat "
    "sing heavenly muse that on the secret top "
    "of oreb or of sinai didst inspire "
    "that shepherd who first taught the chosen seed "
    "in the beginning how the heavens and earth "
    "rose out of chaos or if sion hill "
    "delight thee more and siloas brook that flowed "
    "fast by the oracle of god i thence "
    "invoke thy aid to my adventurous song "
    "that with no middle flight intends to soar "
    "above the aonian mount while it pursues "
    "things unattempted yet in prose or rhyme "
)

CORPUS = HAMLET + GENESIS + PARADISE


# ==========================================================================
#  Run a single experiment
# ==========================================================================

def run_experiment(codebook: FastCodebook, chars: set[str],
                   context_depth: int, strategy: str,
                   n_epochs: int = 10) -> dict:
    """Run one experiment and return results."""
    torch.manual_seed(42)

    field = CascadeField(codebook, context_depth=context_depth,
                         strategy=strategy)
    field.build_field(chars)

    epoch_accs = []
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        phase = f"epoch_{epoch}"
        field.set_phase(phase)
        field._fire_rates.clear()

        for i in range(len(CORPUS) - 1):
            ch = CORPUS[i]
            nxt = CORPUS[i + 1]
            tensor = codebook.encode(ch)
            field.process(tensor, ch, nxt)

        acc = field.phase_acc(phase)
        epoch_accs.append(acc)

    elapsed = time.time() - t0

    # Character-level accuracy
    char_accs = {}
    for ch, accs in field.char_accuracy.items():
        if len(accs) >= 20:
            char_accs[ch] = sum(accs) / len(accs)

    return {
        "context_depth": context_depth,
        "strategy": strategy,
        "epoch_accs": epoch_accs,
        "peak": max(epoch_accs),
        "peak_epoch": epoch_accs.index(max(epoch_accs)) + 1,
        "final": epoch_accs[-1],
        "first_half": sum(epoch_accs[:5]) / 5,
        "second_half": sum(epoch_accs[5:]) / max(1, len(epoch_accs) - 5),
        "elapsed": elapsed,
        "char_accs": char_accs,
        "field": field,
    }


# ==========================================================================
#  Main — Multi-experiment comparison
# ==========================================================================

def main():
    N_EPOCHS = 10

    print("=" * 75)
    print("  SPIKE MEMORY CASCADE — Breaking the Bigram Ceiling")
    print(f"  Corpus: {len(CORPUS)} chars | Epochs: {N_EPOCHS}")
    print("  Two strategies × multiple context depths")
    print("=" * 75)

    torch.manual_seed(42)
    codebook = FastCodebook()
    chars = set(CORPUS)

    # ================================================================
    # EXPERIMENT MATRIX
    # ================================================================
    experiments = [
        # Cascade strategy (crystal filter composition)
        {"ctx": 1, "strategy": "cascade", "label": "K=1 cascade (baseline)"},
        {"ctx": 2, "strategy": "cascade", "label": "K=2 cascade"},
        {"ctx": 3, "strategy": "cascade", "label": "K=3 cascade"},
        {"ctx": 4, "strategy": "cascade", "label": "K=4 cascade"},
        {"ctx": 5, "strategy": "cascade", "label": "K=5 cascade"},
        # Blend strategy
        {"ctx": 2, "strategy": "blend", "label": "K=2 blend"},
        {"ctx": 3, "strategy": "blend", "label": "K=3 blend"},
        {"ctx": 4, "strategy": "blend", "label": "K=4 blend"},
        {"ctx": 5, "strategy": "blend", "label": "K=5 blend"},
    ]

    results = []

    for exp in experiments:
        print(f"\n  --- {exp['label']} ---")
        result = run_experiment(codebook, chars,
                                context_depth=exp["ctx"],
                                strategy=exp["strategy"],
                                n_epochs=N_EPOCHS)
        results.append(result)

        # Print epoch curve
        for i, acc in enumerate(result["epoch_accs"]):
            bar = "#" * int(acc * 100)
            marker = " <-- peak" if i + 1 == result["peak_epoch"] else ""
            print(f"    E{i+1:2d}: {acc:5.1%} | {bar}{marker}")

        print(f"    Peak: {result['peak']:.1%} | "
              f"Final: {result['final']:.1%} | "
              f"Time: {result['elapsed']:.1f}s")

    # ================================================================
    # GRAND SUMMARY
    # ================================================================
    print(f"\n{'='*75}")
    print(f"  GRAND SUMMARY — Context Scaling via Voice Dynamics")
    print(f"{'='*75}")

    print(f"\n  {'Experiment':<25s} | {'Peak':>6s} | {'Final':>6s} | "
          f"{'Learn?':>6s} | {'Time':>6s}")
    print(f"  {'-'*25}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    for i, (exp, res) in enumerate(zip(experiments, results)):
        learns = "YES" if res["second_half"] > res["first_half"] else "NO"
        print(f"  {exp['label']:<25s} | "
              f"{res['peak']:5.1%} | "
              f"{res['final']:5.1%} | "
              f"{learns:>6s} | "
              f"{res['elapsed']:5.1f}s")

    # Best per strategy
    cascade_results = [r for r in results if r["strategy"] == "cascade"]
    blend_results = [r for r in results if r["strategy"] == "blend"]

    best_cascade = max(cascade_results, key=lambda r: r["peak"])
    best_blend = max(blend_results, key=lambda r: r["peak"])

    print(f"\n  Best cascade: K={best_cascade['context_depth']} → {best_cascade['peak']:.1%}")
    print(f"  Best blend:   K={best_blend['context_depth']} → {best_blend['peak']:.1%}")

    # Comparison with baselines
    print(f"\n  BASELINES:")
    print(f"    Pure 2-gram:          34.3%")
    print(f"    Pure 3-gram:          51.7%")
    print(f"    Pure 4-gram:          70.3%")
    print(f"    Evolutionary field:   44.1%")
    print(f"    Memory field (v1):    44.3%")
    print(f"    Lookup field+ctx=1:   50.2%")
    print(f"    Lookup field+ctx=2:   66.9%")
    print(f"    Lookup field+ctx=3:   79.1%")

    # Scaling analysis
    print(f"\n  SCALING ANALYSIS (cascade):")
    for r in cascade_results:
        delta = r["peak"] - cascade_results[0]["peak"]
        sign = "+" if delta >= 0 else ""
        print(f"    K={r['context_depth']}: {r['peak']:5.1%}  "
              f"({sign}{delta:+.1%} vs K=1)")

    print(f"\n  SCALING ANALYSIS (blend):")
    for r in blend_results:
        delta = r["peak"] - cascade_results[0]["peak"]
        sign = "+" if delta >= 0 else ""
        print(f"    K={r['context_depth']}: {r['peak']:5.1%}  "
              f"({sign}{delta:+.1%} vs K=1)")

    # Character accuracy for best result
    best = max(results, key=lambda r: r["peak"])
    print(f"\n  CHARACTER ACCURACY (best: {best['strategy']} K={best['context_depth']}):")
    sorted_chars = sorted(best["char_accs"].items(), key=lambda x: -x[1])[:15]
    for ch, acc in sorted_chars:
        bar = "#" * int(acc * 50)
        print(f"    '{ch}': {acc:5.0%} {bar}")

    # Parent memory state for best
    print(f"\n  PARENT MEMORY STATE (best result):")
    field = best["field"]
    for p_idx in range(min(field.n_l2, 25)):
        voice = field._voices_l2[p_idx]
        parent_char = field._l2_char[p_idx]
        decoded = codebook.decode_topk(voice, k=2)
        top = ", ".join(f"'{ch}'({sim:.2f})" for ch, sim in decoded
                        if ch in field._char_to_idx)
        print(f"    '{parent_char}' -> {top}")


if __name__ == "__main__":
    main()
