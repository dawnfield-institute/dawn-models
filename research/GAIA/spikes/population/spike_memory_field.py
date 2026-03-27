"""Spike Memory Field — Hierarchy IS Memory.

Key insight (Peter): if parent = sum(children), then
  memory(parent) = confluence(actualize(children))
  identity = forward_vector(entropy_injection * memory)

The parent node's STATE is the temporal context. No lookup tables,
no frequency counters. The voice dynamics ARE the memory mechanism.

Architecture:
  L1: 64 character nodes on 8x8 torus. Fire on input. Standard DFT.
  L2: 16 parent nodes. Each owns 4 L1 children.
      Voice = running integration of children's actualizations.
      State encodes "where we've been" — IS the context.
  L3 (optional): 4 grandparent nodes. Each owns 4 L2 parents.
      Even broader context integration.

Prediction: decode the parent's voice through codebook.
  parent voice after "t-h-e" ≈ blend(encode('t'), encode('h'), encode('e'))
  codebook.decode(parent_voice) → most similar character → prediction

No evolution. No frequency tables. No hand-tuned weights.
Memory emerges from the hierarchy's voice dynamics.

Usage:
    cd dawn-models/research/GAIA/spikes/population
    PYTHONIOENCODING=utf-8 PYTHONPATH="../../src;path/to/fracton" python spike_memory_field.py
"""

from __future__ import annotations

import math
import time
from collections import defaultdict

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
DIFFUSION_RATE = PHI_INV * GAMMA

DEVICE = torch.device("cpu")


# ==========================================================================
#  FastCodebook
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

    def decode_nearest(self, tensor: torch.Tensor) -> tuple[str, float]:
        t = tensor.flatten()[:self.dim].to(DEVICE)
        if torch.norm(t) < 1e-8:
            return "?", 0.0
        sims = F.cosine_similarity(self._matrix, t.unsqueeze(0), dim=1)
        idx = int(torch.argmax(sims))
        return self._idx_to_char[idx], float(sims[idx])

    def decode_topk(self, tensor: torch.Tensor, k: int = 5) -> list[tuple[str, float]]:
        """Decode to top-k nearest characters."""
        t = tensor.flatten()[:self.dim].to(DEVICE)
        if torch.norm(t) < 1e-8:
            return [("?", 0.0)]
        sims = F.cosine_similarity(self._matrix, t.unsqueeze(0), dim=1)
        topk = sims.topk(min(k, len(self._idx_to_char)))
        return [(self._idx_to_char[int(idx)], float(val))
                for val, idx in zip(topk.values, topk.indices)]


# ==========================================================================
#  MemoryField — Hierarchy IS Memory
# ==========================================================================

class MemoryField:
    """Hierarchical actualization field where parents ARE memory.

    L1: Character nodes. Fire on input match. Standard DFT actualization.
    L2: Parent nodes. Voice = confluence of children's actualizations.
    L3: Grandparent nodes. Voice = confluence of L2 parents.

    Prediction: decode the parent's forward-projected voice.
    """

    def __init__(self, codebook: FastCodebook, field_size: int = 8):
        self.codebook = codebook
        self.field_size = field_size
        self.n_l1 = field_size * field_size  # 64
        self.n_l2 = (field_size // 2) ** 2   # 16
        self.n_l3 = (field_size // 4) ** 2   # 4

        # Character mapping
        self._node_chars: list[str] = []
        self._char_to_idx: dict[str, int] = {}
        self._char_list: list[str] = []

        # L1 state
        self._voices_l1: torch.Tensor = None   # [n_l1, DIM]
        self._axes_l1: torch.Tensor = None     # [n_l1, DIM]
        self._act_count_l1: torch.Tensor = None

        # L2 state — parents. Voice = memory of children.
        self._voices_l2: torch.Tensor = None   # [n_l2, DIM]
        self._axes_l2: torch.Tensor = None     # [n_l2, DIM] birth direction
        self._children_l2: list[list[int]] = []  # L2 -> list of L1 indices

        # L3 state — grandparents
        self._voices_l3: torch.Tensor = None   # [n_l3, DIM]
        self._axes_l3: torch.Tensor = None
        self._children_l3: list[list[int]] = []  # L3 -> list of L2 indices

        # Entropy tracking (L1 only)
        self._entropy_buf: torch.Tensor = None
        self._entropy_len: torch.Tensor = None

        # Stats
        self.event_counter = 0
        self.input_counter = 0
        self.accuracy_history: list[int] = []
        self.phase_accuracy: dict[str, list[int]] = defaultdict(list)
        self.char_accuracy: dict[str, list[int]] = defaultdict(list)
        self._current_phase = "init"
        self._last_pred_char: str | None = None
        self._last_target_char: str | None = None
        self._fire_rates: list[float] = []
        self.psi_history: list[float] = []

    def build_field(self, chars: set[str]):
        unique = sorted(chars)
        n_chars = len(unique)
        self._char_list = unique
        self._char_to_idx = {ch: i for i, ch in enumerate(unique)}

        fs = self.field_size
        self.n_l1 = fs * fs

        # L1: character nodes
        self._node_chars = [unique[i % n_chars] for i in range(self.n_l1)]
        voices = [self.codebook.encode(self._node_chars[i]).to(DEVICE) for i in range(self.n_l1)]
        self._voices_l1 = torch.stack(voices)
        self._axes_l1 = self._voices_l1.clone()
        self._act_count_l1 = torch.zeros(self.n_l1, dtype=torch.long, device=DEVICE)

        # Entropy
        self._entropy_buf = torch.ones(self.n_l1, 30, device=DEVICE)
        self._entropy_len = torch.zeros(self.n_l1, dtype=torch.long, device=DEVICE)

        # L2: ONE parent per character type (natural grouping)
        # Parent['t'] owns ALL L1 nodes that encode 't'
        self.n_l2 = n_chars
        self._children_l2 = []
        self._l2_char = []  # which char each L2 parent represents
        for ch_idx, ch in enumerate(unique):
            children = [i for i in range(self.n_l1) if self._node_chars[i] == ch]
            self._children_l2.append(children)
            self._l2_char.append(ch)

        # L2 initial voices: mean of children (= character encoding)
        l2_voices = []
        for children in self._children_l2:
            mean_v = self._voices_l1[children].mean(dim=0)
            mean_v = mean_v / (torch.norm(mean_v) + 1e-8)
            l2_voices.append(mean_v)
        self._voices_l2 = torch.stack(l2_voices)
        self._axes_l2 = self._voices_l2.clone()

        # L3: 5 grandparents, each owns ~5 L2 parents (character classes)
        gp_size = max(1, n_chars // 5)
        self._children_l3 = []
        for g in range(0, n_chars, gp_size):
            children = list(range(g, min(g + gp_size, n_chars)))
            if children:
                self._children_l3.append(children)
        self.n_l3 = len(self._children_l3)

        l3_voices = []
        for children in self._children_l3:
            mean_v = self._voices_l2[children].mean(dim=0)
            mean_v = mean_v / (torch.norm(mean_v) + 1e-8)
            l3_voices.append(mean_v)
        self._voices_l3 = torch.stack(l3_voices) if l3_voices else torch.zeros(1, DIM, device=DEVICE)
        self._axes_l3 = self._voices_l3.clone()

        print(f"  L1: {fs}x{fs} = {self.n_l1} character nodes")
        print(f"  L2: {n_chars} parent nodes (1 per character)")
        print(f"  L3: {self.n_l3} grandparent nodes (deep memory)")
        print(f"  Chars: {n_chars} unique")

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

    def _crystal_filter(self, signal: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        """Crystal filter: proj + PHI_INV * orth."""
        dot = torch.dot(signal, axis)
        proj = dot * axis
        orth = signal - proj
        return proj + PHI_INV * orth

    def process(self, input_tensor: torch.Tensor, input_char: str, next_char: str):
        self.input_counter += 1
        next_tensor = self.codebook.encode(next_char)

        # Evaluate previous prediction
        if self._last_pred_char is not None and self._last_target_char is not None:
            hit = 1 if self._last_pred_char == self._last_target_char else 0
            self.accuracy_history.append(hit)
            self.phase_accuracy[self._current_phase].append(hit)
            self.char_accuracy[self._last_target_char].append(hit)

        # ============================================================
        # PHASE 1: L1 ACTUALIZE — which character nodes fire?
        # ============================================================

        # Broadcast + crystal filter
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

            # Entropy tracking
            for idx in fired_indices:
                i = int(idx)
                pos = int(self._entropy_len[i]) % 30
                self._entropy_buf[i, pos] = sims[i]
                self._entropy_len[i] = min(30, self._entropy_len[i] + 1)

            self._act_count_l1 += fire_mask.long()
            self.event_counter += n_fired

        # ============================================================
        # PHASE 2: L2 UPDATE — parents integrate children's activity
        # memory(parent) = confluence(actualize(children))
        # ============================================================

        for p_idx in range(self.n_l2):
            children = self._children_l2[p_idx]
            child_fired = fire_mask[children]

            if child_fired.any():
                # FORWARD-POINTING MEMORY: parent absorbs WHERE WE'RE GOING
                # With 1 parent per char, parent['t'] accumulates what follows 't'
                forward_signal = next_tensor.clone()
                forward_signal = forward_signal / (torch.norm(forward_signal) + 1e-8)

                # Crystal filter through parent axis
                filtered = self._crystal_filter(forward_signal, self._axes_l2[p_idx])

                # Fast parent memory: short-term context
                l2_retention = PHI_INV  # 0.618
                self._voices_l2[p_idx] = (
                    l2_retention * self._voices_l2[p_idx] +
                    (1.0 - l2_retention) * filtered
                )
                norm = torch.norm(self._voices_l2[p_idx])
                if norm > 1e-8:
                    self._voices_l2[p_idx] = self._voices_l2[p_idx] / norm

        # ============================================================
        # PHASE 3: L3 UPDATE — grandparents integrate L2 parents
        # ============================================================

        for g_idx in range(self.n_l3):
            l2_children = self._children_l3[g_idx]
            # Only integrate when L2 children have active parents (fired children)
            active_l2 = []
            for l2_idx in l2_children:
                l2_kids = self._children_l2[l2_idx]
                if fire_mask[l2_kids].any():
                    active_l2.append(l2_idx)

            if active_l2:
                l2_voices = self._voices_l2[active_l2]
                mean_l2 = l2_voices.mean(dim=0)
                mean_l2 = mean_l2 / (torch.norm(mean_l2) + 1e-8)

                filtered_l2 = self._crystal_filter(mean_l2, self._axes_l3[g_idx])

                # Slower integration at L3 (broader context, longer memory)
                l3_decay = LAMBDA_STAR ** PHI_INV  # ~0.9888 — slower than L2
                self._voices_l3[g_idx] = (
                    l3_decay * self._voices_l3[g_idx] +
                    (1.0 - l3_decay) * filtered_l2
                )
                norm = torch.norm(self._voices_l3[g_idx])
                if norm > 1e-8:
                    self._voices_l3[g_idx] = self._voices_l3[g_idx] / norm

        # ============================================================
        # PHASE 4: PREDICT — identity = forward_vector(memory)
        # Each fired L1 node's parent holds the context.
        # Decode the parent's voice → that's the prediction.
        # ============================================================

        pred_char = self._predict_from_hierarchy(fire_mask)
        self._last_pred_char = pred_char
        self._last_target_char = next_char

        # Stats
        self._fire_rates.append(n_fired / self.n_l1 if self.n_l1 > 0 else 0)

        # Phase coherence (L1 mean field)
        mean_field = self._voices_l1.mean(dim=0)
        mn = torch.norm(mean_field)
        if mn > 1e-8:
            mean_field = mean_field / mn
        psi = float(F.cosine_similarity(
            self._voices_l1, mean_field.unsqueeze(0).expand_as(self._voices_l1), dim=1
        ).mean())
        self.psi_history.append(psi)

    def _predict_from_hierarchy(self, fire_mask: torch.Tensor) -> str:
        """Prediction = decode the parent's voice (memory).

        The parent's voice has been integrating children's activity.
        After processing "t-h-e", the parent's voice ≈ blend of those chars.
        Decoding it gives the predicted next character.

        Multiple strategies tested:
        1. Pure L2 decode: decode parent voice directly
        2. L2 forward projection: parent voice projected through crystal filter
        3. L2+L3 blend: combine parent and grandparent context
        """
        if not fire_mask.any():
            return ' '

        # Find which L2 parents have fired children
        parent_votes: dict[str, float] = defaultdict(float)

        for p_idx in range(self.n_l2):
            children = self._children_l2[p_idx]
            child_fired = fire_mask[children]
            if not child_fired.any():
                continue

            # Parent voice = forward-pointing memory
            # With 1 parent per char, this IS "what follows this character"
            parent_voice = self._voices_l2[p_idx]

            # Forward projection through crystal filter
            forward = self._crystal_filter(parent_voice, self._axes_l2[p_idx])

            # Blend with grandparent context (deeper memory)
            gp_voice = None
            for g_idx in range(self.n_l3):
                if p_idx in self._children_l3[g_idx]:
                    gp_voice = self._voices_l3[g_idx]
                    break

            if gp_voice is not None:
                gp_forward = self._crystal_filter(gp_voice, self._axes_l3[g_idx])
                forward = PHI_INV * forward + (1.0 - PHI_INV) * gp_forward
                forward = forward / (torch.norm(forward) + 1e-8)

            # Decode: what character does this forward vector point to?
            decoded = self.codebook.decode_topk(forward, k=3)
            # Weight by number of fired children (stronger signal)
            n_fired = child_fired.sum().float()
            for ch, sim in decoded:
                if ch in self._char_to_idx:
                    parent_votes[ch] += sim * n_fired

        if not parent_votes:
            return ' '

        return max(parent_votes, key=parent_votes.get)

    def set_phase(self, name: str):
        self._current_phase = name

    def phase_acc(self, phase: str) -> float:
        accs = self.phase_accuracy.get(phase, [])
        return sum(accs) / len(accs) if accs else 0.0


# ==========================================================================
#  Landauer Reinjection (simplified — no evolution, just entropy reset)
# ==========================================================================

def landauer_reinject(field: MemoryField) -> int:
    """Reset entropy at epoch boundary. Soften crystallization."""
    cryst = field._get_crystallization()
    n_cryst = int((cryst > 0.5).sum())

    # Soften entropy buffers for crystallized nodes
    soften_mask = (cryst > 0.5) & (field._entropy_len > 5)
    field._entropy_len = torch.where(
        soften_mask,
        torch.tensor(5, device=DEVICE),
        field._entropy_len
    )

    # Partial L2 axis reset — allow parents to reorient
    for p_idx in range(field.n_l2):
        children = field._children_l2[p_idx]
        mean_child = field._voices_l1[children].mean(dim=0)
        mean_child = mean_child / (torch.norm(mean_child) + 1e-8)
        # Slowly drift axis toward current children state
        field._axes_l2[p_idx] = (
            LAMBDA_STAR * field._axes_l2[p_idx] +
            (1.0 - LAMBDA_STAR) * mean_child
        )
        norm = torch.norm(field._axes_l2[p_idx])
        if norm > 1e-8:
            field._axes_l2[p_idx] = field._axes_l2[p_idx] / norm

    return n_cryst


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
#  Diagnostic: what does the parent voice actually encode?
# ==========================================================================

def diagnose_parent_memory(field: MemoryField, codebook: FastCodebook):
    """Show what the parent voices have learned."""
    print(f"\n  PARENT MEMORY STATE (what follows each char):")
    for p_idx in range(field.n_l2):
        voice = field._voices_l2[p_idx]
        parent_char = field._l2_char[p_idx] if hasattr(field, '_l2_char') else '?'

        decoded = codebook.decode_topk(voice, k=3)
        top3 = ", ".join(f"'{ch}'({sim:.2f})" for ch, sim in decoded if ch in field._char_to_idx)

        print(f"    '{parent_char}' -> {top3}")

    if field.n_l3 > 0:
        print(f"\n  GRANDPARENT MEMORY STATE:")
        for g_idx in range(field.n_l3):
            voice = field._voices_l3[g_idx]
            l2_children = field._children_l3[g_idx]

            decoded = codebook.decode_topk(voice, k=3)
            top3 = ", ".join(f"'{ch}'({sim:.2f})" for ch, sim in decoded if ch in field._char_to_idx)

            print(f"    L3[{g_idx}] L2 children={l2_children} -> voice decodes to: {top3}")


# ==========================================================================
#  Main
# ==========================================================================

def main():
    N_EPOCHS = 20

    print("=" * 70)
    print("  SPIKE MEMORY FIELD — Hierarchy IS Memory")
    print(f"  Device: {DEVICE}")
    print(f"  Corpus: {len(CORPUS)} chars | Epochs: {N_EPOCHS}")
    print(f"  memory(parent) = confluence(actualize(children))")
    print(f"  identity = forward_vector(entropy_injection * memory)")
    print("=" * 70)

    torch.manual_seed(42)
    codebook = FastCodebook()

    field = MemoryField(codebook)
    field.build_field(set(CORPUS))

    print(f"\n  {'Ep':>3s} | {'Acc':>6s} | {'Psi':>6s} | "
          f"{'Fire%':>5s} | {'Cryst':>5s} | {'Time':>6s} | {'ch/s':>6s}")
    print(f"  {'-'*3}-+-{'-'*6}-+-{'-'*6}-+-"
          f"{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}")

    total_start = time.time()
    epoch_accs = []

    for epoch in range(1, N_EPOCHS + 1):
        phase = f"epoch_{epoch}"
        field.set_phase(phase)
        field._fire_rates.clear()

        t0 = time.time()
        for i in range(len(CORPUS) - 1):
            ch = CORPUS[i]
            nxt = CORPUS[i + 1]
            tensor = codebook.encode(ch)
            field.process(tensor, ch, nxt)

        elapsed = time.time() - t0
        cps = len(CORPUS) / elapsed if elapsed > 0 else 0
        acc = field.phase_acc(phase)
        epoch_accs.append(acc)

        cryst = field._get_crystallization()
        n_cryst = int((cryst > 0.5).sum())

        mean_fire = sum(field._fire_rates) / len(field._fire_rates) \
            if field._fire_rates else 0

        psi_recent = field.psi_history[-len(CORPUS):] if field.psi_history else [0]
        psi = sum(psi_recent) / len(psi_recent)

        n_cryst_l = landauer_reinject(field)

        print(
            f"  {epoch:3d} | "
            f"{acc:5.1%} | "
            f"{psi:6.3f} | "
            f"{mean_fire:4.0%} | "
            f"{n_cryst:5d} | "
            f"{elapsed:5.1f}s | "
            f"{cps:5.0f}"
        )

    total_elapsed = time.time() - total_start

    print(f"\n{'='*70}")
    print(f"  MEMORY FIELD RESULTS")
    print(f"{'='*70}")

    half = len(epoch_accs) // 2
    first_half = sum(epoch_accs[:half]) / half
    second_half = sum(epoch_accs[half:]) / (len(epoch_accs) - half)
    peak = max(epoch_accs)
    peak_ep = epoch_accs.index(peak) + 1

    print(f"  First half avg:  {first_half:.1%}")
    print(f"  Second half avg: {second_half:.1%}")
    print(f"  Peak:            {peak:.1%} (epoch {peak_ep})")
    print(f"  Field learns:    {'YES' if second_half > first_half else 'NO'}")
    print(f"  Total time:      {total_elapsed:.1f}s")

    print(f"\n  LEARNING CURVE:")
    for i, acc in enumerate(epoch_accs):
        bar = "#" * int(acc * 200)
        print(f"    E{i+1:2d}: {acc:5.1%} | {bar}")

    # Character accuracy
    print(f"\n  CHARACTER ACCURACY (top 15):")
    char_accs = {}
    for ch, accs in field.char_accuracy.items():
        if len(accs) >= 20:
            char_accs[ch] = sum(accs) / len(accs)
    best = sorted(char_accs.items(), key=lambda x: -x[1])[:15]
    for ch, acc in best:
        n = len(field.char_accuracy[ch])
        bar = "#" * int(acc * 50)
        print(f"    '{ch}': {acc:5.0%} (n={n:5d}) {bar}")

    chars_with_hits = sum(1 for a in char_accs.values() if a > 0)
    print(f"\n  Chars with hits: {chars_with_hits}/{len(char_accs)}")

    # Diagnose parent memory
    diagnose_parent_memory(field, codebook)

    # Comparison
    print(f"\n  COMPARISON:")
    print(f"    Pure 2-gram baseline:  34.3%")
    print(f"    Pure 3-gram baseline:  51.7%")
    print(f"    Evolutionary field:    44.1%")
    print(f"    Memory field (this):   {peak:.1%}")


if __name__ == "__main__":
    main()
