"""Spike K CUDA -- Optimized Crystal Colony at Scale.

Strips the full GAIA pipeline down to the physics core:
  - LightCell: voice tensor + tree links, no 5-module agent
  - Vectorized navigation: batch cosine similarity on GPU
  - CUDA-accelerated tensor ops
  - Same physics: PAC tree growth, SEC navigation, potential field, Landauer epochs

The GAIA pipeline proved the physics works. Now we need speed to scale.

Usage:
    cd dawn-models/research/GAIA/spikes/population
    PYTHONPATH="../../src;path/to/fracton" python spike_k_cuda.py
"""

from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

# DFT Constants
XI_SEC = 0.0618033988749895
PHI_INV = 0.618033988749895
LAMBDA_STAR = 0.9816
GAMMA = 1.0 - LAMBDA_STAR
LN_PHI = math.log((1 + math.sqrt(5)) / 2)  # 0.4812
DIM = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================================
#  Codebook (GPU-resident)
# ==========================================================================

class FastCodebook:
    """Character codebook with vectorized decode on GPU."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self._char_to_idx: dict[str, int] = {}
        self._idx_to_char: list[str] = []
        self._vectors: list[torch.Tensor] = []
        self._matrix: torch.Tensor | None = None  # [n_chars, dim] for batch decode
        self._build()

    def _build(self):
        classes = {
            "vowel": list("aeiou"),
            "consonant": list("bcdfghjklmnpqrstvwxyz"),
            "digit": list("0123456789"),
            "space": [" ", "\n", "\t"],
            "punct": list(".,!?;:'\"()-"),
        }

        for class_idx, (class_name, chars) in enumerate(classes.items()):
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

        # Uppercase
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

        # Build matrix on device
        self._matrix = torch.stack(self._vectors).to(DEVICE)  # [n_chars, dim]

    def encode(self, char: str) -> torch.Tensor:
        if char in self._char_to_idx:
            return self._matrix[self._char_to_idx[char]].clone()
        # Hash fallback
        torch.manual_seed(hash(char) % 2**31)
        v = torch.randn(self.dim, device=DEVICE)
        return v / (torch.norm(v) + 1e-8)

    def decode_nearest(self, tensor: torch.Tensor) -> tuple[str, float]:
        """Vectorized decode: cosine similarity against all chars at once."""
        t = tensor.flatten()[:self.dim].to(DEVICE)
        t_norm = torch.norm(t)
        if t_norm < 1e-8:
            return "?", 0.0
        # [n_chars] = matmul([n_chars, dim], [dim])
        sims = F.cosine_similarity(self._matrix, t.unsqueeze(0), dim=1)
        idx = int(torch.argmax(sims))
        return self._idx_to_char[idx], float(sims[idx])

    def encode_string(self, text: str) -> torch.Tensor:
        """Encode full string as [len, dim] tensor."""
        indices = [self._char_to_idx.get(ch, -1) for ch in text]
        vecs = []
        for i, ch in enumerate(text):
            if indices[i] >= 0:
                vecs.append(self._matrix[indices[i]])
            else:
                vecs.append(self.encode(ch))
        return torch.stack(vecs)  # [len, dim]


# ==========================================================================
#  Light Cell (no GAIA pipeline -- pure physics)
# ==========================================================================

@dataclass
class LightCell:
    """Minimal cell: voice tensor + tree links + fixed crystal axis.

    Each cell has a fixed 'axis' — the normalized birth voice, set once
    and never modified. This is the cell's identity: what it was born to
    recognize. The crystal filter projects input along this fixed axis,
    creating CONSISTENT transformations → stable entropy → crystallization.

    The 'voice' evolves separately for resonance matching and codebook decode.
    """
    name: str
    voice: torch.Tensor  # [dim] evolving — for resonance and decode
    axis: torch.Tensor   # [dim] fixed — birth direction, never changes
    parent: LightCell | None = None
    children: list[LightCell] = field(default_factory=list)
    birth_tick: int = 0
    access_count: int = 0


# ==========================================================================
#  Fast Crystal Colony
# ==========================================================================

class FastCrystalColony:
    """Optimized CrystalColony: vectorized ops, GPU tensors, no GAIA pipeline.

    Same physics as CrystalColony:
    - PAC tree growth from residuals
    - SEC collapse navigation
    - Potential field prediction
    - Landauer epoch reinjection

    Cell processing is now just the voice update equation:
      voice = decay * old_voice + (1 - decay) * input
    No 5-module pipeline. The physics IS the processing.
    """

    def __init__(self, codebook: FastCodebook, max_cells: int = 500):
        self.codebook = codebook
        self.max_cells = max_cells
        self.tick = 0

        # Tree
        self.cells: dict[str, LightCell] = {}
        self.roots: list[str] = []
        self._next_id = 0

        # Physics state
        self._entropy_history: dict[str, list[float]] = {}
        self._potential: dict[str, float] = {}

        # Voice matrix cache for vectorized operations
        self._voice_matrix_dirty = True
        self._voice_names: list[str] = []
        self._voice_matrix: torch.Tensor = torch.zeros(0, DIM, device=DEVICE)

        # Tracking (must be before _spawn_root)
        self._pending_parent: LightCell | None = None
        self._last_pred_tensor: torch.Tensor | None = None
        self._last_target_char: str | None = None
        self._path_depths: list[int] = []
        self._residual_history: list[float] = []
        self._root_spawns = 0
        self._child_spawns = 0
        self._growth_checks = 0
        self._growth_hits = 0
        self._growth_zero_voice = 0
        self._growth_dead_parent = 0
        self._growth_no_pending = 0

        # Metrics
        self.accuracy_history: list[bool] = []
        self.phase_accuracy: dict[str, list[bool]] = defaultdict(list)
        self._current_phase = ""
        self.char_accuracy: dict[str, list[bool]] = defaultdict(list)

        # Leaf residual tracking (for Landauer)
        self.leaf_residuals: dict[str, torch.Tensor] = {}
        self.leaf_residual_counts: dict[str, int] = {}

        # Spawn seed (after all state initialized)
        seed = self._spawn_root(torch.zeros(DIM, device=DEVICE))

    def _rebuild_voice_matrix(self):
        """Rebuild the [n_cells, dim] voice matrix for vectorized ops."""
        self._voice_names = list(self.cells.keys())
        if self._voice_names:
            self._voice_matrix = torch.stack(
                [self.cells[n].voice for n in self._voice_names]
            )  # [n, dim]
        else:
            self._voice_matrix = torch.zeros(0, DIM, device=DEVICE)
        self._voice_matrix_dirty = False

    def _make_axis(self, voice: torch.Tensor) -> torch.Tensor:
        """Create fixed crystal axis from birth voice (normalized direction)."""
        norm = torch.norm(voice)
        if norm > 1e-8:
            return (voice / norm).clone()
        # Zero voice: random axis
        axis = torch.randn(DIM, device=DEVICE)
        return axis / (torch.norm(axis) + 1e-8)

    def _spawn_root(self, voice: torch.Tensor) -> LightCell:
        name = f"c{self._next_id}"
        self._next_id += 1
        v = voice.to(DEVICE).clone()
        cell = LightCell(name=name, voice=v, axis=self._make_axis(v))
        cell.birth_tick = self.tick
        self.cells[name] = cell
        self.roots.append(name)
        self._entropy_history[name] = []
        self._potential[name] = float(torch.norm(voice)) + 0.1
        self._voice_matrix_dirty = True
        self._root_spawns += 1
        return cell

    def _spawn_child(self, parent: LightCell, voice: torch.Tensor) -> LightCell:
        name = f"c{self._next_id}"
        self._next_id += 1
        v = voice.to(DEVICE).clone()
        cell = LightCell(name=name, voice=v, axis=self._make_axis(v), parent=parent)
        cell.birth_tick = self.tick
        parent.children.append(cell)
        self.cells[name] = cell
        self._entropy_history[name] = []
        self._potential[name] = float(torch.norm(voice))
        self._voice_matrix_dirty = True
        self._child_spawns += 1
        return cell

    # ------------------------------------------------------------------
    #  Vectorized navigation
    # ------------------------------------------------------------------

    def _batch_resonance(self, signal: torch.Tensor, names: list[str]) -> list[float]:
        """Compute cosine similarity of signal against a batch of cells."""
        if not names:
            return []
        voices = torch.stack([self.cells[n].voice for n in names])  # [k, dim]
        signal_norm = torch.norm(signal)
        if signal_norm < 1e-8:
            return [1.0] * len(names)
        sims = F.cosine_similarity(voices, signal.unsqueeze(0), dim=1)
        return [max(0.0, float(s)) for s in sims]

    def _navigate(self, input_tensor: torch.Tensor) -> list[tuple[str, float]]:
        """SEC collapse navigation with layered signal filtering."""
        if not self.roots:
            return []

        # Collapse exponent from crystallization fraction
        n_cryst = sum(1 for n in self.cells if self._get_entropy_var(n) < XI_SEC)
        cryst_frac = n_cryst / max(1, len(self.cells))
        collapse_exp = 1.0 + cryst_frac * (1.0 / XI_SEC - 1.0)

        signal = input_tensor
        valid_roots = [r for r in self.roots if r in self.cells]
        if not valid_roots:
            return []

        # Score roots
        scores = self._batch_resonance(signal, valid_roots)
        if collapse_exp > 1.01:
            collapsed = [s ** collapse_exp for s in scores]
        else:
            collapsed = scores

        best_idx = max(range(len(collapsed)), key=lambda i: collapsed[i])
        if scores[best_idx] < XI_SEC:
            return []

        path = [(valid_roots[best_idx], scores[best_idx])]
        current = self.cells[valid_roots[best_idx]]

        # Drill down with layered filtering
        while current.children:
            # Filter signal through parent
            pv_norm = float(torch.norm(current.voice))
            if pv_norm > 1e-6:
                signal = signal - current.voice

            child_names = [c.name for c in current.children if c.name in self.cells]
            if not child_names:
                break

            child_scores = self._batch_resonance(signal, child_names)
            if collapse_exp > 1.01:
                collapsed_cs = [s ** collapse_exp for s in child_scores]
            else:
                collapsed_cs = child_scores

            best_ci = max(range(len(collapsed_cs)), key=lambda i: collapsed_cs[i])
            if child_scores[best_ci] < XI_SEC:
                break

            path.append((child_names[best_ci], child_scores[best_ci]))
            current = self.cells[child_names[best_ci]]

        return path

    # ------------------------------------------------------------------
    #  Entropy / crystallization
    # ------------------------------------------------------------------

    def _get_entropy_var(self, name: str) -> float:
        hist = self._entropy_history.get(name, [])
        if len(hist) < 5:
            # Young cells: assume neutral (not crystallized, not maximally unstable)
            # Prevents newborns from being killed by dissipation before they can prove themselves
            return XI_SEC
        recent = hist[-20:]
        mean_h = sum(recent) / len(recent)
        return sum((h - mean_h) ** 2 for h in recent) / len(recent)

    def _compute_entropy(self, voice: torch.Tensor) -> float:
        v = torch.abs(voice) + 1e-10
        p = v / v.sum()
        return -float(torch.sum(p * torch.log(p)))

    # ------------------------------------------------------------------
    #  Potential field
    # ------------------------------------------------------------------

    def _redistribute_potential(self, path: list[tuple[str, float]], absorption: float):
        if not path:
            return
        leaf_name = path[-1][0]
        leaf_pot = self._potential.get(leaf_name, 0.0)
        consumed = leaf_pot * absorption
        unactualized = leaf_pot - consumed
        self._potential[leaf_name] = consumed

        if unactualized < 1e-8:
            return

        remaining = unactualized
        cryst = sum(1 for n in self.cells if self._get_entropy_var(n) < XI_SEC) / max(1, len(self.cells))
        collapse_exp = 1.0 + cryst * (1.0 / XI_SEC - 1.0)

        for i in range(len(path) - 1, -1, -1):
            cell_name = path[i][0]
            cell = self.cells.get(cell_name)
            if cell is None:
                continue

            if cell.parent is not None:
                siblings = [c for c in cell.parent.children
                            if c.name != cell_name and c.name in self.cells]
            else:
                siblings = [self.cells[r] for r in self.roots
                            if r != cell_name and r in self.cells]

            if not siblings:
                continue

            level_frac = PHI_INV if i == len(path) - 1 else PHI_INV ** 2
            level_amount = remaining * level_frac
            remaining -= level_amount

            sib_names = [s.name for s in siblings]
            leaf_voice = self.cells[leaf_name].voice
            resonances = self._batch_resonance(leaf_voice, sib_names)
            resonances = [max(XI_SEC, r) for r in resonances]

            weights = [r ** collapse_exp for r in resonances]
            total_w = sum(weights) + 1e-10
            for sname, w in zip(sib_names, weights):
                self._potential[sname] = self._potential.get(sname, 0.0) + level_amount * (w / total_w)

        if remaining > 1e-8:
            self._potential[leaf_name] += remaining

    def _predict_from_potential(self, path: list[tuple[str, float]]) -> tuple[str, torch.Tensor]:
        if not self._potential or not path:
            return "?", torch.zeros(DIM, device=DEVICE)

        leaf_name = path[-1][0]
        leaf_cell = self.cells.get(leaf_name)
        if leaf_cell is None:
            return "?", torch.zeros(DIM, device=DEVICE)

        candidates: list[str] = []
        candidates.extend(c.name for c in leaf_cell.children if c.name in self.cells)
        for cell_name, _ in path:
            cell = self.cells.get(cell_name)
            if cell is None:
                continue
            if cell.parent is not None:
                candidates.extend(c.name for c in cell.parent.children
                                  if c.name != cell_name and c.name in self.cells)
            else:
                candidates.extend(r for r in self.roots
                                  if r != cell_name and r in self.cells)

        if not candidates:
            path_root = path[0][0] if path else None
            candidates = [r for r in self.roots if r != path_root and r in self.cells]

        if not candidates:
            return "?", torch.zeros(DIM, device=DEVICE)

        best = max(candidates, key=lambda n: self._potential.get(n, 0.0))
        pred_voice = self.cells[best].voice
        ch, conf = self.codebook.decode_nearest(pred_voice)
        return ch, pred_voice.clone()

    # ------------------------------------------------------------------
    #  Step
    # ------------------------------------------------------------------

    def step(self, input_tensor: torch.Tensor, current_char: str, next_char: str):
        self.tick += 1

        # 1. Evaluate last prediction
        if self._last_pred_tensor is not None and self._last_target_char is not None:
            pred_char, _ = self.codebook.decode_nearest(self._last_pred_tensor)
            correct = (pred_char == self._last_target_char)
            self.accuracy_history.append(correct)
            self.phase_accuracy[self._current_phase].append(correct)
            self.char_accuracy[self._last_target_char].append(correct)

        # 2. Navigate
        path = self._navigate(input_tensor)
        if not path:
            if len(self.cells) < self.max_cells:
                root = self._spawn_root(input_tensor)
                path = [(root.name, 1.0)]
            else:
                valid_roots = [r for r in self.roots if r in self.cells]
                if valid_roots:
                    scores = self._batch_resonance(input_tensor, valid_roots)
                    best_idx = max(range(len(scores)), key=lambda i: scores[i])
                    path = [(valid_roots[best_idx], 0.0)]
                else:
                    return

        self._path_depths.append(len(path))
        active = {name for name, _ in path}

        # 3. Process path (crystal filter: project through voice direction)
        cell_outputs: dict[str, torch.Tensor] = {}
        for i, (cell_name, _) in enumerate(path):
            cell = self.cells[cell_name]
            old_voice = cell.voice.clone()

            if i == 0:
                cell_input = input_tensor
            else:
                parent_name = path[i - 1][0]
                cell_input = cell_outputs[parent_name]

            # Crystal filter: project input along FIXED birth axis
            # Fixed axis → consistent transformation → stable entropy → crystallization
            proj = torch.dot(cell_input, cell.axis) * cell.axis
            orth = cell_input - proj
            signal = proj + PHI_INV * orth

            # Voice update = the physics
            ev = self._get_entropy_var(cell_name)
            crystallization = max(0.0, min(1.0, 1.0 - ev / XI_SEC))
            decay = LAMBDA_STAR + (1.0 - LAMBDA_STAR) * crystallization
            cell.voice = decay * old_voice + (1.0 - decay) * signal
            cell_outputs[cell_name] = signal
            cell.access_count += 1

            # Track entropy
            entropy = self._compute_entropy(cell.voice)
            self._entropy_history[cell_name].append(entropy)

        # 4. Residual at leaf
        leaf_name = path[-1][0]
        leaf_output = cell_outputs[leaf_name]
        leaf_input = cell_outputs[path[-2][0]] if len(path) > 1 else input_tensor
        residual = leaf_input - leaf_output
        res_energy = float(torch.norm(residual))
        input_energy = float(torch.norm(leaf_input)) + 1e-8
        absorption = 1.0 - (res_energy / input_energy)
        self._residual_history.append(res_energy)

        # Track leaf residual for Landauer
        if res_energy > XI_SEC:
            if leaf_name not in self.leaf_residuals:
                self.leaf_residuals[leaf_name] = torch.zeros(DIM, device=DEVICE)
                self.leaf_residual_counts[leaf_name] = 0
            self.leaf_residuals[leaf_name] += (input_tensor - self.cells[leaf_name].voice)
            self.leaf_residual_counts[leaf_name] += 1

        # 5. Redistribute potential
        self._redistribute_potential(path, max(0.0, absorption))

        # 6. Predict
        pred_char, pred_tensor = self._predict_from_potential(path)
        self._last_pred_tensor = pred_tensor
        self._last_target_char = next_char

        # 7. Deferred growth
        if self._pending_parent is not None and len(self.cells) < self.max_cells:
            pp = self._pending_parent
            if pp.name in self.cells:
                pv_norm = float(torch.norm(pp.voice))
                if pv_norm > 1e-6:
                    trans_res = input_tensor - pp.voice
                    trans_energy = float(torch.norm(trans_res))
                    trans_abs = 1.0 - (trans_energy / (float(torch.norm(input_tensor)) + 1e-8))
                    if trans_abs < PHI_INV:
                        self._spawn_child(pp, trans_res.clone())
                    self._growth_checks += 1
                    if trans_abs < PHI_INV:
                        self._growth_hits += 1
                else:
                    self._growth_zero_voice += 1
            else:
                self._growth_dead_parent += 1
        elif self._pending_parent is None:
            self._growth_no_pending += 1

        self._pending_parent = self.cells.get(leaf_name)

        # 8. Dissipation (with birth grace period)
        for name in list(self.cells.keys()):
            if name in active:
                continue
            cell = self.cells[name]
            age = self.tick - cell.birth_tick
            if age < 50:  # Grace period: newborns don't dissipate
                continue
            ev = self._get_entropy_var(name)
            instability = min(1.0, ev / XI_SEC)
            eff_gamma = GAMMA * (1.0 + instability * 9.0)
            cell.voice *= (1.0 - eff_gamma)
            if name in self._potential:
                self._potential[name] *= (1.0 - eff_gamma)

        # 9. Dissolution
        dead = []
        for name, cell in self.cells.items():
            if name in active:
                continue
            if float(torch.norm(cell.voice)) < XI_SEC:
                dead.append(name)

        if len(dead) >= len(self.cells):
            dead = dead[:-1]

        for name in dead:
            cell = self.cells[name]
            # Reparent: adopt orphans into roots
            for child in cell.children:
                if cell.parent is not None and cell.parent.name in self.cells:
                    child.parent = cell.parent
                    cell.parent.children.append(child)
                else:
                    best_adopter, best_r = None, -1.0
                    for rname in self.roots:
                        if rname == name or rname in dead or rname == child.name:
                            continue
                        rcell = self.cells.get(rname)
                        if rcell is None:
                            continue
                        sim = float(F.cosine_similarity(
                            child.voice.unsqueeze(0), rcell.voice.unsqueeze(0)))
                        if sim > best_r:
                            best_r = sim
                            best_adopter = rcell
                    if best_adopter is not None and best_r > XI_SEC:
                        child.parent = best_adopter
                        best_adopter.children.append(child)
                    else:
                        child.parent = None
                        self.roots.append(child.name)

            if cell.parent is not None:
                cell.parent.children = [c for c in cell.parent.children if c.name != name]
            if name in self.roots:
                self.roots = [r for r in self.roots if r != name]

            del self.cells[name]
            self._entropy_history.pop(name, None)
            self._potential.pop(name, None)
            self._voice_matrix_dirty = True

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def set_phase(self, name: str):
        self._current_phase = name

    def get_max_depth(self) -> int:
        max_d = 0
        for cell in self.cells.values():
            d = 0
            p = cell.parent
            while p is not None:
                d += 1
                p = p.parent
            if d > max_d:
                max_d = d
        return max_d

    def phase_acc(self, phase: str) -> float:
        accs = self.phase_accuracy.get(phase, [])
        return sum(accs) / len(accs) if accs else 0.0

    def rolling_acc(self, window: int = 50) -> float:
        recent = self.accuracy_history[-window:]
        return sum(recent) / len(recent) if recent else 0.0


# ==========================================================================
#  Landauer Epoch Manager (fast version)
# ==========================================================================

def landauer_reinject(colony: FastCrystalColony):
    """Spawn informed children + soften entropy history."""
    # 1. Spawn children from leaf residuals
    scored = []
    for leaf_name, res_sum in colony.leaf_residuals.items():
        cell = colony.cells.get(leaf_name)
        if cell is None:
            continue
        ev = colony._get_entropy_var(leaf_name)
        if ev >= XI_SEC:
            continue
        count = colony.leaf_residual_counts.get(leaf_name, 1)
        res_energy = float(torch.norm(res_sum))
        scored.append((leaf_name, cell, res_sum, res_energy, count))

    scored.sort(key=lambda x: -x[3])
    spawned = 0
    available = colony.max_cells - len(colony.cells)

    for leaf_name, cell, res_sum, res_energy, count in scored:
        if spawned >= available:
            break
        mean_res = res_sum / max(1, count)
        norm = float(torch.norm(mean_res))
        if norm < XI_SEC:
            continue
        child_voice = (mean_res / (norm + 1e-8)) * math.sqrt(norm)
        colony._spawn_child(cell, child_voice.clone())
        spawned += 1

    # 2. Soften entropy history (phase transition)
    for name in list(colony._entropy_history.keys()):
        hist = colony._entropy_history[name]
        if len(hist) > 5:
            colony._entropy_history[name] = hist[-5:]

    # 3. Reset leaf tracking
    colony.leaf_residuals.clear()
    colony.leaf_residual_counts.clear()

    return spawned


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

CORPUS = HAMLET + GENESIS + PARADISE  # ~3500 chars


# ==========================================================================
#  Main
# ==========================================================================

def main():
    MAX_CELLS = 500
    N_EPOCHS = 12

    print("=" * 70)
    print("  SPIKE K CUDA -- Optimized Crystal Colony at Scale")
    print(f"  Device: {DEVICE}")
    print(f"  Corpus: {len(CORPUS)} chars | Epochs: {N_EPOCHS} | Max cells: {MAX_CELLS}")
    print("=" * 70)

    torch.manual_seed(42)
    codebook = FastCodebook()
    colony = FastCrystalColony(codebook, max_cells=MAX_CELLS)

    # Unique chars
    unique = sorted(set(CORPUS))
    print(f"  Unique characters: {len(unique)}")

    # Run epochs
    print(f"\n  {'Ep':>3s} | {'Acc':>6s} | {'Cells':>5s} | {'Roots':>3s} | {'Depth':>3s} | "
          f"{'Cryst':>5s} | {'Spawned':>7s} | {'Time':>6s} | {'ch/s':>6s}")
    print(f"  {'-'*3}-+-{'-'*6}-+-{'-'*5}-+-{'-'*3}-+-{'-'*3}-+-"
          f"{'-'*5}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}")

    total_start = time.time()
    epoch_accs = []

    for epoch in range(1, N_EPOCHS + 1):
        phase = f"epoch_{epoch}"
        colony.set_phase(phase)

        t0 = time.time()
        text = CORPUS
        for i in range(len(text) - 1):
            ch = text[i]
            nxt = text[i + 1]
            tensor = codebook.encode(ch)
            colony.step(tensor, ch, nxt)

        elapsed = time.time() - t0
        chars_per_sec = len(text) / elapsed if elapsed > 0 else 0
        acc = colony.phase_acc(phase)
        epoch_accs.append(acc)
        depth = colony.get_max_depth()

        # Landauer reinjection between epochs
        n_cryst = sum(1 for n in colony.cells if colony._get_entropy_var(n) < XI_SEC)
        cryst_pct = n_cryst / max(1, len(colony.cells))
        spawned = landauer_reinject(colony)

        print(
            f"  {epoch:3d} | "
            f"{acc:5.1%} | "
            f"{len(colony.cells):5d} | "
            f"{len(colony.roots):3d} | "
            f"{depth:3d} | "
            f"{cryst_pct:5.1%} | "
            f"{spawned:7d} | "
            f"{elapsed:5.1f}s | "
            f"{chars_per_sec:5.0f}"
        )

    total_elapsed = time.time() - total_start

    # Analysis
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")

    half = len(epoch_accs) // 2
    first_half = sum(epoch_accs[:half]) / half
    second_half = sum(epoch_accs[half:]) / (len(epoch_accs) - half)
    peak = max(epoch_accs)
    peak_ep = epoch_accs.index(peak) + 1

    print(f"  First half avg:  {first_half:.1%}")
    print(f"  Second half avg: {second_half:.1%}")
    print(f"  Peak:            {peak:.1%} (epoch {peak_ep})")
    print(f"  Colony learns:   {'YES' if second_half > first_half else 'NO'}")
    print(f"  Total time:      {total_elapsed:.1f}s")
    print(f"  Total ticks:     {colony.tick}")

    # Learning curve
    print(f"\n  LEARNING CURVE:")
    for i, acc in enumerate(epoch_accs):
        bar = "#" * int(acc * 200)
        print(f"    E{i+1:2d}: {acc:5.1%} | {bar}")

    # Character accuracy
    print(f"\n  CHARACTER ACCURACY (top 15):")
    char_accs = {}
    for ch, accs in colony.char_accuracy.items():
        if len(accs) >= 20:
            char_accs[ch] = sum(accs) / len(accs)
    best = sorted(char_accs.items(), key=lambda x: -x[1])[:15]
    for ch, acc in best:
        n = len(colony.char_accuracy[ch])
        bar = "#" * int(acc * 50)
        print(f"    '{ch}': {acc:5.0%} (n={n:5d}) {bar}")

    # Tree
    print(f"\n  FINAL TREE:")
    print(f"    Cells: {len(colony.cells)} / {MAX_CELLS}")
    print(f"    Roots: {len(colony.roots)}")
    print(f"    Max depth: {colony.get_max_depth()}")
    if colony._path_depths:
        mean_pd = sum(colony._path_depths) / len(colony._path_depths)
        pct_deep = sum(1 for d in colony._path_depths if d > 1) / len(colony._path_depths)
        print(f"    Mean path depth: {mean_pd:.1f}")
        print(f"    Deep paths (>1): {pct_deep:.1%}")

    n_cryst = sum(1 for n in colony.cells if colony._get_entropy_var(n) < XI_SEC)
    print(f"    Crystallized: {n_cryst}/{len(colony.cells)}")
    print(f"    Device: {DEVICE}")

    print(f"\n  GROWTH DIAGNOSTICS:")
    print(f"    Growth checks (pp alive, voice>0): {colony._growth_checks}")
    print(f"    Growth hits (trans_abs < PHI_INV):  {colony._growth_hits}")
    print(f"    Blocked: zero voice: {colony._growth_zero_voice}")
    print(f"    Blocked: dead parent: {colony._growth_dead_parent}")
    print(f"    Blocked: no pending: {colony._growth_no_pending}")
    print(f"    Root spawns: {colony._root_spawns}")
    print(f"    Child spawns: {colony._child_spawns}")
    deaths = colony.tick - len(colony.accuracy_history) - 1  # approximate
    print(f"    Net: spawned {colony._root_spawns + colony._child_spawns}, final {len(colony.cells)}")


if __name__ == "__main__":
    main()
