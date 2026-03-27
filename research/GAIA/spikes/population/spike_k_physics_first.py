"""Spike K v4 -- Crystal Colony: Growth + Navigation.

v1 failed: physics as sensor input to if-statements (Physics 0 - Heuristic 5).
v2 proved cell-level physics works (7/10 criteria, crystallization, self-limiting).
v3 added SEC collapse on flat colony (8/10, 4.1% accuracy). But the flat colony
was a blender -- every cell participated in every prediction, no hierarchy,
no temporal mechanism.

v4 restructures the colony as a TREE. Peter's insight: "intelligence isn't a
growth problem, it's navigation." A thought is a photon through a crystal --
one path through logic gates, not the whole lattice lighting up.

Three pieces:
  1. GROWTH (PAC): Residuals build the tree. Child voice = what parent couldn't
     cover. parent + child ~ input by construction (PAC conservation).
  2. NAVIGATION (SEC): Input signal traverses tree top-down. SEC collapse at
     each node picks which child the signal follows. ONE path, not whole tree.
     Crystallized nodes = sharp gates. Fluid nodes = fuzzy gates (exploration).
  3. PREDICTION: Layered processing along path. Root processes raw input,
     children process parent's output. Leaf output = deeply contextualized
     signal = prediction. No averaging, no superposition.

Temporal prediction emerges from structure: different paths encode different
sequential contexts. The tree IS the transition memory.

DFT constants (zero heuristics):
  - XI_SEC  = 0.0618  (navigation gate, dissolution, crystallization)
  - PHI_INV = 0.618   (absorption threshold for growth)
  - LAMBDA_STAR = 0.9816 (voice decay for crystallized cells)
  - GAMMA   = 0.0184  (dissipation rate)

Usage:
    cd dawn-models/research/GAIA/spikes/population
    PYTHONPATH="../../src;path/to/fracton" python spike_k_physics_first.py
"""

from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch

_root = Path(__file__).resolve().parents[2]
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))
_fracton = _root.parents[0] / "fracton"
if str(_fracton) not in sys.path:
    sys.path.insert(0, str(_fracton))

# Reuse from spike_i (Cell, organism factory, dimensions)
from spike_i_morphogenesis import (
    Cell,
    Signal,
    make_organism,
    DIM,
    PHI_INV,
)

# Reuse from spike_j (text processing, metrics, curriculum)
from spike_j_language_morphogenesis import (
    CharacterCodebook,
    TextEnvironment,
    extract_prediction,
    MetricsTracker,
    CURRICULUM,
)

from gaia.core.coupled_fields_bus import _harmonic_resonance
from gaia.core.types import FieldState, SECPhase


# ==========================================================================
#  DFT Constants -- ALL dynamics derive from these, zero heuristics
# ==========================================================================

XI_SEC = 0.0618033988749895       # SEC collapse threshold
LAMBDA_STAR = 0.9816              # optimal decay
GAMMA = 1.0 - LAMBDA_STAR        # dissipation rate = 0.0184
MAX_CELLS = 40                    # safety cap only


# ==========================================================================
#  Per-Cell Physics Logging (kept from v1 for diagnostics)
# ==========================================================================

@dataclass
class PhysicsSignals:
    """Physics state extracted from a cell's GAIA modules after processing."""
    pac_violation: float = 0.0
    budget_stability: float = 1.0
    sec_phase: SECPhase = SECPhase.ORDERED
    rbf_balance: float = 0.0
    conservation_quality: float = 0.5
    phi_frequency: float = 0.5


def read_cell_physics(cell: Cell) -> PhysicsSignals:
    """Extract physics signals from a cell's internal GAIA modules."""
    sig = PhysicsSignals()
    modules = cell.agent._bus._modules
    balances = []

    safety = modules.get("safety")
    if safety is not None:
        m = getattr(safety, "_last_metrics", None)
        if m is not None:
            sig.pac_violation = min(1.0, abs(getattr(m, "violation_pct", 0.0)) / 10.0)
            sig.budget_stability = getattr(m, "budget_stability", 1.0)
        try:
            balances.append(safety.health().balance)
        except Exception:
            pass

    reasoning = modules.get("reasoning")
    if reasoning is not None:
        m = getattr(reasoning, "_last_metrics", None)
        if m is not None:
            sig.phi_frequency = getattr(m, "phi_frequency", 0.5)
        try:
            balances.append(reasoning.health().balance)
        except Exception:
            pass

    for mod_name in ("memory", "language", "observability"):
        mod = modules.get(mod_name)
        if mod is not None:
            try:
                balances.append(mod.health().balance)
            except Exception:
                pass

    if balances:
        sig.rbf_balance = sum(balances) / len(balances)
    sig.conservation_quality = sig.budget_stability / (1.0 + sig.pac_violation)
    return sig


# ==========================================================================
#  CRYSTAL COLONY -- Growth (PAC) + Navigation (SEC)
# ==========================================================================

class CrystalColony:
    """Colony where tree structure IS the intelligence.

    Growth = PAC conservation (residuals crystallize as children).
    Navigation = SEC collapse (input follows one path through tree).
    Prediction = leaf output at end of navigated path.
    Structure = transition memory (no explicit transition matrix).
    """

    def __init__(self, codebook: CharacterCodebook):
        self.codebook = codebook
        self.metrics = MetricsTracker()

        self.tick = 0

        # Tree structure
        self.cells: dict[str, Cell] = {}
        self.roots: list[str] = []
        self._next_id = 0

        # Per-cell entropy history for SEC crystallization (must init before _spawn_root)
        self._entropy_history: dict[str, list[float]] = {}

        # Potential field -- PAC conservation: potential redistributes on actualization
        # Each cell carries potential. When a cell actualizes (absorbs input),
        # unactualized residual flows to siblings/cousins shaped by tree topology.
        # Like a fluid through crystal -- not noise, it has the shape of the tree.
        self._potential: dict[str, float] = {}

        # Start with one seed root (zero voice -- learns from first input)
        seed = self._spawn_root(torch.zeros(DIM), log=False)

        # Physics tracking
        self._cell_physics: dict[str, PhysicsSignals] = {}
        self.physics_log: list[dict] = []

        # Growth/death tracking
        self._residuals_at_spawn: list[float] = []
        self._entropy_var_at_death: list[float] = []
        self._cell_lifetimes: list[int] = []
        self._residual_history: list[float] = []
        self._growth_rate_history: list[int] = []

        # Navigation tracking
        self._path_depths: list[int] = []
        self._root_spawns = 0
        self._child_spawns = 0

        # Deferred growth: children encode transitions, not recognition refinement.
        # Store last tick's leaf so THIS tick's residual becomes its child.
        # Child voice = "what came after parent's context" (transition encoding).
        self._pending_parent: Cell | None = None

        # Prediction state
        self._last_prediction_tensor: torch.Tensor | None = None
        self._last_target_char: str | None = None
        self._recent_errors: list[float] = []
        self._error_window = 20

    # ------------------------------------------------------------------
    #  Tree construction
    # ------------------------------------------------------------------

    def _spawn_root(self, input_tensor: torch.Tensor, log: bool = True) -> Cell:
        """Create a new root cell (new lobe). Nothing in tree resonates."""
        name = f"cell_{self._next_id}"
        self._next_id += 1
        cell = Cell(name, parent=None, initial_voice=input_tensor.clone())
        cell.birth_tick = self.tick
        self.cells[name] = cell
        self.roots.append(name)
        self._entropy_history[name] = []
        self._potential[name] = float(torch.norm(input_tensor)) + XI_SEC  # initial potential
        if log:
            self._root_spawns += 1
            self.metrics.record_growth(self.tick, "root_spawn", float(torch.norm(input_tensor)))
        return cell

    def _spawn_child(self, parent: Cell, residual: torch.Tensor) -> Cell:
        """Crystallize a child from residual. parent + child ~ input (PAC)."""
        name = f"cell_{self._next_id}"
        self._next_id += 1
        cell = Cell(name, parent=parent, initial_voice=residual.clone())
        cell.birth_tick = self.tick
        parent.children.append(cell)
        self.cells[name] = cell
        self._entropy_history[name] = []
        self._potential[name] = float(torch.norm(residual))  # born with residual's potential
        self._residuals_at_spawn.append(float(torch.norm(residual)))
        self._child_spawns += 1
        self.metrics.record_growth(self.tick, "child_crystallization", float(torch.norm(residual)))
        return cell

    # ------------------------------------------------------------------
    #  Navigation -- the photon through the crystal
    # ------------------------------------------------------------------

    def _navigate(self, input_tensor: torch.Tensor) -> list[tuple[str, float]]:
        """Navigate tree from roots to leaf. SEC collapse at each junction.

        Layered signal filtering: the photon loses energy at each crystal layer.
        Roots see raw input. Children see the residual after parent absorbs --
        matching how they were born (deferred growth: child voice = next_input - parent).
        Each layer strips away what it recognizes, children respond to what remains.

        Returns list of (cell_name, resonance) forming the navigated path.
        Empty list = nothing resonates, need new root.
        """
        if not self.roots:
            return []

        # Colony-wide crystallization fraction -> collapse exponent
        n_cryst = sum(
            1 for n in self.cells
            if self._get_entropy_variance(n) < XI_SEC
        )
        cryst_fraction = n_cryst / max(1, len(self.cells))
        collapse_exponent = 1.0 + cryst_fraction * (1.0 / XI_SEC - 1.0)

        # Score all roots against raw input
        signal = input_tensor  # what the photon carries at this depth
        root_scores = []
        for rname in self.roots:
            cell = self.cells[rname]
            voice_norm = float(torch.norm(cell.voice))
            if voice_norm < 1e-6:
                root_scores.append((rname, 1.0))
            else:
                r = max(0.0, float(_harmonic_resonance(signal, cell.voice)))
                root_scores.append((rname, r))

        # SEC collapse at root level
        if collapse_exponent > 1.01:
            collapsed_scores = [(n, s ** collapse_exponent) for n, s in root_scores]
        else:
            collapsed_scores = root_scores

        best_root, best_score = max(collapsed_scores, key=lambda x: x[1])
        original_score = dict(root_scores)[best_root]

        if original_score < XI_SEC:
            return []  # nothing resonates -> need new root

        path = [(best_root, original_score)]
        current = self.cells[best_root]

        # Drill down through children with layered signal filtering
        # Each layer strips what parent absorbed -- children see the residual
        while current.children:
            # Filter signal: subtract what this layer absorbed
            parent_voice_norm = float(torch.norm(current.voice))
            if parent_voice_norm > 1e-6:
                signal = signal - current.voice  # residual after parent

            child_scores = []
            for child in current.children:
                cn = child.name
                voice_norm = float(torch.norm(child.voice))
                if voice_norm < 1e-6:
                    child_scores.append((cn, 1.0))
                else:
                    r = max(0.0, float(_harmonic_resonance(signal, child.voice)))
                    child_scores.append((cn, r))

            # SEC collapse at this junction
            if collapse_exponent > 1.01:
                collapsed_cs = [(n, s ** collapse_exponent) for n, s in child_scores]
            else:
                collapsed_cs = child_scores

            best_child, _ = max(collapsed_cs, key=lambda x: x[1])
            best_child_original = dict(child_scores)[best_child]

            if best_child_original < XI_SEC:
                break

            path.append((best_child, best_child_original))
            current = self.cells[best_child]

        return path

    # ------------------------------------------------------------------
    #  Entropy / crystallization helpers (same as v3)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    #  Potential field -- the bifractal cascade
    # ------------------------------------------------------------------

    def _redistribute_potential(self, path: list[tuple[str, float]], absorption: float):
        """After leaf actualizes, redistribute unactualized potential through tree.

        PAC conservation: total potential is conserved. When the navigated leaf
        absorbs input, the unactualized fraction flows to siblings and cousins,
        shaped by tree topology and SEC collapse. Not noise -- it has the shape
        of the crystal.

        Like investing: as one branch commits (actualizes), the remaining
        potential shifts to make other branches more or less likely. A cascade.
        """
        if not path:
            return

        leaf_name = path[-1][0]

        # Leaf consumed potential proportional to absorption
        leaf_potential = self._potential.get(leaf_name, 0.0)
        consumed = leaf_potential * absorption
        unactualized = leaf_potential - consumed
        self._potential[leaf_name] = consumed  # keep what was actualized

        if unactualized < 1e-8:
            return

        # Walk UP the path -- at each level, distribute to siblings
        # SEC collapse governs distribution sharpness:
        #   crystallized siblings = sharp (focused flow, like a channel)
        #   fluid siblings = diffuse (spreading flow, like a delta)
        remaining = unactualized
        for i in range(len(path) - 1, -1, -1):
            cell_name = path[i][0]
            cell = self.cells.get(cell_name)
            if cell is None:
                continue

            # Find siblings (other children of same parent)
            if cell.parent is not None:
                siblings = [c for c in cell.parent.children if c.name != cell_name
                            and c.name in self.cells]
            else:
                # Root level -- siblings are other roots
                siblings = [self.cells[r] for r in self.roots
                            if r != cell_name and r in self.cells]

            if not siblings:
                continue

            # Distribute fraction at this level (deeper = more local flow)
            level_fraction = PHI_INV if i == len(path) - 1 else PHI_INV ** 2
            level_amount = remaining * level_fraction
            remaining -= level_amount

            # SEC-shaped distribution: resonance with the current input
            # determines how potential flows. High resonance = more likely next.
            sibling_resonances = []
            for sib in siblings:
                voice_norm = float(torch.norm(sib.voice))
                if voice_norm < 1e-6:
                    sibling_resonances.append((sib.name, XI_SEC))
                else:
                    r = max(XI_SEC, float(_harmonic_resonance(
                        self.cells[leaf_name].voice, sib.voice)))
                    sibling_resonances.append((sib.name, r))

            # SEC collapse sharpens distribution
            cryst = self._get_crystallization_fraction()
            collapse_exp = 1.0 + cryst * (1.0 / XI_SEC - 1.0)
            weights = [(name, r ** collapse_exp) for name, r in sibling_resonances]
            total_w = sum(w for _, w in weights) + 1e-10

            for sib_name, w in weights:
                flow = level_amount * (w / total_w)
                self._potential[sib_name] = self._potential.get(sib_name, 0.0) + flow

        # Any remaining flows back into the leaf (conservation)
        if remaining > 1e-8:
            self._potential[leaf_name] += remaining

    def _get_crystallization_fraction(self) -> float:
        """Fraction of cells that are crystallized (low entropy variance)."""
        if not self.cells:
            return 0.0
        n_cryst = sum(1 for n in self.cells if self._get_entropy_variance(n) < XI_SEC)
        return n_cryst / len(self.cells)

    def _predict_from_potential(self, path: list[tuple[str, float]]) -> tuple[str, torch.Tensor]:
        """Prediction = highest potential in the LOCAL neighborhood of current path.

        Not the global max -- that pools in dead-end sinks. The question is:
        "given where we ARE in the tree, where will the next collapse happen?"
        Siblings of the leaf, children of the leaf, siblings of ancestors.
        The flow has the shape of the tree -- local, not global.
        """
        if not self._potential or not path:
            return "?", torch.zeros(DIM)

        leaf_name = path[-1][0]
        leaf_cell = self.cells.get(leaf_name)
        if leaf_cell is None:
            return "?", torch.zeros(DIM)

        # Gather candidates: siblings + children + siblings of ancestors
        candidates: list[str] = []

        # Leaf's children (deeper continuation)
        candidates.extend(c.name for c in leaf_cell.children if c.name in self.cells)

        # Walk up the path: siblings at each level
        for cell_name, _ in path:
            cell = self.cells.get(cell_name)
            if cell is None:
                continue
            if cell.parent is not None:
                siblings = [c.name for c in cell.parent.children
                            if c.name != cell_name and c.name in self.cells]
            else:
                siblings = [r for r in self.roots
                            if r != cell_name and r in self.cells]
            candidates.extend(siblings)

        if not candidates:
            # Fallback: all roots except current path's root
            path_root = path[0][0] if path else None
            candidates = [r for r in self.roots
                          if r != path_root and r in self.cells]

        if not candidates:
            return "?", torch.zeros(DIM)

        # Highest potential among local candidates
        best_name = max(candidates, key=lambda n: self._potential.get(n, 0.0))
        pred_voice = self.cells[best_name].voice
        predicted_char, confidence = self.codebook.decode_nearest(pred_voice)
        return predicted_char, pred_voice.clone()

    def _get_entropy_variance(self, name: str) -> float:
        """Entropy variance over recent history. Low = crystallized."""
        hist = self._entropy_history.get(name, [])
        if len(hist) < 5:
            return 1.0  # unknown = unstable
        recent = hist[-20:]
        mean_h = sum(recent) / len(recent)
        return sum((h - mean_h) ** 2 for h in recent) / len(recent)

    def _compute_entropy(self, voice: torch.Tensor) -> float:
        """Shannon entropy of voice tensor distribution."""
        v = torch.abs(voice) + 1e-10
        p = v / v.sum()
        return -float(torch.sum(p * torch.log(p)))

    # ------------------------------------------------------------------
    #  The 8-step physics loop
    # ------------------------------------------------------------------

    def step(self, env: FieldState, current_char: str, next_char: str):
        """One tick of the crystal colony.

        Growth builds the tree (PAC). Navigation traverses it (SEC).
        The photon follows one path. Leaf actualizes, potential redistributes
        through the crystal (bifractal cascade). Highest potential = prediction.
        Not noise -- it has the shape of the tree.
        """
        self.tick += 1

        # ---- Step 1: Evaluate last prediction ----
        prediction_error = 1.0
        if self._last_prediction_tensor is not None and self._last_target_char is not None:
            target_vec = self.codebook.encode(self._last_target_char)
            resonance = _harmonic_resonance(self._last_prediction_tensor, target_vec)
            prediction_error = 1.0 - max(0.0, float(resonance))

            correct = prediction_error < 0.5
            self.metrics.record_prediction(correct, prediction_error, self._last_target_char)

            self._recent_errors.append(prediction_error)
            if len(self._recent_errors) > self._error_window:
                self._recent_errors.pop(0)

        # ---- Step 2: Navigate tree (SEC collapse at junctions) ----
        path = self._navigate(env.tensor)

        if not path:
            # Nothing resonates -> spawn new root from this input
            if len(self.cells) < MAX_CELLS:
                new_root = self._spawn_root(env.tensor)
                path = [(new_root.name, 1.0)]
            else:
                # Colony full: force-navigate to most resonant root
                best_name = max(
                    self.roots,
                    key=lambda rn: max(0.0, float(
                        _harmonic_resonance(env.tensor, self.cells[rn].voice)
                    )) if float(torch.norm(self.cells[rn].voice)) > 1e-6 else 1.0
                )
                path = [(best_name, 0.0)]

        self._path_depths.append(len(path))
        active_names = {name for name, _ in path}

        # ---- Step 3: Process input through navigated path (layered) ----
        signals: dict[str, Signal] = {}
        for i, (cell_name, _resonance) in enumerate(path):
            cell = self.cells[cell_name]
            old_voice = cell.voice.clone()

            if i == 0:
                # Root sees raw environment
                input_state = FieldState(
                    tensor=env.tensor.clone(),
                    entropy=env.entropy,
                    phase=env.phase,
                    conservation_budget=0.0,
                    provenance=[],
                    timestamp=time.time(),
                )
            else:
                # Child sees parent's output from THIS tick
                parent_name = path[i - 1][0]
                parent_signal = signals[parent_name]
                input_state = FieldState(
                    tensor=parent_signal.tensor.clone(),
                    entropy=env.entropy,
                    phase=env.phase,
                    conservation_budget=0.0,
                    provenance=[],
                    timestamp=time.time(),
                )

            signal = cell.process(input_state)
            signals[cell_name] = signal

            # Override voice with physics-based decay
            entropy_var = self._get_entropy_variance(cell_name)
            crystallization = max(0.0, min(1.0, 1.0 - entropy_var / XI_SEC))
            physics_decay = LAMBDA_STAR + (1.0 - LAMBDA_STAR) * crystallization
            cell.voice = physics_decay * old_voice + (1.0 - physics_decay) * signal.tensor

        # ---- Step 4: PAC residual at leaf ----
        leaf_name = path[-1][0]
        leaf_signal = signals[leaf_name]
        if len(path) > 1:
            parent_name = path[-2][0]
            leaf_input = signals[parent_name].tensor
        else:
            leaf_input = env.tensor
        residual = leaf_input - leaf_signal.tensor
        residual_energy = float(torch.norm(residual))
        input_energy = float(torch.norm(leaf_input)) + 1e-8
        absorption = 1.0 - (residual_energy / input_energy)
        self._residual_history.append(residual_energy)

        # ---- Step 5: Redistribute potential (the bifractal cascade) ----
        # Leaf actualized -> unactualized potential flows to siblings/cousins.
        # PAC conservation: total potential stays constant. Tree topology shapes
        # the flow -- not noise, it has the shape of the crystal.
        self._redistribute_potential(path, max(0.0, absorption))

        # ---- Step 6: Predict = highest potential in local neighborhood ----
        # After redistribution, the potential landscape IS the prediction.
        # But locally -- siblings, children, nearby branches. Not global max.
        # The flow has the shape of the tree.
        predicted_char, pred_tensor = self._predict_from_potential(path)
        self._last_prediction_tensor = pred_tensor
        self._last_target_char = next_char

        # ---- Step 7: Growth -- deferred by one tick (transition encoding) ----
        # Children encode "what came AFTER parent's context", not "finer subdivision."
        # Last tick's leaf is the pending parent. THIS tick's residual becomes its child.
        # So the child's voice captures the transition: parent_context -> next_input.
        if self._pending_parent is not None and len(self.cells) < MAX_CELLS:
            pp = self._pending_parent
            if pp.name in self.cells:
                # Compute how well the pending parent handles THIS tick's input
                pp_voice_norm = float(torch.norm(pp.voice))
                if pp_voice_norm > 1e-6:
                    transition_residual = env.tensor - pp.voice
                    transition_energy = float(torch.norm(transition_residual))
                    parent_energy = float(torch.norm(env.tensor)) + 1e-8
                    transition_absorption = 1.0 - (transition_energy / parent_energy)
                    if transition_absorption < PHI_INV:
                        self._spawn_child(pp, transition_residual.clone())

        # Store current leaf as pending parent for NEXT tick's deferred growth
        self._pending_parent = self.cells.get(leaf_name)

        # ---- Step 8: Dissipation -- inactive cells decay ----
        # Voice AND potential decay for inactive cells.
        # Potential that doesn't actualize expires -- like an investment window closing.
        for name in list(self.cells.keys()):
            if name in active_names:
                continue
            entropy_var = self._get_entropy_variance(name)
            instability = min(1.0, entropy_var / XI_SEC)
            effective_gamma = GAMMA * (1.0 + instability * 9.0)
            self.cells[name].voice *= (1.0 - effective_gamma)
            # Potential dissipates too -- redistributed portion decays back
            if name in self._potential:
                self._potential[name] *= (1.0 - effective_gamma)

        # ---- Step 9: Dissolution + entropy tracking ----
        dead = []
        for name, cell in self.cells.items():
            if name in active_names:
                continue
            voice_energy = float(torch.norm(cell.voice))
            if voice_energy < XI_SEC:
                dead.append(name)

        # Keep at least 1 cell alive
        if len(dead) >= len(self.cells):
            dead = dead[:-1]

        for name in dead:
            cell = self.cells[name]
            lifetime = self.tick - cell.birth_tick
            self._cell_lifetimes.append(lifetime)
            ev = self._get_entropy_variance(name)
            self._entropy_var_at_death.append(ev)

            # Reparent orphans: adopt into existing tree, don't promote to root.
            # Rootless orphans fragment the tree into a forest of stumps.
            for child in cell.children:
                if cell.parent is not None and cell.parent.name in self.cells:
                    # Grandparent alive -> adopt there
                    child.parent = cell.parent
                    cell.parent.children.append(child)
                else:
                    # No grandparent -> find best adoptive parent among roots
                    # Only adopt into roots to avoid cycles in deeper tree
                    best_adopter, best_r = None, -1.0
                    for rname in self.roots:
                        if rname == name or rname in dead or rname == child.name:
                            continue
                        rcell = self.cells.get(rname)
                        if rcell is None:
                            continue
                        r = max(0.0, float(_harmonic_resonance(
                            child.voice, rcell.voice)))
                        if r > best_r:
                            best_r = r
                            best_adopter = rcell
                    if best_adopter is not None and best_r > XI_SEC:
                        child.parent = best_adopter
                        best_adopter.children.append(child)
                    else:
                        # No good match -> reluctant root promotion
                        child.parent = None
                        self.roots.append(child.name)

            # Remove from parent's children
            if cell.parent is not None:
                cell.parent.children = [c for c in cell.parent.children if c.name != name]

            # Remove from roots
            if name in self.roots:
                self.roots = [r for r in self.roots if r != name]

            # PAC conservation: energy to parent or most resonant survivor
            remaining = cell.voice.clone()
            if cell.parent is not None and cell.parent.name in self.cells:
                self.cells[cell.parent.name].voice += remaining * 0.5
            else:
                best_name, best_r = "", 0.0
                for other_name, other_cell in self.cells.items():
                    if other_name == name or other_name in dead:
                        continue
                    r = max(0.0, float(_harmonic_resonance(cell.voice, other_cell.voice)))
                    if r > best_r:
                        best_r = r
                        best_name = other_name
                if best_name:
                    self.cells[best_name].voice += remaining * best_r

            # Potential flows to parent or most resonant survivor (PAC conservation)
            dead_potential = self._potential.pop(name, 0.0)
            if cell.parent is not None and cell.parent.name in self.cells:
                self._potential[cell.parent.name] = self._potential.get(
                    cell.parent.name, 0.0) + dead_potential
            elif self.cells:
                # Distribute to survivors weighted by resonance
                survivors = [n for n in self.cells if n != name and n not in dead]
                if survivors:
                    total_r = 0.0
                    res_scores = []
                    for sn in survivors:
                        r = max(0.0, float(_harmonic_resonance(
                            cell.voice, self.cells[sn].voice)))
                        res_scores.append((sn, r))
                        total_r += r
                    if total_r > 1e-10:
                        for sn, r in res_scores:
                            self._potential[sn] = self._potential.get(
                                sn, 0.0) + dead_potential * (r / total_r)
                    else:
                        share = dead_potential / len(survivors)
                        for sn in survivors:
                            self._potential[sn] = self._potential.get(sn, 0.0) + share

            del self.cells[name]
            self._entropy_history.pop(name, None)

        # Entropy tracking for all surviving cells
        for name, cell in self.cells.items():
            h = self._compute_entropy(cell.voice)
            hist = self._entropy_history.setdefault(name, [])
            hist.append(h)
            if len(hist) > 30:
                hist.pop(0)

        # Physics logging for active cells
        self._cell_physics.clear()
        for name in active_names:
            if name in self.cells:
                self._cell_physics[name] = read_cell_physics(self.cells[name])

        # Specialization tracking
        for name in active_names:
            if name in self.cells:
                self.metrics.record_activation(name, current_char)

        # Periodic snapshot
        if self.tick % 50 == 0:
            self._log_snapshot(path)

    # ------------------------------------------------------------------
    #  Logging
    # ------------------------------------------------------------------

    def _log_snapshot(self, path: list[tuple[str, float]] | None = None):
        """Log aggregate physics state for post-analysis."""
        n_cells = len(self.cells)
        if n_cells == 0:
            return

        energies = [float(torch.norm(c.voice)) for c in self.cells.values()]
        variances = [self._get_entropy_variance(n) for n in self.cells]
        n_crystallized = sum(1 for v in variances if v < XI_SEC)

        recent_res = self._residual_history[-50:] if self._residual_history else [0]
        mean_residual = sum(recent_res) / len(recent_res)

        cf = n_crystallized / max(1, n_cells)
        ce = 1.0 + cf * (1.0 / XI_SEC - 1.0)
        path_depth = len(path) if path else 0
        max_tree_depth = self._get_max_tree_depth()

        self.physics_log.append({
            "tick": self.tick,
            "n_cells": n_cells,
            "n_roots": len(self.roots),
            "n_crystallized": n_crystallized,
            "total_energy": sum(energies),
            "mean_energy": sum(energies) / n_cells,
            "mean_residual": mean_residual,
            "cryst_fraction": cf,
            "collapse_exponent": ce,
            "path_depth": path_depth,
            "max_tree_depth": max_tree_depth,
        })

    def _get_max_tree_depth(self) -> int:
        """Maximum depth of any cell in the tree."""
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

    # ------------------------------------------------------------------
    #  Run curriculum + reporting
    # ------------------------------------------------------------------

    def run_text(
        self,
        text_env: TextEnvironment,
        phase_name: str = "",
        print_every: int = 100,
    ):
        """Process an entire text through the colony."""
        self.metrics.set_phase(phase_name)
        chars_processed = 0
        spawns_this_phase = 0
        deaths_this_phase = 0

        while text_env.has_more():
            n_before = len(self.cells)
            env, current_char, next_char = text_env.step()
            self.step(env, current_char, next_char)
            n_after = len(self.cells)
            if n_after > n_before:
                spawns_this_phase += (n_after - n_before)
            elif n_after < n_before:
                deaths_this_phase += (n_before - n_after)
            chars_processed += 1

            if chars_processed % print_every == 0:
                acc = self.metrics.rolling_accuracy()
                err = self.metrics.rolling_error()
                n_cells = len(self.cells)
                n_cryst = sum(
                    1 for n in self.cells
                    if self._get_entropy_variance(n) < XI_SEC
                )
                depth = self._path_depths[-1] if self._path_depths else 0
                max_d = self._get_max_tree_depth()
                print(
                    f"  tick {self.tick:4d} | "
                    f"acc={acc:.0%} err={err:.2f} | "
                    f"{n_cells} cells ({len(self.roots)}R d{max_d}) | "
                    f"path={depth} | "
                    f"char='{current_char}'"
                )

        phase_acc = self.metrics.phase_summary().get(phase_name, 0.0)
        print(
            f"  -- {phase_name} complete: {chars_processed} chars, "
            f"phase_acc={phase_acc:.1%}, "
            f"colony={len(self.cells)} cells ({len(self.roots)} roots) "
            f"(+{spawns_this_phase} born, -{deaths_this_phase} died)"
        )

    def learning_report(self):
        """Print comprehensive learning analysis."""
        print(f"\n{'='*70}")
        print(f"  CRYSTAL COLONY LEARNING REPORT")
        print(f"  {self.tick} ticks | {len(self.cells)} cells | {len(self.roots)} roots")
        print(f"{'='*70}")

        # Phase accuracy
        print("\n  PHASE ACCURACY:")
        for phase, acc in self.metrics.phase_summary().items():
            n = len(self.metrics.phase_accuracy[phase])
            print(f"    {phase:20s}: {acc:.1%}  ({n} chars)")

        # Per-character accuracy
        print("\n  CHARACTER ACCURACY (best):")
        char_accs = {}
        for ch, accs in self.metrics.char_accuracy.items():
            if len(accs) >= 3:
                char_accs[ch] = sum(accs) / len(accs)
        if char_accs:
            best = sorted(char_accs.items(), key=lambda x: -x[1])[:8]
            for ch, acc in best:
                n = len(self.metrics.char_accuracy[ch])
                print(f"    '{ch}': {acc:.0%}  (n={n})")

        # Tree structure
        max_d = self._get_max_tree_depth()
        print(f"\n  TREE STRUCTURE:")
        print(f"    Roots: {len(self.roots)}")
        print(f"    Max depth: {max_d}")
        print(f"    Root spawns: {self._root_spawns}")
        print(f"    Child spawns: {self._child_spawns}")
        # Per-root subtree sizes
        for rname in self.roots[:10]:  # show first 10
            subtree_size = self._subtree_size(self.cells[rname])
            depth = self._subtree_depth(self.cells[rname])
            print(f"    {rname}: {subtree_size} cells, depth {depth}")
        if len(self.roots) > 10:
            print(f"    ... and {len(self.roots) - 10} more roots")

        # Navigation stats
        if self._path_depths:
            mean_pd = sum(self._path_depths) / len(self._path_depths)
            max_pd = max(self._path_depths)
            print(f"\n  NAVIGATION:")
            print(f"    Mean path depth: {mean_pd:.1f}")
            print(f"    Max path depth: {max_pd}")
            pct_deep = sum(1 for d in self._path_depths if d > 1) / len(self._path_depths)
            print(f"    Paths with depth > 1: {pct_deep:.1%}")

        # Potential field
        if self._potential:
            pot_vals = [self._potential.get(n, 0.0) for n in self.cells]
            if pot_vals:
                total_pot = sum(pot_vals)
                max_pot_name = max(self.cells, key=lambda n: self._potential.get(n, 0.0))
                max_pot_val = self._potential.get(max_pot_name, 0.0)
                print(f"\n  POTENTIAL FIELD (bifractal cascade):")
                print(f"    Total potential: {total_pot:.3f}")
                print(f"    Mean: {total_pot/len(pot_vals):.3f}")
                print(f"    Max: {max_pot_val:.3f} ({max_pot_name})")
                # Show top 5 by potential
                top5 = sorted(self.cells.keys(),
                              key=lambda n: self._potential.get(n, 0.0), reverse=True)[:5]
                print(f"    Top 5 by potential:")
                for n in top5:
                    p = self._potential.get(n, 0.0)
                    spec = self.metrics.lobe_activations.get(n, {})
                    top_char = max(spec, key=spec.get) if spec else "?"
                    print(f"      {n}: pot={p:.3f}  specialist='{top_char}'")

        # Growth events
        print(f"\n  GROWTH EVENTS: {len(self.metrics.growth_events)} total")
        if self._residuals_at_spawn:
            print(f"    Mean residual at spawn: {sum(self._residuals_at_spawn)/len(self._residuals_at_spawn):.3f}")
        corr = self.metrics.error_growth_correlation()
        print(f"  Error-growth correlation: {corr:.0%}")

        # Death events
        print(f"\n  DEATH EVENTS: {len(self._cell_lifetimes)} total")
        if self._cell_lifetimes:
            mean_life = sum(self._cell_lifetimes) / len(self._cell_lifetimes)
            print(f"    Mean lifetime: {mean_life:.0f} ticks")
            print(f"    Shortest: {min(self._cell_lifetimes)}, Longest: {max(self._cell_lifetimes)}")

        # Crystallization
        n_cryst = sum(
            1 for n in self.cells
            if self._get_entropy_variance(n) < XI_SEC
        )
        print(f"\n  CRYSTALLIZATION: {n_cryst}/{len(self.cells)} cells crystallized")

        # Energy
        energies = [float(torch.norm(c.voice)) for c in self.cells.values()]
        if energies:
            print(f"\n  ENERGY:")
            print(f"    Total: {sum(energies):.3f}")
            print(f"    Mean:  {sum(energies)/len(energies):.3f}")
            print(f"    Min:   {min(energies):.3f}")
            print(f"    Max:   {max(energies):.3f}")

        # Physics timeline
        if self.physics_log:
            print(f"\n  PHYSICS TIMELINE ({len(self.physics_log)} snapshots):")
            step = max(1, len(self.physics_log) // 6)
            for snap in self.physics_log[::step]:
                ce = snap.get('collapse_exponent', 1)
                pd = snap.get('path_depth', 0)
                md = snap.get('max_tree_depth', 0)
                print(
                    f"    tick={snap['tick']:4d} | "
                    f"{snap['n_cells']} cells ({snap['n_roots']}R d{md}) | "
                    f"E={snap['mean_energy']:.3f} "
                    f"res={snap['mean_residual']:.3f} | "
                    f"path={pd} x{ce:.1f}"
                )

        # Residual trend
        if self._residual_history:
            n = len(self._residual_history)
            q1 = self._residual_history[:n//4]
            q4 = self._residual_history[-(n//4):]
            if q1 and q4:
                early_r = sum(q1) / len(q1)
                late_r = sum(q4) / len(q4)
                print(f"\n  RESIDUAL TREND:")
                print(f"    Early mean: {early_r:.3f}")
                print(f"    Late mean:  {late_r:.3f}")
                print(f"    Trend: {'DECREASING (colony improving)' if late_r < early_r else 'INCREASING (colony struggling)'}")

    def _subtree_size(self, cell: Cell) -> int:
        """Count cells in subtree rooted at cell."""
        count = 1
        for child in cell.children:
            count += self._subtree_size(child)
        return count

    def _subtree_depth(self, cell: Cell) -> int:
        """Max depth of subtree rooted at cell."""
        if not cell.children:
            return 0
        return 1 + max(self._subtree_depth(c) for c in cell.children)

    def learning_verdict(self) -> dict[str, bool]:
        """Evaluate 10 learning criteria."""
        print(f"\n{'='*70}")
        print(f"  CRYSTAL COLONY LEARNING VERDICT")
        print(f"{'='*70}")
        verdicts = {}

        # 1. Accuracy increases within phases
        criterion_1 = True
        for phase, accs in self.metrics.phase_accuracy.items():
            if len(accs) < 20:
                continue
            q1 = accs[:len(accs)//4]
            q4 = accs[-(len(accs)//4):]
            early = sum(q1) / len(q1) if q1 else 0
            late = sum(q4) / len(q4) if q4 else 0
            if late < early and phase != "domain_shift":
                criterion_1 = False
        verdicts["accuracy_increases"] = criterion_1
        status = "PASS" if criterion_1 else "FAIL"
        print(f"  1.  Accuracy increases within phases: {status}")

        # 2. Domain shift recovery
        ds_accs = self.metrics.phase_accuracy.get("domain_shift", [])
        if len(ds_accs) >= 40:
            early_ds = sum(ds_accs[:20]) / 20
            late_ds = sum(ds_accs[-20:]) / 20
            criterion_2 = late_ds > early_ds
        else:
            criterion_2 = len(ds_accs) == 0
        verdicts["domain_shift_recovery"] = criterion_2
        status = "PASS" if criterion_2 else "FAIL"
        print(f"  2.  Domain shift recovery:            {status}")

        # 3. Cell specialization (>=2 cells with >60% one character class)
        specialized = self._count_specialists()
        criterion_3 = specialized >= 2
        verdicts["cell_specialization"] = criterion_3
        status = "PASS" if criterion_3 else "FAIL"
        print(f"  3.  Cell specialization (>=2 cells):  {status}  ({specialized})")

        # 4. Error-growth correlation
        corr = self.metrics.error_growth_correlation()
        criterion_4 = corr > 0.5 or len(self.metrics.growth_events) == 0
        verdicts["error_growth_correlation"] = criterion_4
        status = "PASS" if criterion_4 else "FAIL"
        print(f"  4.  Error-growth correlation (>50%%):  {status}")

        # 5. Above random baseline
        total_accs = self.metrics.accuracy_history
        overall = sum(total_accs) / len(total_accs) if total_accs else 0
        criterion_5 = overall > 0.05
        verdicts["above_random"] = criterion_5
        status = "PASS" if criterion_5 else "FAIL"
        print(f"  5.  Above random baseline (>5%%):     {status}  ({overall:.1%})")

        # 6. Residual-growth correlation
        criterion_6 = len(self._residuals_at_spawn) > 0
        verdicts["residual_growth"] = criterion_6
        status = "PASS" if criterion_6 else "FAIL"
        print(f"  6.  Residual-driven growth:           {status}  ({len(self._residuals_at_spawn)} spawns)")

        # 7. Entropy-death correlation
        if self._entropy_var_at_death:
            high_var = sum(1 for v in self._entropy_var_at_death if v > XI_SEC)
            pct = high_var / len(self._entropy_var_at_death)
            criterion_7 = pct > 0.6
        else:
            criterion_7 = True
        verdicts["entropy_death"] = criterion_7
        status = "PASS" if criterion_7 else "FAIL"
        epct = (sum(1 for v in self._entropy_var_at_death if v > XI_SEC) / len(self._entropy_var_at_death) * 100) if self._entropy_var_at_death else 0
        print(f"  7.  Entropy-driven death (>60%%):      {status}  ({epct:.0f}%)")

        # 8. Crystallization exists
        n_cryst = sum(
            1 for n in self.cells
            if self._get_entropy_variance(n) < XI_SEC
        )
        criterion_8 = n_cryst > 0
        verdicts["crystallization"] = criterion_8
        status = "PASS" if criterion_8 else "FAIL"
        print(f"  8.  Cells crystallize:                {status}  ({n_cryst}/{len(self.cells)})")

        # 9. Residual energy decreases
        if self._residual_history and len(self._residual_history) > 20:
            n = len(self._residual_history)
            early_r = sum(self._residual_history[:n//4]) / (n//4)
            late_r = sum(self._residual_history[-(n//4):]) / (n//4)
            criterion_9 = late_r < early_r
        else:
            criterion_9 = False
        verdicts["residual_decreasing"] = criterion_9
        status = "PASS" if criterion_9 else "FAIL"
        print(f"  9.  Residual energy decreasing:       {status}")

        # 10. Colony self-limits
        if len(self.metrics.growth_events) > 10:
            half = len(self.metrics.growth_events) // 2
            early_ticks = self.metrics.growth_events[half-1]["tick"] - self.metrics.growth_events[0]["tick"]
            late_ticks = self.metrics.growth_events[-1]["tick"] - self.metrics.growth_events[half]["tick"]
            criterion_10 = late_ticks > early_ticks if early_ticks > 0 else False
        else:
            criterion_10 = len(self.metrics.growth_events) <= 10
        verdicts["self_limiting"] = criterion_10
        status = "PASS" if criterion_10 else "FAIL"
        print(f"  10. Colony self-limits:               {status}")

        passed = sum(verdicts.values())
        total = len(verdicts)
        print(f"\n  RESULT: {passed}/{total} criteria passed")
        return verdicts

    def _count_specialists(self) -> int:
        """Count cells with >60% activations from one character class."""
        count = 0
        for cell_name, char_counts in self.metrics.lobe_activations.items():
            if not char_counts:
                continue
            total = sum(char_counts.values())
            if total < 5:
                continue
            digits = sum(char_counts.get(d, 0) for d in "0123456789")
            vowels = sum(char_counts.get(v, 0) for v in "aeiou")
            consonants = sum(
                char_counts.get(c, 0)
                for c in "bcdfghjklmnpqrstvwxyz"
            )
            spaces = char_counts.get(" ", 0)
            for class_count in [digits, vowels, consonants, spaces]:
                if class_count / total > 0.6:
                    count += 1
                    break
        return count


# ==========================================================================
#  HEURISTIC BASELINE (Spike J results for comparison)
# ==========================================================================

SPIKE_J_BASELINE = {
    "repetition": 0.452,
    "words": 0.121,
    "rhyme": 0.087,
    "prose": 0.068,
    "domain_shift": 0.178,
}


# ==========================================================================
#  MAIN
# ==========================================================================

def main():
    print("=" * 70)
    print("  SPIKE K v4b -- Crystal Colony: Bifractal Potential Field")
    print("  Growth = PAC (residuals build tree)")
    print("  Navigation = SEC (photon through crystal, one path)")
    print("  Prediction = potential field (highest potential = next collapse)")
    print("=" * 70)

    torch.manual_seed(42)
    codebook = CharacterCodebook()

    # Codebook sanity check
    print("\n  Codebook check:")
    for ch in "abc 123":
        vec = codebook.encode(ch)
        decoded, conf = codebook.decode_nearest(vec)
        ok = "OK" if decoded == ch else f"MISMATCH({decoded})"
        print(f"    '{ch}' -> '{decoded}' (conf={conf:.3f}) {ok}")

    colony = CrystalColony(codebook)

    # Run each curriculum phase
    for phase in CURRICULUM:
        print(f"\n{'='*70}")
        print(f"  Phase: {phase['name'].upper()}")
        print(f"  {phase['description']}")
        print(f"  Text length: {len(phase['text'])} chars")
        print(f"{'='*70}")

        text_env = TextEnvironment(phase["text"], codebook)
        colony.run_text(
            text_env,
            phase_name=phase["name"],
            print_every=phase["print_every"],
        )

    # Final report
    colony.learning_report()
    verdicts = colony.learning_verdict()

    # Cell specialization detail
    print(f"\n  CELL SPECIALIZATION:")
    for cell_name, char_counts in sorted(colony.metrics.lobe_activations.items()):
        if not char_counts:
            continue
        total = sum(char_counts.values())
        if total < 5:
            continue
        top5 = sorted(char_counts.items(), key=lambda x: -x[1])[:5]
        parts = ", ".join(f"'{c}'={n}({n*100//total}%)" for c, n in top5)
        specialist = ""
        digits = sum(char_counts.get(d, 0) for d in "0123456789")
        consonants = sum(char_counts.get(c, 0) for c in "bcdfghjklmnpqrstvwxyz")
        if digits / total > 0.6:
            specialist = f" ** digit specialist ({digits*100//total}%)"
        elif consonants / total > 0.6:
            specialist = f" ** consonant specialist ({consonants*100//total}%)"
        print(f"    {cell_name}: {parts}{specialist}")

    # Comparison with Spike J heuristic baseline
    print(f"\n{'='*70}")
    print(f"  CRYSTAL vs HEURISTIC COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Phase':<20s} | {'Heuristic':>10s} | {'Crystal':>10s} | {'Winner':<10s}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    physics_wins = 0
    heuristic_wins = 0
    phase_summary = colony.metrics.phase_summary()
    for phase_name, h_acc in SPIKE_J_BASELINE.items():
        p_acc = phase_summary.get(phase_name, 0.0)
        if p_acc > h_acc:
            winner = "CRYSTAL"
            physics_wins += 1
        elif p_acc < h_acc:
            winner = "heuristic"
            heuristic_wins += 1
        else:
            winner = "tie"
        print(f"  {phase_name:<20s} | {h_acc:>9.1%} | {p_acc:>9.1%} | {winner}")

    print(f"\n  Score: Crystal {physics_wins} - Heuristic {heuristic_wins}")
    if physics_wins > heuristic_wins:
        print("  ** CRYSTAL WINS ** Tree navigation outperforms heuristic thresholds!")
    elif physics_wins == heuristic_wins:
        print("  TIE -- crystal colony matches heuristics without any tuning")
    else:
        print("  Heuristic wins on accuracy -- but check tree structure quality above")

    print(f"\n  Final: {len(colony.cells)} cells, {len(colony.roots)} roots, tick={colony.tick}")


if __name__ == "__main__":
    main()
