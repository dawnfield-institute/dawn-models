"""Spike K Landauer -- Epochs as Entropy Reinjection.

The key insight: an epoch isn't "replay the data." It's a Landauer erasure cascade.

P = A + ξ + Θ
  - A = actualized (what the colony absorbed)
  - ξ = structure (crystallized cells -- these SURVIVE)
  - Θ = thermal (dissipated energy -- NOT waste, it's FUEL for the next iteration)

Each epoch:
  1. Run text through colony (PAC conservation within epoch)
  2. Collect Θ (total dissipated energy during epoch)
  3. REINJECT Θ as new potential into fluid (uncrystallized) cells
  4. Crystallized structure stays. Fluid regions get new potential.
  5. New potential enables new children to grow ON TOP of existing crystal.
  6. Those children crystallize, deepening the tree.
  7. Remaining Θ feeds the NEXT epoch.

Like protein folding: each iteration adds structure on top of what's stable.
Like DNA: the backbone holds while new bases pair.

DFT predicts:
  - ~8.5 self-sustaining generations before depletion
  - Structure amplification: 53x over single pass
  - A/(A+ξ) ≈ ln(φ) = 0.4812
  - Depth bounded by floor(φ²) = 2 per iteration

Usage:
    cd dawn-models/research/GAIA/spikes/population
    PYTHONPATH="../../src;path/to/fracton" python spike_k_landauer.py
"""

from __future__ import annotations

import math
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

import torch

_root = Path(__file__).resolve().parents[2]
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))
_fracton = _root.parents[0] / "fracton"
if str(_fracton) not in sys.path:
    sys.path.insert(0, str(_fracton))

from spike_j_language_morphogenesis import (
    CharacterCodebook,
    TextEnvironment,
    MetricsTracker,
)
from spike_k_physics_first import (
    CrystalColony,
    XI_SEC,
    PHI_INV,
    LAMBDA_STAR,
    GAMMA,
    _harmonic_resonance,
    DIM,
)
from gaia.core.types import FieldState, SECPhase

# DFT cascade constants
LN_PHI = math.log((1 + math.sqrt(5)) / 2)  # 0.4812... actualization ratio
PHI_SQ_FLOOR = 2  # max depth gain per iteration: floor(φ²)


# ==========================================================================
#  Landauer Epoch Manager
# ==========================================================================

class LandauerEpochManager:
    """Manages the Landauer erasure cascade across epochs.

    Tracks energy flow: P → A + ξ + Θ at each epoch.
    Reinjects Θ as new potential for the next epoch.
    Monitors cascade sustainability (Θ must stay above threshold).
    """

    def __init__(self, colony: CrystalColony):
        self.colony = colony
        self.epoch_history: list[dict] = []
        # Per-leaf accumulated residuals across an epoch
        # Tracks what each leaf COULDN'T handle -- shapes the next generation
        self._leaf_residuals: dict[str, torch.Tensor] = {}
        self._leaf_residual_counts: dict[str, int] = {}

    def measure_structure(self) -> tuple[float, float, float]:
        """Partition colony energy into ξ (crystallized) and fluid components.

        Returns (xi_energy, fluid_energy, total_energy)
        """
        xi_energy = 0.0  # crystallized structure
        fluid_energy = 0.0  # uncrystallized

        for name, cell in self.colony.cells.items():
            e = float(torch.norm(cell.voice))
            ev = self.colony._get_entropy_variance(name)
            if ev < XI_SEC:
                xi_energy += e
            else:
                fluid_energy += e

        return xi_energy, fluid_energy, xi_energy + fluid_energy

    def collect_theta(self) -> float:
        """Measure total dissipated energy during epoch.

        Θ = energy that left the system through GAMMA dissipation.
        In the code: inactive cells lose (effective_gamma * voice_energy) per tick.
        We track cumulative dissipation.
        """
        # Use residual history as proxy: total residual energy = what wasn't absorbed
        # This IS the thermal component -- energy that passed through without crystallizing
        if not self.colony._residual_history:
            return 0.0

        # Take recent epoch's residuals
        last_epoch_len = self.epoch_history[-1]["n_chars"] if self.epoch_history else len(self.colony._residual_history)
        recent = self.colony._residual_history[-last_epoch_len:]
        return sum(recent)

    def reinject_theta(self, theta: float):
        """Reinject thermal energy by SPAWNING INFORMED CHILDREN on crystallized leaves.

        Like protein folding: the stable backbone (crystallized cells) is the
        scaffold for the NEXT layer. Children's voices are shaped by the epoch's
        ACTUAL RESIDUAL PATTERN -- what each leaf couldn't handle.

        Theta is not random noise. It carries the shape of what was lost.
        Children literally represent "what this leaf kept getting wrong."

        DFT: depth bounded by floor(phi^2) = 2 per iteration.
        Spawn budget: proportional to accumulated residual energy.
        """
        if not self._leaf_residuals:
            return 0.0

        import spike_k_physics_first as skp

        available_slots = skp.MAX_CELLS - len(self.colony.cells)
        if available_slots <= 0:
            return 0.0

        # Score leaves by accumulated residual energy (how much they struggled)
        scored = []
        for leaf_name, residual_sum in self._leaf_residuals.items():
            cell = self.colony.cells.get(leaf_name)
            if cell is None:
                continue
            # Must be crystallized (stable backbone for new growth)
            ev = self.colony._get_entropy_variance(leaf_name)
            if ev >= XI_SEC:
                continue
            res_energy = float(torch.norm(residual_sum))
            count = self._leaf_residual_counts.get(leaf_name, 1)
            scored.append((leaf_name, cell, residual_sum, res_energy, count))

        if not scored:
            return 0.0

        # Sort by residual energy (highest need first)
        scored.sort(key=lambda x: -x[3])

        # Spawn children: voice = mean residual direction at that leaf
        # This is EXACTLY what the leaf couldn't capture -- the missing structure
        spawned = 0
        injected = 0.0
        max_per_leaf = PHI_SQ_FLOOR  # depth gain bounded by DFT

        for leaf_name, cell, residual_sum, res_energy, count in scored:
            if spawned >= available_slots:
                break

            # Mean residual direction = what this leaf consistently missed
            mean_residual = residual_sum / max(1, count)
            mean_res_norm = float(torch.norm(mean_residual))
            if mean_res_norm < XI_SEC:
                continue  # leaf handled its inputs well, no child needed

            # Normalize to unit direction, scale by sqrt(residual) for stability
            child_voice = mean_residual / (mean_res_norm + 1e-8)
            child_voice = child_voice * math.sqrt(mean_res_norm)

            # Spawn using colony's mechanism
            new_cell = self.colony._spawn_child(cell, child_voice.clone())
            spawned += 1
            injected += float(torch.norm(child_voice))

            # If residual has high variance (leaf missed DIFFERENT things),
            # spawn a second child from the orthogonal component
            if max_per_leaf > 1 and spawned < available_slots and count > 10:
                # Variance in residual: if it changes a lot, two children needed
                orth = residual_sum - torch.dot(residual_sum, mean_residual) / (mean_res_norm ** 2 + 1e-8) * mean_residual
                orth_norm = float(torch.norm(orth))
                if orth_norm > XI_SEC * count:
                    orth_voice = orth / (orth_norm + 1e-8) * math.sqrt(orth_norm / count)
                    new_cell2 = self.colony._spawn_child(cell, orth_voice.clone())
                    spawned += 1
                    injected += float(torch.norm(orth_voice))

        return injected

    def _track_leaf_residual(self, leaf_name: str, residual: torch.Tensor):
        """Accumulate residual for a leaf across the epoch."""
        if leaf_name not in self._leaf_residuals:
            self._leaf_residuals[leaf_name] = torch.zeros(DIM)
            self._leaf_residual_counts[leaf_name] = 0
        self._leaf_residuals[leaf_name] += residual
        self._leaf_residual_counts[leaf_name] += 1

    def run_epoch(self, text: str, codebook: CharacterCodebook, epoch: int) -> dict:
        """Run one Landauer epoch: process text, then reinject Theta."""

        # Reset per-epoch residual tracking
        self._leaf_residuals.clear()
        self._leaf_residual_counts.clear()

        # Measure pre-epoch state
        xi_pre, fluid_pre, total_pre = self.measure_structure()
        n_cells_pre = len(self.colony.cells)
        depth_pre = self.colony._get_max_tree_depth()
        residual_start = len(self.colony._residual_history)

        # Monkey-patch colony step to capture per-leaf residuals
        original_step = self.colony.step.__func__

        manager = self

        def tracked_step(colony_self, env, current_char, next_char):
            original_step(colony_self, env, current_char, next_char)
            # After step, capture leaf residual from the residual history
            if colony_self._residual_history and colony_self._path_depths:
                # The last path depth tells us navigation happened
                # Reconstruct leaf name from last navigation (stored in pending parent)
                if colony_self._pending_parent is not None:
                    leaf_name = colony_self._pending_parent.name
                    # Residual direction: approximate from voice vs last input
                    res_energy = colony_self._residual_history[-1]
                    if res_energy > XI_SEC:
                        # Use env tensor - leaf voice as residual direction
                        leaf = colony_self.cells.get(leaf_name)
                        if leaf is not None:
                            residual = env.tensor - leaf.voice
                            manager._track_leaf_residual(leaf_name, residual)

        import types
        self.colony.step = types.MethodType(tracked_step, self.colony)

        # Run text through colony
        text_env = TextEnvironment(text, codebook)
        phase_name = f"epoch_{epoch}"
        self.colony.metrics.set_phase(phase_name)

        n_chars = 0
        while text_env.has_more():
            env, current_char, next_char = text_env.step()
            self.colony.step(env, current_char, next_char)
            n_chars += 1

        # Restore original step
        self.colony.step = types.MethodType(original_step, self.colony)

        # Measure post-epoch state
        xi_post, fluid_post, total_post = self.measure_structure()
        depth_post = self.colony._get_max_tree_depth()

        # Compute PAC partition: P = A + ξ + Θ
        # A = change in actualized energy (voice norms of active cells)
        # ξ = crystallized energy (preserved)
        # Θ = dissipated (residuals that weren't absorbed)
        epoch_residuals = self.colony._residual_history[residual_start:]
        theta = sum(epoch_residuals) if epoch_residuals else 0.0

        # Compute accuracy
        phase_accs = self.colony.metrics.phase_accuracy.get(phase_name, [])
        accuracy = sum(phase_accs) / len(phase_accs) if phase_accs else 0.0

        n_cryst = sum(1 for n in self.colony.cells
                      if self.colony._get_entropy_variance(n) < XI_SEC)

        stats = {
            "epoch": epoch,
            "n_chars": n_chars,
            "accuracy": accuracy,
            "cells": len(self.colony.cells),
            "roots": len(self.colony.roots),
            "max_depth": depth_post,
            "depth_gain": depth_post - depth_pre,
            "crystallized": n_cryst,
            "xi_energy": xi_post,
            "fluid_energy": fluid_post,
            "theta": theta,
            "theta_per_char": theta / max(1, n_chars),
            "xi_ratio": xi_post / max(1e-8, xi_post + fluid_post),
            "tick": self.colony.tick,
        }

        self.epoch_history.append(stats)

        # --- LANDAUER REINJECTION ---
        # Two-phase reinjection:
        # 1. Spawn informed children on crystallized leaves (structure growth)
        # 2. Soften navigation gates (entropy history partial reset)
        #    Voices (structure) preserved. Entropy history shortened so
        #    SEC collapse exponent drops, letting new children compete.
        #    Re-crystallization happens naturally within ~100 ticks if stable.
        injected = self.reinject_theta(theta)
        stats["injected"] = injected if injected else 0.0

        # Phase transition: partially reset entropy history to soften gates
        # Keep last 5 entries (don't fully erase -- structure memory persists)
        # This lowers crystallization fraction -> navigation becomes fluid ->
        # new children can be reached -> they crystallize -> deeper tree
        for name in list(self.colony._entropy_history.keys()):
            hist = self.colony._entropy_history[name]
            if len(hist) > 5:
                self.colony._entropy_history[name] = hist[-5:]

        return stats


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

CORPUS = HAMLET + GENESIS


# ==========================================================================
#  Main
# ==========================================================================

def main():
    MAX_CELLS = 300
    N_EPOCHS = 6  # DFT predicts ~8.5 self-sustaining generations

    print("=" * 70)
    print("  SPIKE K LANDAUER -- Epochs as Entropy Reinjection")
    print(f"  P = A + xi + Theta  (Theta reinjects as fuel for next epoch)")
    print(f"  Corpus: {len(CORPUS)} chars | Epochs: {N_EPOCHS} | Max cells: {MAX_CELLS}")
    print(f"  Cascade constants: ln(phi)={LN_PHI:.4f}, floor(phi^2)={PHI_SQ_FLOOR}")
    print("=" * 70)

    torch.manual_seed(42)
    codebook = CharacterCodebook()

    import spike_k_physics_first as skp
    skp.MAX_CELLS = MAX_CELLS

    colony = CrystalColony(codebook)
    manager = LandauerEpochManager(colony)

    # Header
    print(f"\n  {'Ep':>3s} | {'Acc':>6s} | {'Cells':>5s} | {'Roots':>3s} | {'Depth':>3s} | "
          f"{'Cryst':>5s} | {'ξ ratio':>7s} | {'Θ':>8s} | {'Inject':>8s} | {'Tick':>6s}")
    print(f"  {'-'*3}-+-{'-'*6}-+-{'-'*5}-+-{'-'*3}-+-{'-'*3}-+-"
          f"{'-'*5}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")

    for epoch in range(1, N_EPOCHS + 1):
        stats = manager.run_epoch(CORPUS, codebook, epoch)
        print(
            f"  {stats['epoch']:3d} | "
            f"{stats['accuracy']:5.1%} | "
            f"{stats['cells']:5d} | "
            f"{stats['roots']:3d} | "
            f"{stats['max_depth']:3d} | "
            f"{stats['crystallized']:5d} | "
            f"{stats['xi_ratio']:6.1%} | "
            f"{stats['theta']:8.1f} | "
            f"{stats['injected']:8.2f} | "
            f"{stats['tick']:6d}"
        )

    # --- Analysis ---
    print(f"\n{'='*70}")
    print(f"  LANDAUER CASCADE ANALYSIS")
    print(f"{'='*70}")

    accs = [s["accuracy"] for s in manager.epoch_history]
    thetas = [s["theta"] for s in manager.epoch_history]
    depths = [s["max_depth"] for s in manager.epoch_history]
    xi_ratios = [s["xi_ratio"] for s in manager.epoch_history]

    # Learning curve
    print(f"\n  LEARNING CURVE:")
    print(f"    Epoch 1:  {accs[0]:.1%}")
    print(f"    Peak:     {max(accs):.1%} (epoch {accs.index(max(accs))+1})")
    print(f"    Final:    {accs[-1]:.1%}")
    half = len(accs) // 2
    first_half = sum(accs[:half]) / half
    second_half = sum(accs[half:]) / (len(accs) - half)
    print(f"    First half avg:  {first_half:.1%}")
    print(f"    Second half avg: {second_half:.1%}")
    learns = second_half > first_half
    print(f"    Colony learns:   {'YES' if learns else 'NO'}")

    # Cascade sustainability
    print(f"\n  CASCADE SUSTAINABILITY (Θ per epoch):")
    for s in manager.epoch_history:
        bar = "#" * int(s["theta"] / max(1, max(thetas)) * 40)
        print(f"    E{s['epoch']:2d}: Θ={s['theta']:8.1f} | {bar}")

    theta_ratio = thetas[-1] / thetas[0] if thetas[0] > 0 else 0
    print(f"    Θ retention: {theta_ratio:.1%} (final/first)")
    if theta_ratio > 0.1:
        print(f"    Cascade: SELF-SUSTAINING")
    else:
        depletion = next((i+1 for i, t in enumerate(thetas) if t < thetas[0] * 0.01), len(thetas))
        print(f"    Cascade depletes at epoch ~{depletion}")

    # Depth evolution
    print(f"\n  DEPTH EVOLUTION:")
    for s in manager.epoch_history:
        bar = "|" * s["max_depth"]
        print(f"    E{s['epoch']:2d}: depth={s['max_depth']:2d} | {bar}")
    print(f"    Depth gain: {depths[0]} → {depths[-1]} ({depths[-1] - depths[0]:+d})")

    # ξ ratio evolution (should approach stable value)
    print(f"\n  CRYSTALLIZATION RATIO (ξ/(ξ+fluid)):")
    for s in manager.epoch_history:
        bar = "#" * int(s["xi_ratio"] * 50)
        print(f"    E{s['epoch']:2d}: {s['xi_ratio']:6.1%} | {bar}")

    # PAC partition check
    print(f"\n  PAC PARTITION (DFT predicts A/(A+ξ) ≈ ln(φ) = {LN_PHI:.4f}):")
    for s in manager.epoch_history:
        # Approximate: actualized ≈ total - ξ - θ_normalized
        total = s["xi_energy"] + s["fluid_energy"]
        if total > 0:
            a_approx = s["fluid_energy"]  # fluid ≈ actualized (not yet crystallized)
            ratio = a_approx / total if total > 0 else 0
            deviation = abs(ratio - LN_PHI) / LN_PHI * 100
            print(f"    E{s['epoch']:2d}: A/(A+ξ) = {ratio:.4f}  (deviation from ln(φ): {deviation:.1f}%)")

    # Character accuracy
    print(f"\n  CHARACTER ACCURACY (top 10):")
    char_accs = {}
    for ch, acc_list in colony.metrics.char_accuracy.items():
        if len(acc_list) >= 10:
            char_accs[ch] = sum(acc_list) / len(acc_list)
    if char_accs:
        best = sorted(char_accs.items(), key=lambda x: -x[1])[:10]
        for ch, acc in best:
            n = len(colony.metrics.char_accuracy[ch])
            bar = "#" * int(acc * 50)
            print(f"    '{ch}': {acc:5.0%} (n={n:4d}) {bar}")

    # Tree structure
    print(f"\n  FINAL TREE:")
    print(f"    Cells: {len(colony.cells)} / {MAX_CELLS}")
    print(f"    Roots: {len(colony.roots)}")
    print(f"    Max depth: {colony._get_max_tree_depth()}")
    if colony._path_depths:
        mean_pd = sum(colony._path_depths) / len(colony._path_depths)
        pct_deep = sum(1 for d in colony._path_depths if d > 1) / len(colony._path_depths)
        max_pd = max(colony._path_depths)
        print(f"    Mean path depth: {mean_pd:.1f}")
        print(f"    Max path depth: {max_pd}")
        print(f"    Deep paths (>1): {pct_deep:.1%}")

    # Per-root subtree info
    for rname in colony.roots[:5]:
        if rname in colony.cells:
            ss = colony._subtree_size(colony.cells[rname])
            sd = colony._subtree_depth(colony.cells[rname])
            print(f"    {rname}: {ss} cells, depth {sd}")

    print(f"\n  Total ticks: {colony.tick}")


if __name__ == "__main__":
    main()
