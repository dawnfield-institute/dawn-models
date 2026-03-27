"""Spike K Scaled Landauer -- Full GAIA Colony at Scale.

Uses the original CrystalColony (full GAIA agent pipeline) with Landauer
epoch management on a larger corpus. The LightCell optimization path showed
that the GAIA pipeline can't be simplified without losing accuracy:
- LightCell crystal filter: 9.7% (best lightweight)
- Full GAIA pipeline: 18-20%

CUDA doesn't help because the bottleneck is Python-level sequential
processing (character-by-character through tree navigation), not tensor ops.

This spike scales the PROVEN architecture instead of trying to optimize away
the thing that makes it work.

Usage:
    cd dawn-models/research/GAIA/spikes/population
    PYTHONIOENCODING=utf-8 PYTHONPATH="../../src;path/to/fracton" python spike_k_scaled_landauer.py
"""

from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

_root = Path(__file__).resolve().parents[2]
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))
_fracton = _root.parents[0] / "fracton"
if str(_fracton) not in sys.path:
    sys.path.insert(0, str(_fracton))

from spike_j_language_morphogenesis import CharacterCodebook, TextEnvironment
from spike_k_physics_first import CrystalColony, XI_SEC, PHI_INV, DIM
from spike_k_landauer import LandauerEpochManager, LN_PHI, PHI_SQ_FLOOR
import spike_k_physics_first as skp

# ==========================================================================
#  Corpus -- longer texts for scaling test
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
#  Main
# ==========================================================================

def main():
    MAX_CELLS = 300
    N_EPOCHS = 8

    print("=" * 70)
    print("  SPIKE K SCALED LANDAUER -- Full GAIA Colony at Scale")
    print(f"  P = A + xi + Theta  (Theta reinjects as fuel for next epoch)")
    print(f"  Corpus: {len(CORPUS)} chars | Epochs: {N_EPOCHS} | Max cells: {MAX_CELLS}")
    print(f"  Cascade constants: ln(phi)={LN_PHI:.4f}, floor(phi^2)={PHI_SQ_FLOOR}")
    print("=" * 70)

    torch.manual_seed(42)
    codebook = CharacterCodebook()

    # Set max cells before colony creation
    skp.MAX_CELLS = MAX_CELLS

    colony = CrystalColony(codebook)
    manager = LandauerEpochManager(colony)

    # Header
    print(f"\n  {'Ep':>3s} | {'Acc':>6s} | {'Cells':>5s} | {'Roots':>3s} | {'Depth':>3s} | "
          f"{'Cryst':>5s} | {'xi ratio':>7s} | {'Theta':>8s} | {'Time':>6s} | {'ch/s':>6s}")
    print(f"  {'-'*3}-+-{'-'*6}-+-{'-'*5}-+-{'-'*3}-+-{'-'*3}-+-"
          f"{'-'*5}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}")

    total_start = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        stats = manager.run_epoch(CORPUS, codebook, epoch)
        elapsed = time.time() - t0
        chars_per_sec = stats["n_chars"] / elapsed if elapsed > 0 else 0

        print(
            f"  {stats['epoch']:3d} | "
            f"{stats['accuracy']:5.1%} | "
            f"{stats['cells']:5d} | "
            f"{stats['roots']:3d} | "
            f"{stats['max_depth']:3d} | "
            f"{stats['crystallized']:5d} | "
            f"{stats['xi_ratio']:6.1%} | "
            f"{stats['theta']:8.1f} | "
            f"{elapsed:5.1f}s | "
            f"{chars_per_sec:5.0f}"
        )

    total_elapsed = time.time() - total_start

    # --- Analysis ---
    print(f"\n{'='*70}")
    print(f"  SCALED LANDAUER ANALYSIS")
    print(f"{'='*70}")

    accs = [s["accuracy"] for s in manager.epoch_history]
    thetas = [s["theta"] for s in manager.epoch_history]
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
    print(f"    Total time:      {total_elapsed:.1f}s")

    for i, acc in enumerate(accs):
        bar = "#" * int(acc * 200)
        print(f"    E{i+1:2d}: {acc:5.1%} | {bar}")

    # Crystallization
    print(f"\n  CRYSTALLIZATION:")
    for i, s in enumerate(manager.epoch_history):
        n = s["crystallized"]
        total = s["cells"]
        pct = n / max(1, total) * 100
        bar = "#" * int(pct / 2)
        print(f"    E{i+1:2d}: {pct:5.1f}% ({n}/{total}) | {bar}")

    # Character accuracy
    print(f"\n  CHARACTER ACCURACY (top 15):")
    char_accs = {}
    for ch, acc_list in colony.metrics.char_accuracy.items():
        if len(acc_list) >= 20:
            char_accs[ch] = sum(acc_list) / len(acc_list)
    best = sorted(char_accs.items(), key=lambda x: -x[1])[:15]
    for ch, acc in best:
        n = len(colony.metrics.char_accuracy[ch])
        bar = "#" * int(acc * 50)
        print(f"    '{ch}': {acc:5.0%} (n={n:5d}) {bar}")

    # Tree
    print(f"\n  FINAL TREE:")
    print(f"    Cells: {len(colony.cells)} / {MAX_CELLS}")
    print(f"    Roots: {len(colony.roots)}")
    print(f"    Max depth: {colony._get_max_tree_depth()}")
    if colony._path_depths:
        mean_pd = sum(colony._path_depths) / len(colony._path_depths)
        pct_deep = sum(1 for d in colony._path_depths if d > 1) / len(colony._path_depths)
        print(f"    Mean path depth: {mean_pd:.1f}")
        print(f"    Deep paths (>1): {pct_deep:.1%}")

    # Per-root subtree sizes
    for rname in colony.roots:
        if rname not in colony.cells:
            continue
        root = colony.cells[rname]
        # Count subtree
        count = 0
        max_d = 0
        stack = [(root, 0)]
        while stack:
            node, d = stack.pop()
            count += 1
            if d > max_d:
                max_d = d
            for child in node.children:
                if child.name in colony.cells:
                    stack.append((child, d + 1))
        if count > 1:
            print(f"    {rname}: {count} cells, depth {max_d}")


if __name__ == "__main__":
    main()
