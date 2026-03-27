"""Spike K Epochs -- Crystal Colony Learning Curves.

Does the colony get BETTER with repeated exposure? Run multiple epochs
over the same text and track per-epoch accuracy. This directly tests
whether the crystal structure accumulates useful knowledge.

Also: longer text, more cells, pre-trained n-gram baseline.

Usage:
    cd dawn-models/research/GAIA/spikes/population
    PYTHONPATH="../../src;path/to/fracton" python spike_k_epochs.py
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
from spike_k_physics_first import CrystalColony, XI_SEC
from gaia.core.types import FieldState, SECPhase


# ==========================================================================
#  Longer corpus -- enough data for real learning
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

# Combined: ~2600 chars of real English prose
CORPUS = HAMLET + GENESIS


# ==========================================================================
#  Pre-trained N-gram (trained on full corpus first, then evaluated)
# ==========================================================================

class PretrainedNGram:
    """N-gram model pre-trained on the full corpus, then evaluated per-epoch."""

    def __init__(self, text: str, n: int = 3):
        self.n = n
        self.counts: dict[str, Counter] = defaultdict(Counter)
        self.total: dict[str, int] = defaultdict(int)

        # Pre-train on full corpus
        for i in range(len(text) - 1):
            for order in range(1, n + 1):
                if i >= order:
                    context = text[i - order:i]
                    self.counts[context][text[i]] += 1
                    self.total[context] += 1

    def predict(self, history: str) -> str | None:
        """Predict next char given recent history."""
        for order in range(self.n, 0, -1):
            if len(history) >= order:
                context = history[-order:]
                if context in self.counts and self.total[context] > 0:
                    return self.counts[context].most_common(1)[0][0]
        return None

    def evaluate(self, text: str) -> float:
        """Run through text, return accuracy."""
        correct = 0
        total = 0
        for i in range(self.n, len(text) - 1):
            pred = self.predict(text[:i])
            if pred == text[i]:
                correct += 1
            total += 1
        return correct / total if total > 0 else 0.0


# ==========================================================================
#  Per-epoch evaluation
# ==========================================================================

def run_epoch(colony: CrystalColony, text: str, codebook: CharacterCodebook,
              epoch: int) -> dict:
    """Run one epoch through text, return stats."""
    text_env = TextEnvironment(text, codebook)
    phase_name = f"epoch_{epoch}"
    colony.metrics.set_phase(phase_name)

    correct = 0
    total = 0
    errors = []

    while text_env.has_more():
        env, current_char, next_char = text_env.step()
        colony.step(env, current_char, next_char)
        total += 1

        # Check last prediction
        accs = colony.metrics.phase_accuracy.get(phase_name, [])
        if accs:
            if accs[-1]:
                correct += 1
            errors.append(0.0 if accs[-1] else 1.0)

    acc = correct / total if total > 0 else 0.0
    n_cells = len(colony.cells)
    n_roots = len(colony.roots)
    max_depth = colony._get_max_tree_depth()
    mean_pd = (sum(colony._path_depths[-total:]) / total
               if colony._path_depths else 0)
    cryst = sum(1 for n in colony.cells
                if colony._get_entropy_variance(n) < XI_SEC)

    return {
        "epoch": epoch,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "cells": n_cells,
        "roots": n_roots,
        "max_depth": max_depth,
        "mean_path": mean_pd,
        "crystallized": cryst,
        "tick": colony.tick,
    }


# ==========================================================================
#  Main
# ==========================================================================

def main():
    MAX_CELLS = 300
    N_EPOCHS = 8

    print("=" * 70)
    print("  SPIKE K EPOCHS -- Crystal Colony Learning Curves")
    print(f"  Corpus: {len(CORPUS)} chars (Hamlet + Genesis)")
    print(f"  Epochs: {N_EPOCHS}")
    print(f"  Max cells: {MAX_CELLS}")
    print("=" * 70)

    torch.manual_seed(42)
    codebook = CharacterCodebook()

    # Unique chars
    unique = sorted(set(CORPUS))
    print(f"\n  Unique characters: {len(unique)}")
    print(f"  Chars: {''.join(unique)}")

    # --- Pre-trained baselines ---
    print(f"\n  Pre-trained N-gram baselines (trained on full corpus):")
    for n in [2, 3, 4, 5]:
        model = PretrainedNGram(CORPUS, n=n)
        acc = model.evaluate(CORPUS)
        print(f"    {n}-gram: {acc:.1%}")

    pretrained_trigram = PretrainedNGram(CORPUS, n=3)
    pretrained_acc = pretrained_trigram.evaluate(CORPUS)

    # --- Crystal Colony epochs ---
    import spike_k_physics_first as skp
    skp.MAX_CELLS = MAX_CELLS

    colony = CrystalColony(codebook)
    epoch_results = []

    print(f"\n{'='*70}")
    print(f"  EPOCH TRAINING")
    print(f"{'='*70}")
    print(f"  {'Epoch':>5s} | {'Acc':>6s} | {'Cells':>5s} | {'Roots':>5s} | {'Depth':>5s} | {'Path':>5s} | {'Cryst':>5s} | {'Tick':>6s}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}")

    for epoch in range(1, N_EPOCHS + 1):
        stats = run_epoch(colony, CORPUS, codebook, epoch)
        epoch_results.append(stats)
        print(
            f"  {stats['epoch']:5d} | "
            f"{stats['accuracy']:5.1%} | "
            f"{stats['cells']:5d} | "
            f"{stats['roots']:5d} | "
            f"{stats['max_depth']:5d} | "
            f"{stats['mean_path']:5.1f} | "
            f"{stats['crystallized']:5d} | "
            f"{stats['tick']:6d}"
        )

    # --- Learning curve analysis ---
    print(f"\n{'='*70}")
    print(f"  LEARNING CURVE ANALYSIS")
    print(f"{'='*70}")

    accs = [r["accuracy"] for r in epoch_results]
    first_acc = accs[0]
    last_acc = accs[-1]
    peak_acc = max(accs)
    peak_epoch = accs.index(peak_acc) + 1

    print(f"  First epoch:  {first_acc:.1%}")
    print(f"  Last epoch:   {last_acc:.1%}")
    print(f"  Peak:         {peak_acc:.1%} (epoch {peak_epoch})")
    print(f"  Improvement:  {last_acc - first_acc:+.1%}")

    # Monotonicity check
    improving_epochs = sum(1 for i in range(1, len(accs)) if accs[i] >= accs[i-1])
    print(f"  Improving epochs: {improving_epochs}/{len(accs)-1}")

    # First half vs second half
    half = len(accs) // 2
    first_half = sum(accs[:half]) / half
    second_half = sum(accs[half:]) / (len(accs) - half)
    print(f"  First half avg:  {first_half:.1%}")
    print(f"  Second half avg: {second_half:.1%}")
    learns = second_half > first_half
    print(f"  Colony learns:   {'YES' if learns else 'NO'}")

    # --- vs pre-trained trigram ---
    print(f"\n{'='*70}")
    print(f"  CRYSTAL vs PRE-TRAINED TRIGRAM")
    print(f"{'='*70}")
    print(f"  Pre-trained trigram (sees corpus before eval): {pretrained_acc:.1%}")
    print(f"  Crystal epoch 1 (never seen corpus):           {first_acc:.1%}")
    print(f"  Crystal epoch {peak_epoch} (peak):                       {peak_acc:.1%}")
    print(f"  Crystal epoch {N_EPOCHS} (final):                      {last_acc:.1%}")

    if peak_acc > pretrained_acc:
        print(f"\n  ** CRYSTAL BEATS PRE-TRAINED TRIGRAM **")
        print(f"  Physics-first architecture surpasses statistical counting")
        print(f"  even when the counter sees ALL the data upfront.")
    elif last_acc > pretrained_acc:
        print(f"\n  ** CRYSTAL BEATS PRE-TRAINED TRIGRAM (by final epoch) **")
    else:
        ratio = peak_acc / pretrained_acc if pretrained_acc > 0 else 0
        print(f"\n  Crystal reaches {ratio:.0%} of pre-trained trigram performance")
        print(f"  But crystal has: zero pre-training, zero parameters, full interpretability")

    # --- Tree structure evolution ---
    print(f"\n{'='*70}")
    print(f"  TREE EVOLUTION ACROSS EPOCHS")
    print(f"{'='*70}")
    for r in epoch_results:
        bar = "#" * int(r["accuracy"] * 100)
        print(f"  E{r['epoch']}: {r['accuracy']:5.1%} | {r['cells']}c {r['roots']}r d{r['max_depth']} p{r['mean_path']:.1f} | {bar}")

    # --- Character-level analysis ---
    print(f"\n{'='*70}")
    print(f"  CHARACTER ACCURACY (from all epochs)")
    print(f"{'='*70}")
    char_accs = {}
    for ch, acc_list in colony.metrics.char_accuracy.items():
        if len(acc_list) >= 10:
            char_accs[ch] = sum(acc_list) / len(acc_list)
    if char_accs:
        best = sorted(char_accs.items(), key=lambda x: -x[1])[:15]
        for ch, acc in best:
            n = len(colony.metrics.char_accuracy[ch])
            bar = "#" * int(acc * 50)
            print(f"    '{ch}': {acc:5.0%} (n={n:4d}) {bar}")

    # Final colony state
    print(f"\n{'='*70}")
    print(f"  FINAL STATE")
    print(f"{'='*70}")
    print(f"  Cells: {len(colony.cells)} / {MAX_CELLS}")
    print(f"  Roots: {len(colony.roots)}")
    print(f"  Max depth: {colony._get_max_tree_depth()}")
    if colony._path_depths:
        mean_pd = sum(colony._path_depths) / len(colony._path_depths)
        pct_deep = sum(1 for d in colony._path_depths if d > 1) / len(colony._path_depths)
        print(f"  Mean path depth: {mean_pd:.1f}")
        print(f"  Deep paths (>1): {pct_deep:.1%}")
    n_cryst = sum(1 for n in colony.cells
                  if colony._get_entropy_variance(n) < XI_SEC)
    print(f"  Crystallized: {n_cryst}/{len(colony.cells)}")
    print(f"  Total ticks: {colony.tick}")


if __name__ == "__main__":
    main()
