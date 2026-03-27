"""Spike K Scaled -- Crystal Colony on Real Text.

Takes the validated CrystalColony (v4b: tree growth + SEC navigation + potential field)
and scales it up:
  - Full alphabet codebook (already in CharacterCodebook: 26 lower + 26 upper + 10 digits + punct + space)
  - 200 cells (was 40)
  - Real text: Shakespeare, Wikipedia-style prose, code
  - N-gram baselines (bigram, trigram) for honest comparison
  - Bits-per-character metric

The question: does physics-derived architecture scale, or does it only work on toy sequences?

Usage:
    cd dawn-models/research/GAIA/spikes/population
    PYTHONPATH="../../src;path/to/fracton" python spike_k_scaled.py
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
)


# ==========================================================================
#  N-gram Baselines
# ==========================================================================

class NGramBaseline:
    """Simple character n-gram model for comparison."""

    def __init__(self, n: int = 2):
        self.n = n
        self.counts: dict[str, Counter] = defaultdict(Counter)
        self.total: dict[str, int] = defaultdict(int)
        self._history: list[str] = []

    def observe(self, char: str):
        """Add character to history and update counts."""
        self._history.append(char)
        # Update all n-gram orders up to self.n
        for order in range(1, self.n + 1):
            if len(self._history) >= order + 1:
                context = "".join(self._history[-(order + 1):-1])
                self.counts[context][char] += 1
                self.total[context] += 1

    def predict(self) -> str | None:
        """Predict next character from highest-order matching context."""
        for order in range(self.n, 0, -1):
            if len(self._history) >= order:
                context = "".join(self._history[-order:])
                if context in self.counts and self.total[context] > 0:
                    return self.counts[context].most_common(1)[0][0]
        return None

    def predict_prob(self, char: str) -> float:
        """Probability of char given context (for bits-per-char)."""
        for order in range(self.n, 0, -1):
            if len(self._history) >= order:
                context = "".join(self._history[-order:])
                if context in self.counts and self.total[context] > 0:
                    return self.counts[context].get(char, 0) / self.total[context]
        return 0.0

    def reset_history(self):
        """Reset prediction history but keep learned counts."""
        self._history = []


# ==========================================================================
#  Real Text Corpus
# ==========================================================================

SHAKESPEARE = (
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
)

PROSE = (
    "the crystal grows from seed to structure "
    "each layer adds what the previous could not hold "
    "potential flows like water through channels "
    "carving deeper paths with every passing moment "
    "what was uncertain becomes certain "
    "what was possible narrows to actual "
    "the tree remembers what the forest forgets "
    "and in the branches lies the shape of thought "
    "not all at once but one path at a time "
    "a photon through the lattice finding its way "
    "the intelligence is not in the nodes "
    "it is in the navigation between them "
    "as possibilities collapse new ones emerge "
    "conservation ensures nothing is lost "
    "only transformed from potential to actual "
    "the cascade continues deeper and deeper "
)

CODE_TEXT = (
    "def process(self, input): "
    "result = self.transform(input) "
    "if result.energy > threshold: "
    "self.absorb(result) "
    "return result "
    "else: "
    "residual = input - result "
    "self.spawn(residual) "
    "return None "
) * 3

REPETITION_LONG = "abcdefabcdefabcdefabcdef" * 20

SCALED_CURRICULUM = [
    {
        "name": "repetition_6",
        "text": REPETITION_LONG,
        "description": "6-char cycle (abcdef). Tests scaling from 3-char to 6-char.",
        "print_every": 50,
    },
    {
        "name": "shakespeare",
        "text": SHAKESPEARE,
        "description": "Hamlet soliloquy. Real English, varied vocabulary.",
        "print_every": 100,
    },
    {
        "name": "prose",
        "text": PROSE,
        "description": "DFT-themed prose. Tests on familiar domain.",
        "print_every": 100,
    },
    {
        "name": "code",
        "text": CODE_TEXT,
        "description": "Python-like code. Tests structural pattern learning.",
        "print_every": 50,
    },
    {
        "name": "shakespeare_2",
        "text": SHAKESPEARE,
        "description": "Shakespeare AGAIN. Does the colony remember?",
        "print_every": 100,
    },
]


# ==========================================================================
#  Metrics
# ==========================================================================

def bits_per_char(errors: list[float]) -> float:
    """Convert prediction errors to bits per character.

    BPC = -log2(accuracy). For error rate e, accuracy ~ 1-e.
    Lower is better. Perfect = 0. Random over N chars = log2(N).
    """
    if not errors:
        return float('inf')
    # Use mean accuracy, clamp to avoid log(0)
    mean_acc = max(0.001, 1.0 - sum(errors) / len(errors))
    return -math.log2(mean_acc)


# ==========================================================================
#  Main Experiment
# ==========================================================================

def main():
    MAX_CELLS = 200  # scaled up from 40

    print("=" * 70)
    print("  SPIKE K SCALED -- Crystal Colony on Real Text")
    print(f"  Max cells: {MAX_CELLS}")
    print(f"  Codebook: full alphabet ({len(CharacterCodebook()._vectors)} chars)")
    print(f"  Phases: {len(SCALED_CURRICULUM)}")
    print("=" * 70)

    torch.manual_seed(42)
    codebook = CharacterCodebook()

    # Count unique chars across curriculum
    all_chars = set()
    for phase in SCALED_CURRICULUM:
        all_chars.update(phase["text"])
    print(f"\n  Unique characters in corpus: {len(all_chars)}")
    print(f"  Characters: {''.join(sorted(all_chars))}")

    # Codebook sanity for real chars
    print(f"\n  Codebook check (sample):")
    for ch in "the ":
        vec = codebook.encode(ch)
        decoded, conf = codebook.decode_nearest(vec)
        ok = "OK" if decoded == ch else f"MISMATCH({decoded})"
        print(f"    '{ch}' -> '{decoded}' (conf={conf:.3f}) {ok}")

    # --- N-gram baselines ---
    bigram = NGramBaseline(n=2)
    trigram = NGramBaseline(n=3)
    baseline_results: dict[str, dict[str, float]] = {}

    print(f"\n{'='*70}")
    print(f"  RUNNING N-GRAM BASELINES")
    print(f"{'='*70}")

    for phase in SCALED_CURRICULUM:
        text = phase["text"]
        bi_correct = 0
        tri_correct = 0
        bi_errors = []
        tri_errors = []

        for i in range(len(text) - 1):
            ch = text[i]
            nxt = text[i + 1]

            bi_pred = bigram.predict()
            tri_pred = trigram.predict()

            if bi_pred == nxt:
                bi_correct += 1
                bi_errors.append(0.0)
            else:
                bi_errors.append(1.0)
            if tri_pred == nxt:
                tri_correct += 1
                tri_errors.append(0.0)
            else:
                tri_errors.append(1.0)

            bigram.observe(ch)
            trigram.observe(ch)

        n = len(text) - 1
        bi_acc = bi_correct / n if n > 0 else 0
        tri_acc = tri_correct / n if n > 0 else 0
        baseline_results[phase["name"]] = {
            "bigram": bi_acc,
            "trigram": tri_acc,
            "bigram_bpc": bits_per_char(bi_errors),
            "trigram_bpc": bits_per_char(tri_errors),
        }
        print(f"  {phase['name']:20s} | bigram={bi_acc:.1%} | trigram={tri_acc:.1%}")

    # --- Crystal Colony ---
    print(f"\n{'='*70}")
    print(f"  RUNNING CRYSTAL COLONY (max_cells={MAX_CELLS})")
    print(f"{'='*70}")

    # Patch MAX_CELLS on the imported module
    import spike_k_physics_first as skp
    skp.MAX_CELLS = MAX_CELLS

    colony = CrystalColony(codebook)
    crystal_results: dict[str, float] = {}
    crystal_errors: dict[str, list[float]] = {}

    for phase in SCALED_CURRICULUM:
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

        phase_acc = colony.metrics.phase_summary().get(phase["name"], 0.0)
        crystal_results[phase["name"]] = phase_acc

        # Collect errors for BPC
        phase_accs = colony.metrics.phase_accuracy.get(phase["name"], [])
        crystal_errors[phase["name"]] = [0.0 if a else 1.0 for a in phase_accs]

    # --- Full Report ---
    colony.learning_report()
    verdicts = colony.learning_verdict()

    # --- Comparison Table ---
    print(f"\n{'='*70}")
    print(f"  SCALED COMPARISON: Crystal vs N-grams")
    print(f"{'='*70}")
    print(f"  {'Phase':<20s} | {'Bigram':>8s} | {'Trigram':>8s} | {'Crystal':>8s} | {'Winner':<10s}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")

    crystal_wins = 0
    ngram_wins = 0
    for phase in SCALED_CURRICULUM:
        name = phase["name"]
        bi = baseline_results[name]["bigram"]
        tri = baseline_results[name]["trigram"]
        cr = crystal_results.get(name, 0.0)
        best_ngram = max(bi, tri)

        if cr > best_ngram:
            winner = "CRYSTAL"
            crystal_wins += 1
        elif cr < best_ngram:
            winner = "n-gram"
            ngram_wins += 1
        else:
            winner = "tie"

        print(f"  {name:<20s} | {bi:>7.1%} | {tri:>7.1%} | {cr:>7.1%} | {winner}")

    print(f"\n  Score: Crystal {crystal_wins} - N-gram {ngram_wins}")

    # BPC comparison
    print(f"\n  BITS PER CHARACTER (lower = better):")
    print(f"  {'Phase':<20s} | {'Bigram BPC':>10s} | {'Trigram BPC':>10s} | {'Crystal BPC':>11s}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*11}")
    for phase in SCALED_CURRICULUM:
        name = phase["name"]
        bi_bpc = baseline_results[name]["bigram_bpc"]
        tri_bpc = baseline_results[name]["trigram_bpc"]
        cr_bpc = bits_per_char(crystal_errors.get(name, []))
        print(f"  {name:<20s} | {bi_bpc:>10.2f} | {tri_bpc:>10.2f} | {cr_bpc:>11.2f}")

    # Key scaling metrics
    print(f"\n{'='*70}")
    print(f"  SCALING METRICS")
    print(f"{'='*70}")
    print(f"  Colony size:     {len(colony.cells)} cells / {MAX_CELLS} max")
    print(f"  Roots:           {len(colony.roots)}")
    print(f"  Tree depth:      {colony._get_max_tree_depth()}")
    if colony._path_depths:
        mean_pd = sum(colony._path_depths) / len(colony._path_depths)
        pct_deep = sum(1 for d in colony._path_depths if d > 1) / len(colony._path_depths)
        print(f"  Mean path depth: {mean_pd:.1f}")
        print(f"  Deep paths (>1): {pct_deep:.1%}")
    print(f"  Total ticks:     {colony.tick}")
    print(f"  Unique chars:    {len(all_chars)}")

    # Shakespeare re-exposure analysis
    s1_acc = crystal_results.get("shakespeare", 0.0)
    s2_acc = crystal_results.get("shakespeare_2", 0.0)
    print(f"\n  MEMORY TEST (Shakespeare re-exposure):")
    print(f"    First pass:  {s1_acc:.1%}")
    print(f"    Second pass: {s2_acc:.1%}")
    if s2_acc > s1_acc:
        improvement = (s2_acc - s1_acc) / max(0.001, s1_acc) * 100
        print(f"    Improvement: +{improvement:.0f}% -- colony REMEMBERS")
    else:
        print(f"    No improvement -- colony forgot or saturated")

    print(f"\n  Final: {len(colony.cells)} cells, {len(colony.roots)} roots, tick={colony.tick}")


if __name__ == "__main__":
    main()
