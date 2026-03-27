"""Spike J -- Language Morphogenesis: Prediction-Driven Colony Growth.

The colony from Spike I gets a PURPOSE: predict the next character in text.
Growth is driven by prediction error, not just signal novelty.

  - HIGH error on NOVEL pattern -> spawn new lobe (new capability needed)
  - HIGH error on FAMILIAR pattern -> spawn child (refine existing capability)
  - LOW error -> reinforce cells, accelerate maturity
  - Morphogenesis IS learning: no weight updates, just structural adaptation

The thesis: learning = counting + structural adaptation, not gradient descent.
The TransitionCounters inside cells learn n-grams by counting.
The PACTrees learn pattern clusters by resonance.
The colony's TOPOLOGY adapts to the structure of the language.

Usage:
    cd dawn-models/research/GAIA
    PYTHONPATH="src;../../fracton" python spikes/population/spike_j_language_morphogenesis.py
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

# Import from spike_i -- reuse, don't copy
from spike_i_morphogenesis import (
    Cell,
    GrowingColony,
    Maturity,
    Signal,
    make_organism,
    DIM,
    PHI_INV,
    MAX_COLONY_SIZE,
    ACTIVATION_THRESHOLD,
)

from gaia.core.coupled_fields_bus import _harmonic_resonance
from gaia.core.types import FieldState, SECPhase


# ==========================================================================
#  1. CHARACTER CODEBOOK -- Categorical Embeddings
# ==========================================================================

class CharacterCodebook:
    """Maps characters to DIM-dimensional unit vectors with categorical structure.

    Character classes share a class direction (40%) + individual variation (60%).
    Vowels cluster together, consonants cluster together, etc.
    The colony can discover linguistic categories from the geometry.
    """

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self._vectors: dict[str, torch.Tensor] = {}
        self._build_codebook()

    def _build_codebook(self):
        classes = {
            "vowel": list("aeiou"),
            "consonant": list("bcdfghjklmnpqrstvwxyz"),
            "digit": list("0123456789"),
            "space": [" ", "\n", "\t"],
            "punct": list(".,!?;:'\"()-"),
        }

        for class_idx, (class_name, chars) in enumerate(classes.items()):
            torch.manual_seed(class_idx * 10000 + 42)
            # Shared class direction
            class_dir = torch.randn(self.dim)
            class_dir = class_dir / (torch.norm(class_dir) + 1e-8)

            vecs: list[torch.Tensor] = []
            for i, ch in enumerate(chars):
                torch.manual_seed(class_idx * 10000 + i * 100 + 7)
                v = torch.randn(self.dim)
                # 40% class direction + 60% individual
                v = 0.4 * class_dir + 0.6 * v
                # Gram-Schmidt within class
                for prev in vecs:
                    v = v - torch.dot(v, prev) * prev
                norm = torch.norm(v)
                if norm < 1e-8:
                    torch.manual_seed(class_idx * 10000 + i * 100 + 999)
                    v = torch.randn(self.dim)
                v = v / (torch.norm(v) + 1e-8)
                vecs.append(v)
                self._vectors[ch] = v

        # Uppercase = lowercase + small offset
        torch.manual_seed(99999)
        offset = 0.15 * torch.randn(self.dim)
        for ch in "abcdefghijklmnopqrstuvwxyz":
            upper = ch.upper()
            if ch in self._vectors:
                v = self._vectors[ch] + offset
                v = v / (torch.norm(v) + 1e-8)
                self._vectors[upper] = v

    def encode(self, char: str) -> torch.Tensor:
        """Character -> DIM unit vector. Hash fallback for unknowns."""
        if char in self._vectors:
            return self._vectors[char].clone()
        torch.manual_seed(hash(char) % 2**31)
        v = torch.randn(self.dim)
        v = v / (torch.norm(v) + 1e-8)
        self._vectors[char] = v
        return v.clone()

    def decode_nearest(self, tensor: torch.Tensor) -> tuple[str, float]:
        """Find character with highest cosine similarity to tensor."""
        t = tensor.flatten()[:self.dim]
        t_norm = torch.norm(t)
        if t_norm < 1e-8:
            return "?", 0.0
        best_char, best_sim = "?", -1.0
        for ch, vec in self._vectors.items():
            sim = float(torch.dot(t, vec) / (t_norm * torch.norm(vec) + 1e-8))
            if sim > best_sim:
                best_sim = sim
                best_char = ch
        return best_char, best_sim

    def encode_string(self, text: str) -> list[torch.Tensor]:
        return [self.encode(ch) for ch in text]

    def class_of(self, char: str) -> str:
        """Return the character class name."""
        if char in "aeiouAEIOU":
            return "vowel"
        elif char.lower() in "bcdfghjklmnpqrstvwxyz":
            return "consonant"
        elif char in "0123456789":
            return "digit"
        elif char in " \n\t":
            return "space"
        else:
            return "punct"


# ==========================================================================
#  2. TEXT ENVIRONMENT -- Character-at-a-Time Feeding
# ==========================================================================

class TextEnvironment:
    """Feeds characters one at a time as FieldStates."""

    def __init__(self, text: str, codebook: CharacterCodebook):
        self.text = text
        self.codebook = codebook
        self.position = 0

    def step(self) -> tuple[FieldState, str, str]:
        """Advance position, return (field_state, current_char, next_char)."""
        ch = self.text[self.position]
        nxt = self.text[self.position + 1] if self.position + 1 < len(self.text) else "\0"
        tensor = self.codebook.encode(ch)
        self.position += 1
        return FieldState(
            tensor=tensor, entropy=1.0, phase=SECPhase.ORDERED,
            conservation_budget=0.0, provenance=[], timestamp=time.time(),
        ), ch, nxt

    def has_more(self) -> bool:
        return self.position < len(self.text) - 1

    def reset(self):
        self.position = 0

    def remaining(self) -> int:
        return max(0, len(self.text) - 1 - self.position)


# ==========================================================================
#  3. PREDICTION EXTRACTION
# ==========================================================================

def extract_prediction(
    colony: GrowingColony,
    active_signals: dict[str, Signal],
    env_tensor: torch.Tensor,
    codebook: CharacterCodebook,
) -> tuple[str, float, torch.Tensor]:
    """Extract colony's next-character prediction from active cell signals.

    Weight each cell's output by resonance(input, voice) * trust(maturity).
    Decode the weighted sum to the nearest character.

    Returns: (predicted_char, confidence, prediction_tensor)
    """
    if not active_signals:
        return "?", 0.0, torch.zeros(DIM)

    weighted_sum = torch.zeros(DIM)
    total_weight = 0.0

    for name, signal in active_signals.items():
        if name not in colony.cells:
            continue
        cell = colony.cells[name]
        voice_norm = torch.norm(cell.voice)
        if voice_norm < 1e-8:
            resonance = 0.01
        else:
            resonance = max(0.01, _harmonic_resonance(env_tensor, cell.voice))
        # Trust increases with maturity: 0.2 (SURFACE) to 1.0 (CRYSTALLIZED)
        trust = 0.2 + 0.2 * cell.maturity.value
        weight = resonance * trust
        weighted_sum += weight * signal.tensor
        total_weight += weight

    if total_weight > 1e-8:
        weighted_sum /= total_weight

    predicted_char, confidence = codebook.decode_nearest(weighted_sum)
    return predicted_char, confidence, weighted_sum


# ==========================================================================
#  4. METRICS TRACKER
# ==========================================================================

class MetricsTracker:
    """Tracks prediction accuracy, lobe specialization, and error-growth correlation."""

    def __init__(self, window: int = 50):
        self.window = window
        self.accuracy_history: list[bool] = []
        self.error_history: list[float] = []
        self.char_accuracy: dict[str, list[bool]] = defaultdict(list)
        self.phase_accuracy: dict[str, list[bool]] = defaultdict(list)
        self.growth_events: list[dict] = []
        self.lobe_activations: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._current_phase: str = ""

    def set_phase(self, phase_name: str):
        self._current_phase = phase_name

    def record_prediction(self, correct: bool, error: float, target_char: str):
        self.accuracy_history.append(correct)
        self.error_history.append(error)
        self.char_accuracy[target_char].append(correct)
        if self._current_phase:
            self.phase_accuracy[self._current_phase].append(correct)

    def record_growth(self, tick: int, reason: str, error_at_tick: float):
        self.growth_events.append({
            "tick": tick, "reason": reason, "error": error_at_tick,
        })

    def record_activation(self, lobe_id: str, char: str):
        self.lobe_activations[lobe_id][char] += 1

    def rolling_accuracy(self, n: int | None = None) -> float:
        w = n or self.window
        recent = self.accuracy_history[-w:]
        if not recent:
            return 0.0
        return sum(recent) / len(recent)

    def rolling_error(self, n: int | None = None) -> float:
        w = n or self.window
        recent = self.error_history[-w:]
        if not recent:
            return 1.0
        return sum(recent) / len(recent)

    def phase_summary(self) -> dict[str, float]:
        result = {}
        for phase, accs in self.phase_accuracy.items():
            result[phase] = sum(accs) / len(accs) if accs else 0.0
        return result

    def error_growth_correlation(self) -> float:
        """Fraction of growth events that occurred when recent error > 0.5."""
        if not self.growth_events:
            return 0.0
        high_error_births = sum(1 for e in self.growth_events if e["error"] > 0.5)
        return high_error_births / len(self.growth_events)

    def specialization_report(self, colony: GrowingColony, codebook: CharacterCodebook):
        """Print which lobes activate for which characters."""
        print("\n  LOBE SPECIALIZATION:")
        specialized_lobes = 0
        for lobe_id, char_counts in sorted(self.lobe_activations.items()):
            total = sum(char_counts.values())
            if total < 5:
                continue
            # Top 5 characters
            top = sorted(char_counts.items(), key=lambda x: -x[1])[:5]
            top_str = ", ".join(
                f"'{ch}'={n}({n*100//total}%)" for ch, n in top
            )
            # Check class dominance
            class_counts: dict[str, int] = defaultdict(int)
            for ch, n in char_counts.items():
                class_counts[codebook.class_of(ch)] += n
            dominant_class = max(class_counts.items(), key=lambda x: x[1])
            dominance = dominant_class[1] / total
            if dominance > 0.6:
                specialized_lobes += 1
                tag = f" ** {dominant_class[0]} specialist ({dominance:.0%})"
            else:
                tag = ""
            print(f"    {lobe_id}: {top_str}{tag}")
        print(f"  Specialized lobes (>60% one class): {specialized_lobes}")
        return specialized_lobes


# ==========================================================================
#  5. PREDICTIVE COLONY -- The Core Innovation
# ==========================================================================

class PredictiveColony:
    """Wraps GrowingColony with prediction-error-driven morphogenesis.

    Does NOT subclass GrowingColony -- composes it. Adds prediction/feedback
    logic around each step without duplicating internals.
    """

    def __init__(self, codebook: CharacterCodebook, seed_name: str = "cell_0"):
        self.colony = GrowingColony(seed_name=seed_name)
        self.codebook = codebook
        self.metrics = MetricsTracker()

        # Prediction state
        self._last_prediction_tensor: torch.Tensor | None = None
        self._last_target_char: str | None = None
        self._recent_errors: list[float] = []
        self._error_window = 20

    @property
    def tick(self) -> int:
        return self.colony.tick

    def step_with_prediction(self, env: FieldState, current_char: str, next_char: str):
        """One tick: evaluate prediction, error-driven growth, process, extract prediction."""

        # --- Phase 1: Evaluate last tick's prediction ---
        prediction_error = 1.0  # default for first tick
        if self._last_prediction_tensor is not None and self._last_target_char is not None:
            target_vec = self.codebook.encode(self._last_target_char)
            resonance = _harmonic_resonance(self._last_prediction_tensor, target_vec)
            prediction_error = 1.0 - max(0.0, float(resonance))

            # Track accuracy (correct if error < 0.5)
            correct = prediction_error < 0.5
            self.metrics.record_prediction(correct, prediction_error, self._last_target_char)

            self._recent_errors.append(prediction_error)
            if len(self._recent_errors) > self._error_window:
                self._recent_errors.pop(0)

        # --- Phase 2: Error-driven growth decision ---
        best_name, best_score = self.colony._resonance_to_colony(env.tensor)

        if len(self.colony.cells) < MAX_COLONY_SIZE:
            if prediction_error > 0.8 and best_score < 0.3:
                # HIGH error on NOVEL pattern -> new lobe
                self.colony._spawn_root(env)
                self.colony.growth_log[-1]["reason"] = "high_error_novel"
                self.metrics.record_growth(
                    self.colony.tick, "high_error_novel", prediction_error
                )
            elif prediction_error > 0.6 and best_score > 0.4:
                # HIGH error on FAMILIAR pattern -> refine
                parent = self.colony.cells[best_name]
                if len(parent.children) < 5:
                    self.colony._spawn_child(parent, env)
                    self.colony.growth_log[-1]["reason"] = "high_error_familiar"
                    self.metrics.record_growth(
                        self.colony.tick, "high_error_familiar", prediction_error
                    )
            elif prediction_error < 0.3 and best_name and best_name in self.colony.cells:
                # LOW error -> reinforce (accelerated maturity)
                self.colony.cells[best_name].access_count += 2

        # --- Phase 3: Colony processing (skip internal growth) ---
        self.colony.step(env, skip_growth=True)

        # --- Phase 4: Extract prediction ---
        active_signals = getattr(self.colony, "_last_active_signals", {})
        predicted_char, confidence, pred_tensor = extract_prediction(
            self.colony, active_signals, env.tensor, self.codebook
        )
        self._last_prediction_tensor = pred_tensor
        self._last_target_char = next_char

        # --- Phase 5: Track lobe specialization ---
        for name in active_signals:
            if name in self.colony.cells:
                lobe = self.colony.cells[name].lobe_id
                self.metrics.record_activation(lobe, current_char)

    def run_text(
        self,
        text_env: TextEnvironment,
        phase_name: str = "",
        print_every: int = 100,
    ):
        """Process an entire text through the colony."""
        self.metrics.set_phase(phase_name)
        start_tick = self.colony.tick
        chars_processed = 0

        while text_env.has_more():
            env, current_char, next_char = text_env.step()
            self.step_with_prediction(env, current_char, next_char)
            chars_processed += 1

            if chars_processed % print_every == 0:
                acc = self.metrics.rolling_accuracy()
                err = self.metrics.rolling_error()
                n_cells = len(self.colony.cells)
                n_lobes = len(self.colony.roots)
                act = getattr(self.colony, "_last_activation_ratio", 0.0)
                print(
                    f"  tick {self.colony.tick:4d} | "
                    f"acc={acc:.0%} err={err:.2f} | "
                    f"{n_cells} cells {n_lobes} lobes | "
                    f"{act:.0%} active | "
                    f"char='{current_char}'"
                )

        phase_acc = self.metrics.phase_summary().get(phase_name, 0.0)
        print(
            f"  -- {phase_name} complete: {chars_processed} chars, "
            f"phase_acc={phase_acc:.1%}, "
            f"colony={len(self.colony.cells)} cells / {len(self.colony.roots)} lobes"
        )

    def learning_report(self):
        """Print comprehensive learning analysis."""
        print(f"\n{'='*70}")
        print(f"  LEARNING REPORT")
        print(f"  {self.colony.tick} ticks | "
              f"{len(self.colony.cells)} cells | "
              f"{len(self.colony.roots)} lobes")
        print(f"{'='*70}")

        # Phase accuracy
        print("\n  PHASE ACCURACY:")
        for phase, acc in self.metrics.phase_summary().items():
            n = len(self.metrics.phase_accuracy[phase])
            print(f"    {phase:20s}: {acc:.1%}  ({n} chars)")

        # Per-character accuracy (top 10 best and worst)
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

        print("\n  CHARACTER ACCURACY (worst):")
        if char_accs:
            worst = sorted(char_accs.items(), key=lambda x: x[1])[:8]
            for ch, acc in worst:
                n = len(self.metrics.char_accuracy[ch])
                print(f"    '{ch}': {acc:.0%}  (n={n})")

        # Growth events
        print(f"\n  GROWTH EVENTS: {len(self.metrics.growth_events)} total")
        reason_counts: dict[str, int] = defaultdict(int)
        for e in self.metrics.growth_events:
            reason_counts[e["reason"]] += 1
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
        corr = self.metrics.error_growth_correlation()
        print(f"  Error-growth correlation: {corr:.0%} of births at high error")

        # Colony maturity
        print("\n  MATURITY DISTRIBUTION:")
        mat_counts: dict[str, int] = defaultdict(int)
        for cell in self.colony.cells.values():
            mat_counts[cell.maturity.name] += 1
        for mat_name in ["SURFACE", "SHALLOW", "STABLE", "DEEP", "CRYSTALLIZED"]:
            n = mat_counts.get(mat_name, 0)
            if n:
                print(f"    {mat_name:15s}: {n}")

    def learning_verdict(self) -> dict[str, bool]:
        """Evaluate 5 learning criteria. Print pass/fail for each."""
        print(f"\n{'='*70}")
        print(f"  LEARNING VERDICT")
        print(f"{'='*70}")
        verdicts = {}

        # 1. Accuracy increases: last 25% of each phase > first 25%
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
        print(f"  1. Accuracy increases within phases: {status}")

        # 2. Domain shift recovery: accuracy drops then recovers
        ds_accs = self.metrics.phase_accuracy.get("domain_shift", [])
        if len(ds_accs) >= 40:
            early_ds = sum(ds_accs[:20]) / 20
            late_ds = sum(ds_accs[-20:]) / 20
            criterion_2 = late_ds > early_ds
        else:
            criterion_2 = len(ds_accs) == 0  # skip if no domain shift phase
        verdicts["domain_shift_recovery"] = criterion_2
        status = "PASS" if criterion_2 else "FAIL"
        print(f"  2. Domain shift recovery:            {status}")

        # 3. Lobe specialization: >=2 lobes with >60% from one class
        specialized = self.metrics.specialization_report(self.colony, self.codebook)
        criterion_3 = specialized >= 2
        verdicts["lobe_specialization"] = criterion_3
        status = "PASS" if criterion_3 else "FAIL"
        print(f"  3. Lobe specialization (>=2 lobes):  {status}")

        # 4. Error-growth correlation: >50% of births at high error
        corr = self.metrics.error_growth_correlation()
        criterion_4 = corr > 0.5 or len(self.metrics.growth_events) == 0
        verdicts["error_growth_correlation"] = criterion_4
        status = "PASS" if criterion_4 else "FAIL"
        print(f"  4. Error-growth correlation (>50%%):  {status}")

        # 5. Overall accuracy > random baseline
        total_accs = self.metrics.accuracy_history
        if total_accs:
            overall = sum(total_accs) / len(total_accs)
            # Random baseline for ~70 chars is ~1.4%
            criterion_5 = overall > 0.05  # 5% is well above random
        else:
            criterion_5 = False
        verdicts["above_random"] = criterion_5
        status = "PASS" if criterion_5 else "FAIL"
        print(f"  5. Above random baseline (>5%%):     {status}")

        passed = sum(verdicts.values())
        total = len(verdicts)
        print(f"\n  RESULT: {passed}/{total} criteria passed")
        return verdicts


# ==========================================================================
#  6. CURRICULUM
# ==========================================================================

CURRICULUM = [
    {
        "name": "repetition",
        "text": "abcabcabcabcabcabcabcabcabcabc" * 10,
        "description": "Pure repetition. Colony learns a->b->c->a.",
        "print_every": 100,
    },
    {
        "name": "words",
        "text": "the cat sat on the mat " * 15,
        "description": "Word patterns. Space prediction, word-level bigrams.",
        "print_every": 100,
    },
    {
        "name": "rhyme",
        "text": (
            "twinkle twinkle little star "
            "how i wonder what you are "
            "up above the world so high "
            "like a diamond in the sky "
        ) * 5,
        "description": "Rhyme/rhythm structure. Deeper n-gram learning.",
        "print_every": 100,
    },
    {
        "name": "prose",
        "text": (
            "it was the best of times "
            "it was the worst of times "
            "it was the age of wisdom "
            "it was the age of foolishness "
        ) * 4,
        "description": "Real prose with varied vocabulary.",
        "print_every": 100,
    },
    {
        "name": "domain_shift",
        "text": "0123456789 " * 20 + "aabbccddee " * 20,
        "description": "Abrupt domain shift: digits then letters.",
        "print_every": 50,
    },
]


# ==========================================================================
#  7. MAIN
# ==========================================================================

def main():
    import os

    print("=" * 70)
    print("  SPIKE J -- Language Morphogenesis")
    print("  Prediction-Driven Colony Growth")
    print("  Learning = Counting + Structural Adaptation")
    print("=" * 70)

    codebook = CharacterCodebook()

    # Quick codebook sanity check
    print("\n  Codebook check:")
    for ch in "abc 123.!":
        vec = codebook.encode(ch)
        decoded, conf = codebook.decode_nearest(vec)
        ok = "OK" if decoded == ch else f"MISMATCH({decoded})"
        print(f"    '{ch}' -> encode -> decode = '{decoded}' (conf={conf:.3f}) {ok}")

    colony = PredictiveColony(codebook)

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

        # Checkpoint after rhyme phase (mid-run)
        if phase["name"] == "rhyme":
            from gaia.network.checkpoint import (
                save_colony, load_colony, checkpoint_info,
            )

            checkpoint_dir = Path(_root) / "checkpoints"
            checkpoint_path = checkpoint_dir / "colony_language.pt"

            print(f"\n  CHECKPOINT: saving after rhyme phase...")
            save_colony(colony.colony, checkpoint_path)
            pre_tick = colony.colony.tick
            pre_cells = len(colony.colony.cells)

            info = checkpoint_info(checkpoint_path)
            file_size = os.path.getsize(checkpoint_path)
            print(f"  saved: {file_size/1024:.0f} KB, {info['n_cells']} cells, "
                  f"tick={info['tick']}")

            # Load and swap
            loaded = load_colony(
                checkpoint_path,
                cell_class=Cell,
                colony_class=GrowingColony,
                make_organism_fn=make_organism,
                signal_class=Signal,
            )
            assert loaded.tick == pre_tick, f"tick mismatch"
            assert len(loaded.cells) == pre_cells, f"cell count mismatch"
            colony.colony = loaded
            print(f"  VERIFIED: loaded {pre_cells} cells at tick {pre_tick}")

    # Final report
    colony.learning_report()
    colony.learning_verdict()

    # Save final state
    from gaia.network.checkpoint import save_colony
    final_path = Path(_root) / "checkpoints" / "colony_language_final.pt"
    save_colony(colony.colony, final_path)
    print(f"\n  Final checkpoint: {final_path.name}, tick={colony.colony.tick}")


if __name__ == "__main__":
    main()
