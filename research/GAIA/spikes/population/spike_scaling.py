"""Spike Scaling Series — What scales character prediction?

Four experiments testing different scaling axes:
  A. BASELINE: Pure n-gram frequency tables (no field). What's the ceiling?
  B. MEMORY:   Sliding window context (2,3,4,5-gram). Longer context = better?
  C. SCALE:    Larger field (16x16=256 nodes). More capacity = better?
  D. DEPTH:    2-level hierarchy. Structure = better?

Each experiment processes the same corpus, same 20 epochs, reports accuracy.
No evolution, no hand-tuned weights. Just: what mechanism scales?

Usage:
    cd dawn-models/research/GAIA/spikes/population
    PYTHONIOENCODING=utf-8 PYTHONPATH="../../src;path/to/fracton" python spike_scaling.py
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, Counter

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
#  Corpus (same as spike_k_field)
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
#  EXPERIMENT A: Pure N-gram Baseline
# ==========================================================================

def experiment_baseline():
    """Pure frequency counting. No field, no physics. Just: what's the ceiling?

    Tests bigram (2), trigram (3), 4-gram, 5-gram, and 6-gram prediction.
    Each model: build frequency table from first pass, predict on subsequent passes.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT A: Pure N-gram Baseline (no field)")
    print("=" * 70)

    results = {}

    for n in [2, 3, 4, 5, 6]:
        # Build frequency table from one pass
        table: dict[str, Counter] = defaultdict(Counter)
        for i in range(len(CORPUS) - n):
            context = CORPUS[i:i + n - 1]  # n-1 chars of context
            target = CORPUS[i + n - 1]     # next char
            table[context][target] += 1

        # For each context, find the most common successor
        predictions: dict[str, str] = {}
        for context, counts in table.items():
            predictions[context] = counts.most_common(1)[0][0]

        # Evaluate over 20 "epochs" (same corpus repeated)
        epoch_accs = []
        char_hits: dict[str, list[int]] = defaultdict(list)

        for epoch in range(20):
            hits = 0
            total = 0
            for i in range(len(CORPUS) - n):
                context = CORPUS[i:i + n - 1]
                actual = CORPUS[i + n - 1]
                pred = predictions.get(context, ' ')
                hit = 1 if pred == actual else 0
                hits += hit
                total += 1
                char_hits[actual].append(hit)

            acc = hits / total if total > 0 else 0
            epoch_accs.append(acc)

        avg = sum(epoch_accs) / len(epoch_accs)
        results[n] = avg

        # Count unique predictions and coverage
        unique_preds = len(set(predictions.values()))
        n_contexts = len(predictions)

        # Per-char accuracy
        char_acc = {}
        for ch, hits_list in char_hits.items():
            if len(hits_list) >= 20:
                char_acc[ch] = sum(hits_list) / len(hits_list)

        chars_predicted = sum(1 for a in char_acc.values() if a > 0)

        print(f"\n  {n}-gram: {avg:.1%} accuracy")
        print(f"    Contexts: {n_contexts} | Unique predictions: {unique_preds}")
        print(f"    Chars with hits: {chars_predicted}/{len(char_acc)}")

        # Top 5 char accuracies
        best = sorted(char_acc.items(), key=lambda x: -x[1])[:5]
        for ch, a in best:
            print(f"      '{ch}': {a:.0%}")

    print(f"\n  SUMMARY:")
    for n, acc in sorted(results.items()):
        bar = "#" * int(acc * 100)
        print(f"    {n}-gram: {acc:5.1%} | {bar}")

    return results


# ==========================================================================
#  FastCodebook (shared across field experiments)
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


# ==========================================================================
#  EXPERIMENT B: Memory Depth — How much context helps?
# ==========================================================================

def experiment_memory():
    """Field with varying context window sizes.

    Same 8x8 field, but prediction uses last K characters as context.
    Tests K = 1 (bigram), 2 (trigram), 3 (4-gram), 4 (5-gram).
    No evolution — just context-dependent frequency counting via the field.

    Key question: does the field add anything over pure n-gram counting?
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT B: Field with Varying Context Depth")
    print("=" * 70)

    torch.manual_seed(42)
    codebook = FastCodebook()
    chars = sorted(set(CORPUS))
    n_chars = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}

    results = {}

    for ctx_depth in [1, 2, 3, 4]:
        n_gram = ctx_depth + 1

        # Build field: 8x8, 64 nodes
        n_nodes = 64
        node_chars = [chars[i % n_chars] for i in range(n_nodes)]
        char_indices: dict[str, list[int]] = defaultdict(list)
        for i, ch in enumerate(node_chars):
            char_indices[ch].append(i)

        # Build voices
        voices = torch.stack([codebook.encode(node_chars[i]).to(DEVICE) for i in range(n_nodes)])
        axes = voices.clone()

        # Context-dependent coupling: [n_nodes, n_chars^ctx_depth * n_chars]
        n_context_keys = n_chars ** ctx_depth
        n_genome = n_context_keys * n_chars

        # Cap genome size — for ctx_depth >= 3, use hash-based bucketing
        MAX_BUCKETS = 2048
        use_hashing = n_context_keys > MAX_BUCKETS
        if use_hashing:
            n_context_keys = MAX_BUCKETS
            n_genome = n_context_keys * n_chars

        char_coupling = torch.zeros(n_nodes, n_genome, device=DEVICE)

        def context_key(context: list[str]) -> int:
            """Convert context chars to a single index."""
            if use_hashing:
                h = hash(tuple(context)) % MAX_BUCKETS
                return h
            key = 0
            for ch in context:
                key = key * n_chars + char_to_idx.get(ch, 0)
            return key

        # Process corpus
        epoch_accs = []
        context_buf: list[str] = []

        for epoch in range(20):
            hits = 0
            total = 0
            context_buf.clear()

            for i in range(len(CORPUS) - 1):
                ch = CORPUS[i]
                nxt = CORPUS[i + 1]

                # Determine which nodes fire (top-k by similarity)
                inp = codebook.encode(ch).to(DEVICE)

                # Crystal filter
                dots = (inp.unsqueeze(0).expand(n_nodes, -1) * axes).sum(dim=-1, keepdim=True)
                projs = dots * axes
                orths = inp.unsqueeze(0).expand(n_nodes, -1) - projs
                filtered = projs + PHI_INV * orths

                sims = F.cosine_similarity(voices, filtered, dim=1)
                k = max(1, int(n_nodes * PHI_INV * XI_SEC))
                threshold = float(sims.topk(min(k, n_nodes)).values[-1])
                threshold = max(threshold, XI_SEC)
                fire_mask = sims >= threshold
                fired_indices = fire_mask.nonzero(as_tuple=True)[0]

                if len(context_buf) >= ctx_depth and fired_indices.numel() > 0:
                    ctx_key = context_key(context_buf[-ctx_depth:])
                    ctx_start = ctx_key * n_chars

                    # Predict
                    votes: dict[str, float] = defaultdict(float)
                    for idx in fired_indices:
                        node_i = int(idx)
                        row = char_coupling[node_i, ctx_start:ctx_start + n_chars]
                        if float(row.sum()) > 0:
                            best = int(row.argmax())
                            votes[chars[best]] += float(row[best])

                    if votes:
                        pred = max(votes, key=votes.get)
                    else:
                        pred = ' '

                    hit = 1 if pred == nxt else 0
                    hits += hit
                    total += 1

                    # Learn
                    nxt_idx = char_to_idx.get(nxt, 0)
                    for idx in fired_indices:
                        char_coupling[int(idx), ctx_start + nxt_idx] += 1.0

                context_buf.append(ch)

            acc = hits / total if total > 0 else 0
            epoch_accs.append(acc)

        avg_second_half = sum(epoch_accs[10:]) / 10
        peak = max(epoch_accs)
        results[ctx_depth] = (avg_second_half, peak, epoch_accs)

        genome_kb = n_nodes * n_genome * 4 / 1024  # float32
        print(f"\n  Context depth {ctx_depth} ({n_gram}-gram equivalent):")
        print(f"    Genome: {n_genome} per node ({genome_kb:.0f} KB total)")
        print(f"    {'Hashed' if use_hashing else 'Exact'} context keys: {n_context_keys}")
        print(f"    Second half avg: {avg_second_half:.1%} | Peak: {peak:.1%}")
        print(f"    Learning curve: ", end="")
        for i, a in enumerate(epoch_accs):
            if i % 5 == 0:
                print(f"E{i+1}:{a:.0%}", end=" ")
        print()

    print(f"\n  SUMMARY:")
    for ctx, (avg, peak, _) in sorted(results.items()):
        bar = "#" * int(avg * 100)
        print(f"    ctx={ctx} ({ctx+1}-gram): avg={avg:5.1%} peak={peak:5.1%} | {bar}")

    return results


# ==========================================================================
#  EXPERIMENT C: Scale — Does Larger Field Help?
# ==========================================================================

def experiment_scale():
    """Same mechanism, different field sizes.

    Tests 4x4 (16), 8x8 (64), 12x12 (144), 16x16 (256) nodes.
    All use context depth 1 (trigram equivalent) for fair comparison.
    Key question: does more nodes per character improve accuracy?
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT C: Field Scale (same mechanism, different sizes)")
    print("=" * 70)

    torch.manual_seed(42)
    codebook = FastCodebook()
    chars = sorted(set(CORPUS))
    n_chars = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}

    results = {}

    for field_size in [4, 8, 12, 16]:
        n_nodes = field_size * field_size
        node_chars = [chars[i % n_chars] for i in range(n_nodes)]

        voices = torch.stack([codebook.encode(node_chars[i]).to(DEVICE) for i in range(n_nodes)])
        axes = voices.clone()

        # Context depth 1 (trigram equivalent): [n_nodes, n_chars * n_chars]
        n_genome = n_chars * n_chars
        char_coupling = torch.zeros(n_nodes, n_genome, device=DEVICE)

        prev_char: str | None = None
        epoch_accs = []

        t0 = time.time()

        for epoch in range(20):
            hits = 0
            total = 0
            prev_char = None

            for i in range(len(CORPUS) - 1):
                ch = CORPUS[i]
                nxt = CORPUS[i + 1]
                inp = codebook.encode(ch).to(DEVICE)

                # Crystal filter + actualize
                inp_exp = inp.unsqueeze(0).expand(n_nodes, -1)
                dots = (inp_exp * axes).sum(dim=-1, keepdim=True)
                projs = dots * axes
                filtered = projs + PHI_INV * (inp_exp - projs)

                sims = F.cosine_similarity(voices, filtered, dim=1)
                k = max(1, int(n_nodes * PHI_INV * XI_SEC))
                threshold = float(sims.topk(min(k, n_nodes)).values[-1])
                threshold = max(threshold, XI_SEC)
                fire_mask = sims >= threshold
                fired_indices = fire_mask.nonzero(as_tuple=True)[0]

                if prev_char is not None and fired_indices.numel() > 0:
                    prev_idx = char_to_idx.get(prev_char, 0)
                    ctx_start = prev_idx * n_chars

                    # Predict
                    votes: dict[str, float] = defaultdict(float)
                    for idx in fired_indices:
                        node_i = int(idx)
                        row = char_coupling[node_i, ctx_start:ctx_start + n_chars]
                        if float(row.sum()) > 0:
                            best = int(row.argmax())
                            votes[chars[best]] += float(row[best])

                    if votes:
                        pred = max(votes, key=votes.get)
                    else:
                        pred = ' '

                    hit = 1 if pred == nxt else 0
                    hits += hit
                    total += 1

                    # Learn
                    nxt_idx = char_to_idx.get(nxt, 0)
                    for idx in fired_indices:
                        char_coupling[int(idx), ctx_start + nxt_idx] += 1.0

                prev_char = ch

            acc = hits / total if total > 0 else 0
            epoch_accs.append(acc)

        elapsed = time.time() - t0
        avg_second_half = sum(epoch_accs[10:]) / 10
        peak = max(epoch_accs)
        nodes_per_char = n_nodes / n_chars
        results[field_size] = (avg_second_half, peak, elapsed)

        print(f"\n  {field_size}x{field_size} = {n_nodes} nodes ({nodes_per_char:.1f}/char):")
        print(f"    k (fire count): {max(1, int(n_nodes * PHI_INV * XI_SEC))}")
        print(f"    Second half avg: {avg_second_half:.1%} | Peak: {peak:.1%}")
        print(f"    Time: {elapsed:.1f}s ({len(CORPUS) * 20 / elapsed:.0f} ch/s)")

    print(f"\n  SUMMARY:")
    for fs, (avg, peak, elapsed) in sorted(results.items()):
        n = fs * fs
        bar = "#" * int(avg * 100)
        print(f"    {fs}x{fs}={n:3d}: avg={avg:5.1%} peak={peak:5.1%} ({elapsed:.0f}s) | {bar}")

    return results


# ==========================================================================
#  EXPERIMENT D: Depth — Does MED Hierarchy Help?
# ==========================================================================

def experiment_depth():
    """Two-level field hierarchy. MED depth = 2.

    Level 1: 8x8 = 64 character nodes (same as before)
    Level 2: 4x4 = 16 cluster nodes, each aggregates 4 character nodes

    Prediction: L1 nodes vote, L2 clusters modulate/override.
    Key question: does hierarchical structure improve over flat?
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT D: MED Depth=2 Hierarchy")
    print("=" * 70)

    torch.manual_seed(42)
    codebook = FastCodebook()
    chars = sorted(set(CORPUS))
    n_chars = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}

    # Level 1: 64 character nodes
    n_l1 = 64
    node_chars = [chars[i % n_chars] for i in range(n_l1)]
    voices_l1 = torch.stack([codebook.encode(node_chars[i]).to(DEVICE) for i in range(n_l1)])
    axes_l1 = voices_l1.clone()

    # Level 2: 16 cluster nodes, each owns 4 L1 nodes
    n_l2 = 16
    cluster_members = [list(range(i * 4, (i + 1) * 4)) for i in range(n_l2)]

    # L1: context-dependent coupling [n_l1, n_chars * n_chars]
    n_genome = n_chars * n_chars
    char_coupling_l1 = torch.zeros(n_l1, n_genome, device=DEVICE)

    # L2: cluster-level coupling [n_l2, n_chars * n_chars]
    # Aggregated view — clusters track broader patterns
    char_coupling_l2 = torch.zeros(n_l2, n_genome, device=DEVICE)

    # L2 voices: mean of member voices
    voices_l2 = torch.stack([voices_l1[members].mean(dim=0) for members in cluster_members])
    voices_l2 = voices_l2 / (torch.norm(voices_l2, dim=1, keepdim=True) + 1e-8)

    results = {}

    for blend_mode in ["l1_only", "l2_only", "l1_l2_blend"]:
        # Reset couplings
        char_coupling_l1.zero_()
        char_coupling_l2.zero_()

        prev_char: str | None = None
        epoch_accs = []

        t0 = time.time()

        for epoch in range(20):
            hits = 0
            total = 0
            prev_char = None

            for i in range(len(CORPUS) - 1):
                ch = CORPUS[i]
                nxt = CORPUS[i + 1]
                inp = codebook.encode(ch).to(DEVICE)

                # L1 actualize
                inp_exp = inp.unsqueeze(0).expand(n_l1, -1)
                dots = (inp_exp * axes_l1).sum(dim=-1, keepdim=True)
                projs = dots * axes_l1
                filtered = projs + PHI_INV * (inp_exp - projs)
                sims = F.cosine_similarity(voices_l1, filtered, dim=1)

                k = max(1, int(n_l1 * PHI_INV * XI_SEC))
                threshold = float(sims.topk(min(k, n_l1)).values[-1])
                threshold = max(threshold, XI_SEC)
                fire_mask = sims >= threshold
                fired_indices = fire_mask.nonzero(as_tuple=True)[0]

                # L2: which clusters have fired members?
                cluster_fire = torch.zeros(n_l2, dtype=torch.bool, device=DEVICE)
                for c_idx, members in enumerate(cluster_members):
                    if fire_mask[members].any():
                        cluster_fire[c_idx] = True

                if prev_char is not None and fired_indices.numel() > 0:
                    prev_idx = char_to_idx.get(prev_char, 0)
                    ctx_start = prev_idx * n_chars
                    nxt_idx = char_to_idx.get(nxt, 0)

                    # L1 votes
                    l1_votes: dict[str, float] = defaultdict(float)
                    for idx in fired_indices:
                        node_i = int(idx)
                        row = char_coupling_l1[node_i, ctx_start:ctx_start + n_chars]
                        if float(row.sum()) > 0:
                            best = int(row.argmax())
                            l1_votes[chars[best]] += float(row[best])

                    # L2 votes
                    l2_votes: dict[str, float] = defaultdict(float)
                    fired_clusters = cluster_fire.nonzero(as_tuple=True)[0]
                    for idx in fired_clusters:
                        c_idx = int(idx)
                        row = char_coupling_l2[c_idx, ctx_start:ctx_start + n_chars]
                        if float(row.sum()) > 0:
                            best = int(row.argmax())
                            l2_votes[chars[best]] += float(row[best])

                    # Blend
                    if blend_mode == "l1_only":
                        votes = l1_votes
                    elif blend_mode == "l2_only":
                        votes = l2_votes
                    else:  # l1_l2_blend
                        votes = defaultdict(float)
                        for ch_v, score in l1_votes.items():
                            votes[ch_v] += score
                        for ch_v, score in l2_votes.items():
                            votes[ch_v] += score * PHI_INV  # L2 has less weight

                    if votes:
                        pred = max(votes, key=votes.get)
                    else:
                        pred = ' '

                    hit = 1 if pred == nxt else 0
                    hits += hit
                    total += 1

                    # Learn — both levels
                    for idx in fired_indices:
                        char_coupling_l1[int(idx), ctx_start + nxt_idx] += 1.0
                    for idx in fired_clusters:
                        char_coupling_l2[int(idx), ctx_start + nxt_idx] += 1.0

                prev_char = ch

            acc = hits / total if total > 0 else 0
            epoch_accs.append(acc)

        elapsed = time.time() - t0
        avg_second_half = sum(epoch_accs[10:]) / 10
        peak = max(epoch_accs)
        results[blend_mode] = (avg_second_half, peak, elapsed)

        print(f"\n  {blend_mode}:")
        print(f"    Second half avg: {avg_second_half:.1%} | Peak: {peak:.1%}")
        print(f"    Time: {elapsed:.1f}s")

    print(f"\n  SUMMARY:")
    for mode, (avg, peak, elapsed) in results.items():
        bar = "#" * int(avg * 100)
        print(f"    {mode:15s}: avg={avg:5.1%} peak={peak:5.1%} | {bar}")

    return results


# ==========================================================================
#  Main — Run all experiments
# ==========================================================================

def main():
    print("=" * 70)
    print("  SPIKE SCALING SERIES")
    print(f"  Corpus: {len(CORPUS)} chars | 20 epochs each")
    print(f"  Question: What mechanism scales character prediction?")
    print("=" * 70)

    t_total = time.time()

    baseline = experiment_baseline()
    memory = experiment_memory()
    scale = experiment_scale()
    depth = experiment_depth()

    elapsed = time.time() - t_total

    print("\n" + "=" * 70)
    print("  GRAND SUMMARY")
    print("=" * 70)

    print(f"\n  A. BASELINE (pure n-gram, no field):")
    for n, acc in sorted(baseline.items()):
        print(f"      {n}-gram: {acc:.1%}")

    print(f"\n  B. MEMORY DEPTH (field + varying context):")
    for ctx, (avg, peak, _) in sorted(memory.items()):
        print(f"      ctx={ctx} ({ctx+1}-gram): {avg:.1%} (peak {peak:.1%})")

    print(f"\n  C. SCALE (varying field size, ctx=1):")
    for fs, (avg, peak, _) in sorted(scale.items()):
        print(f"      {fs}x{fs}={fs*fs:3d}: {avg:.1%} (peak {peak:.1%})")

    print(f"\n  D. DEPTH (MED hierarchy, 2 levels):")
    for mode, (avg, peak, _) in depth.items():
        print(f"      {mode:15s}: {avg:.1%} (peak {peak:.1%})")

    print(f"\n  Spike K Evo field (for reference): 44.1% peak")
    print(f"  Pre-evo engineered field:          47.2% peak")

    print(f"\n  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
