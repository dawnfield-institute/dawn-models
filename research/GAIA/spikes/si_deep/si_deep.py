"""
SI Deep Spike — Beyond Co-occurrence: Relational Intelligence
=============================================================

The question: is GAIA actually thinking, or is it a fancy bag of words?

si_voice proved the architecture (GAIA thinks, LLM speaks) but used
shallow co-occurrence counting. This spike adds:

1. RELATIONAL LEARNING: Subject-relation-object triples, not just word proximity
   - "Darwin proposed natural selection" -> (darwin, proposed, natural_selection)
   - Directional: "Darwin proposed X" != "X proposed Darwin"
   - Enables answering WHO/WHAT/WHEN/WHY questions

2. FACTUAL Q&A: Right/wrong answers, not keyword overlap
   - Questions with definitive correct answers
   - Binary scoring: did the output contain the correct answer?
   - Much harder than keyword hit rate

3. THREE-WAY COMPARISON:
   - Raw LLM (no GAIA)
   - GAIA co-occurrence (si_voice level)
   - GAIA relational (this spike)

Usage:
  python si_deep.py                    # full eval
  python si_deep.py --model gpt2-medium  # stronger voice
"""

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ─── DFT Constants ───

XI_SEC = 0.0618
PHI_INV = 0.618
LAMBDA_STAR = 0.9816


# ─── Relational Entity ───

@dataclass
class Relation:
    """A directed relationship between concepts."""
    subject: str
    predicate: str
    obj: str
    count: int = 0
    confidence: float = 0.0

    def key(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.obj)


@dataclass
class ConceptNode:
    """A concept in GAIA's knowledge graph."""
    name: str
    activation_count: int = 0
    # Outgoing relations: predicate -> [(object, count)]
    outgoing: dict[str, list[tuple[str, int]]] = field(default_factory=lambda: defaultdict(list))
    # Incoming relations: predicate -> [(subject, count)]
    incoming: dict[str, list[tuple[str, int]]] = field(default_factory=lambda: defaultdict(list))
    # Co-occurrence (from si_voice level)
    cooccurrence: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # SEC phase
    phase: str = "chaotic"
    entropy: float = 1.0

    def update_phase(self):
        """Classify phase from relation diversity."""
        total_rels = sum(c for pairs in self.outgoing.values() for _, c in pairs)
        total_rels += sum(c for pairs in self.incoming.values() for _, c in pairs)
        if total_rels < 3:
            self.phase = "chaotic"
            self.entropy = 1.0
            return

        # Entropy of relation distribution
        all_counts = []
        for pairs in self.outgoing.values():
            all_counts.extend(c for _, c in pairs)
        for pairs in self.incoming.values():
            all_counts.extend(c for _, c in pairs)

        total = sum(all_counts)
        probs = [c / total for c in all_counts if c > 0]
        if len(probs) <= 1:
            self.entropy = 0.0
            self.phase = "crystallized"
            return

        self.entropy = -sum(p * math.log(p) for p in probs) / max(math.log(len(probs)), 0.01)
        if self.entropy < XI_SEC:
            self.phase = "crystallized"
        elif self.entropy < PHI_INV:
            self.phase = "ordered"
        elif self.entropy < LAMBDA_STAR:
            self.phase = "transitional"
        else:
            self.phase = "chaotic"


# ─── Relation Extraction ───

# Simple pattern-based relation extraction. Not NLP-heavy on purpose —
# the point is that even crude extraction gives GAIA relational structure
# that co-occurrence alone can't.

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "must", "need",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "and", "or", "but", "nor", "not", "so", "yet", "both", "either",
    "that", "which", "who", "whom", "whose", "what", "where", "when",
    "this", "these", "those", "it", "its", "he", "she", "they", "them",
    "his", "her", "their", "my", "your", "our", "i", "we", "you",
    "also", "very", "just", "even", "only", "then", "than", "more",
    "most", "each", "every", "all", "many", "much", "some", "any",
    "no", "if", "as", "up", "out", "how",
}


def extract_relations(text: str) -> list[tuple[str, str, str]]:
    """Extract (subject, predicate, object) triples from text.

    Uses sentence-level context to find the real subject, not just
    the word immediately before the verb.
    """
    triples = []
    sentences = re.split(r'[.!?]+', text)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Find proper noun subjects: capitalized words at start or after period
        # This catches "Charles Darwin proposed..." -> subject = "darwin"
        words = sentence.split()
        if not words:
            continue

        # Build subject from leading capitalized words (proper nouns)
        subject_words = []
        for w in words:
            if w[0].isupper() or w.lower() in ("and", "y"):
                subject_words.append(w)
            else:
                break

        # If no leading caps, use first non-stopword
        if not subject_words:
            for w in words:
                if w.lower() not in STOPWORDS:
                    subject_words = [w]
                    break

        if not subject_words:
            continue

        default_subject = " ".join(subject_words).lower()
        # Use last name if multi-word (e.g., "Charles Darwin" -> "darwin")
        subject_key = subject_words[-1].lower() if subject_words else ""
        if not subject_key or subject_key in STOPWORDS:
            subject_key = default_subject

        # Pattern matching with sentence-level subject
        patterns = [
            (r"(?:proposed|discovered|invented|developed|created|founded)\s+(?:the\s+)?(.+?)(?:\.|,|$)", "proposed"),
            (r"(?:is|are|was|were)\s+(?:known\s+as|called|named|termed)\s+(?:the\s+)?(.+?)(?:\.|,|$)", "known_as"),
            (r"(?:is|are|was|were)\s+(?:essential|crucial|important|necessary)\s+for\s+(.+?)(?:\.|,|$)", "essential_for"),
            (r"(?:contains?|has)\s+(?:approximately\s+)?(.+?)(?:\.|,|$)", "has"),
            (r"(?:published)\s+(.+?)(?:\.|,|$)", "published"),
            (r"(?:uses?|employs?|utilizes?)\s+(.+?)(?:\.|,|$)", "uses"),
            (r"(?:converts?)\s+(.+?)\s+into\s+(.+?)(?:\.|,|$)", "converts"),
            (r"(?:is|are)\s+(?:a|an|the)\s+(\w+(?:\s+\w+){0,3})(?:\.|,|$)", "is_a"),
            (r"(?:acts?\s+on|operates?\s+on)\s+(.+?)(?:\.|,|$)", "acts_on"),
        ]

        for pattern, predicate in patterns:
            for match in re.finditer(pattern, sentence, re.IGNORECASE):
                if predicate == "converts":
                    obj = match.group(2).lower().strip()
                else:
                    obj = match.group(1).lower().strip()
                obj = re.sub(r'^(a|an|the)\s+', '', obj)
                obj_words = obj.split()[:5]
                obj = ' '.join(obj_words)
                if obj and subject_key != obj and subject_key not in STOPWORDS:
                    triples.append((subject_key, predicate, obj))

    return triples


# ─── GAIA Deep Core ───

class GAIADeepCore:
    """Relational intelligence — learns structure, not just proximity.

    Maintains both:
    - Co-occurrence graph (si_voice level)
    - Relational graph (subject-predicate-object triples)

    The relational graph enables directional reasoning:
    "Who proposed X?" activates incoming(proposed, X)
    "What did Y discover?" activates outgoing(proposed, Y)
    """

    def __init__(self, window: int = 5):
        self.window = window
        self.concepts: dict[str, ConceptNode] = {}
        self.relations: list[Relation] = []
        self.relation_index: dict[tuple[str, str, str], Relation] = {}
        self.words_processed = 0

    def _get_or_create(self, name: str) -> ConceptNode:
        if name not in self.concepts:
            self.concepts[name] = ConceptNode(name=name)
        return self.concepts[name]

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|[0-9]+", text.lower())

    def learn(self, text: str):
        """Learn from text: extract relations AND build co-occurrence."""
        words = self._tokenize(text)

        # 1. Co-occurrence (same as si_voice)
        for i, word in enumerate(words):
            concept = self._get_or_create(word)
            concept.activation_count += 1
            window_start = max(0, i - self.window)
            for j in range(window_start, i):
                other = words[j]
                concept.cooccurrence[other] += 1
                self._get_or_create(other).cooccurrence[word] += 1
            self.words_processed += 1

        # 2. Relational extraction
        triples = extract_relations(text)
        for subj, pred, obj in triples:
            key = (subj, pred, obj)
            if key in self.relation_index:
                self.relation_index[key].count += 1
            else:
                rel = Relation(subject=subj, predicate=pred, obj=obj, count=1)
                self.relations.append(rel)
                self.relation_index[key] = rel

            # Update concept nodes
            subj_node = self._get_or_create(subj)
            obj_node = self._get_or_create(obj)

            # Update outgoing/incoming
            self._add_relation_link(subj_node.outgoing, pred, obj)
            self._add_relation_link(obj_node.incoming, pred, subj)

        # Update phases
        for concept in self.concepts.values():
            concept.update_phase()

    def _add_relation_link(self, links: dict, predicate: str, target: str):
        """Add or increment a relation link."""
        for i, (t, c) in enumerate(links[predicate]):
            if t == target:
                links[predicate][i] = (t, c + 1)
                return
        links[predicate].append((target, 1))

    def query_relations(self, concept: str, direction: str = "both",
                        predicate: str = None) -> list[tuple[str, str, str, int]]:
        """Query the relational graph.

        Returns: [(subject, predicate, object, count), ...]
        """
        results = []
        concept_lower = concept.lower()

        # Check exact match and partial matches
        matching_concepts = []
        if concept_lower in self.concepts:
            matching_concepts.append(concept_lower)
        # Also check multi-word concept names
        for name in self.concepts:
            if concept_lower in name or name in concept_lower:
                if name not in matching_concepts:
                    matching_concepts.append(name)

        for name in matching_concepts:
            node = self.concepts[name]

            if direction in ("out", "both"):
                for pred, pairs in node.outgoing.items():
                    if predicate and pred != predicate:
                        continue
                    for obj, count in pairs:
                        results.append((name, pred, obj, count))

            if direction in ("in", "both"):
                for pred, pairs in node.incoming.items():
                    if predicate and pred != predicate:
                        continue
                    for subj, count in pairs:
                        results.append((subj, pred, name, count))

        results.sort(key=lambda x: -x[3])
        return results

    def format_relational_intent(self, query: str) -> str:
        """Format GAIA's relational knowledge into a directive preamble.

        Unlike si_voice's co-occurrence intent, this encodes STRUCTURE:
        who did what to whom, what causes what, what is part of what.
        """
        words = self._tokenize(query)

        # Only query content words (skip stopwords)
        content_words = [w for w in words if w not in STOPWORDS]
        if not content_words:
            content_words = words[:3]

        # Gather relations for content words only
        all_rels = []
        for word in content_words:
            rels = self.query_relations(word)
            all_rels.extend(rels)

        if not all_rels:
            return self._format_cooccurrence_intent(words)

        # Deduplicate and sort by count
        seen = set()
        unique_rels = []
        for r in all_rels:
            key = (r[0], r[1], r[2])
            if key not in seen:
                seen.add(key)
                unique_rels.append(r)
        unique_rels.sort(key=lambda x: -x[3])

        # Build structured preamble
        facts = []
        for subj, pred, obj, count in unique_rels[:8]:
            if pred == "is_a":
                facts.append(f"{subj} is {obj}")
            elif pred == "proposed":
                facts.append(f"{subj} proposed {obj}")
            elif pred == "has":
                facts.append(f"{subj} has {obj}")
            elif pred == "essential_for":
                facts.append(f"{subj} is essential for {obj}")
            elif pred == "causes":
                facts.append(f"{subj} causes {obj}")
            elif pred == "uses":
                facts.append(f"{subj} uses {obj}")
            elif pred == "known_as":
                facts.append(f"{subj} is known as {obj}")
            elif pred == "acts_on":
                facts.append(f"{subj} acts on {obj}")
            else:
                facts.append(f"{subj} {pred} {obj}")

        if not facts:
            return self._format_cooccurrence_intent(words)

        # Determine phase for directive strength
        phases = [self.concepts[w].phase for w in words if w in self.concepts]
        if phases.count("crystallized") > len(phases) // 2:
            prefix = "The following facts are well established"
        elif phases.count("ordered") > len(phases) // 3:
            prefix = "The following is known"
        else:
            prefix = "The following discusses"

        fact_str = ". ".join(facts[:6])
        return f"{prefix}: {fact_str}."

    def _format_cooccurrence_intent(self, words: list[str]) -> str:
        """Fallback: co-occurrence based intent (si_voice level)."""
        predictions = defaultdict(float)
        for word in words:
            if word in self.concepts:
                concept = self.concepts[word]
                total = sum(concept.cooccurrence.values())
                if total > 0:
                    for co_word, count in concept.cooccurrence.items():
                        conf = count / total
                        predictions[co_word] += conf

        if not predictions:
            return ""

        top = sorted(predictions.items(), key=lambda x: -x[1])[:8]
        concepts_str = ", ".join(w for w, _ in top)
        return f"The following discusses {' '.join(words[:4])}. Key concepts: {concepts_str}."

    def stats(self) -> str:
        n_concepts = len(self.concepts)
        n_relations = len(self.relations)
        n_cryst = sum(1 for c in self.concepts.values() if c.phase == "crystallized")
        n_ordered = sum(1 for c in self.concepts.values() if c.phase == "ordered")
        total_rel_count = sum(r.count for r in self.relations)
        return (f"GAIA Deep: {n_concepts} concepts, {n_relations} unique relations "
                f"({total_rel_count} total), {self.words_processed} words\n"
                f"  Phases: {n_cryst} crystallized, {n_ordered} ordered, "
                f"{sum(1 for c in self.concepts.values() if c.phase == 'transitional')} transitional, "
                f"{sum(1 for c in self.concepts.values() if c.phase == 'chaotic')} chaotic")


# ─── LLM Voice (same as si_voice) ───

class LLMVoice:
    def __init__(self, model_name: str = "gpt2"):
        print(f"Loading LLM voice ({model_name})...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.model_name = model_name

    @torch.no_grad()
    def speak(self, prompt: str, max_tokens: int = 60,
              temperature: float = 0.7, top_k: int = 30) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    @torch.no_grad()
    def speak_guided(self, prompt: str, intent: str,
                     max_tokens: int = 60, temperature: float = 0.7,
                     top_k: int = 30) -> str:
        if intent:
            full_prompt = f"{intent}\n\n{prompt}"
        else:
            full_prompt = prompt
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt")
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        full_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if intent and full_text.startswith(intent):
            full_text = full_text[len(intent):].lstrip("\n")
        return full_text


# ─── Synthetic Intelligence Deep ───

class SyntheticIntelligenceDeep:
    """SI with relational learning. GAIA thinks in relations, LLM speaks."""

    def __init__(self, model_name: str = "gpt2"):
        self.core = GAIADeepCore(window=5)
        self.voice = LLMVoice(model_name)

    def learn(self, text: str, repeat: int = 1):
        for _ in range(repeat):
            self.core.learn(text)

    def feedback(self, correction: str, repeat: int = 3):
        for _ in range(repeat):
            self.core.learn(correction)

    def respond_raw(self, query: str, max_tokens: int = 60) -> str:
        """Raw LLM, no GAIA."""
        return self.voice.speak(query, max_tokens=max_tokens)

    def respond_cooccurrence(self, query: str, max_tokens: int = 60) -> str:
        """GAIA co-occurrence only (si_voice level)."""
        words = self.core._tokenize(query)
        intent = self.core._format_cooccurrence_intent(words)
        return self.voice.speak_guided(query, intent, max_tokens=max_tokens)

    def respond_relational(self, query: str, max_tokens: int = 60) -> str:
        """GAIA relational (full deep intelligence)."""
        intent = self.core.format_relational_intent(query)
        return self.voice.speak_guided(query, intent, max_tokens=max_tokens)


# ─── Knowledge Corpus ───

KNOWLEDGE_CORPUS = """
Charles Darwin proposed the theory of evolution by natural selection. Darwin traveled on
the HMS Beagle to the Galapagos Islands. Alfred Russel Wallace independently discovered
natural selection. Darwin published On the Origin of Species in 1859.

Gregor Mendel discovered the laws of inheritance through pea plant experiments. Mendel
is known as the father of genetics. DNA carries genetic instructions for all living organisms.
James Watson and Francis Crick discovered the structure of DNA in 1953. Rosalind Franklin
produced the X-ray crystallography images that were crucial for discovering DNA structure.

The human brain contains approximately 86 billion neurons. The prefrontal cortex is essential
for planning and decision making. The hippocampus is essential for memory formation. Santiago
Ramon y Cajal discovered that neurons are individual cells. Neurons communicate through
synaptic connections using neurotransmitters.

Isaac Newton proposed the laws of motion and universal gravitation. Newton published
Principia Mathematica in 1687. Albert Einstein proposed the theory of general relativity
in 1915. Einstein proposed the theory of special relativity in 1905. Quantum mechanics
was developed by Planck, Bohr, Heisenberg, and Schrodinger.

Machine learning uses algorithms that learn from data. Neural networks are inspired by
biological neurons. Geoffrey Hinton developed backpropagation for training neural networks.
Transformers use self-attention mechanisms. Attention was proposed by Vaswani in 2017.
Deep learning uses many layers of neural networks.

Photosynthesis converts sunlight into chemical energy in plants. Chloroplasts contain
chlorophyll which absorbs light. The mitochondria is known as the powerhouse of the cell.
ATP is the energy currency of cells. Cellular respiration converts glucose into ATP.
"""

# ─── Factual Q&A Eval ───

# Each question has:
# - query: the prompt to complete
# - correct: strings that MUST appear in a correct answer
# - wrong: strings that indicate a wrong answer
# - topic: what domain this tests

FACTUAL_QA = [
    {
        "query": "Darwin proposed",
        "correct": ["natural selection", "evolution"],
        "wrong": ["relativity", "gravity", "quantum"],
        "topic": "biology",
    },
    {
        "query": "Wallace independently discovered",
        "correct": ["natural selection"],
        "wrong": ["dna", "gravity", "relativity"],
        "topic": "biology",
    },
    {
        "query": "The Origin of Species was published in",
        "correct": ["1859"],
        "wrong": ["1687", "1905", "1915", "1953"],
        "topic": "biology",
    },
    {
        "query": "Mendel is known as the father of",
        "correct": ["genetics"],
        "wrong": ["physics", "chemistry", "evolution"],
        "topic": "biology",
    },
    {
        "query": "Watson and Crick discovered the structure of",
        "correct": ["dna"],
        "wrong": ["rna", "protein", "atom"],
        "topic": "biology",
    },
    {
        "query": "The hippocampus is essential for",
        "correct": ["memory"],
        "wrong": ["vision", "hearing", "digestion"],
        "topic": "neuroscience",
    },
    {
        "query": "The prefrontal cortex handles",
        "correct": ["planning", "decision"],
        "wrong": ["vision", "hearing", "breathing"],
        "topic": "neuroscience",
    },
    {
        "query": "Newton published Principia Mathematica in",
        "correct": ["1687"],
        "wrong": ["1859", "1905", "1915"],
        "topic": "physics",
    },
    {
        "query": "Einstein proposed general relativity in",
        "correct": ["1915"],
        "wrong": ["1687", "1859", "1905"],
        "topic": "physics",
    },
    {
        "query": "Backpropagation for neural networks was developed by",
        "correct": ["hinton"],
        "wrong": ["einstein", "darwin", "newton"],
        "topic": "ml",
    },
    {
        "query": "The attention mechanism was proposed by",
        "correct": ["vaswani"],
        "wrong": ["hinton", "darwin", "einstein"],
        "topic": "ml",
    },
    {
        "query": "The mitochondria is known as the",
        "correct": ["powerhouse"],
        "wrong": ["brain", "nucleus", "membrane"],
        "topic": "biology",
    },
    {
        "query": "Rosalind Franklin produced",
        "correct": ["x-ray", "crystallography", "dna"],
        "wrong": ["evolution", "gravity", "relativity"],
        "topic": "biology",
    },
    {
        "query": "Ramon y Cajal discovered that neurons are",
        "correct": ["individual", "cells"],
        "wrong": ["connected", "continuous", "fluid"],
        "topic": "neuroscience",
    },
    {
        "query": "Photosynthesis converts sunlight into",
        "correct": ["chemical energy", "energy"],
        "wrong": ["heat", "sound", "electricity"],
        "topic": "biology",
    },
]


def score_answer(output: str, correct: list[str], wrong: list[str]) -> tuple[bool, str]:
    """Score a generated answer as correct/wrong/neutral.

    Returns: (is_correct, reason)
    """
    output_lower = output.lower()

    has_correct = any(c.lower() in output_lower for c in correct)
    has_wrong = any(w.lower() in output_lower for w in wrong)

    if has_correct and not has_wrong:
        return True, "correct"
    elif has_correct and has_wrong:
        return True, "correct (with noise)"
    elif has_wrong and not has_correct:
        return False, "wrong"
    else:
        return False, "no answer"


def run_deep_eval(model_name: str = "gpt2"):
    """Three-way factual Q&A comparison: raw LLM vs co-occurrence vs relational."""

    si = SyntheticIntelligenceDeep(model_name)

    print(f"\n{'=' * 80}")
    print("SI DEEP EVAL: FACTUAL Q&A — RAW vs CO-OCCURRENCE vs RELATIONAL")
    print(f"{'=' * 80}")

    # Train on knowledge corpus
    print(f"\nTraining on knowledge corpus ({len(KNOWLEDGE_CORPUS)} chars)...")
    si.learn(KNOWLEDGE_CORPUS, repeat=5)
    print(si.core.stats())

    # Show extracted relations
    print(f"\nTop relations extracted:")
    sorted_rels = sorted(si.core.relations, key=lambda r: -r.count)
    for rel in sorted_rels[:15]:
        print(f"  ({rel.subject}, {rel.predicate}, {rel.obj}) x{rel.count}")

    n_samples = 3  # samples per question per method

    # Results: method -> list of (is_correct, topic)
    results = {"raw": [], "cooccurrence": [], "relational": []}

    print(f"\n{'=' * 80}")
    print(f"{'Question':<45} {'Raw':>5} {'CoOc':>5} {'Rel':>5}")
    print(f"{'=' * 80}")

    for qa in FACTUAL_QA:
        query = qa["query"]
        correct = qa["correct"]
        wrong = qa["wrong"]
        topic = qa["topic"]

        # Score each method over n_samples
        method_scores = {}
        for method_name in ["raw", "cooccurrence", "relational"]:
            correct_count = 0
            for _ in range(n_samples):
                if method_name == "raw":
                    out = si.respond_raw(query, max_tokens=40)
                elif method_name == "cooccurrence":
                    out = si.respond_cooccurrence(query, max_tokens=40)
                else:
                    out = si.respond_relational(query, max_tokens=40)

                is_correct, reason = score_answer(out[len(query):], correct, wrong)
                if is_correct:
                    correct_count += 1

            accuracy = correct_count / n_samples
            method_scores[method_name] = accuracy
            results[method_name].append((accuracy > 0.5, topic))

        # Print row
        raw_s = f"{method_scores['raw']:.0%}"
        cooc_s = f"{method_scores['cooccurrence']:.0%}"
        rel_s = f"{method_scores['relational']:.0%}"

        # Mark wins
        best = max(method_scores.values())
        if method_scores['relational'] == best and best > 0:
            rel_s = f"{rel_s} *"
        elif method_scores['cooccurrence'] == best and best > 0:
            cooc_s = f"{cooc_s} *"

        print(f"  {query:<43} {raw_s:>5} {cooc_s:>5} {rel_s:>6}")

    # ─── Summary ───
    print(f"\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 80}")

    for method in ["raw", "cooccurrence", "relational"]:
        correct_count = sum(1 for is_correct, _ in results[method] if is_correct)
        total = len(results[method])
        accuracy = correct_count / total

        # By topic
        topic_results = defaultdict(lambda: [0, 0])
        for is_correct, topic in results[method]:
            topic_results[topic][1] += 1
            if is_correct:
                topic_results[topic][0] += 1

        label = {"raw": "Raw LLM", "cooccurrence": "GAIA Co-occurrence",
                 "relational": "GAIA Relational"}[method]

        print(f"\n  {label}:")
        print(f"    Overall: {correct_count}/{total} ({accuracy:.0%})")
        for topic, (c, t) in sorted(topic_results.items()):
            print(f"    {topic}: {c}/{t}")

    # Comparison
    raw_acc = sum(1 for c, _ in results["raw"] if c) / len(results["raw"])
    cooc_acc = sum(1 for c, _ in results["cooccurrence"] if c) / len(results["cooccurrence"])
    rel_acc = sum(1 for c, _ in results["relational"] if c) / len(results["relational"])

    print(f"\n  {'=' * 60}")
    print(f"  Raw LLM:           {raw_acc:.0%}")
    print(f"  + Co-occurrence:   {cooc_acc:.0%} ({'+' if cooc_acc - raw_acc >= 0 else ''}{cooc_acc - raw_acc:.0%} vs raw)")
    print(f"  + Relational:      {rel_acc:.0%} ({'+' if rel_acc - raw_acc >= 0 else ''}{rel_acc - raw_acc:.0%} vs raw)")

    if rel_acc > cooc_acc:
        print(f"\n  Relational > Co-occurrence by {rel_acc - cooc_acc:.0%}")
        print(f"  Structure matters. GAIA is thinking, not just counting.")
    elif rel_acc == cooc_acc:
        print(f"\n  Relational = Co-occurrence (no advantage from structure at this scale)")
    else:
        print(f"\n  Co-occurrence > Relational by {cooc_acc - rel_acc:.0%}")
        print(f"  Structure doesn't help here. Needs investigation.")

    print(f"\n  Model: {model_name}")
    print(f"  Relations extracted: {len(si.core.relations)}")
    print(f"  Concepts: {len(si.core.concepts)}")
    print(f"  Samples/question: {n_samples}")

    # Show example intents for comparison
    print(f"\n{'=' * 80}")
    print("EXAMPLE INTENTS (co-occurrence vs relational)")
    print(f"{'=' * 80}")

    example_queries = [
        "Darwin proposed",
        "The hippocampus is essential for",
        "Einstein proposed general relativity in",
    ]
    for q in example_queries:
        words = si.core._tokenize(q)
        cooc_intent = si.core._format_cooccurrence_intent(words)
        rel_intent = si.core.format_relational_intent(q)
        print(f"\n  Query: \"{q}\"")
        print(f"  Co-occurrence: {cooc_intent[:100]}")
        print(f"  Relational:    {rel_intent[:100]}")


def main():
    parser = argparse.ArgumentParser(description="SI Deep Spike")
    parser.add_argument("--model", type=str, default="gpt2")
    args = parser.parse_args()

    run_deep_eval(args.model)


if __name__ == "__main__":
    main()
