"""
SI Voice Spike — GAIA thinks, LLM speaks
=========================================

Core idea: GAIA is the intelligence. It learns, remembers, decides.
The LLM is just its voice — rendering GAIA's semantic intent into language.

Architecture:
  1. GAIA CORE: Field-based learning from corpus
     - Builds entity hierarchy from word co-occurrence (no backprop)
     - SEC phase determines crystallization (what's "known" vs "uncertain")
     - Transition distributions encode learned patterns
     - Active entities = current semantic state

  2. LLM VOICE: Takes GAIA's state and renders it as language
     - GAIA's active entities + predictions become a semantic preamble
     - LLM generates fluent text conditioned on GAIA's intent
     - GAIA is NOT policing the LLM — it's TELLING it what to say

  3. The difference from RAG/agents:
     - RAG retrieves documents. GAIA has LEARNED the structure.
     - Agents call tools. GAIA has ALREADY done the reasoning.
     - Conservation-gating polices LLM output. Voice DIRECTS it.

Usage:
  python si_voice.py                     # train on corpus + demo
  python si_voice.py --corpus path.txt   # custom corpus
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


# ─── GAIA Core: Field-Based Intelligence ───

@dataclass
class SemanticEntity:
    """An emergent entity in GAIA's field — a crystallized concept."""
    eid: int
    words: set[str]                           # words that co-activate this entity
    transitions: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    activation_count: int = 0
    level: int = 0                            # hierarchy level (0=word, 1+=emergent)
    phase: str = "chaotic"                    # SEC phase
    entropy: float = 1.0                      # current entropy

    def top_predictions(self, n: int = 5) -> list[tuple[str, float]]:
        """Top-n predicted next words with confidence."""
        total = sum(self.transitions.values())
        if total == 0:
            return []
        sorted_items = sorted(self.transitions.items(), key=lambda x: -x[1])[:n]
        return [(word, count / total) for word, count in sorted_items]

    def update_phase(self):
        """Classify SEC phase from transition entropy."""
        total = sum(self.transitions.values())
        if total < 5:
            self.phase = "chaotic"
            self.entropy = 1.0
            return
        probs = [c / total for c in self.transitions.values() if c > 0]
        self.entropy = -sum(p * math.log(p) for p in probs) / max(math.log(len(probs)), 0.01)
        if self.entropy < XI_SEC:
            self.phase = "crystallized"
        elif self.entropy < PHI_INV:
            self.phase = "ordered"
        elif self.entropy < LAMBDA_STAR:
            self.phase = "transitional"
        else:
            self.phase = "chaotic"


class GAIACore:
    """Field-based intelligence — learns from text via co-occurrence and hierarchy.

    No backprop. No gradients. Just counting, co-activation, and SEC phases.
    This IS the thinking. The LLM just gives it a voice.
    """

    def __init__(self, window: int = 5, min_coactivation: int = 3):
        self.window = window
        self.min_coactivation = min_coactivation

        # L0: one entity per word
        self.word_entities: dict[str, int] = {}
        self.entities: dict[int, SemanticEntity] = {}
        self.next_eid = 0

        # Co-activation tracking for hierarchy building
        self.coactivation: dict[tuple[int, int], int] = defaultdict(int)

        # Word-level transition table (bigram)
        self.word_transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Context window for entity activation
        self.context_window: list[str] = []

        # Stats
        self.words_processed = 0
        self.n_merges = 0

    def _get_or_create_word_entity(self, word: str) -> int:
        if word not in self.word_entities:
            eid = self.next_eid
            self.next_eid += 1
            self.entities[eid] = SemanticEntity(eid=eid, words={word}, level=0)
            self.word_entities[word] = eid
        return self.word_entities[word]

    def _tokenize(self, text: str) -> list[str]:
        """Simple word tokenization."""
        return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|[0-9]+", text.lower())

    def learn(self, text: str):
        """Learn from text — build entities and transitions. No backprop."""
        words = self._tokenize(text)

        for i, word in enumerate(words):
            eid = self._get_or_create_word_entity(word)
            entity = self.entities[eid]
            entity.activation_count += 1

            # Update transitions: what word follows this one?
            if i + 1 < len(words):
                next_word = words[i + 1]
                entity.transitions[next_word] += 1
                self.word_transitions[word][next_word] += 1

            # Track co-activation within window
            window_start = max(0, i - self.window)
            for j in range(window_start, i):
                other_word = words[j]
                other_eid = self.word_entities[other_word]
                pair = (min(eid, other_eid), max(eid, other_eid))
                self.coactivation[pair] += 1

            self.words_processed += 1

        # After learning, update phases and build hierarchy
        for entity in self.entities.values():
            entity.update_phase()

        self._build_hierarchy()

    def _build_hierarchy(self):
        """Merge frequently co-activated entities into higher-level concepts."""
        # Find pairs that co-activate above threshold
        for (eid1, eid2), count in self.coactivation.items():
            if count < self.min_coactivation:
                continue
            if eid1 not in self.entities or eid2 not in self.entities:
                continue

            e1 = self.entities[eid1]
            e2 = self.entities[eid2]

            # Only merge L0 entities into L1 (keep it simple for spike)
            if e1.level > 0 or e2.level > 0:
                continue

            # Check if a merged entity already covers these words
            merged_words = e1.words | e2.words
            already_merged = any(
                e.words == merged_words and e.level > 0
                for e in self.entities.values()
            )
            if already_merged:
                continue

            # Create merged entity
            new_eid = self.next_eid
            self.next_eid += 1
            merged = SemanticEntity(
                eid=new_eid,
                words=merged_words,
                level=max(e1.level, e2.level) + 1,
                activation_count=count,
            )
            # Merge transitions
            for word, c in e1.transitions.items():
                merged.transitions[word] += c
            for word, c in e2.transitions.items():
                merged.transitions[word] += c
            merged.update_phase()

            self.entities[new_eid] = merged
            self.n_merges += 1

    def activate(self, context: list[str]) -> list[SemanticEntity]:
        """Given context words, return activated entities sorted by relevance."""
        context_set = set(w.lower() for w in context)
        activated = []

        for entity in self.entities.values():
            overlap = entity.words & context_set
            if overlap:
                # Relevance = overlap fraction * activation count * level bonus
                relevance = (len(overlap) / len(entity.words)
                             * math.log1p(entity.activation_count)
                             * (1 + entity.level * PHI_INV))
                activated.append((entity, relevance))

        activated.sort(key=lambda x: -x[1])
        return [e for e, _ in activated]

    def semantic_state(self, context: list[str], max_entities: int = 10) -> dict:
        """Get GAIA's current semantic state given context.

        This is what gets communicated to the LLM voice.
        Filters aggressively: only crystallized/ordered entities contribute
        predictions, and only high-confidence predictions survive.
        """
        active = self.activate(context)[:max_entities]

        # Collect predictions — phase-gated: crystallized/ordered entities
        # get full weight, transitional gets reduced, chaotic gets minimal
        phase_weight = {
            "crystallized": 2.0,
            "ordered": 1.0,
            "transitional": 0.3,
            "chaotic": 0.1,
        }
        predictions: dict[str, float] = defaultdict(float)
        for entity in active:
            pw = phase_weight.get(entity.phase, 0.1)
            level_w = 1.0 + entity.level * PHI_INV
            weight = pw * level_w
            preds = entity.top_predictions(10)
            for word, conf in preds:
                predictions[word] += conf * weight

        # Filter: only predictions above median survive (selective intent)
        if predictions:
            values = sorted(predictions.values())
            median = values[len(values) // 2]
            threshold = max(median, 0.1)  # minimum threshold
            filtered = [(w, s) for w, s in predictions.items() if s >= threshold]
            filtered.sort(key=lambda x: -x[1])
            sorted_preds = filtered[:15]
        else:
            sorted_preds = []

        # Determine overall phase (weighted by entity level)
        phase_scores = {"crystallized": 0, "ordered": 0, "transitional": 0, "chaotic": 0}
        for entity in active:
            phase_scores[entity.phase] += 1 + entity.level
        dominant_phase = max(phase_scores, key=phase_scores.get)

        return {
            "active_entities": active,
            "predictions": sorted_preds,
            "phase": dominant_phase,
            "n_active": len(active),
            "context": context,
        }

    def format_intent(self, state: dict) -> str:
        """Format GAIA's semantic state as a natural language intent.

        This becomes the semantic preamble for the LLM.
        """
        preds = state["predictions"]
        if not preds:
            return ""

        # Build concept list from predictions
        top_concepts = [word for word, _ in preds[:8]]
        concept_str = ", ".join(top_concepts)

        # Phase affects how directive the preamble is
        phase = state["phase"]
        if phase == "crystallized":
            prefix = "The following is well-established knowledge about"
        elif phase == "ordered":
            prefix = "The following discusses"
        elif phase == "transitional":
            prefix = "The following explores"
        else:
            prefix = "The following considers"

        # Active entity words give topic context
        topic_words = set()
        for entity in state["active_entities"][:5]:
            topic_words.update(entity.words)

        topic = " ".join(sorted(topic_words)[:6])

        return f"{prefix} {topic}. Key concepts: {concept_str}."

    def stats(self) -> str:
        n_l0 = sum(1 for e in self.entities.values() if e.level == 0)
        n_l1 = sum(1 for e in self.entities.values() if e.level >= 1)
        crystallized = sum(1 for e in self.entities.values() if e.phase == "crystallized")
        ordered = sum(1 for e in self.entities.values() if e.phase == "ordered")
        return (f"GAIA Core: {n_l0} word entities, {n_l1} merged entities, "
                f"{self.n_merges} merges, {self.words_processed} words processed\n"
                f"  Phases: {crystallized} crystallized, {ordered} ordered, "
                f"{sum(1 for e in self.entities.values() if e.phase == 'transitional')} transitional, "
                f"{sum(1 for e in self.entities.values() if e.phase == 'chaotic')} chaotic")


# ─── LLM Voice ───

class LLMVoice:
    """The LLM component — just a voice for GAIA's intelligence."""

    def __init__(self, model_name: str = "gpt2"):
        print(f"Loading LLM voice ({model_name})...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def speak(self, prompt: str, max_tokens: int = 40,
              temperature: float = 0.7, top_k: int = 30) -> str:
        """Generate text from prompt."""
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
    def speak_guided(self, prompt: str, gaia_intent: str,
                     max_tokens: int = 40, temperature: float = 0.7,
                     top_k: int = 30) -> str:
        """Generate text guided by GAIA's semantic intent.

        The intent is prepended as context — GAIA tells the LLM
        what to talk about, the LLM figures out how to say it.
        """
        if gaia_intent:
            full_prompt = f"{gaia_intent}\n\n{prompt}"
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
        # Strip the preamble to return only the response
        if gaia_intent and full_text.startswith(gaia_intent):
            full_text = full_text[len(gaia_intent):].lstrip("\n")
        return full_text


# ─── Unified Device ───

class SyntheticIntelligence:
    """One device, two components. GAIA thinks, LLM speaks."""

    def __init__(self, model_name: str = "gpt2"):
        self.core = GAIACore(window=5, min_coactivation=3)
        self.voice = LLMVoice(model_name)

    def learn(self, text: str):
        """GAIA learns from text. No LLM involved."""
        self.core.learn(text)

    def respond(self, query: str, max_tokens: int = 50) -> dict:
        """Full SI response: GAIA thinks, then LLM speaks.

        Returns both the raw LLM output AND the GAIA-guided output
        for comparison.
        """
        # 1. GAIA thinks: what do I know about this?
        context_words = self.core._tokenize(query)
        state = self.core.semantic_state(context_words)
        intent = self.core.format_intent(state)

        # 2. LLM speaks: render GAIA's intent as language
        raw_output = self.voice.speak(query, max_tokens=max_tokens)
        guided_output = self.voice.speak_guided(query, intent, max_tokens=max_tokens)

        return {
            "query": query,
            "gaia_state": {
                "n_active": state["n_active"],
                "phase": state["phase"],
                "predictions": state["predictions"][:8],
                "intent": intent,
            },
            "raw_llm": raw_output,
            "gaia_guided": guided_output,
        }

    def feedback(self, correction: str, repeat: int = 3):
        """Learn from a correction instantly. O(1) per token, zero backprop.

        The correction is just text that GAIA learns from — same mechanism
        as initial training, but targeted. Repeating reinforces the pattern
        so it competes with existing knowledge.

        Args:
            correction: Natural language correction, e.g.
                "Darwin proposed natural selection, not Newton"
            repeat: How many times to reinforce (more = stronger override)
        """
        for _ in range(repeat):
            self.core.learn(correction)

    def respond_guided_only(self, query: str, max_tokens: int = 50) -> str:
        """Convenience: just the GAIA-guided response text."""
        context_words = self.core._tokenize(query)
        state = self.core.semantic_state(context_words)
        intent = self.core.format_intent(state)
        return self.voice.speak_guided(query, intent, max_tokens=max_tokens)


# ─── Demo ───

DEMO_CORPUS = """
The theory of evolution by natural selection was proposed by Charles Darwin. Darwin observed
that organisms within a population show variation in traits. Those individuals with traits
better suited to the environment are more likely to survive and reproduce. Over many generations,
this process leads to the adaptation of populations to their environment. This mechanism is
known as natural selection.

Darwin traveled on the HMS Beagle to the Galapagos Islands where he observed finches with
different beak shapes. Each species of finch had evolved beak shapes suited to the food
sources available on their particular island. This observation was crucial to developing
his theory.

Genetics later provided the mechanism for inheritance that Darwin lacked. DNA carries the
genetic instructions for organisms. Mutations in DNA create variation. Natural selection
acts on this variation. The modern synthesis combines Darwin's natural selection with
Mendelian genetics.

The human brain contains approximately 86 billion neurons. Each neuron can form thousands
of synaptic connections. The brain consumes about 20 percent of the body's energy despite
being only 2 percent of body weight. Different regions of the brain are specialized for
different functions. The prefrontal cortex handles planning and decision making. The
hippocampus is crucial for memory formation.

Quantum mechanics describes the behavior of particles at very small scales. The uncertainty
principle states that you cannot simultaneously know both the exact position and exact
momentum of a particle. Particles exhibit wave-particle duality. Quantum entanglement
allows particles to be correlated regardless of distance. Measurement collapses the
wave function.

Machine learning is a subset of artificial intelligence where systems learn from data.
Neural networks are inspired by biological neurons. Deep learning uses many layers of
neural networks. Training involves adjusting weights through backpropagation and gradient
descent. Transformers use attention mechanisms to process sequences.
"""

TEST_PROMPTS = [
    "Darwin discovered that",
    "The human brain is capable of",
    "In quantum physics, particles can",
    "Neural networks learn by",
    "Evolution works through",
    "The Galapagos finches showed that",
]


def run_demo(model_name: str = "gpt2", corpus: str = None):
    """Run the SI voice demo."""

    si = SyntheticIntelligence(model_name)

    # Learn
    text = corpus if corpus else DEMO_CORPUS
    print(f"\n{'=' * 80}")
    print("PHASE 1: GAIA LEARNS")
    print(f"{'=' * 80}")
    print(f"Corpus size: {len(text)} chars")

    # Learn multiple passes for stronger patterns
    for epoch in range(5):
        si.learn(text)
    print(si.core.stats())

    # Show some crystallized entities
    crystallized = [e for e in si.core.entities.values()
                    if e.phase in ("crystallized", "ordered") and e.activation_count > 5]
    crystallized.sort(key=lambda e: -e.activation_count)
    print(f"\nTop crystallized concepts:")
    for e in crystallized[:15]:
        preds = e.top_predictions(3)
        pred_str = ", ".join(f"{w}({c:.0%})" for w, c in preds)
        print(f"  L{e.level} [{e.phase[:5]}] {sorted(e.words)} "
              f"(activated {e.activation_count}x) -> {pred_str}")

    # Respond
    print(f"\n{'=' * 80}")
    print("PHASE 2: GAIA THINKS, LLM SPEAKS")
    print(f"{'=' * 80}")

    for prompt in TEST_PROMPTS:
        result = si.respond(prompt, max_tokens=40)

        print(f"\n{'-' * 70}")
        print(f"QUERY: {prompt}")
        print(f"GAIA state: {result['gaia_state']['n_active']} entities active, "
              f"phase={result['gaia_state']['phase']}")
        print(f"GAIA predictions: "
              + ", ".join(f"{w}({s:.2f})" for w, s in result['gaia_state']['predictions'][:6]))
        print(f"GAIA intent: {result['gaia_state']['intent']}")
        print(f"\nRAW LLM:      {result['raw_llm'][len(prompt):]}")
        print(f"GAIA+LLM:     {result['gaia_guided'][len(prompt):]}")


# ─── Feedback Demo ───

FEEDBACK_SCENARIOS = [
    {
        "name": "Correcting attribution",
        "query": "Darwin discovered that",
        "corrections": [
            "Darwin discovered natural selection through careful observation of species variation.",
            "Darwin's key discovery was that species adapt through natural selection over generations.",
            "Natural selection was Darwin's central contribution to biology and evolution.",
        ],
    },
    {
        "name": "Adding new knowledge",
        "query": "The Galapagos finches showed that",
        "corrections": [
            "The Galapagos finches demonstrated adaptive radiation, where a single ancestor "
            "species diversifies into many species, each adapted to different ecological niches.",
            "Finch beak diversity on the Galapagos proved that environment shapes evolution. "
            "Each island's food sources drove beak shape adaptation.",
            "The Galapagos finches are the strongest evidence for natural selection acting "
            "on variation within populations to produce adaptation.",
        ],
    },
    {
        "name": "Domain correction",
        "query": "Neural networks learn by",
        "corrections": [
            "Neural networks learn by adjusting connection weights through backpropagation, "
            "but biological neurons learn through synaptic plasticity without gradients.",
            "Artificial neural networks use gradient descent and backpropagation. "
            "This is fundamentally different from biological learning which uses "
            "local Hebbian rules and spike-timing dependent plasticity.",
        ],
    },
    {
        "name": "Real-time fact update",
        "query": "In quantum physics, particles can",
        "corrections": [
            "Quantum particles exhibit superposition, existing in multiple states simultaneously "
            "until measured. Entanglement correlates particles across any distance instantly.",
            "Quantum decoherence explains how quantum behavior transitions to classical behavior. "
            "The measurement problem remains one of physics' deepest open questions.",
        ],
    },
]


def run_feedback_demo(model_name: str = "gpt2"):
    """Demonstrate real-time feedback: GAIA learns from corrections instantly."""

    si = SyntheticIntelligence(model_name)

    # Initial training (same as main demo)
    print(f"\n{'=' * 80}")
    print("FEEDBACK DEMO: GAIA LEARNS FROM CORRECTIONS IN REAL-TIME")
    print(f"{'=' * 80}")
    print("\nPhase 1: Initial training on corpus...")
    for _ in range(5):
        si.learn(DEMO_CORPUS)
    print(si.core.stats())

    for scenario in FEEDBACK_SCENARIOS:
        query = scenario["query"]
        corrections = scenario["corrections"]

        print(f"\n{'=' * 80}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"{'=' * 80}")
        print(f"Query: {query}")

        # BEFORE: response with current knowledge
        print(f"\n--- BEFORE FEEDBACK ---")
        context_words = si.core._tokenize(query)
        state_before = si.core.semantic_state(context_words)
        preds_before = state_before["predictions"][:6]
        print(f"GAIA predictions: {', '.join(f'{w}({s:.2f})' for w, s in preds_before)}")
        print(f"GAIA phase: {state_before['phase']}")

        before_response = si.respond_guided_only(query, max_tokens=40)
        print(f"GAIA+LLM output: {before_response[len(query):]}")

        # FEEDBACK: user provides corrections
        print(f"\n--- APPLYING FEEDBACK (O(1) per token, zero backprop) ---")
        for i, correction in enumerate(corrections):
            print(f"  Correction {i+1}: \"{correction[:70]}...\"")
            si.feedback(correction, repeat=3)

        n_entities_after = len(si.core.entities)
        print(f"  GAIA entities: {n_entities_after} (learned new concepts instantly)")

        # AFTER: response with corrected knowledge
        print(f"\n--- AFTER FEEDBACK ---")
        state_after = si.core.semantic_state(context_words)
        preds_after = state_after["predictions"][:6]
        print(f"GAIA predictions: {', '.join(f'{w}({s:.2f})' for w, s in preds_after)}")
        print(f"GAIA phase: {state_after['phase']}")

        after_response = si.respond_guided_only(query, max_tokens=40)
        print(f"GAIA+LLM output: {after_response[len(query):]}")

        # Show what changed in predictions
        before_words = {w for w, _ in preds_before}
        after_words = {w for w, _ in preds_after}
        new_preds = after_words - before_words
        if new_preds:
            print(f"\n  NEW predictions after feedback: {', '.join(sorted(new_preds))}")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total corrections applied: {sum(len(s['corrections']) for s in FEEDBACK_SCENARIOS)}")
    print(f"Learning mechanism: O(1) counting per token")
    print(f"Backpropagation: ZERO")
    print(f"Gradient computation: ZERO")
    print(f"LLM weights modified: ZERO")
    print(f"GAIA entities: {len(si.core.entities)}")
    print(f"GAIA merges: {si.core.n_merges}")
    print(f"\nThe LLM never changed. GAIA learned, and the LLM's output changed")
    print(f"because GAIA told it something different. Same weights, better output.")


# ─── Quantitative Evaluation ───

EVAL_CASES = [
    # (query, correction_corpus, expected_keywords, related_query, related_keywords)
    # expected_keywords: words that SHOULD appear in output after learning
    # related_query: a different query that should ALSO improve (knowledge transfer)
    {
        "name": "Evolution basics",
        "query": "Darwin discovered that",
        "corrections": [
            "Darwin discovered natural selection through observation of species variation.",
            "Darwin's theory of evolution explains how species adapt over generations.",
            "Natural selection acts on variation within populations to produce adaptation.",
        ],
        "expected_keywords": {"selection", "natural", "species", "variation", "evolution", "adapt"},
        "related_query": "Evolution works through",
        "related_keywords": {"selection", "natural", "variation", "species", "adapt"},
    },
    {
        "name": "Galapagos evidence",
        "query": "The Galapagos finches showed that",
        "corrections": [
            "Galapagos finches demonstrated adaptive radiation from a single ancestor species.",
            "Finch beak diversity proves environment shapes evolution through natural selection.",
            "Each island's food sources drove beak shape adaptation in finch populations.",
        ],
        "expected_keywords": {"finch", "beak", "island", "adaptation", "species", "evolution"},
        "related_query": "Beak shapes in birds evolved because",
        "related_keywords": {"beak", "adaptation", "food", "environment", "selection"},
    },
    {
        "name": "Quantum mechanics",
        "query": "In quantum physics, particles can",
        "corrections": [
            "Quantum particles exhibit superposition, existing in multiple states simultaneously.",
            "Entanglement correlates particles across any distance, violating classical locality.",
            "The wave function collapses upon measurement, selecting one definite state.",
        ],
        "expected_keywords": {"quantum", "superposition", "entanglement", "measurement", "wave", "states"},
        "related_query": "Measurement in quantum mechanics causes",
        "related_keywords": {"collapse", "wave", "measurement", "quantum", "state"},
    },
    {
        "name": "Brain structure",
        "query": "The human brain is capable of",
        "corrections": [
            "The human brain has 86 billion neurons forming trillions of synaptic connections.",
            "The prefrontal cortex handles planning, decision making, and executive function.",
            "The hippocampus is essential for memory formation and spatial navigation.",
            "Neuroplasticity allows the brain to reorganize connections throughout life.",
        ],
        "expected_keywords": {"neurons", "brain", "memory", "prefrontal", "connections", "planning"},
        "related_query": "Memory formation happens in the",
        "related_keywords": {"hippocampus", "memory", "brain", "neurons", "connections"},
    },
    {
        "name": "Machine learning",
        "query": "Neural networks learn by",
        "corrections": [
            "Neural networks learn by adjusting weights through backpropagation and gradient descent.",
            "Deep learning uses many layers of neural networks with nonlinear activation functions.",
            "Transformers use self-attention to process sequences without recurrence.",
        ],
        "expected_keywords": {"weights", "backpropagation", "gradient", "layers", "learning", "networks"},
        "related_query": "Deep learning differs from traditional ML because",
        "related_keywords": {"layers", "deep", "learning", "networks", "features"},
    },
]


def keyword_hit_rate(text: str, keywords: set[str]) -> tuple[float, set[str]]:
    """What fraction of expected keywords appear in the output?"""
    text_lower = text.lower()
    text_words = set(re.findall(r"[a-z]+", text_lower))
    hits = keywords & text_words
    rate = len(hits) / len(keywords) if keywords else 0.0
    return rate, hits


def run_eval(model_name: str = "gpt2"):
    """Quantitative evaluation: measure keyword hit rate before/after GAIA feedback.

    Tests two things:
    1. DIRECT improvement: query X improves after corrections about X
    2. TRANSFER improvement: query Y improves after corrections about related X
    """
    si = SyntheticIntelligence(model_name)

    # Initial training
    print(f"\n{'=' * 80}")
    print("QUANTITATIVE EVAL: KEYWORD HIT RATE BEFORE/AFTER FEEDBACK")
    print(f"{'=' * 80}")
    print("\nInitial corpus training...")
    for _ in range(5):
        si.learn(DEMO_CORPUS)
    print(si.core.stats())

    n_samples = 3  # generate multiple samples per query for statistical robustness

    direct_before_rates = []
    direct_after_rates = []
    transfer_before_rates = []
    transfer_after_rates = []

    for case in EVAL_CASES:
        print(f"\n{'-' * 70}")
        print(f"CASE: {case['name']}")
        print(f"{'-' * 70}")

        query = case["query"]
        expected = case["expected_keywords"]
        related_query = case["related_query"]
        related_kw = case["related_keywords"]

        # --- BEFORE feedback ---
        # Direct query
        before_hits_direct = []
        for _ in range(n_samples):
            out = si.respond_guided_only(query, max_tokens=50)
            rate, hits = keyword_hit_rate(out, expected)
            before_hits_direct.append(rate)
        avg_before_direct = sum(before_hits_direct) / len(before_hits_direct)

        # Related query (transfer)
        before_hits_transfer = []
        for _ in range(n_samples):
            out = si.respond_guided_only(related_query, max_tokens=50)
            rate, hits = keyword_hit_rate(out, related_kw)
            before_hits_transfer.append(rate)
        avg_before_transfer = sum(before_hits_transfer) / len(before_hits_transfer)

        # --- RAW LLM baseline ---
        raw_hits = []
        for _ in range(n_samples):
            out = si.voice.speak(query, max_tokens=50)
            rate, hits = keyword_hit_rate(out, expected)
            raw_hits.append(rate)
        avg_raw = sum(raw_hits) / len(raw_hits)

        # --- Apply corrections ---
        for correction in case["corrections"]:
            si.feedback(correction, repeat=3)

        # --- AFTER feedback ---
        # Direct query
        after_hits_direct = []
        for _ in range(n_samples):
            out = si.respond_guided_only(query, max_tokens=50)
            rate, hits = keyword_hit_rate(out, expected)
            after_hits_direct.append(rate)
        avg_after_direct = sum(after_hits_direct) / len(after_hits_direct)

        # Related query (transfer)
        after_hits_transfer = []
        for _ in range(n_samples):
            out = si.respond_guided_only(related_query, max_tokens=50)
            rate, hits = keyword_hit_rate(out, related_kw)
            after_hits_transfer.append(rate)
        avg_after_transfer = sum(after_hits_transfer) / len(after_hits_transfer)

        # Track
        direct_before_rates.append(avg_before_direct)
        direct_after_rates.append(avg_after_direct)
        transfer_before_rates.append(avg_before_transfer)
        transfer_after_rates.append(avg_after_transfer)

        # Report
        direct_delta = avg_after_direct - avg_before_direct
        transfer_delta = avg_after_transfer - avg_before_transfer
        d_sign = "+" if direct_delta >= 0 else ""
        t_sign = "+" if transfer_delta >= 0 else ""

        print(f"  Query: \"{query}\"")
        print(f"  Keywords: {sorted(expected)}")
        print(f"  Raw LLM hit rate:          {avg_raw:.0%}")
        print(f"  GAIA+LLM BEFORE feedback:  {avg_before_direct:.0%}")
        print(f"  GAIA+LLM AFTER feedback:   {avg_after_direct:.0%}  ({d_sign}{direct_delta:.0%})")
        print(f"")
        print(f"  Transfer query: \"{related_query}\"")
        print(f"  Transfer keywords: {sorted(related_kw)}")
        print(f"  Transfer BEFORE:           {avg_before_transfer:.0%}")
        print(f"  Transfer AFTER:            {avg_after_transfer:.0%}  ({t_sign}{transfer_delta:.0%})")

    # ─── Summary ───
    avg_raw_all = sum(raw_hits) / len(raw_hits)  # only last case, but directional
    avg_direct_before = sum(direct_before_rates) / len(direct_before_rates)
    avg_direct_after = sum(direct_after_rates) / len(direct_after_rates)
    avg_transfer_before = sum(transfer_before_rates) / len(transfer_before_rates)
    avg_transfer_after = sum(transfer_after_rates) / len(transfer_after_rates)

    direct_improvement = avg_direct_after - avg_direct_before
    transfer_improvement = avg_transfer_after - avg_transfer_before
    direct_wins = sum(1 for b, a in zip(direct_before_rates, direct_after_rates) if a > b)
    transfer_wins = sum(1 for b, a in zip(transfer_before_rates, transfer_after_rates) if a > b)

    print(f"\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(f"")
    print(f"  DIRECT IMPROVEMENT (same query)")
    print(f"    Before feedback:  {avg_direct_before:.1%} avg keyword hit rate")
    print(f"    After feedback:   {avg_direct_after:.1%} avg keyword hit rate")
    print(f"    Improvement:      {'+' if direct_improvement >= 0 else ''}{direct_improvement:.1%}")
    print(f"    Wins:             {direct_wins}/{len(EVAL_CASES)} cases improved")
    print(f"")
    print(f"  KNOWLEDGE TRANSFER (related query)")
    print(f"    Before feedback:  {avg_transfer_before:.1%} avg keyword hit rate")
    print(f"    After feedback:   {avg_transfer_after:.1%} avg keyword hit rate")
    print(f"    Improvement:      {'+' if transfer_improvement >= 0 else ''}{transfer_improvement:.1%}")
    print(f"    Wins:             {transfer_wins}/{len(EVAL_CASES)} cases improved")
    print(f"")
    print(f"  METHOD")
    print(f"    Learning:         O(1) counting per token")
    print(f"    Backpropagation:  ZERO")
    print(f"    LLM weights:      UNCHANGED")
    print(f"    GAIA entities:    {len(si.core.entities)}")
    print(f"    Samples/query:    {n_samples}")
    print(f"")
    if direct_improvement > 0:
        print(f"  GAIA feedback improves LLM output by {direct_improvement:.1%} keyword hit rate")
        print(f"  without touching a single weight.")
    if transfer_improvement > 0:
        print(f"  Knowledge TRANSFERS to related queries (+{transfer_improvement:.1%}).")
        print(f"  GAIA doesn't just memorize corrections — it builds connected structure.")


# ─── Accumulation Demo ───

ACCUMULATION_STEPS = [
    {
        "label": "Correction 1",
        "correction": "Darwin discovered natural selection. Natural selection is the mechanism of evolution.",
    },
    {
        "label": "Correction 2",
        "correction": "Species evolve through natural selection acting on heritable variation in traits.",
    },
    {
        "label": "Correction 3",
        "correction": "Organisms with advantageous traits survive and reproduce. This is survival of the fittest.",
    },
    {
        "label": "Correction 4",
        "correction": "Evolution by natural selection requires variation, inheritance, and differential survival.",
    },
    {
        "label": "Correction 5",
        "correction": "Darwin observed finch beak adaptation on the Galapagos, proving natural selection drives speciation.",
    },
    {
        "label": "Correction 6",
        "correction": "Natural selection combined with Mendelian genetics forms the modern evolutionary synthesis.",
    },
    {
        "label": "Correction 7",
        "correction": "Evolution is not random. Natural selection is directional, favoring traits that increase fitness.",
    },
    {
        "label": "Correction 8",
        "correction": "Darwin's theory of evolution by natural selection is the unifying framework of biology.",
    },
]

ACCUMULATION_QUERY = "Darwin discovered that"
ACCUMULATION_KEYWORDS = {"darwin", "natural", "selection", "evolution", "species", "traits", "variation", "adaptation"}


def run_accumulation(model_name: str = "gpt2"):
    """Show that targeted corrections accumulate: each one reinforces GAIA's understanding."""

    si = SyntheticIntelligence(model_name)

    print(f"\n{'=' * 80}")
    print("ACCUMULATION DEMO: TARGETED CORRECTIONS COMPOUND")
    print(f"{'=' * 80}")

    # Minimal baseline — just the demo corpus
    print("\nBaseline: original corpus only...")
    for _ in range(3):
        si.learn(DEMO_CORPUS)
    print(si.core.stats())

    n_samples = 5  # more samples for smoother signal
    query = ACCUMULATION_QUERY
    keywords = ACCUMULATION_KEYWORDS

    print(f"\nQuery: \"{query}\"")
    print(f"Target keywords: {sorted(keywords)}")
    print(f"Samples per measurement: {n_samples}")

    # Raw LLM baseline (no GAIA at all)
    raw_rates = []
    for _ in range(n_samples):
        out = si.voice.speak(query, max_tokens=50)
        rate, _ = keyword_hit_rate(out, keywords)
        raw_rates.append(rate)
    raw_avg = sum(raw_rates) / len(raw_rates)

    # Track curve: (label, hit_rate, n_entities, n_crystallized)
    curve: list[tuple[str, float, int, int]] = []

    # Measure baseline
    rates = []
    for _ in range(n_samples):
        out = si.respond_guided_only(query, max_tokens=50)
        rate, _ = keyword_hit_rate(out, keywords)
        rates.append(rate)
    avg = sum(rates) / len(rates)
    n_cryst = sum(1 for e in si.core.entities.values() if e.phase == "crystallized")
    curve.append(("Baseline", avg, len(si.core.entities), n_cryst))

    # Apply corrections one at a time, measuring after each
    for step in ACCUMULATION_STEPS:
        si.feedback(step["correction"], repeat=5)

        rates = []
        for _ in range(n_samples):
            out = si.respond_guided_only(query, max_tokens=50)
            rate, _ = keyword_hit_rate(out, keywords)
            rates.append(rate)
        avg = sum(rates) / len(rates)
        n_cryst = sum(1 for e in si.core.entities.values() if e.phase == "crystallized")
        curve.append((step["label"], avg, len(si.core.entities), n_cryst))

    # ─── Results ───
    print(f"\n{'=' * 80}")
    print("ACCUMULATION CURVE")
    print(f"{'=' * 80}")
    print(f"\n  Raw LLM (no GAIA):  {raw_avg:.0%}")
    print(f"")
    print(f"  {'Step':<15} {'Hit Rate':>10} {'Entities':>10} {'Crystallized':>14}  Trend")
    print(f"  {'-' * 75}")

    for label, rate, n_ent, n_cryst in curve:
        bar_len = int(rate * 40)
        bar = "#" * bar_len + "." * (40 - bar_len)
        print(f"  {label:<15} {rate:>9.0%} {n_ent:>10} {n_cryst:>14}  [{bar}]")

    # Overall
    first_rate = curve[0][1]
    last_rate = curve[-1][1]
    total_gain = last_rate - first_rate
    gain_vs_raw = last_rate - raw_avg

    print(f"\n  {'=' * 75}")
    print(f"  Raw LLM:        {raw_avg:.0%}")
    print(f"  GAIA baseline:  {first_rate:.0%}")
    print(f"  GAIA after {len(ACCUMULATION_STEPS)} corrections: {last_rate:.0%}")
    print(f"  Gain vs baseline: {'+' if total_gain >= 0 else ''}{total_gain:.0%}")
    print(f"  Gain vs raw LLM:  {'+' if gain_vs_raw >= 0 else ''}{gain_vs_raw:.0%}")
    print(f"")
    print(f"  Method: O(1) counting per token. Zero backprop. LLM weights unchanged.")
    if total_gain > 0:
        print(f"  Each correction compounds. GAIA gets sharper, the LLM stays frozen.")


def main():
    parser = argparse.ArgumentParser(description="SI Voice Spike")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--corpus", type=str, help="Path to corpus file")
    parser.add_argument("--feedback", action="store_true",
                        help="Run the feedback loop demo")
    parser.add_argument("--eval", action="store_true",
                        help="Run quantitative evaluation")
    parser.add_argument("--accumulation", action="store_true",
                        help="Run accumulation demo (knowledge compounds)")
    args = parser.parse_args()

    if args.eval:
        run_eval(args.model)
    elif args.feedback:
        run_feedback_demo(args.model)
    elif args.accumulation:
        run_accumulation(args.model)
    else:
        corpus = None
        if args.corpus:
            corpus = Path(args.corpus).read_text()
        run_demo(args.model, corpus)


if __name__ == "__main__":
    main()
