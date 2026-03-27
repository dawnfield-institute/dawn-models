"""
SI Scale Spike — SEC-Navigated RAG (GAIA + Retrieval Fusion)
=============================================================

The pivot: GAIA and RAG aren't competitors. They're two halves of one system.
RAG finds text. GAIA understands it. SEC navigates the retrieval.

si_scale v1 showed RAG (74%) crushes standalone GAIA (38%) on factual recall.
That's expected — RAG literally retrieves the answer sentence. But RAG is
dumb retrieval. It can't reason about what it found, can't weight by confidence,
can't cross-reference across domains.

SEC-RAG fuses them:
1. RAG retrieves candidate sentences (TF-IDF)
2. SEC scores each candidate using GAIA's knowledge graph
   - Sentences about crystallized concepts get boosted
   - Sentences with relational links to query concepts get priority
3. Relational facts prepend the retrieved text (structured framing)
4. Phase-gated context: crystallized = compact facts + 1 sentence,
   chaotic = more retrieved sentences (needs coverage)

5-WAY COMPARISON:
  Raw LLM | RAG | GAIA Co-occur | GAIA Relational | SEC-RAG (fusion)

Usage:
  python si_scale.py                       # full eval (gpt2)
  python si_scale.py --model gpt2-medium   # stronger voice
  python si_scale.py --samples 3           # faster (fewer samples)
  python si_scale.py --quick               # quick mode: 20 questions
"""

import argparse
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ─── DFT Constants ───

XI_SEC = 0.0618
PHI_INV = 0.618
LAMBDA_STAR = 0.9816


# ─── Relational Entity (from si_deep) ───

@dataclass
class Relation:
    subject: str
    predicate: str
    obj: str
    count: int = 0

    def key(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.obj)


@dataclass
class ConceptNode:
    name: str
    activation_count: int = 0
    outgoing: dict[str, list[tuple[str, int]]] = field(default_factory=lambda: defaultdict(list))
    incoming: dict[str, list[tuple[str, int]]] = field(default_factory=lambda: defaultdict(list))
    cooccurrence: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    phase: str = "chaotic"
    entropy: float = 1.0

    def update_phase(self):
        total_rels = sum(c for pairs in self.outgoing.values() for _, c in pairs)
        total_rels += sum(c for pairs in self.incoming.values() for _, c in pairs)
        if total_rels < 3:
            self.phase = "chaotic"
            self.entropy = 1.0
            return
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
    triples = []
    sentences = re.split(r'[.!?]+', text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        words = sentence.split()
        if not words:
            continue
        subject_words = []
        for w in words:
            if w[0].isupper() or w.lower() in ("and", "y"):
                subject_words.append(w)
            else:
                break
        if not subject_words:
            for w in words:
                if w.lower() not in STOPWORDS:
                    subject_words = [w]
                    break
        if not subject_words:
            continue
        subject_key = subject_words[-1].lower() if subject_words else ""
        if not subject_key or subject_key in STOPWORDS:
            subject_key = " ".join(subject_words).lower()
        patterns = [
            (r"(?:proposed|discovered|invented|developed|created|founded|formulated|introduced|demonstrated|proved|showed|established)\s+(?:the\s+)?(.+?)(?:\.|,|$)", "proposed"),
            (r"(?:is|are|was|were)\s+(?:known\s+as|called|named|termed)\s+(?:the\s+)?(.+?)(?:\.|,|$)", "known_as"),
            (r"(?:is|are|was|were)\s+(?:essential|crucial|important|necessary|responsible|used)\s+for\s+(.+?)(?:\.|,|$)", "essential_for"),
            (r"(?:contains?|has|produces?)\s+(?:approximately\s+)?(.+?)(?:\.|,|$)", "has"),
            (r"(?:published|wrote|authored)\s+(.+?)(?:\.|,|$)", "published"),
            (r"(?:uses?|employs?|utilizes?|requires?)\s+(.+?)(?:\.|,|$)", "uses"),
            (r"(?:converts?|transforms?)\s+(.+?)\s+(?:into|to)\s+(.+?)(?:\.|,|$)", "converts"),
            (r"(?:is|are)\s+(?:a|an|the)\s+(\w+(?:\s+\w+){0,3})(?:\.|,|$)", "is_a"),
            (r"(?:acts?\s+on|operates?\s+on|regulates?|controls?)\s+(.+?)(?:\.|,|$)", "acts_on"),
            (r"(?:occurs?\s+in|found\s+in|located\s+in|lives?\s+in)\s+(.+?)(?:\.|,|$)", "found_in"),
            (r"(?:causes?|leads?\s+to|results?\s+in)\s+(.+?)(?:\.|,|$)", "causes"),
            (r"(?:consists?\s+of|composed\s+of|made\s+of)\s+(.+?)(?:\.|,|$)", "consists_of"),
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
        words = self._tokenize(text)
        for i, word in enumerate(words):
            concept = self._get_or_create(word)
            concept.activation_count += 1
            window_start = max(0, i - self.window)
            for j in range(window_start, i):
                other = words[j]
                concept.cooccurrence[other] += 1
                self._get_or_create(other).cooccurrence[word] += 1
            self.words_processed += 1
        triples = extract_relations(text)
        for subj, pred, obj in triples:
            key = (subj, pred, obj)
            if key in self.relation_index:
                self.relation_index[key].count += 1
            else:
                rel = Relation(subject=subj, predicate=pred, obj=obj, count=1)
                self.relations.append(rel)
                self.relation_index[key] = rel
            subj_node = self._get_or_create(subj)
            obj_node = self._get_or_create(obj)
            self._add_relation_link(subj_node.outgoing, pred, obj)
            self._add_relation_link(obj_node.incoming, pred, subj)
        for concept in self.concepts.values():
            concept.update_phase()

    def _add_relation_link(self, links: dict, predicate: str, target: str):
        for i, (t, c) in enumerate(links[predicate]):
            if t == target:
                links[predicate][i] = (t, c + 1)
                return
        links[predicate].append((target, 1))

    def query_relations(self, concept: str, direction: str = "both",
                        predicate: str = None) -> list[tuple[str, str, str, int]]:
        results = []
        concept_lower = concept.lower()
        matching_concepts = []
        if concept_lower in self.concepts:
            matching_concepts.append(concept_lower)
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
        words = self._tokenize(query)
        content_words = [w for w in words if w not in STOPWORDS]
        if not content_words:
            content_words = words[:3]
        all_rels = []
        for word in content_words:
            rels = self.query_relations(word)
            all_rels.extend(rels)
        if not all_rels:
            return self._format_cooccurrence_intent(words)
        seen = set()
        unique_rels = []
        for r in all_rels:
            key = (r[0], r[1], r[2])
            if key not in seen:
                seen.add(key)
                unique_rels.append(r)
        unique_rels.sort(key=lambda x: -x[3])
        facts = []
        for subj, pred, obj, count in unique_rels[:8]:
            pred_map = {
                "is_a": "is", "proposed": "proposed", "has": "has",
                "essential_for": "is essential for", "causes": "causes",
                "uses": "uses", "known_as": "is known as",
                "acts_on": "acts on", "found_in": "is found in",
                "consists_of": "consists of", "published": "published",
                "converts": "converts to",
            }
            verb = pred_map.get(pred, pred)
            facts.append(f"{subj} {verb} {obj}")
        if not facts:
            return self._format_cooccurrence_intent(words)
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

    def sec_rerank(self, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Re-rank retrieved sentences using SEC phase knowledge.

        Sentences containing crystallized concepts get boosted.
        Sentences with relational links get additional weight.
        """
        reranked = []
        for sentence, tfidf_score in candidates:
            words = self._tokenize(sentence)
            # SEC phase boost: how crystallized are the concepts in this sentence?
            phase_scores = []
            rel_bonus = 0.0
            for word in words:
                if word in self.concepts and word not in STOPWORDS:
                    node = self.concepts[word]
                    if node.phase == "crystallized":
                        phase_scores.append(2.0)
                    elif node.phase == "ordered":
                        phase_scores.append(1.0)
                    elif node.phase == "transitional":
                        phase_scores.append(0.3)
                    else:
                        phase_scores.append(0.1)
                    # Relational bonus: does this concept have outgoing relations?
                    n_rels = sum(len(pairs) for pairs in node.outgoing.values())
                    if n_rels > 0:
                        rel_bonus += 0.2 * min(n_rels, 5)

            if phase_scores:
                phase_weight = sum(phase_scores) / len(phase_scores)
            else:
                phase_weight = 0.5  # unknown concepts get neutral weight

            # Combined score: TF-IDF * SEC phase * (1 + relational bonus)
            combined = tfidf_score * phase_weight * (1.0 + rel_bonus)
            reranked.append((sentence, combined))

        reranked.sort(key=lambda x: -x[1])
        return reranked

    def format_sec_rag_intent(self, query: str, rag: 'RAGBaseline') -> str:
        """SEC-navigated RAG: GAIA structures, RAG retrieves, SEC navigates.

        1. Retrieve candidate sentences via TF-IDF
        2. Re-rank using SEC phase confidence
        3. Prepend relational facts for structured framing
        4. Phase-gate the context size
        """
        # Get relational facts for the query
        words = self._tokenize(query)
        content_words = [w for w in words if w not in STOPWORDS]
        if not content_words:
            content_words = words[:3]

        # Determine dominant phase of query concepts
        phases = [self.concepts[w].phase for w in content_words if w in self.concepts]
        n_crystallized = phases.count("crystallized") if phases else 0
        is_crystallized = n_crystallized > len(phases) // 2 if phases else False

        # Get relational facts
        all_rels = []
        for word in content_words:
            rels = self.query_relations(word)
            all_rels.extend(rels)

        # Deduplicate relations
        seen = set()
        unique_rels = []
        for r in all_rels:
            key = (r[0], r[1], r[2])
            if key not in seen:
                seen.add(key)
                unique_rels.append(r)
        unique_rels.sort(key=lambda x: -x[3])

        # Format relational facts
        facts = []
        pred_map = {
            "is_a": "is", "proposed": "proposed", "has": "has",
            "essential_for": "is essential for", "causes": "causes",
            "uses": "uses", "known_as": "is known as",
            "acts_on": "acts on", "found_in": "is found in",
            "consists_of": "consists of", "published": "published",
            "converts": "converts to",
        }
        for subj, pred, obj, count in unique_rels[:6]:
            verb = pred_map.get(pred, pred)
            facts.append(f"{subj} {verb} {obj}")

        # Retrieve and SEC-rerank
        candidates = rag.retrieve_scored(query, top_k=6)
        reranked = self.sec_rerank(candidates)

        # Phase-gated context sizing
        if is_crystallized:
            # High confidence: compact facts + top 1 retrieved sentence
            n_retrieve = 1
        else:
            # Low confidence: need more context
            n_retrieve = 3

        top_sentences = [sent for sent, _ in reranked[:n_retrieve]]

        # Build combined intent
        parts = []
        if facts:
            fact_str = ". ".join(facts[:4])
            parts.append(f"Known facts: {fact_str}")
        if top_sentences:
            retrieved_str = ". ".join(top_sentences)
            parts.append(f"Reference: {retrieved_str}")

        if not parts:
            return rag.retrieve(query, top_k=3)

        return ". ".join(parts) + "."

    def stats(self) -> str:
        n_concepts = len(self.concepts)
        n_relations = len(self.relations)
        n_cryst = sum(1 for c in self.concepts.values() if c.phase == "crystallized")
        n_ordered = sum(1 for c in self.concepts.values() if c.phase == "ordered")
        total_rel_count = sum(r.count for r in self.relations)
        return (f"GAIA: {n_concepts} concepts, {n_relations} unique relations "
                f"({total_rel_count} total), {self.words_processed} words\n"
                f"  Phases: {n_cryst} crystallized, {n_ordered} ordered, "
                f"{sum(1 for c in self.concepts.values() if c.phase == 'transitional')} transitional, "
                f"{sum(1 for c in self.concepts.values() if c.phase == 'chaotic')} chaotic")


# ─── RAG Baseline ───

class RAGBaseline:
    """Simple TF-IDF retrieval baseline.

    Splits corpus into sentences, builds TF-IDF vectors, retrieves
    top-k most relevant sentences for each query. This is the "obvious"
    alternative to GAIA — just find and inject relevant text.
    """

    def __init__(self):
        self.documents: list[str] = []
        self.tf_idf: dict[int, dict[str, float]] = {}
        self.idf: dict[str, float] = {}
        self.vocab: set[str] = set()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|[0-9]+", text.lower())

    def index(self, corpus: str):
        """Split corpus into sentences and build TF-IDF index."""
        # Split into sentences
        raw_sentences = re.split(r'[.!?]+', corpus)
        self.documents = [s.strip() for s in raw_sentences if len(s.strip()) > 20]

        # Compute TF per document
        doc_freq: dict[str, int] = defaultdict(int)
        for i, doc in enumerate(self.documents):
            words = self._tokenize(doc)
            word_counts = defaultdict(int)
            for w in words:
                word_counts[w] += 1
            total = len(words) if words else 1
            self.tf_idf[i] = {}
            seen = set()
            for w, c in word_counts.items():
                self.tf_idf[i][w] = c / total
                self.vocab.add(w)
                if w not in seen:
                    doc_freq[w] += 1
                    seen.add(w)

        # Compute IDF
        n_docs = len(self.documents)
        for word, df in doc_freq.items():
            self.idf[word] = math.log((n_docs + 1) / (df + 1)) + 1

        # Apply IDF to TF
        for i in self.tf_idf:
            for word in self.tf_idf[i]:
                self.tf_idf[i][word] *= self.idf.get(word, 1.0)

    def _score_query(self, query: str) -> list[tuple[float, int]]:
        """Score all documents against query. Returns [(score, doc_idx), ...]."""
        query_words = self._tokenize(query)
        query_tfidf = {}
        word_counts = defaultdict(int)
        for w in query_words:
            word_counts[w] += 1
        total = len(query_words) if query_words else 1
        for w, c in word_counts.items():
            query_tfidf[w] = (c / total) * self.idf.get(w, 1.0)

        scores = []
        for i, doc_vec in self.tf_idf.items():
            dot = sum(query_tfidf.get(w, 0) * doc_vec.get(w, 0)
                      for w in set(list(query_tfidf.keys()) + list(doc_vec.keys())))
            mag_q = math.sqrt(sum(v ** 2 for v in query_tfidf.values())) or 1e-10
            mag_d = math.sqrt(sum(v ** 2 for v in doc_vec.values())) or 1e-10
            sim = dot / (mag_q * mag_d)
            scores.append((sim, i))
        scores.sort(reverse=True)
        return scores

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """Retrieve top-k most relevant sentences for query."""
        scores = self._score_query(query)
        top_docs = [self.documents[i] for _, i in scores[:top_k]]
        return ". ".join(top_docs) + "."

    def retrieve_scored(self, query: str, top_k: int = 6) -> list[tuple[str, float]]:
        """Retrieve top-k sentences with their TF-IDF scores."""
        scores = self._score_query(query)
        return [(self.documents[i], score) for score, i in scores[:top_k]]


# ─── LLM Voice ───

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
            input_ids, max_new_tokens=max_tokens, temperature=temperature,
            top_k=top_k, do_sample=True,
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
            input_ids, max_new_tokens=max_tokens, temperature=temperature,
            top_k=top_k, do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        full_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if intent and full_text.startswith(intent):
            full_text = full_text[len(intent):].lstrip("\n")
        return full_text


# ─── Knowledge Corpus (8 domains, ~5000 chars) ───

KNOWLEDGE_CORPUS = {
    "evolution": """
Charles Darwin proposed the theory of evolution by natural selection. Darwin traveled on
the HMS Beagle to the Galapagos Islands where he observed finch beak variations. Alfred
Russel Wallace independently discovered natural selection while studying wildlife in the
Malay Archipelago. Darwin published On the Origin of Species in 1859. Lamarck proposed
an earlier theory of inheritance of acquired characteristics which was later disproved.
Natural selection acts on heritable variation within populations. Speciation occurs when
populations become reproductively isolated.
""",
    "genetics": """
Gregor Mendel discovered the laws of inheritance through pea plant experiments in 1866.
Mendel is known as the father of genetics. DNA carries genetic instructions for all living
organisms. James Watson and Francis Crick discovered the double helix structure of DNA in
1953. Rosalind Franklin produced the X-ray crystallography images crucial for discovering
DNA structure. Barbara McClintock discovered transposable elements (jumping genes) in maize.
The Human Genome Project completed sequencing the human genome in 2003. CRISPR-Cas9 is
used for precise gene editing and was developed by Jennifer Doudna and Emmanuelle Charpentier.
""",
    "neuroscience": """
The human brain contains approximately 86 billion neurons. The prefrontal cortex is essential
for planning and decision making. The hippocampus is essential for memory formation and spatial
navigation. Santiago Ramon y Cajal discovered that neurons are individual cells, establishing
the neuron doctrine. Neurons communicate through synaptic connections using neurotransmitters.
Broca discovered that damage to a specific left frontal region causes speech production deficits.
Wernicke discovered that damage to the left temporal region causes language comprehension deficits.
Dopamine is essential for reward processing and motivation. Serotonin regulates mood and sleep.
""",
    "physics": """
Isaac Newton proposed the laws of motion and universal gravitation. Newton published Principia
Mathematica in 1687. Albert Einstein proposed the theory of general relativity in 1915 and
special relativity in 1905. Einstein proposed the photoelectric effect which demonstrated
the particle nature of light. Max Planck introduced the quantum hypothesis in 1900. Niels
Bohr proposed the atomic model with quantized electron orbits. Werner Heisenberg formulated
the uncertainty principle. Erwin Schrodinger developed the wave equation for quantum mechanics.
Marie Curie discovered radioactivity in polonium and radium.
""",
    "chemistry": """
Dmitri Mendeleev created the periodic table of elements in 1869, organizing elements by
atomic weight and predicting undiscovered elements. Antoine Lavoisier is known as the father
of modern chemistry and discovered the role of oxygen in combustion. John Dalton proposed
the modern atomic theory in 1803. Linus Pauling discovered the nature of chemical bonds
and proposed electronegativity scales. The pH scale measures hydrogen ion concentration.
Water consists of two hydrogen atoms and one oxygen atom. Acids have pH below 7 and bases
have pH above 7.
""",
    "biology_cell": """
The mitochondria is known as the powerhouse of the cell. ATP is the energy currency of cells.
Cellular respiration converts glucose into ATP through glycolysis, the Krebs cycle, and
oxidative phosphorylation. Photosynthesis converts sunlight into chemical energy in plants.
Chloroplasts contain chlorophyll which absorbs light. Robert Hooke discovered cells in 1665
using a microscope. The cell membrane consists of a phospholipid bilayer. Ribosomes are
responsible for protein synthesis. The endoplasmic reticulum is essential for protein folding
and lipid synthesis.
""",
    "ml": """
Machine learning uses algorithms that learn from data without explicit programming. Neural
networks are inspired by biological neurons. Geoffrey Hinton developed backpropagation for
training neural networks. Transformers use self-attention mechanisms and were introduced in
the paper Attention Is All You Need. Attention was proposed by Vaswani in 2017. Deep learning
uses many layers of neural networks. Yann LeCun developed convolutional neural networks for
image recognition. Ian Goodfellow invented generative adversarial networks in 2014. Support
vector machines use kernel functions for classification in high-dimensional spaces.
""",
    "astronomy": """
Galileo Galilei discovered the moons of Jupiter using a telescope in 1610. Johannes Kepler
formulated the three laws of planetary motion. Edwin Hubble discovered that the universe is
expanding in 1929. The Big Bang theory describes the origin of the universe approximately
13.8 billion years ago. Black holes are regions where gravity is so strong that nothing can
escape. Stephen Hawking proposed that black holes emit radiation through quantum effects.
The speed of light in vacuum is approximately 299792458 meters per second. Neutron stars
are the densest known objects in the universe after black holes.
""",
}


# ─── 50 Factual Questions ───

FACTUAL_QA = [
    # ─── Evolution (6) ───
    {"query": "Darwin proposed", "correct": ["natural selection", "evolution"], "wrong": ["relativity", "quantum"], "topic": "evolution"},
    {"query": "Wallace independently discovered", "correct": ["natural selection"], "wrong": ["dna", "gravity"], "topic": "evolution"},
    {"query": "The Origin of Species was published in", "correct": ["1859"], "wrong": ["1687", "1905", "1953"], "topic": "evolution"},
    {"query": "Darwin traveled on the", "correct": ["beagle"], "wrong": ["enterprise", "mayflower"], "topic": "evolution"},
    {"query": "Lamarck proposed", "correct": ["inheritance", "acquired characteristics"], "wrong": ["natural selection", "relativity"], "topic": "evolution"},
    {"query": "Speciation occurs when populations become", "correct": ["reproductively isolated", "isolated"], "wrong": ["larger", "extinct"], "topic": "evolution"},

    # ─── Genetics (7) ───
    {"query": "Mendel is known as the father of", "correct": ["genetics"], "wrong": ["physics", "chemistry"], "topic": "genetics"},
    {"query": "Watson and Crick discovered the structure of", "correct": ["dna"], "wrong": ["rna", "protein"], "topic": "genetics"},
    {"query": "Rosalind Franklin produced", "correct": ["x-ray", "crystallography"], "wrong": ["evolution", "gravity"], "topic": "genetics"},
    {"query": "Mendel discovered the laws of inheritance through", "correct": ["pea"], "wrong": ["fruit flies", "mice"], "topic": "genetics"},
    {"query": "The Human Genome Project completed sequencing in", "correct": ["2003"], "wrong": ["1990", "2010", "1953"], "topic": "genetics"},
    {"query": "CRISPR-Cas9 was developed by", "correct": ["doudna", "charpentier"], "wrong": ["watson", "crick"], "topic": "genetics"},
    {"query": "McClintock discovered", "correct": ["transposable", "jumping genes"], "wrong": ["dna", "rna"], "topic": "genetics"},

    # ─── Neuroscience (7) ───
    {"query": "The hippocampus is essential for", "correct": ["memory"], "wrong": ["vision", "hearing"], "topic": "neuroscience"},
    {"query": "The prefrontal cortex handles", "correct": ["planning", "decision"], "wrong": ["vision", "breathing"], "topic": "neuroscience"},
    {"query": "Ramon y Cajal discovered that neurons are", "correct": ["individual", "cells"], "wrong": ["connected", "continuous"], "topic": "neuroscience"},
    {"query": "Broca discovered that damage to the left frontal region causes", "correct": ["speech"], "wrong": ["blindness", "deafness"], "topic": "neuroscience"},
    {"query": "Wernicke discovered that damage to the left temporal region causes", "correct": ["comprehension", "language"], "wrong": ["blindness", "paralysis"], "topic": "neuroscience"},
    {"query": "Dopamine is essential for", "correct": ["reward", "motivation"], "wrong": ["digestion", "breathing"], "topic": "neuroscience"},
    {"query": "Serotonin regulates", "correct": ["mood", "sleep"], "wrong": ["vision", "hearing"], "topic": "neuroscience"},

    # ─── Physics (7) ───
    {"query": "Newton published Principia Mathematica in", "correct": ["1687"], "wrong": ["1859", "1905"], "topic": "physics"},
    {"query": "Einstein proposed general relativity in", "correct": ["1915"], "wrong": ["1687", "1859"], "topic": "physics"},
    {"query": "Einstein proposed special relativity in", "correct": ["1905"], "wrong": ["1687", "1915"], "topic": "physics"},
    {"query": "Planck introduced", "correct": ["quantum"], "wrong": ["relativity", "gravity"], "topic": "physics"},
    {"query": "Heisenberg formulated the", "correct": ["uncertainty"], "wrong": ["relativity", "gravity"], "topic": "physics"},
    {"query": "Curie discovered radioactivity in", "correct": ["polonium", "radium"], "wrong": ["uranium", "carbon"], "topic": "physics"},
    {"query": "Bohr proposed the atomic model with", "correct": ["quantized", "electron orbits", "orbits"], "wrong": ["strings", "quarks"], "topic": "physics"},

    # ─── Chemistry (6) ───
    {"query": "Mendeleev created", "correct": ["periodic table"], "wrong": ["atomic bomb", "microscope"], "topic": "chemistry"},
    {"query": "Lavoisier is known as the father of", "correct": ["chemistry"], "wrong": ["physics", "biology"], "topic": "chemistry"},
    {"query": "Dalton proposed the modern", "correct": ["atomic theory"], "wrong": ["periodic table", "quantum"], "topic": "chemistry"},
    {"query": "Pauling discovered the nature of", "correct": ["chemical bonds", "bonds"], "wrong": ["atoms", "gravity"], "topic": "chemistry"},
    {"query": "Water consists of", "correct": ["hydrogen", "oxygen"], "wrong": ["nitrogen", "carbon"], "topic": "chemistry"},
    {"query": "The periodic table was created in", "correct": ["1869"], "wrong": ["1687", "1905", "1953"], "topic": "chemistry"},

    # ─── Cell Biology (6) ───
    {"query": "The mitochondria is known as the", "correct": ["powerhouse"], "wrong": ["brain", "nucleus"], "topic": "cell_biology"},
    {"query": "Photosynthesis converts sunlight into", "correct": ["chemical energy", "energy"], "wrong": ["heat", "sound"], "topic": "cell_biology"},
    {"query": "Robert Hooke discovered cells in", "correct": ["1665"], "wrong": ["1859", "1953"], "topic": "cell_biology"},
    {"query": "Cellular respiration converts glucose into", "correct": ["atp"], "wrong": ["dna", "protein"], "topic": "cell_biology"},
    {"query": "Ribosomes are responsible for", "correct": ["protein synthesis", "protein"], "wrong": ["dna replication", "cell division"], "topic": "cell_biology"},
    {"query": "The cell membrane consists of", "correct": ["phospholipid", "bilayer"], "wrong": ["protein", "carbohydrate"], "topic": "cell_biology"},

    # ─── Machine Learning (6) ───
    {"query": "Backpropagation was developed by", "correct": ["hinton"], "wrong": ["einstein", "darwin"], "topic": "ml"},
    {"query": "The attention mechanism was proposed by", "correct": ["vaswani"], "wrong": ["hinton", "lecun"], "topic": "ml"},
    {"query": "LeCun developed", "correct": ["convolutional", "cnn"], "wrong": ["transformer", "gan"], "topic": "ml"},
    {"query": "Goodfellow invented", "correct": ["generative adversarial", "gan"], "wrong": ["transformer", "cnn"], "topic": "ml"},
    {"query": "Transformers use", "correct": ["self-attention", "attention"], "wrong": ["convolution", "recurrence"], "topic": "ml"},
    {"query": "Attention Is All You Need was published in", "correct": ["2017"], "wrong": ["2014", "2012", "2020"], "topic": "ml"},

    # ─── Astronomy (5) ───
    {"query": "Galileo discovered the moons of", "correct": ["jupiter"], "wrong": ["saturn", "mars"], "topic": "astronomy"},
    {"query": "Hubble discovered that the universe is", "correct": ["expanding"], "wrong": ["shrinking", "static"], "topic": "astronomy"},
    {"query": "The Big Bang occurred approximately", "correct": ["13.8", "13", "billion"], "wrong": ["4.5", "million"], "topic": "astronomy"},
    {"query": "Hawking proposed that black holes emit", "correct": ["radiation"], "wrong": ["light", "sound"], "topic": "astronomy"},
    {"query": "Kepler formulated the laws of", "correct": ["planetary motion", "planetary"], "wrong": ["gravity", "thermodynamics"], "topic": "astronomy"},
]


def score_answer(output: str, correct: list[str], wrong: list[str]) -> tuple[bool, str]:
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


# ─── Main Eval ───

def run_scale_eval(model_name: str = "gpt2", n_samples: int = 5,
                   questions: list[dict] = None):
    """5-way comparison: Raw vs RAG vs GAIA Co-occur vs GAIA Relational vs SEC-RAG."""

    if questions is None:
        questions = FACTUAL_QA

    voice = LLMVoice(model_name)
    core = GAIADeepCore(window=5)
    rag = RAGBaseline()

    print(f"\n{'=' * 90}")
    print("SI SCALE EVAL: RAW vs RAG vs GAIA CO-OCCUR vs GAIA REL vs SEC-RAG")
    print(f"{'=' * 90}")

    # Build combined corpus
    full_corpus = "\n".join(KNOWLEDGE_CORPUS.values())
    corpus_chars = len(full_corpus)
    n_questions = len(questions)

    print(f"\nCorpus: {corpus_chars} chars across {len(KNOWLEDGE_CORPUS)} domains")
    print(f"Questions: {n_questions}")
    print(f"Samples per question: {n_samples}")
    print(f"Total generations: {n_questions * n_samples * 5} ({n_questions} x {n_samples} x 5 methods)")
    print(f"Model: {model_name}")

    # Train GAIA
    print(f"\nTraining GAIA on corpus (5 passes)...")
    t0 = time.time()
    for _ in range(5):
        core.learn(full_corpus)
    gaia_time = time.time() - t0
    print(f"  {core.stats()}")
    print(f"  Training time: {gaia_time:.1f}s")

    # Build RAG index
    print(f"\nBuilding RAG index...")
    t0 = time.time()
    rag.index(full_corpus)
    rag_time = time.time() - t0
    print(f"  {len(rag.documents)} sentences indexed")
    print(f"  Index time: {rag_time:.1f}s")

    # Top relations
    print(f"\nTop 10 relations:")
    sorted_rels = sorted(core.relations, key=lambda r: -r.count)
    for rel in sorted_rels[:10]:
        print(f"  ({rel.subject}, {rel.predicate}, {rel.obj}) x{rel.count}")

    # ─── Run eval ───

    methods = ["raw", "rag", "cooccurrence", "relational", "sec_rag"]
    results: dict[str, list[tuple[float, str]]] = {m: [] for m in methods}
    per_question: list[dict[str, float]] = []

    print(f"\n{'=' * 90}")
    print(f"{'Question':<45} {'Raw':>4} {'RAG':>5} {'CoOc':>5} {'Rel':>5} {'S-R':>5}")
    print(f"{'-' * 90}")

    for qi, qa in enumerate(questions):
        query = qa["query"]
        correct = qa["correct"]
        wrong = qa["wrong"]
        topic = qa["topic"]

        q_scores = {}

        for method in methods:
            correct_count = 0
            for _ in range(n_samples):
                if method == "raw":
                    out = voice.speak(query, max_tokens=40)
                elif method == "rag":
                    retrieved = rag.retrieve(query, top_k=3)
                    intent = f"Based on the following information: {retrieved}"
                    out = voice.speak_guided(query, intent, max_tokens=40)
                elif method == "cooccurrence":
                    words = core._tokenize(query)
                    intent = core._format_cooccurrence_intent(words)
                    out = voice.speak_guided(query, intent, max_tokens=40)
                elif method == "relational":
                    intent = core.format_relational_intent(query)
                    out = voice.speak_guided(query, intent, max_tokens=40)
                else:  # sec_rag
                    intent = core.format_sec_rag_intent(query, rag)
                    out = voice.speak_guided(query, intent, max_tokens=40)

                is_correct, _ = score_answer(out[len(query):], correct, wrong)
                if is_correct:
                    correct_count += 1

            accuracy = correct_count / n_samples
            q_scores[method] = accuracy
            results[method].append((accuracy, topic))

        per_question.append(q_scores)

        # Print row
        best = max(q_scores.values())
        parts = []
        for m in methods:
            s = f"{q_scores[m]:.0%}"
            if q_scores[m] == best and best > 0:
                s = f"{s}*"
            parts.append(f"{s:>5}")

        prefix = f"[{qi+1:2d}/{n_questions}]"
        print(f"  {prefix} {query:<41} {' '.join(parts)}")

    # ─── Summary ───

    print(f"\n{'=' * 90}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 90}")

    method_labels = {
        "raw": "Raw LLM",
        "rag": "RAG (TF-IDF)",
        "cooccurrence": "GAIA Co-occurrence",
        "relational": "GAIA Relational",
        "sec_rag": "SEC-RAG (Fusion)",
    }

    overall = {}
    for method in methods:
        # A question is "correct" if accuracy > 0.5 (majority of samples got it right)
        correct_count = sum(1 for acc, _ in results[method] if acc > 0.5)
        total = len(results[method])
        accuracy = correct_count / total
        overall[method] = accuracy

        # Per-topic breakdown
        topic_results = defaultdict(lambda: [0, 0])
        for acc, topic in results[method]:
            topic_results[topic][1] += 1
            if acc > 0.5:
                topic_results[topic][0] += 1

        label = method_labels[method]
        print(f"\n  {label}: {correct_count}/{total} ({accuracy:.0%})")
        for topic in sorted(topic_results.keys()):
            c, t = topic_results[topic]
            print(f"    {topic:<15} {c}/{t} ({c/t:.0%})")

    # ─── Head-to-head ───

    print(f"\n{'=' * 90}")
    print("HEAD-TO-HEAD COMPARISON")
    print(f"{'=' * 90}")

    raw = overall["raw"]
    for method in ["rag", "cooccurrence", "relational", "sec_rag"]:
        acc = overall[method]
        delta = acc - raw
        print(f"  {method_labels[method]:<22} {acc:.0%}  ({'+' if delta >= 0 else ''}{delta:.0%} vs raw)")

    # Key comparisons
    rag_acc = overall["rag"]
    rel_acc = overall["relational"]
    sec_rag_acc = overall["sec_rag"]

    print(f"\n  --- KEY COMPARISON: SEC-RAG vs RAG ---")
    if sec_rag_acc > rag_acc:
        print(f"  SEC-RAG beats RAG by {sec_rag_acc - rag_acc:.0%}")
        print(f"  SEC navigation improves retrieval.")
    elif sec_rag_acc == rag_acc:
        print(f"  SEC-RAG = RAG (tied)")
    else:
        print(f"  RAG beats SEC-RAG by {rag_acc - sec_rag_acc:.0%}")

    print(f"\n  --- SEC-RAG vs GAIA Relational ---")
    if sec_rag_acc > rel_acc:
        print(f"  SEC-RAG beats standalone GAIA by {sec_rag_acc - rel_acc:.0%}")
        print(f"  Fusion > either component alone.")
    else:
        print(f"  GAIA Relational >= SEC-RAG")

    # ─── Per-question wins ───

    wins = {m: 0 for m in methods}
    ties = 0
    for q_scores in per_question:
        best = max(q_scores.values())
        winners = [m for m in methods if q_scores[m] == best]
        if len(winners) > 1:
            ties += 1
        else:
            wins[winners[0]] += 1

    print(f"\n  Question wins:")
    for m in methods:
        print(f"    {method_labels[m]:<22} {wins[m]:>3} wins")
    print(f"    {'Ties':<22} {ties:>3}")

    # ─── GAIA advantage analysis ───

    print(f"\n  --- WHERE SEC-RAG BEATS RAG ---")
    sec_wins = []
    rag_only_wins = []
    for i, (q_scores, qa) in enumerate(zip(per_question, questions)):
        if q_scores["sec_rag"] > q_scores["rag"]:
            sec_wins.append(qa)
        elif q_scores["rag"] > q_scores["sec_rag"]:
            rag_only_wins.append(qa)

    if sec_wins:
        print(f"  SEC-RAG > RAG on {len(sec_wins)} questions:")
        for qa in sec_wins[:10]:
            print(f"    - {qa['query']} [{qa['topic']}]")
        if len(sec_wins) > 10:
            print(f"    ... and {len(sec_wins) - 10} more")

    if rag_only_wins:
        print(f"\n  RAG > SEC-RAG on {len(rag_only_wins)} questions:")
        for qa in rag_only_wins[:8]:
            print(f"    - {qa['query']} [{qa['topic']}]")
        if len(rag_only_wins) > 8:
            print(f"    ... and {len(rag_only_wins) - 8} more")

    sec_tied = len(questions) - len(sec_wins) - len(rag_only_wins)
    print(f"\n  Tied: {sec_tied} questions")

    # ─── Timing ───

    print(f"\n{'=' * 90}")
    print("SYSTEM INFO")
    print(f"{'=' * 90}")
    print(f"  Model: {model_name}")
    print(f"  Corpus: {corpus_chars} chars, {len(KNOWLEDGE_CORPUS)} domains")
    print(f"  Questions: {n_questions}")
    print(f"  Samples/question: {n_samples}")
    print(f"  Total generations: {n_questions * n_samples * 5}")
    print(f"  GAIA: {len(core.concepts)} concepts, {len(core.relations)} relations")
    print(f"  RAG: {len(rag.documents)} indexed sentences")
    print(f"  GAIA training: {gaia_time:.1f}s")
    print(f"  RAG indexing: {rag_time:.1f}s")

    # ─── Example intents ───

    print(f"\n{'=' * 90}")
    print("EXAMPLE INTENTS (RAG vs Relational vs SEC-RAG)")
    print(f"{'=' * 90}")

    examples = [
        "Darwin proposed",
        "The hippocampus is essential for",
        "Mendeleev created",
        "Hawking proposed that black holes emit",
    ]
    for q in examples:
        rel = core.format_relational_intent(q)
        retrieved = rag.retrieve(q, top_k=2)
        sec_rag_intent = core.format_sec_rag_intent(q, rag)
        print(f"\n  Query: \"{q}\"")
        print(f"  RAG:         {retrieved[:120]}...")
        print(f"  Relational:  {rel[:120]}")
        print(f"  SEC-RAG:     {sec_rag_intent[:120]}")


# ─── Continuous Learning Eval ───

# Map question topics to corpus keys
TOPIC_TO_CORPUS = {
    "evolution": "evolution",
    "genetics": "genetics",
    "neuroscience": "neuroscience",
    "physics": "physics",
    "chemistry": "chemistry",
    "cell_biology": "biology_cell",
    "ml": "ml",
    "astronomy": "astronomy",
}


def run_continuous_eval(model_name: str = "gpt2", n_samples: int = 3):
    """Does GAIA get better over time? Does it specialize?

    Feeds domains one at a time. After each domain is learned:
    1. Tests ALL questions (not just the domain just learned)
    2. Tracks in-domain accuracy (specialization)
    3. Tracks cross-domain accuracy (transfer)
    4. Tracks knowledge graph growth (concepts, relations, crystallization)

    This tests the continuous learning thesis: O(1) per token,
    no retraining, cumulative knowledge.
    """

    voice = LLMVoice(model_name)
    core = GAIADeepCore(window=5)
    rag = RAGBaseline()

    # RAG gets the full corpus upfront (it can't learn incrementally)
    full_corpus = "\n".join(KNOWLEDGE_CORPUS.values())
    rag.index(full_corpus)

    print(f"\n{'=' * 90}")
    print("CONTINUOUS LEARNING: DOES GAIA SPECIALIZE OVER TIME?")
    print(f"{'=' * 90}")
    print(f"\nModel: {model_name}")
    print(f"Samples/question: {n_samples}")
    print(f"RAG has full corpus upfront (baseline)")
    print(f"GAIA learns one domain at a time")

    # Domain learning order — start with biology, end with ML
    domain_order = [
        "evolution", "genetics", "neuroscience", "physics",
        "chemistry", "biology_cell", "ml", "astronomy",
    ]

    # Track results over time
    # timeline[step] = {method: {topic: accuracy}}
    timeline = []
    graph_stats = []  # (concepts, relations, crystallized) at each step

    # Step 0: no training (baseline)
    print(f"\n{'=' * 90}")
    print(f"STEP 0: No GAIA training (raw baseline)")
    print(f"{'=' * 90}")

    step_results = _eval_all_questions(voice, core, rag, n_samples)
    timeline.append(("none", step_results))
    graph_stats.append((0, 0, 0))
    _print_step_summary(step_results, "none")

    # Steps 1-8: learn one domain at a time
    for step, domain in enumerate(domain_order, 1):
        corpus_key = TOPIC_TO_CORPUS.get(domain, domain)
        domain_text = KNOWLEDGE_CORPUS[corpus_key]

        print(f"\n{'=' * 90}")
        print(f"STEP {step}: Learn '{domain}' ({len(domain_text)} chars)")
        print(f"{'=' * 90}")

        # GAIA learns this domain (5 passes for crystallization)
        for _ in range(5):
            core.learn(domain_text)

        n_concepts = len(core.concepts)
        n_relations = len(core.relations)
        n_cryst = sum(1 for c in core.concepts.values() if c.phase == "crystallized")
        graph_stats.append((n_concepts, n_relations, n_cryst))
        print(f"  Graph: {n_concepts} concepts, {n_relations} relations, {n_cryst} crystallized")

        # Test all questions
        step_results = _eval_all_questions(voice, core, rag, n_samples)
        timeline.append((domain, step_results))
        _print_step_summary(step_results, domain)

    # ─── Final Analysis ───

    print(f"\n{'=' * 90}")
    print("LEARNING CURVE")
    print(f"{'=' * 90}")

    # Header
    header = f"  {'Step':<4} {'Domain learned':<16}"
    for method in ["sec_rag", "rag", "relational"]:
        label = {"sec_rag": "S-RAG", "rag": "RAG", "relational": "Rel"}[method]
        header += f" {label:>5}"
    header += f"  {'Concepts':>8} {'Rels':>5} {'Cryst':>6}"
    print(header)
    print(f"  {'-' * 80}")

    for i, ((domain, results), (n_c, n_r, n_cr)) in enumerate(zip(timeline, graph_stats)):
        row = f"  {i:<4} {domain:<16}"
        for method in ["sec_rag", "rag", "relational"]:
            # Overall accuracy for this method at this step
            correct = sum(1 for acc in results[method].values() if acc > 0.5)
            total = len(results[method])
            row += f" {correct/total:>5.0%}"
        row += f"  {n_c:>8} {n_r:>5} {n_cr:>6}"
        print(row)

    # ─── Specialization Analysis ───

    print(f"\n{'=' * 90}")
    print("SPECIALIZATION: IN-DOMAIN vs CROSS-DOMAIN ACCURACY")
    print(f"{'=' * 90}")

    print(f"\n  After learning each domain, SEC-RAG accuracy on:")
    print(f"  {'Step':<4} {'Domain':<16} {'In-domain':>10} {'Cross-domain':>12} {'Delta':>8}")
    print(f"  {'-' * 55}")

    for i, (domain, results) in enumerate(timeline[1:], 1):  # skip step 0
        # In-domain: questions matching the domain just learned
        in_domain_scores = []
        cross_domain_scores = []
        for qa_topic, acc in results["sec_rag"].items():
            if qa_topic == domain:
                in_domain_scores.append(acc)
            else:
                cross_domain_scores.append(acc)

        in_avg = sum(in_domain_scores) / len(in_domain_scores) if in_domain_scores else 0
        cross_avg = sum(cross_domain_scores) / len(cross_domain_scores) if cross_domain_scores else 0
        delta = in_avg - cross_avg

        print(f"  {i:<4} {domain:<16} {in_avg:>10.0%} {cross_avg:>12.0%} {'+' if delta >= 0 else ''}{delta:>7.0%}")

    # ─── Domain Transfer Matrix ───

    print(f"\n{'=' * 90}")
    print("TRANSFER MATRIX: How learning domain X affects domain Y")
    print(f"{'=' * 90}")
    print(f"  (SEC-RAG accuracy change when domain is learned)")

    # Build delta matrix: timeline[i] - timeline[i-1] for each topic
    all_topics = sorted(set(qa["topic"] for qa in FACTUAL_QA))
    print(f"\n  {'Learned':<16}", end="")
    for t in all_topics:
        print(f" {t[:6]:>6}", end="")
    print(f" {'Total':>7}")
    print(f"  {'-' * (16 + 7 * (len(all_topics) + 1))}")

    for i in range(1, len(timeline)):
        domain, curr = timeline[i]
        _, prev = timeline[i - 1]
        print(f"  {domain:<16}", end="")
        total_delta = 0
        for t in all_topics:
            curr_acc = curr["sec_rag"].get(t, 0)
            prev_acc = prev["sec_rag"].get(t, 0)
            delta = curr_acc - prev_acc
            total_delta += delta
            if abs(delta) < 0.01:
                print(f"    {'--':>4}", end="")
            elif delta > 0:
                print(f"  {'+' + f'{delta:.0%}':>5}", end="")
            else:
                print(f"  {f'{delta:.0%}':>5}", end="")
        print(f"  {'+' if total_delta >= 0 else ''}{total_delta:.0%}")

    # ─── Key Findings ───

    print(f"\n{'=' * 90}")
    print("KEY FINDINGS")
    print(f"{'=' * 90}")

    # Compare first step vs last step for SEC-RAG
    _, first = timeline[1]  # after first domain
    _, last = timeline[-1]  # after all domains
    first_acc = sum(1 for acc in first["sec_rag"].values() if acc > 0.5) / len(first["sec_rag"])
    last_acc = sum(1 for acc in last["sec_rag"].values() if acc > 0.5) / len(last["sec_rag"])

    print(f"\n  SEC-RAG after 1 domain:  {first_acc:.0%}")
    print(f"  SEC-RAG after 8 domains: {last_acc:.0%}")
    print(f"  Improvement: {'+' if last_acc - first_acc >= 0 else ''}{last_acc - first_acc:.0%}")

    # RAG is constant (has full corpus)
    _, rag_results = timeline[-1]
    rag_acc = sum(1 for acc in rag_results["rag"].values() if acc > 0.5) / len(rag_results["rag"])
    print(f"  RAG (constant):          {rag_acc:.0%}")

    if last_acc > rag_acc:
        print(f"\n  SEC-RAG surpasses RAG after continuous learning.")
    elif last_acc == rag_acc:
        print(f"\n  SEC-RAG matches RAG after continuous learning.")
    else:
        gap = rag_acc - last_acc
        print(f"\n  SEC-RAG approaches RAG (gap: {gap:.0%}) after continuous learning.")

    # Check if graph is still growing
    final_concepts, final_rels, final_cryst = graph_stats[-1]
    print(f"\n  Final graph: {final_concepts} concepts, {final_rels} relations, {final_cryst} crystallized")
    print(f"  Crystallization rate: {final_cryst/final_concepts:.0%}")


def _eval_all_questions(voice, core, rag, n_samples):
    """Evaluate all questions, return {method: {topic: avg_accuracy}}."""
    methods = ["rag", "relational", "sec_rag"]
    # Collect per-topic accuracy
    topic_scores = {m: defaultdict(list) for m in methods}

    for qa in FACTUAL_QA:
        query = qa["query"]
        correct = qa["correct"]
        wrong = qa["wrong"]
        topic = qa["topic"]

        for method in methods:
            correct_count = 0
            for _ in range(n_samples):
                if method == "rag":
                    retrieved = rag.retrieve(query, top_k=3)
                    intent = f"Based on the following information: {retrieved}"
                    out = voice.speak_guided(query, intent, max_tokens=40)
                elif method == "relational":
                    intent = core.format_relational_intent(query)
                    out = voice.speak_guided(query, intent, max_tokens=40)
                else:  # sec_rag
                    intent = core.format_sec_rag_intent(query, rag)
                    out = voice.speak_guided(query, intent, max_tokens=40)

                is_correct, _ = score_answer(out[len(query):], correct, wrong)
                if is_correct:
                    correct_count += 1

            topic_scores[method][topic].append(correct_count / n_samples)

    # Average per topic
    result = {}
    for method in methods:
        result[method] = {}
        for topic, scores in topic_scores[method].items():
            result[method][topic] = sum(scores) / len(scores) if scores else 0
    return result


def _print_step_summary(results, domain_learned):
    """Print compact summary of a step's results."""
    for method in ["sec_rag", "rag", "relational"]:
        label = {"sec_rag": "SEC-RAG", "rag": "RAG", "relational": "Relational"}[method]
        correct = sum(1 for acc in results[method].values() if acc > 0.5)
        total = len(results[method])
        # Show in-domain accuracy if applicable
        in_domain = results[method].get(domain_learned, None)
        suffix = ""
        if in_domain is not None:
            suffix = f"  (in-domain: {in_domain:.0%})"
        print(f"  {label:<12} {correct}/{total} ({correct/total:.0%}){suffix}")


def main():
    parser = argparse.ArgumentParser(description="SI Scale Spike")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--samples", type=int, default=5,
                        help="Samples per question per method (default 5)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 20 questions instead of 50")
    parser.add_argument("--continuous", action="store_true",
                        help="Continuous learning eval: learn domains one at a time")
    args = parser.parse_args()

    if args.continuous:
        run_continuous_eval(args.model, n_samples=args.samples)
    else:
        questions = FACTUAL_QA
        if args.quick:
            by_topic = defaultdict(list)
            for qa in FACTUAL_QA:
                by_topic[qa["topic"]].append(qa)
            questions = []
            for topic, qs in sorted(by_topic.items()):
                questions.extend(qs[:3])
            print(f"Quick mode: {len(questions)} questions")
        run_scale_eval(args.model, n_samples=args.samples, questions=questions)


if __name__ == "__main__":
    main()
