"""
POC-025: GAIA Prime + Kronos Integration
========================================

Experiment 09: Coherent Retrieval - Real Sentences from Knowledge Base

The problem with exp_08: word-by-word generation produces gibberish.

The solution: Don't generate - RETRIEVE coherent passages and compose them.

This experiment:
1. Uses SEC navigation to find relevant patterns
2. Retrieves actual content (real sentences) from patterns
3. Uses learned connections to rank relevance
4. Composes responses from real source material

This is "physics-guided retrieval" - the mesh navigates, the knowledge speaks.
"""

import sys
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\fracton")
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\dawn-models\research\GAIA\src")

import torch
import re
import time
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field

# Force GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Using: {DEVICE}")

# Import from previous experiments
from exp_01_bridge import KronosGAIABridge
from exp_04_repo_index import SimpleTextEmbedder
from exp_05_sec_navigation import SECNavigator
from exp_06_associative_chains import AssociativeMind, WorkingMemory


# ============================================================================
# Passage Extractor - Gets real sentences from content
# ============================================================================

class PassageExtractor:
    """
    Extracts coherent passages (sentences/paragraphs) from content.
    """
    
    # Sentence-ending patterns
    SENTENCE_END = re.compile(r'[.!?]\s+')
    
    @staticmethod
    def extract_sentences(content: str) -> List[str]:
        """Split content into sentences."""
        # Clean up
        content = content.replace('\n', ' ').replace('\r', ' ')
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Split on sentence boundaries
        sentences = PassageExtractor.SENTENCE_END.split(content)
        
        # Filter and clean
        result = []
        for s in sentences:
            s = s.strip()
            # Must be meaningful (not just code/symbols)
            if len(s) > 20 and len(s) < 500:
                # Should have mostly letters
                letters = sum(1 for c in s if c.isalpha())
                if letters > len(s) * 0.5:
                    result.append(s)
        
        return result
    
    @staticmethod
    def extract_key_passages(content: str, max_passages: int = 5) -> List[str]:
        """Extract the most informative passages from content."""
        sentences = PassageExtractor.extract_sentences(content)
        
        if not sentences:
            return []
        
        # Score sentences by informativeness
        scored = []
        for s in sentences:
            score = 0
            
            # Penalize code-like content
            code_chars = sum(1 for c in s if c in '{}[]()=<>|&;:')
            if code_chars > len(s) * 0.1:
                score -= 1.0  # Heavy penalty for code
            
            # Penalize paths and filenames
            if '.py' in s or '.md' in s or '/' in s or '\\' in s:
                score -= 0.5
            
            # Longer sentences often more informative
            score += min(len(s) / 100, 1.0)
            
            # Contains explanatory terms (boost)
            explain_terms = ['is', 'means', 'describes', 'represents', 'shows', 
                            'demonstrates', 'provides', 'creates', 'enables',
                            'when', 'because', 'therefore', 'thus', 'hence',
                            'suggests', 'proposes', 'explores', 'framework']
            for term in explain_terms:
                if f' {term} ' in s.lower():
                    score += 0.3
            
            # Starts with capital (proper sentence)
            if s[0].isupper():
                score += 0.3
            
            # Penalize sentences starting with symbols/lowercase
            if s[0] in '#*-_`':
                score -= 0.3
            
            scored.append((score, s))
        
        # Return top passages
        scored.sort(reverse=True)
        return [s for _, s in scored[:max_passages]]


# ============================================================================
# Semantic Embeddings (from exp_08)
# ============================================================================

class SemanticEmbeddings:
    """
    Hybrid embeddings: keyword matching + n-gram similarity.
    
    Pure n-grams fail for semantic search. We need to:
    1. Extract keywords from query
    2. Boost passages containing those keywords
    3. Use n-gram similarity as secondary signal
    """
    
    def __init__(self, dim: int = 768, device: str = 'cuda', n: int = 3):
        self.dim = dim
        self.device = device
        self.n = n
        self._cache = {}
        
        torch.manual_seed(42)
        self.num_hashes = dim * 2
        self.projection = torch.randn(self.num_hashes, dim, device=device) * 0.1
        
        # Stopwords to ignore in keyword matching
        self.stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
            'until', 'while', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'am', 'it', 'its', "it's", 'about',
        }
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return [w for w in words if w not in self.stopwords]
    
    def keyword_match_score(self, query: str, passage: str) -> float:
        """Score based on keyword presence with domain-specific boosting."""
        query_keywords = set(self.extract_keywords(query))
        passage_lower = passage.lower()
        
        if not query_keywords:
            return 0.0
        
        # Domain-specific synonyms for better matching
        domain_synonyms = {
            'pac': ['conservation', 'potential', 'actualization'],
            'sec': ['entropy', 'collapse', 'structure'],
            'golden': ['phi', 'ratio', '1.618', '0.618', 'fibonacci'],
            'phi': ['golden', 'ratio', '1.618', 'fibonacci'],
            'entropy': ['disorder', 'information', 'thermodynamic'],
            'information': ['entropy', 'structure', 'field'],
            'dawn': ['field', 'theory', 'infodynamics'],
        }
        
        # Expand query keywords with synonyms
        expanded = set(query_keywords)
        for kw in query_keywords:
            if kw in domain_synonyms:
                expanded.update(domain_synonyms[kw])
        
        # Count matches (with expanded terms)
        matches = sum(1 for kw in expanded if kw in passage_lower)
        base_score = matches / len(expanded)
        
        # Boost for exact phrase matches
        query_lower = query.lower()
        if len(query_lower) > 5 and query_lower in passage_lower:
            base_score += 0.3  # Bonus for exact query match
        
        # Penalize code-like passages
        code_chars = sum(1 for c in passage if c in '{}[]()=<>|&;:_')
        code_ratio = code_chars / max(len(passage), 1)
        if code_ratio > 0.08:
            base_score *= 0.3
        
        # Penalize file paths
        if '.py' in passage or '\\' in passage:
            base_score *= 0.7
        
        # Boost explanatory passages
        explain_markers = ['is ', 'means ', 'describes ', 'represents ', 
                          'proposes ', 'suggests ', 'explores ', 'framework']
        if any(m in passage_lower for m in explain_markers):
            base_score *= 1.3
        
        return min(base_score, 1.0)
    
    def _get_ngrams(self, text: str) -> List[str]:
        text = text.lower().strip()
        if len(text) < self.n:
            return [text]
        return [text[i:i+self.n] for i in range(len(text) - self.n + 1)]
    
    def embed(self, text: str) -> torch.Tensor:
        if text in self._cache:
            return self._cache[text]
        
        ngrams = self._get_ngrams(text)
        sparse = torch.zeros(self.num_hashes, device=self.device)
        for ng in ngrams:
            idx = hash(ng) % self.num_hashes
            sparse[idx] += 1.0
        
        emb = torch.matmul(sparse, self.projection)
        emb = emb / (emb.norm() + 1e-9)
        
        self._cache[text] = emb
        return emb
    
    def similarity(self, text1: str, text2: str) -> float:
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return torch.dot(emb1, emb2).item()


# ============================================================================
# Passage Index - Pre-indexes all passages for fast retrieval
# ============================================================================

# Golden ratio for PAC tree resonance
PHI = 1.618033988749895

@dataclass
class IndexedPassage:
    """A passage with its embedding and source info."""
    text: str
    embedding: torch.Tensor
    source_pattern_id: str
    source_file: str
    depth: int = 0  # Depth in PAC tree
    parent_id: Optional[str] = None  # Parent pattern for tree traversal
    score: float = 0.0  # For ranking


class PassageIndex:
    """
    Pre-indexed passages from the knowledge base.
    
    This is the key to coherent retrieval - we search over
    real sentences, not generated tokens.
    """
    
    def __init__(self, embedder: SemanticEmbeddings):
        self.embedder = embedder
        self.passages: List[IndexedPassage] = []
        self.embedding_matrix: Optional[torch.Tensor] = None
    
    @staticmethod 
    def clean_prose(text: str) -> str:
        """Clean markdown artifacts from text."""
        # Remove markdown formatting
        import re
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic*
        text = re.sub(r'`([^`]+)`', r'\1', text)        # `code`
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # [text](url)
        text = re.sub(r'^\s*#+\s*', '', text)           # # headers
        text = re.sub(r'^\s*[-*]\s*', '', text)         # - bullets
        text = re.sub(r'^\s*>\s*', '', text)            # > quotes
        return text.strip()
    
    def _is_prose(self, text: str) -> bool:
        """Check if text is natural prose (not code or markdown syntax)."""
        # Clean first
        cleaned = self.clean_prose(text)
        
        if len(cleaned) < 25:
            return False
        
        # Code indicators (check cleaned text)
        code_chars = sum(1 for c in cleaned if c in '{}[]()=<>|&;_')
        if code_chars > len(cleaned) * 0.05:
            return False  # Too many code characters
        
        # Skip if still looks like a path or URL
        if '.py' in cleaned or '.md' in cleaned or '.yaml' in cleaned:
            return False
        if 'http' in cleaned.lower():
            return False
        
        # Must have spaces between words
        words = cleaned.split()
        if len(words) < 5:
            return False  # Too few words
        
        # Average word length - prose averages 4-6
        avg_len = sum(len(w) for w in words) / len(words)
        if avg_len > 9:
            return False  # Words too long
        
        # Must start with capital (proper sentence)
        if cleaned[0].isalpha() and cleaned[0].isupper():
            return True
        
        # Check for prose indicators
        lower = cleaned.lower()
        return any(word in lower for word in 
            [' is ', ' are ', ' the ', ' that ', ' which ', ' this ',
             ' means ', ' describes ', ' represents ', ' explores ',
             ' proposes ', ' suggests ', ' framework ', ' theory '])
    
    def _compute_depth_from_path(self, file_path: str) -> int:
        """
        Compute depth in conceptual hierarchy from file path.
        
        Root level docs (README.md, MISSION.md) = 0
        Subdirectory docs = 1 per directory level
        """
        path = Path(file_path) if file_path else Path()
        
        # Get relative depth from common roots
        parts = path.parts
        
        # Count meaningful directory levels (skip drive, repos, etc.)
        meaningful_parts = []
        in_repo = False
        for part in parts:
            if 'Dawn Field Institute' in part:
                in_repo = True
                continue
            if in_repo:
                meaningful_parts.append(part)
        
        # Root files (dawn-field-theory/README.md) = 0
        # Subdirectory files (dawn-field-theory/foundational/x.md) = 1
        return max(0, len(meaningful_parts) - 2)  # -2 for repo name + filename
    
    def build_from_kronos(self, kronos: KronosGAIABridge, verbose: bool = True):
        """Index all passages from Kronos patterns with PAC tree depth."""
        extractor = PassageExtractor()
        skipped = 0
        
        for pattern_id, pattern in kronos.pattern_index.items():
            content = pattern.metadata.get('content_preview', '')
            file_path = pattern.metadata.get('file_path', 'unknown')
            
            # Skip .py files entirely - they're code, not prose
            if file_path.endswith('.py'):
                skipped += 1
                continue
            
            # Compute depth from file path hierarchy
            depth = self._compute_depth_from_path(file_path)
            
            # Extract passages
            passages = extractor.extract_key_passages(content, max_passages=3)
            
            for passage in passages:
                # Only index prose passages
                if not self._is_prose(passage):
                    continue
                
                # Store cleaned version
                cleaned = PassageIndex.clean_prose(passage)
                    
                emb = self.embedder.embed(cleaned)
                self.passages.append(IndexedPassage(
                    text=cleaned,
                    embedding=emb,
                    source_pattern_id=pattern_id,
                    source_file=file_path,
                    depth=depth,
                    parent_id=None,
                ))
        
        # Build embedding matrix for fast search
        if self.passages:
            self.embedding_matrix = torch.stack([p.embedding for p in self.passages])
        
        if verbose:
            print(f"Indexed {len(self.passages)} passages from {len(kronos.pattern_index)} patterns")
    
    def search(self, query: str, top_k: int = 5) -> List[IndexedPassage]:
        """
        Find most relevant passages using hybrid scoring:
        1. Keyword matching (most important)
        2. N-gram embedding similarity (secondary)
        """
        if not self.passages:
            return []
        
        # Score all passages with hybrid approach
        scored = []
        for passage in self.passages:
            # Keyword match score (0-1) - weighted heavily
            keyword_score = self.embedder.keyword_match_score(query, passage.text)
            
            # Embedding similarity (0-1)
            query_emb = self.embedder.embed(query)
            emb_score = torch.nn.functional.cosine_similarity(
                query_emb.unsqueeze(0), 
                passage.embedding.unsqueeze(0),
                dim=1
            ).item()
            
            # Hybrid: keywords weighted 3x more than embeddings
            final_score = keyword_score * 0.75 + emb_score * 0.25
            
            scored.append((final_score, keyword_score, passage))
        
        # Sort by final score
        scored.sort(reverse=True, key=lambda x: x[0])
        
        # Return top-k
        results = []
        for final_score, kw_score, passage in scored[:top_k]:
            passage.score = final_score
            results.append(passage)
        
        return results
    
    def _compute_resonance(self, depth: int, query_depth: int = 0) -> float:
        """
        Compute φ-based resonance score from depth.
        
        R(k) = φ^(1 + (k_eq - k)/2)
        
        Deeper nodes get less resonance, following golden ratio decay.
        """
        k_diff = query_depth - depth
        return PHI ** (1 + k_diff / 2)
    
    def search_with_sec(
        self, 
        query: str, 
        navigator: SECNavigator,
        top_k: int = 5,
    ) -> List[IndexedPassage]:
        """
        Search using hybrid scoring + SEC + PAC tree traversal.
        
        Combines:
        1. Keyword matching (primary)
        2. Embedding similarity
        3. SEC spreading activation
        4. φ-based resonance from tree depth (PAC traversal)
        """
        if not self.passages:
            return []
        
        # Get SEC-activated patterns
        nav_result = navigator.navigate(query, initial_seeds=5, max_activated=30)
        
        # Build boost map from SEC activation
        pattern_boost = {}
        for state in nav_result.activated_patterns:
            pattern_boost[state.pattern_id] = state.effective_activation
        
        # Score all passages with hybrid + SEC + PAC depth
        scored = []
        for passage in self.passages:
            # Keyword match score (0-1)
            keyword_score = self.embedder.keyword_match_score(query, passage.text)
            
            # Embedding similarity (0-1)
            query_emb = self.embedder.embed(query)
            emb_score = torch.nn.functional.cosine_similarity(
                query_emb.unsqueeze(0), 
                passage.embedding.unsqueeze(0),
                dim=1
            ).item()
            
            # SEC boost from spreading activation
            sec_boost = pattern_boost.get(passage.source_pattern_id, 0.0)
            
            # φ-resonance from PAC tree depth (normalized to 0-1)
            resonance = self._compute_resonance(passage.depth)
            resonance_score = min(1.0, resonance / 10.0)  # Normalize
            
            # Hybrid: keywords 50%, embeddings 15%, SEC 20%, PAC depth 15%
            final_score = (
                keyword_score * 0.50 + 
                emb_score * 0.15 + 
                sec_boost * 0.20 +
                resonance_score * 0.15
            )
            
            scored.append((final_score, keyword_score, passage))
        
        # Sort and return
        scored.sort(reverse=True, key=lambda x: x[0])
        
        results = []
        for score, kw_score, passage in scored[:top_k]:
            passage.score = score
            results.append(passage)
        
        return results


# ============================================================================
# Coherent Response Generator
# ============================================================================

class CoherentGenerator:
    """
    Generates coherent responses by composing retrieved passages.
    
    This is NOT token-by-token generation - it's intelligent
    passage selection and composition.
    """
    
    def __init__(
        self,
        passage_index: PassageIndex,
        navigator: SECNavigator,
        embedder: SemanticEmbeddings,
    ):
        self.passage_index = passage_index
        self.navigator = navigator
        self.embedder = embedder
    
    def generate(
        self,
        query: str,
        max_passages: int = 3,
        use_sec: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a coherent response to the query.
        
        Returns relevant passages composed into a response.
        """
        start_time = time.perf_counter()
        
        # Retrieve relevant passages
        if use_sec:
            passages = self.passage_index.search_with_sec(
                query, self.navigator, top_k=max_passages * 2
            )
        else:
            passages = self.passage_index.search(query, top_k=max_passages * 2)
        
        # Deduplicate (similar passages from same source)
        unique_passages = self._deduplicate(passages, max_passages)
        
        # Compose response
        response_parts = []
        sources = []
        
        for p in unique_passages:
            # Add period if missing
            text = p.text.strip()
            if text and text[-1] not in '.!?':
                text += '.'
            response_parts.append(text)
            sources.append(p.source_file)
        
        response = ' '.join(response_parts)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            'query': query,
            'response': response,
            'passages_used': len(unique_passages),
            'sources': list(set(sources)),
            'avg_score': sum(p.score for p in unique_passages) / len(unique_passages) if unique_passages else 0,
            'time_ms': elapsed_ms,
        }
    
    def _deduplicate(
        self, 
        passages: List[IndexedPassage], 
        max_count: int
    ) -> List[IndexedPassage]:
        """Remove near-duplicate passages."""
        if not passages:
            return []
        
        unique = [passages[0]]
        
        for p in passages[1:]:
            if len(unique) >= max_count:
                break
            
            # Check similarity to existing
            is_duplicate = False
            for existing in unique:
                sim = self.embedder.similarity(p.text[:100], existing.text[:100])
                if sim > 0.8:  # Too similar
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(p)
        
        return unique
    
    def answer(self, question: str) -> str:
        """Simple interface - just return the response text."""
        result = self.generate(question)
        return result['response']


# ============================================================================
# Full System - Combines everything
# ============================================================================

class CoherentGAIASystem:
    """
    Complete system for coherent knowledge retrieval and response.
    
    Components:
    - Kronos: Knowledge storage
    - PassageIndex: Pre-indexed sentences
    - SECNavigator: Spreading activation
    - CoherentGenerator: Response composition
    """
    
    def __init__(self, index_path: Path, device: str = None):
        self.device = device or str(DEVICE)
        
        # Load knowledge base
        self.kronos = KronosGAIABridge(
            storage_path=index_path,
            namespace="dawn_institute",
            device=self.device,
        )
        
        # Embedder (must match Kronos dimension - 768)
        self.embedder = SemanticEmbeddings(dim=768, device=self.device)
        
        # Build passage index
        print("Building passage index...")
        self.passage_index = PassageIndex(self.embedder)
        self.passage_index.build_from_kronos(self.kronos)
        
        # Build SEC navigator (it builds its own graph internally)
        print("Building SEC navigator...")
        self.navigator = SECNavigator(
            kronos=self.kronos,
            embedder=self.embedder,
            device=self.device,
        )
        
        # Generator
        self.generator = CoherentGenerator(
            self.passage_index,
            self.navigator,
            self.embedder,
        )
        
        print(f"System ready: {len(self.passage_index.passages)} passages indexed")
    
    def ask(self, question: str, verbose: bool = False) -> str:
        """Ask a question and get a coherent response."""
        result = self.generator.generate(question)
        
        if verbose:
            print(f"\nQuery: {question}")
            print(f"Sources: {result['sources']}")
            print(f"Score: {result['avg_score']:.3f}")
            print(f"Time: {result['time_ms']:.1f}ms")
        
        return result['response']
    
    def ask_detailed(self, question: str) -> Dict[str, Any]:
        """Ask with full details returned."""
        return self.generator.generate(question)


# ============================================================================
# Tests
# ============================================================================

def test_passage_extraction():
    """Test passage extraction from content."""
    print("\n" + "=" * 60)
    print("TEST: Passage Extraction")
    print("=" * 60)
    
    sample_content = """
    Dawn Field Theory proposes that information and entropy are fundamental. 
    This framework suggests structure emerges from information dynamics.
    The golden ratio appears naturally in many physical systems.
    PAC conservation means that potential equals the sum of actualized parts.
    When entropy increases, structure typically decreases in complexity.
    """
    
    extractor = PassageExtractor()
    sentences = extractor.extract_sentences(sample_content)
    
    print(f"Extracted {len(sentences)} sentences:")
    for i, s in enumerate(sentences):
        print(f"  {i+1}. {s[:80]}...")
    
    passages = extractor.extract_key_passages(sample_content, max_passages=3)
    print(f"\nTop {len(passages)} passages:")
    for i, p in enumerate(passages):
        print(f"  {i+1}. {p[:80]}...")
    
    print("\n✓ Passage extraction working")
    return True


def test_passage_index():
    """Test passage indexing and search."""
    print("\n" + "=" * 60)
    print("TEST: Passage Index")
    print("=" * 60)
    
    index_path = Path(__file__).parent / "repo_knowledge_index"
    if not index_path.exists():
        print("  No index found. Run exp_04 first.")
        return False
    
    kronos = KronosGAIABridge(
        storage_path=index_path,
        namespace="dawn_institute",
        device=str(DEVICE),
    )
    
    embedder = SemanticEmbeddings(dim=128, device=str(DEVICE))
    index = PassageIndex(embedder)
    index.build_from_kronos(kronos)
    
    # Test search
    queries = [
        "What is Dawn Field Theory?",
        "How does entropy relate to structure?",
        "What is the golden ratio?",
    ]
    
    for query in queries:
        results = index.search(query, top_k=2)
        print(f"\nQuery: '{query}'")
        for i, r in enumerate(results):
            print(f"  {i+1}. [{r.score:.3f}] {r.text[:80]}...")
    
    print("\n✓ Passage index working")
    return True


def test_coherent_generation():
    """Test coherent response generation."""
    print("\n" + "=" * 60)
    print("TEST: Coherent Generation")
    print("=" * 60)
    
    index_path = Path(__file__).parent / "repo_knowledge_index"
    if not index_path.exists():
        return False
    
    system = CoherentGAIASystem(index_path)
    
    questions = [
        "What is Dawn Field Theory about?",
        "How do information and entropy interact?",
        "What role does the golden ratio play?",
        "What is PAC conservation?",
    ]
    
    for q in questions:
        response = system.ask(q, verbose=True)
        print(f"\nResponse: {response[:200]}...")
    
    print("\n✓ Coherent generation working")
    return True


def test_sec_boosted_retrieval():
    """Test SEC-boosted passage retrieval."""
    print("\n" + "=" * 60)
    print("TEST: SEC-Boosted Retrieval")
    print("=" * 60)
    
    index_path = Path(__file__).parent / "repo_knowledge_index"
    if not index_path.exists():
        return False
    
    system = CoherentGAIASystem(index_path)
    
    query = "How does information create structure in Dawn Field Theory?"
    
    # Compare with and without SEC
    result_no_sec = system.generator.generate(query, use_sec=False)
    result_with_sec = system.generator.generate(query, use_sec=True)
    
    print(f"\nQuery: {query}")
    print(f"\nWithout SEC (direct similarity):")
    print(f"  Score: {result_no_sec['avg_score']:.3f}")
    print(f"  Response: {result_no_sec['response'][:150]}...")
    
    print(f"\nWith SEC (spreading activation):")
    print(f"  Score: {result_with_sec['avg_score']:.3f}")
    print(f"  Response: {result_with_sec['response'][:150]}...")
    
    print("\n✓ SEC-boosted retrieval working")
    return True


def demo_coherent_qa():
    """Interactive demo of coherent Q&A."""
    print("\n" + "=" * 70)
    print("DEMO: Coherent Question Answering")
    print("=" * 70)
    
    index_path = Path(__file__).parent / "repo_knowledge_index"
    if not index_path.exists():
        print("No index found.")
        return
    
    system = CoherentGAIASystem(index_path)
    
    questions = [
        "What is Dawn Field Theory?",
        "How does entropy relate to information?",
        "What is the golden ratio's significance?",
        "Explain PAC conservation.",
        "What are the main predictions of the theory?",
        "How does this relate to AI systems?",
    ]
    
    print("\n" + "-" * 70)
    print("Coherent Answers from Knowledge Base")
    print("-" * 70)
    
    for q in questions:
        result = system.ask_detailed(q)
        print(f"\n❓ {q}")
        print(f"📚 Sources: {', '.join(Path(s).name for s in result['sources'][:3])}")
        print(f"📊 Score: {result['avg_score']:.3f} | Time: {result['time_ms']:.1f}ms")
        print(f"💬 {result['response']}")
        print()


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("POC-025: GAIA Prime + Kronos Integration - Experiment 09")
    print("Coherent Retrieval - Real Sentences from Knowledge Base")
    print("=" * 70)
    
    tests = [
        ("Passage Extraction", test_passage_extraction),
        ("Passage Index", test_passage_index),
        ("Coherent Generation", test_coherent_generation),
        ("SEC-Boosted Retrieval", test_sec_boosted_retrieval),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results[name] = "PASS" if passed else "FAIL"
        except Exception as e:
            results[name] = f"FAIL: {e}"
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)
    
    for name, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"  {name}: {result}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        demo_coherent_qa()
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
