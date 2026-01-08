"""
POC-025: GAIA Prime + Kronos Integration
Experiment 11: Knowledge Q&A with Clean Prose Extraction

Key improvements over exp_10:
1. Clean prose extraction - strips code, YAML, tables, headers
2. Semantic paragraph chunking - not raw sections
3. Anthropic Claude for answer synthesis
4. Focused retrieval with coherent responses

Usage:
    python exp_11_knowledge_qa.py
    python exp_11_knowledge_qa.py --query "What is PAC conservation?"
    python exp_11_knowledge_qa.py --interactive
"""

import asyncio
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "fracton"))

from fracton.storage import KronosMemory, NodeType

# Constants
PHI = 1.618033988749895  # Golden ratio
XI_THRESHOLD = 1.0571    # Balance operator
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"


# ============================================================================
# Prose Extraction - Clean, readable content only
# ============================================================================

class ProseExtractor:
    """
    Extracts clean prose from markdown files.
    Filters out code, YAML, tables, and metadata.
    """
    
    # Patterns to remove
    CODE_BLOCK = re.compile(r'```[\s\S]*?```', re.MULTILINE)
    INLINE_CODE = re.compile(r'`[^`]+`')
    YAML_FRONT = re.compile(r'^---[\s\S]*?---\n?', re.MULTILINE)
    HTML_TAGS = re.compile(r'<[^>]+>')
    LINKS_FULL = re.compile(r'\[([^\]]+)\]\([^)]+\)')  # Keep text, remove URL
    IMAGES = re.compile(r'!\[[^\]]*\]\([^)]+\)')
    TABLES = re.compile(r'\|[^\n]+\|', re.MULTILINE)
    HEADERS = re.compile(r'^#{1,6}\s+', re.MULTILINE)
    LIST_MARKERS = re.compile(r'^\s*[-*+]\s+', re.MULTILINE)
    NUMBERED_LIST = re.compile(r'^\s*\d+\.\s+', re.MULTILINE)
    BLOCKQUOTES = re.compile(r'^>\s*', re.MULTILINE)
    EMPHASIS = re.compile(r'\*{1,2}([^*]+)\*{1,2}')
    HR = re.compile(r'^[-*_]{3,}\s*$', re.MULTILINE)
    
    # Metadata patterns (often in brackets or after colons)
    METADATA_LINE = re.compile(r'^[a-z_]+:\s*.*$', re.MULTILINE | re.IGNORECASE)
    BRACKET_TAGS = re.compile(r'\[[^\]]{1,30}\](?!\()')  # Tags like [experimental]
    
    # Definition patterns - high-value content
    DEFINITION_PATTERN = re.compile(
        r'^[-*]\s+\*\*([^*]+)\*\*[:\s]+(.+)$',
        re.MULTILINE
    )
    
    @classmethod
    def extract(cls, content: str, min_paragraph_len: int = 50) -> List[str]:
        """
        Extract clean prose paragraphs from markdown content.
        
        Args:
            content: Raw markdown content
            min_paragraph_len: Minimum length for a paragraph
            
        Returns:
            List of clean prose paragraphs
        """
        # First, extract definitions (high-value content like "- **PAC**: ...")
        definitions = []
        for match in cls.DEFINITION_PATTERN.finditer(content):
            term = match.group(1).strip()
            definition = match.group(2).strip()
            # Clean emphasis markers from definition
            definition = cls.EMPHASIS.sub(r'\1', definition)
            if len(definition) > 20:
                definitions.append(f"{term}: {definition}")
        
        # Remove code blocks first (before other processing)
        text = cls.CODE_BLOCK.sub('', content)
        
        # Remove YAML frontmatter
        text = cls.YAML_FRONT.sub('', text)
        
        # Remove images
        text = cls.IMAGES.sub('', text)
        
        # Keep link text, remove URLs
        text = cls.LINKS_FULL.sub(r'\1', text)
        
        # Remove tables
        text = cls.TABLES.sub('', text)
        
        # Remove HTML
        text = cls.HTML_TAGS.sub('', text)
        
        # Remove headers but keep the text
        text = cls.HEADERS.sub('', text)
        
        # Remove horizontal rules
        text = cls.HR.sub('', text)
        
        # Clean list markers
        text = cls.LIST_MARKERS.sub('', text)
        text = cls.NUMBERED_LIST.sub('', text)
        
        # Clean blockquotes
        text = cls.BLOCKQUOTES.sub('', text)
        
        # Remove inline code
        text = cls.INLINE_CODE.sub('', text)
        
        # Clean emphasis markers but keep text
        text = cls.EMPHASIS.sub(r'\1', text)
        
        # Remove bracket tags
        text = cls.BRACKET_TAGS.sub('', text)
        
        # Split into paragraphs (double newline)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean and filter
        clean_paragraphs = []
        for p in paragraphs:
            # Clean whitespace
            p = ' '.join(p.split())
            
            # Skip if too short
            if len(p) < min_paragraph_len:
                continue
            
            # Skip if looks like metadata
            if cls._is_metadata(p):
                continue
            
            # Skip if too many special chars (likely code or data)
            if cls._has_too_many_special_chars(p):
                continue
            
            clean_paragraphs.append(p)
        
        # Prepend definitions (high-value definitional content)
        return definitions + clean_paragraphs
    
    @classmethod
    def _is_metadata(cls, text: str) -> bool:
        """Check if text looks like metadata."""
        # Starts with common metadata patterns
        meta_starts = [
            'schema_version', 'version:', 'date:', 'author:', 'title:',
            'tags:', 'categories:', 'layout:', 'permalink:', 'status:',
            'CIP-METADATA', 'description:', 'semantic_scope:', 'proficiency'
        ]
        text_lower = text.lower()
        for start in meta_starts:
            if text_lower.startswith(start.lower()):
                return True
        
        # High colon ratio suggests key:value metadata
        colons = text.count(':')
        if colons > 3 and colons / len(text) > 0.02:
            return True
        
        return False
    
    @classmethod
    def _has_too_many_special_chars(cls, text: str) -> bool:
        """Check if text has too many special characters (likely code)."""
        special = sum(1 for c in text if c in '{}[]()<>=;@#$%^&*|\\/')
        if len(text) == 0:
            return True
        ratio = special / len(text)
        return ratio > 0.1  # More than 10% special chars


# ============================================================================
# Knowledge Ingester - Clean prose only
# ============================================================================

@dataclass
class IngestStats:
    """Ingestion statistics."""
    directories: int = 0
    files: int = 0
    paragraphs: int = 0
    skipped_files: int = 0
    
    @property
    def total_nodes(self) -> int:
        return self.directories + self.files + self.paragraphs


class CleanKnowledgeIngester:
    """
    Ingests clean prose from a directory into Kronos.
    Only stores readable paragraph content, no code/metadata.
    """
    
    # Files to focus on (prose-heavy)
    PROSE_EXTENSIONS = {'.md', '.txt', '.rst'}
    
    # Directories to skip
    SKIP_DIRS = {
        '.git', '__pycache__', 'node_modules', '.venv', 'venv',
        '.cip', 'cache', 'logs', '.changelog', 'results', 'data',
        'scripts', 'tools', 'examples', 'tests'
    }
    
    # Focus on documentation directories
    PRIORITY_DIRS = {
        'docs', 'documentation', 'foundational', 'theory',
        'papers', 'preprints', 'experiments'
    }
    
    def __init__(self, memory: KronosMemory):
        self.memory = memory
        self.stats = IngestStats()
        self.node_ids: Dict[str, str] = {}
        self.extractor = ProseExtractor()
    
    async def ingest(
        self,
        root_path: Path,
        graph_name: str = "knowledge",
        max_depth: int = 4,
    ) -> str:
        """Ingest a directory tree."""
        await self.memory.create_graph(graph_name, f"Knowledge from {root_path.name}")
        return await self._ingest_directory(root_path, graph_name, None, max_depth, 0)
    
    async def _ingest_directory(
        self,
        dir_path: Path,
        graph: str,
        parent_id: Optional[str],
        max_depth: int,
        depth: int,
    ) -> str:
        """Recursively ingest a directory."""
        if depth > max_depth:
            return None
        
        if dir_path.name in self.SKIP_DIRS:
            return None
        
        # Create directory node
        dir_id = await self.memory.store(
            content=f"Topic: {dir_path.name}",
            graph=graph,
            node_type=NodeType.CONCEPT,
            parent_id=parent_id,
            metadata={
                "type": "topic",
                "name": dir_path.name,
                "path": str(dir_path),
                "depth": depth,
            }
        )
        self.stats.directories += 1
        
        # Process files first
        for file_path in sorted(dir_path.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in self.PROSE_EXTENSIONS:
                await self._ingest_file(file_path, graph, dir_id, depth + 1)
        
        # Then subdirectories (prioritize docs directories)
        subdirs = [d for d in dir_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        # Sort: priority dirs first
        subdirs.sort(key=lambda d: (0 if d.name.lower() in self.PRIORITY_DIRS else 1, d.name))
        
        for subdir in subdirs:
            await self._ingest_directory(subdir, graph, dir_id, max_depth, depth + 1)
        
        return dir_id
    
    async def _ingest_file(
        self,
        file_path: Path,
        graph: str,
        parent_id: str,
        depth: int,
    ):
        """Ingest a file's prose content."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            self.stats.skipped_files += 1
            return
        
        # Extract clean prose paragraphs
        paragraphs = self.extractor.extract(content)
        
        if not paragraphs:
            self.stats.skipped_files += 1
            return
        
        # Create file node with summary
        summary = paragraphs[0][:300] if paragraphs else ""
        file_id = await self.memory.store(
            content=f"Document: {file_path.stem} — {summary}",
            graph=graph,
            node_type=NodeType.FACT,
            parent_id=parent_id,
            metadata={
                "type": "document",
                "filename": file_path.name,
                "path": str(file_path),
                "depth": depth,
            }
        )
        self.stats.files += 1
        
        # Store each paragraph as knowledge fact
        for i, para in enumerate(paragraphs[:10]):  # Max 10 paragraphs per file
            # Check if this is a definition (high-value content)
            is_definition = self._is_definition(para)
            passage_type = "definition" if is_definition else "passage"
            
            await self.memory.store(
                content=para,
                graph=graph,
                node_type=NodeType.FACT,
                parent_id=file_id,
                metadata={
                    "type": passage_type,
                    "source": file_path.name,
                    "paragraph": i,
                    "depth": depth + 1,
                    "is_definition": is_definition,
                }
            )
            self.stats.paragraphs += 1
    
    @staticmethod
    def _is_definition(text: str) -> bool:
        """Check if text looks like a definition of a key concept."""
        # Key terms that are often defined
        key_terms = ['PAC', 'SEC', 'MED', 'phi', 'Xi', 'golden ratio', 'entropy', 
                     'information', 'conservation', 'collapse', 'GAIA', 'infodynamics']
        
        text_lower = text.lower()
        
        # Definition patterns: "X: definition" or "X is/means/represents"
        for term in key_terms:
            term_lower = term.lower()
            # Pattern: "TERM: description" or "TERM (Full Name): description"
            if f"{term_lower}:" in text_lower or f"{term_lower} (" in text_lower:
                return True
            # Pattern: "TERM is/means/represents"
            if f"{term_lower} is " in text_lower or f"{term_lower} represents " in text_lower:
                return True
        
        # Pattern: starts with bolded term followed by colon (from markdown)
        if ':' in text[:50] and any(t in text[:50] for t in key_terms):
            return True
        
        return False


# ============================================================================
# Knowledge Retrieval with Answer Synthesis
# ============================================================================

@dataclass
class RetrievalResult:
    """A retrieved knowledge passage."""
    content: str
    score: float
    source: str
    passage_type: str
    
    def __str__(self) -> str:
        return f"[{self.score:.3f}] {self.content[:100]}..."


class KnowledgeRetriever:
    """Retrieves relevant knowledge passages using SEC resonance."""
    
    def __init__(self, memory: KronosMemory):
        self.memory = memory
    
    async def retrieve(
        self,
        query: str,
        graph: str = "knowledge",
        limit: int = 10,
    ) -> List[RetrievalResult]:
        """Retrieve relevant passages for a query."""
        results = await self.memory.query(
            query_text=query,
            graphs=[graph],
            limit=limit,
            expand_graph=True,
        )
        
        # Extract key terms from query for boosting
        query_lower = query.lower()
        key_terms = ['pac', 'sec', 'med', 'phi', 'xi', 'golden ratio', 'entropy', 
                     'information', 'conservation', 'collapse', 'infodynamics']
        query_terms = [t for t in key_terms if t in query_lower]
        
        retrieved = []
        for r in results:
            # Skip topic/document nodes, focus on passages
            node_type = r.node.metadata.get('type', '')
            if node_type in ('topic', 'directory'):
                continue
            
            content_lower = r.node.content.lower()
            
            # Boost score for definitions (1.5x)
            is_definition = r.node.metadata.get('is_definition', False)
            boosted_score = r.score * 1.5 if is_definition else r.score
            
            # Boost for query term matches (1.2x per matched term)
            for term in query_terms:
                if f"{term}:" in content_lower or f"{term} (" in content_lower:
                    # Strong term match (definition-style)
                    boosted_score *= 1.4
                elif term in content_lower:
                    # Weak term match
                    boosted_score *= 1.1
            
            retrieved.append(RetrievalResult(
                content=r.node.content,
                score=boosted_score,
                source=r.node.metadata.get('source', r.node.metadata.get('filename', 'unknown')),
                passage_type=node_type,
            ))
        
        # Re-sort by boosted score
        retrieved.sort(key=lambda x: x.score, reverse=True)
        
        return retrieved


class AnswerSynthesizer:
    """
    Synthesizes coherent answers from retrieved passages.
    Uses Claude if available, otherwise rule-based.
    """
    
    def __init__(self, memory: KronosMemory, use_llm: bool = True):
        self.memory = memory
        self.retriever = KnowledgeRetriever(memory)
        self.use_llm = use_llm
        self._anthropic = None
        
        # Try to get API key
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if self.api_key and use_llm:
            try:
                import anthropic
                self._anthropic = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                print("⚠️  anthropic package not installed, using rule-based answers")
    
    async def answer(
        self,
        question: str,
        graph: str = "knowledge",
        max_passages: int = 5,
    ) -> Dict[str, Any]:
        """Generate an answer to a question."""
        start = time.perf_counter()
        
        # Retrieve relevant passages
        passages = await self.retriever.retrieve(question, graph, limit=max_passages * 2)
        
        # Filter for actual content passages
        passages = [p for p in passages if p.passage_type == 'passage'][:max_passages]
        
        if not passages:
            return {
                "answer": "I couldn't find relevant information to answer that question.",
                "sources": [],
                "confidence": 0.0,
                "time_ms": (time.perf_counter() - start) * 1000,
            }
        
        # Generate answer
        if self._anthropic:
            answer = await self._llm_answer(question, passages)
        else:
            answer = self._rule_based_answer(question, passages)
        
        sources = list(set(p.source for p in passages))
        avg_score = sum(p.score for p in passages) / len(passages)
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": avg_score,
            "passages": passages,
            "time_ms": (time.perf_counter() - start) * 1000,
        }
    
    async def _llm_answer(
        self,
        question: str,
        passages: List[RetrievalResult],
    ) -> str:
        """Generate answer using Claude."""
        context = "\n\n".join([
            f"[Source: {p.source}]\n{p.content}"
            for p in passages
        ])
        
        try:
            message = self._anthropic.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                system="""You are a knowledge assistant for Dawn Field Theory, a research framework exploring information and entropy as foundations of reality.

Answer the question using ONLY the provided context passages. Be concise and accurate. If the context doesn't fully answer the question, say what you can determine and note what's missing.

Key concepts you might encounter:
- PAC (Potential-Actualization Conservation): Conservation law for information
- SEC (Symbolic Entropy Collapse): How structure emerges from entropy gradients
- φ (phi): Golden ratio 1.618..., appears as a natural constant
- Ξ (Xi): Balance operator ~1.057, threshold for collapse
- MED: Macro Emergence Dynamics""",
                messages=[
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}"
                    }
                ]
            )
            return message.content[0].text
        except Exception as e:
            print(f"⚠️  LLM error: {e}")
            return self._rule_based_answer(question, passages)
    
    def _rule_based_answer(
        self,
        question: str,
        passages: List[RetrievalResult],
    ) -> str:
        """Generate answer using rule-based approach."""
        if not passages:
            return "No relevant information found."
        
        # Take top passage as primary answer
        best = passages[0]
        
        # If score is low, be honest
        if best.score < 0.4:
            return f"I found some potentially related content, but I'm not confident it answers your question:\n\n{best.content[:500]}"
        
        # Otherwise, present the best passage
        answer = best.content
        
        # Add supporting info if available
        if len(passages) > 1 and passages[1].score > 0.5:
            answer += f"\n\nRelated: {passages[1].content[:200]}"
        
        return answer


# ============================================================================
# Main Experiment
# ============================================================================

async def run_ingestion() -> KronosMemory:
    """Ingest knowledge base."""
    print("\n" + "="*70)
    print("KNOWLEDGE INGESTION")
    print("="*70)
    
    # Fresh storage
    storage_path = Path("./kronos_qa")
    if storage_path.exists():
        import shutil
        shutil.rmtree(storage_path)
    
    memory = KronosMemory(
        storage_path=storage_path,
        namespace="knowledge",
        device=DEVICE,
        embedding_model="mini",
    )
    await memory.connect()
    
    # Ingest
    ingester = CleanKnowledgeIngester(memory)
    dft_path = Path(r"c:\Users\peter\repos\Dawn Field Institute\dawn-field-theory")
    
    print(f"\nIngesting clean prose from: {dft_path.name}")
    await ingester.ingest(dft_path, "knowledge", max_depth=4)
    
    print(f"\n[OK] Ingestion complete:")
    print(f"  Topics: {ingester.stats.directories}")
    print(f"  Documents: {ingester.stats.files}")
    print(f"  Passages: {ingester.stats.paragraphs}")
    print(f"  Skipped: {ingester.stats.skipped_files}")
    print(f"  Total nodes: {ingester.stats.total_nodes}")
    
    return memory


async def run_qa_tests(memory: KronosMemory):
    """Run Q&A tests."""
    print("\n" + "="*70)
    print("Q&A TESTS")
    print("="*70)
    
    synthesizer = AnswerSynthesizer(memory, use_llm=True)
    
    questions = [
        "What is Dawn Field Theory?",
        "What is PAC conservation and how does it work?",
        "How do information and entropy relate in this framework?",
        "What is the significance of the golden ratio phi?",
        "What is the Xi constant?",
    ]
    
    for q in questions:
        print(f"\n{'-'*60}")
        print(f"Q: {q}")
        print(f"{'-'*60}")
        
        result = await synthesizer.answer(q)
        
        # Debug: show raw retrieved passages
        if 'passages' in result:
            print("\nRetrieved passages:")
            for p in result['passages'][:3]:
                preview = p.content[:80].encode('ascii', 'replace').decode('ascii')
                print(f"  [{p.score:.3f}] {p.passage_type}: {preview}...")
        
        print(f"\nAnswer:")
        # Handle unicode encoding issues
        answer_text = result['answer'][:600].encode('ascii', 'replace').decode('ascii')
        print(answer_text)
        if len(result['answer']) > 600:
            print("...")
        
        sources_text = ', '.join(result['sources'][:3])
        print(f"\nSources: {sources_text.encode('ascii', 'replace').decode('ascii')}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Time: {result['time_ms']:.1f}ms")


async def interactive_mode(memory: KronosMemory):
    """Interactive Q&A mode."""
    print("\n" + "="*70)
    print("INTERACTIVE Q&A")
    print("Type 'quit' to exit")
    print("="*70)
    
    synthesizer = AnswerSynthesizer(memory, use_llm=True)
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            if question.lower() in ('quit', 'exit', 'q'):
                break
            if not question:
                continue
            
            result = await synthesizer.answer(question)
            
            print(f"\nAnswer: {result['answer']}")
            print(f"\nSources: {', '.join(result['sources'][:3])}")
            print(f"Confidence: {result['confidence']:.3f}")
            
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Dawn Field Theory Knowledge Q&A")
    parser.add_argument("--query", "-q", type=str, help="Single query to run")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--test", "-t", action="store_true", help="Run test suite")
    args = parser.parse_args()
    
    print("="*70)
    print("POC-025: GAIA Prime + Kronos Integration")
    print("Experiment 11: Knowledge Q&A with Clean Prose Extraction")
    print("="*70)
    print(f"Device: {DEVICE}")
    
    # Ingest knowledge
    memory = await run_ingestion()
    
    if args.query:
        # Single query mode
        synthesizer = AnswerSynthesizer(memory, use_llm=True)
        result = await synthesizer.answer(args.query)
        print(f"\n{'─'*60}")
        print(f"❓ {args.query}")
        print(f"{'─'*60}")
        print(f"\n💬 {result['answer']}")
        print(f"\n📚 Sources: {', '.join(result['sources'][:3])}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
    
    elif args.interactive:
        await interactive_mode(memory)
    
    else:
        # Default: run tests
        await run_qa_tests(memory)
    
    # Stats
    stats = await memory.get_stats()
    print(f"\n📊 Final Stats: {stats.get('total_nodes', 'N/A')} nodes, {stats.get('queries', 'N/A')} queries")


if __name__ == "__main__":
    asyncio.run(main())
