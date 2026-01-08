"""
POC-025: GAIA Prime + Kronos Integration
========================================

Experiment 04: Repository Knowledge Index

Indexes repository files into Kronos, enabling GAIA to answer
questions about content it was never trained on.

This demonstrates:
1. File → embedding → Kronos crystallization
2. Question → embedding → Kronos recall
3. Retrieved context enables "knowledge" GAIA doesn't have

Essentially: RAG with PAC/SEC architecture.
"""

import sys
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\fracton")
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\dawn-models\research\GAIA\src")

import torch
import numpy as np
import time
import shutil
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass

# Import from previous experiments
from exp_01_bridge import KronosGAIABridge, CrystallizedPattern

# GAIA Prime imports
from gaia_prime.validated_constants import PHI, PHI_INV, XI


# ============================================================================
# Simple Text Embedder (for demo - real system would use sentence-transformers)
# ============================================================================

class SimpleTextEmbedder:
    """
    Simple deterministic text embedder for demo purposes.
    
    In production, this would be replaced with:
    - sentence-transformers
    - OpenAI embeddings
    - GAIA's own grafted embeddings
    
    This version creates consistent embeddings from text using
    a hash-based approach that preserves some semantic signal
    (similar words have similar hashes of n-grams).
    """
    
    def __init__(self, dim: int = 768, device: str = 'cuda'):
        self.dim = dim
        self.device = device
        
        # Pre-compute random projection matrix (fixed seed for consistency)
        torch.manual_seed(42)
        self.projection = torch.randn(10000, dim, device=device) / np.sqrt(dim)
    
    def embed(self, text: str) -> torch.Tensor:
        """
        Embed text into a vector.
        
        Uses bag-of-ngrams with hash trick for consistent embeddings.
        """
        # Normalize text
        text = text.lower().strip()
        
        # Generate n-grams (1, 2, 3)
        words = text.split()
        ngrams = []
        for n in [1, 2, 3]:
            for i in range(len(words) - n + 1):
                ngrams.append(' '.join(words[i:i+n]))
        
        # Also add character 3-grams for robustness
        for i in range(len(text) - 2):
            ngrams.append(text[i:i+3])
        
        if not ngrams:
            # Empty text - return zero vector
            return torch.zeros(self.dim, device=self.device)
        
        # Hash each n-gram to an index
        indices = []
        for ng in ngrams:
            h = int(hashlib.md5(ng.encode()).hexdigest(), 16)
            indices.append(h % 10000)
        
        # Sum the corresponding projection rows
        embedding = self.projection[indices].sum(dim=0)
        
        # Normalize
        norm = embedding.norm()
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        """Embed multiple texts."""
        return torch.stack([self.embed(t) for t in texts])


# ============================================================================
# Repository Knowledge Index
# ============================================================================

@dataclass
class IndexedDocument:
    """A document indexed in Kronos."""
    doc_id: str
    file_path: str
    content: str
    chunk_index: int
    total_chunks: int
    
    def summary(self, max_len: int = 100) -> str:
        """Get a short summary."""
        if len(self.content) <= max_len:
            return self.content
        return self.content[:max_len] + "..."


class RepoKnowledgeIndex:
    """
    Index repository files into Kronos for retrieval.
    
    This turns any repo into a knowledge base that GAIA can query.
    
    Features:
    1. Recursive file scanning
    2. Chunking for long documents
    3. Embedding and crystallization
    4. Semantic search
    """
    
    def __init__(
        self,
        storage_path: Path,
        namespace: str = "repo_index",
        device: str = 'cuda',
        embed_dim: int = 768,
        chunk_size: int = 500,  # words per chunk
        chunk_overlap: int = 50,
    ):
        self.storage_path = Path(storage_path)
        self.namespace = namespace
        self.device = device
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize Kronos bridge
        self.kronos = KronosGAIABridge(
            storage_path=storage_path,
            namespace=namespace,
            device=device,
            embed_dim=embed_dim,
        )
        
        # Initialize embedder
        self.embedder = SimpleTextEmbedder(dim=embed_dim, device=device)
        
        # Document metadata store
        self.documents: Dict[str, IndexedDocument] = {}
        
        # File extensions to index
        self.indexable_extensions = {
            '.md', '.txt', '.py', '.yaml', '.yml', '.json',
            '.rst', '.html', '.css', '.js', '.ts',
        }
        
        # Stats
        self.stats = {
            'files_indexed': 0,
            'chunks_created': 0,
            'queries_processed': 0,
        }
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        
        if len(words) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks
    
    def index_file(self, file_path: Path) -> int:
        """
        Index a single file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Number of chunks indexed
        """
        if file_path.suffix.lower() not in self.indexable_extensions:
            return 0
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
            return 0
        
        # Skip empty files
        if not content.strip():
            return 0
        
        # Chunk the content
        chunks = self._chunk_text(content)
        
        # Index each chunk
        for i, chunk in enumerate(chunks):
            doc_id = f"{file_path.stem}_{i:04d}"
            
            # Embed the chunk
            embedding = self.embedder.embed(chunk)
            
            # Create document record
            doc = IndexedDocument(
                doc_id=doc_id,
                file_path=str(file_path),
                content=chunk,
                chunk_index=i,
                total_chunks=len(chunks),
            )
            self.documents[doc_id] = doc
            
            # Crystallize in Kronos
            self.kronos.crystallize(
                pattern_id=doc_id,
                delta=embedding,
                potential=1.0,
                importance=PHI,  # All indexed content is important
                metadata={
                    'file_path': str(file_path),
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'content_preview': chunk[:200],
                },
            )
        
        self.stats['files_indexed'] += 1
        self.stats['chunks_created'] += len(chunks)
        
        return len(chunks)
    
    def index_directory(
        self,
        dir_path: Path,
        recursive: bool = True,
        max_files: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Index all files in a directory.
        
        Args:
            dir_path: Directory to index
            recursive: Whether to recurse into subdirs
            max_files: Maximum files to index (for testing)
            
        Returns:
            Stats about indexing
        """
        dir_path = Path(dir_path)
        
        if recursive:
            files = list(dir_path.rglob('*'))
        else:
            files = list(dir_path.glob('*'))
        
        # Filter to indexable files
        files = [f for f in files if f.is_file() and f.suffix.lower() in self.indexable_extensions]
        
        if max_files:
            files = files[:max_files]
        
        print(f"Indexing {len(files)} files from {dir_path}")
        
        total_chunks = 0
        for i, file_path in enumerate(files):
            chunks = self.index_file(file_path)
            total_chunks += chunks
            if (i + 1) % 10 == 0:
                print(f"  Indexed {i+1}/{len(files)} files ({total_chunks} chunks)")
        
        print(f"Done! {len(files)} files, {total_chunks} chunks")
        
        return {
            'files_indexed': len(files),
            'chunks_created': total_chunks,
        }
    
    def query(
        self,
        question: str,
        top_k: int = 5,
    ) -> List[Tuple[IndexedDocument, float]]:
        """
        Query the index with a question.
        
        Args:
            question: Natural language question
            top_k: Number of results
            
        Returns:
            List of (document, similarity_score) tuples
        """
        # Embed the question
        query_embedding = self.embedder.embed(question)
        
        # Recall from Kronos
        result = self.kronos.recall(query_embedding, top_k=top_k)
        
        self.stats['queries_processed'] += 1
        
        # Map back to documents
        docs_with_scores = []
        for pattern, score in zip(result.patterns, result.similarity_scores):
            if pattern.id in self.documents:
                docs_with_scores.append((self.documents[pattern.id], score))
        
        return docs_with_scores
    
    def answer_question(
        self,
        question: str,
        top_k: int = 3,
    ) -> str:
        """
        Answer a question using retrieved context.
        
        This is a simple demo - real implementation would use
        GAIA's generator with the retrieved context.
        
        Args:
            question: Question to answer
            top_k: Number of context chunks to use
            
        Returns:
            Answer with sources
        """
        results = self.query(question, top_k=top_k)
        
        if not results:
            return "I don't have any relevant information about that."
        
        # Build answer from retrieved context
        answer_parts = [f"Based on the indexed documents, here's what I found:\n"]
        
        for i, (doc, score) in enumerate(results):
            source = Path(doc.file_path).name
            answer_parts.append(f"\n**Source {i+1}** ({source}, relevance: {score:.2%}):")
            answer_parts.append(f"```\n{doc.summary(300)}\n```")
        
        return '\n'.join(answer_parts)
    
    def sync(self):
        """Save index to disk."""
        self.kronos.sync()


# ============================================================================
# Tests
# ============================================================================

def test_basic_indexing():
    """Test basic file indexing."""
    print("\n" + "=" * 60)
    print("TEST: Basic Indexing")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_path = Path(__file__).parent / "test_index"
    
    # Clean up
    if test_path.exists():
        shutil.rmtree(test_path)
    
    # Create test files
    test_docs = test_path / "docs"
    test_docs.mkdir(parents=True)
    
    (test_docs / "readme.md").write_text("""
# Dawn Field Theory

Dawn Field Theory explores the hypothesis that information and entropy
are generative foundations of reality. The central question: what if 
structure emerges from information dynamics?

## Key Constants

- PHI: Golden ratio, 1.618...
- XI: Balance operator, 1.0571
- LAMBDA_STAR: Critical decay threshold
""", encoding='utf-8')
    
    (test_docs / "pac.md").write_text("""
# PAC - Potential-Actualization Conservation

PAC is the first pillar of Dawn Field Theory.

Formula: f(Parent) = Sum of f(Children)

When potential becomes actual, the total is conserved but redistributed.
This applies across value, complexity, and effect.
""", encoding='utf-8')
    
    (test_docs / "sec.md").write_text("""
# SEC - Symbolic Entropy Collapse

SEC describes how structure forms from entropy gradients.

Formula: dS/dt = alpha * grad(I) - beta * grad(H)

Structure forms when information gradient dominates.
Collapse occurs when entropy gradient overtakes.
""", encoding='utf-8')
    
    # Create index
    index = RepoKnowledgeIndex(
        storage_path=test_path / "kronos",
        namespace="test",
        device=device,
    )
    
    # Index the docs
    stats = index.index_directory(test_docs)
    
    print(f"\nIndexed: {stats}")
    
    assert stats['files_indexed'] == 3
    print("\n✓ Files indexed correctly")
    
    # Clean up
    shutil.rmtree(test_path)
    return True


def test_semantic_query():
    """Test semantic querying."""
    print("\n" + "=" * 60)
    print("TEST: Semantic Query")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_path = Path(__file__).parent / "test_index"
    
    # Clean up
    if test_path.exists():
        shutil.rmtree(test_path)
    
    # Create test files with distinct topics
    test_docs = test_path / "docs"
    test_docs.mkdir(parents=True)
    
    (test_docs / "physics.md").write_text("""
Quantum mechanics describes the behavior of particles at small scales.
The wave function represents the probability amplitude of finding a particle.
Heisenberg's uncertainty principle limits simultaneous measurement precision.
""")
    
    (test_docs / "cooking.md").write_text("""
To make a perfect omelette, start with fresh eggs at room temperature.
Whisk the eggs thoroughly until the yolks and whites are combined.
Cook over medium heat with butter for best results.
""")
    
    (test_docs / "gardening.md").write_text("""
Tomatoes need full sun and consistent watering to thrive.
Plant seedlings after the last frost date in your area.
Support plants with stakes or cages as they grow.
""")
    
    # Create index
    index = RepoKnowledgeIndex(
        storage_path=test_path / "kronos",
        namespace="test",
        device=device,
    )
    
    index.index_directory(test_docs)
    
    # Query about physics
    results = index.query("What is the uncertainty principle?", top_k=3)
    
    print("\nQuery: 'What is the uncertainty principle?'")
    for doc, score in results:
        source = Path(doc.file_path).name
        print(f"  {source}: {score:.3f}")
    
    # Physics should be top result
    top_source = Path(results[0][0].file_path).name
    assert top_source == "physics.md", f"Expected physics.md, got {top_source}"
    print("\n✓ Correct document retrieved for physics query")
    
    # Query about cooking
    results = index.query("How do I cook eggs?", top_k=3)
    
    print("\nQuery: 'How do I cook eggs?'")
    for doc, score in results:
        source = Path(doc.file_path).name
        print(f"  {source}: {score:.3f}")
    
    top_source = Path(results[0][0].file_path).name
    assert top_source == "cooking.md", f"Expected cooking.md, got {top_source}"
    print("\n✓ Correct document retrieved for cooking query")
    
    # Clean up
    shutil.rmtree(test_path)
    return True


def test_real_repo_index():
    """Test indexing real repo files."""
    print("\n" + "=" * 60)
    print("TEST: Real Repo Index (Dawn Field Theory)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_path = Path(__file__).parent / "test_index"
    
    # Clean up
    if test_path.exists():
        shutil.rmtree(test_path)
    
    # Index real repo docs
    repo_path = Path(r"c:\Users\peter\repos\Dawn Field Institute\dawn-field-theory")
    
    if not repo_path.exists():
        print("  Repo not found, skipping")
        return True
    
    # Create index
    index = RepoKnowledgeIndex(
        storage_path=test_path / "kronos",
        namespace="dawn_theory",
        device=device,
    )
    
    # Index just the top-level markdown files (quick test)
    stats = index.index_directory(repo_path, recursive=False, max_files=10)
    
    print(f"\nIndexed: {stats}")
    
    # Test some queries about Dawn Field Theory
    questions = [
        "What is XI constant?",
        "What is PAC conservation?",
        "What is the golden ratio?",
        "What is infodynamics?",
    ]
    
    print("\n--- Knowledge Retrieval Demo ---\n")
    
    for q in questions:
        print(f"Q: {q}")
        results = index.query(q, top_k=2)
        if results:
            for doc, score in results[:2]:
                source = Path(doc.file_path).name
                preview = doc.content[:150].replace('\n', ' ')
                print(f"  [{score:.2%}] {source}: {preview}...")
        else:
            print("  No relevant documents found")
        print()
    
    print("✓ Real repo indexed and queryable")
    
    # Clean up
    shutil.rmtree(test_path)
    return True


def test_answer_generation():
    """Test answer generation from retrieved context."""
    print("\n" + "=" * 60)
    print("TEST: Answer Generation")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_path = Path(__file__).parent / "test_index"
    
    # Clean up
    if test_path.exists():
        shutil.rmtree(test_path)
    
    # Create knowledge base
    test_docs = test_path / "docs"
    test_docs.mkdir(parents=True)
    
    (test_docs / "constants.md").write_text("""
# Dawn Field Theory Constants

## XI - The Balance Operator

XI is derived from the Mobius spectral ratio:
XI = 1 + pi/F10 = 1 + pi/55 = 1.0571

This is NOT a fitting parameter - it emerges from topology.
The 55 comes from the 10th Fibonacci number.

## PHI - Golden Ratio

PHI = (1 + sqrt(5)) / 2 = 1.618

PHI appears throughout nature and Dawn Field Theory as the
optimal growth constant.
""", encoding='utf-8')
    
    # Create index
    index = RepoKnowledgeIndex(
        storage_path=test_path / "kronos",
        namespace="test",
        device=device,
    )
    
    index.index_directory(test_docs)
    
    # Ask a question
    question = "What is the XI constant and where does it come from?"
    answer = index.answer_question(question, top_k=2)
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{answer}")
    
    # Answer should mention key concepts
    assert "1.0571" in answer or "Mobius" in answer or "topology" in answer or "Fibonacci" in answer
    print("\n✓ Answer contains relevant information")
    
    # Clean up
    shutil.rmtree(test_path)
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("POC-025: GAIA Prime + Kronos Integration - Experiment 04")
    print("Repository Knowledge Index")
    print("=" * 70)
    
    tests = [
        ("Basic Indexing", test_basic_indexing),
        ("Semantic Query", test_semantic_query),
        ("Real Repo Index", test_real_repo_index),
        ("Answer Generation", test_answer_generation),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            import traceback
            results.append((name, False, str(e)))
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    passed = 0
    for name, success, error in results:
        status = "✓ PASS" if success else f"✗ FAIL: {error}"
        print(f"  {name}: {status}")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(tests)} tests passed")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
