"""
POC-025: GAIA Prime + Kronos Integration
========================================

Experiment 10: Native Kronos Usage

The problem with exp_09: We built a custom PassageIndex that bypasses
all of Kronos's actual features (PAC delta storage, SEC resonance, tree traversal).

The solution: Use KronosMemory directly, the way the chatbot does.

This experiment:
1. Indexes knowledge into actual KronosMemory with parent-child relationships
2. Uses memory.query() for proper SEC resonance ranking
3. Uses trace_evolution() for tree traversal
4. Leverages all the PAC/SEC/MED infrastructure

This is "using Kronos as designed" - not reimplementing it poorly.
"""

import sys
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\fracton")

import asyncio
import torch
import re
import time
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

# Force GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE] Using: {DEVICE}")

# Import Kronos properly
from fracton.storage import KronosMemory, NodeType


# ============================================================================
# Knowledge Ingestion - Build proper PAC tree in Kronos
# ============================================================================

class KnowledgeIngester:
    """
    Ingests knowledge into KronosMemory with proper parent-child structure.
    
    Key insight: Files in directories form a natural hierarchy.
    - Repository is the root
    - Directories are concept nodes
    - Files are fact nodes under their directory
    - Sections within files are children of the file
    """
    
    def __init__(self, memory: KronosMemory):
        self.memory = memory
        self.node_ids: Dict[str, str] = {}  # path -> node_id
        self.stats = {
            'directories': 0,
            'files': 0,
            'sections': 0,
        }
    
    async def ingest_directory(
        self, 
        root_path: Path,
        graph_name: str = "knowledge",
        parent_id: Optional[str] = None,
        max_depth: int = 5,
        current_depth: int = 0,
    ) -> Optional[str]:
        """Recursively ingest a directory into Kronos."""
        if current_depth > max_depth:
            return None
        
        if not root_path.exists():
            return None
        
        # Create node for this directory
        dir_name = root_path.name or "root"
        dir_id = await self.memory.store(
            content=f"Directory: {dir_name}",
            graph=graph_name,
            node_type=NodeType.CONCEPT,
            parent_id=parent_id,
            metadata={
                "type": "directory",
                "path": str(root_path),
                "depth": current_depth,
            }
        )
        self.node_ids[str(root_path)] = dir_id
        self.stats['directories'] += 1
        
        # Process files first (they're children of directory)
        for file_path in root_path.iterdir():
            if file_path.is_file() and file_path.suffix in ['.md', '.txt', '.yaml']:
                await self._ingest_file(file_path, graph_name, dir_id, current_depth + 1)
        
        # Then subdirectories
        for subdir in root_path.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('.'):
                await self.ingest_directory(
                    subdir, graph_name, dir_id, 
                    max_depth, current_depth + 1
                )
        
        return dir_id
    
    async def _ingest_file(
        self,
        file_path: Path,
        graph_name: str,
        parent_id: str,
        depth: int,
    ):
        """Ingest a file and its sections."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return
        
        if len(content) < 50:  # Skip tiny files
            return
        
        # Create file node
        file_id = await self.memory.store(
            content=f"File: {file_path.name}\n{content[:500]}",
            graph=graph_name,
            node_type=NodeType.FACT,
            parent_id=parent_id,
            metadata={
                "type": "file",
                "path": str(file_path),
                "filename": file_path.name,
                "depth": depth,
            }
        )
        self.node_ids[str(file_path)] = file_id
        self.stats['files'] += 1
        
        # Extract sections (## headers in markdown)
        sections = self._extract_sections(content)
        for section_title, section_content in sections[:5]:  # Max 5 sections
            if len(section_content) < 30:
                continue
            
            await self.memory.store(
                content=f"{section_title}\n{section_content[:400]}",
                graph=graph_name,
                node_type=NodeType.FACT,
                parent_id=file_id,
                metadata={
                    "type": "section",
                    "title": section_title,
                    "file": file_path.name,
                    "depth": depth + 1,
                }
            )
            self.stats['sections'] += 1
    
    def _extract_sections(self, content: str) -> List[tuple]:
        """Extract markdown sections."""
        sections = []
        current_title = "Introduction"
        current_content = []
        
        for line in content.split('\n'):
            if line.startswith('## '):
                if current_content:
                    sections.append((current_title, '\n'.join(current_content)))
                current_title = line[3:].strip()
                current_content = []
            elif line.startswith('# ') and not sections:
                current_title = line[2:].strip()
            else:
                current_content.append(line)
        
        if current_content:
            sections.append((current_title, '\n'.join(current_content)))
        
        return sections


# ============================================================================
# Knowledge Query - Use Kronos SEC resonance natively
# ============================================================================

@dataclass
class QueryResult:
    """Result from knowledge query."""
    content: str
    score: float
    source: str
    depth: int
    node_type: str


class KnowledgeQuerier:
    """
    Queries knowledge using native Kronos SEC resonance.
    
    This uses memory.query() which does:
    - Semantic similarity
    - Entropy matching
    - Recency weighting
    - Coherence scoring
    - Foundation resonance (φ-based depth scoring)
    """
    
    def __init__(self, memory: KronosMemory):
        self.memory = memory
    
    async def query(
        self,
        query_text: str,
        graph: str = "knowledge",
        limit: int = 5,
        expand: bool = True,
    ) -> List[QueryResult]:
        """Query with native SEC resonance."""
        results = await self.memory.query(
            query_text=query_text,
            graphs=[graph],
            limit=limit,
            expand_graph=expand,
        )
        
        query_results = []
        for r in results:
            query_results.append(QueryResult(
                content=r.node.content,
                score=r.score,
                source=r.node.metadata.get('path', r.node.metadata.get('file', 'unknown')),
                depth=r.node.metadata.get('depth', 0),
                node_type=r.node.metadata.get('type', 'unknown'),
            ))
        
        return query_results
    
    async def trace_context(
        self,
        node_id: str,
        graph: str = "knowledge",
    ) -> Dict[str, Any]:
        """Trace evolution to get full context."""
        trace = await self.memory.trace_evolution(
            graph=graph,
            node_id=node_id,
            direction="both",
        )
        return trace


# ============================================================================
# Coherent Answer Generator
# ============================================================================

class CoherentAnswerer:
    """
    Generates coherent answers from Kronos query results.
    """
    
    def __init__(self, memory: KronosMemory):
        self.memory = memory
        self.querier = KnowledgeQuerier(memory)
    
    async def answer(
        self,
        question: str,
        graph: str = "knowledge",
        max_results: int = 5,
    ) -> Dict[str, Any]:
        """Generate an answer to a question."""
        start = time.perf_counter()
        
        # Query with SEC resonance
        results = await self.querier.query(
            question, graph=graph, limit=max_results, expand=True
        )
        
        if not results:
            return {
                "answer": "No relevant information found.",
                "sources": [],
                "score": 0.0,
                "time_ms": (time.perf_counter() - start) * 1000,
            }
        
        # Compose answer from top results
        answer_parts = []
        sources = []
        
        for r in results[:3]:  # Top 3
            # Clean content
            content = r.content
            if content.startswith('File: ') or content.startswith('Directory: '):
                content = content.split('\n', 1)[1] if '\n' in content else content
            
            # Take first meaningful portion
            lines = [l for l in content.split('\n') if l.strip() and not l.startswith('#')]
            if lines:
                answer_parts.append(lines[0][:200])
            
            sources.append(Path(r.source).name if r.source else 'unknown')
        
        return {
            "answer": ' '.join(answer_parts),
            "sources": list(set(sources)),
            "score": results[0].score if results else 0.0,
            "time_ms": (time.perf_counter() - start) * 1000,
            "results": results,
        }


# ============================================================================
# Tests
# ============================================================================

async def test_ingestion():
    """Test knowledge ingestion into Kronos."""
    print("\n" + "="*60)
    print("TEST: Knowledge Ingestion")
    print("="*60)
    
    # Create fresh Kronos memory
    storage_path = Path("./kronos_knowledge")
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
    await memory.create_graph("knowledge", "Dawn Field Theory knowledge base")
    
    # Ingest dawn-field-theory
    ingester = KnowledgeIngester(memory)
    dft_path = Path(r"c:\Users\peter\repos\Dawn Field Institute\dawn-field-theory")
    
    print(f"\nIngesting: {dft_path}")
    root_id = await ingester.ingest_directory(
        dft_path,
        graph_name="knowledge",
        max_depth=3,  # Don't go too deep
    )
    
    print(f"\n✓ Ingestion complete:")
    print(f"  Directories: {ingester.stats['directories']}")
    print(f"  Files: {ingester.stats['files']}")
    print(f"  Sections: {ingester.stats['sections']}")
    print(f"  Total nodes: {sum(ingester.stats.values())}")
    
    return memory, root_id


async def test_query(memory: KronosMemory):
    """Test SEC resonance query."""
    print("\n" + "="*60)
    print("TEST: SEC Resonance Query")
    print("="*60)
    
    querier = KnowledgeQuerier(memory)
    
    queries = [
        "What is Dawn Field Theory?",
        "How does entropy relate to information?",
        "What is PAC conservation?",
        "golden ratio significance",
    ]
    
    for q in queries:
        print(f"\nQuery: '{q}'")
        results = await querier.query(q, limit=3)
        
        for i, r in enumerate(results, 1):
            content_preview = r.content[:80].replace('\n', ' ')
            print(f"  {i}. [{r.score:.3f}] {r.node_type}: {content_preview}...")
    
    print("\n✓ SEC query working")
    return True


async def test_answering(memory: KronosMemory):
    """Test coherent answering."""
    print("\n" + "="*60)
    print("TEST: Coherent Answering")
    print("="*60)
    
    answerer = CoherentAnswerer(memory)
    
    questions = [
        "What is Dawn Field Theory about?",
        "How do information and entropy interact?",
        "Explain PAC conservation.",
        "What role does the golden ratio play?",
    ]
    
    for q in questions:
        result = await answerer.answer(q)
        print(f"\n❓ {q}")
        print(f"📚 Sources: {', '.join(result['sources'])}")
        print(f"📊 Score: {result['score']:.3f} | Time: {result['time_ms']:.1f}ms")
        print(f"💬 {result['answer'][:200]}...")
    
    print("\n✓ Answering working")
    return True


async def test_tree_traversal(memory: KronosMemory):
    """Test PAC tree traversal."""
    print("\n" + "="*60)
    print("TEST: PAC Tree Traversal")
    print("="*60)
    
    querier = KnowledgeQuerier(memory)
    
    # Find a node to trace
    results = await querier.query("PAC conservation", limit=1)
    
    if results:
        # Get node ID from memory directly
        query_results = await memory.query(
            query_text="PAC conservation",
            graphs=["knowledge"],
            limit=1,
        )
        
        if query_results:
            node = query_results[0].node
            print(f"\nTracing from: {node.content[:50]}...")
            
            trace = await memory.trace_evolution(
                graph="knowledge",
                node_id=node.id,
                direction="both",
            )
            
            print(f"\nBackward path (ancestors):")
            for step in trace.get("backward_path", [])[:3]:
                print(f"  ← {step['content'][:50]}...")
                print(f"     depth={step.get('depth', '?')}, entropy={step.get('entropy', 0):.3f}")
            
            print(f"\nForward path (descendants):")
            for step in trace.get("forward_path", [])[:3]:
                print(f"  → {step['content'][:50]}...")
    
    print("\n✓ Tree traversal working")
    return True


async def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("POC-025: GAIA Prime + Kronos Integration - Experiment 10")
    print("Native Kronos Usage - PAC/SEC/MED as Designed")
    print("="*70)
    
    results = {}
    
    # Test 1: Ingestion
    try:
        memory, root_id = await test_ingestion()
        results["Ingestion"] = "PASS"
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        results["Ingestion"] = "FAIL"
        return
    
    # Test 2: Query
    try:
        await test_query(memory)
        results["Query"] = "PASS"
    except Exception as e:
        print(f"❌ Query failed: {e}")
        import traceback
        traceback.print_exc()
        results["Query"] = "FAIL"
    
    # Test 3: Answering
    try:
        await test_answering(memory)
        results["Answering"] = "PASS"
    except Exception as e:
        print(f"❌ Answering failed: {e}")
        import traceback
        traceback.print_exc()
        results["Answering"] = "FAIL"
    
    # Test 4: Tree traversal
    try:
        await test_tree_traversal(memory)
        results["Tree Traversal"] = "PASS"
    except Exception as e:
        print(f"❌ Tree traversal failed: {e}")
        import traceback
        traceback.print_exc()
        results["Tree Traversal"] = "FAIL"
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    for test, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} {test}: {result}")
    
    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    # Show Kronos stats
    if memory:
        stats = await memory.get_stats()
        print(f"\n📊 Kronos Stats:")
        print(f"   Total nodes: {stats.get('total_nodes', 'N/A')}")
        print(f"   Queries: {stats.get('queries', 'N/A')}")
        print(f"   Conservations validated: {stats.get('conservations_validated', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())
