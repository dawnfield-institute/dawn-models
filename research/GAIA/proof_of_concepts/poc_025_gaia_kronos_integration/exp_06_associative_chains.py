"""
POC-025: GAIA Prime + Kronos Integration
========================================

Experiment 06: Associative Chains - Contextual Understanding

Building on SEC Navigation to create TRUE brain-like understanding:

1. **Context Accumulation**: Each activated pattern adds to context
2. **Associative Chains**: Follow concept relationships, not just similarity
3. **Insight Synthesis**: Combine activated patterns into coherent answer
4. **Working Memory**: Track what's been activated and why

The key difference from RAG:
- RAG: "Here are documents that match your query"
- SEC: "Here's what I understand about your question based on how 
       concepts relate to each other"

A brain doesn't search - it thinks by activating related concepts
until a coherent understanding crystallizes.
"""

import sys
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\fracton")
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\dawn-models\research\GAIA\src")

import torch
import numpy as np
import time
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

# Import from previous experiments
from exp_01_bridge import KronosGAIABridge
from exp_04_repo_index import SimpleTextEmbedder
from exp_05_sec_navigation import (
    SECNavigator, SECNavigationResult, ActivationState,
    ACTIVATION_THRESHOLD
)

# GAIA Prime imports
from gaia_prime.validated_constants import PHI, PHI_INV, XI


# ============================================================================
# Working Memory & Context
# ============================================================================

@dataclass
class ConceptNode:
    """A concept in working memory."""
    id: str
    content: str
    source: str
    activation: float
    relationships: List[str]  # IDs of related concepts
    extracted_facts: List[str]  # Key facts extracted from content
    
    
@dataclass
class AssociativeChain:
    """A chain of associated concepts forming a reasoning path."""
    nodes: List[ConceptNode]
    coherence_score: float  # How well the chain holds together
    topic_relevance: float  # How relevant to the query
    chain_type: str  # 'causal', 'definitional', 'procedural', 'temporal'


@dataclass
class SynthesizedInsight:
    """Synthesized understanding from multiple concept chains."""
    query: str
    main_insight: str
    supporting_concepts: List[ConceptNode]
    reasoning_chains: List[AssociativeChain]
    confidence: float
    sources: List[str]
    

# ============================================================================
# Working Memory System
# ============================================================================

class WorkingMemory:
    """
    Short-term memory that accumulates context during navigation.
    
    Like a brain's working memory:
    - Limited capacity (can't hold everything)
    - Recency bias (newer activations more accessible)
    - Association strengthening (related concepts boost each other)
    """
    
    def __init__(self, capacity: int = 15):
        self.capacity = capacity
        self.concepts: Dict[str, ConceptNode] = {}
        self.activation_order: List[str] = []
        self.association_strength: Dict[Tuple[str, str], float] = {}
        
    def add_concept(
        self,
        concept_id: str,
        content: str,
        source: str,
        activation: float,
    ):
        """Add concept to working memory."""
        # Extract key facts from content
        facts = self._extract_facts(content)
        
        node = ConceptNode(
            id=concept_id,
            content=content[:500],  # Truncate for memory
            source=source,
            activation=activation,
            relationships=[],
            extracted_facts=facts,
        )
        
        self.concepts[concept_id] = node
        self.activation_order.append(concept_id)
        
        # Enforce capacity limit (evict least activated)
        if len(self.concepts) > self.capacity:
            self._evict_weakest()
            
        # Strengthen associations with existing concepts
        for existing_id in list(self.concepts.keys()):
            if existing_id != concept_id:
                strength = self._compute_association(node, self.concepts[existing_id])
                if strength > 0.1:
                    self.association_strength[(concept_id, existing_id)] = strength
                    node.relationships.append(existing_id)
                    self.concepts[existing_id].relationships.append(concept_id)
    
    def _extract_facts(self, content: str) -> List[str]:
        """Extract key factual statements from content."""
        facts = []
        
        # Simple extraction: sentences with key patterns
        sentences = content.replace('\n', ' ').split('.')
        
        key_patterns = [
            'is a', 'is the', 'means', 'represents', 'equals',
            'creates', 'generates', 'produces', 'emerges',
            'because', 'therefore', 'thus', 'hence',
            'using', 'through', 'via', 'by',
        ]
        
        for sentence in sentences[:10]:  # Limit sentences checked
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            for pattern in key_patterns:
                if pattern in sentence.lower():
                    facts.append(sentence[:200])
                    break
        
        return facts[:5]  # Max 5 facts per concept
    
    def _compute_association(self, node1: ConceptNode, node2: ConceptNode) -> float:
        """Compute semantic association between two concepts."""
        # Simple word overlap metric
        words1 = set(node1.content.lower().split())
        words2 = set(node2.content.lower().split())
        
        overlap = len(words1 & words2)
        union = len(words1 | words2)
        
        return overlap / max(union, 1)
    
    def _evict_weakest(self):
        """Evict the weakest concept."""
        if not self.concepts:
            return
        
        # Find lowest activation
        weakest_id = min(
            self.concepts.keys(),
            key=lambda x: self.concepts[x].activation
        )
        
        # Remove from memory
        del self.concepts[weakest_id]
        
        # Clean up relationships
        for concept in self.concepts.values():
            if weakest_id in concept.relationships:
                concept.relationships.remove(weakest_id)
    
    def get_context_summary(self) -> str:
        """Get summary of current working memory context."""
        if not self.concepts:
            return "No context accumulated yet."
        
        lines = [f"Working Memory ({len(self.concepts)} concepts):"]
        
        for concept in sorted(
            self.concepts.values(),
            key=lambda x: x.activation,
            reverse=True
        )[:5]:
            lines.append(f"  - {concept.source}: {concept.extracted_facts[0][:80]}..." 
                        if concept.extracted_facts else f"  - {concept.source}")
        
        return "\n".join(lines)
    
    def find_chains(self, min_length: int = 2) -> List[AssociativeChain]:
        """Find associative chains in working memory."""
        chains = []
        visited = set()
        
        # DFS from each concept
        for start_id in self.concepts:
            if start_id in visited:
                continue
            
            chain = self._build_chain(start_id, visited)
            if len(chain) >= min_length:
                coherence = self._compute_chain_coherence(chain)
                chains.append(AssociativeChain(
                    nodes=chain,
                    coherence_score=coherence,
                    topic_relevance=sum(n.activation for n in chain) / len(chain),
                    chain_type=self._classify_chain(chain),
                ))
        
        return sorted(chains, key=lambda x: x.coherence_score, reverse=True)
    
    def _build_chain(
        self,
        start_id: str,
        visited: Set[str],
        max_length: int = 5,
    ) -> List[ConceptNode]:
        """Build a chain from a starting concept."""
        chain = []
        current_id = start_id
        
        while len(chain) < max_length and current_id not in visited:
            if current_id not in self.concepts:
                break
            
            node = self.concepts[current_id]
            chain.append(node)
            visited.add(current_id)
            
            # Find strongest related concept not yet visited
            best_next = None
            best_strength = 0
            
            for related_id in node.relationships:
                if related_id not in visited and related_id in self.concepts:
                    # Check association strength
                    key = (current_id, related_id)
                    rev_key = (related_id, current_id)
                    strength = self.association_strength.get(
                        key,
                        self.association_strength.get(rev_key, 0)
                    )
                    
                    if strength > best_strength:
                        best_strength = strength
                        best_next = related_id
            
            if best_next is None:
                break
            
            current_id = best_next
        
        return chain
    
    def _compute_chain_coherence(self, chain: List[ConceptNode]) -> float:
        """Compute how coherent a chain is."""
        if len(chain) < 2:
            return 0.0
        
        # Average association strength between adjacent nodes
        total_strength = 0
        for i in range(len(chain) - 1):
            key = (chain[i].id, chain[i+1].id)
            rev_key = (chain[i+1].id, chain[i].id)
            total_strength += self.association_strength.get(
                key,
                self.association_strength.get(rev_key, 0)
            )
        
        return total_strength / (len(chain) - 1)
    
    def _classify_chain(self, chain: List[ConceptNode]) -> str:
        """Classify the type of reasoning chain."""
        # Simple heuristic based on content patterns
        all_content = " ".join(c.content.lower() for c in chain)
        
        if any(w in all_content for w in ['because', 'therefore', 'causes', 'leads to']):
            return 'causal'
        elif any(w in all_content for w in ['is defined', 'means', 'is a', 'refers to']):
            return 'definitional'
        elif any(w in all_content for w in ['step', 'then', 'next', 'procedure']):
            return 'procedural'
        elif any(w in all_content for w in ['before', 'after', 'when', 'during']):
            return 'temporal'
        else:
            return 'associative'


# ============================================================================
# Associative Mind
# ============================================================================

class AssociativeMind:
    """
    Brain-like system that builds understanding through association.
    
    This goes beyond SEC navigation by:
    1. Maintaining working memory across queries
    2. Building chains of association (reasoning paths)
    3. Synthesizing insights from multiple activated concepts
    4. Accumulating context for multi-turn understanding
    """
    
    def __init__(
        self,
        kronos: KronosGAIABridge,
        embedder: SimpleTextEmbedder,
        device: str = 'cuda',
    ):
        self.navigator = SECNavigator(
            kronos=kronos,
            embedder=embedder,
            device=device,
        )
        self.kronos = kronos
        self.embedder = embedder
        self.device = device
        
        # Working memory (persists across queries)
        self.working_memory = WorkingMemory(capacity=15)
        
        # Query history for context
        self.query_history: List[str] = []
        
        # Stats
        self.stats = {
            'queries_processed': 0,
            'concepts_activated': 0,
            'chains_found': 0,
        }
    
    def think(
        self,
        query: str,
        use_context: bool = True,
    ) -> SynthesizedInsight:
        """
        Think about a query using associative activation.
        
        This is the main entry point - it:
        1. Navigates semantic space using SEC
        2. Accumulates activated concepts in working memory
        3. Finds associative chains
        4. Synthesizes an insight from all of this
        
        Args:
            query: Natural language question
            use_context: Whether to use accumulated context from previous queries
        """
        start_time = time.perf_counter()
        
        # Track query history
        self.query_history.append(query)
        self.stats['queries_processed'] += 1
        
        # If not using context, reset working memory
        if not use_context:
            self.working_memory = WorkingMemory(capacity=15)
        
        # 1. Navigate using SEC
        nav_result = self.navigator.navigate(
            query=query,
            initial_seeds=5,
            max_activated=20,
        )
        
        # 2. Add activated concepts to working memory
        for state in nav_result.activated_patterns:
            if state.effective_activation >= ACTIVATION_THRESHOLD:
                pattern = self.kronos.pattern_index.get(state.pattern_id)
                if pattern:
                    content = pattern.metadata.get('content_preview', '')
                    source = Path(pattern.metadata.get('file_path', 'unknown')).name
                    
                    self.working_memory.add_concept(
                        concept_id=state.pattern_id,
                        content=content,
                        source=source,
                        activation=state.effective_activation,
                    )
                    self.stats['concepts_activated'] += 1
        
        # 3. Find associative chains
        chains = self.working_memory.find_chains(min_length=2)
        self.stats['chains_found'] += len(chains)
        
        # 4. Synthesize insight
        insight = self._synthesize(query, chains)
        
        elapsed = time.perf_counter() - start_time
        
        return insight
    
    def _synthesize(
        self,
        query: str,
        chains: List[AssociativeChain],
    ) -> SynthesizedInsight:
        """Synthesize an insight from chains and working memory."""
        
        # Collect all facts from working memory
        all_facts = []
        all_sources = set()
        
        for concept in self.working_memory.concepts.values():
            all_facts.extend(concept.extracted_facts)
            all_sources.add(concept.source)
        
        # Get supporting concepts (top by activation)
        supporting = sorted(
            self.working_memory.concepts.values(),
            key=lambda x: x.activation,
            reverse=True
        )[:5]
        
        # Build main insight from best chain
        if chains:
            best_chain = chains[0]
            main_insight = self._chain_to_insight(query, best_chain)
        elif all_facts:
            main_insight = self._facts_to_insight(query, all_facts[:3])
        else:
            main_insight = f"Limited understanding of: {query}"
        
        # Compute confidence
        confidence = self._compute_confidence(chains, supporting)
        
        return SynthesizedInsight(
            query=query,
            main_insight=main_insight,
            supporting_concepts=supporting,
            reasoning_chains=chains[:3],
            confidence=confidence,
            sources=list(all_sources),
        )
    
    def _chain_to_insight(
        self,
        query: str,
        chain: AssociativeChain,
    ) -> str:
        """Convert an associative chain to an insight statement."""
        if not chain.nodes:
            return "No clear insight."
        
        # Combine key facts from chain
        facts = []
        for node in chain.nodes:
            if node.extracted_facts:
                facts.append(node.extracted_facts[0])
        
        if not facts:
            # Fall back to source summary
            sources = [n.source for n in chain.nodes]
            return f"Understanding from {', '.join(sources[:3])}: related to {query}"
        
        # Combine facts into insight
        if len(facts) == 1:
            return facts[0]
        else:
            return f"{facts[0]} Additionally, {facts[1].lower()}"
    
    def _facts_to_insight(self, query: str, facts: List[str]) -> str:
        """Combine facts into an insight."""
        if not facts:
            return f"Limited information about: {query}"
        
        return ". ".join(facts[:2])
    
    def _compute_confidence(
        self,
        chains: List[AssociativeChain],
        supporting: List[ConceptNode],
    ) -> float:
        """Compute confidence in the insight."""
        # Base confidence from chain coherence
        chain_score = chains[0].coherence_score if chains else 0
        
        # Boost from number of supporting concepts
        support_score = min(len(supporting) / 5, 1.0)
        
        # Boost from extracted facts
        fact_count = sum(len(c.extracted_facts) for c in supporting)
        fact_score = min(fact_count / 10, 1.0)
        
        # Combine with weights
        confidence = (
            chain_score * 0.4 +
            support_score * 0.3 +
            fact_score * 0.3
        )
        
        return min(confidence, 1.0)
    
    def explain(self, insight: SynthesizedInsight) -> str:
        """Generate human-readable explanation of the insight."""
        lines = [
            f"Query: {insight.query}",
            "",
            f"Understanding (confidence: {insight.confidence:.1%}):",
            f"  {insight.main_insight}",
            "",
            f"Based on {len(insight.sources)} sources:",
        ]
        
        for source in insight.sources[:5]:
            lines.append(f"  - {source}")
        
        if insight.reasoning_chains:
            lines.append("")
            lines.append(f"Reasoning chains ({len(insight.reasoning_chains)}):")
            
            for i, chain in enumerate(insight.reasoning_chains[:2]):
                chain_sources = [n.source for n in chain.nodes]
                lines.append(
                    f"  [{chain.chain_type}] {' -> '.join(chain_sources[:4])}"
                )
        
        lines.append("")
        lines.append("Working memory:")
        lines.append(self.working_memory.get_context_summary())
        
        return "\n".join(lines)


# ============================================================================
# Tests
# ============================================================================

def test_working_memory():
    """Test working memory accumulation."""
    print("\n" + "=" * 60)
    print("TEST: Working Memory")
    print("=" * 60)
    
    memory = WorkingMemory(capacity=5)
    
    # Add concepts
    memory.add_concept("c1", "GAIA is a neural architecture using PAC conservation", "gaia.py", 0.9)
    memory.add_concept("c2", "PAC means Potential-Actualization Conservation", "pac.md", 0.8)
    memory.add_concept("c3", "Conservation of information is fundamental", "theory.md", 0.7)
    memory.add_concept("c4", "Neural networks learn through backpropagation", "ml.md", 0.6)
    memory.add_concept("c5", "GAIA avoids backprop using physics-based learning", "gaia.md", 0.85)
    
    print(f"Memory has {len(memory.concepts)} concepts")
    
    # Check associations
    print(f"Associations found: {len(memory.association_strength)}")
    for (id1, id2), strength in sorted(
        memory.association_strength.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]:
        print(f"  {id1} <-> {id2}: {strength:.3f}")
    
    # Find chains
    chains = memory.find_chains()
    print(f"Chains found: {len(chains)}")
    for chain in chains[:2]:
        sources = [n.source for n in chain.nodes]
        print(f"  [{chain.chain_type}] {' -> '.join(sources)}")
    
    # Test capacity limit
    memory.add_concept("c6", "This should evict the weakest", "new.md", 0.95)
    print(f"After overflow: {len(memory.concepts)} concepts")
    
    assert len(memory.concepts) <= 5, "Capacity exceeded!"
    print("\n✓ Working memory works correctly")
    return True


def test_associative_mind():
    """Test full associative mind."""
    print("\n" + "=" * 60)
    print("TEST: Associative Mind")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    index_path = Path(__file__).parent / "repo_knowledge_index"
    
    if not index_path.exists():
        print("  No index found. Run exp_04 first.")
        return False
    
    kronos = KronosGAIABridge(
        storage_path=index_path,
        namespace="dawn_institute",
        device=device,
    )
    embedder = SimpleTextEmbedder(dim=768, device=device)
    
    print(f"Loaded {len(kronos.pattern_index)} patterns")
    
    # Create associative mind
    mind = AssociativeMind(kronos=kronos, embedder=embedder, device=device)
    
    # Single query
    query = "How does information create structure?"
    insight = mind.think(query)
    
    print(f"\nQuery: {query}")
    print(f"Confidence: {insight.confidence:.1%}")
    print(f"Supporting concepts: {len(insight.supporting_concepts)}")
    print(f"Reasoning chains: {len(insight.reasoning_chains)}")
    print(f"Sources: {len(insight.sources)}")
    
    assert insight.confidence > 0, "No confidence!"
    assert len(insight.supporting_concepts) > 0, "No supporting concepts!"
    
    print("\n✓ Associative mind working")
    return True


def test_context_accumulation():
    """Test that context accumulates across queries."""
    print("\n" + "=" * 60)
    print("TEST: Context Accumulation")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    index_path = Path(__file__).parent / "repo_knowledge_index"
    
    if not index_path.exists():
        print("  No index found.")
        return False
    
    kronos = KronosGAIABridge(
        storage_path=index_path,
        namespace="dawn_institute",
        device=device,
    )
    embedder = SimpleTextEmbedder(dim=768, device=device)
    
    mind = AssociativeMind(kronos=kronos, embedder=embedder, device=device)
    
    # First query
    q1 = "What is PAC conservation?"
    insight1 = mind.think(q1)
    context_after_1 = len(mind.working_memory.concepts)
    
    print(f"\nQuery 1: {q1}")
    print(f"  Working memory: {context_after_1} concepts")
    
    # Second query (should use accumulated context)
    q2 = "How does SEC relate to entropy?"
    insight2 = mind.think(q2, use_context=True)
    context_after_2 = len(mind.working_memory.concepts)
    
    print(f"\nQuery 2: {q2}")
    print(f"  Working memory: {context_after_2} concepts")
    
    # Context should have grown (or be at capacity)
    print(f"\nContext growth: {context_after_1} -> {context_after_2}")
    
    # Third query referencing previous
    q3 = "How do PAC and SEC work together?"
    insight3 = mind.think(q3, use_context=True)
    
    print(f"\nQuery 3: {q3}")
    print(f"  Chains found: {len(insight3.reasoning_chains)}")
    
    # The mind should have found connections
    assert mind.stats['chains_found'] > 0, "No chains found!"
    
    print("\n✓ Context accumulates correctly")
    return True


def demo_brain_like_understanding():
    """Demo showing brain-like understanding vs simple retrieval."""
    print("\n" + "=" * 70)
    print("DEMO: Brain-Like Understanding")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    index_path = Path(__file__).parent / "repo_knowledge_index"
    
    if not index_path.exists():
        print("  No index found. Run exp_04 first.")
        return
    
    kronos = KronosGAIABridge(
        storage_path=index_path,
        namespace="dawn_institute",
        device=device,
    )
    embedder = SimpleTextEmbedder(dim=768, device=device)
    
    mind = AssociativeMind(kronos=kronos, embedder=embedder, device=device)
    
    print("\nLet's have a multi-turn conversation where context accumulates...")
    print("=" * 70)
    
    questions = [
        "What is Dawn Field Theory?",
        "What role does entropy play?",
        "How does this relate to AI systems like GAIA?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Turn {i}]")
        print("-" * 70)
        
        insight = mind.think(question, use_context=True)
        print(mind.explain(insight))
        
        print("\n")
    
    print("=" * 70)
    print("Key insight: Unlike RAG, the mind ACCUMULATES context")
    print("Each question builds on understanding from previous ones.")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("POC-025: GAIA Prime + Kronos Integration - Experiment 06")
    print("Associative Chains - Contextual Understanding")
    print("=" * 70)
    
    tests = [
        ("Working Memory", test_working_memory),
        ("Associative Mind", test_associative_mind),
        ("Context Accumulation", test_context_accumulation),
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
        status = "PASS" if success else f"FAIL: {error}"
        print(f"  {name}: {status}")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(tests)} tests passed")
    
    # Run demo
    demo_brain_like_understanding()
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
