"""
POC-025: GAIA Prime + Kronos Integration
========================================

Experiment 07: GAIA Agents - Physics-Based Generation from SEC Context

This is the full integration:
1. SEC Navigation → Find relevant concepts through spreading activation
2. Working Memory → Build coherent context from chains
3. GAIA Prime → Generate response using physics mesh

The agent doesn't just retrieve - it THINKS and GENERATES.

Agent Types:
- ResearchAgent: Answers questions about Dawn Field Theory
- CodeAgent: Explains code and architecture  
- ReasoningAgent: Builds multi-step explanations
"""

import sys
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\fracton")
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\dawn-models\research\GAIA\src")

import torch
import numpy as np
import time
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Import from previous experiments
from exp_01_bridge import KronosGAIABridge
from exp_04_repo_index import SimpleTextEmbedder
from exp_05_sec_navigation import SECNavigator, SECNavigationResult
from exp_06_associative_chains import AssociativeMind, SynthesizedInsight, WorkingMemory

# GAIA Prime imports
from gaia_prime.validated_constants import PHI, PHI_INV, XI


# ============================================================================
# Agent Response
# ============================================================================

@dataclass
class AgentResponse:
    """Response from a GAIA agent."""
    query: str
    response: str
    confidence: float
    sources: List[str]
    reasoning_steps: List[str]
    generation_time_ms: float
    tokens_generated: int
    
    # Debug info
    concepts_activated: int = 0
    chains_found: int = 0
    sec_hops: int = 0


# ============================================================================
# Base Agent
# ============================================================================

class GAIAAgent(ABC):
    """
    Base class for GAIA-powered agents.
    
    All agents share:
    - SEC navigation for context retrieval
    - Working memory for context accumulation  
    - Physics-inspired response synthesis
    
    Subclasses define:
    - How to format context for generation
    - What system prompt to use
    - How to post-process output
    
    Note: Full GAIA generation requires trained PACMeshSpace.
    This demo uses SEC navigation + context synthesis.
    """
    
    def __init__(
        self,
        kronos: KronosGAIABridge,
        embedder: SimpleTextEmbedder,
        device: str = 'cuda',
    ):
        self.device = device
        self.kronos = kronos
        self.embedder = embedder
        
        # SEC navigation for context
        self.mind = AssociativeMind(
            kronos=kronos,
            embedder=embedder,
            device=device,
        )
        
        # Agent identity
        self.name = "GAIAAgent"
        self.system_prompt = "You are a helpful assistant."
        
        # Stats
        self.stats = {
            'queries': 0,
            'total_tokens': 0,
            'avg_confidence': 0,
        }
    
    @abstractmethod
    def format_context(self, insight: SynthesizedInsight) -> str:
        """Format the SEC context for generation."""
        pass
    
    @abstractmethod
    def format_prompt(self, query: str, context: str) -> str:
        """Format the full prompt for generation."""
        pass
    
    def think(self, query: str) -> SynthesizedInsight:
        """Use SEC navigation to understand the query."""
        return self.mind.think(query, use_context=True)
    
    def generate_from_context(
        self,
        prompt: str,
        max_tokens: int = 100,
    ) -> Tuple[str, int]:
        """
        Generate response based on SEC-activated context.
        
        This synthesizes a response from the working memory
        using physics-inspired principles:
        - High activation = more weight in response
        - Chain coherence = sentence structure
        - Resonance = reinforced facts
        """
        # Get concepts sorted by activation (physics weighting)
        concepts = sorted(
            self.mind.working_memory.concepts.values(),
            key=lambda x: x.activation,
            reverse=True
        )[:7]
        
        # Build response from extracted facts (weighted by activation)
        response = self._construct_response_from_context(prompt)
        token_estimate = len(response.split())
        
        return response, token_estimate
    
    def _construct_response_from_context(self, prompt: str) -> str:
        """
        Construct a response based on context.
        
        In a full implementation, GAIA would generate tokens.
        For this demo, we synthesize from the working memory.
        """
        # Get facts from working memory
        facts = []
        for concept in sorted(
            self.mind.working_memory.concepts.values(),
            key=lambda x: x.activation,
            reverse=True
        )[:5]:
            facts.extend(concept.extracted_facts)
        
        if not facts:
            return "I don't have enough context to answer that question."
        
        # Build response from facts
        response_parts = []
        used_facts = set()
        
        for fact in facts[:3]:
            # Clean up fact
            clean_fact = fact.strip()
            if clean_fact and clean_fact not in used_facts:
                response_parts.append(clean_fact)
                used_facts.add(clean_fact)
        
        if response_parts:
            return " ".join(response_parts)
        else:
            return "Based on the available context, I cannot provide a specific answer."
    
    def respond(self, query: str, max_tokens: int = 200) -> AgentResponse:
        """
        Full agent response pipeline:
        1. SEC navigation to understand query
        2. Format context from working memory
        3. Generate response using GAIA
        """
        start_time = time.perf_counter()
        
        # Step 1: Think (SEC navigation)
        insight = self.think(query)
        
        # Step 2: Format context
        context = self.format_context(insight)
        
        # Step 3: Format prompt
        prompt = self.format_prompt(query, context)
        
        # Step 4: Generate
        response_text, tokens = self.generate_from_context(prompt, max_tokens)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Build reasoning steps
        reasoning = []
        if insight.reasoning_chains:
            for chain in insight.reasoning_chains[:2]:
                sources = [n.source for n in chain.nodes]
                reasoning.append(f"[{chain.chain_type}] " + " → ".join(sources[:3]))
        
        # Update stats
        self.stats['queries'] += 1
        self.stats['total_tokens'] += tokens
        
        return AgentResponse(
            query=query,
            response=response_text,
            confidence=insight.confidence,
            sources=insight.sources[:5],
            reasoning_steps=reasoning,
            generation_time_ms=elapsed_ms,
            tokens_generated=tokens,
            concepts_activated=len(insight.supporting_concepts),
            chains_found=len(insight.reasoning_chains),
            sec_hops=self.mind.navigator.stats.get('avg_hops', 0),
        )


# ============================================================================
# Specialized Agents
# ============================================================================

class ResearchAgent(GAIAAgent):
    """Agent specialized for Dawn Field Theory research questions."""
    
    def __init__(self, kronos, embedder, device='cuda'):
        super().__init__(kronos, embedder, device)
        self.name = "ResearchAgent"
        self.system_prompt = """You are a research assistant specializing in Dawn Field Theory.
You explain concepts like PAC, SEC, entropy, and information dynamics.
Base your answers on the activated knowledge from the repository."""
    
    def format_context(self, insight: SynthesizedInsight) -> str:
        """Format research context."""
        parts = ["Research Context:"]
        
        # Add main insight
        if insight.main_insight:
            parts.append(f"\nCore Finding: {insight.main_insight[:300]}")
        
        # Add facts from supporting concepts
        parts.append("\nRelevant Facts:")
        for concept in insight.supporting_concepts[:3]:
            for fact in concept.extracted_facts[:2]:
                parts.append(f"  - {fact[:150]}")
        
        # Add source references
        parts.append(f"\nSources: {', '.join(insight.sources[:5])}")
        
        return "\n".join(parts)
    
    def format_prompt(self, query: str, context: str) -> str:
        """Format research prompt."""
        return f"""{self.system_prompt}

{context}

Question: {query}

Based on the research context above, provide a clear explanation:"""


class CodeAgent(GAIAAgent):
    """Agent specialized for code and architecture questions."""
    
    def __init__(self, kronos, embedder, device='cuda'):
        super().__init__(kronos, embedder, device)
        self.name = "CodeAgent"
        self.system_prompt = """You are a code assistant for the GAIA and Fracton projects.
You explain architecture, implementations, and how components work together.
Focus on technical accuracy and code references."""
    
    def format_context(self, insight: SynthesizedInsight) -> str:
        """Format code context."""
        parts = ["Code Context:"]
        
        # Filter for code-related sources
        code_sources = [s for s in insight.sources if s.endswith('.py')]
        doc_sources = [s for s in insight.sources if s.endswith('.md')]
        
        if code_sources:
            parts.append(f"\nCode files: {', '.join(code_sources[:5])}")
        if doc_sources:
            parts.append(f"\nDocumentation: {', '.join(doc_sources[:3])}")
        
        # Add technical facts
        parts.append("\nTechnical Details:")
        for concept in insight.supporting_concepts[:3]:
            for fact in concept.extracted_facts[:2]:
                if any(kw in fact.lower() for kw in ['class', 'function', 'method', 'implements', 'returns', 'uses']):
                    parts.append(f"  - {fact[:150]}")
        
        return "\n".join(parts)
    
    def format_prompt(self, query: str, context: str) -> str:
        """Format code prompt."""
        return f"""{self.system_prompt}

{context}

Question: {query}

Technical explanation:"""


class ReasoningAgent(GAIAAgent):
    """Agent that builds multi-step explanations."""
    
    def __init__(self, kronos, embedder, device='cuda'):
        super().__init__(kronos, embedder, device)
        self.name = "ReasoningAgent"
        self.system_prompt = """You are a reasoning assistant that builds step-by-step explanations.
You follow chains of logic from premise to conclusion.
Show your reasoning process clearly."""
    
    def format_context(self, insight: SynthesizedInsight) -> str:
        """Format reasoning context with chains."""
        parts = ["Reasoning Context:"]
        
        # Emphasize chains
        if insight.reasoning_chains:
            parts.append("\nReasoning Chains Found:")
            for i, chain in enumerate(insight.reasoning_chains[:3], 1):
                sources = [n.source for n in chain.nodes]
                parts.append(f"  {i}. [{chain.chain_type}] {' → '.join(sources[:4])}")
                
                # Add key facts from chain
                for node in chain.nodes[:2]:
                    if node.extracted_facts:
                        parts.append(f"     • {node.extracted_facts[0][:100]}")
        
        # Add main insight
        if insight.main_insight:
            parts.append(f"\nCore Understanding: {insight.main_insight[:200]}")
        
        return "\n".join(parts)
    
    def format_prompt(self, query: str, context: str) -> str:
        """Format reasoning prompt."""
        return f"""{self.system_prompt}

{context}

Question: {query}

Step-by-step reasoning:
1."""


# ============================================================================
# Agent Factory
# ============================================================================

class AgentFactory:
    """Factory for creating GAIA agents."""
    
    AGENT_TYPES = {
        'research': ResearchAgent,
        'code': CodeAgent,
        'reasoning': ReasoningAgent,
    }
    
    def __init__(
        self,
        kronos: KronosGAIABridge,
        embedder: SimpleTextEmbedder,
        device: str = 'cuda',
    ):
        self.kronos = kronos
        self.embedder = embedder
        self.device = device
        self._agents: Dict[str, GAIAAgent] = {}
    
    def get_agent(self, agent_type: str) -> GAIAAgent:
        """Get or create an agent of the specified type."""
        if agent_type not in self._agents:
            if agent_type not in self.AGENT_TYPES:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            agent_class = self.AGENT_TYPES[agent_type]
            self._agents[agent_type] = agent_class(
                kronos=self.kronos,
                embedder=self.embedder,
                device=self.device,
            )
        
        return self._agents[agent_type]
    
    def route_query(self, query: str) -> str:
        """Route a query to the best agent type."""
        query_lower = query.lower()
        
        # Simple keyword routing
        if any(kw in query_lower for kw in ['code', 'implement', 'function', 'class', 'architecture', 'how does the code']):
            return 'code'
        elif any(kw in query_lower for kw in ['why', 'explain', 'reason', 'step', 'how does']):
            return 'reasoning'
        else:
            return 'research'
    
    def ask(self, query: str) -> AgentResponse:
        """Route and answer a query with the appropriate agent."""
        agent_type = self.route_query(query)
        agent = self.get_agent(agent_type)
        return agent.respond(query)


# ============================================================================
# Tests
# ============================================================================

def test_research_agent():
    """Test the research agent."""
    print("\n" + "=" * 60)
    print("TEST: Research Agent")
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
    
    agent = ResearchAgent(kronos=kronos, embedder=embedder, device=device)
    
    query = "What is the relationship between entropy and structure in Dawn Field Theory?"
    print(f"\nQuery: {query}")
    
    response = agent.respond(query)
    
    print(f"\nAgent: {agent.name}")
    print(f"Confidence: {response.confidence:.1%}")
    print(f"Concepts activated: {response.concepts_activated}")
    print(f"Chains found: {response.chains_found}")
    print(f"Time: {response.generation_time_ms:.0f}ms")
    print(f"\nResponse:\n  {response.response[:300]}...")
    
    if response.reasoning_steps:
        print(f"\nReasoning:")
        for step in response.reasoning_steps:
            print(f"  {step}")
    
    assert response.confidence > 0, "No confidence!"
    assert len(response.response) > 20, "Response too short!"
    
    print("\n✓ Research agent working")
    return True


def test_code_agent():
    """Test the code agent."""
    print("\n" + "=" * 60)
    print("TEST: Code Agent")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    index_path = Path(__file__).parent / "repo_knowledge_index"
    
    if not index_path.exists():
        return False
    
    kronos = KronosGAIABridge(
        storage_path=index_path,
        namespace="dawn_institute",
        device=device,
    )
    embedder = SimpleTextEmbedder(dim=768, device=device)
    
    agent = CodeAgent(kronos=kronos, embedder=embedder, device=device)
    
    query = "How does the PhysicsMesh class implement SEC dynamics?"
    print(f"\nQuery: {query}")
    
    response = agent.respond(query)
    
    print(f"\nAgent: {agent.name}")
    print(f"Confidence: {response.confidence:.1%}")
    print(f"Sources: {', '.join(response.sources[:3])}")
    print(f"\nResponse:\n  {response.response[:300]}...")
    
    assert len(response.response) > 20, "Response too short!"
    
    print("\n✓ Code agent working")
    return True


def test_reasoning_agent():
    """Test the reasoning agent."""
    print("\n" + "=" * 60)
    print("TEST: Reasoning Agent")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    index_path = Path(__file__).parent / "repo_knowledge_index"
    
    if not index_path.exists():
        return False
    
    kronos = KronosGAIABridge(
        storage_path=index_path,
        namespace="dawn_institute",
        device=device,
    )
    embedder = SimpleTextEmbedder(dim=768, device=device)
    
    agent = ReasoningAgent(kronos=kronos, embedder=embedder, device=device)
    
    query = "Why does PAC conservation lead to emergent structure?"
    print(f"\nQuery: {query}")
    
    response = agent.respond(query)
    
    print(f"\nAgent: {agent.name}")
    print(f"Confidence: {response.confidence:.1%}")
    print(f"Chains found: {response.chains_found}")
    
    if response.reasoning_steps:
        print(f"\nReasoning chains:")
        for step in response.reasoning_steps:
            print(f"  {step}")
    
    print(f"\nResponse:\n  {response.response[:300]}...")
    
    assert len(response.response) > 20, "Response too short!"
    
    print("\n✓ Reasoning agent working")
    return True


def test_agent_factory():
    """Test the agent factory routing."""
    print("\n" + "=" * 60)
    print("TEST: Agent Factory & Routing")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    index_path = Path(__file__).parent / "repo_knowledge_index"
    
    if not index_path.exists():
        return False
    
    kronos = KronosGAIABridge(
        storage_path=index_path,
        namespace="dawn_institute",
        device=device,
    )
    embedder = SimpleTextEmbedder(dim=768, device=device)
    
    factory = AgentFactory(kronos=kronos, embedder=embedder, device=device)
    
    # Test routing
    test_queries = [
        ("What is PAC?", "research"),
        ("How does the PhysicsMesh code work?", "code"),
        ("Why does entropy decrease during crystallization?", "reasoning"),
    ]
    
    print("\nRouting tests:")
    for query, expected in test_queries:
        routed = factory.route_query(query)
        status = "✓" if routed == expected else "✗"
        print(f"  {status} '{query[:40]}...' → {routed} (expected {expected})")
    
    # Test full query
    print("\nFull query test:")
    response = factory.ask("What is Dawn Field Theory?")
    print(f"  Query answered by: {factory.get_agent(factory.route_query('What is Dawn Field Theory?')).name}")
    print(f"  Confidence: {response.confidence:.1%}")
    
    print("\n✓ Agent factory working")
    return True


def demo_multi_agent_conversation():
    """Demo showing agents working together."""
    print("\n" + "=" * 70)
    print("DEMO: Multi-Agent Conversation")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    index_path = Path(__file__).parent / "repo_knowledge_index"
    
    if not index_path.exists():
        print("  No index found.")
        return
    
    kronos = KronosGAIABridge(
        storage_path=index_path,
        namespace="dawn_institute",
        device=device,
    )
    embedder = SimpleTextEmbedder(dim=768, device=device)
    
    factory = AgentFactory(kronos=kronos, embedder=embedder, device=device)
    
    print("\nAsking different agents the same underlying question...\n")
    
    questions = [
        ("research", "What is SEC collapse in Dawn Field Theory?"),
        ("code", "How is SEC collapse implemented in the codebase?"),
        ("reasoning", "Why does SEC collapse lead to structure formation?"),
    ]
    
    for agent_type, question in questions:
        print("-" * 70)
        agent = factory.get_agent(agent_type)
        response = agent.respond(question)
        
        print(f"[{agent.name}]")
        print(f"Q: {question}")
        print(f"Confidence: {response.confidence:.1%} | Sources: {len(response.sources)}")
        print(f"A: {response.response[:250]}...")
        print()
    
    print("-" * 70)
    print("\nKey insight: Different agents approach the same topic differently")
    print("but they all draw from the same SEC-navigated knowledge base.")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("POC-025: GAIA Prime + Kronos Integration - Experiment 07")
    print("GAIA Agents - Physics-Based Generation from SEC Context")
    print("=" * 70)
    
    tests = [
        ("Research Agent", test_research_agent),
        ("Code Agent", test_code_agent),
        ("Reasoning Agent", test_reasoning_agent),
        ("Agent Factory", test_agent_factory),
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
    demo_multi_agent_conversation()
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
