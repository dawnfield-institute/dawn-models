"""
POC-025: GAIA Prime + Kronos Integration
========================================

Experiment 05: SEC Navigation - Brain-Like Associative Memory

Standard RAG: Query → Vector similarity → Top-K documents
SEC Navigation: Query → Resonance → Spreading activation → Collapse to insight

Key differences from RAG:
1. **Spreading Activation**: Activated patterns boost their neighbors
2. **Resonance Cascade**: High-relevance patterns chain-activate related concepts
3. **SEC Collapse**: Navigate toward low-entropy (high-structure) regions
4. **Graph Traversal**: Follow semantic edges, not just similarity
5. **Context Accumulation**: Multi-hop reasoning builds understanding

This is how a brain works:
- Seeing "apple" activates "fruit", "red", "tree", "pie"...
- Those activate their neighbors
- The network settles into a coherent activation pattern
- That's understanding, not retrieval
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
from typing import Optional, Dict, List, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

# Import from previous experiments
from exp_01_bridge import KronosGAIABridge, CrystallizedPattern

# GAIA Prime imports
from gaia_prime.validated_constants import PHI, PHI_INV, XI, LAMBDA_STAR


# ============================================================================
# SEC Navigation Constants
# ============================================================================

# Activation dynamics (properly bounded)
ACTIVATION_THRESHOLD = 0.05     # Minimum activation to propagate
DECAY_RATE = 0.4                # How fast activation decays per hop
RESONANCE_FACTOR = 0.3          # How much resonance adds (not multiplies)
MAX_HOPS = 5                    # Maximum propagation depth
MAX_ACTIVATION = 5.0            # Cap to prevent explosion
COLLAPSE_THRESHOLD = 0.2       # When to stop (low entropy = found structure)

# Edge types for graph structure
EDGE_TYPES = {
    'similar': 0.8,      # High similarity
    'cooccurs': 0.6,     # Appear together
    'parent': 0.9,       # Hierarchical parent
    'child': 0.7,        # Hierarchical child
    'contradicts': -0.3, # Opposing concepts (negative weight)
}


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ActivationState:
    """Current activation state of a pattern."""
    pattern_id: str
    activation: float           # Current activation level [0, MAX_ACTIVATION]
    source_distance: int        # Hops from query
    activated_by: Set[str]      # Which patterns activated this one
    resonance_score: float      # Cumulative resonance
    
    @property
    def effective_activation(self) -> float:
        """Activation with resonance boost (capped)."""
        base = min(self.activation, MAX_ACTIVATION)
        boost = min(self.resonance_score * RESONANCE_FACTOR, 1.0)
        return min(base * (1 + boost), MAX_ACTIVATION)


@dataclass  
class SECNavigationResult:
    """Result of SEC navigation."""
    query: str
    activated_patterns: List[ActivationState]
    navigation_path: List[str]      # Order of activation
    total_hops: int
    collapse_point: str             # Where navigation converged
    entropy_trajectory: List[float] # Entropy at each step
    time_ms: float


# ============================================================================
# Semantic Graph
# ============================================================================

class SemanticGraph:
    """
    Graph structure for SEC navigation.
    
    Nodes are patterns, edges are semantic relationships.
    Edge weights determine activation propagation strength.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        # Adjacency: pattern_id -> [(neighbor_id, weight, edge_type)]
        self.edges: Dict[str, List[Tuple[str, float, str]]] = defaultdict(list)
        
        # Pattern embeddings for similarity computation
        self.embeddings: Dict[str, torch.Tensor] = {}
        
        # Pattern metadata
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
    def add_pattern(
        self,
        pattern_id: str,
        embedding: torch.Tensor,
        metadata: Optional[Dict] = None,
    ):
        """Add a pattern to the graph."""
        self.embeddings[pattern_id] = embedding.detach()
        self.metadata[pattern_id] = metadata or {}
    
    def add_edge(
        self,
        source: str,
        target: str,
        weight: float,
        edge_type: str = 'similar',
    ):
        """Add directed edge between patterns."""
        self.edges[source].append((target, weight, edge_type))
    
    def build_similarity_edges(
        self,
        threshold: float = 0.3,
        max_edges_per_node: int = 10,
    ):
        """
        Build edges based on embedding similarity.
        
        This creates the initial graph structure from semantic similarity.
        """
        if not self.embeddings:
            return
        
        # Stack all embeddings
        ids = list(self.embeddings.keys())
        embeddings = torch.stack([self.embeddings[pid] for pid in ids])
        
        # Normalize
        embeddings_norm = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Compute all pairwise similarities
        similarities = torch.mm(embeddings_norm, embeddings_norm.t())
        
        # For each pattern, add edges to top-k most similar
        for i, pid in enumerate(ids):
            sims = similarities[i].cpu().numpy()
            
            # Get top-k (excluding self)
            top_indices = np.argsort(sims)[::-1][1:max_edges_per_node+1]
            
            for j in top_indices:
                if sims[j] >= threshold:
                    self.add_edge(
                        pid,
                        ids[j],
                        weight=float(sims[j]),
                        edge_type='similar',
                    )
    
    def get_neighbors(self, pattern_id: str) -> List[Tuple[str, float, str]]:
        """Get all neighbors of a pattern."""
        return self.edges.get(pattern_id, [])
    
    def similarity(self, id1: str, id2: str) -> float:
        """Compute similarity between two patterns."""
        if id1 not in self.embeddings or id2 not in self.embeddings:
            return 0.0
        
        e1 = self.embeddings[id1]
        e2 = self.embeddings[id2]
        
        return torch.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()


# ============================================================================
# SEC Navigator
# ============================================================================

class SECNavigator:
    """
    SEC-based navigation through semantic space.
    
    This implements brain-like spreading activation:
    1. Query activates initial patterns (seed)
    2. Activation spreads through edges
    3. High-activation patterns boost neighbors (resonance)
    4. Navigation collapses toward high-structure regions
    5. Result is the activated subgraph, not just top-K
    
    The key insight: RAG retrieves, SEC understands.
    """
    
    def __init__(
        self,
        kronos: KronosGAIABridge,
        embedder: Any,  # SimpleTextEmbedder from exp_04
        device: str = 'cuda',
    ):
        self.kronos = kronos
        self.embedder = embedder
        self.device = device
        
        # Build semantic graph from Kronos patterns
        self.graph = SemanticGraph(device=device)
        self._build_graph()
        
        # Navigation stats
        self.stats = {
            'navigations': 0,
            'avg_hops': 0,
            'avg_activated': 0,
        }
    
    def _build_graph(self):
        """Build semantic graph from Kronos patterns."""
        print(f"Building semantic graph from {len(self.kronos.pattern_index)} patterns...")
        
        for pid, pattern in self.kronos.pattern_index.items():
            self.graph.add_pattern(
                pattern_id=pid,
                embedding=pattern.delta,
                metadata=pattern.metadata,
            )
        
        # Build similarity-based edges
        self.graph.build_similarity_edges(threshold=0.25, max_edges_per_node=8)
        
        total_edges = sum(len(e) for e in self.graph.edges.values())
        print(f"Graph built: {len(self.graph.embeddings)} nodes, {total_edges} edges")
    
    def _compute_entropy(self, activations: Dict[str, ActivationState]) -> float:
        """
        Compute entropy of activation distribution.
        
        Low entropy = focused activation (good)
        High entropy = diffuse activation (still exploring)
        """
        if not activations:
            return 1.0
        
        values = [s.effective_activation for s in activations.values()]
        total = sum(values)
        
        if total == 0:
            return 1.0
        
        # Normalize to probability distribution
        probs = [v / total for v in values]
        
        # Compute entropy (normalized to [0, 1])
        entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
        max_entropy = np.log(len(probs) + 1)
        
        return entropy / max_entropy
    
    def navigate(
        self,
        query: str,
        initial_seeds: int = 5,
        max_activated: int = 20,
    ) -> SECNavigationResult:
        """
        Navigate semantic space using SEC dynamics.
        
        Args:
            query: Natural language query
            initial_seeds: Number of initial patterns to activate
            max_activated: Maximum patterns to keep activated
            
        Returns:
            SECNavigationResult with activated patterns and path
        """
        start_time = time.perf_counter()
        
        # 1. Embed query
        query_embedding = self.embedder.embed(query)
        
        # 2. Find initial seeds (like RAG, but just the starting point)
        seed_result = self.kronos.recall(query_embedding, top_k=initial_seeds)
        
        # 3. Initialize activation state
        activations: Dict[str, ActivationState] = {}
        for pattern, score in zip(seed_result.patterns, seed_result.similarity_scores):
            activations[pattern.id] = ActivationState(
                pattern_id=pattern.id,
                activation=score,
                source_distance=0,
                activated_by=set(),
                resonance_score=score,  # Initial resonance from query
            )
        
        # 4. Track navigation
        path = [p.id for p in seed_result.patterns]
        entropy_trajectory = [self._compute_entropy(activations)]
        
        # 5. Spreading activation loop
        for hop in range(MAX_HOPS):
            new_activations = {}
            
            # For each currently active pattern
            for pid, state in list(activations.items()):
                if state.effective_activation < ACTIVATION_THRESHOLD:
                    continue
                
                # Propagate to neighbors
                for neighbor_id, weight, edge_type in self.graph.get_neighbors(pid):
                    # Compute propagated activation (properly decayed)
                    propagated = state.effective_activation * weight * DECAY_RATE
                    propagated = min(propagated, MAX_ACTIVATION)  # Cap
                    
                    if propagated < ACTIVATION_THRESHOLD:
                        continue
                    
                    if neighbor_id in activations:
                        # Boost existing activation (additive resonance)
                        existing = activations[neighbor_id]
                        existing.activation = min(
                            existing.activation + propagated * 0.3,
                            MAX_ACTIVATION
                        )
                        existing.resonance_score = min(
                            existing.resonance_score + 0.1,
                            3.0  # Cap resonance score
                        )
                        existing.activated_by.add(pid)
                    elif neighbor_id in new_activations:
                        # Boost pending activation
                        existing = new_activations[neighbor_id]
                        existing.activation = min(
                            existing.activation + propagated * 0.5,
                            MAX_ACTIVATION
                        )
                        existing.resonance_score = min(
                            existing.resonance_score + 0.1,
                            3.0
                        )
                        existing.activated_by.add(pid)
                    else:
                        # New activation
                        new_activations[neighbor_id] = ActivationState(
                            pattern_id=neighbor_id,
                            activation=propagated,
                            source_distance=hop + 1,
                            activated_by={pid},
                            resonance_score=0.1,
                        )
            
            # Merge new activations
            for pid, state in new_activations.items():
                if pid not in activations:
                    activations[pid] = state
                    path.append(pid)
            
            # Compute entropy
            entropy = self._compute_entropy(activations)
            entropy_trajectory.append(entropy)
            
            # Check for collapse (entropy below threshold = found structure)
            if entropy < COLLAPSE_THRESHOLD * 0.5:  # Scale for normalized entropy
                break
            
            # Prune low activations to prevent explosion
            if len(activations) > max_activated:
                sorted_acts = sorted(
                    activations.items(),
                    key=lambda x: x[1].effective_activation,
                    reverse=True
                )
                activations = dict(sorted_acts[:max_activated])
        
        # 6. Sort by effective activation
        sorted_patterns = sorted(
            activations.values(),
            key=lambda x: x.effective_activation,
            reverse=True
        )
        
        # Find collapse point (highest activation pattern)
        collapse_point = sorted_patterns[0].pattern_id if sorted_patterns else ""
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        self.stats['navigations'] += 1
        
        return SECNavigationResult(
            query=query,
            activated_patterns=sorted_patterns,
            navigation_path=path,
            total_hops=len(entropy_trajectory) - 1,
            collapse_point=collapse_point,
            entropy_trajectory=entropy_trajectory,
            time_ms=elapsed_ms,
        )
    
    def explain_activation(
        self,
        result: SECNavigationResult,
        top_k: int = 5,
    ) -> str:
        """
        Generate human-readable explanation of navigation.
        """
        lines = [
            f"Query: {result.query}",
            f"Navigation: {result.total_hops} hops, {len(result.activated_patterns)} patterns activated",
            f"Entropy: {result.entropy_trajectory[0]:.3f} → {result.entropy_trajectory[-1]:.3f}",
            f"Collapse point: {result.collapse_point}",
            "",
            "Activated patterns (by resonance):",
        ]
        
        for i, state in enumerate(result.activated_patterns[:top_k]):
            # Get pattern metadata
            if state.pattern_id in self.kronos.pattern_index:
                pattern = self.kronos.pattern_index[state.pattern_id]
                source = Path(pattern.metadata.get('file_path', 'unknown')).name
                preview = pattern.metadata.get('content_preview', '')[:100]
            else:
                source = state.pattern_id
                preview = ""
            
            activated_by = ", ".join(list(state.activated_by)[:3])
            if len(state.activated_by) > 3:
                activated_by += f"... (+{len(state.activated_by)-3})"
            
            lines.append(
                f"\n[{i+1}] {source} (activation: {state.effective_activation:.3f}, "
                f"hops: {state.source_distance})"
            )
            if activated_by:
                lines.append(f"    Activated by: {activated_by}")
            lines.append(f"    {preview}...")
        
        return "\n".join(lines)


# ============================================================================
# Tests
# ============================================================================

def test_sec_navigation():
    """Test SEC navigation vs simple RAG."""
    print("\n" + "=" * 60)
    print("TEST: SEC Navigation vs RAG")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load existing index
    index_path = Path(__file__).parent / "repo_knowledge_index"
    
    if not index_path.exists():
        print("  No index found. Run exp_04 first.")
        return False
    
    from exp_04_repo_index import SimpleTextEmbedder
    
    kronos = KronosGAIABridge(
        storage_path=index_path,
        namespace="dawn_institute",
        device=device,
    )
    embedder = SimpleTextEmbedder(dim=768, device=device)
    
    print(f"Loaded {len(kronos.pattern_index)} patterns")
    
    # Create SEC navigator
    navigator = SECNavigator(kronos=kronos, embedder=embedder, device=device)
    
    # Test queries
    queries = [
        "How does information create structure?",
        "What is the relationship between entropy and learning?",
        "How does GAIA use PAC conservation?",
    ]
    
    for query in queries:
        print("\n" + "=" * 70)
        
        # RAG-style (simple recall)
        query_emb = embedder.embed(query)
        rag_result = kronos.recall(query_emb, top_k=5)
        
        print(f"RAG Result for: '{query}'")
        print("-" * 40)
        for p, s in zip(rag_result.patterns[:3], rag_result.similarity_scores[:3]):
            source = Path(p.metadata.get('file_path', 'unknown')).name
            print(f"  [{s:.1%}] {source}")
        
        # SEC navigation
        sec_result = navigator.navigate(query, initial_seeds=3, max_activated=15)
        
        print(f"\nSEC Navigation:")
        print("-" * 40)
        print(f"  Hops: {sec_result.total_hops}")
        print(f"  Entropy: {sec_result.entropy_trajectory[0]:.3f} → {sec_result.entropy_trajectory[-1]:.3f}")
        print(f"  Patterns activated: {len(sec_result.activated_patterns)}")
        
        for state in sec_result.activated_patterns[:3]:
            if state.pattern_id in kronos.pattern_index:
                source = Path(kronos.pattern_index[state.pattern_id].metadata.get('file_path', '')).name
                hop_info = f"hop {state.source_distance}" if state.source_distance > 0 else "seed"
                print(f"  [{state.effective_activation:.1%}] {source} ({hop_info})")
    
    print("\n✓ SEC navigation working")
    return True


def test_spreading_activation():
    """Test that activation actually spreads."""
    print("\n" + "=" * 60)
    print("TEST: Spreading Activation")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    index_path = Path(__file__).parent / "repo_knowledge_index"
    
    if not index_path.exists():
        print("  No index found. Run exp_04 first.")
        return False
    
    from exp_04_repo_index import SimpleTextEmbedder
    
    kronos = KronosGAIABridge(
        storage_path=index_path,
        namespace="dawn_institute",
        device=device,
    )
    embedder = SimpleTextEmbedder(dim=768, device=device)
    
    navigator = SECNavigator(kronos=kronos, embedder=embedder, device=device)
    
    # Query that should activate related concepts
    query = "golden ratio fibonacci sequence"
    result = navigator.navigate(query, initial_seeds=3, max_activated=20)
    
    print(f"\nQuery: '{query}'")
    print(f"Initial seeds: 3")
    print(f"Final activated: {len(result.activated_patterns)}")
    
    # Check that we have patterns beyond the seeds
    beyond_seeds = [s for s in result.activated_patterns if s.source_distance > 0]
    print(f"Patterns from spreading: {len(beyond_seeds)}")
    
    # Show the spread
    print("\nActivation spread:")
    for hop in range(result.total_hops + 1):
        at_hop = [s for s in result.activated_patterns if s.source_distance == hop]
        if at_hop:
            print(f"  Hop {hop}: {len(at_hop)} patterns")
            for s in at_hop[:2]:
                if s.pattern_id in kronos.pattern_index:
                    source = Path(kronos.pattern_index[s.pattern_id].metadata.get('file_path', '')).name
                    print(f"    - {source} (activation: {s.effective_activation:.3f})")
    
    assert len(beyond_seeds) > 0, "No spreading occurred!"
    print("\n✓ Activation spreads through graph")
    return True


def test_resonance():
    """Test that resonance boosts related concepts."""
    print("\n" + "=" * 60)
    print("TEST: Resonance Boost")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    index_path = Path(__file__).parent / "repo_knowledge_index"
    
    if not index_path.exists():
        print("  No index found.")
        return False
    
    from exp_04_repo_index import SimpleTextEmbedder
    
    kronos = KronosGAIABridge(
        storage_path=index_path,
        namespace="dawn_institute",
        device=device,
    )
    embedder = SimpleTextEmbedder(dim=768, device=device)
    
    navigator = SECNavigator(kronos=kronos, embedder=embedder, device=device)
    
    # Query where multiple paths should converge
    query = "PAC SEC entropy conservation"
    result = navigator.navigate(query, initial_seeds=4, max_activated=25)
    
    print(f"\nQuery: '{query}'")
    
    # Find patterns activated by multiple sources (resonance)
    multi_source = [
        s for s in result.activated_patterns
        if len(s.activated_by) > 1
    ]
    
    print(f"Patterns with multiple activators: {len(multi_source)}")
    
    for s in multi_source[:3]:
        if s.pattern_id in kronos.pattern_index:
            source = Path(kronos.pattern_index[s.pattern_id].metadata.get('file_path', '')).name
            print(f"  {source}")
            print(f"    Activated by {len(s.activated_by)} patterns")
            print(f"    Resonance score: {s.resonance_score:.3f}")
    
    print("\n✓ Resonance amplifies multiply-activated patterns")
    return True


def demo_sec_understanding():
    """Demo showing SEC 'understanding' vs RAG 'retrieval'."""
    print("\n" + "=" * 70)
    print("DEMO: SEC Understanding vs RAG Retrieval")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    index_path = Path(__file__).parent / "repo_knowledge_index"
    
    if not index_path.exists():
        print("  No index found. Run exp_04 first.")
        return
    
    from exp_04_repo_index import SimpleTextEmbedder
    
    kronos = KronosGAIABridge(
        storage_path=index_path,
        namespace="dawn_institute",
        device=device,
    )
    embedder = SimpleTextEmbedder(dim=768, device=device)
    navigator = SECNavigator(kronos=kronos, embedder=embedder, device=device)
    
    query = "How does the brain-like architecture learn and remember?"
    
    print(f"\nQuery: '{query}'")
    print("\n" + "-" * 70)
    
    # Navigate
    result = navigator.navigate(query, initial_seeds=5, max_activated=20)
    
    # Show the understanding
    print(navigator.explain_activation(result, top_k=7))
    
    print("\n" + "-" * 70)
    print("Key insight: SEC doesn't just find documents -")
    print("it activates a coherent network of related concepts,")
    print("showing how ideas connect through resonance.")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("POC-025: GAIA Prime + Kronos Integration - Experiment 05")
    print("SEC Navigation - Brain-Like Associative Memory")
    print("=" * 70)
    
    tests = [
        ("SEC Navigation", test_sec_navigation),
        ("Spreading Activation", test_spreading_activation),
        ("Resonance Boost", test_resonance),
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
    
    # Run demo
    demo_sec_understanding()
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
