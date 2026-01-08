"""
POC-025: GAIA Prime + Kronos Integration
========================================

Interactive Demo: Brain-Like Understanding

This demo shows the difference between:
- Traditional RAG: "Here are documents matching your query"
- SEC Navigation: "Here's what I understand by thinking about your question"

Usage:
    python demo_sec_brain.py

Then ask questions and watch the brain think!
"""

import sys
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\fracton")
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\dawn-models\research\GAIA\src")

import torch
from pathlib import Path

from exp_01_bridge import KronosGAIABridge
from exp_04_repo_index import SimpleTextEmbedder
from exp_05_sec_navigation import SECNavigator
from exp_06_associative_chains import AssociativeMind


def print_header():
    """Print demo header."""
    print("=" * 70)
    print("   GAIA + KRONOS: Brain-Like Understanding Demo")
    print("=" * 70)
    print("""
This is NOT a chatbot. This is a demonstration of SEC-based navigation
through semantic space - how a brain might "think" about a question by
activating related concepts until understanding crystallizes.

Traditional RAG: Query → Vector Match → Top-K Documents
SEC Navigation:  Query → Spreading Activation → Resonance → Insight

Commands:
  - Type a question to think about it
  - "compare <question>" to see RAG vs SEC side by side
  - "memory" to see current working memory
  - "chains" to see reasoning chains
  - "reset" to clear working memory
  - "quit" to exit
""")
    print("=" * 70)


def compare_rag_vs_sec(question: str, mind: AssociativeMind):
    """Show side-by-side comparison of RAG vs SEC."""
    print("\n" + "=" * 70)
    print(f"COMPARING: {question}")
    print("=" * 70)
    
    # RAG-style (simple vector search)
    print("\n[RAG] Vector Similarity Search:")
    print("-" * 35)
    
    query_emb = mind.embedder.embed(question)
    rag_result = mind.kronos.recall(query_emb, top_k=5)
    
    for i, (p, s) in enumerate(zip(rag_result.patterns, rag_result.similarity_scores)):
        source = Path(p.metadata.get('file_path', 'unknown')).name
        print(f"  {i+1}. [{s:.1%}] {source}")
    
    # SEC-style (spreading activation)
    print("\n[SEC] Spreading Activation + Resonance:")
    print("-" * 40)
    
    insight = mind.think(question, use_context=True)
    
    print(f"  Confidence: {insight.confidence:.1%}")
    print(f"  Concepts activated: {len(insight.supporting_concepts)}")
    print(f"  Reasoning chains: {len(insight.reasoning_chains)}")
    
    if insight.reasoning_chains:
        for chain in insight.reasoning_chains[:2]:
            sources = [n.source for n in chain.nodes]
            print(f"  [{chain.chain_type}] {' → '.join(sources[:4])}")
    
    print("\n  Understanding:")
    # Wrap text
    words = insight.main_insight.split()
    line = "    "
    for word in words[:50]:
        if len(line) + len(word) > 68:
            print(line)
            line = "    "
        line += word + " "
    if line.strip():
        print(line)


def main():
    """Run the demo."""
    print_header()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    index_path = Path(__file__).parent / "repo_knowledge_index"
    
    if not index_path.exists():
        print("ERROR: No index found. Run exp_04_repo_index.py first.")
        return
    
    print("Loading knowledge base...")
    kronos = KronosGAIABridge(
        storage_path=index_path,
        namespace="dawn_institute",
        device=device,
    )
    embedder = SimpleTextEmbedder(dim=768, device=device)
    
    print(f"Loaded {len(kronos.pattern_index)} knowledge patterns")
    
    print("Building semantic graph...")
    mind = AssociativeMind(kronos=kronos, embedder=embedder, device=device)
    
    print("\nReady! Ask a question...\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if user_input.lower() == 'memory':
            print("\n" + mind.working_memory.get_context_summary())
            print()
            continue
        
        if user_input.lower() == 'chains':
            chains = mind.working_memory.find_chains()
            print(f"\nReasoning chains ({len(chains)}):")
            for chain in chains[:5]:
                sources = [n.source for n in chain.nodes]
                print(f"  [{chain.chain_type}] {' → '.join(sources[:4])}")
            print()
            continue
        
        if user_input.lower() == 'reset':
            from exp_06_associative_chains import WorkingMemory
            mind.working_memory = WorkingMemory(capacity=15)
            print("Working memory cleared.\n")
            continue
        
        if user_input.lower().startswith('compare '):
            question = user_input[8:].strip()
            compare_rag_vs_sec(question, mind)
            print()
            continue
        
        # Think about the question
        print("\n[Thinking...]")
        
        insight = mind.think(user_input, use_context=True)
        
        print(mind.explain(insight))
        print()


if __name__ == "__main__":
    main()
