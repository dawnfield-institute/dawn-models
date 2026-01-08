"""
POC-025: GAIA Prime + Kronos Integration
========================================

Interactive Demo: Query the Dawn Field Institute Repository

This indexes the full repo and lets you ask questions about it.
GAIA retrieves relevant context from Kronos to "know" things it wasn't trained on.
"""

import sys
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\fracton")
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\dawn-models\research\GAIA\src")

import torch
import shutil
from pathlib import Path

from exp_04_repo_index import RepoKnowledgeIndex


def main():
    """Interactive demo."""
    print("=" * 70)
    print("GAIA + Kronos: Repository Knowledge Demo")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Storage path
    index_path = Path(__file__).parent / "repo_knowledge_index"
    
    # Check if index already exists
    if index_path.exists():
        print(f"\nLoading existing index from {index_path}")
        index = RepoKnowledgeIndex(
            storage_path=index_path,
            namespace="dawn_institute",
            device=device,
        )
        print(f"Loaded {len(index.kronos.pattern_index)} patterns")
    else:
        print("\nBuilding new index...")
        index = RepoKnowledgeIndex(
            storage_path=index_path,
            namespace="dawn_institute",
            device=device,
            chunk_size=300,  # Smaller chunks for better precision
        )
        
        # Index key directories
        repo_root = Path(r"c:\Users\peter\repos\Dawn Field Institute")
        
        dirs_to_index = [
            repo_root / "dawn-field-theory",
            repo_root / "dawn-models" / "research" / "GAIA" / "proof_of_concepts",
            repo_root / "fracton" / "docs",
        ]
        
        for dir_path in dirs_to_index:
            if dir_path.exists():
                print(f"\nIndexing: {dir_path.relative_to(repo_root)}")
                # Index up to 200 files per dir for deeper PAC coverage
                stats = index.index_directory(dir_path, recursive=True, max_files=200)
                print(f"  {stats['files_indexed']} files, {stats['chunks_created']} chunks")
        
        index.sync()
        print(f"\nIndex built: {len(index.kronos.pattern_index)} total patterns")
    
    # Interactive query loop
    print("\n" + "=" * 70)
    print("Ask questions about the Dawn Field Institute repository!")
    print("Type 'quit' to exit, 'stats' for index stats, 'rebuild' to reindex")
    print("=" * 70)
    
    while True:
        print()
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not question:
            continue
        
        if question.lower() == 'quit':
            break
        
        if question.lower() == 'stats':
            print(f"\nIndex Statistics:")
            print(f"  Patterns: {len(index.kronos.pattern_index)}")
            print(f"  Documents: {len(index.documents)}")
            print(f"  Queries processed: {index.stats['queries_processed']}")
            continue
        
        if question.lower() == 'rebuild':
            if index_path.exists():
                shutil.rmtree(index_path)
            print("Index cleared. Restart to rebuild.")
            break
        
        # Query the index
        results = index.query(question, top_k=5)
        
        if not results:
            print("\nNo relevant documents found.")
            continue
        
        print(f"\nFound {len(results)} relevant chunks:\n")
        
        for i, (doc, score) in enumerate(results):
            source = Path(doc.file_path).name
            # Get a clean preview
            preview = doc.content[:250].replace('\n', ' ').strip()
            
            print(f"[{i+1}] {source} (relevance: {score:.1%})")
            print(f"    {preview}...")
            print()
    
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
