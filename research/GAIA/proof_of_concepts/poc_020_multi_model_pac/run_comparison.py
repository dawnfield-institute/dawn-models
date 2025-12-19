"""
POC-020: Full Multi-Model PAC Comparison Pipeline

Extract multiple models into PAC trees and compare their structures.
"""

import os
import sys
import torch
from multi_model_extractor import ModelToPACExtractor, save_pac_trees
from pac_tree_comparator import PACTreeComparator, generate_report


def main():
    print("=" * 70)
    print("POC-020: MULTI-MODEL PAC TREE COMPARISON")
    print("=" * 70)
    print("\nExtracting multiple models into PAC trees and comparing")
    print("their confluence structures to find universal patterns.\n")
    
    # Step 1: Extract models
    print("=" * 70)
    print("STEP 1: EXTRACT MODELS INTO PAC TREES")
    print("=" * 70)
    
    extractor = ModelToPACExtractor()
    
    # Models to extract
    models = ['gpt2', 'bert', 'pythia']
    print(f"\nExtracting: {models}")
    
    all_trees = extractor.extract_all(models)
    
    if not all_trees:
        print("❌ No models extracted successfully")
        return
        
    # Save trees
    save_pac_trees(all_trees, "extracted_trees")
    
    # Step 2: Compare structures
    print("\n" + "=" * 70)
    print("STEP 2: COMPARE PAC TREE STRUCTURES")
    print("=" * 70)
    
    comparator = PACTreeComparator()
    
    # Load saved trees
    print("\n📂 Loading saved PAC trees...")
    trees = comparator.load_trees_json("extracted_trees")
    
    # Analyze structure
    print("\n📊 Analyzing tree structures...")
    analysis = comparator.analyze_structure(trees)
    
    for model, stats in analysis.items():
        print(f"\n  {model.upper()}:")
        print(f"    Nodes: {stats['total_nodes']}, Depth: {stats['max_depth']}")
        print(f"    Types: {list(stats['types'].keys())}")
    
    # Find patterns
    print("\n\n🔍 Finding patterns...")
    
    universal = comparator.find_universal_patterns(trees)
    print(f"  Universal patterns: {len(universal)}")
    
    arch_specific = comparator.find_architecture_patterns(trees)
    print(f"  Architecture-specific: {len(arch_specific)}")
    
    complementary = comparator.find_complementary_patterns(trees)
    print(f"  Complementary relationships: {len(complementary)}")
    
    # Step 3: Build unified tree
    print("\n" + "=" * 70)
    print("STEP 3: BUILD UNIFIED META-PAC TREE")
    print("=" * 70)
    
    unified = comparator.build_unified_tree(trees)
    print(f"\n  Unified tree nodes: {len(unified)}")
    
    # Step 4: Generate report
    print("\n" + "=" * 70)
    print("STEP 4: ANALYSIS REPORT")
    print("=" * 70)
    
    report = generate_report(analysis, universal, arch_specific, complementary, unified)
    print(report)
    
    # Save all results
    os.makedirs("results", exist_ok=True)
    
    with open("results/comparison_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)
        
    import json
    with open("results/unified_pac_tree.json", 'w', encoding='utf-8') as f:
        json.dump(unified, f, indent=2)
        
    patterns_data = {
        'universal': [p.to_dict() for p in universal],
        'architecture_specific': [p.to_dict() for p in arch_specific],
        'complementary': complementary
    }
    with open("results/patterns.json", 'w', encoding='utf-8') as f:
        json.dump(patterns_data, f, indent=2)
        
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n✅ Extracted {len(all_trees)} models into PAC trees")
    print(f"✅ Found {len(universal)} universal patterns")
    print(f"✅ Found {len(arch_specific)} architecture-specific patterns")
    print(f"✅ Found {len(complementary)} complementary relationships")
    print(f"✅ Built unified meta-PAC tree with {len(unified)} nodes")
    print(f"\n💾 Results saved to results/")
    
    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
Different models encode knowledge in different structures:
- GPT-2: Sequential attention layers (decoder-only)
- BERT: Bidirectional confluence points (encoder)
- Pythia: Clean scaling patterns (training transparency)

By comparing PAC trees, we can:
1. Find UNIVERSAL patterns all intelligence shares
2. Identify COMPLEMENTARY knowledge to combine
3. Build UNIFIED representations transcending architecture
""")


if __name__ == "__main__":
    main()
