"""
Test PAC Extraction from Pythia-70M
====================================

Extract learned knowledge from Pythia-70M as a PAC tree.
This is the first step toward training-free knowledge transfer.

Usage:
    python test_extraction.py [--full]
"""

import argparse
import torch
from pathlib import Path
import sys
import os

from extractor import ModelToPACExtractor, ExtractionConfig


def test_pythia_extraction(full_extraction: bool = False):
    """Extract PAC tree from Pythia-70M."""
    
    print("="*70)
    print("POC-016: PAC Extraction from Pythia-70M")
    print("="*70)
    print("\nHypothesis: Trained model capabilities can be extracted as")
    print("architecture-agnostic PAC trees without needing training data.\n")
    
    # Configure extraction
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    config = ExtractionConfig(
        model_name="EleutherAI/pythia-70m",  # 70M params, Apache 2.0 license
        device=device,
        probe_samples=100 if full_extraction else 30,
        entropy_bins=50,
        min_activation_strength=0.3,
        field_dim=256,  # Match GAIA-1 field dimension
        batch_size=8 if device == "cuda" else 4
    )
    
    print(f"Probe samples: {config.probe_samples}")
    print(f"Field dimension: {config.field_dim}")
    
    # Create extractor
    extractor = ModelToPACExtractor(config)
    
    # Extract PAC tree
    pac_nodes = extractor.extract()
    
    # Save extracted PAC
    output_path = Path(__file__).parent / "extracted" / "pythia_70m"
    save_path = extractor.save_pac_tree(output_path)
    
    # Display results
    print("\n" + "="*70)
    print("EXTRACTION RESULTS")
    print("="*70)
    
    print(f"\nSource Model: {config.model_name}")
    print(f"Model Parameters: {extractor.extraction_metadata['model_params']:,}")
    print(f"PAC Nodes Created: {len(pac_nodes)}")
    
    # Show capability zones
    print(f"\nCapability Zones Detected:")
    for zone in extractor.extraction_metadata['capability_zones']:
        print(f"  • {zone['type']}")
        print(f"    - Layers: {zone['num_layers']}")
        print(f"    - Learning Strength: {zone['learning_strength']:.3f}")
        print(f"    - Entropy: {zone['entropy']:.3f}")
    
    # Show tree structure
    print(f"\nPAC Tree Structure:")
    root_node = pac_nodes.get('root')
    if root_node:
        print(f"  {root_node.label} (root)")
        for child_id in root_node.children:
            child = pac_nodes.get(child_id)
            if child:
                status = "🔷" if child.crystallized else "⚪"
                print(f"    {status} {child.label} (importance={child.importance:.3f})")
                for subchild_id in child.children[:3]:
                    subchild = pac_nodes.get(subchild_id)
                    if subchild:
                        substatus = "🔷" if subchild.crystallized else "⚪"
                        print(f"        {substatus} {subchild.label}")
    
    # Statistics
    n_crystallized = sum(1 for n in pac_nodes.values() if n.crystallized)
    avg_importance = sum(n.importance for n in pac_nodes.values()) / len(pac_nodes)
    
    print(f"\nStatistics:")
    print(f"  • Crystallized nodes: {n_crystallized}/{len(pac_nodes)}")
    print(f"  • Average importance: {avg_importance:.3f}")
    print(f"  • Output saved to: {save_path}")
    
    print("\n" + "="*70)
    print("✅ POC-016 EXTRACTION SUCCESSFUL")
    print("="*70)
    print("\nNext Step: POC-017 - Import this PAC tree into GAIA-1")
    print("Goal: GAIA acquires language capabilities WITHOUT training!")
    
    return pac_nodes, extractor.extraction_metadata


def verify_saved_pac(output_path: Path):
    """Verify the saved PAC tree can be loaded."""
    import json
    
    print("\nVerifying saved PAC tree...")
    
    # Load patterns
    patterns_path = output_path / "patterns.pt"
    patterns = torch.load(patterns_path)
    print(f"  ✓ Loaded {len(patterns)} patterns")
    
    # Load tree structure
    tree_path = output_path / "tree_structure.json"
    with open(tree_path) as f:
        tree_data = json.load(f)
    print(f"  ✓ Loaded tree with {len(tree_data['nodes'])} nodes")
    
    # Load metadata
    meta_path = output_path / "extraction_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)
    print(f"  ✓ Source model: {metadata['source_model']}")
    
    print("  ✓ PAC tree verification complete!")
    
    return patterns, tree_data, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PAC from Pythia-70M")
    parser.add_argument("--full", action="store_true", help="Full extraction (100 samples)")
    args = parser.parse_args()
    
    nodes, metadata = test_pythia_extraction(full_extraction=args.full)
    
    # Verify saved output
    output_path = Path(__file__).parent / "extracted" / "pythia_70m"
    if output_path.exists():
        verify_saved_pac(output_path)
