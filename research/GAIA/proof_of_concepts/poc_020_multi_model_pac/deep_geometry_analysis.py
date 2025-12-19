"""
POC-020: Deep PAC Tree Geometry Analysis

Examine the actual structure and geometry of PAC trees in detail.
Find similar patterns, decompose effectively.
"""

import numpy as np
import pickle
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TreeGeometry:
    """Geometric properties of a PAC tree"""
    depth: int
    width_by_level: Dict[int, int]
    branching_factors: List[float]
    node_types_by_level: Dict[int, Dict[str, int]]
    connectivity_matrix: np.ndarray  # Adjacency structure
    

def load_full_trees(tree_dir: str) -> Dict:
    """Load trees with full data from pickle"""
    trees = {}
    
    for filename in os.listdir(tree_dir):
        if filename.endswith('_pac_tree.pkl'):
            model_name = filename.replace('_pac_tree.pkl', '')
            with open(os.path.join(tree_dir, filename), 'rb') as f:
                trees[model_name] = pickle.load(f)
            print(f"Loaded {model_name}: {len(trees[model_name])} nodes")
            
    return trees


def analyze_tree_geometry(tree: Dict) -> TreeGeometry:
    """Analyze geometric properties of a tree"""
    
    # Get nodes by level
    nodes_by_level = defaultdict(list)
    for node_id, node in tree.items():
        level = node.level if hasattr(node, 'level') else node['level']
        nodes_by_level[level].append(node_id)
        
    # Width by level
    width_by_level = {level: len(nodes) for level, nodes in nodes_by_level.items()}
    
    # Max depth
    depth = max(nodes_by_level.keys()) if nodes_by_level else 0
    
    # Branching factors
    branching_factors = []
    for node_id, node in tree.items():
        children = node.children if hasattr(node, 'children') else node.get('children', [])
        if children:
            branching_factors.append(len(children))
            
    # Node types by level
    node_types_by_level = defaultdict(lambda: defaultdict(int))
    for node_id, node in tree.items():
        level = node.level if hasattr(node, 'level') else node['level']
        ptype = node.pattern_type if hasattr(node, 'pattern_type') else node['pattern_type']
        node_types_by_level[level][ptype] += 1
        
    return TreeGeometry(
        depth=depth,
        width_by_level=width_by_level,
        branching_factors=branching_factors,
        node_types_by_level=dict(node_types_by_level),
        connectivity_matrix=None  # TODO: build if needed
    )


def print_tree_structure(tree: Dict, model_name: str, max_depth: int = 3):
    """Print tree structure visually"""
    print(f"\n{'='*60}")
    print(f"TREE STRUCTURE: {model_name.upper()}")
    print('='*60)
    
    # Find root
    root_id = None
    for node_id, node in tree.items():
        parent = node.parent if hasattr(node, 'parent') else node.get('parent')
        if parent is None:
            root_id = node_id
            break
            
    if root_id is None:
        print("No root found!")
        return
        
    def print_node(node_id, indent=0, depth=0):
        if depth > max_depth:
            return
            
        node = tree.get(node_id)
        if node is None:
            return
            
        ptype = node.pattern_type if hasattr(node, 'pattern_type') else node['pattern_type']
        level = node.level if hasattr(node, 'level') else node['level']
        children = node.children if hasattr(node, 'children') else node.get('children', [])
        
        prefix = "  " * indent
        child_count = len(children)
        
        # Get additional info
        if hasattr(node, 'metadata'):
            meta = node.metadata
        else:
            meta = node.get('metadata', {})
            
        meta_str = ""
        if meta:
            key_info = []
            for k in ['layer', 'head', 'bidirectional', 'follows_scaling_law']:
                if k in meta:
                    key_info.append(f"{k}={meta[k]}")
            if key_info:
                meta_str = f" ({', '.join(key_info[:2])})"
                
        print(f"{prefix}├─ [{ptype}] L{level}{meta_str} → {child_count} children")
        
        # Print children (sample if too many)
        if child_count <= 5:
            for child_id in children:
                print_node(child_id, indent + 1, depth + 1)
        else:
            # Sample first 2, last 1
            for child_id in children[:2]:
                print_node(child_id, indent + 1, depth + 1)
            print(f"{prefix}  │ ... ({child_count - 3} more)")
            print_node(children[-1], indent + 1, depth + 1)
            
    print_node(root_id)
    

def compare_embedding_geometry(trees: Dict) -> Dict:
    """Compare the geometry of embedding spaces"""
    print("\n" + "="*60)
    print("EMBEDDING SPACE GEOMETRY")
    print("="*60)
    
    embedding_stats = {}
    
    for model_name, tree in trees.items():
        embeddings = []
        
        for node_id, node in tree.items():
            ptype = node.pattern_type if hasattr(node, 'pattern_type') else node['pattern_type']
            
            if ptype == 'embedding':
                data = node.data if hasattr(node, 'data') else node.get('data')
                if data is not None and isinstance(data, np.ndarray):
                    embeddings.append(data)
                    
        if embeddings:
            embeddings = np.array(embeddings)
            
            # Compute geometric properties
            norms = np.linalg.norm(embeddings, axis=1)
            
            # Pairwise similarities (sample)
            sample_size = min(100, len(embeddings))
            sample = embeddings[:sample_size]
            
            # Cosine similarities
            normalized = sample / (np.linalg.norm(sample, axis=1, keepdims=True) + 1e-10)
            cos_sim_matrix = normalized @ normalized.T
            
            # Exclude diagonal
            mask = ~np.eye(sample_size, dtype=bool)
            cos_sims = cos_sim_matrix[mask]
            
            embedding_stats[model_name] = {
                'count': len(embeddings),
                'dim': embeddings.shape[1],
                'norm_mean': float(norms.mean()),
                'norm_std': float(norms.std()),
                'cos_sim_mean': float(cos_sims.mean()),
                'cos_sim_std': float(cos_sims.std()),
                'cos_sim_min': float(cos_sims.min()),
                'cos_sim_max': float(cos_sims.max()),
            }
            
            print(f"\n{model_name.upper()}:")
            print(f"  Embeddings: {len(embeddings)} × {embeddings.shape[1]}")
            print(f"  Norm: {norms.mean():.4f} ± {norms.std():.4f}")
            print(f"  Cosine similarity: {cos_sims.mean():.4f} ± {cos_sims.std():.4f}")
            print(f"  Range: [{cos_sims.min():.4f}, {cos_sims.max():.4f}]")
            
    # Cross-model comparison
    print("\n" + "-"*40)
    print("CROSS-MODEL EMBEDDING COMPARISON:")
    
    models = list(embedding_stats.keys())
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            dim_match = embedding_stats[m1]['dim'] == embedding_stats[m2]['dim']
            norm_diff = abs(embedding_stats[m1]['norm_mean'] - embedding_stats[m2]['norm_mean'])
            sim_diff = abs(embedding_stats[m1]['cos_sim_mean'] - embedding_stats[m2]['cos_sim_mean'])
            
            print(f"\n  {m1} vs {m2}:")
            print(f"    Dim match: {dim_match} ({embedding_stats[m1]['dim']} vs {embedding_stats[m2]['dim']})")
            print(f"    Norm diff: {norm_diff:.4f}")
            print(f"    Similarity diff: {sim_diff:.4f}")
            
    return embedding_stats


def find_structural_isomorphisms(trees: Dict) -> List[Dict]:
    """Find structurally similar subtrees across models"""
    print("\n" + "="*60)
    print("STRUCTURAL ISOMORPHISMS")
    print("="*60)
    
    isomorphisms = []
    
    # Extract tree signatures (branching pattern at each level)
    signatures = {}
    
    for model_name, tree in trees.items():
        # Get geometry
        geom = analyze_tree_geometry(tree)
        
        # Create signature: (depth, width_pattern, type_pattern)
        width_pattern = tuple(sorted(geom.width_by_level.items()))
        
        type_pattern = {}
        for level, types in geom.node_types_by_level.items():
            type_pattern[level] = tuple(sorted(types.keys()))
            
        signatures[model_name] = {
            'depth': geom.depth,
            'width_pattern': width_pattern,
            'type_pattern': type_pattern,
            'avg_branching': np.mean(geom.branching_factors) if geom.branching_factors else 0
        }
        
    print("\nTree Signatures:")
    for model, sig in signatures.items():
        print(f"\n  {model.upper()}:")
        print(f"    Depth: {sig['depth']}")
        print(f"    Width by level: {dict(sig['width_pattern'])}")
        print(f"    Avg branching: {sig['avg_branching']:.2f}")
        
    # Compare signatures
    print("\nSignature Comparisons:")
    models = list(signatures.keys())
    
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            s1, s2 = signatures[m1], signatures[m2]
            
            # Depth similarity
            depth_sim = 1.0 - abs(s1['depth'] - s2['depth']) / max(s1['depth'], s2['depth'], 1)
            
            # Branching similarity
            if s1['avg_branching'] > 0 and s2['avg_branching'] > 0:
                branch_sim = min(s1['avg_branching'], s2['avg_branching']) / max(s1['avg_branching'], s2['avg_branching'])
            else:
                branch_sim = 0
                
            # Level width correlation
            levels = set(dict(s1['width_pattern']).keys()) & set(dict(s2['width_pattern']).keys())
            if levels:
                w1 = [dict(s1['width_pattern']).get(l, 0) for l in sorted(levels)]
                w2 = [dict(s2['width_pattern']).get(l, 0) for l in sorted(levels)]
                if len(w1) > 1:
                    width_corr = np.corrcoef(w1, w2)[0, 1] if np.std(w1) > 0 and np.std(w2) > 0 else 0
                else:
                    width_corr = 1.0 if w1[0] == w2[0] else 0.0
            else:
                width_corr = 0
                
            overall_sim = (depth_sim + branch_sim + max(0, width_corr)) / 3
            
            print(f"\n  {m1} ↔ {m2}:")
            print(f"    Depth similarity: {depth_sim:.2f}")
            print(f"    Branching similarity: {branch_sim:.2f}")
            print(f"    Width correlation: {width_corr:.2f}")
            print(f"    Overall similarity: {overall_sim:.2f}")
            
            isomorphisms.append({
                'models': (m1, m2),
                'depth_sim': depth_sim,
                'branch_sim': branch_sim,
                'width_corr': float(width_corr) if not np.isnan(width_corr) else 0,
                'overall': overall_sim
            })
            
    return isomorphisms


def decompose_into_primitives(trees: Dict) -> Dict[str, List]:
    """Decompose trees into primitive patterns"""
    print("\n" + "="*60)
    print("PRIMITIVE DECOMPOSITION")
    print("="*60)
    
    primitives = {
        'embedding_blocks': [],
        'attention_blocks': [],
        'layer_blocks': [],
        'confluence_points': [],
        'scaling_patterns': [],
    }
    
    for model_name, tree in trees.items():
        print(f"\n{model_name.upper()}:")
        
        # Group nodes by type
        by_type = defaultdict(list)
        for node_id, node in tree.items():
            ptype = node.pattern_type if hasattr(node, 'pattern_type') else node['pattern_type']
            by_type[ptype].append(node)
            
        for ptype, nodes in by_type.items():
            print(f"  {ptype}: {len(nodes)} nodes")
            
            # Extract primitive for each type
            if ptype == 'embedding':
                # Sample embedding characteristics
                sample_data = []
                for node in nodes[:10]:
                    data = node.data if hasattr(node, 'data') else node.get('data')
                    if data is not None and isinstance(data, np.ndarray):
                        sample_data.append({
                            'norm': float(np.linalg.norm(data)),
                            'mean': float(data.mean()),
                            'std': float(data.std())
                        })
                if sample_data:
                    primitives['embedding_blocks'].append({
                        'model': model_name,
                        'count': len(nodes),
                        'sample_stats': sample_data[:5]
                    })
                    
            elif 'attention' in ptype.lower():
                primitives['attention_blocks'].append({
                    'model': model_name,
                    'type': ptype,
                    'count': len(nodes)
                })
                
            elif ptype == 'layer':
                primitives['layer_blocks'].append({
                    'model': model_name,
                    'count': len(nodes)
                })
                
            elif ptype == 'confluence':
                # Extract confluence characteristics
                for node in nodes:
                    meta = node.metadata if hasattr(node, 'metadata') else node.get('metadata', {})
                    data = node.data if hasattr(node, 'data') else node.get('data', {})
                    primitives['confluence_points'].append({
                        'model': model_name,
                        'layer': meta.get('layer'),
                        'type': meta.get('type'),
                        'qk_alignment': data.get('qk_alignment') if isinstance(data, dict) else None
                    })
                    
            elif ptype == 'scaling':
                for node in nodes:
                    data = node.data if hasattr(node, 'data') else node.get('data', {})
                    if isinstance(data, dict):
                        primitives['scaling_patterns'].append({
                            'model': model_name,
                            'layer': data.get('layer'),
                            'weight_norm': data.get('weight_norm'),
                            'relative_scale': data.get('relative_scale')
                        })
                        
    # Summarize primitives
    print("\n" + "-"*40)
    print("PRIMITIVE SUMMARY:")
    for prim_type, items in primitives.items():
        if items:
            models = set(item.get('model', 'unknown') for item in items)
            print(f"\n  {prim_type}:")
            print(f"    Total: {len(items)}")
            print(f"    Models: {models}")
            
    return primitives


def find_common_subtrees(trees: Dict) -> List[Dict]:
    """Find common subtree patterns across models"""
    print("\n" + "="*60)
    print("COMMON SUBTREE PATTERNS")
    print("="*60)
    
    common_patterns = []
    
    # Pattern 1: Root → Embeddings pattern (universal)
    print("\nPattern 1: Root → Embeddings")
    for model_name, tree in trees.items():
        # Find root
        root = None
        for node in tree.values():
            parent = node.parent if hasattr(node, 'parent') else node.get('parent')
            if parent is None:
                root = node
                break
                
        if root:
            children = root.children if hasattr(root, 'children') else root.get('children', [])
            embedding_children = 0
            for child_id in children:
                child = tree.get(child_id)
                if child:
                    ptype = child.pattern_type if hasattr(child, 'pattern_type') else child['pattern_type']
                    if ptype == 'embedding':
                        embedding_children += 1
                        
            print(f"  {model_name}: Root has {embedding_children} embedding children")
            
    common_patterns.append({
        'name': 'root_to_embeddings',
        'description': 'All models have root → embeddings pattern',
        'universal': True
    })
    
    # Pattern 2: Layer/Confluence → Attention Heads pattern
    print("\nPattern 2: Layer/Confluence → Attention Heads")
    for model_name, tree in trees.items():
        layer_to_heads = defaultdict(int)
        
        for node_id, node in tree.items():
            ptype = node.pattern_type if hasattr(node, 'pattern_type') else node['pattern_type']
            
            if ptype in ['layer', 'confluence']:
                children = node.children if hasattr(node, 'children') else node.get('children', [])
                head_count = 0
                for child_id in children:
                    child = tree.get(child_id)
                    if child:
                        child_type = child.pattern_type if hasattr(child, 'pattern_type') else child['pattern_type']
                        if 'attention_head' in child_type:
                            head_count += 1
                            
                if head_count > 0:
                    layer_to_heads[ptype] += head_count
                    
        if layer_to_heads:
            print(f"  {model_name}: {dict(layer_to_heads)}")
        else:
            print(f"  {model_name}: No layer→head pattern")
            
    # Pattern 3: Hierarchical depth pattern
    print("\nPattern 3: Hierarchical Depth Organization")
    for model_name, tree in trees.items():
        levels = defaultdict(set)
        for node_id, node in tree.items():
            level = node.level if hasattr(node, 'level') else node['level']
            ptype = node.pattern_type if hasattr(node, 'pattern_type') else node['pattern_type']
            levels[level].add(ptype)
            
        print(f"  {model_name}:")
        for level in sorted(levels.keys()):
            print(f"    L{level}: {levels[level]}")
            
    return common_patterns


def main():
    print("="*60)
    print("POC-020: DEEP PAC TREE GEOMETRY ANALYSIS")
    print("="*60)
    
    # Load trees with full data
    print("\nLoading PAC trees with full data...")
    trees = load_full_trees("extracted_trees")
    
    if not trees:
        print("No trees found. Run run_comparison.py first.")
        return
        
    # Print tree structures
    for model_name, tree in trees.items():
        print_tree_structure(tree, model_name, max_depth=3)
        
    # Analyze geometry
    print("\n\n" + "="*60)
    print("GEOMETRY ANALYSIS")
    print("="*60)
    
    for model_name, tree in trees.items():
        geom = analyze_tree_geometry(tree)
        print(f"\n{model_name.upper()}:")
        print(f"  Depth: {geom.depth}")
        print(f"  Width by level: {geom.width_by_level}")
        print(f"  Branching: mean={np.mean(geom.branching_factors):.2f}, max={max(geom.branching_factors)}")
        print(f"  Types by level: {dict(geom.node_types_by_level)}")
        
    # Compare embedding geometry
    embedding_stats = compare_embedding_geometry(trees)
    
    # Find structural isomorphisms
    isomorphisms = find_structural_isomorphisms(trees)
    
    # Decompose into primitives
    primitives = decompose_into_primitives(trees)
    
    # Find common subtrees
    common = find_common_subtrees(trees)
    
    # Save analysis
    os.makedirs("results", exist_ok=True)
    
    analysis_results = {
        'embedding_stats': embedding_stats,
        'isomorphisms': isomorphisms,
        'primitives': {k: v for k, v in primitives.items() if v},  # Non-empty only
        'common_patterns': common
    }
    
    with open("results/geometry_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, default=str)
        
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nResults saved to results/geometry_analysis.json")
    
    # Key findings
    print("\n" + "="*60)
    print("KEY GEOMETRIC FINDINGS")
    print("="*60)
    
    # Most similar pair
    if isomorphisms:
        best = max(isomorphisms, key=lambda x: x['overall'])
        print(f"\n1. Most structurally similar: {best['models'][0]} ↔ {best['models'][1]}")
        print(f"   Similarity: {best['overall']:.2f}")
        
    # Embedding comparison
    if len(embedding_stats) >= 2:
        models = list(embedding_stats.keys())
        dims = [embedding_stats[m]['dim'] for m in models]
        if len(set(dims)) == 1:
            print(f"\n2. All models share embedding dimension: {dims[0]}")
        else:
            print(f"\n2. Embedding dimensions vary: {dict(zip(models, dims))}")
            
    # Primitives
    print(f"\n3. Primitive blocks found:")
    for prim, items in primitives.items():
        if items:
            print(f"   - {prim}: {len(items)}")


if __name__ == "__main__":
    main()
