"""
POC-020: Effective Tree Decomposition

Based on geometry analysis findings:
- BERT ↔ GPT-2 are structurally identical (1.00 similarity)
- BERT embeddings are tightly clustered (cos_sim ~0.99)
- GPT-2/Pythia embeddings are more spread

This script decomposes trees into transferable units.
"""

import numpy as np
import pickle
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class TransferableUnit:
    """A unit that can be transferred between models"""
    unit_id: str
    unit_type: str  # 'embedding_cluster', 'attention_pattern', 'confluence_point', etc
    source_model: str
    level: int
    pattern_data: np.ndarray  # The actual transferable pattern
    metadata: Dict = field(default_factory=dict)
    compatible_with: List[str] = field(default_factory=list)
    

def load_trees(tree_dir: str) -> Dict:
    """Load trees with full data"""
    trees = {}
    for filename in os.listdir(tree_dir):
        if filename.endswith('_pac_tree.pkl'):
            model_name = filename.replace('_pac_tree.pkl', '')
            with open(os.path.join(tree_dir, filename), 'rb') as f:
                trees[model_name] = pickle.load(f)
    return trees


def extract_embedding_clusters(tree: Dict, model_name: str, n_clusters: int = 10) -> List[TransferableUnit]:
    """Extract embedding clusters as transferable units"""
    
    # Collect all embeddings
    embeddings = []
    token_ids = []
    
    for node_id, node in tree.items():
        ptype = node.pattern_type if hasattr(node, 'pattern_type') else node['pattern_type']
        if ptype == 'embedding':
            data = node.data if hasattr(node, 'data') else node.get('data')
            if data is not None and isinstance(data, np.ndarray):
                embeddings.append(data)
                meta = node.metadata if hasattr(node, 'metadata') else node.get('metadata', {})
                token_ids.append(meta.get('token_id', len(embeddings)))
                
    if not embeddings:
        return []
        
    embeddings = np.array(embeddings)
    print(f"  {model_name}: {len(embeddings)} embeddings of dim {embeddings.shape[1]}")
    
    # Simple k-means clustering
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    units = []
    for cluster_id in range(kmeans.n_clusters):
        mask = cluster_labels == cluster_id
        cluster_center = kmeans.cluster_centers_[cluster_id]
        cluster_size = mask.sum()
        
        # Get sample tokens in this cluster
        cluster_tokens = [token_ids[i] for i in np.where(mask)[0][:5]]
        
        unit = TransferableUnit(
            unit_id=f"{model_name}_emb_cluster_{cluster_id}",
            unit_type="embedding_cluster",
            source_model=model_name,
            level=1,
            pattern_data=cluster_center,
            metadata={
                'cluster_size': int(cluster_size),
                'sample_tokens': cluster_tokens,
                'inertia': float(np.sum((embeddings[mask] - cluster_center) ** 2))
            }
        )
        units.append(unit)
        
    return units


def extract_attention_patterns(tree: Dict, model_name: str) -> List[TransferableUnit]:
    """Extract attention patterns as transferable units"""
    
    units = []
    
    # Find attention-related nodes
    for node_id, node in tree.items():
        ptype = node.pattern_type if hasattr(node, 'pattern_type') else node['pattern_type']
        
        if ptype in ['confluence', 'layer']:
            meta = node.metadata if hasattr(node, 'metadata') else node.get('metadata', {})
            data = node.data if hasattr(node, 'data') else node.get('data', {})
            
            # Create pattern from available data
            if isinstance(data, dict):
                # Extract numeric values
                pattern_values = []
                for k, v in data.items():
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        pattern_values.append(v)
                        
                if pattern_values:
                    unit = TransferableUnit(
                        unit_id=f"{model_name}_{ptype}_{meta.get('layer', 0)}",
                        unit_type=f"{ptype}_pattern",
                        source_model=model_name,
                        level=2,
                        pattern_data=np.array(pattern_values),
                        metadata={
                            'layer': meta.get('layer'),
                            'type': meta.get('type'),
                            'original_data_keys': list(data.keys())
                        }
                    )
                    units.append(unit)
                    
        elif ptype == 'attention_head':
            meta = node.metadata if hasattr(node, 'metadata') else node.get('metadata', {})
            data = node.data if hasattr(node, 'data') else node.get('data')
            
            if data is not None and isinstance(data, np.ndarray):
                unit = TransferableUnit(
                    unit_id=f"{model_name}_head_L{meta.get('layer', 0)}_H{meta.get('head', 0)}",
                    unit_type="attention_head",
                    source_model=model_name,
                    level=3,
                    pattern_data=data,
                    metadata={
                        'layer': meta.get('layer'),
                        'head': meta.get('head')
                    }
                )
                units.append(unit)
                
    return units


def extract_scaling_patterns(tree: Dict, model_name: str) -> List[TransferableUnit]:
    """Extract scaling patterns (Pythia-specific) as transferable units"""
    
    units = []
    scaling_data = []
    
    for node_id, node in tree.items():
        ptype = node.pattern_type if hasattr(node, 'pattern_type') else node['pattern_type']
        
        if ptype == 'scaling':
            data = node.data if hasattr(node, 'data') else node.get('data', {})
            if isinstance(data, dict):
                scaling_data.append({
                    'layer': data.get('layer', 0),
                    'weight_norm': data.get('weight_norm', 0),
                    'relative_scale': data.get('relative_scale', 1.0)
                })
                
    if scaling_data:
        # Create scaling curve as transferable pattern
        layers = [d['layer'] for d in scaling_data]
        norms = [d['weight_norm'] for d in scaling_data]
        scales = [d['relative_scale'] for d in scaling_data]
        
        unit = TransferableUnit(
            unit_id=f"{model_name}_scaling_curve",
            unit_type="scaling_curve",
            source_model=model_name,
            level=2,
            pattern_data=np.array([layers, norms, scales]),
            metadata={
                'n_layers': len(layers),
                'norm_trend': np.polyfit(layers, norms, 1).tolist() if len(layers) > 1 else [0, 0],
                'scale_trend': np.polyfit(layers, scales, 1).tolist() if len(layers) > 1 else [0, 1]
            }
        )
        units.append(unit)
        
    return units


def find_compatible_units(units: List[TransferableUnit]) -> None:
    """Find which units are compatible across models"""
    
    # Group by type
    by_type = defaultdict(list)
    for unit in units:
        by_type[unit.unit_type].append(unit)
        
    # Check compatibility within each type
    for unit_type, type_units in by_type.items():
        if len(type_units) < 2:
            continue
            
        # Compare units of same type
        for i, u1 in enumerate(type_units):
            for u2 in type_units[i+1:]:
                if u1.source_model == u2.source_model:
                    continue
                    
                # Check dimension compatibility
                if u1.pattern_data.shape == u2.pattern_data.shape:
                    # Compute similarity
                    p1 = u1.pattern_data.flatten()
                    p2 = u2.pattern_data.flatten()
                    
                    norm1 = np.linalg.norm(p1)
                    norm2 = np.linalg.norm(p2)
                    
                    if norm1 > 0 and norm2 > 0:
                        cos_sim = np.dot(p1, p2) / (norm1 * norm2)
                        
                        if cos_sim > 0.5:  # Threshold for compatibility
                            u1.compatible_with.append(u2.unit_id)
                            u2.compatible_with.append(u1.unit_id)


def compute_cross_model_alignment(units: List[TransferableUnit]) -> Dict:
    """Compute alignment between models based on transferable units"""
    
    # Group by model and type
    by_model_type = defaultdict(lambda: defaultdict(list))
    for unit in units:
        by_model_type[unit.source_model][unit.unit_type].append(unit)
        
    alignments = {}
    models = list(by_model_type.keys())
    
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            key = f"{m1}_vs_{m2}"
            alignments[key] = {
                'compatible_units': 0,
                'total_compared': 0,
                'type_alignments': {}
            }
            
            # Compare each type
            common_types = set(by_model_type[m1].keys()) & set(by_model_type[m2].keys())
            
            for unit_type in common_types:
                units1 = by_model_type[m1][unit_type]
                units2 = by_model_type[m2][unit_type]
                
                compatible = 0
                total = 0
                
                for u1 in units1:
                    for u2 in units2:
                        if u1.pattern_data.shape == u2.pattern_data.shape:
                            total += 1
                            p1 = u1.pattern_data.flatten()
                            p2 = u2.pattern_data.flatten()
                            
                            norm1 = np.linalg.norm(p1)
                            norm2 = np.linalg.norm(p2)
                            
                            if norm1 > 0 and norm2 > 0:
                                cos_sim = np.dot(p1, p2) / (norm1 * norm2)
                                if cos_sim > 0.3:
                                    compatible += 1
                                    
                alignments[key]['type_alignments'][unit_type] = {
                    'compatible': compatible,
                    'total': total,
                    'rate': compatible / total if total > 0 else 0
                }
                alignments[key]['compatible_units'] += compatible
                alignments[key]['total_compared'] += total
                
            if alignments[key]['total_compared'] > 0:
                alignments[key]['overall_rate'] = (
                    alignments[key]['compatible_units'] / alignments[key]['total_compared']
                )
            else:
                alignments[key]['overall_rate'] = 0
                
    return alignments


def main():
    print("="*60)
    print("POC-020: EFFECTIVE TREE DECOMPOSITION")
    print("="*60)
    
    # Load trees
    print("\nLoading PAC trees...")
    trees = load_trees("extracted_trees")
    print(f"Loaded {len(trees)} trees")
    
    # Extract all transferable units
    all_units = []
    
    print("\n" + "="*60)
    print("EXTRACTING TRANSFERABLE UNITS")
    print("="*60)
    
    for model_name, tree in trees.items():
        print(f"\n{model_name.upper()}:")
        
        # Embedding clusters
        print("  Extracting embedding clusters...")
        emb_units = extract_embedding_clusters(tree, model_name, n_clusters=20)
        print(f"    → {len(emb_units)} clusters")
        all_units.extend(emb_units)
        
        # Attention patterns
        print("  Extracting attention patterns...")
        attn_units = extract_attention_patterns(tree, model_name)
        print(f"    → {len(attn_units)} patterns")
        all_units.extend(attn_units)
        
        # Scaling patterns
        print("  Extracting scaling patterns...")
        scale_units = extract_scaling_patterns(tree, model_name)
        print(f"    → {len(scale_units)} patterns")
        all_units.extend(scale_units)
        
    print(f"\nTotal transferable units: {len(all_units)}")
    
    # Find compatible units
    print("\n" + "="*60)
    print("FINDING COMPATIBLE UNITS")
    print("="*60)
    
    find_compatible_units(all_units)
    
    # Count compatibilities
    compatible_count = sum(1 for u in all_units if u.compatible_with)
    print(f"\nUnits with cross-model compatibility: {compatible_count}/{len(all_units)}")
    
    # Show examples
    print("\nExample compatible units:")
    for unit in all_units[:20]:
        if unit.compatible_with:
            print(f"  {unit.unit_id}")
            print(f"    Compatible with: {unit.compatible_with[:3]}")
            
    # Compute alignments
    print("\n" + "="*60)
    print("CROSS-MODEL ALIGNMENT")
    print("="*60)
    
    alignments = compute_cross_model_alignment(all_units)
    
    for pair, data in alignments.items():
        print(f"\n{pair}:")
        print(f"  Overall alignment: {data['overall_rate']:.2%}")
        for unit_type, type_data in data['type_alignments'].items():
            if type_data['total'] > 0:
                print(f"  - {unit_type}: {type_data['rate']:.2%} ({type_data['compatible']}/{type_data['total']})")
                
    # Summarize by unit type
    print("\n" + "="*60)
    print("UNIT TYPE SUMMARY")
    print("="*60)
    
    by_type = defaultdict(list)
    for unit in all_units:
        by_type[unit.unit_type].append(unit)
        
    for unit_type, units in by_type.items():
        models = set(u.source_model for u in units)
        compatible = sum(1 for u in units if u.compatible_with)
        
        print(f"\n{unit_type}:")
        print(f"  Total: {len(units)}")
        print(f"  Models: {models}")
        print(f"  Cross-compatible: {compatible}")
        
        # Show sample pattern shapes
        shapes = set(u.pattern_data.shape for u in units)
        print(f"  Pattern shapes: {shapes}")
        
    # Save results
    os.makedirs("results", exist_ok=True)
    
    # Convert units to serializable format
    units_data = []
    for unit in all_units:
        units_data.append({
            'unit_id': unit.unit_id,
            'unit_type': unit.unit_type,
            'source_model': unit.source_model,
            'level': unit.level,
            'pattern_shape': list(unit.pattern_data.shape),
            'pattern_mean': float(unit.pattern_data.mean()),
            'pattern_std': float(unit.pattern_data.std()),
            'metadata': unit.metadata,
            'compatible_with': unit.compatible_with
        })
        
    with open("results/transferable_units.json", 'w', encoding='utf-8') as f:
        json.dump({
            'units': units_data,
            'alignments': alignments,
            'summary': {
                'total_units': len(all_units),
                'compatible_units': compatible_count,
                'unit_types': list(by_type.keys())
            }
        }, f, indent=2)
        
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    # Best alignment pair
    if alignments:
        best_pair = max(alignments.items(), key=lambda x: x[1]['overall_rate'])
        print(f"\n1. Best cross-model alignment: {best_pair[0]}")
        print(f"   Rate: {best_pair[1]['overall_rate']:.2%}")
        
    # Most transferable unit type
    type_compatibility = {}
    for unit_type, units in by_type.items():
        compat = sum(1 for u in units if u.compatible_with)
        type_compatibility[unit_type] = compat / len(units) if units else 0
        
    if type_compatibility:
        best_type = max(type_compatibility.items(), key=lambda x: x[1])
        print(f"\n2. Most transferable unit type: {best_type[0]}")
        print(f"   Compatibility rate: {best_type[1]:.2%}")
        
    print(f"\n3. Total transferable units extracted: {len(all_units)}")
    print(f"\n💾 Results saved to results/transferable_units.json")


if __name__ == "__main__":
    main()
