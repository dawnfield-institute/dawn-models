"""
POC-020: PAC Tree Comparator

Compare PAC tree structures to find universal patterns across models.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
import json
import os
from collections import defaultdict
import pickle


@dataclass
class ConfluencePattern:
    """A pattern found at PAC confluence points"""
    pattern_id: str
    models: Set[str]  # Which models have this
    locations: Dict[str, List[str]]  # model -> [node_ids]
    strength: float  # How strong/consistent the pattern is
    pattern_type: str  # "universal", "architecture-specific", "complementary"
    description: str
    
    def to_dict(self) -> Dict:
        return {
            'pattern_id': self.pattern_id,
            'models': list(self.models),
            'locations': self.locations,
            'strength': self.strength,
            'pattern_type': self.pattern_type,
            'description': self.description
        }


class PACTreeComparator:
    """Compare multiple PAC trees to find patterns"""
    
    def __init__(self):
        self.patterns = []
        self.alignments = {}
        
    def load_trees_json(self, tree_dir: str) -> Dict[str, Dict]:
        """Load saved PAC trees from JSON"""
        trees = {}
        
        for filename in os.listdir(tree_dir):
            if filename.endswith('_pac_tree.json'):
                model_name = filename.replace('_pac_tree.json', '')
                with open(os.path.join(tree_dir, filename), 'r') as f:
                    trees[model_name] = json.load(f)
                print(f"  📂 Loaded {model_name}: {len(trees[model_name])} nodes")
                    
        return trees
        
    def load_trees_pickle(self, tree_dir: str) -> Dict[str, Dict]:
        """Load saved PAC trees from pickle (with full data)"""
        trees = {}
        
        for filename in os.listdir(tree_dir):
            if filename.endswith('_pac_tree.pkl'):
                model_name = filename.replace('_pac_tree.pkl', '')
                with open(os.path.join(tree_dir, filename), 'rb') as f:
                    trees[model_name] = pickle.load(f)
                print(f"  📂 Loaded {model_name}: {len(trees[model_name])} nodes")
                    
        return trees
        
    def analyze_structure(self, trees: Dict) -> Dict:
        """Analyze structural properties of each tree"""
        analysis = {}
        
        for model, tree in trees.items():
            # Count levels
            levels = defaultdict(int)
            types = defaultdict(int)
            
            for node in tree.values():
                if isinstance(node, dict):
                    levels[node['level']] += 1
                    types[node['pattern_type']] += 1
                else:
                    levels[node.level] += 1
                    types[node.pattern_type] += 1
                    
            # Calculate branching factors
            branch_counts = []
            for node in tree.values():
                children = node['children'] if isinstance(node, dict) else node.children
                if children:
                    branch_counts.append(len(children))
                    
            analysis[model] = {
                'total_nodes': len(tree),
                'levels': dict(levels),
                'types': dict(types),
                'max_depth': max(levels.keys()) if levels else 0,
                'avg_branching': np.mean(branch_counts) if branch_counts else 0,
                'max_branching': max(branch_counts) if branch_counts else 0
            }
            
        return analysis
        
    def find_universal_patterns(self, trees: Dict) -> List[ConfluencePattern]:
        """Find patterns that appear in ALL models"""
        patterns = []
        
        # 1. Hierarchical organization (all have levels)
        all_hierarchical = True
        max_depths = {}
        for model, tree in trees.items():
            depths = set()
            for node in tree.values():
                level = node['level'] if isinstance(node, dict) else node.level
                depths.add(level)
            max_depths[model] = max(depths) if depths else 0
            if max(depths) < 2:
                all_hierarchical = False
                
        if all_hierarchical:
            patterns.append(ConfluencePattern(
                pattern_id="universal_hierarchy",
                models=set(trees.keys()),
                locations={m: ["root"] for m in trees.keys()},
                strength=1.0,
                pattern_type="universal",
                description=f"All models organize into hierarchical levels (depths: {max_depths})"
            ))
            
        # 2. Embedding layer (all have token embeddings)
        all_have_embeddings = True
        embedding_counts = {}
        for model, tree in trees.items():
            emb_count = 0
            for node in tree.values():
                ptype = node['pattern_type'] if isinstance(node, dict) else node.pattern_type
                if ptype == "embedding":
                    emb_count += 1
            embedding_counts[model] = emb_count
            if emb_count == 0:
                all_have_embeddings = False
                
        if all_have_embeddings:
            patterns.append(ConfluencePattern(
                pattern_id="universal_embeddings",
                models=set(trees.keys()),
                locations={m: [f"{m}_token_*"] for m in trees.keys()},
                strength=1.0,
                pattern_type="universal",
                description=f"All models have token embeddings (counts: {embedding_counts})"
            ))
            
        # 3. Attention mechanism (all have attention patterns)
        all_have_attention = True
        attention_types = {}
        for model, tree in trees.items():
            attn_found = False
            attn_type = None
            for node in tree.values():
                ptype = node['pattern_type'] if isinstance(node, dict) else node.pattern_type
                if 'attention' in ptype.lower() or 'confluence' in ptype.lower():
                    attn_found = True
                    attn_type = ptype
                    break
            attention_types[model] = attn_type
            if not attn_found:
                all_have_attention = False
                
        if all_have_attention:
            patterns.append(ConfluencePattern(
                pattern_id="universal_attention",
                models=set(trees.keys()),
                locations={m: [f"{m}_*attention*"] for m in trees.keys()},
                strength=0.9,
                pattern_type="universal",
                description=f"All models have attention mechanisms (types: {attention_types})"
            ))
            
        return patterns
        
    def find_architecture_patterns(self, trees: Dict) -> List[ConfluencePattern]:
        """Find patterns specific to each architecture"""
        patterns = []
        
        for model, tree in trees.items():
            # Find unique pattern types
            types_in_model = set()
            for node in tree.values():
                ptype = node['pattern_type'] if isinstance(node, dict) else node.pattern_type
                types_in_model.add(ptype)
                
            # Check if any type is unique to this model
            for ptype in types_in_model:
                unique = True
                for other_model, other_tree in trees.items():
                    if other_model == model:
                        continue
                    for node in other_tree.values():
                        other_ptype = node['pattern_type'] if isinstance(node, dict) else node.pattern_type
                        if other_ptype == ptype:
                            unique = False
                            break
                    if not unique:
                        break
                        
                if unique and ptype not in ['root', 'embedding']:
                    patterns.append(ConfluencePattern(
                        pattern_id=f"{model}_{ptype}",
                        models={model},
                        locations={model: [f"*{ptype}*"]},
                        strength=0.8,
                        pattern_type="architecture-specific",
                        description=f"{model} uniquely has '{ptype}' pattern type"
                    ))
                    
        return patterns
        
    def find_complementary_patterns(self, trees: Dict) -> List[Tuple[str, str, str]]:
        """Find what one model has that another lacks"""
        complementary = []
        
        # Get pattern types per model
        pattern_types = {}
        for model, tree in trees.items():
            types = set()
            for node in tree.values():
                ptype = node['pattern_type'] if isinstance(node, dict) else node.pattern_type
                types.add(ptype)
            pattern_types[model] = types
            
        # Get metadata per model
        metadata_keys = {}
        for model, tree in trees.items():
            keys = set()
            for node in tree.values():
                meta = node.get('metadata', {}) if isinstance(node, dict) else getattr(node, 'metadata', {})
                keys.update(meta.keys())
            metadata_keys[model] = keys
            
        # Find complementary pairs
        models = list(trees.keys())
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                # Pattern types unique to each
                unique_to_1 = pattern_types[model1] - pattern_types[model2]
                unique_to_2 = pattern_types[model2] - pattern_types[model1]
                
                for ptype in unique_to_1:
                    if ptype not in ['root']:
                        complementary.append((
                            model1, model2, 
                            f"{model1} has '{ptype}' that {model2} lacks"
                        ))
                        
                for ptype in unique_to_2:
                    if ptype not in ['root']:
                        complementary.append((
                            model2, model1,
                            f"{model2} has '{ptype}' that {model1} lacks"
                        ))
                        
                # Metadata unique to each
                unique_meta_1 = metadata_keys[model1] - metadata_keys[model2]
                unique_meta_2 = metadata_keys[model2] - metadata_keys[model1]
                
                if unique_meta_1:
                    complementary.append((
                        model1, model2,
                        f"{model1} tracks metadata {unique_meta_1} that {model2} doesn't"
                    ))
                    
                if unique_meta_2:
                    complementary.append((
                        model2, model1,
                        f"{model2} tracks metadata {unique_meta_2} that {model1} doesn't"
                    ))
                    
        return complementary
        
    def build_unified_tree(self, trees: Dict) -> Dict:
        """Build a unified meta-PAC tree combining all models"""
        unified = {}
        
        # Create unified root
        unified['meta_root'] = {
            'id': 'meta_root',
            'level': 0,
            'pattern_type': 'meta',
            'children': [],
            'parent': None,
            'metadata': {
                'source_models': list(trees.keys()),
                'total_source_nodes': sum(len(t) for t in trees.values())
            }
        }
        
        # Create model branches
        for model in trees.keys():
            model_branch = {
                'id': f'branch_{model}',
                'level': 1,
                'pattern_type': 'model_branch',
                'children': [],
                'parent': 'meta_root',
                'metadata': {'source_model': model}
            }
            unified[model_branch['id']] = model_branch
            unified['meta_root']['children'].append(model_branch['id'])
            
        # Group nodes by pattern type across models
        by_pattern = defaultdict(list)
        for model, tree in trees.items():
            for node_id, node in tree.items():
                ptype = node['pattern_type'] if isinstance(node, dict) else node.pattern_type
                level = node['level'] if isinstance(node, dict) else node.level
                key = (ptype, level)
                by_pattern[key].append((model, node_id))
                
        # Create unified pattern nodes
        for (pattern_type, level), sources in by_pattern.items():
            if pattern_type == 'root':
                continue
                
            unified_id = f"unified_{pattern_type}_{level}"
            unified[unified_id] = {
                'id': unified_id,
                'level': level + 1,  # Offset by 1 due to meta_root
                'pattern_type': f'unified_{pattern_type}',
                'children': [],
                'parent': 'meta_root',
                'metadata': {
                    'source_nodes': sources,
                    'model_count': len(set(m for m, _ in sources)),
                    'is_universal': len(set(m for m, _ in sources)) == len(trees)
                }
            }
            unified['meta_root']['children'].append(unified_id)
            
        return unified
        
    def compare_embeddings(self, trees: Dict) -> Dict:
        """Compare embedding patterns across models"""
        # This requires pickle data with actual embeddings
        embedding_comparison = {}
        
        for model, tree in trees.items():
            embeddings = []
            for node in tree.values():
                if hasattr(node, 'pattern_type') and node.pattern_type == 'embedding':
                    if node.data is not None and isinstance(node.data, np.ndarray):
                        embeddings.append(node.data)
                        
            if embeddings:
                embeddings = np.array(embeddings[:100])  # Sample 100
                embedding_comparison[model] = {
                    'mean': float(embeddings.mean()),
                    'std': float(embeddings.std()),
                    'norm_mean': float(np.linalg.norm(embeddings, axis=1).mean()),
                    'shape': embeddings.shape
                }
                
        # Compare across models
        if len(embedding_comparison) > 1:
            models = list(embedding_comparison.keys())
            for i, m1 in enumerate(models):
                for m2 in models[i+1:]:
                    norm_diff = abs(embedding_comparison[m1]['norm_mean'] - 
                                   embedding_comparison[m2]['norm_mean'])
                    embedding_comparison[f'{m1}_vs_{m2}'] = {
                        'norm_difference': norm_diff,
                        'similar': norm_diff < 1.0
                    }
                    
        return embedding_comparison


def generate_report(analysis: Dict, universal: List, arch_specific: List, 
                   complementary: List, unified: Dict) -> str:
    """Generate analysis report"""
    lines = []
    lines.append("=" * 70)
    lines.append("PAC TREE COMPARISON REPORT")
    lines.append("=" * 70)
    
    # Tree statistics
    lines.append("\n📊 TREE STATISTICS:")
    for model, stats in analysis.items():
        lines.append(f"\n  {model.upper()}:")
        lines.append(f"    Total nodes: {stats['total_nodes']}")
        lines.append(f"    Max depth: {stats['max_depth']}")
        lines.append(f"    Avg branching: {stats['avg_branching']:.2f}")
        lines.append(f"    Pattern types: {list(stats['types'].keys())}")
        
    # Universal patterns
    lines.append("\n\n🌍 UNIVERSAL PATTERNS (found in ALL models):")
    for pattern in universal:
        lines.append(f"\n  ✓ {pattern.pattern_id}")
        lines.append(f"    {pattern.description}")
        lines.append(f"    Strength: {pattern.strength:.2f}")
        
    # Architecture-specific patterns
    lines.append("\n\n🏗️ ARCHITECTURE-SPECIFIC PATTERNS:")
    for pattern in arch_specific:
        lines.append(f"\n  • {pattern.pattern_id}")
        lines.append(f"    {pattern.description}")
        
    # Complementary relationships
    lines.append("\n\n🔄 COMPLEMENTARY PATTERNS:")
    for m1, m2, desc in complementary[:10]:  # First 10
        lines.append(f"  • {desc}")
        
    # Unified tree stats
    lines.append(f"\n\n🌳 UNIFIED META-PAC TREE:")
    lines.append(f"  Total unified nodes: {len(unified)}")
    
    universal_nodes = sum(1 for n in unified.values() 
                         if isinstance(n, dict) and n.get('metadata', {}).get('is_universal'))
    lines.append(f"  Universal pattern nodes: {universal_nodes}")
    
    # Key insights
    lines.append("\n\n💡 KEY INSIGHTS:")
    
    if len(universal) >= 3:
        lines.append("  ✓ Found strong universal patterns across all architectures")
        
    if len(complementary) > 0:
        lines.append("  ✓ Models have complementary knowledge that could be combined")
        
    # Check for confluence
    has_confluence = any('confluence' in p.pattern_id for p in arch_specific)
    if has_confluence:
        lines.append("  ✓ Some models have explicit confluence points (BERT)")
        
    lines.append("\n" + "=" * 70)
    
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print("POC-020: PAC TREE COMPARISON")
    print("=" * 70)
    
    comparator = PACTreeComparator()
    
    # Load trees
    print("\n📂 Loading PAC trees...")
    trees = comparator.load_trees_json("extracted_trees")
    
    if not trees:
        print("No trees found. Run multi_model_extractor.py first.")
        exit(1)
        
    # Analyze structure
    print("\n📊 Analyzing tree structures...")
    analysis = comparator.analyze_structure(trees)
    
    # Find patterns
    print("\n🔍 Finding universal patterns...")
    universal = comparator.find_universal_patterns(trees)
    
    print("\n🔍 Finding architecture-specific patterns...")
    arch_specific = comparator.find_architecture_patterns(trees)
    
    print("\n🔍 Finding complementary patterns...")
    complementary = comparator.find_complementary_patterns(trees)
    
    # Build unified tree
    print("\n🌳 Building unified meta-PAC tree...")
    unified = comparator.build_unified_tree(trees)
    
    # Generate report
    report = generate_report(analysis, universal, arch_specific, complementary, unified)
    print(report)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    
    with open("results/comparison_report.txt", 'w') as f:
        f.write(report)
        
    with open("results/unified_pac_tree.json", 'w') as f:
        json.dump(unified, f, indent=2)
        
    patterns_data = {
        'universal': [p.to_dict() for p in universal],
        'architecture_specific': [p.to_dict() for p in arch_specific],
        'complementary': complementary
    }
    with open("results/patterns.json", 'w') as f:
        json.dump(patterns_data, f, indent=2)
        
    print("\n💾 Results saved to results/")
