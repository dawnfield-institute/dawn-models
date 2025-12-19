"""
POC-020: Proper PAC Tree Extraction Using Fracton PAC System

The previous approach was wrong - we were reinventing PAC trees.
This uses the ACTUAL fracton.core.pac_system which is:
1. Dimension-agnostic (delta-only storage)
2. Scale-invariant (PAC conservation regardless of tensor size)
3. Properly integrated with the Dawn Field ecosystem

A PAC tree is just a PAC tree - the underlying dimensions don't matter
because we store DELTAS, not absolute values.
"""

import sys
import os
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\fracton")

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import pickle

# Use the ACTUAL fracton PAC system
from fracton.core.pac_system import PACSystem
from fracton.core.pac_node import PACNode

# For model loading
from transformers import AutoModel, AutoTokenizer


@dataclass  
class ModelPACMapping:
    """Maps model components to PAC node IDs"""
    model_name: str
    root_id: int
    component_map: Dict[str, int]  # component_name -> node_id
    metadata: Dict


class ModelToPACExtractor:
    """
    Extracts neural network models into proper PAC trees using fracton.
    
    Key insight: A PAC tree is dimension-agnostic because:
    - Each node stores DELTA from parent, not absolute value
    - Reconstruction = sum of deltas from root
    - Conservation: parent = Σ(children deltas) + residual
    
    This means a 768-dim embedding and 512-dim embedding are BOTH
    just PAC nodes with different delta shapes - the tree structure
    is what matters, not the dimensions.
    """
    
    def __init__(self, device: str = 'auto'):
        device_str = 'cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu'
        self.device = device_str
        
        # Create a shared PAC system for all models
        # This is crucial - all models live in the SAME PAC space
        self.pac_system = PACSystem(
            device=device_str,
            hot_cache_size=50000,
            warm_cache_size=200000
        )
        
        # Track model mappings
        self.model_mappings: Dict[str, ModelPACMapping] = {}
        
        print(f"✓ PAC System initialized on {device_str}")
        
    def extract_model(self, model_name: str, sample_tokens: int = 100) -> ModelPACMapping:
        """
        Extract a model into the shared PAC tree.
        
        The key insight: we inject model components as deltas from
        a common root, making them comparable regardless of dimension.
        
        Structure:
          root (model signature)
            ├── embedding_hub (embedding dimension)
            │     └── token embeddings
            ├── layer_hub (layer signature dimension)
            │     └── layers
            └── attention_hub (attention signature dimension)
                  └── attention patterns
        """
        print(f"\n{'='*60}")
        print(f"Extracting: {model_name}")
        print(f"{'='*60}")
        
        # Load model
        model = AutoModel.from_pretrained(model_name)
        model.to(self.device)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get embedding dimension for this model
        embed_dim = self._get_embedding_dim(model)
        
        # Create root node for this model
        # Root delta is a scalar "signature" of the model
        model_signature = self._compute_model_signature(model)
        root_id = self.pac_system.inject(
            model_signature,
            label=f"model:{model_name}",
            importance=1.0  # Models are high importance
        )
        
        print(f"  Created root node: {root_id}")
        
        component_map = {}
        component_map['root'] = root_id
        
        # Create hub nodes for each component type
        # This allows different dimensions under the same root
        
        # 1. Embedding hub - dimension matches embeddings
        embedding_hub = torch.zeros(embed_dim, device=self.device)
        embedding_hub_id = self.pac_system.inject(
            embedding_hub,
            label=f"hub:embeddings:{model_name}",
            importance=0.8
        )
        component_map['embedding_hub'] = embedding_hub_id
        
        # 2. Layer hub - 256-dim signatures
        layer_hub = torch.zeros(256, device=self.device)
        layer_hub_id = self.pac_system.inject(
            layer_hub,
            label=f"hub:layers:{model_name}",
            importance=0.8
        )
        component_map['layer_hub'] = layer_hub_id
        
        # 3. Attention hub - 256-dim signatures
        attention_hub = torch.zeros(256, device=self.device)
        attention_hub_id = self.pac_system.inject(
            attention_hub,
            label=f"hub:attention:{model_name}",
            importance=0.8
        )
        component_map['attention_hub'] = attention_hub_id
        
        # Now extract components under their respective hubs
        
        # 1. Extract embeddings as children of embedding hub
        print(f"\n  Extracting embeddings...")
        embedding_ids = self._extract_embeddings(model, tokenizer, embedding_hub_id, sample_tokens)
        component_map['embeddings'] = embedding_ids
        print(f"    → {len(embedding_ids)} embedding nodes")
        
        # 2. Extract layer structure
        print(f"\n  Extracting layers...")
        layer_ids = self._extract_layers(model, layer_hub_id)
        component_map['layers'] = layer_ids
        print(f"    → {len(layer_ids)} layer nodes")
        
        # 3. Extract attention patterns (if available)
        print(f"\n  Extracting attention patterns...")
        attention_ids = self._extract_attention(model, attention_hub_id)
        component_map['attention'] = attention_ids
        print(f"    → {len(attention_ids)} attention nodes")
        
        # Create mapping
        mapping = ModelPACMapping(
            model_name=model_name,
            root_id=root_id,
            component_map=component_map,
            metadata={
                'device': self.device,
                'sample_tokens': sample_tokens,
                'total_nodes': len(embedding_ids) + len(layer_ids) + len(attention_ids) + 1,
                'embedding_dim': self._get_embedding_dim(model)
            }
        )
        
        self.model_mappings[model_name] = mapping
        
        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return mapping
    
    def _compute_model_signature(self, model) -> torch.Tensor:
        """
        Compute a dimension-agnostic signature for the model.
        This becomes the root delta.
        
        The signature captures the model's "identity" in a fixed size,
        regardless of the model's internal dimensions.
        """
        # Aggregate statistics from all parameters
        signature_parts = []
        
        for name, param in model.named_parameters():
            if param.numel() > 0:
                flat = param.flatten().float()
                # Compute statistics
                stats = torch.tensor([
                    flat.mean(),
                    flat.std(),
                    flat.min(),
                    flat.max(),
                    float(param.numel()),
                    float(len(param.shape))
                ], device=self.device)
                signature_parts.append(stats)
        
        # Stack and reduce to fixed size signature (256-dim)
        all_stats = torch.stack(signature_parts)  # [n_params, 6]
        
        # Reduce to 256-dim via learned compression
        # For now, use simple stats
        signature = torch.cat([
            all_stats.mean(dim=0),    # 6
            all_stats.std(dim=0),     # 6  
            all_stats.min(dim=0)[0],  # 6
            all_stats.max(dim=0)[0],  # 6
        ])  # 24 total
        
        # Pad to 256 for consistency
        if len(signature) < 256:
            signature = torch.cat([
                signature,
                torch.zeros(256 - len(signature), device=self.device)
            ])
            
        return signature[:256]
    
    def _get_embedding_dim(self, model) -> int:
        """Get embedding dimension from model"""
        if hasattr(model, 'config'):
            if hasattr(model.config, 'hidden_size'):
                return model.config.hidden_size
            if hasattr(model.config, 'd_model'):
                return model.config.d_model
        return 0
    
    def _extract_embeddings(self, model, tokenizer, parent_id: int, 
                            sample_tokens: int) -> List[int]:
        """
        Extract token embeddings as PAC children.
        
        Each embedding is injected as a delta from the parent,
        regardless of its dimension.
        """
        node_ids = []
        
        # Get embedding layer
        if hasattr(model, 'embeddings'):
            embed_weight = model.embeddings.word_embeddings.weight
        elif hasattr(model, 'wte'):
            embed_weight = model.wte.weight
        elif hasattr(model, 'embed_tokens'):
            embed_weight = model.embed_tokens.weight
        elif hasattr(model, 'gpt_neox'):
            if hasattr(model.gpt_neox, 'embed_in'):
                embed_weight = model.gpt_neox.embed_in.weight
            elif hasattr(model.gpt_neox, 'wte'):
                embed_weight = model.gpt_neox.wte.weight
            else:
                # Try to find embedding in children
                for name, module in model.gpt_neox.named_children():
                    if 'embed' in name.lower():
                        embed_weight = module.weight
                        break
                else:
                    print("    ⚠ Could not find embedding in gpt_neox")
                    return []
        else:
            # Generic search for embedding
            for name, module in model.named_modules():
                if 'embed' in name.lower() and hasattr(module, 'weight'):
                    if len(module.weight.shape) == 2:  # [vocab, dim]
                        embed_weight = module.weight
                        print(f"    Found embedding at: {name}")
                        break
            else:
                print("    ⚠ Could not find embedding layer")
                return []
        
        embed_weight = embed_weight.detach()
        vocab_size = embed_weight.shape[0]
        
        # Sample tokens
        sample_ids = np.random.choice(vocab_size, min(sample_tokens, vocab_size), replace=False)
        
        for token_id in sample_ids:
            embedding = embed_weight[token_id]
            
            # Get token text for label
            try:
                token_text = tokenizer.decode([token_id])
            except:
                token_text = f"token_{token_id}"
            
            # Inject into PAC system as child of parent
            # The key: we inject the FULL embedding, and the system
            # stores it as a DELTA from parent
            node_id = self.pac_system.inject(
                embedding,
                parent_id=parent_id,
                label=f"emb:{token_text}",
                importance=0.3
            )
            node_ids.append(node_id)
            
        return node_ids
    
    def _extract_layers(self, model, parent_id: int) -> List[int]:
        """
        Extract layer representations as PAC children.
        
        Each layer is reduced to a signature and injected.
        """
        node_ids = []
        
        # Find encoder/decoder layers
        layers = None
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            layers = model.encoder.layer
            layer_type = 'encoder'
        elif hasattr(model, 'h'):  # GPT-2 style
            layers = model.h
            layer_type = 'transformer'
        elif hasattr(model, 'layers'):
            layers = model.layers
            layer_type = 'layers'
        elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
            layers = model.gpt_neox.layers
            layer_type = 'neox'
            
        if layers is None:
            print("    ⚠ Could not find layers")
            return []
            
        for i, layer in enumerate(layers):
            # Compute layer signature (dimension-agnostic)
            layer_sig = self._compute_layer_signature(layer)
            
            node_id = self.pac_system.inject(
                layer_sig,
                parent_id=parent_id,
                label=f"layer:{layer_type}_{i}",
                importance=0.5
            )
            node_ids.append(node_id)
            
        return node_ids
    
    def _compute_layer_signature(self, layer) -> torch.Tensor:
        """Compute dimension-agnostic signature for a layer"""
        stats = []
        
        for name, param in layer.named_parameters():
            if param.numel() > 0:
                flat = param.flatten().float()
                stats.extend([
                    flat.mean().item(),
                    flat.std().item(),
                    flat.norm().item() / np.sqrt(param.numel())
                ])
        
        sig = torch.tensor(stats[:256], device=self.device)
        
        # Pad to 256
        if len(sig) < 256:
            sig = torch.cat([sig, torch.zeros(256 - len(sig), device=self.device)])
            
        return sig[:256]
    
    def _extract_attention(self, model, parent_id: int) -> List[int]:
        """
        Extract attention patterns as PAC children.
        """
        node_ids = []
        
        # Find attention layers
        attention_modules = []
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                if hasattr(module, 'num_attention_heads') or hasattr(module, 'num_heads'):
                    attention_modules.append((name, module))
        
        for name, attn in attention_modules[:12]:  # Limit to 12
            attn_sig = self._compute_attention_signature(attn)
            
            node_id = self.pac_system.inject(
                attn_sig,
                parent_id=parent_id,
                label=f"attn:{name}",
                importance=0.4
            )
            node_ids.append(node_id)
            
        return node_ids
    
    def _compute_attention_signature(self, attn_module) -> torch.Tensor:
        """Compute signature for attention module"""
        stats = []
        
        for name, param in attn_module.named_parameters():
            if param.numel() > 0:
                flat = param.flatten().float()
                stats.extend([
                    flat.mean().item(),
                    flat.std().item()
                ])
        
        sig = torch.tensor(stats[:256], device=self.device)
        
        if len(sig) < 256:
            sig = torch.cat([sig, torch.zeros(256 - len(sig), device=self.device)])
            
        return sig[:256]
    
    def compare_models_in_pac_space(self) -> Dict:
        """
        Compare models by their PAC tree structure.
        
        Since all models are in the SAME PAC system, we can:
        1. Find resonant nodes across models
        2. Compare tree structures
        3. Identify shared knowledge
        """
        print("\n" + "="*60)
        print("COMPARING MODELS IN UNIFIED PAC SPACE")
        print("="*60)
        
        results = {
            'model_pairs': {},
            'cross_model_resonance': []
        }
        
        models = list(self.model_mappings.keys())
        
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                pair_key = f"{m1}_vs_{m2}"
                print(f"\n  {m1} ↔ {m2}")
                
                map1 = self.model_mappings[m1]
                map2 = self.model_mappings[m2]
                
                # Compare root signatures
                root1 = self.pac_system.reconstruct(map1.root_id)
                root2 = self.pac_system.reconstruct(map2.root_id)
                root_sim = self._cosine_similarity(root1, root2)
                print(f"    Root similarity: {root_sim:.4f}")
                
                # Find cross-model resonance
                resonances = self._find_cross_resonance(map1, map2)
                
                results['model_pairs'][pair_key] = {
                    'root_similarity': float(root_sim),
                    'embedding_resonance': resonances['embeddings'],
                    'layer_resonance': resonances['layers'],
                    'attention_resonance': resonances['attention']
                }
                
                print(f"    Embedding resonance: {resonances['embeddings']['avg_score']:.4f}")
                print(f"    Layer resonance: {resonances['layers']['avg_score']:.4f}")
                
        return results
    
    def _find_cross_resonance(self, map1: ModelPACMapping, 
                               map2: ModelPACMapping) -> Dict:
        """Find resonant nodes between two model mappings"""
        results = {}
        
        for component in ['embeddings', 'layers', 'attention']:
            ids1 = map1.component_map.get(component, [])
            ids2 = map2.component_map.get(component, [])
            
            if not ids1 or not ids2:
                results[component] = {'avg_score': 0.0, 'pairs': []}
                continue
            
            # Sample for efficiency
            sample1 = ids1[:min(20, len(ids1))]
            sample2 = ids2[:min(20, len(ids2))]
            
            similarities = []
            best_pairs = []
            
            for id1 in sample1:
                v1 = self.pac_system.reconstruct(id1)
                best_score = 0
                best_id2 = -1
                
                for id2 in sample2:
                    v2 = self.pac_system.reconstruct(id2)
                    sim = self._cosine_similarity(v1, v2)
                    
                    if sim > best_score:
                        best_score = sim
                        best_id2 = id2
                
                similarities.append(best_score)
                if best_score > 0.5:
                    best_pairs.append((id1, best_id2, float(best_score)))
            
            results[component] = {
                'avg_score': float(np.mean(similarities)) if similarities else 0.0,
                'max_score': float(max(similarities)) if similarities else 0.0,
                'high_resonance_pairs': len([s for s in similarities if s > 0.5]),
                'best_pairs': sorted(best_pairs, key=lambda x: -x[2])[:5]
            }
            
        return results
    
    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute cosine similarity between two tensors of any shape"""
        a_flat = a.flatten().float()
        b_flat = b.flatten().float()
        
        # Handle different sizes by truncating to minimum
        min_len = min(len(a_flat), len(b_flat))
        a_flat = a_flat[:min_len]
        b_flat = b_flat[:min_len]
        
        dot = torch.dot(a_flat, b_flat)
        norm_a = torch.norm(a_flat)
        norm_b = torch.norm(b_flat)
        
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
            
        return float(dot / (norm_a * norm_b))
    
    def get_pac_system_stats(self) -> Dict:
        """Get statistics about the unified PAC system"""
        cache_stats = self.pac_system.cache.stats()
        
        return {
            'total_nodes': len(self.pac_system._node_ids),
            'root_count': len(self.pac_system._roots),
            'cache_stats': cache_stats,
            'models_loaded': list(self.model_mappings.keys()),
            'inject_count': self.pac_system._inject_count,
            'reconstruct_count': self.pac_system._reconstruct_count
        }
    
    def save_state(self, path: str):
        """Save the PAC system state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            'model_mappings': {
                name: {
                    'model_name': m.model_name,
                    'root_id': m.root_id,
                    'component_map': m.component_map,
                    'metadata': m.metadata
                }
                for name, m in self.model_mappings.items()
            },
            'stats': self.get_pac_system_stats()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
            
        print(f"✓ State saved to {path}")


def main():
    print("="*60)
    print("POC-020: PROPER PAC EXTRACTION")
    print("Using fracton.core.pac_system")
    print("="*60)
    
    # Create extractor with shared PAC system
    extractor = ModelToPACExtractor(device='auto')
    
    # Extract multiple models into the SAME PAC space
    models_to_extract = [
        "gpt2",
        "bert-base-uncased",
        "EleutherAI/pythia-70m"
    ]
    
    for model_name in models_to_extract:
        mapping = extractor.extract_model(model_name, sample_tokens=50)
        print(f"\n  Total nodes: {mapping.metadata['total_nodes']}")
        print(f"  Embedding dim: {mapping.metadata['embedding_dim']}")
    
    # Compare in unified PAC space
    comparison = extractor.compare_models_in_pac_space()
    
    # Print summary
    print("\n" + "="*60)
    print("UNIFIED PAC SPACE SUMMARY")
    print("="*60)
    
    stats = extractor.get_pac_system_stats()
    print(f"\n📊 PAC System Stats:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Root nodes: {stats['root_count']} (one per model)")
    print(f"  Cache: {stats['cache_stats']}")
    
    print(f"\n📊 Cross-Model Resonance:")
    for pair, data in comparison['model_pairs'].items():
        print(f"\n  {pair}:")
        print(f"    Root similarity: {data['root_similarity']:.4f}")
        print(f"    Embedding resonance: {data['embedding_resonance']['avg_score']:.4f}")
        print(f"    Layer resonance: {data['layer_resonance']['avg_score']:.4f}")
        high_res = data['embedding_resonance'].get('high_resonance_pairs', 0)
        print(f"    High-resonance pairs: {high_res}")
    
    # Save state
    os.makedirs("results", exist_ok=True)
    extractor.save_state("results/unified_pac_state.json")
    
    # Key insight
    print("\n" + "="*60)
    print("KEY INSIGHT")
    print("="*60)
    print("""
By using the PROPER fracton PAC system:
1. All models exist in the SAME PAC space
2. Dimensions don't matter - we store DELTAS
3. Comparison is direct: find_resonant() across models
4. Knowledge transfer = copy PAC subtrees

A PAC tree is just a PAC tree!
""")


if __name__ == "__main__":
    main()
