"""
POC-020: Multi-Model PAC Tree Extraction

Extract multiple models into PAC trees for comparison.
Find universal patterns vs architecture-specific patterns.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
import os
import pickle


@dataclass
class PACNode:
    """Universal PAC node structure"""
    id: str
    level: int  # 0=root, 1=token, 2=phrase, 3=layer, etc
    pattern_type: str  # "embedding", "attention", "mlp", "confluence"
    data: Any = None  # The actual pattern data
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self, include_data: bool = False) -> Dict:
        """Convert to serializable dict"""
        result = {
            'id': self.id,
            'level': self.level,
            'pattern_type': self.pattern_type,
            'children': self.children,
            'parent': self.parent,
            'metadata': self.metadata,
        }
        if include_data and self.data is not None:
            if isinstance(self.data, np.ndarray):
                result['data_shape'] = list(self.data.shape)
                result['data_mean'] = float(self.data.mean())
                result['data_std'] = float(self.data.std())
            elif isinstance(self.data, dict):
                result['data_keys'] = list(self.data.keys())
            else:
                result['data_type'] = str(type(self.data).__name__)
        return result


class ModelToPACExtractor:
    """Extract any model into PAC tree structure"""
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📍 Using device: {self.device}")
        
    def extract_gpt2(self, model_name: str = "gpt2") -> Dict[str, PACNode]:
        """Extract GPT-2 into PAC tree"""
        from transformers import GPT2Model, GPT2Tokenizer
        
        print(f"\n🔄 Extracting {model_name}...")
        
        model = GPT2Model.from_pretrained(model_name).to(self.device)
        model.eval()
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        pac_tree = {}
        
        # Root node
        root = PACNode(
            id="gpt2_root",
            level=0,
            pattern_type="root",
            data={"model": model_name, "vocab_size": model.config.vocab_size},
            metadata={
                "architecture": "transformer-decoder",
                "n_layers": model.config.n_layer,
                "n_heads": model.config.n_head,
                "hidden_size": model.config.n_embd
            }
        )
        pac_tree[root.id] = root
        
        # Extract embeddings as token-level nodes
        with torch.no_grad():
            embeddings = model.wte.weight.cpu().numpy()
            
        print(f"  📦 Extracting {embeddings.shape[0]} token embeddings...")
        
        # Sample embeddings (full vocab is too large)
        sample_size = min(1000, embeddings.shape[0])
        for i in range(sample_size):
            node = PACNode(
                id=f"gpt2_token_{i}",
                level=1,
                pattern_type="embedding",
                data=embeddings[i],
                parent=root.id,
                metadata={"token_id": i, "token": tokenizer.decode([i]) if i < len(tokenizer) else f"[{i}]"}
            )
            pac_tree[node.id] = node
            root.children.append(node.id)
            
        # Extract layer patterns
        print(f"  📦 Extracting {model.config.n_layer} layers...")
        
        for layer_idx, layer in enumerate(model.h):
            with torch.no_grad():
                # Attention projection weights
                attn_weight = layer.attn.c_attn.weight.cpu().numpy()
                mlp_weight = layer.mlp.c_fc.weight.cpu().numpy()
                
            # Create layer node
            layer_node = PACNode(
                id=f"gpt2_layer_{layer_idx}",
                level=2,
                pattern_type="layer",
                data={
                    "attn_shape": attn_weight.shape,
                    "mlp_shape": mlp_weight.shape,
                    "attn_mean": float(attn_weight.mean()),
                    "mlp_mean": float(mlp_weight.mean())
                },
                parent=root.id,
                metadata={"layer": layer_idx}
            )
            pac_tree[layer_node.id] = layer_node
            root.children.append(layer_node.id)
            
            # Attention heads as children
            n_heads = model.config.n_head
            head_dim = model.config.n_embd // n_heads
            
            for head in range(n_heads):
                start = head * head_dim
                end = (head + 1) * head_dim
                head_pattern = attn_weight[:, start:end]
                
                head_node = PACNode(
                    id=f"gpt2_l{layer_idx}_h{head}",
                    level=3,
                    pattern_type="attention_head",
                    data=np.mean(head_pattern, axis=0),
                    parent=layer_node.id,
                    metadata={"layer": layer_idx, "head": head}
                )
                pac_tree[head_node.id] = head_node
                layer_node.children.append(head_node.id)
                
        print(f"  ✅ Created {len(pac_tree)} PAC nodes for GPT-2")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if self.device == 'cuda' else None
        
        return pac_tree
        
    def extract_bert(self, model_name: str = "bert-base-uncased") -> Dict[str, PACNode]:
        """Extract BERT into PAC tree"""
        from transformers import BertModel, BertTokenizer
        
        print(f"\n🔄 Extracting {model_name}...")
        
        model = BertModel.from_pretrained(model_name).to(self.device)
        model.eval()
        tokenizer = BertTokenizer.from_pretrained(model_name)
        
        pac_tree = {}
        
        # Root node
        root = PACNode(
            id="bert_root",
            level=0,
            pattern_type="root",
            data={"model": model_name, "vocab_size": model.config.vocab_size},
            metadata={
                "architecture": "transformer-encoder",
                "bidirectional": True,
                "n_layers": model.config.num_hidden_layers,
                "n_heads": model.config.num_attention_heads,
                "hidden_size": model.config.hidden_size
            }
        )
        pac_tree[root.id] = root
        
        # Extract embeddings
        with torch.no_grad():
            embeddings = model.embeddings.word_embeddings.weight.cpu().numpy()
            
        print(f"  📦 Extracting {embeddings.shape[0]} token embeddings...")
        
        sample_size = min(1000, embeddings.shape[0])
        for i in range(sample_size):
            node = PACNode(
                id=f"bert_token_{i}",
                level=1,
                pattern_type="embedding",
                data=embeddings[i],
                parent=root.id,
                metadata={"token_id": i, "bidirectional": True}
            )
            pac_tree[node.id] = node
            root.children.append(node.id)
            
        # Extract encoder layers - BERT's key is bidirectional attention
        print(f"  📦 Extracting {model.config.num_hidden_layers} encoder layers...")
        
        for layer_idx, layer in enumerate(model.encoder.layer):
            with torch.no_grad():
                # Self-attention Q, K, V
                query_weight = layer.attention.self.query.weight.cpu().numpy()
                key_weight = layer.attention.self.key.weight.cpu().numpy()
                value_weight = layer.attention.self.value.weight.cpu().numpy()
                
            # BERT's confluence point: where Q meets K
            confluence_node = PACNode(
                id=f"bert_confluence_{layer_idx}",
                level=2,
                pattern_type="confluence",
                data={
                    "query_mean": float(query_weight.mean()),
                    "key_mean": float(key_weight.mean()),
                    "value_mean": float(value_weight.mean()),
                    "qk_alignment": float(np.corrcoef(query_weight.flatten()[:1000], 
                                                       key_weight.flatten()[:1000])[0, 1])
                },
                parent=root.id,
                metadata={
                    "layer": layer_idx, 
                    "type": "bidirectional_confluence",
                    "description": "Q-K attention confluence"
                }
            )
            pac_tree[confluence_node.id] = confluence_node
            root.children.append(confluence_node.id)
            
            # Attention heads
            n_heads = model.config.num_attention_heads
            head_dim = model.config.hidden_size // n_heads
            
            for head in range(n_heads):
                start = head * head_dim
                end = (head + 1) * head_dim
                
                head_node = PACNode(
                    id=f"bert_l{layer_idx}_h{head}",
                    level=3,
                    pattern_type="attention_head",
                    data={
                        "q_slice_mean": float(query_weight[start:end].mean()),
                        "k_slice_mean": float(key_weight[start:end].mean())
                    },
                    parent=confluence_node.id,
                    metadata={"layer": layer_idx, "head": head}
                )
                pac_tree[head_node.id] = head_node
                confluence_node.children.append(head_node.id)
                
        print(f"  ✅ Created {len(pac_tree)} PAC nodes for BERT")
        
        del model
        torch.cuda.empty_cache() if self.device == 'cuda' else None
        
        return pac_tree
        
    def extract_pythia(self, model_name: str = "EleutherAI/pythia-70m") -> Dict[str, PACNode]:
        """Extract Pythia into PAC tree"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"\n🔄 Extracting {model_name}...")
        
        model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        pac_tree = {}
        
        # Root node
        root = PACNode(
            id="pythia_root",
            level=0,
            pattern_type="root",
            data={"model": model_name, "vocab_size": model.config.vocab_size},
            metadata={
                "architecture": "gpt-neox",
                "clean_scaling": True,
                "n_layers": model.config.num_hidden_layers,
                "n_heads": model.config.num_attention_heads,
                "hidden_size": model.config.hidden_size
            }
        )
        pac_tree[root.id] = root
        
        # Extract embeddings
        with torch.no_grad():
            if hasattr(model, 'gpt_neox'):
                embeddings = model.gpt_neox.embed_in.weight.cpu().numpy()
            else:
                embeddings = model.embed_in.weight.cpu().numpy()
                
        print(f"  📦 Extracting {embeddings.shape[0]} token embeddings...")
        
        sample_size = min(1000, embeddings.shape[0])
        for i in range(sample_size):
            node = PACNode(
                id=f"pythia_token_{i}",
                level=1,
                pattern_type="embedding",
                data=embeddings[i],
                parent=root.id,
                metadata={"token_id": i}
            )
            pac_tree[node.id] = node
            root.children.append(node.id)
            
        # Extract layers - Pythia follows clean scaling laws
        if hasattr(model, 'gpt_neox'):
            layers = model.gpt_neox.layers
        else:
            layers = model.layers
            
        print(f"  📦 Extracting {len(layers)} layers with scaling patterns...")
        
        layer_stats = []
        for layer_idx, layer in enumerate(layers):
            with torch.no_grad():
                if hasattr(layer, 'attention'):
                    attn = layer.attention
                    if hasattr(attn, 'query_key_value'):
                        qkv_weight = attn.query_key_value.weight.cpu().numpy()
                        weight_norm = float(np.linalg.norm(qkv_weight))
                    else:
                        weight_norm = 0.0
                else:
                    weight_norm = 0.0
                    
            layer_stats.append(weight_norm)
            
            # Create scaling node (Pythia's contribution)
            scale_node = PACNode(
                id=f"pythia_layer_{layer_idx}",
                level=2,
                pattern_type="scaling",
                data={
                    "layer": layer_idx,
                    "weight_norm": weight_norm,
                    "relative_scale": weight_norm / layer_stats[0] if layer_stats[0] > 0 else 1.0
                },
                parent=root.id,
                metadata={"follows_scaling_law": True}
            )
            pac_tree[scale_node.id] = scale_node
            root.children.append(scale_node.id)
            
        # Add scaling analysis
        if len(layer_stats) > 1:
            scaling_trend = np.polyfit(range(len(layer_stats)), layer_stats, 1)
            root.metadata["scaling_slope"] = float(scaling_trend[0])
            root.metadata["scaling_intercept"] = float(scaling_trend[1])
            
        print(f"  ✅ Created {len(pac_tree)} PAC nodes for Pythia")
        
        del model
        torch.cuda.empty_cache() if self.device == 'cuda' else None
        
        return pac_tree
        
    def extract_all(self, models: List[str] = None) -> Dict[str, Dict[str, PACNode]]:
        """Extract all specified models"""
        if models is None:
            models = ['gpt2', 'bert', 'pythia']
            
        all_trees = {}
        
        extractors = {
            'gpt2': self.extract_gpt2,
            'bert': self.extract_bert,
            'pythia': self.extract_pythia,
        }
        
        for model_type in models:
            if model_type in extractors:
                try:
                    tree = extractors[model_type]()
                    all_trees[model_type] = tree
                except Exception as e:
                    print(f"  ❌ Error extracting {model_type}: {e}")
                    import traceback
                    traceback.print_exc()
                    
        return all_trees


def save_pac_trees(trees: Dict[str, Dict[str, PACNode]], output_dir: str):
    """Save PAC trees to disk"""
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, tree in trees.items():
        # Save JSON (structure only)
        serializable = {node_id: node.to_dict(include_data=True) 
                       for node_id, node in tree.items()}
        
        json_path = os.path.join(output_dir, f"{model_name}_pac_tree.json")
        with open(json_path, 'w') as f:
            json.dump(serializable, f, indent=2)
            
        # Save pickle (full data)
        pkl_path = os.path.join(output_dir, f"{model_name}_pac_tree.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(tree, f)
            
        print(f"  💾 Saved {model_name} PAC tree ({len(tree)} nodes)")


if __name__ == "__main__":
    print("=" * 70)
    print("POC-020: MULTI-MODEL PAC EXTRACTION")
    print("=" * 70)
    
    extractor = ModelToPACExtractor()
    
    # Extract models
    models_to_extract = ['gpt2', 'bert', 'pythia']
    all_trees = extractor.extract_all(models_to_extract)
    
    # Save trees
    save_pac_trees(all_trees, "extracted_trees")
    
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    
    # Summary
    for model, tree in all_trees.items():
        levels = defaultdict(int)
        types = defaultdict(int)
        
        for node in tree.values():
            levels[node.level] += 1
            types[node.pattern_type] += 1
            
        print(f"\n{model.upper()}:")
        print(f"  Total nodes: {len(tree)}")
        print(f"  Levels: {dict(levels)}")
        print(f"  Types: {dict(types)}")
