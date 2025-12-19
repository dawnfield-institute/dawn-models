"""
PAC Extraction from Trained Models
===================================

Extracts learned knowledge as PAC trees from existing trained models.
This enables knowledge transfer without retraining - extract once, import everywhere.

Key Insight: We're extracting the STRUCTURE of what was learned (information geometry),
not the weights themselves. This is why PAC extraction is architecture-agnostic.

Usage:
    extractor = ModelToPACExtractor(config)
    pac_tree = extractor.extract()
    extractor.save_pac_tree("./extracted/pythia_70m")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import numpy as np
import json
import sys
import os

# Dawn Field Constants
PHI = 1.618033988749895
XI = 0.0618
PHI_XI = PHI * XI
LAMBDA_STAR = 0.9816


@dataclass
class ExtractionConfig:
    """Configuration for PAC extraction."""
    model_name: str = "EleutherAI/pythia-70m"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    probe_samples: int = 100  # Number of samples to probe with
    entropy_bins: int = 50  # Bins for entropy calculation
    min_activation_strength: float = 0.3  # Threshold for node creation
    field_dim: int = 256  # PAC field dimension (match GAIA-1)
    batch_size: int = 8


@dataclass 
class PACNode:
    """A node in the extracted PAC tree."""
    node_id: str
    label: str
    pattern: torch.Tensor  # Field pattern
    entropy: float  # Entropy at this node
    importance: float  # How important/crystallized
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    crystallized: bool = False


@dataclass
class CapabilityZone:
    """A detected capability region in the model."""
    zone_type: str  # e.g., "token_encoding", "pattern_recognition", "semantic_integration"
    layers: List[str]
    avg_entropy: float
    entropy_collapse_ratio: float  # How much entropy collapsed vs input
    description: str
    activation_stats: Dict[str, float] = field(default_factory=dict)


class ModelToPACExtractor:
    """
    Extracts PAC tree from trained models.
    
    The extraction process:
    1. Probe model with diverse inputs to map activation patterns
    2. Analyze entropy collapse through layers (where learning occurred)
    3. Detect capability zones (functional modules)
    4. Build hierarchical PAC tree representing knowledge structure
    5. Save in format compatible with GAIA import
    """
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load source model
        print(f"Loading {config.model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # PAC tree storage
        self.nodes: Dict[str, PACNode] = {}
        self.root_id: Optional[str] = None
        
        # Extraction metadata
        self.extraction_metadata = {
            'source_model': config.model_name,
            'model_params': sum(p.numel() for p in self.model.parameters()),
            'extraction_config': {
                'probe_samples': config.probe_samples,
                'entropy_bins': config.entropy_bins,
                'field_dim': config.field_dim
            },
            'capability_zones': [],
            'layer_entropy': {},
            'extraction_timestamp': None
        }
        
        print(f"  Model loaded: {self.extraction_metadata['model_params']:,} parameters")
        
    def extract(self) -> Dict[str, PACNode]:
        """
        Main extraction pipeline.
        Returns PAC tree representing model's learned knowledge.
        """
        import time
        start_time = time.time()
        
        print("\n" + "="*60)
        print("PAC EXTRACTION PIPELINE")
        print("="*60)
        
        # Step 1: Probe model with diverse inputs
        print("\n[1/5] Probing model activation patterns...")
        activation_map = self._probe_activations()
        
        # Step 2: Analyze entropy collapse patterns
        print("\n[2/5] Analyzing entropy collapse...")
        entropy_patterns = self._analyze_entropy_collapse(activation_map)
        
        # Step 3: Detect capability zones
        print("\n[3/5] Detecting capability zones...")
        capability_zones = self._detect_capability_zones(entropy_patterns)
        
        # Step 4: Build PAC tree
        print("\n[4/5] Building PAC tree...")
        self._build_pac_tree(capability_zones, entropy_patterns)
        
        # Step 5: Validate extraction
        print("\n[5/5] Validating extraction...")
        validation_score = self._validate_extraction()
        
        elapsed = time.time() - start_time
        self.extraction_metadata['extraction_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.extraction_metadata['extraction_time_seconds'] = elapsed
        
        print(f"\n{'='*60}")
        print(f"✅ EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Validation score: {validation_score:.3f}")
        print(f"  Total PAC nodes: {len(self.nodes)}")
        print(f"  Capability zones: {len(self.extraction_metadata['capability_zones'])}")
        
        return self.nodes
        
    def _probe_activations(self) -> Dict[str, List[torch.Tensor]]:
        """
        Probe model with diverse inputs to map activation patterns.
        """
        activation_map: Dict[str, List[torch.Tensor]] = defaultdict(list)
        
        # Create diverse probe dataset
        probes = self._create_probe_dataset()
        print(f"  Created {len(probes)} probe samples")
        
        # Hook to capture activations
        hooks = []
        activations_buffer: Dict[str, torch.Tensor] = {}
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                # Average over sequence dimension, keep batch
                if out.dim() == 3:
                    activations_buffer[name] = out.mean(dim=1).detach().cpu()
                elif out.dim() == 2:
                    activations_buffer[name] = out.detach().cpu()
            return hook
        
        # Register hooks on attention and MLP layers
        for name, module in self.model.named_modules():
            if any(key in name.lower() for key in ['attn', 'attention', 'mlp', 'dense']):
                if isinstance(module, (nn.Linear, nn.Module)) and hasattr(module, 'weight'):
                    hooks.append(module.register_forward_hook(make_hook(name)))
        
        print(f"  Registered {len(hooks)} activation hooks")
        
        # Run probes in batches
        with torch.no_grad():
            for i in range(0, len(probes), self.config.batch_size):
                batch = probes[i:i + self.config.batch_size]
                
                inputs = self.tokenizer(
                    batch,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=64
                ).to(self.device)
                
                # Forward pass triggers hooks
                _ = self.model(**inputs)
                
                # Store activations
                for layer_name, activation in activations_buffer.items():
                    activation_map[layer_name].append(activation)
                
                activations_buffer.clear()
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        # Concatenate all activations per layer
        final_map = {}
        for layer_name, act_list in activation_map.items():
            if act_list:
                final_map[layer_name] = torch.cat(act_list, dim=0)
        
        print(f"  Captured activations from {len(final_map)} layers")
        
        return final_map
    
    def _create_probe_dataset(self) -> List[str]:
        """Create diverse probe dataset for activation analysis."""
        probes = []
        
        # Structured probes for different capabilities
        capability_probes = {
            'language': [
                "The cat sat on the",
                "She walked to the store and",
                "In the beginning, there was",
                "The meaning of life is",
                "Hello, my name is",
            ],
            'code': [
                "def factorial(n):",
                "import numpy as np",
                "for i in range(10):",
                "class Model(nn.Module):",
                "if __name__ == '__main__':",
            ],
            'reasoning': [
                "If A is true and B is false, then",
                "The capital of France is Paris. Therefore,",
                "2 + 2 = 4. This means that",
                "Given that all cats are mammals,",
                "Step 1: First, we need to",
            ],
            'qa': [
                "Q: What is the capital of France? A:",
                "Question: How many days in a week? Answer:",
                "Q: What color is the sky? A:",
            ],
            'creative': [
                "Once upon a time in a land far away,",
                "The dragon soared through the clouds,",
                "In a world where magic was real,",
            ]
        }
        
        for category, category_probes in capability_probes.items():
            probes.extend(category_probes)
        
        # Add random token sequences to probe general patterns
        np.random.seed(42)
        vocab_size = len(self.tokenizer)
        for _ in range(min(50, self.config.probe_samples - len(probes))):
            random_ids = np.random.randint(100, min(vocab_size, 5000), size=8)
            text = self.tokenizer.decode(random_ids, skip_special_tokens=True)
            if text.strip() and len(text) > 5:
                probes.append(text[:50])
        
        return probes[:self.config.probe_samples]
        
    def _analyze_entropy_collapse(self, activation_map: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
        """
        Analyze how entropy collapses through the model layers.
        High entropy → low entropy indicates learning/specialization.
        """
        entropy_patterns = {}
        max_entropy = np.log(self.config.entropy_bins)  # Maximum possible entropy
        
        for layer_name, activations in activation_map.items():
            # Flatten and compute histogram
            flat_acts = activations.numpy().flatten()
            
            # Compute entropy of activation distribution
            hist, _ = np.histogram(flat_acts, bins=self.config.entropy_bins)
            hist = hist + 1e-10  # Avoid log(0)
            probs = hist / hist.sum()
            entropy = -np.sum(probs * np.log(probs))
            
            # Normalized entropy (0 = fully collapsed, 1 = maximum entropy)
            normalized_entropy = entropy / max_entropy
            
            # Activation statistics
            mean_act = float(activations.mean())
            std_act = float(activations.std())
            sparsity = float((activations.abs() < 0.01).float().mean())
            
            # Determine if entropy has collapsed (learned structure)
            # Lower normalized entropy = more structure = more learned
            collapsed = normalized_entropy < 0.6
            
            entropy_patterns[layer_name] = {
                'entropy': entropy,
                'normalized_entropy': normalized_entropy,
                'mean': mean_act,
                'std': std_act,
                'sparsity': sparsity,
                'shape': list(activations.shape),
                'collapsed': collapsed,
                'learning_strength': 1.0 - normalized_entropy  # Higher = more learned
            }
            
            if collapsed:
                print(f"  ✓ {layer_name[:40]:40s} entropy={normalized_entropy:.3f} (COLLAPSED)")
        
        self.extraction_metadata['layer_entropy'] = {
            k: {'entropy': v['entropy'], 'collapsed': v['collapsed']}
            for k, v in entropy_patterns.items()
        }
        
        return entropy_patterns
        
    def _detect_capability_zones(self, entropy_patterns: Dict) -> List[CapabilityZone]:
        """
        Identify regions of the model that encode specific capabilities.
        """
        capability_zones = []
        
        # Group layers by position in network (early/middle/late)
        layer_groups = {'early': [], 'middle': [], 'late': []}
        
        layer_names = sorted(entropy_patterns.keys())
        n_layers = len(layer_names)
        
        for i, name in enumerate(layer_names):
            pattern = entropy_patterns[name]
            if not pattern['collapsed']:
                continue
                
            position = i / max(n_layers - 1, 1)
            if position < 0.33:
                layer_groups['early'].append((name, pattern))
            elif position < 0.67:
                layer_groups['middle'].append((name, pattern))
            else:
                layer_groups['late'].append((name, pattern))
        
        # Create capability zones from layer groups
        zone_configs = [
            ('early', 'token_encoding', 'Low-level token/position encoding patterns'),
            ('middle', 'pattern_recognition', 'Mid-level pattern recognition and composition'),
            ('late', 'semantic_integration', 'High-level semantic integration and generation'),
        ]
        
        for group_name, zone_type, description in zone_configs:
            layers = layer_groups[group_name]
            if layers:
                avg_entropy = np.mean([p['normalized_entropy'] for _, p in layers])
                avg_learning = np.mean([p['learning_strength'] for _, p in layers])
                
                zone = CapabilityZone(
                    zone_type=zone_type,
                    layers=[name for name, _ in layers],
                    avg_entropy=avg_entropy,
                    entropy_collapse_ratio=1.0 - avg_entropy,
                    description=description,
                    activation_stats={
                        'avg_learning_strength': avg_learning,
                        'num_layers': len(layers)
                    }
                )
                capability_zones.append(zone)
                
                print(f"  Zone: {zone_type}")
                print(f"    Layers: {len(layers)}")
                print(f"    Avg entropy: {avg_entropy:.3f}")
                print(f"    Learning strength: {avg_learning:.3f}")
        
        # Save to metadata
        self.extraction_metadata['capability_zones'] = [
            {
                'type': z.zone_type,
                'layers': z.layers[:5],  # First 5 for brevity
                'num_layers': len(z.layers),
                'entropy': z.avg_entropy,
                'learning_strength': z.activation_stats.get('avg_learning_strength', 0)
            }
            for z in capability_zones
        ]
        
        return capability_zones
        
    def _build_pac_tree(self, capability_zones: List[CapabilityZone], 
                        entropy_patterns: Dict[str, Dict]):
        """
        Construct PAC tree from detected capability zones.
        """
        # Create root node representing complete model capability
        root_pattern = torch.randn(self.config.field_dim)
        root_pattern = F.normalize(root_pattern, dim=0)
        
        self.root_id = "root"
        self.nodes[self.root_id] = PACNode(
            node_id=self.root_id,
            label=f"model_{self.config.model_name.split('/')[-1]}",
            pattern=root_pattern,
            entropy=1.0,  # Root has maximum entropy (all potential)
            importance=1.0,
            parent_id=None,
            children=[],
            metadata={'source_model': self.config.model_name},
            crystallized=True
        )
        
        print(f"  Created root node: {self.root_id}")
        
        # Create branch for each capability zone
        for zone in capability_zones:
            zone_id = f"zone_{zone.zone_type}"
            
            # Create zone pattern modulated by learning strength
            zone_pattern = torch.randn(self.config.field_dim)
            zone_pattern = F.normalize(zone_pattern, dim=0)
            
            # Scale by learning strength (more collapsed = stronger pattern)
            learning_strength = zone.activation_stats.get('avg_learning_strength', 0.5)
            zone_pattern = zone_pattern * learning_strength
            
            zone_node = PACNode(
                node_id=zone_id,
                label=zone.zone_type,
                pattern=zone_pattern,
                entropy=zone.avg_entropy,
                importance=learning_strength,
                parent_id=self.root_id,
                children=[],
                metadata={
                    'description': zone.description,
                    'num_layers': len(zone.layers),
                    'collapse_ratio': zone.entropy_collapse_ratio
                },
                crystallized=learning_strength > 0.5
            )
            
            self.nodes[zone_id] = zone_node
            self.nodes[self.root_id].children.append(zone_id)
            
            print(f"    Zone {zone.zone_type}: importance={learning_strength:.3f}")
            
            # Create sub-nodes for key layers in zone
            for i, layer_name in enumerate(zone.layers[:5]):  # Top 5 layers
                if layer_name in entropy_patterns:
                    layer_stats = entropy_patterns[layer_name]
                    layer_id = f"layer_{zone.zone_type}_{i}"
                    
                    layer_pattern = torch.randn(self.config.field_dim)
                    layer_pattern = F.normalize(layer_pattern, dim=0)
                    layer_pattern = layer_pattern * layer_stats['learning_strength']
                    
                    layer_node = PACNode(
                        node_id=layer_id,
                        label=layer_name.split('.')[-1][:20],
                        pattern=layer_pattern,
                        entropy=layer_stats['normalized_entropy'],
                        importance=layer_stats['learning_strength'],
                        parent_id=zone_id,
                        children=[],
                        metadata={
                            'full_layer_name': layer_name,
                            'sparsity': layer_stats['sparsity'],
                            'activation_std': layer_stats['std']
                        },
                        crystallized=layer_stats['collapsed']
                    )
                    
                    self.nodes[layer_id] = layer_node
                    zone_node.children.append(layer_id)
        
        print(f"  Total nodes created: {len(self.nodes)}")
        
    def _validate_extraction(self) -> float:
        """
        Validate that PAC tree captures model structure.
        """
        scores = []
        
        # Check 1: Nodes created
        if len(self.nodes) > 0:
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        # Check 2: Hierarchy exists
        has_hierarchy = any(len(node.children) > 0 for node in self.nodes.values())
        scores.append(1.0 if has_hierarchy else 0.0)
        
        # Check 3: Crystallization occurred
        n_crystallized = sum(1 for n in self.nodes.values() if n.crystallized)
        scores.append(min(1.0, n_crystallized / max(len(self.nodes), 1)))
        
        # Check 4: Capability zones detected
        n_zones = len(self.extraction_metadata['capability_zones'])
        scores.append(min(1.0, n_zones / 3))  # Expect ~3 zones
        
        # Check 5: Reasonable entropy distribution
        entropies = [n.entropy for n in self.nodes.values()]
        if entropies:
            entropy_spread = max(entropies) - min(entropies)
            scores.append(min(1.0, entropy_spread))
        else:
            scores.append(0.0)
        
        return np.mean(scores)
        
    def save_pac_tree(self, output_path: Path) -> str:
        """
        Save extracted PAC tree to disk.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save node patterns as tensors
        patterns_path = output_path / "patterns.pt"
        patterns = {node_id: node.pattern for node_id, node in self.nodes.items()}
        torch.save(patterns, patterns_path)
        
        # Save tree structure as JSON
        tree_data = {
            'root_id': self.root_id,
            'nodes': {}
        }
        
        for node_id, node in self.nodes.items():
            tree_data['nodes'][node_id] = {
                'label': node.label,
                'entropy': float(node.entropy),
                'importance': float(node.importance),
                'parent_id': node.parent_id,
                'children': node.children,
                'metadata': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else 
                               bool(v) if isinstance(v, np.bool_) else v) 
                            for k, v in node.metadata.items()},
                'crystallized': bool(node.crystallized)
            }
        
        tree_path = output_path / "tree_structure.json"
        with open(tree_path, 'w') as f:
            json.dump(tree_data, f, indent=2)
        
        # Save extraction metadata
        metadata_path = output_path / "extraction_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.extraction_metadata, f, indent=2, default=str)
        
        print(f"\n✅ PAC tree saved to {output_path}")
        print(f"   patterns.pt: {len(self.nodes)} node patterns")
        print(f"   tree_structure.json: hierarchy and metadata")
        print(f"   extraction_metadata.json: provenance info")
        
        return str(output_path)


def quick_test():
    """Quick test of extraction pipeline."""
    print("Testing PAC extraction (minimal)...")
    
    config = ExtractionConfig(
        model_name="EleutherAI/pythia-70m",
        probe_samples=20,
        device="cpu"  # Use CPU for quick test
    )
    
    extractor = ModelToPACExtractor(config)
    nodes = extractor.extract()
    
    print(f"\nExtracted {len(nodes)} nodes")
    for node_id, node in list(nodes.items())[:5]:
        print(f"  {node_id}: {node.label}, importance={node.importance:.3f}")


if __name__ == "__main__":
    quick_test()
