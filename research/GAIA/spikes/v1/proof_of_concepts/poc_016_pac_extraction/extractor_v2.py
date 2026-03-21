"""
PAC Extraction V2 - Proper Weight Structure Extraction
=======================================================

V1 PROBLEM: We extracted 4,864 random numbers from a 70M parameter model.
            That's like recording average volume levels instead of the symphony.

V2 SOLUTION: Extract actual learned structure via:
1. Weight matrix decomposition (SVD) to capture learned transformations
2. Attention pattern extraction to capture learned relationships
3. Embedding subspace extraction to capture semantic structure
4. Proper PAC tree with conservation properties

The output should be MASSIVE - megabytes, not kilobytes.
Importing should take significant time because we're moving real structure.

Uses existing PACEngine infrastructure from dawn-field-theory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import json
import sys
import os
import time

# Dawn Field Constants
PHI = 1.618033988749895
XI = 0.0618
LAMBDA_STAR = 0.9816


@dataclass
class ExtractionConfigV2:
    """V2 extraction configuration - extracts actual structure."""
    model_name: str = "EleutherAI/pythia-70m"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Compression settings
    svd_rank: int = 64  # Rank for SVD compression of weight matrices
    attention_samples: int = 50  # Samples for attention pattern extraction
    embedding_subspace_dim: int = 256  # Dimension for embedding subspace
    
    # What to extract
    extract_embeddings: bool = True
    extract_attention: bool = True
    extract_mlp: bool = True
    extract_layer_norms: bool = True
    
    # PAC settings  
    pac_depth: int = 4  # Tree depth


@dataclass
class ExtractedLayer:
    """Extracted structure from a single layer."""
    layer_idx: int
    layer_name: str
    
    # Attention components (if extracted)
    attn_qkv_basis: Optional[torch.Tensor] = None  # SVD basis for Q,K,V
    attn_qkv_scales: Optional[torch.Tensor] = None  # Singular values
    attn_out_basis: Optional[torch.Tensor] = None
    attn_out_scales: Optional[torch.Tensor] = None
    attn_patterns: Optional[torch.Tensor] = None  # Mean attention patterns
    
    # MLP components (if extracted)
    mlp_up_basis: Optional[torch.Tensor] = None
    mlp_up_scales: Optional[torch.Tensor] = None
    mlp_down_basis: Optional[torch.Tensor] = None
    mlp_down_scales: Optional[torch.Tensor] = None
    
    # Layer norms (exact - these are small)
    ln_weight: Optional[torch.Tensor] = None
    ln_bias: Optional[torch.Tensor] = None
    
    # Entropy metrics
    entropy_in: float = 0.0
    entropy_out: float = 0.0
    entropy_collapse_ratio: float = 0.0
    
    # Metadata for Vh matrices (right singular vectors)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedPAC:
    """Complete PAC extraction result."""
    source_model: str
    source_params: int
    extraction_timestamp: str
    
    # Core structures
    embedding_matrix: torch.Tensor  # Full or compressed embeddings
    output_projection: torch.Tensor  # LM head
    layers: List[ExtractedLayer] = field(default_factory=list)
    
    # PAC metadata
    total_extracted_params: int = 0
    compression_ratio: float = 0.0
    entropy_profile: Dict[str, float] = field(default_factory=dict)
    capability_zones: List[Dict] = field(default_factory=list)


class ModelToPACExtractorV2:
    """
    V2 Extractor - Extracts ACTUAL learned structure from models.
    
    Key differences from V1:
    - Extracts weight matrices (compressed via SVD)
    - Captures attention patterns from forward passes
    - Preserves layer norm parameters exactly
    - Output is megabytes, not kilobytes
    """
    
    def __init__(self, config: ExtractionConfigV2):
        self.config = config
        self.device = torch.device(config.device)
        
        print(f"Loading {config.model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.source_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Loaded: {self.source_params:,} parameters")
        
        self.extracted: Optional[ExtractedPAC] = None
        
    def extract(self) -> ExtractedPAC:
        """
        Main extraction pipeline - extracts actual learned structure.
        """
        start_time = time.time()
        
        print("\n" + "="*70)
        print("PAC EXTRACTION V2 - PROPER STRUCTURE EXTRACTION")
        print("="*70)
        
        # 1. Extract embeddings
        print("\n[1/5] Extracting embedding matrix...")
        embedding_matrix = self._extract_embeddings()
        
        # 2. Extract output projection (LM head)
        print("\n[2/5] Extracting output projection...")
        output_projection = self._extract_output_projection()
        
        # 3. Extract transformer layers
        print("\n[3/5] Extracting transformer layers...")
        layers = self._extract_layers()
        
        # 4. Capture attention patterns via forward passes
        print("\n[4/5] Capturing attention patterns...")
        self._capture_attention_patterns(layers)
        
        # 5. Build PAC structure with entropy analysis
        print("\n[5/5] Analyzing entropy collapse...")
        entropy_profile, capability_zones = self._analyze_entropy_structure(layers)
        
        # Calculate totals
        total_extracted = self._count_extracted_params(embedding_matrix, output_projection, layers)
        compression_ratio = self.source_params / total_extracted
        
        elapsed = time.time() - start_time
        
        self.extracted = ExtractedPAC(
            source_model=self.config.model_name,
            source_params=self.source_params,
            extraction_timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            embedding_matrix=embedding_matrix,
            output_projection=output_projection,
            layers=layers,
            total_extracted_params=total_extracted,
            compression_ratio=compression_ratio,
            entropy_profile=entropy_profile,
            capability_zones=capability_zones,
        )
        
        print(f"\n{'='*70}")
        print("✅ EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Source parameters: {self.source_params:,}")
        print(f"  Extracted parameters: {total_extracted:,}")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Layers extracted: {len(layers)}")
        
        return self.extracted
    
    def _extract_embeddings(self) -> torch.Tensor:
        """Extract embedding matrix (token embeddings)."""
        # Find embedding layer
        if hasattr(self.model, 'gpt_neox'):
            embed = self.model.gpt_neox.embed_in.weight.data.clone().cpu()
        elif hasattr(self.model, 'transformer'):
            embed = self.model.transformer.wte.weight.data.clone().cpu()
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            embed = self.model.model.embed_tokens.weight.data.clone().cpu()
        else:
            # Generic search
            for name, param in self.model.named_parameters():
                if 'embed' in name.lower() and 'weight' in name.lower():
                    embed = param.data.clone().cpu()
                    break
            else:
                raise ValueError("Could not find embedding matrix")
        
        print(f"  Embedding shape: {embed.shape}")
        print(f"  Parameters: {embed.numel():,}")
        return embed
    
    def _extract_output_projection(self) -> torch.Tensor:
        """Extract LM head (output projection to vocabulary)."""
        if hasattr(self.model, 'embed_out'):
            proj = self.model.embed_out.weight.data.clone().cpu()
        elif hasattr(self.model, 'lm_head'):
            proj = self.model.lm_head.weight.data.clone().cpu()
        else:
            # Often tied to embeddings
            for name, param in self.model.named_parameters():
                if 'lm_head' in name.lower() or 'embed_out' in name.lower():
                    proj = param.data.clone().cpu()
                    break
            else:
                # Use embedding matrix as fallback (weight tying)
                print("  Using tied embeddings for output projection")
                return self._extract_embeddings()
        
        print(f"  Output projection shape: {proj.shape}")
        print(f"  Parameters: {proj.numel():,}")
        return proj
    
    def _extract_layers(self) -> List[ExtractedLayer]:
        """Extract all transformer layers with SVD compression."""
        layers = []
        
        # Find transformer layers
        if hasattr(self.model, 'gpt_neox'):
            transformer_layers = self.model.gpt_neox.layers
        elif hasattr(self.model, 'transformer'):
            transformer_layers = self.model.transformer.h
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            transformer_layers = self.model.model.layers
        else:
            raise ValueError("Could not find transformer layers")
        
        n_layers = len(transformer_layers)
        print(f"  Found {n_layers} transformer layers")
        
        for idx, layer in enumerate(transformer_layers):
            extracted_layer = self._extract_single_layer(idx, layer)
            layers.append(extracted_layer)
            
            if (idx + 1) % 2 == 0 or idx == n_layers - 1:
                print(f"    Extracted layer {idx + 1}/{n_layers}")
        
        return layers
    
    def _extract_single_layer(self, idx: int, layer: nn.Module) -> ExtractedLayer:
        """Extract a single transformer layer."""
        extracted = ExtractedLayer(
            layer_idx=idx,
            layer_name=f"layer_{idx}"
        )
        
        rank = self.config.svd_rank
        
        # Extract attention weights
        if self.config.extract_attention:
            attn_weights = self._find_attention_weights(layer)
            if attn_weights is not None:
                # SVD compression for Q, K, V projection
                qkv = attn_weights['qkv']
                if qkv is not None:
                    U, S, Vh = torch.linalg.svd(qkv.float(), full_matrices=False)
                    # Store U and Vh separately to preserve structure
                    extracted.attn_qkv_basis = U[:, :rank].cpu()  # Left singular vectors
                    extracted.attn_qkv_scales = S[:rank].cpu()    # Singular values
                    # Store right vectors in metadata if needed
                    extracted.metadata['attn_qkv_Vh'] = Vh[:rank, :].cpu()
                
                # Output projection
                out = attn_weights['out']
                if out is not None:
                    U, S, Vh = torch.linalg.svd(out.float(), full_matrices=False)
                    extracted.attn_out_basis = U[:, :rank].cpu()
                    extracted.attn_out_scales = S[:rank].cpu()
                    extracted.metadata['attn_out_Vh'] = Vh[:rank, :].cpu()
        
        # Extract MLP weights
        if self.config.extract_mlp:
            mlp_weights = self._find_mlp_weights(layer)
            if mlp_weights is not None:
                # Up projection
                up = mlp_weights['up']
                if up is not None:
                    U, S, Vh = torch.linalg.svd(up.float(), full_matrices=False)
                    extracted.mlp_up_basis = U[:, :rank].cpu()
                    extracted.mlp_up_scales = S[:rank].cpu()
                    extracted.metadata['mlp_up_Vh'] = Vh[:rank, :].cpu()
                
                # Down projection
                down = mlp_weights['down']
                if down is not None:
                    U, S, Vh = torch.linalg.svd(down.float(), full_matrices=False)
                    extracted.mlp_down_basis = U[:, :rank].cpu()
                    extracted.mlp_down_scales = S[:rank].cpu()
                    extracted.metadata['mlp_down_Vh'] = Vh[:rank, :].cpu()
        
        # Extract layer norms (exact - small)
        if self.config.extract_layer_norms:
            ln = self._find_layer_norm(layer)
            if ln is not None:
                extracted.ln_weight = ln['weight'].cpu()
                extracted.ln_bias = ln['bias'].cpu() if ln['bias'] is not None else None
        
        return extracted
    
    def _find_attention_weights(self, layer: nn.Module) -> Optional[Dict[str, torch.Tensor]]:
        """Find attention weight matrices in a layer."""
        result = {'qkv': None, 'out': None}
        
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                name_lower = name.lower()
                if any(x in name_lower for x in ['query_key_value', 'qkv', 'c_attn']):
                    result['qkv'] = module.weight.data.clone()
                elif any(x in name_lower for x in ['dense', 'c_proj', 'out_proj']):
                    if 'attn' in name_lower or 'attention' in name_lower:
                        result['out'] = module.weight.data.clone()
        
        # Fallback: look for patterns
        for name, param in layer.named_parameters():
            name_lower = name.lower()
            if result['qkv'] is None and 'query_key_value' in name_lower and 'weight' in name_lower:
                result['qkv'] = param.data.clone()
            if result['out'] is None and 'dense' in name_lower and 'attention' in name_lower:
                result['out'] = param.data.clone()
        
        return result if any(v is not None for v in result.values()) else None
    
    def _find_mlp_weights(self, layer: nn.Module) -> Optional[Dict[str, torch.Tensor]]:
        """Find MLP weight matrices in a layer."""
        result = {'up': None, 'down': None}
        
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                name_lower = name.lower()
                if any(x in name_lower for x in ['dense_h_to_4h', 'fc1', 'c_fc', 'up', 'gate']):
                    result['up'] = module.weight.data.clone()
                elif any(x in name_lower for x in ['dense_4h_to_h', 'fc2', 'c_proj', 'down']):
                    if 'mlp' in name_lower or 'fc' in name_lower:
                        result['down'] = module.weight.data.clone()
        
        # Fallback
        for name, param in layer.named_parameters():
            name_lower = name.lower()
            if result['up'] is None and 'dense_h_to_4h' in name_lower:
                result['up'] = param.data.clone()
            if result['down'] is None and 'dense_4h_to_h' in name_lower:
                result['down'] = param.data.clone()
        
        return result if any(v is not None for v in result.values()) else None
    
    def _find_layer_norm(self, layer: nn.Module) -> Optional[Dict[str, torch.Tensor]]:
        """Find layer norm parameters."""
        for name, module in layer.named_modules():
            if isinstance(module, nn.LayerNorm):
                return {
                    'weight': module.weight.data.clone(),
                    'bias': module.bias.data.clone() if module.bias is not None else None
                }
        return None
    
    def _capture_attention_patterns(self, layers: List[ExtractedLayer]):
        """Capture attention patterns via forward passes."""
        print(f"  Running {self.config.attention_samples} forward passes...")
        
        # Diverse prompts for attention pattern sampling
        prompts = [
            "The meaning of life is",
            "In mathematics, we can prove that",
            "Once upon a time there was",
            "The function returns",
            "To solve this problem, first",
        ] * (self.config.attention_samples // 5)
        
        attention_accumulator = {idx: [] for idx in range(len(layers))}
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=32)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            
            if outputs.attentions:
                for idx, attn in enumerate(outputs.attentions):
                    # Average over heads and batch
                    attn_pattern = attn.mean(dim=(0, 1)).cpu()
                    attention_accumulator[idx].append(attn_pattern)
        
        # Store mean attention patterns
        for idx, patterns in attention_accumulator.items():
            if patterns and idx < len(layers):
                mean_pattern = torch.stack(patterns).mean(dim=0)
                layers[idx].attn_patterns = mean_pattern
        
        print(f"  Captured attention patterns for {len([l for l in layers if l.attn_patterns is not None])} layers")
    
    def _analyze_entropy_structure(self, layers: List[ExtractedLayer]) -> Tuple[Dict, List[Dict]]:
        """Analyze entropy collapse across layers."""
        entropy_profile = {}
        
        for layer in layers:
            # Calculate entropy from SVD singular values (if extracted)
            if layer.attn_qkv_scales is not None:
                # Normalize to probability distribution
                scales = layer.attn_qkv_scales.float()
                probs = scales / scales.sum()
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                layer.entropy_in = entropy
                entropy_profile[f"layer_{layer.layer_idx}_attn"] = entropy
            
            if layer.mlp_up_scales is not None:
                scales = layer.mlp_up_scales.float()
                probs = scales / scales.sum()
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                layer.entropy_out = entropy
                entropy_profile[f"layer_{layer.layer_idx}_mlp"] = entropy
        
        # Detect capability zones by entropy gradient
        capability_zones = []
        
        # Early layers (token encoding)
        early_entropy = np.mean([layers[i].entropy_in for i in range(min(2, len(layers))) 
                                if layers[i].entropy_in > 0]) if layers else 0
        capability_zones.append({
            'zone': 'token_encoding',
            'layers': list(range(min(2, len(layers)))),
            'entropy': early_entropy
        })
        
        # Middle layers (pattern recognition)
        mid_start = len(layers) // 3
        mid_end = 2 * len(layers) // 3
        mid_entropy = np.mean([layers[i].entropy_in for i in range(mid_start, mid_end)
                              if layers[i].entropy_in > 0]) if layers else 0
        capability_zones.append({
            'zone': 'pattern_recognition',
            'layers': list(range(mid_start, mid_end)),
            'entropy': mid_entropy
        })
        
        # Late layers (semantic integration)
        late_entropy = np.mean([layers[i].entropy_in for i in range(mid_end, len(layers))
                               if layers[i].entropy_in > 0]) if layers else 0
        capability_zones.append({
            'zone': 'semantic_integration',
            'layers': list(range(mid_end, len(layers))),
            'entropy': late_entropy
        })
        
        return entropy_profile, capability_zones
    
    def _count_extracted_params(self, embeddings: torch.Tensor, 
                                output_proj: torch.Tensor,
                                layers: List[ExtractedLayer]) -> int:
        """Count total extracted parameters."""
        total = embeddings.numel() + output_proj.numel()
        
        for layer in layers:
            for attr in ['attn_qkv_basis', 'attn_qkv_scales', 'attn_out_basis', 'attn_out_scales',
                        'mlp_up_basis', 'mlp_up_scales', 'mlp_down_basis', 'mlp_down_scales',
                        'ln_weight', 'ln_bias', 'attn_patterns']:
                tensor = getattr(layer, attr, None)
                if tensor is not None:
                    total += tensor.numel()
        
        return total
    
    def save(self, output_dir: str):
        """Save extracted PAC to disk."""
        if self.extracted is None:
            raise ValueError("No extraction to save. Call extract() first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving extraction to {output_path}...")
        
        # Save tensors
        tensors = {
            'embedding_matrix': self.extracted.embedding_matrix,
            'output_projection': self.extracted.output_projection,
        }
        
        # Add layer tensors
        for layer in self.extracted.layers:
            prefix = f"layer_{layer.layer_idx}"
            for attr in ['attn_qkv_basis', 'attn_qkv_scales', 'attn_out_basis', 'attn_out_scales',
                        'mlp_up_basis', 'mlp_up_scales', 'mlp_down_basis', 'mlp_down_scales',
                        'ln_weight', 'ln_bias', 'attn_patterns']:
                tensor = getattr(layer, attr, None)
                if tensor is not None:
                    tensors[f"{prefix}_{attr}"] = tensor
        
        # Save all tensors
        torch.save(tensors, output_path / "pac_structure.pt")
        
        # Save metadata
        metadata = {
            'source_model': self.extracted.source_model,
            'source_params': self.extracted.source_params,
            'extraction_timestamp': self.extracted.extraction_timestamp,
            'total_extracted_params': self.extracted.total_extracted_params,
            'compression_ratio': self.extracted.compression_ratio,
            'entropy_profile': self.extracted.entropy_profile,
            'capability_zones': self.extracted.capability_zones,
            'n_layers': len(self.extracted.layers),
            'config': {
                'svd_rank': self.config.svd_rank,
                'attention_samples': self.config.attention_samples,
                'embedding_subspace_dim': self.config.embedding_subspace_dim,
            }
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Report file sizes
        pt_size = (output_path / "pac_structure.pt").stat().st_size
        json_size = (output_path / "metadata.json").stat().st_size
        total_size = pt_size + json_size
        
        print(f"  pac_structure.pt: {pt_size / 1024 / 1024:.2f} MB")
        print(f"  metadata.json: {json_size / 1024:.1f} KB")
        print(f"  Total: {total_size / 1024 / 1024:.2f} MB")
        print(f"  Source model size: ~{self.source_params * 4 / 1024 / 1024:.0f} MB")
        print(f"  Compression: {self.source_params * 4 / total_size:.1f}x")


def test_extraction():
    """Test V2 extraction on Pythia-70M."""
    config = ExtractionConfigV2(
        model_name="EleutherAI/pythia-70m",
        svd_rank=64,
        attention_samples=20,
    )
    
    extractor = ModelToPACExtractorV2(config)
    pac = extractor.extract()
    
    # Save
    output_dir = Path(__file__).parent / "extracted_v2" / "pythia_70m"
    extractor.save(str(output_dir))
    
    return pac


if __name__ == "__main__":
    test_extraction()
