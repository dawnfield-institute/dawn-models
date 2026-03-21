"""
PAC-Lazy Knowledge Extractor V3
================================

Extracts learned knowledge from trained models into PAC-Lazy format.

What we're extracting:
1. Token embeddings → vocab_deltas for PACLazyTransformer
2. Attention patterns → neighbor link weights
3. Layer transformations → SEC expansion templates
4. Position encodings → causal sequence biases

This integrates with POC-011's PACLazyTransformer infrastructure.

Output format:
- pac_vocab.pt: Token embeddings as vocab_deltas
- pac_structure.pt: Attention pattern weights, layer transforms
- pac_metadata.json: Source info, capability zones, entropy profile
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
import time
import sys
import os

# Add POC-011 to path for PAC-Lazy imports
poc_011_path = Path(__file__).parent.parent / "poc_011_pac_lazy_transformer" / "scripts"
if poc_011_path.exists():
    sys.path.insert(0, str(poc_011_path))

# Dawn Field Constants
PHI = 1.618033988749895
XI = 0.0618
PHI_XI = 1.710
LAMBDA_STAR = 0.9816


@dataclass  
class PACLazyExtractionConfig:
    """Configuration for PAC-Lazy extraction."""
    model_name: str = "EleutherAI/pythia-70m"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Attention pattern sampling
    attention_samples: int = 100
    max_seq_len: int = 64
    
    # What to extract
    extract_embeddings: bool = True
    extract_attention_patterns: bool = True
    extract_mlp_templates: bool = True
    
    # Compression (for MLP templates)
    svd_rank: int = 64


class PACLazyExtractor:
    """
    Extracts knowledge from trained models into PAC-Lazy format.
    
    The output can be directly loaded into PACLazyTransformer:
    - vocab_deltas: Dictionary of token_id -> delta tensor
    - neighbor_weights: Attention-derived causal link weights
    - expansion_templates: MLP-derived SEC expansion patterns
    """
    
    def __init__(self, config: PACLazyExtractionConfig):
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
        print(f"  Vocab size: {len(self.tokenizer):,}")
        
        # Extraction results
        self.vocab_deltas: Dict[int, torch.Tensor] = {}
        self.attention_patterns: List[torch.Tensor] = []
        self.mlp_templates: List[Dict[str, torch.Tensor]] = []
        self.entropy_profile: Dict[str, float] = {}
        
    def extract(self):
        """Main extraction pipeline."""
        start_time = time.time()
        
        print("\n" + "="*70)
        print("PAC-LAZY KNOWLEDGE EXTRACTION")
        print("="*70)
        
        # 1. Extract token embeddings as vocab_deltas
        if self.config.extract_embeddings:
            print("\n[1/4] Extracting token embeddings as vocab_deltas...")
            self._extract_embeddings()
        
        # 2. Extract attention patterns (neighbor link weights)
        if self.config.extract_attention_patterns:
            print("\n[2/4] Extracting attention patterns...")
            self._extract_attention_patterns()
        
        # 3. Extract MLP templates (SEC expansion)
        if self.config.extract_mlp_templates:
            print("\n[3/4] Extracting MLP expansion templates...")
            self._extract_mlp_templates()
        
        # 4. Analyze entropy structure
        print("\n[4/4] Analyzing entropy collapse structure...")
        self._analyze_entropy()
        
        elapsed = time.time() - start_time
        
        # Summary
        total_params = sum(t.numel() for t in self.vocab_deltas.values())
        total_params += sum(t.numel() for t in self.attention_patterns)
        for tmpl in self.mlp_templates:
            total_params += sum(t.numel() for t in tmpl.values())
        
        print(f"\n{'='*70}")
        print("✅ EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Vocab deltas: {len(self.vocab_deltas):,} tokens")
        print(f"  Attention patterns: {len(self.attention_patterns)} layers")
        print(f"  MLP templates: {len(self.mlp_templates)} layers")
        print(f"  Total extracted: {total_params:,} parameters")
        print(f"  Compression: {self.source_params / total_params:.1f}x")
        
        return self
    
    def _extract_embeddings(self):
        """Extract token embeddings as vocab_deltas."""
        # Find embedding layer
        if hasattr(self.model, 'gpt_neox'):
            embed_weight = self.model.gpt_neox.embed_in.weight.data
        elif hasattr(self.model, 'transformer'):
            embed_weight = self.model.transformer.wte.weight.data
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            embed_weight = self.model.model.embed_tokens.weight.data
        else:
            for name, param in self.model.named_parameters():
                if 'embed' in name.lower() and 'weight' in name.lower():
                    embed_weight = param.data
                    break
            else:
                raise ValueError("Could not find embedding matrix")
        
        # Convert to vocab_deltas dictionary
        vocab_size = embed_weight.shape[0]
        embed_dim = embed_weight.shape[1]
        
        print(f"  Embedding shape: {embed_weight.shape}")
        print(f"  Converting {vocab_size:,} tokens to vocab_deltas...")
        
        # Store each token's embedding as a delta
        for token_id in range(vocab_size):
            self.vocab_deltas[token_id] = embed_weight[token_id].cpu().clone()
        
        print(f"  ✓ Extracted {len(self.vocab_deltas):,} vocab_deltas")
        print(f"  ✓ Delta dimension: {embed_dim}")
    
    def _extract_attention_patterns(self):
        """Extract attention patterns as neighbor link weights."""
        print(f"  Running {self.config.attention_samples} forward passes...")
        
        # Diverse prompts
        prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "In the beginning, there was light and darkness.",
            "The function returns the sum of all elements in the array.",
            "Once upon a time in a land far away, there lived a princess.",
            "To understand recursion, you must first understand recursion.",
            "The weather today is sunny with a chance of rain later.",
            "Machine learning models learn patterns from data.",
            "The capital of France is Paris, known for the Eiffel Tower.",
        ]
        
        # Accumulate attention patterns per layer
        layer_patterns = {}
        
        for i, prompt in enumerate(prompts * (self.config.attention_samples // len(prompts) + 1)):
            if i >= self.config.attention_samples:
                break
                
            inputs = self.tokenizer(
                prompt, 
                return_tensors='pt', 
                truncation=True, 
                max_length=self.config.max_seq_len
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            
            if outputs.attentions:
                for layer_idx, attn in enumerate(outputs.attentions):
                    # attn shape: [batch, heads, seq, seq]
                    # Average over batch and heads
                    attn_pattern = attn.mean(dim=(0, 1)).cpu()
                    
                    if layer_idx not in layer_patterns:
                        layer_patterns[layer_idx] = []
                    layer_patterns[layer_idx].append(attn_pattern)
        
        # Average patterns per layer
        for layer_idx in sorted(layer_patterns.keys()):
            patterns = layer_patterns[layer_idx]
            # Pad to same size and average
            max_len = max(p.shape[0] for p in patterns)
            padded = []
            for p in patterns:
                if p.shape[0] < max_len:
                    pad = torch.zeros(max_len, max_len)
                    pad[:p.shape[0], :p.shape[1]] = p
                    padded.append(pad)
                else:
                    padded.append(p[:max_len, :max_len])
            
            mean_pattern = torch.stack(padded).mean(dim=0)
            self.attention_patterns.append(mean_pattern)
        
        print(f"  ✓ Extracted {len(self.attention_patterns)} attention pattern matrices")
        if self.attention_patterns:
            print(f"  ✓ Pattern shape: {self.attention_patterns[0].shape}")
    
    def _extract_mlp_templates(self):
        """Extract MLP weight templates via SVD."""
        print(f"  Extracting with SVD rank {self.config.svd_rank}...")
        
        # Find transformer layers
        if hasattr(self.model, 'gpt_neox'):
            layers = self.model.gpt_neox.layers
        elif hasattr(self.model, 'transformer'):
            layers = self.model.transformer.h
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        else:
            print("  ⚠ Could not find transformer layers for MLP extraction")
            return
        
        n_layers = len(layers)
        rank = self.config.svd_rank
        
        for idx, layer in enumerate(layers):
            template = {}
            
            # Find MLP weights
            for name, param in layer.named_parameters():
                name_lower = name.lower()
                
                # Up projection
                if any(x in name_lower for x in ['dense_h_to_4h', 'fc1', 'c_fc', 'up_proj']):
                    if 'weight' in name_lower:
                        W = param.data.float()
                        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                        template['up_U'] = U[:, :rank].cpu()
                        template['up_S'] = S[:rank].cpu()
                        template['up_Vh'] = Vh[:rank, :].cpu()
                
                # Down projection  
                if any(x in name_lower for x in ['dense_4h_to_h', 'fc2', 'c_proj', 'down_proj']):
                    if 'weight' in name_lower:
                        W = param.data.float()
                        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                        template['down_U'] = U[:, :rank].cpu()
                        template['down_S'] = S[:rank].cpu()
                        template['down_Vh'] = Vh[:rank, :].cpu()
            
            if template:
                self.mlp_templates.append(template)
        
        print(f"  ✓ Extracted {len(self.mlp_templates)} MLP templates")
    
    def _analyze_entropy(self):
        """Analyze entropy collapse structure."""
        # Compute entropy of embedding space
        if self.vocab_deltas:
            sample_size = min(1000, len(self.vocab_deltas))
            sample_ids = list(self.vocab_deltas.keys())[:sample_size]
            embeddings = torch.stack([self.vocab_deltas[i] for i in sample_ids])
            
            # Compute entropy via singular value distribution
            U, S, Vh = torch.linalg.svd(embeddings, full_matrices=False)
            S_norm = S / S.sum()
            embed_entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-10)).item()
            
            self.entropy_profile['embedding_entropy'] = embed_entropy
            self.entropy_profile['effective_rank'] = (S > S.max() * 0.01).sum().item()
        
        # Compute entropy of attention patterns
        for idx, pattern in enumerate(self.attention_patterns):
            # Flatten and compute entropy
            flat = pattern.flatten()
            flat = flat / (flat.sum() + 1e-10)
            attn_entropy = -torch.sum(flat * torch.log(flat + 1e-10)).item()
            self.entropy_profile[f'attention_layer_{idx}'] = attn_entropy
        
        print(f"  ✓ Computed entropy for {len(self.entropy_profile)} components")
        if 'embedding_entropy' in self.entropy_profile:
            print(f"  ✓ Embedding entropy: {self.entropy_profile['embedding_entropy']:.3f}")
            print(f"  ✓ Effective rank: {self.entropy_profile.get('effective_rank', 'N/A')}")
    
    def save(self, output_dir: str):
        """Save extraction to disk in PAC-Lazy format."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving to {output_path}...")
        
        # 1. Save vocab_deltas - THE KEY EXTRACTION
        # This is what makes the knowledge transfer work!
        vocab_tensor = torch.stack([self.vocab_deltas[i] for i in range(len(self.vocab_deltas))])
        torch.save({
            'vocab_deltas': vocab_tensor,
            'vocab_size': len(self.vocab_deltas),
            'embed_dim': vocab_tensor.shape[1],
        }, output_path / "pac_vocab.pt")
        
        vocab_size_mb = vocab_tensor.numel() * 4 / 1024 / 1024
        print(f"  pac_vocab.pt: {vocab_size_mb:.2f} MB ({len(self.vocab_deltas):,} tokens)")
        
        # 2. Save attention patterns
        if self.attention_patterns:
            torch.save({
                'patterns': self.attention_patterns,
                'n_layers': len(self.attention_patterns),
            }, output_path / "pac_attention.pt")
            
            attn_size = sum(p.numel() for p in self.attention_patterns) * 4 / 1024 / 1024
            print(f"  pac_attention.pt: {attn_size:.2f} MB ({len(self.attention_patterns)} layers)")
        
        # 3. Save MLP templates
        if self.mlp_templates:
            torch.save({
                'templates': self.mlp_templates,
                'n_layers': len(self.mlp_templates),
                'svd_rank': self.config.svd_rank,
            }, output_path / "pac_mlp.pt")
            
            mlp_size = sum(
                sum(t.numel() for t in tmpl.values()) 
                for tmpl in self.mlp_templates
            ) * 4 / 1024 / 1024
            print(f"  pac_mlp.pt: {mlp_size:.2f} MB ({len(self.mlp_templates)} layers)")
        
        # 4. Save metadata
        metadata = {
            'source_model': self.config.model_name,
            'source_params': self.source_params,
            'extraction_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'vocab_size': len(self.vocab_deltas),
            'embed_dim': self.vocab_deltas[0].shape[0] if self.vocab_deltas else 0,
            'n_attention_layers': len(self.attention_patterns),
            'n_mlp_layers': len(self.mlp_templates),
            'svd_rank': self.config.svd_rank,
            'entropy_profile': self.entropy_profile,
            'config': {
                'attention_samples': self.config.attention_samples,
                'max_seq_len': self.config.max_seq_len,
            }
        }
        
        with open(output_path / "pac_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  pac_metadata.json: saved")
        
        # Total size
        total_size_mb = vocab_size_mb
        if self.attention_patterns:
            total_size_mb += attn_size
        if self.mlp_templates:
            total_size_mb += mlp_size
        
        print(f"\n  Total extracted: {total_size_mb:.2f} MB")
        print(f"  Source model: ~{self.source_params * 4 / 1024 / 1024:.0f} MB")
        print(f"  Compression: {self.source_params * 4 / 1024 / 1024 / total_size_mb:.1f}x")


def main():
    """Run PAC-Lazy extraction on Pythia-70M."""
    config = PACLazyExtractionConfig(
        model_name="EleutherAI/pythia-70m",
        attention_samples=50,
        max_seq_len=64,
        svd_rank=64,
    )
    
    extractor = PACLazyExtractor(config)
    extractor.extract()
    
    # Save to extracted_v3 directory
    output_dir = Path(__file__).parent / "extracted_v3" / "pythia_70m"
    extractor.save(str(output_dir))


if __name__ == "__main__":
    main()
