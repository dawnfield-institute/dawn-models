"""
GAIA Geometric Training with PAC Tree Learnings
================================================

Train GAIA using extracted model patterns with geometric loss.
Loss measures how well GAIA's structure converges to source geometry.
NO BACKPROP - uses SEC-PAC dynamics only.
"""

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\fracton")

# Constants from PAC theory
PHI = (1 + np.sqrt(5)) / 2
XI_CRITICAL = 1.0571
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


@dataclass
class GeometricTarget:
    """Target geometry from source models"""
    layer_count: int
    embed_dim: int
    attention_heads: int
    layer_signatures: Dict[int, np.ndarray]
    embedding_clusters: int
    confluence_depth: int


@dataclass 
class GeometricLoss:
    """Geometric loss components"""
    layer_ratio: float
    dim_ratio: float
    pattern_similarity: float
    cluster_similarity: float
    confluence_match: float
    total: float


class SourceGeometryExtractor:
    """Extract target geometry from source models"""
    
    def __init__(self, device):
        self.device = device
        self.geometries = {}
        
    def extract_gpt2(self) -> GeometricTarget:
        """Extract GPT-2 geometry"""
        from transformers import GPT2Model
        model = GPT2Model.from_pretrained('gpt2')
        
        layer_sigs = {}
        for i, layer in enumerate(model.h):
            # Use layer norm as signature
            ln = layer.ln_1.weight.detach().cpu().numpy()
            layer_sigs[i] = ln
            
        del model
        torch.cuda.empty_cache()
            
        return GeometricTarget(
            layer_count=12,
            embed_dim=768,
            attention_heads=12,
            layer_signatures=layer_sigs,
            embedding_clusters=50,
            confluence_depth=4
        )
        
    def extract_pythia(self) -> GeometricTarget:
        """Extract Pythia geometry"""
        from transformers import GPTNeoXForCausalLM
        model = GPTNeoXForCausalLM.from_pretrained('EleutherAI/pythia-70m')
        
        layer_sigs = {}
        layers = model.gpt_neox.layers
        
        for i, layer in enumerate(layers):
            if hasattr(layer, 'input_layernorm'):
                ln = layer.input_layernorm.weight.detach().cpu().numpy()
                layer_sigs[i] = ln
                
        del model
        torch.cuda.empty_cache()
                
        return GeometricTarget(
            layer_count=6,
            embed_dim=512,
            attention_heads=8,
            layer_signatures=layer_sigs,
            embedding_clusters=30,
            confluence_depth=3
        )
        
    def extract_all(self) -> Dict[str, GeometricTarget]:
        """Extract geometry from all source models"""
        print("Extracting source model geometries...")
        
        try:
            self.geometries['gpt2'] = self.extract_gpt2()
            print(f"  GPT-2: {self.geometries['gpt2'].layer_count} layers, {self.geometries['gpt2'].embed_dim} dim")
        except Exception as e:
            print(f"  GPT-2 failed: {e}")
            
        try:
            self.geometries['pythia'] = self.extract_pythia()
            print(f"  Pythia: {self.geometries['pythia'].layer_count} layers, {self.geometries['pythia'].embed_dim} dim")
        except Exception as e:
            print(f"  Pythia failed: {e}")
            
        return self.geometries
        
    def compute_target(self) -> GeometricTarget:
        """Compute unified target geometry"""
        if not self.geometries:
            self.extract_all()
            
        avg_layers = int(np.mean([g.layer_count for g in self.geometries.values()]))
        avg_dim = int(np.mean([g.embed_dim for g in self.geometries.values()]))
        avg_heads = int(np.mean([g.attention_heads for g in self.geometries.values()]))
        
        # Find nearest Fibonacci for layers
        target_layers = min(FIBONACCI, key=lambda x: abs(x - avg_layers) if x >= avg_layers else float('inf'))
        
        print(f"\nTarget geometry (PAC-aligned):")
        print(f"  Layers: {target_layers} (from avg {avg_layers})")
        print(f"  Dim: {avg_dim}")
        print(f"  Heads: {avg_heads}")
        
        return GeometricTarget(
            layer_count=target_layers,
            embed_dim=avg_dim,
            attention_heads=avg_heads,
            layer_signatures={},
            embedding_clusters=40,
            confluence_depth=4
        )


class PACLazyLayer:
    """Lazy transformer layer - materializes on demand"""
    
    def __init__(self, dim: int, device: torch.device):
        self.dim = dim
        self.device = device
        self.materialized = False
        self.weights = None
        self.layer_norm = None
        self.pattern = None
        self.activations = 0
        
    def materialize(self, template: np.ndarray = None):
        """Materialize layer weights"""
        if self.materialized:
            return
            
        with torch.no_grad():
            scale = 1.0 / np.sqrt(self.dim)
            
            if template is not None:
                # Initialize from template
                t = torch.tensor(template, dtype=torch.float32, device=self.device)
                if len(t) < self.dim:
                    t = torch.cat([t, torch.zeros(self.dim - len(t), device=self.device)])
                elif len(t) > self.dim:
                    t = t[:self.dim]
                self.layer_norm = t
            else:
                self.layer_norm = torch.ones(self.dim, device=self.device)
                
            self.weights = torch.randn(self.dim, self.dim, device=self.device) * scale
            self.materialized = True
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        if not self.materialized:
            self.materialize()
            
        self.activations += 1
        
        with torch.no_grad():
            # Normalize
            normed = x * self.layer_norm
            # Transform
            out = torch.matmul(normed, self.weights.T)
            # Track pattern
            self.pattern = out.mean(dim=0).cpu().numpy()
            # Residual + activation
            return x + torch.relu(out)


class GAIAGeometricModel:
    """GAIA model with geometric growth"""
    
    def __init__(self, target: GeometricTarget, device: torch.device):
        self.target = target
        self.device = device
        
        # Start small
        self.embed_dim = min(256, target.embed_dim)
        self.max_layers = target.layer_count
        self.current_layers = 1
        
        # Embeddings
        self.vocab_size = 50257
        self.embeddings = torch.randn(self.vocab_size, self.embed_dim, device=device) * 0.02
        
        # Lazy layers
        self.layers = [PACLazyLayer(self.embed_dim, device) for _ in range(self.max_layers)]
        self.layers[0].materialize()
        
        # PAC confluence
        self.confluence = defaultdict(lambda: defaultdict(float))
        self.confluence_depth = 0
        
        # Output
        self.output_proj = torch.randn(self.embed_dim, self.vocab_size, device=device) * 0.02
        
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        with torch.no_grad():
            x = self.embeddings[input_ids]
            
            for i in range(self.current_layers):
                x = self.layers[i].forward(x)
                
            logits = torch.matmul(x, self.output_proj)
            return x, logits
            
    def compute_geometric_loss(self, source_geometries: Dict[str, GeometricTarget]) -> GeometricLoss:
        """Compute geometric loss"""
        
        # Layer ratio
        layer_ratio = self.current_layers / self.target.layer_count
        
        # Dim ratio
        dim_ratio = self.embed_dim / self.target.embed_dim
        
        # Pattern similarity to source layer signatures
        pattern_sims = []
        for i in range(self.current_layers):
            if self.layers[i].pattern is not None:
                for source in source_geometries.values():
                    if i in source.layer_signatures:
                        src_sig = source.layer_signatures[i]
                        my_pattern = self.layers[i].pattern
                        
                        # Flatten if 2D
                        if len(my_pattern.shape) > 1:
                            my_pattern = my_pattern.flatten()
                        if len(src_sig.shape) > 1:
                            src_sig = src_sig.flatten()
                        
                        min_len = min(len(src_sig), len(my_pattern))
                        if min_len > 1:
                            corr = np.corrcoef(src_sig[:min_len], my_pattern[:min_len])
                            if not np.isnan(corr[0, 1]):
                                pattern_sims.append(max(0, corr[0, 1]))
                                
        pattern_similarity = np.mean(pattern_sims) if pattern_sims else 0.0
        
        # Cluster similarity (embedding space structure)
        with torch.no_grad():
            sample = self.embeddings[:1000]
            norms = sample.norm(dim=1, keepdim=True)
            normalized = sample / (norms + 1e-8)
            sims = torch.matmul(normalized, normalized.T)
            clusters = (sims > 0.8).sum(dim=1).float().mean().item()
            cluster_similarity = 1.0 - abs(clusters - self.target.embedding_clusters) / max(self.target.embedding_clusters, 1)
            cluster_similarity = max(0, cluster_similarity)
            
        # Confluence match
        confluence_match = 1.0 - abs(self.confluence_depth - self.target.confluence_depth) / max(self.target.confluence_depth, 1)
        confluence_match = max(0, confluence_match)
        
        # Weighted total
        total = (
            0.25 * layer_ratio +
            0.15 * dim_ratio +
            0.3 * pattern_similarity +
            0.15 * cluster_similarity +
            0.15 * confluence_match
        )
        
        return GeometricLoss(
            layer_ratio=layer_ratio,
            dim_ratio=dim_ratio,
            pattern_similarity=pattern_similarity,
            cluster_similarity=cluster_similarity,
            confluence_match=confluence_match,
            total=total
        )
        
    def grow_layer(self, source_geometries: Dict[str, GeometricTarget]) -> bool:
        """Grow a new layer if needed"""
        if self.current_layers >= self.max_layers:
            return False
            
        next_idx = self.current_layers
        
        # Use source template if available
        template = None
        for source in source_geometries.values():
            if next_idx in source.layer_signatures:
                template = source.layer_signatures[next_idx]
                break
                
        self.layers[next_idx].materialize(template)
        self.current_layers += 1
        return True
        
    def update_confluence(self, context: Tuple[int, ...], next_token: int):
        """Update PAC confluence tree"""
        ctx_hash = hash(context) % 100000
        self.confluence[ctx_hash][next_token] += 1.0
        self.confluence_depth = max(self.confluence_depth, len(context))


class GAIAGeometricTrainer:
    """Train GAIA with geometric loss - NO BACKPROP"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        # Extract geometries
        self.extractor = SourceGeometryExtractor(self.device)
        self.source_geometries = self.extractor.extract_all()
        self.target = self.extractor.compute_target()
        
        # Create model
        self.model = GAIAGeometricModel(self.target, self.device)
        
        # Stats
        self.losses = []
        self.growths = []
        
    def transfer_embeddings(self):
        """Transfer embeddings from source models"""
        print("\nTransferring source embeddings...")
        
        embeddings = []
        
        try:
            from transformers import GPT2Model
            gpt2 = GPT2Model.from_pretrained('gpt2')
            gpt2_emb = gpt2.wte.weight.detach()[:self.model.vocab_size, :self.model.embed_dim]
            embeddings.append(gpt2_emb)
            print(f"  GPT-2: {gpt2_emb.shape}")
            del gpt2
        except Exception as e:
            print(f"  GPT-2 failed: {e}")
            
        try:
            from transformers import GPTNeoXForCausalLM
            pythia = GPTNeoXForCausalLM.from_pretrained('EleutherAI/pythia-70m')
            pythia_emb = pythia.gpt_neox.embed_in.weight.detach()[:self.model.vocab_size]
            if pythia_emb.shape[1] < self.model.embed_dim:
                pad = torch.zeros(pythia_emb.shape[0], self.model.embed_dim - pythia_emb.shape[1])
                pythia_emb = torch.cat([pythia_emb, pad], dim=1)
            else:
                pythia_emb = pythia_emb[:, :self.model.embed_dim]
            embeddings.append(pythia_emb)
            print(f"  Pythia: {pythia_emb.shape}")
            del pythia
        except Exception as e:
            print(f"  Pythia failed: {e}")
            
        torch.cuda.empty_cache()
            
        if embeddings:
            with torch.no_grad():
                avg = torch.stack(embeddings).mean(dim=0).to(self.device)
                self.model.embeddings[:avg.shape[0]] = avg
                print(f"  Transferred avg of {len(embeddings)} models")
                
    def train_epoch(self, texts: List[str], tokenizer) -> Tuple[GeometricLoss, Dict]:
        """Train one epoch - NO BACKPROP"""
        
        stats = {'tokens': 0, 'confluence_updates': 0}
        
        for text in texts:
            tokens = tokenizer.encode(text, max_length=128, truncation=True)
            if len(tokens) < 2:
                continue
                
            input_ids = torch.tensor([tokens[:-1]], device=self.device)
            
            # Forward
            hidden, logits = self.model.forward(input_ids)
            
            # PAC learning (no backprop)
            with torch.no_grad():
                for i in range(len(tokens) - 1):
                    context = tuple(tokens[max(0, i-4):i+1])
                    self.model.update_confluence(context, tokens[i + 1])
                    stats['confluence_updates'] += 1
                    
            stats['tokens'] += len(tokens)
            
        loss = self.model.compute_geometric_loss(self.source_geometries)
        return loss, stats
        
    def train(self, epochs: int = 15):
        """Full training loop"""
        
        print("\n" + "=" * 60)
        print("GAIA Geometric Training - NO BACKPROP")
        print("=" * 60)
        
        self.transfer_embeddings()
        
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        texts = [
            "The cat sat on the mat.",
            "The dog ran in the park.",
            "Birds fly in the sky.",
            "Fish swim in the water.",
            "The sun shines brightly during the day.",
            "The moon glows at night in the dark sky.",
            "Trees grow tall in the forest.",
            "Flowers bloom in spring when the weather warms.",
            "Mountains rise high above the clouds.",
            "Rivers flow to the sea carrying water downstream.",
            "Scientists study the natural world to understand it.",
            "Artists create beautiful works that inspire others.",
            "Musicians compose melodies that touch the soul.",
            "Writers tell stories that captivate readers.",
            "Engineers build machines that improve our lives.",
            "Teachers share knowledge with their students.",
            "Doctors heal the sick and help the injured.",
            "Farmers grow food to feed the population.",
            "The history of humanity is long and complex.",
            "Technology changes how we live and work.",
        ]
        
        print(f"\nTraining: {len(texts)} texts, {epochs} epochs")
        print(f"Target: {self.target.layer_count} layers, {self.target.embed_dim} dim")
        print(f"Start: {self.model.current_layers} layers, {self.model.embed_dim} dim")
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            
            loss, stats = self.train_epoch(texts, tokenizer)
            self.losses.append(loss)
            
            print(f"  Geometric Loss: {loss.total:.4f}")
            print(f"    Layer ratio:    {loss.layer_ratio:.3f} ({self.model.current_layers}/{self.target.layer_count})")
            print(f"    Dim ratio:      {loss.dim_ratio:.3f}")
            print(f"    Pattern sim:    {loss.pattern_similarity:.3f}")
            print(f"    Cluster sim:    {loss.cluster_similarity:.3f}")
            print(f"    Confluence:     {loss.confluence_match:.3f} (depth={self.model.confluence_depth})")
            print(f"  Tokens: {stats['tokens']}, Confluence updates: {stats['confluence_updates']}")
            
            # Grow based on geometric loss
            should_grow = (
                self.model.current_layers < self.target.layer_count and
                loss.layer_ratio < 0.9 and
                epoch > 0 and
                epoch % 3 == 0  # Grow every 3 epochs
            )
            
            if should_grow:
                if self.model.grow_layer(self.source_geometries):
                    print(f"  🌱 Grew to {self.model.current_layers} layers")
                    self.growths.append({'epoch': epoch, 'layers': self.model.current_layers})
                    
        return self.get_summary()
        
    def get_summary(self) -> Dict:
        """Get training summary"""
        final_loss = self.losses[-1] if self.losses else GeometricLoss(0,0,0,0,0,0)
        return {
            'final_layers': self.model.current_layers,
            'target_layers': self.target.layer_count,
            'final_dim': self.model.embed_dim,
            'target_dim': self.target.embed_dim,
            'confluence_depth': self.model.confluence_depth,
            'confluence_contexts': len(self.model.confluence),
            'growth_events': len(self.growths),
            'final_loss': {
                'total': final_loss.total,
                'layer_ratio': final_loss.layer_ratio,
                'pattern_sim': final_loss.pattern_similarity,
            }
        }
        
    def generate(self, prompt: str, max_len: int = 30) -> str:
        """Generate text"""
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        tokens = tokenizer.encode(prompt)
        
        with torch.no_grad():
            for _ in range(max_len):
                input_ids = torch.tensor([tokens[-32:]], device=self.device)
                hidden, logits = self.model.forward(input_ids)
                
                # Try confluence first
                ctx = tuple(tokens[-5:])
                ctx_hash = hash(ctx) % 100000
                
                if ctx_hash in self.model.confluence and self.model.confluence[ctx_hash]:
                    candidates = self.model.confluence[ctx_hash]
                    next_token = max(candidates, key=candidates.get)
                else:
                    # Sample from logits with temperature
                    probs = torch.softmax(logits[0, -1] / 0.8, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                    
                tokens.append(next_token)
                
                if next_token == tokenizer.eos_token_id:
                    break
                    
        return tokenizer.decode(tokens)
        
    def compare_geometry(self):
        """Compare final vs target geometry"""
        print("\n" + "=" * 60)
        print("GEOMETRY COMPARISON")
        print("=" * 60)
        
        print(f"\nGAIA Final:")
        print(f"  Layers: {self.model.current_layers}")
        print(f"  Dimension: {self.model.embed_dim}")
        print(f"  Confluence depth: {self.model.confluence_depth}")
        print(f"  Confluence contexts: {len(self.model.confluence)}")
        
        print(f"\nTarget:")
        print(f"  Layers: {self.target.layer_count}")
        print(f"  Dimension: {self.target.embed_dim}")
        print(f"  Confluence depth: {self.target.confluence_depth}")
        
        print(f"\nSource Models:")
        for name, geom in self.source_geometries.items():
            ratio_l = self.model.current_layers / geom.layer_count
            ratio_d = self.model.embed_dim / geom.embed_dim
            print(f"  vs {name}: layers={ratio_l:.2f}x, dim={ratio_d:.2f}x")
            
        print(f"\nFibonacci Check:")
        for i, fib in enumerate(FIBONACCI[:10]):
            if self.model.current_layers == fib:
                print(f"  ✓ {self.model.current_layers} layers = F({i})")
                break
        else:
            print(f"  ○ {self.model.current_layers} layers not Fibonacci")
            
        # Per-layer stats
        print(f"\nLayer Activations:")
        for i in range(self.model.current_layers):
            layer = self.model.layers[i]
            print(f"  Layer {i}: {layer.activations} activations")


def main():
    print("=" * 60)
    print("GAIA Geometric Training with PAC Learnings")
    print("NO BACKPROP - Geometric Loss Only")
    print("=" * 60)
    
    trainer = GAIAGeometricTrainer()
    summary = trainer.train(epochs=30)  # More epochs for better convergence
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Layers: {summary['final_layers']} / {summary['target_layers']}")
    print(f"  Dim: {summary['final_dim']} / {summary['target_dim']}")
    print(f"  Confluence: {summary['confluence_contexts']} contexts, depth {summary['confluence_depth']}")
    print(f"  Growths: {summary['growth_events']}")
    print(f"  Final loss: {summary['final_loss']['total']:.4f}")
    
    trainer.compare_geometry()
    
    print("\n" + "=" * 60)
    print("GENERATION TEST")
    print("=" * 60)
    
    prompts = ["The cat", "Scientists", "The sun", "Birds fly", "In the future"]
    
    for prompt in prompts:
        out = trainer.generate(prompt, max_len=25)
        print(f"\n'{prompt}' → {out}")
        
    # Save
    output_path = Path(__file__).parent.parent / "results" / "gaia_geometric_training.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {output_path}")
    
    return trainer


if __name__ == "__main__":
    trainer = main()
