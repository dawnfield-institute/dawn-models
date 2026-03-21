"""
PAC-Lazy Transformer with Full Knowledge Import
================================================

This version imports BOTH:
1. vocab_deltas - WHAT tokens mean (embeddings)
2. navigation structure - HOW to use that info (attention patterns → neighbor weights)

The key insight: Attention patterns encode which positions should influence which.
We import this as the causal link structure of the PAC tree.

Instead of uniform causal propagation, we use Pythia's attention patterns
to weight which nodes should propagate to which.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque
import sys
import json
import math

# Add POC-011 scripts to path
poc_011_path = Path(__file__).parent.parent / "poc_011_pac_lazy_transformer" / "scripts"
sys.path.insert(0, str(poc_011_path))

# Dawn Field Constants
PHI = 1.618033988749895
XI = 0.0618
PHI_XI = 1.710
LAMBDA_STAR = 0.9816


@dataclass
class NavigationNode:
    """A PAC node with navigation weights (from attention patterns)."""
    nid: str
    delta: torch.Tensor = None  # What this token represents
    potential: float = 0.0
    
    # Navigation structure (from attention)
    neighbor_weights: Dict[str, float] = field(default_factory=dict)  # nid -> attention weight
    
    # Hierarchical structure (for SEC)
    children: List[str] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)
    
    # State
    active: bool = False
    position: int = 0  # Position in sequence
    token_id: Optional[int] = None


class NavigablePACSystem:
    """
    PAC system with learned navigation weights.
    
    Key difference from vanilla PAC-Lazy:
    - Propagation uses LEARNED attention weights, not uniform decay
    - Neighbor structure reflects what Pythia learned about language
    """
    
    def __init__(self, 
                 embed_dim: int = 512,
                 total_potential: float = 100.0,
                 device: str = 'cuda'):
        
        self.embed_dim = embed_dim
        self.total_potential = total_potential
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.nodes: Dict[str, NavigationNode] = {}
        self.active_nodes: Set[str] = set()
        self.allocated_potential = 0.0
        
        # Imported navigation patterns (from attention)
        self.attention_patterns: List[torch.Tensor] = []
        self.n_layers = 0
        
        # Imported expansion patterns (from MLP)
        self.mlp_templates: List[Dict[str, torch.Tensor]] = []
        
    def load_navigation(self, pac_path: Path):
        """Load attention patterns as navigation structure."""
        attn_file = pac_path / "pac_attention.pt"
        
        if attn_file.exists():
            data = torch.load(attn_file, weights_only=True)
            self.attention_patterns = data['patterns']
            self.n_layers = len(self.attention_patterns)
            print(f"  ✓ Loaded {self.n_layers} attention navigation layers")
        
        mlp_file = pac_path / "pac_mlp.pt"
        if mlp_file.exists():
            data = torch.load(mlp_file, weights_only=True)
            self.mlp_templates = data['templates']
            print(f"  ✓ Loaded {len(self.mlp_templates)} MLP expansion templates")
    
    def add_node(self, nid: str, position: int, token_id: int = None) -> NavigationNode:
        """Add a node at a specific position."""
        node = NavigationNode(nid=nid, position=position, token_id=token_id)
        node.delta = torch.zeros(self.embed_dim, device=self.device)
        self.nodes[nid] = node
        return node
    
    def inject_delta(self, nid: str, delta: torch.Tensor, 
                    potential_cost: float = 1.0) -> bool:
        """Inject a delta into a node."""
        if self.allocated_potential + potential_cost > self.total_potential:
            return False
        
        node = self.nodes[nid]
        node.delta = node.delta + delta.to(self.device)
        node.potential += potential_cost
        node.active = True
        
        self.active_nodes.add(nid)
        self.allocated_potential += potential_cost
        return True
    
    def build_navigation_links(self, sequence_nodes: List[str]):
        """
        Build navigation links between nodes using attention patterns.
        
        This is the KEY import: we're using Pythia's learned attention
        to define which nodes should propagate to which.
        """
        seq_len = len(sequence_nodes)
        
        if not self.attention_patterns or seq_len == 0:
            # Fall back to simple causal links
            for i, nid in enumerate(sequence_nodes[1:], 1):
                prev_nid = sequence_nodes[i-1]
                self.nodes[nid].neighbor_weights[prev_nid] = 0.5
                self.nodes[prev_nid].neighbor_weights[nid] = 0.5
            return
        
        # Use averaged attention pattern across layers
        # Shape: [max_len, max_len]
        combined_pattern = torch.zeros(seq_len, seq_len)
        
        for layer_pattern in self.attention_patterns:
            # Interpolate pattern to match sequence length
            if layer_pattern.shape[0] != seq_len:
                # Resize pattern to match
                pattern = F.interpolate(
                    layer_pattern.unsqueeze(0).unsqueeze(0),
                    size=(seq_len, seq_len),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
            else:
                pattern = layer_pattern
            
            combined_pattern += pattern
        
        combined_pattern /= len(self.attention_patterns)
        
        # Apply causal mask (can only attend to past)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        combined_pattern = combined_pattern * causal_mask
        
        # Normalize each row
        row_sums = combined_pattern.sum(dim=1, keepdim=True)
        combined_pattern = combined_pattern / (row_sums + 1e-10)
        
        # Set neighbor weights based on attention
        threshold = 0.05  # Only keep significant attention
        
        for i, to_nid in enumerate(sequence_nodes):
            for j, from_nid in enumerate(sequence_nodes):
                if i != j:
                    weight = combined_pattern[i, j].item()
                    if weight > threshold:
                        self.nodes[to_nid].neighbor_weights[from_nid] = weight
        
        # Count links
        total_links = sum(len(n.neighbor_weights) for n in self.nodes.values())
        print(f"  ✓ Built {total_links} weighted navigation links")
    
    def propagate_with_navigation(self) -> torch.Tensor:
        """
        Propagate using learned attention weights.
        
        Instead of uniform decay, each node receives weighted
        contributions from its neighbors based on attention patterns.
        """
        if not self.active_nodes:
            return None
        
        # Collect contributions for each active node
        new_deltas = {}
        
        for nid in self.active_nodes:
            node = self.nodes[nid]
            
            # Sum weighted contributions from neighbors
            contribution = torch.zeros(self.embed_dim, device=self.device)
            total_weight = 0.0
            
            for neighbor_nid, weight in node.neighbor_weights.items():
                if neighbor_nid in self.nodes:
                    neighbor = self.nodes[neighbor_nid]
                    contribution += weight * neighbor.delta
                    total_weight += weight
            
            if total_weight > 0:
                contribution = contribution / total_weight
                new_deltas[nid] = node.delta + 0.5 * contribution  # Blend with current
        
        # Apply updates
        for nid, delta in new_deltas.items():
            self.nodes[nid].delta = delta
        
        # Compose final representation from all active nodes
        if self.active_nodes:
            active_deltas = [self.nodes[nid].delta for nid in self.active_nodes]
            # Position-weighted average (recent positions matter more)
            positions = [self.nodes[nid].position for nid in self.active_nodes]
            max_pos = max(positions) if positions else 1
            weights = torch.tensor([(p + 1) / (max_pos + 1) for p in positions], device=self.device)
            weights = weights / weights.sum()
            
            composed = torch.stack(active_deltas) * weights.unsqueeze(1)
            return composed.sum(dim=0)
        
        return None
    
    def apply_mlp_transform(self, delta: torch.Tensor) -> torch.Tensor:
        """Apply MLP transformation using imported templates."""
        if not self.mlp_templates:
            return delta
        
        # Use first template (could layer them for depth)
        template = self.mlp_templates[0]
        
        if 'up_U' in template and 'down_U' in template:
            # Reconstruct MLP: up then down
            up_W = template['up_U'].to(self.device) @ torch.diag(template['up_S'].to(self.device)) @ template['up_Vh'].to(self.device)
            down_W = template['down_U'].to(self.device) @ torch.diag(template['down_S'].to(self.device)) @ template['down_Vh'].to(self.device)
            
            # Apply: x -> up -> GELU -> down -> residual
            h = delta @ up_W.T  # [embed] @ [4h, embed].T -> [4h]
            h = F.gelu(h)
            out = h @ down_W.T  # [4h] @ [embed, 4h].T -> [embed]
            
            return delta + out  # Residual connection
        
        return delta


class NavigablePACTransformer:
    """
    PAC Transformer that uses learned navigation.
    
    Combines:
    - Imported embeddings (what tokens mean)
    - Imported attention (which tokens to look at)
    - Imported MLP (how to transform representations)
    """
    
    def __init__(self, pac_path: Path, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.pac_path = pac_path
        
        # Load vocab
        vocab_data = torch.load(pac_path / "pac_vocab.pt", weights_only=True)
        self.vocab_deltas = vocab_data['vocab_deltas'].to(self.device)
        self.embed_dim = self.vocab_deltas.shape[1]
        self.vocab_size = self.vocab_deltas.shape[0]
        
        print(f"  Loaded vocab: {self.vocab_size:,} tokens x {self.embed_dim} dim")
        
        # Create navigable PAC system
        self.system = NavigablePACSystem(
            embed_dim=self.embed_dim,
            total_potential=500.0,
            device=device
        )
        self.system.load_navigation(pac_path)
        
        # Sequence tracking
        self.sequence_nodes: List[str] = []
        self.next_node_id = 0
    
    def reset(self):
        """Reset for new sequence."""
        self.system.nodes.clear()
        self.system.active_nodes.clear()
        self.system.allocated_potential = 0.0
        self.sequence_nodes = []
    
    def process_sequence(self, token_ids: List[int]) -> torch.Tensor:
        """
        Process a sequence of tokens.
        
        1. Create nodes for each token
        2. Inject embeddings as deltas
        3. Build navigation links from attention patterns
        4. Propagate with navigation weights
        5. Apply MLP transform
        """
        self.reset()
        
        # Create nodes for each token
        for i, tid in enumerate(token_ids):
            nid = f"tok_{self.next_node_id}"
            self.next_node_id += 1
            
            node = self.system.add_node(nid, position=i, token_id=tid)
            
            # Inject embedding as delta
            if tid < self.vocab_size:
                embedding = self.vocab_deltas[tid]
            else:
                embedding = torch.randn(self.embed_dim, device=self.device)
            
            self.system.inject_delta(nid, embedding, potential_cost=1.0)
            self.sequence_nodes.append(nid)
        
        # Build navigation links from attention patterns
        self.system.build_navigation_links(self.sequence_nodes)
        
        # Propagate using navigation weights
        composed = self.system.propagate_with_navigation()
        
        # Apply MLP transform
        if composed is not None:
            composed = self.system.apply_mlp_transform(composed)
        
        return composed
    
    def predict_next(self, token_ids: List[int], top_k: int = 10) -> List[Tuple[int, float, str]]:
        """
        Predict next token using full navigation.
        
        Returns list of (token_id, score, token_str) tuples.
        """
        # Process sequence
        representation = self.process_sequence(token_ids)
        
        if representation is None:
            return []
        
        # Normalize representation
        representation = F.normalize(representation.unsqueeze(0), dim=1).squeeze()
        
        # Score against all vocab
        vocab_norm = F.normalize(self.vocab_deltas, dim=1)
        scores = vocab_norm @ representation
        
        # Get top-k
        top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
        
        return [(idx.item(), score.item()) for idx, score in zip(top_indices, top_scores)]
    
    def generate(self, token_ids: List[int], max_new_tokens: int = 20,
                temperature: float = 0.8, top_k: int = 50) -> List[int]:
        """Generate continuation."""
        generated = list(token_ids)
        
        for _ in range(max_new_tokens):
            predictions = self.predict_next(generated, top_k=top_k)
            
            if not predictions:
                break
            
            # Temperature scaling and sampling
            scores = torch.tensor([p[1] for p in predictions]) / temperature
            probs = F.softmax(scores, dim=0)
            idx = torch.multinomial(probs, 1).item()
            
            next_token = predictions[idx][0]
            generated.append(next_token)
        
        return generated


def test_navigable_transformer():
    """Test the navigable PAC transformer."""
    from transformers import GPT2Tokenizer
    
    print("="*70)
    print("NAVIGABLE PAC TRANSFORMER TEST")
    print("="*70)
    
    # Path to extraction
    pac_path = Path(__file__).parent.parent / "poc_016_pac_extraction" / "extracted_v3" / "pythia_70m"
    
    if not pac_path.exists():
        print(f"❌ PAC not found at {pac_path}")
        return
    
    # Create navigable transformer
    print("\nLoading navigable transformer...")
    model = NavigablePACTransformer(pac_path)
    
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Test predictions
    prompts = [
        "The weather today is",
        "The meaning of life is",
        "Once upon a time",
        "In the beginning",
        "To be or not to",
    ]
    
    print("\n" + "="*70)
    print("GENERATION WITH NAVIGATION")
    print("="*70)
    
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt)
        
        print(f"\nPrompt: '{prompt}'")
        
        # Get predictions
        predictions = model.predict_next(token_ids, top_k=5)
        
        print("  Top 5 predictions:")
        for i, (tid, score) in enumerate(predictions[:5]):
            token = tokenizer.decode([tid])
            print(f"    {i+1}. '{token}' (score={score:.4f})")
        
        # Generate
        generated_ids = model.generate(token_ids, max_new_tokens=10, temperature=0.7)
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"  Generated: {generated_text}")
    
    # Compare to vanilla (no navigation)
    print("\n" + "="*70)
    print("COMPARISON: With Navigation vs Without")
    print("="*70)
    
    test_prompt = "The weather today"
    token_ids = tokenizer.encode(test_prompt)
    
    # With navigation (our model)
    nav_preds = model.predict_next(token_ids, top_k=5)
    
    # Without navigation (just embedding similarity)
    last_embed = model.vocab_deltas[token_ids[-1]]
    last_embed_norm = F.normalize(last_embed.unsqueeze(0), dim=1).squeeze()
    vocab_norm = F.normalize(model.vocab_deltas, dim=1)
    no_nav_scores = vocab_norm @ last_embed_norm
    top_no_nav = torch.topk(no_nav_scores, 5)
    
    print(f"\nPrompt: '{test_prompt}'")
    print("\nWith Navigation (context-aware):")
    for i, (tid, score) in enumerate(nav_preds[:5]):
        token = tokenizer.decode([tid])
        print(f"  {i+1}. '{token}' ({score:.4f})")
    
    print("\nWithout Navigation (last-token only):")
    for i, (score, idx) in enumerate(zip(top_no_nav.values, top_no_nav.indices)):
        token = tokenizer.decode([idx.item()])
        print(f"  {i+1}. '{token}' ({score.item():.4f})")


if __name__ == "__main__":
    test_navigable_transformer()
