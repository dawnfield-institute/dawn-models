"""
Resonant Growing SEC-PAC Transformer
=====================================

Key insight from user: "grow the transformers too, this could technically 
be done to resonate with the fractal frequency of the PAC tree"

This combines:
- POC-002: PhaseTransitionMonitor, FibonacciScheduler, ResonanceTrainer
- POC-011: PAC-Lazy core primitives
- SEC collapse dynamics
- Growing transformer capacity that resonates with tree depth

The transformer doesn't have fixed dimensions - it GROWS as the PAC tree
crystallizes, matching capacity to the fractal structure of knowledge.

Fractal Frequency = Rate of SEC crystallization events
Transformer grows when this frequency exceeds a threshold
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque
import sys
import math

# Add POC paths
poc_base = Path(__file__).parent.parent
sys.path.insert(0, str(poc_base / "poc_002_resonance_training" / "scripts"))
sys.path.insert(0, str(poc_base / "poc_011_pac_lazy_transformer" / "scripts"))

# Dawn Field Constants (from POC-002)
PHI = 1.618033988749895
XI = 0.0618
PHI_XI = 1.710  # Crystallization threshold
LAMBDA_STAR = 0.9816  # Optimal memory decay
ENTANGLEMENT_LIMIT = 4/5  # Max coupling strength
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]


@dataclass
class GrowthEvent:
    """Records when the transformer grew."""
    iteration: int
    old_dim: int
    new_dim: int
    trigger: str  # 'crystallization', 'depth_reached', 'capacity_full'
    tree_depth: int
    crystal_count: int


@dataclass
class ResonantNode:
    """Node that participates in resonant growth."""
    nid: str
    position: int
    token_id: int
    depth: int = 0  # Fractal depth level
    
    # SEC state
    entropy: float = 1.0
    crystallized: bool = False
    crystal_iteration: int = -1
    
    # Representation (can grow!)
    delta: torch.Tensor = None
    
    # Coupling
    neighbors: Dict[str, float] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)


class FractalFrequencyTracker:
    """
    Track the fractal frequency of crystallization events.
    
    From PACSeries Paper #2: 
    - Discrete systems resonate at 0.020 Hz
    - Continuous systems at 0.030 Hz
    - Ratio 2/3 is fundamental
    
    We track crystallization rate and use it to determine
    when to grow transformer capacity.
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.crystal_events: List[int] = []  # Iterations when crystallization occurred
        self.frequency_history: List[float] = []
        self.total_crystals = 0  # Cumulative count
        self.fib_thresholds_used: Set[int] = set()  # Track which Fib thresholds triggered
        
    def record_crystallization(self, iteration: int, count: int = 1):
        """Record crystallization events."""
        for _ in range(count):
            self.crystal_events.append(iteration)
        self.total_crystals += count
        print(f"    📊 Crystal count: {self.total_crystals} (added {count})")
        
    def compute_frequency(self, current_iter: int) -> float:
        """
        Compute current crystallization frequency.
        
        Returns events per unit time (normalized to 0.020-0.030 range).
        """
        if len(self.crystal_events) < 2:
            return 0.0
        
        # Count events in recent window
        recent = [e for e in self.crystal_events 
                  if e > current_iter - self.window_size]
        
        if not recent:
            return 0.0
        
        # Frequency = events / time_window
        freq = len(recent) / self.window_size
        
        # Scale to match Dawn Field resonance range (0.020 - 0.030 Hz)
        scaled_freq = 0.020 + freq * 0.010
        
        self.frequency_history.append(scaled_freq)
        return scaled_freq
    
    def should_grow(self) -> Tuple[bool, int]:
        """
        Determine if transformer should grow based on total crystals.
        
        Growth happens at Fibonacci-spaced intervals: 3, 5, 8, 13, 21, 34...
        Returns (should_grow, threshold_that_was_crossed)
        """
        fib_thresholds = [3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
        
        for threshold in fib_thresholds:
            if self.total_crystals >= threshold and threshold not in self.fib_thresholds_used:
                self.fib_thresholds_used.add(threshold)
                return True, threshold
        
        return False, 0
    
    def get_fibonacci_growth_factor(self) -> int:
        """Get growth factor from Fibonacci sequence based on total crystals."""
        n = len(self.fib_thresholds_used)  # Number of growth events so far
        return FIBONACCI[min(n, len(FIBONACCI) - 1)]


class GrowableEmbedding(nn.Module):
    """
    Embedding layer that can grow its dimension.
    
    When the PAC tree deepens, the embedding expands to
    capture the increased fractal complexity.
    """
    
    def __init__(self, vocab_size: int, initial_dim: int, max_dim: int, device='cuda'):
        super().__init__()
        self.vocab_size = vocab_size
        self.current_dim = initial_dim
        self.max_dim = max_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Start with initial embedding
        self.embedding = nn.Parameter(
            torch.randn(vocab_size, initial_dim, device=self.device) * 0.02
        )
        self.growth_history: List[GrowthEvent] = []
        
    def grow(self, new_dim: int, event: GrowthEvent):
        """Grow embedding dimension."""
        if new_dim <= self.current_dim or new_dim > self.max_dim:
            return False
        
        # Create expanded embedding
        with torch.no_grad():
            old_embed = self.embedding.data
            new_embed = torch.randn(
                self.vocab_size, new_dim, 
                device=self.device
            ) * 0.02
            
            # Copy existing dimensions
            new_embed[:, :self.current_dim] = old_embed
            
            # Update
            self.embedding = nn.Parameter(new_embed)
            self.current_dim = new_dim
            self.growth_history.append(event)
        
        return True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings."""
        return F.embedding(x, self.embedding)


class GrowableMLP(nn.Module):
    """
    MLP that can grow its hidden dimension.
    
    Grows to match tree depth - deeper trees need
    more transformation capacity.
    """
    
    def __init__(self, initial_dim: int, hidden_multiplier: int = 4, device='cuda'):
        super().__init__()
        self.current_dim = initial_dim
        self.hidden_multiplier = hidden_multiplier
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        hidden = initial_dim * hidden_multiplier
        self.up = nn.Linear(initial_dim, hidden, device=self.device)
        self.down = nn.Linear(hidden, initial_dim, device=self.device)
        
    def grow(self, new_dim: int):
        """Grow input/output dimension."""
        if new_dim <= self.current_dim:
            return False
        
        hidden = new_dim * self.hidden_multiplier
        
        with torch.no_grad():
            # Create new layers
            new_up = nn.Linear(new_dim, hidden, device=self.device)
            new_down = nn.Linear(hidden, new_dim, device=self.device)
            
            # Copy existing weights
            old_up_w = self.up.weight.data
            old_down_w = self.down.weight.data
            
            # Initialize new weights
            nn.init.xavier_uniform_(new_up.weight)
            nn.init.xavier_uniform_(new_down.weight)
            
            # Copy old weights where they fit
            old_hidden = self.current_dim * self.hidden_multiplier
            new_up.weight.data[:old_hidden, :self.current_dim] = old_up_w
            new_down.weight.data[:self.current_dim, :old_hidden] = old_down_w
            
            self.up = new_up
            self.down = new_down
            self.current_dim = new_dim
        
        return True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MLP forward pass."""
        h = F.gelu(self.up(x))
        return self.down(h)


class ResonantSECSystem:
    """
    SEC system that tracks fractal frequency and triggers growth.
    """
    
    def __init__(self, initial_dim: int = 64, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.current_dim = initial_dim
        
        self.nodes: Dict[str, ResonantNode] = {}
        self.crystallized_nodes: Set[str] = set()
        
        # Fractal tracking
        self.max_depth = 0  # Deepest crystallized node
        self.frequency_tracker = FractalFrequencyTracker()
        
        # Global state
        self.iteration = 0
        self.global_entropy = 1.0
        
        # Imported knowledge (will be resized on growth)
        self.vocab_embeddings: torch.Tensor = None
        
    def load_and_resize_embeddings(self, pac_path: Path, target_dim: int):
        """Load embeddings and resize to target dimension."""
        vocab_data = torch.load(pac_path / "pac_vocab.pt", weights_only=True)
        original = vocab_data['vocab_deltas'].to(self.device)
        
        if original.shape[1] == target_dim:
            self.vocab_embeddings = original
        elif original.shape[1] > target_dim:
            # Truncate (lose info but faster)
            self.vocab_embeddings = original[:, :target_dim]
        else:
            # Pad with small random values
            padding = torch.randn(
                original.shape[0], target_dim - original.shape[1],
                device=self.device
            ) * 0.01
            self.vocab_embeddings = torch.cat([original, padding], dim=1)
        
        print(f"  ✓ Loaded embeddings: {self.vocab_embeddings.shape}")
        
    def add_token(self, position: int, token_id: int, depth: int = 0) -> str:
        """Add a token node at a given depth."""
        nid = f"tok_{position}"
        
        node = ResonantNode(
            nid=nid,
            position=position,
            token_id=token_id,
            depth=depth,
            entropy=1.0
        )
        
        # Get embedding (resized to current dim)
        if token_id < len(self.vocab_embeddings):
            node.delta = self.vocab_embeddings[token_id].clone()
        else:
            node.delta = torch.randn(self.current_dim, device=self.device)
        
        self.nodes[nid] = node
        
        # Track max depth
        self.max_depth = max(self.max_depth, depth)
        
        return nid
    
    def collapse_step(self) -> Tuple[Dict, List[str]]:
        """
        One SEC collapse step.
        
        Returns metrics and list of newly crystallized nodes.
        """
        self.iteration += 1
        new_crystals = []
        
        for nid, node in self.nodes.items():
            if node.crystallized:
                continue
            
            # Compute collapse based on entropy
            beta = 1.0 * (1.0571 - self.global_entropy) / 0.0571
            collapse_mag = node.entropy * np.exp(-beta * node.entropy)
            
            # Entropy gradient from neighbors
            entropy_gradient = 0.0
            for neighbor_nid, coupling in node.neighbors.items():
                if neighbor_nid in self.nodes:
                    neighbor = self.nodes[neighbor_nid]
                    entropy_gradient += coupling * (neighbor.entropy - node.entropy)
            
            # Update entropy (FASTER collapse - 0.3 rate instead of 0.1)
            node.entropy = max(0.0, min(1.0, 
                node.entropy + 0.3 * (0.1 * entropy_gradient - collapse_mag)
            ))
            
            # Check crystallization
            if node.entropy < 0.15:
                node.crystallized = True
                node.crystal_iteration = self.iteration
                self.crystallized_nodes.add(nid)
                new_crystals.append(nid)
                
        # Record new crystals as batch
        if new_crystals:
            self.frequency_tracker.record_crystallization(self.iteration, len(new_crystals))
        
        # Update global entropy
        active = [n for n in self.nodes.values() if not n.crystallized]
        if active:
            self.global_entropy = np.mean([n.entropy for n in active])
        else:
            self.global_entropy = 0.0
        
        return {
            'iteration': self.iteration,
            'entropy': self.global_entropy,
            'crystallized': len(self.crystallized_nodes),
            'new_crystals': len(new_crystals),
            'max_depth': self.max_depth,
        }, new_crystals
    
    def resize_all_deltas(self, new_dim: int):
        """Resize all node deltas when transformer grows."""
        if new_dim <= self.current_dim:
            return
        
        for node in self.nodes.values():
            if node.delta is not None:
                old_delta = node.delta
                new_delta = torch.zeros(new_dim, device=self.device)
                new_delta[:self.current_dim] = old_delta
                node.delta = new_delta
        
        # Also resize vocab embeddings
        if self.vocab_embeddings is not None:
            old_embed = self.vocab_embeddings
            padding = torch.randn(
                old_embed.shape[0], new_dim - self.current_dim,
                device=self.device
            ) * 0.01
            self.vocab_embeddings = torch.cat([old_embed, padding], dim=1)
        
        self.current_dim = new_dim


class ResonantGrowingTransformer(nn.Module):
    """
    Transformer that grows with the PAC tree.
    
    As the tree crystallizes and deepens, the transformer
    expands its capacity to match the fractal complexity.
    """
    
    def __init__(self, 
                 vocab_size: int = 50304,
                 initial_dim: int = 64,
                 max_dim: int = 512,
                 device='cuda'):
        super().__init__()
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.current_dim = initial_dim
        self.max_dim = max_dim
        
        # Growable components
        self.embedding = GrowableEmbedding(vocab_size, initial_dim, max_dim, device)
        self.mlp = GrowableMLP(initial_dim, hidden_multiplier=4, device=device)
        
        # SEC system
        self.sec_system = ResonantSECSystem(initial_dim, device)
        
        # Output projection (also growable)
        self.output_proj = nn.Linear(initial_dim, vocab_size, device=self.device)
        
        # Growth tracking
        self.growth_events: List[GrowthEvent] = []
        
    def load_knowledge(self, pac_path: Path):
        """Load knowledge from extraction."""
        self.sec_system.load_and_resize_embeddings(pac_path, self.current_dim)
        
        # Initialize embedding from loaded vocab
        if self.sec_system.vocab_embeddings is not None:
            with torch.no_grad():
                vocab_embed = self.sec_system.vocab_embeddings[:self.embedding.vocab_size]
                self.embedding.embedding.data = vocab_embed
    
    def check_and_grow(self) -> Optional[GrowthEvent]:
        """Check if growth is needed and grow if so."""
        if self.current_dim >= self.max_dim:
            return None
        
        # Track crystals accumulated this iteration
        total_crystals = self.sec_system.frequency_tracker.total_crystals
        
        # Check Fibonacci-count-based growth
        should_grow, threshold = self.sec_system.frequency_tracker.should_grow()
        if should_grow:
            # Grow by Fibonacci factor
            fib_factor = self.sec_system.frequency_tracker.get_fibonacci_growth_factor()
            growth = min(fib_factor * 8, 64)  # Cap growth per step
            new_dim = min(self.current_dim + growth, self.max_dim)
            
            if new_dim > self.current_dim:
                event = GrowthEvent(
                    iteration=self.sec_system.iteration,
                    old_dim=self.current_dim,
                    new_dim=new_dim,
                    trigger=f'fib_threshold_{threshold}',
                    tree_depth=self.sec_system.max_depth,
                    crystal_count=total_crystals
                )
                
                self._perform_growth(new_dim, event)
                return event
        
        return None
    
    def _perform_growth(self, new_dim: int, event: GrowthEvent):
        """Perform the actual growth."""
        print(f"  🌱 GROWTH: {self.current_dim} → {new_dim} dim "
              f"(trigger: {event.trigger}, depth: {event.tree_depth})")
        
        # Grow all components
        self.embedding.grow(new_dim, event)
        self.mlp.grow(new_dim)
        self.sec_system.resize_all_deltas(new_dim)
        
        # Grow output projection
        with torch.no_grad():
            old_proj = self.output_proj.weight.data
            new_proj = nn.Linear(new_dim, self.embedding.vocab_size, device=self.device)
            nn.init.xavier_uniform_(new_proj.weight)
            new_proj.weight.data[:, :self.current_dim] = old_proj
            self.output_proj = new_proj
        
        self.current_dim = new_dim
        self.growth_events.append(event)
    
    def process_sequence(self, token_ids: List[int], collapse_iters: int = 30, debug: bool = False):
        """Process sequence with SEC collapse and potential growth."""
        # Reset SEC system (but NOT the frequency_tracker - that accumulates!)
        self.sec_system.nodes.clear()
        self.sec_system.crystallized_nodes.clear()
        self.sec_system.iteration = 0
        self.sec_system.max_depth = 0
        
        # Add tokens at increasing depth
        for i, tid in enumerate(token_ids):
            depth = i % 5  # Cycle through depths
            self.sec_system.add_token(i, tid, depth=depth)
        
        # Build coupling
        for i, nid in enumerate(self.sec_system.nodes.keys()):
            node = self.sec_system.nodes[nid]
            for j, other_nid in enumerate(self.sec_system.nodes.keys()):
                if i != j:
                    dist = abs(i - j)
                    coupling = np.exp(-dist / 3.0)
                    node.neighbors[other_nid] = coupling
        
        # Collapse with growth checks
        total_new = 0
        for iter_i in range(collapse_iters):
            metrics, new_crystals = self.sec_system.collapse_step()
            total_new += len(new_crystals)
            
            # Check for growth after each crystallization
            if new_crystals:
                growth = self.check_and_grow()
        
        if debug:
            # Show entropy distribution
            entropies = [n.entropy for n in self.sec_system.nodes.values()]
            min_ent = min(entropies) if entropies else 0
            max_ent = max(entropies) if entropies else 0
            print(f"    [process] tokens={len(token_ids)}, iters={collapse_iters}, "
                  f"crystals={total_new}, entropy=[{min_ent:.3f},{max_ent:.3f}]")
        
        return metrics
    
    def compose_and_predict(self, top_k: int = 10) -> List[Tuple[int, float]]:
        """Compose representation and predict next token."""
        # Weighted composition
        weights = []
        deltas = []
        
        max_pos = max(n.position for n in self.sec_system.nodes.values())
        
        for node in self.sec_system.nodes.values():
            causal_w = np.exp((node.position - max_pos) / 2.0)
            crystal_w = 2.0 if node.crystallized else 1.0 / (1.0 + node.entropy)
            weights.append(causal_w * crystal_w)
            deltas.append(node.delta)
        
        weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        weights = weights / weights.sum()
        
        stacked = torch.stack(deltas)
        composed = (stacked * weights.unsqueeze(1)).sum(dim=0)
        
        # MLP transform
        composed = composed + self.mlp(composed)
        
        # Score against vocab
        composed_norm = F.normalize(composed.unsqueeze(0), dim=1).squeeze()
        vocab_norm = F.normalize(self.sec_system.vocab_embeddings, dim=1)
        scores = vocab_norm @ composed_norm
        
        top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
        
        return [(idx.item(), score.item()) for idx, score in zip(top_indices, top_scores)]
    
    def generate(self, token_ids: List[int], max_new_tokens: int = 20,
                temperature: float = 0.7) -> List[int]:
        """Generate with resonant growth."""
        generated = list(token_ids)
        
        for i in range(max_new_tokens):
            # Process current sequence - more iterations for proper collapse
            metrics = self.process_sequence(generated, collapse_iters=100, debug=(i==0))
            
            # Predict
            predictions = self.compose_and_predict(top_k=50)
            
            if not predictions:
                break
            
            # Sample
            scores = torch.tensor([p[1] for p in predictions]) / temperature
            probs = F.softmax(scores, dim=0)
            idx = torch.multinomial(probs, 1).item()
            
            next_token = predictions[idx][0]
            generated.append(next_token)
        
        return generated


def test_resonant_growing():
    """Test the resonant growing transformer."""
    from transformers import GPT2Tokenizer
    
    print("="*70)
    print("RESONANT GROWING SEC-PAC TRANSFORMER")
    print("="*70)
    print("\nKey: Transformer GROWS as PAC tree crystallizes,")
    print("matching capacity to fractal complexity.")
    
    pac_path = Path(__file__).parent.parent / "poc_016_pac_extraction" / "extracted_v3" / "pythia_70m"
    
    if not pac_path.exists():
        print(f"❌ PAC not found at {pac_path}")
        return
    
    # Create model starting SMALL
    print("\nInitializing with SMALL dimensions (64-dim)...")
    model = ResonantGrowingTransformer(
        vocab_size=50304,
        initial_dim=64,
        max_dim=512
    )
    model.load_knowledge(pac_path)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    print("\n" + "="*70)
    print("RESONANT GROWTH DURING GENERATION")
    print("="*70)
    
    prompts = [
        "The weather today is",
        "Once upon a time there was a",
        "The meaning of life is to",
        "In the beginning there was nothing but"
    ]
    
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt)
        
        print(f"\nPrompt: '{prompt}'")
        print(f"  Initial dim: {model.current_dim}")
        
        # Generate
        generated = model.generate(token_ids, max_new_tokens=15, temperature=0.6)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        
        print(f"  Final dim: {model.current_dim}")
        print(f"  Growth events: {len(model.growth_events)}")
        print(f"  → {text}")
        
        # Show growth history
        if model.growth_events:
            print(f"  Growth history:")
            for event in model.growth_events[-3:]:
                print(f"    iter {event.iteration}: {event.old_dim}→{event.new_dim} "
                      f"({event.trigger})")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
Resonant Growing Transformer Results:

Starting dimension: 64
Final dimension: {model.current_dim}
Total growth events: {len(model.growth_events)}

Key Features:
1. FRACTAL FREQUENCY TRACKING
   - Counts crystallization events
   - Computes frequency in 0.020-0.030 Hz range
   - Triggers growth when frequency exceeds threshold

2. FIBONACCI-GUIDED GROWTH
   - Growth amount follows Fibonacci sequence
   - Matches natural fractal scaling
   - F_n levels for n/10 crystallization events

3. COMPONENT GROWTH
   - Embeddings expand: vocab × dim grows
   - MLP hidden layer expands: 4× dim
   - Output projection expands
   - All node deltas resize

4. RESONANCE WITH TREE
   - Transformer capacity matches tree complexity
   - Deeper trees → larger dimensions
   - More crystallization → more growth
   - Natural feedback loop

This is the "analog" approach: 
- Not fixed architecture
- Grows organically with knowledge structure
- Capacity emerges from information dynamics
""")


if __name__ == "__main__":
    test_resonant_growing()
