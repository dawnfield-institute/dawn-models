"""
POC-023 Experiment 05: Convergence Dashboard
Multi-dimensional convergence metrics during generation.

Key insight: Each metric is a different projection of crystallization.
When metrics diverge from each other, the divergence pattern is diagnostic.

Metrics tracked:
1. Chord concentration (harmonic coherence) - do depths agree?
2. Concentration velocity (dC/dt) - is harmony stable or drifting?
3. Xi balance (energy/structure ratio) - local vs global stability
4. Collapse events - sudden discontinuities
5. Recovery rate - how fast does system re-stabilize?

Goal: Build convergence maintenance dashboard for real-time monitoring.
"""

import json
import time
import random
from datetime import datetime
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from transformers import GPT2Tokenizer

# Dawn Field constants
PHI = 1.6180339887
PHI_INV = 1 / PHI
XI = 1.0571428  # From Navier-Stokes work
HALF = 0.5


@dataclass
class ConvergenceMetrics:
    """Multi-dimensional convergence state at a single token."""
    token_id: int
    token_text: str
    position: int
    
    # Harmonic metrics
    concentration: Optional[float] = None  # Chord concentration
    depths_available: int = 0
    depths_agreeing: int = 0
    
    # Trend metrics
    concentration_velocity: float = 0.0  # dC/dt
    rolling_mean: float = 0.0  # Mean of last N
    
    # Stability metrics
    is_collapse: bool = False  # Sudden drop
    is_recovery: bool = False  # Coming back from collapse
    stability_streak: int = 0  # Tokens since last collapse
    
    # Xi-like balance
    xi_balance: float = 0.0  # Ratio of local to global agreement


@dataclass
class ConvergenceState:
    """Rolling convergence state tracker."""
    window_size: int = 10
    collapse_threshold: float = 0.3  # Drop > this = collapse
    stability_threshold: float = 0.6  # Above this = stable
    
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    concentration_history: deque = field(default_factory=lambda: deque(maxlen=10))
    collapse_count: int = 0
    recovery_count: int = 0
    current_streak: int = 0
    in_collapse: bool = False
    
    def update(self, metrics: ConvergenceMetrics) -> ConvergenceMetrics:
        """Update state and compute derived metrics."""
        
        if metrics.concentration is not None:
            # Compute velocity
            if len(self.concentration_history) > 0:
                prev = self.concentration_history[-1]
                metrics.concentration_velocity = metrics.concentration - prev
            
            # Rolling mean
            self.concentration_history.append(metrics.concentration)
            metrics.rolling_mean = sum(self.concentration_history) / len(self.concentration_history)
            
            # Detect collapse
            if metrics.concentration_velocity < -self.collapse_threshold:
                metrics.is_collapse = True
                self.collapse_count += 1
                self.in_collapse = True
                self.current_streak = 0
            
            # Detect recovery
            if self.in_collapse and metrics.concentration > self.stability_threshold:
                metrics.is_recovery = True
                self.recovery_count += 1
                self.in_collapse = False
            
            # Stability streak
            if metrics.concentration > self.stability_threshold:
                self.current_streak += 1
            else:
                self.current_streak = 0
            metrics.stability_streak = self.current_streak
            
            # Xi balance: ratio of adjacent-depth agreement to distant-depth agreement
            # This measures local vs global coherence
            if metrics.depths_available >= 3:
                # Approximation: high concentration with few depths = local
                # high concentration with many depths = global
                local_signal = metrics.concentration
                global_signal = metrics.depths_agreeing / metrics.depths_available
                if global_signal > 0:
                    metrics.xi_balance = local_signal / global_signal
                else:
                    metrics.xi_balance = 0.0
        
        self.history.append(metrics)
        return metrics


class PACNode:
    """PAC node with detailed statistics."""
    __slots__ = ['token_id', 'counts', 'total', 'children', 'depth']
    
    def __init__(self, token_id: int, depth: int):
        self.token_id = token_id
        self.depth = depth
        self.counts = {}
        self.total = 0
        self.children = {}
    
    def observe(self, target: int):
        self.counts[target] = self.counts.get(target, 0) + 1
        self.total += 1
    
    def get_or_create_child(self, token_id: int) -> 'PACNode':
        if token_id not in self.children:
            self.children[token_id] = PACNode(token_id, self.depth + 1)
        return self.children[token_id]
    
    def predict_top_k(self, k: int = 10) -> list:
        if not self.counts:
            return []
        sorted_preds = sorted(self.counts.items(), key=lambda x: -x[1])
        return [(t, c) for t, c in sorted_preds[:k]]
    
    def sample(self, temperature: float = 1.0) -> int:
        if not self.counts:
            return None
        tokens = list(self.counts.keys())
        counts = list(self.counts.values())
        if temperature != 1.0:
            counts = [c ** (1.0 / temperature) for c in counts]
        total = sum(counts)
        probs = [c / total for c in counts]
        return random.choices(tokens, weights=probs, k=1)[0]


class PACTree:
    """PAC tree with convergence tracking."""
    
    def __init__(self, vocab_size: int, max_depth: int, device: torch.device):
        self.vocab_size = vocab_size
        self.max_depth = max_depth
        self.device = device
        self.roots = {}
        self.node_count = 0
    
    def learn(self, tokens: torch.Tensor):
        tokens_list = tokens.cpu().tolist()
        for i in range(1, len(tokens_list)):
            target = tokens_list[i]
            for depth in range(1, min(self.max_depth + 1, i + 1)):
                context_start = i - depth
                context = tokens_list[context_start:i]
                first_token = context[0]
                if first_token not in self.roots:
                    self.roots[first_token] = PACNode(first_token, 0)
                    self.node_count += 1
                current = self.roots[first_token]
                for token in context[1:]:
                    if token not in current.children:
                        self.node_count += 1
                    current = current.get_or_create_child(token)
                current.observe(target)
    
    def get_node_at_depth(self, context: list, depth: int) -> PACNode:
        if depth > len(context) or depth < 1:
            return None
        ctx = context[-depth:]
        first_token = ctx[0]
        if first_token not in self.roots:
            return None
        current = self.roots[first_token]
        for token in ctx[1:]:
            if token not in current.children:
                return None
            current = current.children[token]
        return current
    
    def compute_detailed_concentration(self, context: list) -> tuple:
        """Compute concentration with detailed breakdown."""
        top1_by_depth = {}
        
        for d in range(1, self.max_depth + 1):
            node = self.get_node_at_depth(context, d)
            if node and node.total > 0:
                preds = node.predict_top_k(1)
                if preds:
                    top1_by_depth[d] = preds[0][0]
        
        depths_available = len(top1_by_depth)
        if depths_available < 2:
            return None, depths_available, 0
        
        # Count how many agree with the majority
        pred_counts = {}
        for pred in top1_by_depth.values():
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        max_agreement = max(pred_counts.values())
        unique_preds = len(pred_counts)
        
        concentration = 1.0 - (unique_preds - 1) / (depths_available - 1)
        
        return concentration, depths_available, max_agreement
    
    def generate_with_dashboard(self, seed_tokens: list, length: int,
                                 temperature: float = 1.0,
                                 tokenizer=None) -> tuple:
        """Generate with full convergence tracking."""
        context = seed_tokens.copy()
        state = ConvergenceState()
        all_metrics = []
        
        for pos in range(length):
            # Compute concentration
            conc, depths_avail, depths_agree = self.compute_detailed_concentration(context)
            
            # Get deepest node for sampling
            node = None
            for d in range(self.max_depth, 0, -1):
                node = self.get_node_at_depth(context, d)
                if node and node.total > 0:
                    break
            
            if node is None or node.total == 0:
                break
            
            # Sample
            next_token = node.sample(temperature)
            if next_token is None:
                break
            
            # Build metrics
            token_text = tokenizer.decode([next_token]) if tokenizer else str(next_token)
            metrics = ConvergenceMetrics(
                token_id=next_token,
                token_text=token_text,
                position=pos,
                concentration=conc,
                depths_available=depths_avail,
                depths_agreeing=depths_agree,
            )
            
            # Update state and compute derived metrics
            metrics = state.update(metrics)
            all_metrics.append(metrics)
            
            context.append(next_token)
        
        return all_metrics, state


def load_wikitext() -> tuple:
    """Load WikiText-2 from cache."""
    cache_path = Path(__file__).parent.parent.parent / "poc_022_scale_stress_test" / "data_cache" / "wikitext2_10000_64.pt"
    if cache_path.exists():
        data = torch.load(cache_path, weights_only=False)
        tokens = data["sequences"].flatten().cpu().tolist()
        return tokens, data["vocab_size"]
    raise FileNotFoundError(f"Cache not found: {cache_path}")


def print_dashboard(metrics: List[ConvergenceMetrics], state: ConvergenceState):
    """Print convergence dashboard."""
    print("\n" + "=" * 80)
    print("CONVERGENCE DASHBOARD")
    print("=" * 80)
    
    # Summary stats
    concs = [m.concentration for m in metrics if m.concentration is not None]
    if concs:
        print(f"\nMean concentration: {sum(concs)/len(concs):.3f}")
        print(f"Min concentration:  {min(concs):.3f}")
        print(f"Max concentration:  {max(concs):.3f}")
    
    print(f"\nCollapse events: {state.collapse_count}")
    print(f"Recovery events: {state.recovery_count}")
    
    # Timeline visualization
    print("\n" + "-" * 80)
    print("TIMELINE (C=concentration, V=velocity)")
    print("-" * 80)
    
    for m in metrics:
        # Build status indicators
        status = ""
        if m.is_collapse:
            status = " [COLLAPSE]"
        elif m.is_recovery:
            status = " [RECOVERY]"
        elif m.stability_streak >= 5:
            status = " [STABLE]"
        
        if m.concentration is not None:
            bar_len = int(m.concentration * 20)
            bar = "#" * bar_len + "." * (20 - bar_len)
            v_char = "+" if m.concentration_velocity > 0.1 else ("-" if m.concentration_velocity < -0.1 else "=")
            
            print(f"{m.position:3d}: [{bar}] C={m.concentration:.2f} V={m.concentration_velocity:+.2f} {v_char} "
                  f"Xi={m.xi_balance:.2f} '{m.token_text}'{status}")
        else:
            print(f"{m.position:3d}: [????????????????????] '{m.token_text}'")
    
    # Divergence analysis
    print("\n" + "-" * 80)
    print("DIVERGENCE ANALYSIS")
    print("-" * 80)
    
    xi_values = [m.xi_balance for m in metrics if m.xi_balance > 0]
    if xi_values:
        mean_xi = sum(xi_values) / len(xi_values)
        print(f"\nMean Xi balance: {mean_xi:.3f}")
        print(f"Expected Xi:     {XI:.3f}")
        print(f"Deviation:       {abs(mean_xi - XI):.3f}")
        
        # Check for divergence patterns
        last_5_conc = concs[-5:] if len(concs) >= 5 else concs
        last_5_xi = xi_values[-5:] if len(xi_values) >= 5 else xi_values
        
        conc_trend = sum(last_5_conc) / len(last_5_conc) if last_5_conc else 0
        xi_trend = sum(last_5_xi) / len(last_5_xi) if last_5_xi else 0
        
        print(f"\nRecent trends:")
        print(f"  Concentration: {conc_trend:.3f}")
        print(f"  Xi balance:    {xi_trend:.3f}")
        
        if conc_trend < 0.5 and xi_trend > XI:
            print("\n  Pattern: Harmony low, Xi high -> Local instability")
        elif conc_trend > 0.7 and xi_trend < XI * 0.8:
            print("\n  Pattern: Harmony high, Xi low -> Over-regularizing")
        elif conc_trend < 0.5 and xi_trend < XI * 0.8:
            print("\n  Pattern: Both metrics falling -> Possible collapse")
        else:
            print("\n  Pattern: Metrics in normal range")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("POC-023 Exp 05: Convergence Dashboard")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"PHI = {PHI:.4f}, XI = {XI:.4f}")
    print()
    print("Multi-dimensional convergence tracking during generation")
    print()
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    all_tokens, vocab_size = load_wikitext()
    print(f"Total tokens: {len(all_tokens):,}")
    
    split = int(len(all_tokens) * 0.8)
    train_tokens = all_tokens[:split]
    test_tokens = all_tokens[split:]
    
    max_depth = 5
    
    print("\nTraining...")
    t0 = time.time()
    tree = PACTree(vocab_size, max_depth, device)
    tree.learn(torch.tensor(train_tokens, device=device))
    print(f"Training time: {time.time() - t0:.1f}s")
    print(f"Total nodes: {tree.node_count:,}")
    
    # Generate with dashboard
    print("\n" + "=" * 60)
    print("GENERATION WITH CONVERGENCE TRACKING")
    print("=" * 60)
    
    num_sequences = 3
    gen_length = 40
    temperature = 0.8
    
    all_results = []
    
    for seq_idx in range(num_sequences):
        seed_start = random.randint(0, len(test_tokens) - max_depth - 1)
        seed_tokens = test_tokens[seed_start:seed_start + max_depth]
        
        print(f"\n{'='*60}")
        print(f"SEQUENCE {seq_idx + 1}")
        print(f"{'='*60}")
        print(f"Seed: {tokenizer.decode(seed_tokens)}")
        
        metrics, state = tree.generate_with_dashboard(
            seed_tokens, gen_length, temperature, tokenizer
        )
        
        print_dashboard(metrics, state)
        
        all_results.append({
            "seed": tokenizer.decode(seed_tokens),
            "generated": "".join(m.token_text for m in metrics),
            "collapse_count": state.collapse_count,
            "recovery_count": state.recovery_count,
            "mean_concentration": sum(m.concentration for m in metrics if m.concentration) / len([m for m in metrics if m.concentration]) if metrics else 0,
        })
    
    # Save results
    output = {
        "experiment": "exp_05_convergence_dashboard",
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "vocab_size": vocab_size,
        "max_depth": max_depth,
        "gen_length": gen_length,
        "temperature": temperature,
        "constants": {"PHI": PHI, "XI": XI},
        "sequences": all_results,
    }
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"exp_05_convergence_dashboard_{timestamp}.json"
    
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n\nSaved: {results_path}")


if __name__ == "__main__":
    main()
