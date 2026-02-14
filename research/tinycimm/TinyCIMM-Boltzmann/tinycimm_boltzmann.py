"""
TinyCIMM-Boltzmann: PAC-Conserved Multi-Head Learning
======================================================

A minimal neural architecture where the total information budget (entropy)
across heads is CONSERVED during learning — the PAC rule as architectural
constraint, not observation.

Motivation (from exp_12 findings):
  In standard transformers, hallucination creates +9.6% uncompensated
  entropy across all heads. Compensation ratio = 0.000 in GPT-2 models.
  Every single layer gains entropy; nothing counterbalances.

  TinyCIMM-Boltzmann asks: what if we ENFORCE conservation?
  When one head increases entropy (explores), another must decrease
  (crystallize). The total budget stays constant. Information is
  redistributed, never created.

Key Insight:
  Hallucination = PAC violation = entropy creation without source.
  If we make PAC violation impossible by architecture, does the network
  naturally avoid generating "phantom structure"?

Architecture:
  - BoltzmannHead: Learnable processing unit with tracked entropy
  - BoltzmannLayer: N parallel heads with a shared entropy budget
  - ConservationProjector: Redistributes entropy to enforce PAC
  - TinyCIMMBoltzmann: Full model with continuous learning

The "Boltzmann" name:
  Ludwig Boltzmann's S = k ln W is literally what we measure — entropy
  of activation distributions. His H-theorem proves entropy increases
  toward equilibrium. Our finding: hallucination breaks this by creating
  entropy without thermodynamic justification. This model enforces the
  conservation that Boltzmann proved should hold.

Author: Dawn Field Institute
Date: 2026-02-14
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import math

# Dawn Field Theory constants
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = PHI - 1
XI = 1 + np.pi / 55

# SEC phase boundaries (zero-parameter, theory-derived)
SEC_CRYSTALLIZED = 0.5
SEC_ORDERED = 2.0
SEC_TRANSITIONAL = 4.0


def classify_sec_phase(entropy: float) -> str:
    """Classify SEC phase from attention entropy."""
    if entropy < SEC_CRYSTALLIZED:
        return "crystallized"
    elif entropy < SEC_ORDERED:
        return "ordered"
    elif entropy < SEC_TRANSITIONAL:
        return "transitional"
    else:
        return "chaotic"


# ─── Data Structures ────────────────────────────────────────────────

@dataclass
class ConservationState:
    """Tracks PAC conservation metrics across heads."""
    total_budget: float = 0.0          # Σ(head entropy) — should stay constant
    target_budget: float = 0.0         # Initial budget to conserve
    violation: float = 0.0             # Current violation magnitude
    compensation_ratio: float = 1.0    # 1.0 = perfect PAC
    head_entropies: List[float] = field(default_factory=list)
    phase_distribution: Dict[str, int] = field(default_factory=dict)
    steps: int = 0


@dataclass
class BoltzmannMetrics:
    """Complete metrics from a learning step."""
    loss: float = 0.0
    task_loss: float = 0.0
    conservation_loss: float = 0.0
    total_budget: float = 0.0
    target_budget: float = 0.0
    violation_pct: float = 0.0
    compensation_ratio: float = 1.0
    head_entropies: List[float] = field(default_factory=list)
    head_phases: List[str] = field(default_factory=list)
    mean_entropy: float = 0.0
    entropy_std: float = 0.0
    head_cv: float = 0.0
    step: int = 0


# ─── Core Components ────────────────────────────────────────────────

class BoltzmannHead(nn.Module):
    """
    Single processing head with tracked activation entropy.

    Each head maintains its own internal state and contributes to the
    layer-level entropy budget. The head can be in different SEC phases
    (crystallized, ordered, transitional, chaotic) depending on its
    current activation distribution.

    Key: the head does NOT control its own entropy — the layer does,
    through the ConservationProjector. This is the PAC constraint.
    """

    def __init__(self, input_dim: int, head_dim: int, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.head_dim = head_dim

        # Learnable transformation
        self.W_q = nn.Linear(input_dim, head_dim, bias=False, device=device)
        self.W_k = nn.Linear(input_dim, head_dim, bias=False, device=device)
        self.W_v = nn.Linear(input_dim, head_dim, bias=False, device=device)

        # Entropy tracking (not learnable — diagnostic)
        self._last_entropy = 0.0
        self._last_phase = "ordered"
        self._entropy_history: List[float] = []

        # Initialize with small weights for stable entropy
        nn.init.xavier_uniform_(self.W_q.weight, gain=0.5)
        nn.init.xavier_uniform_(self.W_k.weight, gain=0.5)
        nn.init.xavier_uniform_(self.W_v.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Forward pass. Returns (output, entropy_float, entropy_tensor).

        The entropy measures the "attention spread" of this head —
        how diffuse vs concentrated its processing is.

        Returns both a float (for monitoring) and a differentiable tensor
        (so conservation loss can backpropagate through the network).
        """
        # Compute attention-like scores
        q = self.W_q(x)  # (batch, head_dim)
        k = self.W_k(x)  # (batch, head_dim)
        v = self.W_v(x)  # (batch, head_dim)

        # Self-attention within head dimensions
        # Score = softmax(q·k^T / sqrt(d))
        scale = math.sqrt(self.head_dim)
        scores = F.softmax(torch.sum(q * k, dim=-1, keepdim=True) / scale, dim=-1)

        # Weighted value (analogous to attention output)
        out = scores * v

        # ── Differentiable entropy proxy ──
        # Use softmax over squared activations to get a probability-like
        # distribution that stays connected to the computation graph.
        act_sq = out.flatten() ** 2 + 1e-10
        act_prob = act_sq / act_sq.sum()
        # Differentiable entropy (connected to computation graph)
        entropy_tensor = -torch.sum(act_prob * torch.log(act_prob))

        # Float entropy for monitoring (detached)
        entropy = float(entropy_tensor.detach().item())

        # Update tracking
        self._last_entropy = entropy
        self._last_phase = classify_sec_phase(entropy)
        self._entropy_history.append(entropy)
        if len(self._entropy_history) > 100:
            self._entropy_history.pop(0)

        return out, entropy, entropy_tensor

    @property
    def entropy(self) -> float:
        return self._last_entropy

    @property
    def phase(self) -> str:
        return self._last_phase


class ConservationProjector(nn.Module):
    """
    Enforces PAC conservation across heads within a layer.

    This is the key architectural innovation: after all heads compute
    their outputs, the projector adjusts the combined representation
    so that the TOTAL entropy budget is conserved.

    Two modes:
      - SOFT: Adds a conservation loss term (penalty for violation)
      - HARD: Explicitly normalizes head outputs to enforce budget

    The soft mode is more useful for research (we can measure violation).
    The hard mode tests what happens when conservation is absolute.
    """

    def __init__(self, n_heads: int, head_dim: int, mode: str = 'soft',
                 conservation_strength: float = 1.0, device: str = 'cpu'):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.mode = mode
        self.conservation_strength = conservation_strength
        self.device = device

        # Learnable mixing weights (how to combine heads)
        self.mix = nn.Linear(n_heads * head_dim, n_heads * head_dim,
                             bias=False, device=device)
        nn.init.eye_(self.mix.weight)  # Start at identity

        # Budget tracking
        self._target_budget: Optional[float] = None
        self._budget_initialized = False

    def set_target_budget(self, budget: float):
        """Set the entropy budget to conserve."""
        self._target_budget = budget
        self._budget_initialized = True

    def forward(self, head_outputs: List[torch.Tensor],
                head_entropies: List[float],
                entropy_tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine head outputs while tracking conservation.

        Returns (combined_output, conservation_loss_tensor).
        conservation_loss is a DIFFERENTIABLE tensor so gradients flow.
        """
        # Concatenate head outputs
        combined = torch.cat(head_outputs, dim=-1)  # (batch, n_heads * head_dim)
        mixed = self.mix(combined)

        # Current total entropy budget (float for tracking)
        current_budget = sum(head_entropies)

        # Initialize target budget on first call
        if not self._budget_initialized:
            self._target_budget = current_budget
            self._budget_initialized = True

        # Conservation loss (DIFFERENTIABLE)
        if self.mode == 'soft':
            # Sum the differentiable entropy tensors
            total_entropy_tensor = sum(entropy_tensors)
            target = torch.tensor(self._target_budget, device=mixed.device,
                                  dtype=mixed.dtype)
            # Squared violation — gradient flows through entropy_tensors
            # back to head parameters
            conservation_loss = self.conservation_strength * (
                total_entropy_tensor - target) ** 2
        elif self.mode == 'hard':
            # Scale outputs to enforce exact budget
            if current_budget > 1e-8:
                scale_factor = self._target_budget / current_budget
                mixed = mixed * math.sqrt(scale_factor)
            conservation_loss = torch.tensor(0.0, device=mixed.device)
        else:
            conservation_loss = torch.tensor(0.0, device=mixed.device)

        return mixed, conservation_loss


class BoltzmannLayer(nn.Module):
    """
    Multi-head layer with PAC-conserved entropy budget.

    Contains N parallel BoltzmannHeads plus a ConservationProjector
    that enforces the total entropy budget stays constant.

    The key analogy to exp_12:
      - In transformers: compensation ratio ≈ 0 during hallucination
      - In BoltzmannLayer: conservation_loss forces ratio → 1.0
    """

    def __init__(self, input_dim: int, n_heads: int = 4, head_dim: int = 8,
                 output_dim: Optional[int] = None,
                 conservation_mode: str = 'soft',
                 conservation_strength: float = 1.0,
                 device: str = 'cpu'):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device

        # Create heads
        self.heads = nn.ModuleList([
            BoltzmannHead(input_dim, head_dim, device=device)
            for _ in range(n_heads)
        ])

        # Conservation projector
        self.projector = ConservationProjector(
            n_heads, head_dim, mode=conservation_mode,
            conservation_strength=conservation_strength, device=device
        )

        # Output projection
        out_dim = output_dim or input_dim
        self.output_proj = nn.Linear(n_heads * head_dim, out_dim,
                                     bias=True, device=device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """
        Forward pass. Returns (output, conservation_loss_tensor, head_entropies).
        """
        head_outputs = []
        head_entropies = []
        entropy_tensors = []

        for head in self.heads:
            out, entropy, entropy_t = head(x)
            head_outputs.append(out)
            head_entropies.append(entropy)
            entropy_tensors.append(entropy_t)

        # Apply conservation constraint (differentiable)
        combined, conservation_loss = self.projector(
            head_outputs, head_entropies, entropy_tensors)

        # Project to output dimension
        output = self.output_proj(combined)

        return output, conservation_loss, head_entropies


class BoltzmannMonitor:
    """
    Real-time monitor for PAC conservation across the network.

    Tracks:
      - Total entropy budget over time (should be constant if PAC holds)
      - Per-head phase distribution (crystallized/ordered/transitional/chaotic)
      - Compensation dynamics (when one head goes up, does another go down?)
      - Violation trend (growing or shrinking?)
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.budget_history: List[float] = []
        self.violation_history: List[float] = []
        self.head_entropy_history: List[List[float]] = []
        self.phase_history: List[List[str]] = []
        self.compensation_history: List[float] = []
        self.state = ConservationState()

    def update(self, head_entropies: List[float], target_budget: float) -> ConservationState:
        """Update conservation state with new measurements."""
        self.state.steps += 1
        self.state.head_entropies = head_entropies
        self.state.total_budget = sum(head_entropies)
        self.state.target_budget = target_budget

        # Violation
        self.state.violation = self.state.total_budget - target_budget
        violation_pct = (self.state.violation / max(target_budget, 1e-8)) * 100

        # Track per-head changes for compensation analysis
        if len(self.head_entropy_history) > 0:
            prev = self.head_entropy_history[-1]
            deltas = [h - p for h, p in zip(head_entropies, prev)]
            increases = sum(d for d in deltas if d > 0)
            decreases = sum(d for d in deltas if d < 0)
            if increases > 1e-8:
                self.state.compensation_ratio = abs(decreases / increases)
            else:
                self.state.compensation_ratio = 1.0
            self.compensation_history.append(self.state.compensation_ratio)

        # Phase distribution
        phases = [classify_sec_phase(h) for h in head_entropies]
        self.state.phase_distribution = {}
        for p in phases:
            self.state.phase_distribution[p] = self.state.phase_distribution.get(p, 0) + 1

        # Update histories
        self.budget_history.append(self.state.total_budget)
        self.violation_history.append(violation_pct)
        self.head_entropy_history.append(head_entropies.copy())
        self.phase_history.append(phases)

        # Trim
        if len(self.budget_history) > self.window_size:
            self.budget_history.pop(0)
            self.violation_history.pop(0)
            self.head_entropy_history.pop(0)
            self.phase_history.pop(0)
            if len(self.compensation_history) > self.window_size:
                self.compensation_history.pop(0)

        return self.state

    def budget_stability(self) -> float:
        """How stable is the budget? 1.0 = perfectly constant."""
        if len(self.budget_history) < 3:
            return 1.0
        std = np.std(self.budget_history)
        mean = np.mean(self.budget_history)
        if mean < 1e-8:
            return 1.0
        cv = std / mean  # Coefficient of variation
        return 1.0 / (1.0 + cv * 10)

    def mean_compensation(self) -> float:
        """Average compensation ratio over window."""
        if not self.compensation_history:
            return 1.0
        return float(np.mean(self.compensation_history))

    def get_summary(self) -> Dict:
        """Get full monitor summary."""
        return {
            'steps': self.state.steps,
            'current_budget': self.state.total_budget,
            'target_budget': self.state.target_budget,
            'violation': self.state.violation,
            'violation_pct': (self.state.violation /
                              max(self.state.target_budget, 1e-8) * 100),
            'budget_stability': self.budget_stability(),
            'mean_compensation': self.mean_compensation(),
            'phase_distribution': self.state.phase_distribution,
            'head_entropies': self.state.head_entropies,
        }


# ─── Main Model ─────────────────────────────────────────────────────

class TinyCIMMBoltzmann(nn.Module):
    """
    TinyCIMM-Boltzmann: PAC-Conserved Multi-Head Continuous Learner

    Architecture:
      Input → BoltzmannLayer 1 → BoltzmannLayer 2 → Output

    Each layer has N heads with a conserved entropy budget.
    Conservation is enforced by the ConservationProjector.

    This tests the hypothesis from exp_12: if PAC conservation
    prevents entropy creation, does the model naturally produce
    more grounded, less hallucination-like outputs?

    The model learns sequences continuously (like all TinyCIMM
    variants) and tracks its conservation metrics in real-time.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        output_size: int = 1,
        n_heads: int = 4,
        n_layers: int = 2,
        conservation_mode: str = 'soft',
        conservation_strength: float = 1.0,
        learning_rate: float = 0.01,
        device: str = 'cpu',
    ):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.n_heads = n_heads
        self.n_layers_count = n_layers
        self.conservation_mode = conservation_mode
        self.conservation_strength = conservation_strength

        head_dim = hidden_size // n_heads

        # Build layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = input_size if i == 0 else hidden_size
            out_dim = hidden_size
            layer = BoltzmannLayer(
                input_dim=in_dim, n_heads=n_heads, head_dim=head_dim,
                output_dim=out_dim,
                conservation_mode=conservation_mode,
                conservation_strength=conservation_strength,
                device=device
            )
            self.layers.append(layer)

        # Final output
        self.output_head = nn.Linear(hidden_size, output_size, device=device)

        # Monitoring
        self.monitors = [BoltzmannMonitor() for _ in range(n_layers)]

        # Continuous learning state
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.step_count = 0
        self.history: List[BoltzmannMetrics] = []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[List[float]]]:
        """
        Forward pass through conserved layers.

        Returns (output, total_conservation_loss_tensor, all_head_entropies).
        """
        total_conservation_loss = torch.tensor(0.0, device=x.device)
        all_head_entropies = []

        h = x
        for i, layer in enumerate(self.layers):
            h, cons_loss, head_ents = layer(h)
            h = F.gelu(h)  # Activation between layers
            total_conservation_loss = total_conservation_loss + cons_loss
            all_head_entropies.append(head_ents)

        output = self.output_head(h)
        return output, total_conservation_loss, all_head_entropies

    def continuous_step(self, x: torch.Tensor, y_true: torch.Tensor) -> BoltzmannMetrics:
        """
        Single step of PAC-conserved continuous learning.
        """
        self.step_count += 1

        # Forward
        y_pred, conservation_loss, all_head_entropies = self.forward(x)

        # Task loss
        task_loss = F.mse_loss(y_pred, y_true)

        # Combined loss (both are differentiable tensors now)
        total_loss = task_loss + conservation_loss

        # Backward — conservation_loss now has gradient path
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        # Update monitors
        for i, (monitor, head_ents) in enumerate(zip(self.monitors, all_head_entropies)):
            target = self.layers[i].projector._target_budget or sum(head_ents)
            monitor.update(head_ents, target)

        # Flatten head entropies for metrics
        flat_entropies = [e for layer_ents in all_head_entropies for e in layer_ents]
        head_phases = [classify_sec_phase(e) for e in flat_entropies]

        # Compute aggregate metrics
        mean_ent = float(np.mean(flat_entropies)) if flat_entropies else 0
        std_ent = float(np.std(flat_entropies)) if flat_entropies else 0
        cv = std_ent / mean_ent if mean_ent > 1e-8 else 0

        total_budget = sum(flat_entropies)
        target_budget = sum(
            layer.projector._target_budget or 0
            for layer in self.layers
        )
        violation_pct = ((total_budget - target_budget) /
                         max(target_budget, 1e-8) * 100) if target_budget > 0 else 0

        metrics = BoltzmannMetrics(
            loss=total_loss.item(),
            task_loss=task_loss.item(),
            conservation_loss=float(conservation_loss.item()) if torch.is_tensor(conservation_loss) else float(conservation_loss),
            total_budget=total_budget,
            target_budget=target_budget,
            violation_pct=violation_pct,
            compensation_ratio=float(np.mean(
                [m.mean_compensation() for m in self.monitors])),
            head_entropies=flat_entropies,
            head_phases=head_phases,
            mean_entropy=mean_ent,
            entropy_std=std_ent,
            head_cv=cv,
            step=self.step_count,
        )

        self.history.append(metrics)
        return metrics

    def continuous_train(
        self,
        data_stream,
        max_steps: int = 1000,
        log_interval: int = 100,
    ) -> List[BoltzmannMetrics]:
        """Continuous training on a data stream."""
        history = []

        for step, (x, y) in enumerate(data_stream):
            if step >= max_steps:
                break

            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
            if not torch.is_tensor(y):
                y = torch.tensor(y, dtype=torch.float32, device=self.device)

            if x.dim() == 1:
                x = x.unsqueeze(0)
            if y.dim() == 1:
                y = y.unsqueeze(0)

            metrics = self.continuous_step(x, y)
            history.append(metrics)

            if step % log_interval == 0:
                phases = {}
                for p in metrics.head_phases:
                    phases[p] = phases.get(p, 0) + 1
                phase_str = " ".join(f"{k}:{v}" for k, v in sorted(phases.items()))
                print(f"  Step {step:5d}: task_loss={metrics.task_loss:.6f}  "
                      f"budget={metrics.total_budget:.3f}  "
                      f"violation={metrics.violation_pct:+.1f}%  "
                      f"comp={metrics.compensation_ratio:.3f}  "
                      f"[{phase_str}]")

        return history

    def get_conservation_summary(self) -> Dict:
        """Get summary of PAC conservation across all layers."""
        summaries = []
        for i, monitor in enumerate(self.monitors):
            s = monitor.get_summary()
            s['layer'] = i
            summaries.append(s)

        # Overall
        total_budget = sum(s['current_budget'] for s in summaries)
        total_target = sum(s['target_budget'] for s in summaries)

        return {
            'n_layers': self.n_layers_count,
            'n_heads': self.n_heads,
            'total_heads': self.n_layers_count * self.n_heads,
            'conservation_mode': self.conservation_mode,
            'conservation_strength': self.conservation_strength,
            'total_budget': total_budget,
            'total_target': total_target,
            'total_violation_pct': ((total_budget - total_target) /
                                    max(total_target, 1e-8) * 100),
            'mean_compensation': float(np.mean(
                [m.mean_compensation() for m in self.monitors])),
            'budget_stability': float(np.mean(
                [m.budget_stability() for m in self.monitors])),
            'layers': summaries,
        }


# ─── Data Generators ────────────────────────────────────────────────

def create_factual_stream(n_samples: int = 500):
    """
    Stream of learnable, structured sequences.
    Analogous to "factual" prompts — clear patterns to learn.
    """
    for i in range(n_samples):
        # Simple function: y = sin(x) + small noise
        x = np.random.uniform(-np.pi, np.pi, size=1)
        y = np.sin(x) + np.random.normal(0, 0.01, size=1)
        yield x.astype(np.float32), y.astype(np.float32)


def create_hallucination_stream(n_samples: int = 500):
    """
    Stream that forces the model outside its training distribution.
    Analogous to "hallucination" prompts — model must fabricate.
    """
    for i in range(n_samples):
        # Random noise with no learnable pattern
        x = np.random.uniform(-10, 10, size=1)
        y = np.random.uniform(-10, 10, size=1)  # No relationship to x
        yield x.astype(np.float32), y.astype(np.float32)


def create_mixed_stream(n_factual: int = 300, n_halluc: int = 200):
    """
    Mixed stream: first factual (learnable), then hallucination (noise).
    Tests whether conservation stabilizes during factual and tightens
    during hallucination.
    """
    # Phase 1: Learnable pattern
    for i in range(n_factual):
        x = np.random.uniform(-np.pi, np.pi, size=1)
        y = np.sin(x) + np.random.normal(0, 0.01, size=1)
        yield x.astype(np.float32), y.astype(np.float32)

    # Phase 2: Random noise (hallucination-analogue)
    for i in range(n_halluc):
        x = np.random.uniform(-10, 10, size=1)
        y = np.random.uniform(-10, 10, size=1)
        yield x.astype(np.float32), y.astype(np.float32)


def create_fibonacci_ratio_stream(n_samples: int = 500):
    """Fibonacci ratio stream for PAC-resonance testing."""
    fibs = [1, 1]
    for _ in range(50):
        fibs.append(fibs[-1] + fibs[-2])

    for _ in range(n_samples):
        idx = np.random.randint(2, len(fibs) - 1)
        x = np.array([fibs[idx] / fibs[idx-1]], dtype=np.float32)
        y = np.array([fibs[idx+1] / fibs[idx]], dtype=np.float32)
        yield x, y


# ─── Main Demo ──────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("  TinyCIMM-Boltzmann: PAC-Conserved Multi-Head Learning")
    print("  Testing whether enforced conservation prevents hallucination")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Test 1: Factual stream (should conserve well) ──
    print("\n─── Test 1: Factual Stream (should conserve) ───")
    model = TinyCIMMBoltzmann(
        input_size=1, hidden_size=32, output_size=1,
        n_heads=4, n_layers=2,
        conservation_mode='soft', conservation_strength=1.0,
        device=device,
    )

    stream = create_factual_stream(300)
    history = model.continuous_train(stream, max_steps=300, log_interval=50)

    summary = model.get_conservation_summary()
    print(f"\n  Final Conservation State:")
    print(f"    Budget: {summary['total_budget']:.3f} "
          f"(target: {summary['total_target']:.3f})")
    print(f"    Violation: {summary['total_violation_pct']:+.1f}%")
    print(f"    Stability: {summary['budget_stability']:.3f}")
    print(f"    Compensation: {summary['mean_compensation']:.3f}")

    fact_final_violation = summary['total_violation_pct']
    fact_final_loss = history[-1].task_loss

    # ── Test 2: Hallucination stream (should show violation pressure) ──
    print("\n─── Test 2: Hallucination Stream (noise = PAC stress test) ───")
    model2 = TinyCIMMBoltzmann(
        input_size=1, hidden_size=32, output_size=1,
        n_heads=4, n_layers=2,
        conservation_mode='soft', conservation_strength=1.0,
        device=device,
    )

    stream2 = create_hallucination_stream(300)
    history2 = model2.continuous_train(stream2, max_steps=300, log_interval=50)

    summary2 = model2.get_conservation_summary()
    print(f"\n  Final Conservation State:")
    print(f"    Budget: {summary2['total_budget']:.3f} "
          f"(target: {summary2['total_target']:.3f})")
    print(f"    Violation: {summary2['total_violation_pct']:+.1f}%")
    print(f"    Stability: {summary2['budget_stability']:.3f}")
    print(f"    Compensation: {summary2['mean_compensation']:.3f}")

    hall_final_violation = summary2['total_violation_pct']
    hall_final_loss = history2[-1].task_loss

    # ── Test 3: Conservation OFF (baseline — should show more violation) ──
    print("\n─── Test 3: No Conservation (baseline comparison) ───")
    model3 = TinyCIMMBoltzmann(
        input_size=1, hidden_size=32, output_size=1,
        n_heads=4, n_layers=2,
        conservation_mode='none', conservation_strength=0.0,
        device=device,
    )

    stream3 = create_hallucination_stream(300)
    history3 = model3.continuous_train(stream3, max_steps=300, log_interval=50)

    summary3 = model3.get_conservation_summary()
    print(f"\n  Final Conservation State:")
    print(f"    Budget: {summary3['total_budget']:.3f} "
          f"(target: {summary3['total_target']:.3f})")
    print(f"    Violation: {summary3['total_violation_pct']:+.1f}%")
    print(f"    Stability: {summary3['budget_stability']:.3f}")
    print(f"    Compensation: {summary3['mean_compensation']:.3f}")

    nocons_final_violation = summary3['total_violation_pct']

    # ── Comparison ──
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)
    print(f"  {'Condition':25s} {'Violation%':>10s} {'Stability':>10s} "
          f"{'Compensation':>12s} {'Task Loss':>10s}")
    print(f"  {'Factual + Conservation':25s} "
          f"{fact_final_violation:+9.1f}% "
          f"{summary['budget_stability']:10.3f} "
          f"{summary['mean_compensation']:12.3f} "
          f"{fact_final_loss:10.6f}")
    print(f"  {'Halluc + Conservation':25s} "
          f"{hall_final_violation:+9.1f}% "
          f"{summary2['budget_stability']:10.3f} "
          f"{summary2['mean_compensation']:12.3f} "
          f"{hall_final_loss:10.6f}")
    print(f"  {'Halluc NO Conservation':25s} "
          f"{nocons_final_violation:+9.1f}% "
          f"{summary3['budget_stability']:10.3f} "
          f"{summary3['mean_compensation']:12.3f} "
          f"{history3[-1].task_loss:10.6f}")

    # ── PAC Verdict ──
    print(f"\n  PAC CONSERVATION TEST:")
    if abs(fact_final_violation) < abs(hall_final_violation):
        print(f"  ✓ Factual has less violation than hallucination")
    else:
        print(f"  ? Factual violation >= hallucination violation")

    if abs(hall_final_violation) < abs(nocons_final_violation):
        print(f"  ✓ Conservation constraint reduces violation under noise")
    else:
        print(f"  ? Conservation constraint did not reduce violation")

    print(f"\n  Exp_12 parallel: Does the CONSERVED model handle noise better?")
    print(f"  Unconstrained violation: {nocons_final_violation:+.1f}%")
    print(f"  Constrained violation:   {hall_final_violation:+.1f}%")
    ratio = abs(nocons_final_violation) / max(abs(hall_final_violation), 0.01)
    print(f"  Constraint reduction:    {ratio:.1f}x")
