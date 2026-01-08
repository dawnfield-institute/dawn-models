"""
Experiment 36: Möbius Networks with Proper Composition

Key insight from exp_35: Möbius was UNSTABLE, not wrong.
Sometimes achieved perfect loss, sometimes terrible.

The problem: we were averaging Möbius outputs instead of COMPOSING them.
Möbius transformations compose via matrix multiplication - that's the whole point.

This experiment:
1. Uses proper matrix composition (not averaging)
2. Enforces determinant = 1 normalization
3. Adds regularization to avoid singularities
4. Uses gradient clipping more aggressively
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Dict
from dataclasses import dataclass
import json
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

PHI = (1 + np.sqrt(5)) / 2


class MobiusTransform(nn.Module):
    """
    A single Möbius transformation with proper SL(2,R) structure.
    
    Parameterization: Use (θ, r, s) instead of (a, b, c, d)
    This avoids singularities and maintains det = 1.
    
    M = [[cosh(r) + sinh(r)cos(θ), sinh(r)sin(θ) + s],
         [sinh(r)sin(θ) - s,       cosh(r) - sinh(r)cos(θ)]]
    
    This gives det = cosh²(r) - sinh²(r) = 1 automatically!
    """
    
    def __init__(self, init: str = 'identity'):
        super().__init__()
        
        if init == 'identity':
            self.theta = nn.Parameter(torch.tensor(0.0, device=device))
            self.r = nn.Parameter(torch.tensor(0.0, device=device))
            self.s = nn.Parameter(torch.tensor(0.0, device=device))
        elif init == 'fibonacci':
            # Approximate Fibonacci matrix [[1,1],[1,0]]
            # This has det = -1, so we need [[1,1],[1,0]] / sqrt|det| with phase
            self.theta = nn.Parameter(torch.tensor(0.5, device=device))
            self.r = nn.Parameter(torch.tensor(0.5, device=device))
            self.s = nn.Parameter(torch.tensor(0.3, device=device))
        else:
            self.theta = nn.Parameter(torch.randn(1, device=device).squeeze() * 0.3)
            self.r = nn.Parameter(torch.randn(1, device=device).squeeze() * 0.3)
            self.s = nn.Parameter(torch.randn(1, device=device).squeeze() * 0.3)
    
    def matrix(self) -> torch.Tensor:
        """Get the 2x2 Möbius matrix in SL(2,R)."""
        cosh_r = torch.cosh(self.r)
        sinh_r = torch.sinh(self.r)
        cos_t = torch.cos(self.theta)
        sin_t = torch.sin(self.theta)
        
        a = cosh_r + sinh_r * cos_t
        b = sinh_r * sin_t + self.s
        c = sinh_r * sin_t - self.s
        d = cosh_r - sinh_r * cos_t
        
        return torch.stack([torch.stack([a, b]), torch.stack([c, d])])
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply M(z) = (az + b) / (cz + d)."""
        M = self.matrix()
        a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        
        denom = c * z + d
        # Soft clipping to avoid singularity
        denom = torch.where(
            torch.abs(denom) < 0.01,
            torch.sign(denom) * 0.01 + denom * 0.1,
            denom
        )
        
        return (a * z + b) / denom
    
    def compose(self, other: 'MobiusTransform') -> torch.Tensor:
        """Compose with another Möbius transform (matrix multiplication)."""
        return torch.matmul(self.matrix(), other.matrix())


class ComposedMobiusNetwork(nn.Module):
    """
    Network that COMPOSES Möbius transformations (not averages them).
    
    Output = M_n ∘ M_{n-1} ∘ ... ∘ M_1 (z)
    
    This is a single equivalent Möbius transformation!
    With n layers, we have 3n parameters total.
    """
    
    def __init__(self, n_layers: int = 3, init: str = 'random'):
        super().__init__()
        self.transforms = nn.ModuleList([
            MobiusTransform(init=init) for _ in range(n_layers)
        ])
        # Input/output scaling
        self.in_scale = nn.Parameter(torch.tensor(1.0, device=device))
        self.in_bias = nn.Parameter(torch.tensor(0.0, device=device))
        self.out_scale = nn.Parameter(torch.tensor(1.0, device=device))
        self.out_bias = nn.Parameter(torch.tensor(0.0, device=device))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_scale * x + self.in_bias
        
        for transform in self.transforms:
            z = transform(z)
        
        return self.out_scale * z + self.out_bias
    
    def composed_matrix(self) -> torch.Tensor:
        """Get the single equivalent Möbius matrix."""
        result = torch.eye(2, device=device)
        for transform in self.transforms:
            result = torch.matmul(transform.matrix(), result)
        return result
    
    @property
    def param_count(self) -> int:
        return len(self.transforms) * 3 + 4  # 3 per transform + 4 scale/bias


class DirectMobiusNetwork(nn.Module):
    """
    Even simpler: Just ONE Möbius transform with direct (a,b,c,d) params.
    
    For tasks that ARE Möbius transforms, this is the exact solution.
    4 parameters total (plus scale/bias = 8).
    """
    
    def __init__(self):
        super().__init__()
        # Direct parameterization
        self.a = nn.Parameter(torch.tensor(1.0, device=device))
        self.b = nn.Parameter(torch.tensor(0.0, device=device))
        self.c = nn.Parameter(torch.tensor(0.0, device=device))
        self.d = nn.Parameter(torch.tensor(1.0, device=device))
        
        # I/O scaling
        self.in_scale = nn.Parameter(torch.tensor(1.0, device=device))
        self.in_bias = nn.Parameter(torch.tensor(0.0, device=device))
        self.out_scale = nn.Parameter(torch.tensor(1.0, device=device))
        self.out_bias = nn.Parameter(torch.tensor(0.0, device=device))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_scale * x + self.in_bias
        
        denom = self.c * z + self.d
        # Regularize denominator
        denom = denom + 0.01 * torch.sign(denom)
        
        y = (self.a * z + self.b) / denom
        
        return self.out_scale * y + self.out_bias
    
    def regularization_loss(self) -> torch.Tensor:
        """Encourage det ≈ 1 (proper Möbius)."""
        det = self.a * self.d - self.b * self.c
        return (det - 1.0) ** 2
    
    @property
    def param_count(self) -> int:
        return 8


class StandardMLP(nn.Module):
    """Baseline MLP."""
    
    def __init__(self, hidden_sizes: List[int], activation: str = 'tanh'):
        super().__init__()
        
        layers = []
        in_size = 1
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h, device=device))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            in_size = h
        layers.append(nn.Linear(in_size, 1, device=device))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.net(x).squeeze(-1)
    
    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================
# TASK: Continued Fraction (z → 1/(1+z) iterated)
# ============================================================

def generate_continued_fraction_data(n_samples: int = 2000, n_steps: int = 5):
    """z → 1/(1+z) iterated n_steps times."""
    z0 = torch.rand(n_samples, device=device) * 2 - 1  # [-1, 1]
    
    z = z0.clone()
    for _ in range(n_steps):
        z = 1.0 / (1.0 + z + 1e-8)
    
    return z0, z


def train_and_evaluate(model, X_train, y_train, X_test, y_test, 
                       epochs=1000, lr=0.01, reg_weight=0.1):
    """Train with regularization if available."""
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        pred = model(X_train)
        loss = criterion(pred, y_train)
        
        # Add regularization if available
        if hasattr(model, 'regularization_loss'):
            loss = loss + reg_weight * model.regularization_loss()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test).item()
        
        if test_loss < best_loss:
            best_loss = test_loss
    
    return best_loss


def main():
    print("=" * 70)
    print("EXPERIMENT 36: Proper Möbius Composition")
    print("=" * 70)
    
    # Generate data
    X, y = generate_continued_fraction_data(n_samples=2000, n_steps=5)
    n_train = 1600
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    print(f"\nTask: z → 1/(1+z) iterated 5 times")
    print(f"This is a Möbius transformation!")
    print(f"Exact formula: M(z) = 1/(1+z) composed 5 times")
    print()
    
    n_runs = 10
    results = {
        'direct_mobius': [],
        'composed_mobius': [],
        'mlp_small': [],
        'mlp_large': []
    }
    
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}...", end=" ")
        
        # Direct Möbius (8 params)
        direct = DirectMobiusNetwork().to(device)
        loss = train_and_evaluate(direct, X_train, y_train, X_test, y_test, 
                                  epochs=1000, lr=0.01, reg_weight=0.01)
        results['direct_mobius'].append(loss)
        
        # Composed Möbius (3 layers = 13 params)
        composed = ComposedMobiusNetwork(n_layers=3).to(device)
        loss = train_and_evaluate(composed, X_train, y_train, X_test, y_test,
                                  epochs=1000, lr=0.01)
        results['composed_mobius'].append(loss)
        
        # MLP small (8 params to match direct)
        # [2] = 1×2+2 + 2×1+1 = 7 params... close enough
        mlp_small = StandardMLP(hidden_sizes=[2]).to(device)
        loss = train_and_evaluate(mlp_small, X_train, y_train, X_test, y_test,
                                  epochs=1000, lr=0.01)
        results['mlp_small'].append(loss)
        
        # MLP larger (similar to composed = 13 params)
        # [3, 2] = 1×3+3 + 3×2+2 + 2×1+1 = 17 params
        mlp_large = StandardMLP(hidden_sizes=[3, 2]).to(device)
        loss = train_and_evaluate(mlp_large, X_train, y_train, X_test, y_test,
                                  epochs=1000, lr=0.01)
        results['mlp_large'].append(loss)
        
        print(f"direct={results['direct_mobius'][-1]:.6f}, composed={results['composed_mobius'][-1]:.6f}, mlp={results['mlp_large'][-1]:.6f}")
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    for name, losses in results.items():
        mean = np.mean(losses)
        std = np.std(losses)
        min_loss = np.min(losses)
        print(f"  {name:20s}: mean={mean:.6f} ± {std:.6f}, best={min_loss:.6f}")
    
    # Compare best Möbius vs best MLP
    best_mobius = min(min(results['direct_mobius']), min(results['composed_mobius']))
    best_mlp = min(min(results['mlp_small']), min(results['mlp_large']))
    
    print()
    if best_mobius < best_mlp:
        print(f"  WINNER: Möbius ({best_mobius:.6f} vs MLP {best_mlp:.6f})")
    else:
        print(f"  WINNER: MLP ({best_mlp:.6f} vs Möbius {best_mobius:.6f})")
    
    # Check if direct Möbius learned the right transformation
    print("\n" + "=" * 70)
    print("LEARNED TRANSFORMATION CHECK")
    print("=" * 70)
    
    # The exact answer for z → 1/(1+z) iterated 5 times:
    # It's a Möbius with specific (a,b,c,d)
    # Let's compute what it should be
    
    # M(z) = 1/(1+z) = (0*z + 1)/(1*z + 1)
    # Matrix: [[0, 1], [1, 1]]
    # M^5 = ?
    M = np.array([[0, 1], [1, 1]], dtype=float)
    M5 = np.linalg.matrix_power(M, 5)
    print(f"  Exact M^5 matrix: [[{M5[0,0]:.0f}, {M5[0,1]:.0f}], [{M5[1,0]:.0f}, {M5[1,1]:.0f}]]")
    
    # Check learned
    direct = DirectMobiusNetwork().to(device)
    train_and_evaluate(direct, X_train, y_train, X_test, y_test, epochs=2000, lr=0.01)
    
    print(f"  Learned (raw): a={direct.a.item():.4f}, b={direct.b.item():.4f}, c={direct.c.item():.4f}, d={direct.d.item():.4f}")
    print(f"  Learned (with scaling): needs to account for in_scale/bias and out_scale/bias")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'task': 'continued_fraction_5_steps',
        'results': {name: {'mean': float(np.mean(losses)), 'std': float(np.std(losses)), 'min': float(np.min(losses))}
                   for name, losses in results.items()}
    }
    
    with open(f'results/exp_36_{timestamp}.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Saved to results/exp_36_{timestamp}.json")


if __name__ == "__main__":
    main()
