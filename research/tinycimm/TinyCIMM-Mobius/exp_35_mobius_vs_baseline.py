"""
Experiment 35: Möbius Neural Networks vs Conventional Baselines

HYPOTHESIS: Möbius networks learn certain tasks faster/better because:
1. Built-in nonlinearity (no activation needed)
2. Composition structure (depth = matrix multiplication)
3. Natural φ-attractors from Fibonacci seeding

TEST TASKS:
1. Möbius inversion: Learn to invert M(z) - trivial for Möbius, hard for standard
2. Continued fraction dynamics: z → 1/(z+1) iteration - natural Möbius structure
3. Phase estimation: Extract phase from noisy complex signal
4. Logistic map: r*x*(1-x) at edge of chaos - connects to Feigenbaum

METRICS:
- Convergence speed (epochs to reach threshold)
- Final accuracy (MSE on test set)
- Parameter efficiency (accuracy per parameter)
- Stability (variance across runs)

BASELINES:
- Standard MLP with same parameter count
- Complex-valued MLP (fair comparison for complex tasks)
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
PHI_INV = PHI - 1


# ============================================================
# MÖBIUS NETWORK (PyTorch version)
# ============================================================

class MobiusNeuronTorch(nn.Module):
    """Single Möbius neuron: M(z) = (az+b)/(cz+d)"""
    
    def __init__(self, init: str = 'fibonacci'):
        super().__init__()
        
        if init == 'fibonacci':
            self.a = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device))
            self.b = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device))
            self.c = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device))
            self.d = nn.Parameter(torch.tensor(0.01, dtype=torch.float32, device=device))
        elif init == 'identity':
            self.a = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device))
            self.b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
            self.c = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
            self.d = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device))
        else:
            self.a = nn.Parameter(torch.randn(1, device=device).squeeze() * 0.5 + 0.5)
            self.b = nn.Parameter(torch.randn(1, device=device).squeeze() * 0.3)
            self.c = nn.Parameter(torch.randn(1, device=device).squeeze() * 0.3)
            self.d = nn.Parameter(torch.randn(1, device=device).squeeze() * 0.5 + 0.5)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply Möbius transformation."""
        denom = self.c * z + self.d
        return (self.a * z + self.b) / (denom + 1e-8 * torch.sign(denom))
    
    @property
    def param_count(self) -> int:
        return 4


class MobiusLayer(nn.Module):
    """Layer of parallel Möbius neurons with aggregation."""
    
    def __init__(self, n_neurons: int = 4, init: str = 'random'):
        super().__init__()
        self.neurons = nn.ModuleList([
            MobiusNeuronTorch(init=init) for _ in range(n_neurons)
        ])
        self.weights = nn.Parameter(torch.ones(n_neurons, device=device) / n_neurons)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        outputs = torch.stack([n(z) for n in self.neurons], dim=-1)
        weights = torch.softmax(self.weights, dim=0)
        return (outputs * weights).sum(dim=-1)
    
    @property
    def param_count(self) -> int:
        return len(self.neurons) * 4 + len(self.neurons)


class MobiusNetwork(nn.Module):
    """Multi-layer Möbius network."""
    
    def __init__(self, n_layers: int = 2, neurons_per_layer: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            MobiusLayer(neurons_per_layer) for _ in range(n_layers)
        ])
        self.output_scale = nn.Parameter(torch.tensor(1.0, device=device))
        self.output_bias = nn.Parameter(torch.tensor(0.0, device=device))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        for layer in self.layers:
            z = layer(z)
        return self.output_scale * z + self.output_bias
    
    @property
    def param_count(self) -> int:
        return sum(l.param_count for l in self.layers) + 2


# ============================================================
# BASELINE: Standard MLP
# ============================================================

class StandardMLP(nn.Module):
    """Conventional MLP with matched parameter count."""
    
    def __init__(self, hidden_sizes: List[int], activation: str = 'relu'):
        super().__init__()
        
        layers = []
        in_size = 1
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h, device=device))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'gelu':
                layers.append(nn.GELU())
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
# TEST TASK 1: Continued Fraction Dynamics
# ============================================================

def generate_continued_fraction_data(n_samples: int = 1000, n_steps: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate continued fraction iteration: z → 1/(1+z)
    
    This is a Möbius transformation! The Möbius network should learn it exactly.
    After n_steps iterations, z_n converges to 1/φ = 0.618...
    
    Task: Given z_0, predict z_n
    """
    z0 = torch.rand(n_samples, device=device) * 2 - 1  # [-1, 1]
    
    # Iterate: z → 1/(1+z)
    z = z0.clone()
    for _ in range(n_steps):
        z = 1.0 / (1.0 + z + 1e-8)
    
    return z0, z


def continued_fraction_exact(z: torch.Tensor, n_steps: int = 5) -> torch.Tensor:
    """Exact continued fraction iteration (for validation)."""
    for _ in range(n_steps):
        z = 1.0 / (1.0 + z + 1e-8)
    return z


# ============================================================
# TEST TASK 2: Möbius Inversion
# ============================================================

def generate_mobius_inversion_data(n_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Generate random Möbius transformation and learn to invert it.
    
    Given: y = M(x) = (ax+b)/(cx+d)
    Learn: x = M^{-1}(y) = (dy-b)/(-cy+a)
    
    Returns: (y, x, params) where params are the true (a,b,c,d)
    """
    # Random Möbius parameters
    a, b, c, d = 1.5, 0.7, 0.3, 1.2
    
    # Generate x values
    x = torch.rand(n_samples, device=device) * 2 - 1  # [-1, 1]
    
    # Forward transformation
    y = (a * x + b) / (c * x + d + 1e-8)
    
    return y, x, {'a': a, 'b': b, 'c': c, 'd': d}


# ============================================================
# TEST TASK 3: Golden Ratio Convergence
# ============================================================

def generate_golden_convergence_data(n_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Task: Learn the mapping from initial condition to convergence rate toward φ.
    
    For z_0, iterate z → (z+1)/z until close to φ.
    Output: number of steps (normalized) to reach |z - φ| < 0.01
    """
    z0 = torch.rand(n_samples, device=device) * 3 + 0.5  # [0.5, 3.5]
    
    steps = torch.zeros(n_samples, device=device)
    z = z0.clone()
    
    for i in range(50):
        dist = torch.abs(z - PHI)
        not_converged = dist > 0.01
        steps += not_converged.float()
        z = torch.where(not_converged, (z + 1) / (z + 1e-8), z)
    
    # Normalize to [0, 1]
    steps = steps / 50.0
    
    return z0, steps


# ============================================================
# TRAINING AND EVALUATION
# ============================================================

@dataclass
class TrainingResult:
    model_name: str
    task_name: str
    param_count: int
    epochs_to_threshold: int  # -1 if never reached
    final_loss: float
    loss_history: List[float]
    convergence_threshold: float = 0.001


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    model_name: str,
    task_name: str,
    max_epochs: int = 500,
    lr: float = 0.01,
    threshold: float = 0.001,
    verbose: bool = False
) -> TrainingResult:
    """Train model and track metrics."""
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    loss_history = []
    epochs_to_threshold = -1
    
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        
        pred = model(X_train)
        loss = criterion(pred, y_train)
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test).item()
        
        loss_history.append(test_loss)
        
        if test_loss < threshold and epochs_to_threshold == -1:
            epochs_to_threshold = epoch
        
        if verbose and epoch % 100 == 0:
            print(f"  Epoch {epoch}: train={loss.item():.6f}, test={test_loss:.6f}")
    
    return TrainingResult(
        model_name=model_name,
        task_name=task_name,
        param_count=model.param_count if hasattr(model, 'param_count') else sum(p.numel() for p in model.parameters()),
        epochs_to_threshold=epochs_to_threshold,
        final_loss=loss_history[-1],
        loss_history=loss_history,
        convergence_threshold=threshold
    )


def run_comparison(task_name: str, X: torch.Tensor, y: torch.Tensor, 
                   n_runs: int = 5, verbose: bool = True) -> Dict:
    """Run comparison between Möbius and MLP on a task."""
    
    # Split data
    n = len(X)
    n_train = int(0.8 * n)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    results = {'mobius': [], 'mlp': []}
    
    for run in range(n_runs):
        if verbose:
            print(f"\n--- Run {run+1}/{n_runs} ---")
        
        # Möbius network: 2 layers × 4 neurons = 42 params
        mobius = MobiusNetwork(n_layers=2, neurons_per_layer=4).to(device)
        mobius_params = mobius.param_count
        
        # MLP with similar param count: [8, 4] hidden = 1×8+8 + 8×4+4 + 4×1+1 = 57 params
        # Adjust to match: [6, 4] = 1×6+6 + 6×4+4 + 4×1+1 = 41 params
        mlp = StandardMLP(hidden_sizes=[6, 4], activation='tanh').to(device)
        mlp_params = mlp.param_count
        
        if verbose:
            print(f"  Möbius params: {mobius_params}")
            print(f"  MLP params: {mlp_params}")
        
        # Train both
        if verbose:
            print(f"  Training Möbius...")
        mobius_result = train_model(
            mobius, X_train, y_train, X_test, y_test,
            "MobiusNetwork", task_name, max_epochs=500, lr=0.01
        )
        results['mobius'].append(mobius_result)
        
        if verbose:
            print(f"  Training MLP...")
        mlp_result = train_model(
            mlp, X_train, y_train, X_test, y_test,
            "StandardMLP", task_name, max_epochs=500, lr=0.01
        )
        results['mlp'].append(mlp_result)
        
        if verbose:
            print(f"  Möbius final: {mobius_result.final_loss:.6f}, epochs to threshold: {mobius_result.epochs_to_threshold}")
            print(f"  MLP final: {mlp_result.final_loss:.6f}, epochs to threshold: {mlp_result.epochs_to_threshold}")
    
    return results


def analyze_results(results: Dict) -> Dict:
    """Analyze comparison results."""
    
    analysis = {}
    
    for model_name in ['mobius', 'mlp']:
        final_losses = [r.final_loss for r in results[model_name]]
        epochs_to_thresh = [r.epochs_to_threshold for r in results[model_name]]
        
        # Filter out -1 (never converged)
        converged_epochs = [e for e in epochs_to_thresh if e >= 0]
        
        analysis[model_name] = {
            'mean_final_loss': np.mean(final_losses),
            'std_final_loss': np.std(final_losses),
            'mean_epochs_to_threshold': np.mean(converged_epochs) if converged_epochs else -1,
            'convergence_rate': len(converged_epochs) / len(epochs_to_thresh),
            'param_count': results[model_name][0].param_count
        }
    
    # Compute relative metrics
    if analysis['mlp']['mean_final_loss'] > 0:
        analysis['loss_ratio'] = analysis['mobius']['mean_final_loss'] / analysis['mlp']['mean_final_loss']
    else:
        analysis['loss_ratio'] = float('inf')
    
    if analysis['mlp']['mean_epochs_to_threshold'] > 0 and analysis['mobius']['mean_epochs_to_threshold'] > 0:
        analysis['speed_ratio'] = analysis['mlp']['mean_epochs_to_threshold'] / analysis['mobius']['mean_epochs_to_threshold']
    else:
        analysis['speed_ratio'] = None
    
    return analysis


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    print("=" * 70)
    print("EXPERIMENT 35: Möbius Neural Networks vs Conventional Baselines")
    print("=" * 70)
    
    all_results = {}
    
    # Task 1: Continued Fraction Dynamics
    print("\n" + "=" * 50)
    print("TASK 1: Continued Fraction Dynamics")
    print("z → 1/(1+z) iterated 5 times")
    print("This IS a Möbius transformation - Möbius should have advantage")
    print("=" * 50)
    
    X, y = generate_continued_fraction_data(n_samples=2000, n_steps=5)
    results_cf = run_comparison("continued_fraction", X, y, n_runs=5)
    analysis_cf = analyze_results(results_cf)
    all_results['continued_fraction'] = analysis_cf
    
    print(f"\n  RESULTS:")
    print(f"    Möbius: loss={analysis_cf['mobius']['mean_final_loss']:.6f} ± {analysis_cf['mobius']['std_final_loss']:.6f}")
    print(f"    MLP:    loss={analysis_cf['mlp']['mean_final_loss']:.6f} ± {analysis_cf['mlp']['std_final_loss']:.6f}")
    print(f"    Loss ratio (Möbius/MLP): {analysis_cf['loss_ratio']:.4f}")
    if analysis_cf['speed_ratio']:
        print(f"    Speed ratio (MLP/Möbius): {analysis_cf['speed_ratio']:.2f}x")
    
    # Task 2: Möbius Inversion
    print("\n" + "=" * 50)
    print("TASK 2: Möbius Inversion")
    print("Given y = M(x), learn x = M^{-1}(y)")
    print("Möbius can represent inverse exactly with 4 params")
    print("=" * 50)
    
    y, X, params = generate_mobius_inversion_data(n_samples=2000)
    results_inv = run_comparison("mobius_inversion", y, X, n_runs=5)
    analysis_inv = analyze_results(results_inv)
    all_results['mobius_inversion'] = analysis_inv
    
    print(f"\n  RESULTS:")
    print(f"    Möbius: loss={analysis_inv['mobius']['mean_final_loss']:.6f} ± {analysis_inv['mobius']['std_final_loss']:.6f}")
    print(f"    MLP:    loss={analysis_inv['mlp']['mean_final_loss']:.6f} ± {analysis_inv['mlp']['std_final_loss']:.6f}")
    print(f"    Loss ratio (Möbius/MLP): {analysis_inv['loss_ratio']:.4f}")
    
    # Task 3: Golden Convergence
    print("\n" + "=" * 50)
    print("TASK 3: Golden Ratio Convergence")
    print("Predict convergence speed to φ")
    print("Möbius has φ as natural attractor")
    print("=" * 50)
    
    X, y = generate_golden_convergence_data(n_samples=2000)
    results_gold = run_comparison("golden_convergence", X, y, n_runs=5)
    analysis_gold = analyze_results(results_gold)
    all_results['golden_convergence'] = analysis_gold
    
    print(f"\n  RESULTS:")
    print(f"    Möbius: loss={analysis_gold['mobius']['mean_final_loss']:.6f} ± {analysis_gold['mobius']['std_final_loss']:.6f}")
    print(f"    MLP:    loss={analysis_gold['mlp']['mean_final_loss']:.6f} ± {analysis_gold['mlp']['std_final_loss']:.6f}")
    print(f"    Loss ratio (Möbius/MLP): {analysis_gold['loss_ratio']:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    mobius_wins = 0
    for task, analysis in all_results.items():
        winner = "Möbius" if analysis['loss_ratio'] < 1.0 else "MLP"
        advantage = abs(1 - analysis['loss_ratio']) * 100
        if winner == "Möbius":
            mobius_wins += 1
        print(f"  {task}: {winner} wins by {advantage:.1f}%")
    
    print(f"\n  Overall: Möbius wins {mobius_wins}/3 tasks")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'device': str(device),
        'tasks': {}
    }
    
    for task, analysis in all_results.items():
        output['tasks'][task] = {
            'mobius_loss': analysis['mobius']['mean_final_loss'],
            'mlp_loss': analysis['mlp']['mean_final_loss'],
            'loss_ratio': analysis['loss_ratio'],
            'mobius_params': analysis['mobius']['param_count'],
            'mlp_params': analysis['mlp']['param_count']
        }
    
    with open(f'results/exp_35_{timestamp}.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Results saved to results/exp_35_{timestamp}.json")
    
    return all_results


if __name__ == "__main__":
    main()
