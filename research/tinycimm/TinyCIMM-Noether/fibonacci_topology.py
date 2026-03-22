"""
Fibonacci Topology Builder for TinyCIMM-Noether

Constructs network topology where:
- Depth D=3 (from five independent Milestone 1 derivation paths)
- Layer widths follow Fibonacci ratios: F_n → F_{n-1} → F_{n-2} → F_{n-3}
- Fibonacci index n is the ONLY free parameter, chosen by MED bounds
- No hyperparameter search — topology is derived, not searched

The PAC recursion Ψ(k) = Ψ(k+1) + Ψ(k+2) has unique solution Ψ(k) = φ^(−k),
making the golden ratio a mathematical necessity for conservation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Golden ratio constants
PHI = (1 + np.sqrt(5)) / 2       # φ ≈ 1.6180339887
PHI_INV = 1.0 / PHI              # 1/φ ≈ 0.6180339887


def fibonacci(n: int) -> int:
    """Compute nth Fibonacci number (F_1=1, F_2=1, F_3=2, ...)."""
    if n <= 0:
        raise ValueError(f"Fibonacci index must be positive, got {n}")
    if n <= 2:
        return 1
    a, b = 1, 1
    for _ in range(n - 2):
        a, b = b, a + b
    return b


def fibonacci_sequence(start_index: int, count: int) -> List[int]:
    """Return count Fibonacci numbers starting from F_{start_index} descending."""
    return [fibonacci(start_index - i) for i in range(count)]


@dataclass
class FibonacciTopology:
    """
    Complete topology specification for a TinyCIMM-Noether network.

    Attributes:
        n: Fibonacci index (the only free parameter)
        depth: Network depth (always 3 for Phase A)
        layer_widths: List of layer sizes [input, hidden1, hidden2, output]
        pac_ratios: Expected PAC value ratios between layers
        conservation_pairs: List of (parent, child1, child2) index triples
            defining the Fibonacci PAC recursion structure
    """
    n: int
    depth: int
    layer_widths: List[int]
    pac_ratios: List[float]
    conservation_pairs: List[Tuple[int, int, int]]

    @property
    def num_layers(self) -> int:
        return len(self.layer_widths)

    @property
    def total_params(self) -> int:
        """Total number of weight parameters (weights + biases)."""
        total = 0
        for i in range(len(self.layer_widths) - 1):
            total += self.layer_widths[i] * self.layer_widths[i + 1]  # weights
            total += self.layer_widths[i + 1]  # biases
        return total

    def expected_value_ratio(self, layer_k: int) -> float:
        """Expected V(layer_k) / V(layer_0) from PAC recursion."""
        return PHI_INV ** layer_k


def med_select_index(input_dim: int, output_dim: int,
                     min_index: int = 4, max_index: int = 15) -> int:
    """
    Select Fibonacci index n using MED (Macro Emergence Dynamics) bounds.

    MED constrains: depth ≤ 1, nodes ≤ 3 at each level.
    For a network, n is chosen so that:
    - F_n >= input_dim (input layer can represent input)
    - F_{n-3} >= output_dim (output layer can represent output)
    - The topology is as compact as possible (minimize n)

    Parameters:
        input_dim: Dimension of input data
        output_dim: Dimension of output/target
        min_index: Minimum Fibonacci index to consider
        max_index: Maximum Fibonacci index to consider

    Returns:
        Optimal Fibonacci index n
    """
    for n in range(min_index, max_index + 1):
        f_n = fibonacci(n)
        f_n3 = fibonacci(max(1, n - 3))
        if f_n >= input_dim and f_n3 >= output_dim:
            return n
    return max_index


def build_topology(n: int, depth: int = 3,
                   input_dim: Optional[int] = None,
                   output_dim: Optional[int] = None) -> FibonacciTopology:
    """
    Build a Fibonacci topology for TinyCIMM-Noether.

    Parameters:
        n: Fibonacci index (the only free parameter)
        depth: Network depth (default 3, from M1 derivation)
        input_dim: Override input dimension (default: F_n)
        output_dim: Override output dimension (default: F_{n-depth})

    Returns:
        FibonacciTopology specification
    """
    if depth < 1:
        raise ValueError(f"Depth must be >= 1, got {depth}")

    # Layer widths follow Fibonacci descent
    widths = fibonacci_sequence(n, depth + 1)

    # Override input/output dims if specified
    if input_dim is not None:
        widths[0] = input_dim
    if output_dim is not None:
        widths[-1] = output_dim

    # PAC ratios: V(k)/V(0) = φ^(-k)
    pac_ratios = [PHI_INV ** k for k in range(depth + 1)]

    # Conservation pairs from Fibonacci recursion: Ψ(k) = Ψ(k+1) + Ψ(k+2)
    # Each pair is (parent_layer, child1_layer, child2_layer)
    conservation_pairs = []
    for k in range(depth - 1):
        conservation_pairs.append((k, k + 1, k + 2))

    return FibonacciTopology(
        n=n,
        depth=depth,
        layer_widths=widths,
        pac_ratios=pac_ratios,
        conservation_pairs=conservation_pairs,
    )


def build_topology_for_data(input_dim: int, output_dim: int,
                            depth: int = 3) -> FibonacciTopology:
    """
    Build a topology that fits the given data dimensions.

    Selects Fibonacci index n via MED bounds, then builds the topology
    with input/output overrides to match data dimensions exactly.

    Parameters:
        input_dim: Input feature dimension
        output_dim: Output/target dimension
        depth: Network depth (default 3)

    Returns:
        FibonacciTopology specification
    """
    n = med_select_index(input_dim, output_dim)
    return build_topology(n, depth=depth,
                          input_dim=input_dim,
                          output_dim=output_dim)


def topology_summary(topo: FibonacciTopology) -> str:
    """Human-readable topology summary."""
    lines = [
        f"TinyCIMM-Noether Fibonacci Topology (n={topo.n}, D={topo.depth})",
        f"  Layer widths: {' → '.join(map(str, topo.layer_widths))}",
        f"  Total parameters: {topo.total_params}",
        f"  PAC ratios: {' : '.join(f'{r:.4f}' for r in topo.pac_ratios)}",
        f"  Conservation pairs: {topo.conservation_pairs}",
    ]
    return '\n'.join(lines)
