"""
Hard Conservation Enforcement for TinyCIMM-Noether

This is NOT a soft loss penalty. After each forward pass, PAC conservation
is verified and enforced within tolerance ε. If violated, correction terms
are applied until conservation is satisfied.

The key insight: PAC conservation cannot be "approximately" satisfied and
still give correct physics. Hard enforcement ensures the network's internal
structure exactly mirrors the conservation law.

Enforcement mechanism:
1. After forward pass, compute V(k) at each layer
2. Check Fibonacci recursion: V(k) = V(k+1) + V(k+2) within ε
3. If violated, apply additive correction to layer activations
4. Re-check until all violations < ε or max iterations reached
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from fibonacci_topology import PHI, PHI_INV, FibonacciTopology
from pac_descent import LayerState, compute_value


@dataclass
class ConservationStatus:
    """Result of a conservation check."""
    satisfied: bool                      # All violations < epsilon
    violations: Dict[int, float]         # Per-pair violations
    max_violation: float                 # Worst violation
    total_violation: float               # Sum of |violations|
    corrections_applied: int             # Number of correction iterations
    layer_values: List[float]            # V(k) at each layer


@dataclass
class EnforcementConfig:
    """Configuration for hard conservation enforcement."""
    epsilon: float = 1e-4                # Conservation tolerance
    max_iterations: int = 20             # Max correction iterations
    correction_strength: float = 0.5     # How aggressively to correct (0-1)
    value_fn: str = 'l1'                 # Must match PACDescentConfig


class ConservationEnforcer:
    """
    Hard enforcement of PAC conservation in TinyCIMM-Noether.

    After each forward pass, checks whether V(k) = V(k+1) + V(k+2)
    holds within tolerance ε. If not, applies corrections to activations
    until it does.
    """

    def __init__(self, topology: FibonacciTopology,
                 config: Optional[EnforcementConfig] = None):
        self.topology = topology
        self.config = config or EnforcementConfig()
        self.enforcement_history: List[ConservationStatus] = []

    def check_conservation(self, layer_states: List[LayerState]
                           ) -> ConservationStatus:
        """
        Check whether PAC conservation holds across all layers.

        Parameters:
            layer_states: States from a forward pass

        Returns:
            ConservationStatus with violation details
        """
        values = [s.value for s in layer_states]
        violations = {}

        for parent, child1, child2 in self.topology.conservation_pairs:
            delta = values[parent] - values[child1] - values[child2]
            violations[parent] = delta

        max_viol = max(abs(v) for v in violations.values()) if violations else 0.0
        total_viol = sum(abs(v) for v in violations.values())
        satisfied = max_viol < self.config.epsilon

        return ConservationStatus(
            satisfied=satisfied,
            violations=violations,
            max_violation=max_viol,
            total_violation=total_viol,
            corrections_applied=0,
            layer_values=values,
        )

    def enforce(self, layer_states: List[LayerState]) -> ConservationStatus:
        """
        Enforce PAC conservation by correcting layer activations.

        This modifies activations IN PLACE to satisfy:
            V(k) = V(k+1) + V(k+2)  for all conservation pairs

        The correction distributes the violation between children
        proportional to the golden ratio (maintaining Fibonacci structure).

        Parameters:
            layer_states: States from a forward pass (modified in place)

        Returns:
            ConservationStatus after enforcement
        """
        status = self.check_conservation(layer_states)

        if status.satisfied:
            self.enforcement_history.append(status)
            return status

        iterations = 0
        while not status.satisfied and iterations < self.config.max_iterations:
            # Apply corrections from top (input) to bottom (output)
            for parent, child1, child2 in self.topology.conservation_pairs:
                delta = status.violations[parent]

                if abs(delta) < self.config.epsilon:
                    continue

                # Distribute correction to children proportional to φ
                # child1 gets φ/(1+φ) of the correction
                # child2 gets 1/(1+φ) of the correction
                # This preserves the Fibonacci ratio between children
                correction = delta * self.config.correction_strength
                frac1 = PHI / (1 + PHI)  # ≈ 0.618
                frac2 = 1.0 / (1 + PHI)  # ≈ 0.382

                # Apply additive correction to activations
                a1 = layer_states[child1].activations
                a2 = layer_states[child2].activations

                # Scale correction by number of neurons to distribute evenly
                if a1.size > 0:
                    per_neuron_1 = (correction * frac1) / max(1, a1.shape[-1])
                    layer_states[child1].activations = a1 + per_neuron_1
                    layer_states[child1].value = compute_value(
                        layer_states[child1].activations, self.config.value_fn)

                if a2.size > 0:
                    per_neuron_2 = (correction * frac2) / max(1, a2.shape[-1])
                    layer_states[child2].activations = a2 + per_neuron_2
                    layer_states[child2].value = compute_value(
                        layer_states[child2].activations, self.config.value_fn)

            iterations += 1
            status = self.check_conservation(layer_states)
            status.corrections_applied = iterations

        self.enforcement_history.append(status)
        return status

    def verify_fibonacci_ratios(self, layer_states: List[LayerState],
                                tolerance: float = 0.1) -> Dict[str, float]:
        """
        Verify that layer value ratios follow the Fibonacci/golden ratio pattern.

        Expected: V(k)/V(k+1) ≈ φ for all k

        Parameters:
            layer_states: States from a forward pass
            tolerance: Relative tolerance for ratio check

        Returns:
            Dict with ratio analysis
        """
        values = [s.value for s in layer_states]
        ratios = []
        deviations = []

        for k in range(len(values) - 1):
            if values[k + 1] > 1e-10:
                ratio = values[k] / values[k + 1]
                ratios.append(ratio)
                deviations.append(abs(ratio - PHI) / PHI)
            else:
                ratios.append(float('inf'))
                deviations.append(float('inf'))

        mean_deviation = np.mean([d for d in deviations if d != float('inf')])
        all_close = all(d < tolerance for d in deviations)

        return {
            'ratios': ratios,
            'deviations': deviations,
            'mean_deviation': mean_deviation,
            'fibonacci_consistent': all_close,
            'expected_ratio': PHI,
        }

    def conservation_report(self, layer_states: List[LayerState]) -> str:
        """Generate a human-readable conservation report."""
        status = self.check_conservation(layer_states)
        ratio_info = self.verify_fibonacci_ratios(layer_states)

        lines = [
            "=== PAC Conservation Report ===",
            f"Conservation satisfied: {'YES' if status.satisfied else 'NO'}",
            f"Max violation: {status.max_violation:.2e} (ε = {self.config.epsilon:.2e})",
            f"Total violation: {status.total_violation:.2e}",
            "",
            "Layer values:",
        ]
        for k, v in enumerate(status.layer_values):
            target = status.layer_values[0] * PHI_INV ** k
            lines.append(f"  V({k}) = {v:.6f}  (target: {target:.6f})")

        lines.append("")
        lines.append("Fibonacci ratios (expected φ ≈ 1.6180):")
        for k, (r, d) in enumerate(zip(ratio_info['ratios'],
                                        ratio_info['deviations'])):
            mark = '✓' if d < 0.1 else '✗'
            lines.append(f"  V({k})/V({k+1}) = {r:.4f}  (dev: {d:.4f}) {mark}")

        return '\n'.join(lines)
