"""
Measurement instruments for TinyCIMM-Genesis.

Detect emergent constants (phi, Xi, gamma/ln(phi)) from dynamics.
These functions COMPARE measured values against DFT predictions
but never INSERT the predictions into the dynamics.
"""

import numpy as np
from typing import Dict, List, Tuple

from spectral_utils import (
    PHI, PHI_INV, LN_PHI, GAMMA_EM, XI, SCOPE_RATIO,
    eigenvalue_ratios, phi_enrichment, cascade_depth,
    hierarchy_entropy,
)


def measure_modulation_convergence(diagnostics: List[Dict], N: int,
                                    window: int = 500) -> Dict:
    """
    Measure where weak_factor converged and compare to phi^(-1/N).

    Args:
        diagnostics: list of snapshot dicts from GenesisSystem.run()
        N: system size
        window: averaging window for final value

    Returns dict with measured, predicted, error.
    """
    if len(diagnostics) < window:
        window = len(diagnostics)

    weak_values = [d['weak_factor'] for d in diagnostics[-window:]]
    measured = float(np.mean(weak_values))
    std = float(np.std(weak_values))
    predicted = PHI ** (-1.0 / N)
    error_pct = abs(measured - predicted) / predicted * 100

    return {
        'measured_weak': measured,
        'measured_std': std,
        'predicted_weak': predicted,
        'error_pct': error_pct,
        'N': N,
        'converged': std < 0.01,  # Low variance = converged
    }


def measure_spectral_radius_convergence(diagnostics: List[Dict],
                                          window: int = 500) -> Dict:
    """
    Measure where spectral radius stabilized and compare to gamma/ln(phi).
    """
    if len(diagnostics) < window:
        window = len(diagnostics)

    sr_values = [d['spectral_radius'] for d in diagnostics[-window:]]
    measured = float(np.mean(sr_values))
    std = float(np.std(sr_values))
    error_pct = abs(measured - SCOPE_RATIO) / SCOPE_RATIO * 100

    return {
        'measured_sr': measured,
        'measured_std': std,
        'predicted_sr': SCOPE_RATIO,
        'error_pct': error_pct,
        'converged': std < 0.05,
    }


def measure_phi_in_eigenvalues(W: np.ndarray) -> Dict:
    """
    Check eigenvalue ratios for phi enrichment.

    Returns dict with ratios, enrichment fraction, and breakdown.
    """
    ratios = eigenvalue_ratios(W)
    enrichment = phi_enrichment(ratios)

    # Also check 1/phi enrichment (inverse ratios)
    inv_enrichment = phi_enrichment(1.0 / ratios[ratios > 1e-10]) if len(ratios) > 0 else 0.0

    return {
        'n_ratios': len(ratios),
        'phi_enrichment': enrichment,
        'phi_inv_enrichment': inv_enrichment,
        'combined_enrichment': enrichment + inv_enrichment,
        'ratios': ratios.tolist() if len(ratios) < 50 else ratios[:50].tolist(),
        'mean_ratio': float(np.mean(ratios)) if len(ratios) > 0 else 0.0,
    }


def measure_cascade_transition(W_func, N: int, weak_range=(0.80, 0.99),
                                 n_points: int = 40, n_steps: int = 500,
                                 n_seeds: int = 20) -> Dict:
    """
    Scan weak_factor and measure cascade depth at each value.
    Detect first-order transition (discontinuous jump in active modes).

    Args:
        W_func: callable(N, seed, weak_factor) -> (final_W, final_state)
            runs a Genesis system with FIXED weak_factor for n_steps
        N: system size
        weak_range: scan range for weak_factor
        n_points: number of scan points
        n_seeds: seeds per point

    Returns dict with scan results and transition detection.
    """
    weak_values = np.linspace(weak_range[0], weak_range[1], n_points)
    mean_depths = []
    std_depths = []
    min_spacings = []

    for weak in weak_values:
        depths = []
        spacings = []
        for seed in range(n_seeds):
            W_final, _ = W_func(N, seed, float(weak))
            d = cascade_depth(W_final)
            depths.append(d)
            # Minimum eigenvalue spacing
            eigvals = np.sort(np.abs(np.linalg.eigvalsh(W_final)))[::-1]
            if len(eigvals) > 1:
                diffs = np.diff(eigvals)
                spacings.append(float(np.min(np.abs(diffs[diffs != 0])))
                                if np.any(diffs != 0) else 0.0)
        mean_depths.append(float(np.mean(depths)))
        std_depths.append(float(np.std(depths)))
        min_spacings.append(float(np.mean(spacings)) if spacings else 0.0)

    # Detect first-order transition: largest gap in mean_depths
    depth_diffs = np.diff(mean_depths)
    max_jump_idx = int(np.argmax(np.abs(depth_diffs)))
    max_jump = float(np.abs(depth_diffs[max_jump_idx]))
    transition_weak = float((weak_values[max_jump_idx] + weak_values[max_jump_idx + 1]) / 2)

    return {
        'weak_values': weak_values.tolist(),
        'mean_depths': mean_depths,
        'std_depths': std_depths,
        'min_spacings': min_spacings,
        'max_jump': max_jump,
        'transition_weak': transition_weak,
        'is_first_order': max_jump > 3.0,
        'N': N,
    }


def measure_xi_from_dynamics(trajectory: np.ndarray, W_history: List[np.ndarray]) -> Dict:
    """
    Measure per-mode-switch information cost and compare to Xi.

    A "mode switch" occurs when the dominant eigenvalue index changes.
    The information cost is the entropy change across that switch.

    Args:
        trajectory: [n_steps, N] state trajectory
        W_history: list of W matrices at each step

    Returns dict with measured Xi and comparison.
    """
    n_steps = len(trajectory)
    if n_steps < 100 or len(W_history) < n_steps:
        return {'measured_xi': 0.0, 'n_switches': 0, 'error_pct': 999.0}

    # Track dominant mode index
    dominant_indices = []
    entropies = []
    for t in range(n_steps):
        eigvals = np.linalg.eigvalsh(W_history[t])
        projections = np.abs(eigvals)
        dominant_indices.append(int(np.argmax(projections)))
        entropies.append(hierarchy_entropy(eigvals))

    # Find mode switches
    switches = []
    for t in range(1, n_steps):
        if dominant_indices[t] != dominant_indices[t - 1]:
            # Entropy change at switch
            delta_h = abs(entropies[t] - entropies[t - 1])
            switches.append(delta_h)

    if len(switches) < 5:
        return {
            'measured_xi': 0.0,
            'n_switches': len(switches),
            'error_pct': 999.0,
            'note': 'too few switches for reliable measurement',
        }

    measured_xi = float(np.mean(switches))
    error_pct = abs(measured_xi - XI) / XI * 100

    return {
        'measured_xi': measured_xi,
        'xi_std': float(np.std(switches)),
        'n_switches': len(switches),
        'predicted_xi': XI,
        'error_pct': error_pct,
    }
