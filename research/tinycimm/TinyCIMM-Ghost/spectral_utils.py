"""
Spectral utilities for TinyCIMM-Genesis.

Eigenvalue analysis, symmetric matrix operations, hierarchy measurement,
and DFT constants. Self-contained — no cross-variant dependencies.
"""

import numpy as np
from math import sqrt, log

# DFT constants (not inserted into dynamics — only used for measurement/comparison)
PHI = (1 + sqrt(5)) / 2                    # 1.6180339887...
PHI_INV = 1 / PHI                          # 0.6180339887...
LN_PHI = log(PHI)                          # 0.4812118250...
GAMMA_EM = 0.5772156649015329               # Euler-Mascheroni
XI = GAMMA_EM + LN_PHI                     # 1.0584274900...
SCOPE_RATIO = GAMMA_EM / LN_PHI            # 1.1995043240...


def symmetric_eigendecomposition(W):
    """Eigendecompose symmetric matrix. Returns (eigenvalues, eigenvectors)."""
    eigvals, eigvecs = np.linalg.eigh(W)
    return eigvals, eigvecs


def spectral_radius(W):
    """Max absolute eigenvalue of symmetric W."""
    eigvals = np.linalg.eigvalsh(W)
    return float(np.max(np.abs(eigvals)))


def anti_hebbian_modulate(eigvals, eigvecs, state, weak_factor, strong_factor,
                          target_sr=None):
    """
    Anti-Hebbian eigenvalue modulation.

    Active modes (projection > 2× mean) weakened by weak_factor.
    Inactive modes (projection < 0.5× mean) strengthened by strong_factor.
    Optionally re-normalizes spectral radius to target_sr.

    Returns new W = eigvecs @ diag(new_eigvals) @ eigvecs.T
    """
    N = len(eigvals)
    projections = (eigvecs.T @ state) ** 2
    total = np.sum(projections) + 1e-10
    activities = projections / total
    mean_act = 1.0 / N

    modulation = np.ones(N)
    modulation[activities > 2.0 * mean_act] = weak_factor
    modulation[activities < 0.5 * mean_act] = strong_factor
    new_eigvals = eigvals * modulation

    if target_sr is not None:
        sr = np.max(np.abs(new_eigvals))
        if sr > 1e-10:
            new_eigvals = new_eigvals * (target_sr / sr)

    W_new = eigvecs @ np.diag(new_eigvals) @ eigvecs.T
    return W_new, new_eigvals


def eigenvalue_ratios(W):
    """
    Consecutive ratios of sorted (descending) absolute eigenvalues.
    Returns array of |λ_k| / |λ_{k+1}| for k = 0..N-2.
    """
    eigvals = np.abs(np.linalg.eigvalsh(W))
    eigvals = np.sort(eigvals)[::-1]  # Descending
    # Avoid division by zero
    ratios = []
    for k in range(len(eigvals) - 1):
        if eigvals[k + 1] > 1e-12:
            ratios.append(eigvals[k] / eigvals[k + 1])
    return np.array(ratios)


def phi_enrichment(ratios, tol=0.15):
    """Fraction of eigenvalue ratios within tol of phi."""
    if len(ratios) == 0:
        return 0.0
    near_phi = np.sum(np.abs(ratios - PHI) / PHI < tol)
    return float(near_phi / len(ratios))


def cascade_depth(W, threshold=0.01):
    """Number of active eigenvalue modes (|λ| > threshold × max|λ|)."""
    eigvals = np.abs(np.linalg.eigvalsh(W))
    max_val = np.max(eigvals)
    if max_val < 1e-15:
        return 0
    return int(np.sum(eigvals > threshold * max_val))


def hierarchy_entropy(eigenvalues):
    """Shannon entropy of normalized absolute eigenvalue spectrum."""
    absvals = np.abs(eigenvalues)
    total = np.sum(absvals)
    if total < 1e-15:
        return 0.0
    p = absvals / total
    p = p[p > 1e-15]  # Remove zeros
    return float(-np.sum(p * np.log(p)))


def measure_hierarchical_structure(trajectory):
    """
    Analyze trajectory for multi-scale hierarchical structure.
    Adapted from M10 foundations.py.

    Returns dict with:
        has_hierarchy: bool
        n_active_scales: int
        scale_persistence: float [0,1]
        mean_complexity: float (participation ratio)
    """
    traj = np.array(trajectory)
    n_steps, N = traj.shape

    if n_steps < 10:
        return {
            'has_hierarchy': False, 'n_active_scales': 0,
            'scale_persistence': 0.0, 'mean_complexity': 0.0,
        }

    # SVD of trajectory covariance
    centered = traj - traj.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return {
            'has_hierarchy': False, 'n_active_scales': 0,
            'scale_persistence': 0.0, 'mean_complexity': 0.0,
        }

    S_norm = S / (S[0] + 1e-15) if S[0] > 1e-15 else S
    n_active = int(np.sum(S_norm > 0.05))

    # Participation ratio (effective dimension)
    S2 = S ** 2
    pr = (np.sum(S2)) ** 2 / (np.sum(S2 ** 2) + 1e-15) if np.sum(S2) > 0 else 0

    # Windowed persistence
    window = max(10, n_steps // 10)
    n_windows = max(1, n_steps // window)
    persistent = 0
    for w in range(n_windows):
        start = w * window
        end = min(start + window, n_steps)
        chunk = centered[start:end]
        if len(chunk) < 3:
            continue
        try:
            _, S_w, _ = np.linalg.svd(chunk, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        S_w_norm = S_w / (S_w[0] + 1e-15) if S_w[0] > 1e-15 else S_w
        if np.sum(S_w_norm > 0.05) >= 3:
            persistent += 1
    persistence = persistent / n_windows if n_windows > 0 else 0

    # Non-stationarity check
    first_half = np.std(traj[:n_steps // 2], axis=0)
    second_half = np.std(traj[n_steps // 2:], axis=0)
    variation = np.mean(np.abs(first_half - second_half))
    non_stationary = variation > 0.01

    has_hierarchy = (n_active >= 3) and (persistence > 0.5) and non_stationary

    return {
        'has_hierarchy': has_hierarchy,
        'n_active_scales': n_active,
        'scale_persistence': float(persistence),
        'mean_complexity': float(pr),
    }
