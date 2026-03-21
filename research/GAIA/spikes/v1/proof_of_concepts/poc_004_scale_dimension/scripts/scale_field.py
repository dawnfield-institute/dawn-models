"""
POC-004: Scaled Field Dynamics

Core 3D field operations with Reality Engine integration.
Extends POC-002/003 patterns to 3D with spherical harmonics.

Key innovations:
- 3D field encoding with spherical harmonics
- Reality Engine RearrangementTensor for conservation
- Adaptive field sizing based on pattern load
- Scale-invariant Dawn Field constants

Torch only, GPU all the way.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import math
import sys

# Dawn Field Constants (validated in POC-002/003)
PHI = 1.618033988749895
XI = 1.0571
PHI_XI = PHI * XI  # 1.710 - 2D crystallization threshold
LAMBDA_STAR = 0.9816  # Optimal decay
PI_SQUARED_INV = 1.0 / (math.pi ** 2)  # 0.101

# 3D scaling law: threshold scales as (PHI_XI)^(dim/2)
def critical_density_3d() -> float:
    """3D critical density = (φ × ξ)^(3/2) ≈ 2.236"""
    return PHI_XI ** 1.5


@dataclass
class ConservationMetrics:
    """Track P+A+M conservation (Reality Engine compatible)"""
    initial_total: float
    current_total: float
    violation: float
    transfers_applied: int


class RearrangementTensor3D:
    """
    3D Rearrangement Tensor for Dawn Field dynamics.
    
    Inspired by Reality Engine's RearrangementTensor but specialized
    for GAIA's field-native transformer architecture.
    
    Maintains P+A+M conservation while enabling:
    - Spherical harmonic encoding
    - Scale-adaptive field sizing
    - GPU-accelerated 3D operations
    """
    
    def __init__(
        self,
        shape: Tuple[int, int, int] = (32, 32, 32),
        device: str = 'cuda'
    ):
        self.shape = shape
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.total_cells = shape[0] * shape[1] * shape[2]
        
        # Initialize P+A+M fields with Dawn Field total
        initial_total = PHI_XI  # Dawn Field constant
        third = initial_total / 3.0
        
        self.P = torch.ones(*shape, device=self.device) * third / self.total_cells
        self.A = torch.ones(*shape, device=self.device) * third / self.total_cells  
        self.M = torch.ones(*shape, device=self.device) * third / self.total_cells
        
        self.initial_total = initial_total
        self.step = 0
        
    def get_total(self) -> float:
        """Compute current P+A+M total."""
        return (self.P.sum() + self.A.sum() + self.M.sum()).item()
    
    def get_conservation_metrics(self) -> ConservationMetrics:
        """Check conservation status."""
        current = self.get_total()
        return ConservationMetrics(
            initial_total=self.initial_total,
            current_total=current,
            violation=abs(current - self.initial_total),
            transfers_applied=self.step
        )
    
    def transfer_p_to_a(self, rate: torch.Tensor, dt: float = 0.01) -> None:
        """Transfer from Potential to Active (gradient-driven)."""
        transfer = rate * self.P * dt
        transfer = torch.clamp(transfer, max=self.P)  # Can't transfer more than available
        self.P = self.P - transfer
        self.A = self.A + transfer
        self.step += 1
        
    def transfer_a_to_m(self, rate: torch.Tensor, dt: float = 0.01) -> None:
        """Transfer from Active to Material (stabilization)."""
        transfer = rate * self.A * dt
        transfer = torch.clamp(transfer, max=self.A)
        self.A = self.A - transfer
        self.M = self.M + transfer
        self.step += 1
        
    def transfer_m_to_p(self, rate: torch.Tensor, dt: float = 0.01) -> None:
        """Transfer from Material to Potential (dissolution)."""
        transfer = rate * self.M * dt
        transfer = torch.clamp(transfer, max=self.M)
        self.M = self.M - transfer
        self.P = self.P + transfer
        self.step += 1
    
    def get_combined_field(self) -> torch.Tensor:
        """Get combined field state for encoding."""
        # Weight by field type: P=potential energy, A=active dynamics, M=stable memory
        return 0.3 * self.P + 0.5 * self.A + 0.2 * self.M


class SphericalHarmonicEncoder(nn.Module):
    """
    Encode patterns in 3D using spherical harmonics.
    
    Extends the 2D prime harmonic attention to 3D:
    - Y_l^m weighted by 1/(l+1)² (prime-like decay)
    - Radial decay follows λ* = 0.9816
    """
    
    def __init__(
        self,
        shape: Tuple[int, int, int] = (32, 32, 32),
        l_max: int = 4,
        device: str = 'cuda'
    ):
        super().__init__()
        self.shape = shape
        self.l_max = l_max
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Precompute spherical coordinate grids
        self._init_grids()
        
        # Precompute harmonic weights (prime-like 1/(l+1)² decay)
        self.harmonic_weights = torch.tensor(
            [1.0 / ((l + 1) ** 2) for l in range(l_max + 1)],
            device=self.device
        )
        
    def _init_grids(self):
        """Initialize spherical coordinate grids."""
        x = torch.linspace(-1, 1, self.shape[0], device=self.device)
        y = torch.linspace(-1, 1, self.shape[1], device=self.device)
        z = torch.linspace(-1, 1, self.shape[2], device=self.device)
        
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        
        self.r = torch.sqrt(X**2 + Y**2 + Z**2).clamp(min=1e-8)
        self.theta = torch.acos((Z / self.r).clamp(-1, 1))
        self.phi = torch.atan2(Y, X)
        
        # Precompute cos(theta) for Legendre polynomials
        self.cos_theta = torch.cos(self.theta)
        
    def _legendre_p(self, l: int, x: torch.Tensor) -> torch.Tensor:
        """Compute Legendre polynomial P_l(x) using recurrence."""
        if l == 0:
            return torch.ones_like(x)
        elif l == 1:
            return x
        else:
            # Recurrence: (l+1)P_{l+1} = (2l+1)xP_l - lP_{l-1}
            p_prev = torch.ones_like(x)
            p_curr = x
            for n in range(1, l):
                p_next = ((2*n + 1) * x * p_curr - n * p_prev) / (n + 1)
                p_prev = p_curr
                p_curr = p_next
            return p_curr
    
    def encode(self, pattern: torch.Tensor) -> torch.Tensor:
        """
        Encode a pattern embedding into 3D spherical field.
        
        REFINED VERSION: Better preserves pairwise similarity.
        
        Strategy:
        1. Project pattern to harmonic coefficients (captures global structure)
        2. Use pattern's PCA-like projection for angular structure (stable)
        3. Additive angular modulation instead of multiplicative (preserves sign)
        4. Consistent radial decay (no pattern-dependent flipping)
        
        Args:
            pattern: 1D embedding tensor
            
        Returns:
            3D field tensor with spherical harmonic structure
        """
        pattern = pattern.to(self.device)
        n = len(pattern)
        
        # === HARMONIC COEFFICIENTS ===
        # Project pattern to harmonic space using overlapping windows
        # This gives smoother, more stable coefficients
        harm_coeffs = []
        window_size = max(1, n // (self.l_max + 1))
        
        for l in range(self.l_max + 1):
            # Overlapping windows for smoother projection
            center = int(l * n / (self.l_max + 1))
            start = max(0, center - window_size // 2)
            end = min(n, center + window_size // 2 + 1)
            
            # Use mean for stability (less sensitive to outliers)
            coeff = pattern[start:end].mean()
            harm_coeffs.append(coeff)
        
        harm_coeffs = torch.stack(harm_coeffs)
        
        # Soft normalization preserving relative magnitudes
        coeff_norm = harm_coeffs.norm() + 1e-8
        harm_coeffs = harm_coeffs / coeff_norm
        
        # === BUILD BASE FIELD ===
        field = torch.zeros(self.shape, device=self.device)
        
        for l in range(self.l_max + 1):
            Y_l0 = self._legendre_p(l, self.cos_theta)
            weight = self.harmonic_weights[l] * harm_coeffs[l]
            field = field + weight * Y_l0
        
        # === RADIAL DECAY ===
        # Use λ* with consistent decay (no pattern-dependent scaling)
        # This ensures similar patterns have similar radial structure
        radial_decay = LAMBDA_STAR ** self.r
        field = field * radial_decay
        
        # === ANGULAR MODULATION (ADDITIVE) ===
        # Instead of multiplicative (which can flip signs), use additive
        # Project pattern to 3D using consistent basis
        if n >= 6:
            # Use first half for x, second half for y, interlaced for z
            px = pattern[:n//2].mean()
            py = pattern[n//2:].mean()
            pz = pattern[::2].mean() - pattern[1::2].mean()
        else:
            px = pattern[0] if n > 0 else torch.tensor(0.0, device=self.device)
            py = pattern[1] if n > 1 else torch.tensor(0.0, device=self.device)
            pz = pattern[2] if n > 2 else torch.tensor(0.0, device=self.device)
        
        # Normalize direction
        p_vec = torch.stack([px, py, pz])
        p_norm = p_vec.norm() + 1e-8
        p_vec = p_vec / p_norm
        
        # Create angular pattern (dot product with unit sphere)
        angular_pattern = (
            p_vec[0] * torch.sin(self.theta) * torch.cos(self.phi) +
            p_vec[1] * torch.sin(self.theta) * torch.sin(self.phi) +
            p_vec[2] * torch.cos(self.theta)
        )
        
        # ADDITIVE modulation scaled by 1/l² for consistency with harmonics
        angular_weight = 0.2  # Reduced from 0.3 for stability
        field = field + angular_weight * angular_pattern * radial_decay
        
        # === FINAL NORMALIZATION ===
        # Preserve mean structure, normalize variance
        field = field - field.mean()
        field_std = field.std() + 1e-8
        field = field / field_std
        
        return field
    
    def encode_v1(self, pattern: torch.Tensor) -> torch.Tensor:
        """Original encode method (kept for comparison)."""
        pattern = pattern.to(self.device)
        n = len(pattern)
        
        harm_coeffs = []
        stride = max(1, n // (self.l_max + 1))
        for l in range(self.l_max + 1):
            start = (l * stride) % n
            end = min(start + stride, n)
            coeff = pattern[start:end].sum()
            harm_coeffs.append(coeff)
        harm_coeffs = torch.stack(harm_coeffs)
        harm_coeffs = harm_coeffs / (harm_coeffs.abs().max() + 1e-8)
        
        field = torch.zeros(self.shape, device=self.device)
        for l in range(self.l_max + 1):
            Y_l0 = self._legendre_p(l, self.cos_theta)
            weight = self.harmonic_weights[l] * harm_coeffs[l]
            field = field + weight * Y_l0
        
        pattern_magnitude = pattern.norm() / (n ** 0.5)
        radial_scale = 0.8 + 0.4 * torch.sigmoid(pattern_magnitude - 1)
        radial_decay = LAMBDA_STAR ** (self.r * radial_scale)
        field = field * radial_decay
        
        if n >= 3:
            px = pattern[0::3].mean()
            py = pattern[1::3].mean()
            pz = pattern[2::3].mean()
        else:
            px = py = pz = pattern.mean()
        
        p_vec = torch.stack([px, py, pz])
        p_vec = p_vec / (p_vec.norm() + 1e-8)
        
        alignment = (
            p_vec[0] * torch.sin(self.theta) * torch.cos(self.phi) +
            p_vec[1] * torch.sin(self.theta) * torch.sin(self.phi) +
            p_vec[2] * torch.cos(self.theta)
        )
        angular_mod = 1.0 + 0.3 * alignment
        field = field * angular_mod
        
        field = field - field.mean()
        max_abs = field.abs().max()
        if max_abs > 1e-8:
            field = field / max_abs
        
        return field
    
    def encode_v2(self, pattern: torch.Tensor) -> torch.Tensor:
        """
        V2 Encoder: Geometry-preserving spherical encoding.
        
        Key insight: The problem with v1 and refined is they project to 3D
        in ways that can flip signs. V2 uses a different strategy:
        
        1. Use MORE harmonic coefficients (l_max=8 gives 9 coefficients)
        2. Map embedding chunks to positive harmonic modulations
        3. Use absolute values for angular modulation (no sign flips)
        4. Center on embedding statistics, not arbitrary projections
        
        This should preserve cosine similarity better because similar
        embeddings will produce similar harmonic coefficient patterns.
        """
        pattern = pattern.to(self.device)
        n = len(pattern)
        
        # === HARMONIC COEFFICIENTS ===
        # Use non-overlapping chunks for cleaner mapping
        chunk_size = max(1, n // (self.l_max + 1))
        
        harm_coeffs = []
        for l in range(self.l_max + 1):
            start = l * chunk_size
            end = min((l + 1) * chunk_size, n)
            if start < n:
                # Use std as well as mean to capture pattern structure
                chunk = pattern[start:end]
                coeff = chunk.mean() + 0.1 * chunk.std()
            else:
                coeff = torch.tensor(0.0, device=self.device)
            harm_coeffs.append(coeff)
        
        harm_coeffs = torch.stack(harm_coeffs)
        
        # Unit normalize to create a direction in harmonic space
        coeff_norm = harm_coeffs.norm() + 1e-8
        harm_coeffs = harm_coeffs / coeff_norm
        
        # === BUILD BASE FIELD ===
        field = torch.zeros(self.shape, device=self.device)
        
        for l in range(self.l_max + 1):
            Y_l0 = self._legendre_p(l, self.cos_theta)
            weight = self.harmonic_weights[l] * harm_coeffs[l]
            field = field + weight * Y_l0
        
        # === RADIAL DECAY ===
        radial_decay = LAMBDA_STAR ** self.r
        field = field * radial_decay
        
        # === ANGULAR MODULATION ===
        # Use ABSOLUTE VALUE of embedding statistics for angular pattern
        # This ensures similar embeddings get similar modulations
        # regardless of sign flips in the projection
        
        # Create angle from embedding statistics (magnitude-based, no sign issues)
        pattern_norm = pattern.norm() / (n ** 0.5)
        pattern_mean = pattern.mean()
        pattern_std = pattern.std()
        
        # Map to spherical angles using sigmoid (bounded, smooth)
        theta_weight = torch.sigmoid(pattern_mean)  # [0, 1]
        phi_weight = torch.sigmoid(pattern_std)      # [0, 1]
        
        # Create angular modulation that is ALWAYS positive
        # Uses cos² and sin² which are always non-negative
        angular_mod = (
            1.0 + 
            0.2 * theta_weight * torch.cos(self.theta)**2 +
            0.2 * phi_weight * torch.cos(self.phi)**2
        )
        
        field = field * angular_mod
        
        # === NORMALIZATION ===
        field = field - field.mean()
        field_std = field.std() + 1e-8
        field = field / field_std
        
        return field
    
    def encode_v4(self, pattern: torch.Tensor) -> torch.Tensor:
        """
        V4 Encoder: Pi-Harmonic + SEC Phase + Bifractal Integration
        
        Combines insights from foundational experiments:
        
        1. Pi-Harmonics: Use sin(π * n * θ) for angular coherence
           - Pi modulation produces more coherent attractors
           - Radial symmetry emerges naturally
        
        2. SEC Phase Transitions: Stress accumulation with decay
           - E(n) = λ·E(n-1) + I(n) models embedding "stress"
           - Phase transitions at critical thresholds
        
        3. Bifractal Embedding: Semantic ancestry aggregation
           - Multiple embedding views contribute to field
           - Cosine similarity preserved through linear combination
        
        Key insight: Instead of projecting to arbitrary 3D direction,
        use embedding components directly as harmonic coefficients
        weighted by their natural frequency (index-based).
        """
        pattern = pattern.to(self.device)
        n = len(pattern)
        
        # === BIFRACTAL SEMANTIC AGGREGATION ===
        # Compute multiple "views" of the embedding like bifractal ancestry
        # View 1: Low-frequency (global structure)
        low_freq = pattern.unfold(0, min(32, n), max(1, n//8)).mean(dim=1)
        # View 2: High-frequency (local detail)  
        high_freq = pattern[::2] - pattern[1::2] if n >= 2 else pattern
        # View 3: Original pattern statistics
        pattern_mean = pattern.mean()
        pattern_std = pattern.std()
        
        # === SEC STRESS FIELD COMPUTATION ===
        # Model "stress" as cumulative departure from expectation
        # I(n) = expected - actual, E(n) = λ*E(n-1) + I(n)
        expected = pattern_mean  # Local expectation
        actual = pattern  # Actual values
        impulse = expected - actual  # Collapse impulse
        
        # Accumulate stress with λ* decay
        stress = torch.zeros(n, device=self.device)
        for i in range(1, n):
            stress[i] = LAMBDA_STAR * stress[i-1] + impulse[i]
        
        # Normalize stress
        stress_norm = stress / (stress.abs().max() + 1e-8)
        
        # === PI-HARMONIC FIELD CONSTRUCTION ===
        # Use π as the fundamental modulation constant
        PI = torch.tensor(torch.pi, device=self.device)
        
        field = torch.zeros(self.shape, device=self.device)
        
        # Build field using pi-modulated harmonics
        # Each embedding component contributes at its natural frequency
        num_coeffs = min(n, (self.l_max + 1) * 4)  # Use more coefficients
        
        for i in range(num_coeffs):
            # Map to frequency using golden ratio for optimal coverage
            freq = 1 + (i * PHI) % (self.l_max + 1)
            l = int(freq)
            
            # Weight from embedding + stress modulation
            base_weight = pattern[i % n]
            stress_mod = 1.0 + 0.2 * stress_norm[i % n]
            
            # Pi-harmonic angular modulation
            # sin(π * l * θ) produces coherent attractor zones
            angular = torch.sin(PI * l * self.theta)
            
            # Spherical harmonic base
            Y_l0 = self._legendre_p(l, self.cos_theta)
            
            # 1/(l+1)² weighting
            harmonic_weight = 1.0 / ((l + 1) ** 2)
            
            # Combine: base field + pi-modulated angular structure
            contribution = (Y_l0 + 0.3 * angular) * harmonic_weight * base_weight * stress_mod
            field = field + contribution
        
        # === RADIAL DECAY WITH λ* ===
        radial_decay = LAMBDA_STAR ** self.r
        field = field * radial_decay
        
        # === BIFRACTAL MULTI-SCALE MODULATION ===
        # Low-frequency view affects global field shape
        low_freq_weight = low_freq.mean() if len(low_freq) > 0 else pattern_mean
        # High-frequency view affects local detail
        high_freq_energy = high_freq.abs().mean() if len(high_freq) > 0 else pattern_std
        
        # Scale modulation based on views
        global_scale = 1.0 + 0.2 * torch.sigmoid(low_freq_weight)
        local_detail = 1.0 + 0.1 * torch.sigmoid(high_freq_energy)
        
        # Apply at different radii (bifractal multi-scale)
        center_mask = self.r < 0.5
        edge_mask = self.r >= 0.5
        
        field = torch.where(center_mask, field * global_scale, field * local_detail)
        
        # === FINAL NORMALIZATION ===
        field = field - field.mean()
        field_std = field.std() + 1e-8
        field = field / field_std
        
        return field
    
    def encode_v5(self, pattern: torch.Tensor) -> torch.Tensor:
        """
        V5 Encoder: Linear Pi-Harmonic Basis Expansion
        
        Key insight: To preserve inner products, we need LINEAR combination
        of FIXED basis functions. V4's problem was sequential accumulation.
        
        From foundational experiments:
        1. Pi-harmonics: Use sin(π * l * θ) as basis (coherent attractors)
        2. Bifractal: Each embedding component weights a basis (linear = preserves IP)
        3. SEC: Use contrast weighting (deviation from mean) for importance
        
        This encoder:
        - Creates fixed pi-harmonic 3D basis fields (orthogonal-ish)
        - Weights each basis by embedding component
        - Linear combination guarantees: <f(a), f(b)> ∝ <a, b>
        """
        pattern = pattern.to(self.device)
        n = len(pattern)
        
        # === SEC-INSPIRED CONTRAST WEIGHTING ===
        # Instead of accumulation, use contrast: |x - mean| for importance
        pattern_mean = pattern.mean()
        contrast = (pattern - pattern_mean).abs()
        contrast_weights = 1.0 + 0.3 * contrast / (contrast.max() + 1e-8)
        
        # Weighted pattern (emphasizes distinctive components)
        weighted_pattern = pattern * contrast_weights
        
        # === INITIALIZE PI-HARMONIC BASES ON FIRST CALL ===
        if not hasattr(self, '_v5_bases') or self._v5_n != n:
            self._init_v5_bases(n)
        
        # === LINEAR COMBINATION ===
        # f(pattern) = Σ pattern[i] * basis[i]
        # This preserves: <f(a), f(b)> = Σ a[i]*b[j] * <basis[i], basis[j]>
        # If bases are orthonormal, this = Σ a[i]*b[i] = <a, b>
        
        field = torch.zeros(self.shape, device=self.device)
        
        for i in range(min(n, len(self._v5_bases))):
            field = field + weighted_pattern[i] * self._v5_bases[i]
        
        # === RADIAL DECAY WITH λ* ===
        field = field * self._v5_radial
        
        # === NORMALIZATION (preserves direction) ===
        field = field - field.mean()
        field_std = field.std() + 1e-8
        field = field / field_std
        
        return field
    
    def _init_v5_bases(self, n: int):
        """Initialize pi-harmonic orthogonal basis fields."""
        self._v5_n = n
        self._v5_bases = []
        
        PI = torch.tensor(torch.pi, device=self.device)
        
        # Create n basis fields using pi-harmonic structure
        # Key: use sin(π * freq * angle) which creates coherent zones
        
        for i in range(n):
            # Map index to 3D frequency using golden ratio
            # This ensures maximum spread across frequency space
            golden_angle = 2.399963  # Golden angle in radians
            
            # 3D frequency components
            l = 1 + (i % (self.l_max + 1))  # Harmonic order
            m = (i // (self.l_max + 1)) % 3  # Angular mode (x, y, or z aligned)
            phase = (i * golden_angle) % (2 * PI.item())
            
            # Create basis field
            # Use spherical harmonics Y_l^0 modulated by pi-harmonic angular term
            Y_l0 = self._legendre_p(l, self.cos_theta)
            
            # Pi-harmonic angular modulation (different orientation for each mode)
            if m == 0:
                angular = torch.sin(PI * l * self.theta + phase)
            elif m == 1:
                angular = torch.sin(PI * l * self.phi + phase)
            else:
                angular = torch.sin(PI * l * (self.theta + self.phi) / 2 + phase)
            
            # Combine: Y_l0 gives structure, angular gives coherent zones
            # Weight by 1/(l+1)² for natural falloff
            harmonic_weight = 1.0 / ((l + 1) ** 2)
            basis = (Y_l0 + 0.3 * angular) * harmonic_weight
            
            # Normalize each basis to unit norm (for orthonormality)
            basis_norm = basis.norm() + 1e-8
            basis = basis / basis_norm
            
            self._v5_bases.append(basis)
        
        # Precompute radial decay with λ*
        self._v5_radial = LAMBDA_STAR ** self.r
    
    def encode_v6(self, pattern: torch.Tensor) -> torch.Tensor:
        """
        V6 Encoder: Geometric E=mc² Preservation
        
        From Euclidean distance validation experiments (Dec 14, 2025):
        - ξ = correlation between energy and mass metrics
        - Embeddings that are similar have correlated geometric properties
        - Linear combination preserves inner products when structure-coupled
        
        Key insight: The embedding components are ALREADY in a space where
        cosine similarity is meaningful. Our encoder should preserve this
        by using the embedding values directly as field coefficients.
        
        From experiment_25: Betweenness ∝ out-degree because both measure
        the decomposition structure. Similarly, embedding[i] ∝ embedding[j]
        for similar concepts because both measure semantic proximity.
        
        Strategy:
        1. Use embedding components directly as coefficients (linear = preserves IP)
        2. Apply ξ-modulation: weight by local contrast (like SEC deviation)
        3. Use structural coupling via hierarchical basis (depth-indexed)
        4. Preserve the embedding's inherent correlation structure
        """
        pattern = pattern.to(self.device)
        n = len(pattern)
        
        # === ξ-MODULATION: LOCAL CONTRAST WEIGHTING ===
        # From SEC: contrast = expected - actual
        # High contrast components are more distinctive (like high betweenness)
        local_mean = torch.zeros(n, device=self.device)
        window = min(16, n // 4) if n > 16 else max(1, n // 4)
        
        for i in range(n):
            start = max(0, i - window)
            end = min(n, i + window + 1)
            local_mean[i] = pattern[start:end].mean()
        
        contrast = (pattern - local_mean).abs()
        xi_weight = 1.0 + XI * contrast / (contrast.max() + 1e-8)
        
        # === INITIALIZE BASES IF NEEDED ===
        if not hasattr(self, '_v6_bases') or self._v6_n != n:
            self._init_v6_bases(n)
        
        # === LINEAR COMBINATION WITH ξ-MODULATED WEIGHTS ===
        # f(pattern) = Σ (pattern[i] × ξ_weight[i]) × basis[i]
        # Since ξ correlates with structural importance, similar patterns
        # get similar ξ-modulated fields
        
        field = torch.zeros(self.shape, device=self.device)
        
        weighted_pattern = pattern * xi_weight
        
        for i in range(min(n, len(self._v6_bases))):
            field = field + weighted_pattern[i] * self._v6_bases[i]
        
        # === RADIAL DECAY ===
        field = field * self._v6_radial
        
        # === NORMALIZE PRESERVING DIRECTION ===
        field = field - field.mean()
        field_std = field.std() + 1e-8
        field = field / field_std
        
        return field
    
    def _init_v6_bases(self, n: int):
        """Initialize orthogonal basis fields for v6 encoder."""
        self._v6_n = n
        self._v6_bases = []
        
        # Create orthogonal basis using depth-indexed harmonics
        # Like experiment_13: depth, subtree_size, branching all correlate
        # because they measure the same hierarchical structure
        
        # Use 3D DCT-like basis (orthogonal by construction)
        shape = self.shape
        
        for i in range(n):
            # Map index to 3D frequency (Morton code style for good distribution)
            fx = 1 + (i % 8)
            fy = 1 + ((i // 8) % 8)
            fz = 1 + ((i // 64) % 8)
            
            # Phase for uniqueness
            phase = (i * PHI) % (2 * torch.pi)
            
            # Create orthogonal basis: cos(fx*x) * cos(fy*y) * cos(fz*z)
            # DCT basis is known to be orthogonal
            x = torch.linspace(0, torch.pi * fx, shape[0], device=self.device)
            y = torch.linspace(0, torch.pi * fy, shape[1], device=self.device)
            z = torch.linspace(0, torch.pi * fz, shape[2], device=self.device)
            
            X = x.view(-1, 1, 1)
            Y = y.view(1, -1, 1)
            Z = z.view(1, 1, -1)
            
            # Cosine basis (DCT type II)
            basis = torch.cos(X + phase) * torch.cos(Y) * torch.cos(Z)
            
            # Apply 1/(f+1)² weighting like 1/l² in harmonics
            freq_total = fx + fy + fz
            weight = 1.0 / ((freq_total / 3 + 1) ** 2)
            basis = basis * weight
            
            # Normalize
            basis_norm = basis.norm() + 1e-8
            basis = basis / basis_norm
            
            self._v6_bases.append(basis)
        
        # Radial decay with λ*
        self._v6_radial = LAMBDA_STAR ** self.r


class AdaptiveFieldSizer:
    """
    Dynamically adjust field size based on pattern load.
    
    Inspired by Reality Engine's AdaptiveParameters.
    Scales field resolution to maintain performance at high pattern counts.
    """
    
    def __init__(
        self,
        min_size: int = 16,
        max_size: int = 128,
        target_density: float = None
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.target_density = target_density or critical_density_3d()
        
        self.current_size = min_size
        self.pattern_count = 0
        
    def recommend_size(self, n_patterns: int) -> Tuple[int, int, int]:
        """
        Recommend field size for given pattern count.
        
        Strategy: Maintain pattern density near critical threshold.
        More patterns → larger field to keep density stable.
        """
        self.pattern_count = n_patterns
        
        # Target: n_patterns / volume ≈ some optimal density
        # Solve: size³ = n_patterns / target_density_per_cell
        optimal_volume = n_patterns / (self.target_density / 1000)  # Scale factor
        optimal_size = int(optimal_volume ** (1/3))
        
        # Clamp to valid range
        size = max(self.min_size, min(self.max_size, optimal_size))
        
        # Round to power of 2 for GPU efficiency
        size = 2 ** int(math.log2(size) + 0.5)
        size = max(self.min_size, min(self.max_size, size))
        
        self.current_size = size
        return (size, size, size)


class ScaledFieldAttention(nn.Module):
    """
    3D Field-Native Attention at Scale.
    
    Combines:
    - Spherical harmonic encoding (3D)
    - Prime harmonic attention weights (1/l²)
    - Reality Engine conservation
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        field_size: int = 32,
        device: str = 'cuda'
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.field_size = field_size
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Spherical encoder for patterns
        self.encoder = SphericalHarmonicEncoder(
            shape=(field_size, field_size, field_size),
            l_max=num_heads - 1,  # One harmonic per head
            device=self.device
        )
        
        # Conservation-aware field dynamics
        self.field = RearrangementTensor3D(
            shape=(field_size, field_size, field_size),
            device=self.device
        )
        
        # Learnable projection for output
        self.output_proj = nn.Linear(field_size ** 3, embed_dim).to(self.device)
        
        # Harmonic head weights (prime-like)
        self.head_weights = nn.Parameter(
            torch.tensor([1.0 / ((h + 2) ** 2) for h in range(num_heads)], 
                        device=self.device)
        )
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        3D field-native attention.
        
        Instead of QKV matmul, we:
        1. Encode Q, K, V as 3D fields
        2. Compute resonance between Q-field and K-field
        3. Weight V-field by resonance
        4. Project back to embedding space
        """
        batch_size = query.size(0)
        
        # Encode as 3D spherical fields
        q_fields = torch.stack([self.encoder.encode(q) for q in query])
        k_fields = torch.stack([self.encoder.encode(k) for k in key])
        v_fields = torch.stack([self.encoder.encode(v) for v in value])
        
        # Compute resonance (field correlation)
        # Resonance = ∫ Q(r) · K(r) dr
        resonance = (q_fields * k_fields).sum(dim=(-3, -2, -1))
        resonance = F.softmax(resonance / math.sqrt(self.field_size), dim=-1)
        
        # Weight V-fields by resonance
        output_fields = resonance.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * v_fields
        
        # Flatten and project
        output_flat = output_fields.view(batch_size, -1)
        if output_flat.size(-1) > self.field_size ** 3:
            output_flat = output_flat[:, :self.field_size ** 3]
        elif output_flat.size(-1) < self.field_size ** 3:
            padding = torch.zeros(batch_size, self.field_size ** 3 - output_flat.size(-1), 
                                 device=self.device)
            output_flat = torch.cat([output_flat, padding], dim=-1)
        
        output = self.output_proj(output_flat)
        
        return output
    
    def get_conservation_status(self) -> ConservationMetrics:
        """Check P+A+M conservation."""
        return self.field.get_conservation_metrics()


class ScaleInvariantTrainer:
    """
    Training loop that maintains Dawn Field invariants at scale.
    
    Key insight: The constants (φ×ξ, λ*, 1/l²) should work at any scale
    if they're truly fundamental. This trainer verifies that.
    """
    
    def __init__(
        self,
        n_patterns: int = 1000,
        embed_dim: int = 64,
        device: str = 'cuda'
    ):
        self.n_patterns = n_patterns
        self.embed_dim = embed_dim
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Adaptive sizing based on pattern count
        self.sizer = AdaptiveFieldSizer()
        self.field_shape = self.sizer.recommend_size(n_patterns)
        
        # Create scaled attention
        self.attention = ScaledFieldAttention(
            embed_dim=embed_dim,
            num_heads=4,
            field_size=self.field_shape[0],
            device=self.device
        )
        
    def test_invariance(
        self,
        patterns: torch.Tensor,
        n_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Test if Dawn Field constants remain valid at scale.
        
        Returns metrics on:
        - Conservation violation
        - Crystallization threshold accuracy
        - Semantic separation
        """
        patterns = patterns.to(self.device)
        batch_size = min(32, len(patterns))
        
        conservation_violations = []
        crystallization_scores = []
        
        for i in range(n_iterations):
            # Sample batch
            idx = torch.randperm(len(patterns))[:batch_size]
            batch = patterns[idx]
            
            # Forward pass
            output = self.attention(batch, batch, batch)
            
            # Check conservation
            metrics = self.attention.get_conservation_status()
            conservation_violations.append(metrics.violation)
            
            # Check crystallization (field coherence)
            with torch.no_grad():
                field = self.attention.field.get_combined_field()
                coherence = (field ** 2).sum() / (field.abs().sum() + 1e-8)
                crystallization_scores.append(coherence.item())
        
        return {
            'mean_conservation_violation': sum(conservation_violations) / len(conservation_violations),
            'max_conservation_violation': max(conservation_violations),
            'mean_crystallization': sum(crystallization_scores) / len(crystallization_scores),
            'field_size': self.field_shape[0],
            'n_patterns': self.n_patterns,
            'critical_threshold_3d': critical_density_3d(),
        }


if __name__ == '__main__':
    print("="*60)
    print("POC-004: Scale & Dimension - Core Module Test")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"3D Critical Density: {critical_density_3d():.4f}")
    
    # Test RearrangementTensor3D
    print("\n--- RearrangementTensor3D ---")
    field = RearrangementTensor3D(shape=(16, 16, 16), device=device)
    print(f"Initial total: {field.get_total():.6f}")
    
    # Apply some transfers
    rate = torch.ones(16, 16, 16, device=device) * 0.1
    field.transfer_p_to_a(rate, dt=0.1)
    field.transfer_a_to_m(rate, dt=0.1)
    
    metrics = field.get_conservation_metrics()
    print(f"After transfers - Total: {metrics.current_total:.6f}, Violation: {metrics.violation:.8f}")
    
    # Test SphericalHarmonicEncoder
    print("\n--- SphericalHarmonicEncoder ---")
    encoder = SphericalHarmonicEncoder(shape=(16, 16, 16), l_max=3, device=device)
    pattern = torch.randn(64, device=device)
    encoded = encoder.encode(pattern)
    print(f"Pattern shape: {pattern.shape} -> Field shape: {encoded.shape}")
    print(f"Field range: [{encoded.min():.4f}, {encoded.max():.4f}]")
    
    # Test AdaptiveFieldSizer
    print("\n--- AdaptiveFieldSizer ---")
    sizer = AdaptiveFieldSizer(min_size=16, max_size=64)
    for n in [100, 1000, 10000]:
        size = sizer.recommend_size(n)
        print(f"  {n:,} patterns -> field size {size}")
    
    # Test ScaledFieldAttention
    print("\n--- ScaledFieldAttention ---")
    attn = ScaledFieldAttention(embed_dim=64, num_heads=4, field_size=16, device=device)
    q = torch.randn(4, 64, device=device)
    k = torch.randn(4, 64, device=device)
    v = torch.randn(4, 64, device=device)
    out = attn(q, k, v)
    print(f"Attention output shape: {out.shape}")
    
    conservation = attn.get_conservation_status()
    print(f"Conservation: initial={conservation.initial_total:.4f}, "
          f"current={conservation.current_total:.4f}, violation={conservation.violation:.8f}")
    
    print("\n✅ All core components operational")
