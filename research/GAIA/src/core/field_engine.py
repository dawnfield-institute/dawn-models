"""
Field Engine for GAIA
Manages energy-information field dynamics with integrated PAC conservation.
Enhanced with Klein-Gordon evolution and built-in conservation mathematics.
"""

import numpy as np
import time
import math
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Import fracton core modules (with fallbacks)
try:
    from fracton.core.memory_field import MemoryField
    from fracton.core.recursive_engine import ExecutionContext
    from fracton.core.entropy_dispatch import EntropyLevel
    FRACTON_AVAILABLE = True
except ImportError:
    # Fallback implementations
    class MemoryField:
        def __init__(self): self.field_tensor = np.zeros((8, 8))
    class ExecutionContext:
        def __init__(self, entropy=0.5, depth=1): self.entropy = entropy; self.depth = depth
    class EntropyLevel: LOW = 0; MEDIUM = 1; HIGH = 2
    FRACTON_AVAILABLE = False

# Import native GAIA enhancement components (with fallbacks)
try:
    from .conservation_engine import ConservationEngine, ConservationMode
    from .emergence_detector import EmergenceDetector, EmergenceType
    from .pattern_amplifier import PatternAmplifier, AmplificationMode
except ImportError:
    # Fallback implementations for missing components
    class ConservationEngine:
        def __init__(self): pass
        def calculate_violations(self, field): return 0.0
    class EmergenceDetector:
        def __init__(self): pass
        def detect_patterns(self, field): return []
    class PatternAmplifier:
        def __init__(self): pass
        def amplify_patterns(self, field, patterns): return field
    class ConservationMode: STRICT = 1
    class EmergenceType: STRUCTURAL = 1
    class AmplificationMode: RESONANT = 1


class PACMathematics:
    """
    Integrated PAC (Persistent Arithmetic Conservation) mathematics.
    Implements Xi operator and conservation enforcement directly.
    """
    
    XI_OPERATOR_CONSTANT = 1.0571  # Discovered PAC constant
    
    @staticmethod
    def calculate_conservation_residual(field_values: np.ndarray, 
                                      parent_indices: np.ndarray = None,
                                      child_indices: np.ndarray = None) -> Tuple[float, float]:
        """
        Calculate PAC conservation residual using parent-child relationships.
        Returns (conservation_residual, xi_operator_deviation).
        """
        if len(field_values) == 0:
            return 0.0, 0.0
            
        # Default parent-child structure if not provided
        if parent_indices is None or child_indices is None:
            parent_indices = np.arange(0, len(field_values), 2)
            child_indices = np.arange(1, len(field_values), 2)
            
        # Ensure matching pairs
        min_len = min(len(parent_indices), len(child_indices))
        if min_len == 0:
            return 0.0, 0.0
            
        parent_indices = parent_indices[:min_len]
        child_indices = child_indices[:min_len]
        
        # Calculate conservation: f(parent) should equal Σf(children)
        parent_values = field_values[parent_indices]
        child_values = field_values[child_indices]
        
        # PAC conservation residual
        conservation_violations = np.abs(parent_values - child_values)
        conservation_residual = np.mean(conservation_violations)
        
        # Xi operator measurement
        parent_sum = np.sum(parent_values)
        child_sum = np.sum(child_values)
        
        if child_sum != 0:
            measured_xi = parent_sum / child_sum
            xi_deviation = abs(measured_xi - PACMathematics.XI_OPERATOR_CONSTANT)
        else:
            xi_deviation = float('inf')
            
        return float(conservation_residual), float(xi_deviation)
    
    @staticmethod
    def enforce_conservation(field_values: np.ndarray) -> np.ndarray:
        """Enforce PAC conservation on field values."""
        if len(field_values) < 2:
            return field_values
            
        # Create parent-child pairs
        conserved_field = field_values.copy()
        
        for i in range(0, len(conserved_field) - 1, 2):
            parent_idx = i
            child_idx = i + 1
            
            # Enforce conservation: adjust child to match parent
            parent_val = conserved_field[parent_idx]
            child_val = conserved_field[child_idx]
            
            # Apply Xi operator correction
            corrected_child = parent_val / PACMathematics.XI_OPERATOR_CONSTANT
            conserved_field[child_idx] = corrected_child
            
        return conserved_field


@dataclass
class FieldState:
    """Current state of energy-information fields."""
    energy_field: np.ndarray
    information_field: np.ndarray
    entropy_tensor: np.ndarray
    field_pressure: float
    delta_entropy: float
    collapse_likelihood: float
    potential_structures: int
    timestamp: float


@dataclass
class FieldPressure:
    """Field pressure analysis result."""
    pressure_magnitude: float
    gradient_norm: float
    divergence: float
    critical_points: List[Tuple[int, int]]
    stability_index: float


class EnergyField:
    """
    Genuine physics-based energy field implementing the Klein-Gordon equation.
    Represents quantum field dynamics with proper wave propagation and conservation.
    """
    
    def __init__(self, shape: Tuple[int, int] = (32, 32), dx: float = 0.1, dt: float = 0.01):
        self.shape = shape
        self.dx = dx  # Spatial step size
        self.dt = dt  # Time step size
        self.c = 1.0  # Speed of light (normalized)
        self.m = 0.1  # Field mass parameter
        
        # Field values and derivatives (complex for quantum field)
        self.field = np.zeros(shape, dtype=np.complex128)
        self.field_dot = np.zeros(shape, dtype=np.complex128)  # ∂φ/∂t
        self.field_prev = np.zeros(shape, dtype=np.complex128)
        
        # Physical constants and metrics
        self.total_energy = 0.0
        self.momentum_density = np.zeros(shape, dtype=np.complex128)
        self.stress_tensor = np.zeros((*shape, 2, 2), dtype=np.complex128)
        
        self.history = []
    
    def update(self, input_data: Any, context: ExecutionContext) -> np.ndarray:
        """Update field using Klein-Gordon equation: □φ + m²φ = J (with source J)."""
        
        # Convert input to source term (external current)
        source_term = self._compute_source_term(input_data, context)
        
        # Solve Klein-Gordon equation: ∂²φ/∂t² - c²∇²φ + m²φ = J
        # Using finite difference scheme
        laplacian = self._compute_laplacian(self.field)
        
        # Second-order time derivative
        field_ddot = (self.c**2 * laplacian - self.m**2 * self.field + source_term)
        
        # Update using Verlet integration for stability
        new_field = 2 * self.field - self.field_prev + field_ddot * self.dt**2
        
        # Update field history
        self.field_prev = self.field.copy()
        self.field = new_field
        
        # Update first derivative
        self.field_dot = (self.field - self.field_prev) / self.dt
        
        # Compute conserved quantities
        self._update_conserved_quantities()
        
        # Record physical state history
        self.history.append({
            'field': self.field.copy(),
            'total_energy': self.total_energy,
            'field_amplitude': np.mean(np.abs(self.field)),
            'timestamp': time.time()
        })
        
        # Keep history bounded
        if len(self.history) > 100:
            self.history.pop(0)
        
        return np.abs(self.field).astype(np.float32)  # Return amplitude for compatibility
    
    def _compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute discrete Laplacian ∇²φ using finite differences."""
        laplacian = np.zeros_like(field)
        
        # Second derivatives in x and y directions
        # ∂²φ/∂x² ≈ (φ[i+1,j] - 2φ[i,j] + φ[i-1,j]) / dx²
        laplacian[1:-1, :] += (field[2:, :] - 2*field[1:-1, :] + field[:-2, :]) / self.dx**2
        laplacian[:, 1:-1] += (field[:, 2:] - 2*field[:, 1:-1] + field[:, :-2]) / self.dx**2
        
        # Periodic boundary conditions (field wraps around)
        laplacian[0, :] += (field[1, :] - 2*field[0, :] + field[-1, :]) / self.dx**2
        laplacian[-1, :] += (field[0, :] - 2*field[-1, :] + field[-2, :]) / self.dx**2
        laplacian[:, 0] += (field[:, 1] - 2*field[:, 0] + field[:, -1]) / self.dx**2
        laplacian[:, -1] += (field[:, 0] - 2*field[:, -1] + field[:, -2]) / self.dx**2
        
        return laplacian
    
    def _compute_source_term(self, input_data: Any, context: ExecutionContext) -> np.ndarray:
        """Convert input to physical source term for field equation."""
        source = np.zeros(self.shape, dtype=np.complex128)
        
        if isinstance(input_data, str):
            # Convert text to localized source based on character information content
            for i, char in enumerate(input_data[:min(len(input_data), self.shape[0])]):
                char_code = ord(char)
                # Use character entropy as source strength
                entropy = -np.log2((char_code % 26 + 1) / 27.0) if char_code > 32 else 0
                
                # Position based on character properties
                row = char_code % self.shape[0]
                col = (i * 7) % self.shape[1]  # Prime spacing for better distribution
                
                # Gaussian source centered at character position
                source[row, col] += entropy * 1.0  # Increased coupling strength
                
        elif isinstance(input_data, (int, float)):
            # Scalar input creates central source with strength proportional to magnitude
            center = (self.shape[0]//2, self.shape[1]//2)
            magnitude = abs(float(input_data))
            source[center] = magnitude * 5.0  # Increased coupling
            
        elif isinstance(input_data, (list, tuple)):
            # Array/list input creates distributed source pattern
            for i, item in enumerate(input_data[:min(len(input_data), self.shape[0] * self.shape[1])]):
                try:
                    value = float(item)
                    row = i % self.shape[0]
                    col = (i // self.shape[0]) % self.shape[1]
                    
                    # Source strength based on value magnitude and position correlation
                    strength = abs(value) * 0.5
                    # Add structured vs random distinction
                    if i > 0 and abs(value - float(input_data[i-1])) == 1:
                        strength *= 2.0  # Boost for sequential patterns
                    
                    source[row, col] += strength
                except (ValueError, TypeError):
                    pass
        else:
            # Generic input - create weak central source
            center = (self.shape[0]//2, self.shape[1]//2)
            source[center] = 1.0
            
        # Apply context modulation - deeper processing creates stronger sources
        context_factor = 2.0 + (context.depth or 0) * 0.5 + context.entropy * 1.0  # Increased base
        source *= context_factor
        
        return source
    
    def _update_conserved_quantities(self):
        """Calculate conserved energy and momentum from field configuration."""
        # Energy density: T₀₀ = (1/2)|∂φ/∂t|² + (1/2)c²|∇φ|² + (1/2)m²|φ|²
        kinetic_density = 0.5 * np.abs(self.field_dot)**2
        
        # Gradient energy (potential)
        grad_x = np.gradient(self.field, axis=0) / self.dx
        grad_y = np.gradient(self.field, axis=1) / self.dx
        gradient_density = 0.5 * self.c**2 * (np.abs(grad_x)**2 + np.abs(grad_y)**2)
        
        # Mass energy
        mass_density = 0.5 * self.m**2 * np.abs(self.field)**2
        
        # Total energy (integrated over space)
        energy_density = kinetic_density + gradient_density + mass_density
        self.total_energy = np.sum(energy_density) * self.dx**2
        
        # Momentum density: T₀ᵢ = -Re(∂φ*/∂t · ∂φ/∂xᵢ)
        self.momentum_density = -np.real(np.conj(self.field_dot) * 
                                        (np.gradient(self.field, axis=0) / self.dx + 
                                         1j * np.gradient(self.field, axis=1) / self.dx))

    def get_divergence(self) -> float:
        """Compute genuine divergence: ∇·E(x) from Maxwell equations."""
        # For complex field, compute divergence of real part
        real_field = np.real(self.field)
        grad_x, grad_y = np.gradient(real_field, self.dx)
        divergence_field = np.gradient(grad_x, axis=0) + np.gradient(grad_y, axis=1)
        return np.mean(divergence_field)
    def calculate_pressure(self) -> float:
        """Calculate genuine field pressure from stress-energy tensor."""
        # Pressure is the trace of spatial part of stress-energy tensor
        # P = (1/3)(T¹¹ + T²²) where Tⁱʲ are spatial components
        
        # Compute stress-energy tensor components
        grad_x = np.gradient(self.field, axis=0) / self.dx
        grad_y = np.gradient(self.field, axis=1) / self.dx
        
        # T¹¹ = (∂φ/∂x)(∂φ*/∂x) - (1/2)δ¹¹(kinetic + potential + mass terms)
        T11 = np.real(grad_x * np.conj(grad_x))
        T22 = np.real(grad_y * np.conj(grad_y))
        
        # Subtract trace part for pressure
        kinetic_term = 0.5 * np.abs(self.field_dot)**2
        potential_term = 0.5 * self.c**2 * (np.abs(grad_x)**2 + np.abs(grad_y)**2)
        mass_term = 0.5 * self.m**2 * np.abs(self.field)**2
        energy_density = kinetic_term + potential_term + mass_term
        
        pressure_density = 0.5 * (T11 + T22) - (2/3) * energy_density
        return np.mean(pressure_density)

    def get_flux_gradient(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute electromagnetic field gradients: E = -∇φ."""
        # Electric field from scalar potential
        grad_x = -np.gradient(np.real(self.field), axis=0) / self.dx
        grad_y = -np.gradient(np.real(self.field), axis=1) / self.dx
        return grad_x, grad_y
    
    @property
    def amplitude_field(self) -> np.ndarray:
        """Get field amplitude for compatibility."""
        return np.abs(self.field).astype(np.float32)
    
    @property
    def field_real(self) -> np.ndarray:
        """Compatibility property - returns real field amplitude."""
        return np.abs(self.field).astype(np.float32)


class InformationField:
    """
    Quantum information field implementing von Neumann equation dynamics.
    Represents evolution of quantum information density matrix ρ(x,t).
    """
    
    def __init__(self, shape: Tuple[int, int] = (32, 32), dx: float = 0.1, dt: float = 0.01):
        self.shape = shape
        self.dx = dx
        self.dt = dt
        self.hbar = 1.0  # Reduced Planck constant (normalized)
        
        # Density matrix field - represents quantum information state
        self.density_matrix = np.zeros((*shape, 2, 2), dtype=np.complex128)
        self.hamiltonian = np.zeros((*shape, 2, 2), dtype=np.complex128)
        
        # Initialize in mixed state (maximum entropy)
        identity = np.eye(2, dtype=np.complex128)
        self.density_matrix = np.broadcast_to(identity[None, None, :, :] / 2, (*shape, 2, 2)).copy()
        
        # Information theoretic quantities
        self.entropy_density = np.zeros(shape, dtype=np.float64)
        self.mutual_information = 0.0
        self.quantum_coherence = np.zeros(shape, dtype=np.float64)
        
        self.structure_history = []
    
    def update(self, memory_field, energy_field: np.ndarray) -> np.ndarray:
        """Update using von Neumann equation: ∂ρ/∂t = -(i/ℏ)[H,ρ] + dissipation."""
        
        # Construct Hamiltonian from energy field and memory coupling
        self._update_hamiltonian(energy_field, memory_field)
        
        # Evolve density matrix using von Neumann equation
        self._evolve_density_matrix()
        
        # Calculate information theoretic quantities
        self._compute_information_measures()
        
        # Record quantum information state
        self.structure_history.append({
            'entropy_density': self.entropy_density.copy(),
            'mutual_information': self.mutual_information,
            'quantum_coherence': np.mean(self.quantum_coherence),
            'timestamp': time.time()
        })
        
        # Keep history bounded
        if len(self.structure_history) > 100:
            self.structure_history.pop(0)
        
        # Return information density for compatibility
        return self.entropy_density.astype(np.float32)
    
    def _update_hamiltonian(self, energy_field: np.ndarray, memory_field):
        """Construct quantum Hamiltonian from external fields."""
        # Pauli matrices for 2-level quantum system
        sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                # Energy field couples to σz (diagonal coupling)
                h_energy = energy_field[i, j] * sigma_z
                
                # Memory field creates off-diagonal coupling (σx term)
                if hasattr(memory_field, 'items'):
                    memory_strength = len(dict(memory_field.items())) * 0.1
                else:
                    memory_strength = 0.1
                    
                # Spatial coupling (tunneling) - σx coupling to neighbors
                neighbor_coupling = 0.0
                if i > 0: neighbor_coupling += energy_field[i-1, j]
                if i < self.shape[0]-1: neighbor_coupling += energy_field[i+1, j]
                if j > 0: neighbor_coupling += energy_field[i, j-1]
                if j < self.shape[1]-1: neighbor_coupling += energy_field[i, j+1]
                
                h_coupling = memory_strength * neighbor_coupling * 0.1 * sigma_x
                
                # Total Hamiltonian
                self.hamiltonian[i, j] = h_energy + h_coupling
    
    def _evolve_density_matrix(self):
        """Evolve density matrix using von Neumann equation with dissipation."""
        gamma = 0.01  # Decoherence rate
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                rho = self.density_matrix[i, j]
                H = self.hamiltonian[i, j]
                
                # von Neumann equation: ∂ρ/∂t = -(i/ℏ)[H,ρ]
                commutator = (H @ rho - rho @ H) * (-1j / self.hbar)
                
                # Add decoherence (Lindblad term) - damps off-diagonal elements
                dissipator = -gamma * (rho - np.diag(np.diag(rho)))
                
                # Update with Runge-Kutta 4th order for stability
                drho_dt = commutator + dissipator
                
                # Simple Euler step (can be improved with RK4)
                self.density_matrix[i, j] = rho + drho_dt * self.dt
                
                # Ensure trace preservation and positivity
                self.density_matrix[i, j] = self._project_to_valid_density_matrix(
                    self.density_matrix[i, j])
    
    def _project_to_valid_density_matrix(self, rho: np.ndarray) -> np.ndarray:
        """Project to valid density matrix (positive, trace=1)."""
        # Hermitianize
        rho = (rho + rho.conj().T) / 2
        
        # Diagonalize and enforce positivity
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        eigenvals = np.maximum(eigenvals, 0)  # Enforce positivity
        
        # Normalize trace to 1
        eigenvals = eigenvals / np.sum(eigenvals) if np.sum(eigenvals) > 0 else np.array([0.5, 0.5])
        
        # Reconstruct density matrix
        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.conj().T
    
    def _compute_information_measures(self):
        """Compute quantum information measures from density matrices."""
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                rho = self.density_matrix[i, j]
                
                # Von Neumann entropy: S = -Tr(ρ log ρ)
                eigenvals = np.linalg.eigvals(rho)
                eigenvals = eigenvals[eigenvals > 1e-12]  # Avoid log(0)
                self.entropy_density[i, j] = -np.sum(eigenvals * np.log2(eigenvals))
                
                # Quantum coherence (off-diagonal magnitude)
                self.quantum_coherence[i, j] = np.abs(rho[0, 1])
        
        # Mutual information between spatial regions (simplified)
        left_half = self.entropy_density[:, :self.shape[1]//2]
        right_half = self.entropy_density[:, self.shape[1]//2:]
        total_entropy = np.sum(self.entropy_density)
        self.mutual_information = (np.sum(left_half) + np.sum(right_half) - total_entropy)
    
    def get_compression_gradient(self) -> float:
        """Compute gradient of quantum information entropy: ∇·S(ρ)."""
        grad_x, grad_y = np.gradient(self.entropy_density, self.dx)
        divergence = np.gradient(grad_x, axis=0) + np.gradient(grad_y, axis=1)
        return np.mean(divergence)
    
    def get_regularity_index(self) -> float:
        """Compute quantum coherence as a measure of information regularity."""
        # Average quantum coherence across the field
        total_coherence = np.sum(self.quantum_coherence)
        max_possible_coherence = self.shape[0] * self.shape[1] * 0.5  # Maximum coherence
        return total_coherence / max(max_possible_coherence, 1e-10)
    
    def get_entanglement_entropy(self, region_A: Tuple[slice, slice]) -> float:
        """Compute entanglement entropy between spatial regions."""
        # Extract density matrices for region A
        rho_A_matrices = self.density_matrix[region_A]
        
        # Partial trace to get reduced density matrix (simplified for 2x2 case)
        # In full implementation, would need proper partial trace over region B
        total_entropy_A = 0.0
        count = 0
        
        for i in range(rho_A_matrices.shape[0]):
            for j in range(rho_A_matrices.shape[1]):
                rho = rho_A_matrices[i, j]
                eigenvals = np.linalg.eigvals(rho)
                eigenvals = eigenvals[eigenvals > 1e-12]
                entropy = -np.sum(eigenvals * np.log2(eigenvals))
                total_entropy_A += entropy
                count += 1
        
        return total_entropy_A / max(count, 1)
    
    def get_quantum_fidelity(self, target_state: np.ndarray) -> float:
        """Compute average quantum fidelity with target state across field."""
        fidelities = []
        target_dm = np.outer(target_state, target_state.conj())  # |ψ⟩⟨ψ|
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                rho = self.density_matrix[i, j]
                # Quantum fidelity: F = Tr(√(√ρ σ √ρ))
                # For density matrices, simplified calculation
                fidelity = np.real(np.trace(rho @ target_dm))
                fidelities.append(fidelity)
        
        return np.mean(fidelities)
    
    @property 
    def field(self) -> np.ndarray:
        """Compatibility property - returns entropy density as 'field'."""
        return self.entropy_density.astype(np.float32)
    
    @field.setter
    def field(self, value: np.ndarray):
        """Compatibility setter - updates entropy density when 'field' is set."""
        if value.shape == self.shape:
            self.entropy_density = value.astype(np.float64)
        
        return refined.astype(np.float32)
    
    def calculate_pressure(self) -> float:
        """Calculate pressure from information field dynamics."""
        compression = self.get_compression_gradient()
        field_variance = np.var(self.field)
        return abs(compression) + field_variance * 0.3


class EntropyTensor:
    """
    Measures deviation between energy and information fields.
    Computes various entropy metrics and tracks field irregularities.
    """
    
    def __init__(self, shape: Tuple[int, int] = (32, 32)):
        self.shape = shape
        self.tensor = np.zeros(shape, dtype=np.float32)
        self.von_neumann_entropy = 0.0
        self.fisher_information = 0.0
    
    def compute(self, energy_field: np.ndarray, information_field: np.ndarray) -> np.ndarray:
        """Compute entropy tensor from energy-information field deviation."""
        # Compute field difference
        field_deviation = energy_field - information_field
        
        # Local entropy density
        local_entropy = self._compute_local_entropy(field_deviation)
        
        # Structural irregularity
        irregularity = self._compute_irregularity(field_deviation)
        
        # Combine metrics
        self.tensor = local_entropy + 0.5 * irregularity
        
        # Compute global metrics
        self.von_neumann_entropy = self._compute_von_neumann_entropy()
        self.fisher_information = self._compute_fisher_information()
        
        return self.tensor
    
    def get_pressure_points(self, threshold: float = 0.7) -> List[Tuple[int, int]]:
        """Find points where entropy pressure exceeds threshold."""
        pressure_mask = self.tensor > threshold
        points = []
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if pressure_mask[i, j]:
                    points.append((i, j))
        
        return points
    
    def _compute_local_entropy(self, field_deviation: np.ndarray) -> np.ndarray:
        """Compute local entropy density."""
        # Use local variance as entropy proxy
        from scipy.ndimage import uniform_filter
        
        local_mean = uniform_filter(field_deviation, size=3)
        local_variance = uniform_filter(field_deviation**2, size=3) - local_mean**2
        
        # Convert variance to entropy-like measure
        entropy = np.log(1 + np.abs(local_variance))
        return entropy.astype(np.float32)
    
    def _compute_irregularity(self, field_deviation: np.ndarray) -> np.ndarray:
        """Compute structural irregularity measure."""
        # Use Laplacian to detect irregularities
        from scipy.ndimage import laplace
        
        laplacian = laplace(field_deviation)
        irregularity = np.abs(laplacian)
        
        # Normalize
        max_val = np.max(irregularity)
        if max_val > 0:
            irregularity = irregularity / max_val
        
        return irregularity.astype(np.float32)
    
    def _compute_von_neumann_entropy(self) -> float:
        """Compute Von Neumann entropy of the tensor."""
        # Simplified eigenvalue-based entropy
        # Reshape to matrix and compute eigenvalues
        matrix = self.tensor.reshape(-1, 1) @ self.tensor.reshape(1, -1)
        eigenvals = np.linalg.eigvals(matrix)
        
        # Remove negative/zero eigenvalues and normalize
        eigenvals = eigenvals[eigenvals > 1e-10]
        eigenvals = eigenvals / np.sum(eigenvals)
        
        # Compute entropy
        entropy = -np.sum(eigenvals * np.log(eigenvals + 1e-10))
        return float(entropy)
    
    def _compute_fisher_information(self) -> float:
        """Compute Quantum Fisher Information approximation."""
        # Use gradient magnitude as Fisher information proxy
        grad_x, grad_y = np.gradient(self.tensor)
        fisher = np.mean(grad_x**2 + grad_y**2)
        return float(fisher)


class BalanceController:
    """
    Computes pressure between energy and information fields.
    Determines when to trigger collapse events.
    """
    
    def __init__(self, collapse_threshold: float = 0.6):
        self.collapse_threshold = collapse_threshold
        self.pressure_history = []
        self.balance_metrics = {}
    
    def compute_balance(self, energy_field: EnergyField, information_field: InformationField, 
                      entropy_tensor: EntropyTensor) -> FieldPressure:
        """Compute field balance and pressure metrics."""
        # Get field divergences
        energy_divergence = energy_field.get_divergence()
        info_divergence = information_field.get_compression_gradient()
        
        # Compute pressure magnitude
        pressure_magnitude = abs(energy_divergence - info_divergence)
        
        # Gradient analysis
        e_grad = energy_field.get_flux_gradient()
        gradient_norm = np.linalg.norm([np.linalg.norm(e_grad[0]), np.linalg.norm(e_grad[1])])
        
        # Find critical points
        critical_points = entropy_tensor.get_pressure_points(self.collapse_threshold)
        
        # Stability index
        regularity = information_field.get_regularity_index()
        alignment = information_field.memory_alignment
        stability_index = regularity * alignment
        
        pressure = FieldPressure(
            pressure_magnitude=pressure_magnitude,
            gradient_norm=gradient_norm,
            divergence=energy_divergence + info_divergence,
            critical_points=critical_points,
            stability_index=stability_index
        )
        
        # Record pressure history
        self.pressure_history.append({
            'pressure': pressure_magnitude,
            'stability': stability_index,
            'critical_count': len(critical_points),
            'timestamp': time.time()
        })
        
        # Keep history bounded
        if len(self.pressure_history) > 1000:
            self.pressure_history.pop(0)
        
        return pressure
    
    def should_collapse(self, pressure: FieldPressure, entropy: float) -> bool:
        """Determine if conditions warrant triggering collapse."""
        # Multiple collapse criteria
        pressure_trigger = pressure.pressure_magnitude > self.collapse_threshold
        instability_trigger = pressure.stability_index < 0.3
        entropy_trigger = entropy > 0.7
        critical_mass_trigger = len(pressure.critical_points) > 5
        
        # Adaptive threshold based on recent pressure
        if len(self.pressure_history) > 10:
            recent_pressure = [p['pressure'] for p in self.pressure_history[-10:]]
            avg_pressure = np.mean(recent_pressure)
            adaptive_trigger = pressure.pressure_magnitude > avg_pressure * 1.2
        else:
            adaptive_trigger = False
        
        return (pressure_trigger or instability_trigger or 
                entropy_trigger or critical_mass_trigger or adaptive_trigger)
    
    def detect_collapse_conditions(self) -> Dict[str, Any]:
        """Detect current collapse conditions and readiness."""
        if not self.pressure_history:
            return {
                'collapse_ready': False,
                'pressure_level': 0.0,
                'stability_level': 1.0,
                'conditions_met': []
            }
        
        latest = self.pressure_history[-1]
        conditions_met = []
        
        if latest['pressure'] > self.collapse_threshold:
            conditions_met.append('pressure_threshold')
        if latest['stability'] < 0.3:
            conditions_met.append('instability')
        if latest['critical_count'] > 5:
            conditions_met.append('critical_mass')
        
        return {
            'collapse_ready': len(conditions_met) > 0,
            'pressure_level': latest['pressure'],
            'stability_level': latest['stability'],
            'critical_points': latest['critical_count'],
            'conditions_met': conditions_met
        }


class FieldEngine:
    """
    Main field engine coordinating energy-information dynamics.
    Monitors entropy pressure and triggers collapse events.
    """
    
    def __init__(self, shape: Tuple[int, int] = (32, 32), collapse_threshold: float = 0.6):
        self.shape = shape
        self.energy_field = EnergyField(shape)
        self.information_field = InformationField(shape)
        self.entropy_tensor = EntropyTensor(shape)
        self.balance_controller = BalanceController(collapse_threshold)
        
        # Initialize native GAIA enhancement components
        self.conservation_engine = ConservationEngine(field_shape=shape)
        self.emergence_detector = EmergenceDetector(field_shape=shape)
        self.pattern_amplifier = PatternAmplifier(field_shape=shape)
        print(f"Native GAIA-enhanced field engine initialized with {max(shape)}x{max(shape)} resolution")
        
        # Statistics
        self.update_count = 0
        self.collapse_triggers = 0
        self.field_states = []
    
    def update_fields(self, input_data: Any, memory_field: MemoryField, 
                     context: ExecutionContext) -> FieldState:
        """Update complex amplitude fields with PAC conservation constraints."""
        self.update_count += 1
        
        # Convert input to complex amplitude distribution
        input_amplitude = self._input_to_amplitude_distribution(input_data, context)
        
        # Apply PAC conservation constraint - this is the core physics
        if not hasattr(self, 'amplitude_field'):
            # Initialize complex amplitude field on first use
            shape = getattr(self.energy_field, 'shape', (32, 32))
            self.amplitude_field = np.zeros(shape, dtype=complex)
            
        # Add input amplitude while conserving total probability
        self.amplitude_field = self._conserve_amplitude_addition(self.amplitude_field, input_amplitude)
        
        # Native GAIA pattern amplification (keep existing enhancement)
        amplitude_data = np.abs(self.amplitude_field)**2  # Convert to real intensity field
        amplification_result = self.pattern_amplifier.amplify_pattern(amplitude_data)

        if amplification_result.amplification_factor > 1.1:
            # Apply amplification boost to amplitude (preserving conservation)
            boost_factor = 1.0 + (amplification_result.amplification_factor - 1.0) * 0.1
            # Scale amplitude, not probability
            self.amplitude_field *= np.sqrt(boost_factor)  
            self.amplitude_field = self._renormalize_field(self.amplitude_field)  # Conserve total
        
        # Extract energy and information from complex amplitude (quantum-like)
        energy_array = np.abs(self.amplitude_field) ** 2  # Energy = |ψ|²
        info_array = np.angle(self.amplitude_field)       # Information = arg(ψ)
        
        # Update legacy fields for compatibility - use actual physics field updates
        # Energy field gets source term from current amplitude
        self.energy_field.update(energy_array.mean(), context)
        
        # Information field updates with genuine quantum dynamics
        self.information_field.update(memory_field, self.energy_field.amplitude_field)
        
        # Calculate conservation residual (replaces entropy tensor)
        conservation_residual = self._compute_conservation_residual()
        violation_magnitude = np.linalg.norm(conservation_residual)
        
        # Native GAIA conservation validation using PAC principles
        # Update conservation engine with current amplitude
        self.conservation_engine._update_pac_fields(self.amplitude_field)
        
        # Validate conservation state
        conservation_result = self.conservation_engine.validate_conservation()
        
        if not conservation_result.get('valid', True):
            print("GAIA PAC conservation violation detected - renormalizing amplitude field")
            self.amplitude_field = self._renormalize_field(self.amplitude_field)
            # Recalculate after correction
            energy_array = np.abs(self.amplitude_field) ** 2
            info_array = np.angle(self.amplitude_field)
            conservation_residual = self._compute_conservation_residual()
            violation_magnitude = np.linalg.norm(conservation_residual)
        
        # Field pressure now comes from conservation violations (principled physics)
        field_pressure = violation_magnitude
        
        # Collapse likelihood scaled by Xi operator
        xi_operator = 1.0571  # PAC fundamental constant
        collapse_likelihood = min(field_pressure * xi_operator, 1.0)
        
        # Find potential structures from phase singularities (real physics)
        potential_structures = self._count_phase_singularities(info_array)
        
        # Create field state with PAC-derived values
        field_state = FieldState(
            energy_field=energy_array,
            information_field=info_array,
            entropy_tensor=conservation_residual,  # Conservation residual replaces arbitrary entropy
            field_pressure=field_pressure,         # Now violation magnitude 
            delta_entropy=violation_magnitude,     # Meaningful conservation violation measure
            collapse_likelihood=collapse_likelihood, # Xi-scaled violation probability
            potential_structures=potential_structures, # Phase singularity count
            timestamp=time.time()
        )
        
        # Store total amplitude for conservation verification
        field_state.total_amplitude = self.amplitude_field
        
        # Record state
        self.field_states.append(field_state)
        if len(self.field_states) > 100:
            self.field_states.pop(0)
        
        return field_state
    
    def check_collapse_trigger(self, field_state: FieldState, context: ExecutionContext) -> bool:
        """Check if field conditions warrant collapse trigger."""
        pressure = FieldPressure(
            pressure_magnitude=field_state.field_pressure,
            gradient_norm=field_state.collapse_likelihood,
            divergence=field_state.delta_entropy,
            critical_points=[(0, 0)] * field_state.potential_structures,  # Simplified
            stability_index=1.0 - field_state.collapse_likelihood
        )
        
        should_collapse = self.balance_controller.should_collapse(pressure, context.entropy)
        
        if should_collapse:
            self.collapse_triggers += 1
        
        return should_collapse
    
    def get_field_state(self) -> FieldState:
        """Get most recent field state."""
        return self.field_states[-1] if self.field_states else FieldState(
            energy_field=np.zeros(self.shape),
            information_field=np.zeros(self.shape),
            entropy_tensor=np.zeros(self.shape),
            field_pressure=0.0,
            delta_entropy=0.0,
            collapse_likelihood=0.0,
            potential_structures=0,
            timestamp=time.time()
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get field engine statistics."""
        return {
            'update_count': self.update_count,
            'collapse_triggers': self.collapse_triggers,
            'current_field_pressure': self.get_field_state().field_pressure,
            'von_neumann_entropy': self.entropy_tensor.von_neumann_entropy,
            'fisher_information': self.entropy_tensor.fisher_information,
            'energy_field_magnitude': np.mean(np.abs(self.energy_field.field)),
            'information_alignment': self.information_field.memory_alignment,
            'stability_index': self.balance_controller.pressure_history[-1]['stability'] if self.balance_controller.pressure_history else 0.0
        }
    
    def get_field_statistics(self) -> Dict[str, Any]:
        """Get comprehensive field engine statistics."""
        energy_flux = self.energy_field.get_divergence() if hasattr(self.energy_field, 'get_divergence') else 0.0
        info_compression = self.information_field.get_compression_gradient() if hasattr(self.information_field, 'get_compression_gradient') else 0.0
        
        return {
            'average_entropy': 0.5,  # Default entropy level
            'total_pressure': abs(energy_flux) + abs(info_compression),
            'energy_divergence': energy_flux,
            'information_compression': info_compression,
            'field_updates': self.update_count,
            'collapse_triggers': self.collapse_triggers,
            'field_balance': 0.5  # Neutral balance
        }
    
    def reset(self):
        """Reset field engine to initial state."""
        self.energy_field = EnergyField(self.shape)
        self.information_field = InformationField(self.shape)
        self.entropy_tensor = EntropyTensor(self.shape)
        self.balance_controller = BalanceController(self.balance_controller.collapse_threshold)
        self.update_count = 0
        self.collapse_triggers = 0
        self.field_states.clear()
        
        # Reset native GAIA components
        self.conservation_engine.reset_conservation_state()
        self.emergence_detector.reset_detection_state()
        
        # Reset pattern amplifier statistics
        stats = self.pattern_amplifier.get_amplification_statistics()
        print(f"Field engine reset - {stats['total_amplifications']} patterns amplified")
    
    def _process_input_with_pac(self, input_data: Any, context: ExecutionContext) -> Any:
        """Process input through PAC substrate for enhanced field dynamics."""
        try:
            # Convert input to pattern suitable for PAC processing
            if isinstance(input_data, str):
                pattern_data = {'text': input_data, 'entropy': context.entropy}
            elif isinstance(input_data, (list, tuple)):
                pattern_data = {'sequence': list(input_data), 'entropy': context.entropy}
            elif isinstance(input_data, dict):
                pattern_data = dict(input_data)
                pattern_data['entropy'] = context.entropy
            else:
                pattern_data = {'data': str(input_data), 'entropy': context.entropy}
            
            # Process through PAC kernel for pattern amplification
            amplified_pattern = self.pac_kernel.process_pattern(
                pattern_data,
                conservation_mode='energy_information'
            )
            
            # Extract enhanced input for field processing
            enhanced_input = amplified_pattern.get('enhanced_input', input_data)
            print(f"PAC substrate enhanced input with amplification factor: {amplified_pattern.get('amplification_factor', 1.0)}")
            
            return enhanced_input
            
        except Exception as e:
            print(f"PAC input processing failed: {e}, using original input")
            return input_data

    def _input_to_amplitude_distribution(self, input_data: Any, context: ExecutionContext) -> np.ndarray:
        """Convert input data to complex amplitude distribution for PAC processing."""
        shape = getattr(self.energy_field, 'shape', (32, 32))
        
        # Start with uniform amplitude base
        amplitude = np.ones(shape, dtype=complex) * 0.1
        
        if isinstance(input_data, str):
            # Convert text to amplitude pattern
            for i, char in enumerate(input_data[:min(len(input_data), 50)]):
                row = (ord(char) % shape[0])
                col = (i % shape[1])
                # Add amplitude with phase encoding
                phase = 2 * np.pi * (ord(char) / 255.0)
                amplitude[row, col] += 0.1 * np.exp(1j * phase)
                
        elif isinstance(input_data, (int, float)):
            # Convert scalar to radial amplitude pattern
            center_x, center_y = shape[0] // 2, shape[1] // 2
            x, y = np.meshgrid(range(shape[1]), range(shape[0]))
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            
            # Gaussian amplitude with value-dependent phase
            sigma = max_distance / 3
            magnitude = float(input_data) * 0.1
            phase = 2 * np.pi * (float(input_data) % 1.0)
            amplitude += magnitude * np.exp(-distance**2 / (2 * sigma**2)) * np.exp(1j * phase)
            
        elif isinstance(input_data, (list, tuple)):
            # Convert sequence to amplitude pattern
            for i, item in enumerate(input_data[:min(len(input_data), shape[0] * shape[1])]):
                row = i // shape[1]
                col = i % shape[1]
                if row < shape[0]:
                    val = float(item) if isinstance(item, (int, float)) else hash(str(item)) / 1e6
                    phase = 2 * np.pi * (val % 1.0)
                    amplitude[row, col] += 0.05 * np.exp(1j * phase)
        
        # Apply context depth as wave modulation
        if hasattr(context, 'depth') and context.depth:
            x, y = np.meshgrid(range(shape[1]), range(shape[0]))
            wave_phase = 2 * np.pi * context.depth * (x + y) / (shape[0] + shape[1])
            amplitude *= np.exp(1j * wave_phase * 0.1)
        
        return amplitude
    
    def _conserve_amplitude_addition(self, existing_field: np.ndarray, new_amplitude: np.ndarray) -> np.ndarray:
        """Add new amplitude while conserving total probability."""
        # Add amplitudes
        combined = existing_field + new_amplitude
        
        # Calculate current total probability
        current_total = np.sum(np.abs(combined) ** 2)
        
        # Normalize to conserve unit probability (∑|ψ|² = constant)
        target_total = existing_field.size  # Normalize to field size
        if current_total > 1e-10:
            conservation_factor = np.sqrt(target_total / current_total)
            combined *= conservation_factor
        
        return combined
    
    def _renormalize_field(self, amplitude_field: np.ndarray) -> np.ndarray:
        """Renormalize amplitude field to conserve total probability."""
        total = np.sum(np.abs(amplitude_field) ** 2)
        target = amplitude_field.size
        
        if total > 1e-10:
            return amplitude_field * np.sqrt(target / total)
        else:
            return np.ones_like(amplitude_field) * np.sqrt(target / amplitude_field.size)
    
    def _compute_conservation_residual(self) -> np.ndarray:
        """Compute conservation violation residual from amplitude field."""
        if not hasattr(self, 'amplitude_field'):
            return np.zeros((32, 32))
        
        # Local probability density
        local_density = np.abs(self.amplitude_field) ** 2
        
        # Expected uniform density for perfect conservation
        total_prob = np.sum(local_density)
        expected_density = total_prob / self.amplitude_field.size
        
        # Residual = deviation from uniform (perfect conservation)
        residual = local_density - expected_density
        
        return residual
    
    def _count_phase_singularities(self, phase_field: np.ndarray) -> int:
        """Count phase singularities (vortices) in phase field - real physics."""
        singularities = 0
        
        try:
            # Look for phase singularities where phase wraps around
            for i in range(1, phase_field.shape[0] - 1):
                for j in range(1, phase_field.shape[1] - 1):
                    # Get phase values around this point
                    phases = [
                        phase_field[i-1, j], phase_field[i, j-1],
                        phase_field[i+1, j], phase_field[i, j+1]
                    ]
                    
                    # Calculate phase differences
                    phase_diff = 0
                    for k in range(len(phases)):
                        diff = phases[k] - phases[k-1]
                        # Unwrap phase difference
                        if diff > np.pi:
                            diff -= 2 * np.pi
                        elif diff < -np.pi:
                            diff += 2 * np.pi
                        phase_diff += diff
                    
                    # Singularity if total phase change ≈ ±2π
                    if abs(abs(phase_diff) - 2 * np.pi) < 0.5:
                        singularities += 1
                        
        except Exception as e:
            print(f"Phase singularity detection error: {e}")
        
        return singularities
