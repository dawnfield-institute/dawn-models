"""
Field Engine for GAIA
Manages energy-information field dynamics and triggers collapse events.
See docs/architecture/modules/field_engine.md for design details.
"""

import numpy as np
import time
import math
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Import fracton core modules
from fracton.core.memory_field import MemoryField
from fracton.core.recursive_engine import ExecutionContext
from fracton.core.entropy_dispatch import EntropyLevel


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
    Represents kinetic potential and activation flux in the system.
    Handles energy field computation and gradient analysis.
    """
    
    def __init__(self, shape: Tuple[int, int] = (32, 32)):
        self.shape = shape
        self.field = np.zeros(shape, dtype=np.float32)
        self.history = []
        self.flux_threshold = 0.5
    
    def update(self, input_data: Any, context: ExecutionContext) -> np.ndarray:
        """Update energy field based on input and context."""
        # Convert input to energy field activation
        if isinstance(input_data, (int, float)):
            activation = self._scalar_to_field(float(input_data))
        elif isinstance(input_data, str):
            activation = self._string_to_field(input_data)
        elif isinstance(input_data, (list, tuple)):
            activation = self._sequence_to_field(input_data)
        else:
            activation = self._generic_to_field(input_data)
        
        # Apply context-dependent modulation
        depth_factor = 1.0 + 0.1 * (context.depth or 0)
        entropy_factor = 1.0 + context.entropy
        
        # Update field with temporal decay and new activation
        decay_rate = 0.95
        self.field = self.field * decay_rate + activation * depth_factor * entropy_factor
        
        # Record history
        self.history.append({
            'field': self.field.copy(),
            'timestamp': time.time(),
            'flux_magnitude': np.mean(np.abs(self.field))
        })
        
        # Keep history bounded
        if len(self.history) > 100:
            self.history.pop(0)
        
        return self.field
    
    def get_flux_gradient(self) -> np.ndarray:
        """Compute local flux gradient across field."""
        return np.gradient(self.field)
    
    def get_divergence(self) -> float:
        """Compute divergence: ∇·E(x)"""
        grad_x, grad_y = np.gradient(self.field)
        return np.mean(grad_x + grad_y)
    
    def _scalar_to_field(self, value: float) -> np.ndarray:
        """Convert scalar to energy field activation."""
        # Create radial activation pattern
        center_x, center_y = self.shape[0] // 2, self.shape[1] // 2
        x, y = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Gaussian activation
        sigma = max_distance / 3
        activation = value * np.exp(-distance**2 / (2 * sigma**2))
        return activation.astype(np.float32)
    
    def _string_to_field(self, text: str) -> np.ndarray:
        """Convert string to energy field activation."""
        # Use character frequencies and positions
        activation = np.zeros(self.shape, dtype=np.float32)
        
        for i, char in enumerate(text[:min(len(text), 100)]):
            # Map character to field position
            row = (ord(char) % self.shape[0])
            col = (i % self.shape[1])
            activation[row, col] += 0.1
        
        # Smooth with Gaussian filter
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(activation, sigma=1.0)
    
    def _sequence_to_field(self, sequence: Union[List, Tuple]) -> np.ndarray:
        """Convert sequence to energy field activation."""
        activation = np.zeros(self.shape, dtype=np.float32)
        
        for i, item in enumerate(sequence[:min(len(sequence), 50)]):
            if isinstance(item, (int, float)):
                row = int(abs(float(item)) * 10) % self.shape[0]
                col = i % self.shape[1]
                activation[row, col] += 0.1
        
        return activation
    
    def _generic_to_field(self, data: Any) -> np.ndarray:
        """Generic conversion for unknown data types."""
        # Use hash-based activation
        data_hash = hash(str(data))
        value = (data_hash % 1000) / 1000.0
        return self._scalar_to_field(value)
    
    def calculate_pressure(self) -> float:
        """Calculate pressure from field dynamics."""
        divergence = self.get_divergence()
        gradient = self.get_flux_gradient()
        gradient_magnitude = np.sqrt(np.mean(gradient[0]**2 + gradient[1]**2))
        return abs(divergence) + gradient_magnitude * 0.5


class InformationField:
    """
    Represents structured potential and symbolic alignment.
    Manages information compression and structural regularity.
    """
    
    def __init__(self, shape: Tuple[int, int] = (32, 32)):
        self.shape = shape
        self.field = np.zeros(shape, dtype=np.float32)
        self.structure_history = []
        self.memory_alignment = 0.0
    
    def update(self, memory_field, energy_field: np.ndarray) -> np.ndarray:
        """Update information field based on memory structures and energy."""
        # Get structured information from memory
        # Handle both MemoryField and MemoryFieldTensor
        if hasattr(memory_field, 'items'):
            structures = dict(memory_field.items())
        else:
            # For MemoryFieldTensor, use empty structures for now
            structures = {}
        
        structure_contribution = self._structures_to_field(structures)
        
        # Compute alignment with energy field
        alignment = self._compute_alignment(energy_field)
        
        # Apply recursive refinement
        refined_field = self._apply_refinement(structure_contribution, alignment)
        
        # Update field with temporal integration
        integration_rate = 0.1
        self.field = (1 - integration_rate) * self.field + integration_rate * refined_field
        
        # Record structure evolution
        self.structure_history.append({
            'field': self.field.copy(),
            'alignment': alignment,
            'structure_count': len(structures),
            'timestamp': time.time()
        })
        
        # Keep history bounded
        if len(self.structure_history) > 100:
            self.structure_history.pop(0)
        
        return self.field
    
    def get_compression_gradient(self) -> float:
        """Compute information compression gradient: ∇·I(x)"""
        grad_x, grad_y = np.gradient(self.field)
        return np.mean(grad_x + grad_y)
    
    def get_regularity_index(self) -> float:
        """Measure structural regularity in information field."""
        # Use FFT to detect periodic structures
        fft = np.fft.fft2(self.field)
        power_spectrum = np.abs(fft)**2
        
        # Higher energy in low frequencies indicates more regularity
        low_freq_power = np.sum(power_spectrum[:self.shape[0]//4, :self.shape[1]//4])
        total_power = np.sum(power_spectrum)
        
        return low_freq_power / max(total_power, 1e-10)
    
    def _structures_to_field(self, structures: Dict[str, Any]) -> np.ndarray:
        """Convert memory structures to information field."""
        field = np.zeros(self.shape, dtype=np.float32)
        
        for key, data in structures.items():
            if 'coordinates' in data:
                x, y = data['coordinates']
                # Map coordinates to field indices
                row = int(abs(y) * self.shape[0] / 20) % self.shape[0]
                col = int(abs(x) * self.shape[1] / 20) % self.shape[1]
                
                # Add structure contribution
                strength = data.get('entropy_resolved', 0.1)
                field[row, col] += strength
        
        # Smooth the field
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(field, sigma=2.0)
    
    def _compute_alignment(self, energy_field: np.ndarray) -> float:
        """Compute alignment between information and energy fields."""
        # Normalized correlation
        e_norm = energy_field / (np.linalg.norm(energy_field) + 1e-10)
        i_norm = self.field / (np.linalg.norm(self.field) + 1e-10)
        
        alignment = np.sum(e_norm * i_norm)
        self.memory_alignment = alignment
        return alignment
    
    def _apply_refinement(self, structure_field: np.ndarray, alignment: float) -> np.ndarray:
        """Apply recursive refinement to information field."""
        # Strengthen areas with good alignment
        refinement_factor = 1.0 + 0.2 * alignment
        
        # Apply local enhancement
        refined = structure_field * refinement_factor
        
        # Add small random perturbations for exploration
        noise = np.random.normal(0, 0.01, self.shape)
        refined += noise
        
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
        
        # Statistics
        self.update_count = 0
        self.collapse_triggers = 0
        self.field_states = []
    
    def update_fields(self, input_data: Any, memory_field: MemoryField, 
                     context: ExecutionContext) -> FieldState:
        """Update all fields and compute current state."""
        self.update_count += 1
        
        # Update energy field
        energy_array = self.energy_field.update(input_data, context)
        
        # Update information field
        info_array = self.information_field.update(memory_field, energy_array)
        
        # Compute entropy tensor
        entropy_array = self.entropy_tensor.compute(energy_array, info_array)
        
        # Analyze field balance
        pressure = self.balance_controller.compute_balance(
            self.energy_field, self.information_field, self.entropy_tensor
        )
        
        # Compute derived metrics
        field_pressure = pressure.pressure_magnitude
        delta_entropy = np.mean(entropy_array)
        collapse_likelihood = min(field_pressure / max(self.balance_controller.collapse_threshold, 0.1), 1.0)
        potential_structures = len(pressure.critical_points)
        
        # Create field state
        field_state = FieldState(
            energy_field=energy_array,
            information_field=info_array,
            entropy_tensor=entropy_array,
            field_pressure=field_pressure,
            delta_entropy=delta_entropy,
            collapse_likelihood=collapse_likelihood,
            potential_structures=potential_structures,
            timestamp=time.time()
        )
        
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
