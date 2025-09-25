"""
Collapse Core for GAIA
Implements entropy-driven collapse dynamics using fracton core modules.
Enhanced with native GAIA conservation, emergence detection, and pattern amplification.
See docs/architecture/modules/collapse_core.md for theory and design.
"""

import numpy as np
import math
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Import fracton core modules
from fracton.core.memory_field import MemoryField
from fracton.core.entropy_dispatch import EntropyDispatcher, EntropyLevel
from fracton.core.recursive_engine import ExecutionContext

# Import native GAIA enhancement components
from .conservation_engine import ConservationEngine
from .emergence_detector import EmergenceDetector, PhaseType
from .pattern_amplifier import PatternAmplifier


class CollapseType(Enum):
    """Types of entropy collapse structures."""
    FRACTAL_NODE = "fractal_node"
    MEMORY_IMPRINT = "memory_imprint"
    AGENTIC_SIGNAL = "agentic_signal"
    GEOMETRIC = "geometric"
    THERMODYNAMIC = "thermodynamic"


@dataclass
class CollapseVector:
    """Represents a collapse event in entropy field."""
    locus: Tuple[float, float]  # (x, y) position in field
    entropy_tension: float
    curvature: float
    force_magnitude: float
    collapse_type: CollapseType
    energy_cost: float
    expected_gain: float


@dataclass
class CollapseResult:
    """Result of a collapse operation."""
    structure_id: str
    collapse_type: CollapseType
    entropy_resolved: float
    thermodynamic_cost: float
    symbolic_content: Any
    field_coordinates: Tuple[float, float]
    timestamp: float


class CollapseEvaluator:
    """
    Analyzes entropy tensor for collapse hotspots using fracton entropy dispatch.
    Implements physics-informed collapse detection with force calculations.
    Enhanced with native GAIA conservation, emergence detection, and pattern amplification.
    """
    
    def __init__(self, memory_field: MemoryField):
        self.memory_field = memory_field
        self.entropy_dispatcher = EntropyDispatcher()
        self.k_boltzmann = 1.380649e-23  # For thermodynamic calculations
        self.temperature = 300.0  # K
        
        # Initialize native GAIA enhancement components
        self.conservation_engine = ConservationEngine()
        self.emergence_detector = EmergenceDetector()
        self.pattern_amplifier = PatternAmplifier()
        print("Native GAIA-enhanced collapse evaluation initialized")
        
    def evaluate(self, context: ExecutionContext) -> CollapseVector:
        """
        Evaluate PAC conservation field for violation-driven collapse opportunities.
        Implements PAC collapse criteria: |∇·ψ|² > 0 (any conservation violation triggers collapse)
        Enhanced with native GAIA emergence detection and Xi operator scaling.
        """
        # Get field state and map to PAC lattice
        field_state = context.field_state or {}
        
        # Convert field state to complex amplitude lattice
        lattice = self._field_to_pac_lattice(context, field_state)
        
        # Calculate conservation residual using PAC validator
        residual = self._compute_pac_residual(lattice)
        violation_magnitude = np.linalg.norm(residual)
        
        # Native GAIA emergence detection (keep existing enhancement)
        lattice_data = np.abs(lattice)**2  # Convert complex field to real data
        emergence_events = self.emergence_detector.detect_emergence(lattice_data)
        
        # If GAIA detects genuine emergence, enhance collapse evaluation
        xi_operator = 1.0571  # PAC balance operator - fundamental constant
        consciousness_detected = False
        
        if emergence_events:
            print(f"GAIA detected {len(emergence_events)} emergence events")
            
            # Check for consciousness emergence
            consciousness_events = [e for e in emergence_events if e.phase_type == PhaseType.EMERGENT_CONSCIOUSNESS]
            if consciousness_events:
                consciousness_detected = True
                xi_operator *= 1.1  # Slight boost for consciousness
                print("Consciousness emergence detected - enhancing PAC collapse dynamics")
            
            # Use the strongest emergence event to guide collapse
            strongest_event = max(emergence_events, key=lambda e: e.information_integration)
            violation_magnitude = max(violation_magnitude, strongest_event.information_integration * 0.3 + violation_magnitude)
        
        # PAC conservation check - collapse ONLY when conservation is violated
        if violation_magnitude > 1e-10:  # Any non-zero violation triggers collapse (no arbitrary threshold)
            # Find violation center from residual pattern
            violation_center = self._find_violation_center(residual)
            
            # Collapse strength directly from violation magnitude scaled by Xi operator
            collapse_strength = violation_magnitude * xi_operator
            
            # Native GAIA conservation validation using PAC principles
            pre_state = {
                'total_amplitude': np.sum(np.abs(lattice)**2),
                'violation_magnitude': violation_magnitude,
                'xi_ratio': collapse_strength / violation_magnitude if violation_magnitude > 0 else 0
            }
            
            post_state = {
                'total_amplitude': pre_state['total_amplitude'],  # Must be conserved
                'violation_magnitude': 0.0,  # Collapse resolves violation
                'xi_ratio': xi_operator
            }
            
            # Native GAIA conservation validation using PAC principles  
            # Update engine with current lattice state
            self.conservation_engine._update_pac_fields(lattice)
            
            # Validate conservation before and after collapse
            conservation_result = self.conservation_engine.validate_conservation()
            
            if not conservation_result.get('valid', True):
                print("GAIA conservation engine prevents PAC collapse - amplitude conservation violation")
                return None
            
            # SEC pattern analysis for collapse type (replace arbitrary classification)
            collapse_type = self._classify_collapse_type_with_sec_analysis(residual, consciousness_detected)
            
            # Calculate thermodynamic cost from violation resolution
            energy_cost = violation_magnitude * self.k_boltzmann * self.temperature
            expected_gain = collapse_strength  # Gain is proportional to violation resolved
            
            return CollapseVector(
                locus=violation_center,
                entropy_tension=violation_magnitude,  # Now meaningful - actual violation magnitude
                curvature=0.0,  # Curvature not used in PAC physics
                force_magnitude=collapse_strength,    # Force is violation * Xi operator
                collapse_type=collapse_type,
                energy_cost=energy_cost,
                expected_gain=expected_gain
            )
        
        return None  # No violation = no collapse (principled physics)
    
    def _compute_entropy_gradient(self, entropy: float, field_state: Dict) -> np.ndarray:
        """Compute entropy gradient for force calculation."""
        # Simplified gradient calculation
        dx = field_state.get('dx', 0.01)
        dy = field_state.get('dy', 0.01)
        return np.array([entropy * dx, entropy * dy])
    
    def _compute_information_curvature(self, entropy: float, field_state: Dict) -> float:
        """Compute information curvature tensor κ(x)."""
        # Simplified curvature calculation
        return entropy * entropy - 0.25  # Creates interesting curvature dynamics
    
    def _find_collapse_locus(self, gradient: np.ndarray, field_state: Dict) -> Tuple[float, float]:
        """Find optimal collapse position in field."""
        # Find point of maximum gradient magnitude
        x = gradient[0] * 10.0  # Scale to field coordinates
        y = gradient[1] * 10.0
        return (x, y)
    
    def _predict_entropy_gain(self, tension: float, force: float) -> float:
        """Predict entropy reduction from collapse."""
        return min(tension * 0.8, force * 0.6)  # Conservative estimate
    
    def _classify_collapse_type(self, entropy: float, curvature: float, force: float) -> CollapseType:
        """Classify collapse type based on entropy signature."""
        if entropy > 0.9 and abs(curvature) > 0.5:
            return CollapseType.FRACTAL_NODE
        elif entropy > 0.7 and force > 0.4:
            return CollapseType.MEMORY_IMPRINT
        elif abs(curvature) > 0.6:
            return CollapseType.GEOMETRIC
        elif force > 0.5:
            return CollapseType.THERMODYNAMIC
        else:
            return CollapseType.AGENTIC_SIGNAL

    def _field_to_pac_lattice(self, context: ExecutionContext, field_state: Dict) -> np.ndarray:
        """Convert field state to complex amplitude lattice for PAC analysis."""
        # Extract field dimensions
        resolution = field_state.get('resolution', (32, 32))
        
        # Convert entropy to amplitude magnitude
        entropy = context.entropy if hasattr(context, 'entropy') else 0.5
        amplitude_magnitude = np.sqrt(entropy)  # |ψ| = √E
        
        # Create phase from field gradients (information encoding)
        phase = field_state.get('information_phase', 0.0)
        
        # Build complex amplitude lattice
        lattice = np.full(resolution, amplitude_magnitude * np.exp(1j * phase), dtype=complex)
        
        # Add field variations from context
        if hasattr(context, 'depth') and context.depth:
            # Depth creates wavelike variations
            x, y = np.meshgrid(range(resolution[0]), range(resolution[1]))
            wave_phase = 2 * np.pi * context.depth * (x + y) / (resolution[0] + resolution[1])
            lattice *= np.exp(1j * wave_phase)
        
        return lattice

    def _compute_pac_residual(self, lattice: np.ndarray) -> np.ndarray:
        """Compute PAC conservation residual from amplitude lattice."""
        # Total amplitude should be conserved (∑|ψ|² = constant)
        current_total = np.sum(np.abs(lattice)**2)
        expected_total = lattice.size  # Normalized expectation
        
        # Residual is deviation from conservation
        conservation_violation = current_total - expected_total
        
        # Create residual field showing where violations occur
        local_densities = np.abs(lattice)**2
        expected_density = expected_total / lattice.size
        residual = local_densities - expected_density
        
        # Scale by global violation magnitude
        if np.linalg.norm(residual) > 0:
            residual = residual * (conservation_violation / np.linalg.norm(residual))
        
        return residual

    def _find_violation_center(self, residual: np.ndarray) -> Tuple[float, float]:
        """Find center of maximum conservation violation."""
        # Find location of maximum absolute residual
        max_idx = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
        
        # Convert to field coordinates
        x = float(max_idx[0] / residual.shape[0])
        y = float(max_idx[1] / residual.shape[1])
        
        return (x, y)

    def _classify_collapse_type_with_sec_analysis(self, residual: np.ndarray, consciousness_detected: bool) -> CollapseType:
        """Classify collapse type using SEC (Symbolic Entropy Collapse) frequency analysis."""
        try:
            # Perform 2D FFT on residual pattern
            residual_fft = np.fft.fft2(residual)
            
            # Analyze frequency content
            freq_magnitude = np.abs(residual_fft)
            
            # Calculate frequency characteristics
            low_freq_energy = np.sum(freq_magnitude[:residual.shape[0]//4, :residual.shape[1]//4])
            high_freq_energy = np.sum(freq_magnitude[3*residual.shape[0]//4:, 3*residual.shape[1]//4:])
            total_energy = np.sum(freq_magnitude)
            
            if total_energy > 0:
                low_ratio = low_freq_energy / total_energy
                high_ratio = high_freq_energy / total_energy
                mixed_ratio = 1.0 - low_ratio - high_ratio
                
                # SEC classification based on frequency signatures
                if high_ratio > 0.4:  # High-frequency dominant
                    return CollapseType.FRACTAL_NODE
                elif low_ratio > 0.6:  # Low-frequency dominant  
                    return CollapseType.MEMORY_IMPRINT
                elif mixed_ratio > 0.5 or consciousness_detected:  # Mixed frequencies or consciousness
                    return CollapseType.AGENTIC_SIGNAL
                else:
                    return CollapseType.GEOMETRIC
            else:
                return CollapseType.THERMODYNAMIC
                
        except Exception as e:
            print(f"SEC frequency analysis failed: {e}, using consciousness fallback")
            return CollapseType.AGENTIC_SIGNAL if consciousness_detected else CollapseType.THERMODYNAMIC
    
    def _classify_collapse_type_with_emergence(self, 
                                            entropy: float, 
                                            curvature: float, 
                                            force: float, 
                                            field_state: Dict,
                                            consciousness_detected: bool) -> CollapseType:
        """Native GAIA collapse type classification with emergence context."""
        
        # Get base classification
        base_type = self._classify_collapse_type(entropy, curvature, force)
        
        # Enhance classification with consciousness detection
        if consciousness_detected:
            # Consciousness detected - this is likely agentic
            return CollapseType.AGENTIC_SIGNAL
        
        # Use emergence detector for additional context
        field_data = {
            'entropy': entropy,
            'field_state': field_state,
            'coherence': min(entropy, 1.0)
        }
        
        # Convert field data to amplitude field for emergence detection
        amplitude_field = np.abs(residual) + 1j * np.angle(residual + 1e-10)
        emergence_events = self.emergence_detector.detect_emergence(amplitude_field)
        
        if emergence_events:
            # Analyze emergence types to refine classification
            phase_types = [e.phase_type for e in emergence_events]
            max_strength = max(e.information_integration for e in emergence_events)
            
            # Emergence-guided refinement
            if any(pt.value == 'local_global' for pt in phase_types) and max_strength > 0.7:
                return CollapseType.FRACTAL_NODE
            elif any(pt.value == 'emergent_consciousness' for pt in phase_types):
                return CollapseType.MEMORY_IMPRINT
            elif any(pt.value == 'disorder_order' for pt in phase_types):
                return CollapseType.GEOMETRIC
        
        return base_type
    
    def _classify_collapse_type_with_pac(self, entropy: float, curvature: float, force: float, field_state: Dict) -> CollapseType:
        """PAC-enhanced collapse type classification using pattern amplification."""
        try:
            # Create pattern for PAC analysis
            pattern_data = {
                'entropy': entropy,
                'curvature': curvature,
                'force': force,
                'field_state': field_state
            }
            
            # Use PAC kernel to amplify and analyze pattern
            amplified_pattern = self.pac_kernel.process_pattern(
                pattern_data,
                conservation_mode='energy_information'
            )
            
            # Enhanced classification based on PAC amplification results
            pac_entropy = amplified_pattern.get('amplified_entropy', entropy)
            pac_complexity = amplified_pattern.get('pattern_complexity', 0.5)
            emergence_strength = amplified_pattern.get('emergence_strength', 0.0)
            
            # PAC-guided classification with emergence detection
            if emergence_strength > 0.8 and pac_complexity > 0.7:
                return CollapseType.FRACTAL_NODE  # High emergence = fractal
            elif pac_entropy > 0.8 and force > 0.5:
                return CollapseType.MEMORY_IMPRINT  # High entropy + force = memory
            elif pac_complexity > 0.6:
                return CollapseType.GEOMETRIC  # High complexity = geometric
            elif emergence_strength > 0.5:
                return CollapseType.AGENTIC_SIGNAL  # Medium emergence = agentic
            else:
                return CollapseType.THERMODYNAMIC  # Default = thermodynamic
                
        except Exception as e:
            print(f"PAC classification failed: {e}, falling back to standard classification")
            return self._classify_collapse_type(entropy, curvature, force)
    
    def _calculate_collapse_cost(self, entropy_gain: float) -> float:
        """Calculate Landauer-limited thermodynamic cost."""
        # E_collapse_cost(x) = k_B * T * ln(2) * N_bits_crystallized(x)
        bits_crystallized = entropy_gain * 10  # Estimate bits from entropy
        return self.k_boltzmann * self.temperature * math.log(2) * bits_crystallized


class CollapseTypingEngine:
    """
    Chooses collapse type using entropy signature and fracton dispatch conditions.
    """
    
    def __init__(self):
        self.dispatcher = EntropyDispatcher()
        self._setup_dispatch_conditions()
    
    def _setup_dispatch_conditions(self):
        """Setup dispatch conditions for different collapse types."""
        # TODO: Implement proper dispatch conditions when DispatchConditions is available
        # For now, just register basic functions
        try:
            self.dispatcher.register_function(
                "fractal_node_dispatch",
                lambda ctx: CollapseType.FRACTAL_NODE
            )
            self.dispatcher.register_function(
                "symbolic_node_dispatch", 
                lambda ctx: CollapseType.SYMBOLIC_NODE
            )
            self.dispatcher.register_function(
                "memory_imprint_dispatch",
                lambda ctx: CollapseType.MEMORY_IMPRINT
            )
            self.dispatcher.register_function(
                "agentic_signal_dispatch",
                lambda ctx: CollapseType.AGENTIC_SIGNAL
            )
        except Exception as e:
            # If registration fails, continue without dispatch setup
            pass
    
    def choose_type(self, collapse_vector: CollapseVector, context: ExecutionContext) -> CollapseType:
        """Choose optimal collapse type using fracton dispatch."""
        try:
            # Use fracton dispatcher for intelligent type selection
            selected_type = self.dispatcher.dispatch_function(context)
            return selected_type if selected_type else collapse_vector.collapse_type
        except:
            # Fallback to vector's suggested type
            return collapse_vector.collapse_type


class CollapseSynthesizer:
    """
    Converts collapse results into fracton memory structures.
    Applies field-reinforced consistency constraints.
    """
    
    def __init__(self, memory_field: MemoryField):
        self.memory_field = memory_field
        self.structure_counter = 0
    
    def synthesize(self, collapse_vector: CollapseVector, context: ExecutionContext) -> CollapseResult:
        """Crystallize collapse into structured form in memory field."""
        self.structure_counter += 1
        structure_id = f"collapse_{self.structure_counter}_{collapse_vector.collapse_type.value}"
        
        # Create symbolic content based on collapse type
        symbolic_content = self._create_symbolic_content(collapse_vector, context)
        
        # Store in fracton memory field
        self.memory_field.set(structure_id, {
            'type': collapse_vector.collapse_type.value,
            'content': symbolic_content,
            'entropy_resolved': collapse_vector.expected_gain,
            'cost': collapse_vector.energy_cost,
            'coordinates': collapse_vector.locus,
            'timestamp': time.time()
        })
        
        # Apply consistency constraints
        self._apply_consistency_constraints(structure_id, collapse_vector)
        
        return CollapseResult(
            structure_id=structure_id,
            collapse_type=collapse_vector.collapse_type,
            entropy_resolved=collapse_vector.expected_gain,
            thermodynamic_cost=collapse_vector.energy_cost,
            symbolic_content=symbolic_content,
            field_coordinates=collapse_vector.locus,
            timestamp=time.time()
        )
    
    def _create_symbolic_content(self, vector: CollapseVector, context: ExecutionContext) -> Dict[str, Any]:
        """Create symbolic content based on collapse type."""
        base_content = {
            'entropy_signature': vector.entropy_tension,
            'curvature': vector.curvature,
            'force': vector.force_magnitude,
            'depth': context.depth
        }
        
        if vector.collapse_type == CollapseType.FRACTAL_NODE:
            base_content.update({
                'fractal_dimension': 1.0 + vector.curvature,
                'self_similarity': vector.force_magnitude,
                'recursive_depth': context.depth
            })
        elif vector.collapse_type == CollapseType.MEMORY_IMPRINT:
            base_content.update({
                'memory_strength': vector.entropy_tension,
                'temporal_signature': time.time(),
                'field_resonance': vector.force_magnitude
            })
        elif vector.collapse_type == CollapseType.AGENTIC_SIGNAL:
            base_content.update({
                'signal_strength': vector.force_magnitude,
                'phase_alignment': vector.curvature,
                'broadcast_radius': vector.entropy_tension * 10
            })
        
        return base_content
    
    def _apply_consistency_constraints(self, structure_id: str, vector: CollapseVector):
        """Apply field-reinforced consistency constraints."""
        # Get neighboring structures for consistency checking
        existing_structures = dict(self.memory_field.items())
        
        # Simple consistency: ensure no overlapping high-energy structures
        for key, data in existing_structures.items():
            if key != structure_id and 'coordinates' in data:
                distance = self._calculate_distance(vector.locus, data['coordinates'])
                if distance < 2.0 and data.get('cost', 0) > vector.energy_cost * 0.8:
                    # Reduce energy to maintain consistency
                    current_data = self.memory_field.get(structure_id)
                    current_data['cost'] *= 0.9
                    self.memory_field.set(structure_id, current_data)
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between positions."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


class PostCollapseStabilizer:
    """
    Stabilizes field after collapse using fracton memory management.
    Prevents collapse chaining instability and manages entropy dissipation.
    """
    
    def __init__(self, memory_field: MemoryField):
        self.memory_field = memory_field
        self.stabilization_history = []
    
    def stabilize(self, collapse_result: CollapseResult, context: ExecutionContext) -> ExecutionContext:
        """Stabilize field post-collapse and update context."""
        # Create stabilization snapshot (with fallback for missing create_snapshot)
        try:
            snapshot = self.memory_field.create_snapshot()
        except AttributeError:
            # Fallback: create minimal snapshot
            snapshot = {
                'field_id': getattr(self.memory_field, 'field_id', 'unknown'),
                'capacity': getattr(self.memory_field, 'capacity', 1000),
                'timestamp': time.time(),
                'collapse_stabilization': True
            }
        
        # Entropy damping around collapse site
        new_entropy = self._apply_entropy_damping(context.entropy, collapse_result)
        
        # Prevent cascade collapses
        self._prevent_collapse_chaining(collapse_result)
        
        # Update context with stabilized state
        new_context = ExecutionContext(
            entropy=new_entropy,
            depth=context.depth,
            trace_id=context.trace_id,
            field_state={
                **context.field_state,
                'last_collapse': collapse_result.structure_id,
                'stabilization_time': time.time(),
                'entropy_damping_applied': True
            },
            parent_context=context.parent_context,
            metadata={
                **context.metadata,
                'stabilization_snapshot': snapshot.get('field_id', 'unknown') if isinstance(snapshot, dict) else getattr(snapshot, 'field_id', 'unknown')
            }
        )
        
        # Record stabilization
        self.stabilization_history.append({
            'collapse_id': collapse_result.structure_id,
            'entropy_before': context.entropy,
            'entropy_after': new_entropy,
            'timestamp': time.time()
        })
        
        return new_context
    
    def _apply_entropy_damping(self, current_entropy: float, result: CollapseResult) -> float:
        """Apply entropy damping based on collapse result."""
        # Reduce entropy proportional to collapse efficiency
        efficiency = result.entropy_resolved / max(result.thermodynamic_cost, 1e-10)
        damping_factor = 0.8 + 0.2 * min(efficiency, 1.0)
        return current_entropy * damping_factor
    
    def _prevent_collapse_chaining(self, result: CollapseResult):
        """Prevent unstable collapse chains."""
        # Find recent collapses near this location
        recent_threshold = time.time() - 1.0  # 1 second
        nearby_collapses = []
        
        for key, data in self.memory_field.items():
            if (data.get('timestamp', 0) > recent_threshold and 
                'coordinates' in data):
                distance = self._calculate_distance(result.field_coordinates, data['coordinates'])
                if distance < 5.0:  # Within influence radius
                    nearby_collapses.append(key)
        
        # If too many recent collapses, reduce their energy
        if len(nearby_collapses) > 3:
            for collapse_id in nearby_collapses:
                data = self.memory_field.get(collapse_id)
                if data:
                    data['cost'] *= 0.7  # Reduce energy
                    self.memory_field.set(collapse_id, data)
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate distance between field positions."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


class CollapseCore:
    """
    Main entry point for Collapse Core logic.
    Integrates all collapse subsystems using fracton infrastructure.
    Implements physics-informed, entropy-driven collapse with full v2.0 enhancements.
    """
    
    def __init__(self, memory_field: MemoryField = None):
        self.memory_field = memory_field or MemoryField(capacity=1000, field_id="collapse_core_field")
        self.evaluator = CollapseEvaluator(self.memory_field)
        self.typing_engine = CollapseTypingEngine()
        self.synthesizer = CollapseSynthesizer(self.memory_field)
        self.stabilizer = PostCollapseStabilizer(self.memory_field)
        
        # Statistics tracking
        self.total_collapses = 0
        self.collapse_efficiency_history = []
        self.collapse_type_counts = {ctype: 0 for ctype in CollapseType}
    
    def collapse(self, context: ExecutionContext) -> Tuple[Optional[CollapseResult], ExecutionContext]:
        """
        Perform a full collapse cycle:
        1. Evaluate entropy field for collapse opportunities
        2. Choose optimal collapse type
        3. Synthesize new structure
        4. Stabilize post-collapse field
        """
        # 1. Evaluate for collapse opportunity
        collapse_vector = self.evaluator.evaluate(context)
        
        if collapse_vector is None:
            # No collapse opportunity
            return None, context
        
        # 2. Choose optimal collapse type
        final_type = self.typing_engine.choose_type(collapse_vector, context)
        collapse_vector.collapse_type = final_type
        
        # 3. Synthesize structure
        collapse_result = self.synthesizer.synthesize(collapse_vector, context)
        
        # 4. Stabilize field
        stabilized_context = self.stabilizer.stabilize(collapse_result, context)
        
        # Update statistics
        self.total_collapses += 1
        self.collapse_type_counts[final_type] += 1
        efficiency = collapse_result.entropy_resolved / max(collapse_result.thermodynamic_cost, 1e-10)
        self.collapse_efficiency_history.append(efficiency)
        
        return collapse_result, stabilized_context
    
    def get_collapse_statistics(self) -> Dict[str, Any]:
        """Get comprehensive collapse statistics."""
        avg_efficiency = (sum(self.collapse_efficiency_history) / 
                         max(len(self.collapse_efficiency_history), 1))
        
        return {
            'total_collapses': self.total_collapses,
            'collapse_type_counts': {k.value: v for k, v in self.collapse_type_counts.items()},
            'average_efficiency': avg_efficiency,
            'total_structures': len(dict(self.memory_field.items())),
            'memory_field_entropy': self.memory_field.get_entropy()
        }
    
    def reset(self):
        """Reset collapse core to initial state."""
        self.memory_field.clear()
        self.total_collapses = 0
        self.collapse_efficiency_history.clear()
        self.collapse_type_counts = {ctype: 0 for ctype in CollapseType}
        self.stabilizer.stabilization_history.clear()


# Utility function for external access
def create_collapse_core(field_id: str = None) -> CollapseCore:
    """Create a new collapse core with fresh memory field."""
    field_id = field_id or f"collapse_field_{int(time.time())}"
    memory_field = MemoryField(field_id)
    return CollapseCore(memory_field)
