"""
Collapse Core for GAIA
Implements entropy-driven collapse dynamics using fracton core modules.
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
    """
    
    def __init__(self, memory_field: MemoryField):
        self.memory_field = memory_field
        self.entropy_dispatcher = EntropyDispatcher()
        self.k_boltzmann = 1.380649e-23  # For thermodynamic calculations
        self.temperature = 300.0  # K
        
    def evaluate(self, context: ExecutionContext) -> CollapseVector:
        """
        Evaluate entropy field for collapse opportunities.
        Implements collapse criteria: ΔS(x) ≥ σ AND ∇²S(x) ≠ 0 AND |κ(x)| > κ_critical
        """
        # Get entropy field state
        entropy = context.entropy
        field_state = context.field_state or {}
        
        # Calculate entropy gradient and curvature
        entropy_gradient = self._compute_entropy_gradient(entropy, field_state)
        curvature = self._compute_information_curvature(entropy, field_state)
        
        # Compute collapse force: F_collapse(x) = -∇S_field(x)
        force_magnitude = np.linalg.norm(entropy_gradient)
        
        # Dynamic collapse threshold with pi-harmonic modulation
        depth_factor = context.depth if context.depth else 1
        sigma_threshold = 0.6 * (1 + 0.1 * math.sin(math.pi * depth_factor))
        
        # Critical curvature threshold
        kappa_critical = 0.3
        
        # Check collapse conditions
        entropy_tension = entropy
        meets_threshold = entropy_tension >= sigma_threshold
        has_curvature = abs(curvature) > 0.01  # Non-flat zone
        exceeds_critical = abs(curvature) > kappa_critical
        
        if meets_threshold and has_curvature and exceeds_critical:
            # Calculate collapse locus (simplified 2D coordinates)
            locus = self._find_collapse_locus(entropy_gradient, field_state)
            
            # Predict entropy gain
            expected_gain = self._predict_entropy_gain(entropy_tension, force_magnitude)
            
            # Determine collapse type based on entropy signature
            collapse_type = self._classify_collapse_type(entropy, curvature, force_magnitude)
            
            # Calculate thermodynamic cost
            energy_cost = self._calculate_collapse_cost(expected_gain)
            
            return CollapseVector(
                locus=locus,
                entropy_tension=entropy_tension,
                curvature=curvature,
                force_magnitude=force_magnitude,
                collapse_type=collapse_type,
                energy_cost=energy_cost,
                expected_gain=expected_gain
            )
        
        return None
    
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
        # Create stabilization snapshot
        snapshot = self.memory_field.create_snapshot()
        
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
                'stabilization_snapshot': snapshot.field_id
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
