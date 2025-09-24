"""
Meta-Cognition Layer for GAIA
Provides meta-cognitive ancestry tracking, epistemic repair, and collapse front visualization.
Enhanced with native GAIA consciousness emergence detection and pattern amplification.
See docs/architecture/modules/meta_cognition_layer.md for design details.
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque

from fracton.core.recursive_engine import ExecutionContext
from fracton.core.memory_field import MemoryField

# Import native GAIA enhancement components
from .conservation_engine import ConservationEngine, ConservationMode
from .emergence_detector import EmergenceDetector, EmergenceType
from .pattern_amplifier import PatternAmplifier, AmplificationMode


@dataclass
class CognitiveTrace:
    """Represents a trace of cognitive operations."""
    trace_id: str
    operation_type: str
    timestamp: float
    context_snapshot: Dict[str, Any]
    entropy_before: float
    entropy_after: float
    structures_created: List[str]
    parent_trace_id: Optional[str]
    depth: int


@dataclass
class EpistemicRepair:
    """Represents an epistemic repair operation."""
    repair_id: str
    target_structure: str
    inconsistency_type: str
    repair_action: str
    confidence: float
    timestamp: float


class MetaCognitionLayer:
    """
    Provides meta-cognitive oversight and epistemic integrity monitoring.
    Tracks cognitive ancestry and performs epistemic repairs.
    """
    
    def __init__(self):
        self.cognitive_traces = {}
        self.ancestry_graph = defaultdict(list)
        self.epistemic_repairs = []
        self.collapse_front_data = deque(maxlen=1000)
        
        # Initialize native GAIA consciousness detection components
        self.emergence_detector = EmergenceDetector(
            consciousness_threshold=0.85,  # High threshold for meta-cognition
            coherence_threshold=0.7
        )
        self.pattern_amplifier = PatternAmplifier(
            max_amplification=1.5,  # Conservative amplification for meta-cognition
            energy_budget=0.5
        )
        self.conservation_engine = ConservationEngine(
            mode=ConservationMode.FULL_THERMODYNAMIC,  # Strictest conservation for meta-cognition
            tolerance=0.05
        )
        print("Native GAIA-enhanced meta-cognition initialized with consciousness detection")
        
        # Statistics
        self.total_traces = 0
        self.total_repairs = 0
        self.integrity_score = 1.0
    
    def track_operation(self, operation_type: str, context: ExecutionContext,
                       entropy_before: float, entropy_after: float,
                       structures_created: List[str] = None) -> str:
        """Track a cognitive operation and add to ancestry."""
        self.total_traces += 1
        trace_id = f"trace_{self.total_traces}_{int(time.time())}"
        
        # Find parent trace
        parent_trace_id = context.trace_id if hasattr(context, 'trace_id') else None
        
        trace = CognitiveTrace(
            trace_id=trace_id,
            operation_type=operation_type,
            timestamp=time.time(),
            context_snapshot={
                'entropy': context.entropy,
                'depth': context.depth or 0,
                'field_state': context.field_state or {}
            },
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            structures_created=structures_created or [],
            parent_trace_id=parent_trace_id,
            depth=context.depth or 0
        )
        
        self.cognitive_traces[trace_id] = trace
        
        # Update ancestry graph
        if parent_trace_id:
            self.ancestry_graph[parent_trace_id].append(trace_id)
        
        # Update collapse front
        self._update_collapse_front(trace)
        
        return trace_id
    
    def perform_epistemic_repair(self, target_structure: str, 
                                inconsistency_type: str) -> EpistemicRepair:
        """Perform epistemic repair on inconsistent structures."""
        self.total_repairs += 1
        repair_id = f"repair_{self.total_repairs}"
        
        # Determine repair action based on inconsistency type
        if inconsistency_type == "entropy_divergence":
            repair_action = "entropy_normalization"
            confidence = 0.8
        elif inconsistency_type == "structural_contradiction":
            repair_action = "structure_reconciliation"
            confidence = 0.7
        elif inconsistency_type == "temporal_inconsistency":
            repair_action = "temporal_alignment"
            confidence = 0.6
        else:
            repair_action = "general_stabilization"
            confidence = 0.5
        
        repair = EpistemicRepair(
            repair_id=repair_id,
            target_structure=target_structure,
            inconsistency_type=inconsistency_type,
            repair_action=repair_action,
            confidence=confidence,
            timestamp=time.time()
        )
        
        self.epistemic_repairs.append(repair)
        
        # Update integrity score
        self._update_integrity_score()
        
        return repair
    
    def get_ancestry_lineage(self, trace_id: str) -> List[str]:
        """Get complete ancestry lineage for a trace."""
        lineage = []
        current_id = trace_id
        
        while current_id and current_id in self.cognitive_traces:
            lineage.append(current_id)
            trace = self.cognitive_traces[current_id]
            current_id = trace.parent_trace_id
        
        return lineage[::-1]  # Return in chronological order
    
    def visualize_collapse_front(self) -> Dict[str, Any]:
        """Provide collapse front visualization data."""
        if not self.collapse_front_data:
            return {'status': 'no_data'}
        
        # Analyze recent collapse activity
        recent_collapses = list(self.collapse_front_data)[-50:]  # Last 50 operations
        
        # Calculate front metrics
        entropy_trend = self._calculate_entropy_trend(recent_collapses)
        operation_frequency = self._calculate_operation_frequency(recent_collapses)
        depth_distribution = self._calculate_depth_distribution(recent_collapses)
        
        return {
            'status': 'active',
            'recent_operations': len(recent_collapses),
            'entropy_trend': entropy_trend,
            'operation_frequency': operation_frequency,
            'depth_distribution': depth_distribution,
            'integrity_score': self.integrity_score
        }
    
    def get_meta_statistics(self) -> Dict[str, Any]:
        """Get comprehensive meta-cognition statistics."""
        # Calculate average entropy change
        avg_entropy_change = 0.0
        if self.cognitive_traces:
            entropy_changes = [
                trace.entropy_after - trace.entropy_before 
                for trace in self.cognitive_traces.values()
            ]
            avg_entropy_change = sum(entropy_changes) / len(entropy_changes)
        
        # Calculate operation type distribution
        operation_counts = defaultdict(int)
        for trace in self.cognitive_traces.values():
            operation_counts[trace.operation_type] += 1
        
        # Calculate repair success rate
        recent_repairs = [r for r in self.epistemic_repairs if time.time() - r.timestamp < 300]
        avg_repair_confidence = 0.0
        if recent_repairs:
            avg_repair_confidence = sum(r.confidence for r in recent_repairs) / len(recent_repairs)
        
        return {
            'total_traces': self.total_traces,
            'total_repairs': self.total_repairs,
            'integrity_score': self.integrity_score,
            'average_entropy_change': avg_entropy_change,
            'operation_type_distribution': dict(operation_counts),
            'recent_repair_confidence': avg_repair_confidence,
            'ancestry_depth': self._calculate_max_ancestry_depth()
        }
    
    def _update_collapse_front(self, trace: CognitiveTrace):
        """Update collapse front data with new trace."""
        front_data = {
            'timestamp': trace.timestamp,
            'operation_type': trace.operation_type,
            'entropy_change': trace.entropy_after - trace.entropy_before,
            'depth': trace.depth,
            'structures_count': len(trace.structures_created)
        }
        
        self.collapse_front_data.append(front_data)
    
    def _update_integrity_score(self):
        """Update overall epistemic integrity score."""
        if not self.epistemic_repairs:
            self.integrity_score = 1.0
            return
        
        # Recent repairs with high confidence boost integrity
        recent_repairs = [r for r in self.epistemic_repairs if time.time() - r.timestamp < 600]
        
        if recent_repairs:
            avg_confidence = sum(r.confidence for r in recent_repairs) / len(recent_repairs)
            repair_frequency = len(recent_repairs) / 600.0  # repairs per second
            
            # High confidence and low frequency = high integrity
            frequency_penalty = min(repair_frequency * 100, 0.5)  # Cap penalty
            self.integrity_score = max(0.1, avg_confidence - frequency_penalty)
        else:
            # No recent repairs - integrity gradually recovers
            self.integrity_score = min(1.0, self.integrity_score * 1.01)
    
    def _calculate_entropy_trend(self, collapses: List[Dict]) -> str:
        """Calculate overall entropy trend."""
        if len(collapses) < 2:
            return "stable"
        
        entropy_changes = [c['entropy_change'] for c in collapses]
        recent_avg = sum(entropy_changes[-10:]) / min(len(entropy_changes), 10)
        
        if recent_avg > 0.05:
            return "increasing"
        elif recent_avg < -0.05:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_operation_frequency(self, collapses: List[Dict]) -> Dict[str, int]:
        """Calculate frequency of different operation types."""
        frequency = defaultdict(int)
        for collapse in collapses:
            frequency[collapse['operation_type']] += 1
        return dict(frequency)
    
    def _calculate_depth_distribution(self, collapses: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of operation depths."""
        depths = [c['depth'] for c in collapses]
        
        # Group into ranges
        distribution = {
            'shallow (0-2)': sum(1 for d in depths if 0 <= d <= 2),
            'medium (3-5)': sum(1 for d in depths if 3 <= d <= 5),
            'deep (6+)': sum(1 for d in depths if d >= 6)
        }
        
        return distribution
    
    def _calculate_max_ancestry_depth(self) -> int:
        """Calculate maximum ancestry depth."""
        max_depth = 0
        
        for trace_id in self.cognitive_traces:
            lineage = self.get_ancestry_lineage(trace_id)
            max_depth = max(max_depth, len(lineage))
        
        return max_depth
    
    def get_meta_cognitive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive meta-cognitive statistics."""
        # Calculate max trace depth
        max_depth = 0
        for trace_id in self.cognitive_traces:
            lineage = self.get_ancestry_lineage(trace_id) if hasattr(self, 'get_ancestry_lineage') else []
            max_depth = max(max_depth, len(lineage))
        
        return {
            'integrity_score': self.integrity_score,
            'total_traces': self.total_traces,
            'total_repairs': self.total_repairs,
            'active_traces': len(self.cognitive_traces),
            'ancestry_branches': len(self.ancestry_graph),
            'epistemic_repairs': len(self.epistemic_repairs),
            'collapse_front_size': len(self.collapse_front_data),
            'max_trace_depth': max_depth
        }
    
    def track_cognitive_operation(self, operation: Dict[str, Any]):
        """Track a cognitive operation for meta-analysis."""
        trace_id = f"trace_{self.total_traces}_{time.time()}"
        
        trace = CognitiveTrace(
            trace_id=trace_id,
            operation_type=operation.get('type', 'unknown'),
            timestamp=time.time(),
            context_snapshot=operation.get('context', {}),
            entropy_before=operation.get('entropy_before', 0.5),
            entropy_after=operation.get('entropy_after', 0.5),
            structures_created=operation.get('structures', []),
            parent_trace_id=operation.get('parent_id', None),
            depth=operation.get('depth', 1)
        )
        
        self.cognitive_traces[trace_id] = trace
        self.total_traces += 1
        
        # Update ancestry graph if there's a parent
        if trace.parent_trace_id:
            self.ancestry_graph[trace.parent_trace_id].append(trace_id)
    
    def calculate_cognitive_integrity(self) -> float:
        """Calculate current cognitive integrity score."""
        if not self.cognitive_traces:
            return 1.0
        
        # Base integrity on repair rate and trace consistency
        repair_ratio = len(self.epistemic_repairs) / max(len(self.cognitive_traces), 1)
        trace_depth_consistency = min(1.0, self.total_traces / 100.0)
        
        # Lower integrity if too many repairs needed
        integrity = 1.0 - (repair_ratio * 0.3) + (trace_depth_consistency * 0.1)
        
        # Update stored integrity score
        self.integrity_score = max(0.1, min(1.0, integrity))
        return self.integrity_score
    
    def detect_epistemic_inconsistencies(self, structures: Optional[List[Any]] = None) -> List[str]:
        """Detect inconsistencies in cognitive structures."""
        inconsistencies = []
        
        # Use provided structures or default to empty list
        if structures is None:
            structures = []
        
        # Simple heuristic-based inconsistency detection
        if len(structures) > 10:
            inconsistencies.append("structure_overload")
        
        # Check for rapid changes indicating instability
        if len(self.cognitive_traces) > 50:
            recent_traces = list(self.cognitive_traces.values())[-10:]
            entropy_changes = [abs(t.entropy_after - t.entropy_before) for t in recent_traces]
            if sum(entropy_changes) > 5.0:
                inconsistencies.append("entropy_instability")
        
        return inconsistencies
    
    def reset(self):
        """Reset meta-cognition layer to initial state."""
        self.cognitive_traces.clear()
        self.ancestry_graph.clear()
        self.epistemic_repairs.clear()
        self.collapse_front_data.clear()
        
        self.total_traces = 0
        self.total_repairs = 0
        self.integrity_score = 1.0
