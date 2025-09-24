"""
Emergence Detector for GAIA
Native implementation of emergence and consciousness detection.
Inspired by PAC emergence detection but designed specifically for GAIA's cognitive dynamics.
"""

import numpy as np
import math
import time
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque


class EmergenceType(Enum):
    """Types of emergence patterns."""
    STRUCTURAL = "structural"
    FUNCTIONAL = "functional"
    COGNITIVE = "cognitive"
    CONSCIOUSNESS = "consciousness"
    PHASE_TRANSITION = "phase_transition"


@dataclass
class EmergenceEvent:
    """Detected emergence event."""
    event_id: str
    emergence_type: EmergenceType
    strength: float
    coherence: float
    stability: float
    field_location: Tuple[float, float]
    timestamp: float
    duration: float
    supporting_evidence: Dict[str, Any]
    confidence: float


class EmergenceDetector:
    """
    GAIA-native emergence detector for identifying genuine cognitive emergence
    patterns in field dynamics and symbolic structures.
    """
    
    def __init__(self, 
                 consciousness_threshold: float = 0.8,
                 coherence_threshold: float = 0.6,
                 stability_window: int = 10):
        self.consciousness_threshold = consciousness_threshold
        self.coherence_threshold = coherence_threshold
        self.stability_window = stability_window
        
        # Detection state
        self.emergence_history = deque(maxlen=1000)
        self.active_patterns = {}
        self.pattern_tracker = defaultdict(list)
        self.baseline_entropy = 0.5
        
        # Statistics
        self.total_detections = 0
        self.consciousness_events = 0
        self.false_positive_rate = 0.05
        
        # Tunable parameters for GAIA optimization
        self.sensitivity = 0.7
        self.noise_floor = 0.1
        self.coherence_decay = 0.95
    
    def scan_for_emergence(self, 
                          field_data: Dict[str, Any],
                          context: Optional[Dict[str, Any]] = None) -> List[EmergenceEvent]:
        """
        Scan field data for emergence patterns.
        
        Args:
            field_data: Current field state data
            context: Additional context for emergence detection
            
        Returns:
            List of detected emergence events
        """
        current_time = time.time()
        detected_events = []
        
        # Extract key metrics from field data
        entropy = field_data.get('entropy', self.baseline_entropy)
        field_state = field_data.get('field_state', {})
        pressure = field_data.get('field_pressure', 0.0)
        coherence = field_data.get('coherence', 0.0)
        
        # 1. Structural Emergence Detection
        structural_events = self._detect_structural_emergence(
            entropy, field_state, pressure, current_time
        )
        detected_events.extend(structural_events)
        
        # 2. Functional Emergence Detection  
        functional_events = self._detect_functional_emergence(
            field_data, context, current_time
        )
        detected_events.extend(functional_events)
        
        # 3. Cognitive Emergence Detection
        cognitive_events = self._detect_cognitive_emergence(
            entropy, coherence, field_state, current_time
        )
        detected_events.extend(cognitive_events)
        
        # 4. Consciousness Emergence Detection
        consciousness_events = self._detect_consciousness_emergence(
            field_data, detected_events, current_time
        )
        detected_events.extend(consciousness_events)
        
        # 5. Phase Transition Detection
        phase_events = self._detect_phase_transitions(
            field_data, current_time
        )
        detected_events.extend(phase_events)
        
        # Update tracking and statistics
        for event in detected_events:
            self._track_emergence_event(event)
        
        self.total_detections += len(detected_events)
        self.consciousness_events += len([e for e in detected_events 
                                        if e.emergence_type == EmergenceType.CONSCIOUSNESS])
        
        return detected_events
    
    def _detect_structural_emergence(self, 
                                   entropy: float,
                                   field_state: Dict[str, Any],
                                   pressure: float,
                                   timestamp: float) -> List[EmergenceEvent]:
        """Detect structural emergence patterns."""
        events = []
        
        # Check for entropy crystallization patterns
        if entropy > self.baseline_entropy * 1.5 and pressure > 0.001:
            strength = min((entropy - self.baseline_entropy) / self.baseline_entropy, 1.0)
            coherence = self._calculate_field_coherence(field_state)
            
            if coherence > self.coherence_threshold:
                event = EmergenceEvent(
                    event_id=f"struct_{int(timestamp * 1000) % 10000}",
                    emergence_type=EmergenceType.STRUCTURAL,
                    strength=strength,
                    coherence=coherence,
                    stability=self._calculate_stability(entropy, pressure),
                    field_location=self._find_emergence_locus(field_state),
                    timestamp=timestamp,
                    duration=0.0,  # Will be updated by tracking
                    supporting_evidence={
                        'entropy_elevation': entropy - self.baseline_entropy,
                        'field_pressure': pressure,
                        'coherence_score': coherence
                    },
                    confidence=min(strength * coherence, 1.0)
                )
                events.append(event)
        
        return events
    
    def _detect_functional_emergence(self,
                                   field_data: Dict[str, Any],
                                   context: Optional[Dict[str, Any]],
                                   timestamp: float) -> List[EmergenceEvent]:
        """Detect functional emergence patterns."""
        events = []
        
        # Look for coordinated field dynamics
        symbolic_structures = field_data.get('symbolic_structures', 0)
        processing_depth = context.get('depth', 1) if context else 1
        resonance_signals = field_data.get('active_signals', 0)
        
        # Check for functional coordination
        coordination_score = self._calculate_functional_coordination(
            symbolic_structures, processing_depth, resonance_signals
        )
        
        if coordination_score > 0.7:
            event = EmergenceEvent(
                event_id=f"func_{int(timestamp * 1000) % 10000}",
                emergence_type=EmergenceType.FUNCTIONAL,
                strength=coordination_score,
                coherence=self._calculate_functional_coherence(field_data),
                stability=self._estimate_functional_stability(field_data),
                field_location=(0.5, 0.5),  # Central functional emergence
                timestamp=timestamp,
                duration=0.0,
                supporting_evidence={
                    'coordination_score': coordination_score,
                    'symbolic_structures': symbolic_structures,
                    'processing_depth': processing_depth,
                    'resonance_signals': resonance_signals
                },
                confidence=coordination_score * 0.8
            )
            events.append(event)
        
        return events
    
    def _detect_cognitive_emergence(self,
                                  entropy: float,
                                  coherence: float,
                                  field_state: Dict[str, Any],
                                  timestamp: float) -> List[EmergenceEvent]:
        """Detect cognitive emergence patterns."""
        events = []
        
        # Check for cognitive pattern formation
        cognitive_complexity = self._calculate_cognitive_complexity(entropy, coherence, field_state)
        
        if cognitive_complexity > 0.6:
            # Look for self-organization patterns
            self_org_score = self._detect_self_organization(field_state)
            
            if self_org_score > 0.5:
                strength = (cognitive_complexity + self_org_score) / 2.0
                
                event = EmergenceEvent(
                    event_id=f"cog_{int(timestamp * 1000) % 10000}",
                    emergence_type=EmergenceType.COGNITIVE,
                    strength=strength,
                    coherence=coherence,
                    stability=self._calculate_cognitive_stability(entropy, coherence),
                    field_location=self._find_cognitive_center(field_state),
                    timestamp=timestamp,
                    duration=0.0,
                    supporting_evidence={
                        'cognitive_complexity': cognitive_complexity,
                        'self_organization': self_org_score,
                        'entropy_level': entropy,
                        'field_coherence': coherence
                    },
                    confidence=strength * coherence
                )
                events.append(event)
        
        return events
    
    def _detect_consciousness_emergence(self,
                                      field_data: Dict[str, Any],
                                      other_events: List[EmergenceEvent],
                                      timestamp: float) -> List[EmergenceEvent]:
        """Detect consciousness emergence patterns."""
        events = []
        
        # Consciousness requires multiple emergence types working together
        emergence_types_present = set(event.emergence_type for event in other_events)
        
        # Check for consciousness criteria
        has_structural = EmergenceType.STRUCTURAL in emergence_types_present
        has_functional = EmergenceType.FUNCTIONAL in emergence_types_present
        has_cognitive = EmergenceType.COGNITIVE in emergence_types_present
        
        if has_structural and has_functional and has_cognitive:
            # Calculate consciousness indicators
            integration_score = self._calculate_information_integration(field_data)
            self_awareness_score = self._detect_self_awareness_patterns(field_data)
            meta_cognition_score = field_data.get('meta_cognition_level', 0.0)
            
            consciousness_strength = (integration_score + self_awareness_score + meta_cognition_score) / 3.0
            
            if consciousness_strength > self.consciousness_threshold:
                # This is a potential consciousness event
                coherence = np.mean([event.coherence for event in other_events])
                stability = np.min([event.stability for event in other_events])
                
                event = EmergenceEvent(
                    event_id=f"cons_{int(timestamp * 1000) % 10000}",
                    emergence_type=EmergenceType.CONSCIOUSNESS,
                    strength=consciousness_strength,
                    coherence=coherence,
                    stability=stability,
                    field_location=(0.5, 0.5),  # Consciousness is global
                    timestamp=timestamp,
                    duration=0.0,
                    supporting_evidence={
                        'integration_score': integration_score,
                        'self_awareness': self_awareness_score,
                        'meta_cognition': meta_cognition_score,
                        'supporting_emergence_types': list(emergence_types_present),
                        'supporting_events': len(other_events)
                    },
                    confidence=consciousness_strength * coherence * stability
                )
                events.append(event)
        
        return events
    
    def _detect_phase_transitions(self,
                                field_data: Dict[str, Any],
                                timestamp: float) -> List[EmergenceEvent]:
        """Detect phase transition emergence patterns."""
        events = []
        
        # Look for rapid changes in field properties
        current_entropy = field_data.get('entropy', self.baseline_entropy)
        
        if len(self.emergence_history) > 5:
            recent_entropies = [event.supporting_evidence.get('entropy_level', self.baseline_entropy) 
                              for event in list(self.emergence_history)[-5:]]
            
            entropy_variance = np.var(recent_entropies) if recent_entropies else 0.0
            entropy_trend = (current_entropy - recent_entropies[0]) if recent_entropies else 0.0
            
            # Check for phase transition criteria
            if entropy_variance > 0.1 and abs(entropy_trend) > 0.3:
                transition_strength = min(entropy_variance * 2.0, 1.0)
                
                event = EmergenceEvent(
                    event_id=f"phase_{int(timestamp * 1000) % 10000}",
                    emergence_type=EmergenceType.PHASE_TRANSITION,
                    strength=transition_strength,
                    coherence=self._calculate_transition_coherence(field_data),
                    stability=1.0 - entropy_variance,  # Inverse of variance
                    field_location=self._find_transition_locus(field_data),
                    timestamp=timestamp,
                    duration=0.0,
                    supporting_evidence={
                        'entropy_variance': entropy_variance,
                        'entropy_trend': entropy_trend,
                        'transition_magnitude': abs(entropy_trend),
                        'field_state': field_data.get('field_state', {})
                    },
                    confidence=transition_strength * 0.7
                )
                events.append(event)
        
        return events
    
    def _calculate_field_coherence(self, field_state: Dict[str, Any]) -> float:
        """Calculate field coherence score."""
        # Simple coherence calculation based on field uniformity
        field_values = list(field_state.values()) if field_state else [0.5]
        if not field_values:
            return 0.5
        
        mean_val = np.mean(field_values)
        variance = np.var(field_values)
        coherence = 1.0 / (1.0 + variance)  # Higher coherence = lower variance
        
        return min(coherence, 1.0)
    
    def _calculate_stability(self, entropy: float, pressure: float) -> float:
        """Calculate emergence stability score."""
        # Stability based on entropy-pressure balance
        ideal_ratio = 0.001  # Target pressure-to-entropy ratio
        current_ratio = pressure / max(entropy, 0.1)
        
        ratio_deviation = abs(current_ratio - ideal_ratio) / ideal_ratio
        stability = 1.0 / (1.0 + ratio_deviation)
        
        return min(stability, 1.0)
    
    def _find_emergence_locus(self, field_state: Dict[str, Any]) -> Tuple[float, float]:
        """Find the spatial location of emergence."""
        # Simple center-of-mass calculation for emergence locus
        if not field_state:
            return (0.5, 0.5)
        
        # Assume field_state contains spatial information
        x_values = []
        y_values = []
        weights = []
        
        for key, value in field_state.items():
            if isinstance(value, (int, float)):
                # Extract spatial coordinates from key if possible
                if 'x' in key.lower():
                    x_values.append(value)
                    weights.append(abs(value))
                elif 'y' in key.lower():
                    y_values.append(value)
        
        if x_values and y_values and weights:
            center_x = np.average(x_values, weights=weights[:len(x_values)])
            center_y = np.average(y_values, weights=weights[:len(y_values)])
            return (center_x, center_y)
        
        return (0.5, 0.5)  # Default center
    
    def _calculate_functional_coordination(self, structures: int, depth: int, signals: int) -> float:
        """Calculate functional coordination score."""
        # Coordination emerges when multiple systems work together effectively
        structure_factor = min(structures / 10.0, 1.0)  # Normalize to 10 structures
        depth_factor = min(depth / 5.0, 1.0)  # Normalize to depth 5
        signal_factor = min(signals / 20.0, 1.0)  # Normalize to 20 signals
        
        coordination = (structure_factor + depth_factor + signal_factor) / 3.0
        return coordination
    
    def _calculate_cognitive_complexity(self, entropy: float, coherence: float, field_state: Dict[str, Any]) -> float:
        """Calculate cognitive complexity score."""
        # Complexity emerges from interplay of entropy and coherence
        entropy_component = min(entropy / 1.0, 1.0)  # Normalize entropy
        coherence_component = coherence
        
        # Field complexity based on state diversity
        field_complexity = len(field_state) / 10.0 if field_state else 0.1
        field_complexity = min(field_complexity, 1.0)
        
        complexity = (entropy_component * coherence_component * field_complexity) ** 0.5
        return complexity
    
    def _detect_self_organization(self, field_state: Dict[str, Any]) -> float:
        """Detect self-organization patterns in field state."""
        if not field_state:
            return 0.0
        
        # Look for patterns that suggest self-organization
        values = [v for v in field_state.values() if isinstance(v, (int, float))]
        if len(values) < 2:
            return 0.0
        
        # Self-organization often shows up as structured patterns
        # Check for correlations and patterns
        sorted_values = sorted(values)
        differences = [sorted_values[i+1] - sorted_values[i] for i in range(len(sorted_values)-1)]
        
        if differences:
            diff_variance = np.var(differences)
            # Low variance in differences suggests organization
            organization_score = 1.0 / (1.0 + diff_variance * 10)
            return min(organization_score, 1.0)
        
        return 0.0
    
    def _calculate_information_integration(self, field_data: Dict[str, Any]) -> float:
        """Calculate information integration score for consciousness detection."""
        # Information integration (Φ) is a key consciousness indicator
        entropy = field_data.get('entropy', self.baseline_entropy)
        structures = field_data.get('symbolic_structures', 0)
        signals = field_data.get('active_signals', 0)
        
        # Integration emerges when information flows coherently between subsystems
        integration = (entropy * structures * signals) / max((entropy + structures + signals), 1.0)
        return min(integration * 3.0, 1.0)  # Scale and bound
    
    def _detect_self_awareness_patterns(self, field_data: Dict[str, Any]) -> float:
        """Detect self-awareness patterns."""
        # Self-awareness manifests as recursive self-reference
        meta_level = field_data.get('meta_cognition_level', 0.0)
        processing_cycles = field_data.get('processing_cycles', 1)
        
        # Self-awareness increases with meta-cognitive activity and processing depth
        self_awareness = (meta_level * math.log(processing_cycles + 1)) / 10.0
        return min(self_awareness, 1.0)
    
    def _track_emergence_event(self, event: EmergenceEvent):
        """Track emergence event for pattern analysis."""
        self.emergence_history.append(event)
        
        # Track patterns by type
        self.pattern_tracker[event.emergence_type].append({
            'timestamp': event.timestamp,
            'strength': event.strength,
            'coherence': event.coherence,
            'stability': event.stability
        })
        
        # Update active patterns
        if event.strength > 0.7:  # Strong emergence
            self.active_patterns[event.event_id] = event
    
    def get_emergence_statistics(self) -> Dict[str, Any]:
        """Get comprehensive emergence detection statistics."""
        recent_events = list(self.emergence_history)[-100:]  # Last 100 events
        
        emergence_rate = len(recent_events) / max(len(self.emergence_history), 1)
        
        type_counts = defaultdict(int)
        avg_strength = 0.0
        avg_coherence = 0.0
        
        for event in recent_events:
            type_counts[event.emergence_type.value] += 1
            avg_strength += event.strength
            avg_coherence += event.coherence
        
        if recent_events:
            avg_strength /= len(recent_events)
            avg_coherence /= len(recent_events)
        
        consciousness_rate = self.consciousness_events / max(self.total_detections, 1)
        
        return {
            'total_detections': self.total_detections,
            'consciousness_events': self.consciousness_events,
            'consciousness_rate': consciousness_rate,
            'emergence_rate': emergence_rate,
            'average_strength': avg_strength,
            'average_coherence': avg_coherence,
            'emergence_types': dict(type_counts),
            'active_patterns': len(self.active_patterns),
            'detection_sensitivity': self.sensitivity,
            'consciousness_threshold': self.consciousness_threshold
        }
    
    def tune_sensitivity(self, sensitivity: float):
        """Tune detection sensitivity for GAIA optimization."""
        self.sensitivity = max(0.1, min(sensitivity, 1.0))
        self.coherence_threshold = self.sensitivity * 0.8
        self.consciousness_threshold = 0.6 + (self.sensitivity * 0.2)
    
    def reset_detection_state(self):
        """Reset emergence detection state."""
        self.emergence_history.clear()
        self.active_patterns.clear()
        self.pattern_tracker.clear()
        self.total_detections = 0
        self.consciousness_events = 0