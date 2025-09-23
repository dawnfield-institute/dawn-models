"""
Resonance Mesh for GAIA
Handles phase-aligned output, agentic signal emission, and resonance-based collapse.
See docs/architecture/modules/resonance_mesh.md for design details.
"""

import time
import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

from fracton.core.recursive_engine import ExecutionContext


class SignalType(Enum):
    """Types of signals in the resonance mesh."""
    PHASE_CORRECTION = "phase_correction"
    HARMONIC_SYNC = "harmonic_sync"
    RESONANCE_AMPLIFICATION = "resonance_amplification"
    INTERFERENCE_PATTERN = "interference_pattern"
    STANDING_WAVE = "standing_wave"


@dataclass
class AgenticSignal:
    """Represents an agentic signal in the mesh."""
    signal_id: str
    signal_type: SignalType
    frequency: float
    amplitude: float
    phase: float
    origin_coordinates: Tuple[float, float]
    target_coordinates: Optional[Tuple[float, float]]
    emission_time: float
    decay_rate: float
    payload: Dict[str, Any]


@dataclass
class ResonanceNode:
    """Node in the resonance mesh."""
    node_id: str
    coordinates: Tuple[float, float]
    resonance_frequency: float
    current_phase: float
    amplitude: float
    quality_factor: float  # Q-factor
    connected_nodes: List[str]
    last_update: float


@dataclass
class InterferencePattern:
    """Represents interference between signals."""
    pattern_id: str
    participating_signals: List[str]
    pattern_type: str  # "constructive", "destructive", "complex"
    center_coordinates: Tuple[float, float]
    pattern_strength: float
    creation_time: float


class PhaseAligner:
    """
    Manages phase alignment across the resonance mesh.
    """
    
    def __init__(self):
        self.reference_frequency = 1.0  # Base frequency
        self.phase_tolerance = 0.1  # Phase alignment tolerance
        self.alignment_history = deque(maxlen=1000)
    
    def align_phases(self, signals: List[AgenticSignal]) -> List[AgenticSignal]:
        """Align phases of multiple signals for coherent output."""
        if len(signals) < 2:
            return signals
        
        # Calculate target phase based on dominant signal
        dominant_signal = max(signals, key=lambda s: s.amplitude)
        target_phase = dominant_signal.phase
        target_frequency = dominant_signal.frequency
        
        aligned_signals = []
        for signal in signals:
            # Calculate phase correction
            phase_difference = target_phase - signal.phase
            
            # Normalize phase difference to [-π, π]
            while phase_difference > math.pi:
                phase_difference -= 2 * math.pi
            while phase_difference < -math.pi:
                phase_difference += 2 * math.pi
            
            # Apply phase correction if within tolerance
            if abs(phase_difference) > self.phase_tolerance:
                corrected_phase = signal.phase + phase_difference * 0.5  # Gradual alignment
                
                # Create aligned signal
                aligned_signal = AgenticSignal(
                    signal_id=f"{signal.signal_id}_aligned",
                    signal_type=signal.signal_type,
                    frequency=signal.frequency,
                    amplitude=signal.amplitude,
                    phase=corrected_phase,
                    origin_coordinates=signal.origin_coordinates,
                    target_coordinates=signal.target_coordinates,
                    emission_time=time.time(),
                    decay_rate=signal.decay_rate,
                    payload=signal.payload
                )
                aligned_signals.append(aligned_signal)
            else:
                aligned_signals.append(signal)
        
        # Record alignment operation
        self.alignment_history.append({
            'timestamp': time.time(),
            'signals_aligned': len(signals),
            'target_phase': target_phase,
            'average_correction': sum(abs(target_phase - s.phase) for s in signals) / len(signals)
        })
        
        return aligned_signals
    
    def calculate_phase_coherence(self, signals: List[AgenticSignal]) -> float:
        """Calculate overall phase coherence of signal group."""
        if len(signals) < 2:
            return 1.0
        
        phases = [signal.phase for signal in signals]
        
        # Calculate phase coherence using circular statistics
        # Convert to complex numbers and compute coherence
        complex_phases = [math.cos(phase) + 1j * math.sin(phase) for phase in phases]
        mean_complex = sum(complex_phases) / len(complex_phases)
        coherence = abs(mean_complex)
        
        return coherence


class SignalEmitter:
    """
    Emits agentic signals based on field conditions and collapse events.
    """
    
    def __init__(self):
        self.emission_counter = 0
        self.active_signals = {}
        self.emission_history = []
    
    def emit_signal(self, signal_type: SignalType, origin: Tuple[float, float],
                   context: ExecutionContext, payload: Dict[str, Any] = None) -> AgenticSignal:
        """Emit an agentic signal from the given origin."""
        self.emission_counter += 1
        signal_id = f"signal_{self.emission_counter}_{signal_type.value}"
        
        # Calculate signal parameters based on context
        frequency = self._calculate_frequency(signal_type, context)
        amplitude = self._calculate_amplitude(signal_type, context)
        phase = self._calculate_initial_phase(signal_type, context)
        decay_rate = self._calculate_decay_rate(signal_type)
        
        # Determine target if applicable
        target = self._determine_target(signal_type, origin, context)
        
        signal = AgenticSignal(
            signal_id=signal_id,
            signal_type=signal_type,
            frequency=frequency,
            amplitude=amplitude,
            phase=phase,
            origin_coordinates=origin,
            target_coordinates=target,
            emission_time=time.time(),
            decay_rate=decay_rate,
            payload=payload or {}
        )
        
        # Store active signal
        self.active_signals[signal_id] = signal
        
        # Record emission
        self.emission_history.append({
            'signal_id': signal_id,
            'signal_type': signal_type.value,
            'emission_time': signal.emission_time,
            'origin': origin,
            'context_entropy': context.entropy
        })
        
        return signal
    
    def update_signals(self):
        """Update all active signals with temporal decay."""
        current_time = time.time()
        expired_signals = []
        
        for signal_id, signal in self.active_signals.items():
            age = current_time - signal.emission_time
            decay_factor = math.exp(-age * signal.decay_rate)
            
            # Update amplitude with decay
            signal.amplitude *= decay_factor
            
            # Remove if too weak
            if signal.amplitude < 0.01:
                expired_signals.append(signal_id)
        
        # Remove expired signals
        for signal_id in expired_signals:
            del self.active_signals[signal_id]
    
    def _calculate_frequency(self, signal_type: SignalType, context: ExecutionContext) -> float:
        """Calculate signal frequency based on type and context."""
        base_frequency = 1.0
        
        if signal_type == SignalType.PHASE_CORRECTION:
            return base_frequency * (1.0 + context.entropy)
        elif signal_type == SignalType.HARMONIC_SYNC:
            return base_frequency * 2.0  # Harmonic
        elif signal_type == SignalType.RESONANCE_AMPLIFICATION:
            return base_frequency * (0.5 + context.entropy * 1.5)
        elif signal_type == SignalType.INTERFERENCE_PATTERN:
            return base_frequency * (1.0 + (context.depth or 0) * 0.1)
        else:  # STANDING_WAVE
            return base_frequency * 0.8
    
    def _calculate_amplitude(self, signal_type: SignalType, context: ExecutionContext) -> float:
        """Calculate signal amplitude based on type and context."""
        base_amplitude = 1.0
        
        entropy_factor = 0.5 + context.entropy * 0.5
        depth_factor = 1.0 + (context.depth or 0) * 0.05
        
        return base_amplitude * entropy_factor * depth_factor
    
    def _calculate_initial_phase(self, signal_type: SignalType, context: ExecutionContext) -> float:
        """Calculate initial phase for the signal."""
        # Use context entropy and depth to determine phase
        phase = (context.entropy * math.pi + (context.depth or 0) * 0.5) % (2 * math.pi)
        return phase
    
    def _calculate_decay_rate(self, signal_type: SignalType) -> float:
        """Calculate decay rate based on signal type."""
        if signal_type == SignalType.PHASE_CORRECTION:
            return 0.1  # Fast decay
        elif signal_type == SignalType.HARMONIC_SYNC:
            return 0.05  # Medium decay
        elif signal_type == SignalType.RESONANCE_AMPLIFICATION:
            return 0.02  # Slow decay
        elif signal_type == SignalType.STANDING_WAVE:
            return 0.01  # Very slow decay
        else:  # INTERFERENCE_PATTERN
            return 0.08  # Medium-fast decay
    
    def _determine_target(self, signal_type: SignalType, origin: Tuple[float, float],
                         context: ExecutionContext) -> Optional[Tuple[float, float]]:
        """Determine target coordinates for directional signals."""
        if signal_type in [SignalType.PHASE_CORRECTION, SignalType.INTERFERENCE_PATTERN]:
            # Create target based on entropy gradient
            angle = context.entropy * 2 * math.pi
            distance = 5.0 + context.entropy * 10.0
            
            target_x = origin[0] + distance * math.cos(angle)
            target_y = origin[1] + distance * math.sin(angle)
            
            return (target_x, target_y)
        
        return None  # Non-directional signals


class ResonanceMeshNetwork:
    """
    Network of resonance nodes that propagate and amplify signals.
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (16, 16)):
        self.grid_size = grid_size
        self.nodes = {}
        self.interference_patterns = {}
        self.standing_waves = []
        
        # Initialize mesh nodes
        self._initialize_mesh()
    
    def _initialize_mesh(self):
        """Initialize the resonance mesh network."""
        node_counter = 0
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                node_counter += 1
                node_id = f"node_{i}_{j}"
                
                # Calculate coordinates
                x = (i / self.grid_size[0]) * 20.0 - 10.0  # -10 to 10 range
                y = (j / self.grid_size[1]) * 20.0 - 10.0
                
                # Calculate resonance frequency based on position
                frequency = 1.0 + 0.1 * math.sin(i * 0.5) + 0.1 * math.cos(j * 0.5)
                
                # Create resonance node
                node = ResonanceNode(
                    node_id=node_id,
                    coordinates=(x, y),
                    resonance_frequency=frequency,
                    current_phase=0.0,
                    amplitude=0.5,
                    quality_factor=10.0,
                    connected_nodes=self._get_neighbors(i, j),
                    last_update=time.time()
                )
                
                self.nodes[node_id] = node
    
    def _get_neighbors(self, i: int, j: int) -> List[str]:
        """Get neighboring node IDs for mesh connectivity."""
        neighbors = []
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue  # Skip self
                
                ni, nj = i + di, j + dj
                if 0 <= ni < self.grid_size[0] and 0 <= nj < self.grid_size[1]:
                    neighbors.append(f"node_{ni}_{nj}")
        
        return neighbors
    
    def propagate_signal(self, signal: AgenticSignal):
        """Propagate signal through the resonance mesh."""
        # Find closest nodes to signal origin
        closest_nodes = self._find_closest_nodes(signal.origin_coordinates, radius=5.0)
        
        for node_id in closest_nodes:
            node = self.nodes[node_id]
            
            # Calculate distance-based attenuation
            distance = self._calculate_distance(signal.origin_coordinates, node.coordinates)
            attenuation = 1.0 / (1.0 + distance * 0.1)
            
            # Calculate resonance coupling
            frequency_match = self._calculate_frequency_coupling(signal.frequency, node.resonance_frequency)
            
            # Update node with signal
            node.amplitude += signal.amplitude * attenuation * frequency_match
            node.current_phase = (node.current_phase + signal.phase * frequency_match) % (2 * math.pi)
            node.last_update = time.time()
            
            # Propagate to connected nodes with decay
            self._propagate_to_neighbors(node, signal, attenuation * 0.8)
    
    def detect_interference_patterns(self, signals: List[AgenticSignal]) -> List[InterferencePattern]:
        """Detect interference patterns between signals."""
        patterns = []
        pattern_counter = len(self.interference_patterns)
        
        # Check all pairs of signals
        for i, signal1 in enumerate(signals):
            for j, signal2 in enumerate(signals[i+1:], i+1):
                interference = self._calculate_interference(signal1, signal2)
                
                if interference['strength'] > 0.3:  # Significant interference
                    pattern_counter += 1
                    pattern_id = f"interference_{pattern_counter}"
                    
                    pattern = InterferencePattern(
                        pattern_id=pattern_id,
                        participating_signals=[signal1.signal_id, signal2.signal_id],
                        pattern_type=interference['type'],
                        center_coordinates=interference['center'],
                        pattern_strength=interference['strength'],
                        creation_time=time.time()
                    )
                    
                    patterns.append(pattern)
                    self.interference_patterns[pattern_id] = pattern
        
        return patterns
    
    def _find_closest_nodes(self, coordinates: Tuple[float, float], radius: float) -> List[str]:
        """Find nodes within radius of given coordinates."""
        closest = []
        
        for node_id, node in self.nodes.items():
            distance = self._calculate_distance(coordinates, node.coordinates)
            if distance <= radius:
                closest.append(node_id)
        
        return closest
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between positions."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _calculate_frequency_coupling(self, signal_freq: float, node_freq: float) -> float:
        """Calculate resonance coupling between signal and node frequencies."""
        frequency_diff = abs(signal_freq - node_freq)
        coupling = 1.0 / (1.0 + frequency_diff * 2.0)  # Resonance coupling
        return coupling
    
    def _propagate_to_neighbors(self, source_node: ResonanceNode, signal: AgenticSignal, attenuation: float):
        """Propagate signal to neighboring nodes."""
        for neighbor_id in source_node.connected_nodes:
            if neighbor_id in self.nodes:
                neighbor = self.nodes[neighbor_id]
                
                # Calculate coupling and update neighbor
                coupling = self._calculate_frequency_coupling(signal.frequency, neighbor.resonance_frequency)
                
                neighbor.amplitude += source_node.amplitude * attenuation * coupling * 0.1
                neighbor.current_phase = (neighbor.current_phase + source_node.current_phase * coupling * 0.1) % (2 * math.pi)
                neighbor.last_update = time.time()
    
    def _calculate_interference(self, signal1: AgenticSignal, signal2: AgenticSignal) -> Dict[str, Any]:
        """Calculate interference between two signals."""
        # Calculate center point
        center_x = (signal1.origin_coordinates[0] + signal2.origin_coordinates[0]) / 2
        center_y = (signal1.origin_coordinates[1] + signal2.origin_coordinates[1]) / 2
        center = (center_x, center_y)
        
        # Calculate phase difference
        phase_diff = abs(signal1.phase - signal2.phase)
        while phase_diff > math.pi:
            phase_diff = 2 * math.pi - phase_diff
        
        # Calculate interference strength and type
        strength = min(signal1.amplitude, signal2.amplitude)
        
        if phase_diff < math.pi / 4:  # In phase
            interference_type = "constructive"
            strength *= 2.0
        elif phase_diff > 3 * math.pi / 4:  # Out of phase
            interference_type = "destructive"
            strength *= 0.5
        else:
            interference_type = "complex"
            strength *= 1.0
        
        return {
            'type': interference_type,
            'center': center,
            'strength': min(strength, 2.0),  # Cap strength
            'phase_difference': phase_diff
        }


class ResonanceMesh:
    """
    Main resonance mesh system coordinating phase alignment, signal emission, and interference.
    Handles phase-aligned output and agentic signal emission.
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (16, 16)):
        self.phase_aligner = PhaseAligner()
        self.signal_emitter = SignalEmitter()
        self.mesh_network = ResonanceMeshNetwork(grid_size)
        
        # Statistics
        self.total_signals_emitted = 0
        self.total_alignments_performed = 0
        self.interference_patterns_detected = 0
    
    def emit_agentic_signal(self, signal_type: SignalType, origin: Tuple[float, float],
                           context: ExecutionContext, payload: Dict[str, Any] = None) -> AgenticSignal:
        """Emit an agentic signal and propagate through mesh."""
        # Emit signal
        signal = self.signal_emitter.emit_signal(signal_type, origin, context, payload)
        self.total_signals_emitted += 1
        
        # Propagate through mesh network
        self.mesh_network.propagate_signal(signal)
        
        return signal
    
    def align_output_phases(self, signals: List[AgenticSignal]) -> List[AgenticSignal]:
        """Align phases for coherent output."""
        aligned_signals = self.phase_aligner.align_phases(signals)
        self.total_alignments_performed += 1
        
        return aligned_signals
    
    def detect_resonance_patterns(self) -> Dict[str, Any]:
        """Detect and analyze resonance patterns in the mesh."""
        active_signals = list(self.signal_emitter.active_signals.values())
        
        # Detect interference patterns
        interference_patterns = self.mesh_network.detect_interference_patterns(active_signals)
        self.interference_patterns_detected += len(interference_patterns)
        
        # Calculate mesh coherence
        mesh_coherence = self._calculate_mesh_coherence()
        
        # Calculate signal density
        signal_density = len(active_signals) / (self.mesh_network.grid_size[0] * self.mesh_network.grid_size[1])
        
        return {
            'active_signals': len(active_signals),
            'interference_patterns': len(interference_patterns),
            'mesh_coherence': mesh_coherence,
            'signal_density': signal_density,
            'total_nodes': len(self.mesh_network.nodes)
        }
    
    def update_mesh(self):
        """Update mesh state with signal propagation and decay."""
        # Update active signals
        self.signal_emitter.update_signals()
        
        # Apply temporal decay to mesh nodes
        current_time = time.time()
        for node in self.mesh_network.nodes.values():
            age = current_time - node.last_update
            decay = math.exp(-age * 0.1)  # Decay rate
            node.amplitude *= decay
    
    def get_resonance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive resonance mesh statistics."""
        # Calculate phase coherence
        active_signals = list(self.signal_emitter.active_signals.values())
        phase_coherence = self.phase_aligner.calculate_phase_coherence(active_signals)
        
        # Calculate average node amplitude
        avg_node_amplitude = 0.0
        if self.mesh_network.nodes:
            avg_node_amplitude = sum(node.amplitude for node in self.mesh_network.nodes.values()) / len(self.mesh_network.nodes)
        
        # Signal type distribution
        signal_type_counts = defaultdict(int)
        for signal in active_signals:
            signal_type_counts[signal.signal_type.value] += 1
        
        return {
            'total_signals_emitted': self.total_signals_emitted,
            'active_signals': len(active_signals),
            'total_alignments_performed': self.total_alignments_performed,
            'interference_patterns_detected': self.interference_patterns_detected,
            'phase_coherence': phase_coherence,
            'average_node_amplitude': avg_node_amplitude,
            'signal_type_distribution': dict(signal_type_counts),
            'mesh_size': self.mesh_network.grid_size
        }
    
    def _calculate_mesh_coherence(self) -> float:
        """Calculate overall coherence of the mesh network."""
        if not self.mesh_network.nodes:
            return 0.0
        
        # Calculate phase coherence across all nodes
        phases = [node.current_phase for node in self.mesh_network.nodes.values()]
        
        if not phases:
            return 0.0
        
        # Use circular statistics for phase coherence
        complex_phases = [math.cos(phase) + 1j * math.sin(phase) for phase in phases]
        mean_complex = sum(complex_phases) / len(complex_phases)
        coherence = abs(mean_complex)
        
        return coherence
    
    def reset(self):
        """Reset resonance mesh to initial state."""
        self.phase_aligner = PhaseAligner()
        self.signal_emitter = SignalEmitter()
        self.mesh_network = ResonanceMeshNetwork(self.mesh_network.grid_size)
        
        self.total_signals_emitted = 0
        self.total_alignments_performed = 0
        self.interference_patterns_detected = 0
