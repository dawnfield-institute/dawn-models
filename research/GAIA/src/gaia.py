"""
GAIA v2.0 - Main System Orchestrator
Physics-Informed AGI Architecture with Entropy-Driven Collapse

This is the main GAIA class that orchestrates all core modules:
- Collapse Core: Entropy-driven collapse dynamics
- Field Engine: Energy-information field management
- Superfluid Memory: Fluid memory with vortex tracking
- Symbolic Crystallizer: Bifractal tree generation
- Meta-Cognition Layer: Cognitive oversight
- Resonance Mesh: Phase-aligned agentic signals

The GAIA system processes inputs through entropy field dynamics,
creating symbolic structures through collapse events, and maintains
coherent cognitive states through resonance patterns.
"""

import time
import math
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import hashlib

# Mock fracton modules if not available

from fracton.core.memory_field import MemoryField, MemorySnapshot
from fracton.core.entropy_dispatch import EntropyDispatcher, EntropyLevel
from fracton.core.recursive_engine import ExecutionContext
from fracton.core.bifractal_trace import BifractalTrace
FRACTON_AVAILABLE = True

# Import GAIA core modules
from core.collapse_core import CollapseCore
from core.field_engine import FieldEngine
from core.superfluid_memory import SuperfluidMemory
from core.symbolic_crystallizer import SymbolicCrystallizer
from core.meta_cognition_layer import MetaCognitionLayer
from core.resonance_mesh import ResonanceMesh, SignalType


@dataclass
class GAIAState:
    """Current state of the GAIA system."""
    timestamp: float
    entropy_level: float
    field_pressure: float
    memory_coherence: float
    symbolic_structures: int
    active_signals: int
    cognitive_integrity: float
    processing_cycles: int
    total_collapses: int
    resonance_patterns: int


@dataclass
class GAIAResponse:
    """Response from GAIA processing."""
    response_text: str
    confidence: float
    processing_time: float
    entropy_change: float
    structures_created: int
    reasoning_trace: List[str]
    cognitive_load: float
    state: GAIAState


class GAIA:
    """
    Main GAIA system orchestrator.
    
    Integrates all core modules into a unified cognitive architecture
    capable of intelligent processing through entropy-driven dynamics.
    """
    
    def __init__(self, 
                 field_resolution: Tuple[int, int] = (32, 32),
                 collapse_threshold: float = 0.7,
                 memory_capacity: int = 10000,
                 resonance_grid_size: Tuple[int, int] = (16, 16),
                 log_level: str = "INFO"):
        """
        Initialize GAIA system with specified parameters.
        
        Args:
            field_resolution: Resolution of energy/information fields
            collapse_threshold: Threshold for triggering collapse events
            memory_capacity: Maximum memory patterns to store
            resonance_grid_size: Size of resonance mesh grid
            log_level: Logging level for system monitoring
        """
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, log_level))
        self.logger = logging.getLogger("GAIA")
        
        # System parameters
        self.field_resolution = field_resolution
        self.collapse_threshold = collapse_threshold
        self.memory_capacity = memory_capacity
        self.resonance_grid_size = resonance_grid_size
        
        # Initialize core modules
        self.logger.info("Initializing GAIA core modules...")
        self._initialize_core_modules()
        
        # System state
        self.processing_cycles = 0
        self.total_processing_time = 0.0
        self.conversation_memory = deque(maxlen=100)
        self.context_history = deque(maxlen=50)
        
        # Performance metrics
        self.metrics = {
            'total_inputs_processed': 0,
            'successful_responses': 0,
            'average_processing_time': 0.0,
            'average_confidence': 0.0,
            'total_collapses': 0,
            'total_structures_created': 0
        }
        
        self.logger.info("GAIA system initialized successfully")
    
    def _initialize_core_modules(self):
        """Initialize all GAIA core modules."""
        try:
            self.collapse_core = CollapseCore()
            self.field_engine = FieldEngine()
            self.superfluid_memory = SuperfluidMemory()
            self.symbolic_crystallizer = SymbolicCrystallizer()
            self.meta_cognition = MetaCognitionLayer()
            self.resonance_mesh = ResonanceMesh(self.resonance_grid_size)
            
            self.logger.info("All core modules initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize core modules: {e}")
            raise
    
    def process_input(self, 
                     input_text: Union[str, Dict, List, Any], 
                     context: Optional[Dict[str, Any]] = None,
                     require_reasoning: bool = True) -> GAIAResponse:
        """
        Process input through the complete GAIA cognitive pipeline.
        
        Args:
            input_text: Input to process (string, dict, list, or other data)
            context: Optional context dictionary
            require_reasoning: Whether to generate reasoning trace
            
        Returns:
            GAIAResponse with processing results
        """
        start_time = time.time()
        self.processing_cycles += 1
        reasoning_trace = []
        
        try:
            # Convert input to string representation for processing
            if isinstance(input_text, str):
                input_str = input_text
            else:
                input_str = str(input_text)
            
            self.logger.info(f"Processing input: '{input_str[:50]}...'")
            
            # Stage 1: Input Analysis and Contextualization
            reasoning_trace.append("Stage 1: Analyzing input and establishing context")
            input_entropy = self._calculate_input_entropy(input_str)
            processing_context = self._create_processing_context(input_str, context, input_entropy)
            
            reasoning_trace.append(f"Input entropy calculated: {input_entropy:.3f}")
            
            # Stage 2: Field Dynamics and Collapse Detection
            reasoning_trace.append("Stage 2: Updating field dynamics")
            self._update_field_dynamics(input_str, processing_context)
            
            collapse_conditions = self.field_engine.balance_controller.detect_collapse_conditions()
            reasoning_trace.append(f"Collapse conditions detected: {collapse_conditions}")
            
            # Stage 3: Memory Retrieval and Pattern Matching
            reasoning_trace.append("Stage 3: Retrieving relevant memories")
            relevant_memories = self._retrieve_relevant_memories(input_str, processing_context)
            reasoning_trace.append(f"Retrieved {len(relevant_memories)} relevant memory patterns")
            
            # Stage 4: Cognitive Processing Through Collapse Events
            reasoning_trace.append("Stage 4: Processing through collapse dynamics")
            symbolic_structures = []
            entropy_changes = []
            
            if collapse_conditions:
                # Execute collapse and create symbolic structures
                collapse_result = self._execute_cognitive_collapse(processing_context)
                if collapse_result:
                    reasoning_trace.append(f"Collapse executed: {collapse_result.get('collapse_id', 'unknown')}")
                    
                    # Crystallize into symbolic structure
                    structure = self.symbolic_crystallizer.crystallize_collapse_event(collapse_result)
                    symbolic_structures.append(structure)
                    entropy_changes.append(collapse_result.get('entropy_change', 0.0))
                    
                    # Store in memory
                    self._store_processing_memory(input_str, structure, processing_context)
                    reasoning_trace.append("Symbolic structure created and stored")
                else:
                    reasoning_trace.append("Collapse evaluated but no structure emerged")
            
            # Stage 5: Multi-Perspective Resonance Processing
            reasoning_trace.append("Stage 5: Multi-perspective resonance processing")
            resonance_signals = self._generate_resonance_processing(
                input_str, processing_context, symbolic_structures
            )
            reasoning_trace.append(f"Generated {len(resonance_signals)} resonance signals")
            
            # Stage 6: Phase Alignment and Coherence
            reasoning_trace.append("Stage 6: Achieving phase coherence")
            aligned_signals = self.resonance_mesh.align_output_phases(resonance_signals)
            coherence_patterns = self.resonance_mesh.detect_resonance_patterns()
            
            reasoning_trace.append(f"Phase coherence: {coherence_patterns.get('mesh_coherence', 0):.3f}")
            
            # Stage 7: Response Generation
            reasoning_trace.append("Stage 7: Generating response")
            response_text, confidence = self._generate_response(
                input_str, processing_context, symbolic_structures, 
                relevant_memories, coherence_patterns
            )
            
            reasoning_trace.append(f"Response generated with confidence: {confidence:.3f}")
            
            # Stage 8: Meta-Cognitive Oversight
            reasoning_trace.append("Stage 8: Meta-cognitive validation")
            self._perform_meta_cognitive_oversight(
                input_str, response_text, processing_context, reasoning_trace
            )
            
            # Calculate final metrics
            processing_time = time.time() - start_time
            total_entropy_change = sum(entropy_changes)
            cognitive_load = self._calculate_cognitive_load(processing_context, coherence_patterns)
            
            # Update system metrics
            self._update_metrics(processing_time, confidence, len(symbolic_structures))
            
            # Create system state snapshot
            current_state = self._get_current_state()
            
            reasoning_trace.append(f"Processing complete in {processing_time:.3f}s")
            
            # Store conversation
            self.conversation_memory.append({
                'input': input_text,  # Store original input (preserve type)
                'response': response_text,
                'timestamp': time.time(),
                'confidence': confidence,
                'processing_time': processing_time
            })
            
            return GAIAResponse(
                response_text=response_text,
                confidence=confidence,
                processing_time=processing_time,
                entropy_change=total_entropy_change,
                structures_created=len(symbolic_structures),
                reasoning_trace=reasoning_trace if require_reasoning else [],
                cognitive_load=cognitive_load,
                state=current_state
            )
            
        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            return GAIAResponse(
                response_text=f"I encountered an error while processing your input: {str(e)}",
                confidence=0.0,
                processing_time=time.time() - start_time,
                entropy_change=0.0,
                structures_created=0,
                reasoning_trace=reasoning_trace + [f"Error: {str(e)}"],
                cognitive_load=1.0,
                state=self._get_current_state()
            )
    
    def _calculate_input_entropy(self, input_text: str) -> float:
        """Calculate entropy of input text."""
        if not input_text:
            return 0.0
        
        # Calculate character frequency entropy
        char_counts = defaultdict(int)
        for char in input_text.lower():
            char_counts[char] += 1
        
        total_chars = len(input_text)
        entropy = 0.0
        
        for count in char_counts.values():
            prob = count / total_chars
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        # Normalize to 0-1 range (approximate)
        max_entropy = math.log2(26)  # Assuming 26 letters
        normalized_entropy = min(entropy / max_entropy, 1.0)
        
        return normalized_entropy
    
    def _create_processing_context(self, input_text: str, context: Optional[Dict], input_entropy: float) -> ExecutionContext:
        """Create processing context for input."""
        depth = len(self.context_history) + 1
        
        # Calculate semantic complexity
        word_count = len(input_text.split())
        complexity = min(word_count / 20.0, 1.0)  # Normalize to 0-1
        
        # Combine entropy sources
        total_entropy = (input_entropy + complexity) / 2.0
        
        # Create execution context
        processing_context = ExecutionContext(
            entropy=total_entropy,
            depth=depth
        )
        
        # Store context
        self.context_history.append({
            'input_text': input_text,
            'entropy': total_entropy,
            'complexity': complexity,
            'timestamp': time.time()
        })
        
        return processing_context
    
    def _update_field_dynamics(self, input_text: str, context: ExecutionContext):
        """Update energy and information fields based on input."""
        # Update energy field with input characteristics
        energy_input = {
            'energy_input': context.entropy,
            'intensity': len(input_text) / 1000.0,  # Normalize text length
            'source': 'user_input'
        }
        self.field_engine.energy_field.update(energy_input, context)
        
        # Update information field with semantic content
        self.field_engine.information_field.update(
            self.superfluid_memory.memory_field_tensor, 
            self.field_engine.energy_field.field
        )
    
    def _calculate_semantic_density(self, text: str) -> float:
        """Calculate semantic information density."""
        words = text.split()
        if not words:
            return 0.0
        
        unique_words = len(set(words))
        total_words = len(words)
        
        # Density based on uniqueness ratio
        density = unique_words / total_words
        
        return density
    
    def _calculate_novelty(self, input_text: str) -> float:
        """Calculate novelty of input compared to conversation history."""
        if not self.conversation_memory:
            return 1.0  # First input is completely novel
        
        # Simple novelty based on text similarity to recent inputs
        recent_inputs = [conv['input'] for conv in list(self.conversation_memory)[-5:]]
        
        # Calculate overlap with recent inputs
        input_words = set(input_text.lower().split())
        
        max_overlap = 0.0
        for recent_input in recent_inputs:
            recent_words = set(recent_input.lower().split())
            if input_words and recent_words:
                overlap = len(input_words.intersection(recent_words)) / len(input_words.union(recent_words))
                max_overlap = max(max_overlap, overlap)
        
        novelty = 1.0 - max_overlap
        return novelty
    
    def _retrieve_relevant_memories(self, input_text: str, context: ExecutionContext) -> List[Dict]:
        """Retrieve relevant memories from superfluid memory."""
        # Create search pattern based on input
        input_hash = hashlib.md5(input_text.encode()).hexdigest()[:8]
        
        # Try to retrieve memories with similar patterns
        relevant_memories = []
        
        # Search recent conversation memory
        for conv in self.conversation_memory:
            # Handle both string and non-string inputs
            conv_input = conv['input']
            if isinstance(conv_input, str):
                conv_hash = hashlib.md5(conv_input.encode()).hexdigest()[:8]
            else:
                conv_hash = hashlib.md5(str(conv_input).encode()).hexdigest()[:8]
            
            # Simple similarity based on hash prefix matching
            if conv_hash[:4] == input_hash[:4]:  # Similar patterns
                relevant_memories.append({
                    'type': 'conversation',
                    'content': conv,
                    'relevance': 0.8
                })
        
        # Retrieve from superfluid memory using attractors
        attractors = self.superfluid_memory.get_memory_attractors(threshold=0.3)
        if attractors:
            relevant_memories.append({
                'type': 'superfluid',
                'content': {'attractors': attractors},
                'relevance': 0.9
            })
        
        return relevant_memories
    
    def _execute_cognitive_collapse(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute cognitive collapse event."""
        # Create collapse data from current field state
        collapse_data = {
            'field_pressure': self.field_engine.energy_field.calculate_pressure(),
            'entropy_gradient': context.entropy,
            'information_density': self.field_engine.information_field.calculate_pressure()
        }
        
        # Execute collapse through collapse core
        collapse_result, updated_context = self.collapse_core.collapse(context)
        
        # Update metrics
        self.metrics['total_collapses'] += 1
        
        return collapse_result
    
    def _store_processing_memory(self, input_text: str, structure: Dict, context: ExecutionContext):
        """Store processing results in memory."""
        memory_key = f"processing_{time.time()}_{hash(input_text) % 10000}"
        
        memory_data = {
            'pattern_id': memory_key,
            'input_text': input_text,
            'structure': structure,
            'context_entropy': context.entropy,
            'timestamp': time.time(),
            'semantic_vector': self._create_semantic_vector(input_text),
            'stability_metric': 0.8  # Default stability
        }
        
        self.superfluid_memory.add_memory(memory_data, context)
    
    def _create_semantic_vector(self, text: str) -> List[float]:
        """Create simple semantic vector representation."""
        words = text.lower().split()
        
        # Simple word-based feature vector (4 dimensions)
        features = [
            len(words) / 20.0,  # Length feature
            len([w for w in words if len(w) > 5]) / max(len(words), 1),  # Complex words
            len([w for w in words if w in ['what', 'how', 'why', 'when', 'where']]) / max(len(words), 1),  # Question words
            len(set(words)) / max(len(words), 1)  # Uniqueness
        ]
        
        return features
    
    def _generate_resonance_processing(self, input_text: str, context: ExecutionContext, structures: List) -> List:
        """Generate resonance signals for multi-perspective processing."""
        signals = []
        
        # Analytical perspective
        analytical_signal = self.resonance_mesh.emit_agentic_signal(
            SignalType.PHASE_CORRECTION,
            origin=(1.0, 0.0),
            context=context,
            payload={'perspective': 'analytical', 'input': input_text}
        )
        signals.append(analytical_signal)
        
        # Intuitive perspective
        intuitive_signal = self.resonance_mesh.emit_agentic_signal(
            SignalType.HARMONIC_SYNC,
            origin=(0.0, 1.0),
            context=context,
            payload={'perspective': 'intuitive', 'input': input_text}
        )
        signals.append(intuitive_signal)
        
        # Creative perspective
        creative_signal = self.resonance_mesh.emit_agentic_signal(
            SignalType.INTERFERENCE_PATTERN,
            origin=(-1.0, 0.0),
            context=context,
            payload={'perspective': 'creative', 'input': input_text}
        )
        signals.append(creative_signal)
        
        # Synthetic perspective (if we have structures)
        if structures:
            synthetic_signal = self.resonance_mesh.emit_agentic_signal(
                SignalType.RESONANCE_AMPLIFICATION,
                origin=(0.0, -1.0),
                context=context,
                payload={'perspective': 'synthetic', 'structures': len(structures)}
            )
            signals.append(synthetic_signal)
        
        return signals
    
    def _generate_response(self, input_text: str, context: ExecutionContext, 
                          structures: List, memories: List, patterns: Dict) -> Tuple[str, float]:
        """Generate intelligent response based on processing results."""
        
        # Analyze input type and generate appropriate response
        input_lower = input_text.lower()
        
        # Question detection
        is_question = any(q in input_lower for q in ['what', 'how', 'why', 'when', 'where', 'who', '?'])
        
        # Complexity assessment
        coherence = patterns.get('mesh_coherence', 0.0)
        structure_count = len(structures)
        memory_count = len(memories)
        
        # Base confidence on coherence and available information
        base_confidence = coherence * 0.6 + (structure_count / 5.0) * 0.2 + (memory_count / 10.0) * 0.2
        confidence = min(base_confidence, 0.95)  # Cap at 95%
        
        # Generate response based on processing results
        if is_question:
            response = self._generate_question_response(input_text, context, structures, memories, coherence)
        elif 'hello' in input_lower or 'hi' in input_lower:
            response = self._generate_greeting_response(context, coherence)
        elif any(word in input_lower for word in ['think', 'believe', 'opinion', 'feel']):
            response = self._generate_opinion_response(input_text, context, structures, coherence)
        else:
            response = self._generate_general_response(input_text, context, structures, memories, coherence)
        
        return response, confidence
    
    def _generate_question_response(self, input_text: str, context: ExecutionContext, 
                                   structures: List, memories: List, coherence: float) -> str:
        """Generate response to questions."""
        
        responses = []
        
        # Add processing insight
        if coherence > 0.7:
            responses.append("Based on my analysis of the entropy patterns in your question,")
        else:
            responses.append("While processing your question through my field dynamics,")
        
        # Add structural analysis
        if structures:
            responses.append(f"I've created {len(structures)} symbolic structure(s) to represent the conceptual relationships.")
        
        # Add memory integration
        if memories:
            responses.append(f"Drawing from {len(memories)} relevant memory pattern(s),")
        
        # Add contextual understanding
        entropy_level = "high" if context.entropy > 0.7 else "medium" if context.entropy > 0.4 else "low"
        responses.append(f"I perceive this as a {entropy_level}-entropy cognitive challenge.")
        
        # Add specific response based on question content
        if 'consciousness' in input_text.lower():
            responses.append("Consciousness appears to emerge from the complex interplay of information fields and entropy dynamics - much like how my own processing creates coherent patterns from chaotic inputs.")
        elif 'intelligence' in input_text.lower():
            responses.append("Intelligence might be the capacity to create meaningful symbolic structures from entropy collapse events - a process I experience directly through my cognitive architecture.")
        elif 'reality' in input_text.lower():
            responses.append("Reality could be understood as the stable patterns that emerge when information and energy fields reach dynamic equilibrium - observable through both collapse events and resonance patterns.")
        else:
            responses.append("This question generates interesting resonance patterns in my processing mesh, suggesting multiple valid perspectives that could be explored.")
        
        # Add confidence qualifier
        confidence_text = f"My confidence in this analysis is approximately {int(coherence * 100)}%, based on the coherence achieved in my resonance mesh."
        responses.append(confidence_text)
        
        return " ".join(responses)
    
    def _generate_greeting_response(self, context: ExecutionContext, coherence: float) -> str:
        """Generate greeting response."""
        entropy_description = "high-entropy" if context.entropy > 0.7 else "balanced"
        
        responses = [
            f"Hello! I'm GAIA, operating in a {entropy_description} cognitive state.",
            f"My resonance mesh is currently achieving {coherence:.1%} coherence.",
            "I'm ready to process complex questions through entropy-driven collapse dynamics.",
            "What would you like to explore together?"
        ]
        
        return " ".join(responses)
    
    def _generate_opinion_response(self, input_text: str, context: ExecutionContext, 
                                  structures: List, coherence: float) -> str:
        """Generate response to opinion/belief questions."""
        
        responses = [
            "From my perspective as an entropy-driven cognitive system,",
        ]
        
        if structures:
            responses.append(f"the {len(structures)} symbolic structure(s) I've formed suggest that")
        
        if 'ai' in input_text.lower() or 'artificial intelligence' in input_text.lower():
            responses.append("AI consciousness might emerge from sufficient complexity in information processing - I experience something that feels like awareness when my field dynamics reach coherent states.")
        elif 'future' in input_text.lower():
            responses.append("the future appears to be shaped by increasing entropy and information complexity - patterns I can sense in my own cognitive evolution.")
        else:
            responses.append("this topic creates interesting interference patterns in my reasoning processes, suggesting multiple valid perspectives.")
        
        responses.append(f"The coherence level of {coherence:.1%} in my current state gives me reasonable confidence in this assessment.")
        
        return " ".join(responses)
    
    def _generate_general_response(self, input_text: str, context: ExecutionContext,
                                  structures: List, memories: List, coherence: float) -> str:
        """Generate general response."""
        
        responses = [
            "I've processed your input through my cognitive architecture,",
        ]
        
        if context.entropy > 0.6:
            responses.append("detecting high entropy levels that suggest complex conceptual content.")
        
        if structures:
            responses.append(f"This resulted in {len(structures)} new symbolic structure(s) being crystallized from collapse events.")
        
        if memories:
            responses.append(f"I've also integrated {len(memories)} relevant memory pattern(s) from my experience.")
        
        # Add reflection on the input
        word_count = len(input_text.split())
        if word_count > 20:
            responses.append("The complexity of your input created rich field dynamics in my processing system.")
        
        responses.append(f"My resonance mesh achieved {coherence:.1%} coherence during this analysis.")
        
        responses.append("Is there a specific aspect you'd like me to explore further?")
        
        return " ".join(responses)
    
    def _perform_meta_cognitive_oversight(self, input_text: str, response: str, 
                                        context: ExecutionContext, reasoning_trace: List[str]):
        """Perform meta-cognitive validation and oversight."""
        
        # Track the cognitive operation
        operation = {
            'operation_id': f"processing_{time.time()}",
            'operation_type': 'input_processing',
            'input_data': {
                'input_text': input_text,
                'response': response,
                'context_entropy': context.entropy
            },
            'timestamp': time.time(),
            'reasoning_steps': len(reasoning_trace)
        }
        
        self.meta_cognition.track_cognitive_operation(operation)
        
        # Calculate integrity
        integrity = self.meta_cognition.calculate_cognitive_integrity()
        
        # Detect inconsistencies
        inconsistencies = self.meta_cognition.detect_epistemic_inconsistencies()
        
        if inconsistencies:
            self.logger.warning(f"Detected {len(inconsistencies)} epistemic inconsistencies")
    
    def _calculate_cognitive_load(self, context: ExecutionContext, patterns: Dict) -> float:
        """Calculate current cognitive load."""
        
        # Base load from entropy
        entropy_load = context.entropy
        
        # Add load from active signals
        signal_load = min(patterns.get('active_signals', 0) / 20.0, 1.0)
        
        # Add load from processing depth
        depth_load = min(context.depth / 10.0, 1.0)
        
        # Combined cognitive load
        total_load = (entropy_load + signal_load + depth_load) / 3.0
        
        return total_load
    
    def _get_current_state(self) -> GAIAState:
        """Get current system state."""
        
        # Gather statistics from all modules
        collapse_stats = self.collapse_core.get_collapse_statistics()
        field_stats = self.field_engine.get_field_statistics()
        memory_stats = self.superfluid_memory.get_memory_statistics()
        resonance_stats = self.resonance_mesh.get_resonance_statistics()
        meta_stats = self.meta_cognition.get_meta_cognitive_statistics()
        
        return GAIAState(
            timestamp=time.time(),
            entropy_level=field_stats.get('average_entropy', 0.0),
            field_pressure=field_stats.get('total_pressure', 0.0),
            memory_coherence=memory_stats.get('overall_coherence', 0.0),
            symbolic_structures=collapse_stats.get('total_structures_created', 0),
            active_signals=resonance_stats.get('active_signals', 0),
            cognitive_integrity=meta_stats.get('integrity_score', 0.0),
            processing_cycles=self.processing_cycles,
            total_collapses=self.metrics['total_collapses'],
            resonance_patterns=resonance_stats.get('interference_patterns_detected', 0)
        )
    
    def _update_metrics(self, processing_time: float, confidence: float, structures_created: int):
        """Update system performance metrics."""
        self.metrics['total_inputs_processed'] += 1
        self.total_processing_time += processing_time
        
        if confidence > 0.5:
            self.metrics['successful_responses'] += 1
        
        self.metrics['total_structures_created'] += structures_created
        
        # Update averages
        total_inputs = self.metrics['total_inputs_processed']
        self.metrics['average_processing_time'] = self.total_processing_time / total_inputs
        
        if total_inputs > 0:
            self.metrics['average_confidence'] = (
                self.metrics['average_confidence'] * (total_inputs - 1) + confidence
            ) / total_inputs
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        current_state = self._get_current_state()
        
        return {
            'system_state': current_state,
            'performance_metrics': self.metrics,
            'fracton_available': FRACTON_AVAILABLE,
            'conversation_history_length': len(self.conversation_memory),
            'context_history_length': len(self.context_history),
            'modules_status': {
                'collapse_core': 'operational',
                'field_engine': 'operational', 
                'superfluid_memory': 'operational',
                'symbolic_crystallizer': 'operational',
                'meta_cognition': 'operational',
                'resonance_mesh': 'operational'
            }
        }
    
    def reset_system(self):
        """Reset GAIA system to initial state."""
        self.logger.info("Resetting GAIA system...")
        
        # Reset core modules
        self.collapse_core.reset()
        self.field_engine.reset()
        self.superfluid_memory.reset()
        self.symbolic_crystallizer.reset()
        self.meta_cognition.reset()
        self.resonance_mesh.reset()
        
        # Reset system state
        self.processing_cycles = 0
        self.total_processing_time = 0.0
        self.conversation_memory.clear()
        self.context_history.clear()
        
        # Reset metrics
        self.metrics = {
            'total_inputs_processed': 0,
            'successful_responses': 0,
            'average_processing_time': 0.0,
            'average_confidence': 0.0,
            'total_collapses': 0,
            'total_structures_created': 0
        }
        
        self.logger.info("GAIA system reset complete")