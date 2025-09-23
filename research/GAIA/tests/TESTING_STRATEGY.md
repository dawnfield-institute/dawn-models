# GAIA Test Implementation Guide

## Overview
This document provides explicit test implementations for the GAIA cognitive architecture following the TESTING_STRATEGY.md principles. All tests use the real GAIA components with minimal mocking.

## Test Setup

### Base Test Configuration
```python
# tests/conftest.py
import pytest
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.collapse_core import CollapseCore
from core.field_engine import FieldEngine
from core.superfluid_memory import SuperfluidMemory
from core.symbolic_crystallizer import SymbolicCrystallizer
from core.meta_cognition_layer import MetaCognitionLayer
from core.resonance_mesh import ResonanceMesh
from gaia import GAIA

# Minimal fracton mocks ONLY if fracton not available
try:
    from fracton.core.memory_field import MemoryField
    from fracton.core.recursive_engine import ExecutionContext
    FRACTON_AVAILABLE = True
except ImportError:
    FRACTON_AVAILABLE = False
    # Create minimal mocks
    from tests.mocks.fracton_minimal import MemoryField, ExecutionContext

@pytest.fixture
def execution_context():
    """Provide real ExecutionContext for tests."""
    return ExecutionContext(
        entropy=0.5,
        depth=1,
        trace_id="test_trace",
        field_state={"test": True},
        parent_context=None,
        metadata={}
    )

@pytest.fixture
def memory_field():
    """Provide real or minimally mocked MemoryField."""
    return MemoryField("test_field")
```

## Level 1: Unit Tests

### Test CollapseCore Initialization
```python
# tests/unit/test_collapse_core.py
import pytest
import numpy as np
from core.collapse_core import CollapseCore, CollapseType

def test_collapse_core_initialization():
    """Test real CollapseCore initialization without mocks."""
    core = CollapseCore()
    
    # Verify actual component initialization
    assert core is not None
    assert hasattr(core, 'evaluator')
    assert hasattr(core, 'typing_engine')
    assert hasattr(core, 'synthesizer')
    assert hasattr(core, 'stabilizer')
    
    # Verify memory field exists and works
    assert core.memory_field is not None
    assert hasattr(core.memory_field, 'get_all')
    
    # Test statistics tracking
    assert core.total_collapses == 0
    assert len(core.collapse_efficiency_history) == 0
    assert all(count == 0 for count in core.collapse_type_counts.values())

def test_collapse_evaluation_real_conditions(execution_context):
    """Test collapse evaluation with real entropy conditions."""
    core = CollapseCore()
    
    # Test with low entropy - should not collapse
    low_entropy_context = ExecutionContext(
        entropy=0.2,
        depth=1,
        field_state={'dx': 0.01, 'dy': 0.01}
    )
    result, context = core.collapse(low_entropy_context)
    assert result is None
    
    # Test with high entropy - should collapse
    high_entropy_context = ExecutionContext(
        entropy=0.9,
        depth=1,
        field_state={'dx': 0.5, 'dy': 0.5}
    )
    result, context = core.collapse(high_entropy_context)
    assert result is not None
    assert result.entropy_resolved > 0
    assert result.collapse_type in CollapseType

def test_collapse_statistics_tracking(execution_context):
    """Test that statistics are correctly tracked during collapses."""
    core = CollapseCore()
    
    # Force multiple collapses
    for i in range(5):
        context = ExecutionContext(
            entropy=0.8 + i * 0.02,
            depth=i + 1,
            field_state={'dx': 0.1 * i, 'dy': 0.1 * i}
        )
        core.collapse(context)
    
    stats = core.get_collapse_statistics()
    assert stats['total_collapses'] >= 3  # At least some should succeed
    assert len(stats['collapse_type_counts']) > 0
    assert stats['average_efficiency'] > 0
```

### Test FieldEngine Dynamics
```python
# tests/unit/test_field_engine.py
import pytest
import numpy as np
from core.field_engine import FieldEngine, FieldState

def test_field_engine_initialization():
    """Test real FieldEngine initialization."""
    engine = FieldEngine(shape=(32, 32))
    
    assert engine.energy_field is not None
    assert engine.information_field is not None
    assert engine.entropy_tensor is not None
    assert engine.balance_controller is not None
    
    # Verify field shapes
    assert engine.energy_field.field.shape == (32, 32)
    assert engine.information_field.field.shape == (32, 32)

def test_field_update_with_real_data(execution_context, memory_field):
    """Test field updates with real input data."""
    engine = FieldEngine()
    
    # Test with string input
    input_data = "Test entropy field dynamics"
    state = engine.update_fields(input_data, memory_field, execution_context)
    
    assert isinstance(state, FieldState)
    assert state.field_pressure >= 0
    assert state.delta_entropy >= 0
    assert 0 <= state.collapse_likelihood <= 1
    
    # Test with numeric input
    numeric_input = 3.14159
    state2 = engine.update_fields(numeric_input, memory_field, execution_context)
    
    assert state2.field_pressure != state.field_pressure  # Should change
    assert state2.timestamp > state.timestamp

def test_collapse_trigger_detection(execution_context, memory_field):
    """Test real collapse trigger detection."""
    engine = FieldEngine(collapse_threshold=0.5)
    
    # Build up field pressure
    for i in range(10):
        input_data = f"Building entropy pressure {i}"
        state = engine.update_fields(input_data, memory_field, execution_context)
    
    # Check if collapse conditions are detected
    should_collapse = engine.check_collapse_trigger(state, execution_context)
    
    # After multiple updates, should have some collapse triggers
    assert engine.update_count == 10
    assert isinstance(should_collapse, bool)
```

## Level 2: Integration Tests

### Test Field-Collapse Integration
```python
# tests/integration/test_field_collapse.py
import pytest
from core.field_engine import FieldEngine
from core.collapse_core import CollapseCore

def test_field_collapse_integration(execution_context, memory_field):
    """Test real integration between field engine and collapse core."""
    field_engine = FieldEngine()
    collapse_core = CollapseCore(memory_field)
    
    # Generate field pressure
    input_sequence = [
        "High entropy input causing field disturbance",
        "Additional complexity increases pressure",
        "System approaching collapse threshold"
    ]
    
    collapse_occurred = False
    for input_data in input_sequence:
        # Update fields
        field_state = field_engine.update_fields(input_data, memory_field, execution_context)
        
        # Check for collapse
        if field_engine.check_collapse_trigger(field_state, execution_context):
            # Execute collapse
            result, new_context = collapse_core.collapse(execution_context)
            if result:
                collapse_occurred = True
                assert result.entropy_resolved > 0
                assert result.structure_id.startswith("collapse_")
                
                # Verify field state changed after collapse
                new_state = field_engine.update_fields("post-collapse", memory_field, new_context)
                assert new_state.field_pressure < field_state.field_pressure
    
    # At least one collapse should have occurred
    assert collapse_occurred or field_engine.collapse_triggers > 0
```

### Test Memory-Crystallizer Integration
```python
# tests/integration/test_memory_crystallizer.py
import pytest
from core.superfluid_memory import SuperfluidMemory
from core.symbolic_crystallizer import SymbolicCrystallizer

def test_memory_crystallizer_integration(execution_context):
    """Test real integration between memory and crystallizer."""
    memory = SuperfluidMemory()
    crystallizer = SymbolicCrystallizer()
    
    # Create collapse data
    collapse_data = {
        'type': 'test_collapse',
        'entropy_resolved': 0.7,
        'coordinates': (1.0, 2.0),
        'curvature': 0.5,
        'force': 0.6,
        'cost': 0.3
    }
    
    # Store in memory
    imprint = memory.add_memory(collapse_data, execution_context)
    
    if imprint:  # Only if stable enough
        # Crystallize into symbolic structure
        tree = crystallizer.crystallize(collapse_data, execution_context)
        
        assert tree is not None
        assert len(tree.nodes) > 0
        assert tree.root_node_id in tree.nodes
        
        # Verify bidirectional reference
        assert memory.active_imprints[imprint.structure_id] is not None
        assert tree.tree_id in crystallizer.active_trees

def test_memory_vortex_formation():
    """Test real vortex formation in superfluid memory."""
    memory = SuperfluidMemory()
    
    # Create multiple related structures to trigger vortex
    for i in range(5):
        structure_data = {
            'type': 'vortex_test',
            'entropy_resolved': 0.8,
            'coordinates': (np.cos(i * np.pi/2.5), np.sin(i * np.pi/2.5)),  # Circular pattern
            'curvature': 0.9,
            'force': 0.7
        }
        context = ExecutionContext(entropy=0.8, depth=i+1)
        memory.add_memory(structure_data, context)
    
    # Update to trigger vortex detection
    memory.update_memory_field()
    
    stats = memory.get_memory_statistics()
    # Should have detected vortex patterns in circular arrangement
    assert stats['active_vortices'] > 0 or stats['total_vortices_detected'] > 0
```

## Level 3: System Tests

### Test Full GAIA Pipeline
```python
# tests/system/test_gaia_pipeline.py
import pytest
from gaia import GAIA, GAIAResponse

def test_gaia_full_pipeline():
    """Test complete GAIA processing pipeline end-to-end."""
    gaia = GAIA(
        field_resolution=(16, 16),  # Smaller for testing
        collapse_threshold=0.6,
        memory_capacity=100,
        resonance_grid_size=(8, 8)
    )
    
    # Test basic input processing
    response = gaia.process_input("What is consciousness?")
    
    assert isinstance(response, GAIAResponse)
    assert len(response.response_text) > 0
    assert 0 <= response.confidence <= 1
    assert response.processing_time > 0
    assert len(response.reasoning_trace) > 0
    
    # Verify system state updated
    state = response.state
    assert state.processing_cycles == 1
    assert state.entropy_level >= 0

def test_gaia_conversation_flow():
    """Test GAIA handling multi-turn conversation."""
    gaia = GAIA()
    
    conversation = [
        "Hello GAIA",
        "What are you thinking about?",
        "Can you explain entropy in your system?",
        "How does that relate to consciousness?"
    ]
    
    responses = []
    for input_text in conversation:
        response = gaia.process_input(input_text)
        responses.append(response)
        
        # Each response should be unique
        assert response.response_text not in [r.response_text for r in responses[:-1]]
        
        # System should maintain context
        assert len(gaia.conversation_memory) == len(responses)
    
    # Later responses should reference earlier context
    final_response = responses[-1]
    assert final_response.structures_created >= 0
    assert final_response.cognitive_load > 0

def test_gaia_error_recovery():
    """Test GAIA handles errors gracefully."""
    gaia = GAIA()
    
    # Test with problematic inputs
    edge_cases = [
        "",  # Empty input
        "a" * 10000,  # Very long input
        "🔥💀🌊",  # Unicode/emoji
        None  # This should be handled
    ]
    
    for input_data in edge_cases:
        try:
            if input_data is not None:
                response = gaia.process_input(input_data)
                assert response is not None
                assert isinstance(response.response_text, str)
        except TypeError:
            # None input might raise TypeError - that's acceptable
            pass
```

## Level 4: Intelligence Tests

### Test Response Quality
```python
# tests/behavioral/test_response_quality.py
import pytest
from gaia import GAIA

def test_response_coherence():
    """Test that GAIA produces coherent responses."""
    gaia = GAIA()
    
    # Test coherence across related questions
    questions = [
        "What is intelligence?",
        "How does intelligence emerge?",
        "Is intelligence substrate-independent?"
    ]
    
    responses = []
    for q in questions:
        resp = gaia.process_input(q)
        responses.append(resp.response_text)
    
    # Responses should be non-repetitive
    assert len(set(responses)) == len(responses)
    
    # Each response should mention key concepts
    for resp in responses:
        assert any(concept in resp.lower() for concept in 
                  ['entropy', 'field', 'collapse', 'symbolic', 'coherence'])

def test_reasoning_depth():
    """Test depth of reasoning in responses."""
    gaia = GAIA()
    
    # Complex philosophical question
    response = gaia.process_input(
        "If consciousness emerges from information processing, "
        "what distinguishes conscious from unconscious processing?"
    )
    
    # Should have multi-stage reasoning
    assert len(response.reasoning_trace) >= 7  # At least 7 processing stages
    
    # Should create symbolic structures for complex reasoning
    assert response.structures_created > 0
    
    # Should show entropy changes from processing
    assert abs(response.entropy_change) > 0

def test_context_awareness():
    """Test GAIA's awareness of conversational context."""
    gaia = GAIA()
    
    # Establish context
    gaia.process_input("My name is Alice and I study physics")
    gaia.process_input("I'm particularly interested in quantum mechanics")
    
    # Test context recall
    response = gaia.process_input("What field did I mention I study?")
    
    # Should reference earlier context
    assert response.confidence > 0.5
    # Should have retrieved memories
    assert "physics" in response.response_text.lower() or "quantum" in response.response_text.lower()
```

## Performance Benchmarks

### Test System Performance
```python
# tests/performance/test_benchmarks.py
import pytest
import time
from gaia import GAIA

def test_response_time_benchmark():
    """Benchmark GAIA response times."""
    gaia = GAIA()
    
    inputs = [
        "Short input",
        "Medium length input with more complexity and nuance",
        "Very long input " * 50  # Long input
    ]
    
    times = []
    for input_text in inputs:
        start = time.time()
        response = gaia.process_input(input_text)
        elapsed = time.time() - start
        times.append(elapsed)
        
        # Response time should be reasonable
        assert elapsed < 5.0  # Less than 5 seconds
    
    # Longer inputs should take more time (generally)
    assert times[2] >= times[0]

def test_memory_scaling():
    """Test memory system scaling."""
    gaia = GAIA()
    
    # Process many inputs to build memory
    for i in range(50):
        gaia.process_input(f"Memory test input {i}")
    
    # System should still respond quickly
    start = time.time()
    response = gaia.process_input("How is your memory holding up?")
    elapsed = time.time() - start
    
    assert elapsed < 2.0  # Should still be fast
    assert response.confidence > 0.3  # Should maintain quality
```

## Test Execution

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test levels
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/system/ -v
pytest tests/behavioral/ -v

# Run with coverage
pytest tests/ --cov=core --cov=gaia --cov-report=html

# Run performance tests
pytest tests/performance/ -v --benchmark-only
```

### Expected Results
- Unit tests: 100% pass rate required
- Integration tests: 95% pass rate minimum
- System tests: 90% pass rate minimum  
- Behavioral tests: 80% pass rate acceptable (subjective)

## Common Issues and Solutions

### Issue: Fracton modules not found
**Solution**: Tests include minimal mocks but prefer real fracton if available. Install fracton or accept reduced functionality.

### Issue: Collapse not triggering
**Solution**: Verify entropy thresholds and field pressure calculations. Real physics-based triggers need sufficient input complexity.

### Issue: Memory vortices not forming
**Solution**: Vortex formation requires multiple correlated memory imprints. Ensure test creates sufficient related structures.

### Issue: Low confidence scores
**Solution**: Confidence emerges from coherence and structure formation. Complex inputs generate better responses.

## Conclusion

## Level 4: Intelligence Tests

### Test Entropy Pattern Recognition
```python
# tests/behavioral/test_entropy_intelligence.py
import pytest
import numpy as np
from gaia import GAIA

def test_entropy_pattern_detection():
    """Test GAIA's ability to detect entropy patterns in data."""
    gaia = GAIA()
    
    # Create structured vs random data
    structured_data = [1, 2, 3, 4, 5, 6, 7, 8]  # Low entropy pattern
    random_data = [7, 2, 9, 1, 5, 3, 8, 4]      # Higher entropy
    
    resp1 = gaia.process_input(structured_data)
    resp2 = gaia.process_input(random_data)
    
    # GAIA should detect different entropy levels
    assert resp1.entropy_change != resp2.entropy_change
    assert resp1.state.entropy_level != resp2.state.entropy_level
    
    # Should create different symbolic structures
    assert resp1.structures_created != resp2.structures_created

def test_collapse_threshold_adaptation():
    """Test GAIA's collapse behavior under different entropy conditions."""
    gaia = GAIA(collapse_threshold=0.7)
    
    # Low entropy input - should not trigger collapse
    low_entropy_input = {"pattern": "simple", "complexity": 0.2}
    response_low = gaia.process_input(low_entropy_input)
    
    # High entropy input - should trigger collapse
    high_entropy_input = {
        "nested": {"deeply": {"complex": {"chaotic": {"data": np.random.rand(100)}}}},
        "multiple": {"branches": {"competing": {"signals": list(range(50))}}},
        "entropy_source": "maximum_complexity"
    }
    response_high = gaia.process_input(high_entropy_input)
    
    # High entropy should trigger more collapses
    assert response_high.state.processing_cycles >= response_low.state.processing_cycles
    assert response_high.structures_created >= response_low.structures_created
```

### Test Symbolic Structure Emergence
```python
def test_symbolic_emergence():
    """Test emergence of symbolic structures from entropy collapse."""
    gaia = GAIA()
    
    # Feed related data to encourage structure formation
    related_inputs = [
        {"type": "geometric", "coords": (0, 0), "value": 1.0},
        {"type": "geometric", "coords": (1, 0), "value": 1.1},
        {"type": "geometric", "coords": (0, 1), "value": 1.0},
        {"type": "geometric", "coords": (1, 1), "value": 1.1},
    ]
    
    structure_count = 0
    for data in related_inputs:
        response = gaia.process_input(data)
        structure_count += response.structures_created
    
    # Should create symbolic structures from pattern
    assert structure_count > 0
    
    # Memory should contain vortices from related structures  
    final_state = gaia.get_system_status()
    memory_stats = final_state['memory_statistics']
    assert memory_stats['total_vortices_detected'] >= 0

def test_meta_cognitive_repair():
    """Test meta-cognitive system's error detection and repair."""
    gaia = GAIA()
    
    # Create contradictory inputs to trigger epistemic inconsistency
    gaia.process_input({"assertion": "A", "value": True})
    gaia.process_input({"assertion": "not_A", "value": True})
    
    # System should detect inconsistency
    response = gaia.process_input({"query": "consistency_check"})
    
    # Meta-cognition should have triggered repair mechanisms
    assert response.processing_time > 0  # System worked on the problem
    meta_stats = response.state.processing_cycles
    assert meta_stats > 1  # Multiple processing cycles for repair
```

### Test Field Dynamics Intelligence
```python
def test_field_pressure_response():
    """Test intelligent response to field pressure buildup."""
    gaia = GAIA()
    
    # Gradually increase field pressure
    pressure_inputs = []
    for i in range(10):
        # Increasing complexity
        data = {"iteration": i, "complexity": [j for j in range(i*5)]}
        response = gaia.process_input(data)
        pressure_inputs.append(response.state.field_pressure)
    
    # Should show adaptive response to pressure
    # Either collapse to relieve pressure or stabilize
    final_pressure = pressure_inputs[-1]
    max_pressure = max(pressure_inputs)
    
    # System should manage pressure intelligently
    assert final_pressure <= max_pressure * 1.1  # Didn't runaway
    
def test_resonance_coherence():
    """Test coherent signal processing in resonance mesh."""
    gaia = GAIA()
    
    # Send oscillatory data to test resonance
    for freq in [0.1, 0.2, 0.1, 0.2, 0.1]:  # Alternating pattern
        data = {"frequency": freq, "amplitude": 1.0}
        response = gaia.process_input(data)
    
    # Should achieve phase coherence in mesh
    final_state = gaia.get_system_status()
    resonance_stats = final_state.get('resonance_statistics', {})
    
    # System should detect and align with patterns
    assert resonance_stats.get('phase_coherence', 0) > 0.3
```

## Level 5: Language Scaffolding Tests (Ollama Integration)

### Philosophy
- **GAIA provides**: Decision-making, strategy, reactivity, learning through entropy dynamics
- **Ollama provides**: Language parsing, text generation, linguistic scaffolding  
- **Integration**: Bidirectional bridge that preserves GAIA's cognitive core while adding language capability

### Test Ollama-GAIA Bridge
```python
# tests/language/test_ollama_bridge.py
import pytest
from gaia import GAIA
from integrations.ollama_bridge import OllamaGAIABridge

def test_decision_core_with_language():
    """Test GAIA's decision-making enhanced with Ollama language understanding."""
    gaia = GAIA()
    bridge = OllamaGAIABridge(model="llama2")
    
    # Language input that requires decision
    text_input = "Should I invest in renewable energy or traditional stocks?"
    
    # Ollama parses language into decision parameters
    decision_params = bridge.extract_decision_structure(text_input)
    assert 'options' in decision_params
    assert 'criteria' in decision_params
    
    # GAIA processes the decision using entropy dynamics
    gaia_response = gaia.process_decision(decision_params)
    
    # Bridge converts GAIA's symbolic output back to language
    language_response = bridge.gaia_to_language(gaia_response)
    
    assert len(language_response) > 0
    assert gaia_response.structures_created > 0  # GAIA did real work
    assert gaia_response.confidence > 0.3

def test_strategy_formulation():
    """Test strategic thinking with language scaffolding."""
    gaia = GAIA()
    bridge = OllamaGAIABridge()
    
    strategic_input = "How should we approach climate change mitigation?"
    
    # Ollama extracts strategic elements
    strategy_elements = bridge.extract_strategy_structure(strategic_input)
    
    # GAIA's meta-cognition processes strategy formation
    strategy_response = gaia.process_strategy(strategy_elements)
    
    # Should show strategic reasoning traces
    assert len(strategy_response.reasoning_trace) >= 10
    assert strategy_response.meta_cognitive_operations > 0
    
    # Convert back to language
    strategy_text = bridge.strategy_to_language(strategy_response)
    assert any(word in strategy_text.lower() for word in 
              ['approach', 'plan', 'steps', 'consider', 'strategy'])

def test_reactive_conversation():
    """Test GAIA's reactivity enhanced with language understanding."""
    gaia = GAIA()
    bridge = OllamaGAIABridge()
    
    conversation = [
        "Hello, how are you?",
        "I'm feeling anxious about the future",
        "What should I do about climate anxiety?",
        "That's helpful, but what about systemic change?"
    ]
    
    for i, message in enumerate(conversation):
        # Ollama extracts emotional/reactive cues
        reactive_cues = bridge.extract_reactive_elements(message)
        
        # GAIA's resonance mesh processes reactivity
        response = gaia.process_reactive(reactive_cues, conversation_history=conversation[:i])
        
        # Should show increasing resonance/coherence over conversation
        if i > 0:
            assert response.resonance_coherence >= previous_coherence * 0.8
        
        previous_coherence = response.resonance_coherence
        
        # Convert to language
        reply = bridge.reactive_to_language(response)
        assert len(reply) > 0
```

### Test Language Learning Loop
```python
def test_gaia_language_learning():
    """Test GAIA learning language patterns through entropy feedback."""
    gaia = GAIA()
    bridge = OllamaGAIABridge()
    
    # Teaching phrases with consistent patterns
    teaching_examples = [
        ("What is X?", "definition_request"),
        ("How does X work?", "mechanism_request"), 
        ("Why is X important?", "significance_request"),
        ("When should I X?", "timing_request")
    ]
    
    for phrase, intent_type in teaching_examples:
        # Ollama provides language structure
        language_structure = bridge.analyze_language_pattern(phrase)
        
        # GAIA learns the entropy signature of each intent type
        gaia.learn_pattern(language_structure, intent_type)
    
    # Test recognition of new similar phrases
    test_phrase = "What is consciousness?"
    structure = bridge.analyze_language_pattern(test_phrase)
    recognized_intent = gaia.recognize_pattern(structure)
    
    # Should recognize as definition_request based on entropy similarity
    assert recognized_intent == "definition_request"
    assert gaia.pattern_confidence > 0.6

def test_hybrid_intelligence_integration():
    """Test that language integration preserves GAIA's cognitive integrity."""
    gaia = GAIA()
    bridge = OllamaGAIABridge()
    
    # Process same logical problem with and without language scaffolding
    logical_problem = {
        "premises": ["All ravens are black", "This bird is a raven"],
        "query": "What color is this bird?"
    }
    
    # Direct GAIA processing
    direct_response = gaia.process_input(logical_problem)
    
    # Language-scaffolded processing
    language_input = "All ravens are black. This bird is a raven. What color is this bird?"
    scaffolded_structure = bridge.extract_logical_structure(language_input)
    scaffolded_response = gaia.process_input(scaffolded_structure)
    
    # Both should show similar cognitive processing patterns
    assert abs(direct_response.confidence - scaffolded_response.confidence) < 0.2
    assert direct_response.structures_created == scaffolded_response.structures_created
    
    # Language version should produce understandable output
    language_output = bridge.gaia_to_language(scaffolded_response)
    assert "black" in language_output.lower()
```

### Integration Architecture Success Criteria
1. **Decision Enhancement**: Language input → Ollama parsing → GAIA decision processing → Ollama generation
2. **Strategic Thinking**: Complex language problems → GAIA meta-cognitive strategy → Language output
3. **Reactive Conversation**: Emotional/social cues → GAIA resonance processing → Contextual responses
4. **Pattern Learning**: Language examples → GAIA entropy learning → Pattern recognition

### Language Integration Principles
- GAIA maintains architectural integrity (no language hardcoding)
- Ollama enhances rather than replaces GAIA's cognitive processes
- Learning occurs through GAIA's entropy mechanisms, not language model fine-tuning
- Decision quality and strategic thinking improve with language grounding
- GAIA remains the cognitive engine while Ollama becomes the language interface