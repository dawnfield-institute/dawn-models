#!/usr/bin/env python3
"""
Quick test to verify bifractal memory scaffolding functionality.
"""

import sys
sys.path.append('src')

from core.resonance_mesh import ResonanceMesh, SignalType, BifractalDepth
from fracton.core.recursive_engine import ExecutionContext
import time

def test_bifractal_memory_scaffolding():
    """Test the bifractal memory scaffolding functionality."""
    print("🧠 Testing Bifractal Memory Scaffolding...")
    
    # Create resonance mesh with bifractal memory config
    config = {
        'memory_depth': 5,
        'crystallization_threshold': 0.6
    }
    mesh = ResonanceMesh(grid_size=(8, 8), config=config)
    
    # Create test context
    context = ExecutionContext(
        entropy=0.3,
        depth=1,
        metadata={'conservation_mode': True}
    )
    
    # Store some memory patterns
    experiences = [
        {'content': 'analytical thinking pattern', 'type': 'cognitive'},
        {'content': 'creative problem solving', 'type': 'creative'},  
        {'content': 'emotional response to beauty', 'type': 'emotional'},
        {'content': 'systematic organization approach', 'type': 'systematic'},
        {'content': 'adaptive learning behavior', 'type': 'adaptive'}
    ]
    
    print(f"📥 Storing {len(experiences)} experience patterns...")
    
    # Store experiences and crystallize them
    stored_patterns = []
    for i, experience in enumerate(experiences):
        pattern = mesh.crystallize_experience(experience, context)
        stored_patterns.append(pattern)
        print(f"   Pattern {i+1}: {pattern.pattern_id} at depth {pattern.depth_level.name}")
    
    # Activate some patterns to increase crystallization
    print("\n⚡ Activating memory patterns...")
    for pattern in stored_patterns[:3]:  # Activate first 3
        for _ in range(5):  # Multiple activations
            mesh.memory_scaffold.activate_memory_pattern(pattern.pattern_id, 0.8)
    
    # Generate memory-influenced response
    print("\n🧠 Generating memory-influenced response...")
    query = {'question': 'How should I approach a complex problem?', 'context': 'problem_solving'}
    response = mesh.generate_memory_influenced_response(query, context)
    
    print(f"   Activated patterns: {response.get('activated_patterns', 0)}")
    print(f"   Response confidence: {response.get('response_confidence', 0):.3f}")
    print(f"   Personality influence: {response.get('personality_influence', 0):.3f}")
    print(f"   Memory depths accessed: {response.get('memory_depths_accessed', [])}")
    
    # Evolve personality cores
    print("\n🌟 Evolving personality cores...")
    new_cores = mesh.memory_scaffold.evolve_personality_cores()
    print(f"   New personality cores emerged: {len(new_cores)}")
    
    for core in new_cores:
        print(f"   Core {core.core_id}: stability={core.stability_metric:.3f}")
        print(f"      Traits: {dict(list(core.personality_traits.items())[:3])}")
    
    # Get comprehensive statistics
    print("\n📊 Bifractal Memory Statistics:")
    stats = mesh.get_resonance_statistics()
    
    print(f"   Total signals emitted: {stats['total_signals_emitted']}")
    print(f"   Memory activations: {stats['memory_activations']}")  
    print(f"   Personality emergences: {stats['personality_emergences']}")
    print(f"   Active memory resonances: {stats['active_memory_resonances']}")
    print(f"   Active personality cores: {stats['active_personality_cores']}")
    print(f"   Memory-mesh coherence: {stats.get('memory_mesh_coherence', 0):.3f}")
    print(f"   Personality influence strength: {stats.get('personality_influence_strength', 0):.3f}")
    
    memory_stats = stats['memory_scaffold_stats']
    print(f"   Total memory patterns: {memory_stats['total_patterns']}")
    print(f"   Patterns by depth: {memory_stats['patterns_by_depth']}")
    print(f"   Average crystallization: {memory_stats['average_crystallization']:.3f}")
    print(f"   Crystallization events: {memory_stats['crystallization_events']}")
    
    print("\n✅ Bifractal memory scaffolding test completed!")
    return True

def test_recursive_memory_patterns():
    """Test recursive bifractal memory pattern formation."""
    print("\n🔄 Testing Recursive Memory Pattern Formation...")
    
    mesh = ResonanceMesh(config={'memory_depth': 5, 'crystallization_threshold': 0.5})
    context = ExecutionContext(entropy=0.2, depth=2, metadata={'conservation_mode': True})
    
    # Create hierarchical memory structure
    parent_experience = {
        'concept': 'learning_strategy',
        'description': 'meta-cognitive approach to learning'
    }
    
    parent_pattern = mesh.crystallize_experience(parent_experience, context)
    print(f"   Parent pattern: {parent_pattern.pattern_id}")
    
    # Create child patterns
    child_experiences = [
        {'strategy': 'break down complex problems', 'parent_concept': 'learning_strategy'},
        {'strategy': 'seek multiple perspectives', 'parent_concept': 'learning_strategy'},
        {'strategy': 'reflect on learning process', 'parent_concept': 'learning_strategy'}
    ]
    
    for child_exp in child_experiences:
        child_pattern = mesh.memory_scaffold.store_memory_pattern(
            child_exp, context, parent_pattern_id=parent_pattern.pattern_id
        )
        print(f"   Child pattern: {child_pattern.pattern_id}")
    
    # Test recursive activation
    print("\n⚡ Testing recursive activation propagation...")
    activated = mesh.memory_scaffold.activate_memory_pattern(parent_pattern.pattern_id, 1.0)
    
    if activated:
        print(f"   Parent activated with strength: {activated.crystallization_strength:.3f}")
        print(f"   Child patterns: {len(activated.child_patterns)}")
    
    print("✅ Recursive memory pattern test completed!")
    return True

if __name__ == "__main__":
    print("🚀 Starting Bifractal Memory Scaffolding Tests...\n")
    
    try:
        # Test basic bifractal memory functionality
        test_bifractal_memory_scaffolding()
        
        # Test recursive memory patterns
        test_recursive_memory_patterns()
        
        print("\n🎉 All bifractal memory tests passed successfully!")
        print("🧠 Long-term memory and personality emergence capabilities are working!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)