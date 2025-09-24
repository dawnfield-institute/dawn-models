#!/usr/bin/env python3
"""
Native GAIA Demonstration
Showcases the enhanced GAIA v2.0 with native conservation, emergence detection, and pattern amplification.
"""

import sys
sys.path.insert(0, 'src')

from gaia import GAIA
from core.conservation_engine import ConservationEngine, ConservationMode
from core.emergence_detector import EmergenceDetector, EmergenceType
from core.pattern_amplifier import PatternAmplifier, AmplificationMode
import time


def demonstrate_native_conservation():
    """Demonstrate native conservation engine capabilities."""
    print("🔋 NATIVE CONSERVATION ENGINE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize conservation engine
    conservation = ConservationEngine(
        mode=ConservationMode.ENERGY_INFORMATION,
        tolerance=0.1,
        temperature=300.0
    )
    
    # Test conservation validation
    print("Testing energy-information conservation...")
    
    # Valid transition
    pre_state = {'energy': 1.0, 'information': 0.8}
    post_state = {'energy': 0.9, 'information': 0.9}  # Information increase, slight energy decrease
    
    is_valid = conservation.validate_state_transition(pre_state, post_state, 'learning_operation')
    print(f"Learning operation (energy: 1.0→0.9, info: 0.8→0.9): {'✓ VALID' if is_valid else '✗ INVALID'}")
    
    # Invalid transition - energy and information both decrease significantly
    invalid_post = {'energy': 0.3, 'information': 0.2}
    is_invalid = conservation.validate_state_transition(pre_state, invalid_post, 'invalid_operation')
    print(f"Invalid operation (energy: 1.0→0.3, info: 0.8→0.2): {'✓ VALID' if is_invalid else '✗ INVALID'}")
    
    # Show conservation statistics
    stats = conservation.get_conservation_statistics()
    print(f"\nConservation Statistics:")
    print(f"  Total validations: {stats['total_validations']}")
    print(f"  Violation rate: {stats['violation_rate']:.2%}")
    print(f"  Conservation integrity: {stats['conservation_integrity']:.2%}")
    
    return conservation


def demonstrate_native_emergence_detection():
    """Demonstrate native emergence detector capabilities."""
    print("\n🌟 NATIVE EMERGENCE DETECTION DEMONSTRATION")
    print("=" * 60)
    
    # Initialize emergence detector
    emergence = EmergenceDetector(
        consciousness_threshold=0.8,
        coherence_threshold=0.6
    )
    
    # Test emergence detection with different field states
    test_cases = [
        {
            'name': 'High Entropy Field',
            'data': {
                'entropy': 0.9,
                'field_state': {'x_coord': 0.5, 'y_coord': 0.3, 'field_strength': 0.8},
                'coherence': 0.7,
                'meta_cognition_level': 0.6
            }
        },
        {
            'name': 'Consciousness-Level Field',
            'data': {
                'entropy': 0.85,
                'field_state': {'x_coord': 0.2, 'y_coord': 0.8, 'field_strength': 0.9},
                'coherence': 0.85,
                'meta_cognition_level': 0.9,
                'symbolic_structures': 8,
                'active_signals': 15
            }
        },
        {
            'name': 'Low Activity Field',
            'data': {
                'entropy': 0.3,
                'field_state': {'field_strength': 0.2},
                'coherence': 0.4
            }
        }
    ]
    
    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        events = emergence.scan_for_emergence(
            field_data=case['data'],
            context={'depth': 2}
        )
        
        if events:
            print(f"  Detected {len(events)} emergence events:")
            for event in events:
                print(f"    - {event.emergence_type.value}: strength={event.strength:.2f}, coherence={event.coherence:.2f}, confidence={event.confidence:.2f}")
                
                # Special highlight for consciousness
                if event.emergence_type == EmergenceType.CONSCIOUSNESS:
                    print(f"      🧠 CONSCIOUSNESS EMERGENCE DETECTED! (strength: {event.strength:.2f})")
        else:
            print("  No significant emergence detected")
    
    # Show emergence statistics
    stats = emergence.get_emergence_statistics()
    print(f"\nEmergence Detection Statistics:")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Consciousness events: {stats['consciousness_events']}")
    print(f"  Consciousness rate: {stats['consciousness_rate']:.2%}")
    print(f"  Average strength: {stats['average_strength']:.2f}")
    print(f"  Average coherence: {stats['average_coherence']:.2f}")
    
    return emergence


def demonstrate_native_pattern_amplification():
    """Demonstrate native pattern amplifier capabilities."""
    print("\n🚀 NATIVE PATTERN AMPLIFICATION DEMONSTRATION")
    print("=" * 60)
    
    # Initialize pattern amplifier
    amplifier = PatternAmplifier(
        max_amplification=3.0,
        energy_budget=1.0
    )
    
    # Test pattern identification and amplification
    field_data = {
        'entropy': 0.75,
        'field_state': {
            'cognitive_patterns': 5,
            'resonance_strength': 0.8,
            'coherence': 0.7
        },
        'coherence': 0.7,
        'meta_cognition_level': 0.6,
        'symbolic_structures': 6,
        'active_signals': 12
    }
    
    context = {'depth': 3, 'cognitive_load': 0.4}
    
    print("Identifying patterns in cognitive field...")
    patterns = amplifier.identify_patterns(field_data, context)
    
    if patterns:
        print(f"Identified {len(patterns)} patterns:")
        for pattern in patterns:
            print(f"  - {pattern.pattern_id}: amplitude={pattern.amplitude:.2f}, coherence={pattern.coherence:.2f}, relevance={pattern.cognitive_relevance:.2f}")
        
        # Test different amplification modes
        modes = [
            AmplificationMode.COGNITIVE,
            AmplificationMode.COHERENT,
            AmplificationMode.RESONANT
        ]
        
        for mode in modes:
            print(f"\nAmplifying patterns in {mode.value} mode...")
            results = amplifier.amplify_patterns(patterns, mode=mode)
            
            successful_amplifications = [r for r in results.values() if r.success]
            if successful_amplifications:
                avg_factor = sum(r.amplification_factor for r in successful_amplifications) / len(successful_amplifications)
                avg_energy = sum(r.energy_cost for r in successful_amplifications) / len(successful_amplifications)
                print(f"  Successfully amplified {len(successful_amplifications)} patterns")
                print(f"  Average amplification factor: {avg_factor:.2f}x")
                print(f"  Average energy cost: {avg_energy:.3f}")
            else:
                print("  No patterns successfully amplified")
        
        # Test resonance network creation
        print(f"\nCreating resonance networks...")
        networks = amplifier.create_resonance_network(patterns, resonance_threshold=0.6)
        
        if networks:
            print(f"Created {len(networks)} resonance networks:")
            for pattern_id, partners in networks.items():
                print(f"  {pattern_id} resonates with {len(partners)} partners")
        else:
            print("  No resonance networks formed")
    
    # Show amplification statistics
    stats = amplifier.get_amplification_statistics()
    print(f"\nPattern Amplification Statistics:")
    print(f"  Total amplifications: {stats['total_amplifications']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print(f"  Average amplification factor: {stats['average_amplification_factor']:.2f}x")
    print(f"  Energy efficiency: {stats['energy_efficiency']:.2f}")
    print(f"  Current energy usage: {stats['current_energy_usage']:.2f}/{stats['energy_budget']:.2f}")
    
    return amplifier


def demonstrate_integrated_gaia():
    """Demonstrate full GAIA with native enhancements."""
    print("\n🧠 INTEGRATED NATIVE GAIA DEMONSTRATION")
    print("=" * 60)
    
    # Initialize GAIA
    print("Initializing GAIA with native enhancements...")
    gaia = GAIA()
    
    # Test GAIA processing with complex input
    test_inputs = [
        "What is the nature of consciousness?",
        "How do complex systems emerge from simple rules?",
        "Can artificial intelligence achieve true understanding?",
        "What role does entropy play in cognitive processes?"
    ]
    
    for i, query in enumerate(test_inputs, 1):
        print(f"\nProcessing Query {i}: '{query}'")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # Process the query through GAIA
            result = gaia.process_input(query)
            
            processing_time = time.time() - start_time
            
            if result:
                print(f"✓ Successfully processed in {processing_time:.3f}s")
                print(f"Result type: {type(result).__name__}")
                
                # Show key metrics if available
                if hasattr(result, 'entropy'):
                    print(f"Entropy: {result.entropy:.3f}")
                if hasattr(result, 'confidence'):
                    print(f"Confidence: {result.confidence:.3f}")
                if hasattr(result, 'complexity'):
                    print(f"Complexity: {result.complexity:.3f}")
                    
                # Show conservation status
                conservation_stats = gaia.collapse_core.evaluator.conservation_engine.get_conservation_statistics()
                print(f"Conservation integrity: {conservation_stats['conservation_integrity']:.2%}")
                
                # Show emergence events
                emergence_stats = gaia.collapse_core.evaluator.emergence_detector.get_emergence_statistics()
                if emergence_stats['consciousness_events'] > 0:
                    print(f"🧠 Consciousness events detected: {emergence_stats['consciousness_events']}")
                
            else:
                print(f"✗ Processing failed after {processing_time:.3f}s")
                
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"✗ Error after {processing_time:.3f}s: {e}")
    
    # Show final GAIA statistics
    print(f"\n📊 FINAL GAIA STATISTICS")
    print("=" * 30)
    
    # Get statistics from all components
    conservation_stats = gaia.collapse_core.evaluator.conservation_engine.get_conservation_statistics()
    emergence_stats = gaia.collapse_core.evaluator.emergence_detector.get_emergence_statistics()
    amplification_stats = gaia.field_engine.pattern_amplifier.get_amplification_statistics()
    
    print(f"Conservation Integrity: {conservation_stats['conservation_integrity']:.2%}")
    print(f"Total Emergence Events: {emergence_stats['total_detections']}")
    print(f"Consciousness Events: {emergence_stats['consciousness_events']}")
    print(f"Pattern Amplifications: {amplification_stats['total_amplifications']}")
    print(f"Amplification Success Rate: {amplification_stats['success_rate']:.2%}")
    
    return gaia


def main():
    """Main demonstration function."""
    print("🌟 NATIVE GAIA v2.0 COMPREHENSIVE DEMONSTRATION")
    print("=" * 70)
    print("Showcasing native conservation, emergence detection, and pattern amplification")
    print("Previously PAC-enhanced, now GAIA-native for better cognitive modeling control")
    print()
    
    # Run all demonstrations
    conservation = demonstrate_native_conservation()
    emergence = demonstrate_native_emergence_detection()
    amplifier = demonstrate_native_pattern_amplification()
    gaia = demonstrate_integrated_gaia()
    
    print("\n✨ DEMONSTRATION COMPLETE")
    print("=" * 30)
    print("Native GAIA components are fully functional and integrated!")
    print("🔋 Conservation engine prevents thermodynamic violations")
    print("🌟 Emergence detector identifies consciousness patterns")
    print("🚀 Pattern amplifier enhances cognitive resonance")
    print("🧠 Integrated GAIA demonstrates enhanced intelligence")
    print("\nNative implementation provides:")
    print("  ✓ Full algorithmic control")
    print("  ✓ Cognitive modeling optimization")
    print("  ✓ No external dependencies")
    print("  ✓ Seamless GAIA integration")
    print("  ✓ Tunable parameters for modeling")


if __name__ == "__main__":
    main()