#!/usr/bin/env python3
"""
GAIA-PAC Integration Demonstration
Shows enhanced cognitive capabilities with PAC Engine integration.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from gaia import GAIA
import time

def demonstrate_pac_enhanced_gaia():
    """Demonstrate GAIA with PAC enhancements."""
    print("=" * 60)
    print("🧠 GAIA v2.0 with PAC Engine Integration Demo")
    print("=" * 60)
    
    # Initialize GAIA
    print("\n🔧 Initializing GAIA with PAC enhancements...")
    gaia = GAIA()
    
    print("\n📊 Testing different input patterns:")
    
    # Test 1: Structured mathematical sequence
    print("\n1. 🔢 Mathematical sequence [1,2,3,4,5,6,7,8]")
    structured_data = [1, 2, 3, 4, 5, 6, 7, 8]
    start_time = time.time()
    response1 = gaia.process_input(structured_data)
    process_time1 = time.time() - start_time
    
    print(f"   Response: {response1.response_text[:80]}...")
    print(f"   Confidence: {response1.confidence:.3f}")
    print(f"   Field Pressure: {response1.state.field_pressure:.6f}")
    print(f"   Cognitive Load: {response1.cognitive_load:.3f}")
    print(f"   Processing Time: {process_time1:.3f}s")
    
    # Test 2: Random data
    print("\n2. 🎲 Random sequence [7,2,9,1,5,3,8,4]")
    random_data = [7, 2, 9, 1, 5, 3, 8, 4]
    start_time = time.time()
    response2 = gaia.process_input(random_data)
    process_time2 = time.time() - start_time
    
    print(f"   Response: {response2.response_text[:80]}...")
    print(f"   Confidence: {response2.confidence:.3f}")
    print(f"   Field Pressure: {response2.state.field_pressure:.6f}")
    print(f"   Cognitive Load: {response2.cognitive_load:.3f}")
    print(f"   Processing Time: {process_time2:.3f}s")
    
    # Test 3: Complex structured data
    print("\n3. 📈 Complex pattern {'pattern': 'fibonacci', 'values': [1,1,2,3,5,8]}")
    complex_data = {"pattern": "fibonacci", "values": [1, 1, 2, 3, 5, 8]}
    start_time = time.time()
    response3 = gaia.process_input(complex_data)
    process_time3 = time.time() - start_time
    
    print(f"   Response: {response3.response_text[:80]}...")
    print(f"   Confidence: {response3.confidence:.3f}")
    print(f"   Field Pressure: {response3.state.field_pressure:.6f}")
    print(f"   Cognitive Load: {response3.cognitive_load:.3f}")
    print(f"   Processing Time: {process_time3:.3f}s")
    
    # Analysis
    print("\n" + "=" * 60)
    print("📊 PATTERN ANALYSIS")
    print("=" * 60)
    
    print(f"🔍 Pattern Detection Capabilities:")
    print(f"   Structured vs Random Confidence Difference: {abs(response1.confidence - response2.confidence):.3f}")
    print(f"   Field Pressure Variation: {max(response1.state.field_pressure, response2.state.field_pressure, response3.state.field_pressure) - min(response1.state.field_pressure, response2.state.field_pressure, response3.state.field_pressure):.6f}")
    print(f"   Cognitive Load Adaptation: {max(response1.cognitive_load, response2.cognitive_load, response3.cognitive_load) - min(response1.cognitive_load, response2.cognitive_load, response3.cognitive_load):.3f}")
    
    # System State
    print(f"\n🧠 Current GAIA System State:")
    print(f"   Total Processing Cycles: {response3.state.processing_cycles}")
    print(f"   Total Collapses: {response3.state.total_collapses}")
    print(f"   Active Signals: {response3.state.active_signals}")
    print(f"   Cognitive Integrity: {response3.state.cognitive_integrity:.3f}")
    print(f"   Memory Coherence: {response3.state.memory_coherence:.3f}")
    
    # PAC Enhancement Status
    print(f"\n⚡ PAC Enhancement Status:")
    try:
        from PACEngine.core.conservation_math import ConservationLaws
        print("   ✅ PAC Engine Available - Enhanced mode active")
        print("   🔬 Conservation laws: ENFORCED")
        print("   🌟 Emergence detection: ACTIVE")
        print("   📐 Geometric enhancement: ENABLED")
    except ImportError:
        print("   ⚠️  PAC Engine Not Available - Standard mode")
        print("   📊 Fallback dynamics: ACTIVE")
        print("   🔄 Graceful degradation: SUCCESSFUL")
    
    print("\n" + "=" * 60)
    print("✨ GAIA v2.0 Demonstration Complete")

if __name__ == "__main__":
    demonstrate_pac_enhanced_gaia()