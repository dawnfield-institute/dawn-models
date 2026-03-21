"""
Experiment: Stress Test Entropy Dynamics

Question: Does the auto-collapse system actually trigger under real chaos?

Method:
1. Feed mesh with high-entropy input (random tokens, contradictory patterns)
2. Monitor entropy trajectory
3. Verify collapse triggers at appropriate thresholds
4. Measure entropy reduction effectiveness

Success Criteria:
- Entropy spikes above threshold trigger collapse
- Collapse actually reduces entropy
- System stabilizes at "edge of chaos" (PHI_INV to PHI range)
"""

import torch
import random
import sys
sys.path.insert(0, 'c:/Users/peter/repos/core_workspace/dawn-models/research/GAIA/src')

from gaia_prime.pac_mesh import PACMeshSpace, MeshNode
from gaia_prime.physics_mesh import PhysicsMesh
from gaia_prime.validated_constants import XI, PHI, PHI_INV
from gaia_prime.auto_collapse import AutoCollapseEngine, EntropyBalancer


def generate_chaotic_input(n_tokens: int, vocab_size: int = 1000) -> list:
    """Generate random token sequences with no coherent structure."""
    return [random.randint(0, vocab_size - 1) for _ in range(n_tokens)]


def generate_contradictory_patterns(n_patterns: int = 50) -> list:
    """
    Generate patterns where same prefix leads to different continuations.
    This is pure entropy - no learnable structure.
    """
    patterns = []
    prefixes = ["A", "B", "C", "D", "E"]
    
    for _ in range(n_patterns):
        prefix = random.choice(prefixes)
        # Same prefix, random continuation
        continuation = [random.randint(0, 99) for _ in range(5)]
        patterns.append((prefix, continuation))
    
    return patterns


def inject_chaos(mesh: PACMeshSpace, n_nodes: int = 100, embed_dim: int = 64):
    """
    Inject chaotic nodes with random connections.
    
    This simulates information overload - many nodes, sparse structure.
    """
    nodes = []
    
    for i in range(n_nodes):
        # Random embedding (no semantic structure)
        embedding = torch.randn(embed_dim)
        
        # Random token
        token_id = random.randint(0, 999)
        token_str = f"chaos_{i}"
        
        node = mesh.get_or_create_root(token_id + i * 1000, token_str, embedding, "chaos")
        node.confidence = random.uniform(0.1, 0.5)  # Low confidence
        nodes.append(node)
    
    # Create chaotic connections (many weak links)
    for node in nodes:
        # Connect to random subset
        n_connections = random.randint(3, 10)
        targets = random.sample(nodes, min(n_connections, len(nodes)))
        for target in targets:
            if target != node:
                node.add_child(target)
    
    return nodes


def run_stress_test():
    """Main stress test."""
    print("=" * 60)
    print("ENTROPY STRESS TEST")
    print("=" * 60)
    
    # Create mesh
    embed_dim = 64
    mesh = PACMeshSpace(embed_dim=embed_dim)
    physics = PhysicsMesh(mesh)
    auto = AutoCollapseEngine(physics)
    balancer = EntropyBalancer(physics)
    
    # Configure for more sensitive triggering
    auto.config.entropy_threshold = 1.0  # Lower threshold
    auto.config.min_collapse_interval = 0.0  # No cooldown
    auto.config.min_nodes_for_collapse = 10
    
    # Track metrics
    entropy_history = []
    collapse_events = []
    
    print("\n1. BASELINE (empty mesh)")
    print(f"   Entropy: {physics.state.entropy:.4f}")
    print(f"   Regime: {balancer.get_regime()}")
    
    # Phase 1: Inject chaos gradually
    print("\n2. INJECTING CHAOS (waves of random nodes)")
    
    for wave in range(5):
        print(f"\n   Wave {wave + 1}:")
        
        # Inject chaos
        inject_chaos(mesh, n_nodes=50, embed_dim=embed_dim)
        
        # Step physics multiple times
        for _ in range(10):
            physics.step()
            entropy_history.append(physics.state.entropy)
        
        current_entropy = physics.state.entropy
        regime = balancer.get_regime()
        
        print(f"   Nodes: {len(mesh.nodes)}")
        print(f"   Entropy: {current_entropy:.4f}")
        print(f"   Regime: {regime}")
        
        # Check if collapse triggers
        result = auto.step()
        if result:
            collapse_events.append({
                'wave': wave,
                'strategy': result.strategy.value,
                'reduction': result.entropy_reduction,
                'before': result.entropy_before,
                'after': result.entropy_after
            })
            print(f"   COLLAPSE TRIGGERED: {result.strategy.value}")
            print(f"   Entropy: {result.entropy_before:.4f} -> {result.entropy_after:.4f}")
    
    # Phase 2: Run until stable
    print("\n3. RUNNING UNTIL STABLE")
    
    stabilization_results = auto.run_until_stable(max_iterations=20)
    
    for i, result in enumerate(stabilization_results):
        collapse_events.append({
            'phase': 'stabilization',
            'iteration': i,
            'strategy': result.strategy.value,
            'reduction': result.entropy_reduction
        })
        print(f"   Collapse {i+1}: {result.strategy.value} "
              f"({result.entropy_before:.4f} -> {result.entropy_after:.4f})")
    
    # Phase 3: Analysis
    print("\n4. ANALYSIS")
    print("-" * 40)
    
    final_entropy = physics.state.entropy
    final_regime = balancer.get_regime()
    
    print(f"   Total nodes: {len(mesh.nodes)}")
    print(f"   Final entropy: {final_entropy:.4f}")
    print(f"   Final regime: {final_regime}")
    print(f"   Collapse events: {len(collapse_events)}")
    
    if entropy_history:
        max_entropy = max(entropy_history)
        min_entropy = min(entropy_history)
        avg_entropy = sum(entropy_history) / len(entropy_history)
        
        print(f"\n   Entropy trajectory:")
        print(f"   Max: {max_entropy:.4f}")
        print(f"   Min: {min_entropy:.4f}")
        print(f"   Avg: {avg_entropy:.4f}")
    
    # Check if we're in optimal range
    in_optimal = PHI_INV <= final_entropy <= PHI
    print(f"\n   Edge of chaos range: [{PHI_INV:.3f}, {PHI:.3f}]")
    print(f"   In optimal range: {in_optimal}")
    
    # Strategy breakdown
    if collapse_events:
        strategies = {}
        for event in collapse_events:
            s = event['strategy']
            strategies[s] = strategies.get(s, 0) + 1
        
        print(f"\n   Strategy usage:")
        for strategy, count in sorted(strategies.items(), key=lambda x: -x[1]):
            print(f"   - {strategy}: {count}")
    
    # Verdict
    print("\n5. VERDICT")
    print("-" * 40)
    
    if len(collapse_events) == 0:
        print("   ❌ FAIL: No collapses triggered despite chaos injection")
        print("   -> Entropy threshold may be too high, or chaos not chaotic enough")
    elif final_entropy > PHI * 1.5:
        print("   ⚠️ PARTIAL: Collapses triggered but system still chaotic")
        print("   -> Collapse strategies may not be effective enough")
    elif final_entropy < PHI_INV * 0.5:
        print("   ⚠️ PARTIAL: System over-collapsed (too rigid)")
        print("   -> May need to reduce collapse aggressiveness")
    else:
        print("   ✅ PASS: System reached stable edge-of-chaos regime")
        print(f"   -> Final entropy {final_entropy:.4f} in range [{PHI_INV:.3f}, {PHI:.3f}]")
    
    return {
        'entropy_history': entropy_history,
        'collapse_events': collapse_events,
        'final_entropy': final_entropy,
        'final_regime': final_regime,
        'in_optimal_range': in_optimal
    }


def run_contradiction_test():
    """Test with contradictory patterns (same input -> different outputs)."""
    print("\n" + "=" * 60)
    print("CONTRADICTION STRESS TEST")
    print("=" * 60)
    
    embed_dim = 64
    mesh = PACMeshSpace(embed_dim=embed_dim)
    physics = PhysicsMesh(mesh)
    auto = AutoCollapseEngine(physics)
    
    auto.config.entropy_threshold = 0.5
    auto.config.min_collapse_interval = 0.0
    auto.config.min_nodes_for_collapse = 5
    
    # Create prefix nodes
    print("\n1. Creating prefix nodes...")
    prefixes = {}
    for i, prefix in enumerate(["START", "A", "B", "C"]):
        emb = torch.randn(embed_dim)
        node = mesh.get_or_create_root(i, prefix, emb, "test")
        prefixes[prefix] = node
    
    # Add contradictory continuations
    print("\n2. Adding contradictory continuations...")
    
    for _ in range(20):
        # Same prefix, different random continuation
        prefix_node = prefixes["A"]
        
        # Random continuation
        cont_id = random.randint(1000, 9999)
        cont_emb = torch.randn(embed_dim)
        cont_node = mesh.get_or_create_root(cont_id, f"cont_{cont_id}", cont_emb, "test")
        
        # Link prefix to continuation
        prefix_node.add_child(cont_node)
    
    # Check entropy of prefix node
    print("\n3. Measuring prefix node entropy...")
    
    # The prefix "A" should have high entropy (many equally likely children)
    prefix_a = prefixes["A"]
    if prefix_a.children:
        counts = [c for (_, c) in prefix_a.children.values()]
        total = sum(counts)
        if total > 0:
            probs = [c/total for c in counts]
            import math
            entropy = -sum(p * math.log(p + 1e-9) for p in probs)
            print(f"   Prefix 'A' has {len(prefix_a.children)} children")
            print(f"   Local entropy: {entropy:.4f}")
            print(f"   (Max possible: {math.log(len(prefix_a.children)):.4f})")
    
    # Step physics and check for collapse
    print("\n4. Running physics...")
    
    for step in range(10):
        physics.step()
        result = auto.step()
        if result:
            print(f"   Step {step}: COLLAPSE - {result.strategy.value}")
    
    print(f"\n   Final mesh entropy: {physics.state.entropy:.4f}")
    print(f"   Attractors formed: {len(physics.attractors)}")


if __name__ == "__main__":
    results = run_stress_test()
    run_contradiction_test()
