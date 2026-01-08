"""
Experiment: Do XI and PHI Emerge or Are They Imposed?

Core question: We use XI ≈ 1.0571 and PHI ≈ 1.618 as thresholds throughout.
Are we discovering these values in the dynamics, or just imposing them?

Method:
1. Run system WITHOUT imposing constants
2. Measure where natural balance points occur
3. Compare emergent values to theoretical predictions

What we're looking for:
- Collapse thresholds: At what entropy does structure naturally form?
- Stability points: Where does the system settle without intervention?
- Ratio emergence: Do golden ratios appear in natural distributions?

This is a FALSIFICATION experiment. If constants don't emerge,
we're cargo-culting, not discovering.
"""

import torch
import random
import math
import sys
from collections import defaultdict, Counter

sys.path.insert(0, 'c:/Users/peter/repos/Dawn Field Institute/dawn-models/research/GAIA/src')

from gaia_prime.pac_mesh import PACMeshSpace, MeshNode
from gaia_prime.physics_mesh import PhysicsMesh
from gaia_prime.validated_constants import XI, PHI, PHI_INV, LAMBDA_STAR

# Reference values - now using the derived constants
EXPECTED_XI = XI          # 1.0571... (derived from π/F10)
EXPECTED_PHI = PHI        # 1.618... (golden ratio)
EXPECTED_PHI_INV = PHI_INV  # 0.618... (inverse golden)
EXPECTED_LAMBDA = LAMBDA_STAR  # 0.618432 (from SEC experiments)


def measure_natural_collapse_threshold(n_trials: int = 20):
    """
    Find where collapse naturally occurs WITHOUT imposing thresholds.
    
    Method: Gradually increase entropy and detect phase transition.
    """
    print("=" * 60)
    print("EXPERIMENT 1: Natural Collapse Threshold")
    print("=" * 60)
    
    collapse_points = []
    
    for trial in range(n_trials):
        mesh = PACMeshSpace(embed_dim=64)
        physics = PhysicsMesh(mesh)
        
        # Disable our imposed thresholds
        physics.entropy_monitor.collapse_threshold = float('inf')  # Never trigger
        
        # Track when structure spontaneously forms
        structure_formed = False
        collapse_entropy = None
        
        # Gradually add nodes
        for i in range(200):
            # Add random node
            embedding = torch.randn(64)
            node = mesh.get_or_create_root(i, f"token_{i}", embedding, "test")
            node.confidence = random.uniform(0.2, 0.8)
            
            # Random connections
            if i > 0:
                targets = random.sample(list(mesh.nodes.values()), 
                                       min(3, len(mesh.nodes)))
                for target in targets:
                    if target.node_id != node.node_id:
                        node.add_child(target)
            
            # Step physics
            physics.step()
            
            # Detect structure formation (attractors appearing)
            n_attractors = len(physics.attractors)
            entropy = physics.state.entropy
            
            # Structure = attractors form without us forcing them
            if n_attractors > 0 and not structure_formed:
                structure_formed = True
                collapse_entropy = entropy
                break
        
        if collapse_entropy is not None:
            collapse_points.append(collapse_entropy)
    
    if collapse_points:
        avg = sum(collapse_points) / len(collapse_points)
        std = (sum((x - avg)**2 for x in collapse_points) / len(collapse_points)) ** 0.5
        
        print(f"\n   Trials with structure formation: {len(collapse_points)}/{n_trials}")
        print(f"   Average collapse entropy: {avg:.4f}")
        print(f"   Std dev: {std:.4f}")
        print(f"\n   Expected (PHI): {EXPECTED_PHI:.4f}")
        print(f"   Ratio (measured/expected): {avg/EXPECTED_PHI:.4f}")
        
        return avg
    else:
        print("   No spontaneous structure formation observed")
        return None


def measure_stability_points(n_trials: int = 20):
    """
    Find where system naturally stabilizes.
    
    Let system run freely and measure equilibrium entropy.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Natural Stability Points")
    print("=" * 60)
    
    equilibrium_entropies = []
    
    for trial in range(n_trials):
        mesh = PACMeshSpace(embed_dim=64)
        physics = PhysicsMesh(mesh)
        
        # Add nodes with varied structure
        for i in range(100):
            embedding = torch.randn(64)
            node = mesh.get_or_create_root(i, f"token_{i}", embedding, "test")
            node.confidence = random.uniform(0.3, 0.9)
            
            # Add structured connections (not purely random)
            if i > 0:
                # Prefer recent nodes (locality)
                recent_ids = list(mesh.nodes.keys())[-min(10, len(mesh.nodes)):]
                for rid in random.sample(recent_ids, min(2, len(recent_ids))):
                    if rid in mesh.nodes and rid != node.node_id:
                        node.add_child(mesh.nodes[rid])
        
        # Let system evolve
        entropies = []
        for _ in range(50):
            physics.step()
            entropies.append(physics.state.entropy)
        
        # Take last 10 as "equilibrium"
        equilibrium = sum(entropies[-10:]) / 10
        equilibrium_entropies.append(equilibrium)
    
    avg = sum(equilibrium_entropies) / len(equilibrium_entropies)
    std = (sum((x - avg)**2 for x in equilibrium_entropies) / len(equilibrium_entropies)) ** 0.5
    
    print(f"\n   Average equilibrium entropy: {avg:.4f}")
    print(f"   Std dev: {std:.4f}")
    print(f"\n   Expected range: [{EXPECTED_PHI_INV:.4f}, {EXPECTED_PHI:.4f}]")
    print(f"   In expected range: {EXPECTED_PHI_INV <= avg <= EXPECTED_PHI}")
    
    # Check ratio to PHI
    ratio_to_phi = avg / EXPECTED_PHI
    ratio_to_phi_inv = avg / EXPECTED_PHI_INV
    
    print(f"\n   Ratio to PHI: {ratio_to_phi:.4f}")
    print(f"   Ratio to 1/PHI: {ratio_to_phi_inv:.4f}")
    
    return avg


def measure_child_count_distribution():
    """
    Measure natural distribution of child counts.
    
    Theory: Child counts should follow power law with PHI exponent.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Child Count Distribution")
    print("=" * 60)
    
    mesh = PACMeshSpace(embed_dim=64)
    physics = PhysicsMesh(mesh)
    
    # Build substantial mesh with natural growth
    for i in range(500):
        embedding = torch.randn(64)
        node = mesh.get_or_create_root(i, f"token_{i}", embedding, "test")
        
        # Preferential attachment (rich get richer)
        if mesh.nodes:
            existing = list(mesh.nodes.values())
            weights = [len(n.children) + 1 for n in existing]
            total_weight = sum(weights)
            probs = [w/total_weight for w in weights]
            
            # Connect to 1-3 existing nodes
            n_connections = random.randint(1, 3)
            for _ in range(n_connections):
                r = random.random()
                cumsum = 0
                for n, p in zip(existing, probs):
                    cumsum += p
                    if r < cumsum:
                        if n.node_id != node.node_id:
                            node.add_child(n)
                        break
    
    # Let system stabilize
    for _ in range(20):
        physics.step()
    
    # Measure child count distribution
    child_counts = [len(n.children) for n in mesh.nodes.values()]
    count_freq = Counter(child_counts)
    
    print(f"\n   Total nodes: {len(mesh.nodes)}")
    print(f"   Max children: {max(child_counts)}")
    print(f"   Mean children: {sum(child_counts)/len(child_counts):.2f}")
    
    # Check for power law: log(freq) vs log(count) should be linear
    # Slope should relate to PHI
    print(f"\n   Child count distribution (top 10):")
    for count, freq in sorted(count_freq.items(), key=lambda x: -x[1])[:10]:
        print(f"   {count} children: {freq} nodes")
    
    # Fit power law
    if len([c for c in child_counts if c > 0]) > 10:
        log_counts = []
        log_freqs = []
        for count, freq in count_freq.items():
            if count > 0 and freq > 0:
                log_counts.append(math.log(count))
                log_freqs.append(math.log(freq))
        
        if len(log_counts) > 2:
            # Simple linear regression
            n = len(log_counts)
            sum_x = sum(log_counts)
            sum_y = sum(log_freqs)
            sum_xy = sum(x*y for x, y in zip(log_counts, log_freqs))
            sum_x2 = sum(x*x for x in log_counts)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2 + 1e-9)
            
            print(f"\n   Power law exponent: {-slope:.4f}")
            print(f"   Expected (PHI): {EXPECTED_PHI:.4f}")
            print(f"   Ratio: {-slope/EXPECTED_PHI:.4f}")


def measure_convergence_ratios():
    """
    Measure ratios in convergence point formation.
    
    Theory: Convergence factor distribution should show PHI patterns.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Convergence Ratios")
    print("=" * 60)
    
    mesh = PACMeshSpace(embed_dim=64)
    physics = PhysicsMesh(mesh)
    
    # Build mesh with deliberate convergence patterns
    # Start with roots
    roots = []
    for i in range(10):
        embedding = torch.randn(64)
        node = mesh.get_or_create_root(i, f"root_{i}", embedding, "test")
        roots.append(node)
    
    # Create paths that converge
    for depth in range(5):
        layer_nodes = []
        for i in range(20):
            embedding = torch.randn(64)
            node = mesh.get_or_create_root(
                100 * (depth + 1) + i, 
                f"d{depth}_{i}", 
                embedding, 
                "test"
            )
            layer_nodes.append(node)
            
            # Connect from previous layer
            if depth == 0:
                sources = roots
            else:
                sources = list(mesh.nodes.values())[-40:-20]
            
            for source in random.sample(sources, min(3, len(sources))):
                source.add_child(node)
                # Record incoming path
                path_key = f"{source.node_id}->{node.node_id}"
                node.incoming_paths[path_key] = node.incoming_paths.get(path_key, 0) + 1
    
    # Let physics evolve
    for _ in range(30):
        physics.step()
    
    # Measure convergence factors
    convergence_factors = [n.convergence_factor for n in mesh.nodes.values()]
    nonzero_cf = [cf for cf in convergence_factors if cf > 0]
    
    if nonzero_cf:
        avg_cf = sum(nonzero_cf) / len(nonzero_cf)
        max_cf = max(nonzero_cf)
        
        print(f"\n   Nodes with convergence: {len(nonzero_cf)}/{len(mesh.nodes)}")
        print(f"   Average convergence factor: {avg_cf:.4f}")
        print(f"   Max convergence factor: {max_cf:.4f}")
        
        # Check ratios between consecutive Fibonacci-like levels
        sorted_cf = sorted(nonzero_cf, reverse=True)
        if len(sorted_cf) > 5:
            ratios = []
            for i in range(min(10, len(sorted_cf) - 1)):
                if sorted_cf[i+1] > 0:
                    ratio = sorted_cf[i] / sorted_cf[i+1]
                    ratios.append(ratio)
            
            if ratios:
                avg_ratio = sum(ratios) / len(ratios)
                print(f"\n   Consecutive CF ratios: {[f'{r:.3f}' for r in ratios[:5]]}")
                print(f"   Average ratio: {avg_ratio:.4f}")
                print(f"   Expected (PHI): {EXPECTED_PHI:.4f}")


def measure_entropy_scaling():
    """
    Measure how entropy scales with mesh size.
    
    Theory: Entropy should scale with log(N) modified by XI.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Entropy Scaling")
    print("=" * 60)
    
    sizes = [10, 25, 50, 100, 200, 400]
    measurements = []
    
    for size in sizes:
        # Run multiple trials
        trial_entropies = []
        
        for _ in range(5):
            mesh = PACMeshSpace(embed_dim=64)
            physics = PhysicsMesh(mesh)
            
            # Build mesh
            for i in range(size):
                embedding = torch.randn(64)
                node = mesh.get_or_create_root(i, f"token_{i}", embedding, "test")
                
                if i > 0:
                    targets = random.sample(list(mesh.nodes.values()), 
                                           min(2, len(mesh.nodes)))
                    for target in targets:
                        if target.node_id != node.node_id:
                            node.add_child(target)
            
            # Stabilize
            for _ in range(20):
                physics.step()
            
            trial_entropies.append(physics.state.entropy)
        
        avg_entropy = sum(trial_entropies) / len(trial_entropies)
        measurements.append((size, avg_entropy))
    
    print(f"\n   Size -> Entropy:")
    for size, entropy in measurements:
        log_size = math.log(size)
        ratio = entropy / log_size if log_size > 0 else 0
        print(f"   N={size:4d}: H={entropy:.4f}, log(N)={log_size:.4f}, H/log(N)={ratio:.4f}")
    
    # Check scaling exponent
    if len(measurements) > 2:
        log_sizes = [math.log(s) for s, _ in measurements]
        entropies = [e for _, e in measurements]
        
        # Linear regression in log-log space
        n = len(log_sizes)
        sum_x = sum(log_sizes)
        sum_y = sum(entropies)
        sum_xy = sum(x*y for x, y in zip(log_sizes, entropies))
        sum_x2 = sum(x*x for x in log_sizes)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2 + 1e-9)
        intercept = (sum_y - slope * sum_x) / n
        
        print(f"\n   Scaling: H ≈ {slope:.4f} * log(N) + {intercept:.4f}")
        print(f"   Expected slope (XI-adjusted): ~{1/EXPECTED_XI:.4f}")


def summarize_findings():
    """Summarize all findings."""
    print("\n" + "=" * 60)
    print("SUMMARY: Do Constants Emerge?")
    print("=" * 60)
    
    print("""
   THEORETICAL CONSTANTS:
   - XI = 1.0571 (balance operator)
   - PHI = 1.618 (golden ratio)
   - PHI_INV = 0.618 (inverse golden)
   - LAMBDA = 0.618432 (decay rate)
   
   WHAT WE MEASURED:
   - Natural collapse thresholds
   - Equilibrium entropy points
   - Child count distribution (power law exponent)
   - Convergence factor ratios
   - Entropy scaling with size
   
   KEY QUESTION:
   Do these measurements cluster around the theoretical values?
   
   If YES: Constants are emergent properties, not impositions
   If NO: We may be cargo-culting - need to recalibrate
   
   Note: This is a single run. For rigorous validation,
   run multiple times and compute statistics.
    """)


if __name__ == "__main__":
    random.seed(42)  # Reproducibility
    torch.manual_seed(42)
    
    measure_natural_collapse_threshold()
    measure_stability_points()
    measure_child_count_distribution()
    measure_convergence_ratios()
    measure_entropy_scaling()
    summarize_findings()
