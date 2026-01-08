"""
exp_33_prime_crystallization.py
================================

THE PRIME CRYSTALLIZATION MODEL

Hypothesis: Primes are not the mathematical structure - they are
the CRYSTALLIZATION EVENTS where an entropic seed (the 1/log(n)
density function) meets the lattice constraint (divisibility/sieve).

Key findings:
- Lattice constant: 6 (equilibrium gap)
- Two facets: 6k+1 and 6k-1 residue classes
- Facet symmetry: 0.9954 (near-perfect)
- Balance ratio: 0.25 (4x more balanced than random)
- Restoring force: k = 0.054 (Dirichlet theorem in action)
- Defect density: 25.4% (twin + cousin primes)

The analogy:
- Physical crystal atoms = Primes
- Lattice constant = 6
- Crystal facets = 6k±1 residue classes
- Crystal defects = Twin primes (gap=2), cousin primes (gap=4)
- Supersaturation = 1/log(n) density
- Crystallization process = Sieve of Eratosthenes
"""

import numpy as np
import json
import os
from datetime import datetime


def sieve(limit):
    """Generate boolean primality array up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return is_prime


def analyze_crystallization(limit=100000):
    """Full crystallization analysis of primes."""
    
    is_prime = sieve(limit)
    primes = np.where(is_prime)[0]
    gaps = np.diff(primes[2:])  # Skip 2, 3
    
    results = {
        "limit": limit,
        "num_primes": len(primes),
        "num_gaps": len(gaps)
    }
    
    # 1. Lattice constant (equilibrium gap)
    gap_counts = {}
    for g in gaps:
        gap_counts[int(g)] = gap_counts.get(int(g), 0) + 1
    
    # Find equilibrium by stress minimization
    candidate_equilibria = [4, 6, 8, 10, 12]
    stresses = {}
    for eq in candidate_equilibria:
        stress = np.mean(np.abs(gaps - eq))
        stresses[eq] = stress
    
    equilibrium_gap = min(stresses, key=stresses.get)
    results["lattice_constant"] = equilibrium_gap
    results["gap_distribution"] = {int(k): v for k, v in sorted(gap_counts.items())[:10]}
    
    # 2. Facet analysis
    facets = np.array([p % 6 for p in primes[2:]])
    f1_count = int(np.sum(facets == 1))
    f5_count = int(np.sum(facets == 5))
    symmetry = min(f1_count, f5_count) / max(f1_count, f5_count)
    
    results["facet_counts"] = {"6k+1": f1_count, "6k-1": f5_count}
    results["facet_symmetry"] = symmetry
    
    # 3. Facet transitions
    transitions = {}
    for i in range(len(facets) - 1):
        key = f"{facets[i]}->{facets[i+1]}"
        transitions[key] = transitions.get(key, 0) + 1
    
    same_facet = transitions.get("1->1", 0) + transitions.get("5->5", 0)
    diff_facet = transitions.get("1->5", 0) + transitions.get("5->1", 0)
    alternation_preference = diff_facet / same_facet if same_facet > 0 else 0
    
    results["facet_transitions"] = transitions
    results["same_facet_ratio"] = same_facet / (same_facet + diff_facet)
    results["alternation_preference"] = alternation_preference
    
    # 4. Balance analysis (random walk)
    walk = np.cumsum(np.where(facets == 1, 1, -1))
    actual_rms = float(np.sqrt(np.mean(walk**2)))
    expected_rms = float(np.sqrt(len(walk)))
    balance_ratio = actual_rms / expected_rms
    
    results["walk_rms_actual"] = actual_rms
    results["walk_rms_expected"] = expected_rms
    results["balance_ratio"] = balance_ratio
    results["balance_factor"] = 1 / balance_ratio  # How many times better than random
    
    # 5. Restoring force
    facet_sign = np.where(facets[1:] == 1, 1, -1)
    walk_before = walk[:-1]
    corr = float(np.corrcoef(walk_before, facet_sign)[0, 1])
    
    results["restoring_force_k"] = abs(corr)
    results["restoring_correlation"] = corr
    
    # 6. Defect analysis
    twin_count = int(np.sum(gaps == 2))
    cousin_count = int(np.sum(gaps == 4))
    sexy_count = int(np.sum(gaps == 6))
    defect_count = twin_count + cousin_count
    defect_density = defect_count / len(gaps)
    
    results["defects"] = {
        "twin_primes_gap2": twin_count,
        "cousin_primes_gap4": cousin_count,
        "sexy_primes_gap6": sexy_count,
        "total_defects": defect_count,
        "defect_density": defect_density
    }
    
    # 7. Gap mod 6 structure (lattice constraint)
    gap_mod6 = {}
    for g in gaps:
        m = int(g % 6)
        gap_mod6[m] = gap_mod6.get(m, 0) + 1
    
    results["gap_mod6_distribution"] = gap_mod6
    
    # 8. PNT verification (the entropic seed)
    expected_primes = limit / np.log(limit)
    pnt_ratio = len(primes) / expected_primes
    
    results["pnt_expected"] = expected_primes
    results["pnt_ratio"] = pnt_ratio
    
    # 9. Seed entropy (is the 6k+1 vs 6k-1 choice random?)
    seed = np.where(facets == 1, 1, -1)
    seed_autocorr = np.correlate(seed[:1000], seed[:1000], mode='full')
    seed_autocorr = seed_autocorr[len(seed_autocorr)//2:]
    seed_autocorr = seed_autocorr / seed_autocorr[0]
    
    results["seed_autocorrelation"] = {
        "lag_1": float(seed_autocorr[1]),
        "lag_2": float(seed_autocorr[2]),
        "lag_3": float(seed_autocorr[3])
    }
    
    return results


def main():
    print("=" * 60)
    print("PRIME CRYSTALLIZATION MODEL")
    print("=" * 60)
    print()
    
    results = analyze_crystallization(100000)
    
    print("THE CRYSTAL ANALOGY:")
    print("-" * 60)
    print()
    print(f"Physical Crystal    | Prime Crystal")
    print(f"-" * 20 + "|" + "-" * 39)
    print(f"Atoms               | Primes")
    print(f"Lattice constant    | {results['lattice_constant']} (gap equilibrium)")
    print(f"Facets              | 6k+1 and 6k-1 residue classes")
    print(f"Defects             | Twin/cousin primes (gap=2,4)")
    print(f"Surface energy      | Restoring force k = {results['restoring_force_k']:.4f}")
    print()
    
    print("QUANTIFIED PROPERTIES:")
    print("-" * 60)
    print()
    print(f"1. LATTICE CONSTANT: {results['lattice_constant']}")
    print(f"   Gap=6 count: {results['defects']['sexy_primes_gap6']}")
    print()
    print(f"2. FACET SYMMETRY: {results['facet_symmetry']:.4f}")
    print(f"   6k+1: {results['facet_counts']['6k+1']}, 6k-1: {results['facet_counts']['6k-1']}")
    print()
    print(f"3. DEFECT DENSITY: {results['defects']['defect_density']*100:.1f}%")
    print(f"   Twins: {results['defects']['twin_primes_gap2']}, Cousins: {results['defects']['cousin_primes_gap4']}")
    print()
    print(f"4. BALANCE RATIO: {results['balance_ratio']:.4f}")
    print(f"   Primes are {results['balance_factor']:.2f}x more balanced than random")
    print()
    print(f"5. RESTORING FORCE: k = {results['restoring_force_k']:.4f}")
    print(f"   (Dirichlet theorem in action)")
    print()
    
    print("SEED ANALYSIS:")
    print("-" * 60)
    print(f"Seed autocorrelation lag 1: {results['seed_autocorrelation']['lag_1']:.4f}")
    print(f"   (Negative = alternating pattern in facets)")
    print()
    
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print()
    print("PRIMES = (entropic seed 1/log(n)) × (sieve lattice constraint)")
    print()
    print("The seed determines HOW MANY primes exist.")
    print("The lattice determines WHERE they can appear.")
    print("The primes ARE the crystallization events.")
    print()
    print(">>> PRIMES ARE THE CRYSTAL FACETS OF NUMBER <<<")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f"exp_33_crystallization_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    main()
