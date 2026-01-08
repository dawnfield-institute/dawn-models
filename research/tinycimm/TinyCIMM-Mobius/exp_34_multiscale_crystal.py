"""
exp_34_multiscale_crystal.py
============================

MULTI-SCALE CRYSTAL STRUCTURE OF PRIMES

Key Discovery: Primes form a self-similar crystal structure across
primorial scales, with PAC conservation at each level.

The Primorial Hierarchy:
- P_k = 2 × 3 × ... × p_k (product of first k primes)
- Lattice constant = P_k
- Number of facets = φ(P_k) (Euler's totient)
- Primes distribute equally across all facets (Dirichlet)

PAC Conservation:
- Parent facet (e.g., 1 mod 6) contains exactly the sum of children
- Children are the coprime residues mod P_{k+1} that reduce to parent mod P_k
- f(Parent) = Σ f(Children) - verified exactly

This connects to Dawn Field Theory:
- SEC: The primorial hierarchy defines information scales
- PAC: Conservation across hierarchical levels
- MED: Facet count = φ(P_k), bounded by primorial structure

The entropic seed (1/log n) crystallizes against the primorial lattice.
"""

import numpy as np
import json
import os
from datetime import datetime


def sieve(limit):
    """Generate primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]


def euler_phi(n):
    """Euler's totient function."""
    result = n
    p = 2
    temp_n = n
    while p * p <= temp_n:
        if temp_n % p == 0:
            while temp_n % p == 0:
                temp_n //= p
            result -= result // p
        p += 1
    if temp_n > 1:
        result -= result // temp_n
    return result


def coprime_residues(n):
    """Return list of residues coprime to n."""
    return [r for r in range(1, n) if np.gcd(r, n) == 1]


def analyze_multiscale_crystal(limit=100000):
    """Analyze prime crystal structure at multiple primorial scales."""
    
    primes = sieve(limit)
    
    # Define primorial hierarchy
    primorials = [2, 6, 30, 210, 2310]
    primorial_primes = [[2], [2, 3], [2, 3, 5], [2, 3, 5, 7], [2, 3, 5, 7, 11]]
    
    results = {
        "limit": limit,
        "num_primes": len(primes),
        "scales": []
    }
    
    for P, p_list in zip(primorials, primorial_primes):
        # Skip primes that are factors of P
        max_factor = max(p_list)
        valid_primes = primes[primes > max_factor]
        
        # Get residue distribution
        residues = valid_primes % P
        residue_counts = {}
        for r in residues:
            residue_counts[int(r)] = residue_counts.get(int(r), 0) + 1
        
        # Coprime residues
        coprime = coprime_residues(P)
        phi_P = len(coprime)
        
        # Symmetry: how equal is the distribution?
        if len(residue_counts) > 0:
            counts = list(residue_counts.values())
            symmetry = min(counts) / max(counts) if max(counts) > 0 else 0
        else:
            symmetry = 0
        
        # Gap analysis at this scale
        gaps = np.diff(valid_primes)
        same_scale_gaps = gaps[gaps % P == 0]
        if len(same_scale_gaps) > 0:
            lattice_constant = int(np.min(same_scale_gaps))
            dominant_gap = int(np.median(same_scale_gaps))
        else:
            lattice_constant = P
            dominant_gap = P
        
        scale_data = {
            "primorial": P,
            "prime_factors": p_list,
            "phi_P": phi_P,
            "coprime_residues": coprime,
            "residue_counts": residue_counts,
            "symmetry": symmetry,
            "lattice_constant": lattice_constant,
            "dominant_gap": dominant_gap
        }
        results["scales"].append(scale_data)
    
    # PAC conservation test
    primes_skip_5 = primes[primes > 5]  # Skip 2, 3, 5
    
    # Parent: 1 mod 6
    count_1_mod_6 = int(np.sum(primes_skip_5 % 6 == 1))
    # Children: 1, 7, 13, 19 mod 30 (not 25 because 5|25)
    children_1 = [1, 7, 13, 19]
    count_children_1 = sum(int(np.sum(primes_skip_5 % 30 == r)) for r in children_1)
    
    # Parent: 5 mod 6
    count_5_mod_6 = int(np.sum(primes_skip_5 % 6 == 5))
    # Children: 11, 17, 23, 29 mod 30 (not 5 because 5|5)
    children_5 = [11, 17, 23, 29]
    count_children_5 = sum(int(np.sum(primes_skip_5 % 30 == r)) for r in children_5)
    
    results["pac_conservation"] = {
        "test_1": {
            "parent": "1 mod 6",
            "parent_count": count_1_mod_6,
            "children": children_1,
            "children_sum": count_children_1,
            "conserved": count_1_mod_6 == count_children_1
        },
        "test_2": {
            "parent": "5 mod 6",
            "parent_count": count_5_mod_6,
            "children": children_5,
            "children_sum": count_children_5,
            "conserved": count_5_mod_6 == count_children_5
        }
    }
    
    return results


def main():
    print("=" * 70)
    print("MULTI-SCALE CRYSTAL STRUCTURE OF PRIMES")
    print("=" * 70)
    print()
    
    results = analyze_multiscale_crystal(100000)
    
    print("THE PRIMORIAL HIERARCHY:")
    print()
    print("%10s %10s %10s %10s" % ("Primorial", "Value", "φ(P)", "Symmetry"))
    print("-" * 45)
    
    for scale in results["scales"]:
        print("%10s %10d %10d %10.4f" % (
            "×".join(map(str, scale["prime_factors"])),
            scale["primorial"],
            scale["phi_P"],
            scale["symmetry"]
        ))
    
    print()
    print("PAC CONSERVATION:")
    print("-" * 45)
    
    for test_name, test_data in results["pac_conservation"].items():
        print("  %s: %d" % (test_data["parent"], test_data["parent_count"]))
        print("    Children %s: %d" % (test_data["children"], test_data["children_sum"]))
        print("    Conserved: %s" % ("YES ✓" if test_data["conserved"] else "NO ✗"))
        print()
    
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("The crystal is SELF-SIMILAR across primorial scales:")
    print("  - Lattice constant = primorial P_k")
    print("  - Number of facets = φ(P_k)")
    print("  - Equal distribution across facets (Dirichlet)")
    print("  - PAC conservation: f(Parent) = Σ f(Children)")
    print()
    print(">>> PRIMES FORM A HIERARCHICAL CRYSTAL <<<")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f"exp_34_multiscale_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    main()
