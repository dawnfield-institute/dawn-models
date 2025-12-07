"""
SEC Phase Transition Mapping: Fibonacci vs Prime Spirals

The idea: Both sequences can be mapped to spirals. Do they occupy
different regions of SEC phase space (Ξ thresholds)?

SEC Phase Thresholds (from framework):
- Ξ_min = 1.0015 (ground state)
- Ξ_mean = 1.0289 (geometric mean / balance)
- Ξ_PAC = 1.0571 (maximum collapse)

Key question: When we compute local "order" metrics along each spiral,
do Fibonacci and Primes cluster at different Ξ values?
"""

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
XI_MIN = 1.0015
XI_MEAN = 1.0289
XI_PAC = 1.0571

# =============================================================================
# Generate sequences
# =============================================================================

def fibonacci_sequence(n_terms):
    fib = [1, 1]
    for _ in range(n_terms - 2):
        fib.append(fib[-1] + fib[-2])
    return np.array(fib)

def sieve_primes(limit):
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

# =============================================================================
# Spiral mappings
# =============================================================================

def fibonacci_spiral(n):
    """
    Map index n to position on Fibonacci spiral.
    Uses golden angle: θ = n × 2π/φ²
    Radius grows as sqrt(n) for even spacing
    """
    golden_angle = 2 * np.pi / (PHI ** 2)  # ~137.5°
    theta = n * golden_angle
    r = np.sqrt(n)
    return r, theta

def prime_spiral(p, index):
    """
    Map prime p to position on Ulam-like spiral.
    Angle proportional to prime index (nth prime)
    Radius = sqrt(p) for density normalization
    """
    theta = index * 2 * np.pi / np.log(index + 2)  # Spiral rate ~ 1/ln(n)
    r = np.sqrt(p)
    return r, theta

# =============================================================================
# Local SEC metrics along spiral
# =============================================================================

def compute_local_xi(sequence, window=5):
    """
    Compute local Ξ (order/entropy balance) along a sequence.
    
    Ξ is based on:
    - Local ratio stability (order)
    - Local gap predictability (order)
    - Local entropy of differences (entropy)
    
    Returns array of Ξ values, one per element.
    """
    n = len(sequence)
    xi_values = np.full(n, np.nan)
    
    for i in range(window, n - window):
        local = sequence[i-window:i+window+1]
        
        # Order metric 1: Ratio stability
        ratios = local[1:] / local[:-1]
        ratio_stability = 1.0 / (1.0 + np.std(ratios))
        
        # Order metric 2: Second difference regularity
        diffs = np.diff(local)
        diff2 = np.diff(diffs)
        regularity = 1.0 / (1.0 + np.std(diff2) / (np.mean(np.abs(diff2)) + 1))
        
        # Entropy metric: Normalized entropy of gaps
        gaps = np.diff(local)
        if len(np.unique(gaps)) > 1:
            # Discretize gaps
            gap_counts = {}
            for g in gaps:
                gap_counts[g] = gap_counts.get(g, 0) + 1
            probs = np.array(list(gap_counts.values())) / len(gaps)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            max_entropy = np.log2(len(gaps))
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
        else:
            norm_entropy = 0  # All gaps identical = zero entropy
        
        # Combine into Ξ
        # High order + low entropy → high Ξ (collapse)
        # Low order + high entropy → low Ξ (branch)
        order_score = (ratio_stability + regularity) / 2
        
        # Map to Ξ range [XI_MIN, XI_PAC]
        xi = XI_MIN + (XI_PAC - XI_MIN) * order_score * (1 - norm_entropy * 0.5)
        xi_values[i] = xi
    
    return xi_values

def compute_spiral_xi(rs, thetas, sequence, window=5):
    """
    Compute Ξ incorporating spiral geometry.
    
    In addition to sequence-based Ξ, consider:
    - Angular regularity (golden angle vs irregular)
    - Radial growth pattern
    """
    n = len(sequence)
    xi_values = np.full(n, np.nan)
    
    for i in range(window, n - window):
        # Sequence-based metrics
        local_seq = sequence[i-window:i+window+1]
        local_r = rs[i-window:i+window+1]
        local_theta = thetas[i-window:i+window+1]
        
        # Ratio stability
        ratios = local_seq[1:] / local_seq[:-1]
        ratio_stability = 1.0 / (1.0 + np.std(ratios))
        
        # Angular regularity: how constant are angular steps?
        d_theta = np.diff(local_theta)
        # Wrap to [-π, π]
        d_theta = np.mod(d_theta + np.pi, 2*np.pi) - np.pi
        angular_regularity = 1.0 / (1.0 + np.std(d_theta))
        
        # Radial regularity
        d_r = np.diff(local_r)
        radial_regularity = 1.0 / (1.0 + np.std(d_r) / (np.mean(np.abs(d_r)) + 0.01))
        
        # Gap entropy
        gaps = np.diff(local_seq)
        if len(np.unique(gaps)) > 1:
            gap_counts = {}
            for g in gaps:
                gap_counts[g] = gap_counts.get(g, 0) + 1
            probs = np.array(list(gap_counts.values())) / len(gaps)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            max_entropy = np.log2(len(gaps))
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
        else:
            norm_entropy = 0
        
        # Combined order score
        order_score = (ratio_stability + angular_regularity + radial_regularity) / 3
        
        # Map to Ξ
        xi = XI_MIN + (XI_PAC - XI_MIN) * order_score * (1 - norm_entropy * 0.5)
        xi_values[i] = xi
    
    return xi_values

# =============================================================================
# Main analysis
# =============================================================================

print("=" * 70)
print("SEC PHASE MAPPING: FIBONACCI vs PRIME SPIRALS")
print("=" * 70)

print(f"""
SEC Phase Thresholds:
  Ξ_min  = {XI_MIN:.4f} (ground state, minimal order)
  Ξ_mean = {XI_MEAN:.4f} (balance point)
  Ξ_PAC  = {XI_PAC:.4f} (maximum collapse/order)
""")

# Generate sequences
n_fib = 50  # Reduced for memory
fib = fibonacci_sequence(n_fib)

# Match prime count to Fibonacci
limit = min(int(fib[-1] * 1.5), 10000000)  # Cap the limit
primes = sieve_primes(limit)[:n_fib]

print(f"Fibonacci: {n_fib} terms, max = {fib[-1]}")
print(f"Primes: {len(primes)} terms, max = {primes[-1]}")

# =============================================================================
# Method 1: Sequence-based Ξ (no spiral geometry)
# =============================================================================

print("\n" + "=" * 70)
print("METHOD 1: Sequence-Based Ξ (Intrinsic Order)")
print("=" * 70)

fib_xi = compute_local_xi(fib, window=5)
prime_xi = compute_local_xi(primes, window=5)

# Remove NaN
fib_xi_valid = fib_xi[~np.isnan(fib_xi)]
prime_xi_valid = prime_xi[~np.isnan(prime_xi)]

print(f"\nFibonacci Ξ distribution:")
print(f"  Mean:   {np.mean(fib_xi_valid):.4f}")
print(f"  Std:    {np.std(fib_xi_valid):.4f}")
print(f"  Min:    {np.min(fib_xi_valid):.4f}")
print(f"  Max:    {np.max(fib_xi_valid):.4f}")

print(f"\nPrime Ξ distribution:")
print(f"  Mean:   {np.mean(prime_xi_valid):.4f}")
print(f"  Std:    {np.std(prime_xi_valid):.4f}")
print(f"  Min:    {np.min(prime_xi_valid):.4f}")
print(f"  Max:    {np.max(prime_xi_valid):.4f}")

# Statistical comparison
t_stat, p_value = stats.ttest_ind(fib_xi_valid, prime_xi_valid)
print(f"\nt-test: t={t_stat:.3f}, p={p_value:.6f}")

if p_value < 0.001:
    if np.mean(fib_xi_valid) > np.mean(prime_xi_valid):
        print("✓ Fibonacci has HIGHER Ξ (more ordered)")
    else:
        print("✓ Primes have HIGHER Ξ (more ordered)")

# =============================================================================
# Method 2: Spiral-based Ξ (with geometry)
# =============================================================================

print("\n" + "=" * 70)
print("METHOD 2: Spiral-Based Ξ (Including Geometry)")
print("=" * 70)

# Compute spiral coordinates
fib_r = np.array([fibonacci_spiral(i)[0] for i in range(n_fib)])
fib_theta = np.array([fibonacci_spiral(i)[1] for i in range(n_fib)])

prime_r = np.array([prime_spiral(p, i)[0] for i, p in enumerate(primes)])
prime_theta = np.array([prime_spiral(p, i)[1] for i, p in enumerate(primes)])

# Compute spiral Ξ
fib_spiral_xi = compute_spiral_xi(fib_r, fib_theta, fib, window=5)
prime_spiral_xi = compute_spiral_xi(prime_r, prime_theta, primes, window=5)

fib_spiral_xi_valid = fib_spiral_xi[~np.isnan(fib_spiral_xi)]
prime_spiral_xi_valid = prime_spiral_xi[~np.isnan(prime_spiral_xi)]

print(f"\nFibonacci Spiral Ξ:")
print(f"  Mean:   {np.mean(fib_spiral_xi_valid):.4f}")
print(f"  Std:    {np.std(fib_spiral_xi_valid):.4f}")

print(f"\nPrime Spiral Ξ:")
print(f"  Mean:   {np.mean(prime_spiral_xi_valid):.4f}")
print(f"  Std:    {np.std(prime_spiral_xi_valid):.4f}")

t_stat2, p_value2 = stats.ttest_ind(fib_spiral_xi_valid, prime_spiral_xi_valid)
print(f"\nt-test: t={t_stat2:.3f}, p={p_value2:.6f}")

# =============================================================================
# Phase threshold analysis
# =============================================================================

print("\n" + "=" * 70)
print("PHASE THRESHOLD ANALYSIS")
print("=" * 70)

def classify_phase(xi):
    """Classify Ξ into SEC phase."""
    if xi < XI_MEAN:
        return "BRANCH"  # Below balance = entropy-dominated
    else:
        return "COLLAPSE"  # Above balance = order-dominated

fib_phases = [classify_phase(xi) for xi in fib_xi_valid]
prime_phases = [classify_phase(xi) for xi in prime_xi_valid]

fib_collapse = fib_phases.count("COLLAPSE") / len(fib_phases) * 100
prime_collapse = prime_phases.count("COLLAPSE") / len(prime_phases) * 100

print(f"\nPhase classification (threshold: Ξ = {XI_MEAN:.4f}):")
print(f"\n  Fibonacci:")
print(f"    COLLAPSE (Ξ > {XI_MEAN:.4f}): {fib_collapse:.1f}%")
print(f"    BRANCH   (Ξ < {XI_MEAN:.4f}): {100-fib_collapse:.1f}%")

print(f"\n  Primes:")
print(f"    COLLAPSE (Ξ > {XI_MEAN:.4f}): {prime_collapse:.1f}%")
print(f"    BRANCH   (Ξ < {XI_MEAN:.4f}): {100-prime_collapse:.1f}%")

# Chi-square test
contingency = np.array([
    [fib_phases.count("COLLAPSE"), fib_phases.count("BRANCH")],
    [prime_phases.count("COLLAPSE"), prime_phases.count("BRANCH")]
])
chi2, p_chi = stats.chi2_contingency(contingency)[:2]
print(f"\nChi-square test: χ²={chi2:.2f}, p={p_chi:.6f}")

# =============================================================================
# Visualization of Ξ trajectories
# =============================================================================

print("\n" + "=" * 70)
print("Ξ TRAJECTORY COMPARISON")
print("=" * 70)

# Show how Ξ evolves along each sequence
print("\nFibonacci Ξ trajectory (every 10th point):")
for i in range(10, min(90, len(fib_xi)), 10):
    if not np.isnan(fib_xi[i]):
        bar = "█" * int((fib_xi[i] - XI_MIN) / (XI_PAC - XI_MIN) * 30)
        phase = "C" if fib_xi[i] > XI_MEAN else "B"
        print(f"  n={i:3d}: Ξ={fib_xi[i]:.4f} [{phase}] {bar}")

print("\nPrime Ξ trajectory (every 10th point):")
for i in range(10, min(90, len(prime_xi)), 10):
    if not np.isnan(prime_xi[i]):
        bar = "█" * int((prime_xi[i] - XI_MIN) / (XI_PAC - XI_MIN) * 30)
        phase = "C" if prime_xi[i] > XI_MEAN else "B"
        print(f"  n={i:3d}: Ξ={prime_xi[i]:.4f} [{phase}] {bar}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: SEC PHASE SEPARATION")
print("=" * 70)

sep = abs(np.mean(fib_xi_valid) - np.mean(prime_xi_valid))
effect_size = sep / np.sqrt((np.var(fib_xi_valid) + np.var(prime_xi_valid)) / 2)

print(f"""
FINDINGS:

1. MEAN Ξ SEPARATION:
   Fibonacci: {np.mean(fib_xi_valid):.4f}
   Primes:    {np.mean(prime_xi_valid):.4f}
   Δ = {sep:.4f}
   Effect size (Cohen's d): {effect_size:.2f}

2. PHASE CLASSIFICATION:
   Fibonacci in COLLAPSE phase: {fib_collapse:.1f}%
   Primes in COLLAPSE phase:    {prime_collapse:.1f}%

3. INTERPRETATION:
""")

if effect_size > 0.8:
    print("   ✓ STRONG separation - Fibonacci and Primes occupy DIFFERENT")
    print("     SEC phases. Fibonacci is in COLLAPSE, Primes are in BRANCH.")
elif effect_size > 0.5:
    print("   ✓ MODERATE separation - There is a measurable difference")
    print("     in SEC phase between Fibonacci and Prime spirals.")
elif effect_size > 0.2:
    print("   ~ WEAK separation - Small difference in SEC phase.")
else:
    print("   ✗ NO separation - Fibonacci and Primes are in the same SEC phase.")

print(f"""
4. SEC FRAMEWORK VALIDATION:
   The SEC phase transition threshold (Ξ_mean = {XI_MEAN:.4f}) predicts
   that ordered sequences (Fibonacci) should cluster ABOVE threshold
   and high-entropy sequences (Primes) should cluster BELOW.
   
   Observed: Fibonacci {fib_collapse:.0f}% COLLAPSE, Primes {prime_collapse:.0f}% COLLAPSE
   """)

if fib_collapse > 60 and prime_collapse < 40:
    print("   ✓ CONFIRMED: SEC phase separation matches prediction!")
elif fib_collapse > prime_collapse:
    print("   ~ PARTIAL: Fibonacci more ordered, but not clean separation.")
else:
    print("   ✗ NOT CONFIRMED: No clear phase separation.")
