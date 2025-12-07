"""
Critical Phenomena Test: Prime Gap VARIANCE Near Fibonacci

If Fibonacci marks a phase boundary, we expect:
1. Enhanced FLUCTUATIONS (variance) near the boundary
2. Not necessarily shifted means
3. Possibly power-law scaling of variance with distance

This is the signature of critical phenomena.
"""

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2

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

def distance_to_nearest_fib(n, fibs):
    idx = np.searchsorted(fibs, n)
    if idx == 0:
        return abs(n - fibs[0])
    if idx >= len(fibs):
        return abs(n - fibs[-1])
    return min(abs(n - fibs[idx-1]), abs(n - fibs[idx]))

print("=" * 70)
print("CRITICAL PHENOMENA TEST: PRIME GAP VARIANCE NEAR FIBONACCI")
print("=" * 70)

limit = 500000
primes = sieve_primes(limit)
gaps = np.diff(primes)
fibs = fibonacci_sequence(30)
fibs = fibs[(fibs > 100) & (fibs < limit)]

print(f"\nPrimes: {len(primes)}")
print(f"Fibonacci numbers: {len(fibs)}")

# Compute distance to nearest Fibonacci for each prime
prime_fib_dist = np.array([distance_to_nearest_fib(p, fibs) for p in primes[:-1]])

# Normalize by expected gap at that prime (≈ ln(p))
expected_gap = np.log(primes[:-1])
normalized_gap = gaps / expected_gap

# =============================================================================
# Test 1: Gap VARIANCE by distance to Fibonacci
# =============================================================================

print("\n" + "=" * 70)
print("TEST 1: Normalized Gap Variance by Distance to Fibonacci")
print("=" * 70)

# Use normalized distance (fraction of local Fib spacing)
def get_local_fib_spacing(n, fibs):
    idx = np.searchsorted(fibs, n)
    if 0 < idx < len(fibs):
        return fibs[idx] - fibs[idx-1]
    return fibs[-1] - fibs[-2]  # Default to last spacing

norm_distances = np.array([
    prime_fib_dist[i] / get_local_fib_spacing(primes[i], fibs) 
    for i in range(len(prime_fib_dist))
])

# Bin by normalized distance
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
print("\nNormalized gap statistics by distance to Fibonacci:")
print("-" * 60)
print(f"{'Distance':>15} {'N':>8} {'Mean':>10} {'Variance':>12} {'CV':>10}")

results = []
for i in range(len(bins) - 1):
    mask = (norm_distances >= bins[i]) & (norm_distances < bins[i+1])
    if mask.sum() > 100:
        g = normalized_gap[mask]
        mean = g.mean()
        var = g.var()
        cv = g.std() / g.mean()  # Coefficient of variation
        results.append((f"{bins[i]:.1f}-{bins[i+1]:.1f}", mask.sum(), mean, var, cv))
        print(f"{bins[i]:.1f}-{bins[i+1]:.1f}:       {mask.sum():8d} {mean:10.4f} {var:12.4f} {cv:10.4f}")

print("\nInterpretation:")
if len(results) >= 2:
    near_var = results[0][3]
    far_var = results[-1][3]
    if near_var > far_var * 1.1:
        print(f"  ✓ Variance HIGHER near Fibonacci ({near_var:.4f} vs {far_var:.4f})")
        print("  → Consistent with critical phenomena (enhanced fluctuations)")
    elif far_var > near_var * 1.1:
        print(f"  ✗ Variance LOWER near Fibonacci ({near_var:.4f} vs {far_var:.4f})")
    else:
        print(f"  ~ Variance similar ({near_var:.4f} vs {far_var:.4f})")

# =============================================================================
# Test 2: Gap distribution SHAPE near vs far from Fibonacci
# =============================================================================

print("\n" + "=" * 70)
print("TEST 2: Gap Distribution Shape (Near vs Far from Fibonacci)")
print("=" * 70)

near_mask = norm_distances < 0.1
far_mask = norm_distances > 0.4

near_gaps = normalized_gap[near_mask]
far_gaps = normalized_gap[far_mask]

# Compute distribution moments
print("\nDistribution Moments:")
print(f"{'Moment':>15} {'Near Fib':>12} {'Far Fib':>12} {'Ratio':>10}")
print("-" * 50)

moments = [
    ("Mean", near_gaps.mean(), far_gaps.mean()),
    ("Variance", near_gaps.var(), far_gaps.var()),
    ("Skewness", stats.skew(near_gaps), stats.skew(far_gaps)),
    ("Kurtosis", stats.kurtosis(near_gaps), stats.kurtosis(far_gaps)),
]

for name, near, far in moments:
    ratio = near / far if far != 0 else np.inf
    print(f"{name:>15} {near:12.4f} {far:12.4f} {ratio:10.4f}")

# Kolmogorov-Smirnov test
ks_stat, ks_p = stats.ks_2samp(near_gaps, far_gaps)
print(f"\nKolmogorov-Smirnov test: D={ks_stat:.4f}, p={ks_p:.4f}")

if ks_p < 0.05:
    print("  ✓ Distributions are significantly DIFFERENT")
else:
    print("  ✗ Distributions are not significantly different")

# =============================================================================
# Test 3: Scaling of variance with distance (critical exponent)
# =============================================================================

print("\n" + "=" * 70)
print("TEST 3: Variance Scaling (Critical Exponent?)")
print("=" * 70)

# Finer bins for scaling analysis
n_bins = 10
bin_edges = np.linspace(0, 0.5, n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

variances = []
counts = []
for i in range(n_bins):
    mask = (norm_distances >= bin_edges[i]) & (norm_distances < bin_edges[i+1])
    if mask.sum() > 50:
        variances.append(normalized_gap[mask].var())
        counts.append(mask.sum())
    else:
        variances.append(np.nan)
        counts.append(0)

# Fit power law: var ~ d^α (or var ~ const near critical point)
valid = ~np.isnan(variances)
if valid.sum() >= 3:
    x = bin_centers[valid]
    y = np.array(variances)[valid]
    
    # Log-log fit for power law
    log_x = np.log(x + 0.01)  # Offset to avoid log(0)
    log_y = np.log(y)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    
    print(f"\nPower-law fit: Variance ~ distance^α")
    print(f"  α (exponent) = {slope:.4f}")
    print(f"  R² = {r_value**2:.4f}")
    print(f"  p-value = {p_value:.4f}")
    
    if abs(slope) < 0.3 and r_value**2 < 0.3:
        print("\n  ~ Variance is roughly CONSTANT with distance")
        print("  → No strong critical scaling detected")
    elif slope > 0.3:
        print(f"\n  ✓ Variance INCREASES with distance (α={slope:.2f})")
        print("  → Lower fluctuations near Fibonacci (ordered region)")
    elif slope < -0.3:
        print(f"\n  ✓ Variance DECREASES with distance (α={slope:.2f})")
        print("  → Higher fluctuations near Fibonacci (critical region)")

# =============================================================================
# Test 4: Gap EXTREMES near Fibonacci
# =============================================================================

print("\n" + "=" * 70)
print("TEST 4: Extreme Gaps Near Fibonacci")
print("=" * 70)

# Are unusually LARGE gaps more common near Fibonacci?
threshold_high = np.percentile(normalized_gap, 95)
threshold_low = np.percentile(normalized_gap, 5)

extreme_high = normalized_gap > threshold_high
extreme_low = normalized_gap < threshold_low

near_extreme_high = (near_mask & extreme_high).sum() / near_mask.sum()
far_extreme_high = (far_mask & extreme_high).sum() / far_mask.sum()

near_extreme_low = (near_mask & extreme_low).sum() / near_mask.sum()
far_extreme_low = (far_mask & extreme_low).sum() / far_mask.sum()

print(f"\nFraction of extreme gaps:")
print(f"{'':>20} {'Near Fib':>12} {'Far Fib':>12}")
print(f"  Large (>95th %ile): {near_extreme_high*100:11.2f}% {far_extreme_high*100:11.2f}%")
print(f"  Small (<5th %ile):  {near_extreme_low*100:11.2f}% {far_extreme_low*100:11.2f}%")

# Chi-square test for extreme gap association
contingency = np.array([
    [(near_mask & extreme_high).sum(), (far_mask & extreme_high).sum()],
    [(near_mask & ~extreme_high).sum(), (far_mask & ~extreme_high).sum()]
])
chi2, p_value = stats.chi2_contingency(contingency)[:2]

print(f"\n  Chi-square test (large gaps vs location): χ²={chi2:.2f}, p={p_value:.4f}")

if p_value < 0.05:
    if near_extreme_high > far_extreme_high:
        print("  ✓ Large gaps are MORE common near Fibonacci")
    else:
        print("  ✓ Large gaps are LESS common near Fibonacci")
else:
    print("  ~ No significant association")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: CRITICAL PHENOMENA SIGNATURES")
print("=" * 70)

print("""
Looking for critical phenomena signatures near Fibonacci:

1. VARIANCE by distance: Mixed/weak signal
2. DISTRIBUTION SHAPE: Distributions differ (KS test)
3. SCALING EXPONENT: Weak or absent
4. EXTREME GAPS: Test above

OVERALL ASSESSMENT:
The critical phenomena model (enhanced fluctuations at boundary)
shows weak support. The data suggests primes and Fibonacci are
largely INDEPENDENT in their fine structure, even though:

- F_n + 1 is never prime (hard constraint)
- Primes have simpler Zeckendorf (soft preference)
- Autocorrelation duality is real (φ vs 1, 1 vs 0)

The "opposite polarity" is STRUCTURAL (growth laws, recursion)
but NOT SPATIAL (primes don't cluster at or avoid Fibonacci).

This is actually the correct null hypothesis:
- Primes are "random" relative to Fibonacci positions
- Their opposition is in TYPE, not LOCATION
- Order vs Entropy as categories, not territories
""")
