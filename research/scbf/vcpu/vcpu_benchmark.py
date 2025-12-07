"""
vCPU Benchmark: GPU-accelerated vCPU vs CPU-bound operations

Tests whether the vCPU paradigm can outperform traditional CPU computation
for specific workloads that match its architecture.

Test cases:
1. Fibonacci recursion (PAC-native operation)
2. Balance field computation (RBF-native)
3. Phase synchronization (network-native)
4. Symbolic entropy collapse (SEC-native)
"""

import torch
import time
import math
from typing import Tuple, List, Dict
from dataclasses import dataclass

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"vCPU Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Constants
PHI = (1 + math.sqrt(5)) / 2
TWO_THIRDS = 2/3


# ============================================================================
# BENCHMARK UTILITIES
# ============================================================================
@dataclass
class BenchmarkResult:
    name: str
    cpu_time: float
    vcpu_time: float
    speedup: float
    cpu_result: float
    vcpu_result: float
    match: bool


def warm_up_gpu():
    """Warm up GPU to avoid cold-start penalties"""
    if device.type == 'cuda':
        x = torch.randn(1000, 1000, device=device)
        for _ in range(10):
            x = torch.matmul(x, x)
        torch.cuda.synchronize()


# ============================================================================
# TEST 1: FIBONACCI RECURSION (PAC-native)
# ============================================================================
def cpu_fibonacci_field(n_elements: int, n_iterations: int) -> float:
    """
    CPU: Compute Fibonacci-structured field evolution
    This is what PAC naturally does - Fibonacci recursion
    """
    # Initialize field
    field = [0.0] * n_elements
    field[0] = 1.0
    field[1] = 1.0
    
    for _ in range(n_iterations):
        new_field = [0.0] * n_elements
        for i in range(2, n_elements):
            # Fibonacci recursion: F(n) = F(n-1) + F(n-2)
            new_field[i] = field[i-1] + field[i-2]
            # Normalize to prevent overflow
            if new_field[i] > 1e10:
                new_field[i] = new_field[i] / 1e10
        # Boundary
        new_field[0] = field[1]
        new_field[1] = field[2] if n_elements > 2 else field[1]
        field = new_field
    
    return sum(field) / n_elements


def vcpu_fibonacci_field(n_elements: int, n_iterations: int) -> float:
    """
    vCPU: Fibonacci field using tensor operations
    Leverages PAC structure for parallel Fibonacci
    """
    field = torch.zeros(n_elements, device=device)
    field[0] = 1.0
    field[1] = 1.0
    
    for _ in range(n_iterations):
        # Parallel Fibonacci using roll operations
        f_prev1 = torch.roll(field, 1)
        f_prev2 = torch.roll(field, 2)
        new_field = f_prev1 + f_prev2
        
        # Mask out first two elements (boundary)
        mask = torch.ones(n_elements, device=device)
        mask[0] = 0
        mask[1] = 0
        new_field = new_field * mask + field * (1 - mask)
        
        # Normalize
        new_field = torch.where(new_field > 1e10, new_field / 1e10, new_field)
        
        # Update boundary
        new_field[0] = field[1]
        new_field[1] = field[2] if n_elements > 2 else field[1]
        field = new_field
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    return field.mean().item()


# ============================================================================
# TEST 2: BALANCE FIELD COMPUTATION (RBF-native)
# ============================================================================
def cpu_balance_field(n_nodes: int, n_iterations: int) -> float:
    """
    CPU: Compute recursive balance field B = λ[(E-I)/(1+αM)]Φ
    """
    # Initialize
    I_vals = [1.0 + 0.1 * i for i in range(n_nodes)]
    E_vals = [1.0 - 0.05 * i for i in range(n_nodes)]
    M_vals = [0.0] * n_nodes
    phases = [2 * math.pi * i / n_nodes for i in range(n_nodes)]
    
    lambda_rbf = 1.0
    alpha = 0.1
    
    for _ in range(n_iterations):
        B_vals = []
        for i in range(n_nodes):
            # Compute Φ (Fibonacci harmonics)
            phi = math.cos(phases[i])
            phi += (1/PHI) * math.cos(PHI * phases[i])
            phi += (1/PHI**2) * math.cos(PHI**2 * phases[i])
            
            # RBF equation
            imbalance = E_vals[i] - I_vals[i]
            dampened = imbalance / (1 + alpha * abs(M_vals[i]))
            B = lambda_rbf * dampened * phi
            B_vals.append(B)
            
            # Update memory
            M_vals[i] = M_vals[i] * 0.99 + abs(B) * 0.01
            
            # Update phase
            phases[i] = (phases[i] + 0.1) % (2 * math.pi)
        
        # Update I, E based on B
        for i in range(n_nodes):
            I_vals[i] += 0.01 * B_vals[i]
            E_vals[i] -= 0.01 * B_vals[i]
    
    return sum(B_vals) / n_nodes


def vcpu_balance_field(n_nodes: int, n_iterations: int) -> float:
    """
    vCPU: Compute RBF using parallel tensor operations
    """
    I_vals = torch.tensor([1.0 + 0.1 * i for i in range(n_nodes)], device=device)
    E_vals = torch.tensor([1.0 - 0.05 * i for i in range(n_nodes)], device=device)
    M_vals = torch.zeros(n_nodes, device=device)
    phases = torch.tensor([2 * math.pi * i / n_nodes for i in range(n_nodes)], device=device)
    
    lambda_rbf = 1.0
    alpha = 0.1
    
    for _ in range(n_iterations):
        # Compute Φ (vectorized)
        phi = torch.cos(phases)
        phi = phi + (1/PHI) * torch.cos(PHI * phases)
        phi = phi + (1/PHI**2) * torch.cos(PHI**2 * phases)
        
        # RBF equation (vectorized)
        imbalance = E_vals - I_vals
        dampened = imbalance / (1 + alpha * torch.abs(M_vals))
        B_vals = lambda_rbf * dampened * phi
        
        # Update memory
        M_vals = M_vals * 0.99 + torch.abs(B_vals) * 0.01
        
        # Update phase
        phases = (phases + 0.1) % (2 * math.pi)
        
        # Update I, E
        I_vals = I_vals + 0.01 * B_vals
        E_vals = E_vals - 0.01 * B_vals
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    return B_vals.mean().item()


# ============================================================================
# TEST 3: PHASE SYNCHRONIZATION (Network-native)
# ============================================================================
def cpu_phase_sync(n_nodes: int, n_iterations: int) -> float:
    """
    CPU: Kuramoto synchronization with Fibonacci coupling
    """
    phases = [2 * math.pi * i / n_nodes for i in range(n_nodes)]
    omegas = [0.1 * (1 + 0.3 * math.sin(2 * math.pi * i / PHI)) for i in range(n_nodes)]
    
    # Build Fibonacci adjacency
    fib = [1, 1, 2, 3, 5, 8]
    adj = [[0.0] * n_nodes for _ in range(n_nodes)]
    for i in range(n_nodes):
        for f in fib:
            j = (i + f) % n_nodes
            if i != j:
                adj[i][j] = 1.0 / math.sqrt(f)
        # Normalize
        s = sum(adj[i])
        if s > 0:
            for j in range(n_nodes):
                adj[i][j] /= s
    
    K = 0.5  # Coupling strength
    
    for _ in range(n_iterations):
        new_phases = []
        for i in range(n_nodes):
            coupling = 0.0
            for j in range(n_nodes):
                coupling += adj[i][j] * math.sin(phases[j] - phases[i])
            new_phase = phases[i] + omegas[i] + K * coupling
            new_phases.append(new_phase % (2 * math.pi))
        phases = new_phases
    
    # Order parameter
    real_sum = sum(math.cos(p) for p in phases)
    imag_sum = sum(math.sin(p) for p in phases)
    r = math.sqrt(real_sum**2 + imag_sum**2) / n_nodes
    return r


def vcpu_phase_sync(n_nodes: int, n_iterations: int) -> float:
    """
    vCPU: Kuramoto synchronization with tensor operations
    """
    phases = torch.tensor([2 * math.pi * i / n_nodes for i in range(n_nodes)], device=device)
    omegas = torch.tensor([0.1 * (1 + 0.3 * math.sin(2 * math.pi * i / PHI)) 
                           for i in range(n_nodes)], device=device)
    
    # Build Fibonacci adjacency
    adj = torch.zeros((n_nodes, n_nodes), device=device)
    fib = [1, 1, 2, 3, 5, 8]
    for i in range(n_nodes):
        for f in fib:
            j = (i + f) % n_nodes
            if i != j:
                adj[i, j] = 1.0 / math.sqrt(f)
    # Normalize rows
    row_sums = adj.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1
    adj = adj / row_sums
    
    K = 0.5
    
    for _ in range(n_iterations):
        # Vectorized coupling computation
        phase_diff = phases.unsqueeze(0) - phases.unsqueeze(1)  # [n, n]
        sin_diff = torch.sin(phase_diff)  # [n, n]
        coupling = torch.sum(adj * sin_diff, dim=1)  # [n]
        
        phases = (phases + omegas + K * coupling) % (2 * math.pi)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Order parameter
    r = torch.abs(torch.mean(torch.exp(1j * phases.to(torch.complex64))))
    return r.item()


# ============================================================================
# TEST 4: SYMBOLIC ENTROPY COLLAPSE (SEC-native)
# ============================================================================
def cpu_entropy_collapse(n_symbols: int, n_iterations: int) -> float:
    """
    CPU: SEC collapse C(S) = S * e^(-β*S)
    """
    S_vals = [1.0 + 0.5 * (i / n_symbols) for i in range(n_symbols)]
    beta = 0.5
    
    for _ in range(n_iterations):
        for i in range(n_symbols):
            # SEC collapse
            collapse = S_vals[i] * math.exp(-beta * S_vals[i])
            S_vals[i] = max(0.01, S_vals[i] - 0.1 * collapse)
            
            # Fibonacci modulation of beta
            phase = 2 * math.pi * i / n_symbols
            phi_mod = math.cos(phase) + (1/PHI) * math.cos(PHI * phase)
            beta_dynamic = beta * (1 + 0.3 * phi_mod)
    
    return sum(S_vals) / n_symbols


def vcpu_entropy_collapse(n_symbols: int, n_iterations: int) -> float:
    """
    vCPU: SEC collapse with tensor operations
    """
    S_vals = torch.tensor([1.0 + 0.5 * (i / n_symbols) for i in range(n_symbols)], device=device)
    beta = 0.5
    phases = torch.tensor([2 * math.pi * i / n_symbols for i in range(n_symbols)], device=device)
    
    for _ in range(n_iterations):
        # Fibonacci modulation
        phi_mod = torch.cos(phases) + (1/PHI) * torch.cos(PHI * phases)
        beta_dynamic = beta * (1 + 0.3 * phi_mod)
        
        # SEC collapse (vectorized)
        collapse = S_vals * torch.exp(-beta_dynamic * S_vals)
        S_vals = torch.clamp(S_vals - 0.1 * collapse, min=0.01)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    return S_vals.mean().item()


# ============================================================================
# TEST 5: FULL vCPU CYCLE (All components)
# ============================================================================
def cpu_full_cycle(n_nodes: int, n_iterations: int) -> float:
    """
    CPU: Full vCPU cycle with QBE + RBF + SEC + PAC + Xi
    """
    # State per node
    states = []
    for i in range(n_nodes):
        states.append({
            'I': 1.0, 'E': 1.0, 'S': 1.0,
            'P': 1.0, 'A': 0.0, 'xi': 1.028,
            'phase': 2 * math.pi * i / n_nodes,
            'M': 0.0, 'B': 0.0
        })
    
    for _ in range(n_iterations):
        for i, state in enumerate(states):
            # QBE
            qpl = math.cos(state['phase']) + (1/PHI) * math.cos(PHI * state['phase'])
            dI = -0.3 * 0.5 * qpl * 0.1
            dE = 0.3 * 0.5 * qpl * 0.1
            
            # RBF
            phi = math.cos(state['phase']) + (1/PHI) * math.cos(PHI * state['phase'])
            imbalance = state['E'] - state['I']
            B = imbalance / (1 + 0.1 * abs(state['M'])) * phi
            state['B'] = B
            state['M'] = state['M'] * 0.99 + abs(B) * 0.01
            
            # I-E balance from RBF
            ie_ratio = state['I'] / (state['E'] + 1e-6)
            flux = 0.5 * math.tanh(-math.log(ie_ratio + 1e-6)) * 0.1
            dI += flux
            dE -= flux
            
            # SEC
            beta = 0.3 * (1 + abs(state['xi'] - 1.028) / 0.0556)
            dS = -beta * state['S'] * 0.1
            
            # PAC
            C = state['P'] + state['A']
            pa_ratio = state['A'] / C if C > 0 else 0
            transfer = 0.3 * (TWO_THIRDS - pa_ratio) * C * 0.1
            state['P'] = max(0.01, state['P'] - transfer)
            state['A'] = max(0.01, state['A'] + transfer)
            
            # Apply
            state['I'] = max(0.1, state['I'] + dI)
            state['E'] = max(0.1, state['E'] + dE)
            state['S'] = max(0.1, state['S'] + dS)
            
            # Xi
            state['xi'] = max(1.0015, min(1.0571, state['xi'] - 0.1 * (state['xi'] - 1.028) * 0.1))
            
            # Phase
            state['phase'] = (state['phase'] + 0.1) % (2 * math.pi)
    
    return sum(s['xi'] for s in states) / n_nodes


def vcpu_full_cycle(n_nodes: int, n_iterations: int) -> float:
    """
    vCPU: Full cycle with tensor operations
    """
    # Vectorized state
    I = torch.ones(n_nodes, device=device)
    E = torch.ones(n_nodes, device=device)
    S = torch.ones(n_nodes, device=device)
    P = torch.ones(n_nodes, device=device)
    A = torch.zeros(n_nodes, device=device)
    xi = torch.full((n_nodes,), 1.028, device=device)
    phases = torch.tensor([2 * math.pi * i / n_nodes for i in range(n_nodes)], device=device)
    M = torch.zeros(n_nodes, device=device)
    
    dt = 0.1
    
    for _ in range(n_iterations):
        # QBE (vectorized)
        qpl = torch.cos(phases) + (1/PHI) * torch.cos(PHI * phases)
        dI = -0.3 * 0.5 * qpl * dt
        dE = 0.3 * 0.5 * qpl * dt
        
        # RBF (vectorized)
        phi = torch.cos(phases) + (1/PHI) * torch.cos(PHI * phases)
        imbalance = E - I
        B = imbalance / (1 + 0.1 * torch.abs(M)) * phi
        M = M * 0.99 + torch.abs(B) * 0.01
        
        # I-E balance flux
        ie_ratio = I / (E + 1e-6)
        flux = 0.5 * torch.tanh(-torch.log(ie_ratio + 1e-6)) * dt
        dI = dI + flux
        dE = dE - flux
        
        # SEC (vectorized)
        beta = 0.3 * (1 + torch.abs(xi - 1.028) / 0.0556)
        dS = -beta * S * dt
        
        # PAC (vectorized)
        C = P + A
        pa_ratio = A / (C + 1e-6)
        transfer = 0.3 * (TWO_THIRDS - pa_ratio) * C * dt
        P = torch.clamp(P - transfer, min=0.01)
        A = torch.clamp(A + transfer, min=0.01)
        
        # Apply
        I = torch.clamp(I + dI, min=0.1)
        E = torch.clamp(E + dE, min=0.1)
        S = torch.clamp(S + dS, min=0.1)
        
        # Xi
        xi = torch.clamp(xi - 0.1 * (xi - 1.028) * dt, min=1.0015, max=1.0571)
        
        # Phase
        phases = (phases + 0.1) % (2 * math.pi)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    return xi.mean().item()


# ============================================================================
# RUN BENCHMARKS
# ============================================================================
def run_benchmark(name: str, cpu_func, vcpu_func, *args) -> BenchmarkResult:
    """Run a single benchmark comparing CPU vs vCPU"""
    
    # CPU timing
    start = time.perf_counter()
    cpu_result = cpu_func(*args)
    cpu_time = time.perf_counter() - start
    
    # vCPU timing
    start = time.perf_counter()
    vcpu_result = vcpu_func(*args)
    vcpu_time = time.perf_counter() - start
    
    speedup = cpu_time / vcpu_time if vcpu_time > 0 else float('inf')
    match = abs(cpu_result - vcpu_result) < 0.1 * max(abs(cpu_result), abs(vcpu_result), 0.01)
    
    return BenchmarkResult(
        name=name,
        cpu_time=cpu_time,
        vcpu_time=vcpu_time,
        speedup=speedup,
        cpu_result=cpu_result,
        vcpu_result=vcpu_result,
        match=match
    )


def run_all_benchmarks():
    """Run complete benchmark suite"""
    print("=" * 70)
    print("vCPU BENCHMARK: GPU-Accelerated vCPU vs CPU")
    print("=" * 70)
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Warm up
    print("Warming up GPU...")
    warm_up_gpu()
    print()
    
    # Test configurations (reduced to avoid CPU timeout on O(n²) ops)
    configs = [
        ("Small", 100, 500),
        ("Medium", 300, 1000),
        ("Large", 500, 2000),
    ]
    
    all_results = []
    
    for size_name, n_nodes, n_iters in configs:
        print(f"\n{'='*70}")
        print(f"SIZE: {size_name} (nodes={n_nodes}, iterations={n_iters})")
        print(f"{'='*70}")
        
        results = []
        
        # Test 1: Fibonacci
        print("\n1. Fibonacci Field (PAC-native)...")
        r = run_benchmark("Fibonacci", cpu_fibonacci_field, vcpu_fibonacci_field, n_nodes, n_iters)
        results.append(r)
        print(f"   CPU: {r.cpu_time:.4f}s | vCPU: {r.vcpu_time:.4f}s | Speedup: {r.speedup:.2f}x")
        
        # Test 2: Balance Field
        print("\n2. Balance Field (RBF-native)...")
        r = run_benchmark("RBF", cpu_balance_field, vcpu_balance_field, n_nodes, n_iters)
        results.append(r)
        print(f"   CPU: {r.cpu_time:.4f}s | vCPU: {r.vcpu_time:.4f}s | Speedup: {r.speedup:.2f}x")
        
        # Test 3: Phase Sync
        print("\n3. Phase Synchronization (Network-native)...")
        r = run_benchmark("Phase Sync", cpu_phase_sync, vcpu_phase_sync, n_nodes, n_iters)
        results.append(r)
        print(f"   CPU: {r.cpu_time:.4f}s | vCPU: {r.vcpu_time:.4f}s | Speedup: {r.speedup:.2f}x")
        
        # Test 4: Entropy Collapse
        print("\n4. Entropy Collapse (SEC-native)...")
        r = run_benchmark("SEC", cpu_entropy_collapse, vcpu_entropy_collapse, n_nodes, n_iters)
        results.append(r)
        print(f"   CPU: {r.cpu_time:.4f}s | vCPU: {r.vcpu_time:.4f}s | Speedup: {r.speedup:.2f}x")
        
        # Test 5: Full Cycle
        print("\n5. Full vCPU Cycle (All components)...")
        r = run_benchmark("Full Cycle", cpu_full_cycle, vcpu_full_cycle, n_nodes, n_iters)
        results.append(r)
        print(f"   CPU: {r.cpu_time:.4f}s | vCPU: {r.vcpu_time:.4f}s | Speedup: {r.speedup:.2f}x")
        
        all_results.append((size_name, results))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Test':<20} {'Small':>12} {'Medium':>12} {'Large':>12}")
    print("-" * 58)
    
    test_names = ["Fibonacci", "RBF", "Phase Sync", "SEC", "Full Cycle"]
    for test_name in test_names:
        row = f"{test_name:<20}"
        for size_name, results in all_results:
            for r in results:
                if r.name == test_name:
                    row += f" {r.speedup:>10.2f}x"
                    break
        print(row)
    
    # Averages
    print("-" * 58)
    row = f"{'AVERAGE':<20}"
    for size_name, results in all_results:
        avg_speedup = sum(r.speedup for r in results) / len(results)
        row += f" {avg_speedup:>10.2f}x"
    print(row)
    
    # Results match check
    print("\n" + "-" * 58)
    print("Result Validation:")
    all_match = True
    for size_name, results in all_results:
        for r in results:
            if not r.match:
                print(f"  ⚠ {r.name} ({size_name}): CPU={r.cpu_result:.4f}, vCPU={r.vcpu_result:.4f}")
                all_match = False
    if all_match:
        print("  ✓ All results match between CPU and vCPU")
    
    print("\n" + "=" * 70)
    
    # Final verdict
    final_avg = sum(r.speedup for _, results in all_results for r in results) / sum(len(results) for _, results in all_results)
    if final_avg > 1.0:
        print(f"🚀 vCPU is {final_avg:.2f}x FASTER than CPU on average!")
    else:
        print(f"📊 CPU is {1/final_avg:.2f}x faster than vCPU on average")
    print("=" * 70)


if __name__ == "__main__":
    run_all_benchmarks()
