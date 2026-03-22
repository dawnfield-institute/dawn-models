# TinyCIMM-Noether: PAC Conservation as Learning Rule

Phase A — Minimal proof-of-concept where PAC conservation IS the learning rule.

## Core Idea

Standard neural networks use gradient descent: minimize a loss function via backpropagation. TinyCIMM-Noether replaces this entirely with **PAC conservation descent** — local reduction of PAC violations at each layer. No backprop, no optimizer, no learning rate schedule.

Named after Emmy Noether: every conservation law corresponds to a symmetry. If PAC conservation is the correct symmetry of the data (established across Milestones 1–4), then the optimal update at each step is the one that minimally violates conservation — not the one that minimally reduces cross-entropy.

## Architecture (Phase A)

### Fibonacci Topology
- **Depth D=3** — derived from five independent M1 paths (not a hyperparameter)
- **Layer widths follow Fibonacci ratios**: F_n → F_{n-1} → F_{n-2} → F_{n-3}
- **Fibonacci index n** is the only free parameter, chosen by MED bounds
- No hyperparameter search — topology is derived, not searched

### PAC Conservation Descent
- At each layer k, compute PAC violation: δ = V(k) - V(k+1) - V(k+2)
- Update weights to reduce δ locally — no global gradient
- All layers update in parallel (cf. NoProp's local denoising)
- Conservation correction (scale weights) + direction correction (Hebbian)

### Hard Conservation Enforcement
- Not a soft loss penalty — exact correction terms applied after each pass
- V(k) = V(k+1) + V(k+2) must hold within tolerance ε
- Fibonacci ratios verified: V(k)/V(k+1) ≈ φ

### Forward-Only Training
- Training and inference use the same forward pass
- No separate phases, no optimizer state
- Each forward pass simultaneously infers and checks conservation

## Files

| File | Description |
|------|-------------|
| `fibonacci_topology.py` | Topology builder — Fibonacci widths, MED index selection |
| `pac_descent.py` | PAC violation reduction update rule |
| `conservation_enforcement.py` | Hard conservation checker/corrector |
| `TinyCIMM_Noether.py` | Main architecture class + SGD baseline for comparison |

## Usage

```python
from TinyCIMM_Noether import NoetherNetwork

# Create network — topology derived from input/output dims
net = NoetherNetwork(input_dim=5, output_dim=1)

# Train using PAC conservation descent
history = net.fit(X_train, Y_train, epochs=200, verbose=True)

# Predict (same forward pass as training)
Y_pred = net.predict(X_test)

# Check conservation
print(net.conservation_report(X_test[:1]))
```

## Block A Experiments

In `experiments/block_a/`:

| Experiment | Purpose | Criterion |
|-----------|---------|-----------|
| `exp_01_baseline_conservation.py` | Verify PAC holds during training | Violations decrease, enforcement works |
| `exp_02_fibonacci_topology.py` | Confirm D=3 is optimal | D=3 beats D=2 and D=4 on MSE |
| `exp_03_pac_descent_convergence.py` | Convergence vs SGD | Converges, within 3x of SGD, lower violations |
| `exp_04_conservation_vs_gradient.py` | Noether vs SGD on physics data | Noether wins on power-law + Fibonacci data |
| `exp_05_forward_only_training.py` | Verify no backward flow | Perturbation test, local dependency, frozen layers |

Run all:
```bash
cd experiments/block_a
for exp in exp_0*.py; do echo "=== $exp ===" && python "$exp"; done
```

## Dependencies

- Python 3.8+
- NumPy
- SciPy (optional, for future experiments)

No PyTorch or other deep learning frameworks — Phase A is pure Python + NumPy.

## Phase B (pending Phase A validation)

- Möbius neurons: M(z) = (az+b)/(cz+d) — 12,000× MLP advantage from TinyCIMM-Möbius
- SEC phase gates — zero-parameter, from TinyCIMM-Boltzmann
- Landauer-bounded updates — thermodynamic floor on information erasure
- KAN 2.0 integration — PAC equations compiled into topology
