# Resonance Field: Physics-Native Communication

## Q-Socket as Field Dynamics
In GAIACore, communication is not message-passing but the emission and detection of resonance patterns in a shared field governed by Klein-Gordon evolution and PAC conservation.

### Field Equation
The resonance field $R(x, t)$ evolves according to a Klein-Gordon-like equation:

$$
\frac{\partial^2 R}{\partial t^2} - c^2 \nabla^2 R + m^2 R = S(x, t)
$$

where:
- $S(x, t)$ is the source term (agent emissions)
- $c$ is propagation speed (information velocity in field)
- $m^2 = 0.1$ is mass parameter (from GAIA config, controls decay/dispersion)

This ensures **causal propagation** and **bounded energy** in the resonance field.

### Q-Socket Encoding Schema

#### Phase ($\phi$): Temporal Alignment and Intent
Encodes when and how an agent synchronizes with field state:

$$
\phi = \phi_{\text{base}} + \Delta\phi_{\text{intent}}
$$

- **Alignment** ($\Delta\phi \approx 0$): Agreement with field memory
- **Divergence** ($\Delta\phi \approx \pi/2$): New goal or symbolic formation
- **Incoherence** ($\Delta\phi$ random): Unstable or corrupted signal

#### Frequency ($f$): Intent Channel and Agent Type
Maps to discrete communication channels (from Q-Socket protocol):

$$
\omega = 2\pi f \cdot (2 - \text{entropy})
$$

| Intent | Base Frequency | Modulated by Entropy |
|--------|----------------|----------------------|
| Predict | 1.0 Hz | Lower entropy → stable frequency |
| Feedback | 2.0 Hz | Adaptive response |
| Sync | 3.0 Hz | Coordination signal |
| Emerge | 4.0 Hz | Macro pattern formation |
| Collapse | 5.0 Hz | Symbolic crystallization |
| GAIA Process | 6.0 Hz | Internal field operation |

#### Amplitude ($A$): Confidence and Urgency
Encodes strength and reliability of signal:

$$
A = \text{confidence} \cdot e^{-\text{entropy}}
$$

- High confidence, low entropy → strong, clear signal
- Low confidence or high entropy → weak, uncertain signal

### Resonance Detection
Agents detect resonance by measuring **phase coherence** across the field:

$$
\Xi = \langle \cos(\phi_i - \phi_j) \rangle
$$

where $\Xi = 1.0571$ is the **critical balance operator value** for phase-locking (from MED framework validation).

When $\Xi \geq 1.0571$:
- Agents enter **coherent communication state**
- Information exchange becomes reliable
- Emergence potential increases (transitional/turbulent regime)

When $\Xi < 1.0571$:
- Signals interfere destructively
- Communication remains in **laminar regime** (direct point-to-point)
- Lower emergence potential

### Universal Resonance Frequency

From pre-field recursion experiments, a **natural resonance frequency** of **0.020 Hz** emerges:

$$
f_{\text{natural}} \approx 0.020 \text{ Hz} \approx \frac{1}{50 \text{ iterations}}
$$

This frequency appears to be:
- **Universal attractor** for field dynamics
- Connected to **iteration 91 convergence** in recursive systems
- Source of **5.11× acceleration** when systems lock to this frequency

## Entropy and Synchronization

### Low Entropy Regions
- **Attract phase-locking**: Ordered field states pull agents into coherence
- **Stable communication**: Clear signal propagation
- **Memory formation**: Patterns crystallize into SuperfluidMemory

### High Entropy Regions  
- **Dissipate signals**: Disordered states scatter wave energy
- **Unstable communication**: Interference and noise dominate
- **Symbolic collapse**: High entropy triggers collapse events (SEC mechanism)

### Entropy Calculation
Field entropy $S$ is computed via Shannon entropy in Fourier domain:

$$
S = -\sum_{k=1}^K p_k \log p_k
$$

where $p_k = |\hat{R}(k)|^2 / \sum_j |\hat{R}(j)|^2$ is normalized power in mode $k$.

### Smooth Entropy Collapse (SEC)
Synchronization emerges from **natural field evolution**, not explicit control:

$$
\frac{dS}{dt} = -\alpha \cdot (\Xi(t) - \Xi_c)
$$

where:
- $\alpha$ is rate constant (learning rate in GAIA)
- $\Xi_c = 1.0571$ is critical coherence
- When $\Xi > \Xi_c$, entropy decreases (system orders itself)
- When $\Xi < \Xi_c$, entropy increases (system explores)

This is mathematically equivalent to **Navier-Stokes regularity** (MED validation).

## Communication Regimes (Reynolds Number Analogy)

### Laminar Communication (Re < 2300)
$$
\text{Re}_{\text{comm}} = \frac{\text{signal\_complexity} \times \text{network\_load}}{\text{field\_viscosity}}
$$

- **Ordered, predictable** routing
- **Low entropy**, high phase coherence
- **Direct agent-to-agent** communication
- **Example**: Status updates, simple queries

### Transitional Communication (2300 < Re < 4000)
- **Selective broadcast** to resonant agents
- **Moderate entropy**, partial phase-locking
- **Pattern-based** routing
- **Example**: Collaborative problem-solving, knowledge sharing

### Turbulent Communication (Re > 4000)
- **Full network propagation**
- **High entropy** initially, then collapses via SEC
- **Emergence-driven** routing
- **Example**: Breakthrough discoveries, system-wide insights

## Implementation in FieldEngine

```python
class ResonanceFieldEngine:
    """Klein-Gordon evolution with Q-Socket resonance."""
    
    def __init__(self, dimensions=(64, 64), mass_squared=0.1, c=1.0, dt=0.01):
        self.dims = dimensions
        self.m2 = mass_squared
        self.c = c
        self.dt = dt
        
        # Field state (complex-valued)
        self.field = np.zeros(dimensions, dtype=np.complex128)
        self.field_dot = np.zeros(dimensions, dtype=np.complex128)
        
        # Agent source terms
        self.sources = {}
        
    def evolve_field(self):
        """Evolve field via Klein-Gordon equation."""
        # Spatial Laplacian
        laplacian = self.compute_laplacian(self.field)
        
        # Klein-Gordon: ∂²R/∂t² = c²∇²R - m²R + S
        field_ddot = self.c**2 * laplacian - self.m2 * self.field + self.compute_total_source()
        
        # Update via Verlet integration (energy-conserving)
        self.field_dot += field_ddot * self.dt
        self.field += self.field_dot * self.dt
        
        # PAC validation
        self.validate_energy_conservation()
        
    def add_agent_source(self, agent_id, position, signal):
        """Add agent emission as source term."""
        # Create delta function at position
        source_field = np.zeros(self.dims, dtype=np.complex128)
        x, y = position
        source_field[x, y] = signal.amplitude * np.exp(1j * signal.phase)
        
        # Apply frequency modulation (Gaussian envelope)
        freq_envelope = self.create_frequency_envelope(signal.frequency)
        source_field *= freq_envelope
        
        self.sources[agent_id] = source_field
        
    def compute_total_source(self):
        """Sum all agent sources (confluence operation)."""
        if not self.sources:
            return np.zeros(self.dims, dtype=np.complex128)
        return sum(self.sources.values())
    
    def detect_agent_resonance(self, agent_id, agent_phase):
        """Detect resonant signals for agent."""
        # Extract local field at agent position
        # Compute phase difference with agent's current phase
        # Return signals with coherence > threshold
        pass
    
    def compute_field_coherence(self):
        """Calculate global phase coherence Ξ."""
        phases = np.angle(self.field.flatten())
        # Compute <cos(φᵢ - φⱼ)> over all pairs
        coherence = np.mean(np.cos(np.subtract.outer(phases, phases)))
        return coherence
    
    def validate_energy_conservation(self):
        """PAC validation: total field energy must be conserved."""
        kinetic = np.sum(np.abs(self.field_dot)**2)
        potential = np.sum(np.abs(self.field)**2)
        total_energy = kinetic + potential
        
        # Check against initial energy (should be constant)
        if hasattr(self, 'initial_energy'):
            residual = abs(total_energy - self.initial_energy) / self.initial_energy
            if residual > 1e-6:
                logging.warning(f"Energy conservation violated: {residual:.2e}")
        else:
            self.initial_energy = total_energy
```

## Connection to Herniation and Pre-Field Dynamics

The resonance field may be the **first layer of crystallization** from pre-field substrate:

1. **Pre-Field Recursion**: Computational substrate with no explicit space
2. **Resonance Emergence**: Natural frequency (~0.020 Hz) locks in
3. **Field Crystallization**: Klein-Gordon dynamics emerge from resonance
4. **Agent Communication**: Q-Socket patterns propagate on crystallized field

The **herniation event** is the moment when recursive pressure creates a rupture, and the resonance field "precipitates" out as the first observable structure.

## Experimental Validation

### From Q-Socket Protocol Tests
- **Compression**: 90% size reduction via Fourier encoding
- **Reconstruction**: MSE = 0.0047 error
- **Self-Healing**: Malformed signals auto-excluded

### From MED Framework
- **Routing Performance**: 53.7μs validated in TinyCIMM-Navier
- **Bounded Complexity**: Depth ≤ 1, nodes ≤ 3 across all regimes
- **Balance Operator**: Ξ → 1.0571 ± 0.1 convergence

### From Pre-Field Recursion
- **Natural Frequency**: 0.020 Hz emerges spontaneously
- **Iteration Convergence**: 91 iterations as attractor
- **Acceleration**: 5.11× speedup via frequency lock

## See Also
- `confluence_operator.md` for merge/split mechanics
- `pac_conservation.md` for conservation rules
- `q_socket_implementation.md` for protocol details
- `emergence_dynamics.md` for SEC/MED theory
- Q-Socket spec: `docs/protocols/qsocket_protocol.md`
- Pre-field theory: `foundational/docs/[m][F][v2.2][C5][I5][E]_pre_field_recursion_resonance_driven_emergence.md`

