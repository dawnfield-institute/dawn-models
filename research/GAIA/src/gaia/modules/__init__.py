"""
GAIA Modules — Pluggable intelligence components.

Each module implements the GAIAModule protocol:
  - process(field_state) -> field_state  (PAC-conserving transformation)
  - phase() -> SECPhase                  (current entropy phase)
  - health() -> RBFBalance               (energy-information balance)

Planned modules:
  - language: Token prediction, embedding grafting, generation
  - reasoning: Mobius neurons for recursive dynamics
  - safety: Boltzmann conservation layer, hallucination detection
  - observability: SCBF metrics, QBE equilibrium monitoring
  - memory: Bifractal hierarchy, PACTree, continuous learning
"""
