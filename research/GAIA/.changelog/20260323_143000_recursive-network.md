# Recursive Multi-Agent Network — Same Physics, Every Scale

## Summary

Created `src/gaia/network/` package implementing recursive multi-agent architecture.
CoupledFieldsBus is used at every level — modules within agents, agents within networks.
Same PAC/SEC/RBF/QBE physics at every recursion level, producing emergent specialization.

## What it does

`RecursiveEntity` wraps a `CoupledFieldsBus` as a `GAIAModule`, enabling fractal nesting.
`GAIAAgent` extends this with persistent identity (spectral lens evolution), self-modification
(add/remove modules at runtime), and sub-agent spawning. `GAIANetwork` couples agents via
a network-level CoupledFieldsBus — the QSocket concept realized without new physics.

Key emergence: identical agents processing diverse signals develop spontaneously divergent
spectral lenses (validated in test_five_agent_network, 100 ticks, 5 agents).

## Why

Peter's insight: "the same fundamental rules recurring themselves and every recursion,
they become smarter." The CoupledFieldsBus architecture already implements full DFT physics.
Wrapping it as a GAIAModule and registering it in a higher-level bus makes the recursion
explicit. No new physics needed — same conservation, same coupling, same emergence.

## Files changed

- `src/gaia/network/__init__.py` — package exports
- `src/gaia/network/recursive_entity.py` — CoupledFieldsBus wrapped as GAIAModule
- `src/gaia/network/identity.py` — AgentIdentity (spectral lens, specialization tracking)
- `src/gaia/network/agent.py` — GAIAAgent (identity + self-modification + sub-agent spawning)
- `src/gaia/network/network.py` — GAIANetwork (CoupledFieldsBus of agents)
- `tests/test_recursive_entity.py` — 15 tests across 4 test classes
- `tests/test_network.py` — 29 tests across 11 test classes
- `.spec/gaia-v2.spec.md` — added Section 10: Recursive Architecture
