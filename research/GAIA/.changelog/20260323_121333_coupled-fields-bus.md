# Coupled Fields Bus — Spike F promoted to core

## Summary

Promoted CoupledFieldsBus from `spikes/resonance_bus/spike_f_coupled_fields.py` to
`src/gaia/core/coupled_fields_bus.py`. This is the third bus variant alongside
ConservationBus (sequential) and ResonanceBus (broadcast+merge).

## What it does

Each module maintains a persistent resonance state that acts as a spectral lens,
filtering the raw input to give each module a unique perspective. Modules are coupled
like oscillators on a shared medium — they influence each other's evolution but maintain
distinct frequency domains. QBE regulates coupling strength via QPL oscillation.

## Why

Spikes C-E (broadcast+superpose) solved the "who gets heard" problem but created a new
one: superposition homogenizes the signal. Memory's PAC tree couldn't differentiate
stimulus classes (retrieval resonance = 1.000 for all classes, zero preference signal).

Coupled Fields solves this by giving each module its own channel (Peter's "WiFi" insight).

## Benchmark results vs Sequential baseline

- Preference: -7% (vs -80% to -90% for all other broadcast buses)
- Surprise: +783% (best of all spikes)
- Habituation: +16% (only spike that improves on Sequential)
- Adaptation: 18 ticks (matches Sequential's 17, vs 25 for all others)
- Memory preference range: 0.2381 (vs 0.000 for all other broadcast buses)

## Files changed

- `src/gaia/core/coupled_fields_bus.py` — new bus implementation
- `src/gaia/core/__init__.py` — added CoupledFieldsBus, CoupledWeight, CoupledFieldState exports
- `src/gaia/body/ablation.py` — added `make_coupled_fields_bus()` factory
- `tests/test_coupled_fields_bus.py` — 36 tests across 15 test classes
- `.spec/gaia-v2.spec.md` — added Section 2.5 Dispatch Variants
