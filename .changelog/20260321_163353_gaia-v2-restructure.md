# GAIA v2 Restructure — Modular Intelligence Architecture

**Date**: 2026-03-21
**Scope**: research/GAIA/, roadmaps/, CLAUDE.md, README.md, meta.yaml

## What Changed

### GAIA folder restructure
- Moved ALL v1 code to `research/GAIA/spikes/v1/` (gaia_prime, legacy, 25 POCs, tests, benchmarks, docs, training, usecases)
- Created v2 scaffold: `src/gaia/` with `core/`, `modules/`, `interfaces/` packages
- Created `tests/` scaffold for v2

### New design documents
- `research/GAIA/.spec/gaia-v2.spec.md` — full v2 architecture spec (module protocol, conservation bus, SEC routing, planned modules, evaluation strategy)
- `roadmaps/gaia-v2-roadmap.md` — 9 milestones (M0-M8) from foundation through GRIM integration
- `roadmaps/gaia_roadmap.md` renamed to `gaia_roadmap_v1.md` (archived)

### Updated documentation
- `CLAUDE.md` — rewritten for v2 modular architecture, fracton as hard dependency
- `README.md` — rewritten with v2 architecture diagram, updated structure, TinyCIMM as ancestors
- `meta.yaml` — updated description and semantic scope for v2
- `research/GAIA/meta.yaml` — updated for v2 structure
- `research/GAIA/README.md` — new v2-focused README

### Vault FDOs (PENDING — Kronos MCP down during session)
- CREATE: `gaia-v2-architecture` (modelling), `proj-gaia-v2` (projects), `adr-gaia-modular-architecture` (decisions)
- UPDATE: `gaia-prime`, `gaia-poc-corpus`, `gaia-field-intelligence`, `gaia-physics-mesh`, `gaia-continuous-learning`, `tinycimm-family`, `tinycimm-boltzmann`, `tinycimm-mobius`

## Why

GAIA v1 proved PAC-conserving, zero-backprop architectures work (25 POCs validated). But the codebase was a monolithic prototype with ~120KB of reimplemented Fracton physics. The v2 redesign treats intelligence as modular composition — specialized modules connected by a conservation bus — built natively on Fracton SDK 2.1. This also supports the long-term vision of a "spinal column" for hardware control and GRIM integration.

## Nothing was deleted

All v1 code is preserved in `spikes/v1/`. The 25 validated POCs, 8.4K lines of gaia_prime, 93 docs, and all tests/benchmarks remain accessible as reference material for v2 development.
