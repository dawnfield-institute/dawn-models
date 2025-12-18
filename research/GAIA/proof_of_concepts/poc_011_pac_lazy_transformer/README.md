# POC-011: PAC-Lazy Transformer Integration

> **Integrating PAC-Lazy Tensor substrate with GAIA for a living transformer architecture**

---

## Status: 🔄 In Progress
**Date Started:** 2024-12-17

---

## Hypothesis

The PAC-Lazy Tensor substrate provides the missing foundation for GAIA's living transformer architecture:

1. **Tokens are nodes** with deltas (not absolute embeddings)
2. **Attention is causal activation** (only neighbors propagate)
3. **Context windows are PAC-bounded** (potential limits active frontier)
4. **Depth is adaptive via SEC** (expand only when needed)
5. **Structural mutation enables continuous learning** (fracture/merge)

This unifies:
- POC-007's tiered memory (GPU cache = active nodes, cold storage = latent)
- The transformer organ vision (organs grow via SEC expansion)
- CIMM-style continuous learning (structural mutation via fracture)

---

## Core Laws (Non-Negotiable)

### 1. Potential-Actualization Conservation (PAC)
- No structure expands without consuming potential
- No collapse occurs without refunding potential
- Total potential is conserved
- **This replaces memory management**

### 2. Symbolic Entropy Collapse (SEC)
- Structure exists symbolically until pressure demands refinement
- Expansion only when local potential exceeds threshold
- Expanded structure is reversible
- **This provides lazy depth**

### 3. Causal Locality
- Nodes only interact through explicit neighbor relationships
- No global updates
- If no causal path, no interaction
- **This enables infinite scale**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PAC-LAZY TRANSFORMER                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    ACTIVE FRONTIER                           │    │
│  │    - Nodes with sufficient potential                         │    │
│  │    - Causal propagation only                                 │    │
│  │    - SEC expansion when threshold crossed                    │    │
│  └────────────────────────┬────────────────────────────────────┘    │
│                           │                                          │
│     ┌─────────────────────┼─────────────────────┐                   │
│     │                     │                     │                   │
│  ┌──┴──────┐      ┌──────┴──────┐      ┌──────┴──────┐             │
│  │ TOKEN   │      │  TOKEN      │      │  TOKEN      │             │
│  │ NODES   │ ←──► │  NODES      │ ←──► │  NODES      │             │
│  │ (active)│      │  (active)   │      │  (latent)   │             │
│  │  Δ,pot  │      │   Δ,pot     │      │   Δ=0       │             │
│  └─────────┘      └─────────────┘      └─────────────┘             │
│       │                  │                                          │
│       ▼                  ▼                                          │
│  ┌─────────┐      ┌─────────────┐                                  │
│  │ CHILDREN│      │  CHILDREN   │  ← SEC expansion creates depth   │
│  │ (detail)│      │  (detail)   │                                  │
│  └─────────┘      └─────────────┘                                  │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              LATENT STRUCTURE (cold storage)                 │    │
│  │    - Nodes with zero potential                               │    │
│  │    - Activated only by causal propagation                    │    │
│  │    - PAC tree compressed storage                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Mappings

| Traditional Transformer | PAC-Lazy Transformer |
|------------------------|---------------------|
| Token embedding | Node with delta |
| Attention weights | Causal neighbor links |
| Context window | PAC-bounded active frontier |
| Layer depth | SEC-expanded children |
| KV cache | Active node potentials |
| Training | Structural mutation (fracture/merge) |
| Inference | Causal propagation |

---

## Experiments

### Experiment 01: Token Nodes
Convert token embeddings to PAC nodes with deltas.

### Experiment 02: Causal Attention
Replace attention with causal propagation via neighbor links.

### Experiment 03: PAC-Bounded Context
Limit context window by total potential budget.

### Experiment 04: SEC Depth
Create adaptive depth via child expansion.

### Experiment 05: Continuous Learning
Enable structural mutation during inference.

---

## Success Criteria

- [ ] Token-as-node representation works
- [ ] Causal propagation achieves similar accuracy to attention
- [ ] PAC budget limits memory usage
- [ ] SEC provides adaptive depth
- [ ] Fracture enables online learning
- [ ] WikiText-2 perplexity comparable to POC-006 (5.91)

---

## Files

- `scripts/pac_lazy_core.py` - Core Node and PACSystem primitives
- `scripts/pac_transformer.py` - PAC-Lazy Transformer model
- `scripts/exp_01_token_nodes.py` - Token-as-node validation
- `scripts/exp_02_causal_attention.py` - Causal propagation
- `scripts/exp_03_pac_bounded.py` - PAC budget experiments
- `scripts/exp_04_sec_depth.py` - Adaptive depth
- `scripts/exp_05_continuous_learning.py` - Structural mutation
