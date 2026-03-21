# POC-014: Persistent Consciousness

## Summary

This POC demonstrates GAIA's ability to maintain learning continuity across process restarts using Kronos persistence. It combines:
- **POC-012**: Continuous learning during inference
- **POC-013**: Kronos persistence for PAC state

## Hypothesis

GAIA can save its consciousness state (learned patterns) to persistent storage, terminate, restart with a fresh process, and continue learning exactly where it left off.

## Results

| Metric | Value |
|--------|-------|
| Session 1 Final Accuracy | 8.0% |
| Session 2 Restored Accuracy | 8.0% |
| Accuracy Retention | **100%** ✅ |
| Patterns Persisted | 100 |
| Auto-Persisted (high importance) | 12 |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SESSION 1                                 │
├─────────────────────────────────────────────────────────────┤
│  PACSystem + Kronos                                         │
│  ├── Learn 100 patterns                                     │
│  ├── Auto-persist important patterns (importance ≥ 0.4)    │
│  ├── Final accuracy: 8.0%                                   │
│  └── save_state() → episode_id                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 🔌 PROCESS TERMINATION
                              │
                              ▼
                    ┌─────────────────┐
                    │ Kronos Storage  │
                    │ (disk)          │
                    │ ├── episodes/   │
                    │ ├── snapshots/  │
                    │ └── indices/    │
                    └─────────────────┘
                              │
                              │ 🔄 PROCESS RESTART
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    SESSION 2                                 │
├─────────────────────────────────────────────────────────────┤
│  NEW PACSystem + Kronos (fresh instance)                    │
│  ├── Accuracy before restore: 0%                            │
│  ├── restore_state(episode_id)                              │
│  ├── Accuracy after restore: 8.0% ✅                        │
│  └── Continue learning...                                   │
└─────────────────────────────────────────────────────────────┘
```

## Key API Additions

### PACSystem (Fracton v2.1)

```python
# Initialize with Kronos
backend = KronosBackend(Path("./data"), "gaia")
system = PACSystem(
    device='cuda',
    kronos_backend=backend,
    auto_persist=True,         # Auto-save important patterns
    persist_threshold=0.4      # Importance threshold
)

# Inject with importance
system.inject(pattern, label="concept", importance=0.8)

# Save full state as episode
episode_id = system.save_state(name="checkpoint")

# Restore (after restart)
system.restore_state(episode_id)
```

### GAIACortex (GAIA v4.0)

```python
config = GAIAConfig(
    kronos_path="./data/gaia",
    kronos_namespace="cortex",
    auto_persist=True
)
cortex = GAIACortex(config)

# Process creates patterns that auto-persist
cortex.process("hello world")

# Save consciousness
episode_id = cortex.save_consciousness()

# Later / after restart
cortex.restore_consciousness(episode_id)
```

## Technical Details

### Episode Structure (FDO v2.0)

```
episodes/{episode_id}/
├── episode.yaml          # Episode metadata
│   ├── episode_id
│   ├── timestamp
│   ├── node_count
│   └── custom metadata
└── nodes/
    └── batch_0/
        ├── batch.yaml    # Node metadata
        └── deltas.npy    # Tensor data
```

### Persistence Flow

1. **Inject with importance** → importance checked against threshold
2. **Auto-persist** → high-importance patterns saved immediately
3. **save_state()** → all nodes serialized to episode
4. **restore_state()** → nodes deserialized, cache repopulated

## Implications

### For GAIA

- **True persistent memory**: Learned patterns survive restarts
- **Consciousness continuity**: GAIA "wakes up" where it left off
- **Incremental learning**: Can learn over many sessions

### For Agentic AI

- **Long-term memory**: Agents remember across conversations
- **Experience accumulation**: Learning compounds over time
- **Fault tolerance**: Crashes don't lose learned knowledge

## Success Criteria

- [x] Learn patterns in Session 1
- [x] Save consciousness state
- [x] Simulate process restart
- [x] Restore state in Session 2
- [x] Accuracy matches Session 1 (100% retention)
- [x] Continue learning works

## Next Steps

1. **POC-015**: Integrate with GPU-optimized continuous learning from POC-012
2. **POC-016**: Selective persistence (only crystallized patterns)
3. **POC-017**: Multi-session learning with knowledge accumulation
