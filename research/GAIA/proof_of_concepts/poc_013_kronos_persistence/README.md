# POC-013: Kronos Persistent Memory

## Objective

Demonstrate Fracton's Kronos storage backend for persistent PAC memory - enabling patterns to survive process restarts.

## Key Features

1. **Save/Load PAC Nodes** - Individual nodes persist to disk
2. **Episode Tracking** - Save complete substrate state as episodes
3. **Temporal Queries** - Find patterns by creation time
4. **Crystallized Patterns** - Important patterns marked and indexed

## Architecture

```
kronos_data/
└── {namespace}/
    ├── indices/
    │   ├── temporal_index.json    # Time → node mapping
    │   └── crystallized_index.json # Important patterns
    ├── snapshots/
    │   ├── {doc_id}.fdo.yaml      # Node metadata
    │   └── {doc_id}_delta.npy     # Delta tensor
    └── episodes/
        └── {episode_id}/
            ├── episode.yaml       # Episode metadata
            └── nodes/             # Snapshot of all nodes
```

## FDO v2.0 Format

Field Differential Object schema for PAC nodes:

```yaml
schema_version: "fdo-v2.0"
doc_type: "pac_node"
doc_id: "0000000000000042"

node:
  id: 66
  parent_id: -1
  children_ids: []
  label: "greeting_pattern"

physics:
  potential: 0.847
  phase: "STABLE"

metadata:
  created: "2024-12-17T20:30:00Z"
  crystallized: true
  importance: 1.618

delta_file: "0000000000000042_delta.npy"
```

## Usage

```python
from fracton.storage import KronosBackend
from fracton.core import PACNode

# Create backend
backend = KronosBackend(Path("./kronos_data"), namespace="gaia")

# Save node
doc_id = backend.save_node(node, crystallized=True, importance=1.0)

# Load node
node = backend.load_node(node_id)

# Save episode (full state snapshot)
episode_id = backend.save_episode(nodes, name="learning_session")

# Restore from episode
nodes, metadata = backend.load_episode(episode_id)

# Query
recent = backend.query_recent(10)
crystallized = backend.query_crystallized(min_importance=0.5)
```

## Running the POC

```bash
cd dawn-models/research/GAIA/proof_of_concepts/poc_013_kronos_persistence
python poc_013_kronos_persistence.py
```

## Results

All tests pass:
- ✓ All node IDs match after restore
- ✓ All field values preserved exactly  
- ✓ Crystallized patterns indexed correctly
- ✓ Data survives "restart" (new backend instance)

## Integration with GAIA

```python
from fracton.storage import KronosBackend
from v4 import GAIACortex, GAIAConfig

# GAIA with persistent memory
config = GAIAConfig(field_dim=64)
kronos = KronosBackend(Path("./memory"), "gaia_cortex")

cortex = GAIACortex(config, kronos_backend=kronos)

# Now patterns persist between sessions
cortex.process("Hello")

# Save state
episode_id = cortex.save_state()

# Later: restore
cortex.restore_state(episode_id)
```

## Future Enhancements

1. **Resonance Index** - Fast similarity search on stored patterns
2. **Incremental Episodes** - Only store deltas between episodes
3. **Distributed Storage** - Multi-node Kronos for swarm agents
4. **Compression** - Sparse delta compression for large fields
