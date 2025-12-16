# POC-006: Memory Persistence

## Status: 🔄 In Progress

## Hypothesis

Field patterns should persist and remain retrievable over extended sequences,
unlike attention-based memory which degrades with distance.

## Key Questions

1. Do stored patterns survive many new pattern injections?
2. Can we retrieve old patterns accurately after 100+ new ones?
3. Does field superposition preserve individual pattern identity?
4. Is there graceful degradation or catastrophic forgetting?

## Building On

- **POC-004 v6 encoder**: Geometric preservation
- **POC-005 generation**: Transition learning persistence
- **POC-003 attention**: Resonance-based retrieval

## Success Criteria

- [ ] Retrieve patterns after 100+ intervening patterns
- [ ] Pattern accuracy > 80% at depth 100
- [ ] No catastrophic forgetting (gradual decay only)
- [ ] Conservation maintained through storage/retrieval

## Experiments

| Exp | Name | Goal | Status |
|-----|------|------|--------|
| 01 | Storage Depth | How many patterns can we store? | 🔄 |
| 02 | Retrieval Accuracy | Can we recall old patterns? | 📝 |
| 03 | Interference | Do new patterns destroy old? | 📝 |
| 04 | Long-Range | 1000+ pattern persistence | 📝 |
