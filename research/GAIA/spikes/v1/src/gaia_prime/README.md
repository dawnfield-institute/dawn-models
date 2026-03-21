# GAIA-PAC v1.0.0

**Generative AI via Information Architecture - PAC-native implementation**

The first language model built entirely from POC-validated mechanisms. No backpropagation. No gradients. Pure PAC conservation.

## Validated Components

| Component | POC | Key Finding |
|-----------|-----|-------------|
| PAC Tree (delta-only) | 007, 020 | 12.5x memory savings |
| Embedding Grafting | 016, 017, 020 | 100% cross-model success |
| No-Backprop Learning | 019, 021 | Pure counting works |
| Transition Matrix | 021, 022 | 65% hit rate at 100K vocab |
| Concentration Monitor | 023 | λ≈0.5 universal, +3.6% quality |
| φ Threshold | 024 | Critical transition at depth 4 |

## Quick Start

```python
from gaia_pac import GAIA_PAC

# Create model by grafting GPT-2 embeddings
model = GAIA_PAC.from_gpt2()

# Learn from text (no backprop!)
model.learn("Your training text here...")

# Generate with quality control
result = model.generate("Once upon a time")
print(result.text)
```

## Architecture

```
┌─────────────────────────────────────────┐
│            GAIA-PAC Model               │
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────────────────────────────┐   │
│  │     GraftedEmbeddings           │   │ ← STOLEN from GPT-2/Pythia
│  │     (frozen, 50257 × 768)       │   │    (POC-016, 017, 020)
│  └─────────────────────────────────┘   │
│              ↓                         │
│  ┌─────────────────────────────────┐   │
│  │         PACTree                 │   │ ← Delta-only storage
│  │   (byref nodes, conservation)   │   │    (POC-007, 020)
│  └─────────────────────────────────┘   │
│              ↓                         │
│  ┌─────────────────────────────────┐   │
│  │     TransitionMatrix            │   │ ← N-gram counting
│  │   (GPU-accelerated, sparse)     │   │    (POC-021, 022)
│  └─────────────────────────────────┘   │
│              ↓                         │
│  ┌─────────────────────────────────┐   │
│  │   ConcentrationMonitor          │   │ ← λ≈0.5 quality gate
│  │   (reject-resample if low)      │   │    (POC-023)
│  └─────────────────────────────────┘   │
│              ↓                         │
│  ┌─────────────────────────────────┐   │
│  │       PACGenerator              │   │ ← Text output
│  │   (greedy/sample/beam)          │   │
│  └─────────────────────────────────┘   │
│                                         │
└─────────────────────────────────────────┘
```

## Key Insight

Traditional LLMs:
- Learn embeddings from scratch (slow, expensive)
- Use backpropagation (O(n²) memory, requires gradients)
- Hope quality emerges from scale

GAIA-PAC:
- **Steal** embeddings from pretrained models (instant, free)
- **Count** transitions, no gradients (O(1) memory per token)
- **Monitor** quality with concentration (reject-resample bad outputs)

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports |
| `pac_tree.py` | Delta-only hierarchical storage |
| `embeddings.py` | Graft from GPT-2/Pythia |
| `transitions.py` | GPU-accelerated n-gram counting |
| `concentration.py` | λ≈0.5 quality monitoring |
| `generator.py` | Text generation with reject-resample |
| `model.py` | GAIA_PAC main class |
| `demo.py` | Example usage scripts |

## Dawn Field Theory Constants

```python
PHI = 1.618033988749895      # Golden ratio
PHI_INV = 0.618033988749895  # 1/φ - critical threshold
LAMBDA_HALF = 0.5            # Universal eigenvalue
```

These emerge from PAC conservation at hierarchical boundaries (POC-023, 024).

## API Reference

### GAIA_PAC

```python
# Creation
model = GAIA_PAC.from_gpt2('gpt2')          # GPT-2 embeddings
model = GAIA_PAC.from_pythia('pythia-70m')  # Pythia embeddings

# Learning
stats = model.learn(text="Training text")
stats = model.learn(tokens=token_tensor)

# Generation
result = model.generate(
    prompt="Once upon",
    max_tokens=100,
    temperature=1.0,
    top_k=50,
    top_p=0.9
)
print(result.text)
print(result.stats)

# Evaluation
perplexity = model.get_perplexity(text="Test text")
stats = model.get_statistics()

# Persistence
model.save("./checkpoint")
model = GAIA_PAC.load("./checkpoint")
```

### GraftedEmbeddings

```python
from gaia_pac import GraftedEmbeddings

embeddings = GraftedEmbeddings.from_gpt2('gpt2')
tokens = embeddings.encode("Hello world")
text = embeddings.decode(tokens)
```

### TransitionMatrix

```python
from gaia_pac import TransitionMatrix

transitions = TransitionMatrix(vocab_size=50257, context_size=5)
stats = transitions.learn_batch(token_sequences)
next_tokens, probs = transitions.predict(context)
```

### ConcentrationMonitor

```python
from gaia_pac import ConcentrationMonitor

monitor = ConcentrationMonitor(threshold=0.618)
result = monitor.analyze(depth_predictions)
if not result.is_high_quality:
    resample()
```

## Performance

Based on POC validations:

| Metric | Value | POC |
|--------|-------|-----|
| Learning speed | O(1) per token | 019 |
| Hit rate | 65% at 100K vocab | 022 |
| Quality boost | +3.6% with reject-resample | 023 |
| Memory savings | 12.5x with delta storage | 020 |

## Requirements

```
torch>=2.0
transformers>=4.30  # For embedding extraction
```

## License

MIT License - See repository root.
