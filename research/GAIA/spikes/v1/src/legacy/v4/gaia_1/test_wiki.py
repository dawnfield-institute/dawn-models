"""Test with real Wikipedia text."""
import torch
from model import GAIA1

model = GAIA1.load('./checkpoints/overnight_run/gaia1_best.pt', device='cuda')

# Try with a longer sequence from Wikipedia
text = 'Valentin Louis Georges Eugène Marcel Proust was a French novelist'
input_ids = model.encode_text(text)
print(f'Input ({input_ids.shape[1]} tokens): {text}')

patterns = model.vocab.encode(input_ids)
hidden = model.generator.forward_causal(patterns)

# Check predictions at each position
print('\nPredictions:')
for i in range(min(10, input_ids.shape[1])):
    h = hidden[0, i, :]
    _, logits = model.vocab.decode_resonance(h)
    pred = logits.argmax().item()
    actual = input_ids[0, i+1].item() if i+1 < input_ids.shape[1] else None
    input_tok = model.tokenizer.decode([input_ids[0, i].item()])
    pred_tok = model.tokenizer.decode([pred])
    if actual:
        actual_tok = model.tokenizer.decode([actual])
        match = '✓' if pred == actual else '✗'
        print(f'  {match} After "{input_tok}" → pred "{pred_tok}" (actual "{actual_tok}")')
    else:
        print(f'  After "{input_tok}" → pred "{pred_tok}"')
