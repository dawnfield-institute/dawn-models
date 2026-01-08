"""Debug generation path."""
import torch
from model import GAIA1

model = GAIA1.load('./checkpoints/overnight_run/gaia1_best.pt', device='cuda')

# Simple step-by-step generation test
print("Step-by-step generation test:")
print("="*50)

# Start with "The quick brown"
prompt = "The quick brown"
input_ids = model.encode_text(prompt)
print(f"Prompt: '{prompt}'")
print(f"Token IDs: {input_ids.squeeze().tolist()}")

# What should come next?
patterns = model.vocab.encode(input_ids)
hidden = model.generator.forward_causal(patterns)
last_hidden = hidden[0, -1, :]

# Decode via resonance
probs, logits = model.vocab.decode_resonance(last_hidden, temperature=0.1)
print(f"Probs shape: {probs.shape}, sum: {probs.sum().item():.4f}")
print(f"Top 5 logits: {torch.topk(logits, 5)}")

# Get argmax
argmax_token = logits.argmax().item()
print(f"Argmax prediction: {argmax_token} = '{model.tokenizer.decode([argmax_token])}'")

# Now sample
sampled = model.vocab.sample(last_hidden, temperature=0.1, top_k=10)
print(f"Sampled token: {sampled.item()} = '{model.tokenizer.decode([sampled.item()])}'")

# Try 10 samples
print("\n10 samples at temp=0.1:")
for i in range(10):
    s = model.vocab.sample(last_hidden, temperature=0.1, top_k=10)
    print(f"  {i+1}: {model.tokenizer.decode([s.item()])}", end="")
print()

print("\n10 samples at temp=0.5:")
for i in range(10):
    s = model.vocab.sample(last_hidden, temperature=0.5, top_k=50)
    print(f"  {i+1}: {model.tokenizer.decode([s.item()])}", end="")
print()
