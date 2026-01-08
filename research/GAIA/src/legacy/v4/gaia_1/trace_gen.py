"""Trace through generation step by step."""
import torch
from model import GAIA1

model = GAIA1.load('./checkpoints/overnight_run/gaia1_best.pt', device='cuda')

# Use the Wikipedia example that worked
prompt = 'Valentin Louis Georges'
input_ids = model.encode_text(prompt)
print(f"Prompt: '{prompt}'")
print(f"Token IDs: {input_ids.squeeze().tolist()}")

# Step 1: Encode to patterns
patterns = model.vocab.encode(input_ids)
print(f"Patterns shape: {patterns.shape}")

# Step 2: Forward causal
hidden = model.generator.forward_causal(patterns)
print(f"Hidden shape: {hidden.shape}")

# Step 3: Get last position
last_hidden = hidden[:, -1, :]
print(f"Last hidden shape: {last_hidden.shape}")

# Step 4: Decode resonance
probs, logits = model.vocab.decode_resonance(last_hidden.squeeze(0), temperature=1.0)
print(f"Logits shape: {logits.shape}")
print(f"Top 5 logits: {torch.topk(logits, 5)}")

# Argmax should give us the next token
argmax = logits.argmax().item()
print(f"Argmax: {argmax} = '{model.tokenizer.decode([argmax])}'")

# Now what does sample do?
print("\nTrying sample function:")
for temp in [0.1, 0.5, 1.0]:
    samples = []
    for _ in range(5):
        s = model.vocab.sample(last_hidden.squeeze(0), temperature=temp, top_k=50)
        samples.append(model.tokenizer.decode([s.item()]))
    print(f"  temp={temp}: {samples}")

# Check if sample is using the right hidden state
print("\nCompare: what does position 2 predict vs what sample gets:")
pos2_hidden = hidden[0, 2, :]  # Position 2 = "Georges"
_, logits2 = model.vocab.decode_resonance(pos2_hidden)
print(f"Position 2 argmax: {model.tokenizer.decode([logits2.argmax().item()])}")

# Position 3 = last position
pos3_hidden = hidden[0, -1, :]
_, logits3 = model.vocab.decode_resonance(pos3_hidden)
print(f"Position -1 argmax: {model.tokenizer.decode([logits3.argmax().item()])}")
