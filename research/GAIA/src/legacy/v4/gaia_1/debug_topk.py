"""Debug the top-k filtering."""
import torch
import math
from model import GAIA1

model = GAIA1.load('./checkpoints/overnight_run/gaia1_best.pt', device='cuda')

prompt = 'Valentin Louis Georges'
input_ids = model.encode_text(prompt)
patterns = model.vocab.encode(input_ids)
hidden = model.generator.forward_causal(patterns)
last_hidden = hidden[0, -1, :]  # (256,)

# Simulate decode_resonance step by step
field = last_hidden.unsqueeze(0)  # (1, 256)
field_norm = torch.norm(field, dim=1, keepdim=True)
logits = torch.mm(field, model.vocab.patterns.T)  # (1, vocab)
logits = logits / (field_norm * model.vocab._pattern_norms.unsqueeze(0) + 1e-8)
logits = logits * math.sqrt(256)  # Scale by sqrt(dim)

print("After cosine + scale, logits stats:")
print(f"  min: {logits.min().item():.4f}, max: {logits.max().item():.4f}")

# Apply temperature
temperature = 0.1
logits = logits / temperature
print(f"\nAfter temperature {temperature}:")
print(f"  min: {logits.min().item():.4f}, max: {logits.max().item():.4f}")

# Top-k filtering
top_k = 50
values, indices = torch.topk(logits, top_k, dim=-1)
print(f"\nTop {top_k} values: {values[0, :5]}")

mask = torch.zeros_like(logits).scatter_(-1, indices, 1.0)
logits_masked = logits * mask + (1 - mask) * float('-inf')
print(f"\nAfter masking, non-inf count: {(logits_masked > float('-inf')).sum().item()}")

# Softmax
probs = torch.softmax(logits_masked, dim=-1)
print(f"\nProbs stats:")
print(f"  sum: {probs.sum().item():.4f}")
print(f"  has nan: {torch.isnan(probs).any().item()}")
print(f"  has inf: {torch.isinf(probs).any().item()}")
print(f"  top 5: {torch.topk(probs.squeeze(), 5)}")

# The issue is temperature 0.1 makes logits too large
# 9.2878 / 0.1 = 92.878 → exp(92.878) overflows!
print(f"\nProblem: exp({values[0, 0].item():.2f}) = overflow!")
print(f"  exp(700) overflows, exp(100) = {math.exp(100):.2e}")
