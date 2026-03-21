"""Debug the sample function."""
import torch
from model import GAIA1

model = GAIA1.load('./checkpoints/overnight_run/gaia1_best.pt', device='cuda')

prompt = 'Valentin Louis Georges'
input_ids = model.encode_text(prompt)
patterns = model.vocab.encode(input_ids)
hidden = model.generator.forward_causal(patterns)
last_hidden = hidden[0, -1, :]  # (256,)

print("last_hidden shape:", last_hidden.shape)

# Manual decode_resonance
field = last_hidden.unsqueeze(0)  # (1, 256)
field_norm = torch.norm(field, dim=1, keepdim=True)
logits = torch.mm(field, model.vocab.patterns.T)  # (1, vocab)
logits = logits / (field_norm * model.vocab._pattern_norms.unsqueeze(0) + 1e-8)
logits = logits * (256 ** 0.5)  # Scale by sqrt(dim)

print("Raw logits (before temp) top 5:", torch.topk(logits.squeeze(), 5))

# With temperature 0.1
logits_temp = logits / 0.1
probs = torch.softmax(logits_temp, dim=-1)
print("Probs sum:", probs.sum().item())
print("Probs top 5:", torch.topk(probs.squeeze(), 5))

# Sample from probs
print("\nSampling from probs directly (5 tries):")
for i in range(5):
    s = torch.multinomial(probs.squeeze(), 1).item()
    print(f"  {i+1}: {s} = '{model.tokenizer.decode([s])}'")

# Now what does vocab.sample do?
print("\nWhat vocab.sample does:")
probs2, logits2 = model.vocab.decode_resonance(last_hidden, temperature=0.1, top_k=50)
print("decode_resonance probs top 5:", torch.topk(probs2, 5))

# Sample
for i in range(5):
    s = torch.multinomial(probs2, 1).item()
    print(f"  {i+1}: {s} = '{model.tokenizer.decode([s])}'")
