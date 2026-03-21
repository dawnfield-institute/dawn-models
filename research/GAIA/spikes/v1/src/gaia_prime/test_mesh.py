"""Test multi-model PAC mesh learning."""

from gaia_prime import PACMesh
import time

print("=" * 60)
print("PAC Mesh: Multi-Model Learning Test")
print("=" * 60)

# Create mesh
print("\n[1] Creating PAC Mesh...")
mesh = PACMesh(device='cpu')

# Learn from GPT-2
print("\n[2] Learning from GPT-2...")
start = time.time()
stats1 = mesh.learn_from_gpt2('gpt2')
print(f"    Time: {time.time() - start:.1f}s")

# Learn from GPT-2 Medium (larger model, same vocab)
print("\n[3] Learning from GPT-2 Medium...")
start = time.time()
stats2 = mesh.learn_from_gpt2('gpt2-medium')
print(f"    Time: {time.time() - start:.1f}s")

# Merge
print("\n[4] Merging models...")
mesh.merge()

# Statistics
print("\n[5] Mesh Statistics:")
stats = mesh.get_statistics()
print(f"    Sources: {[s['name'] for s in stats['sources']]}")
print(f"    Unified vocab: {stats['unified_vocab_size']}")
print(f"    Reinforced nodes: {stats['reinforced_nodes']}")
print(f"    Gap-filled nodes: {stats['gap_filled_nodes']}")
print(f"    Reinforcement ratio: {stats['reinforcement_ratio']:.1%}")

# Build model
print("\n[6] Building model from mesh...")
model = mesh.build_model()
print(f"    {model}")

# Learn from text
print("\n[7] Learning from text...")
text = """
Machine learning is a subset of artificial intelligence. 
Natural language processing uses machine learning techniques.
Deep learning is a type of machine learning with neural networks.
""" * 20
model.learn(text)

# Generate
print("\n[8] Generating text...")
prompts = ["Machine learning", "Natural language", "Deep learning"]
for prompt in prompts:
    result = model.generate(prompt, max_tokens=12)
    print(f'    "{prompt}" -> "{result.text}"')

print("\n" + "=" * 60)
print("SUCCESS! Multi-model mesh working!")
print("=" * 60)
