"""Test extracting learned knowledge from a pretrained model."""

from gaia_prime.pac_mesh import ModelKnowledgeExtractor, MultiModelMesh

print("="*60)
print("KNOWLEDGE EXTRACTION DEMO")
print("="*60)
print()
print("The idea: Instead of just taking embeddings, we can QUERY")
print("a pretrained model for its actual predictions and use those")
print("to build our transition structure.")
print()
print("This is like asking: 'What have you learned about language?'")
print("and capturing that knowledge in our own format.")
print()

# Create extractor
extractor = ModelKnowledgeExtractor('gpt2', device='cpu')

# Test contexts - what does GPT-2 predict after these?
contexts = [
    "The quick brown",
    "Machine learning is",
    "The capital of France is",
    "In the beginning",
]

print("Extracting transition knowledge from GPT-2...")
print()

transitions = extractor.extract_transitions(contexts, top_k=5)

for ctx_ids, predictions in transitions.items():
    ctx_text = extractor.tokenizer.decode(list(ctx_ids))
    print(f'Context: "{ctx_text}"')
    print(f'  GPT-2 predicts:')
    for token_id, prob in sorted(predictions.items(), key=lambda x: -x[1]):
        token_str = extractor.tokenizer.decode([token_id])
        print(f'    "{token_str}": {prob:.1%}')
    print()

# Now show how we can use this to "teach" our mesh
print("="*60)
print("TEACHING THE MESH")
print("="*60)
print()
print("We can use extracted transitions to teach our mesh what")
print("GPT-2 has learned - without training ourselves!")
print()

# Create mesh and extract to it
mesh = MultiModelMesh(device='cpu')
mesh.learn_from_model('gpt2')

sample_texts = [
    "The quick brown fox jumps over",
    "Machine learning is a powerful",
]

print(f"Teaching mesh from {len(sample_texts)} sample texts...")
stats = extractor.teach_mesh(mesh, sample_texts, top_k=3)
print(f"  Contexts queried: {stats['contexts_queried']}")
print(f"  Transitions added: {stats['transitions_added']}")
print()

# Show the mesh now has extracted knowledge
mesh.summary()

print()
print("="*60)
print("KEY INSIGHT")
print("="*60)
print()
print("We just 'ripped' GPT-2's learned knowledge:")
print("  1. Embeddings (semantic space)")
print("  2. Transition probabilities (what comes next)")
print()
print("All without training a single gradient ourselves.")
print("This is knowledge extraction → structure building.")
