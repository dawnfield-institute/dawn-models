"""Test PAC mesh convergence with multi-model learning."""

from gaia_prime.pac_mesh import MultiModelMesh

mesh = MultiModelMesh(device='cpu')

# Learn from two models
stats1 = mesh.learn_from_model('gpt2')
stats2 = mesh.learn_from_model('gpt2-medium')  # Same vocab, different weights

print(f'\nAfter gpt2: {stats1}')
print(f'After gpt2-medium: {stats2}')

# Learn text
text = """
The quick brown fox jumps over the lazy dog.
A quick brown fox leaps over the lazy dog.
The fast brown fox jumps over a lazy dog.
""" * 10

stats = mesh.learn_from_text(text, source='corpus')
print(f'After text: {stats}')

mesh.summary()

# Show convergences
convergences = mesh.mesh.find_convergences(min_factor=2)
print(f'\nConvergence points (byref shared nodes):')
for node in convergences[:10]:
    print(f'  "{node.token_str}" <- {node.convergence_factor} paths, conf={node.confidence:.2f}, sources={node.sources}')

# Build model and generate
model = mesh.build_model()
print(f'\nBuilt: {model}')

result = model.generate("The quick", max_tokens=10)
print(f'Generated: "{result.text}"')

# Show what the mesh knows about convergence
print('\n' + '='*60)
print('BYREF CONVERGENCE ANALYSIS')
print('='*60)

print('\nMeaning of byref in this context:')
print('- Each node stores a reference to child nodes')
print('- When "The quick" and "A quick" both lead to "brown"')
print('- They share the SAME "brown" node (byref, not copy)')
print('- This is convergence: multiple paths → same destination')

# Show the actual convergence
if convergences:
    node = convergences[0]
    print(f'\nExample convergence: "{node.token_str}"')
    print(f'  Incoming paths from: {list(node.incoming_paths.keys())}')
    print(f'  Each path count: {list(node.incoming_paths.values())}')
    print(f'  Total traversals: {node.total_incoming}')
