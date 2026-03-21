"""Quick test of GAIA-PAC v1.0.0"""

from gaia_prime import gaia_prime
import time

print('='*60)
print('GAIA-PAC v1.0.0 Test')
print('='*60)

# Create model
print()
print('[1] Creating model from GPT-2...')
start = time.time()
model = gaia_prime.from_gpt2('gpt2', device='cpu')
print(f'    Created in {time.time()-start:.2f}s')
print(f'    {model}')

# Train
print()
print('[2] Learning from text...')
training_text = '''
Machine learning is a subset of artificial intelligence that enables systems 
to learn from data. Natural language processing uses machine learning to 
understand human language. Deep learning is a type of machine learning that 
uses neural networks with many layers. The field has grown rapidly.
''' * 20

start = time.time()
stats = model.learn(training_text)
print(f'    Learned in {time.time()-start:.2f}s')
print(f'    Tokens: {stats["tokens_processed"]}')

# Generate
print()
print('[3] Generating text...')
prompts = ['Machine learning is', 'Natural language', 'Deep learning']
for prompt in prompts:
    result = model.generate(prompt, max_tokens=15, temperature=1.0)
    print(f'    "{prompt}" -> "{result.text}"')

# Stats
print()
print('[4] Statistics:')
all_stats = model.get_statistics()
print(f'    Tokens learned: {all_stats["metadata"]["tokens_learned"]}')
print(f'    High quality rate: {all_stats["concentration"]["high_quality_rate"]:.1%}')
print(f'    Mean concentration: {all_stats["concentration"]["mean_concentration"]:.2f}')

print()
print('='*60)
print('SUCCESS!')
print('='*60)
