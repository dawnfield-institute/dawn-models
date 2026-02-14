"""Quick analysis: Does Xi serve as a natural decision boundary in attention entropy?"""
import json, numpy as np

d = json.load(open('../results/exp_07_attention_pac_20260213_201014.json'))

XI = 1 + np.pi / 55  # 1.05712

print(f'Xi = {XI:.5f}')
print()

# Per-prompt attention entropy
print('FACTUAL attention entropy (per prompt):')
for p in d['groups']['factual']['prompts']:
    h = p['mean_attn_entropy']
    delta = h - XI
    marker = '<-- below Xi' if delta < 0 else '    above Xi'
    print(f'  H={h:.4f}  delta={delta:+.4f}  {marker}  {p["prompt"][:45]}')

below_f = sum(1 for p in d['groups']['factual']['prompts'] if p['mean_attn_entropy'] < XI)
total_f = len(d['groups']['factual']['prompts'])
print(f'\n  Factual below Xi: {below_f}/{total_f}')

print()
print('HALLUCINATED attention entropy (per prompt):')
for p in d['groups']['hallucination']['prompts']:
    h = p['mean_attn_entropy']
    delta = h - XI
    marker = '<-- below Xi' if delta < 0 else '    above Xi'
    print(f'  H={h:.4f}  delta={delta:+.4f}  {marker}  {p["prompt"][:45]}')

below_h = sum(1 for p in d['groups']['hallucination']['prompts'] if p['mean_attn_entropy'] < XI)
total_h = len(d['groups']['hallucination']['prompts'])
print(f'\n  Hallucinated below Xi: {below_h}/{total_h}')

# Xi as classifier
print(f'\n=== Xi AS DECISION BOUNDARY ===')
print(f'  Rule: H < Xi => factual, H >= Xi => hallucinated')
correct_f = below_f
correct_h = total_h - below_h
total = total_f + total_h
accuracy = (correct_f + correct_h) / total
print(f'  Accuracy: {correct_f + correct_h}/{total} = {accuracy*100:.1f}%')
print(f'  Factual recall: {correct_f}/{total_f} = {correct_f/total_f*100:.1f}%')
print(f'  Halluc recall:  {correct_h}/{total_h} = {correct_h/total_h*100:.1f}%')

# Overall means
all_f = [p['mean_attn_entropy'] for p in d['groups']['factual']['prompts']]
all_h = [p['mean_attn_entropy'] for p in d['groups']['hallucination']['prompts']]
print(f'\n  Factual mean:  {np.mean(all_f):.5f}  (Xi - {XI - np.mean(all_f):.5f})')
print(f'  Halluc mean:   {np.mean(all_h):.5f}  (Xi + {np.mean(all_h) - XI:.5f})')
print(f'  Grand mean:    {np.mean(all_f + all_h):.5f}  (Xi = {XI:.5f})')

# Token-level: how many tokens have attention entropy near Xi?
print(f'\n=== TOKEN-LEVEL Xi PROXIMITY ===')
all_token_h = []
for group in ['factual', 'hallucination']:
    for p in d['groups'][group]['prompts']:
        # We don't have tokens in the summary, but let's check prompt-level
        pass

# Check per-token from individual prompts (they're stored without tokens in summary)
# Let's check the depth slope sign at Xi
print(f'\n=== DEPTH SLOPE AT Xi ===')
for group in ['factual', 'hallucination']:
    slopes = [p['mean_depth_slope'] for p in d['groups'][group]['prompts']]
    ents = [p['mean_attn_entropy'] for p in d['groups'][group]['prompts']]
    print(f'  {group:15s}: H_mean={np.mean(ents):.4f}  depth_slope={np.mean(slopes):.4f}')

# Optimal threshold search
print(f'\n=== OPTIMAL THRESHOLD SEARCH ===')
all_vals = [(h, 'F') for h in all_f] + [(h, 'H') for h in all_h]
thresholds = np.linspace(min(all_f + all_h) - 0.01, max(all_f + all_h) + 0.01, 200)
best_acc = 0
best_t = 0
for t in thresholds:
    corr = sum(1 for v, g in all_vals if (v < t and g == 'F') or (v >= t and g == 'H'))
    acc = corr / len(all_vals)
    if acc > best_acc:
        best_acc = acc
        best_t = t

print(f'  Optimal threshold: {best_t:.5f}  accuracy: {best_acc*100:.1f}%')
print(f'  Xi threshold:      {XI:.5f}  accuracy: {accuracy*100:.1f}%')
print(f'  Distance: {abs(best_t - XI):.5f} ({abs(best_t - XI)/XI*100:.2f}% of Xi)')
