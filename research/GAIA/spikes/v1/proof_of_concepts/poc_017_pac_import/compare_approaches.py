"""
SEC-PAC vs Baseline Comparison
===============================

Compare:
1. Vanilla (last token embedding only)
2. Programmatic navigation (hard-coded thresholds)
3. SEC-PAC (entropy collapse with analog mixing)

This demonstrates the value of using SEC dynamics
for "analog density" of detail.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Add the test files to path
poc_path = Path(__file__).parent
sys.path.insert(0, str(poc_path))

from test_sec_pac import SECPACTransformer, SECPACSystem
from test_navigable_pac import NavigablePACTransformer


def compare_approaches():
    """Compare all three approaches."""
    from transformers import GPT2Tokenizer
    
    print("="*70)
    print("SEC-PAC vs BASELINE COMPARISON")
    print("="*70)
    
    pac_path = poc_path.parent / "poc_016_pac_extraction" / "extracted_v3" / "pythia_70m"
    
    if not pac_path.exists():
        print(f"❌ PAC not found at {pac_path}")
        return
    
    # Load models
    print("\nLoading models...")
    sec_model = SECPACTransformer(pac_path)
    nav_model = NavigablePACTransformer(pac_path)
    
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Test prompts
    prompts = [
        "The weather today is",
        "Once upon a time",
        "In the beginning",
        "To be or not to",
        "The quick brown fox"
    ]
    
    print("\n" + "="*70)
    print("PREDICTION COMPARISON")
    print("="*70)
    
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt)
        
        print(f"\n{'='*60}")
        print(f"Prompt: '{prompt}'")
        print(f"{'='*60}")
        
        # 1. Vanilla: Just use last token embedding
        print("\n1. VANILLA (last token only):")
        last_embed = sec_model.system.vocab_embeddings[token_ids[-1]]
        last_embed_norm = F.normalize(last_embed.unsqueeze(0), dim=1).squeeze()
        vocab_norm = F.normalize(sec_model.system.vocab_embeddings, dim=1)
        vanilla_scores = vocab_norm @ last_embed_norm
        top_vanilla = torch.topk(vanilla_scores, 5)
        
        for i, (score, idx) in enumerate(zip(top_vanilla.values, top_vanilla.indices)):
            token = tokenizer.decode([idx.item()])
            print(f"   {i+1}. '{token}' ({score.item():.4f})")
        
        # 2. Programmatic navigation
        print("\n2. PROGRAMMATIC NAVIGATION (threshold-based):")
        nav_preds = nav_model.predict_next(token_ids, top_k=5)
        for i, (tid, score) in enumerate(nav_preds[:5]):
            token = tokenizer.decode([tid])
            print(f"   {i+1}. '{token}' ({score:.4f})")
        
        # 3. SEC-PAC (entropy collapse)
        print("\n3. SEC-PAC (entropy collapse dynamics):")
        sec_preds = sec_model.predict_next(token_ids, top_k=5, collapse_iters=30)
        for i, (tid, score, metrics) in enumerate(sec_preds[:5]):
            token = tokenizer.decode([tid])
            print(f"   {i+1}. '{token}' ({score:.4f})")
        print(f"   [Xi={metrics['final_xi']:.4f}, entropy={metrics['final_entropy']:.3f}]")
    
    # Analyze what SEC-PAC does differently
    print("\n" + "="*70)
    print("SEC COLLAPSE ANALYSIS")
    print("="*70)
    
    test_prompt = "The meaning of life is"
    token_ids = tokenizer.encode(test_prompt)
    
    print(f"\nPrompt: '{test_prompt}'")
    print(f"Tokens: {[tokenizer.decode([t]) for t in token_ids]}")
    
    # Run SEC with detailed tracking
    sec_model.system.initialize_from_sequence(token_ids)
    
    print("\nSEC Collapse History:")
    print(f"{'Iter':>5} | {'Entropy':>8} | {'Xi':>8} | {'Collapsed':>10}")
    print("-"*45)
    
    for i in range(30):
        metrics = sec_model.system.sec_collapse_step()
        if i < 5 or i % 5 == 4:
            print(f"{metrics['iteration']:>5} | {metrics['global_entropy']:>8.4f} | "
                  f"{metrics['global_xi']:>8.4f} | {metrics['collapsed_nodes']:>10}")
    
    # Show per-token entropy after collapse
    print("\nPer-Token Entropy After Collapse:")
    for nid, node in sec_model.system.nodes.items():
        pos = int(nid.split('_')[1])
        token = tokenizer.decode([token_ids[pos]])
        print(f"  {nid}: '{token}' -> entropy={node.entropy:.4f}, xi={node.xi_local:.4f}")
    
    # Show delta norms (how much each token "contributes")
    print("\nDelta Contributions:")
    total_norm = sum(node.delta.norm().item() for node in sec_model.system.nodes.values())
    for nid, node in sec_model.system.nodes.items():
        pos = int(nid.split('_')[1])
        token = tokenizer.decode([token_ids[pos]])
        contribution = node.delta.norm().item() / total_norm * 100
        print(f"  {nid}: '{token}' -> {contribution:.1f}% of total")
    
    # Compare generation quality
    print("\n" + "="*70)
    print("GENERATION QUALITY COMPARISON")
    print("="*70)
    
    test_prompts = [
        "The weather today is",
        "Once upon a time there was",
    ]
    
    for prompt in test_prompts:
        token_ids = tokenizer.encode(prompt)
        print(f"\nPrompt: '{prompt}'")
        
        # SEC generation
        print("\nSEC-PAC Generation:")
        sec_gen = sec_model.generate(token_ids, max_new_tokens=15, temperature=0.6)
        sec_text = tokenizer.decode(sec_gen, skip_special_tokens=True)
        print(f"  {sec_text}")
        
        # Navigable generation
        print("\nProgrammatic Navigation Generation:")
        nav_gen = nav_model.generate(token_ids, max_new_tokens=15, temperature=0.6)
        nav_text = tokenizer.decode(nav_gen, skip_special_tokens=True)
        print(f"  {nav_text}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key Differences:

1. VANILLA: Only uses last token's embedding
   - Fast but no context
   - Predicts based on single word similarity

2. PROGRAMMATIC: Uses hard-coded attention thresholds
   - Includes context but discretely
   - Binary decisions: include or exclude

3. SEC-PAC: Uses entropy collapse dynamics
   - Continuous "analog" blending
   - Structure EMERGES through collapse
   - Xi-bounded complexity (natural regularization)
   - Information flows based on entropy gradients

The SEC approach preserves "analog density of detail" because:
- No hard thresholds (continuous values)
- Structure crystallizes naturally
- Collapse history affects contribution weights
- Neighbor coupling creates smooth information flow
""")


if __name__ == "__main__":
    compare_approaches()
