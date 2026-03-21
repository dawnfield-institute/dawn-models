"""
POC-021: Test Text Generations from Multi-Oracle System
Compare generation quality between small and large oracle configurations
"""

import sys
sys.path.insert(0, '.')

from unified_full_system import UnifiedFullSystem

def test_generations():
    print("=" * 70)
    print("POC-021: TEXT GENERATION COMPARISON")
    print("=" * 70)
    
    prompts = [
        "The cat",
        "Scientists discovered",
        "In the forest",
        "Love is",
        "The future of AI",
        "Water flows through",
        "Music creates",
        "Knowledge helps us",
    ]
    
    # Test with small oracles first
    print("\n" + "=" * 70)
    print("SMALL ORACLES (GPT-2 + Pythia = 194M params)")
    print("=" * 70)
    
    small_system = UnifiedFullSystem()
    small_system.build(include_large_models=False)
    
    print("\nGenerating text...")
    small_generations = {}
    for prompt in prompts:
        text = small_system.generate(prompt, max_tokens=30)
        small_generations[prompt] = text
        print(f"\n  '{prompt}':")
        print(f"    → {text}")
    
    # Clean up to free memory
    del small_system
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Test with large oracles
    print("\n" + "=" * 70)
    print("LARGE ORACLES (GPT-2 + Pythia + Qwen 1.5B + SmolLM2 = 2.05B params)")
    print("=" * 70)
    
    large_system = UnifiedFullSystem()
    large_system.build(include_large_models=True)
    
    print("\nGenerating text...")
    large_generations = {}
    for prompt in prompts:
        text = large_system.generate(prompt, max_tokens=30)
        large_generations[prompt] = text
        print(f"\n  '{prompt}':")
        print(f"    → {text}")
    
    # Side-by-side comparison
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 70)
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print(f"  Small: {small_generations[prompt][:80]}...")
        print(f"  Large: {large_generations[prompt][:80]}...")
    
    # Test multi-oracle prediction if available
    if hasattr(large_system, 'multi_oracle_predict'):
        print("\n" + "=" * 70)
        print("MULTI-ORACLE CONSENSUS PREDICTION")
        print("=" * 70)
        
        test_contexts = [
            "The quick brown",
            "Scientists have discovered that",
            "In nature, we observe",
        ]
        
        for context in test_contexts:
            # Get predictions from multi-oracle consensus
            input_ids = large_system.tokenizer.encode(context, return_tensors='pt')
            logits = large_system.multi_oracle_predict(input_ids)
            
            # Get top 5 predictions
            import torch
            top_k = torch.topk(logits[0, -1], k=5)
            
            print(f"\n  Context: '{context}'")
            print(f"  Top 5 next tokens (weighted consensus):")
            for i, (prob, idx) in enumerate(zip(top_k.values, top_k.indices)):
                token = large_system.tokenizer.decode([idx.item()])
                print(f"    {i+1}. '{token}' (score: {prob.item():.3f})")
    
    print("\n" + "=" * 70)
    print("GENERATION TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_generations()
