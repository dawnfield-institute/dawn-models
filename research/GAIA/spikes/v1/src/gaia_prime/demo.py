"""
GAIA-PAC Demo: Full training and generation pipeline.

This demonstrates the complete GAIA-PAC workflow:
1. Graft embeddings from GPT-2
2. Learn from WikiText-2
3. Generate text
4. Benchmark against baseline
"""

import torch
import time
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gaia_prime import gaia_prime


def download_wikitext2():
    """Download WikiText-2 sample for training."""
    try:
        from datasets import load_dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        # Join first 1000 samples
        text = '\n'.join(dataset['text'][:1000])
        return text
    except ImportError:
        print("datasets not installed, using sample text")
        return """
        The quick brown fox jumps over the lazy dog. Machine learning is a subset
        of artificial intelligence that enables systems to learn from data without
        being explicitly programmed. Natural language processing combines linguistics
        and computer science to help machines understand human language. Deep learning
        uses neural networks with many layers to learn complex patterns. The field
        of AI has grown rapidly in recent years, with applications ranging from
        image recognition to language translation.
        """ * 100


def demo_basic():
    """Basic demo: create model, learn, generate."""
    print("=" * 60)
    print("GAIA-PAC v1.0 Demo")
    print("=" * 60)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Create model
    print("\n1. Creating GAIA-PAC from GPT-2...")
    start = time.time()
    model = gaia_prime.from_gpt2('gpt2', device=device)
    print(f"   Created in {time.time() - start:.2f}s")
    print(f"   {model}")
    
    # Learn from text
    print("\n2. Learning from text...")
    text = download_wikitext2()
    print(f"   Text length: {len(text):,} characters")
    
    start = time.time()
    stats = model.learn(text)
    learn_time = time.time() - start
    
    print(f"   Learned in {learn_time:.2f}s")
    print(f"   Tokens processed: {stats['tokens_processed']:,}")
    print(f"   Tokens/sec: {stats['tokens_processed'] / learn_time:,.0f}")
    
    # Generate text
    print("\n3. Generating text...")
    prompts = [
        "The quick brown",
        "Machine learning is",
        "Natural language processing",
    ]
    
    for prompt in prompts:
        print(f"\n   Prompt: '{prompt}'")
        start = time.time()
        result = model.generate(prompt, max_tokens=50)
        gen_time = time.time() - start
        
        print(f"   Generated: '{result.text}'")
        print(f"   Time: {gen_time:.2f}s, Tokens: {result.tokens.numel()}")
    
    # Statistics
    print("\n4. Model Statistics:")
    stats = model.get_statistics()
    print(f"   Tokens learned: {stats['metadata']['tokens_learned']:,}")
    print(f"   Transition hit rate: {stats['transitions'].get('hit_rate', 'N/A')}")
    print(f"   Concentration high-quality rate: {stats['concentration'].get('high_quality_rate', 0):.1%}")
    
    # Save model
    save_path = Path(__file__).parent / 'checkpoints' / 'gaia_prime_demo'
    print(f"\n5. Saving model to {save_path}...")
    model.save(save_path)
    
    # Load and verify
    print("\n6. Loading model back...")
    model2 = gaia_prime.load(save_path, device=device)
    print(f"   Loaded: {model2}")
    
    result2 = model2.generate("The quick", max_tokens=20)
    print(f"   Test generation: '{result2.text}'")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


def demo_comparison():
    """Compare GAIA-PAC to raw GPT-2."""
    print("=" * 60)
    print("GAIA-PAC vs GPT-2 Comparison")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get some text
    text = download_wikitext2()
    
    # Test prompts
    prompts = ["The", "Machine learning", "In the"]
    
    # ---- GAIA-PAC ----
    print("\n[GAIA-PAC]")
    model = gaia_prime.from_gpt2('gpt2', device=device)
    model.learn(text)
    
    for prompt in prompts:
        result = model.generate(prompt, max_tokens=30)
        print(f"  '{prompt}' → '{result.text[:50]}...'")
    
    # ---- Raw GPT-2 ----
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        print("\n[GPT-2 (untrained)]")
        gpt2 = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            outputs = gpt2.generate(
                **inputs, 
                max_new_tokens=30,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"  '{prompt}' → '{text[:50]}...'")
    
    except ImportError:
        print("\n[Skipping GPT-2 comparison - transformers not installed]")
    
    print("\n" + "=" * 60)


def demo_metrics():
    """Detailed metrics and benchmarks."""
    print("=" * 60)
    print("GAIA-PAC Metrics & Benchmarks")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = gaia_prime.from_gpt2('gpt2', device=device)
    
    # Train with increasing data
    text = download_wikitext2()
    chunks = [text[:1000], text[:5000], text[:20000], text]
    
    print("\nLearning curve:")
    for chunk in chunks:
        model_fresh = gaia_prime.from_gpt2('gpt2', device=device)
        
        start = time.time()
        stats = model_fresh.learn(chunk)
        learn_time = time.time() - start
        
        # Get hit rate
        model_stats = model_fresh.get_statistics()
        
        print(f"  {len(chunk):>6,} chars | "
              f"{stats['tokens_processed']:>6,} tokens | "
              f"{learn_time:.2f}s | "
              f"{stats['tokens_processed']/learn_time:>8,.0f} tok/s")
    
    # Generation speed
    print("\nGeneration speed:")
    model = gaia_prime.from_gpt2('gpt2', device=device)
    model.learn(text)
    
    for length in [10, 50, 100, 200]:
        times = []
        for _ in range(3):
            start = time.time()
            model.generate("The quick", max_tokens=length)
            times.append(time.time() - start)
        avg_time = sum(times) / len(times)
        print(f"  {length:>3} tokens: {avg_time:.3f}s ({length/avg_time:.1f} tok/s)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GAIA-PAC Demo")
    parser.add_argument('--mode', choices=['basic', 'compare', 'metrics'], 
                       default='basic', help='Demo mode')
    args = parser.parse_args()
    
    if args.mode == 'basic':
        demo_basic()
    elif args.mode == 'compare':
        demo_comparison()
    elif args.mode == 'metrics':
        demo_metrics()
