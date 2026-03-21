"""
POC-021: Large Model Comparison
===============================

Compare oracle distillation from:
1. Small models only: GPT-2 (124M) + Pythia (70M)
2. With large models: + Qwen2.5-3B + Phi-3-mini (3.8B)

Questions:
- Do larger oracles improve generation quality?
- Does multi-oracle consensus improve hit rate?
- Can we combine knowledge from all models effectively?
"""

import sys
sys.path.insert(0, ".")

from unified_full_system import UnifiedFullSystem
from datetime import datetime
import json
from pathlib import Path


def test_small_oracles():
    """Test with GPT-2 + Pythia only"""
    print("\n" + "="*70)
    print("TEST 1: SMALL ORACLES ONLY (GPT-2 + Pythia)")
    print("="*70)
    
    system = UnifiedFullSystem(dim=256, max_layers=13)
    system.build(include_large_models=False)
    
    return run_eval(system, "small_oracles")


def test_large_oracles():
    """Test with all oracles including Qwen and Phi"""
    print("\n" + "="*70)
    print("TEST 2: ALL ORACLES (GPT-2 + Pythia + Qwen + Phi)")
    print("="*70)
    
    system = UnifiedFullSystem(dim=256, max_layers=13)
    system.build(include_large_models=True)
    
    return run_eval(system, "large_oracles")


def run_eval(system, name: str):
    """Run standard evaluation suite"""
    
    test_prompts = [
        # Simple
        "The cat",
        "A dog",
        "Birds fly",
        # Science
        "Scientists study",
        "Research shows",
        "Experiments reveal",
        # Abstract
        "Love is",
        "Time is",
        "Knowledge helps",
        # Nature
        "Trees grow",
        "Water flows",
        "Fire burns",
        # Complex
        "The future of",
        "In nature we",
        "History teaches us",
    ]
    
    results = []
    total_hits = 0
    total_tokens = 0
    
    print(f"\n  Evaluating {len(test_prompts)} prompts...")
    
    for prompt in test_prompts:
        output, stats = system.generate_with_learning(prompt, max_tokens=20, temperature=0.7)
        
        hits = stats.get('hits', 0)
        misses = stats.get('misses', 0)
        hit_rate = stats.get('hit_rate', 0)
        
        total_hits += hits
        total_tokens += hits + misses
        
        results.append({
            'prompt': prompt,
            'output': output[:60] + "..." if len(output) > 60 else output,
            'hit_rate': hit_rate,
            'hits': hits,
            'misses': misses,
        })
        
        print(f"    '{prompt}' → {hit_rate:.1f}%")
    
    overall_hit_rate = total_hits / max(1, total_tokens) * 100
    
    print(f"\n  Overall hit rate: {overall_hit_rate:.1f}%")
    print(f"  Oracles: {list(system.oracles.keys())}")
    
    return {
        'name': name,
        'oracles': list(system.oracles.keys()),
        'num_oracles': len(system.oracles),
        'overall_hit_rate': overall_hit_rate,
        'total_transitions': system.transitions.num_transitions(),
        'crystallized': system.transitions.crystallized,
        'results': results,
    }


def compare_results(small_results, large_results):
    """Compare small vs large oracle performance"""
    print("\n" + "="*70)
    print("COMPARISON: SMALL vs LARGE ORACLES")
    print("="*70)
    
    print(f"\n  {'Metric':<25} {'Small (2)':<15} {'Large (4)':<15} {'Delta':<10}")
    print("  " + "-"*65)
    
    metrics = [
        ('Overall hit rate', 'overall_hit_rate', '%'),
        ('Transitions', 'total_transitions', ''),
        ('Crystallized', 'crystallized', ''),
    ]
    
    for label, key, suffix in metrics:
        small_val = small_results[key]
        large_val = large_results[key]
        
        if suffix == '%':
            delta = f"+{large_val - small_val:.1f}%"
        else:
            delta = f"+{large_val - small_val}"
            
        print(f"  {label:<25} {small_val:<15.1f} {large_val:<15.1f} {delta:<10}")
    
    # Per-prompt comparison
    print(f"\n  Per-prompt breakdown:")
    print(f"  {'Prompt':<20} {'Small':<10} {'Large':<10} {'Winner':<10}")
    print("  " + "-"*50)
    
    small_wins = 0
    large_wins = 0
    ties = 0
    
    for s, l in zip(small_results['results'], large_results['results']):
        prompt = s['prompt'][:18]
        s_rate = s['hit_rate']
        l_rate = l['hit_rate']
        
        if l_rate > s_rate:
            winner = "Large ✓"
            large_wins += 1
        elif s_rate > l_rate:
            winner = "Small ✓"
            small_wins += 1
        else:
            winner = "Tie"
            ties += 1
            
        print(f"  {prompt:<20} {s_rate:<10.1f} {l_rate:<10.1f} {winner:<10}")
    
    print(f"\n  Summary: Small wins: {small_wins}, Large wins: {large_wins}, Ties: {ties}")
    
    return {
        'small_wins': small_wins,
        'large_wins': large_wins,
        'ties': ties,
    }


def main():
    """Run full comparison"""
    print("="*70)
    print("POC-021: LARGE MODEL COMPARISON")
    print("="*70)
    print("Comparing oracle distillation quality")
    print("  Small: GPT-2 (124M) + Pythia (70M)")
    print("  Large: + Qwen2.5-3B + Phi-3-mini (3.8B)")
    
    # Test small oracles first
    small_results = test_small_oracles()
    
    # Test large oracles
    print("\n" + "="*70)
    print("Loading large models... (this may take a few minutes)")
    print("="*70)
    large_results = test_large_oracles()
    
    # Compare
    comparison = compare_results(small_results, large_results)
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'small_oracles': small_results,
        'large_oracles': large_results,
        'comparison': comparison,
    }
    
    with open(output_dir / "large_model_comparison.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to {output_dir / 'large_model_comparison.json'}")
    
    # Final summary
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if comparison['large_wins'] > comparison['small_wins']:
        print("\n  ✓ LARGE ORACLES IMPROVE DISTILLATION")
        print(f"    Large models won {comparison['large_wins']}/{comparison['large_wins']+comparison['small_wins']+comparison['ties']} prompts")
    elif comparison['small_wins'] > comparison['large_wins']:
        print("\n  ✗ Small oracles performed better (unexpected)")
    else:
        print("\n  = Tie - similar performance")
    
    print(f"\n  Hit rate improvement: {large_results['overall_hit_rate'] - small_results['overall_hit_rate']:.1f}%")


if __name__ == "__main__":
    main()
