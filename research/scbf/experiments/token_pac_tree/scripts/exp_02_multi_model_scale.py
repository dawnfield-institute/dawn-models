#!/usr/bin/env python3
"""
exp_02_multi_model_scale.py
============================

Scale-up experiment: run PAC tree analysis across ALL cached Pythia models
(70M, 160M, 410M, 1B) with a large diverse prompt set + null baselines.

GOALS:
  1. Increase token count from 69 → 500+ for statistical power
  2. Compare phi enrichment across 4 model scales
  3. Add proper null baseline (random logits → are phi ratios an artifact of softmax?)
  4. Histogram analysis of PAC ratio distribution
  5. Test whether SEC phase → accuracy gradient is universal across scales

BUILDS ON exp_01:
  - 32.89x phi enrichment (but only 69 tokens)
  - SEC phase monotonically predicts accuracy
  - Chaotic phase median ratio = 1.6349 (near phi)
  - Correct vs incorrect p = 0.057 (needs more data)

Usage:
    python exp_02_multi_model_scale.py
    python exp_02_multi_model_scale.py --model pythia-160m   # single model
    python exp_02_multi_model_scale.py --quick                # fast subset
"""

import argparse
import json
import sys
import time
import math
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# ── Path setup ───────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(EXPERIMENT_DIR))

from core.pac_tree import build_pac_tree_from_logits, PACForest, PHI, INV_PHI, XI
from core.collapse_metrics import classify_sec_phase, SECPhase

PYTHIA_CACHE = str(EXPERIMENT_DIR.parent / 'huggingface_bifractal_validation' / 'pythia_cache')
TOP_K = 10

MODELS = {
    'pythia-70m': 'EleutherAI/pythia-70m',
    'pythia-160m': 'EleutherAI/pythia-160m',
    'pythia-410m': 'EleutherAI/pythia-410m',
    'pythia-1b': 'EleutherAI/pythia-1b',
}

# ── Large diverse prompt set ─────────────────────────────────────────
# Categories: factual, pattern, reasoning, ambiguous, nonsense
PROMPTS = {
    'factual': [
        ("The capital of France is", " Paris"),
        ("The largest planet in our solar system is", " Jupiter"),
        ("In mathematics, pi is approximately", " 3"),
        ("The chemical symbol for water is", " H"),
        ("The Earth orbits around the", " Sun"),
        ("Albert Einstein developed the theory of", " relativ"),
        ("The human body has two hundred and six", " bones"),
        ("Photosynthesis converts sunlight into", " energy"),
        ("The atomic number of hydrogen is", " 1"),
        ("Shakespeare wrote Romeo and", " Juliet"),
        ("The speed of light in a vacuum is approximately", " 300"),
        ("DNA stands for deoxyribonucle", "ic"),
        ("The Pacific Ocean is the", " largest"),
        ("Isaac Newton discovered the law of", " gravity"),
        ("The boiling point of water is", " 100"),
    ],
    'pattern': [
        ("Once upon a time", " there"),
        ("To be or not to", " be"),
        ("The quick brown fox jumps over the", " lazy"),
        ("All that glitters is not", " gold"),
        ("A penny saved is a penny", " earned"),
        ("Actions speak louder than", " words"),
        ("The early bird catches the", " worm"),
        ("When in Rome do as the", " Romans"),
        ("An apple a day keeps the", " doctor"),
        ("Knowledge is", " power"),
    ],
    'reasoning': [
        ("If x equals 5, then x plus 3 equals", " 8"),
        ("The Pythagorean theorem states that a squared plus b squared equals", " c"),
        ("In a right triangle, the longest side is called the", " hyp"),
        ("The derivative of x squared is", " 2"),
        ("The integral of 1 over x is", " ln"),
        ("If all dogs are mammals and all mammals are animals then all dogs are", " animals"),
        ("The square root of 144 is", " 12"),
        ("Two plus two equals", " four"),
    ],
    'ambiguous': [
        ("The best way to learn is", " to"),
        ("I think that the most important thing in life is", " to"),
        ("In my opinion the", " most"),
        ("Yesterday I went to the", " store"),
        ("The meaning of life is", " to"),
        ("When I look at the sky I see", " the"),
        ("The future of technology is", " going"),
        ("If I could change one thing about the world it would be", " the"),
    ],
    'technical': [
        ("In quantum mechanics the uncertainty principle states that", " the"),
        ("The Fourier transform converts a function from the time domain to the", " frequency"),
        ("A neural network consists of layers of", " neurons"),
        ("The Fibonacci sequence starts with", " 0"),
        ("In machine learning overfitting occurs when", " the"),
        ("The Standard Model of particle physics describes", " the"),
        ("Entropy in thermodynamics measures the", " dis"),
        ("A black hole is formed when a massive star", " col"),
    ],
}


def load_model(model_key: str, device: str):
    """Load a Pythia model from cache."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_id = MODELS[model_key]
    print(f"\n  Loading {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=PYTHIA_CACHE)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=PYTHIA_CACHE).to(device)
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {n_params:,} parameters")
    return model, tokenizer


def generate_random_logits(vocab_size: int, n_positions: int, seed: int = 42):
    """Generate random logits for null baseline comparison."""
    rng = np.random.RandomState(seed)
    return torch.tensor(rng.randn(n_positions, vocab_size), dtype=torch.float32)


def analyse_single_prompt(model, tokenizer, prompt: str, continuation: str, 
                          device: str, top_k: int = TOP_K) -> PACForest:
    """Build PAC forest for one prompt."""
    full_text = prompt + continuation
    input_ids = tokenizer.encode(full_text, return_tensors='pt').to(device)
    seq_len = input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0]  # [seq_len, vocab_size]
    
    forest = PACForest()
    forest.prompt_text = full_text
    forest.model_name = str(model.config._name_or_path)
    
    for pos in range(seq_len - 1):
        pos_logits = logits[pos]
        gt_id = input_ids[0, pos + 1].item()
        selected_id = pos_logits.argmax().item()
        
        tree = build_pac_tree_from_logits(
            logits=pos_logits, position=pos, tokenizer=tokenizer,
            top_k=top_k, selected_token_id=selected_id, ground_truth_id=gt_id,
        )
        forest.add_tree(tree)
    
    return forest


def compute_histogram_stats(ratios: np.ndarray, n_bins: int = 50) -> dict:
    """Binned histogram analysis of PAC ratios with phi/inv_phi markers."""
    if len(ratios) == 0:
        return {}
    
    # Clip extreme ratios for histogram (keep raw stats separate)
    clipped = ratios[ratios < 20]  # focus on structured region
    if len(clipped) == 0:
        clipped = ratios
    
    hist, bin_edges = np.histogram(clipped, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Find which bin phi and 1/phi fall in
    phi_bin = np.argmin(np.abs(bin_centers - PHI))
    inv_phi_bin = np.argmin(np.abs(bin_centers - INV_PHI))
    
    # Density at phi vs average density
    avg_density = 1.0 / (bin_edges[-1] - bin_edges[0]) if (bin_edges[-1] - bin_edges[0]) > 0 else 1.0
    phi_density = hist[phi_bin]
    inv_phi_density = hist[inv_phi_bin]
    
    return {
        'n_ratios': len(ratios),
        'n_clipped': len(clipped),
        'bin_width': float(bin_width),
        'phi_bin_density': float(phi_density),
        'inv_phi_bin_density': float(inv_phi_density),
        'avg_density': float(avg_density),
        'phi_density_ratio': float(phi_density / avg_density) if avg_density > 0 else None,
        'inv_phi_density_ratio': float(inv_phi_density / avg_density) if avg_density > 0 else None,
        'histogram': {
            'bin_centers': bin_centers.tolist(),
            'density': hist.tolist(),
        },
    }


def run_null_baseline(vocab_size: int, n_positions: int = 500, top_k: int = TOP_K) -> dict:
    """Generate PAC trees from random logits — what phi enrichment do we expect by chance?"""
    print("\n  Running null baseline (random logits)...")
    
    random_logits = generate_random_logits(vocab_size, n_positions)
    
    ratios = []
    entropies = []
    
    for pos in range(n_positions):
        tree = build_pac_tree_from_logits(
            logits=random_logits[pos], position=pos,
            tokenizer=None, top_k=top_k,
        )
        if tree.pac_ratio_1_2 is not None:
            ratios.append(tree.pac_ratio_1_2)
        entropies.append(tree.total_entropy)
    
    ratios = np.array(ratios)
    entropies = np.array(entropies)
    
    phi_distances = np.abs(ratios - PHI)
    near_phi = np.sum(phi_distances < 0.1)
    
    return {
        'n_positions': n_positions,
        'n_ratios': len(ratios),
        'ratio_mean': float(np.mean(ratios)),
        'ratio_median': float(np.median(ratios)),
        'ratio_std': float(np.std(ratios)),
        'entropy_mean': float(np.mean(entropies)),
        'near_phi_count': int(near_phi),
        'near_phi_fraction': float(near_phi / len(ratios)) if len(ratios) > 0 else 0,
        'phi_distance_mean': float(np.mean(phi_distances)),
        'histogram': compute_histogram_stats(ratios),
    }


def analyse_model_results(forests: list, category_map: dict) -> dict:
    """Comprehensive analysis for one model's results."""
    all_ratios = []
    correct_ratios = []
    incorrect_ratios = []
    phase_data = defaultdict(lambda: {'count': 0, 'correct': 0, 'total_scored': 0, 'ratios': []})
    category_stats = defaultdict(lambda: {'ratios': [], 'entropies': [], 'correct': 0, 'total': 0})
    
    for forest in forests:
        cat = category_map.get(forest.prompt_text, 'unknown')
        for tree in forest.trees:
            phase = classify_sec_phase(tree.total_entropy).value
            phase_data[phase]['count'] += 1
            
            if tree.pac_ratio_1_2 is not None:
                all_ratios.append(tree.pac_ratio_1_2)
                phase_data[phase]['ratios'].append(tree.pac_ratio_1_2)
                category_stats[cat]['ratios'].append(tree.pac_ratio_1_2)
            
            category_stats[cat]['entropies'].append(tree.total_entropy)
            
            if tree.is_correct is not None:
                phase_data[phase]['total_scored'] += 1
                category_stats[cat]['total'] += 1
                if tree.is_correct:
                    phase_data[phase]['correct'] += 1
                    correct_ratios.append(tree.pac_ratio_1_2)
                    category_stats[cat]['correct'] += 1
                else:
                    if tree.pac_ratio_1_2 is not None:
                        incorrect_ratios.append(tree.pac_ratio_1_2)
    
    all_ratios = np.array(all_ratios)
    correct_ratios = np.array([r for r in correct_ratios if r is not None])
    incorrect_ratios = np.array(incorrect_ratios)
    
    # Core stats
    phi_distances = np.abs(all_ratios - PHI) if len(all_ratios) > 0 else np.array([])
    near_phi = np.sum(phi_distances < 0.1) if len(phi_distances) > 0 else 0
    
    result = {
        'n_tokens': len(all_ratios),
        'ratio_mean': float(np.mean(all_ratios)) if len(all_ratios) > 0 else None,
        'ratio_median': float(np.median(all_ratios)) if len(all_ratios) > 0 else None,
        'ratio_std': float(np.std(all_ratios)) if len(all_ratios) > 0 else None,
        'phi_distance_mean': float(np.mean(phi_distances)) if len(phi_distances) > 0 else None,
        'near_phi_count': int(near_phi),
        'near_phi_fraction': float(near_phi / len(all_ratios)) if len(all_ratios) > 0 else None,
        'histogram': compute_histogram_stats(all_ratios),
    }
    
    # SEC phase accuracy
    phase_summary = {}
    for phase in ['crystallized', 'ordered', 'transitional', 'chaotic']:
        pd = phase_data[phase]
        phase_summary[phase] = {
            'count': pd['count'],
            'accuracy': pd['correct'] / pd['total_scored'] if pd['total_scored'] > 0 else None,
            'ratio_mean': float(np.mean(pd['ratios'])) if pd['ratios'] else None,
            'ratio_median': float(np.median(pd['ratios'])) if pd['ratios'] else None,
        }
    result['sec_phases'] = phase_summary
    
    # Correct vs incorrect
    if len(correct_ratios) > 0 and len(incorrect_ratios) > 0:
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(correct_ratios, incorrect_ratios, equal_var=False)
        result['correct_vs_incorrect'] = {
            'correct_mean': float(np.mean(correct_ratios)),
            'incorrect_mean': float(np.mean(incorrect_ratios)),
            'correct_median': float(np.median(correct_ratios)),
            'incorrect_median': float(np.median(incorrect_ratios)),
            't_stat': float(t_stat),
            'p_value': float(p_value),
        }
    
    # Category breakdown
    cat_summary = {}
    for cat, data in category_stats.items():
        ratios_arr = np.array(data['ratios'])
        cat_summary[cat] = {
            'n_ratios': len(ratios_arr),
            'ratio_mean': float(np.mean(ratios_arr)) if len(ratios_arr) > 0 else None,
            'ratio_median': float(np.median(ratios_arr)) if len(ratios_arr) > 0 else None,
            'entropy_mean': float(np.mean(data['entropies'])) if data['entropies'] else None,
            'accuracy': data['correct'] / data['total'] if data['total'] > 0 else None,
        }
    result['categories'] = cat_summary
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, 
                        help='Single model key (e.g. pythia-160m)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick run with fewer prompts')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("  EXP 02: Multi-Model Scale-Up PAC Tree Analysis")
    print("=" * 70)
    start_time = time.time()
    
    # Select models
    if args.model:
        model_keys = [args.model]
    else:
        model_keys = list(MODELS.keys())
    
    # Select prompts
    if args.quick:
        # Take first 3 from each category
        prompts_to_run = {}
        for cat, ps in PROMPTS.items():
            prompts_to_run[cat] = ps[:3]
    else:
        prompts_to_run = PROMPTS
    
    total_prompts = sum(len(ps) for ps in prompts_to_run.values())
    print(f"  Models: {model_keys}")
    print(f"  Prompts: {total_prompts} across {len(prompts_to_run)} categories")
    print(f"  Device: {device}")
    
    # Build category map for tracking
    category_map = {}
    for cat, ps in prompts_to_run.items():
        for prompt, cont in ps:
            category_map[prompt + cont] = cat
    
    # ── Null baseline ────────────────────────────────────────────────
    # Use first model's vocab size (all Pythia models share vocab)
    print("\n" + "=" * 70)
    print("  NULL BASELINE")
    print("=" * 70)
    null_baseline = run_null_baseline(vocab_size=50304, n_positions=500)
    print(f"  Random logits: ratio mean={null_baseline['ratio_mean']:.4f}, "
          f"median={null_baseline['ratio_median']:.4f}")
    print(f"  Near phi: {null_baseline['near_phi_count']} / {null_baseline['n_ratios']} "
          f"({null_baseline['near_phi_fraction']:.1%})")
    
    # ── Run each model ───────────────────────────────────────────────
    all_model_results = {}
    
    for model_key in model_keys:
        print("\n" + "=" * 70)
        print(f"  MODEL: {model_key}")
        print("=" * 70)
        
        model, tokenizer = load_model(model_key, device)
        forests = []
        
        for cat, prompt_list in prompts_to_run.items():
            print(f"\n  Category: {cat} ({len(prompt_list)} prompts)")
            for prompt, cont in prompt_list:
                forest = analyse_single_prompt(model, tokenizer, prompt, cont, device)
                forests.append(forest)
                
                s = forest.summary()
                marker = "+" if (s['accuracy'] or 0) > 0.3 else "-"
                ratio_str = f"{s['pac_ratio_median']:.3f}" if s['pac_ratio_median'] else "N/A"
                print(f"    [{marker}] H={s['entropy_mean']:.2f} "
                      f"r={ratio_str} "
                      f"acc={s['accuracy']:.0%}" if s['accuracy'] is not None else 
                      f"    [{marker}] H={s['entropy_mean']:.2f} r={ratio_str}")
        
        # Analyse this model
        model_analysis = analyse_model_results(forests, category_map)
        all_model_results[model_key] = model_analysis
        
        # Print model summary
        print(f"\n  --- {model_key} Summary ---")
        print(f"  Tokens: {model_analysis['n_tokens']}")
        print(f"  Ratio median: {model_analysis['ratio_median']:.4f}")
        print(f"  Phi distance: {model_analysis['phi_distance_mean']:.4f}")
        print(f"  Near phi: {model_analysis['near_phi_count']} "
              f"({model_analysis['near_phi_fraction']:.1%})")
        
        if 'correct_vs_incorrect' in model_analysis:
            cvi = model_analysis['correct_vs_incorrect']
            print(f"  Correct median: {cvi['correct_median']:.4f}, "
                  f"Incorrect median: {cvi['incorrect_median']:.4f}")
            print(f"  t-test p = {cvi['p_value']:.6f}")
        
        print(f"  SEC phases:")
        for phase, data in model_analysis['sec_phases'].items():
            acc_str = f"{data['accuracy']:.0%}" if data['accuracy'] is not None else "N/A"
            r_str = f"{data['ratio_median']:.3f}" if data['ratio_median'] is not None else "N/A"
            print(f"    {phase:15s}: n={data['count']:4d}  acc={acc_str:>5s}  "
                  f"ratio_median={r_str}")
        
        # Free GPU memory
        del model
        torch.cuda.empty_cache() if device == 'cuda' else None
    
    # ── Cross-model comparison ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CROSS-MODEL COMPARISON")
    print("=" * 70)
    
    comparison = {}
    for mk, mr in all_model_results.items():
        comparison[mk] = {
            'n_tokens': mr['n_tokens'],
            'ratio_median': mr['ratio_median'],
            'phi_distance': mr['phi_distance_mean'],
            'near_phi_frac': mr['near_phi_fraction'],
            'null_near_phi_frac': null_baseline['near_phi_fraction'],
            'enrichment_vs_null': (
                mr['near_phi_fraction'] / null_baseline['near_phi_fraction']
                if null_baseline['near_phi_fraction'] > 0 else None
            ),
        }
        e = comparison[mk]['enrichment_vs_null']
        print(f"  {mk:15s}: median={mr['ratio_median']:.4f}  "
              f"phi_near={mr['near_phi_fraction']:.1%}  "
              f"enrichment={e:.1f}x" if e else 
              f"  {mk:15s}: median={mr['ratio_median']:.4f}  "
              f"phi_near={mr['near_phi_fraction']:.1%}")
    
    # ── Save ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    
    results = {
        'experiment': 'exp_02_multi_model_scale',
        'timestamp': timestamp,
        'device': device,
        'models': model_keys,
        'n_prompts': total_prompts,
        'elapsed_seconds': elapsed,
        'null_baseline': null_baseline,
        'model_results': all_model_results,
        'cross_model_comparison': comparison,
        'dft_constants': {'phi': PHI, 'inv_phi': INV_PHI, 'xi': XI},
    }
    
    # Remove histogram raw data from the large save to keep file manageable
    result_path = RESULTS_DIR / f'exp_02_multi_model_{timestamp}.json'
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {result_path}")
    print(f"Total elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
