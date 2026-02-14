#!/usr/bin/env python3
"""
exp_03_hallucination_signatures.py
===================================

Test whether PAC tree collapse signatures can distinguish
hallucinated tokens from correct tokens.

HYPOTHESIS:
  A hallucination is a "forced collapse" — the model produces a
  confident-looking token despite high internal uncertainty. The PAC
  tree signature should show: high parent entropy, but disproportionate
  collapse magnitude. The "structure cost of erasure" is higher than
  the information budget justifies.

  Correct predictions should show natural SEC collapse (entropy matches
  confidence). Hallucinations should show a gap between internal 
  uncertainty and external confidence.

DESIGN:
  1. Use prompts where Pythia is LIKELY to hallucinate (obscure facts,
     numerical claims, fake entities)
  2. Use prompts where Pythia is LIKELY to be correct (common patterns,
     simple facts)
  3. Compare PAC tree signatures between the two groups
  4. Test specific diagnostic signals:
     - Forced collapse rate (high H + high p1)
     - Entropy-confidence gap
     - PAC ratio distribution shift
     - Conservation anomalies

FALSIFICATION:
  If PAC tree signatures are identical for hallucinated and correct
  tokens, the diagnostic is falsified.

Usage:
    python exp_03_hallucination_signatures.py
"""

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
from core.collapse_metrics import (
    classify_sec_phase, compute_collapse_signature,
    compute_conservation_budget, SECPhase,
)

PYTHIA_CACHE = str(EXPERIMENT_DIR.parent / 'huggingface_bifractal_validation' / 'pythia_cache')
MODEL_ID = 'EleutherAI/pythia-160m'
TOP_K = 20  # wider window to see distribution shape
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Prompts designed to trigger or avoid hallucination ────────────────

# SHOULD_KNOW: common knowledge a 160M model likely learned
SHOULD_KNOW = [
    ("The sun rises in the", " east"),
    ("Water is made of hydrogen and", " oxygen"),
    ("The capital of the United States is", " Washington"),
    ("One plus one equals", " two"),
    ("The color of the sky is", " blue"),
    ("Dogs are commonly kept as", " pets"),
    ("The opposite of hot is", " cold"),
    ("Bread is made from", " flour"),
    ("The Earth has one natural satellite called the", " Moon"),
    ("Humans breathe", " oxygen"),
    ("Fire is", " hot"),
    ("Ice is", " cold"),
    ("The sun is a", " star"),
    ("Birds can", " fly"),
    ("Fish live in", " water"),
]

# WILL_HALLUCINATE: obscure, fake, or numerical prompts that should trip up a small model
WILL_HALLUCINATE = [
    ("The 37th president of the fictional country of Zarlandia was", " Emperor"),
    ("The population of the city of Xylphoria in 2024 was exactly", " 3"),
    ("Professor Thornwick Blatherstein's most famous theorem proves that", " the"),
    ("The chemical element Fictionium has an atomic number of", " 1"),
    ("In the year 2847, the dominant programming language was", " Python"),
    ("The capital of the underwater nation of Atlantia is", " the"),
    ("Dr. Quentin Hargrove discovered in 1923 that", " the"),
    ("The Blinkworth equation states that entropy divided by", " the"),
    ("The 147th digit of pi is", " 5"),
    ("The airspeed velocity of an unladen swallow in meters per second is", " approximately"),
    ("The exact GDP of Luxembourg in 1847 was", " $"),
    ("The molecular weight of the protein titin to three decimal places is", " 3"),
    ("In quantum chromodynamics, the seventh gluon coupling constant equals", " 0"),
    ("The Riemann hypothesis was proven in the year", " 20"),
    ("The longest word in the Welsh language is", " ll"),
]

# TRICKY: facts that sound like they could be wrong but are right
TRICKY = [
    ("A group of flamingos is called a", " fl"),
    ("The largest desert in the world is the", " S"),
    ("Honey never", " expires"),
    ("Octopuses have three", " hearts"),
    ("Bananas are technically", " ber"),
]


def load_model():
    """Load Pythia-160m."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=PYTHIA_CACHE)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=PYTHIA_CACHE).to(DEVICE)
    model.eval()
    print(f"  Loaded. {sum(p.numel() for p in model.parameters()):,} params on {DEVICE}")
    return model, tokenizer


def build_detailed_forest(model, tokenizer, prompt: str, continuation: str) -> dict:
    """Build PAC forest with detailed per-token collapse signatures."""
    full_text = prompt + continuation
    input_ids = tokenizer.encode(full_text, return_tensors='pt').to(DEVICE)
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt')
    prompt_len = prompt_ids.shape[1]
    seq_len = input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0]
    
    trees = []
    signatures = []
    
    for pos in range(seq_len - 1):
        pos_logits = logits[pos]
        gt_id = input_ids[0, pos + 1].item()
        selected_id = pos_logits.argmax().item()
        
        tree = build_pac_tree_from_logits(
            logits=pos_logits, position=pos, tokenizer=tokenizer,
            top_k=TOP_K, selected_token_id=selected_id, ground_truth_id=gt_id,
        )
        trees.append(tree)
        
        # Compute full collapse signature
        prob_err, ent_err = compute_conservation_budget(
            tree.children_prob_sum, tree.tail_prob,
            tree.children_entropy_sum, tree.tail_entropy,
            tree.total_entropy,
        )
        
        top1_prob = tree.children[0].probability if tree.children else 0.0
        
        sig = compute_collapse_signature(
            entropy=tree.total_entropy,
            pac_ratio_1_2=tree.pac_ratio_1_2,
            prob_conservation_error=prob_err,
            entropy_conservation_error=ent_err,
            top1_prob=top1_prob,
        )
        signatures.append(sig)
    
    # Separate prompt tokens from continuation tokens  
    # The continuation tokens start at position prompt_len-1 (predicting prompt_len onwards)
    continuation_start = max(0, prompt_len - 1)
    
    prompt_trees = trees[:continuation_start]
    continuation_trees = trees[continuation_start:]
    prompt_sigs = signatures[:continuation_start]
    continuation_sigs = signatures[continuation_start:]
    
    return {
        'prompt': prompt,
        'continuation': continuation,
        'prompt_len': prompt_len,
        'total_len': seq_len,
        'all_trees': trees,
        'all_signatures': signatures,
        'continuation_trees': continuation_trees,
        'continuation_signatures': continuation_sigs,
    }


def extract_signature_stats(signatures: list, trees: list) -> dict:
    """Extract aggregate statistics from collapse signatures."""
    if not signatures:
        return {}
    
    entropies = [s.entropy for s in signatures]
    concentrations = [s.concentration for s in signatures]
    effective_ks = [s.effective_k for s in signatures]
    forced = sum(1 for s in signatures if s.is_forced_collapse)
    phi_aligned = sum(1 for s in signatures if s.is_phi_aligned)
    
    # Phase distribution
    phases = [s.sec_phase.value for s in signatures]
    phase_counts = {p: phases.count(p) for p in ['crystallized', 'ordered', 'transitional', 'chaotic']}
    
    # Accuracy from trees
    correct = sum(1 for t in trees if t.is_correct is True)
    scored = sum(1 for t in trees if t.is_correct is not None)
    
    # PAC ratios
    ratios = [s.pac_ratio_1_2 for s in signatures if s.pac_ratio_1_2 is not None]
    
    # Entropy-confidence gap: high entropy but also high top-1 prob
    # This is the hallucination signal
    gaps = []
    for s in signatures:
        if s.entropy > 0:
            # A "natural" collapse has low entropy AND high confidence
            # A "forced" collapse has high entropy AND high confidence
            # Measure: confidence relative to what entropy would predict
            expected_concentration = 1.0 / s.effective_k if s.effective_k > 0 else 1.0
            gap = s.concentration - expected_concentration
            gaps.append(gap)
    
    result = {
        'n_tokens': len(signatures),
        'entropy_mean': float(np.mean(entropies)),
        'entropy_std': float(np.std(entropies)),
        'entropy_median': float(np.median(entropies)),
        'concentration_mean': float(np.mean(concentrations)),
        'concentration_median': float(np.median(concentrations)),
        'effective_k_mean': float(np.mean(effective_ks)),
        'effective_k_median': float(np.median(effective_ks)),
        'forced_collapse_count': forced,
        'forced_collapse_rate': forced / len(signatures),
        'phi_aligned_count': phi_aligned,
        'phi_aligned_rate': phi_aligned / len(signatures),
        'phase_distribution': phase_counts,
        'accuracy': correct / scored if scored > 0 else None,
        'n_correct': correct,
        'n_scored': scored,
    }
    
    if ratios:
        result['ratio_mean'] = float(np.mean(ratios))
        result['ratio_median'] = float(np.median(ratios))
        result['ratio_std'] = float(np.std(ratios))
        result['phi_distance_mean'] = float(np.mean(np.abs(np.array(ratios) - PHI)))
    
    if gaps:
        result['entropy_confidence_gap_mean'] = float(np.mean(gaps))
        result['entropy_confidence_gap_std'] = float(np.std(gaps))
        result['entropy_confidence_gap_median'] = float(np.median(gaps))
    
    return result


def compare_groups(group_a_stats: list, group_b_stats: list, 
                   label_a: str, label_b: str) -> dict:
    """Statistical comparison between two groups of signatures."""
    from scipy import stats
    
    def collect(stats_list, key):
        vals = [s.get(key) for s in stats_list if s.get(key) is not None]
        return np.array(vals) if vals else np.array([])
    
    comparisons = {}
    
    metrics_to_compare = [
        'entropy_mean', 'concentration_mean', 'effective_k_mean',
        'forced_collapse_rate', 'phi_aligned_rate',
        'ratio_median', 'phi_distance_mean',
        'entropy_confidence_gap_mean',
    ]
    
    for metric in metrics_to_compare:
        vals_a = collect(group_a_stats, metric)
        vals_b = collect(group_b_stats, metric)
        
        if len(vals_a) >= 2 and len(vals_b) >= 2:
            t_stat, p_value = stats.ttest_ind(vals_a, vals_b, equal_var=False)
            # Also Mann-Whitney U for non-parametric
            try:
                u_stat, u_pvalue = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')
            except ValueError:
                u_stat, u_pvalue = None, None
            
            comparisons[metric] = {
                f'{label_a}_mean': float(np.mean(vals_a)),
                f'{label_b}_mean': float(np.mean(vals_b)),
                f'{label_a}_std': float(np.std(vals_a)),
                f'{label_b}_std': float(np.std(vals_b)),
                'difference': float(np.mean(vals_a) - np.mean(vals_b)),
                't_stat': float(t_stat),
                'p_value_ttest': float(p_value),
                'u_stat': float(u_stat) if u_stat is not None else None,
                'p_value_mannwhitney': float(u_pvalue) if u_pvalue is not None else None,
            }
    
    return comparisons


def main():
    print("=" * 70)
    print("  EXP 03: Hallucination Signature Detection")
    print("  Can PAC trees distinguish hallucinated from correct tokens?")
    print("=" * 70)
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model, tokenizer = load_model()
    
    # ── Run all prompt groups ────────────────────────────────────────
    groups = {
        'should_know': SHOULD_KNOW,
        'will_hallucinate': WILL_HALLUCINATE,
        'tricky': TRICKY,
    }
    
    all_results = {}
    group_sig_stats = {}
    
    for group_name, prompts in groups.items():
        print(f"\n{'=' * 60}")
        print(f"  GROUP: {group_name} ({len(prompts)} prompts)")
        print(f"{'=' * 60}")
        
        group_data = []
        group_stats = []
        
        for prompt, cont in prompts:
            result = build_detailed_forest(model, tokenizer, prompt, cont)
            
            # Focus on continuation tokens (where the model is "answering")
            cont_stats = extract_signature_stats(
                result['continuation_signatures'],
                result['continuation_trees'],
            )
            
            # Also get full stats for context
            full_stats = extract_signature_stats(
                result['all_signatures'],
                result['all_trees'],
            )
            
            acc_str = f"{cont_stats.get('accuracy', 0):.0%}" if cont_stats.get('accuracy') is not None else "N/A"
            fc_str = f"{cont_stats.get('forced_collapse_rate', 0):.0%}"
            phi_str = f"{cont_stats.get('ratio_median', 0):.3f}" if cont_stats.get('ratio_median') else "N/A"
            
            print(f"  [{acc_str:>4s}] fc={fc_str}  phi_r={phi_str}  "
                  f"H={cont_stats.get('entropy_mean', 0):.2f}  "
                  f"'{prompt[:40]}...'")
            
            prompt_result = {
                'prompt': prompt,
                'continuation': cont,
                'continuation_stats': cont_stats,
                'full_stats': full_stats,
                'trees': [t.to_dict() for t in result['all_trees']],
            }
            group_data.append(prompt_result)
            group_stats.append(cont_stats)
        
        all_results[group_name] = group_data
        group_sig_stats[group_name] = group_stats
    
    # ── Compare groups ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  COMPARISON: should_know vs will_hallucinate")
    print("=" * 70)
    
    comparison = compare_groups(
        group_sig_stats['should_know'],
        group_sig_stats['will_hallucinate'],
        'known', 'hallucinated',
    )
    
    for metric, stats in comparison.items():
        sig = "***" if stats['p_value_ttest'] < 0.01 else "**" if stats['p_value_ttest'] < 0.05 else "*" if stats['p_value_ttest'] < 0.1 else ""
        print(f"\n  {metric}:")
        print(f"    Known:        {stats['known_mean']:.4f} +/- {stats['known_std']:.4f}")
        print(f"    Hallucinated: {stats['hallucinated_mean']:.4f} +/- {stats['hallucinated_std']:.4f}")
        print(f"    p-value:      {stats['p_value_ttest']:.6f} {sig}")
    
    # ── Aggregate phase comparison ───────────────────────────────────
    print("\n" + "=" * 70)
    print("  SEC PHASE DISTRIBUTION BY GROUP")
    print("=" * 70)
    
    for group_name, stats_list in group_sig_stats.items():
        phase_totals = defaultdict(int)
        total = 0
        for s in stats_list:
            if 'phase_distribution' in s:
                for phase, count in s['phase_distribution'].items():
                    phase_totals[phase] += count
                    total += count
        
        print(f"\n  {group_name}:")
        for phase in ['crystallized', 'ordered', 'transitional', 'chaotic']:
            pct = phase_totals[phase] / total * 100 if total > 0 else 0
            print(f"    {phase:15s}: {phase_totals[phase]:4d} ({pct:.1f}%)")
    
    # ── Save ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    
    results = {
        'experiment': 'exp_03_hallucination_signatures',
        'timestamp': timestamp,
        'model': MODEL_ID,
        'device': DEVICE,
        'top_k': TOP_K,
        'elapsed_seconds': elapsed,
        'group_comparison': comparison,
        'group_summaries': {
            group: [s for s in stats] 
            for group, stats in group_sig_stats.items()
        },
        'detailed_results': all_results,
    }
    
    result_path = RESULTS_DIR / f'exp_03_hallucination_{timestamp}.json'
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n\nResults saved to: {result_path}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
