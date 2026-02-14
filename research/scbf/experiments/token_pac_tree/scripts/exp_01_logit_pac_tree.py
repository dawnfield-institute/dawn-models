#!/usr/bin/env python3
"""
exp_01_logit_pac_tree.py
========================

First experiment: Build PAC trees from Pythia token predictions
and test whether DFT constants (phi, 1/phi, xi) appear in the
logit collapse dynamics at inference time.

HYPOTHESIS:
  When an LLM predicts a token, the softmax distribution collapses
  from a broad potential to a single actualization — structurally
  identical to SEC collapse. If PAC conservation governs this process,
  then the ratio p1/p2 (top two token probabilities) should cluster
  near phi (1.618) or 1/phi (0.618), and the SEC phase signature
  should differ between correct predictions and hallucinations.

WHAT THIS SCRIPT DOES:
  1. Loads Pythia-160m (cached locally from prior experiments)
  2. Runs it on factual prompts where ground truth is known
  3. At each token position, builds a PAC tree from the logit distribution
  4. Computes: PAC ratios, SEC phases, conservation budgets, collapse signatures
  5. Tests: Do ratios cluster near phi? Does SEC phase predict correctness?
  6. Saves full results + summary statistics as JSON

FALSIFICATION:
  If pac_ratio distributions are indistinguishable from uniform random,
  or if SEC phase has zero correlation with prediction correctness,
  the hypothesis is falsified.

Usage:
    python exp_01_logit_pac_tree.py
"""

import json
import sys
import time
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# ── Path setup ───────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
CORE_DIR = EXPERIMENT_DIR / 'core'
RESULTS_DIR = EXPERIMENT_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(EXPERIMENT_DIR))

from core.pac_tree import build_pac_tree_from_logits, PACForest, PHI, INV_PHI, XI
from core.collapse_metrics import (
    compute_collapse_signature,
    compute_conservation_budget,
    classify_sec_phase,
)

# ── Constants ────────────────────────────────────────────────────────
PYTHIA_CACHE = str(EXPERIMENT_DIR.parent / 'huggingface_bifractal_validation' / 'pythia_cache')
MODEL_ID = 'EleutherAI/pythia-160m'
TOP_K = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Test prompts with known continuations ────────────────────────────
# Format: (prompt, expected_continuation)
# We use factual prompts where the model should know the answer,
# and nonsense prompts where it should be uncertain.
TEST_PROMPTS = [
    # Factual — model should be confident and correct
    ("The capital of France is", " Paris"),
    ("Water freezes at", " zero"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("In mathematics, pi is approximately", " 3"),
    ("The speed of light is approximately", " 300"),

    # Common patterns — high confidence expected
    ("Once upon a time", " there"),
    ("The quick brown fox", " jumps"),
    ("To be or not to", " be"),

    # Technical — moderate confidence
    ("The Pythagorean theorem states that", " the"),
    ("In quantum mechanics, the wave function", " is"),

    # Ambiguous — should show high entropy / broad distribution
    ("The best way to", " get"),
    ("I think that", " the"),
    ("Yesterday I went to the", " store"),
]


def load_model_and_tokenizer():
    """Load Pythia-160m with local cache."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_ID} from cache...")
    print(f"  Cache dir: {PYTHIA_CACHE}")
    print(f"  Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=PYTHIA_CACHE
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=PYTHIA_CACHE
    ).to(DEVICE)
    model.eval()

    print(f"  Loaded. Vocab size: {tokenizer.vocab_size}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def analyse_prompt(
    model,
    tokenizer,
    prompt: str,
    expected_continuation: str,
    top_k: int = TOP_K,
) -> PACForest:
    """Run model on a prompt and build PAC trees for each predicted token position.

    We encode the full string (prompt + expected continuation) and then
    analyse every position where the model is predicting the next token.
    """
    full_text = prompt + expected_continuation
    input_ids = tokenizer.encode(full_text, return_tensors='pt').to(DEVICE)
    seq_len = input_ids.shape[1]

    # Forward pass — get logits for all positions
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0]  # [seq_len, vocab_size]

    # Build PAC forest
    forest = PACForest()
    forest.prompt_text = full_text
    forest.model_name = MODEL_ID

    # For each position, the model predicts position+1
    for pos in range(seq_len - 1):
        pos_logits = logits[pos]  # logits predicting token at pos+1
        gt_token_id = input_ids[0, pos + 1].item()

        # Greedy selection
        selected_id = pos_logits.argmax().item()

        tree = build_pac_tree_from_logits(
            logits=pos_logits,
            position=pos,
            tokenizer=tokenizer,
            top_k=top_k,
            selected_token_id=selected_id,
            ground_truth_id=gt_token_id,
        )
        forest.add_tree(tree)

    return forest


def analyse_pac_ratios(forests: list) -> dict:
    """Statistical analysis of PAC ratios across all forests."""
    all_ratios = []
    correct_ratios = []
    incorrect_ratios = []

    for forest in forests:
        for tree in forest.trees:
            if tree.pac_ratio_1_2 is not None:
                all_ratios.append(tree.pac_ratio_1_2)
                if tree.is_correct is True:
                    correct_ratios.append(tree.pac_ratio_1_2)
                elif tree.is_correct is False:
                    incorrect_ratios.append(tree.pac_ratio_1_2)

    all_ratios = np.array(all_ratios)
    correct_ratios = np.array(correct_ratios)
    incorrect_ratios = np.array(incorrect_ratios)

    # Distance from phi
    phi_distances = np.abs(all_ratios - PHI)
    inv_phi_distances = np.abs(all_ratios - INV_PHI)

    # Binned distribution: how many ratios fall near phi vs elsewhere
    phi_window = 0.1  # within 0.1 of phi
    near_phi = np.sum(phi_distances < phi_window)
    near_inv_phi = np.sum(inv_phi_distances < phi_window)

    # Null test: if ratios were uniform on [0, max(ratios)],
    # what fraction would land near phi by chance?
    max_ratio = np.max(all_ratios) if len(all_ratios) > 0 else 1.0
    expected_near_phi = (2 * phi_window / max_ratio) if max_ratio > 0 else 0

    result = {
        'n_total': len(all_ratios),
        'n_correct': len(correct_ratios),
        'n_incorrect': len(incorrect_ratios),
        'ratio_mean': float(np.mean(all_ratios)) if len(all_ratios) > 0 else None,
        'ratio_median': float(np.median(all_ratios)) if len(all_ratios) > 0 else None,
        'ratio_std': float(np.std(all_ratios)) if len(all_ratios) > 0 else None,
        'phi_distance_mean': float(np.mean(phi_distances)) if len(phi_distances) > 0 else None,
        'inv_phi_distance_mean': float(np.mean(inv_phi_distances)) if len(inv_phi_distances) > 0 else None,
        'near_phi_count': int(near_phi),
        'near_inv_phi_count': int(near_inv_phi),
        'near_phi_fraction': float(near_phi / len(all_ratios)) if len(all_ratios) > 0 else None,
        'expected_near_phi_fraction': expected_near_phi,
        'enrichment_vs_null': (
            float((near_phi / len(all_ratios)) / expected_near_phi)
            if len(all_ratios) > 0 and expected_near_phi > 0 else None
        ),
    }

    # Correct vs incorrect comparison
    if len(correct_ratios) > 0 and len(incorrect_ratios) > 0:
        result['correct_ratio_mean'] = float(np.mean(correct_ratios))
        result['incorrect_ratio_mean'] = float(np.mean(incorrect_ratios))
        result['correct_phi_distance'] = float(np.mean(np.abs(correct_ratios - PHI)))
        result['incorrect_phi_distance'] = float(np.mean(np.abs(incorrect_ratios - PHI)))

        # t-test: are correct and incorrect ratios drawn from different distributions?
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(correct_ratios, incorrect_ratios, equal_var=False)
        result['correct_vs_incorrect_t'] = float(t_stat)
        result['correct_vs_incorrect_p'] = float(p_value)

    return result


def analyse_sec_phases(forests: list) -> dict:
    """SEC phase distribution and correlation with correctness."""
    phase_counts = {'crystallized': 0, 'ordered': 0, 'transitional': 0, 'chaotic': 0}
    phase_correct = {'crystallized': 0, 'ordered': 0, 'transitional': 0, 'chaotic': 0}
    phase_total = {'crystallized': 0, 'ordered': 0, 'transitional': 0, 'chaotic': 0}

    forced_collapse_count = 0
    total_tokens = 0

    for forest in forests:
        for tree in forest.trees:
            total_tokens += 1
            phase = classify_sec_phase(tree.total_entropy)
            phase_name = phase.value
            phase_counts[phase_name] += 1

            if tree.is_correct is not None:
                phase_total[phase_name] += 1
                if tree.is_correct:
                    phase_correct[phase_name] += 1

            # Forced collapse check
            if tree.total_entropy > 2.0 and len(tree.children) > 0:
                if tree.children[0].probability > 0.5:
                    forced_collapse_count += 1

    # Accuracy by phase
    phase_accuracy = {}
    for phase in phase_counts:
        if phase_total[phase] > 0:
            phase_accuracy[phase] = phase_correct[phase] / phase_total[phase]
        else:
            phase_accuracy[phase] = None

    return {
        'total_tokens': total_tokens,
        'phase_distribution': phase_counts,
        'phase_accuracy': phase_accuracy,
        'forced_collapse_count': forced_collapse_count,
        'forced_collapse_fraction': forced_collapse_count / total_tokens if total_tokens > 0 else 0,
    }


def analyse_phi_alignment(forests: list) -> dict:
    """Check for phi alignment patterns across different conditions."""
    # Group by SEC phase
    phase_ratios = {'crystallized': [], 'ordered': [], 'transitional': [], 'chaotic': []}

    for forest in forests:
        for tree in forest.trees:
            if tree.pac_ratio_1_2 is not None:
                phase = classify_sec_phase(tree.total_entropy)
                phase_ratios[phase.value].append(tree.pac_ratio_1_2)

    result = {}
    for phase, ratios in phase_ratios.items():
        if len(ratios) > 0:
            ratios_arr = np.array(ratios)
            result[f'{phase}_n'] = len(ratios)
            result[f'{phase}_ratio_mean'] = float(np.mean(ratios_arr))
            result[f'{phase}_ratio_median'] = float(np.median(ratios_arr))
            result[f'{phase}_phi_distance_mean'] = float(np.mean(np.abs(ratios_arr - PHI)))
            result[f'{phase}_inv_phi_distance_mean'] = float(np.mean(np.abs(ratios_arr - INV_PHI)))

    return result


def main():
    """Run the full experiment."""
    print("=" * 70)
    print("  EXP 01: Token-Level PAC Tree Analysis")
    print("  Hypothesis: LLM logit collapse follows PAC conservation")
    print("=" * 70)
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Run all prompts
    forests = []
    print(f"\nAnalysing {len(TEST_PROMPTS)} prompts...")

    for i, (prompt, expected) in enumerate(TEST_PROMPTS):
        print(f"\n--- Prompt {i+1}/{len(TEST_PROMPTS)} ---")
        print(f"  '{prompt}' -> '{expected}'")
        forest = analyse_prompt(model, tokenizer, prompt, expected, top_k=TOP_K)
        forests.append(forest)

        summary = forest.summary()
        print(f"  Tokens: {summary['n_tokens']}")
        print(f"  Entropy: mean={summary['entropy_mean']:.3f}, "
              f"range=[{summary['entropy_min']:.3f}, {summary['entropy_max']:.3f}]")
        if summary['pac_ratio_mean'] is not None:
            print(f"  PAC ratio (p1/p2): mean={summary['pac_ratio_mean']:.4f}, "
                  f"median={summary['pac_ratio_median']:.4f}")
            print(f"  Phi distance: {summary['phi_distance_mean']:.4f}")
        print(f"  Accuracy: {summary['accuracy']}")

    # ── Aggregate analysis ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  AGGREGATE ANALYSIS")
    print("=" * 70)

    pac_results = analyse_pac_ratios(forests)
    sec_results = analyse_sec_phases(forests)
    phi_results = analyse_phi_alignment(forests)

    print(f"\n[PAC Ratios]")
    print(f"  Total tokens analysed: {pac_results['n_total']}")
    print(f"  Mean p1/p2 ratio: {pac_results['ratio_mean']:.4f}")
    print(f"  Median p1/p2 ratio: {pac_results['ratio_median']:.4f}")
    print(f"  Mean distance from phi: {pac_results['phi_distance_mean']:.4f}")
    print(f"  Mean distance from 1/phi: {pac_results['inv_phi_distance_mean']:.4f}")
    print(f"  Near phi (within 0.1): {pac_results['near_phi_count']} "
          f"({pac_results['near_phi_fraction']:.1%})")
    print(f"  Expected by chance: {pac_results['expected_near_phi_fraction']:.1%}")
    if pac_results.get('enrichment_vs_null'):
        print(f"  Enrichment vs null: {pac_results['enrichment_vs_null']:.2f}x")
    if pac_results.get('correct_vs_incorrect_p') is not None:
        print(f"\n  Correct ratio mean: {pac_results['correct_ratio_mean']:.4f}")
        print(f"  Incorrect ratio mean: {pac_results['incorrect_ratio_mean']:.4f}")
        print(f"  t-test p-value: {pac_results['correct_vs_incorrect_p']:.6f}")

    print(f"\n[SEC Phases]")
    for phase, count in sec_results['phase_distribution'].items():
        acc = sec_results['phase_accuracy'].get(phase)
        acc_str = f'{acc:.1%}' if acc is not None else 'N/A'
        print(f"  {phase:15s}: {count:4d} tokens, accuracy={acc_str}")
    print(f"  Forced collapses: {sec_results['forced_collapse_count']} "
          f"({sec_results['forced_collapse_fraction']:.1%})")

    print(f"\n[Phi Alignment by SEC Phase]")
    for key, val in phi_results.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")

    # ── Save results ─────────────────────────────────────────────────
    elapsed = time.time() - start_time

    results = {
        'experiment': 'exp_01_logit_pac_tree',
        'timestamp': timestamp,
        'model': MODEL_ID,
        'device': DEVICE,
        'top_k': TOP_K,
        'n_prompts': len(TEST_PROMPTS),
        'elapsed_seconds': elapsed,
        'dft_constants': {
            'phi': PHI,
            'inv_phi': INV_PHI,
            'xi': XI,
        },
        'pac_ratio_analysis': pac_results,
        'sec_phase_analysis': sec_results,
        'phi_alignment_by_phase': phi_results,
        'per_prompt_summaries': [f.summary() for f in forests],
        'per_prompt_trees': [f.to_dict() for f in forests],
    }

    result_path = RESULTS_DIR / f'exp_01_logit_pac_tree_{timestamp}.json'
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {result_path}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
