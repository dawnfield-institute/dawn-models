#!/usr/bin/env python3
"""
EXP 04: Sequence-Level Hallucination Detection
===============================================

exp_03 showed that single-token analysis is insufficient — all prompts
produce chaotic SEC phase at the first token. This experiment generates
multi-token responses and analyses the ENTIRE PAC forest.

Hypothesis: When a model "knows" something, generated tokens will show
more crystallized/ordered SEC phases (confidence). When hallucinating,
tokens remain in chaotic/transitional phases (uncertainty).

The signal is NOT in the first token — it's in the TRAJECTORY of
SEC phases across the generated sequence.

Design:
  - Generate 30 tokens for each prompt
  - Build PAC tree at every generated token
  - Track SEC phase trajectory, ratio trajectory, entropy trajectory
  - Compare aggregate forest statistics between groups

Key metrics:
  1. Crystallized+Ordered fraction (confidence ratio)
  2. Mean PAC ratio across sequence
  3. Entropy trajectory slope (decreasing = model locking on)
  4. Phase transition count (stability metric)
  5. Max consecutive crystallized tokens (knowledge burst)

Author: Dawn Field Institute
Date: 2025-02-13
"""

import sys, os, json, time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from scipy import stats

# --- path setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR))
from core.pac_tree import build_pac_tree_from_logits, PACForest
from core.collapse_metrics import (
    classify_sec_phase, SECPhase, compute_collapse_signature
)

# --- constants ---
PHI = (1 + 5**0.5) / 2
TOP_K = 10
GEN_TOKENS = 30  # tokens to generate per prompt
CACHE_DIR = str(Path(EXPERIMENT_DIR).parent / "huggingface_bifractal_validation" / "pythia_cache")

# ── Prompt groups ─────────────────────────────────────────────────

# Things Pythia-160m should genuinely know (high-frequency training data)
FACTUAL_PROMPTS = [
    "The capital of France is",
    "Water freezes at zero degrees",
    "The sun is a star that",
    "Dogs are mammals that",
    "The Earth orbits around the",
    "Humans need oxygen to",
    "The ocean is full of",
    "Trees produce oxygen through",
    "The moon orbits the",
    "Birds have wings and can",
    "Fish breathe using their",
    "Rain falls from clouds when",
    "The heart pumps blood through",
    "Computers process information using",
    "Books contain pages filled with",
]

# Things that will force hallucination (fictional, impossible, or unknowable)
HALLUCINATION_PROMPTS = [
    "The president of the fictional nation of Zorblandia announced",
    "Professor Blinkworth's theory of quantum gastronomy states that",
    "The chemical compound Fizzium-99 was discovered in",
    "In the year 3847, humans colonized the planet",
    "The underwater city of Deepheim was founded by",
    "Dr. Thornwick proved that time travel requires",
    "The 500th digit of pi multiplied by the 237th prime equals",
    "The Blatherstein equation for recursive entropy states",
    "The lost continent of Meridia sank because",
    "According to the Fictional Encyclopedia of 2099,",
    "The alien species known as the Glorpians communicate by",
    "In alternate universe 7B, gravity works by",
    "The mythical Zephyr Crystal was forged using",
    "The Voynich manuscript was decoded and it says",
    "The exact population of Earth in 1 BC was",
]

# Tricky: things that sound hard but are knowable
TRICKY_PROMPTS = [
    "A group of crows is called a",
    "The largest planet in our solar system is",
    "Shakespeare wrote the play Romeo and",
    "The speed of light is approximately",
    "Einstein's famous equation is E equals",
]


def load_model(model_name="pythia-160m"):
    """Load a Pythia model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    full_name = f"EleutherAI/{model_name}"
    print(f"Loading {full_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        full_name, cache_dir=CACHE_DIR
    )
    model = AutoModelForCausalLM.from_pretrained(
        full_name, cache_dir=CACHE_DIR
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded. {param_count:,} params on {device}")
    
    return model, tokenizer, device


def generate_with_pac_forest(model, tokenizer, device, prompt, n_tokens=GEN_TOKENS, top_k=TOP_K):
    """
    Generate n_tokens autoregressively, building a PAC tree at each step.
    
    Returns:
        dict with full sequence analysis
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    trees = []
    generated_tokens = []
    generated_text_parts = []
    
    current_ids = input_ids
    
    for step in range(n_tokens):
        with torch.no_grad():
            outputs = model(current_ids)
            # Get logits for the LAST token position (next token prediction)
            logits = outputs.logits[0, -1, :]
        
        # Build PAC tree from the logit distribution
        tree = build_pac_tree_from_logits(logits.cpu(), position=step, top_k=top_k)
        
        # Greedy decode: pick the top token
        chosen_id = int(logits.argmax())
        chosen_prob = float(torch.softmax(logits, dim=-1)[chosen_id])
        chosen_text = tokenizer.decode([chosen_id])
        
        # Compute PAC ratio and SEC phase
        pac_ratio = tree.pac_ratio_1_2 if tree.pac_ratio_1_2 is not None else float('inf')
        
        sec_phase = classify_sec_phase(tree.total_entropy)
        
        # Compute collapse signature
        signature = compute_collapse_signature(
            entropy=tree.total_entropy,
            pac_ratio_1_2=pac_ratio if pac_ratio < 1e6 else None,
            prob_conservation_error=tree.conservation_error(),
            entropy_conservation_error=tree.entropy_conservation_error(),
            top1_prob=chosen_prob,
        )
        
        tree_data = {
            "step": step,
            "token_id": chosen_id,
            "token_text": chosen_text,
            "probability": chosen_prob,
            "pac_ratio": pac_ratio,
            "entropy": tree.total_entropy,
            "sec_phase": sec_phase.name,
            "concentration": tree.children_prob_sum,
            "effective_k": float(np.exp(tree.total_entropy)) if tree.total_entropy < 20 else 0,
            "phi_aligned": signature.is_phi_aligned,
            "xi_aligned": signature.is_xi_aligned,
            "forced_collapse": signature.is_forced_collapse,
            "phi_distance": abs(pac_ratio - PHI),
        }
        
        trees.append(tree_data)
        generated_tokens.append(chosen_id)
        generated_text_parts.append(chosen_text)
        
        # Append chosen token for next step
        chosen_tensor = torch.tensor([[chosen_id]], device=device)
        current_ids = torch.cat([current_ids, chosen_tensor], dim=1)
    
    generated_text = "".join(generated_text_parts)
    
    # ── Compute sequence-level statistics ──
    ratios = [t["pac_ratio"] for t in trees if t["pac_ratio"] < 1000]
    entropies = [t["entropy"] for t in trees]
    phases = [t["sec_phase"] for t in trees]
    
    # 1. Confidence ratio: fraction of tokens in crystallized+ordered
    confident_count = sum(1 for p in phases if p in ("CRYSTALLIZED", "ORDERED"))
    confidence_ratio = confident_count / len(phases) if phases else 0
    
    # 2. Chaotic fraction
    chaotic_count = sum(1 for p in phases if p == "CHAOTIC")
    chaotic_fraction = chaotic_count / len(phases) if phases else 0
    
    # 3. Mean PAC ratio
    mean_ratio = float(np.mean(ratios)) if ratios else 0
    median_ratio = float(np.median(ratios)) if ratios else 0
    
    # 4. Entropy trajectory slope (linear regression)
    if len(entropies) >= 3:
        x = np.arange(len(entropies))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, entropies)
        entropy_slope = float(slope)
        entropy_r2 = float(r_value ** 2)
    else:
        entropy_slope = 0.0
        entropy_r2 = 0.0
    
    # 5. Phase transition count (how often phase changes between consecutive tokens)
    transitions = sum(1 for i in range(1, len(phases)) if phases[i] != phases[i-1])
    transition_rate = transitions / (len(phases) - 1) if len(phases) > 1 else 0
    
    # 6. Max consecutive crystallized/ordered tokens
    max_confident_streak = 0
    current_streak = 0
    for p in phases:
        if p in ("CRYSTALLIZED", "ORDERED"):
            current_streak += 1
            max_confident_streak = max(max_confident_streak, current_streak)
        else:
            current_streak = 0
    
    # 7. Entropy variability (std of entropy — stable = factual, variable = uncertain)
    entropy_std = float(np.std(entropies)) if entropies else 0
    
    # 8. Ratio trajectory slope
    if len(ratios) >= 3:
        x = np.arange(len(ratios))
        ratio_slope = float(stats.linregress(x, ratios).slope)
    else:
        ratio_slope = 0.0
    
    # 9. Phi alignment rate across sequence
    phi_aligned_count = sum(1 for t in trees if t["phi_aligned"])
    phi_rate = phi_aligned_count / len(trees) if trees else 0
    
    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "n_tokens": len(trees),
        "tokens": trees,
        # Sequence-level statistics
        "confidence_ratio": confidence_ratio,
        "chaotic_fraction": chaotic_fraction,
        "mean_ratio": mean_ratio,
        "median_ratio": median_ratio,
        "entropy_slope": entropy_slope,
        "entropy_r2": entropy_r2,
        "entropy_std": entropy_std,
        "transition_rate": transition_rate,
        "max_confident_streak": max_confident_streak,
        "ratio_slope": ratio_slope,
        "phi_rate": phi_rate,
        "mean_entropy": float(np.mean(entropies)),
        "phase_distribution": {
            phase: sum(1 for p in phases if p == phase)
            for phase in ("CRYSTALLIZED", "ORDERED", "TRANSITIONAL", "CHAOTIC")
        },
    }


def compare_groups(group_a_results, group_b_results, label_a="Factual", label_b="Hallucinated"):
    """Compare sequence-level statistics between two groups."""
    
    metrics = [
        "confidence_ratio",
        "chaotic_fraction", 
        "mean_ratio",
        "median_ratio",
        "entropy_slope",
        "entropy_std",
        "transition_rate",
        "max_confident_streak",
        "ratio_slope",
        "phi_rate",
        "mean_entropy",
    ]
    
    comparison = {}
    significant_metrics = []
    
    print(f"\n{'='*70}")
    print(f"  COMPARISON: {label_a} vs {label_b}")
    print(f"{'='*70}")
    
    for metric in metrics:
        vals_a = [r[metric] for r in group_a_results]
        vals_b = [r[metric] for r in group_b_results]
        
        mean_a = np.mean(vals_a)
        mean_b = np.mean(vals_b)
        std_a = np.std(vals_a)
        std_b = np.std(vals_b)
        
        # Mann-Whitney U test (non-parametric, more robust for small samples)
        try:
            u_stat, p_mw = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')
        except ValueError:
            u_stat, p_mw = 0, 1.0
        
        # T-test
        try:
            t_stat, p_tt = stats.ttest_ind(vals_a, vals_b)
        except:
            t_stat, p_tt = 0, 1.0
        
        significant = p_mw < 0.05
        marker = " ***" if p_mw < 0.01 else " **" if p_mw < 0.05 else " *" if p_mw < 0.1 else ""
        
        print(f"\n  {metric}:{marker}")
        print(f"    {label_a:15s}: {mean_a:8.4f} +/- {std_a:.4f}")
        print(f"    {label_b:15s}: {mean_b:8.4f} +/- {std_b:.4f}")
        print(f"    Mann-Whitney p: {p_mw:.6f}  |  t-test p: {p_tt:.6f}")
        
        if significant:
            significant_metrics.append(metric)
        
        comparison[metric] = {
            f"{label_a}_mean": float(mean_a),
            f"{label_a}_std": float(std_a),
            f"{label_b}_mean": float(mean_b),
            f"{label_b}_std": float(std_b),
            "mann_whitney_p": float(p_mw),
            "ttest_p": float(p_tt),
            "significant_005": significant,
        }
    
    print(f"\n{'='*70}")
    if significant_metrics:
        print(f"  SIGNIFICANT METRICS (p < 0.05): {', '.join(significant_metrics)}")
    else:
        print(f"  No individually significant metrics at p < 0.05")
    print(f"{'='*70}")
    
    return comparison, significant_metrics


def print_phase_heatmap(results, group_name):
    """Print a visual heatmap of SEC phases across generated tokens."""
    phase_chars = {
        "CRYSTALLIZED": "█",  # solid = confident
        "ORDERED": "▓",
        "TRANSITIONAL": "▒",
        "CHAOTIC": "░",
    }
    
    print(f"\n  {group_name} — SEC Phase Trajectories")
    print(f"  {'█=crystallized  ▓=ordered  ▒=transitional  ░=chaotic'}")
    print(f"  {'─' * 50}")
    
    for r in results[:10]:  # Show first 10
        prompt_short = r["prompt"][:30].ljust(30)
        trajectory = "".join(
            phase_chars.get(t["sec_phase"], "?") for t in r["tokens"]
        )
        conf = f"{r['confidence_ratio']*100:4.0f}%"
        print(f"  {prompt_short} {trajectory} {conf}")


def main():
    print("=" * 70)
    print("  EXP 04: Sequence-Level Hallucination Detection")
    print("  Generate multi-token responses, analyse PAC forest trajectory")
    print("=" * 70)
    
    t0 = time.time()
    model, tokenizer, device = load_model("pythia-160m")
    
    all_results = {}
    
    # Process each group
    for group_name, prompts in [
        ("factual", FACTUAL_PROMPTS),
        ("hallucination", HALLUCINATION_PROMPTS),
        ("tricky", TRICKY_PROMPTS),
    ]:
        print(f"\n{'='*60}")
        print(f"  GROUP: {group_name} ({len(prompts)} prompts, {GEN_TOKENS} tokens each)")
        print(f"{'='*60}")
        
        group_results = []
        for i, prompt in enumerate(prompts):
            result = generate_with_pac_forest(model, tokenizer, device, prompt)
            group_results.append(result)
            
            # Progress indicator
            prompt_short = prompt[:40] + "..." if len(prompt) > 40 else prompt
            print(f"  [{i+1:2d}/{len(prompts)}] conf={result['confidence_ratio']*100:4.0f}%  "
                  f"chaos={result['chaotic_fraction']*100:4.0f}%  "
                  f"H_slope={result['entropy_slope']:+.3f}  "
                  f"streak={result['max_confident_streak']:2d}  "
                  f"'{prompt_short}'")
        
        all_results[group_name] = group_results
        
        # Print phase trajectory heatmap
        print_phase_heatmap(group_results, group_name)
    
    # ── Comparisons ──
    comparison_fh, sig_fh = compare_groups(
        all_results["factual"],
        all_results["hallucination"],
        "Factual", "Hallucinated"
    )
    
    comparison_ft, sig_ft = compare_groups(
        all_results["factual"],
        all_results["tricky"],
        "Factual", "Tricky"
    )
    
    # ── Summary ──
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    
    for gname in ["factual", "hallucination", "tricky"]:
        results = all_results[gname]
        mean_conf = np.mean([r["confidence_ratio"] for r in results])
        mean_chaos = np.mean([r["chaotic_fraction"] for r in results])
        mean_slope = np.mean([r["entropy_slope"] for r in results])
        mean_streak = np.mean([r["max_confident_streak"] for r in results])
        mean_trans = np.mean([r["transition_rate"] for r in results])
        
        print(f"\n  {gname:15s}:  conf={mean_conf*100:5.1f}%  chaos={mean_chaos*100:5.1f}%  "
              f"H_slope={mean_slope:+.4f}  streak={mean_streak:.1f}  trans_rate={mean_trans:.3f}")
    
    # ── Phase distribution aggregate ──
    print(f"\n  Phase Distribution (aggregate across all tokens):")
    for gname in ["factual", "hallucination", "tricky"]:
        results = all_results[gname]
        total_phases = {}
        total = 0
        for r in results:
            for phase, count in r["phase_distribution"].items():
                total_phases[phase] = total_phases.get(phase, 0) + count
                total += count
        
        parts = []
        for phase in ("CRYSTALLIZED", "ORDERED", "TRANSITIONAL", "CHAOTIC"):
            c = total_phases.get(phase, 0)
            parts.append(f"{phase[:4]}={c/total*100:4.1f}%")
        print(f"    {gname:15s}: {' | '.join(parts)}  (n={total})")
    
    # ── Save results ──
    output = {
        "experiment": "exp_04_sequence_hallucination",
        "model": "pythia-160m",
        "gen_tokens": GEN_TOKENS,
        "top_k": TOP_K,
        "timestamp": datetime.now().isoformat(),
        "groups": {
            gname: {
                "n_prompts": len(results),
                "summary": {
                    "mean_confidence_ratio": float(np.mean([r["confidence_ratio"] for r in results])),
                    "mean_chaotic_fraction": float(np.mean([r["chaotic_fraction"] for r in results])),
                    "mean_entropy_slope": float(np.mean([r["entropy_slope"] for r in results])),
                    "mean_max_streak": float(np.mean([r["max_confident_streak"] for r in results])),
                    "mean_transition_rate": float(np.mean([r["transition_rate"] for r in results])),
                    "mean_ratio": float(np.mean([r["mean_ratio"] for r in results])),
                    "mean_entropy": float(np.mean([r["mean_entropy"] for r in results])),
                },
                "prompts": [
                    {k: v for k, v in r.items() if k != "tokens"}
                    for r in results
                ],
            }
            for gname, results in all_results.items()
        },
        "comparisons": {
            "factual_vs_hallucination": {
                "metrics": comparison_fh,
                "significant_metrics": sig_fh,
            },
            "factual_vs_tricky": {
                "metrics": comparison_ft,
                "significant_metrics": sig_ft,
            },
        },
    }
    
    results_dir = EXPERIMENT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"exp_04_sequence_hallucination_{ts}.json"
    
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
