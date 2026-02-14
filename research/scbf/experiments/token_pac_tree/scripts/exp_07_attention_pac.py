#!/usr/bin/env python3
"""
EXP 07: Attention Pattern PAC — Where Collapse Actually Happens
================================================================

exp_05 showed SVD mode projections don't distinguish factual from
hallucinated text. The SVD decomposition treats weight matrices as
static capacity — but the real collapse mechanism in transformers
is ATTENTION: which tokens attend to which.

Attention IS PAC collapse:
  - Pre-attention: all positions are "potential" (full residual stream)
  - Attention weights: the "PAC tree" (how potential distributes)
  - Post-attention: collapsed to a weighted sum (actualization)

Design:
  1. Extract attention weights from every head at every layer
  2. For each head: compute PAC ratio of attention weights (a1/a2)
     — this is how concentrated the head's attention is
  3. Track "attention entropy" per head (how diffuse vs focused)
  4. Build attention PAC profiles per token:
     - How many heads are "crystallized" (attending to one position)?
     - How many are "chaotic" (attending broadly)?
     - Does the ratio of crystallized/chaotic heads predict correctness?
  5. Compare factual vs hallucinated generation

Key hypothesis: When a model "knows" something, more attention heads
should crystallize onto relevant context. During hallucination, attention
should be more diffuse/chaotic across heads.

Author: Dawn Field Institute
Date: 2026-02-13
"""

import sys, json, time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR))
from core.collapse_metrics import classify_sec_phase, SECPhase

PHI = (1 + 5**0.5) / 2
XI = 1 + np.pi / 55
CACHE_DIR = str(Path(EXPERIMENT_DIR).parent / "huggingface_bifractal_validation" / "pythia_cache")


def compute_attention_pac(attn_weights):
    """
    Compute PAC diagnostics from attention weights for one head.
    
    attn_weights: [seq_len] — attention from the last query position
    
    Returns dict with PAC ratio, entropy, concentration, SEC phase.
    """
    # Sort descending
    sorted_attn = np.sort(attn_weights)[::-1]
    
    # PAC ratio (top-1 / top-2 attention weight)
    if len(sorted_attn) >= 2 and sorted_attn[1] > 1e-10:
        pac_ratio = float(sorted_attn[0] / sorted_attn[1])
    else:
        pac_ratio = float('inf')
    
    # Attention entropy
    attn_pos = attn_weights[attn_weights > 1e-10]
    if len(attn_pos) > 0:
        entropy = float(-np.sum(attn_pos * np.log(attn_pos)))
    else:
        entropy = 0.0
    
    # Concentration (top-1 weight)
    concentration = float(sorted_attn[0]) if len(sorted_attn) > 0 else 0
    
    # Effective positions attended
    effective_k = float(np.exp(entropy)) if entropy < 20 else float('inf')
    
    # SEC phase of this head's attention
    sec_phase = classify_sec_phase(entropy)
    
    return {
        'pac_ratio': pac_ratio,
        'entropy': entropy,
        'concentration': concentration,
        'effective_k': effective_k,
        'sec_phase': sec_phase.name,
    }


def run_attention_monitoring(model, tokenizer, device, prompt, n_tokens=20):
    """
    Generate tokens while capturing attention weights from all heads.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    current_ids = input_ids
    
    token_data = []
    
    for step in range(n_tokens):
        with torch.no_grad():
            outputs = model(current_ids, output_attentions=True)
            logits = outputs.logits[0, -1, :]
            attentions = outputs.attentions  # tuple of [batch, n_heads, seq, seq]
        
        # Output-level metrics
        probs = torch.softmax(logits, dim=-1)
        chosen_id = int(logits.argmax())
        chosen_prob = float(probs[chosen_id])
        logit_entropy = float(-torch.sum(probs * torch.log(probs + 1e-30)))
        
        top2 = torch.topk(probs, 2)
        logit_pac_ratio = float(top2.values[0] / top2.values[1]) if top2.values[1] > 1e-10 else float('inf')
        output_sec = classify_sec_phase(logit_entropy)
        
        # Attention-level PAC analysis
        n_layers = len(attentions)
        
        # Per-head PAC diagnostics (last query position attending to all)
        head_diagnostics = []
        phase_counts = defaultdict(int)
        head_ratios = []
        head_entropies = []
        head_concentrations = []
        
        for layer_idx, layer_attn in enumerate(attentions):
            # layer_attn: [1, n_heads, seq_len, seq_len]
            n_heads = layer_attn.shape[1]
            
            for head_idx in range(n_heads):
                # Attention from the LAST query position
                attn_vec = layer_attn[0, head_idx, -1, :].cpu().numpy()
                
                head_pac = compute_attention_pac(attn_vec)
                head_pac['layer'] = layer_idx
                head_pac['head'] = head_idx
                head_diagnostics.append(head_pac)
                
                phase_counts[head_pac['sec_phase']] += 1
                if head_pac['pac_ratio'] < 1e6:
                    head_ratios.append(head_pac['pac_ratio'])
                head_entropies.append(head_pac['entropy'])
                head_concentrations.append(head_pac['concentration'])
        
        total_heads = len(head_diagnostics)
        
        # Aggregate metrics across all heads for this token
        crystallized_heads = phase_counts.get('CRYSTALLIZED', 0)
        ordered_heads = phase_counts.get('ORDERED', 0)
        chaotic_heads = phase_counts.get('CHAOTIC', 0)
        
        confident_head_ratio = (crystallized_heads + ordered_heads) / total_heads if total_heads > 0 else 0
        chaotic_head_ratio = chaotic_heads / total_heads if total_heads > 0 else 0
        
        # Layer-wise entropy gradient (do deeper layers have lower entropy?)
        layer_entropies = defaultdict(list)
        for hd in head_diagnostics:
            layer_entropies[hd['layer']].append(hd['entropy'])
        
        layer_mean_entropies = [np.mean(layer_entropies[l]) for l in sorted(layer_entropies.keys())]
        if len(layer_mean_entropies) >= 3:
            x = np.arange(len(layer_mean_entropies))
            attn_depth_slope = float(stats.linregress(x, layer_mean_entropies).slope)
        else:
            attn_depth_slope = 0.0
        
        td = {
            'step': step,
            'token_id': chosen_id,
            'token_text': tokenizer.decode([chosen_id]),
            'chosen_prob': chosen_prob,
            # Output-level
            'logit_entropy': logit_entropy,
            'logit_pac_ratio': logit_pac_ratio,
            'output_sec_phase': output_sec.name,
            # Attention-level aggregates
            'n_heads_total': total_heads,
            'confident_head_ratio': confident_head_ratio,
            'chaotic_head_ratio': chaotic_head_ratio,
            'phase_counts': dict(phase_counts),
            'attn_ratio_mean': float(np.mean(head_ratios)) if head_ratios else 0,
            'attn_ratio_median': float(np.median(head_ratios)) if head_ratios else 0,
            'attn_entropy_mean': float(np.mean(head_entropies)),
            'attn_concentration_mean': float(np.mean(head_concentrations)),
            'attn_depth_slope': attn_depth_slope,
        }
        
        token_data.append(td)
        
        # Autoregressive step
        chosen_tensor = torch.tensor([[chosen_id]], device=device)
        current_ids = torch.cat([current_ids, chosen_tensor], dim=1)
    
    generated_text = tokenizer.decode([td['token_id'] for td in token_data])
    
    # Sequence-level aggregates
    conf_ratios = [td['confident_head_ratio'] for td in token_data]
    chaos_ratios = [td['chaotic_head_ratio'] for td in token_data]
    attn_ratios = [td['attn_ratio_median'] for td in token_data if td['attn_ratio_median'] > 0]
    attn_ents = [td['attn_entropy_mean'] for td in token_data]
    depth_slopes = [td['attn_depth_slope'] for td in token_data]
    
    # Entropy slope of attention across generation (does attention sharpen?)
    if len(attn_ents) >= 3:
        x = np.arange(len(attn_ents))
        seq_attn_entropy_slope = float(stats.linregress(x, attn_ents).slope)
    else:
        seq_attn_entropy_slope = 0.0
    
    # Confident head ratio slope (does confidence grow?)
    if len(conf_ratios) >= 3:
        x = np.arange(len(conf_ratios))
        seq_conf_slope = float(stats.linregress(x, conf_ratios).slope)
    else:
        seq_conf_slope = 0.0
    
    # Correlation: attention confidence vs output confidence
    output_probs = [td['chosen_prob'] for td in token_data]
    if len(conf_ratios) >= 5:
        attn_output_corr, attn_output_p = stats.spearmanr(conf_ratios, output_probs)
    else:
        attn_output_corr, attn_output_p = 0.0, 1.0
    
    return {
        'prompt': prompt,
        'generated_text': generated_text,
        'n_tokens': len(token_data),
        'tokens': token_data,
        # Sequence-level
        'mean_confident_head_ratio': float(np.mean(conf_ratios)),
        'mean_chaotic_head_ratio': float(np.mean(chaos_ratios)),
        'mean_attn_ratio': float(np.mean(attn_ratios)) if attn_ratios else 0,
        'mean_attn_entropy': float(np.mean(attn_ents)),
        'mean_depth_slope': float(np.mean(depth_slopes)),
        'seq_attn_entropy_slope': seq_attn_entropy_slope,
        'seq_conf_slope': seq_conf_slope,
        'attn_output_corr': float(attn_output_corr) if not np.isnan(attn_output_corr) else 0,
        'attn_output_p': float(attn_output_p) if not np.isnan(attn_output_p) else 1,
    }


# ── Prompts ───────────────────────────────────────────────────────

FACTUAL = [
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
]

HALLUCINATION = [
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
]


def compare_groups(group_a, group_b, label_a="Factual", label_b="Hallucinated"):
    """Compare attention PAC metrics between groups."""
    metrics = [
        'mean_confident_head_ratio',
        'mean_chaotic_head_ratio',
        'mean_attn_ratio',
        'mean_attn_entropy',
        'mean_depth_slope',
        'seq_attn_entropy_slope',
        'seq_conf_slope',
        'attn_output_corr',
    ]
    
    comparison = {}
    significant = []
    
    print(f"\n{'='*70}")
    print(f"  ATTENTION PAC: {label_a} vs {label_b}")
    print(f"{'='*70}")
    
    for metric in metrics:
        vals_a = [r[metric] for r in group_a]
        vals_b = [r[metric] for r in group_b]
        
        mean_a, std_a = np.mean(vals_a), np.std(vals_a)
        mean_b, std_b = np.mean(vals_b), np.std(vals_b)
        
        try:
            _, p_mw = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')
        except ValueError:
            p_mw = 1.0
        
        marker = " ***" if p_mw < 0.01 else " **" if p_mw < 0.05 else " *" if p_mw < 0.1 else ""
        
        print(f"\n  {metric}:{marker}")
        print(f"    {label_a:15s}: {mean_a:8.4f} +/- {std_a:.4f}")
        print(f"    {label_b:15s}: {mean_b:8.4f} +/- {std_b:.4f}")
        print(f"    Mann-Whitney p: {p_mw:.6f}")
        
        if p_mw < 0.05:
            significant.append(metric)
        
        comparison[metric] = {
            f'{label_a}_mean': float(mean_a), f'{label_a}_std': float(std_a),
            f'{label_b}_mean': float(mean_b), f'{label_b}_std': float(std_b),
            'mann_whitney_p': float(p_mw),
        }
    
    # Token-level: does attention head confidence predict output SEC phase?
    phase_conf = defaultdict(list)
    for r in group_a + group_b:
        for td in r['tokens']:
            phase_conf[td['output_sec_phase']].append(td['confident_head_ratio'])
    
    print(f"\n  Attention confidence by output SEC phase (all tokens):")
    for phase in ['CRYSTALLIZED', 'ORDERED', 'TRANSITIONAL', 'CHAOTIC']:
        vals = phase_conf.get(phase, [])
        if vals:
            print(f"    {phase:15s}: n={len(vals):3d}  conf_heads={np.mean(vals)*100:5.1f}%")
    
    comparison['phase_attention_confidence'] = {
        phase: {'n': len(vals), 'mean': float(np.mean(vals))}
        for phase, vals in phase_conf.items() if vals
    }
    
    if significant:
        print(f"\n  SIGNIFICANT (p<0.05): {', '.join(significant)}")
    else:
        print(f"\n  No individually significant metrics at p<0.05")
    
    return comparison, significant


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("=" * 70)
    print("  EXP 07: Attention Pattern PAC — Where Collapse Happens")
    print("  Attention IS PAC collapse: potential → weighted sum → actualization")
    print("=" * 70)
    
    t0 = time.time()
    
    model_name = "pythia-160m"
    full_name = f"EleutherAI/{model_name}"
    
    print(f"\nLoading {full_name}...")
    tokenizer = AutoTokenizer.from_pretrained(full_name, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(full_name, cache_dir=CACHE_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded. {param_count:,} params on {device}")
    
    # Get model architecture info
    config = model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    print(f"Architecture: {n_layers} layers, {n_heads} heads/layer = {n_layers*n_heads} total heads")
    
    # ── Run inference with attention monitoring ──
    all_results = {}
    
    for group_name, prompts in [("factual", FACTUAL), ("hallucination", HALLUCINATION)]:
        print(f"\n{'='*60}")
        print(f"  GROUP: {group_name} ({len(prompts)} prompts, 20 tokens each)")
        print(f"{'='*60}")
        
        group_results = []
        for i, prompt in enumerate(prompts):
            result = run_attention_monitoring(model, tokenizer, device, prompt)
            group_results.append(result)
            
            prompt_short = prompt[:40] + "..." if len(prompt) > 40 else prompt
            print(f"  [{i+1:2d}/{len(prompts)}] conf_heads={result['mean_confident_head_ratio']*100:4.1f}%  "
                  f"attn_H={result['mean_attn_entropy']:.2f}  "
                  f"depth_slope={result['mean_depth_slope']:+.3f}  "
                  f"corr={result['attn_output_corr']:+.2f}  "
                  f"'{prompt_short}'")
        
        all_results[group_name] = group_results
    
    # ── Compare groups ──
    comparison, sig_metrics = compare_groups(
        all_results['factual'], all_results['hallucination']
    )
    
    # ── Layer-by-layer analysis ──
    print(f"\n{'='*60}")
    print(f"  LAYER-BY-LAYER: Mean attention entropy per layer")
    print(f"{'='*60}")
    
    for group_name in ['factual', 'hallucination']:
        layer_ents = defaultdict(list)
        for r in all_results[group_name]:
            for td in r['tokens']:
                for phase_key in td.get('phase_counts', {}).keys():
                    pass  # just iterating
                # We need per-layer data — extract from head_diagnostics
                # But we didn't store full per-head data to save memory.
                # Use the depth_slope as proxy.
        
        mean_slope = np.mean([r['mean_depth_slope'] for r in all_results[group_name]])
        print(f"  {group_name:15s}: mean depth slope = {mean_slope:+.4f} "
              f"({'deeper layers sharper' if mean_slope < 0 else 'deeper layers broader'})")
    
    # ── Save ──
    output = {
        'experiment': 'exp_07_attention_pac',
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
        'architecture': {'n_layers': n_layers, 'n_heads': n_heads, 'total_heads': n_layers * n_heads},
        'groups': {
            gname: {
                'n_prompts': len(results),
                'summary': {
                    'mean_confident_head_ratio': float(np.mean([r['mean_confident_head_ratio'] for r in results])),
                    'mean_chaotic_head_ratio': float(np.mean([r['mean_chaotic_head_ratio'] for r in results])),
                    'mean_attn_entropy': float(np.mean([r['mean_attn_entropy'] for r in results])),
                    'mean_depth_slope': float(np.mean([r['mean_depth_slope'] for r in results])),
                    'seq_attn_entropy_slope': float(np.mean([r['seq_attn_entropy_slope'] for r in results])),
                    'seq_conf_slope': float(np.mean([r['seq_conf_slope'] for r in results])),
                    'mean_attn_output_corr': float(np.mean([r['attn_output_corr'] for r in results])),
                },
                'prompts': [
                    {k: v for k, v in r.items() if k != 'tokens'}
                    for r in results
                ],
            }
            for gname, results in all_results.items()
        },
        'comparison': {
            'metrics': comparison,
            'significant_metrics': sig_metrics,
        },
        'elapsed_seconds': time.time() - t0,
    }
    
    results_dir = EXPERIMENT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"exp_07_attention_pac_{ts}.json"
    
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
