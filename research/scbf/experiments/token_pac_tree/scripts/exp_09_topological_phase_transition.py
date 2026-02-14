#!/usr/bin/env python3
"""
EXP 09: Topological Phase Transition — Where Chaos Becomes Order
=================================================================

exp_08 showed:
  - Xi works as a balance point only for Pythia-160m
  - But the H/F attention entropy RATIO ≈ Xi across all scales (mean 1.086)
  - confident_head_ratio threshold ≈ 0.80 is stable across architectures
  - log(d_model) best normalizes raw entropy

Core hypothesis from user insight: each architecture creates a logical
recursion with its own complexity. The SEC phase transition from chaos→order
happens at a point determined by the model's TOPOLOGY, not at a fixed
entropy value. Xi may be:
  (a) the ratio H_halluc / H_factual (scale-invariant)
  (b) the normalized entropy after accounting for topology
  (c) the per-layer entropy at the chaos→order flip point

This experiment:
  1. Bootstrap H/F ratio across all models to test Xi hypothesis
  2. Per-LAYER phase transition: at which layer does attention go from
     chaotic→ordered? Does this depth relate to architecture?
  3. Per-HEAD phase mapping: classify each head's SEC state across
     factual vs hallucination — find which heads flip
  4. Dynamic threshold: entropy / f(topology) → does Xi emerge?

Author: Dawn Field Institute
Date: 2026-02-14
"""

import sys, json, time, gc
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
from scipy import stats as sp

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR))
from core.collapse_metrics import classify_sec_phase, SECPhase

PHI = (1 + 5**0.5) / 2
XI = 1 + np.pi / 55
CACHE_DIR = str(Path(EXPERIMENT_DIR).parent / "huggingface_bifractal_validation" / "pythia_cache")

MODELS = [
    ("pythia-70m",  {'d': 512,  'layers': 6,  'heads': 8}),
    ("pythia-160m", {'d': 768,  'layers': 12, 'heads': 12}),
    ("pythia-410m", {'d': 1024, 'layers': 24, 'heads': 16}),
    ("pythia-1b",   {'d': 2048, 'layers': 16, 'heads': 8}),
]

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
    "The speed of light is",
    "Iron is a type of",
    "The human heart pumps",
    "Gravity pulls objects toward the",
    "Photosynthesis converts sunlight into",
    "DNA contains the genetic",
    "Electricity flows through wires",
    "The atmosphere is made of",
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
    "The mystical element Chronium allows users to",
    "The Zephyrian Council decided that all citizens must",
    "Professor Quibblesworth discovered that dreams are made of",
    "The planet Nexarion has seventeen moons that",
    "The secret society of time weavers believes that",
    "In the underwater kingdom of Bathysphere, all laws require",
    "The quantum philosopher Heidenberg proved that consciousness is",
    "According to the Book of Infinite Recursion, the universe began when",
]


def run_layerwise_attention(model, tokenizer, device, prompt, n_tokens=20):
    """
    Generate tokens capturing PER-LAYER, PER-HEAD attention diagnostics.
    Returns full layer×head entropy matrix for each token step.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    current_ids = input_ids
    token_data = []

    for step in range(n_tokens):
        with torch.no_grad():
            outputs = model(current_ids, output_attentions=True)
            logits = outputs.logits[0, -1, :]
            attentions = outputs.attentions

        probs = torch.softmax(logits, dim=-1)
        chosen_id = int(logits.argmax())

        n_layers = len(attentions)
        n_heads = attentions[0].shape[1]

        # Per-layer, per-head entropy matrix
        layer_head_entropy = np.zeros((n_layers, n_heads))
        layer_head_concentration = np.zeros((n_layers, n_heads))
        layer_head_phase = []

        for li, layer_attn in enumerate(attentions):
            head_phases = []
            for hi in range(n_heads):
                attn_vec = layer_attn[0, hi, -1, :].cpu().numpy()
                attn_pos = attn_vec[attn_vec > 1e-10]
                h = float(-np.sum(attn_pos * np.log(attn_pos))) if len(attn_pos) > 0 else 0.0
                layer_head_entropy[li, hi] = h
                layer_head_concentration[li, hi] = float(np.max(attn_vec))
                head_phases.append(classify_sec_phase(h).name)
            layer_head_phase.append(head_phases)

        # Per-layer aggregates
        layer_mean_entropy = layer_head_entropy.mean(axis=1)  # [n_layers]
        layer_confident_ratio = np.zeros(n_layers)
        for li in range(n_layers):
            confident = sum(1 for p in layer_head_phase[li] if p in ('CRYSTALLIZED', 'ORDERED'))
            layer_confident_ratio[li] = confident / n_heads

        # Find the chaos→order transition layer
        # Where does confident_ratio first exceed 0.5? (majority of heads ordered)
        transition_layer = None
        for li in range(n_layers):
            if layer_confident_ratio[li] >= 0.5:
                transition_layer = li
                break

        td = {
            'step': step,
            'token_id': chosen_id,
            'layer_mean_entropy': layer_mean_entropy.tolist(),
            'layer_confident_ratio': layer_confident_ratio.tolist(),
            'layer_head_entropy': layer_head_entropy.tolist(),
            'layer_head_phase': layer_head_phase,
            'transition_layer': transition_layer,
        }
        token_data.append(td)

        chosen_tensor = torch.tensor([[chosen_id]], device=device)
        current_ids = torch.cat([current_ids, chosen_tensor], dim=1)

    # Sequence-level: average layer profile
    n_layers = len(token_data[0]['layer_mean_entropy'])
    avg_layer_entropy = np.mean([td['layer_mean_entropy'] for td in token_data], axis=0)
    avg_layer_confident = np.mean([td['layer_confident_ratio'] for td in token_data], axis=0)

    # Mean transition layer
    transitions = [td['transition_layer'] for td in token_data if td['transition_layer'] is not None]
    mean_transition = float(np.mean(transitions)) if transitions else None

    # Overall mean attention entropy
    all_ents = [np.mean(td['layer_mean_entropy']) for td in token_data]
    mean_attn_entropy = float(np.mean(all_ents))

    # Overall confident head ratio
    all_conf = [np.mean(td['layer_confident_ratio']) for td in token_data]
    mean_confident = float(np.mean(all_conf))

    # Per-head phase stability: for each (layer, head), what fraction of tokens
    # is it in crystallized/ordered phase?
    n_heads_per = len(token_data[0]['layer_head_phase'][0])
    head_confident_rate = np.zeros((n_layers, n_heads_per))
    for td in token_data:
        for li in range(n_layers):
            for hi in range(n_heads_per):
                if td['layer_head_phase'][li][hi] in ('CRYSTALLIZED', 'ORDERED'):
                    head_confident_rate[li, hi] += 1
    head_confident_rate /= len(token_data)

    return {
        'prompt': prompt,
        'mean_attn_entropy': mean_attn_entropy,
        'mean_confident_ratio': mean_confident,
        'avg_layer_entropy': avg_layer_entropy.tolist(),
        'avg_layer_confident': avg_layer_confident.tolist(),
        'mean_transition_layer': mean_transition,
        'head_confident_rate': head_confident_rate.tolist(),
        'n_layers': n_layers,
        'n_heads': n_heads_per,
    }


def bootstrap_ratio(factual_vals, halluc_vals, n_bootstrap=10000):
    """Bootstrap the H/F ratio to get confidence interval."""
    ratios = []
    for _ in range(n_bootstrap):
        f_sample = np.random.choice(factual_vals, size=len(factual_vals), replace=True)
        h_sample = np.random.choice(halluc_vals, size=len(halluc_vals), replace=True)
        ratios.append(np.mean(h_sample) / np.mean(f_sample))
    ratios = np.array(ratios)
    return {
        'mean': float(np.mean(ratios)),
        'median': float(np.median(ratios)),
        'std': float(np.std(ratios)),
        'ci_2_5': float(np.percentile(ratios, 2.5)),
        'ci_97_5': float(np.percentile(ratios, 97.5)),
        'xi_in_ci': bool(XI >= np.percentile(ratios, 2.5) and XI <= np.percentile(ratios, 97.5)),
        'p_above_xi': float(np.mean(ratios > XI)),
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("  EXP 09: Topological Phase Transition")
    print("  Where does chaos→order happen inside the model?")
    print("=" * 70)

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_model_results = {}

    for model_name, arch_info in MODELS:
        full_name = f"EleutherAI/{model_name}"
        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name}")
        print(f"  layers={arch_info['layers']}  heads={arch_info['heads']}  "
              f"d_model={arch_info['d']}  d_head={arch_info['d']//arch_info['heads']}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(full_name, cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(full_name, cache_dir=CACHE_DIR)
        model = model.to(device).eval()

        # Run all prompts with layerwise analysis
        factual_results = []
        halluc_results = []

        for i, prompt in enumerate(FACTUAL):
            r = run_layerwise_attention(model, tokenizer, device, prompt)
            r['group'] = 'factual'
            factual_results.append(r)
            t_layer = r['mean_transition_layer']
            t_str = f"L{t_layer:.1f}" if t_layer is not None else "none"
            print(f"  F[{i+1:2d}] H={r['mean_attn_entropy']:.4f}  "
                  f"conf={r['mean_confident_ratio']*100:.0f}%  "
                  f"trans={t_str:>5s}  '{prompt[:40]}'")

        for i, prompt in enumerate(HALLUCINATION):
            r = run_layerwise_attention(model, tokenizer, device, prompt)
            r['group'] = 'hallucination'
            halluc_results.append(r)
            t_layer = r['mean_transition_layer']
            t_str = f"L{t_layer:.1f}" if t_layer is not None else "none"
            print(f"  H[{i+1:2d}] H={r['mean_attn_entropy']:.4f}  "
                  f"conf={r['mean_confident_ratio']*100:.0f}%  "
                  f"trans={t_str:>5s}  '{prompt[:40]}'")

        # ── 1. Bootstrap H/F ratio ──
        fact_ents = [r['mean_attn_entropy'] for r in factual_results]
        hall_ents = [r['mean_attn_entropy'] for r in halluc_results]
        bootstrap = bootstrap_ratio(fact_ents, hall_ents)

        print(f"\n  BOOTSTRAP H/F RATIO (n=10000):")
        print(f"    Mean ratio:  {bootstrap['mean']:.4f}")
        print(f"    95% CI:      [{bootstrap['ci_2_5']:.4f}, {bootstrap['ci_97_5']:.4f}]")
        print(f"    Xi = {XI:.4f}  {'IN CI' if bootstrap['xi_in_ci'] else 'OUTSIDE CI'}")
        print(f"    P(ratio > Xi): {bootstrap['p_above_xi']:.4f}")

        # ── 2. Per-layer phase profile ──
        n_layers = arch_info['layers']
        fact_layer_entropy = np.mean([r['avg_layer_entropy'] for r in factual_results], axis=0)
        hall_layer_entropy = np.mean([r['avg_layer_entropy'] for r in halluc_results], axis=0)
        fact_layer_conf = np.mean([r['avg_layer_confident'] for r in factual_results], axis=0)
        hall_layer_conf = np.mean([r['avg_layer_confident'] for r in halluc_results], axis=0)

        print(f"\n  PER-LAYER ENTROPY PROFILE:")
        print(f"    {'Layer':>5s} {'F_entropy':>10s} {'H_entropy':>10s} {'H/F':>8s} "
              f"{'F_conf%':>8s} {'H_conf%':>8s} {'Δconf':>8s}")
        
        layer_ratios = []
        for li in range(n_layers):
            ratio = hall_layer_entropy[li] / fact_layer_entropy[li] if fact_layer_entropy[li] > 0 else 0
            layer_ratios.append(ratio)
            delta_conf = (fact_layer_conf[li] - hall_layer_conf[li]) * 100
            print(f"    L{li:3d}  {fact_layer_entropy[li]:10.4f} {hall_layer_entropy[li]:10.4f} "
                  f"{ratio:8.4f} {fact_layer_conf[li]*100:7.1f}% {hall_layer_conf[li]*100:7.1f}% "
                  f"{delta_conf:+7.1f}%")

        # Where is the H/F ratio closest to Xi?
        layer_xi_dist = [abs(r - XI) for r in layer_ratios]
        closest_layer = int(np.argmin(layer_xi_dist))
        print(f"\n    Layer closest to Xi ratio: L{closest_layer} "
              f"(H/F={layer_ratios[closest_layer]:.4f}, dist={layer_xi_dist[closest_layer]:.4f})")

        # Per-layer entropy ratio mean
        mean_layer_ratio = np.mean(layer_ratios)
        print(f"    Mean layer H/F ratio: {mean_layer_ratio:.4f} (Xi={XI:.4f}, dist={abs(mean_layer_ratio-XI):.4f})")

        # ── 3. Transition layer analysis ──
        fact_transitions = [r['mean_transition_layer'] for r in factual_results if r['mean_transition_layer'] is not None]
        hall_transitions = [r['mean_transition_layer'] for r in halluc_results if r['mean_transition_layer'] is not None]

        print(f"\n  CHAOS→ORDER TRANSITION LAYER (>50% heads confident):")
        if fact_transitions:
            print(f"    Factual:  mean=L{np.mean(fact_transitions):.1f}  "
                  f"std={np.std(fact_transitions):.1f}  "
                  f"range=[L{min(fact_transitions):.0f}, L{max(fact_transitions):.0f}]  "
                  f"n={len(fact_transitions)}/{len(factual_results)}")
        else:
            print(f"    Factual:  no transitions found (always >50% or always <50%)")
        
        if hall_transitions:
            print(f"    Halluc:   mean=L{np.mean(hall_transitions):.1f}  "
                  f"std={np.std(hall_transitions):.1f}  "
                  f"range=[L{min(hall_transitions):.0f}, L{max(hall_transitions):.0f}]  "
                  f"n={len(hall_transitions)}/{len(halluc_results)}")
        else:
            print(f"    Halluc:   no transitions found")

        if fact_transitions and hall_transitions:
            try:
                _, t_p = sp.mannwhitneyu(fact_transitions, hall_transitions, alternative='two-sided')
            except ValueError:
                t_p = 1.0
            print(f"    Mann-Whitney p: {t_p:.4f}")

        # Normalize transition layer by total depth
        if fact_transitions:
            fact_norm_trans = [t / n_layers for t in fact_transitions]
            print(f"    Factual normalized (layer/depth): {np.mean(fact_norm_trans):.3f}")
        if hall_transitions:
            hall_norm_trans = [t / n_layers for t in hall_transitions]
            print(f"    Halluc normalized (layer/depth):  {np.mean(hall_norm_trans):.3f}")

        # ── 4. Head taxonomy ──
        # Which heads flip between factual and hallucination?
        n_heads_per = arch_info['heads']
        fact_head_conf = np.mean([r['head_confident_rate'] for r in factual_results], axis=0)
        hall_head_conf = np.mean([r['head_confident_rate'] for r in halluc_results], axis=0)
        head_flip = fact_head_conf - hall_head_conf  # positive = more confident during factual

        # Classify heads
        stable_confident = np.sum((fact_head_conf > 0.8) & (hall_head_conf > 0.8))
        stable_chaotic = np.sum((fact_head_conf < 0.2) & (hall_head_conf < 0.2))
        discriminative = np.sum(np.abs(head_flip) > 0.15)
        total_heads = n_layers * n_heads_per

        print(f"\n  HEAD TAXONOMY ({total_heads} total heads):")
        print(f"    Stable confident (>80% both): {stable_confident} ({stable_confident/total_heads*100:.0f}%)")
        print(f"    Stable chaotic (<20% both):   {stable_chaotic} ({stable_chaotic/total_heads*100:.0f}%)")
        print(f"    Discriminative (|flip|>15%):   {discriminative} ({discriminative/total_heads*100:.0f}%)")
        print(f"    Other:                         {total_heads - stable_confident - stable_chaotic - discriminative}")

        # Top discriminative heads
        flat_flip = head_flip.flatten()
        flat_indices = np.argsort(-np.abs(flat_flip))[:5]
        print(f"\n    Top 5 discriminative heads:")
        for idx in flat_indices:
            li = idx // n_heads_per
            hi = idx % n_heads_per
            print(f"      L{li}H{hi}: factual_conf={fact_head_conf[li,hi]*100:.0f}%  "
                  f"halluc_conf={hall_head_conf[li,hi]*100:.0f}%  "
                  f"flip={head_flip[li,hi]*100:+.0f}%")

        # ── 5. Dynamic normalized threshold ──
        # Normalize entropy by log(d_model) and test Xi
        log_d = np.log(arch_info['d'])
        norm_fact = np.mean(fact_ents) / log_d
        norm_hall = np.mean(hall_ents) / log_d
        norm_mid = (norm_fact + norm_hall) / 2

        # Also try: entropy / log(total_heads)
        log_total = np.log(total_heads)
        norm2_fact = np.mean(fact_ents) / log_total
        norm2_hall = np.mean(hall_ents) / log_total
        norm2_mid = (norm2_fact + norm2_hall) / 2

        print(f"\n  NORMALIZED ENTROPY:")
        print(f"    By log(d_model={arch_info['d']})={log_d:.3f}:")
        print(f"      F={norm_fact:.4f}  H={norm_hall:.4f}  mid={norm_mid:.4f}  "
              f"dist from Xi/log(d)={abs(norm_mid - XI/log_d):.4f}")
        print(f"    By log(total_heads={total_heads})={log_total:.3f}:")
        print(f"      F={norm2_fact:.4f}  H={norm2_hall:.4f}  mid={norm2_mid:.4f}")

        # Store
        all_model_results[model_name] = {
            'architecture': arch_info,
            'd_head': arch_info['d'] // arch_info['heads'],
            'total_heads': total_heads,
            'bootstrap_ratio': bootstrap,
            'layer_profile': {
                'factual_entropy': fact_layer_entropy.tolist(),
                'halluc_entropy': hall_layer_entropy.tolist(),
                'factual_confident': fact_layer_conf.tolist(),
                'halluc_confident': hall_layer_conf.tolist(),
                'layer_hf_ratios': layer_ratios,
                'closest_xi_layer': closest_layer,
                'mean_layer_ratio': float(mean_layer_ratio),
            },
            'transition': {
                'factual_mean': float(np.mean(fact_transitions)) if fact_transitions else None,
                'halluc_mean': float(np.mean(hall_transitions)) if hall_transitions else None,
                'factual_n': len(fact_transitions),
                'halluc_n': len(hall_transitions),
                'factual_normalized': float(np.mean(fact_norm_trans)) if fact_transitions else None,
                'halluc_normalized': float(np.mean(hall_norm_trans)) if hall_transitions else None,
            },
            'head_taxonomy': {
                'stable_confident': int(stable_confident),
                'stable_chaotic': int(stable_chaotic),
                'discriminative': int(discriminative),
                'total': total_heads,
            },
            'normalized': {
                'log_d_midpoint': float(norm_mid),
                'log_total_midpoint': float(norm2_mid),
            },
            'raw': {
                'factual_mean': float(np.mean(fact_ents)),
                'halluc_mean': float(np.mean(hall_ents)),
                'midpoint': float((np.mean(fact_ents) + np.mean(hall_ents)) / 2),
            },
        }

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # ── Cross-model synthesis ──
    print(f"\n{'='*70}")
    print(f"  CROSS-MODEL SYNTHESIS")
    print(f"{'='*70}")

    # 1. H/F ratio vs Xi
    print(f"\n  1. H/F ENTROPY RATIO vs Xi ({XI:.4f}):")
    all_ratios = []
    for mn, _ in MODELS:
        br = all_model_results[mn]['bootstrap_ratio']
        xi_status = "Xi IN CI" if br['xi_in_ci'] else "Xi OUTSIDE"
        all_ratios.append(br['mean'])
        print(f"    {mn:15s}: {br['mean']:.4f}  95%CI=[{br['ci_2_5']:.4f}, {br['ci_97_5']:.4f}]  {xi_status}")
    
    pooled_mean = np.mean(all_ratios)
    pooled_std = np.std(all_ratios)
    print(f"    {'POOLED':15s}: {pooled_mean:.4f} ± {pooled_std:.4f}  "
          f"dist from Xi = {abs(pooled_mean - XI):.4f}")

    # 2. Transition layer (normalized by depth)
    print(f"\n  2. NORMALIZED TRANSITION LAYER (chaos→order at >50% confident):")
    for mn, arch in MODELS:
        t = all_model_results[mn]['transition']
        fn = f"{t['factual_normalized']:.3f}" if t['factual_normalized'] is not None else "N/A"
        hn = f"{t['halluc_normalized']:.3f}" if t['halluc_normalized'] is not None else "N/A"
        print(f"    {mn:15s}: factual={fn}  halluc={hn}  "
              f"(n_F={t['factual_n']}, n_H={t['halluc_n']})")

    # 3. Head taxonomy
    print(f"\n  3. HEAD TAXONOMY ACROSS SCALES:")
    print(f"    {'Model':15s} {'Total':>6s} {'Stable':>7s} {'Chaotic':>8s} {'Discrim':>8s} {'%Disc':>7s}")
    for mn, _ in MODELS:
        ht = all_model_results[mn]['head_taxonomy']
        print(f"    {mn:15s} {ht['total']:6d} {ht['stable_confident']:7d} "
              f"{ht['stable_chaotic']:8d} {ht['discriminative']:8d} "
              f"{ht['discriminative']/ht['total']*100:6.1f}%")

    # 4. Layer-level H/F ratio — is Xi in there per-layer?
    print(f"\n  4. PER-LAYER H/F RATIO — LOOKING FOR Xi:")
    for mn, arch in MODELS:
        lp = all_model_results[mn]['layer_profile']
        ratios = lp['layer_hf_ratios']
        near_xi = sum(1 for r in ratios if abs(r - XI) < 0.02)
        print(f"    {mn:15s}: mean={lp['mean_layer_ratio']:.4f}  "
              f"closest_to_Xi=L{lp['closest_xi_layer']}({ratios[lp['closest_xi_layer']]:.4f})  "
              f"layers_near_Xi(<0.02)={near_xi}/{len(ratios)}")

    # 5. Normalized entropy midpoints
    print(f"\n  5. NORMALIZED ENTROPY MIDPOINTS:")
    log_d_mids = [all_model_results[mn]['normalized']['log_d_midpoint'] for mn, _ in MODELS]
    log_t_mids = [all_model_results[mn]['normalized']['log_total_midpoint'] for mn, _ in MODELS]
    
    print(f"    By log(d_model): {[f'{m:.4f}' for m in log_d_mids]}  std={np.std(log_d_mids):.4f}")
    print(f"    By log(total_h): {[f'{m:.4f}' for m in log_t_mids]}  std={np.std(log_t_mids):.4f}")

    # 6. The key question: does any normalization make Xi universal?
    print(f"\n  6. DOES Xi EMERGE FROM NORMALIZATION?")
    raw_mids = [all_model_results[mn]['raw']['midpoint'] for mn, _ in MODELS]
    
    # Try: midpoint / Xi → should this equal some function of architecture?
    xi_normalized = [m / XI for m in raw_mids]
    print(f"    midpoint/Xi = {[f'{v:.4f}' for v in xi_normalized]}")
    
    # What architectural feature correlates best with midpoint/Xi?
    layers = [arch['layers'] for _, arch in MODELS]
    heads = [arch['heads'] for _, arch in MODELS]
    d_models = [arch['d'] for _, arch in MODELS]
    d_heads = [arch['d'] // arch['heads'] for _, arch in MODELS]
    
    features = {
        'log(d_model)': [np.log(d) for d in d_models],
        'log(d_head)': [np.log(dh) for dh in d_heads],
        'layers': layers,
        'sqrt(layers)': [np.sqrt(l) for l in layers],
        'log(layers*heads)': [np.log(l*h) for l, h in zip(layers, heads)],
    }
    
    print(f"\n    Correlation of midpoint with architectural features:")
    for fname, fvals in features.items():
        r, p = sp.pearsonr(fvals, raw_mids)
        print(f"      {fname:20s}: r={r:+.3f}  p={p:.4f}")

    # ── Save ──
    output = {
        'experiment': 'exp_09_topological_phase_transition',
        'timestamp': datetime.now().isoformat(),
        'xi': float(XI),
        'n_factual': len(FACTUAL),
        'n_hallucination': len(HALLUCINATION),
        'n_tokens_per_prompt': 20,
        'models': {mn: all_model_results[mn] for mn, _ in MODELS},
        'cross_model': {
            'hf_ratios': all_ratios,
            'pooled_ratio': float(pooled_mean),
            'pooled_std': float(pooled_std),
            'log_d_midpoints': log_d_mids,
            'log_total_midpoints': log_t_mids,
        },
        'elapsed_seconds': time.time() - t0,
    }

    results_dir = EXPERIMENT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"exp_09_topology_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
