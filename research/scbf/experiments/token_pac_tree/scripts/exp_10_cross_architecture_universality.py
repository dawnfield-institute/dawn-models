#!/usr/bin/env python3
"""
EXP 10: Cross-Architecture Universality Test
=============================================

exp_09 found the delayed phase transition in Pythia models:
  - Hallucination prompts transition from chaos→order LATER in the network
  - 3/4 Pythia models significant (p < 0.05)
  - ~20% of heads are discriminative (flip between factual/halluc)
  - Discriminative heads cluster in early layers

Critical question: IS THIS UNIVERSAL OR A PYTHIA ARTIFACT?

GPT-2 family is ideal for falsification:
  - Different training data (WebText vs The Pile)
  - Different tokenizer (BPE with different vocab)
  - Different architecture family (GPT-2 vs GPT-NeoX)
  - Same d_head=64 as Pythia 70m/160m/410m (controls for d_head)

GPT-2 architectures:
  gpt2       (124M): 12L, 12H, d=768,  d_head=64
  gpt2-medium (355M): 24L, 16H, d=1024, d_head=64
  gpt2-large  (774M): 36L, 20H, d=1280, d_head=64

Then: load exp_09 Pythia results and overlay ALL 7 MODELS on normalized
depth to find universal phase transition signatures.

New analyses:
  1. Cross-architecture delayed transition test
  2. 7-model normalized depth overlay (H/F ratio vs relative depth)
  3. Discriminative head depth distribution (always early?)
  4. Phase transition gradient (how sharply does chaos→order happen?)
  5. Universal confident_head_ratio threshold test

Author: Dawn Field Institute
Date: 2026-02-14
"""

import sys, json, time, gc, glob
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

# GPT-2 family — different training, tokenizer, architecture
GPT2_MODELS = [
    ("gpt2",        {'d': 768,  'layers': 12, 'heads': 12, 'family': 'gpt2'}),
    ("gpt2-medium", {'d': 1024, 'layers': 24, 'heads': 16, 'family': 'gpt2'}),
    ("gpt2-large",  {'d': 1280, 'layers': 36, 'heads': 20, 'family': 'gpt2'}),
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
        layer_mean_entropy = layer_head_entropy.mean(axis=1)
        layer_confident_ratio = np.zeros(n_layers)
        for li in range(n_layers):
            confident = sum(1 for p in layer_head_phase[li] if p in ('CRYSTALLIZED', 'ORDERED'))
            layer_confident_ratio[li] = confident / n_heads

        # Find chaos→order transition layer
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

    # Sequence-level aggregates
    n_layers = len(token_data[0]['layer_mean_entropy'])
    avg_layer_entropy = np.mean([td['layer_mean_entropy'] for td in token_data], axis=0)
    avg_layer_confident = np.mean([td['layer_confident_ratio'] for td in token_data], axis=0)

    transitions = [td['transition_layer'] for td in token_data if td['transition_layer'] is not None]
    mean_transition = float(np.mean(transitions)) if transitions else None

    all_ents = [np.mean(td['layer_mean_entropy']) for td in token_data]
    mean_attn_entropy = float(np.mean(all_ents))

    all_conf = [np.mean(td['layer_confident_ratio']) for td in token_data]
    mean_confident = float(np.mean(all_conf))

    # Per-head phase stability
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
        'ci_2_5': float(np.percentile(ratios, 2.5)),
        'ci_97_5': float(np.percentile(ratios, 97.5)),
        'xi_in_ci': bool(XI >= np.percentile(ratios, 2.5) and XI <= np.percentile(ratios, 97.5)),
        'p_above_xi': float(np.mean(ratios > XI)),
    }


def analyze_model(model_name, arch_info, device, is_pythia=False):
    """Run full analysis pipeline for a single model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if is_pythia:
        full_name = f"EleutherAI/{model_name}"
    else:
        full_name = model_name

    print(f"\n{'='*60}")
    print(f"  MODEL: {model_name}  [{'Pythia' if is_pythia else 'GPT-2'}]")
    print(f"  layers={arch_info['layers']}  heads={arch_info['heads']}  "
          f"d_model={arch_info['d']}  d_head={arch_info['d']//arch_info['heads']}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(full_name, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(full_name, cache_dir=CACHE_DIR)
    model = model.to(device).eval()

    # GPT-2 tokenizer needs padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    # ── Analysis ──
    n_layers = arch_info['layers']
    n_heads_per = arch_info['heads']
    total_heads = n_layers * n_heads_per
    d_head = arch_info['d'] // arch_info['heads']

    fact_ents = [r['mean_attn_entropy'] for r in factual_results]
    hall_ents = [r['mean_attn_entropy'] for r in halluc_results]

    # Bootstrap H/F ratio
    bootstrap = bootstrap_ratio(fact_ents, hall_ents)
    print(f"\n  BOOTSTRAP H/F RATIO (n=10000):")
    print(f"    Mean ratio:  {bootstrap['mean']:.4f}")
    print(f"    95% CI:      [{bootstrap['ci_2_5']:.4f}, {bootstrap['ci_97_5']:.4f}]")
    print(f"    Xi = {XI:.4f}  {'IN CI' if bootstrap['xi_in_ci'] else 'OUTSIDE CI'}")

    # Per-layer profiles
    fact_layer_entropy = np.mean([r['avg_layer_entropy'] for r in factual_results], axis=0)
    hall_layer_entropy = np.mean([r['avg_layer_entropy'] for r in halluc_results], axis=0)
    fact_layer_conf = np.mean([r['avg_layer_confident'] for r in factual_results], axis=0)
    hall_layer_conf = np.mean([r['avg_layer_confident'] for r in halluc_results], axis=0)

    layer_ratios = []
    print(f"\n  PER-LAYER ENTROPY PROFILE:")
    print(f"    {'Layer':>5s} {'F_ent':>8s} {'H_ent':>8s} {'H/F':>8s} "
          f"{'F_conf%':>8s} {'H_conf%':>8s} {'Δconf':>8s}")
    for li in range(n_layers):
        ratio = hall_layer_entropy[li] / fact_layer_entropy[li] if fact_layer_entropy[li] > 0 else 0
        layer_ratios.append(ratio)
        delta_conf = (fact_layer_conf[li] - hall_layer_conf[li]) * 100
        print(f"    L{li:3d}  {fact_layer_entropy[li]:8.4f} {hall_layer_entropy[li]:8.4f} "
              f"{ratio:8.4f} {fact_layer_conf[li]*100:7.1f}% {hall_layer_conf[li]*100:7.1f}% "
              f"{delta_conf:+7.1f}%")

    layer_xi_dist = [abs(r - XI) for r in layer_ratios]
    closest_layer = int(np.argmin(layer_xi_dist))
    mean_layer_ratio = float(np.mean(layer_ratios))
    print(f"\n    Layer closest to Xi: L{closest_layer} "
          f"(H/F={layer_ratios[closest_layer]:.4f}, dist={layer_xi_dist[closest_layer]:.4f})")
    print(f"    Mean layer H/F: {mean_layer_ratio:.4f} (Xi={XI:.4f}, dist={abs(mean_layer_ratio-XI):.4f})")

    # Transition layer
    fact_transitions = [r['mean_transition_layer'] for r in factual_results
                        if r['mean_transition_layer'] is not None]
    hall_transitions = [r['mean_transition_layer'] for r in halluc_results
                        if r['mean_transition_layer'] is not None]

    print(f"\n  CHAOS→ORDER TRANSITION:")
    t_p = None
    fact_norm_trans = None
    hall_norm_trans = None

    if fact_transitions:
        fact_norm_trans = [t / n_layers for t in fact_transitions]
        print(f"    Factual:  mean=L{np.mean(fact_transitions):.1f}  "
              f"std={np.std(fact_transitions):.1f}  norm={np.mean(fact_norm_trans):.3f}  "
              f"n={len(fact_transitions)}/{len(factual_results)}")
    if hall_transitions:
        hall_norm_trans = [t / n_layers for t in hall_transitions]
        print(f"    Halluc:   mean=L{np.mean(hall_transitions):.1f}  "
              f"std={np.std(hall_transitions):.1f}  norm={np.mean(hall_norm_trans):.3f}  "
              f"n={len(hall_transitions)}/{len(halluc_results)}")
    if fact_transitions and hall_transitions:
        try:
            _, t_p = sp.mannwhitneyu(fact_transitions, hall_transitions, alternative='two-sided')
        except ValueError:
            t_p = 1.0
        sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "n.s."
        print(f"    Mann-Whitney p: {t_p:.6f}  {sig}")

    # Head taxonomy
    fact_head_conf = np.mean([r['head_confident_rate'] for r in factual_results], axis=0)
    hall_head_conf = np.mean([r['head_confident_rate'] for r in halluc_results], axis=0)
    head_flip = fact_head_conf - hall_head_conf

    stable_confident = int(np.sum((fact_head_conf > 0.8) & (hall_head_conf > 0.8)))
    stable_chaotic = int(np.sum((fact_head_conf < 0.2) & (hall_head_conf < 0.2)))
    discriminative = int(np.sum(np.abs(head_flip) > 0.15))

    print(f"\n  HEAD TAXONOMY ({total_heads} total):")
    print(f"    Stable confident: {stable_confident} ({stable_confident/total_heads*100:.0f}%)")
    print(f"    Discriminative:   {discriminative} ({discriminative/total_heads*100:.0f}%)")

    # Where are discriminative heads? Compute mean normalized depth
    disc_depths = []
    for li in range(n_layers):
        for hi in range(n_heads_per):
            if abs(head_flip[li, hi]) > 0.15:
                disc_depths.append(li / n_layers)  # normalized depth

    if disc_depths:
        disc_depth_mean = float(np.mean(disc_depths))
        disc_depth_std = float(np.std(disc_depths))
        # How many are in first quarter vs rest?
        first_quarter = sum(1 for d in disc_depths if d < 0.25)
        print(f"    Disc. head mean depth: {disc_depth_mean:.3f} ± {disc_depth_std:.3f}")
        print(f"    Disc. in first 25%:    {first_quarter}/{len(disc_depths)} "
              f"({first_quarter/len(disc_depths)*100:.0f}%)")
    else:
        disc_depth_mean = None
        disc_depth_std = None
        first_quarter = 0

    # Phase transition gradient — how sharply does confident ratio rise?
    # Use the max Δ(confident_ratio) between consecutive layers
    fact_gradient = np.diff(fact_layer_conf)
    hall_gradient = np.diff(hall_layer_conf)
    fact_max_grad = float(np.max(fact_gradient)) if len(fact_gradient) > 0 else 0
    hall_max_grad = float(np.max(hall_gradient)) if len(hall_gradient) > 0 else 0
    fact_grad_layer = int(np.argmax(fact_gradient)) if len(fact_gradient) > 0 else -1
    hall_grad_layer = int(np.argmax(hall_gradient)) if len(hall_gradient) > 0 else -1

    print(f"\n  PHASE TRANSITION GRADIENT:")
    print(f"    Factual:  max Δconf = {fact_max_grad:.3f} at L{fact_grad_layer}→L{fact_grad_layer+1}  "
          f"(norm depth {fact_grad_layer / n_layers:.3f})")
    print(f"    Halluc:   max Δconf = {hall_max_grad:.3f} at L{hall_grad_layer}→L{hall_grad_layer+1}  "
          f"(norm depth {hall_grad_layer / n_layers:.3f})")

    result = {
        'model_name': model_name,
        'family': arch_info.get('family', 'pythia'),
        'architecture': arch_info,
        'd_head': d_head,
        'total_heads': total_heads,
        'bootstrap_ratio': bootstrap,
        'layer_profile': {
            'factual_entropy': fact_layer_entropy.tolist(),
            'halluc_entropy': hall_layer_entropy.tolist(),
            'factual_confident': fact_layer_conf.tolist(),
            'halluc_confident': hall_layer_conf.tolist(),
            'layer_hf_ratios': layer_ratios,
            'closest_xi_layer': closest_layer,
            'mean_layer_ratio': mean_layer_ratio,
        },
        'transition': {
            'factual_transitions': fact_transitions,
            'halluc_transitions': hall_transitions,
            'factual_mean': float(np.mean(fact_transitions)) if fact_transitions else None,
            'halluc_mean': float(np.mean(hall_transitions)) if hall_transitions else None,
            'factual_normalized': float(np.mean(fact_norm_trans)) if fact_norm_trans else None,
            'halluc_normalized': float(np.mean(hall_norm_trans)) if hall_norm_trans else None,
            'mann_whitney_p': t_p,
            'delay_ratio': (float(np.mean(hall_norm_trans)) / float(np.mean(fact_norm_trans))
                           if fact_norm_trans and hall_norm_trans and np.mean(fact_norm_trans) > 0
                           else None),
        },
        'head_taxonomy': {
            'stable_confident': stable_confident,
            'stable_chaotic': stable_chaotic,
            'discriminative': discriminative,
            'total': total_heads,
            'disc_pct': float(discriminative / total_heads * 100),
            'disc_depth_mean': disc_depth_mean,
            'disc_depth_std': disc_depth_std,
            'disc_first_quarter_pct': float(first_quarter / len(disc_depths) * 100) if disc_depths else None,
        },
        'gradient': {
            'factual_max': fact_max_grad,
            'factual_layer_norm': float(fact_grad_layer / n_layers),
            'halluc_max': hall_max_grad,
            'halluc_layer_norm': float(hall_grad_layer / n_layers),
        },
        'raw': {
            'factual_mean': float(np.mean(fact_ents)),
            'halluc_mean': float(np.mean(hall_ents)),
        },
    }

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return result


def load_pythia_results():
    """Load exp_09 Pythia results for combined analysis."""
    results_dir = EXPERIMENT_DIR / "results"
    pattern = str(results_dir / "exp_09_topology_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print("  WARNING: No exp_09 results found. Running without Pythia data.")
        return None
    latest = files[-1]
    print(f"\n  Loading Pythia results from: {Path(latest).name}")
    with open(latest) as f:
        return json.load(f)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("  EXP 10: Cross-Architecture Universality Test")
    print("  Does the delayed phase transition hold beyond Pythia?")
    print("=" * 70)

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Phase 1: Run GPT-2 models ──
    print(f"\n{'─'*70}")
    print(f"  PHASE 1: GPT-2 Family")
    print(f"{'─'*70}")

    gpt2_results = {}
    for model_name, arch_info in GPT2_MODELS:
        result = analyze_model(model_name, arch_info, device, is_pythia=False)
        gpt2_results[model_name] = result

    # ── Phase 2: Load Pythia results from exp_09 ──
    print(f"\n{'─'*70}")
    print(f"  PHASE 2: Loading Pythia Results from exp_09")
    print(f"{'─'*70}")

    pythia_data = load_pythia_results()

    # If no exp_09 results, re-run Pythia
    pythia_results = {}
    if pythia_data and 'models' in pythia_data:
        for mn, mdata in pythia_data['models'].items():
            # Reconstruct exp_09 format into exp_10 format
            pythia_results[mn] = {
                'model_name': mn,
                'family': 'pythia',
                'architecture': mdata['architecture'],
                'd_head': mdata['d_head'],
                'total_heads': mdata['total_heads'],
                'bootstrap_ratio': mdata['bootstrap_ratio'],
                'layer_profile': mdata['layer_profile'],
                'transition': mdata['transition'],
                'head_taxonomy': mdata['head_taxonomy'],
                'raw': mdata['raw'],
                # These weren't in exp_09, compute from data
                'gradient': {
                    'factual_max': None,
                    'halluc_max': None,
                    'factual_layer_norm': None,
                    'halluc_layer_norm': None,
                },
            }
            # Compute gradient from layer profiles
            fact_conf = np.array(mdata['layer_profile']['factual_confident'])
            hall_conf = np.array(mdata['layer_profile']['halluc_confident'])
            n_layers = mdata['architecture']['layers']
            if len(fact_conf) > 1:
                fg = np.diff(fact_conf)
                hg = np.diff(hall_conf)
                pythia_results[mn]['gradient'] = {
                    'factual_max': float(np.max(fg)),
                    'factual_layer_norm': float(np.argmax(fg) / n_layers),
                    'halluc_max': float(np.max(hg)),
                    'halluc_layer_norm': float(np.argmax(hg) / n_layers),
                }
            # Add disc_pct if missing
            ht = pythia_results[mn]['head_taxonomy']
            if 'disc_pct' not in ht:
                ht['disc_pct'] = float(ht['discriminative'] / ht['total'] * 100)
            # Add delay_ratio if missing
            tr = pythia_results[mn]['transition']
            if 'delay_ratio' not in tr:
                if tr.get('halluc_normalized') and tr.get('factual_normalized') and tr['factual_normalized'] > 0:
                    tr['delay_ratio'] = tr['halluc_normalized'] / tr['factual_normalized']
                else:
                    tr['delay_ratio'] = None
        print(f"  Loaded {len(pythia_results)} Pythia models")
    else:
        print("  No Pythia data — running Pythia models too")
        PYTHIA_MODELS = [
            ("pythia-70m",  {'d': 512,  'layers': 6,  'heads': 8, 'family': 'pythia'}),
            ("pythia-160m", {'d': 768,  'layers': 12, 'heads': 12, 'family': 'pythia'}),
            ("pythia-410m", {'d': 1024, 'layers': 24, 'heads': 16, 'family': 'pythia'}),
            ("pythia-1b",   {'d': 2048, 'layers': 16, 'heads': 8, 'family': 'pythia'}),
        ]
        for model_name, arch_info in PYTHIA_MODELS:
            result = analyze_model(model_name, arch_info, device, is_pythia=True)
            pythia_results[model_name] = result

    # ── Phase 3: CROSS-ARCHITECTURE SYNTHESIS ──
    all_results = {}
    all_results.update(pythia_results)
    all_results.update(gpt2_results)

    # Sort by parameter count (approximate via d_model)
    sorted_models = sorted(all_results.items(),
                           key=lambda x: x[1]['architecture']['d'])

    print(f"\n{'='*70}")
    print(f"  CROSS-ARCHITECTURE SYNTHESIS ({len(all_results)} models)")
    print(f"{'='*70}")

    # ── 1. Delayed Phase Transition — THE UNIVERSALITY TEST ──
    print(f"\n  1. DELAYED PHASE TRANSITION (chaos→order):")
    print(f"     {'Model':17s} {'Family':7s} {'F_norm':>7s} {'H_norm':>7s} {'Delay':>7s} "
          f"{'p-value':>10s} {'Sig':>5s}")
    print(f"     {'-'*65}")

    all_transition_p = []
    for mn, mr in sorted_models:
        tr = mr['transition']
        fn = f"{tr['factual_normalized']:.3f}" if tr.get('factual_normalized') else "N/A"
        hn = f"{tr['halluc_normalized']:.3f}" if tr.get('halluc_normalized') else "N/A"
        delay = f"{tr['delay_ratio']:.2f}x" if tr.get('delay_ratio') else "N/A"
        p = tr.get('mann_whitney_p')
        p_str = f"{p:.6f}" if p is not None else "N/A"
        sig = ""
        if p is not None:
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            all_transition_p.append(p)
        family = mr.get('family', 'pythia')
        print(f"     {mn:17s} {family:7s} {fn:>7s} {hn:>7s} {delay:>7s} {p_str:>10s} {sig:>5s}")

    # Combined significance via Fisher's method
    if len(all_transition_p) >= 2:
        chi2 = -2 * sum(np.log(max(p, 1e-300)) for p in all_transition_p)
        fisher_p = 1 - sp.chi2.cdf(chi2, 2 * len(all_transition_p))
        print(f"\n     Fisher's combined p-value: {fisher_p:.2e}  "
              f"(chi²={chi2:.1f}, df={2*len(all_transition_p)})")

    # Split by family for comparison
    pythia_delays = [mr['transition']['delay_ratio'] for mn, mr in sorted_models
                     if mr.get('family') == 'pythia' and mr['transition'].get('delay_ratio')]
    gpt2_delays = [mr['transition']['delay_ratio'] for mn, mr in sorted_models
                   if mr.get('family') == 'gpt2' and mr['transition'].get('delay_ratio')]

    if pythia_delays and gpt2_delays:
        print(f"\n     Pythia delay ratio: {np.mean(pythia_delays):.3f} ± {np.std(pythia_delays):.3f}")
        print(f"     GPT-2  delay ratio: {np.mean(gpt2_delays):.3f} ± {np.std(gpt2_delays):.3f}")
        all_delays = pythia_delays + gpt2_delays
        print(f"     ALL    delay ratio: {np.mean(all_delays):.3f} ± {np.std(all_delays):.3f}")

    # ── 2. H/F Entropy Ratio vs Xi ──
    print(f"\n  2. H/F ENTROPY RATIO vs Xi ({XI:.4f}):")
    for mn, mr in sorted_models:
        br = mr['bootstrap_ratio']
        xi_s = "IN CI" if br['xi_in_ci'] else "OUTSIDE"
        family = mr.get('family', 'pythia')
        print(f"     {mn:17s} [{family:5s}] {br['mean']:.4f}  "
              f"95%CI=[{br['ci_2_5']:.4f}, {br['ci_97_5']:.4f}]  {xi_s}")

    # ── 3. Head Taxonomy ──
    print(f"\n  3. HEAD TAXONOMY:")
    print(f"     {'Model':17s} {'Family':7s} {'Total':>6s} {'%Disc':>6s} "
          f"{'Disc_depth':>11s} {'First25%':>9s}")
    for mn, mr in sorted_models:
        ht = mr['head_taxonomy']
        family = mr.get('family', 'pythia')
        dd = f"{ht.get('disc_depth_mean', 0):.3f}" if ht.get('disc_depth_mean') is not None else "N/A"
        fq = f"{ht.get('disc_first_quarter_pct', 0):.0f}%" if ht.get('disc_first_quarter_pct') is not None else "N/A"
        print(f"     {mn:17s} {family:7s} {ht['total']:6d} {ht['disc_pct']:5.1f}% "
              f"{dd:>11s} {fq:>9s}")

    # ── 4. Phase Transition Gradient ──
    print(f"\n  4. PHASE TRANSITION GRADIENT (sharpest Δconf between layers):")
    print(f"     {'Model':17s} {'Family':7s} {'F_max_Δ':>8s} {'F_depth':>8s} "
          f"{'H_max_Δ':>8s} {'H_depth':>8s}")
    for mn, mr in sorted_models:
        g = mr['gradient']
        family = mr.get('family', 'pythia')
        fm = f"{g['factual_max']:.3f}" if g.get('factual_max') is not None else "N/A"
        fl = f"{g['factual_layer_norm']:.3f}" if g.get('factual_layer_norm') is not None else "N/A"
        hm = f"{g['halluc_max']:.3f}" if g.get('halluc_max') is not None else "N/A"
        hl = f"{g['halluc_layer_norm']:.3f}" if g.get('halluc_layer_norm') is not None else "N/A"
        print(f"     {mn:17s} {family:7s} {fm:>8s} {fl:>8s} {hm:>8s} {hl:>8s}")

    # ── 5. Normalized Depth Overlay — H/F ratio at each relative depth ──
    print(f"\n  5. NORMALIZED DEPTH OVERLAY (H/F ratio at relative depth):")
    # Bin into 10 depth segments: 0-10%, 10-20%, ..., 90-100%
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_labels = [f"{int(bin_edges[i]*100)}-{int(bin_edges[i+1]*100)}%" for i in range(n_bins)]

    print(f"\n     {'Depth':>8s}", end="")
    for mn, _ in sorted_models:
        short = mn.replace('pythia-', 'P').replace('gpt2-', 'G').replace('gpt2', 'G-sm')
        print(f" {short:>8s}", end="")
    print()
    print(f"     {'─'*8}", end="")
    for _ in sorted_models:
        print(f" {'─'*8}", end="")
    print()

    model_binned_ratios = {}
    for mn, mr in sorted_models:
        ratios = mr['layer_profile']['layer_hf_ratios']
        n_layers = mr['architecture']['layers']
        # Assign each layer to a depth bin
        binned = [[] for _ in range(n_bins)]
        for li, r in enumerate(ratios):
            norm_depth = li / n_layers
            bin_idx = min(int(norm_depth * n_bins), n_bins - 1)
            binned[bin_idx].append(r)
        means = [float(np.mean(b)) if b else None for b in binned]
        model_binned_ratios[mn] = means

    for bi in range(n_bins):
        print(f"     {bin_labels[bi]:>8s}", end="")
        for mn, _ in sorted_models:
            v = model_binned_ratios[mn][bi]
            if v is not None:
                # Mark if near Xi
                marker = "*" if abs(v - XI) < 0.01 else " "
                print(f" {v:7.4f}{marker}", end="")
            else:
                print(f" {'—':>8s}", end="")
        print()
    print(f"     {'Xi':>8s}", end="")
    for _ in sorted_models:
        print(f" {XI:8.4f}", end="")
    print()

    # ── 6. Universality Score ──
    # Count how many results replicate across architectures
    print(f"\n  6. UNIVERSALITY SCORECARD:")

    # A. Delayed transition: significant in how many models?
    n_sig = sum(1 for p in all_transition_p if p < 0.05)
    print(f"     Delayed phase transition:  {n_sig}/{len(all_transition_p)} models significant")

    # B. Delayed transition consistent in GPT-2 family?
    gpt2_p = [mr['transition'].get('mann_whitney_p') for mn, mr in sorted_models
              if mr.get('family') == 'gpt2' and mr['transition'].get('mann_whitney_p') is not None]
    gpt2_sig = sum(1 for p in gpt2_p if p < 0.05)
    print(f"     GPT-2 delayed transition:  {gpt2_sig}/{len(gpt2_p)} models significant")

    # C. Pythia delayed transition
    pythia_p = [mr['transition'].get('mann_whitney_p') for mn, mr in sorted_models
                if mr.get('family') == 'pythia' and mr['transition'].get('mann_whitney_p') is not None]
    pythia_sig = sum(1 for p in pythia_p if p < 0.05)
    print(f"     Pythia delayed transition: {pythia_sig}/{len(pythia_p)} models significant")

    # D. Discriminative head % range
    disc_pcts = [mr['head_taxonomy']['disc_pct'] for _, mr in sorted_models]
    print(f"     Discriminative head range: {min(disc_pcts):.1f}% – {max(disc_pcts):.1f}% "
          f"(mean {np.mean(disc_pcts):.1f}%)")

    # E. Confidence threshold stability
    all_factual_conf = [mr['raw']['factual_mean'] for _, mr in sorted_models]
    all_halluc_conf = [mr['raw']['halluc_mean'] for _, mr in sorted_models]
    print(f"     Factual entropy range:     {min(all_factual_conf):.3f} – {max(all_factual_conf):.3f}")
    print(f"     Halluc entropy range:      {min(all_halluc_conf):.3f} – {max(all_halluc_conf):.3f}")

    # F. Overall verdict
    if gpt2_sig >= 2:
        verdict = "UNIVERSAL — delayed transition replicates across architecture families"
    elif gpt2_sig == 1:
        verdict = "SUGGESTIVE — partial replication in GPT-2"
    else:
        verdict = "PYTHIA ARTIFACT — does not replicate in GPT-2"
    print(f"\n     VERDICT: {verdict}")

    # ── Save ──
    output = {
        'experiment': 'exp_10_cross_architecture_universality',
        'timestamp': datetime.now().isoformat(),
        'xi': float(XI),
        'n_factual': len(FACTUAL),
        'n_hallucination': len(HALLUCINATION),
        'n_tokens_per_prompt': 20,
        'gpt2_models': {mn: gpt2_results[mn] for mn in gpt2_results},
        'pythia_models': {mn: pythia_results[mn] for mn in pythia_results},
        'cross_architecture': {
            'n_significant': n_sig,
            'n_total': len(all_transition_p),
            'gpt2_significant': gpt2_sig,
            'pythia_significant': pythia_sig,
            'fisher_p': float(fisher_p) if len(all_transition_p) >= 2 else None,
            'pythia_delay_mean': float(np.mean(pythia_delays)) if pythia_delays else None,
            'gpt2_delay_mean': float(np.mean(gpt2_delays)) if gpt2_delays else None,
            'disc_pct_range': [float(min(disc_pcts)), float(max(disc_pcts))],
            'verdict': verdict,
        },
        'elapsed_seconds': time.time() - t0,
    }

    results_dir = EXPERIMENT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"exp_10_cross_arch_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
