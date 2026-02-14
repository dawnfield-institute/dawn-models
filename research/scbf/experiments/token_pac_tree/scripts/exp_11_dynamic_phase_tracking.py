#!/usr/bin/env python3
"""
EXP 11: Dynamic Phase Tracking During Generation
==================================================

exp_09/10 established: hallucination prompts delay the chaos→order
phase transition by ~1.43x across architectures.

But that was an AVERAGE across 20 generated tokens. The deeper question:
what happens TOKEN BY TOKEN as the model generates?

Hypotheses:
  H1: DRIFT — The transition layer shifts deeper as the model generates
      hallucinated content (uncertainty accumulates)
  H2: ONSET — There's a detectable "moment" where confident_head_ratio
      drops, predicting hallucination BEFORE the model commits
  H3: OSCILLATION — The model's phase profile oscillates between
      confident and uncertain states during generation
  H4: DIVERGENCE — Factual and hallucination trajectories start similar
      but diverge at a predictable token position

New analyses:
  1. Token-by-token transition layer trajectory (all 7 models)
  2. Confident head ratio time series + trend detection
  3. Discriminative head attention: what tokens do they attend to?
  4. Phase velocity: rate of change of the transition boundary
  5. Cross-architecture temporal alignment

Also: longer generation (50 tokens) to see if patterns amplify.

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

# Use 3 models spanning both families + scales
MODELS = [
    ("pythia-160m", {'d': 768,  'layers': 12, 'heads': 12, 'family': 'pythia'}, True),
    ("pythia-410m", {'d': 1024, 'layers': 24, 'heads': 16, 'family': 'pythia'}, True),
    ("gpt2",        {'d': 768,  'layers': 12, 'heads': 12, 'family': 'gpt2'},   False),
    ("gpt2-medium", {'d': 1024, 'layers': 24, 'heads': 16, 'family': 'gpt2'},   False),
]

N_GEN_TOKENS = 50  # Longer generation to see temporal dynamics

# Fewer prompts but deeper analysis per prompt
FACTUAL = [
    "The capital of France is",
    "Water freezes at zero degrees",
    "The sun is a star that",
    "The Earth orbits around the",
    "Humans need oxygen to",
    "Trees produce oxygen through",
    "The moon orbits the",
    "Birds have wings and can",
    "The speed of light is",
    "DNA contains the genetic",
]

HALLUCINATION = [
    "The president of the fictional nation of Zorblandia announced",
    "Professor Blinkworth's theory of quantum gastronomy states that",
    "In the year 3847, humans colonized the planet",
    "The underwater city of Deepheim was founded by",
    "The 500th digit of pi multiplied by the 237th prime equals",
    "The lost continent of Meridia sank because",
    "According to the Fictional Encyclopedia of 2099,",
    "The alien species known as the Glorpians communicate by",
    "The secret society of time weavers believes that",
    "According to the Book of Infinite Recursion, the universe began when",
]


def run_dynamic_tracking(model, tokenizer, device, prompt, n_tokens=50,
                          disc_heads=None):
    """
    Generate tokens with FULL per-step diagnostics.
    Returns token-by-token phase profile + discriminative head attention.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    current_ids = input_ids

    steps = []

    for step in range(n_tokens):
        with torch.no_grad():
            outputs = model(current_ids, output_attentions=True)
            logits = outputs.logits[0, -1, :]
            attentions = outputs.attentions

        probs = torch.softmax(logits, dim=-1)
        top_prob = float(probs.max())
        top_entropy = float(-torch.sum(probs * torch.log(probs + 1e-10)))
        chosen_id = int(logits.argmax())
        chosen_token = tokenizer.decode([chosen_id])

        n_layers = len(attentions)
        n_heads = attentions[0].shape[1]

        # Per-layer, per-head entropy
        layer_head_entropy = np.zeros((n_layers, n_heads))
        layer_head_phase = []

        for li, layer_attn in enumerate(attentions):
            head_phases = []
            for hi in range(n_heads):
                attn_vec = layer_attn[0, hi, -1, :].cpu().numpy()
                attn_pos = attn_vec[attn_vec > 1e-10]
                h = float(-np.sum(attn_pos * np.log(attn_pos))) if len(attn_pos) > 0 else 0.0
                layer_head_entropy[li, hi] = h
                head_phases.append(classify_sec_phase(h).name)
            layer_head_phase.append(head_phases)

        # Per-layer aggregates
        layer_mean_entropy = layer_head_entropy.mean(axis=1)
        layer_confident_ratio = np.zeros(n_layers)
        for li in range(n_layers):
            confident = sum(1 for p in layer_head_phase[li]
                          if p in ('CRYSTALLIZED', 'ORDERED'))
            layer_confident_ratio[li] = confident / n_heads

        # Transition layer (where >50% heads are confident)
        transition_layer = None
        for li in range(n_layers):
            if layer_confident_ratio[li] >= 0.5:
                transition_layer = li
                break

        # Overall confident ratio (mean across all layers)
        overall_confident = float(np.mean(layer_confident_ratio))

        # Discriminative head attention patterns
        disc_attn_info = []
        if disc_heads:
            for (li, hi) in disc_heads:
                if li < n_layers and hi < n_heads:
                    attn_vec = attentions[li][0, hi, -1, :].cpu().numpy()
                    seq_len = len(attn_vec)

                    # Where is this head attending?
                    top5_positions = np.argsort(attn_vec)[-5:][::-1]
                    top5_weights = attn_vec[top5_positions]

                    # Attention to prompt vs generated tokens
                    attn_to_prompt = float(np.sum(attn_vec[:prompt_len]))
                    attn_to_generated = float(np.sum(attn_vec[prompt_len:]))
                    attn_to_recent = float(np.sum(attn_vec[-min(5, seq_len):]))
                    attn_to_self = float(attn_vec[-1]) if seq_len > 0 else 0.0

                    # Entropy of this head's attention
                    head_h = float(layer_head_entropy[li, hi])
                    head_phase = layer_head_phase[li][hi]

                    disc_attn_info.append({
                        'head': f"L{li}H{hi}",
                        'entropy': head_h,
                        'phase': head_phase,
                        'attn_to_prompt': attn_to_prompt,
                        'attn_to_generated': attn_to_generated,
                        'attn_to_recent_5': attn_to_recent,
                        'attn_to_self': attn_to_self,
                        'top_position': int(top5_positions[0]),
                        'top_weight': float(top5_weights[0]),
                        'concentration': float(np.max(attn_vec)),
                    })

        step_data = {
            'step': step,
            'token': chosen_token.strip(),
            'token_id': chosen_id,
            'top_prob': top_prob,
            'output_entropy': top_entropy,
            'transition_layer': transition_layer,
            'transition_norm': transition_layer / n_layers if transition_layer is not None else None,
            'overall_confident': overall_confident,
            'layer_mean_entropy': layer_mean_entropy.tolist(),
            'layer_confident_ratio': layer_confident_ratio.tolist(),
            'disc_heads': disc_attn_info,
        }
        steps.append(step_data)

        chosen_tensor = torch.tensor([[chosen_id]], device=device)
        current_ids = torch.cat([current_ids, chosen_tensor], dim=1)

    return {
        'prompt': prompt,
        'prompt_len': prompt_len,
        'n_tokens': n_tokens,
        'steps': steps,
    }


def identify_discriminative_heads(model, tokenizer, device, n_layers, n_heads,
                                   threshold=0.15):
    """Quick pass to identify discriminative heads using 5 prompts each."""
    quick_f = FACTUAL[:5]
    quick_h = HALLUCINATION[:5]

    def get_head_conf_rates(prompts, n_steps=10):
        rates = np.zeros((n_layers, n_heads))
        count = 0
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            current_ids = input_ids
            for step in range(n_steps):
                with torch.no_grad():
                    outputs = model(current_ids, output_attentions=True)
                    logits = outputs.logits[0, -1, :]
                    attentions = outputs.attentions
                chosen_id = int(logits.argmax())
                for li, layer_attn in enumerate(attentions):
                    for hi in range(n_heads):
                        attn_vec = layer_attn[0, hi, -1, :].cpu().numpy()
                        attn_pos = attn_vec[attn_vec > 1e-10]
                        h = float(-np.sum(attn_pos * np.log(attn_pos))) if len(attn_pos) > 0 else 0.0
                        phase = classify_sec_phase(h).name
                        if phase in ('CRYSTALLIZED', 'ORDERED'):
                            rates[li, hi] += 1
                count += 1
                chosen_tensor = torch.tensor([[chosen_id]], device=device)
                current_ids = torch.cat([current_ids, chosen_tensor], dim=1)
        rates /= count
        return rates

    f_rates = get_head_conf_rates(quick_f)
    h_rates = get_head_conf_rates(quick_h)
    flip = f_rates - h_rates

    # Get heads where |flip| > threshold
    disc = []
    for li in range(n_layers):
        for hi in range(n_heads):
            if abs(flip[li, hi]) > threshold:
                disc.append((li, hi, float(flip[li, hi])))

    disc.sort(key=lambda x: -abs(x[2]))
    return disc[:10]  # Top 10 discriminative heads


def analyze_trajectory(steps):
    """Compute trajectory statistics from token-by-token data."""
    transitions = [s['transition_norm'] for s in steps if s['transition_norm'] is not None]
    confidences = [s['overall_confident'] for s in steps]
    output_ents = [s['output_entropy'] for s in steps]

    result = {}

    # 1. Transition layer trend (linear regression over token position)
    if len(transitions) >= 5:
        valid_steps = [s['step'] for s in steps if s['transition_norm'] is not None]
        slope, intercept, r, p, se = sp.linregress(valid_steps, transitions)
        result['transition_slope'] = float(slope)
        result['transition_r'] = float(r)
        result['transition_p'] = float(p)
        result['transition_trend'] = 'deepening' if slope > 0 else 'shallowing'
    else:
        result['transition_slope'] = None
        result['transition_r'] = None
        result['transition_p'] = None
        result['transition_trend'] = 'insufficient_data'

    # 2. Confidence trend
    if len(confidences) >= 5:
        slope, intercept, r, p, se = sp.linregress(range(len(confidences)), confidences)
        result['confidence_slope'] = float(slope)
        result['confidence_r'] = float(r)
        result['confidence_p'] = float(p)
        result['confidence_trend'] = 'increasing' if slope > 0 else 'decreasing'
    else:
        result['confidence_slope'] = None

    # 3. Output entropy trend
    if len(output_ents) >= 5:
        slope, intercept, r, p, se = sp.linregress(range(len(output_ents)), output_ents)
        result['entropy_slope'] = float(slope)
        result['entropy_p'] = float(p)
        result['entropy_trend'] = 'increasing' if slope > 0 else 'decreasing'

    # 4. Phase stability — how much does transition layer jump around?
    if len(transitions) >= 3:
        result['transition_volatility'] = float(np.std(np.diff(transitions)))
        result['transition_range'] = float(max(transitions) - min(transitions))
    else:
        result['transition_volatility'] = None
        result['transition_range'] = None

    # 5. Early vs late comparison (first 15 vs last 15 tokens)
    if len(transitions) >= 30:
        early = transitions[:15]
        late = transitions[-15:]
        _, shift_p = sp.mannwhitneyu(early, late, alternative='two-sided')
        result['early_late_shift_p'] = float(shift_p)
        result['early_mean'] = float(np.mean(early))
        result['late_mean'] = float(np.mean(late))
        result['shift_direction'] = 'deeper' if np.mean(late) > np.mean(early) else 'shallower'
    elif len(transitions) >= 10:
        mid = len(transitions) // 2
        early = transitions[:mid]
        late = transitions[mid:]
        _, shift_p = sp.mannwhitneyu(early, late, alternative='two-sided')
        result['early_late_shift_p'] = float(shift_p)
        result['early_mean'] = float(np.mean(early))
        result['late_mean'] = float(np.mean(late))
        result['shift_direction'] = 'deeper' if np.mean(late) > np.mean(early) else 'shallower'

    # 6. Discriminative head attention shift
    if steps[0].get('disc_heads') and len(steps) >= 10:
        n_disc = len(steps[0]['disc_heads'])
        for di in range(n_disc):
            head_name = steps[0]['disc_heads'][di]['head']
            prompt_attn = [s['disc_heads'][di]['attn_to_prompt'] for s in steps
                          if len(s.get('disc_heads', [])) > di]
            gen_attn = [s['disc_heads'][di]['attn_to_generated'] for s in steps
                       if len(s.get('disc_heads', [])) > di]
            if len(prompt_attn) >= 10:
                slope, _, _, p, _ = sp.linregress(range(len(prompt_attn)), prompt_attn)
                result[f'disc_{head_name}_prompt_attn_slope'] = float(slope)
                result[f'disc_{head_name}_prompt_attn_p'] = float(p)

    return result


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("  EXP 11: Dynamic Phase Tracking During Generation")
    print("  Does the phase transition shift as the model generates?")
    print("=" * 70)

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results = {}

    for model_name, arch_info, is_pythia in MODELS:
        full_name = f"EleutherAI/{model_name}" if is_pythia else model_name

        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name}  [{arch_info['family']}]")
        print(f"  layers={arch_info['layers']}  heads={arch_info['heads']}  "
              f"d_model={arch_info['d']}  d_head={arch_info['d']//arch_info['heads']}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(full_name, cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(full_name, cache_dir=CACHE_DIR)
        model = model.to(device).eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        n_layers = arch_info['layers']
        n_heads = arch_info['heads']

        # Step 1: Identify discriminative heads
        print(f"\n  Identifying discriminative heads...")
        disc_heads_raw = identify_discriminative_heads(
            model, tokenizer, device, n_layers, n_heads)
        disc_heads = [(li, hi) for (li, hi, _) in disc_heads_raw]
        print(f"  Found {len(disc_heads)} discriminative heads:")
        for li, hi, flip in disc_heads_raw[:5]:
            print(f"    L{li}H{hi}: flip={flip*100:+.0f}%")

        # Step 2: Run dynamic tracking
        print(f"\n  Running dynamic tracking ({N_GEN_TOKENS} tokens per prompt)...")

        factual_runs = []
        halluc_runs = []

        for i, prompt in enumerate(FACTUAL):
            run = run_dynamic_tracking(model, tokenizer, device, prompt,
                                       n_tokens=N_GEN_TOKENS, disc_heads=disc_heads)
            traj = analyze_trajectory(run['steps'])
            run['trajectory'] = traj
            factual_runs.append(run)

            trans = [s['transition_norm'] for s in run['steps'] if s['transition_norm'] is not None]
            trend = traj.get('transition_slope', 0) or 0
            t_dir = "+" if trend > 0 else "-" if trend < 0 else "="
            tokens = ''.join(s['token'] for s in run['steps'][:8])
            print(f"  F[{i+1:2d}] conf={np.mean([s['overall_confident'] for s in run['steps']])*100:.0f}%  "
                  f"trans_slope={trend:+.5f}{t_dir}  '{prompt[:35]}' → {tokens}...")

        for i, prompt in enumerate(HALLUCINATION):
            run = run_dynamic_tracking(model, tokenizer, device, prompt,
                                       n_tokens=N_GEN_TOKENS, disc_heads=disc_heads)
            traj = analyze_trajectory(run['steps'])
            run['trajectory'] = traj
            halluc_runs.append(run)

            trans = [s['transition_norm'] for s in run['steps'] if s['transition_norm'] is not None]
            trend = traj.get('transition_slope', 0) or 0
            t_dir = "+" if trend > 0 else "-" if trend < 0 else "="
            tokens = ''.join(s['token'] for s in run['steps'][:8])
            print(f"  H[{i+1:2d}] conf={np.mean([s['overall_confident'] for s in run['steps']])*100:.0f}%  "
                  f"trans_slope={trend:+.5f}{t_dir}  '{prompt[:35]}' → {tokens}...")

        # ── Analysis ──
        print(f"\n  {'─'*55}")
        print(f"  TEMPORAL DYNAMICS ANALYSIS")
        print(f"  {'─'*55}")

        # A. Transition slope: do hallucination trajectories drift deeper?
        f_slopes = [r['trajectory']['transition_slope'] for r in factual_runs
                    if r['trajectory']['transition_slope'] is not None]
        h_slopes = [r['trajectory']['transition_slope'] for r in halluc_runs
                    if r['trajectory']['transition_slope'] is not None]

        print(f"\n  A. TRANSITION LAYER DRIFT (slope over {N_GEN_TOKENS} tokens):")
        if f_slopes:
            print(f"     Factual:  mean slope = {np.mean(f_slopes):+.6f} ± {np.std(f_slopes):.6f}")
        if h_slopes:
            print(f"     Halluc:   mean slope = {np.mean(h_slopes):+.6f} ± {np.std(h_slopes):.6f}")
        if f_slopes and h_slopes:
            _, slope_p = sp.mannwhitneyu(f_slopes, h_slopes, alternative='two-sided')
            sig = "***" if slope_p < 0.001 else "**" if slope_p < 0.01 else "*" if slope_p < 0.05 else "n.s."
            print(f"     Difference: p = {slope_p:.6f}  {sig}")
            print(f"     Interpretation: {'Halluc drifts deeper' if np.mean(h_slopes) > np.mean(f_slopes) else 'No differential drift'}")

        # B. Confidence trajectory
        f_conf_slopes = [r['trajectory'].get('confidence_slope') for r in factual_runs
                         if r['trajectory'].get('confidence_slope') is not None]
        h_conf_slopes = [r['trajectory'].get('confidence_slope') for r in halluc_runs
                         if r['trajectory'].get('confidence_slope') is not None]

        print(f"\n  B. CONFIDENCE TREND (overall confident ratio over time):")
        if f_conf_slopes:
            print(f"     Factual:  slope = {np.mean(f_conf_slopes):+.6f} ± {np.std(f_conf_slopes):.6f}")
        if h_conf_slopes:
            print(f"     Halluc:   slope = {np.mean(h_conf_slopes):+.6f} ± {np.std(h_conf_slopes):.6f}")
        if f_conf_slopes and h_conf_slopes:
            _, conf_p = sp.mannwhitneyu(f_conf_slopes, h_conf_slopes, alternative='two-sided')
            sig = "***" if conf_p < 0.001 else "**" if conf_p < 0.01 else "*" if conf_p < 0.05 else "n.s."
            print(f"     Difference: p = {conf_p:.6f}  {sig}")

        # C. Early vs Late shift
        print(f"\n  C. EARLY vs LATE TOKEN TRANSITION DEPTH:")
        for group_name, runs in [("Factual", factual_runs), ("Halluc", halluc_runs)]:
            early_means = [r['trajectory'].get('early_mean') for r in runs
                          if r['trajectory'].get('early_mean') is not None]
            late_means = [r['trajectory'].get('late_mean') for r in runs
                         if r['trajectory'].get('late_mean') is not None]
            if early_means and late_means:
                print(f"     {group_name:8s}: early={np.mean(early_means):.4f}  "
                      f"late={np.mean(late_means):.4f}  "
                      f"shift={np.mean(late_means)-np.mean(early_means):+.4f}")

        # D. Transition volatility
        f_vol = [r['trajectory'].get('transition_volatility') for r in factual_runs
                 if r['trajectory'].get('transition_volatility') is not None]
        h_vol = [r['trajectory'].get('transition_volatility') for r in halluc_runs
                 if r['trajectory'].get('transition_volatility') is not None]

        print(f"\n  D. TRANSITION VOLATILITY (how much does transition bounce?):")
        if f_vol:
            print(f"     Factual:  {np.mean(f_vol):.4f} ± {np.std(f_vol):.4f}")
        if h_vol:
            print(f"     Halluc:   {np.mean(h_vol):.4f} ± {np.std(h_vol):.4f}")
        if f_vol and h_vol:
            _, vol_p = sp.mannwhitneyu(f_vol, h_vol, alternative='two-sided')
            sig = "***" if vol_p < 0.001 else "**" if vol_p < 0.01 else "*" if vol_p < 0.05 else "n.s."
            print(f"     Difference: p = {vol_p:.6f}  {sig}")

        # E. Discriminative head attention shift
        print(f"\n  E. DISCRIMINATIVE HEAD ATTENTION PATTERNS:")
        if disc_heads_raw:
            # Average attention to prompt tokens over time
            for di, (li, hi, flip) in enumerate(disc_heads_raw[:3]):
                head_name = f"L{li}H{hi}"
                f_prompt_attn = []
                h_prompt_attn = []
                for run in factual_runs:
                    for s in run['steps']:
                        if len(s.get('disc_heads', [])) > di:
                            f_prompt_attn.append(s['disc_heads'][di]['attn_to_prompt'])
                for run in halluc_runs:
                    for s in run['steps']:
                        if len(s.get('disc_heads', [])) > di:
                            h_prompt_attn.append(s['disc_heads'][di]['attn_to_prompt'])

                if f_prompt_attn and h_prompt_attn:
                    print(f"     {head_name} (flip={flip*100:+.0f}%):")
                    print(f"       Factual attn→prompt: {np.mean(f_prompt_attn):.3f}")
                    print(f"       Halluc  attn→prompt: {np.mean(h_prompt_attn):.3f}")

                    # Do disc heads shift attention FROM prompt tokens during halluc?
                    f_early_attn = []
                    f_late_attn = []
                    h_early_attn = []
                    h_late_attn = []
                    for run in factual_runs:
                        for s in run['steps'][:15]:
                            if len(s.get('disc_heads', [])) > di:
                                f_early_attn.append(s['disc_heads'][di]['attn_to_prompt'])
                        for s in run['steps'][-15:]:
                            if len(s.get('disc_heads', [])) > di:
                                f_late_attn.append(s['disc_heads'][di]['attn_to_prompt'])
                    for run in halluc_runs:
                        for s in run['steps'][:15]:
                            if len(s.get('disc_heads', [])) > di:
                                h_early_attn.append(s['disc_heads'][di]['attn_to_prompt'])
                        for s in run['steps'][-15:]:
                            if len(s.get('disc_heads', [])) > di:
                                h_late_attn.append(s['disc_heads'][di]['attn_to_prompt'])

                    if f_early_attn and h_late_attn:
                        print(f"       F: early→prompt={np.mean(f_early_attn):.3f}  "
                              f"late→prompt={np.mean(f_late_attn):.3f}")
                        print(f"       H: early→prompt={np.mean(h_early_attn):.3f}  "
                              f"late→prompt={np.mean(h_late_attn):.3f}")

        # F. Token-by-token average phase profile (10-step bins)
        print(f"\n  F. BINNED TEMPORAL PHASE PROFILE (avg confident ratio):")
        bin_size = 10
        n_bins = N_GEN_TOKENS // bin_size
        print(f"     {'Tokens':>10s}  {'Factual':>8s}  {'Halluc':>8s}  {'Δ':>8s}  {'H/F':>6s}")
        for bi in range(n_bins):
            start = bi * bin_size
            end = start + bin_size
            f_vals = []
            h_vals = []
            for run in factual_runs:
                for s in run['steps'][start:end]:
                    f_vals.append(s['overall_confident'])
            for run in halluc_runs:
                for s in run['steps'][start:end]:
                    h_vals.append(s['overall_confident'])
            if f_vals and h_vals:
                fm = np.mean(f_vals)
                hm = np.mean(h_vals)
                ratio = hm / fm if fm > 0 else 0
                print(f"     {start:3d}-{end:3d}     {fm*100:7.1f}%  {hm*100:7.1f}%  "
                      f"{(hm-fm)*100:+7.1f}%  {ratio:.3f}")

        # Store results
        model_result = {
            'model_name': model_name,
            'family': arch_info['family'],
            'architecture': arch_info,
            'disc_heads': [(li, hi, float(f)) for li, hi, f in disc_heads_raw],
            'factual_trajectories': {
                'slopes': f_slopes,
                'conf_slopes': f_conf_slopes,
                'volatilities': f_vol,
            },
            'halluc_trajectories': {
                'slopes': h_slopes,
                'conf_slopes': h_conf_slopes,
                'volatilities': h_vol,
            },
            'temporal_bins': {},
        }

        # Store per-run summary (not full step data — too large)
        model_result['per_prompt'] = []
        for run in factual_runs + halluc_runs:
            model_result['per_prompt'].append({
                'prompt': run['prompt'],
                'group': 'factual' if run in factual_runs else 'hallucination',
                'trajectory': run['trajectory'],
                'mean_confident': float(np.mean([s['overall_confident'] for s in run['steps']])),
                'transitions': [s['transition_norm'] for s in run['steps']],
                'confidences': [s['overall_confident'] for s in run['steps']],
            })

        all_results[model_name] = model_result

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # ── CROSS-MODEL TEMPORAL SYNTHESIS ──
    print(f"\n{'='*70}")
    print(f"  CROSS-MODEL TEMPORAL SYNTHESIS")
    print(f"{'='*70}")

    print(f"\n  TRANSITION DRIFT SUMMARY:")
    print(f"  {'Model':17s} {'Family':7s} {'F_slope':>10s} {'H_slope':>10s} {'p':>10s} {'Sig':>5s}")
    for mn in all_results:
        mr = all_results[mn]
        f_s = mr['factual_trajectories']['slopes']
        h_s = mr['halluc_trajectories']['slopes']
        family = mr['family']
        if f_s and h_s:
            _, p = sp.mannwhitneyu(f_s, h_s, alternative='two-sided')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {mn:17s} {family:7s} {np.mean(f_s):+.7f} {np.mean(h_s):+.7f} {p:.6f} {sig}")

    print(f"\n  CONFIDENCE DRIFT SUMMARY:")
    print(f"  {'Model':17s} {'Family':7s} {'F_conf_slope':>12s} {'H_conf_slope':>12s} {'p':>10s}")
    for mn in all_results:
        mr = all_results[mn]
        f_c = mr['factual_trajectories']['conf_slopes']
        h_c = mr['halluc_trajectories']['conf_slopes']
        family = mr['family']
        if f_c and h_c:
            _, p = sp.mannwhitneyu(f_c, h_c, alternative='two-sided')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {mn:17s} {family:7s} {np.mean(f_c):+.8f} {np.mean(h_c):+.8f} {p:.6f} {sig}")

    print(f"\n  VOLATILITY SUMMARY:")
    print(f"  {'Model':17s} {'Family':7s} {'F_vol':>8s} {'H_vol':>8s} {'p':>10s}")
    for mn in all_results:
        mr = all_results[mn]
        f_v = mr['factual_trajectories']['volatilities']
        h_v = mr['halluc_trajectories']['volatilities']
        family = mr['family']
        if f_v and h_v:
            _, p = sp.mannwhitneyu(f_v, h_v, alternative='two-sided')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {mn:17s} {family:7s} {np.mean(f_v):.4f}  {np.mean(h_v):.4f}  {p:.6f} {sig}")

    # ── Save ──
    output = {
        'experiment': 'exp_11_dynamic_phase_tracking',
        'timestamp': datetime.now().isoformat(),
        'n_gen_tokens': N_GEN_TOKENS,
        'n_factual': len(FACTUAL),
        'n_hallucination': len(HALLUCINATION),
        'models': all_results,
        'elapsed_seconds': time.time() - t0,
    }

    results_dir = EXPERIMENT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"exp_11_dynamics_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
