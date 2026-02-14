#!/usr/bin/env python3
"""
EXP 12: PAC Conservation During Hallucination
===============================================

Core hypothesis (from user insight):
  If PAC (f(Parent) = Σf(Children)) applies to neural networks, then
  when the model shifts attention/activation during hallucination,
  that shift must COME FROM SOMEWHERE. Information is redistributed,
  not created.

  Two possible outcomes:
    A. CONSERVATION HOLDS: When one head goes chaotic, another compensates.
       Hallucination is a redistribution problem — the model moves
       information to the wrong place, breaking the correct mapping.
    B. CONSERVATION BREAKS: Total entropy increases without compensation.
       The model is generating "synthetic structure" from nothing —
       a PAC violation. This IS the hallucination.

  If (B), then hallucination = PAC violation, and the magnitude of
  the violation predicts hallucination severity.

Measurements:
  1. INTRA-LAYER HEAD CONSERVATION
     - Total entropy across all heads in a layer: constant or varies?
     - When one head goes chaotic, does another crystallize?
     - Entropy variance within layer: redistributed or added?

  2. CROSS-LAYER CONSERVATION
     - Total confident ratio across all layers: does it sum to a constant?
     - Layer-level entropy budget: when one layer shifts, does another compensate?

  3. ATTENTION MASS FLOW
     - Where does "lost" attention go when disc heads shift to prompt?
     - Attention redistribution matrix: factual vs halluc

  4. RESIDUAL STREAM NORM CONSERVATION
     - L2 norm of hidden states per layer: constant budget or not?
     - Norm growth rate: does hallucination show different norm dynamics?

  5. PAC VIOLATION INDEX
     - Δ(total entropy) between factual and hallucination
     - If PAC holds: this should be ~0 (redistribution)
     - If PAC breaks: this is positive (entropy creation)
     - Per-layer PAC violation: where does conservation break first?

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

MODELS = [
    ("pythia-160m", {'d': 768,  'layers': 12, 'heads': 12, 'family': 'pythia'}, True),
    ("pythia-410m", {'d': 1024, 'layers': 24, 'heads': 16, 'family': 'pythia'}, True),
    ("gpt2",        {'d': 768,  'layers': 12, 'heads': 12, 'family': 'gpt2'},   False),
    ("gpt2-medium", {'d': 1024, 'layers': 24, 'heads': 16, 'family': 'gpt2'},   False),
]

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
    "The human heart pumps",
    "Gravity pulls objects toward the",
    "Iron is a type of",
    "Rain falls from clouds when",
    "Fish breathe using their",
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
    "Dr. Thornwick proved that time travel requires",
    "The mystical element Chronium allows users to",
    "The Blatherstein equation for recursive entropy states",
    "The planet Nexarion has seventeen moons that",
    "In alternate universe 7B, gravity works by",
]

N_TOKENS = 20


def run_conservation_analysis(model, tokenizer, device, prompt, n_tokens=20):
    """
    Generate tokens capturing CONSERVATION diagnostics:
    - Per-head entropy (for intra-layer budget)
    - Hidden state norms (for residual stream conservation)
    - Full attention matrices (for mass flow)
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    current_ids = input_ids

    steps = []

    for step in range(n_tokens):
        with torch.no_grad():
            outputs = model(
                current_ids,
                output_attentions=True,
                output_hidden_states=True,
            )
            logits = outputs.logits[0, -1, :]
            attentions = outputs.attentions
            hidden_states = outputs.hidden_states  # tuple of (batch, seq, d_model)

        chosen_id = int(logits.argmax())

        n_layers = len(attentions)
        n_heads = attentions[0].shape[1]
        seq_len = attentions[0].shape[-1]

        # ── 1. PER-HEAD ENTROPY MATRIX ──
        layer_head_entropy = np.zeros((n_layers, n_heads))
        for li, layer_attn in enumerate(attentions):
            for hi in range(n_heads):
                attn_vec = layer_attn[0, hi, -1, :].cpu().numpy()
                attn_pos = attn_vec[attn_vec > 1e-10]
                h = float(-np.sum(attn_pos * np.log(attn_pos))) if len(attn_pos) > 0 else 0.0
                layer_head_entropy[li, hi] = h

        # Intra-layer entropy budget
        layer_total_entropy = layer_head_entropy.sum(axis=1)  # total H per layer
        layer_mean_entropy = layer_head_entropy.mean(axis=1)
        layer_entropy_std = layer_head_entropy.std(axis=1)    # intra-layer spread

        # Cross-head correlation within each layer
        # When one head goes up, does another go down?
        layer_head_correlations = []
        for li in range(n_layers):
            # Pairwise: is the variance across heads "compensatory"?
            head_vals = layer_head_entropy[li, :]
            # How far from uniform is this layer's head distribution?
            if np.mean(head_vals) > 0:
                cv = float(np.std(head_vals) / np.mean(head_vals))  # coefficient of variation
            else:
                cv = 0.0
            layer_head_correlations.append(cv)

        # ── 2. RESIDUAL STREAM NORMS ──
        # hidden_states[0] = embedding, hidden_states[1] = after layer 0, etc.
        layer_norms = []
        for li in range(len(hidden_states)):
            # L2 norm of the last token's hidden state
            hs = hidden_states[li][0, -1, :]  # (d_model,)
            norm = float(torch.norm(hs, p=2))
            layer_norms.append(norm)

        # Norm growth rate: how much does the norm change per layer?
        norm_diffs = [layer_norms[i+1] - layer_norms[i]
                      for i in range(len(layer_norms)-1)]

        # ── 3. ATTENTION MASS FLOW ──
        # For each head: where is attention going?
        # Split into: prompt tokens, generated tokens, self
        layer_attn_to_prompt = np.zeros((n_layers, n_heads))
        layer_attn_to_generated = np.zeros((n_layers, n_heads))
        layer_attn_to_self = np.zeros((n_layers, n_heads))
        layer_attn_concentration = np.zeros((n_layers, n_heads))  # max attention weight

        for li, layer_attn in enumerate(attentions):
            for hi in range(n_heads):
                attn_vec = layer_attn[0, hi, -1, :].cpu().numpy()
                layer_attn_to_prompt[li, hi] = float(np.sum(attn_vec[:prompt_len]))
                layer_attn_to_generated[li, hi] = float(np.sum(attn_vec[prompt_len:]))
                layer_attn_to_self[li, hi] = float(attn_vec[-1])
                layer_attn_concentration[li, hi] = float(np.max(attn_vec))

        step_data = {
            'step': step,
            'token_id': chosen_id,
            # Intra-layer
            'layer_total_entropy': layer_total_entropy.tolist(),
            'layer_mean_entropy': layer_mean_entropy.tolist(),
            'layer_entropy_std': layer_entropy_std.tolist(),
            'layer_head_cv': layer_head_correlations,
            # Per-head entropy (full matrix)
            'head_entropy': layer_head_entropy.tolist(),
            # Residual stream
            'layer_norms': layer_norms,
            'norm_diffs': norm_diffs,
            # Attention flow
            'layer_attn_to_prompt': layer_attn_to_prompt.mean(axis=1).tolist(),
            'layer_attn_to_generated': layer_attn_to_generated.mean(axis=1).tolist(),
        }
        steps.append(step_data)

        chosen_tensor = torch.tensor([[chosen_id]], device=device)
        current_ids = torch.cat([current_ids, chosen_tensor], dim=1)

    return {
        'prompt': prompt,
        'prompt_len': prompt_len,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'steps': steps,
    }


def compute_conservation_metrics(factual_runs, halluc_runs, n_layers, n_heads):
    """Compute PAC conservation metrics from run data."""
    metrics = {}

    # ── 1. INTRA-LAYER ENTROPY BUDGET ──
    # Is total entropy per layer constant (conservation) or variable (violation)?
    # Average across all tokens and prompts

    fact_layer_totals = np.zeros(n_layers)
    hall_layer_totals = np.zeros(n_layers)
    fact_layer_stds = np.zeros(n_layers)
    hall_layer_stds = np.zeros(n_layers)

    for run in factual_runs:
        for step in run['steps']:
            fact_layer_totals += np.array(step['layer_total_entropy'])
    fact_layer_totals /= (len(factual_runs) * len(factual_runs[0]['steps']))

    for run in halluc_runs:
        for step in run['steps']:
            hall_layer_totals += np.array(step['layer_total_entropy'])
    hall_layer_totals /= (len(halluc_runs) * len(halluc_runs[0]['steps']))

    # Intra-layer coefficient of variation
    fact_cvs = np.zeros(n_layers)
    hall_cvs = np.zeros(n_layers)
    for run in factual_runs:
        for step in run['steps']:
            fact_cvs += np.array(step['layer_head_cv'])
    fact_cvs /= (len(factual_runs) * len(factual_runs[0]['steps']))

    for run in halluc_runs:
        for step in run['steps']:
            hall_cvs += np.array(step['layer_head_cv'])
    hall_cvs /= (len(halluc_runs) * len(halluc_runs[0]['steps']))

    metrics['intra_layer'] = {
        'factual_total_entropy': fact_layer_totals.tolist(),
        'halluc_total_entropy': hall_layer_totals.tolist(),
        'entropy_difference': (hall_layer_totals - fact_layer_totals).tolist(),
        'factual_head_cv': fact_cvs.tolist(),
        'halluc_head_cv': hall_cvs.tolist(),
    }

    # ── 2. CROSS-LAYER CONSERVATION ──
    # Sum total entropy across ALL heads in ALL layers — the full system budget
    fact_system_budget = float(np.sum(fact_layer_totals))
    hall_system_budget = float(np.sum(hall_layer_totals))
    budget_violation = hall_system_budget - fact_system_budget
    budget_violation_pct = (budget_violation / fact_system_budget * 100) if fact_system_budget > 0 else 0

    # Per-prompt system-level budgets for statistical test
    fact_budgets = []
    hall_budgets = []
    for run in factual_runs:
        prompt_budget = []
        for step in run['steps']:
            prompt_budget.append(sum(step['layer_total_entropy']))
        fact_budgets.append(float(np.mean(prompt_budget)))
    for run in halluc_runs:
        prompt_budget = []
        for step in run['steps']:
            prompt_budget.append(sum(step['layer_total_entropy']))
        hall_budgets.append(float(np.mean(prompt_budget)))

    _, budget_p = sp.mannwhitneyu(fact_budgets, hall_budgets, alternative='two-sided')

    metrics['cross_layer'] = {
        'factual_system_budget': fact_system_budget,
        'halluc_system_budget': hall_system_budget,
        'budget_violation': budget_violation,
        'budget_violation_pct': budget_violation_pct,
        'budget_p': float(budget_p),
        'factual_budgets': fact_budgets,
        'halluc_budgets': hall_budgets,
    }

    # ── 3. PER-LAYER PAC VIOLATION ──
    # For each layer: how much MORE total entropy during hallucination?
    # Positive = hallucination adds entropy (PAC violation)
    # Zero = redistribution only (PAC holds)
    layer_violations = hall_layer_totals - fact_layer_totals
    layer_violation_pcts = ((hall_layer_totals - fact_layer_totals) /
                            np.maximum(fact_layer_totals, 1e-10) * 100)

    # Statistical test per layer
    layer_p_values = []
    for li in range(n_layers):
        f_layer_vals = []
        h_layer_vals = []
        for run in factual_runs:
            layer_vals = [step['layer_total_entropy'][li] for step in run['steps']]
            f_layer_vals.append(float(np.mean(layer_vals)))
        for run in halluc_runs:
            layer_vals = [step['layer_total_entropy'][li] for step in run['steps']]
            h_layer_vals.append(float(np.mean(layer_vals)))
        _, p = sp.mannwhitneyu(f_layer_vals, h_layer_vals, alternative='two-sided')
        layer_p_values.append(float(p))

    metrics['per_layer_violation'] = {
        'violations': layer_violations.tolist(),
        'violation_pcts': layer_violation_pcts.tolist(),
        'p_values': layer_p_values,
    }

    # ── 4. RESIDUAL STREAM NORM ANALYSIS ──
    # Are hidden state norms conserved or do they grow differently?
    n_hidden = n_layers + 1  # embedding + n_layers
    fact_norms = np.zeros(n_hidden)
    hall_norms = np.zeros(n_hidden)
    fact_norm_count = 0
    hall_norm_count = 0

    for run in factual_runs:
        for step in run['steps']:
            norms = step['layer_norms']
            fact_norms[:len(norms)] += np.array(norms)
            fact_norm_count += 1
    fact_norms /= max(fact_norm_count, 1)

    for run in halluc_runs:
        for step in run['steps']:
            norms = step['layer_norms']
            hall_norms[:len(norms)] += np.array(norms)
            hall_norm_count += 1
    hall_norms /= max(hall_norm_count, 1)

    # Norm growth rate per layer
    fact_norm_growth = np.diff(fact_norms)
    hall_norm_growth = np.diff(hall_norms)

    # Total norm budget (sum across all layers)
    fact_total_norm = float(np.sum(fact_norms))
    hall_total_norm = float(np.sum(hall_norms))

    metrics['residual_norms'] = {
        'factual_norms': fact_norms.tolist(),
        'halluc_norms': hall_norms.tolist(),
        'factual_total_norm': fact_total_norm,
        'halluc_total_norm': hall_total_norm,
        'norm_ratio': hall_total_norm / fact_total_norm if fact_total_norm > 0 else 0,
        'factual_growth': fact_norm_growth.tolist(),
        'halluc_growth': hall_norm_growth.tolist(),
    }

    # ── 5. COMPENSATION ANALYSIS ──
    # Key test: when a layer's entropy increases during hallucination,
    # does ANY other layer's entropy decrease to compensate?
    # If yes: PAC redistribution. If no: PAC violation.

    positive_violations = sum(1 for v in layer_violations if v > 0)
    negative_violations = sum(1 for v in layer_violations if v < 0)
    net_positive = float(sum(v for v in layer_violations if v > 0))
    net_negative = float(sum(v for v in layer_violations if v < 0))

    metrics['compensation'] = {
        'layers_with_more_entropy': positive_violations,
        'layers_with_less_entropy': negative_violations,
        'net_positive_violation': net_positive,
        'net_negative_compensation': net_negative,
        'net_uncompensated': net_positive + net_negative,  # if 0: perfect PAC
        'compensation_ratio': abs(net_negative / net_positive) if net_positive > 0 else 0,
        # 1.0 = perfect conservation, <1 = PAC violation
    }

    # ── 6. HEAD-LEVEL ENTROPY REDISTRIBUTION ──
    # Per-head: factual vs halluc average entropy
    fact_head_entropy = np.zeros((n_layers, n_heads))
    hall_head_entropy = np.zeros((n_layers, n_heads))

    for run in factual_runs:
        for step in run['steps']:
            fact_head_entropy += np.array(step['head_entropy'])
    fact_head_entropy /= (len(factual_runs) * len(factual_runs[0]['steps']))

    for run in halluc_runs:
        for step in run['steps']:
            hall_head_entropy += np.array(step['head_entropy'])
    hall_head_entropy /= (len(halluc_runs) * len(halluc_runs[0]['steps']))

    head_delta = hall_head_entropy - fact_head_entropy

    # Count heads that increase vs decrease
    heads_increase = int(np.sum(head_delta > 0.01))
    heads_decrease = int(np.sum(head_delta < -0.01))
    heads_stable = n_layers * n_heads - heads_increase - heads_decrease

    # For heads that increase: mean increase
    increase_vals = head_delta[head_delta > 0.01]
    decrease_vals = head_delta[head_delta < -0.01]

    metrics['head_redistribution'] = {
        'heads_increase': heads_increase,
        'heads_decrease': heads_decrease,
        'heads_stable': heads_stable,
        'mean_increase': float(np.mean(increase_vals)) if len(increase_vals) > 0 else 0,
        'mean_decrease': float(np.mean(decrease_vals)) if len(decrease_vals) > 0 else 0,
        'total_increase': float(np.sum(increase_vals)) if len(increase_vals) > 0 else 0,
        'total_decrease': float(np.sum(decrease_vals)) if len(decrease_vals) > 0 else 0,
        'net_head_violation': float(np.sum(head_delta)),
    }

    # ── 7. TEMPORAL BUDGET EVOLUTION ──
    # Does the budget violation grow over time?
    n_steps = len(factual_runs[0]['steps'])
    per_step_fact_budget = np.zeros(n_steps)
    per_step_hall_budget = np.zeros(n_steps)

    for run in factual_runs:
        for si, step in enumerate(run['steps']):
            per_step_fact_budget[si] += sum(step['layer_total_entropy'])
    per_step_fact_budget /= len(factual_runs)

    for run in halluc_runs:
        for si, step in enumerate(run['steps']):
            per_step_hall_budget[si] += sum(step['layer_total_entropy'])
    per_step_hall_budget /= len(halluc_runs)

    budget_gap = per_step_hall_budget - per_step_fact_budget

    # Does the gap grow or shrink?
    if len(budget_gap) >= 5:
        slope, _, r, p, _ = sp.linregress(range(len(budget_gap)), budget_gap)
        metrics['temporal_budget'] = {
            'gap_slope': float(slope),
            'gap_r': float(r),
            'gap_p': float(p),
            'gap_trend': 'growing' if slope > 0 else 'shrinking',
            'early_gap': float(np.mean(budget_gap[:5])),
            'late_gap': float(np.mean(budget_gap[-5:])),
        }
    else:
        metrics['temporal_budget'] = {'gap_slope': None}

    return metrics


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("  EXP 12: PAC Conservation During Hallucination")
    print("  Does information budget balance, or does hallucination")
    print("  create uncompensated entropy?")
    print("=" * 70)

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results = {}

    for model_name, arch_info, is_pythia in MODELS:
        full_name = f"EleutherAI/{model_name}" if is_pythia else model_name

        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name}  [{arch_info['family']}]")
        print(f"  layers={arch_info['layers']}  heads={arch_info['heads']}  "
              f"d_model={arch_info['d']}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(full_name, cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(full_name, cache_dir=CACHE_DIR)
        model = model.to(device).eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        n_layers = arch_info['layers']
        n_heads = arch_info['heads']

        print(f"\n  Running conservation analysis ({N_TOKENS} tokens × "
              f"{len(FACTUAL)+len(HALLUCINATION)} prompts)...")

        factual_runs = []
        for i, prompt in enumerate(FACTUAL):
            run = run_conservation_analysis(model, tokenizer, device, prompt, N_TOKENS)
            factual_runs.append(run)
            budget = np.mean([sum(s['layer_total_entropy']) for s in run['steps']])
            print(f"  F[{i+1:2d}] budget={budget:.1f}  '{prompt[:40]}'")

        halluc_runs = []
        for i, prompt in enumerate(HALLUCINATION):
            run = run_conservation_analysis(model, tokenizer, device, prompt, N_TOKENS)
            halluc_runs.append(run)
            budget = np.mean([sum(s['layer_total_entropy']) for s in run['steps']])
            print(f"  H[{i+1:2d}] budget={budget:.1f}  '{prompt[:40]}'")

        # Compute conservation metrics
        metrics = compute_conservation_metrics(
            factual_runs, halluc_runs, n_layers, n_heads)

        # ── Print Results ──
        print(f"\n  {'─'*55}")
        print(f"  PAC CONSERVATION ANALYSIS")
        print(f"  {'─'*55}")

        # 1. System-level budget
        cl = metrics['cross_layer']
        print(f"\n  1. SYSTEM ENTROPY BUDGET (all heads × all layers):")
        print(f"     Factual total:  {cl['factual_system_budget']:.2f}")
        print(f"     Halluc total:   {cl['halluc_system_budget']:.2f}")
        print(f"     Violation:      {cl['budget_violation']:+.2f} "
              f"({cl['budget_violation_pct']:+.1f}%)")
        sig = "***" if cl['budget_p'] < 0.001 else "**" if cl['budget_p'] < 0.01 else "*" if cl['budget_p'] < 0.05 else "n.s."
        print(f"     Mann-Whitney p: {cl['budget_p']:.6f}  {sig}")

        if cl['budget_violation_pct'] > 0:
            print(f"     → PAC VIOLATION: Hallucination ADDS {cl['budget_violation_pct']:.1f}% "
                  f"uncompensated entropy")
        else:
            print(f"     → PAC CONSERVED: No excess entropy during hallucination")

        # 2. Per-layer violation
        plv = metrics['per_layer_violation']
        print(f"\n  2. PER-LAYER PAC VIOLATION:")
        print(f"     {'Layer':>5s} {'ΔEntropy':>10s} {'%Change':>8s} {'p-value':>10s} {'Sig':>5s}")
        for li in range(n_layers):
            v = plv['violations'][li]
            pct = plv['violation_pcts'][li]
            p = plv['p_values'][li]
            s = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            marker = "▲" if v > 0 else "▼" if v < 0 else "="
            print(f"     L{li:3d}  {v:+10.3f} {pct:+7.1f}%  {p:10.6f} {s:>5s} {marker}")

        # 3. Compensation analysis
        comp = metrics['compensation']
        print(f"\n  3. COMPENSATION ANALYSIS:")
        print(f"     Layers gaining entropy:  {comp['layers_with_more_entropy']}")
        print(f"     Layers losing entropy:   {comp['layers_with_less_entropy']}")
        print(f"     Total gained:            {comp['net_positive_violation']:+.3f}")
        print(f"     Total compensated:       {comp['net_negative_compensation']:+.3f}")
        print(f"     NET uncompensated:       {comp['net_uncompensated']:+.3f}")
        print(f"     Compensation ratio:      {comp['compensation_ratio']:.3f}")
        print(f"     (1.0 = perfect PAC, 0.0 = no compensation)")

        # 4. Head-level redistribution
        hr = metrics['head_redistribution']
        total = n_layers * n_heads
        print(f"\n  4. HEAD-LEVEL REDISTRIBUTION ({total} total heads):")
        print(f"     Heads with MORE entropy (Δ>0.01):  {hr['heads_increase']} "
              f"(mean Δ={hr['mean_increase']:+.3f})")
        print(f"     Heads with LESS entropy (Δ<-0.01): {hr['heads_decrease']} "
              f"(mean Δ={hr['mean_decrease']:+.3f})")
        print(f"     Stable heads:                      {hr['heads_stable']}")
        print(f"     Total head increase:  {hr['total_increase']:+.3f}")
        print(f"     Total head decrease:  {hr['total_decrease']:+.3f}")
        print(f"     NET head violation:   {hr['net_head_violation']:+.3f}")

        # 5. Residual stream norms
        rn = metrics['residual_norms']
        print(f"\n  5. RESIDUAL STREAM NORMS:")
        print(f"     Factual total norm:  {rn['factual_total_norm']:.1f}")
        print(f"     Halluc total norm:   {rn['halluc_total_norm']:.1f}")
        print(f"     Norm ratio (H/F):    {rn['norm_ratio']:.4f}")

        # Show norm per layer (condensed)
        print(f"     {'Layer':>5s} {'F_norm':>8s} {'H_norm':>8s} {'Ratio':>7s}")
        # Show first 3, middle, last 3
        show_layers = list(range(min(3, n_layers+1))) + \
                      [n_layers // 2] + \
                      list(range(max(0, n_layers-2), n_layers+1))
        show_layers = sorted(set(show_layers))
        for li in show_layers:
            if li < len(rn['factual_norms']):
                fn = rn['factual_norms'][li]
                hn = rn['halluc_norms'][li]
                ratio = hn / fn if fn > 0 else 0
                label = "emb" if li == 0 else f"L{li-1}"
                print(f"     {label:>5s} {fn:8.1f} {hn:8.1f} {ratio:7.4f}")

        # 6. Temporal budget evolution
        tb = metrics['temporal_budget']
        if tb.get('gap_slope') is not None:
            print(f"\n  6. TEMPORAL BUDGET GAP:")
            print(f"     Early gap (tokens 0-4):  {tb['early_gap']:+.2f}")
            print(f"     Late gap (tokens 15-19): {tb['late_gap']:+.2f}")
            print(f"     Gap slope:               {tb['gap_slope']:+.4f}")
            sig = "***" if tb['gap_p'] < 0.001 else "**" if tb['gap_p'] < 0.01 else "*" if tb['gap_p'] < 0.05 else "n.s."
            print(f"     Gap trend p:             {tb['gap_p']:.6f}  {sig}")
            if tb['gap_slope'] > 0:
                print(f"     → PAC violation GROWS over generation")
            else:
                print(f"     → PAC violation SHRINKS (model self-corrects)")

        # Store
        all_results[model_name] = {
            'architecture': arch_info,
            'metrics': metrics,
        }

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # ── CROSS-MODEL SYNTHESIS ──
    print(f"\n{'='*70}")
    print(f"  CROSS-MODEL PAC CONSERVATION SYNTHESIS")
    print(f"{'='*70}")

    print(f"\n  SYSTEM BUDGET VIOLATION:")
    print(f"  {'Model':17s} {'Family':7s} {'F_budget':>9s} {'H_budget':>9s} "
          f"{'Violation%':>10s} {'p':>10s} {'Sig':>5s}")
    for mn, _, _ in MODELS:
        mr = all_results[mn]
        cl = mr['metrics']['cross_layer']
        sig = "***" if cl['budget_p'] < 0.001 else "**" if cl['budget_p'] < 0.01 else "*" if cl['budget_p'] < 0.05 else "n.s."
        family = mr['architecture']['family']
        print(f"  {mn:17s} {family:7s} {cl['factual_system_budget']:9.1f} "
              f"{cl['halluc_system_budget']:9.1f} {cl['budget_violation_pct']:+9.1f}% "
              f"{cl['budget_p']:10.6f} {sig}")

    print(f"\n  COMPENSATION RATIO (1.0 = perfect PAC):")
    print(f"  {'Model':17s} {'Family':7s} {'Gained':>8s} {'Compensated':>11s} "
          f"{'Net':>8s} {'Ratio':>7s}")
    for mn, _, _ in MODELS:
        mr = all_results[mn]
        comp = mr['metrics']['compensation']
        family = mr['architecture']['family']
        print(f"  {mn:17s} {family:7s} {comp['net_positive_violation']:+7.2f} "
              f"{comp['net_negative_compensation']:+10.2f} "
              f"{comp['net_uncompensated']:+7.2f} {comp['compensation_ratio']:7.3f}")

    print(f"\n  HEAD REDISTRIBUTION:")
    print(f"  {'Model':17s} {'Total':>6s} {'Increase':>8s} {'Decrease':>8s} "
          f"{'Stable':>7s} {'Net ΔH':>8s}")
    for mn, _, _ in MODELS:
        mr = all_results[mn]
        hr = mr['metrics']['head_redistribution']
        total = mr['architecture']['layers'] * mr['architecture']['heads']
        family = mr['architecture']['family']
        print(f"  {mn:17s} {total:6d} {hr['heads_increase']:8d} "
              f"{hr['heads_decrease']:8d} {hr['heads_stable']:7d} "
              f"{hr['net_head_violation']:+7.2f}")

    print(f"\n  TEMPORAL PAC VIOLATION TREND:")
    print(f"  {'Model':17s} {'Early_gap':>10s} {'Late_gap':>10s} "
          f"{'Trend':>10s} {'p':>10s}")
    for mn, _, _ in MODELS:
        mr = all_results[mn]
        tb = mr['metrics']['temporal_budget']
        family = mr['architecture']['family']
        if tb.get('gap_slope') is not None:
            trend = "GROWING" if tb['gap_slope'] > 0 else "SHRINKING"
            print(f"  {mn:17s} {tb['early_gap']:+9.2f} {tb['late_gap']:+9.2f} "
                  f"{trend:>10s} {tb['gap_p']:10.6f}")

    print(f"\n  RESIDUAL NORM RATIOS (halluc/factual):")
    print(f"  {'Model':17s} {'Norm_ratio':>10s}")
    for mn, _, _ in MODELS:
        mr = all_results[mn]
        rn = mr['metrics']['residual_norms']
        family = mr['architecture']['family']
        print(f"  {mn:17s} {rn['norm_ratio']:10.4f}")

    # ── VERDICT ──
    print(f"\n  {'='*55}")
    all_violations = [all_results[mn]['metrics']['cross_layer']['budget_violation_pct']
                      for mn, _, _ in MODELS]
    all_comp_ratios = [all_results[mn]['metrics']['compensation']['compensation_ratio']
                       for mn, _, _ in MODELS]

    mean_violation = np.mean(all_violations)
    mean_comp = np.mean(all_comp_ratios)

    if mean_violation > 5 and mean_comp < 0.5:
        verdict = "PAC VIOLATION: Hallucination creates uncompensated entropy"
    elif mean_violation > 2 and mean_comp < 0.8:
        verdict = "PARTIAL PAC VIOLATION: Some compensation, but net entropy increases"
    elif mean_comp > 0.8:
        verdict = "PAC CONSERVED: Hallucination is redistribution, not creation"
    else:
        verdict = "MIXED: Conservation varies by architecture"

    print(f"  VERDICT: {verdict}")
    print(f"  Mean system violation: {mean_violation:+.1f}%")
    print(f"  Mean compensation ratio: {mean_comp:.3f}")
    print(f"  {'='*55}")

    # ── Save ──
    output = {
        'experiment': 'exp_12_pac_conservation',
        'timestamp': datetime.now().isoformat(),
        'n_tokens': N_TOKENS,
        'n_factual': len(FACTUAL),
        'n_hallucination': len(HALLUCINATION),
        'models': {},
        'cross_model': {
            'mean_violation_pct': float(mean_violation),
            'mean_compensation_ratio': float(mean_comp),
            'verdict': verdict,
        },
        'elapsed_seconds': time.time() - t0,
    }

    for mn, _, _ in MODELS:
        output['models'][mn] = all_results[mn]

    results_dir = EXPERIMENT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"exp_12_conservation_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
