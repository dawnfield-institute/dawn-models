#!/usr/bin/env python3
"""
EXP 05: Weight PAC Tree — Internal Activation Monitoring
=========================================================

Prior experiments (01–04) analysed the OUTPUT of collapse (logits/softmax).
This experiment looks INSIDE the model: builds a PAC tree from the actual
weight matrices and monitors how activations flow through it during inference.

Core idea:
  - Each layer's weight matrix encodes "potential" (what the layer CAN do)
  - During inference, the activation vector selects a subset of that potential
  - This IS a PAC collapse: f(layer_potential) → f(activated_subset)

Design:
  1. For each layer, SVD-decompose the weight matrix: W = UΣV^T
     - Singular values Σ = the PAC "children" (ranked potential)
     - Each σ_i represents one structural mode of the layer
  2. During inference, project activations onto the right-singular vectors V
     - Activation projection α_i = |x · v_i| gives "how much mode i fires"
     - The ratio α_1/α_2 is the INTERNAL PAC ratio (cf. logit ratio p1/p2)
  3. Track these internal PAC ratios layer-by-layer for each token
     - Does the internal ratio predict correctness?
     - Does it correlate with the output-level SEC phase?
     - Does it differ for factual vs hallucinated generation?

This bridges the gap between:
  - POC-016/017/020 (static weight trees) and
  - exp_01-04 (output-level PAC analysis)

Author: Dawn Field Institute
Date: 2026-02-13
"""

import sys, os, json, time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
from scipy import stats

# --- path setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR))
from core.collapse_metrics import classify_sec_phase, SECPhase

# --- constants ---
PHI = (1 + 5**0.5) / 2
INV_PHI = PHI - 1
XI = 1 + np.pi / 55
TOP_K_MODES = 10          # Top singular value modes to track
CACHE_DIR = str(Path(EXPERIMENT_DIR).parent / "huggingface_bifractal_validation" / "pythia_cache")


# ── Weight PAC Tree ───────────────────────────────────────────────

class WeightPACNode:
    """One singular mode of a weight matrix."""
    __slots__ = ['mode_index', 'singular_value', 'right_vector', 'left_vector',
                 'relative_weight', 'cumulative_energy']
    
    def __init__(self, mode_index, singular_value, right_vector, left_vector,
                 relative_weight, cumulative_energy):
        self.mode_index = mode_index
        self.singular_value = singular_value
        self.right_vector = right_vector     # v_i — projects input
        self.left_vector = left_vector       # u_i — produces output
        self.relative_weight = relative_weight  # σ_i / Σσ
        self.cumulative_energy = cumulative_energy  # Σ(σ_1..σ_i)² / Σσ²


class LayerWeightPAC:
    """PAC tree built from one layer's weight matrix via SVD."""
    
    def __init__(self, layer_name, weight_matrix, top_k=TOP_K_MODES):
        self.layer_name = layer_name
        self.top_k = top_k
        
        # SVD decomposition
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(weight_matrix.float(), full_matrices=False)
        
        self.singular_values = S.cpu().numpy()
        self.total_energy = float(np.sum(self.singular_values ** 2))
        self.total_weight = float(np.sum(self.singular_values))
        
        # Build PAC children (top-k modes)
        cumulative_energy = 0.0
        self.modes = []
        for i in range(min(top_k, len(S))):
            sv = float(S[i])
            cumulative_energy += sv ** 2
            
            node = WeightPACNode(
                mode_index=i,
                singular_value=sv,
                right_vector=Vh[i].cpu(),    # input projection
                left_vector=U[:, i].cpu(),    # output projection
                relative_weight=sv / self.total_weight,
                cumulative_energy=cumulative_energy / self.total_energy,
            )
            self.modes.append(node)
        
        # PAC ratios between successive modes
        self.weight_pac_ratios = []
        for i in range(len(self.modes) - 1):
            r = self.modes[i].singular_value / max(self.modes[i+1].singular_value, 1e-10)
            self.weight_pac_ratios.append(r)
        
        # Entropy of singular value distribution
        probs = self.singular_values / max(self.total_weight, 1e-10)
        probs = probs[probs > 0]
        self.weight_entropy = float(-np.sum(probs * np.log(probs + 1e-30)))
        
        # Effective rank (exp of entropy)
        self.effective_rank = float(np.exp(self.weight_entropy))
    
    def project_activation(self, activation):
        """
        Project an activation vector onto the SVD modes.
        Returns mode activations α_i = |x · v_i| for each mode.
        """
        # activation: [hidden_dim]
        with torch.no_grad():
            projections = []
            for mode in self.modes:
                # How much does this activation align with this mode?
                v = mode.right_vector.to(activation.device)
                # Handle dimension mismatch
                if v.shape[0] != activation.shape[-1]:
                    return None  # Skip if dimensions don't match
                proj = torch.abs(torch.dot(activation.float().flatten()[:v.shape[0]], v.float()))
                projections.append(float(proj))
            return projections
    
    def compute_activation_pac(self, activation):
        """
        Compute PAC diagnostics for how an activation uses this layer's modes.
        
        Returns dict with:
          - mode_activations: raw projections
          - activation_pac_ratios: α_i/α_{i+1}
          - activation_entropy: H(normalised projections)
          - concentration: α_1 / sum(α)
          - dominant_mode: which mode fires most
          - activation_vs_weight_alignment: do activations follow weight structure?
        """
        projections = self.project_activation(activation)
        if projections is None:
            return None
            
        total = sum(projections) + 1e-30
        normed = [p / total for p in projections]
        
        # Activation PAC ratios
        act_ratios = []
        for i in range(len(projections) - 1):
            r = projections[i] / max(projections[i+1], 1e-30)
            act_ratios.append(r)
        
        # Activation entropy
        act_entropy = -sum(p * np.log(p + 1e-30) for p in normed if p > 0)
        
        # Concentration: how much does top mode dominate?
        concentration = projections[0] / total if total > 1e-30 else 0
        
        # Dominant mode
        dominant = int(np.argmax(projections))
        
        # Alignment: correlation between weight structure and activation structure
        weight_profile = [m.relative_weight for m in self.modes]
        if len(weight_profile) >= 3 and len(normed) >= 3:
            corr, _ = stats.spearmanr(weight_profile, normed)
        else:
            corr = 0.0
        
        return {
            'mode_activations': projections,
            'activation_pac_ratios': act_ratios,
            'activation_entropy': act_entropy,
            'concentration': concentration,
            'dominant_mode': dominant,
            'weight_activation_alignment': float(corr) if not np.isnan(corr) else 0.0,
        }


class ModelWeightPAC:
    """Full model weight PAC tree — one LayerWeightPAC per MLP/attention layer."""
    
    def __init__(self, model, model_name="pythia"):
        self.model_name = model_name
        self.layer_pacs = {}
        
        print(f"  Building weight PAC tree from {model_name}...")
        
        # Extract MLP and attention weight matrices
        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue  # skip biases, norms
            
            # Focus on the key projection matrices
            is_mlp = 'mlp' in name and ('dense' in name or 'fc' in name)
            is_attn = ('attention' in name or 'attn' in name) and ('query' in name or 'key' in name or 'value' in name or 'dense' in name or 'proj' in name)
            
            if is_mlp or is_attn:
                layer_pac = LayerWeightPAC(name, param.data)
                self.layer_pacs[name] = layer_pac
        
        print(f"  Built PAC tree for {len(self.layer_pacs)} weight matrices")
        
        # Aggregate statistics
        if self.layer_pacs:
            all_ratios = []
            for lp in self.layer_pacs.values():
                all_ratios.extend(lp.weight_pac_ratios)
            
            self.weight_ratio_mean = float(np.mean(all_ratios)) if all_ratios else 0
            self.weight_ratio_median = float(np.median(all_ratios)) if all_ratios else 0
            
            near_phi = sum(1 for r in all_ratios if abs(r - PHI) / PHI < 0.05)
            self.weight_phi_rate = near_phi / len(all_ratios) if all_ratios else 0
            
            print(f"  Weight PAC ratio median: {self.weight_ratio_median:.4f}")
            print(f"  Weight phi-alignment rate: {self.weight_phi_rate*100:.1f}%")
            print(f"  Weight SVD entropy (mean): {np.mean([lp.weight_entropy for lp in self.layer_pacs.values()]):.2f}")
    
    def get_summary(self):
        """Summary statistics for the static weight PAC tree."""
        summaries = {}
        for name, lp in self.layer_pacs.items():
            summaries[name] = {
                'singular_values_top5': [float(m.singular_value) for m in lp.modes[:5]],
                'weight_pac_ratios': lp.weight_pac_ratios,
                'weight_entropy': lp.weight_entropy,
                'effective_rank': lp.effective_rank,
                'top_mode_energy': lp.modes[0].cumulative_energy if lp.modes else 0,
                'top3_energy': lp.modes[2].cumulative_energy if len(lp.modes) > 2 else 0,
            }
        return summaries


# ── Inference Hooks ───────────────────────────────────────────────

class ActivationCapture:
    """Captures activations at key points during forward pass."""
    
    def __init__(self, model, weight_pac):
        self.model = model
        self.weight_pac = weight_pac
        self.hooks = []
        self.captured = {}
        
        # Register hooks on layers we have PAC trees for
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        for name, module in self.model.named_modules():
            # Match module name to weight PAC layer names
            for pac_name in self.weight_pac.layer_pacs:
                # The parameter name is like "gpt_neox.layers.0.mlp.dense_h_to_4h.weight"
                # The module name is like "gpt_neox.layers.0.mlp.dense_h_to_4h"
                module_from_param = pac_name.rsplit('.weight', 1)[0]
                if name == module_from_param:
                    hook = module.register_forward_hook(
                        self._make_hook(pac_name)
                    )
                    self.hooks.append(hook)
                    break
    
    def _make_hook(self, layer_name):
        def hook_fn(module, input, output):
            # Capture the INPUT to the layer (pre-transformation)
            if isinstance(input, tuple):
                act = input[0]
            else:
                act = input
            
            # Take the last token position's activation
            if act.dim() == 3:  # [batch, seq, hidden]
                act = act[0, -1, :]  # last token
            elif act.dim() == 2:  # [batch, hidden]
                act = act[0, :]
            
            self.captured[layer_name] = act.detach()
        return hook_fn
    
    def clear(self):
        self.captured = {}
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def get_layer_pac_diagnostics(self):
        """
        For each captured activation, project onto the weight PAC tree
        and compute activation PAC diagnostics.
        """
        diagnostics = {}
        for layer_name, activation in self.captured.items():
            if layer_name in self.weight_pac.layer_pacs:
                lp = self.weight_pac.layer_pacs[layer_name]
                diag = lp.compute_activation_pac(activation)
                if diag is not None:
                    diagnostics[layer_name] = diag
        return diagnostics


# ── Prompts ───────────────────────────────────────────────────────

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
]

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
]


# ── Main Experiment ───────────────────────────────────────────────

def run_inference_with_monitoring(model, tokenizer, device, weight_pac, prompt, 
                                  n_tokens=20):
    """
    Generate tokens while monitoring internal PAC activation patterns.
    """
    capture = ActivationCapture(model, weight_pac)
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    current_ids = input_ids
    
    token_diagnostics = []
    
    try:
        for step in range(n_tokens):
            capture.clear()
            
            with torch.no_grad():
                outputs = model(current_ids)
                logits = outputs.logits[0, -1, :]
            
            # Get layer-by-layer PAC diagnostics
            layer_diags = capture.get_layer_pac_diagnostics()
            
            # Logit-level info (from exp_01-04)
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_ids = torch.topk(probs, 10)
            chosen_id = int(logits.argmax())
            chosen_prob = float(probs[chosen_id])
            
            logit_entropy = float(-torch.sum(probs * torch.log(probs + 1e-30)))
            
            if len(top_probs) >= 2:
                logit_pac_ratio = float(top_probs[0] / top_probs[1])
            else:
                logit_pac_ratio = float('inf')
            
            sec_phase = classify_sec_phase(logit_entropy)
            
            # Aggregate internal PAC ratios across layers
            internal_ratios_1_2 = []
            internal_concentrations = []
            internal_entropies = []
            internal_alignments = []
            
            for lname, diag in layer_diags.items():
                if diag['activation_pac_ratios']:
                    internal_ratios_1_2.append(diag['activation_pac_ratios'][0])
                internal_concentrations.append(diag['concentration'])
                internal_entropies.append(diag['activation_entropy'])
                internal_alignments.append(diag['weight_activation_alignment'])
            
            token_data = {
                'step': step,
                'token_id': chosen_id,
                'token_text': tokenizer.decode([chosen_id]),
                'chosen_prob': chosen_prob,
                'logit_entropy': logit_entropy,
                'logit_pac_ratio': logit_pac_ratio,
                'sec_phase': sec_phase.name,
                # Internal PAC aggregates
                'n_layers_captured': len(layer_diags),
                'internal_ratio_mean': float(np.mean(internal_ratios_1_2)) if internal_ratios_1_2 else 0,
                'internal_ratio_median': float(np.median(internal_ratios_1_2)) if internal_ratios_1_2 else 0,
                'internal_concentration_mean': float(np.mean(internal_concentrations)) if internal_concentrations else 0,
                'internal_entropy_mean': float(np.mean(internal_entropies)) if internal_entropies else 0,
                'internal_alignment_mean': float(np.mean(internal_alignments)) if internal_alignments else 0,
                # Per-layer detail (for deep analysis)
                'per_layer': {
                    lname: {
                        'act_ratio_1_2': diag['activation_pac_ratios'][0] if diag['activation_pac_ratios'] else None,
                        'concentration': diag['concentration'],
                        'entropy': diag['activation_entropy'],
                        'alignment': diag['weight_activation_alignment'],
                        'dominant_mode': diag['dominant_mode'],
                    }
                    for lname, diag in layer_diags.items()
                },
            }
            
            token_diagnostics.append(token_data)
            
            # Autoregressive step
            chosen_tensor = torch.tensor([[chosen_id]], device=device)
            current_ids = torch.cat([current_ids, chosen_tensor], dim=1)
    finally:
        capture.remove_hooks()
    
    generated_text = tokenizer.decode([td['token_id'] for td in token_diagnostics])
    
    # Sequence-level aggregates
    all_internal_ratios = [td['internal_ratio_mean'] for td in token_diagnostics if td['internal_ratio_mean'] > 0]
    all_logit_ratios = [td['logit_pac_ratio'] for td in token_diagnostics if td['logit_pac_ratio'] < 1e6]
    all_alignments = [td['internal_alignment_mean'] for td in token_diagnostics]
    
    # Key question: does internal ratio predict output SEC phase?
    phase_internal_ratios = defaultdict(list)
    for td in token_diagnostics:
        if td['internal_ratio_mean'] > 0:
            phase_internal_ratios[td['sec_phase']].append(td['internal_ratio_mean'])
    
    # Correlation between internal and output PAC ratios
    if len(all_internal_ratios) >= 3 and len(all_logit_ratios) >= 3:
        min_len = min(len(all_internal_ratios), len(all_logit_ratios))
        internal_vs_output_corr, internal_vs_output_p = stats.spearmanr(
            all_internal_ratios[:min_len], all_logit_ratios[:min_len]
        )
    else:
        internal_vs_output_corr = 0.0
        internal_vs_output_p = 1.0
    
    # Entropy trajectory of internal activations
    int_entropies = [td['internal_entropy_mean'] for td in token_diagnostics if td['internal_entropy_mean'] > 0]
    if len(int_entropies) >= 3:
        x = np.arange(len(int_entropies))
        int_entropy_slope = float(stats.linregress(x, int_entropies).slope)
    else:
        int_entropy_slope = 0.0
    
    return {
        'prompt': prompt,
        'generated_text': generated_text,
        'n_tokens': len(token_diagnostics),
        'tokens': token_diagnostics,
        # Sequence-level
        'internal_ratio_mean': float(np.mean(all_internal_ratios)) if all_internal_ratios else 0,
        'internal_ratio_median': float(np.median(all_internal_ratios)) if all_internal_ratios else 0,
        'internal_entropy_slope': int_entropy_slope,
        'internal_alignment_mean': float(np.mean(all_alignments)) if all_alignments else 0,
        'internal_vs_output_corr': float(internal_vs_output_corr) if not np.isnan(internal_vs_output_corr) else 0,
        'internal_vs_output_p': float(internal_vs_output_p) if not np.isnan(internal_vs_output_p) else 1,
        'phase_internal_ratios': {
            phase: {
                'mean': float(np.mean(ratios)),
                'median': float(np.median(ratios)),
                'count': len(ratios),
            }
            for phase, ratios in phase_internal_ratios.items()
        },
    }


def compare_groups(group_a, group_b, label_a="Factual", label_b="Hallucinated"):
    """Compare internal PAC metrics between groups."""
    metrics = [
        'internal_ratio_mean',
        'internal_ratio_median',
        'internal_entropy_slope',
        'internal_alignment_mean',
        'internal_vs_output_corr',
    ]
    
    comparison = {}
    significant = []
    
    print(f"\n{'='*70}")
    print(f"  INTERNAL PAC: {label_a} vs {label_b}")
    print(f"{'='*70}")
    
    for metric in metrics:
        vals_a = [r[metric] for r in group_a]
        vals_b = [r[metric] for r in group_b]
        
        mean_a, std_a = np.mean(vals_a), np.std(vals_a)
        mean_b, std_b = np.mean(vals_b), np.std(vals_b)
        
        try:
            u_stat, p_mw = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')
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
    
    # Also compare token-level internal ratios vs logit ratios
    all_int = []
    all_log = []
    for r in group_a + group_b:
        for td in r['tokens']:
            if td['internal_ratio_mean'] > 0 and td['logit_pac_ratio'] < 1e6:
                all_int.append(td['internal_ratio_mean'])
                all_log.append(td['logit_pac_ratio'])
    
    if len(all_int) >= 10:
        corr, p = stats.spearmanr(all_int, all_log)
        print(f"\n  INTERNAL vs OUTPUT PAC ratio (all tokens):")
        print(f"    Spearman r = {corr:.4f}, p = {p:.6f}")
        comparison['global_internal_vs_output'] = {
            'spearman_r': float(corr), 'p_value': float(p), 'n_tokens': len(all_int)
        }
    
    if significant:
        print(f"\n  SIGNIFICANT (p<0.05): {', '.join(significant)}")
    else:
        print(f"\n  No individually significant metrics at p<0.05")
    
    return comparison, significant


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("=" * 70)
    print("  EXP 05: Weight PAC Tree — Internal Activation Monitoring")
    print("  Build PAC tree from weight SVD, monitor activations during inference")
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
    
    # ── Step 1: Build weight PAC tree (static) ──
    print(f"\n{'='*60}")
    print(f"  STEP 1: Build Weight PAC Tree (SVD)")
    print(f"{'='*60}")
    
    weight_pac = ModelWeightPAC(model, model_name)
    
    # Print weight PAC structure
    print(f"\n  Weight PAC structure:")
    for name, lp in list(weight_pac.layer_pacs.items())[:6]:
        svs = [f"{m.singular_value:.1f}" for m in lp.modes[:5]]
        ratios = [f"{r:.3f}" for r in lp.weight_pac_ratios[:3]]
        print(f"    {name[:60]:60s}  σ=[{','.join(svs)}...]  ratio=[{','.join(ratios)}...]  H={lp.weight_entropy:.2f}")
    if len(weight_pac.layer_pacs) > 6:
        print(f"    ... and {len(weight_pac.layer_pacs) - 6} more layers")
    
    # ── Step 2: Check weight PAC ratios against phi ──
    print(f"\n{'='*60}")
    print(f"  STEP 2: Weight PAC Ratio Analysis")
    print(f"{'='*60}")
    
    all_weight_ratios = []
    per_layer_data = {}
    for name, lp in weight_pac.layer_pacs.items():
        all_weight_ratios.extend(lp.weight_pac_ratios)
        per_layer_data[name] = {
            'ratios': lp.weight_pac_ratios,
            'entropy': lp.weight_entropy,
            'effective_rank': lp.effective_rank,
            'top_mode_energy': lp.modes[0].cumulative_energy if lp.modes else 0,
        }
    
    if all_weight_ratios:
        near_phi = sum(1 for r in all_weight_ratios if abs(r - PHI) / PHI < 0.05)
        near_inv_phi = sum(1 for r in all_weight_ratios if abs(r - INV_PHI) / INV_PHI < 0.05)
        near_xi = sum(1 for r in all_weight_ratios if abs(r - XI) / XI < 0.05)
        
        print(f"  Total weight PAC ratios: {len(all_weight_ratios)}")
        print(f"  Mean: {np.mean(all_weight_ratios):.4f}")
        print(f"  Median: {np.median(all_weight_ratios):.4f}")
        print(f"  Near phi ({PHI:.3f}): {near_phi} ({near_phi/len(all_weight_ratios)*100:.1f}%)")
        print(f"  Near 1/phi ({INV_PHI:.3f}): {near_inv_phi} ({near_inv_phi/len(all_weight_ratios)*100:.1f}%)")
        print(f"  Near xi ({XI:.3f}): {near_xi} ({near_xi/len(all_weight_ratios)*100:.1f}%)")
        
        # Histogram of weight ratios
        hist, edges = np.histogram(all_weight_ratios, bins=20)
        print(f"\n  Weight ratio distribution:")
        max_count = max(hist)
        for i, count in enumerate(hist):
            bar = '█' * int(count / max(max_count, 1) * 30)
            center = (edges[i] + edges[i+1]) / 2
            phi_marker = " ←φ" if abs(center - PHI) < (edges[1]-edges[0]) else ""
            print(f"    {center:6.3f}: {bar} ({count}){phi_marker}")
    
    # ── Step 3: Run inference with monitoring ──
    print(f"\n{'='*60}")
    print(f"  STEP 3: Inference with Internal PAC Monitoring")
    print(f"{'='*60}")
    
    all_results = {}
    
    for group_name, prompts in [("factual", FACTUAL_PROMPTS), ("hallucination", HALLUCINATION_PROMPTS)]:
        print(f"\n  GROUP: {group_name} ({len(prompts)} prompts)")
        group_results = []
        
        for i, prompt in enumerate(prompts):
            result = run_inference_with_monitoring(model, tokenizer, device, weight_pac, prompt)
            group_results.append(result)
            
            prompt_short = prompt[:40] + "..." if len(prompt) > 40 else prompt
            corr_str = f"r={result['internal_vs_output_corr']:+.2f}" if result['internal_vs_output_corr'] != 0 else "r=N/A"
            print(f"    [{i+1:2d}/{len(prompts)}] int_ratio={result['internal_ratio_median']:.3f}  "
                  f"align={result['internal_alignment_mean']:.3f}  "
                  f"{corr_str}  '{prompt_short}'")
        
        all_results[group_name] = group_results
    
    # ── Step 4: Compare groups ──
    comparison, sig_metrics = compare_groups(
        all_results['factual'], all_results['hallucination']
    )
    
    # ── Step 5: Internal ratio by SEC phase ──
    print(f"\n{'='*60}")
    print(f"  STEP 5: Internal PAC Ratio by SEC Phase")
    print(f"{'='*60}")
    
    phase_data = defaultdict(list)
    for group in all_results.values():
        for r in group:
            for td in r['tokens']:
                if td['internal_ratio_mean'] > 0:
                    phase_data[td['sec_phase']].append(td['internal_ratio_mean'])
    
    for phase in ['CRYSTALLIZED', 'ORDERED', 'TRANSITIONAL', 'CHAOTIC']:
        ratios = phase_data.get(phase, [])
        if ratios:
            print(f"  {phase:15s}: n={len(ratios):3d}  mean={np.mean(ratios):.4f}  median={np.median(ratios):.4f}")
        else:
            print(f"  {phase:15s}: n=  0")
    
    # ── Save results ──
    output = {
        'experiment': 'exp_05_weight_pac_activation',
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
        'top_k_modes': TOP_K_MODES,
        'n_tokens_per_prompt': 20,
        'weight_pac_summary': {
            'n_layers': len(weight_pac.layer_pacs),
            'ratio_mean': weight_pac.weight_ratio_mean,
            'ratio_median': weight_pac.weight_ratio_median,
            'phi_rate': weight_pac.weight_phi_rate,
            'total_weight_ratios': len(all_weight_ratios),
        },
        'groups': {
            gname: {
                'n_prompts': len(results),
                'summary': {
                    'mean_internal_ratio': float(np.mean([r['internal_ratio_mean'] for r in results])),
                    'mean_internal_entropy_slope': float(np.mean([r['internal_entropy_slope'] for r in results])),
                    'mean_alignment': float(np.mean([r['internal_alignment_mean'] for r in results])),
                    'mean_corr': float(np.mean([r['internal_vs_output_corr'] for r in results])),
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
        'phase_internal_ratios': {
            phase: {
                'mean': float(np.mean(ratios)),
                'median': float(np.median(ratios)),
                'count': len(ratios),
            }
            for phase, ratios in phase_data.items()
            if ratios
        },
        'elapsed_seconds': time.time() - t0,
    }
    
    results_dir = EXPERIMENT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"exp_05_weight_pac_{ts}.json"
    
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
