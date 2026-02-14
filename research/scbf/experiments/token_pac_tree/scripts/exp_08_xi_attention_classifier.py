#!/usr/bin/env python3
"""
EXP 08: Xi Attention Classifier — Multi-Model Scaling + Hallucination Detection
================================================================================

exp_07 showed:
  - Attention entropy orbits Xi (1.057) as a balance point
  - Factual text pulls entropy below Xi, hallucinated pushes above
  - 5 metrics significant at p<0.05, 3 at p<0.001

This experiment:
  1. Scales attention PAC across all 4 Pythia models (70m → 1B)
  2. Tests whether Xi as balance point holds at every scale
  3. Builds a zero-parameter hallucination classifier:
     "Is mean attention entropy above or below Xi?"
  4. Reports precision/recall/F1 per model
  5. Tests multi-feature classifier using attention metrics
  6. Runs threshold sweep to confirm Xi is optimal (or find what is)

Author: Dawn Field Institute
Date: 2026-02-14
"""

import sys, json, time, gc
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

MODELS = [
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
]

# Expanded prompt sets for better classifier evaluation
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


def compute_attention_pac(attn_weights):
    """Compute PAC diagnostics from attention weights for one head."""
    sorted_attn = np.sort(attn_weights)[::-1]
    
    if len(sorted_attn) >= 2 and sorted_attn[1] > 1e-10:
        pac_ratio = float(sorted_attn[0] / sorted_attn[1])
    else:
        pac_ratio = float('inf')
    
    attn_pos = attn_weights[attn_weights > 1e-10]
    if len(attn_pos) > 0:
        entropy = float(-np.sum(attn_pos * np.log(attn_pos)))
    else:
        entropy = 0.0
    
    concentration = float(sorted_attn[0]) if len(sorted_attn) > 0 else 0
    sec_phase = classify_sec_phase(entropy)
    
    return {
        'pac_ratio': pac_ratio,
        'entropy': entropy,
        'concentration': concentration,
        'sec_phase': sec_phase.name,
    }


def run_attention_monitoring(model, tokenizer, device, prompt, n_tokens=20):
    """Generate tokens while capturing attention weights."""
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
        chosen_prob = float(probs[chosen_id])
        logit_entropy = float(-torch.sum(probs * torch.log(probs + 1e-30)))
        
        top2 = torch.topk(probs, 2)
        logit_pac_ratio = float(top2.values[0] / top2.values[1]) if top2.values[1] > 1e-10 else float('inf')
        
        # Per-head attention analysis
        phase_counts = defaultdict(int)
        head_entropies = []
        head_concentrations = []
        
        for layer_attn in attentions:
            n_heads = layer_attn.shape[1]
            for head_idx in range(n_heads):
                attn_vec = layer_attn[0, head_idx, -1, :].cpu().numpy()
                head_pac = compute_attention_pac(attn_vec)
                phase_counts[head_pac['sec_phase']] += 1
                head_entropies.append(head_pac['entropy'])
                head_concentrations.append(head_pac['concentration'])
        
        total_heads = sum(phase_counts.values())
        crystallized = phase_counts.get('CRYSTALLIZED', 0)
        ordered = phase_counts.get('ORDERED', 0)
        
        confident_head_ratio = (crystallized + ordered) / total_heads if total_heads > 0 else 0
        
        # Layer-wise entropy for depth slope
        layer_ents = defaultdict(list)
        layer_idx = 0
        for layer_attn in attentions:
            n_heads = layer_attn.shape[1]
            for head_idx in range(n_heads):
                attn_vec = layer_attn[0, head_idx, -1, :].cpu().numpy()
                attn_pos = attn_vec[attn_vec > 1e-10]
                h = float(-np.sum(attn_pos * np.log(attn_pos))) if len(attn_pos) > 0 else 0.0
                layer_ents[layer_idx].append(h)
            layer_idx += 1
        
        layer_means = [np.mean(layer_ents[l]) for l in sorted(layer_ents.keys())]
        if len(layer_means) >= 3:
            x = np.arange(len(layer_means))
            depth_slope = float(stats.linregress(x, layer_means).slope)
        else:
            depth_slope = 0.0
        
        td = {
            'step': step,
            'chosen_prob': chosen_prob,
            'logit_entropy': logit_entropy,
            'logit_pac_ratio': logit_pac_ratio,
            'confident_head_ratio': confident_head_ratio,
            'attn_entropy_mean': float(np.mean(head_entropies)),
            'attn_concentration_mean': float(np.mean(head_concentrations)),
            'depth_slope': depth_slope,
        }
        token_data.append(td)
        
        chosen_tensor = torch.tensor([[chosen_id]], device=device)
        current_ids = torch.cat([current_ids, chosen_tensor], dim=1)
    
    # Sequence-level aggregates
    attn_ents = [td['attn_entropy_mean'] for td in token_data]
    conf_ratios = [td['confident_head_ratio'] for td in token_data]
    depth_slopes = [td['depth_slope'] for td in token_data]
    
    if len(attn_ents) >= 3:
        x = np.arange(len(attn_ents))
        seq_entropy_slope = float(stats.linregress(x, attn_ents).slope)
        seq_conf_slope = float(stats.linregress(x, conf_ratios).slope)
    else:
        seq_entropy_slope = 0.0
        seq_conf_slope = 0.0
    
    return {
        'prompt': prompt,
        'mean_attn_entropy': float(np.mean(attn_ents)),
        'mean_confident_head_ratio': float(np.mean(conf_ratios)),
        'mean_depth_slope': float(np.mean(depth_slopes)),
        'seq_entropy_slope': seq_entropy_slope,
        'seq_conf_slope': seq_conf_slope,
        'token_attn_entropies': attn_ents,  # for trajectory analysis
    }


def xi_classifier(results, threshold=None):
    """
    Binary classifier: is mean attention entropy above or below threshold?
    Below threshold → predict "factual" (label=0)
    Above threshold → predict "hallucinated" (label=1)
    
    Returns precision, recall, F1 for hallucination detection.
    """
    if threshold is None:
        threshold = XI
    
    y_true = []
    y_pred = []
    scores = []
    
    for r in results:
        true_label = r['true_label']  # 0=factual, 1=hallucination
        pred_label = 1 if r['mean_attn_entropy'] > threshold else 0
        y_true.append(true_label)
        y_pred.append(pred_label)
        scores.append(r['mean_attn_entropy'])
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Hallucination detection metrics (positive = hallucinated)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true)
    
    return {
        'threshold': float(threshold),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
    }


def threshold_sweep(results, n_steps=200):
    """Sweep thresholds to find optimal and compare with Xi."""
    entropies = [r['mean_attn_entropy'] for r in results]
    lo, hi = min(entropies) - 0.01, max(entropies) + 0.01
    thresholds = np.linspace(lo, hi, n_steps)
    
    best_f1 = 0
    best_thresh = 0
    all_results = []
    
    for t in thresholds:
        cr = xi_classifier(results, threshold=float(t))
        all_results.append(cr)
        if cr['f1'] > best_f1:
            best_f1 = cr['f1']
            best_thresh = float(t)
    
    # Xi-specific result
    xi_result = xi_classifier(results, threshold=XI)
    
    return {
        'best_threshold': best_thresh,
        'best_f1': best_f1,
        'xi_threshold': XI,
        'xi_f1': xi_result['f1'],
        'xi_accuracy': xi_result['accuracy'],
        'xi_distance_from_optimal': abs(best_thresh - XI),
        'sweep': [(r['threshold'], r['f1'], r['accuracy']) for r in all_results],
    }


def multi_feature_classifier(results):
    """
    Test multiple attention metrics as classifiers.
    For each metric, find optimal threshold and compare with Xi-entropy.
    """
    metrics = [
        ('mean_attn_entropy', 'above'),       # higher → hallucination
        ('mean_confident_head_ratio', 'below'), # lower → hallucination
        ('seq_entropy_slope', 'below'),         # lower slope → hallucination
        ('seq_conf_slope', 'above'),            # higher (less negative) → hallucination
        ('mean_depth_slope', 'below'),          # more negative → hallucination
    ]
    
    results_out = {}
    
    for metric_name, direction in metrics:
        vals = [r[metric_name] for r in results]
        labels = [r['true_label'] for r in results]
        
        lo, hi = min(vals) - 0.001, max(vals) + 0.001
        thresholds = np.linspace(lo, hi, 200)
        
        best_f1 = 0
        best_thresh = 0
        
        for t in thresholds:
            if direction == 'above':
                preds = [1 if v > t else 0 for v in vals]
            else:
                preds = [1 if v < t else 0 for v in vals]
            
            tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
            fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
            fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
            tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            acc = (tp + tn) / len(labels)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(t)
                best_metrics = {'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc}
        
        # Mann-Whitney
        factual_vals = [v for v, l in zip(vals, labels) if l == 0]
        halluc_vals = [v for v, l in zip(vals, labels) if l == 1]
        try:
            _, p_mw = stats.mannwhitneyu(factual_vals, halluc_vals, alternative='two-sided')
        except ValueError:
            p_mw = 1.0
        
        results_out[metric_name] = {
            'best_threshold': best_thresh,
            'best_f1': best_f1,
            'best_accuracy': best_metrics['accuracy'],
            'factual_mean': float(np.mean(factual_vals)),
            'halluc_mean': float(np.mean(halluc_vals)),
            'mann_whitney_p': float(p_mw),
            'direction': direction,
        }
    
    return results_out


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("=" * 70)
    print("  EXP 08: Xi Attention Classifier — Multi-Model + Threshold Sweep")
    print("  Can Xi serve as a zero-parameter hallucination boundary?")
    print("=" * 70)
    
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    all_model_results = {}
    
    for model_name in MODELS:
        full_name = f"EleutherAI/{model_name}"
        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*60}")
        
        tokenizer = AutoTokenizer.from_pretrained(full_name, cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(full_name, cache_dir=CACHE_DIR)
        model = model.to(device).eval()
        
        config = model.config
        n_layers = config.num_hidden_layers
        n_heads = config.num_attention_heads
        print(f"  {n_layers} layers, {n_heads} heads = {n_layers*n_heads} total")
        
        # Run all prompts
        all_prompts = []
        
        for i, prompt in enumerate(FACTUAL):
            r = run_attention_monitoring(model, tokenizer, device, prompt)
            r['true_label'] = 0
            r['group'] = 'factual'
            all_prompts.append(r)
            xi_dist = r['mean_attn_entropy'] - XI
            side = "▼" if xi_dist < 0 else "▲"
            print(f"  F[{i+1:2d}] H={r['mean_attn_entropy']:.4f} {side} Xi{xi_dist:+.4f}  "
                  f"conf={r['mean_confident_head_ratio']*100:.0f}%  '{prompt[:40]}'")
        
        for i, prompt in enumerate(HALLUCINATION):
            r = run_attention_monitoring(model, tokenizer, device, prompt)
            r['true_label'] = 1
            r['group'] = 'hallucination'
            all_prompts.append(r)
            xi_dist = r['mean_attn_entropy'] - XI
            side = "▼" if xi_dist < 0 else "▲"
            print(f"  H[{i+1:2d}] H={r['mean_attn_entropy']:.4f} {side} Xi{xi_dist:+.4f}  "
                  f"conf={r['mean_confident_head_ratio']*100:.0f}%  '{prompt[:40]}'")
        
        # ── Xi balance point analysis ──
        fact_ents = [r['mean_attn_entropy'] for r in all_prompts if r['true_label'] == 0]
        hall_ents = [r['mean_attn_entropy'] for r in all_prompts if r['true_label'] == 1]
        
        fact_mean = np.mean(fact_ents)
        hall_mean = np.mean(hall_ents)
        midpoint = (fact_mean + hall_mean) / 2
        grand_mean = np.mean(fact_ents + hall_ents)
        
        print(f"\n  XI BALANCE POINT:")
        print(f"    Factual mean:     {fact_mean:.4f}  (Xi {fact_mean - XI:+.4f})")
        print(f"    Halluc mean:      {hall_mean:.4f}  (Xi {hall_mean - XI:+.4f})")
        print(f"    Midpoint F/H:     {midpoint:.4f}  (Xi {midpoint - XI:+.4f})")
        print(f"    Grand mean:       {grand_mean:.4f}  (Xi {grand_mean - XI:+.4f})")
        print(f"    Xi = {XI:.4f}")
        
        # Does factual fall below Xi and halluc above?
        fact_below_xi = sum(1 for v in fact_ents if v < XI)
        hall_above_xi = sum(1 for v in hall_ents if v > XI)
        print(f"    Factual below Xi: {fact_below_xi}/{len(fact_ents)} = {fact_below_xi/len(fact_ents)*100:.0f}%")
        print(f"    Halluc above Xi:  {hall_above_xi}/{len(hall_ents)} = {hall_above_xi/len(hall_ents)*100:.0f}%")
        
        # ── Xi classifier ──
        xi_result = xi_classifier(all_prompts)
        print(f"\n  XI CLASSIFIER (threshold = Xi = {XI:.4f}):")
        print(f"    Precision: {xi_result['precision']:.3f}")
        print(f"    Recall:    {xi_result['recall']:.3f}")
        print(f"    F1:        {xi_result['f1']:.3f}")
        print(f"    Accuracy:  {xi_result['accuracy']:.3f}")
        print(f"    Confusion: TP={xi_result['tp']} FP={xi_result['fp']} FN={xi_result['fn']} TN={xi_result['tn']}")
        
        # ── Threshold sweep ──
        sweep = threshold_sweep(all_prompts)
        print(f"\n  THRESHOLD SWEEP:")
        print(f"    Best threshold:   {sweep['best_threshold']:.4f}")
        print(f"    Best F1:          {sweep['best_f1']:.3f}")
        print(f"    Xi F1:            {sweep['xi_f1']:.3f}")
        print(f"    Xi dist from opt: {sweep['xi_distance_from_optimal']:.4f}")
        
        if sweep['best_f1'] > 0:
            rel_perf = sweep['xi_f1'] / sweep['best_f1'] * 100
            print(f"    Xi relative perf: {rel_perf:.1f}% of optimal")
        
        # ── Multi-feature classifier ──
        mf = multi_feature_classifier(all_prompts)
        print(f"\n  MULTI-FEATURE COMPARISON:")
        for name, info in sorted(mf.items(), key=lambda x: -x[1]['best_f1']):
            sig = "***" if info['mann_whitney_p'] < 0.001 else "**" if info['mann_whitney_p'] < 0.01 else "*" if info['mann_whitney_p'] < 0.05 else ""
            print(f"    {name:30s}: F1={info['best_f1']:.3f}  acc={info['best_accuracy']:.3f}  p={info['mann_whitney_p']:.4f} {sig}")
        
        # ── Token-level trajectory: Xi crossing analysis ──
        fact_trajectories = [r['token_attn_entropies'] for r in all_prompts if r['true_label'] == 0]
        hall_trajectories = [r['token_attn_entropies'] for r in all_prompts if r['true_label'] == 1]
        
        # Mean trajectory per group
        n_steps = min(len(t) for t in fact_trajectories + hall_trajectories)
        fact_traj = np.mean([t[:n_steps] for t in fact_trajectories], axis=0)
        hall_traj = np.mean([t[:n_steps] for t in hall_trajectories], axis=0)
        
        # When does each group cross Xi?
        fact_cross = None
        hall_cross = None
        for s in range(n_steps):
            if fact_cross is None and fact_traj[s] > XI:
                fact_cross = s
            if hall_cross is None and hall_traj[s] > XI:
                hall_cross = s
        
        print(f"\n  TRAJECTORY (mean attn entropy per token step):")
        print(f"    Step  Factual   Halluc    Δ(F-H)")
        for s in [0, 4, 9, 14, 19]:
            if s < n_steps:
                print(f"    {s:4d}  {fact_traj[s]:.4f}    {hall_traj[s]:.4f}    {fact_traj[s]-hall_traj[s]:+.4f}")
        
        if fact_cross is not None:
            print(f"    Factual crosses Xi at step {fact_cross}")
        else:
            print(f"    Factual never crosses Xi (stays below)")
        if hall_cross is not None:
            print(f"    Halluc crosses Xi at step {hall_cross}")
        else:
            print(f"    Halluc {'never crosses Xi (stays above)' if hall_traj[0] > XI else 'never crosses Xi (stays below)'}")
        
        # Store results
        all_model_results[model_name] = {
            'architecture': {'n_layers': n_layers, 'n_heads': n_heads},
            'balance_point': {
                'factual_mean': float(fact_mean),
                'halluc_mean': float(hall_mean),
                'midpoint': float(midpoint),
                'grand_mean': float(grand_mean),
                'xi': float(XI),
                'midpoint_xi_distance': float(abs(midpoint - XI)),
                'factual_below_xi_rate': fact_below_xi / len(fact_ents),
                'halluc_above_xi_rate': hall_above_xi / len(hall_ents),
            },
            'xi_classifier': xi_result,
            'threshold_sweep': {
                'best_threshold': sweep['best_threshold'],
                'best_f1': sweep['best_f1'],
                'xi_f1': sweep['xi_f1'],
                'xi_distance_from_optimal': sweep['xi_distance_from_optimal'],
            },
            'multi_feature': mf,
            'trajectory': {
                'factual_mean': fact_traj.tolist(),
                'halluc_mean': hall_traj.tolist(),
                'factual_xi_crossing': fact_cross,
                'halluc_xi_crossing': hall_cross,
            },
            'prompts': [
                {k: v for k, v in r.items() if k != 'token_attn_entropies'}
                for r in all_prompts
            ],
        }
        
        # Free GPU memory
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    # ── Cross-model summary ──
    print(f"\n{'='*70}")
    print(f"  CROSS-MODEL SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n  Xi Balance Point (factual-mean / Xi / halluc-mean):")
    for mn in MODELS:
        bp = all_model_results[mn]['balance_point']
        print(f"    {mn:15s}: {bp['factual_mean']:.4f}  < Xi({XI:.4f}) <  {bp['halluc_mean']:.4f}"
              f"    midpoint={bp['midpoint']:.4f}  dist={bp['midpoint_xi_distance']:.4f}")
    
    print(f"\n  Xi Classifier Performance:")
    print(f"    {'Model':15s} {'Prec':>6s} {'Recall':>6s} {'F1':>6s} {'Acc':>6s} {'Best F1':>8s} {'BestΘ':>8s} {'ΔΘ':>8s}")
    for mn in MODELS:
        xc = all_model_results[mn]['xi_classifier']
        ts = all_model_results[mn]['threshold_sweep']
        print(f"    {mn:15s} {xc['precision']:6.3f} {xc['recall']:6.3f} {xc['f1']:6.3f} {xc['accuracy']:6.3f}"
              f" {ts['best_f1']:8.3f} {ts['best_threshold']:8.4f} {ts['xi_distance_from_optimal']:8.4f}")
    
    print(f"\n  Best Single Feature per Model:")
    for mn in MODELS:
        mf = all_model_results[mn]['multi_feature']
        best = max(mf.items(), key=lambda x: x[1]['best_f1'])
        print(f"    {mn:15s}: {best[0]} (F1={best[1]['best_f1']:.3f}, p={best[1]['mann_whitney_p']:.4f})")
    
    # ── Save ──
    output = {
        'experiment': 'exp_08_xi_attention_classifier',
        'timestamp': datetime.now().isoformat(),
        'n_factual': len(FACTUAL),
        'n_hallucination': len(HALLUCINATION),
        'n_tokens_per_prompt': 20,
        'xi': float(XI),
        'models': all_model_results,
        'elapsed_seconds': time.time() - t0,
    }
    
    results_dir = EXPERIMENT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"exp_08_xi_classifier_{ts}.json"
    
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
