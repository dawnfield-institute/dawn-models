#!/usr/bin/env python3
"""
EXP 06: Xi Clustering in Weight Spectra — Trained vs Untrained
===============================================================

exp_05 found 70.8% of weight SVD ratio σ_i/σ_{i+1} cluster near Xi (1.057).
Is this a property of TRAINED networks, or does it emerge from random
initialisation / the Marchenko-Pastur distribution of random matrices?

Design:
  1. Load Pythia-160m (trained) — compute SVD ratios for all weight matrices
  2. Create a randomly-initialised model with IDENTICAL architecture
  3. Create Marchenko-Pastur null (random Gaussian matrices of same shape)
  4. Compare Xi clustering rates across all three
  5. Also check across all 4 Pythia scales (70m, 160m, 410m, 1B)

If Xi clustering is equally present in random matrices → softmax/SVD artifact
If Xi clustering is ONLY in trained networks → training creates it
If Xi clustering grows with model size → structural organising principle

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
CACHE_DIR = str(Path(EXPERIMENT_DIR).parent / "huggingface_bifractal_validation" / "pythia_cache")

PHI = (1 + 5**0.5) / 2
INV_PHI = PHI - 1
XI = 1 + np.pi / 55
TOLERANCE = 0.05  # 5% relative tolerance


def compute_svd_ratios(weight_matrix):
    """Compute consecutive singular value ratios for a weight matrix."""
    with torch.no_grad():
        _, S, _ = torch.linalg.svd(weight_matrix.float().cpu(), full_matrices=False)
    
    S = S.numpy()
    ratios = []
    for i in range(len(S) - 1):
        if S[i+1] > 1e-10:
            ratios.append(float(S[i] / S[i+1]))
    return ratios, S


def analyse_ratios(ratios, label):
    """Compute clustering statistics for a set of ratios."""
    if not ratios:
        return {}
    
    ratios = np.array(ratios)
    
    near_xi = np.sum(np.abs(ratios - XI) / XI < TOLERANCE)
    near_phi = np.sum(np.abs(ratios - PHI) / PHI < TOLERANCE)
    near_inv_phi = np.sum(np.abs(ratios - INV_PHI) / INV_PHI < TOLERANCE)
    near_1 = np.sum(np.abs(ratios - 1.0) < TOLERANCE)
    
    # Also check at different tolerances
    xi_at_1pct = np.sum(np.abs(ratios - XI) / XI < 0.01)
    xi_at_10pct = np.sum(np.abs(ratios - XI) / XI < 0.10)
    
    result = {
        'n_ratios': len(ratios),
        'mean': float(np.mean(ratios)),
        'median': float(np.median(ratios)),
        'std': float(np.std(ratios)),
        'min': float(np.min(ratios)),
        'max': float(np.max(ratios)),
        'near_xi_5pct': int(near_xi),
        'near_xi_5pct_rate': float(near_xi / len(ratios)),
        'near_xi_1pct': int(xi_at_1pct),
        'near_xi_1pct_rate': float(xi_at_1pct / len(ratios)),
        'near_xi_10pct': int(xi_at_10pct),
        'near_xi_10pct_rate': float(xi_at_10pct / len(ratios)),
        'near_phi_5pct': int(near_phi),
        'near_phi_5pct_rate': float(near_phi / len(ratios)),
        'near_1_5pct': int(near_1),
        'near_1_5pct_rate': float(near_1 / len(ratios)),
    }
    
    return result


def extract_weight_ratios(model, label="model"):
    """Extract SVD ratios from all 2D+ parameter matrices."""
    all_ratios = []
    per_type = defaultdict(list)
    
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        
        is_mlp = 'mlp' in name.lower()
        is_attn = 'attention' in name.lower() or 'attn' in name.lower()
        is_embed = 'embed' in name.lower()
        
        if is_mlp or is_attn:
            ratios, _ = compute_svd_ratios(param.data)
            all_ratios.extend(ratios)
            
            if is_mlp:
                per_type['mlp'].extend(ratios)
            elif is_attn:
                per_type['attention'].extend(ratios)
    
    return all_ratios, dict(per_type)


def create_random_model(model):
    """Create a model with same architecture but random (re-initialised) weights."""
    import copy
    random_model = copy.deepcopy(model)
    
    for name, param in random_model.named_parameters():
        if param.dim() >= 2:
            # Standard Xavier/Glorot initialisation
            torch.nn.init.xavier_normal_(param.data)
        elif param.dim() == 1:
            torch.nn.init.zeros_(param.data)
    
    return random_model


def marchenko_pastur_null(model, n_samples=5):
    """
    Generate random Gaussian matrices with same shapes as model weights.
    Average over multiple samples for stability.
    """
    all_ratios = []
    
    for _ in range(n_samples):
        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue
            
            is_mlp = 'mlp' in name.lower()
            is_attn = 'attention' in name.lower() or 'attn' in name.lower()
            
            if is_mlp or is_attn:
                # Random Gaussian matrix with same shape
                random_w = torch.randn_like(param.data)
                ratios, _ = compute_svd_ratios(random_w)
                all_ratios.extend(ratios)
    
    return all_ratios


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    
    print("=" * 70)
    print("  EXP 06: Xi Clustering — Trained vs Untrained vs Random")
    print("=" * 70)
    
    t0 = time.time()
    
    # ── Part 1: Pythia-160m three-way comparison ──
    print(f"\n{'='*60}")
    print(f"  PART 1: Pythia-160m — Trained vs Reinitialised vs Random")
    print(f"{'='*60}")
    
    model_name = "EleutherAI/pythia-160m"
    print(f"\n  Loading trained {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR)
    model.eval()
    
    # 1a: Trained weights
    print("  Extracting trained weight SVD ratios...")
    trained_ratios, trained_per_type = extract_weight_ratios(model, "trained")
    trained_stats = analyse_ratios(trained_ratios, "trained")
    
    print(f"    Trained: n={trained_stats['n_ratios']}, "
          f"Xi@5%={trained_stats['near_xi_5pct_rate']*100:.1f}%, "
          f"phi@5%={trained_stats['near_phi_5pct_rate']*100:.1f}%, "
          f"median={trained_stats['median']:.4f}")
    
    # 1b: Re-initialised weights (same architecture)
    print("  Creating re-initialised model...")
    random_model = create_random_model(model)
    reinit_ratios, reinit_per_type = extract_weight_ratios(random_model, "reinit")
    reinit_stats = analyse_ratios(reinit_ratios, "reinit")
    
    print(f"    Reinit:  n={reinit_stats['n_ratios']}, "
          f"Xi@5%={reinit_stats['near_xi_5pct_rate']*100:.1f}%, "
          f"phi@5%={reinit_stats['near_phi_5pct_rate']*100:.1f}%, "
          f"median={reinit_stats['median']:.4f}")
    
    del random_model
    
    # 1c: Pure random matrices (Marchenko-Pastur null)
    print("  Computing Marchenko-Pastur null (5 samples)...")
    mp_ratios = marchenko_pastur_null(model, n_samples=5)
    mp_stats = analyse_ratios(mp_ratios, "random")
    
    print(f"    Random:  n={mp_stats['n_ratios']}, "
          f"Xi@5%={mp_stats['near_xi_5pct_rate']*100:.1f}%, "
          f"phi@5%={mp_stats['near_phi_5pct_rate']*100:.1f}%, "
          f"median={mp_stats['median']:.4f}")
    
    # Statistical test: trained vs random Xi rates
    trained_near = [1 if abs(r - XI)/XI < TOLERANCE else 0 for r in trained_ratios]
    mp_near = [1 if abs(r - XI)/XI < TOLERANCE else 0 for r in mp_ratios]
    
    # Chi-squared test for proportion difference
    from scipy.stats import chi2_contingency
    trained_xi = sum(trained_near)
    trained_not = len(trained_near) - trained_xi
    mp_xi = sum(mp_near)
    mp_not = len(mp_near) - mp_xi
    
    contingency = [[trained_xi, trained_not], [mp_xi, mp_not]]
    chi2, p_chi, dof, expected = chi2_contingency(contingency)
    
    print(f"\n  Chi-squared test (trained vs random Xi rate):")
    print(f"    Trained Xi: {trained_xi}/{len(trained_near)} = {trained_xi/len(trained_near)*100:.1f}%")
    print(f"    Random Xi:  {mp_xi}/{len(mp_near)} = {mp_xi/len(mp_near)*100:.1f}%")
    print(f"    chi² = {chi2:.2f}, p = {p_chi:.6f}")
    
    # ── Part 2: Cross-model scaling ──
    print(f"\n{'='*60}")
    print(f"  PART 2: Xi Clustering Across Model Scales")
    print(f"{'='*60}")
    
    model_names = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b"]
    scale_results = {}
    
    # We already have 160m
    scale_results["pythia-160m"] = {
        'stats': trained_stats,
        'per_type': {k: analyse_ratios(v, k) for k, v in trained_per_type.items()},
    }
    
    del model  # free memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    for mname in model_names:
        if mname == "pythia-160m":
            continue
        
        print(f"\n  Loading EleutherAI/{mname}...")
        m = AutoModelForCausalLM.from_pretrained(f"EleutherAI/{mname}", cache_dir=CACHE_DIR)
        m.eval()
        
        ratios, per_type = extract_weight_ratios(m, mname)
        s = analyse_ratios(ratios, mname)
        
        scale_results[mname] = {
            'stats': s,
            'per_type': {k: analyse_ratios(v, k) for k, v in per_type.items()},
        }
        
        print(f"    {mname}: n={s['n_ratios']}, "
              f"Xi@5%={s['near_xi_5pct_rate']*100:.1f}%, "
              f"phi@5%={s['near_phi_5pct_rate']*100:.1f}%, "
              f"median={s['median']:.4f}")
        
        del m
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ── Summary table ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n  Three-way comparison (Pythia-160m):")
    print(f"  {'Source':15s} {'n':>6s} {'Xi@5%':>8s} {'Xi@1%':>8s} {'phi@5%':>8s} {'~1@5%':>8s} {'median':>8s}")
    print(f"  {'-'*60}")
    for label, s in [("Trained", trained_stats), ("Reinitialised", reinit_stats), ("Random", mp_stats)]:
        print(f"  {label:15s} {s['n_ratios']:6d} {s['near_xi_5pct_rate']*100:7.1f}% "
              f"{s['near_xi_1pct_rate']*100:7.1f}% {s['near_phi_5pct_rate']*100:7.1f}% "
              f"{s['near_1_5pct_rate']*100:7.1f}% {s['median']:8.4f}")
    
    print(f"\n  Cross-model scaling:")
    print(f"  {'Model':15s} {'n':>6s} {'Xi@5%':>8s} {'Xi@1%':>8s} {'phi@5%':>8s} {'median':>8s} {'MLP Xi%':>8s} {'Attn Xi%':>8s}")
    print(f"  {'-'*75}")
    for mname in model_names:
        s = scale_results[mname]['stats']
        pt = scale_results[mname]['per_type']
        mlp_xi = pt.get('mlp', {}).get('near_xi_5pct_rate', 0) * 100
        attn_xi = pt.get('attention', {}).get('near_xi_5pct_rate', 0) * 100
        print(f"  {mname:15s} {s['n_ratios']:6d} {s['near_xi_5pct_rate']*100:7.1f}% "
              f"{s['near_xi_1pct_rate']*100:7.1f}% {s['near_phi_5pct_rate']*100:7.1f}% "
              f"{s['median']:8.4f} {mlp_xi:7.1f}% {attn_xi:7.1f}%")
    
    # ── Per-type analysis ──
    print(f"\n  MLP vs Attention (Pythia-160m):")
    for ttype in ['mlp', 'attention']:
        if ttype in trained_per_type:
            ts = analyse_ratios(trained_per_type[ttype], ttype)
            print(f"    {ttype:12s}: n={ts['n_ratios']}, Xi@5%={ts['near_xi_5pct_rate']*100:.1f}%, "
                  f"median={ts['median']:.4f}")
    
    # ── Save ──
    output = {
        'experiment': 'exp_06_xi_weight_clustering',
        'timestamp': datetime.now().isoformat(),
        'constants': {'phi': PHI, 'inv_phi': INV_PHI, 'xi': XI, 'tolerance': TOLERANCE},
        'three_way': {
            'trained': trained_stats,
            'reinitialised': reinit_stats,
            'random_mp': mp_stats,
            'chi_squared': {
                'chi2': float(chi2), 'p': float(p_chi), 'dof': int(dof),
                'trained_xi_count': trained_xi, 'trained_total': len(trained_near),
                'random_xi_count': mp_xi, 'random_total': len(mp_near),
            },
        },
        'cross_model': {
            mname: scale_results[mname]
            for mname in model_names
        },
        'elapsed_seconds': time.time() - t0,
    }
    
    results_dir = EXPERIMENT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"exp_06_xi_weight_{ts}.json"
    
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
