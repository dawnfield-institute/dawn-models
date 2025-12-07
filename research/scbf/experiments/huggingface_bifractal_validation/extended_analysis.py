"""
Extended Pythia Checkpoint Analysis

Goes beyond step 512 to see full training trajectory.
Key question: Does the ratio continue converging to phi, stabilize, or diverge?
"""

import torch
from huggingface_hub import hf_hub_download
import numpy as np
from scipy import stats
import os
import time

PHI = 1.618034

def compute_delta_norm(w1, w2):
    """Compute L2 norm of weight delta, filtering non-finite values."""
    delta_sq = 0.0
    keys_used = 0
    for k in w1.keys():
        if k in w2:
            d1 = w1[k].float()
            d2 = w2[k].float()
            if torch.isfinite(d1).all() and torch.isfinite(d2).all():
                delta_sq += ((d2 - d1)**2).sum().item()
                keys_used += 1
    return np.sqrt(delta_sq), keys_used


def load_checkpoint(model_id, step, cache_dir, max_retries=3):
    """Load a single checkpoint with retry logic."""
    for attempt in range(max_retries):
        try:
            path = hf_hub_download(
                repo_id=model_id,
                filename='pytorch_model.bin',
                revision=f'step{step}',
                cache_dir=cache_dir
            )
            return torch.load(path, map_location='cpu', weights_only=True)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return None


def main():
    print('='*70)
    print('PYTHIA-70M EXTENDED CHECKPOINT ANALYSIS')
    print('Steps: 0 -> 143000 (sampling key points)')
    print('='*70)

    model_id = 'EleutherAI/pythia-70m'
    # Extended range - exponential early, then sample later training
    steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 
             1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000]
    cache_dir = './pythia_cache'

    # Load checkpoints
    checkpoints = {}
    for step in steps:
        print(f'Loading step {step}...', end=' ', flush=True)
        ckpt = load_checkpoint(model_id, step, cache_dir)
        if ckpt:
            checkpoints[step] = ckpt
            print('OK')
        else:
            print('SKIP')

    print(f'\nLoaded {len(checkpoints)} checkpoints')

    # Compute deltas
    sorted_steps = sorted(checkpoints.keys())
    deltas = []
    print('\nComputing deltas...')
    for i in range(len(sorted_steps)-1):
        s1, s2 = sorted_steps[i], sorted_steps[i+1]
        norm, keys = compute_delta_norm(checkpoints[s1], checkpoints[s2])
        interval = s2 - s1
        norm_per_step = norm / interval if interval > 0 else norm
        deltas.append({
            'from': s1, 
            'to': s2, 
            'norm': norm, 
            'interval': interval, 
            'norm_per_step': norm_per_step
        })
        print(f'  {s1:>6} -> {s2:>6}: norm={norm:10.4f}  interval={interval:>5}  per_step={norm_per_step:.6f}')

    # Compute ratios
    print('\n' + '='*70)
    print('DELTA RATIOS (consecutive)')
    print('='*70)
    ratios = []
    for i in range(len(deltas)-1):
        d1, d2 = deltas[i], deltas[i+1]
        if d1['norm'] > 1e-10:
            raw_ratio = d2['norm'] / d1['norm']
            if d1['norm_per_step'] > 1e-10:
                norm_ratio = d2['norm_per_step'] / d1['norm_per_step']
            else:
                norm_ratio = float('nan')
            
            dist_phi = abs(raw_ratio - PHI)
            
            ratios.append({
                'step': d1['to'],
                'raw_ratio': raw_ratio,
                'norm_ratio': norm_ratio,
                'dist_phi': dist_phi
            })
            print(f"  Step {d1['to']:>6}: raw_ratio={raw_ratio:8.4f}  norm_ratio={norm_ratio:8.4f}  dist_phi={dist_phi:.4f}")

    # Analysis by training phase
    print('\n' + '='*70)
    print('ANALYSIS BY TRAINING PHASE')
    print('='*70)

    early = [r for r in ratios if r['step'] <= 512]
    mid = [r for r in ratios if 512 < r['step'] <= 16000]
    late = [r for r in ratios if r['step'] > 16000]

    for phase, data in [('Early (0-512)', early), ('Mid (512-16k)', mid), ('Late (16k+)', late)]:
        if data:
            raw_ratios = [r['raw_ratio'] for r in data]
            dists = [r['dist_phi'] for r in data]
            print(f'{phase}:')
            print(f'  Mean raw ratio: {np.mean(raw_ratios):.4f}')
            print(f'  Mean dist from phi: {np.mean(dists):.4f}')
            print(f'  Closest to phi: {min(dists):.4f}')
            print()

    # Convergence target analysis
    print('='*70)
    print('CONVERGENCE TARGET ANALYSIS')
    print('='*70)
    late_ratios = [r['raw_ratio'] for r in ratios if r['step'] > 8000]
    if late_ratios:
        mean_late = np.mean(late_ratios)
        std_late = np.std(late_ratios)
        print(f'Late training ratios (step > 8000):')
        print(f'  Values: {[f"{r:.3f}" for r in late_ratios]}')
        print(f'  Mean: {mean_late:.4f} ± {std_late:.4f}')
        print()
        print('Distance from key values:')
        for target, name in [(PHI, 'phi (1.618)'), (2.0, '2.0'), (1.5, '1.5'), (1.0, '1.0')]:
            print(f'  {name}: {abs(mean_late - target):.4f}')
    
    # Normalized ratios (per-step) - removes interval size effect
    print('\n' + '='*70)
    print('NORMALIZED RATIOS (per-step change rate)')
    print('='*70)
    norm_ratios = [r['norm_ratio'] for r in ratios if not np.isnan(r['norm_ratio'])]
    if norm_ratios:
        print(f'All normalized ratios: {[f"{r:.3f}" for r in norm_ratios[:10]]}...')
        late_norm = [r['norm_ratio'] for r in ratios if r['step'] > 8000 and not np.isnan(r['norm_ratio'])]
        if late_norm:
            print(f'Late normalized mean: {np.mean(late_norm):.4f}')


if __name__ == '__main__':
    main()
