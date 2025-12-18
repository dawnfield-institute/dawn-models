#!/usr/bin/env python
"""
GAIA-1 Overnight Training Configuration

Run this script to train GAIA-1 overnight on WikiText-2.
Expected training time: ~8-12 hours
Expected perplexity: <50 (hopefully <30)

Usage:
    python train_overnight.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def main():
    print("="*60)
    print("GAIA-1 Overnight Training")
    print("="*60)
    print(f"Start time: {datetime.now()}")
    print()
    
    # Training configuration
    # With resonance attention, the model learns much better!
    config = {
        # Model - moderate size for overnight training
        'field_dim': 256,       # 256 is plenty with resonance attention
        'n_layers': 4,          # 4 layers with multi-head attention
        'max_context': 128,     # Keep reasonable for memory
        
        # Training - 20 epochs should be enough to see good results
        'epochs': 20,           # Enough for convergence
        'batch_size': 8,        # Fits comfortably in GPU memory
        'lr': 3e-4,             # Good LR for this architecture
        'accumulation_steps': 4, # Effective batch = 32
        
        # Data
        'dataset': 'wikitext',
        
        # Checkpointing
        'save_dir': './checkpoints/overnight_run',
        'eval_every': 1,
    }
    
    # Build command
    cmd = [
        sys.executable, 'train.py',
        f'--epochs={config["epochs"]}',
        f'--batch_size={config["batch_size"]}',
        f'--lr={config["lr"]}',
        f'--field_dim={config["field_dim"]}',
        f'--n_layers={config["n_layers"]}',
        f'--max_context={config["max_context"]}',
        f'--dataset={config["dataset"]}',
        f'--save_dir={config["save_dir"]}',
        f'--accumulation_steps={config["accumulation_steps"]}',
        f'--eval_every={config["eval_every"]}',
    ]
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    print(f"Command: {' '.join(cmd)}")
    print("="*60)
    print()
    
    # Run training
    subprocess.run(cmd)
    
    print()
    print("="*60)
    print(f"End time: {datetime.now()}")
    print("="*60)


if __name__ == "__main__":
    main()
