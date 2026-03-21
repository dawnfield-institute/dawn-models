"""
Experiment 02 v3: Extended Training
====================================
10 epochs, larger model, more data.
"""

import torch
import torch.nn as nn
import time
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from exp_02_wikitext2_v2 import (
    GPUPACConfig, GPUPACTransformer, 
    load_wikitext2, build_vocab, 
    train_epoch, evaluate
)

def main():
    print('='*60)
    print('Extended Training (10 epochs, larger model)')
    print('='*60)
    
    # Load more data
    train_texts = load_wikitext2('train', max_samples=5000)
    test_texts = load_wikitext2('test', max_samples=1000)
    print(f'Train: {len(train_texts)}, Test: {len(test_texts)}')
    
    word_to_id = build_vocab(train_texts, vocab_size=8000)
    print(f'Vocab: {len(word_to_id)}')
    
    # Larger model (512 dim / 8 heads)
    config = GPUPACConfig(
        vocab_size=len(word_to_id),
        embedding_dim=512,
        hidden_dim=1024,
        n_layers=4,
        dropout=0.1,
        device='cuda'
    )
    model = GPUPACTransformer(config)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    start = time.perf_counter()
    for epoch in range(10):
        result = train_epoch(model, train_texts, word_to_id, optimizer, batch_size=64, max_len=128)
        scheduler.step()
        print(f'Epoch {epoch+1}: loss={result["avg_loss"]:.4f}, ppl={result["perplexity"]:.2f}')
    
    total_time = time.perf_counter() - start
    print(f'\nTotal training time: {total_time:.2f}s')
    
    # Final eval
    eval_result = evaluate(model, test_texts, word_to_id, batch_size=64, max_len=128)
    print(f'\nFinal Test PPL: {eval_result["perplexity"]:.2f}')
    print(f'GAIA baseline: 5.91')
    print(f'Ratio: {eval_result["perplexity"]/5.91:.2f}x')
    
    if eval_result["perplexity"] < 50:
        print('✅ Approaching reasonable perplexity')
    else:
        print('⚠️ Needs more epochs or hyperparameter tuning')

if __name__ == '__main__':
    main()
