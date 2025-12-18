"""
Experiment 02 v2: PAC-Lazy on WikiText-2 (GPU Optimized)
========================================================

Batched GPU operations for efficient training/evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Constants from PAC-Lazy
PHI = 1.6180339887
XI = 0.0618
PHI_XI = 1.710
LAMBDA_STAR = 0.9816


@dataclass
class GPUPACConfig:
    vocab_size: int = 5000
    embedding_dim: int = 384
    hidden_dim: int = 512
    n_layers: int = 2
    dropout: float = 0.1
    total_potential: float = 100.0
    device: str = 'cuda'


class GPUPACTransformer(nn.Module):
    """
    GPU-optimized PAC-Lazy Transformer.
    
    Uses batched operations while maintaining PAC principles:
    - Potential tracking per sequence position
    - SEC-style adaptive computation
    - Causal masking for locality
    """
    
    def __init__(self, config: GPUPACConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_embed = nn.Embedding(512, config.embedding_dim)
        
        # PAC potential per position (learnable)
        self.potential_scale = nn.Parameter(torch.ones(512) * 0.1)
        
        # Transformer layers with causal attention
        # nhead must divide embedding_dim evenly
        nhead = 8 if config.embedding_dim % 8 == 0 else 6
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=nhead,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        
        # Output projection
        self.output_proj = nn.Linear(config.embedding_dim, config.vocab_size)
        
        # Tracking
        self.total_potential = config.total_potential
        self.current_potential = config.total_potential
        self.expansions = 0
        self.collapses = 0
        
        self.to(self.device)
    
    def forward(self, input_ids: torch.Tensor, 
                targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with PAC potential tracking.
        
        Args:
            input_ids: [batch, seq_len] token IDs
            targets: [batch, seq_len] target IDs for loss computation
            
        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: scalar if targets provided
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        
        # PAC: Scale by potential (higher potential = more active)
        potential_weights = torch.sigmoid(self.potential_scale[:seq_len]).unsqueeze(0).unsqueeze(-1)
        x = x * potential_weights
        
        # Causal mask for locality
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=self.device)
        
        # Transform
        hidden = self.transformer(x, mask=causal_mask, is_causal=True)
        
        # Project to vocab
        logits = self.output_proj(hidden)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_targets.view(-1),
                ignore_index=-100
            )
            
            # PAC: Update potential based on loss
            self._update_potential(loss.item(), seq_len)
        
        return logits, loss
    
    def _update_potential(self, loss: float, seq_len: int):
        """Update potential based on loss (SEC dynamics)."""
        # High loss = expansion needed
        if loss > PHI_XI:
            self.expansions += 1
        # Low loss = can collapse
        elif loss < XI:
            self.collapses += 1
            self.current_potential = min(self.current_potential + 0.1, self.total_potential)
    
    def get_stats(self) -> Dict:
        return {
            'current_potential': self.current_potential,
            'utilization': 1 - (self.current_potential / self.total_potential),
            'expansions': self.expansions,
            'collapses': self.collapses
        }


def load_wikitext2(split: str = 'train', max_samples: int = None) -> Optional[List[str]]:
    """Load WikiText-2 dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        texts = [x['text'] for x in ds if x['text'].strip()]
        if max_samples:
            texts = texts[:max_samples]
        return texts
    except ImportError:
        return None


def build_vocab(texts: List[str], vocab_size: int = 5000) -> Dict[str, int]:
    """Build vocabulary from texts."""
    from collections import Counter
    
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)
    
    word_to_id = {'<pad>': 0, '<unk>': 1}
    for word, _ in word_counts.most_common(vocab_size - 2):
        word_to_id[word] = len(word_to_id)
    
    return word_to_id


def tokenize_batch(texts: List[str], word_to_id: Dict[str, int], 
                   max_len: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize batch of texts with padding."""
    batch = []
    for text in texts:
        words = text.lower().split()
        ids = [word_to_id.get(w, 1) for w in words]  # 1 = <unk>
        ids = ids[:max_len]
        batch.append(ids)
    
    # Pad to max length in batch
    max_batch_len = max(len(ids) for ids in batch)
    padded = []
    for ids in batch:
        padded.append(ids + [0] * (max_batch_len - len(ids)))  # 0 = <pad>
    
    input_ids = torch.tensor(padded, dtype=torch.long)
    # Targets are same as inputs (for next-token prediction, loss shifts internally)
    targets = input_ids.clone()
    targets[targets == 0] = -100  # Ignore padding in loss
    
    return input_ids, targets


def train_epoch(model: GPUPACTransformer, 
                texts: List[str],
                word_to_id: Dict[str, int],
                optimizer: torch.optim.Optimizer,
                batch_size: int = 32,
                max_len: int = 128) -> Dict:
    """Train for one epoch with batched GPU operations."""
    model.train()
    
    total_loss = 0
    total_batches = 0
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        if not batch_texts:
            continue
            
        # Filter empty texts
        batch_texts = [t for t in batch_texts if len(t.split()) > 1]
        if not batch_texts:
            continue
        
        input_ids, targets = tokenize_batch(batch_texts, word_to_id, max_len)
        input_ids = input_ids.to(model.device)
        targets = targets.to(model.device)
        
        optimizer.zero_grad()
        _, loss = model(input_ids, targets)
        
        if loss is not None and not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
    
    avg_loss = total_loss / max(total_batches, 1)
    return {
        'avg_loss': avg_loss,
        'perplexity': math.exp(min(avg_loss, 10)),  # Cap to avoid overflow
        'batches': total_batches
    }


def evaluate(model: GPUPACTransformer,
             texts: List[str],
             word_to_id: Dict[str, int],
             batch_size: int = 32,
             max_len: int = 128) -> Dict:
    """Evaluate model with batched GPU operations."""
    model.eval()
    
    total_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_texts = [t for t in batch_texts if len(t.split()) > 1]
            if not batch_texts:
                continue
            
            input_ids, targets = tokenize_batch(batch_texts, word_to_id, max_len)
            input_ids = input_ids.to(model.device)
            targets = targets.to(model.device)
            
            _, loss = model(input_ids, targets)
            
            if loss is not None and not torch.isnan(loss):
                total_loss += loss.item()
                total_batches += 1
    
    avg_loss = total_loss / max(total_batches, 1)
    return {
        'avg_loss': avg_loss,
        'perplexity': math.exp(min(avg_loss, 10)),
        'batches': total_batches
    }


def main():
    print("=" * 60)
    print("POC-011 Exp 02 v2: PAC-Lazy WikiText-2 (GPU Optimized)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Target: GAIA unified perplexity = 5.91")
    
    # Load data
    print("\n1. Loading WikiText-2...")
    train_texts = load_wikitext2('train', max_samples=2000)
    
    if train_texts is None:
        print("  Using synthetic data (datasets library not available)")
        import random
        random.seed(42)
        vocab_size = 1000
        train_texts = [' '.join(f'word{random.randint(0, vocab_size-1)}' for _ in range(50)) 
                       for _ in range(1000)]
        test_texts = [' '.join(f'word{random.randint(0, vocab_size-1)}' for _ in range(50)) 
                      for _ in range(200)]
        word_to_id = {f'word{i}': i for i in range(vocab_size)}
    else:
        test_texts = load_wikitext2('test', max_samples=500)
        print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")
        
        print("\n2. Building vocabulary...")
        word_to_id = build_vocab(train_texts, vocab_size=5000)
        print(f"  Vocab size: {len(word_to_id)}")
    
    # Create model
    print("\n3. Creating GPU PAC Transformer...")
    config = GPUPACConfig(
        vocab_size=len(word_to_id),
        embedding_dim=384,
        hidden_dim=512,
        n_layers=2,
        device=device
    )
    model = GPUPACTransformer(config)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training
    print("\n4. Training (3 epochs)...")
    results = {'epochs': []}
    
    for epoch in range(3):
        start = time.perf_counter()
        train_result = train_epoch(model, train_texts, word_to_id, optimizer, batch_size=32)
        elapsed = time.perf_counter() - start
        
        print(f"  Epoch {epoch+1}: loss={train_result['avg_loss']:.4f}, "
              f"ppl={train_result['perplexity']:.2f}, time={elapsed:.2f}s")
        
        results['epochs'].append({
            'epoch': epoch + 1,
            'train_loss': train_result['avg_loss'],
            'train_ppl': train_result['perplexity'],
            'time_s': elapsed
        })
    
    # Evaluation
    print("\n5. Evaluating...")
    start = time.perf_counter()
    eval_result = evaluate(model, test_texts, word_to_id, batch_size=32)
    eval_time = time.perf_counter() - start
    
    print(f"  Test loss: {eval_result['avg_loss']:.4f}")
    print(f"  Test perplexity: {eval_result['perplexity']:.2f}")
    print(f"  Eval time: {eval_time:.2f}s")
    
    results['test'] = {
        'loss': eval_result['avg_loss'],
        'perplexity': eval_result['perplexity'],
        'time_s': eval_time
    }
    
    # PAC stats
    stats = model.get_stats()
    results['pac_stats'] = stats
    
    print("\n6. PAC Statistics...")
    print(f"  Potential utilization: {stats['utilization']:.1%}")
    print(f"  Expansions: {stats['expansions']}")
    print(f"  Collapses: {stats['collapses']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    baseline = 5.91
    test_ppl = eval_result['perplexity']
    
    print(f"  GAIA baseline:     {baseline:.2f}")
    print(f"  PAC-Lazy GPU:      {test_ppl:.2f}")
    print(f"  Ratio:             {test_ppl/baseline:.2f}x")
    
    if test_ppl < baseline * 1.5:
        print("  ✅ Excellent: Within 1.5x of baseline")
    elif test_ppl < baseline * 2:
        print("  ✅ Good: Within 2x of baseline")
    elif test_ppl < baseline * 5:
        print("  ⚠️ Needs tuning: 2-5x baseline")
    else:
        print("  ❌ Needs work: >5x baseline")
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'baseline': baseline,
        'results': results
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = results_dir / f'exp_02_wikitext2_v2_{timestamp}.json'
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
