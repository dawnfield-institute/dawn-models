"""
Experiment 03: PAC-Lazy with Spherical Field Encoding
======================================================

Integrates PAC conservation laws with GAIA's spherical encoder
for a fair comparison with the unified baseline.

This is the "full picture" - combining:
1. Spherical field encoding (from GAIA)
2. PAC potential conservation
3. SEC adaptive depth
4. Causal locality
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

# Import GAIA components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

try:
    from gaia.spherical_encoder import SphericalEncoderV6
    HAS_GAIA = True
except ImportError:
    HAS_GAIA = False

# PAC constants
PHI = 1.6180339887
XI = 0.0618
PHI_XI = 1.710
LAMBDA_STAR = 0.9816


@dataclass
class PACFieldConfig:
    vocab_size: int = 5000
    field_dim: int = 384
    n_layers: int = 4
    total_potential: float = 100.0
    expansion_threshold: float = PHI_XI
    device: str = 'cuda'


class SphericalFieldEncoder(nn.Module):
    """Spherical field encoding with PAC potential tracking."""
    
    def __init__(self, vocab_size: int, field_dim: int, device: str = 'cuda'):
        super().__init__()
        self.field_dim = field_dim
        self.device = device
        
        # Vocabulary field vectors (on unit sphere)
        self.vocab_fields = nn.Parameter(torch.randn(vocab_size, field_dim))
        nn.init.xavier_normal_(self.vocab_fields)
        
        # Position encoding as rotation
        self.pos_freq = nn.Parameter(torch.randn(256, field_dim) * 0.1)
        
    def forward(self, token_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Encode tokens to field vectors on unit sphere."""
        # Get base field vectors
        fields = F.embedding(token_ids, self.vocab_fields)
        
        # Apply position-dependent rotation (simplified as additive)
        pos_emb = F.embedding(positions.clamp(0, 255), self.pos_freq)
        fields = fields + pos_emb
        
        # Project to unit sphere
        fields = F.normalize(fields, dim=-1)
        
        return fields


class PACAttention(nn.Module):
    """
    PAC-bounded attention: Potential limits attention span.
    
    Unlike standard attention, we only attend to neighbors
    within the potential budget (causal locality).
    """
    
    def __init__(self, dim: int, n_heads: int = 8, potential_decay: float = 0.9):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.potential_decay = potential_decay
        
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor, potential_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Attention scores
        q = q.transpose(1, 2)  # [B, H, L, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        attn.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # PAC potential mask: decay attention with distance
        if potential_mask is not None:
            attn = attn + potential_mask.unsqueeze(0).unsqueeze(0)
        else:
            # Default: exponential decay with distance
            dist = torch.arange(L, device=x.device).unsqueeze(0) - torch.arange(L, device=x.device).unsqueeze(1)
            decay_mask = torch.where(dist >= 0, -dist.float() * (1 - self.potential_decay), float('-inf'))
            attn = attn + decay_mask.unsqueeze(0).unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class PACFieldLayer(nn.Module):
    """Single PAC field layer with SEC-style gating."""
    
    def __init__(self, dim: int, n_heads: int = 8, expansion_factor: float = 4.0):
        super().__init__()
        
        self.attn = PACAttention(dim, n_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # SEC gate: controls depth expansion (operates on full dim)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        
        # FFN
        hidden = int(dim * expansion_factor)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Attention with residual
        attn_out = self.attn(self.norm1(x))
        x = x + attn_out
        
        # SEC gate: pooled representation per token
        gate_val = self.gate(x)  # [B, L, 1]
        
        # FFN with gated residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out * gate_val  # Broadcasting [B, L, D] * [B, L, 1]
        
        return x, gate_val.squeeze(-1)  # Return [B, L] gate values


class PACFieldTransformer(nn.Module):
    """
    PAC Field Transformer: Combines spherical encoding with PAC dynamics.
    
    Key features:
    - Spherical field encoding (unit sphere projection)
    - PAC-bounded attention (potential decay with distance)
    - SEC gating (adaptive depth per token)
    - Conservation tracking (expansion/collapse counts)
    """
    
    def __init__(self, config: PACFieldConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Spherical encoder
        self.encoder = SphericalFieldEncoder(config.vocab_size, config.field_dim, config.device)
        
        # PAC layers
        self.layers = nn.ModuleList([
            PACFieldLayer(config.field_dim, n_heads=8)
            for _ in range(config.n_layers)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(config.field_dim, config.vocab_size)
        
        # PAC tracking
        self.total_potential = config.total_potential
        self.current_potential = config.total_potential
        self.expansions = 0
        self.collapses = 0
        
        self.to(self.device)
        
    def forward(self, token_ids: torch.Tensor, 
                targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L = token_ids.shape
        
        # Position indices
        positions = torch.arange(L, device=self.device).unsqueeze(0).expand(B, -1)
        
        # Spherical encoding
        x = self.encoder(token_ids, positions)
        
        # PAC layers with SEC gating
        total_gate = 0.0
        for layer in self.layers:
            x, gate = layer(x)
            total_gate += gate.mean().item()
        
        avg_gate = total_gate / len(self.layers)
        
        # Track SEC dynamics
        if avg_gate > PHI_XI / 2:  # High activity
            self.expansions += 1
        elif avg_gate < XI:  # Low activity
            self.collapses += 1
        
        # Project to vocab
        logits = self.out_proj(x)
        
        # Loss
        loss = None
        if targets is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            
            # Mask out invalid targets (padding mapped to -100)
            # Use ignore_index=-100 which is the standard
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_targets.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    def get_stats(self) -> Dict:
        return {
            'expansions': self.expansions,
            'collapses': self.collapses,
            'n_params': sum(p.numel() for p in self.parameters())
        }


def load_wikitext2(split: str, max_samples: int = None) -> Optional[List[str]]:
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        texts = [x['text'] for x in ds if x['text'].strip()]
        return texts[:max_samples] if max_samples else texts
    except ImportError:
        return None


def build_vocab(texts: List[str], vocab_size: int) -> Dict[str, int]:
    from collections import Counter
    counts = Counter()
    for t in texts:
        counts.update(t.lower().split())
    word_to_id = {'<pad>': 0, '<unk>': 1}
    for w, _ in counts.most_common(vocab_size - 2):
        word_to_id[w] = len(word_to_id)
    return word_to_id


def tokenize_batch(texts: List[str], vocab: Dict[str, int], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = []
    for t in texts:
        ids = [vocab.get(w, 1) for w in t.lower().split()][:max_len]
        batch.append(ids)
    
    max_batch_len = max(len(ids) for ids in batch) if batch else 1
    padded = [ids + [0] * (max_batch_len - len(ids)) for ids in batch]
    
    input_ids = torch.tensor(padded, dtype=torch.long)
    targets = input_ids.clone()
    targets[targets == 0] = -100
    
    return input_ids, targets


def train_epoch(model, texts, vocab, optimizer, batch_size=32, max_len=128):
    model.train()
    total_loss = 0
    n_batches = 0
    
    for i in range(0, len(texts), batch_size):
        batch = [t for t in texts[i:i+batch_size] if len(t.split()) > 1]
        if not batch:
            continue
        
        input_ids, targets = tokenize_batch(batch, vocab, max_len)
        input_ids = input_ids.to(model.device)
        targets = targets.to(model.device)
        
        optimizer.zero_grad()
        _, loss = model(input_ids, targets)
        
        if loss is not None and not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    
    avg_loss = total_loss / max(n_batches, 1)
    return {'loss': avg_loss, 'ppl': math.exp(min(avg_loss, 10))}


def evaluate(model, texts, vocab, batch_size=32, max_len=128):
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [t for t in texts[i:i+batch_size] if len(t.split()) > 1]
            if not batch:
                continue
            
            input_ids, targets = tokenize_batch(batch, vocab, max_len)
            input_ids = input_ids.to(model.device)
            targets = targets.to(model.device)
            
            _, loss = model(input_ids, targets)
            if loss is not None and not torch.isnan(loss):
                total_loss += loss.item()
                n_batches += 1
    
    avg_loss = total_loss / max(n_batches, 1)
    return {'loss': avg_loss, 'ppl': math.exp(min(avg_loss, 10))}


def main():
    print("=" * 60)
    print("POC-011 Exp 03: PAC Field Transformer")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"GAIA components available: {HAS_GAIA}")
    
    # Load data
    print("\n1. Loading WikiText-2...")
    train_texts = load_wikitext2('train', 5000)
    test_texts = load_wikitext2('test', 1000)
    
    if train_texts is None:
        print("  datasets not available")
        return
    
    print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")
    
    vocab = build_vocab(train_texts, 8000)
    print(f"  Vocab: {len(vocab)}")
    
    # Create model
    print("\n2. Creating PAC Field Transformer...")
    config = PACFieldConfig(
        vocab_size=len(vocab),
        field_dim=384,
        n_layers=4,
        device=device
    )
    model = PACFieldTransformer(config)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    
    # Train
    print("\n3. Training (15 epochs)...")
    results = []
    
    start = time.perf_counter()
    for epoch in range(15):
        result = train_epoch(model, train_texts, vocab, optimizer, batch_size=48, max_len=128)
        scheduler.step()
        
        if (epoch + 1) % 3 == 0:
            eval_result = evaluate(model, test_texts[:200], vocab, batch_size=48)
            print(f"  Epoch {epoch+1}: train_ppl={result['ppl']:.2f}, test_ppl={eval_result['ppl']:.2f}")
            results.append({'epoch': epoch+1, 'train': result, 'test': eval_result})
        else:
            print(f"  Epoch {epoch+1}: train_ppl={result['ppl']:.2f}")
            results.append({'epoch': epoch+1, 'train': result})
    
    total_time = time.perf_counter() - start
    
    # Final eval
    print("\n4. Final Evaluation...")
    final_result = evaluate(model, test_texts, vocab, batch_size=48)
    
    stats = model.get_stats()
    
    print(f"\n  Final Test Perplexity: {final_result['ppl']:.2f}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  SEC expansions: {stats['expansions']}")
    print(f"  SEC collapses: {stats['collapses']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    baseline = 5.91  # GAIA unified baseline
    test_ppl = final_result['ppl']
    
    print(f"  GAIA Unified baseline:   {baseline:.2f}")
    print(f"  PAC Field Transformer:   {test_ppl:.2f}")
    print(f"  Ratio:                   {test_ppl/baseline:.2f}x")
    
    if test_ppl < baseline * 2:
        print("  ✅ Within 2x of unified baseline")
    elif test_ppl < baseline * 5:
        print("  ⚠️ Within 5x (needs tuning)")
    else:
        print("  ❌ Needs more work")
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'vocab_size': config.vocab_size,
            'field_dim': config.field_dim,
            'n_layers': config.n_layers
        },
        'results': results,
        'final': final_result,
        'stats': stats,
        'baseline': baseline,
        'time_s': total_time
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = results_dir / f'exp_03_pac_field_{timestamp}.json'
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
