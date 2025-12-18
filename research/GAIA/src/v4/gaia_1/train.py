"""
GAIA-1 Training Script

Train the field-native language model on text data.
Optimized for overnight training on a single GPU.

Usage:
    python train.py --epochs 10 --batch_size 16
    python train.py --dataset wikitext --epochs 20
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import json
import sys

# Fracton imports
try:
    import fracton
except ImportError:
    _fracton_path = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "fracton"
    if _fracton_path.exists():
        sys.path.insert(0, str(_fracton_path))

from model import GAIA1, GAIA1Config


class TextDataset(Dataset):
    """Simple text dataset for training."""
    
    def __init__(
        self,
        texts: list,
        tokenizer,
        max_length: int = 256,
        min_length: int = 16
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.examples = []
        
        # Tokenize all texts and concatenate
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
        
        # Chunk into max_length sequences with overlap
        stride = max_length // 2
        for i in range(0, len(all_tokens) - max_length, stride):
            chunk = all_tokens[i:i + max_length]
            self.examples.append(torch.tensor(chunk, dtype=torch.long))
        
        # Handle remaining tokens
        if len(all_tokens) >= min_length:
            remaining = all_tokens[-(max_length):]
            if len(remaining) >= min_length:
                # Pad if necessary
                if len(remaining) < max_length:
                    remaining = remaining + [tokenizer.eos_token_id] * (max_length - len(remaining))
                self.examples.append(torch.tensor(remaining[:max_length], dtype=torch.long))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def load_wikitext(split: str = 'train') -> list:
    """Load WikiText-2 dataset."""
    try:
        from datasets import load_dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        texts = [t for t in dataset['text'] if len(t.strip()) > 50]
        return texts
    except ImportError:
        print("Install datasets: pip install datasets")
        raise


def load_simple_corpus() -> list:
    """Load a simple test corpus."""
    # Simple training data for testing - needs to be longer for chunking
    base_texts = [
        "The quick brown fox jumps over the lazy dog. The dog was sleeping peacefully in the warm afternoon sun. When the fox jumped, the dog woke up startled and began to bark loudly. The fox ran away into the forest, never to be seen again.",
        "In the beginning was the word, and the word was with the field. The field expanded and contracted according to the laws of physics. Energy flowed through the substrate, creating patterns of meaning. These patterns crystallized into symbols over time.",
        "Hello world. This is a test of the field-native language model. The model processes text by encoding tokens into field patterns. These patterns evolve according to Klein-Gordon dynamics. The final state is decoded via resonance with vocabulary patterns.",
        "The field evolves according to Klein-Gordon dynamics. The equation describes how waves propagate through spacetime. In our model, the field represents the state of consciousness. As it evolves, meaning emerges from the interplay of patterns.",
        "Intelligence emerges from conservation and balance. The system maintains PAC conservation at all times. This means that the sum of parts equals the whole. Through this constraint, structure naturally arises from chaos.",
        "Patterns crystallize when entropy collapses below threshold. This is the mechanism of learning and memory. Important patterns are marked for persistence. They survive even when the system is restarted.",
        "The consciousness field integrates all organ outputs. Each organ specializes in a different aspect of cognition. The language organ processes text. The reasoning organ handles logic. The memory organ stores experiences.",
        "Learning happens continuously, not in separate training phases. Every interaction modifies the substrate. High importance patterns are automatically persisted. The system grows smarter with each conversation.",
        "The universe is a field of pure potential. Matter and energy are manifestations of field dynamics. Consciousness may emerge from similar principles. Our model explores this hypothesis computationally.",
        "Language is the crystallized form of thought. Words represent compressed patterns of meaning. Grammar emerges from the structure of experience. Semantics arise from resonance between concepts.",
    ]
    # Repeat for more training data
    texts = base_texts * 100
    return texts


def train_epoch(
    model: GAIA1,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: str,
    accumulation_steps: int = 1
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch.to(device)
        
        # Forward pass
        loss = model.compute_loss(input_ids)
        
        # Backward pass (with accumulation)
        loss = loss / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        total_tokens += input_ids.numel()
        
        # Progress
        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: "
                  f"loss={loss.item() * accumulation_steps:.4f}, "
                  f"tokens/sec={tokens_per_sec:.0f}")
    
    avg_loss = total_loss / len(dataloader)
    elapsed = time.time() - start_time
    tokens_per_sec = total_tokens / elapsed
    
    return avg_loss, tokens_per_sec


@torch.no_grad()
def evaluate(
    model: GAIA1,
    dataloader: DataLoader,
    device: str
) -> Tuple[float, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for batch in dataloader:
        input_ids = batch.to(device)
        loss = model.compute_loss(input_ids)
        total_loss += loss.item()
        total_tokens += input_ids.numel()
    
    avg_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity


def sample_generation(model: GAIA1, prompts: list):
    """Generate samples from prompts."""
    print("\n📝 Sample generations:")
    print("-" * 50)
    
    for prompt in prompts:
        output = model.generate(prompt, max_tokens=50, temperature=0.8)
        print(f"Prompt: {prompt}")
        print(f"Output: {output}")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description='Train GAIA-1')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--field_dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--max_context', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='simple', 
                        choices=['simple', 'wikitext'])
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--eval_every', type=int, default=1)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Device: {device}")
    
    # Create config
    config = GAIA1Config(
        field_dim=args.field_dim,
        n_layers=args.n_layers,
        max_context=args.max_context,
        learning_rate=args.lr,
        device=device
    )
    
    # Create model
    print("\n🌀 Creating GAIA-1 model...")
    model = GAIA1(config)
    print(f"Model: {model}")
    print(f"Parameters: {model.get_parameter_count():,}")
    
    # Load data
    print(f"\n📚 Loading {args.dataset} dataset...")
    if args.dataset == 'wikitext':
        train_texts = load_wikitext('train')
        val_texts = load_wikitext('validation')
    else:
        texts = load_simple_corpus()
        split = int(len(texts) * 0.9)
        train_texts = texts[:split]
        val_texts = texts[split:]
    
    print(f"Train texts: {len(train_texts)}")
    print(f"Val texts: {len(val_texts)}")
    
    # Create datasets
    train_dataset = TextDataset(train_texts, model.tokenizer, args.max_context)
    val_dataset = TextDataset(val_texts, model.tokenizer, args.max_context)
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print(f"Total steps: {total_steps}")
    print("-" * 60)
    
    best_val_loss = float('inf')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    training_log = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n📊 Epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, tokens_per_sec = train_epoch(
            model, train_loader, optimizer, scheduler, 
            device, args.accumulation_steps
        )
        
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Tokens/sec: {tokens_per_sec:.0f}")
        
        # Evaluate
        if epoch % args.eval_every == 0:
            val_loss, perplexity = evaluate(model, val_loader, device)
            print(f"  Val loss: {val_loss:.4f}")
            print(f"  Perplexity: {perplexity:.2f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = save_dir / "gaia1_best.pt"
                model.save(checkpoint_path)
                print(f"  💾 Saved best model to {checkpoint_path}")
        else:
            val_loss = None
            perplexity = None
        
        # Log
        training_log.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'perplexity': perplexity,
            'tokens_per_sec': tokens_per_sec
        })
        
        # Sample generation every few epochs
        if epoch % 5 == 0:
            sample_generation(model, [
                "The field",
                "Hello,",
                "In the"
            ])
    
    # Save final model
    final_path = save_dir / "gaia1_final.pt"
    model.save(final_path)
    print(f"\n💾 Saved final model to {final_path}")
    
    # Save training log
    log_path = save_dir / "training_log.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print("\n✅ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Final samples
    sample_generation(model, [
        "Hello, my name is",
        "The meaning of life is",
        "In the beginning",
        "The quick brown fox"
    ])


if __name__ == "__main__":
    main()
