"""
GAIA-1 Adaptive Multi-Corpus Training

Uses entropy-driven adaptive learning rate and PAC tree health monitoring
to dynamically adjust training for optimal generalization.

Inspired by CIMM's quantum-coherent training approach.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
import argparse
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

# Model imports
from model import GAIA1, GAIA1Config

# Adaptive trainer
from adaptive_trainer import AdaptiveGAIATrainer, AdaptiveConfig, create_adaptive_trainer


@dataclass
class CorpusStats:
    """Track per-corpus statistics."""
    name: str
    total_samples: int = 0
    recent_loss: float = 0.0
    samples_seen: int = 0


class TextDataset(Dataset):
    """Simple text dataset."""
    
    def __init__(
        self, 
        texts: List[str], 
        tokenizer, 
        max_length: int = 256,
        corpus_name: str = "unknown"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.corpus_name = corpus_name
        self.examples = []
        
        for text in texts:
            if len(text.strip()) < 10:
                continue
            tokens = tokenizer.encode(text)
            if len(tokens) >= 10:
                self.examples.append(tokens[:max_length])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        # Pad if needed
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.eos_token_id] * (self.max_length - len(tokens))
        return {
            'input_ids': torch.tensor(tokens[:self.max_length], dtype=torch.long),
            'corpus': self.corpus_name
        }


def load_wikitext(tokenizer, max_samples: int = 2000, max_length: int = 256):
    """Load WikiText-2 dataset."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [ex['text'] for ex in dataset if len(ex['text']) > 50][:max_samples]
        print(f"  WikiText-2: {len(texts)} documents")
        return TextDataset(texts, tokenizer, max_length, "wiki")
    except Exception as e:
        print(f"  WikiText-2 not available: {e}")
        return None


def load_dailydialog(tokenizer, max_samples: int = 1000, max_length: int = 256):
    """Load DailyDialog for conversational patterns."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("daily_dialog", split="train", trust_remote_code=True)
        texts = []
        for ex in dataset:
            if 'dialog' in ex:
                dialog = " ".join(ex['dialog'])
                texts.append(dialog)
            if len(texts) >= max_samples:
                break
        print(f"  DailyDialog: {len(texts)} conversations")
        return TextDataset(texts, tokenizer, max_length, "dialog")
    except Exception as e:
        print(f"  DailyDialog not available: {e}")
        # Fallback: simple conversational examples
        fallback_dialogs = [
            "Hello! How are you today? I'm doing great, thanks for asking.",
            "What do you think about the weather? It's quite nice outside.",
            "Can you help me with something? Of course, what do you need?",
        ] * 128
        return TextDataset(fallback_dialogs, tokenizer, max_length, "dialog")


def load_tinystories(tokenizer, max_samples: int = 1000, max_length: int = 256):
    """Load TinyStories for narrative patterns."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        texts = [ex['text'] for ex in dataset if len(ex['text']) > 50][:max_samples]
        print(f"  TinyStories: {len(texts)} stories")
        return TextDataset(texts, tokenizer, max_length, "story")
    except Exception as e:
        print(f"  TinyStories not available: {e}")
        # Fallback stories
        fallback_stories = [
            "Once upon a time, there was a little girl who loved to read books.",
            "The brave knight rode through the forest on his trusty horse.",
            "In a small village, there lived a kind old woman who helped everyone.",
        ] * 144
        return TextDataset(fallback_stories, tokenizer, max_length, "story")


def collate_fn(batch):
    """Collate batch with corpus tracking."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    corpora = [item['corpus'] for item in batch]
    return {'input_ids': input_ids, 'corpora': corpora}


def compute_loss(output, batch, model):
    """Compute language modeling loss."""
    # Model forward returns (logits, hidden_states)
    if isinstance(output, tuple):
        logits = output[0]  # First element is logits
    else:
        logits = output
    
    # Shift for next-token prediction
    input_ids = batch['input_ids']
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    # Cross entropy loss
    eos_id = model.tokenizer.eos_token_id if hasattr(model.tokenizer, 'eos_token_id') else -100
    loss_fn = nn.CrossEntropyLoss(ignore_index=eos_id)
    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    return loss


def train(args):
    """Main training loop with adaptive trainer."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("GAIA-1 ADAPTIVE Multi-Corpus Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Initial LR: {args.lr}")
    print(f"Adaptive: ENABLED")
    print()
    
    # Initialize model
    print("Initializing model...")
    config = GAIA1Config(
        max_context=256,
        field_dim=256,
        n_layers=4,
        evolution_steps=8
    )
    model = GAIA1(config).to(device)
    print(f"Model parameters: {model.get_parameter_count():,}")
    print()
    
    # Load corpora
    print("Loading corpora...")
    tokenizer = model.tokenizer
    
    wiki_dataset = load_wikitext(tokenizer, args.max_samples, config.max_context)
    dialog_dataset = load_dailydialog(tokenizer, args.max_samples // 2, config.max_context)
    story_dataset = load_tinystories(tokenizer, args.max_samples // 2, config.max_context)
    
    # Combine datasets
    datasets = [d for d in [wiki_dataset, dialog_dataset, story_dataset] if d is not None]
    combined_dataset = ConcatDataset(datasets)
    
    print(f"\nCreating datasets...")
    for d in datasets:
        print(f"  {d.corpus_name.capitalize()} examples: {len(d)}")
    print(f"  Combined examples: {len(combined_dataset)}")
    
    dataloader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create adaptive trainer
    print("\nInitializing adaptive trainer...")
    adaptive_config = AdaptiveConfig(
        initial_lr=args.lr,
        min_lr=1e-6,
        max_lr=1e-2,
        adaptation_interval=args.adapt_interval,
        target_entropy=0.916,  # PHI_XI
        target_abstract_ratio=0.4
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(dataloader)
    )
    
    trainer = AdaptiveGAIATrainer(model, optimizer, adaptive_config, device)
    
    # Corpus tracking
    corpus_stats = {
        'wiki': CorpusStats('wiki'),
        'dialog': CorpusStats('dialog'),
        'story': CorpusStats('story')
    }
    
    # Training state
    checkpoint_dir = Path("checkpoints/adaptive_multi_corpus")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    training_log = []
    start_time = time.time()
    
    print("\n" + "=" * 60)
    print("Starting ADAPTIVE training...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_tokens = 0
        
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 60)
        
        for batch_idx, batch in enumerate(dataloader, 1):
            # Move to device
            batch['input_ids'] = batch['input_ids'].to(device)
            
            # Adaptive training step
            loss, metrics = trainer.training_step(
                batch,
                lambda out, b: compute_loss(out, b, model)
            )
            
            # Track per-corpus loss
            for corpus_name in batch['corpora']:
                if corpus_name in corpus_stats:
                    corpus_stats[corpus_name].samples_seen += 1
                    # Exponential moving average
                    alpha = 0.1
                    corpus_stats[corpus_name].recent_loss = (
                        alpha * loss.item() + 
                        (1 - alpha) * corpus_stats[corpus_name].recent_loss
                    )
            
            # Update scheduler (base scheduler, adaptive overlays on top)
            scheduler.step()
            
            epoch_loss += loss.item()
            batch_tokens = batch['input_ids'].numel()
            epoch_tokens += batch_tokens
            
            # Progress logging
            if batch_idx % args.log_interval == 0:
                elapsed = time.time() - epoch_start
                tokens_per_sec = epoch_tokens / elapsed
                
                print(f"  Batch {batch_idx}/{len(dataloader)}: "
                      f"loss={loss.item():.4f}, "
                      f"lr={metrics['lr']:.2e}, "
                      f"entropy={metrics['entropy']:.3f}, "
                      f"abstract={metrics['abstract_ratio']:.3f}, "
                      f"gen_score={metrics['generalization_score']:.3f}")
            
            # Detailed status report
            if batch_idx % args.status_interval == 0:
                print(trainer.get_status_report())
        
        # Epoch complete
        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch} complete:")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Throughput: {epoch_tokens / epoch_time:.0f} tokens/sec")
        print(f"  LR adjustments: {trainer.state.lr_adjustments}")
        print(f"  Noise injections: {trainer.state.noise_injections}")
        
        print(f"\n  Per-corpus performance:")
        for name, stats in corpus_stats.items():
            print(f"    {name}: recent_loss={stats.recent_loss:.4f}, samples={stats.samples_seen}")
        
        # Log
        training_log.append({
            'epoch': epoch,
            'avg_loss': avg_loss,
            'lr': trainer.state.current_lr,
            'entropy': trainer.entropy_tracker.prev_entropy,
            'abstract_ratio': trainer.pac_tracker._compute_abstract_ratio(),
            'generalization_score': trainer.state.generalization_score,
            'lr_adjustments': trainer.state.lr_adjustments,
            'noise_injections': trainer.state.noise_injections,
            'corpus_stats': {k: {'loss': v.recent_loss, 'samples': v.samples_seen} 
                           for k, v in corpus_stats.items()}
        })
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save(checkpoint_dir / "gaia1_best.pt")
            print(f"  ✓ New best model saved: {checkpoint_dir / 'gaia1_best.pt'}")
        
        # Periodic checkpoint
        if epoch % 10 == 0:
            model.save(checkpoint_dir / f"gaia1_epoch_{epoch}.pt")
            print(f"  ✓ Checkpoint saved: {checkpoint_dir / f'gaia1_epoch_{epoch}.pt'}")
    
    # Training complete
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Total LR adjustments: {trainer.state.lr_adjustments}")
    print(f"Total noise injections: {trainer.state.noise_injections}")
    print(f"Final generalization score: {trainer.state.generalization_score:.4f}")
    
    # Save final model
    model.save(checkpoint_dir / "gaia1_final.pt")
    print(f"\nFinal model saved: {checkpoint_dir / 'gaia1_final.pt'}")
    
    # Save training log
    log_path = checkpoint_dir / "training_log.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"Training log saved: {log_path}")
    
    # Save adaptation log
    trainer.save_adaptation_log(checkpoint_dir / "adaptation_log.json")
    print(f"Adaptation log saved: {checkpoint_dir / 'adaptation_log.json'}")
    
    # Final status report
    print(trainer.get_status_report())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAIA-1 Adaptive Training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--max_samples", type=int, default=2000, help="Max samples per corpus")
    parser.add_argument("--log_interval", type=int, default=25, help="Log every N batches")
    parser.add_argument("--status_interval", type=int, default=100, help="Status report every N batches")
    parser.add_argument("--adapt_interval", type=int, default=10, help="Adapt every N batches")
    
    args = parser.parse_args()
    train(args)
