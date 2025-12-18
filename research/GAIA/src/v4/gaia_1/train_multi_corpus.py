"""
GAIA-1 Multi-Corpus Training with Generalization Monitoring

Train on diverse corpora with real-time monitoring to prevent memorization:
- WikiText-2: Encyclopedic knowledge
- DailyDialog: Conversational patterns  
- TinyStories: Narrative structure

Uses fracton.monitoring to detect overfitting and nurture generalization.

Usage:
    python train_multi_corpus.py --epochs 10
    python train_multi_corpus.py --epochs 5 --monitor_interval 50
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import json
import sys
import hashlib

# Fracton imports
_fracton_path = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "fracton"
if _fracton_path.exists():
    sys.path.insert(0, str(_fracton_path))

try:
    from fracton.monitoring import (
        PACTreeMonitor,
        LanguageGeneralizationMonitor,
        GeneralizationNurturingTrainer,
        SCBFBridge,
        visualize_tree_health,
        create_training_dashboard
    )
    HAS_MONITORING = True
except ImportError as e:
    HAS_MONITORING = False
    print(f"Warning: fracton.monitoring not available ({e}), running without monitoring")

from model import GAIA1, GAIA1Config


# =============================================================================
# Dataset Classes
# =============================================================================

class TextDataset(Dataset):
    """Text dataset with corpus type tracking for monitoring."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        corpus_type: str,
        max_length: int = 256,
        min_length: int = 16
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.corpus_type = corpus_type
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
                if len(remaining) < max_length:
                    eos_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 0
                    remaining = remaining + [eos_id] * (max_length - len(remaining))
                self.examples.append(torch.tensor(remaining[:max_length], dtype=torch.long))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.examples[idx],
            'corpus_type': self.corpus_type
        }


class MultiCorpusDataset(Dataset):
    """Combines multiple corpus datasets with balanced sampling."""
    
    def __init__(self, datasets: List[TextDataset]):
        self.datasets = datasets
        self.corpus_types = [d.corpus_type for d in datasets]
        
        # Create interleaved indices for balanced sampling
        self.indices = []
        max_len = max(len(d) for d in datasets)
        
        for i in range(max_len):
            for j, ds in enumerate(datasets):
                if i < len(ds):
                    self.indices.append((j, i))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        ds_idx, item_idx = self.indices[idx]
        return self.datasets[ds_idx][item_idx]


# =============================================================================
# Corpus Loaders
# =============================================================================

def load_wikitext(split: str = 'train', max_samples: int = None) -> List[str]:
    """Load WikiText-2 dataset - encyclopedic knowledge."""
    try:
        from datasets import load_dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        texts = [t for t in dataset['text'] if len(t.strip()) > 50]
        if max_samples:
            texts = texts[:max_samples]
        print(f"  WikiText-2: {len(texts)} documents")
        return texts
    except ImportError:
        print("Install datasets: pip install datasets")
        raise


def load_dailydialog(split: str = 'train', max_samples: int = None) -> List[str]:
    """Load DailyDialog dataset - conversational patterns."""
    try:
        from datasets import load_dataset
        dataset = load_dataset('daily_dialog', split=split, trust_remote_code=True)
        
        texts = []
        for dialog in dataset['dialog']:
            # Join dialog turns with speaker markers
            conversation = []
            for i, turn in enumerate(dialog):
                speaker = "A" if i % 2 == 0 else "B"
                conversation.append(f"{speaker}: {turn}")
            texts.append("\n".join(conversation))
        
        if max_samples:
            texts = texts[:max_samples]
        print(f"  DailyDialog: {len(texts)} conversations")
        return texts
    except Exception as e:
        print(f"  DailyDialog not available: {e}")
        return load_fallback_conversations()


def load_tinystories(split: str = 'train', max_samples: int = 5000) -> List[str]:
    """Load TinyStories dataset - narrative structure."""
    try:
        from datasets import load_dataset
        dataset = load_dataset('roneneldan/TinyStories', split=split)
        
        texts = [t['text'] for t in dataset if len(t['text'].strip()) > 100]
        if max_samples:
            texts = texts[:max_samples]
        print(f"  TinyStories: {len(texts)} stories")
        return texts
    except Exception as e:
        print(f"  TinyStories not available: {e}")
        return load_fallback_stories()


def load_fallback_conversations() -> List[str]:
    """Fallback conversation data if DailyDialog not available."""
    conversations = [
        "A: Hello, how are you doing today?\nB: I'm doing well, thank you for asking. How about yourself?\nA: Pretty good! I was thinking about going to the park later.\nB: That sounds lovely. The weather is perfect for it.",
        "A: Did you finish the report?\nB: Yes, I sent it to the manager this morning.\nA: Great work! What did she say?\nB: She said she'll review it and get back to us by Friday.",
        "A: What's your favorite type of music?\nB: I really enjoy jazz and classical music. How about you?\nA: I'm more into rock and indie music.\nB: That's cool! Music taste is so personal.",
        "A: Have you tried the new restaurant downtown?\nB: Not yet, but I've heard great things about it.\nA: We should go together sometime!\nB: That would be fun. How about this weekend?",
        "A: I'm having trouble with my computer.\nB: What seems to be the problem?\nA: It keeps freezing randomly.\nB: Have you tried restarting it? That often helps.",
    ] * 200
    return conversations


def load_fallback_stories() -> List[str]:
    """Fallback story data if TinyStories not available."""
    stories = [
        "Once upon a time, there was a little rabbit who lived in a cozy burrow. Every morning, the rabbit would hop out to find carrots in the garden. One day, the rabbit met a friendly squirrel who taught it how to climb trees. They became the best of friends and had many adventures together.",
        "There was a young girl named Lily who loved to paint. She would spend hours creating colorful pictures of flowers and butterflies. One day, her paintings came to life and danced around her room. Lily was amazed and invited all her friends to see the magical art.",
        "A curious kitten named Whiskers explored the big house. In the attic, Whiskers found a box full of old toys. Among them was a teddy bear that seemed to wink at the kitten. From that day on, Whiskers and the teddy bear became inseparable companions.",
        "In a small pond lived a wise old frog. Young tadpoles would come to learn from the frog every day. The frog taught them about the world beyond the water. When they grew into frogs themselves, they thanked their teacher and hopped off to explore.",
        "There was a robot who wanted to understand feelings. It watched humans laugh and cry, trying to learn. One day, when a child was sad, the robot gave a gentle hug. The child smiled, and the robot finally understood what happiness meant.",
    ] * 200
    return stories


# =============================================================================
# Monitored Training
# =============================================================================

class MonitoredTrainer:
    """Trainer with integrated generalization monitoring."""
    
    def __init__(
        self,
        model: GAIA1,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: str,
        monitor: Optional[LanguageGeneralizationMonitor] = None,
        scbf_bridge: Optional[SCBFBridge] = None,
        accumulation_steps: int = 1,
        monitor_interval: int = 100,
        verbose: bool = True
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.monitor = monitor
        self.scbf_bridge = scbf_bridge
        self.accumulation_steps = accumulation_steps
        self.monitor_interval = monitor_interval
        self.verbose = verbose
        
        self.step_count = 0
        self.total_tokens = 0
        self.active_interventions = []
        
        # Tracking
        self.corpus_losses: Dict[str, List[float]] = {}
        self.health_history = []
    
    def training_step(self, batch: Dict[str, Any]) -> float:
        """Execute one training step with monitoring."""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        corpus_type = batch['corpus_type'][0]  # Batch should be same corpus
        
        # Forward pass
        loss = self.model.compute_loss(input_ids)
        
        # Backward pass
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        if (self.step_count + 1) % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()
        
        self.step_count += 1
        self.total_tokens += input_ids.numel()
        
        # Track loss by corpus
        if corpus_type not in self.corpus_losses:
            self.corpus_losses[corpus_type] = []
        self.corpus_losses[corpus_type].append(loss.item())
        
        # Monitoring
        if self.monitor and self.step_count % self.monitor_interval == 0:
            self._record_monitoring(input_ids, corpus_type, loss.item())
        
        return loss.item()
    
    def _record_monitoring(
        self, 
        input_ids: torch.Tensor, 
        corpus_type: str,
        loss: float
    ):
        """Record activation patterns for monitoring."""
        # Create context ID from batch content
        batch_hash = hashlib.md5(input_ids.cpu().numpy().tobytes()).hexdigest()[:8]
        context_id = f"{corpus_type}_{batch_hash}"
        
        # Record activations for unique tokens in batch
        unique_tokens = input_ids.unique().tolist()
        for token_id in unique_tokens[:50]:  # Sample top 50
            pattern_id = f"tok_{token_id}"
            self.monitor.record_activation(pattern_id, context_id)
        
        self.monitor.step()
        
        # Periodic health check
        if self.step_count % (self.monitor_interval * 5) == 0:
            self._print_health_report(loss)
    
    def _print_health_report(self, loss: float):
        """Print health report with monitoring data."""
        health = self.monitor.get_health_summary()
        self.health_history.append(health)
        
        metrics = self.monitor.get_tree_metrics()
        
        if self.scbf_bridge:
            scbf = self.scbf_bridge.get_tree_health(self.monitor.tree_monitor)
        else:
            scbf = None
        
        print("\n" + "="*60)
        print(visualize_tree_health(metrics, self.monitor.zones, scbf))
        
        # Corpus-specific losses
        print("\nLoss by Corpus (recent):")
        for corpus, losses in self.corpus_losses.items():
            recent = losses[-100:] if len(losses) > 100 else losses
            avg = sum(recent) / len(recent)
            print(f"  {corpus}: {avg:.4f}")
        
        # Recommendations
        if scbf:
            recs = self.scbf_bridge.get_recommendations(scbf)
            print("\nRecommendations:")
            for rec in recs:
                print(f"  • {rec}")
        
        print("="*60 + "\n")
    
    def get_corpus_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get per-corpus training metrics."""
        result = {}
        for corpus, losses in self.corpus_losses.items():
            result[corpus] = {
                'samples': len(losses),
                'avg_loss': sum(losses) / len(losses) if losses else 0,
                'recent_loss': sum(losses[-100:]) / len(losses[-100:]) if losses else 0,
                'min_loss': min(losses) if losses else 0,
            }
        return result


# =============================================================================
# Main Training Loop
# =============================================================================

def train(
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    max_samples_per_corpus: int = 5000,
    monitor_interval: int = 100,
    checkpoint_dir: str = './checkpoints/multi_corpus',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Main training function with multi-corpus and monitoring."""
    
    print("="*60)
    print("GAIA-1 Multi-Corpus Training with Generalization Monitoring")
    print("="*60)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Monitor interval: {monitor_interval}")
    print()
    
    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    print("Initializing model...")
    config = GAIA1Config(
        vocab_size=50257,
        field_dim=256,
        n_layers=4,
        max_context=256,
        evolution_steps=8,
        device=device
    )
    model = GAIA1(config).to(device)
    tokenizer = model.tokenizer  # Use GPT-2 tokenizer
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load corpora
    print("\nLoading corpora...")
    
    wiki_texts = load_wikitext('train', max_samples_per_corpus)
    dialog_texts = load_dailydialog('train', max_samples_per_corpus)
    story_texts = load_tinystories('train', max_samples_per_corpus)
    
    # Create datasets
    print("\nCreating datasets...")
    
    wiki_dataset = TextDataset(wiki_texts, tokenizer, 'wiki', max_length=256)
    dialog_dataset = TextDataset(dialog_texts, tokenizer, 'dialog', max_length=256)
    story_dataset = TextDataset(story_texts, tokenizer, 'story', max_length=256)
    
    print(f"  Wiki examples: {len(wiki_dataset)}")
    print(f"  Dialog examples: {len(dialog_dataset)}")
    print(f"  Story examples: {len(story_dataset)}")
    
    # Combined dataset with balanced sampling
    combined = MultiCorpusDataset([wiki_dataset, dialog_dataset, story_dataset])
    print(f"  Combined examples: {len(combined)}")
    
    # Dataloader
    def collate_fn(batch):
        input_ids = torch.stack([b['input_ids'] for b in batch])
        corpus_types = [b['corpus_type'] for b in batch]
        return {'input_ids': input_ids, 'corpus_type': corpus_types}
    
    dataloader = DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(dataloader) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=learning_rate * 0.1)
    
    # Initialize monitoring
    if HAS_MONITORING:
        print("\nInitializing generalization monitor...")
        monitor = LanguageGeneralizationMonitor(
            pac_tree=None,  # We'll track patterns manually
            check_interval=monitor_interval,
            min_zone_size=3
        )
        scbf_bridge = SCBFBridge()
    else:
        monitor = None
        scbf_bridge = None
    
    # Create trainer
    trainer = MonitoredTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        monitor=monitor,
        scbf_bridge=scbf_bridge,
        monitor_interval=monitor_interval
    )
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0
        epoch_tokens = 0
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")
        
        for batch_idx, batch in enumerate(dataloader):
            loss = trainer.training_step(batch)
            epoch_loss += loss
            epoch_tokens += batch['input_ids'].numel()
            
            # Progress
            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - epoch_start
                tokens_per_sec = epoch_tokens / elapsed
                avg_loss = epoch_loss / (batch_idx + 1)
                
                if HAS_MONITORING:
                    dashboard = create_training_dashboard(
                        trainer.step_count,
                        loss,
                        trainer.monitor.get_tree_metrics()
                    )
                    print(dashboard)
                else:
                    print(f"  Batch {batch_idx + 1}/{len(dataloader)}: "
                          f"loss={loss:.4f}, avg={avg_loss:.4f}, "
                          f"tokens/sec={tokens_per_sec:.0f}")
        
        # Epoch summary
        epoch_elapsed = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / len(dataloader)
        tokens_per_sec = epoch_tokens / epoch_elapsed
        
        print(f"\nEpoch {epoch + 1} complete:")
        print(f"  Average loss: {avg_epoch_loss:.4f}")
        print(f"  Time: {epoch_elapsed:.1f}s")
        print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
        
        # Per-corpus metrics
        corpus_metrics = trainer.get_corpus_metrics()
        print("\n  Per-corpus performance:")
        for corpus, metrics in corpus_metrics.items():
            print(f"    {corpus}: recent_loss={metrics['recent_loss']:.4f}, "
                  f"samples={metrics['samples']}")
        
        # Save checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_path = checkpoint_path / 'gaia1_best.pt'
            model.save(save_path)
            print(f"\n  ✓ New best model saved: {save_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            save_path = checkpoint_path / f'gaia1_epoch_{epoch+1}.pt'
            model.save(save_path)
            print(f"  ✓ Checkpoint saved: {save_path}")
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Total steps: {trainer.step_count}")
    print(f"Total tokens: {trainer.total_tokens:,}")
    
    if HAS_MONITORING:
        print("\nFinal Health Report:")
        health = trainer.monitor.get_health_summary()
        print(f"  Patterns tracked: {health['total_patterns']}")
        print(f"  Specific ratio: {health['specific_ratio']:.2%}")
        print(f"  Is overfitting: {health['is_overfitting']}")
        print(f"  Generalization zones: {health['num_zones']}")
    
    # Save final model
    final_path = checkpoint_path / 'gaia1_final.pt'
    model.save(final_path)
    print(f"\nFinal model saved: {final_path}")
    
    # Save training log
    log_path = checkpoint_path / 'training_log.json'
    log = {
        'config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'max_samples_per_corpus': max_samples_per_corpus
        },
        'results': {
            'best_loss': best_loss,
            'total_time_hours': total_time / 3600,
            'total_steps': trainer.step_count,
            'total_tokens': trainer.total_tokens
        },
        'corpus_metrics': trainer.get_corpus_metrics()
    }
    
    if HAS_MONITORING:
        log['final_health'] = trainer.monitor.get_health_summary()
    
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2, default=str)
    print(f"Training log saved: {log_path}")
    
    return model, trainer


def main():
    parser = argparse.ArgumentParser(description='GAIA-1 Multi-Corpus Training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=5000, 
                       help='Max samples per corpus')
    parser.add_argument('--monitor_interval', type=int, default=100,
                       help='Steps between monitoring updates')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='./checkpoints/multi_corpus',
                       help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples_per_corpus=args.max_samples,
        monitor_interval=args.monitor_interval,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()
