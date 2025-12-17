"""
GAIA Corpus Trainer
====================

Train GAIA on real text corpora with streaming data loading.

Supported corpora:
- WikiText-2/103 (via HuggingFace datasets)
- Custom text files
- Streaming from large files

Key features:
- Incremental vocabulary building
- Batched training
- Checkpoint saving/loading
- Progress tracking with perplexity
"""

import torch
import json
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator, List, Tuple
from dataclasses import dataclass, field
import sys

# Add src to path
src_path = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(src_path))

from gaia_unified import GAIAUnified, GAIAConfig


@dataclass
class TrainingConfig:
    """Configuration for corpus training."""
    # Corpus settings
    corpus_name: str = "wikitext-2"
    max_vocab_size: int = 10000
    min_word_freq: int = 2
    
    # Training settings
    batch_size: int = 32
    max_sentences: Optional[int] = None  # None = all
    repetitions: int = 3  # Passes over corpus
    
    # Checkpointing
    checkpoint_every: int = 1000
    checkpoint_dir: str = "checkpoints"
    
    # Evaluation
    eval_every: int = 500
    eval_samples: int = 100
    
    # Model settings
    field_shape: Tuple[int, int, int] = (24, 24, 24)
    memory_capacity: int = 50000


@dataclass
class TrainingStats:
    """Track training progress."""
    sentences_processed: int = 0
    tokens_processed: int = 0
    transitions_learned: int = 0
    vocab_size: int = 0
    elapsed_seconds: float = 0.0
    perplexity_history: List[Tuple[int, float]] = field(default_factory=list)
    
    def to_dict(self):
        return {
            'sentences_processed': self.sentences_processed,
            'tokens_processed': self.tokens_processed,
            'transitions_learned': self.transitions_learned,
            'vocab_size': self.vocab_size,
            'elapsed_seconds': self.elapsed_seconds,
            'perplexity_history': self.perplexity_history
        }


class CorpusLoader:
    """Load and stream corpus data."""
    
    def __init__(self, corpus_name: str, split: str = "train"):
        self.corpus_name = corpus_name
        self.split = split
        self._dataset = None
        
    def _load_huggingface(self):
        """Load corpus from HuggingFace datasets."""
        try:
            from datasets import load_dataset
            
            if self.corpus_name == "wikitext-2":
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=self.split)
            elif self.corpus_name == "wikitext-103":
                ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=self.split)
            elif self.corpus_name == "tiny_shakespeare":
                ds = load_dataset("tiny_shakespeare", split=self.split)
            else:
                # Try loading directly
                ds = load_dataset(self.corpus_name, split=self.split)
                
            return ds
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
    
    def sentences(self, max_sentences: Optional[int] = None) -> Iterator[str]:
        """Yield sentences from corpus."""
        if self._dataset is None:
            self._dataset = self._load_huggingface()
            
        count = 0
        for item in self._dataset:
            text = item.get('text', item.get('content', str(item)))
            
            # Split into sentences (simple split on . ! ?)
            for sentence in re.split(r'[.!?]+', text):
                sentence = sentence.strip()
                if len(sentence) > 10:  # Skip very short
                    yield sentence
                    count += 1
                    if max_sentences and count >= max_sentences:
                        return


class VocabularyBuilder:
    """Build vocabulary from corpus with frequency filtering."""
    
    def __init__(self, max_size: int = 10000, min_freq: int = 2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word_counts = {}
        
    def add_sentence(self, sentence: str):
        """Add sentence to word counts."""
        tokens = self._tokenize(sentence)
        for token in tokens:
            self.word_counts[token] = self.word_counts.get(token, 0) + 1
            
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        # Remove punctuation, lowercase, split
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()
    
    def build(self) -> List[str]:
        """Return vocabulary sorted by frequency."""
        # Filter by min frequency
        filtered = {w: c for w, c in self.word_counts.items() if c >= self.min_freq}
        
        # Sort by frequency
        sorted_words = sorted(filtered.items(), key=lambda x: -x[1])
        
        # Take top N
        vocab = [w for w, c in sorted_words[:self.max_size]]
        
        return vocab


class GAIACorpusTrainer:
    """Train GAIA on text corpus."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.stats = TrainingStats()
        self.model: Optional[GAIAUnified] = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def build_vocabulary(self):
        """Build vocabulary from corpus."""
        print(f"\n{'='*60}")
        print(f"Building vocabulary from {self.config.corpus_name}")
        print(f"{'='*60}")
        
        loader = CorpusLoader(self.config.corpus_name)
        builder = VocabularyBuilder(
            max_size=self.config.max_vocab_size,
            min_freq=self.config.min_word_freq
        )
        
        start = time.time()
        count = 0
        
        for sentence in loader.sentences(self.config.max_sentences):
            builder.add_sentence(sentence)
            count += 1
            if count % 10000 == 0:
                print(f"  Processed {count} sentences, {len(builder.word_counts)} unique words")
                
        vocab = builder.build()
        elapsed = time.time() - start
        
        print(f"\nVocabulary built:")
        print(f"  Total unique words: {len(builder.word_counts)}")
        print(f"  After filtering (freq >= {self.config.min_word_freq}): {len(vocab)}")
        print(f"  Time: {elapsed:.1f}s")
        
        return vocab
    
    def initialize_model(self, vocab: List[str]):
        """Initialize GAIA model with vocabulary."""
        print(f"\n{'='*60}")
        print(f"Initializing GAIA model")
        print(f"{'='*60}")
        
        gaia_config = GAIAConfig(
            field_shape=self.config.field_shape,
            memory_capacity=self.config.memory_capacity,
            device=self.device
        )
        
        self.model = GAIAUnified(gaia_config)
        self.model = self.model.to(self.device)
        
        # Add vocabulary
        start = time.time()
        self.model.add_tokens(vocab)
        elapsed = time.time() - start
        
        self.stats.vocab_size = self.model.vocab_size
        
        print(f"  Vocabulary size: {self.model.vocab_size}")
        print(f"  Field shape: {self.config.field_shape}")
        print(f"  Memory capacity: {self.config.memory_capacity}")
        print(f"  Device: {self.device}")
        print(f"  Token encoding time: {elapsed:.2f}s")
        
    def _tokenize(self, sentence: str) -> List[str]:
        """Tokenize sentence (matching vocabulary builder)."""
        text = re.sub(r'[^\w\s]', '', sentence.lower())
        return text.split()
    
    @torch.no_grad()
    def train_sentence(self, tokens: List[str]):
        """Train on a single sentence."""
        # Filter to known vocabulary
        known_tokens = [t for t in tokens if t in self.model.token_to_id]
        
        if len(known_tokens) >= 2:
            # Direct transition learning without re-encoding
            ids = [self.model.token_to_id[t] for t in known_tokens]
            for i in range(len(ids) - 1):
                self.model.memory.learn_transition(ids[i], ids[i + 1])
            self.stats.tokens_processed += len(known_tokens)
            
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate current perplexity on sample - fast version."""
        import math
        
        # Rebuild cache for accurate retrieval
        self.model.memory._cache_dirty = True
        self.model.memory._rebuild_cache()
        
        # Use cached validation data
        if not hasattr(self, '_val_sentences'):
            loader = CorpusLoader(self.config.corpus_name, split="validation")
            self._val_sentences = list(loader.sentences(self.config.eval_samples))
        
        total_log_prob = 0.0
        total_tokens = 0
        hits = 0
        
        for sentence in self._val_sentences[:50]:  # Faster: only 50 sentences
            tokens = self._tokenize(sentence)
            known = [t for t in tokens if t in self.model.token_to_id]
            
            if len(known) >= 3:
                self.model.clear_context()
                
                for j in range(len(known) - 1):
                    self.model.push_context(known[j])
                    
                    if j >= 1:
                        # Fast: only get top 100, not all vocab
                        preds = self.model.predict(top_k=100)
                        
                        next_token = known[j + 1]
                        prob = 1e-6  # Smoothing for OOV
                        
                        for tok, p in preds:
                            if tok == next_token:
                                prob = max(p, 1e-10)
                                hits += 1
                                break
                        
                        total_log_prob += math.log(prob)
                        total_tokens += 1
                            
        if total_tokens > 0:
            avg_log_prob = total_log_prob / total_tokens
            perplexity = math.exp(-avg_log_prob)
        else:
            perplexity = float('inf')
            
        return perplexity
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        # Transitions are (from_id, to_id) -> weight
        transitions_serializable = {
            f"{k[0]},{k[1]}": v 
            for k, v in self.model.memory.transitions.items()
        }
        
        checkpoint = {
            'vocab': list(self.model.token_to_id.keys()),
            'transitions': transitions_serializable,
            'stats': self.stats.to_dict(),
            'config': {
                'corpus_name': self.config.corpus_name,
                'field_shape': self.config.field_shape,
                'memory_capacity': self.config.memory_capacity
            },
            'timestamp': datetime.now().isoformat()
        }
        
        path = self.checkpoint_dir / f"{name}.json"
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
            
        print(f"  Checkpoint saved: {path}")
        
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        with open(path) as f:
            checkpoint = json.load(f)
            
        # Initialize model
        vocab = checkpoint['vocab']
        self.initialize_model(vocab)
        
        # Restore transitions - keys are "from_id,to_id" strings
        for k, weight in checkpoint['transitions'].items():
            parts = k.split(',')
            from_id, to_id = int(parts[0]), int(parts[1])
            self.model.memory.transitions[(from_id, to_id)] = weight
            
        # Restore stats
        stats = checkpoint['stats']
        self.stats.sentences_processed = stats['sentences_processed']
        self.stats.tokens_processed = stats['tokens_processed']
        self.stats.transitions_learned = stats['transitions_learned']
        self.stats.vocab_size = stats['vocab_size']
        self.stats.perplexity_history = stats.get('perplexity_history', [])
        
        print(f"Checkpoint loaded: {path}")
        print(f"  Sentences: {self.stats.sentences_processed}")
        print(f"  Transitions: {len(checkpoint['transitions'])}")
        
    def train(self):
        """Run full training loop."""
        print(f"\n{'='*60}")
        print(f"Training GAIA on {self.config.corpus_name}")
        print(f"{'='*60}")
        
        loader = CorpusLoader(self.config.corpus_name)
        start_time = time.time()
        batch_start = time.time()
        
        for rep in range(self.config.repetitions):
            print(f"\n--- Pass {rep + 1}/{self.config.repetitions} ---")
            
            for sentence in loader.sentences(self.config.max_sentences):
                tokens = self._tokenize(sentence)
                self.train_sentence(tokens)
                self.stats.sentences_processed += 1
                
                # Progress every 1000
                if self.stats.sentences_processed % 1000 == 0:
                    batch_elapsed = time.time() - batch_start
                    rate = 1000 / max(batch_elapsed, 0.001)
                    transitions = len(self.model.memory.transitions)
                    print(f"  {self.stats.sentences_processed} sentences, "
                          f"{transitions} transitions, "
                          f"{rate:.0f} sent/sec")
                    batch_start = time.time()
                    
                    # Clear CUDA cache periodically
                    if torch.cuda.is_available() and self.stats.sentences_processed % 5000 == 0:
                        torch.cuda.empty_cache()
                    
                # Evaluation - less frequent
                if self.stats.sentences_processed % self.config.eval_every == 0:
                    ppl = self.evaluate()
                    self.stats.perplexity_history.append(
                        (self.stats.sentences_processed, ppl)
                    )
                    print(f"  → Perplexity: {ppl:.2f}")
                    batch_start = time.time()  # Reset after eval
                    
                # Checkpoint
                if self.stats.sentences_processed % self.config.checkpoint_every == 0:
                    self.save_checkpoint(
                        f"checkpoint_{self.stats.sentences_processed}"
                    )
                    
            # Reset dataset for next pass
            loader._dataset = None
            
        # Final stats
        self.stats.elapsed_seconds = time.time() - start_time
        self.stats.transitions_learned = len(self.model.memory.transitions)
        
        print(f"\n{'='*60}")
        print(f"Training Complete")
        print(f"{'='*60}")
        print(f"  Sentences: {self.stats.sentences_processed}")
        print(f"  Tokens: {self.stats.tokens_processed}")
        print(f"  Transitions: {self.stats.transitions_learned}")
        print(f"  Time: {self.stats.elapsed_seconds:.1f}s")
        
        # Final evaluation
        final_ppl = self.evaluate()
        print(f"  Final Perplexity: {final_ppl:.2f}")
        
        # Save final checkpoint
        self.save_checkpoint("final")
        
        return self.stats


def train_wikitext2(max_sentences: Optional[int] = None):
    """Train on WikiText-2."""
    config = TrainingConfig(
        corpus_name="wikitext-2",
        max_vocab_size=10000,
        min_word_freq=3,
        batch_size=32,
        max_sentences=max_sentences,
        repetitions=3,
        checkpoint_every=10000,
        eval_every=5000,  # Less frequent eval
        field_shape=(32, 32, 32),
        memory_capacity=100000
    )
    
    trainer = GAIACorpusTrainer(config)
    
    # Build vocabulary
    vocab = trainer.build_vocabulary()
    
    # Initialize model
    trainer.initialize_model(vocab)
    
    # Train
    stats = trainer.train()
    
    return trainer, stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train GAIA on corpus")
    parser.add_argument("--corpus", default="wikitext-2", help="Corpus name")
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--max-vocab", type=int, default=10000)
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    if args.checkpoint:
        config = TrainingConfig(corpus_name=args.corpus)
        trainer = GAIACorpusTrainer(config)
        trainer.load_checkpoint(args.checkpoint)
        trainer.train()
    else:
        trainer, stats = train_wikitext2(max_sentences=args.max_sentences)
