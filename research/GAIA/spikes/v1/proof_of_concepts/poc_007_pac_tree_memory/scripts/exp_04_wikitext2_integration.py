"""
Experiment 04: WikiText-2 Integration
=====================================

Integrates tiered memory cache with GAIA unified for WikiText-2 training.
Tests that we can achieve similar perplexity with limited GPU memory.

Goals:
- Train on WikiText-2 with only 1000-pattern GPU cache
- Compare perplexity to full GPU storage (from POC-006)
- Validate memory savings in practice
"""

import torch
import torch.nn.functional as F
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, r'c:\Users\peter\repos\core_workspace\dawn-models\research\GAIA\src')

from tiered_memory_cache import TieredMemoryCache

# Import GAIA components
try:
    from gaia_unified import GAIAConfig, SphericalEncoderV6, KleinGordonEvolution
    HAS_GAIA = True
except ImportError as e:
    HAS_GAIA = False
    print(f"Warning: gaia_unified not found: {e}")


class TieredFieldMemory:
    """
    Field memory using tiered cache for large vocabularies.
    
    Drop-in replacement for FieldMemory with memory efficiency.
    """
    
    def __init__(self, config, gpu_cache_size: int = 1000):
        self.config = config
        self.device = config.device if torch.cuda.is_available() else 'cpu'
        
        # Use tiered cache
        self.cache = TieredMemoryCache(
            field_shape=config.field_shape,
            device=self.device,
            gpu_cache_size=gpu_cache_size,
            prefetch_k=10
        )
        
        self.next_id = 0
        
    def store(self, field: torch.Tensor, token_id: Optional[int] = None) -> int:
        """Store pattern using tiered cache"""
        if token_id is None:
            token_id = self.next_id
        self.next_id = max(self.next_id, token_id + 1)
        
        self.cache.store(token_id, field)
        return token_id
    
    def retrieve(self, query: torch.Tensor, top_k: int = 5,
                context_ids: Optional[List[int]] = None,
                exclude: Optional[set] = None) -> List[Tuple[int, float]]:
        """Retrieve using tiered cache with transitions"""
        return self.cache.retrieve(
            query, 
            context_ids=context_ids,
            top_k=top_k
        )
    
    def learn_transition(self, from_id: int, to_id: int):
        """Learn transition for prefetching"""
        self.cache.learn_transition(from_id, to_id, strength=0.1)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return self.cache.get_stats()


class TieredGAIA:
    """
    GAIA model with tiered memory.
    """
    
    def __init__(self, config: GAIAConfig, gpu_cache_size: int = 1000):
        self.config = config
        self.device = config.device if torch.cuda.is_available() else 'cpu'
        
        # Components
        self.encoder = SphericalEncoderV6(config).to(self.device)
        self.evolution = KleinGordonEvolution(config).to(self.device)
        self.memory = TieredFieldMemory(config, gpu_cache_size)
        
        # Tokenizer cache
        self._token_embeddings: Dict[int, torch.Tensor] = {}
        
        # Stats
        self.stats = {
            'tokens_processed': 0,
            'predictions_correct': 0,
            'total_loss': 0.0
        }
        
    def process_token(self, token_id: int, embedding: torch.Tensor,
                     context: Optional[List[int]] = None) -> torch.Tensor:
        """Process token and return field"""
        # Encode to field
        field = self.encoder(embedding.to(self.device))
        
        # Store in memory
        self.memory.store(field, token_id)
        
        # Learn transition from context
        if context:
            self.memory.learn_transition(context[-1], token_id)
            
        # Evolve field
        evolved = self.evolution(field)
        
        return evolved
    
    def predict_next(self, evolved_field: torch.Tensor,
                    context: Optional[List[int]] = None,
                    top_k: int = 10) -> List[Tuple[int, float]]:
        """Predict next token from evolved field"""
        candidates = self.memory.retrieve(
            evolved_field,
            top_k=top_k,
            context_ids=context
        )
        return candidates


def train_on_wikitext2(gpu_cache_size: int = 1000, 
                       max_sentences: int = 2000) -> Dict:
    """
    Train on WikiText-2 with tiered memory.
    """
    print(f"\n{'='*60}")
    print(f"WikiText-2 Training with Tiered Memory")
    print(f"GPU Cache Size: {gpu_cache_size}")
    print(f"{'='*60}")
    
    # Load sentence transformer
    try:
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        print("ERROR: sentence-transformers required")
        return {'error': 'no sentence transformers'}
    
    # Load WikiText-2
    try:
        from datasets import load_dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    except ImportError:
        print("ERROR: datasets required")
        return {'error': 'no datasets'}
    
    # Create model with tiered memory
    config = GAIAConfig(
        field_shape=(24, 24, 24),
        embedding_dim=384,
        memory_capacity=100000,  # Large capacity
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    model = TieredGAIA(config, gpu_cache_size=gpu_cache_size)
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    all_tokens = set()
    for i, item in enumerate(dataset):
        if i >= max_sentences:
            break
        text = item['text'].strip()
        if text:
            tokens = text.lower().split()
            all_tokens.update(tokens)
            
    vocab = {token: i for i, token in enumerate(sorted(all_tokens))}
    print(f"Vocabulary size: {len(vocab)}")
    
    # Pre-compute embeddings
    print("Computing embeddings...")
    token_embeddings = {}
    batch_size = 128
    tokens = list(vocab.keys())
    
    for i in range(0, len(tokens), batch_size):
        batch = tokens[i:i+batch_size]
        embeddings = sbert.encode(batch, convert_to_tensor=True)
        for j, token in enumerate(batch):
            token_embeddings[vocab[token]] = embeddings[j]
    
    # Training
    print("\nTraining...")
    start_time = time.perf_counter()
    
    total_predictions = 0
    correct_predictions = 0
    total_loss = 0.0
    sentences_processed = 0
    
    for i, item in enumerate(dataset):
        if i >= max_sentences:
            break
            
        text = item['text'].strip()
        if not text:
            continue
            
        tokens = text.lower().split()
        if len(tokens) < 3:
            continue
            
        token_ids = [vocab.get(t) for t in tokens if t in vocab]
        
        context = []
        for j, tid in enumerate(token_ids[:-1]):
            # Get embedding
            emb = token_embeddings.get(tid)
            if emb is None:
                continue
                
            # Process token
            evolved = model.process_token(tid, emb, context if context else None)
            
            # Predict next
            if j > 0:  # Need at least one context
                predictions = model.predict_next(evolved, context, top_k=10)
                
                target = token_ids[j + 1] if j + 1 < len(token_ids) else None
                
                if predictions and target is not None:
                    total_predictions += 1
                    
                    # Check if correct
                    pred_ids = [p[0] for p in predictions]
                    if target in pred_ids:
                        correct_predictions += 1
                        
                    # Compute loss
                    if predictions[0][0] == target:
                        total_loss += 0.0
                    else:
                        total_loss += 1.0
                        
            context = context[-9:] + [tid]  # Keep last 10
            
        sentences_processed += 1
        
        if sentences_processed % 500 == 0:
            accuracy = correct_predictions / max(total_predictions, 1)
            print(f"  Processed {sentences_processed} sentences, "
                  f"accuracy: {accuracy:.2%}")
    
    elapsed = time.perf_counter() - start_time
    
    # Results
    accuracy = correct_predictions / max(total_predictions, 1)
    avg_loss = total_loss / max(total_predictions, 1)
    
    # Estimate perplexity (rough)
    perplexity = 2 ** avg_loss
    
    cache_stats = model.memory.get_stats()
    
    results = {
        'gpu_cache_size': gpu_cache_size,
        'vocabulary_size': len(vocab),
        'sentences_processed': sentences_processed,
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'accuracy': accuracy,
        'estimated_perplexity': perplexity,
        'training_time_sec': elapsed,
        'tokens_per_sec': total_predictions / elapsed,
        'cache_hit_rate': cache_stats['hit_rate'],
        'cache_evictions': cache_stats['evictions'],
        'memory_savings': cache_stats['pac_tree_stats'].get('compression_ratio', 1.0)
    }
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Vocabulary: {len(vocab)}")
    print(f"Sentences: {sentences_processed}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Est. Perplexity: {perplexity:.2f}")
    print(f"Training time: {elapsed:.1f}s")
    print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
    print(f"Memory savings: {results['memory_savings']:.1f}x")
    
    return results


def main():
    if not HAS_GAIA:
        print("ERROR: gaia_unified.py required")
        print("Please ensure GAIA source is in path")
        return
        
    print("=" * 60)
    print("POC-007 Experiment 04: WikiText-2 Integration")
    print("=" * 60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    all_results = []
    
    # Test different cache sizes
    cache_sizes = [500, 1000, 2000]
    
    for cache_size in cache_sizes:
        try:
            results = train_on_wikitext2(
                gpu_cache_size=cache_size,
                max_sentences=1000
            )
            all_results.append(results)
        except Exception as e:
            print(f"ERROR with cache_size={cache_size}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    if all_results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        print(f"\n{'Cache':<10} {'Vocab':<10} {'Accuracy':<12} {'Perplexity':<12} {'Hit Rate':<10}")
        print("-" * 60)
        
        for r in all_results:
            print(f"{r['gpu_cache_size']:<10} {r['vocabulary_size']:<10} "
                  f"{r['accuracy']:.2%}         {r['estimated_perplexity']:.2f}           "
                  f"{r['cache_hit_rate']:.2%}")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'tests': all_results
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = results_dir / f'exp_04_wikitext2_integration_{timestamp}.json'
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
        
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
