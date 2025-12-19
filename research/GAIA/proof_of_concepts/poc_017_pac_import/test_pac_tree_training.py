"""
PAC Tree Training & Generation Test
=====================================

The REAL test: Train on data, generate coherently, using:
1. PAC tree structure from extracted Pythia model
2. Fracton PACSystem for tree management
3. QBE-regulated growth
4. SEC-triggered expansion

This is not a toy - it's the actual PAC import pipeline.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import our PAC tree importer
from pac_tree_importer import (
    load_extraction, 
    PACTreeBuilder, 
    GrowingPACTransformer,
    ExtractionData
)

# Add fracton
fracton_path = Path(__file__).parent.parent.parent.parent.parent / "fracton"
sys.path.insert(0, str(fracton_path))

from fracton.physics.constants import PHI, XI, PHI_XI


# Training data
TRAINING_SENTENCES = [
    "The weather today is cold and rainy.",
    "The weather today is warm and sunny.",
    "The weather tomorrow will be nice.",
    "Once upon a time there was a princess.",
    "Once upon a time there was a dragon.",
    "Once upon a time there was a kingdom.",
    "The meaning of life is to help others.",
    "The meaning of life is to find happiness.",
    "The meaning of life is love and peace.",
    "In the beginning there was nothing but darkness.",
    "In the beginning there was light and hope.",
    "In the beginning there was only silence.",
    "The cat sat on the mat quietly.",
    "The dog ran through the park quickly.",
    "The bird flew across the blue sky.",
    "She walked to the store to buy bread.",
    "He drove to work early in the morning.",
    "They traveled to Paris for vacation.",
    "The sun rises in the east.",
    "The moon shines bright at night.",
    "The stars twinkle in the dark sky.",
    "Learning requires patience and practice.",
    "Knowledge comes from experience.",
    "Wisdom grows with age and time.",
]


class PACTrainer:
    """
    Trainer for PAC-tree-based transformer.
    
    Key features:
    - Uses PAC tree for embedding reconstruction
    - QBE-regulated entropy for growth triggers
    - SEC expansion when complexity demands it
    """
    
    def __init__(self, 
                 transformer: GrowingPACTransformer,
                 tree: PACTreeBuilder,
                 device: str = 'cpu'):
        self.model = transformer
        self.tree = tree
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            transformer.parameters(),
            lr=1e-3,
            weight_decay=0.01
        )
        
        # Training stats
        self.losses = []
        self.entropies = []
        self.growth_triggers = []
        
        # Growth thresholds (based on QBE)
        self.entropy_floor = 2.0  # Below this = collapsed, needs growth
        self.entropy_ceiling = 8.0  # Above this = chaotic, needs stabilization
        
        # Growth limits
        self.max_dim = 512  # Cap growth to prevent OOM
        self.max_growth_events = 5  # Limit total growth
        
        # Crystal count (Fibonacci-triggered growth)
        self.crystal_count = 0
        self.fib_sequence = [5, 13, 21, 34, 55, 89]  # Start higher, less aggressive
        self.fib_index = 0
    
    def tokenize_simple(self, text: str) -> List[int]:
        """Simple word-level tokenization."""
        words = text.lower().replace('.', ' .').replace(',', ' ,').split()
        # Map to first 10000 token IDs (simple hash)
        return [hash(w) % 10000 for w in words]
    
    def train_step(self, text: str) -> Dict:
        """Single training step on a sentence."""
        self.model.train()
        
        # Tokenize
        tokens = self.tokenize_simple(text)
        if len(tokens) < 2:
            return {'loss': 0, 'entropy': 0}
        
        # Create input/target
        input_ids = torch.tensor([tokens[:-1]], device=self.device)
        target_ids = torch.tensor([tokens[1:]], device=self.device)
        
        # Forward
        logits, metrics = self.model(input_ids)
        
        # Loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Track stats
        self.losses.append(loss.item())
        self.entropies.append(metrics['entropy'])
        
        return {
            'loss': loss.item(),
            'entropy': metrics['entropy'],
            'energy_delta': metrics['energy_delta']
        }
    
    def check_growth(self) -> bool:
        """Check if growth should be triggered based on QBE metrics."""
        if len(self.losses) < 10:
            return False
        
        # Check limits
        if self.model.current_dim >= self.max_dim:
            return False
        if len(self.model.growth_events) >= self.max_growth_events:
            return False
        
        recent_entropy = sum(self.entropies[-10:]) / 10
        recent_loss = sum(self.losses[-10:]) / 10
        
        # Growth triggers:
        # 1. Entropy too low (collapsed, needs more capacity)
        if recent_entropy < self.entropy_floor:
            self.growth_triggers.append(('entropy_low', recent_entropy))
            return True
        
        # 2. Loss plateaued but entropy high (needs structure)
        if len(self.losses) > 50:
            old_loss = sum(self.losses[-50:-40]) / 10
            if abs(old_loss - recent_loss) < 0.1 and recent_entropy > 5.0:
                self.growth_triggers.append(('plateau', recent_loss))
                return True
        
        # 3. Fibonacci crystal count
        self.crystal_count += 1
        if self.fib_index < len(self.fib_sequence):
            if self.crystal_count >= self.fib_sequence[self.fib_index]:
                self.fib_index += 1
                self.growth_triggers.append(('fibonacci', self.crystal_count))
                return True
        
        return False
    
    def train_epoch(self, sentences: List[str], verbose: bool = True) -> Dict:
        """Train one epoch over all sentences."""
        epoch_losses = []
        epoch_entropies = []
        growth_count = 0
        
        for i, sentence in enumerate(sentences):
            result = self.train_step(sentence)
            epoch_losses.append(result['loss'])
            epoch_entropies.append(result['entropy'])
            
            # Check for growth
            if self.check_growth():
                self.model.grow(f"step_{len(self.losses)}")
                growth_count += 1
                
                # Reset optimizer for new parameters
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=1e-3 / (growth_count + 1),  # Decay lr with growth
                    weight_decay=0.01
                )
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_entropy = sum(epoch_entropies) / len(epoch_entropies)
        
        if verbose:
            print(f"  Loss: {avg_loss:.4f}, Entropy: {avg_entropy:.4f}, "
                  f"Growth: {growth_count}, Dim: {self.model.current_dim}")
        
        return {
            'loss': avg_loss,
            'entropy': avg_entropy,
            'growth_count': growth_count,
            'dim': self.model.current_dim
        }
    
    def generate(self, prompt: str, max_tokens: int = 10, temperature: float = 0.8) -> str:
        """Generate text from prompt."""
        self.model.eval()
        
        # Tokenize prompt
        tokens = self.tokenize_simple(prompt)
        generated = tokens.copy()
        
        # Simple word lookup for decoding
        word_to_id = {}
        id_to_word = {}
        for sentence in TRAINING_SENTENCES:
            words = sentence.lower().replace('.', ' .').replace(',', ' ,').split()
            for w in words:
                wid = hash(w) % 10000
                word_to_id[w] = wid
                id_to_word[wid] = w
        
        with torch.no_grad():
            for _ in range(max_tokens):
                input_ids = torch.tensor([generated[-16:]], device=self.device)
                logits, _ = self.model(input_ids)
                
                # Get next token probs
                next_logits = logits[0, -1] / temperature
                probs = F.softmax(next_logits, dim=-1)
                
                # Sample from top-k
                top_k = 50
                top_probs, top_indices = probs.topk(top_k)
                
                # Filter to known words
                valid_indices = []
                valid_probs = []
                for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
                    if idx in id_to_word:
                        valid_indices.append(idx)
                        valid_probs.append(prob)
                
                if valid_indices:
                    # Renormalize
                    total = sum(valid_probs)
                    valid_probs = [p / total for p in valid_probs]
                    
                    # Sample
                    next_token = valid_indices[
                        torch.multinomial(torch.tensor(valid_probs), 1).item()
                    ]
                else:
                    # Fallback to most likely known token
                    for idx in top_indices.tolist():
                        if idx in id_to_word:
                            next_token = idx
                            break
                    else:
                        break
                
                generated.append(next_token)
                
                # Stop on period
                if id_to_word.get(next_token) == '.':
                    break
        
        # Decode
        result_words = []
        for tid in generated[len(tokens):]:
            if tid in id_to_word:
                result_words.append(id_to_word[tid])
        
        return ' '.join(result_words)


def main():
    """Full PAC tree training and generation test."""
    
    print("="*70)
    print("PAC TREE TRAINING & GENERATION TEST")
    print("="*70)
    
    # Load extraction
    extraction_dir = Path(__file__).parent.parent / "poc_016_pac_extraction" / "extracted_v3" / "pythia_70m"
    
    if not extraction_dir.exists():
        print(f"❌ Extraction not found at {extraction_dir}")
        return
    
    extraction = load_extraction(extraction_dir)
    
    # Build PAC tree
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📍 Device: {device}")
    
    tree_builder = PACTreeBuilder(extraction, device=device)
    root_id = tree_builder.build_tree(n_clusters=64)
    
    # Create transformer
    transformer = GrowingPACTransformer(
        tree_builder,
        initial_dim=64,
        n_heads=4,
        device=device
    )
    
    initial_params = sum(p.numel() for p in transformer.parameters())
    print(f"\n📊 Initial parameters: {initial_params:,}")
    
    # Create trainer
    trainer = PACTrainer(transformer, tree_builder, device=device)
    
    # Training
    print("\n" + "="*60)
    print("PHASE 1: TRAINING")
    print("="*60)
    
    n_epochs = 20
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}:")
        result = trainer.train_epoch(TRAINING_SENTENCES)
    
    final_params = sum(p.numel() for p in transformer.parameters())
    print(f"\n📊 Final parameters: {final_params:,}")
    print(f"📈 Parameter growth: {final_params / initial_params:.2f}x")
    print(f"🌱 Total growth events: {len(transformer.growth_events)}")
    
    # Generation
    print("\n" + "="*60)
    print("PHASE 2: GENERATION")
    print("="*60)
    
    prompts = [
        "The weather today is",
        "Once upon a time there was a",
        "The meaning of life is",
        "In the beginning there was",
        "The cat sat on",
        "Learning requires",
    ]
    
    for prompt in prompts:
        generated = trainer.generate(prompt, max_tokens=10)
        print(f'  "{prompt}" → "{generated}"')
    
    # Tree stats
    print("\n" + "="*60)
    print("TREE STATISTICS")
    print("="*60)
    
    print(f"  Tree depth: {tree_builder.stats['tree_depth']}")
    print(f"  Total nodes: {tree_builder.stats['total_nodes']}")
    print(f"  Clusters: {tree_builder.stats['cluster_count']}")
    print(f"  Materialized tokens: {len(tree_builder.token_to_node)}")
    print(f"  Growth events: {len(transformer.growth_events)}")
    
    for i, event in enumerate(transformer.growth_events[:5]):
        print(f"    Growth {i+1}: {event['old_dim']} → {event['new_dim']} ({event['trigger']})")
    
    print("\n" + "="*70)
    print("✅ PAC TREE TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
