"""
PAC-Lazy Transformer with Imported Knowledge
=============================================

Integrates extracted Pythia knowledge with the full PAC-Lazy Transformer.

This uses the actual PAC-Lazy system (causal propagation, SEC expansion)
but initializes with Pythia's learned embeddings.

The hypothesis: By seeding vocab_deltas with trained embeddings,
the PAC system should immediately produce better predictions than
random initialization.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys
import json

# Add POC-011 scripts to path
poc_011_path = Path(__file__).parent.parent / "poc_011_pac_lazy_transformer" / "scripts"
sys.path.insert(0, str(poc_011_path))

from pac_lazy_core import PACLazySystem, PACNode, PHI, XI, PHI_XI, LAMBDA_STAR
from pac_lazy_transformer import PACLazyTransformer, PACTransformerConfig


class ImportedPACLazyTransformer(PACLazyTransformer):
    """
    PAC-Lazy Transformer initialized with extracted knowledge.
    
    Inherits all PAC-Lazy behavior but starts with Pythia's embeddings
    instead of random initialization.
    """
    
    def __init__(self, config: PACTransformerConfig, pac_path: Path):
        # Initialize base class
        super().__init__(config)
        
        # Load extracted vocab_deltas
        self.pac_path = pac_path
        self._load_extracted_knowledge()
        
    def _load_extracted_knowledge(self):
        """Load Pythia's vocab_deltas."""
        vocab_file = self.pac_path / "pac_vocab.pt"
        
        if not vocab_file.exists():
            raise FileNotFoundError(f"No vocab file at {vocab_file}")
        
        vocab_data = torch.load(vocab_file, weights_only=True)
        embeddings = vocab_data['vocab_deltas'].to(self.device)
        
        print(f"Loading {embeddings.shape[0]:,} vocab_deltas from Pythia...")
        
        # Clear existing vocab_deltas and replace with imported
        self.vocab_deltas.clear()
        
        for token_id in range(embeddings.shape[0]):
            self.vocab_deltas[token_id] = embeddings[token_id].clone()
        
        print(f"  ✓ Loaded {len(self.vocab_deltas):,} imported embeddings")
        print(f"  ✓ Embedding dimension: {embeddings.shape[1]}")
    
    def process_token_by_id(self, token_id: int, learn: bool = True) -> torch.Tensor:
        """Process token using imported embedding."""
        # Get embedding from vocab_deltas
        if token_id in self.vocab_deltas:
            embedding = self.vocab_deltas[token_id]
        else:
            # Fallback to random for unknown tokens
            embedding = torch.randn(self.embedding_dim, device=self.device)
        
        return self.process_token(token_id, embedding, learn=learn)
    
    def generate_sequence(self, initial_tokens: List[int], 
                         max_new_tokens: int = 20,
                         temperature: float = 0.8) -> List[int]:
        """Generate a sequence of tokens."""
        self.reset_sequence()
        
        # Process initial tokens
        for token_id in initial_tokens:
            self.process_token_by_id(token_id, learn=False)
        
        generated = list(initial_tokens)
        
        for _ in range(max_new_tokens):
            # Get predictions
            predictions = self.predict_next(top_k=50)
            
            if not predictions:
                break
            
            # Sample with temperature
            scores = torch.tensor([p[1] for p in predictions]) / temperature
            probs = F.softmax(scores, dim=0)
            idx = torch.multinomial(probs, 1).item()
            
            next_token = predictions[idx][0]
            generated.append(next_token)
            
            # Process the generated token
            self.process_token_by_id(next_token, learn=True)
        
        return generated


def test_imported_pac_lazy():
    """Compare imported vs random PAC-Lazy transformers."""
    from transformers import GPT2Tokenizer
    
    print("="*70)
    print("IMPORTED PAC-LAZY TRANSFORMER TEST")
    print("="*70)
    
    # Path to extracted PAC
    pac_path = Path(__file__).parent.parent / "poc_016_pac_extraction" / "extracted_v3" / "pythia_70m"
    
    if not pac_path.exists():
        print(f"❌ PAC not found at {pac_path}")
        return
    
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load metadata to get embedding dim
    with open(pac_path / "pac_metadata.json") as f:
        metadata = json.load(f)
    embed_dim = metadata['embed_dim']
    
    # Create configs
    config_imported = PACTransformerConfig(
        embedding_dim=embed_dim,
        total_potential=200.0,  # More budget for longer sequences
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    config_random = PACTransformerConfig(
        embedding_dim=embed_dim,
        total_potential=200.0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\n[1] Creating imported PAC-Lazy transformer...")
    imported = ImportedPACLazyTransformer(config_imported, pac_path)
    
    print("\n[2] Creating random baseline PAC-Lazy transformer...")
    random_model = PACLazyTransformer(config_random)
    # Initialize random vocab_deltas
    for i in range(50304):  # Pythia vocab size
        random_model.vocab_deltas[i] = torch.randn(embed_dim, device=random_model.device)
    print(f"  ✓ Created random baseline with {len(random_model.vocab_deltas):,} random embeddings")
    
    # Test prompts
    prompts = [
        "The meaning of life is",
        "Once upon a time",
        "In mathematics,",
    ]
    
    print("\n" + "="*70)
    print("GENERATION COMPARISON")
    print("="*70)
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Tokenize
        token_ids = tokenizer.encode(prompt)
        
        # Generate with imported model
        imported.reset_sequence()
        imported_tokens = imported.generate_sequence(token_ids, max_new_tokens=10, temperature=0.7)
        imported_text = tokenizer.decode(imported_tokens, skip_special_tokens=True)
        
        # Generate with random model (use base method)
        random_model.reset_sequence()
        # Process prompt tokens
        for tid in token_ids:
            embedding = torch.randn(embed_dim, device=random_model.device)
            random_model.process_token(tid, embedding, learn=False)
        # Generate
        random_tokens = list(token_ids)
        for _ in range(10):
            preds = random_model.predict_next(top_k=50)
            if not preds:
                break
            scores = torch.tensor([p[1] for p in preds]) / 0.7
            probs = F.softmax(scores, dim=0)
            idx = torch.multinomial(probs, 1).item()
            next_token = preds[idx][0]
            random_tokens.append(next_token)
            embedding = torch.randn(embed_dim, device=random_model.device)
            random_model.process_token(next_token, embedding, learn=False)
        random_text = tokenizer.decode(random_tokens, skip_special_tokens=True)
        
        print(f"  Imported: {imported_text}")
        print(f"  Random:   {random_text}")
    
    # Compare prediction quality
    print("\n" + "="*70)
    print("PREDICTION QUALITY ANALYSIS")
    print("="*70)
    
    test_prompt = "The weather today"
    test_tokens = tokenizer.encode(test_prompt)
    
    # Process prompt
    imported.reset_sequence()
    random_model.reset_sequence()
    
    for tid in test_tokens:
        imported.process_token_by_id(tid, learn=False)
        embedding = torch.randn(embed_dim, device=random_model.device)
        random_model.process_token(tid, embedding, learn=False)
    
    # Get predictions
    imported_preds = imported.predict_next(top_k=10)
    random_preds = random_model.predict_next(top_k=10)
    
    print(f"\nPrompt: '{test_prompt}'")
    print("\nTop 5 Imported predictions:")
    for i, (tid, score) in enumerate(imported_preds[:5]):
        token = tokenizer.decode([tid])
        print(f"  {i+1}. '{token}' (score={score:.4f})")
    
    print("\nTop 5 Random predictions:")
    for i, (tid, score) in enumerate(random_preds[:5]):
        token = tokenizer.decode([tid])
        print(f"  {i+1}. '{token}' (score={score:.4f})")
    
    # Compute prediction entropy
    imported_scores = torch.tensor([p[1] for p in imported_preds])
    random_scores = torch.tensor([p[1] for p in random_preds])
    
    if len(imported_scores) > 0:
        imported_probs = F.softmax(imported_scores, dim=0)
        imported_entropy = -torch.sum(imported_probs * torch.log(imported_probs + 1e-10)).item()
    else:
        imported_entropy = float('inf')
    
    if len(random_scores) > 0:
        random_probs = F.softmax(random_scores, dim=0)
        random_entropy = -torch.sum(random_probs * torch.log(random_probs + 1e-10)).item()
    else:
        random_entropy = float('inf')
    
    print(f"\nPrediction Entropy:")
    print(f"  Imported: {imported_entropy:.3f}")
    print(f"  Random:   {random_entropy:.3f}")
    
    if imported_entropy < random_entropy:
        reduction = (random_entropy - imported_entropy) / random_entropy * 100
        print(f"\n✅ Imported model has {reduction:.1f}% lower entropy (more confident)")
    else:
        print(f"\n⚠️ Similar entropy - may need more context processing")
    
    # Stats
    print("\n" + "="*70)
    print("SYSTEM STATISTICS")
    print("="*70)
    
    imported_stats = imported.get_stats()
    random_stats = random_model.get_stats()
    
    print(f"\nImported model:")
    print(f"  Active nodes: {imported_stats['active_nodes']}")
    print(f"  Potential utilization: {imported_stats['utilization']:.1%}")
    print(f"  Expansions: {imported_stats['expansions']}")
    
    print(f"\nRandom model:")
    print(f"  Active nodes: {random_stats['active_nodes']}")
    print(f"  Potential utilization: {random_stats['utilization']:.1%}")
    print(f"  Expansions: {random_stats['expansions']}")


if __name__ == "__main__":
    test_imported_pac_lazy()
