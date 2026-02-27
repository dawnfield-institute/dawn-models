"""
POC-020: GAIA + PAC Integration Test
=====================================

Load the trained GAIA-1 model and enhance it with PAC grafted patterns.

The idea:
1. Load GAIA-1's vocabulary patterns
2. Extract similar patterns from GPT-2 via PAC
3. Graft the GPT-2 learning into GAIA's space
4. Test if generation improves
"""

import sys
import os
from pathlib import Path

# Add paths
gaia_path = Path(__file__).parent.parent.parent.parent / "src" / "v4" / "gaia_1"
sys.path.insert(0, str(gaia_path))
sys.path.insert(0, r"c:\Users\peter\repos\core_workspace\fracton")

import torch
import torch.nn.functional as F
import numpy as np

# GAIA imports
from model import GAIA1, GAIA1Config

# PAC imports
from fracton.core.pac_system import PACSystem
from fracton.core.pac_node import PACNode


class GAIAWithPAC:
    """GAIA model enhanced with PAC knowledge."""
    
    def __init__(self, checkpoint_path: str = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.pac_system = None
        self.gpt2_patterns = None
        
        if checkpoint_path:
            self.load_gaia(checkpoint_path)
    
    def load_gaia(self, checkpoint_path: str):
        """Load trained GAIA model."""
        print("Loading GAIA-1 model...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        self.model = GAIA1(config).to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        print(f"  ✓ Loaded (field_dim={config.field_dim}, max_context={config.max_context})")
        
    def setup_pac(self):
        """Setup PAC system and extract GPT-2 patterns."""
        print("\nSetting up PAC system...")
        self.pac_system = PACSystem(device=self.device)
        
        # Load GPT-2 for pattern extraction
        from transformers import AutoModel
        print("  Loading GPT-2 for pattern extraction...")
        gpt2 = AutoModel.from_pretrained('gpt2').to(self.device).eval()
        
        # Extract embedding patterns
        gpt2_embeddings = gpt2.wte.weight.detach()  # [vocab, 768]
        
        # Store in PAC
        print("  Injecting GPT-2 embeddings into PAC...")
        self.gpt2_patterns = {}
        
        # Sample top common tokens
        common_tokens = list(range(1000))  # First 1000 tokens
        
        for token_id in common_tokens:
            emb = gpt2_embeddings[token_id]
            node_id = self.pac_system.inject(
                emb,
                label=f"gpt2:token:{token_id}",
                importance=0.5
            )
            self.gpt2_patterns[token_id] = node_id
        
        print(f"  ✓ {len(self.gpt2_patterns)} GPT-2 patterns in PAC")
        
        del gpt2
        torch.cuda.empty_cache()
    
    def find_resonant_gpt2_pattern(self, gaia_pattern: torch.Tensor, top_k: int = 3):
        """Find GPT-2 patterns that resonate with GAIA pattern."""
        gaia_flat = gaia_pattern.flatten()
        
        resonances = []
        for token_id, node_id in self.gpt2_patterns.items():
            node = self.pac_system.cache.get(node_id)
            if node and node.delta is not None:
                gpt2_delta = node.delta.flatten()
                # Project to same size
                min_len = min(len(gaia_flat), len(gpt2_delta))
                g = gaia_flat[:min_len]
                p = gpt2_delta[:min_len]
                
                norm_g = torch.norm(g)
                norm_p = torch.norm(p)
                
                if norm_g > 1e-10 and norm_p > 1e-10:
                    sim = float(torch.dot(g, p) / (norm_g * norm_p))
                    resonances.append((token_id, sim, node))
        
        # Sort by similarity
        resonances.sort(key=lambda x: -x[1])
        return resonances[:top_k]
    
    def enhance_pattern(self, gaia_pattern: torch.Tensor, boost: float = 0.1):
        """Enhance GAIA pattern with resonant GPT-2 knowledge."""
        resonant = self.find_resonant_gpt2_pattern(gaia_pattern, top_k=1)
        
        if not resonant:
            return gaia_pattern
        
        token_id, sim, node = resonant[0]
        
        if sim > 0.3 and node.delta is not None:
            # Blend in GPT-2 pattern (projected to GAIA size)
            gpt2_delta = node.delta.flatten()
            gaia_size = gaia_pattern.shape[-1]
            
            if len(gpt2_delta) > gaia_size:
                projected = gpt2_delta[:gaia_size]
            else:
                projected = F.pad(gpt2_delta, (0, gaia_size - len(gpt2_delta)))
            
            # Normalize and blend
            projected = projected / (torch.norm(projected) + 1e-8)
            enhanced = gaia_pattern + boost * projected.view_as(gaia_pattern)
            
            return enhanced
        
        return gaia_pattern
    
    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 1.0, 
                 use_pac: bool = True, rep_penalty: float = 1.3):
        """Generate text, optionally using PAC enhancement."""
        self.model.eval()
        
        # Tokenize
        tokens = self.model.tokenizer.encode(prompt)
        generated = tokens.copy()
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Get context
                context = generated[-self.model.config.max_context:]
                input_ids = torch.tensor([context], device=self.device)
                
                # Forward pass
                logits, hidden = self.model(input_ids)
                next_logits = logits[0, -1, :]
                
                # PAC enhancement: use hidden state to find resonant patterns
                if use_pac and self.pac_system is not None:
                    hidden_last = hidden[0, -1, :]  # Last position hidden state
                    
                    # Find resonant GPT-2 patterns
                    resonant = self.find_resonant_gpt2_pattern(hidden_last, top_k=5)
                    
                    # Boost logits for resonant tokens
                    for token_id, sim, _ in resonant:
                        if token_id < len(next_logits) and sim > 0.2:
                            next_logits[token_id] += sim * 2.0  # Boost resonant tokens
                
                # Repetition penalty
                for token_id in set(generated[-30:]):
                    next_logits[token_id] /= rep_penalty
                
                # Temperature
                next_logits = next_logits / temperature
                
                # Sample
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                generated.append(next_token)
                
                if next_token == self.model.tokenizer.eos_token_id:
                    break
        
        return self.model.tokenizer.decode(generated)
    
    def compare_generation(self, prompt: str, max_tokens: int = 50):
        """Compare generation with and without PAC."""
        print(f"\nPrompt: '{prompt}'")
        print("-" * 50)
        
        # Without PAC
        output_base = self.generate(prompt, max_tokens=max_tokens, use_pac=False)
        print(f"GAIA (base):     {output_base}")
        
        # With PAC
        output_pac = self.generate(prompt, max_tokens=max_tokens, use_pac=True)
        print(f"GAIA + PAC:      {output_pac}")
        
        return output_base, output_pac


def main():
    print("=" * 70)
    print("POC-020: GAIA + PAC INTEGRATION TEST")
    print("=" * 70)
    
    # Find checkpoint - try overnight_run first (longer training)
    checkpoint_path = gaia_path / "checkpoints" / "overnight_run" / "gaia1_best.pt"
    
    if not checkpoint_path.exists():
        checkpoint_path = gaia_path / "checkpoints" / "adaptive_multi_corpus" / "gaia1_best.pt"
    
    if not checkpoint_path.exists():
        checkpoint_path = gaia_path / "checkpoints" / "gaia1_best.pt"
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    # Create enhanced model
    gaia_pac = GAIAWithPAC(str(checkpoint_path))
    gaia_pac.setup_pac()
    
    # Test prompts
    prompts = [
        "The history of",
        "Hello, how are",
        "In mathematics,",
        "Once upon a time",
        "The meaning of life is",
    ]
    
    print("\n" + "=" * 70)
    print("GENERATION COMPARISON")
    print("=" * 70)
    
    for prompt in prompts:
        gaia_pac.compare_generation(prompt, max_tokens=40)
        print()
    
    print("=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
