"""
exp_10_layer_eigenvalues.py

HYPOTHESIS: GPT-2 layer transitions show eigenvalue structure similar to
Prime Harmonic Manifold (lambda -> 1/2) and PAC depth transitions (lambda_3 ~ 0.49).

APPROACH:
- Build transition matrix: P[i,j] = probability layer i agrees when layer j agrees
- Compute eigenvalues
- Look for eigenvalues near 1/2 or phi-related values
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from typing import List, Dict, Tuple
from datetime import datetime
from pathlib import Path

# Dawn constants
PHI = 1.6180339887
PHI_INV = 0.6180339887
XI = 1.0571428

class GPT2EigenAnalyzer:
    """Analyze eigenvalue structure of GPT-2 layer transitions."""
    
    def __init__(self, model_name: str = 'gpt2'):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        print(f"Loading {model_name}...")
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        self.n_layers = self.model.config.n_layer
        print(f"Loaded: {self.n_layers} layers, device={self.device}")
        
        self.layer_outputs = {}
        self._register_hooks()
    
    def _register_hooks(self):
        def make_hook(layer_idx):
            def hook(module, input, output):
                self.layer_outputs[layer_idx] = output[0].detach()
            return hook
        
        for i, block in enumerate(self.model.transformer.h):
            block.register_forward_hook(make_hook(i))
    
    def _get_layer_logits(self, layer_hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.model.transformer.ln_f(layer_hidden)
        return self.model.lm_head(hidden)
    
    def get_layer_predictions(self, input_ids: torch.Tensor) -> List[int]:
        """Get top prediction from each layer."""
        self.layer_outputs.clear()
        
        with torch.no_grad():
            self.model(input_ids)
        
        predictions = []
        for layer_idx in range(self.n_layers):
            layer_hidden = self.layer_outputs[layer_idx]
            layer_logits = self._get_layer_logits(layer_hidden)[0, -1, :]
            predictions.append(layer_logits.argmax().item())
        
        return predictions
    
    def collect_transition_data(self, texts: List[str], 
                                 max_tokens_per_text: int = 100) -> np.ndarray:
        """Collect layer agreement data for transition matrix."""
        # Count[i,j] = times layer i and j both agree with final
        # Also track total agreements per layer
        
        co_agree = np.zeros((self.n_layers, self.n_layers))
        agree_count = np.zeros(self.n_layers)
        total_samples = 0
        
        for text in texts:
            input_ids = self.tokenizer.encode(text, return_tensors='pt', 
                                              max_length=max_tokens_per_text,
                                              truncation=True).to(self.device)
            
            # Slide through each position
            for pos in range(5, input_ids.shape[1]):
                context = input_ids[:, :pos]
                predictions = self.get_layer_predictions(context)
                
                # Final layer always agrees with itself
                final_pred = predictions[-1]
                
                # Track which layers agree with final
                agrees = [pred == final_pred for pred in predictions]
                
                for i in range(self.n_layers):
                    if agrees[i]:
                        agree_count[i] += 1
                        for j in range(self.n_layers):
                            if agrees[j]:
                                co_agree[i, j] += 1
                
                total_samples += 1
        
        print(f"Collected {total_samples} samples")
        
        # Build transition matrix: P[i,j] = P(layer j agrees | layer i agrees)
        transition = np.zeros((self.n_layers, self.n_layers))
        for i in range(self.n_layers):
            if agree_count[i] > 0:
                for j in range(self.n_layers):
                    transition[i, j] = co_agree[i, j] / agree_count[i]
        
        return transition, agree_count / total_samples

def main():
    print("=" * 60)
    print("EXP 10: LAYER EIGENVALUES")
    print("Searching for harmonic eigenvalue structure in GPT-2")
    print("=" * 60)
    
    analyzer = GPT2EigenAnalyzer('gpt2')
    
    # Sample texts for analysis
    texts = [
        "The history of science is filled with remarkable discoveries that changed our understanding of the world.",
        "In modern cities, people often struggle to find balance between work and personal life.",
        "The ocean contains countless species that remain undiscovered by marine biologists.",
        "Technology has transformed how we communicate, work, and spend our leisure time.",
        "Ancient civilizations built monuments that continue to inspire wonder and curiosity today.",
        "The study of mathematics reveals patterns that appear throughout nature.",
        "Climate change presents one of the greatest challenges facing humanity.",
        "Literature offers a window into the human experience across cultures and centuries.",
        "The brain remains one of the most complex and mysterious organs in the body.",
        "Space exploration continues to reveal new wonders about our universe.",
    ]
    
    print("\nCollecting layer transition data...")
    transition, agree_rates = analyzer.collect_transition_data(texts, max_tokens_per_text=80)
    
    print("\n" + "=" * 60)
    print("LAYER AGREEMENT RATES")
    print("=" * 60)
    
    print("\nIndividual layer agreement with final output:")
    for i, rate in enumerate(agree_rates):
        bar = '#' * int(rate * 40)
        print(f"  Layer {i:2}: [{bar:<40}] {rate:.1%}")
    
    print("\n" + "=" * 60)
    print("TRANSITION MATRIX (P[i,j] = P(j agrees | i agrees))")
    print("=" * 60)
    
    # Print abbreviated transition matrix
    print("\nAbbreviated view (layers 0,3,6,9,11):")
    selected = [0, 3, 6, 9, 11]
    print("      ", end="")
    for j in selected:
        print(f"  L{j:02}  ", end="")
    print()
    
    for i in selected:
        print(f"L{i:02}   ", end="")
        for j in selected:
            print(f" {transition[i,j]:.3f} ", end="")
        print()
    
    # Compute eigenvalues
    print("\n" + "=" * 60)
    print("EIGENVALUE ANALYSIS")
    print("=" * 60)
    
    eigenvalues = np.linalg.eigvals(transition)
    eigenvalues = np.sort(np.real(eigenvalues))[::-1]  # Sort descending
    
    print("\nEigenvalues (real parts, sorted):")
    for i, ev in enumerate(eigenvalues):
        # Check for special values
        special = ""
        if abs(ev - 1.0) < 0.05:
            special = "<- ~1.0 (trivial)"
        elif abs(ev - 0.5) < 0.05:
            special = "<- ~1/2 *** PRIME HARMONIC ***"
        elif abs(ev - PHI_INV) < 0.05:
            special = "<- ~phi_inv (0.618)"
        elif abs(ev - PHI) < 0.05:
            special = "<- ~phi (1.618)"
        elif abs(ev) < 0.05:
            special = "<- ~0"
            
        print(f"  lambda_{i:02}: {ev:+.4f} {special}")
    
    # Look for harmonic structure
    print("\n" + "-" * 40)
    print("HARMONIC ANALYSIS")
    print("-" * 40)
    
    # Distance from key values
    dist_half = min(abs(ev - 0.5) for ev in eigenvalues)
    dist_phi = min(abs(ev - PHI_INV) for ev in eigenvalues)
    
    print(f"\nClosest eigenvalue to 1/2: {dist_half:.4f} away")
    print(f"Closest eigenvalue to 1/phi: {dist_phi:.4f} away")
    
    # Check for eigenvalue ratios
    print("\nEigenvalue ratios (looking for phi-structure):")
    for i in range(min(5, len(eigenvalues)-1)):
        if abs(eigenvalues[i+1]) > 0.01:
            ratio = eigenvalues[i] / eigenvalues[i+1]
            phi_dist = abs(ratio - PHI)
            if phi_dist < 0.3:
                print(f"  lambda_{i}/lambda_{i+1} = {ratio:.4f} (phi={PHI:.4f}, dist={phi_dist:.4f}) ***")
            else:
                print(f"  lambda_{i}/lambda_{i+1} = {ratio:.4f}")
    
    # Adjacent layer analysis (like PAC depth)
    print("\n" + "-" * 40)
    print("ADJACENT LAYER TRANSITIONS")
    print("-" * 40)
    
    print("\nP(layer i+1 agrees | layer i agrees):")
    adjacent_probs = []
    for i in range(analyzer.n_layers - 1):
        p = transition[i, i+1]
        adjacent_probs.append(p)
        print(f"  L{i:02} -> L{i+1:02}: {p:.3f}")
    
    mean_adjacent = np.mean(adjacent_probs)
    print(f"\nMean adjacent transition: {mean_adjacent:.4f}")
    
    # Compare to PAC tree finding
    print("\n" + "=" * 60)
    print("CROSS-DOMAIN COMPARISON")
    print("=" * 60)
    
    print("\nPrime Harmonic Manifold: eigenvalue -> 1/2")
    print(f"PAC Tree Depth Transitions: lambda_3 = 0.490")
    print(f"GPT-2 Layer Transitions: closest to 1/2 = {0.5 - dist_half:.4f} or {0.5 + dist_half:.4f}")
    
    # Check if there's a 1/2 eigenvalue
    half_eigenvalues = [ev for ev in eigenvalues if abs(ev - 0.5) < 0.1]
    if half_eigenvalues:
        print(f"\n*** EIGENVALUE NEAR 1/2 FOUND: {half_eigenvalues[0]:.4f} ***")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_layers': analyzer.n_layers,
        'agree_rates': agree_rates.tolist(),
        'eigenvalues': eigenvalues.tolist(),
        'transition_matrix': transition.tolist(),
        'mean_adjacent_transition': float(mean_adjacent),
        'dist_to_half': float(dist_half),
        'dist_to_phi_inv': float(dist_phi),
        'dawn_constants': {'PHI': PHI, 'PHI_INV': PHI_INV, 'XI': XI}
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f'exp_10_eigenvalues_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved: {results_file.name}")
    
    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION STATUS")
    print("=" * 60)
    
    if dist_half < 0.1:
        print("+ Eigenvalue near 1/2 found - matches Prime Harmonic pattern")
    else:
        print(f"! No eigenvalue within 0.1 of 1/2 (closest: {dist_half:.4f})")
    
    if dist_phi < 0.1:
        print("+ Eigenvalue near 1/phi found - phi structure present")
    else:
        print(f"! No eigenvalue within 0.1 of 1/phi (closest: {dist_phi:.4f})")

if __name__ == '__main__':
    main()
