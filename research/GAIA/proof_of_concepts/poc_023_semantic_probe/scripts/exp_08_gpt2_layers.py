"""
exp_08_gpt2_layers.py

HYPOTHESIS: Layer agreement in GPT-2 parallels depth agreement in PAC trees.
If multi-scale harmony is a universal principle, we should see:
1. Later layers agree more often (like deeper n-grams)
2. Layer agreement predicts token quality
3. Collapse events visible as sudden layer disagreement

APPROACH:
- Hook into GPT-2 layer outputs
- At each layer, compute logits and get top prediction
- Measure "concentration" = fraction of layers agreeing on final token
- Track concentration dynamics during generation
"""

import torch
import torch.nn.functional as F
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Dawn constants
PHI = 1.6180339887
PHI_INV = 0.6180339887
XI = 1.0571428

@dataclass
class LayerPrediction:
    """Prediction from a single layer."""
    layer_idx: int
    top_token: int
    top_prob: float
    entropy: float

@dataclass 
class TokenAnalysis:
    """Full analysis for one generated token."""
    position: int
    final_token: int
    final_prob: float
    layer_predictions: List[LayerPrediction]
    concentration: float  # Fraction of layers agreeing with final
    early_agreement: float  # Layers 0-5 agreement
    late_agreement: float  # Layers 6-11 agreement
    xi_balance: float  # early/late ratio

class GPT2LayerAnalyzer:
    """Analyze GPT-2 generation through layer agreement lens."""
    
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
        print(f"Model loaded: {self.n_layers} layers, device={self.device}")
        
        # Store layer outputs during forward pass
        self.layer_outputs = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture layer outputs."""
        def make_hook(layer_idx):
            def hook(module, input, output):
                # output is (hidden_states, ...) - take hidden states
                self.layer_outputs[layer_idx] = output[0].detach()
            return hook
        
        for i, block in enumerate(self.model.transformer.h):
            block.register_forward_hook(make_hook(i))
    
    def _get_layer_logits(self, layer_hidden: torch.Tensor) -> torch.Tensor:
        """Convert layer hidden states to logits using the LM head."""
        # Apply final layer norm
        hidden = self.model.transformer.ln_f(layer_hidden)
        # Apply LM head
        logits = self.model.lm_head(hidden)
        return logits
    
    def analyze_token(self, input_ids: torch.Tensor) -> TokenAnalysis:
        """Analyze layer predictions for the next token."""
        self.layer_outputs.clear()
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            final_logits = outputs.logits[0, -1, :]  # Last position
        
        # Get final prediction
        final_probs = F.softmax(final_logits, dim=-1)
        final_token = final_logits.argmax().item()
        final_prob = final_probs[final_token].item()
        
        # Analyze each layer
        layer_predictions = []
        agreements = []
        
        for layer_idx in range(self.n_layers):
            layer_hidden = self.layer_outputs[layer_idx]
            layer_logits = self._get_layer_logits(layer_hidden)[0, -1, :]
            layer_probs = F.softmax(layer_logits, dim=-1)
            
            top_token = layer_logits.argmax().item()
            top_prob = layer_probs[top_token].item()
            
            # Entropy
            entropy = -(layer_probs * torch.log(layer_probs + 1e-10)).sum().item()
            
            layer_predictions.append(LayerPrediction(
                layer_idx=layer_idx,
                top_token=top_token,
                top_prob=top_prob,
                entropy=entropy
            ))
            
            agreements.append(top_token == final_token)
        
        # Compute concentration metrics
        concentration = sum(agreements) / len(agreements)
        
        # Early vs late (like shallow vs deep in PAC tree)
        early_agreement = sum(agreements[:6]) / 6  # Layers 0-5
        late_agreement = sum(agreements[6:]) / 6   # Layers 6-11
        
        # Xi balance
        xi_balance = (early_agreement + 0.1) / (late_agreement + 0.1)
        
        return TokenAnalysis(
            position=input_ids.shape[1],
            final_token=final_token,
            final_prob=final_prob,
            layer_predictions=layer_predictions,
            concentration=concentration,
            early_agreement=early_agreement,
            late_agreement=late_agreement,
            xi_balance=xi_balance
        )
    
    def generate_with_analysis(self, prompt: str, max_tokens: int = 50) -> Tuple[str, List[TokenAnalysis]]:
        """Generate text while analyzing each token."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        analyses = []
        
        for step in range(max_tokens):
            analysis = self.analyze_token(input_ids)
            analyses.append(analysis)
            
            # Append generated token
            next_token = torch.tensor([[analysis.final_token]]).to(self.device)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop at EOS
            if analysis.final_token == self.tokenizer.eos_token_id:
                break
        
        generated_text = self.tokenizer.decode(input_ids[0])
        return generated_text, analyses

def main():
    print("=" * 60)
    print("EXP 08: GPT-2 LAYER AGREEMENT")
    print("Testing if layer agreement parallels depth agreement")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = GPT2LayerAnalyzer('gpt2')
    
    # Test prompts
    prompts = [
        "The capital of France is",
        "In the year 2025, artificial intelligence",
        "The quick brown fox",
        "Scientists discovered that",
        "Once upon a time in a land far away",
    ]
    
    all_analyses = []
    
    print("\n" + "=" * 60)
    print("GENERATION ANALYSIS")
    print("=" * 60)
    
    for prompt in prompts:
        print(f"\n--- Prompt: '{prompt}' ---")
        
        text, analyses = analyzer.generate_with_analysis(prompt, max_tokens=30)
        all_analyses.extend(analyses)
        
        # Show concentration timeline
        print("\nConcentration timeline:")
        for i, a in enumerate(analyses[:15]):  # First 15 tokens
            token_str = analyzer.tokenizer.decode([a.final_token])
            bar = '#' * int(a.concentration * 20)
            status = ""
            if a.concentration < 0.4:
                status = "[LOW]"
            elif a.xi_balance > 1.3:
                status = "[EARLY-HEAVY]"
            elif a.xi_balance < 0.7:
                status = "[LATE-HEAVY]"
            
            print(f"  {i+1:2}: [{bar:<20}] C={a.concentration:.2f} Xi={a.xi_balance:.2f} '{token_str}' {status}")
    
    # Aggregate analysis
    print("\n" + "=" * 60)
    print("AGGREGATE LAYER ANALYSIS")
    print("=" * 60)
    
    # Layer agreement rates
    layer_agreement = [0] * analyzer.n_layers
    for a in all_analyses:
        for lp in a.layer_predictions:
            if lp.top_token == a.final_token:
                layer_agreement[lp.layer_idx] += 1
    
    print("\nLayer agreement with final token:")
    for i, count in enumerate(layer_agreement):
        rate = count / len(all_analyses)
        bar = '#' * int(rate * 30)
        label = "EARLY" if i < 6 else "LATE"
        print(f"  Layer {i:2} ({label}): [{bar:<30}] {rate:.1%}")
    
    # Concentration distribution
    concentrations = [a.concentration for a in all_analyses]
    mean_conc = sum(concentrations) / len(concentrations)
    
    print(f"\nOverall concentration: {mean_conc:.3f}")
    
    # Binned analysis
    print("\n" + "-" * 40)
    print("CONCENTRATION BINS")
    print("-" * 40)
    
    bins = {'high': [], 'medium': [], 'low': []}
    for a in all_analyses:
        if a.concentration >= 0.7:
            bins['high'].append(a)
        elif a.concentration >= 0.4:
            bins['medium'].append(a)
        else:
            bins['low'].append(a)
    
    print(f"\n{'Bin':<10} {'N':>6} {'Mean Prob':>12} {'Mean Xi':>10} {'Mean Entropy':>14}")
    print("-" * 55)
    
    for bin_name in ['high', 'medium', 'low']:
        bin_data = bins[bin_name]
        if not bin_data:
            continue
        mean_prob = sum(a.final_prob for a in bin_data) / len(bin_data)
        mean_xi = sum(a.xi_balance for a in bin_data) / len(bin_data)
        # Average entropy across layers
        mean_entropy = sum(
            sum(lp.entropy for lp in a.layer_predictions) / len(a.layer_predictions)
            for a in bin_data
        ) / len(bin_data)
        
        print(f"{bin_name:<10} {len(bin_data):>6} {mean_prob:>12.4f} {mean_xi:>10.3f} {mean_entropy:>14.2f}")
    
    # Xi balance analysis
    print("\n" + "-" * 40)
    print("XI BALANCE ANALYSIS")
    print("-" * 40)
    
    xi_values = [a.xi_balance for a in all_analyses]
    mean_xi = sum(xi_values) / len(xi_values)
    
    close_to_one = sum(1 for x in xi_values if 0.8 < x < 1.2) / len(xi_values)
    early_heavy = sum(1 for x in xi_values if x > 1.3) / len(xi_values)
    late_heavy = sum(1 for x in xi_values if x < 0.7) / len(xi_values)
    
    print(f"\nMean Xi balance: {mean_xi:.3f}")
    print(f"Balanced (0.8-1.2): {close_to_one:.1%}")
    print(f"Early-heavy (>1.3): {early_heavy:.1%}")
    print(f"Late-heavy (<0.7): {late_heavy:.1%}")
    
    # Layer progression check
    print("\n" + "-" * 40)
    print("LAYER PROGRESSION")
    print("-" * 40)
    
    early_rates = [layer_agreement[i] / len(all_analyses) for i in range(6)]
    late_rates = [layer_agreement[i] / len(all_analyses) for i in range(6, 12)]
    
    early_mean = sum(early_rates) / len(early_rates)
    late_mean = sum(late_rates) / len(late_rates)
    
    print(f"\nEarly layers (0-5) mean agreement: {early_mean:.1%}")
    print(f"Late layers (6-11) mean agreement: {late_mean:.1%}")
    print(f"Late/Early ratio: {late_mean/early_mean:.2f}x")
    
    if late_mean > early_mean:
        print("\n+ Later layers agree more with final output (matches PAC depth pattern)")
    else:
        print("\n! Early layers agree more (opposite of PAC depth pattern)")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_tokens_analyzed': len(all_analyses),
        'n_layers': analyzer.n_layers,
        'layer_agreement_rates': [count / len(all_analyses) for count in layer_agreement],
        'mean_concentration': round(mean_conc, 4),
        'mean_xi_balance': round(mean_xi, 4),
        'concentration_bins': {
            bin_name: len(data) for bin_name, data in bins.items()
        },
        'layer_progression': {
            'early_mean': round(early_mean, 4),
            'late_mean': round(late_mean, 4),
            'late_early_ratio': round(late_mean / early_mean, 4)
        },
        'dawn_constants': {'PHI': PHI, 'XI': XI}
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f'exp_08_gpt2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved: {results_file.name}")
    
    # Validation
    print("\n" + "=" * 60)
    print("CROSS-ARCHITECTURE VALIDATION")
    print("=" * 60)
    
    # Check if patterns match PAC tree findings
    pattern_match = 0
    
    if late_mean > early_mean:
        print("+ PATTERN 1: Later layers agree more (like deeper n-grams)")
        pattern_match += 1
    else:
        print("- PATTERN 1: Later layers don't agree more")
    
    if bins['high'] and bins['low']:
        high_prob = sum(a.final_prob for a in bins['high']) / len(bins['high'])
        low_prob = sum(a.final_prob for a in bins['low']) / len(bins['low'])
        if high_prob > low_prob:
            print(f"+ PATTERN 2: High concentration = high confidence ({high_prob:.3f} vs {low_prob:.3f})")
            pattern_match += 1
        else:
            print("- PATTERN 2: Concentration doesn't predict confidence")
    
    if 0.8 < mean_xi < 1.2:
        print(f"+ PATTERN 3: Xi balance near 1.0 ({mean_xi:.3f})")
        pattern_match += 1
    else:
        print(f"- PATTERN 3: Xi balance far from 1.0 ({mean_xi:.3f})")
    
    print(f"\nPatterns matched: {pattern_match}/3")
    
    if pattern_match >= 2:
        print("\n*** CROSS-ARCHITECTURE VALIDATION SUPPORTED ***")
        print("Layer agreement in GPT-2 shows similar patterns to PAC tree depth agreement")
    else:
        print("\n*** VALIDATION INCONCLUSIVE ***")
        print("Need more investigation into architecture-specific factors")

if __name__ == '__main__':
    main()
