"""
exp_09_gpt2_intervention.py

HYPOTHESIS: Reject-resample using layer agreement improves GPT-2 generation.
If layer concentration predicts quality, rejecting low-concentration samples
should improve output.

STRATEGIES:
1. Baseline: standard sampling
2. Layer-reject: reject if too few layers agree, resample
3. Temperature-by-layer: modulate temp based on layer agreement
"""

import torch
import torch.nn.functional as F
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path

# Dawn constants
PHI = 1.6180339887
XI = 1.0571428

@dataclass
class GenerationMetrics:
    """Metrics for a generation run."""
    strategy: str
    tokens: List[int]
    mean_concentration: float
    mean_probability: float
    mean_entropy: float
    low_conc_count: int
    resamples: int

class GPT2Intervener:
    """GPT-2 with layer-based intervention during generation."""
    
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
    
    def compute_layer_concentration(self, input_ids: torch.Tensor, 
                                     candidate_token: int) -> Tuple[float, float]:
        """Compute layer concentration for a candidate token."""
        self.layer_outputs.clear()
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            final_logits = outputs.logits[0, -1, :]
        
        agreements = 0
        total_entropy = 0
        
        for layer_idx in range(self.n_layers):
            layer_hidden = self.layer_outputs[layer_idx]
            layer_logits = self._get_layer_logits(layer_hidden)[0, -1, :]
            layer_probs = F.softmax(layer_logits, dim=-1)
            
            top_token = layer_logits.argmax().item()
            if top_token == candidate_token:
                agreements += 1
            
            entropy = -(layer_probs * torch.log(layer_probs + 1e-10)).sum().item()
            total_entropy += entropy
        
        concentration = agreements / self.n_layers
        mean_entropy = total_entropy / self.n_layers
        
        return concentration, mean_entropy
    
    def sample_baseline(self, input_ids: torch.Tensor, 
                        temperature: float = 1.0, top_k: int = 50) -> Tuple[int, float]:
        """Standard sampling."""
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1).item()
            prob = probs[token].item()
            
        return token, prob
    
    def sample_layer_reject(self, input_ids: torch.Tensor,
                            temperature: float = 1.0, top_k: int = 50,
                            min_concentration: float = 0.25,
                            max_attempts: int = 5) -> Tuple[int, float, int]:
        """Reject samples with low layer concentration."""
        attempts = 0
        
        while attempts < max_attempts:
            token, prob = self.sample_baseline(input_ids, temperature, top_k)
            concentration, _ = self.compute_layer_concentration(input_ids, token)
            
            if concentration >= min_concentration:
                return token, prob, attempts
            
            attempts += 1
            temperature *= 1.1  # Slightly increase temp to explore
        
        # Fallback: return last sample
        return token, prob, attempts
    
    def sample_temp_modulated(self, input_ids: torch.Tensor,
                               base_temp: float = 1.0, top_k: int = 50,
                               prev_concentration: float = 0.5) -> Tuple[int, float, float]:
        """Modulate temperature based on previous concentration."""
        if prev_concentration > 0.6:
            temp = base_temp * 1.2  # More random if crystallized
        elif prev_concentration < 0.3:
            temp = base_temp * 0.7  # More deterministic if drifting
        else:
            temp = base_temp
        
        token, prob = self.sample_baseline(input_ids, temp, top_k)
        return token, prob, temp
    
    def generate_with_strategy(self, prompt: str, strategy: str,
                                max_tokens: int = 30) -> GenerationMetrics:
        """Generate with specified intervention strategy."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        tokens = []
        concentrations = []
        probabilities = []
        entropies = []
        resamples = 0
        prev_concentration = 0.5
        
        for _ in range(max_tokens):
            if strategy == 'baseline':
                token, prob = self.sample_baseline(input_ids)
                concentration, entropy = self.compute_layer_concentration(input_ids, token)
            elif strategy == 'layer_reject':
                token, prob, attempts = self.sample_layer_reject(input_ids)
                concentration, entropy = self.compute_layer_concentration(input_ids, token)
                resamples += attempts
            elif strategy == 'temp_modulated':
                token, prob, _ = self.sample_temp_modulated(input_ids, prev_concentration=prev_concentration)
                concentration, entropy = self.compute_layer_concentration(input_ids, token)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            tokens.append(token)
            concentrations.append(concentration)
            probabilities.append(prob)
            entropies.append(entropy)
            prev_concentration = concentration
            
            next_token = torch.tensor([[token]]).to(self.device)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if token == self.tokenizer.eos_token_id:
                break
        
        low_conc_count = sum(1 for c in concentrations if c < 0.25)
        
        return GenerationMetrics(
            strategy=strategy,
            tokens=tokens,
            mean_concentration=sum(concentrations) / len(concentrations),
            mean_probability=sum(probabilities) / len(probabilities),
            mean_entropy=sum(entropies) / len(entropies),
            low_conc_count=low_conc_count,
            resamples=resamples
        )

def main():
    print("=" * 60)
    print("EXP 09: GPT-2 INTERVENTION")
    print("Testing layer-based reject-resample")
    print("=" * 60)
    
    intervener = GPT2Intervener('gpt2')
    
    prompts = [
        "The future of artificial intelligence",
        "In a small village by the sea",
        "Scientists have discovered that",
        "The most important thing to remember",
        "Once upon a time there was",
    ]
    
    strategies = ['baseline', 'layer_reject', 'temp_modulated']
    n_runs = 3  # Per prompt per strategy
    
    results = {s: [] for s in strategies}
    
    print(f"\nRunning {len(prompts)} prompts x {n_runs} runs x {len(strategies)} strategies...")
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt[:40]}...'")
        for run in range(n_runs):
            for strategy in strategies:
                metrics = intervener.generate_with_strategy(prompt, strategy, max_tokens=25)
                results[strategy].append(metrics)
    
    # Aggregate
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)
    
    print(f"\n{'Strategy':<16} {'Conc':>8} {'Prob':>8} {'Entropy':>10} {'LowConc':>10} {'Resample':>10}")
    print("-" * 65)
    
    summary = {}
    for strategy in strategies:
        strat_results = results[strategy]
        mean_conc = sum(r.mean_concentration for r in strat_results) / len(strat_results)
        mean_prob = sum(r.mean_probability for r in strat_results) / len(strat_results)
        mean_entropy = sum(r.mean_entropy for r in strat_results) / len(strat_results)
        mean_low = sum(r.low_conc_count for r in strat_results) / len(strat_results)
        mean_resample = sum(r.resamples for r in strat_results) / len(strat_results)
        
        print(f"{strategy:<16} {mean_conc:>8.3f} {mean_prob:>8.4f} {mean_entropy:>10.2f} {mean_low:>10.1f} {mean_resample:>10.1f}")
        
        summary[strategy] = {
            'mean_concentration': round(mean_conc, 4),
            'mean_probability': round(mean_prob, 4),
            'mean_entropy': round(mean_entropy, 2),
            'mean_low_conc': round(mean_low, 2),
            'mean_resamples': round(mean_resample, 2)
        }
    
    # Improvement analysis
    print("\n" + "-" * 40)
    print("IMPROVEMENT OVER BASELINE")
    print("-" * 40)
    
    baseline = summary['baseline']
    for strategy in strategies[1:]:
        strat = summary[strategy]
        conc_lift = (strat['mean_concentration'] - baseline['mean_concentration']) / baseline['mean_concentration'] * 100
        prob_lift = (strat['mean_probability'] - baseline['mean_probability']) / baseline['mean_probability'] * 100
        low_reduction = (baseline['mean_low_conc'] - strat['mean_low_conc']) / baseline['mean_low_conc'] * 100 if baseline['mean_low_conc'] > 0 else 0
        
        print(f"\n{strategy}:")
        print(f"  Concentration: {conc_lift:+.1f}%")
        print(f"  Probability: {prob_lift:+.1f}%")
        print(f"  Low-conc reduction: {low_reduction:+.1f}%")
    
    # Sample outputs
    print("\n" + "=" * 60)
    print("SAMPLE GENERATIONS")
    print("=" * 60)
    
    test_prompt = "The key to happiness is"
    print(f"\nPrompt: '{test_prompt}'")
    
    for strategy in strategies:
        metrics = intervener.generate_with_strategy(test_prompt, strategy, max_tokens=20)
        text = intervener.tokenizer.decode(metrics.tokens)
        print(f"\n{strategy}: '{text}'")
        print(f"  Concentration: {metrics.mean_concentration:.3f}, Prob: {metrics.mean_probability:.4f}")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'n_prompts': len(prompts),
        'n_runs': n_runs,
        'strategies': summary,
        'dawn_constants': {'PHI': PHI, 'XI': XI}
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f'exp_09_intervention_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved: {results_file.name}")
    
    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    
    if summary['layer_reject']['mean_concentration'] > baseline['mean_concentration']:
        print("+ Layer-reject improves concentration")
    if summary['layer_reject']['mean_probability'] > baseline['mean_probability']:
        print("+ Layer-reject improves probability")
    if summary['layer_reject']['mean_low_conc'] < baseline['mean_low_conc']:
        print("+ Layer-reject reduces low-concentration tokens")

if __name__ == '__main__':
    main()
