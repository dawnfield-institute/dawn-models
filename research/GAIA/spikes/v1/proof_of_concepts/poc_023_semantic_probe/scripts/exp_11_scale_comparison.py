"""
exp_11_scale_comparison.py

HYPOTHESIS: The eigenvalue pattern (lambda_3 ~ 1/2) scales with model size.

Test on:
- gpt2 (124M, 12 layers) - already done
- gpt2-medium (355M, 24 layers) - if memory allows
- distilgpt2 (82M, 6 layers) - smaller for comparison

Memory-conscious implementation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import gc
from typing import List, Dict
from datetime import datetime
from pathlib import Path

# Dawn constants
PHI = 1.6180339887
PHI_INV = 0.6180339887

def analyze_model(model_name: str, max_texts: int = 5, max_tokens: int = 50):
    """Analyze layer eigenstructure for a model."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*60}")
    
    # Check memory before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        print(f"Free GPU memory: {free_mem / 1e9:.2f} GB")
    
    try:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    n_layers = model.config.n_layer
    print(f"Layers: {n_layers}, Device: {device}")
    
    # Setup hooks
    layer_outputs = {}
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            layer_outputs[layer_idx] = output[0].detach()
        return hook
    
    for i, block in enumerate(model.transformer.h):
        block.register_forward_hook(make_hook(i))
    
    def get_layer_logits(layer_hidden):
        hidden = model.transformer.ln_f(layer_hidden)
        return model.lm_head(hidden)
    
    # Sample texts
    texts = [
        "The history of science reveals remarkable discoveries.",
        "In modern cities, technology shapes daily life.",
        "The ocean contains many undiscovered species.",
        "Ancient civilizations built lasting monuments.",
        "Climate change affects ecosystems worldwide.",
    ][:max_texts]
    
    # Collect transition data
    co_agree = np.zeros((n_layers, n_layers))
    agree_count = np.zeros(n_layers)
    total_samples = 0
    
    with torch.no_grad():
        for text in texts:
            input_ids = tokenizer.encode(text, return_tensors='pt',
                                         max_length=max_tokens,
                                         truncation=True).to(device)
            
            for pos in range(5, input_ids.shape[1]):
                context = input_ids[:, :pos]
                layer_outputs.clear()
                model(context)
                
                # Get predictions per layer
                predictions = []
                for layer_idx in range(n_layers):
                    layer_hidden = layer_outputs[layer_idx]
                    layer_logits = get_layer_logits(layer_hidden)[0, -1, :]
                    predictions.append(layer_logits.argmax().item())
                
                final_pred = predictions[-1]
                agrees = [pred == final_pred for pred in predictions]
                
                for i in range(n_layers):
                    if agrees[i]:
                        agree_count[i] += 1
                        for j in range(n_layers):
                            if agrees[j]:
                                co_agree[i, j] += 1
                
                total_samples += 1
    
    print(f"Samples: {total_samples}")
    
    # Build transition matrix
    transition = np.zeros((n_layers, n_layers))
    for i in range(n_layers):
        if agree_count[i] > 0:
            for j in range(n_layers):
                transition[i, j] = co_agree[i, j] / agree_count[i]
    
    # Eigenvalues
    eigenvalues = np.linalg.eigvals(transition)
    eigenvalues = np.sort(np.real(eigenvalues))[::-1]
    
    # Find key eigenvalues
    dist_half = min(abs(ev - 0.5) for ev in eigenvalues)
    closest_to_half = min(eigenvalues, key=lambda x: abs(x - 0.5))
    
    # Find which index is closest to 1/2
    half_idx = list(eigenvalues).index(closest_to_half)
    
    # Agreement progression
    agree_rates = agree_count / total_samples
    early_agree = np.mean(agree_rates[:n_layers//2])
    late_agree = np.mean(agree_rates[n_layers//2:])
    ratio = late_agree / early_agree if early_agree > 0 else 0
    
    results = {
        'model': model_name,
        'n_layers': n_layers,
        'total_samples': total_samples,
        'eigenvalues': eigenvalues[:8].tolist(),
        'closest_to_half': float(closest_to_half),
        'half_eigenvalue_index': int(half_idx),
        'dist_to_half': float(dist_half),
        'early_agree': float(early_agree),
        'late_agree': float(late_agree),
        'late_early_ratio': float(ratio)
    }
    
    print(f"\nKey eigenvalues:")
    for i, ev in enumerate(eigenvalues[:6]):
        marker = "***" if abs(ev - 0.5) < 0.1 else ""
        print(f"  lambda_{i}: {ev:.4f} {marker}")
    
    print(f"\nClosest to 1/2: lambda_{half_idx} = {closest_to_half:.4f}")
    print(f"Late/Early agreement ratio: {ratio:.2f}x")
    
    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return results

def main():
    print("=" * 60)
    print("EXP 11: SCALE COMPARISON")
    print("Testing eigenvalue pattern across model sizes")
    print("=" * 60)
    
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU Memory: {total_mem:.1f} GB")
    
    models = ['distilgpt2', 'gpt2']
    
    # Try medium if we have memory
    if torch.cuda.is_available():
        free = (torch.cuda.get_device_properties(0).total_memory - 
                torch.cuda.memory_allocated()) / 1e9
        if free > 2.0:  # GPT2-medium needs ~1.5GB
            models.append('gpt2-medium')
            print("Including gpt2-medium")
        else:
            print(f"Skipping gpt2-medium (need ~2GB, have {free:.1f}GB)")
    
    all_results = []
    
    for model_name in models:
        result = analyze_model(model_name)
        if result:
            all_results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("SCALE COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Model':<20} {'Layers':>8} {'λ~1/2':>10} {'λ_idx':>8} {'L/E Ratio':>10}")
    print("-" * 60)
    
    for r in all_results:
        print(f"{r['model']:<20} {r['n_layers']:>8} {r['closest_to_half']:>10.4f} "
              f"{r['half_eigenvalue_index']:>8} {r['late_early_ratio']:>10.2f}x")
    
    # Pattern analysis
    print("\n" + "-" * 40)
    print("PATTERN ANALYSIS")
    print("-" * 40)
    
    half_values = [r['closest_to_half'] for r in all_results]
    mean_half = np.mean(half_values)
    std_half = np.std(half_values)
    
    print(f"\nEigenvalue near 1/2:")
    print(f"  Mean: {mean_half:.4f}")
    print(f"  Std:  {std_half:.4f}")
    print(f"  Range: {min(half_values):.4f} - {max(half_values):.4f}")
    
    # Check if lambda index scales with layers
    print("\nLambda index vs layers:")
    for r in all_results:
        relative_idx = r['half_eigenvalue_index'] / r['n_layers']
        print(f"  {r['model']}: lambda_{r['half_eigenvalue_index']} / {r['n_layers']} = {relative_idx:.2f}")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'models': all_results,
        'summary': {
            'mean_half_eigenvalue': float(mean_half),
            'std_half_eigenvalue': float(std_half)
        },
        'dawn_constants': {'PHI': PHI, 'target_eigenvalue': 0.5}
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f'exp_11_scale_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved: {results_file.name}")
    
    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    
    if all(abs(r['closest_to_half'] - 0.5) < 0.15 for r in all_results):
        print("+ All models have eigenvalue near 1/2")
        print("+ Pattern is SCALE-INVARIANT")
    else:
        print("! Eigenvalue pattern varies with scale")

if __name__ == '__main__':
    main()
