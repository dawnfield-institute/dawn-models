"""
Test PAC Import into GAIA
==========================

Import Pythia-70M's extracted capabilities into a fresh GAIA model.
Test whether GAIA acquires language capabilities WITHOUT any training.

This is the critical test:
- Fresh GAIA (no training) + Pythia PAC → Can it generate coherent text?

Usage:
    python test_import.py
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import os
import json

from importer import PACToGAIAImporter, ImportConfig


def test_import_vs_baseline():
    """
    Compare three models:
    1. Fresh GAIA (no training, no import) - baseline
    2. GAIA + Pythia PAC import - our approach
    3. (Optional) Trained GAIA - gold standard
    """
    
    print("="*70)
    print("POC-017: PAC Import - Training-Free Knowledge Transfer")
    print("="*70)
    print("\nHypothesis: GAIA can acquire language capabilities from")
    print("imported PAC trees WITHOUT any training.\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Path to extracted PAC
    poc_016_path = Path(__file__).parent.parent / "poc_016_pac_extraction"
    pac_path = poc_016_path / "extracted" / "pythia_70m"
    
    if not pac_path.exists():
        print(f"\n❌ ERROR: PAC tree not found at {pac_path}")
        print("Run POC-016 first: python poc_016_pac_extraction/test_extraction.py")
        return None
    
    print(f"\nPAC source: {pac_path}")
    
    # Load tokenizer
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test prompts
    prompts = [
        "The meaning of life is",
        "Once upon a time in a",
        "The weather today is",
        "Hello, my name is",
        "2 + 2 =",
    ]
    
    results = {
        'baseline': {},
        'imported': {},
    }
    
    # =========================================
    # TEST 1: Fresh GAIA (baseline - no import)
    # =========================================
    print("\n" + "="*60)
    print("TEST 1: Fresh GAIA (no training, no import)")
    print("="*60)
    
    config_baseline = ImportConfig(
        pac_path=pac_path,
        device=device,
        field_dim=256,
    )
    
    # Create fresh model without import
    from importer import PACToGAIAImporter
    importer_temp = PACToGAIAImporter(config_baseline)
    baseline_model = importer_temp._create_minimal_model()
    baseline_model.eval()
    
    print("\nGeneration (random weights):")
    print("-" * 40)
    
    baseline_coherence = 0
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            output_ids = baseline_model.generate(input_ids, max_new_tokens=15, temperature=0.8)
        
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results['baseline'][prompt] = output_text
        
        # Simple coherence check: does output contain recognizable words?
        words = output_text.split()
        real_words = sum(1 for w in words if len(w) > 2 and w.isalpha())
        coherence = real_words / max(len(words), 1)
        baseline_coherence += coherence
        
        print(f"Prompt: {prompt}")
        print(f"Output: {output_text[:80]}...")
        print()
    
    baseline_coherence /= len(prompts)
    print(f"Baseline coherence score: {baseline_coherence:.3f}")
    
    # =========================================
    # TEST 2: GAIA + Pythia PAC Import
    # =========================================
    print("\n" + "="*60)
    print("TEST 2: GAIA with Pythia PAC Import (NO TRAINING)")
    print("="*60)
    
    config_import = ImportConfig(
        pac_path=pac_path,
        device=device,
        field_dim=256,
        integration_strength=0.8,
    )
    
    importer = PACToGAIAImporter(config_import)
    imported_model = importer.import_to_gaia()
    imported_model.eval()
    
    print("\nGeneration (with imported knowledge):")
    print("-" * 40)
    
    imported_coherence = 0
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            output_ids = imported_model.generate(input_ids, max_new_tokens=15, temperature=0.8)
        
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results['imported'][prompt] = output_text
        
        # Coherence check
        words = output_text.split()
        real_words = sum(1 for w in words if len(w) > 2 and w.isalpha())
        coherence = real_words / max(len(words), 1)
        imported_coherence += coherence
        
        print(f"Prompt: {prompt}")
        print(f"Output: {output_text[:80]}...")
        print()
    
    imported_coherence /= len(prompts)
    print(f"Imported coherence score: {imported_coherence:.3f}")
    
    # =========================================
    # COMPARISON
    # =========================================
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    print(f"\n{'Model':<30} {'Coherence Score':<20}")
    print("-" * 50)
    print(f"{'Fresh GAIA (baseline)':<30} {baseline_coherence:.3f}")
    print(f"{'GAIA + Pythia PAC':<30} {imported_coherence:.3f}")
    
    improvement = (imported_coherence - baseline_coherence) / max(baseline_coherence, 0.01)
    print(f"\nImprovement: {improvement*100:+.1f}%")
    
    # Statistical analysis
    print("\n" + "-"*50)
    print("Output Entropy Analysis:")
    
    # Measure output distribution entropy
    test_input = tokenizer.encode("The", return_tensors='pt').to(device)
    
    with torch.no_grad():
        baseline_logits = baseline_model(test_input)
        imported_logits = imported_model(test_input)
        
        baseline_probs = F.softmax(baseline_logits[:, -1, :], dim=-1)
        imported_probs = F.softmax(imported_logits[:, -1, :], dim=-1)
        
        baseline_entropy = -torch.sum(baseline_probs * torch.log(baseline_probs + 1e-10)).item()
        imported_entropy = -torch.sum(imported_probs * torch.log(imported_probs + 1e-10)).item()
    
    print(f"  Baseline entropy: {baseline_entropy:.2f}")
    print(f"  Imported entropy: {imported_entropy:.2f}")
    
    # Lower entropy after import suggests more structured predictions
    if imported_entropy < baseline_entropy:
        print(f"  ✓ Import reduced entropy (more structured predictions)")
    else:
        print(f"  ~ Entropy similar (patterns may need tuning)")
    
    # =========================================
    # CONCLUSION
    # =========================================
    print("\n" + "="*70)
    
    if imported_coherence > baseline_coherence * 1.1:
        print("✅ POC-017 SUCCESS: PAC import improved generation!")
        print("   GAIA acquired some capabilities from Pythia WITHOUT training.")
        success = True
    elif imported_coherence >= baseline_coherence:
        print("⚠️  POC-017 PARTIAL: Import didn't hurt, but improvement minimal.")
        print("   May need stronger integration or better pattern mapping.")
        success = False
    else:
        print("❌ POC-017 NEEDS WORK: Import didn't improve over baseline.")
        print("   Need to investigate pattern application strategy.")
        success = False
    
    print("="*70)
    
    # Save results
    output_path = Path(__file__).parent / "results"
    output_path.mkdir(exist_ok=True)
    
    results_data = {
        'baseline_coherence': baseline_coherence,
        'imported_coherence': imported_coherence,
        'improvement_percent': improvement * 100,
        'baseline_entropy': baseline_entropy,
        'imported_entropy': imported_entropy,
        'success': success,
        'samples': {
            'baseline': results['baseline'],
            'imported': results['imported']
        }
    }
    
    with open(output_path / "import_results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to {output_path / 'import_results.json'}")
    
    return results_data


def test_perplexity_comparison():
    """Compare perplexity on test text (more rigorous evaluation)."""
    print("\n" + "="*70)
    print("PERPLEXITY COMPARISON TEST")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test texts
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning, there was nothing but darkness.",
        "Scientists have discovered a new species of butterfly.",
        "The economy grew by three percent last quarter.",
    ]
    
    pac_path = Path(__file__).parent.parent / "poc_016_pac_extraction" / "extracted" / "pythia_70m"
    
    if not pac_path.exists():
        print("PAC not found - run POC-016 first")
        return
    
    # Create models
    config = ImportConfig(pac_path=pac_path, device=device, field_dim=256)
    importer = PACToGAIAImporter(config)
    
    baseline = importer._create_minimal_model()
    imported = importer.import_to_gaia()
    
    baseline.eval()
    imported.eval()
    
    print("\nPerplexity on test texts:")
    print("-" * 50)
    
    baseline_ppls = []
    imported_ppls = []
    
    for text in test_texts:
        input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            # Baseline
            baseline_logits = baseline(input_ids)
            baseline_loss = F.cross_entropy(
                baseline_logits[:, :-1, :].reshape(-1, baseline_logits.size(-1)),
                input_ids[:, 1:].reshape(-1)
            )
            baseline_ppl = torch.exp(baseline_loss).item()
            baseline_ppls.append(baseline_ppl)
            
            # Imported
            imported_logits = imported(input_ids)
            imported_loss = F.cross_entropy(
                imported_logits[:, :-1, :].reshape(-1, imported_logits.size(-1)),
                input_ids[:, 1:].reshape(-1)
            )
            imported_ppl = torch.exp(imported_loss).item()
            imported_ppls.append(imported_ppl)
        
        print(f"Text: {text[:40]}...")
        print(f"  Baseline PPL: {baseline_ppl:.1f}")
        print(f"  Imported PPL: {imported_ppl:.1f}")
        print()
    
    avg_baseline = sum(baseline_ppls) / len(baseline_ppls)
    avg_imported = sum(imported_ppls) / len(imported_ppls)
    
    print(f"Average Perplexity:")
    print(f"  Baseline: {avg_baseline:.1f}")
    print(f"  Imported: {avg_imported:.1f}")
    
    if avg_imported < avg_baseline:
        improvement = (avg_baseline - avg_imported) / avg_baseline * 100
        print(f"\n✅ Import reduced perplexity by {improvement:.1f}%!")
    else:
        print(f"\n⚠️  Perplexity not improved (may need tuning)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--perplexity", action="store_true", help="Run perplexity test")
    args = parser.parse_args()
    
    if args.perplexity:
        test_perplexity_comparison()
    else:
        test_import_vs_baseline()
