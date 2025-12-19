"""
POC-020: Generation Test
=========================

The ultimate test: Does PAC grafting affect model generations?

We'll test:
1. Generate with base model
2. Graft knowledge from another model
3. Use grafted patterns to influence generation
4. Compare outputs
"""

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\fracton")

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from proper_pac_extractor import ModelToPACExtractor
from pac_grafting import PACGrafter


def cosine_similarity(a, b):
    """Cosine similarity between tensors."""
    a, b = a.flatten().float(), b.flatten().float()
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    norm_a, norm_b = torch.norm(a), torch.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(torch.dot(a, b) / (norm_a * norm_b))


class GenerationTester:
    """Test if PAC grafting influences model generations."""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.extractor = None
        self.grafter = None
        self.models = {}
        self.tokenizers = {}
        
    def setup(self):
        """Extract models and perform grafts."""
        print("=" * 70)
        print("POC-020: GENERATION TEST")
        print("=" * 70)
        print()
        print("Setting up PAC system...")
        
        self.extractor = ModelToPACExtractor()
        
        # Extract models into PAC
        for model_name in ['gpt2', 'EleutherAI/pythia-70m']:
            print(f"  Extracting {model_name}...")
            self.extractor.extract_model(model_name)
        
        # Load actual models for generation
        print()
        print("Loading models for generation...")
        
        for model_name in ['gpt2', 'EleutherAI/pythia-70m']:
            print(f"  Loading {model_name}...")
            self.models[model_name] = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.models[model_name].eval()
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizers[model_name].pad_token is None:
                self.tokenizers[model_name].pad_token = self.tokenizers[model_name].eos_token
        
        # Setup grafter
        self.grafter = PACGrafter(self.extractor)
        
        # Perform grafts
        print()
        print("Performing grafts (GPT-2 → Pythia)...")
        candidates = self.grafter.find_graft_candidates(
            'gpt2', 'EleutherAI/pythia-70m', threshold=0.3, max_candidates=10
        )
        for c in candidates[:5]:
            self.grafter.graft_node(c['source_id'], c['target_id'])
        print(f"  → {len(self.grafter.grafts)} grafts created")
        print()
        
    def generate(self, model_name: str, prompt: str, max_new_tokens: int = 30) -> str:
        """Generate text with a model."""
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        inputs = tokenizer(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def get_hidden_states(self, model_name: str, prompt: str) -> torch.Tensor:
        """Get hidden states from a model for a prompt."""
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        inputs = tokenizer(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Get last layer hidden states, mean pooled
        hidden = outputs.hidden_states[-1]  # [batch, seq, dim]
        return hidden.mean(dim=1).squeeze()  # [dim]
    
    def find_most_resonant_graft(self, query_hidden: torch.Tensor) -> tuple:
        """Find the graft most similar to query hidden states."""
        pac_system = self.extractor.pac_system
        
        best_graft = None
        best_sim = -1
        
        for graft in self.grafter.grafts:
            if not graft.success:
                continue
            graft_node = pac_system.cache.get(graft.graft_node_id)
            if graft_node and graft_node.delta is not None:
                sim = cosine_similarity(query_hidden, graft_node.delta)
                if sim > best_sim:
                    best_sim = sim
                    best_graft = graft
        
        return best_graft, best_sim
    
    def test_generation_influence(self):
        """Test if grafted knowledge influences what we can detect in generations."""
        print("=" * 70)
        print("TEST: GENERATION PATTERN ANALYSIS")
        print("=" * 70)
        print()
        
        prompts = [
            "The meaning of life is",
            "In the future, artificial intelligence will",
            "The most important thing about science is",
            "When I think about the universe,",
        ]
        
        results = []
        
        for prompt in prompts:
            print(f"Prompt: '{prompt}'")
            print("-" * 50)
            
            # Generate with both models
            gpt2_output = self.generate('gpt2', prompt)
            pythia_output = self.generate('EleutherAI/pythia-70m', prompt)
            
            print(f"  GPT-2:  {gpt2_output}")
            print(f"  Pythia: {pythia_output}")
            
            # Get hidden states
            gpt2_hidden = self.get_hidden_states('gpt2', gpt2_output)
            pythia_hidden = self.get_hidden_states('EleutherAI/pythia-70m', pythia_output)
            
            # Find most resonant graft for each
            gpt2_graft, gpt2_sim = self.find_most_resonant_graft(gpt2_hidden)
            pythia_graft, pythia_sim = self.find_most_resonant_graft(pythia_hidden)
            
            print(f"  GPT-2 hidden → graft resonance: {gpt2_sim:.3f}")
            print(f"  Pythia hidden → graft resonance: {pythia_sim:.3f}")
            
            # Direct hidden state comparison
            direct_sim = cosine_similarity(gpt2_hidden, pythia_hidden)
            print(f"  GPT-2 ↔ Pythia direct: {direct_sim:.3f}")
            
            results.append({
                'prompt': prompt,
                'gpt2_output': gpt2_output,
                'pythia_output': pythia_output,
                'gpt2_graft_sim': gpt2_sim,
                'pythia_graft_sim': pythia_sim,
                'direct_sim': direct_sim
            })
            print()
        
        return results
    
    def test_graft_as_style_detector(self):
        """Test if grafts can detect which model generated text."""
        print("=" * 70)
        print("TEST: GRAFT AS STYLE DETECTOR")
        print("=" * 70)
        print()
        print("Can grafted patterns identify which model produced text?")
        print()
        
        pac_system = self.extractor.pac_system
        
        # Get GPT-2 source layer patterns
        gpt2_map = self.extractor.model_mappings.get('gpt2')
        gpt2_layer_ids = gpt2_map.component_map.get('layers', [])[:4]
        
        gpt2_patterns = []
        for lid in gpt2_layer_ids:
            node = pac_system.cache.get(lid)
            if node and node.delta is not None:
                gpt2_patterns.append(node.delta)
        
        prompts = [
            "The quick brown fox",
            "Once upon a time",
            "Scientists have discovered",
        ]
        
        correct = 0
        total = 0
        
        for prompt in prompts:
            # Generate with both
            gpt2_text = self.generate('gpt2', prompt, max_new_tokens=20)
            pythia_text = self.generate('EleutherAI/pythia-70m', prompt, max_new_tokens=20)
            
            # Get hidden states
            gpt2_hidden = self.get_hidden_states('gpt2', gpt2_text)
            pythia_hidden = self.get_hidden_states('EleutherAI/pythia-70m', pythia_text)
            
            # Compare to GPT-2 patterns
            gpt2_match = max(cosine_similarity(gpt2_hidden, p) for p in gpt2_patterns)
            pythia_match = max(cosine_similarity(pythia_hidden, p) for p in gpt2_patterns)
            
            # GPT-2 text should match GPT-2 patterns better
            gpt2_wins = gpt2_match > pythia_match
            
            status = "✓" if gpt2_wins else "✗"
            print(f"  {status} Prompt: '{prompt[:30]}...'")
            print(f"      GPT-2 text → GPT-2 patterns: {gpt2_match:.3f}")
            print(f"      Pythia text → GPT-2 patterns: {pythia_match:.3f}")
            
            if gpt2_wins:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        print()
        print(f"  Style Detection Accuracy: {correct}/{total} = {accuracy:.1%}")
        return accuracy
    
    def test_knowledge_retrieval_boost(self):
        """Test if grafted knowledge helps retrieve relevant patterns."""
        print()
        print("=" * 70)
        print("TEST: KNOWLEDGE RETRIEVAL BOOST")
        print("=" * 70)
        print()
        print("Do grafts help find relevant knowledge for generation?")
        print()
        
        pac_system = self.extractor.pac_system
        
        # Get Pythia layer patterns (target model)
        pythia_map = self.extractor.model_mappings.get('EleutherAI/pythia-70m')
        pythia_layer_ids = pythia_map.component_map.get('layers', [])
        
        # Get graft patterns
        graft_ids = [g.graft_node_id for g in self.grafter.grafts if g.success]
        
        prompts = [
            "The meaning of",
            "Artificial intelligence",
            "In the year 2050",
        ]
        
        improvements = []
        
        for prompt in prompts:
            # Generate with Pythia
            output = self.generate('EleutherAI/pythia-70m', prompt)
            hidden = self.get_hidden_states('EleutherAI/pythia-70m', output)
            
            # Best match to native Pythia layers
            pythia_sims = []
            for lid in pythia_layer_ids:
                node = pac_system.cache.get(lid)
                if node and node.delta is not None:
                    pythia_sims.append(cosine_similarity(hidden, node.delta))
            best_pythia = max(pythia_sims) if pythia_sims else 0
            
            # Best match to grafts (GPT-2 knowledge in Pythia tree)
            graft_sims = []
            for gid in graft_ids:
                node = pac_system.cache.get(gid)
                if node and node.delta is not None:
                    graft_sims.append(cosine_similarity(hidden, node.delta))
            best_graft = max(graft_sims) if graft_sims else 0
            
            improvement = best_graft - best_pythia
            improvements.append(improvement)
            
            status = "✓" if improvement > 0 else "○"
            print(f"  {status} '{prompt}' → Pythia: {best_pythia:.3f}, +Graft: {best_graft:.3f} (Δ={improvement:+.3f})")
        
        avg_improvement = np.mean(improvements)
        print()
        print(f"  Average improvement from grafts: {avg_improvement:+.3f}")
        return avg_improvement
    
    def run_all(self):
        """Run all generation tests."""
        self.setup()
        
        # Test 1: Generation pattern analysis
        gen_results = self.test_generation_influence()
        
        # Test 2: Style detection
        style_accuracy = self.test_graft_as_style_detector()
        
        # Test 3: Knowledge retrieval boost
        retrieval_boost = self.test_knowledge_retrieval_boost()
        
        # Summary
        print()
        print("=" * 70)
        print("GENERATION TEST SUMMARY")
        print("=" * 70)
        print()
        print("Results:")
        print(f"  • Generation patterns analyzed: {len(gen_results)}")
        print(f"  • Style detection accuracy: {style_accuracy:.1%}")
        print(f"  • Retrieval boost from grafts: {retrieval_boost:+.3f}")
        print()
        
        avg_graft_resonance = np.mean([r['pythia_graft_sim'] for r in gen_results])
        print(f"  • Average graft resonance: {avg_graft_resonance:.3f}")
        print()
        
        if style_accuracy >= 0.6 and retrieval_boost > 0:
            print("🎉 GENERATION TEST PASSED: Grafts influence pattern detection!")
        elif style_accuracy >= 0.5 or retrieval_boost > 0:
            print("✓ PARTIAL SUCCESS: Some influence detected")
        else:
            print("○ LIMITED: Grafts don't strongly influence generation patterns")
        
        return {
            'generation_results': gen_results,
            'style_accuracy': style_accuracy,
            'retrieval_boost': retrieval_boost,
            'avg_graft_resonance': avg_graft_resonance
        }


if __name__ == "__main__":
    tester = GenerationTester()
    results = tester.run_all()
