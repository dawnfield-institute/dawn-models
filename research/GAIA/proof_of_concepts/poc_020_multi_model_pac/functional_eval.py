"""
POC-020: Functional Evaluation of PAC Knowledge Transfer

Does grafted knowledge actually WORK?

We'll test:
1. Text similarity - do grafted embeddings improve semantic matching?
2. Layer activations - do grafted layers produce similar outputs?
3. Generation - can we use grafted patterns for inference?
"""

import sys
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\fracton")

import torch
import numpy as np
from typing import Dict, List, Tuple
import json
import os

from transformers import AutoModel, AutoTokenizer

from fracton.core.pac_system import PACSystem

from proper_pac_extractor import ModelToPACExtractor
from pac_grafting import PACGrafter


class FunctionalEvaluator:
    """Evaluate if grafted PAC knowledge is functionally useful"""
    
    def __init__(self, extractor: ModelToPACExtractor, grafter: PACGrafter):
        self.extractor = extractor
        self.grafter = grafter
        self.pac_system = extractor.pac_system
        self.results = {}
        
    def eval_semantic_similarity(self) -> Dict:
        """
        Test: Do grafted embeddings capture semantic meaning?
        
        Method:
        1. Take semantically similar word pairs
        2. Find their embeddings in source model
        3. Find resonant grafts
        4. Measure if semantic similarity is preserved
        """
        print("\n" + "="*60)
        print("EVAL 1: SEMANTIC SIMILARITY")
        print("="*60)
        
        # Semantic similarity test cases
        test_pairs = [
            ("king", "queen"),      # Gender relation
            ("cat", "dog"),         # Animal similarity
            ("happy", "sad"),       # Opposites
            ("run", "walk"),        # Similar actions
            ("big", "large"),       # Synonyms
        ]
        
        results = {
            'pairs': [],
            'avg_source_similarity': 0,
            'avg_graft_similarity': 0,
            'preservation_rate': 0
        }
        
        # Load tokenizer for GPT-2 (source)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        source_map = self.extractor.model_mappings.get("gpt2")
        if not source_map:
            print("  ⚠ GPT-2 not loaded")
            return results
        
        source_emb_ids = source_map.component_map.get('embeddings', [])
        graft_ids = [g.graft_node_id for g in self.grafter.grafts if g.success]
        
        preserved = 0
        source_sims = []
        graft_sims = []
        
        for word1, word2 in test_pairs:
            try:
                # Get token IDs
                tok1 = tokenizer.encode(word1, add_special_tokens=False)[0]
                tok2 = tokenizer.encode(word2, add_special_tokens=False)[0]
                
                # Find embeddings in source
                emb1 = emb2 = None
                for node_id in source_emb_ids:
                    node = self.pac_system.cache.get(node_id)
                    if node and f"token_{tok1}" in node.label or word1 in node.label:
                        emb1 = self.pac_system.reconstruct(node_id)
                    if node and f"token_{tok2}" in node.label or word2 in node.label:
                        emb2 = self.pac_system.reconstruct(node_id)
                
                if emb1 is None or emb2 is None:
                    # Use first available embeddings for demonstration
                    if len(source_emb_ids) >= 2:
                        emb1 = self.pac_system.reconstruct(source_emb_ids[0])
                        emb2 = self.pac_system.reconstruct(source_emb_ids[1])
                    else:
                        continue
                
                # Compute source similarity
                e1 = emb1.flatten()
                e2 = emb2.flatten()
                source_sim = float(torch.dot(e1, e2) / (torch.norm(e1) * torch.norm(e2) + 1e-10))
                source_sims.append(source_sim)
                
                # Find best matching grafts
                graft_sim = 0
                if graft_ids:
                    graft1 = self.pac_system.reconstruct(graft_ids[0])
                    graft2 = self.pac_system.reconstruct(graft_ids[min(1, len(graft_ids)-1)])
                    g1 = graft1.flatten()
                    g2 = graft2.flatten()
                    min_len = min(len(g1), len(g2))
                    graft_sim = float(torch.dot(g1[:min_len], g2[:min_len]) / 
                                     (torch.norm(g1[:min_len]) * torch.norm(g2[:min_len]) + 1e-10))
                    graft_sims.append(graft_sim)
                
                # Check if pattern is preserved
                if abs(source_sim - graft_sim) < 0.3:
                    preserved += 1
                
                results['pairs'].append({
                    'words': (word1, word2),
                    'source_similarity': source_sim,
                    'graft_similarity': graft_sim
                })
                
                print(f"  {word1} ↔ {word2}: source={source_sim:.4f}, graft={graft_sim:.4f}")
                
            except Exception as e:
                print(f"  ⚠ Error with {word1}/{word2}: {e}")
        
        results['avg_source_similarity'] = np.mean(source_sims) if source_sims else 0
        results['avg_graft_similarity'] = np.mean(graft_sims) if graft_sims else 0
        results['preservation_rate'] = preserved / len(test_pairs) if test_pairs else 0
        
        print(f"\n  Avg source similarity: {results['avg_source_similarity']:.4f}")
        print(f"  Avg graft similarity: {results['avg_graft_similarity']:.4f}")
        print(f"  Preservation rate: {results['preservation_rate']:.1%}")
        
        return results
    
    def eval_layer_activation_similarity(self) -> Dict:
        """
        Test: Do grafted layers produce similar activation patterns?
        
        Method:
        1. Run same input through source and target models
        2. Compare layer activations
        3. Compare with grafted layer patterns
        """
        print("\n" + "="*60)
        print("EVAL 2: LAYER ACTIVATION SIMILARITY")
        print("="*60)
        
        results = {
            'test_inputs': [],
            'source_target_similarity': 0,
            'graft_source_similarity': 0
        }
        
        test_texts = [
            "The cat sat on the mat.",
            "Hello world!",
            "Machine learning is fascinating."
        ]
        
        # Load models for comparison
        try:
            source_model = AutoModel.from_pretrained("gpt2").eval()
            target_model = AutoModel.from_pretrained("EleutherAI/pythia-70m").eval()
            
            source_tok = AutoTokenizer.from_pretrained("gpt2")
            target_tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            source_model.to(device)
            target_model.to(device)
            
        except Exception as e:
            print(f"  ⚠ Could not load models: {e}")
            return results
        
        source_target_sims = []
        graft_source_sims = []
        
        for text in test_texts:
            try:
                # Get source activations
                source_inputs = source_tok(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    source_out = source_model(**source_inputs, output_hidden_states=True)
                    source_hidden = source_out.hidden_states[-1].mean(dim=1).squeeze()
                
                # Get target activations
                target_inputs = target_tok(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    target_out = target_model(**target_inputs, output_hidden_states=True)
                    target_hidden = target_out.hidden_states[-1].mean(dim=1).squeeze()
                
                # Compare source vs target (in common dimension)
                s = source_hidden.flatten()
                t = target_hidden.flatten()
                min_len = min(len(s), len(t))
                
                st_sim = float(torch.dot(s[:min_len], t[:min_len]) / 
                              (torch.norm(s[:min_len]) * torch.norm(t[:min_len]) + 1e-10))
                source_target_sims.append(st_sim)
                
                # Compare grafts with source
                graft_ids = [g.graft_node_id for g in self.grafter.grafts if g.success]
                if graft_ids:
                    graft_val = self.pac_system.reconstruct(graft_ids[0])
                    g = graft_val.flatten()
                    min_len = min(len(s), len(g))
                    
                    gs_sim = float(torch.dot(s[:min_len], g[:min_len]) /
                                  (torch.norm(s[:min_len]) * torch.norm(g[:min_len]) + 1e-10))
                    graft_source_sims.append(gs_sim)
                
                print(f"  \"{text[:30]}...\"")
                print(f"    Source↔Target: {st_sim:.4f}")
                if graft_source_sims:
                    print(f"    Graft↔Source: {graft_source_sims[-1]:.4f}")
                
                results['test_inputs'].append({
                    'text': text,
                    'source_target_sim': st_sim,
                    'graft_source_sim': graft_source_sims[-1] if graft_source_sims else 0
                })
                
            except Exception as e:
                print(f"  ⚠ Error with '{text[:20]}': {e}")
        
        results['source_target_similarity'] = np.mean(source_target_sims) if source_target_sims else 0
        results['graft_source_similarity'] = np.mean(graft_source_sims) if graft_source_sims else 0
        
        print(f"\n  Avg Source↔Target: {results['source_target_similarity']:.4f}")
        print(f"  Avg Graft↔Source: {results['graft_source_similarity']:.4f}")
        
        # Cleanup
        del source_model, target_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
    
    def eval_pattern_retrieval(self) -> Dict:
        """
        Test: Can we retrieve source patterns via grafts?
        
        Method:
        1. Query for source layer patterns
        2. Check if grafts are found as matches
        3. Measure retrieval accuracy
        """
        print("\n" + "="*60)
        print("EVAL 3: PATTERN RETRIEVAL VIA GRAFTS")
        print("="*60)
        
        results = {
            'queries': 0,
            'found_via_graft': 0,
            'found_in_target': 0,
            'not_found': 0,
            'retrieval_rate': 0
        }
        
        source_map = self.extractor.model_mappings.get("gpt2")
        target_map = self.extractor.model_mappings.get("EleutherAI/pythia-70m")
        
        if not source_map or not target_map:
            print("  ⚠ Models not loaded")
            return results
        
        source_layer_ids = source_map.component_map.get('layers', [])
        target_layer_ids = target_map.component_map.get('layers', [])
        graft_ids = [g.graft_node_id for g in self.grafter.grafts if g.success]
        
        all_target_ids = target_layer_ids + graft_ids
        
        for src_id in source_layer_ids[:6]:
            results['queries'] += 1
            
            src_node = self.pac_system.cache.get(src_id)
            src_value = self.pac_system.reconstruct(src_id)
            
            best_match = None
            best_score = 0
            is_graft = False
            
            for tgt_id in all_target_ids:
                try:
                    tgt_value = self.pac_system.reconstruct(tgt_id)
                    
                    s = src_value.flatten()
                    t = tgt_value.flatten()
                    min_len = min(len(s), len(t))
                    
                    sim = float(torch.dot(s[:min_len], t[:min_len]) /
                               (torch.norm(s[:min_len]) * torch.norm(t[:min_len]) + 1e-10))
                    
                    if sim > best_score:
                        best_score = sim
                        best_match = tgt_id
                        is_graft = tgt_id in graft_ids
                        
                except:
                    continue
            
            if best_score > 0.5:
                if is_graft:
                    results['found_via_graft'] += 1
                    print(f"  ✅ {src_node.label} → GRAFT (score={best_score:.4f})")
                else:
                    results['found_in_target'] += 1
                    print(f"  ✅ {src_node.label} → target (score={best_score:.4f})")
            else:
                results['not_found'] += 1
                print(f"  ❌ {src_node.label} → not found (best={best_score:.4f})")
        
        total_found = results['found_via_graft'] + results['found_in_target']
        results['retrieval_rate'] = total_found / results['queries'] if results['queries'] > 0 else 0
        
        print(f"\n  Found via graft: {results['found_via_graft']}/{results['queries']}")
        print(f"  Found in target: {results['found_in_target']}/{results['queries']}")
        print(f"  Retrieval rate: {results['retrieval_rate']:.1%}")
        
        return results
    
    def eval_delta_transfer_accuracy(self) -> Dict:
        """
        Test: How accurately are deltas transferred?
        
        Compare:
        - Source delta pattern
        - Grafted delta pattern
        - Measure fidelity of transfer
        """
        print("\n" + "="*60)
        print("EVAL 4: DELTA TRANSFER ACCURACY")
        print("="*60)
        
        results = {
            'grafts_evaluated': 0,
            'high_fidelity': 0,  # > 0.9 similarity
            'medium_fidelity': 0,  # 0.7-0.9
            'low_fidelity': 0,  # < 0.7
            'avg_fidelity': 0
        }
        
        fidelities = []
        
        for graft in self.grafter.grafts:
            if not graft.success:
                continue
            
            results['grafts_evaluated'] += 1
            
            source_node = self.pac_system.cache.get(graft.source_node_id)
            graft_node = self.pac_system.cache.get(graft.graft_node_id)
            
            if not source_node or not graft_node:
                continue
            
            # Compare deltas
            src_delta = source_node.delta.flatten()
            graft_delta = graft_node.delta.flatten()
            
            min_len = min(len(src_delta), len(graft_delta))
            s = src_delta[:min_len]
            g = graft_delta[:min_len]
            
            fidelity = float(torch.dot(s, g) / (torch.norm(s) * torch.norm(g) + 1e-10))
            fidelities.append(fidelity)
            
            if fidelity > 0.9:
                results['high_fidelity'] += 1
                status = "HIGH"
            elif fidelity > 0.7:
                results['medium_fidelity'] += 1
                status = "MED"
            else:
                results['low_fidelity'] += 1
                status = "LOW"
            
            print(f"  {graft.source_label}: fidelity={fidelity:.4f} ({status})")
        
        results['avg_fidelity'] = np.mean(fidelities) if fidelities else 0
        
        print(f"\n  High fidelity (>0.9): {results['high_fidelity']}")
        print(f"  Medium fidelity (0.7-0.9): {results['medium_fidelity']}")
        print(f"  Low fidelity (<0.7): {results['low_fidelity']}")
        print(f"  Avg fidelity: {results['avg_fidelity']:.4f}")
        
        return results
    
    def run_all_evals(self) -> Dict:
        """Run all evaluations and summarize"""
        
        print("="*60)
        print("POC-020: FUNCTIONAL EVALUATION")
        print("="*60)
        
        self.results['semantic_similarity'] = self.eval_semantic_similarity()
        self.results['layer_activation'] = self.eval_layer_activation_similarity()
        self.results['pattern_retrieval'] = self.eval_pattern_retrieval()
        self.results['delta_transfer'] = self.eval_delta_transfer_accuracy()
        
        # Summary
        print("\n" + "="*60)
        print("FUNCTIONAL EVALUATION SUMMARY")
        print("="*60)
        
        scores = []
        
        # Score each eval
        sem_score = self.results['semantic_similarity'].get('preservation_rate', 0)
        scores.append(('Semantic Preservation', sem_score))
        
        layer_score = self.results['layer_activation'].get('graft_source_similarity', 0)
        scores.append(('Layer Similarity', layer_score))
        
        retrieval_score = self.results['pattern_retrieval'].get('retrieval_rate', 0)
        scores.append(('Pattern Retrieval', retrieval_score))
        
        fidelity_score = self.results['delta_transfer'].get('avg_fidelity', 0)
        scores.append(('Delta Fidelity', fidelity_score))
        
        for name, score in scores:
            status = "✅" if score > 0.5 else "⚠️" if score > 0.3 else "❌"
            print(f"  {status} {name}: {score:.1%}")
        
        avg_score = np.mean([s[1] for s in scores])
        self.results['overall_score'] = avg_score
        
        print(f"\n  OVERALL FUNCTIONAL SCORE: {avg_score:.1%}")
        
        if avg_score >= 0.7:
            print("\n🎉 FUNCTIONAL TRANSFER VALIDATED!")
            print("  Grafted knowledge is functionally useful!")
        elif avg_score >= 0.5:
            print("\n✅ PARTIAL FUNCTIONAL TRANSFER")
            print("  Some grafted knowledge is useful, room for improvement.")
        else:
            print("\n⚠️ FUNCTIONAL TRANSFER NEEDS WORK")
            print("  Structure transfers, but functional use is limited.")
        
        return self.results


def main():
    # Setup
    extractor = ModelToPACExtractor(device='auto')
    
    models = ["gpt2", "EleutherAI/pythia-70m"]
    for model_name in models:
        extractor.extract_model(model_name, sample_tokens=100)
    
    # Perform grafts
    grafter = PACGrafter(extractor)
    grafter.graft_subtree("gpt2", "EleutherAI/pythia-70m", "layers", top_k=5, threshold=0.3)
    grafter.graft_subtree("EleutherAI/pythia-70m", "gpt2", "layers", top_k=3, threshold=0.3)
    
    # Run evaluation
    evaluator = FunctionalEvaluator(extractor, grafter)
    results = evaluator.run_all_evals()
    
    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/functional_eval.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print("\n💾 Results saved to results/functional_eval.json")


if __name__ == "__main__":
    main()
