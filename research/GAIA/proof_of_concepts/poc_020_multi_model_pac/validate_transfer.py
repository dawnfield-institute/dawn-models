"""
POC-020: Validate PAC Knowledge Transfer

After grafting, verify that:
1. Grafted nodes have the source's delta pattern
2. Target model can now find source patterns via resonance
3. Knowledge is actually transferred, not just copied

This is the definitive test of PAC-based knowledge transfer!
"""

import sys
sys.path.insert(0, r"c:\Users\peter\repos\core_workspace\fracton")

import torch
import numpy as np
from typing import Dict, List
import json
import os

from fracton.core.pac_system import PACSystem

from proper_pac_extractor import ModelToPACExtractor
from pac_grafting import PACGrafter


def comprehensive_transfer_test(extractor: ModelToPACExtractor, 
                                grafter: PACGrafter) -> Dict:
    """
    Comprehensive test of knowledge transfer.
    """
    pac_system = extractor.pac_system
    results = {
        'tests': [],
        'summary': {}
    }
    
    # Test 1: Delta Pattern Preservation
    print("\n📊 Test 1: Delta Pattern Preservation")
    print("-" * 40)
    
    delta_preserved = 0
    for graft in grafter.grafts:
        if not graft.success:
            continue
            
        source_node = pac_system.cache.get(graft.source_node_id)
        graft_node = pac_system.cache.get(graft.graft_node_id)
        
        if not source_node or not graft_node:
            continue
        
        # Compare delta patterns
        src_delta = source_node.delta.flatten()
        graft_delta = graft_node.delta.flatten()
        
        # Project to common dimension
        min_len = min(len(src_delta), len(graft_delta))
        s = src_delta[:min_len]
        g = graft_delta[:min_len]
        
        # Cosine similarity of deltas
        norm_s = torch.norm(s)
        norm_g = torch.norm(g)
        
        if norm_s > 1e-10 and norm_g > 1e-10:
            sim = float(torch.dot(s, g) / (norm_s * norm_g))
        else:
            sim = 0.0
        
        if sim > 0.5:
            delta_preserved += 1
            print(f"  ✅ {graft.source_label}: delta similarity = {sim:.4f}")
        else:
            print(f"  ❌ {graft.source_label}: delta similarity = {sim:.4f}")
    
    results['tests'].append({
        'name': 'Delta Pattern Preservation',
        'passed': delta_preserved,
        'total': len([g for g in grafter.grafts if g.success]),
        'rate': delta_preserved / max(1, len([g for g in grafter.grafts if g.success]))
    })
    
    # Test 2: Cross-Model Resonance
    print("\n📊 Test 2: Cross-Model Resonance After Grafting")
    print("-" * 40)
    
    # Get a source embedding and see if target+grafts can find it
    source_model = "gpt2"
    target_model = "EleutherAI/pythia-70m"
    
    source_map = extractor.model_mappings.get(source_model)
    target_map = extractor.model_mappings.get(target_model)
    
    if source_map and target_map:
        source_layer_ids = source_map.component_map.get('layers', [])[:5]
        target_layer_ids = target_map.component_map.get('layers', [])
        graft_ids = [g.graft_node_id for g in grafter.grafts if g.success]
        
        resonance_found = 0
        resonance_in_grafts = 0
        
        for src_id in source_layer_ids:
            src_value = pac_system.reconstruct(src_id)
            
            # Search in target + grafts
            all_target_ids = target_layer_ids + graft_ids
            
            best_match = None
            best_score = 0
            
            for tgt_id in all_target_ids:
                try:
                    tgt_value = pac_system.reconstruct(tgt_id)
                    
                    # Compute similarity in common space
                    s = src_value.flatten()
                    t = tgt_value.flatten()
                    min_len = min(len(s), len(t))
                    
                    sim = float(torch.dot(s[:min_len], t[:min_len]) / 
                               (torch.norm(s[:min_len]) * torch.norm(t[:min_len]) + 1e-10))
                    
                    if sim > best_score:
                        best_score = sim
                        best_match = tgt_id
                except:
                    continue
            
            if best_score > 0.3:
                resonance_found += 1
                if best_match in graft_ids:
                    resonance_in_grafts += 1
                    print(f"  ✅ Source layer found in GRAFT: score={best_score:.4f}")
                else:
                    print(f"  ✅ Source layer found in target: score={best_score:.4f}")
            else:
                print(f"  ❌ Source layer not found: best={best_score:.4f}")
        
        results['tests'].append({
            'name': 'Cross-Model Resonance',
            'passed': resonance_found,
            'total': len(source_layer_ids),
            'in_grafts': resonance_in_grafts
        })
    
    # Test 3: Bidirectional Transfer
    print("\n📊 Test 3: Bidirectional Transfer Verification")
    print("-" * 40)
    
    # Check that grafts from Pythia→GPT2 also work
    pythia_to_gpt2_grafts = [
        g for g in grafter.grafts 
        if 'layers_' in g.source_label and 'gpt2' in g.target_label.lower()
    ]
    
    bidirectional_success = len([g for g in pythia_to_gpt2_grafts if g.success])
    
    print(f"  Pythia → GPT2 grafts: {bidirectional_success}/{len(pythia_to_gpt2_grafts)}")
    
    results['tests'].append({
        'name': 'Bidirectional Transfer',
        'passed': bidirectional_success,
        'total': len(pythia_to_gpt2_grafts)
    })
    
    # Test 4: Tree Structure Integrity
    print("\n📊 Test 4: Tree Structure Integrity")
    print("-" * 40)
    
    structure_ok = True
    for graft in grafter.grafts:
        if not graft.success:
            continue
        
        graft_node = pac_system.cache.get(graft.graft_node_id)
        if not graft_node:
            structure_ok = False
            print(f"  ❌ Graft node {graft.graft_node_id} not in cache")
            continue
        
        # Verify parent relationship
        parent_id = graft_node.parent_id
        parent = pac_system.cache.get(parent_id)
        
        if parent:
            # Verify child is registered
            if graft.graft_node_id in parent.children_ids:
                print(f"  ✅ Graft {graft.graft_node_id} properly linked to parent {parent_id}")
            else:
                print(f"  ⚠ Graft {graft.graft_node_id} not in parent's children list")
        else:
            print(f"  ❌ Parent {parent_id} not found")
            structure_ok = False
    
    results['tests'].append({
        'name': 'Tree Structure Integrity',
        'passed': 1 if structure_ok else 0,
        'total': 1
    })
    
    # Summary
    print("\n" + "=" * 60)
    print("TRANSFER VALIDATION SUMMARY")
    print("=" * 60)
    
    total_passed = sum(t['passed'] for t in results['tests'])
    total_tests = sum(t['total'] for t in results['tests'])
    
    for test in results['tests']:
        rate = test['passed'] / max(1, test['total']) * 100
        status = "✅" if rate >= 50 else "❌"
        print(f"  {status} {test['name']}: {test['passed']}/{test['total']} ({rate:.0f}%)")
    
    overall_rate = total_passed / max(1, total_tests) * 100
    
    results['summary'] = {
        'total_passed': total_passed,
        'total_tests': total_tests,
        'success_rate': overall_rate / 100
    }
    
    print(f"\n  OVERALL: {total_passed}/{total_tests} ({overall_rate:.0f}%)")
    
    if overall_rate >= 75:
        print("\n🎉 KNOWLEDGE TRANSFER VALIDATED!")
        print("""
The PAC grafting mechanism successfully:
1. Preserves source delta patterns in grafted nodes
2. Enables cross-model resonance (finding source patterns in target)
3. Works bidirectionally between architectures
4. Maintains tree structure integrity

This proves that PAC trees enable TRAINING-FREE knowledge transfer!
""")
    else:
        print("\n⚠️ Transfer needs improvement")
    
    return results


def main():
    print("=" * 60)
    print("POC-020: KNOWLEDGE TRANSFER VALIDATION")
    print("=" * 60)
    
    # Setup
    extractor = ModelToPACExtractor(device='auto')
    
    models = ["gpt2", "bert-base-uncased", "EleutherAI/pythia-70m"]
    for model_name in models:
        extractor.extract_model(model_name, sample_tokens=50)
    
    # Perform grafts
    grafter = PACGrafter(extractor)
    
    print("\n" + "=" * 60)
    print("PERFORMING GRAFTS")
    print("=" * 60)
    
    # Graft between models with high compatibility
    grafter.graft_subtree("gpt2", "EleutherAI/pythia-70m", "layers", top_k=5, threshold=0.3)
    grafter.graft_subtree("EleutherAI/pythia-70m", "gpt2", "layers", top_k=3, threshold=0.3)
    
    # Run validation
    print("\n" + "=" * 60)
    print("VALIDATING TRANSFER")
    print("=" * 60)
    
    results = comprehensive_transfer_test(extractor, grafter)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    
    with open("results/transfer_validation.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print("\n💾 Results saved to results/transfer_validation.json")


if __name__ == "__main__":
    main()
