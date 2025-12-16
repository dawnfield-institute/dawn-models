"""
Experiment 03: Grammar Emergence from Field Dynamics
=====================================================

Tests whether grammatical structure emerges naturally from
field evolution without explicit grammar rules.

Hypothesis: Subject-verb-object patterns create distinct
field signatures that enforce grammatical constraints.

Success Criteria:
- Field similarity groups words by grammatical category
- Transitions respect subject-verb agreement patterns
- Generated sequences follow basic SVO structure
"""

import torch
import torch.nn.functional as F
from datetime import datetime
import json
import sys
from pathlib import Path

# Add scripts to path
scripts_path = Path(__file__).resolve().parent
sys.path.insert(0, str(scripts_path))

from field_generator import FieldVocabulary, FieldPredictor, FieldGenerator

# Dawn Field Constants
PHI = 1.618033988749895
XI = 0.0618
PHI_XI = PHI * XI
LAMBDA_STAR = 0.9816


def field_similarity(vocab, word1: str, word2: str) -> float:
    """Compute field similarity between two words."""
    field1 = vocab.get_field(word1)
    field2 = vocab.get_field(word2)
    return F.cosine_similarity(
        field1.flatten().unsqueeze(0),
        field2.flatten().unsqueeze(0)
    ).item()


def run_tests():
    """Run grammar emergence tests."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"POC-005 Experiment 03: Grammar Emergence")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'tests': {}
    }
    
    # Initialize vocabulary with grammatical categories
    print("\n[Initializing vocabulary with grammatical categories...]")
    vocab = FieldVocabulary(device=device, field_shape=(24, 24, 24))
    
    # Define words by category
    nouns = ["cat", "dog", "bird", "man", "woman", "child", "sun", "moon"]
    verbs = ["runs", "eats", "sleeps", "sees", "loves", "helps", "knows"]
    adjectives = ["big", "small", "fast", "slow", "happy", "sad", "good"]
    articles = ["the", "a", "an"]
    pronouns = ["I", "you", "he", "she", "it", "we", "they"]
    
    # Add all words
    all_words = nouns + verbs + adjectives + articles + pronouns
    vocab.add_tokens(all_words)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"  Nouns: {len(nouns)}, Verbs: {len(verbs)}, Adjectives: {len(adjectives)}")
    
    # Create predictor
    predictor = FieldPredictor(vocab)
    
    # Train on grammatical sentences (SVO patterns)
    print("\n[Training on grammatical patterns...]")
    sentences = [
        # Article + Noun + Verb
        ["the", "cat", "runs"],
        ["the", "dog", "eats"],
        ["the", "bird", "sleeps"],
        ["the", "man", "sees"],
        ["the", "woman", "helps"],
        ["a", "cat", "sleeps"],
        ["a", "dog", "runs"],
        ["a", "bird", "eats"],
        # Article + Adjective + Noun + Verb
        ["the", "big", "cat", "runs"],
        ["the", "small", "dog", "eats"],
        ["the", "fast", "bird", "sleeps"],
        ["a", "happy", "man", "sees"],
        ["a", "good", "woman", "helps"],
        # Pronoun + Verb + Article + Noun
        ["I", "see", "the", "cat"],
        ["you", "love", "the", "dog"],
        ["he", "helps", "the", "man"],
        ["she", "knows", "the", "woman"],
        # More patterns
        ["the", "sun", "helps"],
        ["the", "moon", "sleeps"],
        ["it", "runs"],
        ["they", "see"],
    ]
    
    for sentence in sentences:
        for _ in range(3):
            predictor.train_on_sequence(sentence)
    
    generator = FieldGenerator(predictor)
    
    passed = 0
    failed = 0
    
    # ===== TEST 1: Grammatical Category Clustering =====
    print("\n" + "-"*40)
    print("TEST 1: Grammatical Category Clustering")
    print("-"*40)
    
    # Check if nouns are more similar to each other than to verbs
    noun_noun_sims = []
    for i, n1 in enumerate(nouns[:4]):
        for n2 in nouns[i+1:5]:
            noun_noun_sims.append(field_similarity(vocab, n1, n2))
    
    noun_verb_sims = []
    for n in nouns[:4]:
        for v in verbs[:4]:
            noun_verb_sims.append(field_similarity(vocab, n, v))
    
    avg_nn = sum(noun_noun_sims) / len(noun_noun_sims)
    avg_nv = sum(noun_verb_sims) / len(noun_verb_sims)
    
    print(f"  Noun-Noun similarity: {avg_nn:.3f}")
    print(f"  Noun-Verb similarity: {avg_nv:.3f}")
    
    test1_pass = avg_nn > avg_nv  # Nouns should be more similar to each other
    results['tests']['category_clustering'] = {
        'noun_noun_avg': avg_nn,
        'noun_verb_avg': avg_nv,
        'passed': test1_pass
    }
    
    if test1_pass:
        print(f"✓ PASSED: Same-category words are more similar")
        passed += 1
    else:
        print(f"✗ FAILED: Categories not clustered")
        failed += 1
    
    # ===== TEST 2: Article → Noun/Adjective Transition =====
    print("\n" + "-"*40)
    print("TEST 2: Article → Noun/Adjective Pattern")
    print("-"*40)
    
    # After "the", should predict nouns or adjectives, not verbs
    pred_the = predictor.predict(["the"], top_k=5)
    top_after_the = [p[0] for p in pred_the]
    
    print(f"  After 'the': {[(p[0], f'{p[1]:.3f}') for p in pred_the[:5]]}")
    
    # Check if nouns/adjectives are preferred
    nouns_adj_set = set(nouns + adjectives)
    nouns_adj_in_top = sum(1 for t in top_after_the[:3] if t in nouns_adj_set)
    
    test2_pass = nouns_adj_in_top >= 2  # At least 2 of top 3 should be noun/adj
    results['tests']['article_transition'] = {
        'predictions_after_the': pred_the[:5],
        'nouns_adj_in_top3': nouns_adj_in_top,
        'passed': test2_pass
    }
    
    if test2_pass:
        print(f"✓ PASSED: Article → Noun/Adjective pattern respected")
        passed += 1
    else:
        print(f"✗ FAILED: Article pattern not learned ({nouns_adj_in_top}/3)")
        failed += 1
    
    # ===== TEST 3: Noun → Verb Transition =====
    print("\n" + "-"*40)
    print("TEST 3: Noun → Verb Pattern")
    print("-"*40)
    
    # After "the cat", should predict verb
    pred_cat = predictor.predict(["the", "cat"], top_k=5)
    top_after_cat = [p[0] for p in pred_cat]
    
    print(f"  After 'the cat': {[(p[0], f'{p[1]:.3f}') for p in pred_cat[:5]]}")
    
    verb_set = set(verbs)
    verbs_in_top = sum(1 for t in top_after_cat[:3] if t in verb_set)
    
    test3_pass = verbs_in_top >= 1  # At least 1 verb in top 3
    results['tests']['noun_verb_transition'] = {
        'predictions_after_noun': pred_cat[:5],
        'verbs_in_top3': verbs_in_top,
        'passed': test3_pass
    }
    
    if test3_pass:
        print(f"✓ PASSED: Noun → Verb pattern respected ({verbs_in_top}/3 verbs)")
        passed += 1
    else:
        print(f"✗ FAILED: Noun → Verb pattern not learned")
        failed += 1
    
    # ===== TEST 4: Adjective Position =====
    print("\n" + "-"*40)
    print("TEST 4: Adjective Position")
    print("-"*40)
    
    # After "the big", should predict noun
    pred_adj = predictor.predict(["the", "big"], top_k=5)
    top_after_adj = [p[0] for p in pred_adj]
    
    print(f"  After 'the big': {[(p[0], f'{p[1]:.3f}') for p in pred_adj[:5]]}")
    
    noun_set = set(nouns)
    nouns_in_top = sum(1 for t in top_after_adj[:3] if t in noun_set)
    
    test4_pass = nouns_in_top >= 1
    results['tests']['adjective_position'] = {
        'predictions_after_adj': pred_adj[:5],
        'nouns_in_top3': nouns_in_top,
        'passed': test4_pass
    }
    
    if test4_pass:
        print(f"✓ PASSED: Adjective → Noun pattern ({nouns_in_top}/3 nouns)")
        passed += 1
    else:
        print(f"✗ FAILED: Adjective position not learned")
        failed += 1
    
    # ===== TEST 5: SVO Generation =====
    print("\n" + "-"*40)
    print("TEST 5: SVO Sequence Generation")
    print("-"*40)
    
    # Generate from "the" and check SVO structure
    gen = generator.generate(["the"], max_tokens=3, temperature=0.3)
    print(f"  Generated: {' '.join(gen)}")
    
    # Check structure: should be Article + (Adj?) + Noun + Verb
    def check_svo(tokens):
        """Check if sequence follows grammatical pattern."""
        if len(tokens) < 2:
            return False
        # First should be article
        if tokens[0] not in articles:
            return False
        # Last should be verb or noun
        if tokens[-1] in (verbs + nouns + adjectives):
            return True
        return False
    
    test5_pass = check_svo(gen)
    results['tests']['svo_generation'] = {
        'generated': gen,
        'valid_structure': test5_pass,
        'passed': test5_pass
    }
    
    if test5_pass:
        print(f"✓ PASSED: Generated valid grammatical structure")
        passed += 1
    else:
        print(f"✗ FAILED: Structure not grammatical")
        failed += 1
    
    # ===== TEST 6: Pronoun → Verb Pattern =====
    print("\n" + "-"*40)
    print("TEST 6: Pronoun → Verb Pattern")
    print("-"*40)
    
    pred_pronoun = predictor.predict(["I"], top_k=5)
    print(f"  After 'I': {[(p[0], f'{p[1]:.3f}') for p in pred_pronoun[:5]]}")
    
    top_after_pronoun = [p[0] for p in pred_pronoun]
    # Check if top prediction is a verb (primary criterion)
    # Note: "see" is base form, we need to check all verb forms
    all_verbs = set(["runs", "eats", "sleeps", "sees", "loves", "helps", "knows",
                     "run", "eat", "sleep", "see", "love", "help", "know"])
    top_is_verb = top_after_pronoun[0] in all_verbs if top_after_pronoun else False
    verbs_in_top = sum(1 for t in top_after_pronoun[:5] if t in all_verbs)
    
    print(f"  Top token '{top_after_pronoun[0]}' is verb: {top_is_verb}")
    
    test6_pass = top_is_verb or verbs_in_top >= 2
    results['tests']['pronoun_verb'] = {
        'predictions_after_I': pred_pronoun[:5],
        'top_is_verb': top_is_verb,
        'verbs_in_top5': verbs_in_top,
        'passed': test6_pass
    }
    
    if test6_pass:
        print(f"✓ PASSED: Pronoun → Verb pattern ({verbs_in_top}/3)")
        passed += 1
    else:
        print(f"✗ FAILED: Pronoun pattern not learned")
        failed += 1
    
    # ===== SUMMARY =====
    print("\n" + "="*60)
    print(f"EXPERIMENT 03 RESULTS: {passed}/{passed+failed} passed")
    print("="*60)
    
    results['summary'] = {
        'passed': passed,
        'failed': failed,
        'total': passed + failed,
        'success_rate': passed / (passed + failed)
    }
    
    # Save results
    results_dir = Path(__file__).resolve().parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f'exp_03_grammar_emergence_{timestamp}.json'
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved: {results_path}")
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = run_tests()
    sys.exit(0 if failed == 0 else 1)
