"""
True no-backprop training using only SEC-PAC dynamics

NO OPTIMIZER. NO BACKWARD(). NO GRADIENTS.
Learning through field dynamics alone.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json
from pathlib import Path

# Constants from Dawn Field Theory
XI_CRITICAL = 1.0571  # SEC critical point
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
CRYSTALLIZATION_THRESHOLD = 0.15


class SECCollapseOperator:
    """Pure SEC collapse without gradients"""
    
    def __init__(self, dim: int):
        self.dim = dim
        self.crystallized_patterns = {}
        self.entropy_history = []
        
    def collapse(self, pattern: torch.Tensor, iterations: int = 30) -> torch.Tensor:
        """
        SEC collapse: C(S) = S * exp(-ξ * S)
        No gradients, pure field dynamics
        """
        with torch.no_grad():  # Explicitly no gradients
            current = pattern.clone()
            
            for i in range(iterations):
                # Compute entropy (Shannon)
                probs = torch.softmax(current, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                
                # Collapse operator
                collapse_factor = torch.exp(-XI_CRITICAL * entropy)
                current = current * collapse_factor.unsqueeze(-1)
                
                # Track entropy
                avg_entropy = entropy.mean().item()
                self.entropy_history.append(avg_entropy)
                
                # Crystallization check
                if avg_entropy < CRYSTALLIZATION_THRESHOLD:
                    pattern_key = self._hash_pattern(current)
                    self.crystallized_patterns[pattern_key] = current.clone()
                    
            return current
            
    def _hash_pattern(self, pattern: torch.Tensor) -> str:
        """Create hashable key for pattern"""
        return str(pattern.flatten()[:10].tolist())


class PACConservationField:
    """PAC conservation without optimization"""
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.field = torch.zeros(vocab_size, vocab_size)  # Transition field
        self.conservation_values = {}
        
    def update_field(self, source: int, target: int, resonance: float):
        """
        Update field through resonance, not gradients
        Resonance = how well patterns align
        """
        with torch.no_grad():
            # Current field value
            current = self.field[source, target].item()
            
            # Resonance update (no learning rate, no gradient)
            # New value is weighted average based on resonance strength
            self.field[source, target] = (1 - resonance) * current + resonance
            
            # Maintain conservation: row sums = 1
            self._enforce_conservation()
            
    def _enforce_conservation(self):
        """Ensure f(parent) = Σf(children) without gradients"""
        with torch.no_grad():
            # Normalize rows to maintain probability conservation
            row_sums = self.field.sum(dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            self.field = self.field / row_sums
            
    def get_next(self, token: int) -> int:
        """Get next token based on field dynamics"""
        with torch.no_grad():
            probs = self.field[token]
            if probs.sum() == 0:
                # Uniform if no field established
                probs = torch.ones(self.vocab_size) / self.vocab_size
            return torch.multinomial(probs, 1).item()


class ResonanceSkillLearner:
    """Learn skills through resonance, not backprop"""
    
    def __init__(self):
        self.skills = {}  # pattern -> response mapping
        self.resonance_threshold = 0.7
        
    def compute_resonance(self, pattern1: torch.Tensor, pattern2: torch.Tensor) -> float:
        """
        Compute resonance between patterns (like harmonics)
        High resonance = patterns align well
        """
        with torch.no_grad():
            # Normalized dot product (cosine similarity)
            p1_norm = pattern1 / (pattern1.norm() + 1e-10)
            p2_norm = pattern2 / (pattern2.norm() + 1e-10)
            
            resonance = (p1_norm * p2_norm).sum().item()
            return max(0, resonance)  # Ensure non-negative
            
    def learn_skill(self, input_pattern: torch.Tensor, output_pattern: torch.Tensor):
        """Learn skill if resonance is high enough"""
        with torch.no_grad():
            # Check if input resonates with existing skills
            best_resonance = 0
            best_key = None
            
            for key, skill in self.skills.items():
                res = self.compute_resonance(input_pattern, skill['input'])
                if res > best_resonance:
                    best_resonance = res
                    best_key = key
                    
            # If high resonance with existing, strengthen it
            if best_resonance > self.resonance_threshold and best_key:
                # Blend patterns (no gradients!)
                alpha = best_resonance
                self.skills[best_key]['output'] = (
                    alpha * self.skills[best_key]['output'] + 
                    (1 - alpha) * output_pattern
                )
                self.skills[best_key]['strength'] += 0.1
            else:
                # Create new skill
                new_key = len(self.skills)
                self.skills[new_key] = {
                    'input': input_pattern.clone(),
                    'output': output_pattern.clone(),
                    'strength': 1.0
                }
                
    def apply_skill(self, input_pattern: torch.Tensor) -> Optional[torch.Tensor]:
        """Apply learned skill based on resonance"""
        with torch.no_grad():
            best_resonance = 0
            best_output = None
            
            for skill in self.skills.values():
                res = self.compute_resonance(input_pattern, skill['input'])
                if res > best_resonance:
                    best_resonance = res
                    best_output = skill['output']
                    
            if best_resonance > self.resonance_threshold:
                return best_output
            return None


class NoBackpropTransformer:
    """Transformer that learns without any backprop"""
    
    def __init__(self, vocab_size: int, dim: int = 128, device: str = 'cpu'):
        self.vocab_size = vocab_size
        self.dim = dim
        self.device = device
        
        # Components (no nn.Module, no parameters!)
        self.embeddings = torch.randn(vocab_size, dim, device=device) * 0.02
        self.sec_operator = SECCollapseOperator(dim)
        self.pac_field = PACConservationField(vocab_size)
        self.skill_learner = ResonanceSkillLearner()
        
        # Stats
        self.stats = {
            'crystallized': 0,
            'skills_learned': 0,
            'field_updates': 0,
            'resonance_total': 0.0,
            'resonance_count': 0
        }
        
    def process_sequence(self, tokens: List[int]) -> List[int]:
        """Process sequence without any gradients"""
        output = []
        
        with torch.no_grad():  # Ensure absolutely no gradients
            for i in range(len(tokens) - 1):
                # Get current pattern
                current_embed = self.embeddings[tokens[i]]
                
                # SEC collapse
                collapsed = self.sec_operator.collapse(current_embed, iterations=10)
                
                # Check for skill application
                skill_output = self.skill_learner.apply_skill(collapsed)
                
                if skill_output is not None:
                    # Use skill to predict
                    # Find closest token to skill output
                    distances = ((self.embeddings - skill_output.unsqueeze(0)) ** 2).sum(dim=1)
                    next_token = distances.argmin().item()
                else:
                    # Use PAC field
                    next_token = self.pac_field.get_next(tokens[i])
                    
                output.append(next_token)
                
                # Learn from observation (not loss!)
                if i < len(tokens) - 1:
                    actual_next = tokens[i + 1]
                    
                    # Update PAC field based on observation
                    resonance = 1.0 if next_token == actual_next else 0.3
                    self.pac_field.update_field(tokens[i], actual_next, resonance)
                    self.stats['field_updates'] += 1
                    self.stats['resonance_total'] += resonance
                    self.stats['resonance_count'] += 1
                    
                    # Learn skill if pattern is good
                    if resonance > 0.5:
                        next_embed = self.embeddings[actual_next]
                        self.skill_learner.learn_skill(collapsed, next_embed)
                        self.stats['skills_learned'] = len(self.skill_learner.skills)
                        
        self.stats['crystallized'] = len(self.sec_operator.crystallized_patterns)
        return output
        
    def generate(self, prompt: List[int], max_length: int = 50) -> List[int]:
        """Generate without any gradients"""
        result = prompt.copy()
        
        with torch.no_grad():
            for _ in range(max_length - len(prompt)):
                if len(result) == 0:
                    break
                    
                last_token = result[-1]
                last_embed = self.embeddings[last_token]
                
                # Collapse
                collapsed = self.sec_operator.collapse(last_embed, iterations=5)
                
                # Try skill first
                skill_output = self.skill_learner.apply_skill(collapsed)
                
                if skill_output is not None:
                    distances = ((self.embeddings - skill_output.unsqueeze(0)) ** 2).sum(dim=1)
                    next_token = distances.argmin().item()
                else:
                    # Fall back to PAC field
                    next_token = self.pac_field.get_next(last_token)
                    
                result.append(next_token)
                
        return result


def train_no_backprop(texts: List[str], epochs: int = 10, vocab_size: int = 256):
    """Train without any optimizer or gradients"""
    
    print("="*70)
    print("TRUE NO-BACKPROP TRAINING")
    print("="*70)
    print("\n⚠️  NO OPTIMIZER")
    print("⚠️  NO BACKWARD()")
    print("⚠️  NO GRADIENTS")
    print("✅ Learning through SEC collapse and PAC conservation ONLY\n")
    print("="*70)
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NoBackpropTransformer(vocab_size=vocab_size, dim=64, device=device)
    
    print(f"\n📍 Device: {device}")
    print(f"📊 Vocab size: {vocab_size}, Embedding dim: 64")
    
    for epoch in range(epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*70}")
        
        epoch_correct = 0
        epoch_total = 0
        
        for text in texts:
            # Simple tokenization
            tokens = [ord(c) % vocab_size for c in text]
            
            # Process sequence (learning happens here)
            predicted = model.process_sequence(tokens)
            
            # Compute accuracy (for monitoring only, not for training!)
            correct = sum(1 for p, t in zip(predicted, tokens[1:]) if p == t)
            accuracy = correct / len(predicted) if predicted else 0
            
            epoch_correct += correct
            epoch_total += len(predicted)
            
            print(f"  '{text[:40]}{'...' if len(text) > 40 else ''}' → {accuracy:.1%}")
            
        # Epoch stats
        epoch_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
        avg_resonance = model.stats['resonance_total'] / max(model.stats['resonance_count'], 1)
        
        print(f"\n  📊 Epoch {epoch + 1} Stats:")
        print(f"     Accuracy: {epoch_accuracy:.1%}")
        print(f"     Crystallized: {model.stats['crystallized']}")
        print(f"     Skills: {model.stats['skills_learned']}")
        print(f"     Field updates: {model.stats['field_updates']}")
        print(f"     Avg resonance: {avg_resonance:.3f}")
              
    return model


def test_generation(model: NoBackpropTransformer, prompts: List[str], vocab_size: int = 256):
    """Test generation without gradients"""
    
    print("\n" + "="*70)
    print("GENERATION TEST (NO BACKPROP)")
    print("="*70)
    
    for prompt in prompts:
        tokens = [ord(c) % vocab_size for c in prompt]
        generated = model.generate(tokens, max_length=50)
        
        # Decode
        text = ''.join(chr(min(t, 127)) for t in generated)
        print(f"\n  Prompt: '{prompt}'")
        print(f"  Generated: '{text}'")


if __name__ == "__main__":
    # Training data
    training_texts = [
        "The cat sat on the mat.",
        "The dog ran in the park.",
        "Birds fly in the sky.",
        "Fish swim in the water.",
        "The sun shines brightly.",
        "The moon glows at night.",
        "Trees grow in the forest.",
        "Flowers bloom in spring.",
        "Rain falls from clouds.",
        "Wind blows through trees.",
        "Stars twinkle at night.",
        "The ocean is deep.",
        "Mountains reach high.",
        "Rivers flow to sea.",
        "Children play outside.",
    ]
    
    print("\n📚 Training Texts:")
    for i, text in enumerate(training_texts, 1):
        print(f"  {i}. {text}")
    
    # Train without backprop
    model = train_no_backprop(training_texts, epochs=5, vocab_size=256)
    
    # Test generation
    test_prompts = [
        "The cat",
        "Birds",
        "The sun",
        "Trees"
    ]
    
    test_generation(model, test_prompts, vocab_size=256)
    
    # Final verification
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    print("✅ No torch.optim used")
    print("✅ No loss.backward() called")
    print("✅ No gradients computed")
    print("✅ Learning through field dynamics only")
    print(f"✅ {model.stats['crystallized']} patterns crystallized")
    print(f"✅ {model.stats['skills_learned']} skills learned")
    print(f"✅ {model.stats['field_updates']} field updates")
    print("="*70)
