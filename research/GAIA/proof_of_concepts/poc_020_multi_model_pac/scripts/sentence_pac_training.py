"""
Hierarchical Sentence PAC Training
===================================

Train GAIA on sentence combinations using hierarchical PAC.
Sentences are word combinations, words are token combinations.
NO BACKPROP - builds on our existing PAC geometric training.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Add paths to use our EXISTING work
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "poc_019_true_no_backprop"))
sys.path.insert(0, r"c:\Users\peter\repos\Dawn Field Institute\fracton")

# Constants
PHI = (1 + np.sqrt(5)) / 2
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


class HierarchicalSECCollapse:
    """SEC collapse at different hierarchy levels"""
    
    def __init__(self, dim: int):
        self.dim = dim
        # Different phi powers for different levels
        self.level_scales = {
            0: 1.0,           # Token level
            1: PHI,           # Word level  
            2: PHI ** 2,      # Sentence level
            3: PHI ** 3,      # Paragraph level
        }
        
    def collapse(self, x: torch.Tensor, level: int = 0) -> torch.Tensor:
        """Collapse tensor at specified hierarchy level"""
        scale = self.level_scales.get(level, 1.0)
        
        # Apply scaled normalization
        norm = x.norm(dim=-1, keepdim=True) + 1e-8
        normalized = x / norm
        
        # Scale by level
        return normalized * scale


class HierarchicalSkillLearner:
    """Learn skills at each hierarchy level - NO BACKPROP"""
    
    def __init__(self):
        self.skills = {}
        self.skill_count = 0
        
    def learn_skill(self, input_pattern: torch.Tensor, output_pattern: torch.Tensor, level: int):
        """Learn a skill mapping input to output at given level"""
        skill_id = f"L{level}_{self.skill_count}"
        self.skills[skill_id] = {
            'input': input_pattern.detach().cpu(),
            'output': output_pattern.detach().cpu(),
            'level': level,
            'strength': 1.0
        }
        self.skill_count += 1
        
    def find_matching_skill(self, input_pattern: torch.Tensor, level: int, threshold: float = 0.5) -> Optional[torch.Tensor]:
        """Find skill that matches input pattern"""
        best_match = None
        best_sim = threshold
        
        # Ensure input is on CPU for comparison
        input_cpu = input_pattern.detach().cpu()
        
        for skill_id, skill in self.skills.items():
            if skill['level'] == level:
                sim = torch.cosine_similarity(
                    input_cpu.flatten().unsqueeze(0),
                    skill['input'].flatten().unsqueeze(0)
                ).item()
                
                if sim > best_sim:
                    best_sim = sim
                    best_match = skill['output']
                    
        return best_match


class PACConfluenceTree:
    """PAC confluence tree for pattern storage"""
    
    def __init__(self):
        self.confluence = {}
        self.depth = 0
        
    def update(self, context_hash: int, next_hash: int):
        """Update confluence with context -> next mapping"""
        if context_hash not in self.confluence:
            self.confluence[context_hash] = {}
        self.confluence[context_hash][next_hash] = \
            self.confluence[context_hash].get(next_hash, 0) + 1
            
    def get_candidates(self, context_hash: int) -> Dict[int, int]:
        """Get candidate next items for context"""
        return self.confluence.get(context_hash, {})


class SentencePACModel:
    """GAIA model with hierarchical sentence-level PAC"""
    
    def __init__(self, device):
        self.device = device
        self.vocab_size = 50257
        self.embed_dim = 256
        self.max_layers = 13
        self.current_layers = 1
        
        # Embeddings
        self.embeddings = torch.randn(self.vocab_size, self.embed_dim, device=device) * 0.02
        
        # Hierarchical confluence trees
        self.token_confluence = {}  # Level 0
        self.word_confluence = PACConfluenceTree()  # Level 1
        self.sentence_confluence = PACConfluenceTree()  # Level 2
        
        # Lazy layers
        self.layers = []
        for _ in range(self.max_layers):
            self.layers.append({
                'materialized': False,
                'pattern': None,
                'activations': 0
            })
        self.layers[0]['materialized'] = True
        
        # Target geometry (from our geometric training)
        self.target_layers = 13
        self.target_dim = 640
        
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through materialized layers"""
        x = self.embeddings[input_ids]
        
        for i in range(self.current_layers):
            if self.layers[i]['materialized']:
                # Simple attention-like operation
                attn = torch.softmax(torch.matmul(x, x.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
                x = torch.matmul(attn, x)
                self.layers[i]['activations'] += 1
                
                # Store pattern
                if self.layers[i]['pattern'] is None:
                    self.layers[i]['pattern'] = x[0, -1].detach().cpu().numpy()
                    
        # Project to vocab
        logits = torch.matmul(x, self.embeddings.T)
        return x, logits
        
    def update_token_confluence(self, context: Tuple, next_token: int):
        """Update token-level confluence"""
        context_hash = hash(context) % 100000
        if context_hash not in self.token_confluence:
            self.token_confluence[context_hash] = {}
        self.token_confluence[context_hash][next_token] = \
            self.token_confluence[context_hash].get(next_token, 0) + 1
            
    def grow_layer(self) -> bool:
        """Grow a new layer if needed"""
        if self.current_layers >= self.max_layers:
            return False
            
        # Check if we should grow (geometric criterion)
        layer_ratio = self.current_layers / self.target_layers
        if layer_ratio >= 0.9:
            return False
            
        # Materialize next layer
        self.layers[self.current_layers]['materialized'] = True
        self.current_layers += 1
        return True


class SentencePACTrainer:
    """
    Train GAIA on sentence combinations hierarchically.
    NO BACKPROP - uses SEC collapse and confluence trees.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Create model
        self.model = SentencePACModel(self.device)
        
        # Hierarchical skill learners
        self.token_skills = HierarchicalSkillLearner()
        self.word_skills = HierarchicalSkillLearner()
        self.sentence_skills = HierarchicalSkillLearner()
        
        # SEC collapse for each level
        self.sec_collapse = HierarchicalSECCollapse(self.model.embed_dim)
        
        # Transfer embeddings from source models
        self._transfer_embeddings()
        
    def _transfer_embeddings(self):
        """Transfer embeddings from GPT-2 and Pythia"""
        print("\nTransferring source embeddings...")
        embeddings = []
        
        try:
            from transformers import GPT2Model
            gpt2 = GPT2Model.from_pretrained('gpt2')
            gpt2_emb = gpt2.wte.weight.detach()[:self.model.vocab_size, :self.model.embed_dim]
            embeddings.append(gpt2_emb)
            print(f"  GPT-2: {gpt2_emb.shape}")
            del gpt2
        except Exception as e:
            print(f"  GPT-2 failed: {e}")
            
        try:
            from transformers import GPTNeoXForCausalLM
            pythia = GPTNeoXForCausalLM.from_pretrained('EleutherAI/pythia-70m')
            pythia_emb = pythia.gpt_neox.embed_in.weight.detach()[:self.model.vocab_size]
            if pythia_emb.shape[1] < self.model.embed_dim:
                pad = torch.zeros(pythia_emb.shape[0], self.model.embed_dim - pythia_emb.shape[1])
                pythia_emb = torch.cat([pythia_emb, pad], dim=1)
            else:
                pythia_emb = pythia_emb[:, :self.model.embed_dim]
            embeddings.append(pythia_emb)
            print(f"  Pythia: {pythia_emb.shape}")
            del pythia
        except Exception as e:
            print(f"  Pythia failed: {e}")
            
        torch.cuda.empty_cache()
        
        if embeddings:
            with torch.no_grad():
                avg = torch.stack(embeddings).mean(dim=0).to(self.device)
                self.model.embeddings[:avg.shape[0]] = avg
                print(f"  Transferred avg of {len(embeddings)} models")
                
    def train(self, corpus: List[str], epochs: int = 10):
        """Train on sentence combinations hierarchically"""
        
        print("\n" + "="*60)
        print("Hierarchical Sentence PAC Training")
        print("NO BACKPROP - SEC Collapse + Confluence Trees")
        print("="*60)
        
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        stats_history = []
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            
            epoch_stats = {
                'tokens': 0,
                'words': 0,
                'sentences': 0,
                'token_skills': 0,
                'word_skills': 0,
                'sentence_skills': 0
            }
            
            for text in corpus:
                # Split into sentences
                sentences = [s.strip() for s in text.split('.') if s.strip()]
                epoch_stats['sentences'] += len(sentences)
                
                # Process each sentence
                for sent_idx, sentence in enumerate(sentences):
                    # Tokenize
                    tokens = tokenizer.encode(sentence, max_length=128, truncation=True)
                    if len(tokens) < 2:
                        continue
                        
                    epoch_stats['tokens'] += len(tokens)
                    
                    # Split into words
                    words = sentence.split()
                    epoch_stats['words'] += len(words)
                    
                    # LEVEL 0: Token-level learning
                    token_skills_before = len(self.token_skills.skills)
                    self._learn_token_patterns(tokens, tokenizer)
                    epoch_stats['token_skills'] += len(self.token_skills.skills) - token_skills_before
                    
                    # LEVEL 1: Word-level learning
                    word_skills_before = len(self.word_skills.skills)
                    self._learn_word_patterns(words, tokens, tokenizer)
                    epoch_stats['word_skills'] += len(self.word_skills.skills) - word_skills_before
                    
                    # LEVEL 2: Sentence-level learning
                    if sent_idx > 0:
                        sent_skills_before = len(self.sentence_skills.skills)
                        prev_sentence = sentences[sent_idx - 1]
                        self._learn_sentence_transitions(prev_sentence, sentence, tokenizer)
                        epoch_stats['sentence_skills'] += len(self.sentence_skills.skills) - sent_skills_before
                        
            # Check if model should grow (every 3 epochs)
            grew = False
            if (epoch + 1) % 3 == 0:
                grew = self.model.grow_layer()
                
            # Print epoch stats
            print(f"  Tokens: {epoch_stats['tokens']}, Words: {epoch_stats['words']}, Sentences: {epoch_stats['sentences']}")
            print(f"  New skills: T={epoch_stats['token_skills']}, W={epoch_stats['word_skills']}, S={epoch_stats['sentence_skills']}")
            print(f"  Total skills: T={len(self.token_skills.skills)}, W={len(self.word_skills.skills)}, S={len(self.sentence_skills.skills)}")
            print(f"  Token confluence: {len(self.model.token_confluence)} contexts")
            print(f"  Word confluence: {len(self.model.word_confluence.confluence)} contexts")
            print(f"  Sentence confluence: {len(self.model.sentence_confluence.confluence)} contexts")
            print(f"  Model layers: {self.model.current_layers}")
            
            if grew:
                print(f"  🌱 Grew to {self.model.current_layers} layers!")
                
            stats_history.append(epoch_stats)
            
        return {
            'final_layers': self.model.current_layers,
            'token_skills': len(self.token_skills.skills),
            'word_skills': len(self.word_skills.skills),
            'sentence_skills': len(self.sentence_skills.skills),
            'token_confluence': len(self.model.token_confluence),
            'word_confluence': len(self.model.word_confluence.confluence),
            'sentence_confluence': len(self.model.sentence_confluence.confluence),
            'epochs': epochs,
            'history': stats_history
        }
        
    def _learn_token_patterns(self, tokens: List[int], tokenizer):
        """Learn token-level patterns"""
        with torch.no_grad():
            for i in range(len(tokens) - 1):
                # Update token confluence
                context = tuple(tokens[max(0, i-3):i+1])
                next_token = tokens[i + 1]
                self.model.update_token_confluence(context, next_token)
                
                # Learn token transition skill
                curr_embed = self.model.embeddings[tokens[i]]
                next_embed = self.model.embeddings[tokens[i + 1]]
                
                # SEC collapse at token level
                collapsed = self.sec_collapse.collapse(curr_embed.unsqueeze(0), level=0)
                
                # Check resonance
                resonance = torch.cosine_similarity(
                    collapsed.flatten().unsqueeze(0),
                    next_embed.unsqueeze(0)
                ).item()
                
                # Learn if resonant (selective learning)
                if resonance > 0.3:
                    self.token_skills.learn_skill(collapsed.flatten(), next_embed, level=0)
                    
    def _learn_word_patterns(self, words: List[str], tokens: List[int], tokenizer):
        """Learn word-level patterns"""
        with torch.no_grad():
            # Get word boundaries in token space
            word_embeds = []
            for word in words:
                word_tokens = tokenizer.encode(' ' + word, add_special_tokens=False)
                if word_tokens:
                    # Average embedding for word
                    word_embed = self.model.embeddings[word_tokens].mean(dim=0)
                    word_embeds.append(word_embed)
                    
            # Learn word transitions
            for i in range(len(word_embeds) - 1):
                curr_embed = word_embeds[i]
                next_embed = word_embeds[i + 1]
                
                # SEC collapse at word level
                collapsed = self.sec_collapse.collapse(curr_embed.unsqueeze(0), level=1)
                
                # Learn word transition
                self.word_skills.learn_skill(collapsed.flatten(), next_embed, level=1)
                
                # Update word confluence
                word_context = tuple(words[max(0, i-2):i+1])
                context_hash = hash(word_context) % 100000
                next_hash = hash(words[i + 1]) % 10000
                self.model.word_confluence.update(context_hash, next_hash)
                
    def _learn_sentence_transitions(self, prev_sentence: str, curr_sentence: str, tokenizer):
        """Learn sentence-level transitions"""
        with torch.no_grad():
            # Get sentence embeddings
            prev_tokens = tokenizer.encode(prev_sentence, max_length=64, truncation=True)
            curr_tokens = tokenizer.encode(curr_sentence, max_length=64, truncation=True)
            
            if len(prev_tokens) > 0 and len(curr_tokens) > 0:
                prev_embed = self.model.embeddings[prev_tokens].mean(dim=0)
                curr_embed = self.model.embeddings[curr_tokens].mean(dim=0)
                
                # SEC collapse at sentence level
                collapsed = self.sec_collapse.collapse(prev_embed.unsqueeze(0), level=2)
                
                # Learn sentence transition
                self.sentence_skills.learn_skill(collapsed.flatten(), curr_embed, level=2)
                
                # Update sentence confluence
                context_hash = hash(prev_sentence[:50]) % 100000
                next_hash = hash(curr_sentence[:50]) % 10000
                self.model.sentence_confluence.update(context_hash, next_hash)
                
    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate text using hierarchical patterns with diversity"""
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        tokens = tokenizer.encode(prompt)
        recent_tokens = []  # Track recent tokens to avoid repetition
        
        for step in range(max_tokens):
            next_token = None
            
            # Try multiple context lengths for confluence lookup
            for ctx_len in [5, 4, 3, 2]:
                if len(tokens) >= ctx_len:
                    context = tuple(tokens[-ctx_len:])
                    context_hash = hash(context) % 100000
                    
                    if context_hash in self.model.token_confluence:
                        candidates = self.model.token_confluence[context_hash]
                        if candidates:
                            # Filter out recently used tokens
                            filtered = {k: v for k, v in candidates.items() 
                                       if k not in recent_tokens[-5:]}
                            
                            if filtered:
                                # Sample proportionally with temperature
                                total = sum(filtered.values())
                                items = list(filtered.items())
                                weights = [(v/total) ** 0.8 for _, v in items]
                                weight_sum = sum(weights)
                                weights = [w/weight_sum for w in weights]
                                
                                r = np.random.random()
                                cumsum = 0
                                for (tok, _), w in zip(items, weights):
                                    cumsum += w
                                    if r < cumsum:
                                        next_token = tok
                                        break
                                        
                                if next_token is not None:
                                    break
                                    
            # Fall back to skill-based prediction (but limit to avoid loops)
            if next_token is None and len(tokens) > 0 and step < 15:  # Only use skills early
                # Find matching token skill
                curr_embed = self.model.embeddings[tokens[-1]]
                collapsed = self.sec_collapse.collapse(curr_embed.unsqueeze(0), level=0)
                
                matched = self.token_skills.find_matching_skill(collapsed.flatten(), level=0, threshold=0.6)  # Higher threshold
                if matched is not None:
                    # Find closest token to skill output
                    sims = torch.cosine_similarity(
                        matched.to(self.device).unsqueeze(0),
                        self.model.embeddings
                    )
                    # Filter recent tokens more aggressively
                    for rt in recent_tokens[-10:]:
                        sims[rt] = -1
                    # Also filter common words that cause loops
                    common_tokens = [262, 257, 13, 290, 284, 318, 468, 373, 389, 307]  # the, a, ., and, to, is, has, was, are, be
                    for ct in common_tokens:
                        if ct < len(sims):
                            sims[ct] *= 0.5  # Reduce but don't eliminate
                    next_token = sims.argmax().item()
                    
            # Last resort: use model forward with diversity
            if next_token is None:
                input_ids = torch.tensor([tokens[-32:]], device=self.device)
                hidden, logits = self.model.forward(input_ids)
                
                # Sample from top-k
                top_k = 20
                top_probs, top_indices = torch.topk(logits[0, -1], top_k)
                top_probs = torch.softmax(top_probs / 1.0, dim=0)
                
                # Filter recent
                for i, idx in enumerate(top_indices):
                    if idx.item() in recent_tokens[-5:]:
                        top_probs[i] = 0
                        
                if top_probs.sum() > 0:
                    top_probs = top_probs / top_probs.sum()
                    next_token = top_indices[torch.multinomial(top_probs, 1)].item()
                else:
                    next_token = top_indices[0].item()
                    
            tokens.append(next_token)
            recent_tokens.append(next_token)
            if len(recent_tokens) > 15:
                recent_tokens.pop(0)
            
            # Stop on period or EOS
            decoded = tokenizer.decode([next_token])
            if '.' in decoded or next_token == tokenizer.eos_token_id:
                break
                
        return tokenizer.decode(tokens)


def main():
    """Train GAIA with hierarchical sentence PAC patterns"""
    
    print("="*60)
    print("Hierarchical Sentence PAC Training")
    print("NO BACKPROP - Building on existing PAC work")
    print("="*60)
    
    trainer = SentencePACTrainer()
    
    # Training corpus with multi-sentence texts
    corpus = [
        "The cat sat on the mat. It was a sunny day. Birds sang in the trees.",
        "Scientists study the natural world. They use experiments to test hypotheses. Knowledge grows through research.",
        "The ocean is vast and deep. Many creatures live beneath the waves. Coral reefs are like underwater cities.",
        "Computers process information quickly. They follow instructions called programs. Modern life depends on technology.",
        "Plants need sunlight to grow. They convert light into energy through photosynthesis. Forests produce oxygen for us.",
        "Music brings people together. Different cultures have unique musical traditions. Rhythm and melody are universal.",
        "The human brain is complex. It contains billions of neurons. Consciousness emerges from neural activity.",
        "Stars are giant balls of gas. They generate light through nuclear fusion. Our sun is an average-sized star.",
        "History teaches us about the past. We learn from previous generations' mistakes. Progress builds on what came before.",
        "Language allows us to communicate. Words carry meaning and emotion. Stories connect us across time and space.",
        "Mathematics describes patterns in nature. Numbers and equations reveal hidden truths. Logic underlies all reasoning.",
        "Art expresses human creativity. Artists see the world differently. Beauty exists in many forms.",
        "Food provides energy for life. Different cuisines reflect local ingredients. Cooking is both art and science.",
        "Exercise keeps the body healthy. Physical activity improves mental well-being. Movement is essential for life.",
        "Books contain worlds of knowledge. Reading expands the mind. Libraries preserve human wisdom.",
        "Water flows from mountains to sea. Rivers carve valleys through rock. The water cycle sustains all life.",
        "Birds migrate across continents. They navigate using Earth's magnetic field. Seasons drive their ancient journeys.",
        "Cities grow and change over time. Architecture reflects cultural values. Urban planning shapes how we live.",
        "Dreams occur during sleep. The brain processes memories at night. Science still explores their meaning.",
        "Gravity keeps planets in orbit. Mass bends the fabric of spacetime. Einstein revolutionized our understanding.",
    ]
    
    # Train with sentence combinations
    summary = trainer.train(corpus, epochs=15)
    
    print("\n" + "="*60)
    print("HIERARCHICAL TRAINING COMPLETE")
    print("="*60)
    
    print(f"\nFinal Statistics:")
    print(f"  Model layers: {summary['final_layers']}")
    print(f"  Token skills: {summary['token_skills']}")
    print(f"  Word skills: {summary['word_skills']}")
    print(f"  Sentence skills: {summary['sentence_skills']}")
    print(f"  Token confluence: {summary['token_confluence']} contexts")
    print(f"  Word confluence: {summary['word_confluence']} contexts")
    print(f"  Sentence confluence: {summary['sentence_confluence']} contexts")
    
    # Fibonacci check
    print(f"\nFibonacci Check:")
    for fib in FIBONACCI:
        if summary['final_layers'] == fib:
            print(f"  ✓ {summary['final_layers']} layers = F({FIBONACCI.index(fib)})")
            break
    else:
        print(f"  ○ {summary['final_layers']} layers not Fibonacci-aligned")
        
    # Show hierarchical structure
    print("\n" + "="*60)
    print("HIERARCHICAL PAC STRUCTURE")
    print("="*60)
    
    print(f"\nLevel 0 (Token): {summary['token_skills']} skills, {summary['token_confluence']} contexts")
    print(f"Level 1 (Word):  {summary['word_skills']} skills, {summary['word_confluence']} contexts")
    print(f"Level 2 (Sentence): {summary['sentence_skills']} skills, {summary['sentence_confluence']} contexts")
    
    # Skill ratio analysis
    total_skills = summary['token_skills'] + summary['word_skills'] + summary['sentence_skills']
    print(f"\nSkill Distribution:")
    print(f"  Token:    {summary['token_skills']:5d} ({100*summary['token_skills']/total_skills:.1f}%)")
    print(f"  Word:     {summary['word_skills']:5d} ({100*summary['word_skills']/total_skills:.1f}%)")
    print(f"  Sentence: {summary['sentence_skills']:5d} ({100*summary['sentence_skills']/total_skills:.1f}%)")
    
    # Check if word skills > token skills (expected for natural language)
    if summary['word_skills'] > summary['token_skills']:
        print(f"\n  ✓ Word skills > Token skills (natural hierarchy)")
    else:
        print(f"\n  ○ Token skills dominate (may need more data)")
    
    # Test generation
    print("\n" + "="*60)
    print("GENERATION TEST")
    print("="*60)
    
    prompts = [
        "The cat",
        "Scientists study",
        "The ocean",
        "Music brings",
        "The human brain"
    ]
    
    for prompt in prompts:
        generated = trainer.generate(prompt, max_tokens=30)
        print(f"\n'{prompt}' → {generated}")
        
    # Save results
    results = {
        'summary': summary,
        'corpus_size': len(corpus),
        'total_sentences': sum(len(t.split('.')) for t in corpus)
    }
    
    output_path = Path(__file__).parent.parent / "results" / "sentence_pac_training.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")
    
    return trainer


if __name__ == "__main__":
    trainer = main()
