"""
Integrated Hierarchical PAC-SEC with Oracle Distillation and Skill Composition

Combines:
1. POC-017: Oracle distillation (real Pythia as loss function)
2. POC-018: Hierarchical PAC-SEC (local/non-local governance)
3. Full skill composition chains (word→phrase→sentence→paragraph)

Key insight: Train on sentence COMBINATIONS, not just word sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import sys
import json

# Add fracton to path
fracton_path = Path(__file__).parent.parent.parent.parent / "fracton"
sys.path.insert(0, str(fracton_path))

try:
    from fracton.physics.constants import PHI, XI, PHI_XI, LAMBDA_STAR
    print(f"✓ Using fracton constants: PHI={PHI:.4f}, XI={XI:.4f}")
except ImportError:
    PHI = (1 + np.sqrt(5)) / 2
    XI = 0.0618
    PHI_XI = 0.1
    LAMBDA_STAR = 0.9816
    print(f"⚠ Using fallback constants")

# Constants
XI_CRITICAL = 1.0571
CRYSTALLIZATION_THRESHOLD = 0.15


@dataclass
class ComplexityLevel:
    """Abstraction level in the hierarchy"""
    level: int
    name: str
    min_layers: int
    max_layers: int
    sec_threshold: float
    pac_radius: int  # -1 = global
    
COMPLEXITY_LEVELS = [
    ComplexityLevel(0, "token", 0, 1, 0.9, 1),
    ComplexityLevel(1, "phrase", 1, 2, 0.7, 3),
    ComplexityLevel(2, "sentence", 2, 4, 0.5, 8),
    ComplexityLevel(3, "paragraph", 4, 6, 0.3, 21),
    ComplexityLevel(4, "document", 6, 12, 0.1, -1),
]


@dataclass
class Skill:
    """A learned transformation between abstraction levels (ByRef link)"""
    source_level: int
    target_level: int
    source_pattern: Tuple
    target_pattern: Optional[Tuple] = None
    strength: float = 0.0
    usage_count: int = 0
    success_count: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count
    
    def apply(self, input_pattern: Tuple) -> Optional[Tuple]:
        """Apply skill if input matches source"""
        # Fuzzy matching - check prefix overlap
        min_len = min(len(input_pattern), len(self.source_pattern))
        if input_pattern[:min_len] == self.source_pattern[:min_len]:
            self.usage_count += 1
            return self.target_pattern
        return None


class SkillGraph:
    """
    Graph of skills connecting abstraction levels.
    
    Enables composition: word→phrase→sentence→paragraph
    """
    
    def __init__(self):
        self.skills: Dict[int, List[Skill]] = defaultdict(list)  # source_level -> skills
        self.skill_chains: List[List[Skill]] = []  # Discovered chains
        
    def add_skill(self, skill: Skill):
        """Add a skill to the graph"""
        self.skills[skill.source_level].append(skill)
        
    def compose_chain(self, start_level: int, end_level: int) -> List[List[Skill]]:
        """Find all skill chains from start_level to end_level"""
        if start_level >= end_level:
            return []
            
        chains = []
        
        def dfs(current_level: int, current_chain: List[Skill]):
            if current_level == end_level:
                chains.append(current_chain.copy())
                return
                
            for skill in self.skills[current_level]:
                if skill.target_level > current_level:
                    current_chain.append(skill)
                    dfs(skill.target_level, current_chain)
                    current_chain.pop()
                    
        dfs(start_level, [])
        return chains
    
    def apply_chain(self, chain: List[Skill], input_pattern: Tuple) -> Optional[Tuple]:
        """Apply a chain of skills to transform input"""
        current = input_pattern
        for skill in chain:
            result = skill.apply(current)
            if result is None:
                return None
            current = result
        return current
    
    def get_stats(self) -> Dict:
        """Get skill graph statistics"""
        total_skills = sum(len(s) for s in self.skills.values())
        levels_with_skills = len(self.skills)
        
        return {
            "total_skills": total_skills,
            "levels_with_skills": levels_with_skills,
            "skills_by_level": {k: len(v) for k, v in self.skills.items()},
            "skill_chains_discovered": len(self.skill_chains)
        }


class SECField:
    """Symbolic Entropy Collapse - Local governance"""
    
    def __init__(self, vocab_size: int, device: str = 'cpu'):
        self.vocab_size = vocab_size
        self.device = device
        self.crystallized_patterns = set()
        self.pattern_counts = defaultdict(int)
        
    def compute_entropy(self, tokens: torch.Tensor) -> float:
        if tokens.numel() == 0:
            return 1.0
        flat = tokens.flatten().tolist()
        counts = defaultdict(int)
        for t in flat:
            counts[t] += 1
        total = len(flat)
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        max_entropy = np.log(len(counts)) if len(counts) > 1 else 1.0
        return min(1.0, entropy / (max_entropy + 1e-10))
    
    def collapse_operator(self, entropy: float, xi: float = XI_CRITICAL) -> float:
        return entropy * np.exp(-xi * entropy)
    
    def process(self, tokens: torch.Tensor) -> Tuple[float, bool]:
        entropy = self.compute_entropy(tokens)
        collapsed = self.collapse_operator(entropy)
        is_crystallized = collapsed < CRYSTALLIZATION_THRESHOLD
        
        if is_crystallized:
            pattern = tuple(tokens.flatten().tolist()[:8])
            self.crystallized_patterns.add(pattern)
            self.pattern_counts[pattern] += 1
            
        return collapsed, is_crystallized


class PACTree:
    """Potential-Actualization Conservation - Non-local governance"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.nodes = {}
        self.parent_links = {}
        self.child_links = defaultdict(list)
        self.conservation_deltas = {}
        
    def add_node(self, node_id: int, value: float, parent_id: Optional[int] = None):
        self.nodes[node_id] = value
        if parent_id is not None:
            self.parent_links[node_id] = parent_id
            self.child_links[parent_id].append(node_id)
            self._propagate_conservation(parent_id)
            
    def _propagate_conservation(self, node_id: int):
        if node_id not in self.nodes:
            return
        children = self.child_links[node_id]
        if not children:
            return
        child_sum = sum(self.nodes.get(c, 0) for c in children)
        self.conservation_deltas[node_id] = self.nodes.get(node_id, 0) - child_sum
        self.nodes[node_id] = child_sum
        if node_id in self.parent_links:
            self._propagate_conservation(self.parent_links[node_id])
            
    def compute_entanglement(self, node_a: int, node_b: int) -> float:
        """Nodes sharing ancestors are entangled"""
        ancestors_a = set()
        current = node_a
        while current in self.parent_links:
            ancestors_a.add(current)
            current = self.parent_links[current]
        ancestors_a.add(current)
        
        ancestors_b = set()
        current = node_b
        while current in self.parent_links:
            ancestors_b.add(current)
            current = self.parent_links[current]
        ancestors_b.add(current)
        
        common = ancestors_a & ancestors_b
        if not common:
            return 0.0
            
        return len(common) / (len(ancestors_a) + len(ancestors_b) - len(common))


class LazyTransformerLayer(nn.Module):
    """Transformer layer that materializes on demand"""
    
    def __init__(self, dim: int, n_heads: int = 4, device: str = 'cpu'):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.device = device
        self.materialized = False
        self.usage_count = 0
        self.total_complexity = 0.0
        self._attn = None
        self._ff = None
        self._norm1 = None
        self._norm2 = None
        
    def materialize(self):
        if not self.materialized:
            self._norm1 = nn.LayerNorm(self.dim).to(self.device)
            self._attn = nn.MultiheadAttention(self.dim, self.n_heads, batch_first=True).to(self.device)
            self._norm2 = nn.LayerNorm(self.dim).to(self.device)
            self._ff = nn.Sequential(
                nn.Linear(self.dim, self.dim * 4),
                nn.GELU(),
                nn.Linear(self.dim * 4, self.dim)
            ).to(self.device)
            self.materialized = True
            
    def forward(self, x: torch.Tensor, complexity: float = 1.0) -> torch.Tensor:
        if complexity < 0.2 and self.usage_count > 0:
            return x
        self.materialize()
        self.usage_count += 1
        self.total_complexity += complexity
        
        h = self._norm1(x)
        attn_out, _ = self._attn(h, h, h)
        x = x + attn_out
        h = self._norm2(x)
        ff_out = self._ff(h)
        return x + ff_out
    
    @property
    def avg_complexity(self) -> float:
        return self.total_complexity / max(self.usage_count, 1)


class IntegratedPACTransformer(nn.Module):
    """
    Integrated hierarchical transformer with:
    - SEC for local crystallization
    - PAC for global conservation
    - Lazy layer materialization
    - Skill composition chains
    - Oracle distillation support
    """
    
    def __init__(self,
                 vocab_size: int = 50304,
                 dim: int = 256,
                 n_heads: int = 8,
                 max_layers: int = 12,
                 device: str = 'cpu'):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_heads = n_heads
        self.max_layers = max_layers
        self.device = device
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, dim).to(device)
        self.pos_embedding = nn.Embedding(2048, dim).to(device)
        
        # Lazy layers
        self.layers = nn.ModuleList([
            LazyTransformerLayer(dim, n_heads, device) 
            for _ in range(max_layers)
        ])
        
        # Output
        self.output_norm = nn.LayerNorm(dim).to(device)
        self.output_proj = nn.Linear(dim, vocab_size).to(device)
        
        # SEC-PAC dynamics
        self.sec_field = SECField(vocab_size, device)
        self.pac_tree = PACTree(device)
        
        # Skill graph for composition
        self.skill_graph = SkillGraph()
        
        # Stats
        self.layer_usage = defaultdict(int)
        self.level_usage = defaultdict(int)
        
    def assess_complexity(self, x: torch.Tensor, tokens: torch.Tensor) -> Tuple[float, int, int]:
        """Assess complexity to determine layers needed"""
        seq_len = tokens.shape[1] if tokens.dim() > 1 else len(tokens)
        len_complexity = min(1.0, seq_len / 30.0)
        
        sec_entropy, is_crystallized = self.sec_field.process(tokens)
        sec_complexity = 0.1 if is_crystallized else sec_entropy
        
        complexity = 0.7 * len_complexity + 0.3 * sec_complexity
        
        if complexity < 0.1:
            level_idx = 0
        elif complexity < 0.3:
            level_idx = 1
        elif complexity < 0.5:
            level_idx = 2
        elif complexity < 0.7:
            level_idx = 3
        else:
            level_idx = 4
                
        required_layers = COMPLEXITY_LEVELS[level_idx].max_layers
        return complexity, required_layers, level_idx
    
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, T = input_ids.shape
        
        tok_emb = self.embedding(input_ids)
        pos_emb = self.pos_embedding(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        
        complexity, required_layers, level_idx = self.assess_complexity(x, input_ids)
        self.level_usage[level_idx] += 1
        
        layers_used = 0
        for i in range(min(required_layers, self.max_layers)):
            x = self.layers[i](x, complexity)
            self.layer_usage[i] += 1
            layers_used += 1
            
        # Update PAC tree
        for b in range(B):
            for t in range(T):
                node_id = hash((b, t, input_ids[b, t].item()))
                parent_id = hash((b, t-1, input_ids[b, t-1].item())) if t > 0 else None
                value = x[b, t, 0].item()
                self.pac_tree.add_node(node_id, value, parent_id)
        
        x = self.output_norm(x)
        logits = self.output_proj(x)
        
        info = {
            "complexity": complexity,
            "layers_used": layers_used,
            "level": COMPLEXITY_LEVELS[level_idx].name,
            "sec_crystallized": len(self.sec_field.crystallized_patterns),
            "pac_nodes": len(self.pac_tree.nodes)
        }
        
        return logits, info
    
    def learn_skill(self, source_level: int, target_level: int,
                    source_tokens: torch.Tensor, target_tokens: torch.Tensor):
        """Learn a skill connecting two abstraction levels"""
        src_pattern = tuple(source_tokens.flatten().tolist()[:8])
        tgt_pattern = tuple(target_tokens.flatten().tolist()[:8])
        
        # Ensure we're learning skills that PROGRESS through levels
        # source_level < target_level for composition chains
        skill = Skill(
            source_level=source_level,
            target_level=target_level,
            source_pattern=src_pattern,
            target_pattern=tgt_pattern,
            strength=1.0
        )
        self.skill_graph.add_skill(skill)
        return skill
    
    def learn_hierarchical_skill(self, tokens: torch.Tensor, text: str):
        """Learn skills at all levels from a single text"""
        # Determine the level of this text
        seq_len = tokens.shape[1] if tokens.dim() > 1 else len(tokens)
        if seq_len <= 3:
            current_level = 0  # token
        elif seq_len <= 10:
            current_level = 1  # phrase
        elif seq_len <= 25:
            current_level = 2  # sentence
        elif seq_len <= 60:
            current_level = 3  # paragraph
        else:
            current_level = 4  # document
        
        # Learn skill from previous level to current
        if current_level > 0:
            # Get a subpattern at lower level
            sub_len = [3, 8, 20, 50][min(current_level-1, 3)]
            sub_tokens = tokens[0, :min(sub_len, tokens.shape[1])]
            
            skill = Skill(
                source_level=current_level - 1,
                target_level=current_level,
                source_pattern=tuple(sub_tokens.tolist()[:8]),
                target_pattern=tuple(tokens[0].tolist()[:8]),
                strength=1.0
            )
            self.skill_graph.add_skill(skill)
            return skill
        return None
    
    def compose_skills(self, start_level: int = 0, end_level: int = 3) -> List[List[Skill]]:
        """Find and store skill composition chains"""
        chains = self.skill_graph.compose_chain(start_level, end_level)
        self.skill_graph.skill_chains.extend(chains)
        return chains
    
    def get_stats(self) -> Dict:
        materialized = sum(1 for l in self.layers if l.materialized)
        return {
            "materialized_layers": materialized,
            "max_layers": self.max_layers,
            "layer_usage": dict(self.layer_usage),
            "level_usage": {COMPLEXITY_LEVELS[k].name: v for k, v in self.level_usage.items()},
            "sec_crystallized": len(self.sec_field.crystallized_patterns),
            "pac_nodes": len(self.pac_tree.nodes),
            "skill_stats": self.skill_graph.get_stats()
        }


class IntegratedTrainer:
    """
    Trainer that combines:
    1. Oracle distillation (POC-017)
    2. Hierarchical training (POC-018)
    3. Skill composition learning
    """
    
    def __init__(self, model: IntegratedPACTransformer, oracle=None, tokenizer=None):
        self.model = model
        self.oracle = oracle
        self.tokenizer = tokenizer
        self.device = model.device
        
        # Initialize embeddings from oracle if available
        if oracle is not None and hasattr(oracle, 'gpt_neox'):
            print("  Initializing embeddings from oracle...")
            oracle_emb = oracle.gpt_neox.embed_in.weight.detach()
            student_dim = model.dim
            with torch.no_grad():
                if oracle_emb.shape[1] >= student_dim:
                    model.embedding.weight.copy_(oracle_emb[:, :student_dim].to(self.device))
                else:
                    model.embedding.weight[:, :oracle_emb.shape[1]].copy_(oracle_emb.to(self.device))
            print(f"    Initialized {student_dim} dims from oracle")
    
    def generate_training_data(self, n_samples: int = 100) -> Dict[int, List[str]]:
        """Generate diverse training data at each level using oracle"""
        data = {0: [], 1: [], 2: [], 3: [], 4: []}
        
        if self.oracle is None or self.tokenizer is None:
            # Fallback data - more variety
            data[0] = ["cat", "dog", "sun", "moon", "tree", "bird", "fish", "rain",
                       "love", "hope", "fear", "joy", "peace", "war", "life", "death"]
            data[1] = ["the cat", "big dog", "warm sun", "full moon", "tall tree",
                       "bright stars", "deep ocean", "cold wind", "green grass", "blue sky"]
            data[2] = ["The cat sat on the mat.", "Dogs run fast in parks.", 
                       "Birds fly high in the sky.", "Fish swim in deep water.",
                       "The sun rises in the east.", "Stars twinkle at night.",
                       "Rain falls from clouds.", "Wind blows through trees."]
            data[3] = ["The cat sat on the mat. It was warm and comfortable. The sun streamed through the window.",
                       "Scientists study nature every day. They make discoveries. Knowledge grows over time.",
                       "The city was busy. Cars filled the streets. People rushed to work."]
            data[4] = ["Introduction. This is a story about discovery. It begins with curiosity. Scientists explore. They ask questions. The answers lead to more questions. Knowledge grows. Understanding deepens. The end.",
                       "Chapter one. The world is full of wonder. Nature has many secrets. We study them carefully. Each discovery teaches us something new. Learning never stops."]
            return data
        
        # Generate from oracle with more variety
        prompts = {
            0: ["The", "A", "It", "He", "She", "They", "We", "One", "This", "That",
                "What", "How", "Why", "Where", "When", "I", "You", "My", "Your", "Our"],
            1: ["The quick", "A large", "It was", "He walked", "She said", 
                "They found", "We saw", "One day", "This is", "That was",
                "Looking at", "Running to", "After the", "Before we", "During the"],
            2: ["The weather today is", "Scientists discovered that", "In the morning we",
                "Technology has changed", "The study shows", "Research indicates",
                "According to experts", "In recent years", "The evidence suggests",
                "Many people believe", "Studies have shown", "Data reveals that"],
            3: ["Once upon a time there was", "The research shows that many people",
                "According to experts in the field", "In the beginning of the story",
                "Scientists have long wondered", "The study examines how people",
                "Throughout history humans have", "In recent developments researchers"],
            4: ["This document describes the following important concepts and ideas that",
                "The following analysis examines multiple factors including the key",
                "In conclusion we have examined several important aspects of this topic",
                "This report provides a comprehensive overview of the current state of"]
        }
        
        print("    Generating samples from oracle...")
        for level, level_prompts in prompts.items():
            target_len = [3, 10, 25, 60, 120][level]
            
            for i, prompt in enumerate(level_prompts[:n_samples // 4]):
                try:
                    tokens = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                    
                    with torch.no_grad():
                        for _ in range(target_len):
                            if tokens.shape[1] >= target_len:
                                break
                            logits = self.oracle(tokens).logits
                            probs = F.softmax(logits[0, -1] / 0.7, dim=-1)
                            next_token = torch.multinomial(probs, 1)
                            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
                    
                    text = self.tokenizer.decode(tokens[0])
                    data[level].append(text)
                except Exception as e:
                    continue
            
            print(f"      Level {level}: {len(data[level])} samples")
                
        return data
    
    def train_with_oracle(self, 
                          texts: List[str], 
                          level: int,
                          n_epochs: int = 3,
                          lr: float = 1e-3) -> Dict:
        """Train using oracle as loss function (distillation)"""
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        total_loss = 0.0
        n_batches = 0
        skills_learned = 0
        
        for epoch in range(n_epochs):
            for text in texts:
                if self.tokenizer:
                    tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
                else:
                    tokens = torch.tensor(
                        [ord(c) % self.model.vocab_size for c in text],
                        device=self.device
                    ).unsqueeze(0)
                
                # Learn hierarchical skill on first epoch
                if epoch == 0:
                    skill = self.model.learn_hierarchical_skill(tokens, text)
                    if skill:
                        skills_learned += 1
                
                # Get oracle output
                if self.oracle is not None:
                    with torch.no_grad():
                        oracle_logits = self.oracle(tokens).logits
                else:
                    oracle_logits = None
                    
                # Forward
                student_logits, info = self.model(tokens)
                
                # Combined loss: KL + CLM
                if oracle_logits is not None:
                    # KL divergence
                    kl_loss = F.kl_div(
                        F.log_softmax(student_logits / 2.0, dim=-1),
                        F.softmax(oracle_logits / 2.0, dim=-1),
                        reduction='batchmean'
                    ) * 4.0
                    
                    # CLM loss
                    if tokens.shape[1] > 1:
                        shift_logits = student_logits[:, :-1, :]
                        shift_labels = tokens[:, 1:]
                        clm_loss = F.cross_entropy(
                            shift_logits.reshape(-1, self.model.vocab_size),
                            shift_labels.reshape(-1)
                        )
                    else:
                        clm_loss = 0.0
                    
                    loss = 0.5 * kl_loss + 0.5 * clm_loss
                else:
                    # CLM only
                    if tokens.shape[1] > 1:
                        shift_logits = student_logits[:, :-1, :]
                        shift_labels = tokens[:, 1:]
                        loss = F.cross_entropy(
                            shift_logits.reshape(-1, self.model.vocab_size),
                            shift_labels.reshape(-1)
                        )
                    else:
                        continue
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
                
        return {
            "level": level,
            "avg_loss": total_loss / max(n_batches, 1),
            "n_texts": len(texts),
            "skills_learned": skills_learned
        }
    
    def train_sentence_compositions(self, 
                                     sentence_pairs: List[Tuple[str, str]],
                                     n_epochs: int = 5) -> Dict:
        """Train on sentence combinations and learn skills"""
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4)
        
        total_loss = 0.0
        skills_learned = 0
        
        for epoch in range(n_epochs):
            for sent1, sent2 in sentence_pairs:
                combined = sent1 + " " + sent2
                
                if self.tokenizer:
                    tokens = self.tokenizer.encode(combined, return_tensors='pt').to(self.device)
                    sent1_tokens = self.tokenizer.encode(sent1, return_tensors='pt').to(self.device)
                    sent2_tokens = self.tokenizer.encode(sent2, return_tensors='pt').to(self.device)
                else:
                    tokens = torch.tensor(
                        [ord(c) % self.model.vocab_size for c in combined],
                        device=self.device
                    ).unsqueeze(0)
                    sent1_tokens = torch.tensor(
                        [ord(c) % self.model.vocab_size for c in sent1],
                        device=self.device
                    ).unsqueeze(0)
                    sent2_tokens = torch.tensor(
                        [ord(c) % self.model.vocab_size for c in sent2],
                        device=self.device
                    ).unsqueeze(0)
                
                # Get oracle output
                if self.oracle is not None:
                    with torch.no_grad():
                        oracle_logits = self.oracle(tokens).logits
                else:
                    oracle_logits = None
                
                # Forward
                logits, info = self.model(tokens)
                
                # Loss on second sentence given first
                split_point = sent1_tokens.shape[1]
                
                if tokens.shape[1] > split_point + 1:
                    pred_logits = logits[:, split_point:-1, :]
                    target_tokens = tokens[:, split_point+1:]
                    
                    loss = F.cross_entropy(
                        pred_logits.reshape(-1, self.model.vocab_size),
                        target_tokens.reshape(-1)
                    )
                    
                    # Add KL if oracle available
                    if oracle_logits is not None:
                        oracle_pred = oracle_logits[:, split_point:-1, :]
                        kl_loss = F.kl_div(
                            F.log_softmax(pred_logits / 2.0, dim=-1),
                            F.softmax(oracle_pred / 2.0, dim=-1),
                            reduction='batchmean'
                        ) * 4.0
                        loss = 0.5 * loss + 0.5 * kl_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                # Learn skills on first epoch
                if epoch == 0:
                    # Sentence → Sentence skill
                    self.model.learn_skill(2, 2, sent1_tokens[0], sent2_tokens[0])
                    skills_learned += 1
                    
                    # Also learn phrase-level skills from sentence parts
                    if sent1_tokens.shape[1] > 3:
                        phrase1 = sent1_tokens[0, :3]
                        phrase2 = sent1_tokens[0, 3:6] if sent1_tokens.shape[1] > 6 else sent1_tokens[0, 3:]
                        self.model.learn_skill(1, 1, phrase1, phrase2)
                        skills_learned += 1
                    
        return {
            "avg_loss": total_loss / max(len(sentence_pairs) * n_epochs, 1),
            "skills_learned": skills_learned
        }
    
    def train_skill_chains(self) -> Dict:
        """Build and validate skill composition chains"""
        
        # First ensure we have skills at each level transition
        skill_stats = self.model.skill_graph.get_stats()
        print(f"    Skills by level: {skill_stats['skills_by_level']}")
        
        # Find chains from token to paragraph level
        chains_0_to_3 = self.model.compose_skills(0, 3)
        chains_1_to_3 = self.model.compose_skills(1, 3)
        chains_0_to_2 = self.model.compose_skills(0, 2)
        chains_1_to_2 = self.model.compose_skills(1, 2)
        chains_2_to_3 = self.model.compose_skills(2, 3)
        chains_0_to_1 = self.model.compose_skills(0, 1)
        
        all_chains = chains_0_to_3 + chains_1_to_3 + chains_0_to_2 + chains_1_to_2 + chains_2_to_3 + chains_0_to_1
        
        print(f"    Discovered {len(all_chains)} skill chains:")
        print(f"      Token→Paragraph (0→3): {len(chains_0_to_3)}")
        print(f"      Phrase→Paragraph (1→3): {len(chains_1_to_3)}")
        print(f"      Sentence→Paragraph (2→3): {len(chains_2_to_3)}")
        print(f"      Token→Sentence (0→2): {len(chains_0_to_2)}")
        print(f"      Phrase→Sentence (1→2): {len(chains_1_to_2)}")
        print(f"      Token→Phrase (0→1): {len(chains_0_to_1)}")
        
        return {
            "total_chains": len(all_chains),
            "chains_0_to_3": len(chains_0_to_3),
            "chains_1_to_3": len(chains_1_to_3),
            "chains_2_to_3": len(chains_2_to_3),
            "chains_0_to_2": len(chains_0_to_2),
            "chains_1_to_2": len(chains_1_to_2),
            "chains_0_to_1": len(chains_0_to_1)
        }
    
    def full_hierarchical_training(self, n_samples_per_level: int = 20) -> Dict:
        """Complete hierarchical training pipeline"""
        
        results = {
            "level_results": [],
            "composition_result": None,
            "chain_result": None,
            "total_skills": 0
        }
        
        # Generate training data
        print("\n  Generating training data from oracle...")
        training_data = self.generate_training_data(n_samples_per_level * 5)
        
        # Train each level
        for level in range(5):
            level_name = COMPLEXITY_LEVELS[level].name
            texts = training_data[level][:n_samples_per_level]
            
            if not texts:
                continue
                
            print(f"\n  Training level {level} ({level_name})...")
            result = self.train_with_oracle(texts, level, n_epochs=5)
            results["level_results"].append(result)
            results["total_skills"] += result.get("skills_learned", 0)
            print(f"    Loss: {result['avg_loss']:.4f}, Texts: {result['n_texts']}, Skills: {result.get('skills_learned', 0)}")
        
        # Train sentence compositions
        print("\n  Training sentence compositions...")
        
        # Generate sentence pairs from paragraph-level data
        sentence_pairs = []
        for para in training_data[3]:
            sentences = para.replace('!', '.').replace('?', '.').split('.')
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            for i in range(len(sentences) - 1):
                sentence_pairs.append((sentences[i], sentences[i+1]))
        
        # Add some manual pairs if needed
        manual_pairs = [
            ("The sun was shining.", "It was a beautiful day."),
            ("She opened the door.", "The room was dark inside."),
            ("The cat meowed loudly.", "It wanted to be fed."),
            ("Rain fell on the window.", "The streets became wet."),
            ("He finished his work.", "Now he could relax."),
        ]
        sentence_pairs.extend(manual_pairs)
        
        if sentence_pairs:
            comp_result = self.train_sentence_compositions(sentence_pairs, n_epochs=5)
            results["composition_result"] = comp_result
            print(f"    Loss: {comp_result['avg_loss']:.4f}, Skills: {comp_result['skills_learned']}")
        
        # Build skill chains
        print("\n  Building skill chains...")
        chain_result = self.train_skill_chains()
        results["chain_result"] = chain_result
        
        return results


def demo_integrated():
    """Full demonstration of integrated PAC-SEC with oracle distillation and skill chains"""
    
    print("="*70)
    print("INTEGRATED HIERARCHICAL PAC-SEC TRAINING")
    print("="*70)
    print("\n  1. Oracle distillation (POC-017)")
    print("  2. Hierarchical PAC-SEC (POC-018)")
    print("  3. Skill composition chains (word→phrase→sentence→paragraph)")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📍 Device: {device}")
    
    # Load oracle
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("\nLoading oracle (Pythia-70M)...")
        oracle = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/pythia-70m",
            torch_dtype=torch.float32
        ).to(device)
        oracle.eval()
        for p in oracle.parameters():
            p.requires_grad = False
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        vocab_size = oracle.config.vocab_size
        print(f"  Oracle loaded: {sum(p.numel() for p in oracle.parameters()):,} params")
    except Exception as e:
        print(f"  ⚠️ Oracle not available: {e}")
        oracle = None
        tokenizer = None
        vocab_size = 256
    
    # Create model
    print("\n" + "="*60)
    print("CREATING INTEGRATED PAC TRANSFORMER")
    print("="*60)
    
    model = IntegratedPACTransformer(
        vocab_size=vocab_size,
        dim=256,
        n_heads=8,
        max_layers=12,
        device=device
    )
    
    initial_params = sum(p.numel() for p in model.parameters())
    print(f"  Initial: {initial_params:,} params, {model.max_layers} max layers")
    
    trainer = IntegratedTrainer(model, oracle, tokenizer)
    
    # Full hierarchical training
    print("\n" + "="*60)
    print("FULL HIERARCHICAL TRAINING")
    print("="*60)
    
    results = trainer.full_hierarchical_training(n_samples_per_level=15)
    
    # Show stats
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    stats = model.get_stats()
    
    print(f"\n📊 MODEL STATS:")
    print(f"  Materialized layers: {stats['materialized_layers']}/{stats['max_layers']}")
    print(f"  SEC crystallized: {stats['sec_crystallized']}")
    print(f"  PAC nodes: {stats['pac_nodes']}")
    
    print(f"\n🔗 SKILL STATS:")
    skill_stats = stats['skill_stats']
    print(f"  Total skills: {skill_stats['total_skills']}")
    print(f"  Skill chains: {skill_stats['skill_chains_discovered']}")
    print(f"  Skills by level: {skill_stats['skills_by_level']}")
    
    print(f"\n📈 LAYER USAGE:")
    for layer_idx, count in sorted(stats['layer_usage'].items()):
        bar = "█" * min(count // 2, 50)
        print(f"  Layer {layer_idx:2d}: {count:4d} {bar}")
        
    print(f"\n📊 LEVEL USAGE:")
    for level_name, count in stats['level_usage'].items():
        print(f"  {level_name}: {count}")
    
    # Test efficiency
    print("\n" + "="*60)
    print("EFFICIENCY TEST")
    print("="*60)
    
    test_inputs = [
        ("cat", "Token"),
        ("the big cat", "Phrase"),
        ("The cat sat on the mat.", "Sentence"),
        ("The cat sat on the mat. It purred softly. The sun was warm.", "Paragraph"),
    ]
    
    for text, desc in test_inputs:
        if tokenizer:
            tokens = tokenizer.encode(text, return_tensors='pt').to(device)
        else:
            tokens = torch.tensor([ord(c) % vocab_size for c in text], device=device).unsqueeze(0)
            
        with torch.no_grad():
            _, info = model(tokens)
            
        print(f"\n  {desc}: '{text[:40]}{'...' if len(text) > 40 else ''}'")
        print(f"    Complexity: {info['complexity']:.3f}")
        print(f"    Layers: {info['layers_used']}")
        print(f"    Level: {info['level']}")
    
    # Test generation
    if tokenizer and oracle:
        print("\n" + "="*60)
        print("GENERATION TEST")
        print("="*60)
        
        prompts = [
            "The meaning of life is",
            "Once upon a time",
            "Scientists have discovered that",
        ]
        
        for prompt in prompts:
            tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            with torch.no_grad():
                for _ in range(25):
                    logits, _ = model(tokens)
                    probs = F.softmax(logits[0, -1] / 0.8, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
                    
            output = tokenizer.decode(tokens[0])
            
            # Also get oracle output for comparison
            oracle_tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                for _ in range(25):
                    logits = oracle(oracle_tokens).logits
                    probs = F.softmax(logits[0, -1] / 0.8, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    oracle_tokens = torch.cat([oracle_tokens, next_token.unsqueeze(0)], dim=1)
            oracle_output = tokenizer.decode(oracle_tokens[0])
            
            print(f"\n  Prompt: '{prompt}'")
            print(f"  Student: '{output}'")
            print(f"  Oracle:  '{oracle_output}'")
    
    # Test skill chains
    print("\n" + "="*60)
    print("SKILL CHAIN TEST")
    print("="*60)
    
    chains = model.skill_graph.skill_chains
    print(f"  Total chains discovered: {len(chains)}")
    
    if chains:
        print("\n  Sample chains:")
        for i, chain in enumerate(chains[:5]):
            chain_str = " → ".join([f"L{s.source_level}→L{s.target_level}" for s in chain])
            print(f"    Chain {i+1}: {chain_str}")
    
    print("\n" + "="*70)
    print("✅ INTEGRATED TRAINING COMPLETE")
    print("="*70)
    
    print("""
💡 KEY INSIGHTS:
  1. ORACLE DISTILLATION: Real Pythia as loss function
  2. HIERARCHICAL PAC-SEC: Local crystallization + global conservation
  3. LAZY LAYERS: Only materialize when complexity demands
  4. SKILL CHAINS: Compose word→phrase→sentence→paragraph
  
📐 ARCHITECTURE:
  SEC (Local)                    PAC (Non-local)
  ───────────                    ─────────────────
  Token crystallization          Tree conservation
  Phrase patterns                Entanglement effects
  Immediate neighborhood         Document coherence
  
🔗 SKILL COMPOSITION:
  Level 0 (token) → Level 1 (phrase) → Level 2 (sentence) → Level 3 (paragraph)
  Each ByRef link is a learned transformation between levels
""")
    
    # Save results
    results_path = Path(__file__).parent / "results"
    results_path.mkdir(exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_file = results_path / f"integrated_training_{timestamp}.json"
    
    save_data = {
        "timestamp": timestamp,
        "model_stats": {
            "materialized_layers": stats['materialized_layers'],
            "max_layers": stats['max_layers'],
            "sec_crystallized": stats['sec_crystallized'],
            "pac_nodes": stats['pac_nodes'],
            "skill_stats": stats['skill_stats'],
            "layer_usage": {str(k): v for k, v in stats['layer_usage'].items()},
            "level_usage": stats['level_usage']
        },
        "training_results": {
            "n_levels_trained": len(results['level_results']),
            "skill_chains_discovered": results['chain_result']['total_chains'] if results['chain_result'] else 0
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n📁 Results saved to: {results_file}")


if __name__ == "__main__":
    demo_integrated()
