"""
PAC Tree with ByRef Skills
===========================

The tree IS the computation:
- Nodes = Knowledge (extracted patterns)
- ByRef links = Skills (how to use knowledge)
- Growth = Learning new knowledge
- New ByRef = Learning new skills

Computation flow:
1. Input → find resonant node
2. Follow byref links (skill application)
3. Traverse tree structure
4. Output = terminal node reached

This is fundamentally different from transformers:
- Transformer: knowledge + skill bundled in weights
- PAC+ByRef: knowledge (nodes) separate from skill (links)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add fracton
fracton_path = Path(__file__).parent.parent.parent.parent.parent / "fracton"
sys.path.insert(0, str(fracton_path))

from fracton.core import PACSystem, PACNode
from fracton.physics.constants import PHI, XI, PHI_XI, LAMBDA_STAR

print(f"✓ Using fracton from {fracton_path}")


@dataclass
class ByRefLink:
    """
    A skill connection between tree branches.
    
    Unlike parent-child (structure), byref is (function):
    - source_id: Where the skill starts
    - target_id: Where the skill leads
    - skill_type: What kind of operation (predict, transform, associate)
    - strength: How reliable/practiced this skill is
    - context: When to use this skill (activation pattern)
    """
    source_id: int
    target_id: int
    skill_type: str  # "predict", "transform", "associate", "compose"
    strength: float = 1.0
    usage_count: int = 0
    context: Optional[torch.Tensor] = None  # Activation pattern that triggers this skill
    
    def use(self):
        """Use this skill (strengthens it)."""
        self.usage_count += 1
        # Hebbian-style strengthening
        self.strength = min(10.0, self.strength + 0.1 * LAMBDA_STAR)
    
    def decay(self):
        """Decay unused skill."""
        self.strength *= LAMBDA_STAR


class SkillGraph:
    """
    Graph of byref skill connections overlaid on PAC tree.
    
    This is the "how to use knowledge" layer.
    The tree structure is knowledge, the skill graph is competence.
    """
    
    def __init__(self):
        # Links indexed by source
        self.outgoing: Dict[int, List[ByRefLink]] = defaultdict(list)
        # Links indexed by target
        self.incoming: Dict[int, List[ByRefLink]] = defaultdict(list)
        # Links indexed by skill type
        self.by_type: Dict[str, List[ByRefLink]] = defaultdict(list)
        
        # Statistics
        self.total_links = 0
        self.total_usages = 0
    
    def add_skill(self, 
                  source_id: int, 
                  target_id: int, 
                  skill_type: str = "predict",
                  context: Optional[torch.Tensor] = None) -> ByRefLink:
        """Add a new skill connection."""
        link = ByRefLink(
            source_id=source_id,
            target_id=target_id,
            skill_type=skill_type,
            context=context
        )
        
        self.outgoing[source_id].append(link)
        self.incoming[target_id].append(link)
        self.by_type[skill_type].append(link)
        self.total_links += 1
        
        return link
    
    def get_skills_from(self, source_id: int, skill_type: Optional[str] = None) -> List[ByRefLink]:
        """Get all skills originating from a node."""
        links = self.outgoing.get(source_id, [])
        if skill_type:
            links = [l for l in links if l.skill_type == skill_type]
        return links
    
    def get_best_skill(self, 
                       source_id: int, 
                       skill_type: str = "predict",
                       context: Optional[torch.Tensor] = None) -> Optional[ByRefLink]:
        """Get the strongest matching skill from a node."""
        links = self.get_skills_from(source_id, skill_type)
        
        if not links:
            return None
        
        # If context provided, score by context similarity
        if context is not None:
            best_link = None
            best_score = -1
            for link in links:
                if link.context is not None:
                    sim = F.cosine_similarity(
                        context.flatten().unsqueeze(0),
                        link.context.flatten().unsqueeze(0)
                    ).item()
                    score = sim * link.strength
                else:
                    score = link.strength
                
                if score > best_score:
                    best_score = score
                    best_link = link
            return best_link
        
        # Otherwise return strongest
        return max(links, key=lambda l: l.strength)
    
    def follow_skill(self, link: ByRefLink) -> int:
        """Follow a skill link, strengthening it."""
        link.use()
        self.total_usages += 1
        return link.target_id
    
    def decay_all(self, threshold: float = 0.1):
        """Decay all skills and prune weak ones."""
        to_remove = []
        for source_id, links in self.outgoing.items():
            for link in links:
                link.decay()
                if link.strength < threshold:
                    to_remove.append((source_id, link))
        
        for source_id, link in to_remove:
            self.outgoing[source_id].remove(link)
            self.incoming[link.target_id].remove(link)
            self.by_type[link.skill_type].remove(link)
            self.total_links -= 1


class PACTreeComputer:
    """
    Computation engine using PAC tree + skill graph.
    
    This is the core insight: computation IS tree traversal + skill following.
    
    Forward pass:
    1. Encode input → find resonant node in tree
    2. Follow skill links based on task
    3. Accumulate deltas along path
    4. Output = final node's value
    
    Learning:
    1. Tree growth = new knowledge nodes
    2. Skill creation = new byref links
    3. Skill strengthening = successful traversals
    """
    
    def __init__(self, 
                 device: str = 'cpu',
                 embed_dim: int = 512):
        self.device = device
        self.embed_dim = embed_dim
        
        # PAC tree (knowledge storage)
        self.tree = PACSystem(device=device)
        
        # Skill graph (byref connections)
        self.skills = SkillGraph()
        
        # Node embeddings cache (for fast lookup)
        self.node_embeddings: Dict[int, torch.Tensor] = {}
        
        # Token to node mapping
        self.token_to_node: Dict[int, int] = {}
        
        # Cluster structure
        self.cluster_centers: Dict[int, torch.Tensor] = {}
        self.cluster_nodes: Dict[int, int] = {}
        
        # Transition skills (token A → token B prediction)
        self.transition_skills: Dict[Tuple[int, int], ByRefLink] = {}
        
        # Statistics
        self.stats = {
            'tree_nodes': 0,
            'skill_links': 0,
            'traversals': 0,
            'skill_creations': 0
        }
    
    def inject_knowledge(self, 
                        embedding: torch.Tensor,
                        parent_id: Optional[int] = None,
                        label: str = "") -> int:
        """Add knowledge to the tree."""
        embedding = embedding.to(self.device)
        
        node_id = self.tree.inject(
            embedding,
            parent_id=parent_id,
            label=label
        )
        
        self.node_embeddings[node_id] = embedding
        self.stats['tree_nodes'] += 1
        
        return node_id
    
    def create_skill(self,
                    source_id: int,
                    target_id: int,
                    skill_type: str = "predict",
                    context: Optional[torch.Tensor] = None) -> ByRefLink:
        """Create a skill (byref link) between nodes."""
        link = self.skills.add_skill(source_id, target_id, skill_type, context)
        self.stats['skill_links'] += 1
        self.stats['skill_creations'] += 1
        return link
    
    def find_resonant_node(self, query: torch.Tensor, top_k: int = 1) -> List[Tuple[int, float]]:
        """Find nodes most resonant with query."""
        query = query.to(self.device)
        
        # Use PACSystem's find_resonant
        results = self.tree.find_resonant(query, top_k=top_k, threshold=0.0)
        
        return results
    
    def traverse(self,
                start_node: int,
                skill_type: str = "predict",
                context: Optional[torch.Tensor] = None,
                max_steps: int = 5) -> Tuple[int, List[int], torch.Tensor]:
        """
        Traverse tree following skills.
        
        Returns:
            - Final node ID
            - Path taken
            - Accumulated delta
        """
        current = start_node
        path = [current]
        accumulated = torch.zeros(self.embed_dim, device=self.device)
        
        for step in range(max_steps):
            # Get best skill from current node
            skill = self.skills.get_best_skill(current, skill_type, context)
            
            if skill is None:
                break  # No skill available, stop
            
            # Follow the skill
            next_node = self.skills.follow_skill(skill)
            
            # Accumulate delta
            if next_node in self.node_embeddings:
                accumulated = accumulated + self.node_embeddings[next_node]
            else:
                try:
                    delta = self.tree.reconstruct(next_node)
                    accumulated = accumulated + delta
                except:
                    pass
            
            current = next_node
            path.append(current)
            self.stats['traversals'] += 1
            
            # Context shifts after each step
            if context is not None:
                context = context + accumulated * 0.1
        
        return current, path, accumulated
    
    def compute(self, 
               input_embedding: torch.Tensor,
               skill_type: str = "predict") -> Tuple[torch.Tensor, Dict]:
        """
        Main computation: tree traversal + skill following.
        
        This IS the forward pass - no separate neural network needed.
        """
        input_embedding = input_embedding.to(self.device)
        
        # Step 1: Find resonant starting node
        resonant = self.find_resonant_node(input_embedding, top_k=1)
        
        if not resonant:
            # No resonant node, return input unchanged
            return input_embedding, {'path': [], 'steps': 0}
        
        start_node, resonance = resonant[0]
        
        # Step 2: Traverse following skills
        end_node, path, accumulated = self.traverse(
            start_node,
            skill_type=skill_type,
            context=input_embedding,
            max_steps=5
        )
        
        # Step 3: Output is input + accumulated delta
        output = input_embedding + accumulated
        
        return output, {
            'start_node': start_node,
            'end_node': end_node,
            'path': path,
            'steps': len(path) - 1,
            'resonance': resonance
        }
    
    def learn_transition(self,
                        from_token: int,
                        to_token: int,
                        context: Optional[torch.Tensor] = None,
                        context_tokens: Optional[List[int]] = None):
        """Learn a transition skill (prediction) with optional context."""
        # Get or create nodes for tokens
        if from_token not in self.token_to_node:
            return  # Can't learn without node
        if to_token not in self.token_to_node:
            return
        
        from_node = self.token_to_node[from_token]
        to_node = self.token_to_node[to_token]
        
        # Build context embedding if context tokens provided
        if context_tokens and context is None:
            context_embeds = []
            for tid in context_tokens[-3:]:  # Use last 3 tokens as context
                if tid in self.token_to_node:
                    nid = self.token_to_node[tid]
                    if nid in self.node_embeddings:
                        context_embeds.append(self.node_embeddings[nid])
            if context_embeds:
                context = torch.stack(context_embeds).mean(dim=0)
        
        key = (from_node, to_node)
        
        if key in self.transition_skills:
            # Strengthen existing skill
            self.transition_skills[key].use()
            # Update context if provided (running average)
            if context is not None and self.transition_skills[key].context is not None:
                old_ctx = self.transition_skills[key].context
                self.transition_skills[key].context = 0.9 * old_ctx + 0.1 * context
        else:
            # Create new skill with context
            link = self.create_skill(from_node, to_node, "predict", context)
            self.transition_skills[key] = link
    
    def predict_next(self, 
                    current_tokens: List[int],
                    top_k: int = 10) -> List[Tuple[int, float]]:
        """Predict next token using skill traversal with context."""
        if not current_tokens:
            return []
        
        # Get node for last token
        last_token = current_tokens[-1]
        if last_token not in self.token_to_node:
            return []
        
        current_node = self.token_to_node[last_token]
        
        # Build context embedding
        context = None
        context_embeds = []
        for tid in current_tokens[-3:]:
            if tid in self.token_to_node:
                nid = self.token_to_node[tid]
                if nid in self.node_embeddings:
                    context_embeds.append(self.node_embeddings[nid])
        if context_embeds:
            context = torch.stack(context_embeds).mean(dim=0)
        
        # Get all prediction skills from this node
        skills = self.skills.get_skills_from(current_node, "predict")
        
        if not skills:
            return []
        
        # Score by strength AND context similarity
        candidates = []
        for skill in skills:
            target_node = skill.target_id
            
            # Compute score with context
            score = skill.strength
            if context is not None and skill.context is not None:
                similarity = F.cosine_similarity(
                    context.flatten().unsqueeze(0),
                    skill.context.flatten().unsqueeze(0)
                ).item()
                score = score * (1.0 + similarity)  # Boost by context match
            
            # Find token for this node
            for token_id, node_id in self.token_to_node.items():
                if node_id == target_node:
                    candidates.append((token_id, score))
                    break
        
        # Sort by score
        candidates.sort(key=lambda x: -x[1])
        
        return candidates[:top_k]


def load_extraction_into_computer(
    extraction_dir: Path,
    computer: PACTreeComputer,
    n_clusters: int = 64
) -> int:
    """Load extracted model into PAC tree computer."""
    
    print(f"Loading extraction from {extraction_dir}")
    
    # Load vocab embeddings
    vocab_data = torch.load(extraction_dir / "pac_vocab.pt", weights_only=False)
    vocab_embeddings = vocab_data['vocab_deltas']
    vocab_size = vocab_embeddings.shape[0]
    embed_dim = vocab_embeddings.shape[1]
    
    print(f"  Vocab: {vocab_size} tokens, {embed_dim} dim")
    
    # Create root
    root_embedding = torch.zeros(embed_dim)
    root_id = computer.inject_knowledge(root_embedding, label="root")
    print(f"  Root node: {root_id}")
    
    # Cluster embeddings
    print(f"  Clustering into {n_clusters} groups...")
    
    # Simple k-means
    indices = torch.randperm(vocab_size)[:n_clusters]
    centroids = vocab_embeddings[indices].clone()
    
    for _ in range(10):
        dists = torch.cdist(vocab_embeddings, centroids)
        assignments = dists.argmin(dim=1)
        for c in range(n_clusters):
            mask = assignments == c
            if mask.sum() > 0:
                centroids[c] = vocab_embeddings[mask].mean(dim=0)
    
    # Create cluster nodes
    for c in range(n_clusters):
        cluster_node_id = computer.inject_knowledge(
            centroids[c],
            parent_id=root_id,
            label=f"cluster_{c}"
        )
        computer.cluster_nodes[c] = cluster_node_id
        computer.cluster_centers[c] = centroids[c]
    
    print(f"  Created {n_clusters} cluster nodes")
    
    # Map tokens to clusters and create nodes for common tokens
    common_tokens = 5000  # Only materialize top tokens
    
    for token_id in range(min(common_tokens, vocab_size)):
        # Find cluster
        embedding = vocab_embeddings[token_id]
        dists = torch.cdist(embedding.unsqueeze(0), centroids)
        cluster_id = dists.argmin().item()
        cluster_node_id = computer.cluster_nodes[cluster_id]
        
        # Create token node
        node_id = computer.inject_knowledge(
            embedding,
            parent_id=cluster_node_id,
            label=f"token_{token_id}"
        )
        computer.token_to_node[token_id] = node_id
    
    print(f"  Materialized {len(computer.token_to_node)} token nodes")
    
    # Load attention patterns for skill creation
    attn_data = torch.load(extraction_dir / "pac_attention.pt", weights_only=False)
    attention_patterns = attn_data['patterns']
    
    print(f"  Creating skills from attention patterns...")
    
    # Use attention to create skills between clusters
    if attention_patterns:
        attn = attention_patterns[0]  # First layer
        seq_len = min(attn.shape[0], n_clusters)
        
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j and attn[i, j] > 0.1:
                    node_i = computer.cluster_nodes.get(i)
                    node_j = computer.cluster_nodes.get(j)
                    if node_i is not None and node_j is not None:
                        computer.create_skill(
                            node_i, node_j,
                            skill_type="associate",
                            context=centroids[i]
                        )
    
    print(f"  Created {computer.stats['skill_links']} initial skills")
    
    return root_id


# Training data - with context variety
TRAINING_SENTENCES = [
    "the weather today is cold and rainy",
    "the weather today is warm and sunny",
    "the weather tomorrow will be nice",
    "once upon a time there was a princess",
    "once upon a time there was a dragon",
    "the cat sat on the mat quietly",
    "the dog ran through the park",
    "learning requires patience and practice",
    "knowledge comes from experience",
    # Context-dependent patterns
    "I am happy today",
    "I am sad today", 
    "she is happy today",
    "he is sad today",
    "the food is good",
    "the food is bad",
    "this movie is good",
    "this movie is bad",
]


def simple_tokenize(text: str) -> List[int]:
    """Simple word-level tokenization."""
    words = text.lower().replace('.', '').replace(',', '').split()
    return [hash(w) % 5000 for w in words]


class SkillComposer:
    """
    Composes skills into higher-order skills.
    
    Skill composition:
    - Chain: A→B + B→C = A→C (sequence)
    - Parallel: A→B + A→C = A→(B,C) (options)
    - Context: A→B when X, A→C when Y (conditional)
    """
    
    def __init__(self, computer: PACTreeComputer):
        self.computer = computer
        self.composed_skills: Dict[str, ByRefLink] = {}
        self.skill_chains: Dict[int, List[List[ByRefLink]]] = defaultdict(list)
    
    def discover_chains(self, max_depth: int = 3) -> int:
        """Discover multi-hop skill chains."""
        discovered = 0
        
        # For each node with outgoing skills
        for source_id, links in self.computer.skills.outgoing.items():
            for link in links:
                # Try to extend this skill
                chains = self._find_chains(link, max_depth)
                if chains:
                    self.skill_chains[source_id].extend(chains)
                    discovered += len(chains)
        
        return discovered
    
    def _find_chains(self, start_link: ByRefLink, max_depth: int) -> List[List[ByRefLink]]:
        """Find all chains starting from a link."""
        chains = []
        
        def dfs(current_link: ByRefLink, path: List[ByRefLink], depth: int):
            if depth >= max_depth:
                if len(path) > 1:
                    chains.append(path.copy())
                return
            
            # Find next links
            next_links = self.computer.skills.get_skills_from(
                current_link.target_id, 
                current_link.skill_type
            )
            
            if not next_links:
                if len(path) > 1:
                    chains.append(path.copy())
                return
            
            for next_link in next_links:
                if next_link.target_id not in [l.source_id for l in path]:  # Avoid cycles
                    path.append(next_link)
                    dfs(next_link, path, depth + 1)
                    path.pop()
        
        dfs(start_link, [start_link], 1)
        return chains
    
    def compose_chain(self, chain: List[ByRefLink], name: str) -> ByRefLink:
        """Compose a chain into a single skill."""
        if not chain:
            return None
        
        source = chain[0].source_id
        target = chain[-1].target_id
        
        # Composed strength is product of individual strengths
        strength = 1.0
        for link in chain:
            strength *= link.strength
        
        # Create composed skill
        composed = self.computer.create_skill(
            source, target,
            skill_type="composed",
            context=chain[0].context
        )
        composed.strength = strength
        
        self.composed_skills[name] = composed
        return composed
    
    def create_conditional_skill(self,
                                  source_id: int,
                                  conditions: List[Tuple[torch.Tensor, int]],
                                  name: str) -> List[ByRefLink]:
        """Create context-dependent skills: different targets based on context."""
        skills = []
        
        for context, target_id in conditions:
            link = self.computer.create_skill(
                source_id, target_id,
                skill_type="conditional",
                context=context
            )
            skills.append(link)
        
        return skills


class MultiHopPredictor:
    """
    Prediction using multi-hop skill traversal.
    
    Instead of single-step prediction, follow chains to generate sequences.
    """
    
    def __init__(self, computer: PACTreeComputer, composer: SkillComposer):
        self.computer = computer
        self.composer = composer
    
    def predict_sequence(self, 
                        start_tokens: List[int],
                        max_length: int = 10,
                        temperature: float = 0.8) -> List[int]:
        """Generate a sequence by following skill chains."""
        if not start_tokens:
            return []
        
        generated = list(start_tokens)
        current_token = start_tokens[-1]
        
        for _ in range(max_length):
            if current_token not in self.computer.token_to_node:
                break
            
            current_node = self.computer.token_to_node[current_token]
            
            # Get all prediction skills
            skills = self.computer.skills.get_skills_from(current_node, "predict")
            
            if not skills:
                # Try composed skills
                for chain_list in self.composer.skill_chains.get(current_node, []):
                    if chain_list:
                        # Use first chain
                        target = chain_list[-1].target_id
                        for tid, nid in self.computer.token_to_node.items():
                            if nid == target:
                                generated.append(tid)
                                current_token = tid
                                break
                        break
                else:
                    break
                continue
            
            # Sample from skills based on strength
            if temperature > 0:
                strengths = torch.tensor([s.strength for s in skills])
                probs = F.softmax(strengths / temperature, dim=0)
                idx = torch.multinomial(probs, 1).item()
                chosen_skill = skills[idx]
            else:
                chosen_skill = max(skills, key=lambda s: s.strength)
            
            # Follow skill
            next_node = self.computer.skills.follow_skill(chosen_skill)
            
            # Find token for this node
            next_token = None
            for tid, nid in self.computer.token_to_node.items():
                if nid == next_node:
                    next_token = tid
                    break
            
            if next_token is None:
                break
            
            generated.append(next_token)
            current_token = next_token
        
        return generated[len(start_tokens):]
    
    def beam_search(self,
                   start_tokens: List[int],
                   beam_width: int = 3,
                   max_length: int = 10) -> List[Tuple[List[int], float]]:
        """Beam search over skill graph."""
        if not start_tokens:
            return []
        
        # Each beam: (tokens, score, last_node)
        last_token = start_tokens[-1]
        if last_token not in self.computer.token_to_node:
            return []
        
        initial_node = self.computer.token_to_node[last_token]
        beams = [([], 0.0, initial_node)]
        
        for step in range(max_length):
            candidates = []
            
            for tokens, score, current_node in beams:
                skills = self.computer.skills.get_skills_from(current_node, "predict")
                
                if not skills:
                    candidates.append((tokens, score, current_node))
                    continue
                
                for skill in skills[:beam_width]:
                    next_node = skill.target_id
                    
                    # Find token
                    next_token = None
                    for tid, nid in self.computer.token_to_node.items():
                        if nid == next_node:
                            next_token = tid
                            break
                    
                    if next_token is not None:
                        new_tokens = tokens + [next_token]
                        new_score = score + math.log(skill.strength + 1e-10)
                        candidates.append((new_tokens, new_score, next_node))
            
            # Keep top beams
            candidates.sort(key=lambda x: -x[1])
            beams = candidates[:beam_width]
            
            if not beams:
                break
        
        return [(tokens, score) for tokens, score, _ in beams]


def main():
    """Test PAC tree computation with skills."""
    
    print("="*70)
    print("PAC TREE + BYREF SKILLS - COMPLETE SYSTEM")
    print("="*70)
    print("\nKnowledge = Tree nodes")
    print("Skill = ByRef connections (how to use knowledge)")
    print("Computation = Tree traversal following skills")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📍 Device: {device}")
    
    # Create computer
    computer = PACTreeComputer(device=device, embed_dim=512)
    
    # Load extraction
    extraction_dir = Path(__file__).parent.parent / "poc_016_pac_extraction" / "extracted_v3" / "pythia_70m"
    
    if extraction_dir.exists():
        root_id = load_extraction_into_computer(extraction_dir, computer, n_clusters=64)
    else:
        print("❌ No extraction found, using random init")
        root_id = computer.inject_knowledge(torch.zeros(512), label="root")
    
    # Phase 1: Learn transition skills from training data
    print("\n" + "="*60)
    print("PHASE 1: LEARNING SKILLS (transition prediction)")
    print("="*60)
    
    word_to_id = {}
    id_to_word = {}
    
    for sentence in TRAINING_SENTENCES:
        tokens = simple_tokenize(sentence)
        words = sentence.lower().replace('.', '').replace(',', '').split()
        
        for word, tid in zip(words, tokens):
            word_to_id[word] = tid
            id_to_word[tid] = word
        
        # Learn transitions WITH context
        for i in range(len(tokens) - 1):
            context_tokens = tokens[:i+1]  # All tokens up to current
            computer.learn_transition(
                tokens[i], 
                tokens[i+1],
                context_tokens=context_tokens
            )
    
    print(f"  Learned from {len(TRAINING_SENTENCES)} sentences")
    print(f"  Total skill links: {computer.stats['skill_links']}")
    print(f"  Skill creations: {computer.stats['skill_creations']}")
    
    # Phase 2: Test prediction via skill traversal
    print("\n" + "="*60)
    print("PHASE 2: SINGLE-STEP PREDICTION")
    print("="*60)
    
    test_prompts = [
        "the weather today is",
        "once upon a time there was a",
        "the cat sat on",
        "learning requires",
    ]
    
    for prompt in test_prompts:
        tokens = simple_tokenize(prompt)
        
        # Predict next using skills
        predictions = computer.predict_next(tokens, top_k=5)
        
        if predictions:
            pred_words = []
            for tid, strength in predictions[:3]:
                if tid in id_to_word:
                    pred_words.append(f"{id_to_word[tid]}({strength:.2f})")
            print(f'  "{prompt}" → {", ".join(pred_words) if pred_words else "?"}')
        else:
            print(f'  "{prompt}" → (no skills for this context)')
    
    # Phase 3: Discover and use skill chains
    print("\n" + "="*60)
    print("PHASE 3: MULTI-HOP SKILL CHAINS")
    print("="*60)
    
    composer = SkillComposer(computer)
    chains_found = composer.discover_chains(max_depth=4)
    print(f"  Discovered {chains_found} skill chains")
    
    # Show some chains
    print("\n  Sample chains:")
    shown = 0
    for source_id, chain_lists in composer.skill_chains.items():
        for chain in chain_lists[:2]:
            if len(chain) >= 2:
                chain_words = []
                for link in chain:
                    for tid, nid in computer.token_to_node.items():
                        if nid == link.source_id:
                            if tid in id_to_word:
                                chain_words.append(id_to_word[tid])
                            break
                # Add final target
                final_target = chain[-1].target_id
                for tid, nid in computer.token_to_node.items():
                    if nid == final_target:
                        if tid in id_to_word:
                            chain_words.append(id_to_word[tid])
                        break
                
                if len(chain_words) >= 3:
                    print(f"    {' → '.join(chain_words)}")
                    shown += 1
                    if shown >= 5:
                        break
        if shown >= 5:
            break
    
    # Phase 4: Multi-hop sequence generation
    print("\n" + "="*60)
    print("PHASE 4: SEQUENCE GENERATION (multi-hop)")
    print("="*60)
    
    predictor = MultiHopPredictor(computer, composer)
    
    gen_prompts = [
        "the weather",
        "once upon",
        "the cat",
        "learning",
    ]
    
    for prompt in gen_prompts:
        tokens = simple_tokenize(prompt)
        
        # Generate sequence
        generated_tokens = predictor.predict_sequence(tokens, max_length=8)
        
        generated_words = []
        for tid in generated_tokens:
            if tid in id_to_word:
                generated_words.append(id_to_word[tid])
        
        print(f'  "{prompt}" → "{" ".join(generated_words)}"')
    
    # Phase 5: Beam search
    print("\n" + "="*60)
    print("PHASE 5: BEAM SEARCH (best paths)")
    print("="*60)
    
    for prompt in ["the", "once"]:
        tokens = simple_tokenize(prompt)
        beams = predictor.beam_search(tokens, beam_width=3, max_length=5)
        
        print(f'  "{prompt}":')
        for beam_tokens, score in beams[:3]:
            words = [id_to_word.get(t, "?") for t in beam_tokens]
            print(f'    score={score:.2f}: "{" ".join(words)}"')
    
    # Phase 6: Context-dependent skill selection
    print("\n" + "="*60)
    print("PHASE 6: CONTEXT-DEPENDENT SKILL SELECTION")
    print("="*60)
    
    # Same word "is" but different context should give different predictions
    contexts = [
        ("i am happy", "happy → today"),
        ("i am sad", "sad → today"),
        ("the food is", "is → good/bad"),
        ("this movie is", "is → good/bad"),
    ]
    
    for prompt, desc in contexts:
        tokens = simple_tokenize(prompt)
        predictions = computer.predict_next(tokens, top_k=3)
        
        if predictions:
            pred_words = [f"{id_to_word.get(t, '?')}({s:.2f})" for t, s in predictions[:3]]
            print(f'  "{prompt}" → {", ".join(pred_words)}')
        else:
            print(f'  "{prompt}" → (no predictions)')
    
    # Phase 7: Show skill graph structure
    print("\n" + "="*60)
    print("SKILL GRAPH STATISTICS")
    print("="*60)
    
    print(f"  Tree nodes: {computer.stats['tree_nodes']}")
    print(f"  Skill links: {computer.stats['skill_links']}")
    print(f"  Traversals: {computer.stats['traversals']}")
    
    # Show some actual skills
    print("\n  Sample skills:")
    shown = 0
    for (from_node, to_node), link in list(computer.transition_skills.items())[:10]:
        from_token = None
        to_token = None
        for tid, nid in computer.token_to_node.items():
            if nid == from_node:
                from_token = tid
            if nid == to_node:
                to_token = tid
        
        if from_token in id_to_word and to_token in id_to_word:
            print(f"    {id_to_word[from_token]} → {id_to_word[to_token]} "
                  f"(strength={link.strength:.2f}, used={link.usage_count})")
            shown += 1
            if shown >= 5:
                break
    
    print("\n" + "="*70)
    print("✅ PAC + BYREF SKILL SYSTEM COMPLETE")
    print("="*70)
    print("\n📊 FINAL STATISTICS:")
    print(f"  Tree nodes: {computer.stats['tree_nodes']}")
    print(f"  Skill links: {computer.stats['skill_links']}")
    print(f"  Traversals: {computer.stats['traversals']}")
    print(f"  Skill chains discovered: {chains_found}")
    
    print("\n🎯 KEY ARCHITECTURE:")
    print("  - Knowledge = PAC tree nodes (extracted from Pythia)")
    print("  - Skills = ByRef links (learned from data)")
    print("  - Computation = Graph traversal (not matrix multiply)")
    print("  - Learning = Link creation (not backprop)")
    
    print("\n💡 IMPLICATIONS:")
    print("  - Can add knowledge without retraining skills")
    print("  - Can inspect/edit skills directly")
    print("  - Memory scales with knowledge, not parameters")
    print("  - Skills are composable and transferable")


if __name__ == "__main__":
    main()
