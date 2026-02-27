"""
ByRef PAC System - True PAC Conservation
=========================================

Core insight: Higher-level concepts don't COPY lower-level knowledge.
They REFERENCE (byref) it and store only the DELTA.

Example:
  - "cat" → full PAC tree of cat knowledge
  - "dog" → full PAC tree of dog knowledge  
  - "animal" → byref[cat, dog, ...] + delta[has_life, moves, eats]

The "animal" node conserves PAC by:
1. Pointing to instances (no duplication)
2. Storing only the added abstraction (the delta)
3. Reconstructing full knowledge by traversing refs + applying delta

This maps directly to:
- Embeddings = raw entity knowledge
- Attention = byref connections (what relates to what)
- MLP = delta computation (what's added at this level)
"""

import sys
import json
import random
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib

# Add fracton path
sys.path.insert(0, r"c:\Users\peter\repos\core_workspace\fracton")

# Constants
PHI = (1 + np.sqrt(5)) / 2


@dataclass
class ByRefLink:
    """A reference to another PAC node (no copying, just pointing)"""
    target_id: str  # ID of referenced node
    weight: float = 1.0  # Strength of reference
    relation: str = "instance_of"  # Type of relation
    
    def __hash__(self):
        return hash((self.target_id, self.relation))


@dataclass 
class PACEntity:
    """
    An entity in the PAC tree with byref support.
    
    Stores:
    - delta: What THIS node adds (not inherited)
    - byrefs: References to other nodes (no copying)
    - properties: Computed from delta + byref traversal
    """
    id: str
    name: str
    delta: np.ndarray  # What this node ADDS (the distinct contribution)
    byrefs: List[ByRefLink] = field(default_factory=list)
    level: int = 0  # 0=instance, 1=category, 2=supercategory, etc.
    
    # Cached reconstruction (invalidated when byrefs change)
    _cached_full: Optional[np.ndarray] = None
    _cache_valid: bool = False
    
    def add_byref(self, target_id: str, weight: float = 1.0, relation: str = "instance_of"):
        """Add a byref link to another entity"""
        self.byrefs.append(ByRefLink(target_id, weight, relation))
        self._cache_valid = False
        
    def get_full_representation(self, entity_registry: Dict[str, 'PACEntity'], 
                                  visited: Set[str] = None) -> np.ndarray:
        """
        Reconstruct full representation by:
        1. Collecting all byref targets
        2. Averaging their representations (weighted)
        3. Adding our delta
        
        This is PAC conservation: parent = sum(children) + delta
        """
        if visited is None:
            visited = set()
            
        # Cycle detection
        if self.id in visited:
            return self.delta  # Break cycle by returning just delta
            
        visited.add(self.id)
        
        if self._cache_valid and self._cached_full is not None:
            return self._cached_full
            
        if not self.byrefs:
            # Leaf node: just our delta
            self._cached_full = self.delta
            self._cache_valid = True
            return self.delta
            
        # Collect byref representations
        weighted_sum = np.zeros_like(self.delta)
        total_weight = 0.0
        
        for ref in self.byrefs:
            if ref.target_id in entity_registry and ref.target_id not in visited:
                target = entity_registry[ref.target_id]
                target_repr = target.get_full_representation(entity_registry, visited.copy())
                
                # Handle dimension mismatch
                if target_repr.shape[0] != self.delta.shape[0]:
                    # Project to our dimension
                    min_dim = min(target_repr.shape[0], self.delta.shape[0])
                    target_repr = target_repr[:min_dim]
                    if min_dim < self.delta.shape[0]:
                        target_repr = np.pad(target_repr, (0, self.delta.shape[0] - min_dim))
                        
                weighted_sum += ref.weight * target_repr
                total_weight += ref.weight
                
        if total_weight > 0:
            averaged = weighted_sum / total_weight
        else:
            averaged = np.zeros_like(self.delta)
            
        # Full = byref average + our delta (PAC conservation)
        self._cached_full = averaged + self.delta
        self._cache_valid = True
        
        return self._cached_full


class ByRefPACTree:
    """
    PAC tree with byref support for efficient knowledge representation.
    
    Structure:
    - Level 0: Instances (cat, dog, bird) - full embeddings
    - Level 1: Categories (animal, vehicle) - byref[instances] + delta
    - Level 2: Supercategories (living_thing, object) - byref[categories] + delta
    
    Key properties:
    - No duplication: Higher levels point to lower levels
    - Delta-only storage: Each level stores only what it adds
    - Lazy reconstruction: Full representation computed on demand
    """
    
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.entities: Dict[str, PACEntity] = {}
        self.level_index: Dict[int, Set[str]] = defaultdict(set)
        self.name_to_id: Dict[str, str] = {}
        
        # Semantic clusters discovered during learning
        self.clusters: Dict[str, List[str]] = {}  # cluster_name -> [entity_ids]
        
    def _generate_id(self, name: str) -> str:
        """Generate unique ID from name"""
        return hashlib.md5(name.encode()).hexdigest()[:12]
        
    def add_instance(self, name: str, embedding: np.ndarray) -> str:
        """Add a base instance (level 0) - stores full embedding as delta"""
        entity_id = self._generate_id(name)
        
        # For instances, delta IS the full representation
        if embedding.shape[0] != self.dim:
            if embedding.shape[0] > self.dim:
                embedding = embedding[:self.dim]
            else:
                embedding = np.pad(embedding, (0, self.dim - embedding.shape[0]))
                
        entity = PACEntity(
            id=entity_id,
            name=name,
            delta=embedding,
            level=0
        )
        
        self.entities[entity_id] = entity
        self.level_index[0].add(entity_id)
        self.name_to_id[name] = entity_id
        
        return entity_id
        
    def add_category(self, name: str, instance_names: List[str], 
                     delta: Optional[np.ndarray] = None) -> str:
        """
        Add a category (level 1) that references instances.
        
        The category stores:
        - byrefs to all instances
        - delta = what makes this a "category" (abstractness)
        
        If no delta provided, compute it as the residual from instances.
        """
        entity_id = self._generate_id(name)
        
        # Find instance IDs
        byrefs = []
        instance_embeddings = []
        
        for inst_name in instance_names:
            if inst_name in self.name_to_id:
                inst_id = self.name_to_id[inst_name]
                byrefs.append(ByRefLink(inst_id, weight=1.0, relation="has_instance"))
                instance_embeddings.append(self.entities[inst_id].delta)
                
        if not byrefs:
            print(f"Warning: No instances found for category '{name}'")
            return None
            
        # Compute delta if not provided
        if delta is None:
            # Delta = what the category adds beyond the average of instances
            # This represents the "abstraction" or "category-ness"
            instance_avg = np.mean(instance_embeddings, axis=0)
            
            # The delta should be orthogonal to instances (new information)
            # Use a small learned perturbation scaled by golden ratio
            delta = np.random.randn(self.dim) * 0.1 / PHI
            
            # Make delta somewhat orthogonal to instance average
            projection = np.dot(delta, instance_avg) / (np.linalg.norm(instance_avg) + 1e-8)
            delta = delta - 0.5 * projection * instance_avg / (np.linalg.norm(instance_avg) + 1e-8)
            
        if delta.shape[0] != self.dim:
            delta = delta[:self.dim] if delta.shape[0] > self.dim else np.pad(delta, (0, self.dim - delta.shape[0]))
            
        entity = PACEntity(
            id=entity_id,
            name=name,
            delta=delta,
            byrefs=byrefs,
            level=1
        )
        
        self.entities[entity_id] = entity
        self.level_index[1].add(entity_id)
        self.name_to_id[name] = entity_id
        
        # Track cluster
        self.clusters[name] = [ref.target_id for ref in byrefs]
        
        return entity_id
        
    def add_supercategory(self, name: str, category_names: List[str],
                          delta: Optional[np.ndarray] = None) -> str:
        """Add a supercategory (level 2) that references categories"""
        entity_id = self._generate_id(name)
        
        byrefs = []
        for cat_name in category_names:
            if cat_name in self.name_to_id:
                cat_id = self.name_to_id[cat_name]
                byrefs.append(ByRefLink(cat_id, weight=1.0, relation="has_category"))
                
        if not byrefs:
            print(f"Warning: No categories found for supercategory '{name}'")
            return None
            
        if delta is None:
            delta = np.random.randn(self.dim) * 0.1 / (PHI ** 2)
            
        if delta.shape[0] != self.dim:
            delta = delta[:self.dim] if delta.shape[0] > self.dim else np.pad(delta, (0, self.dim - delta.shape[0]))
            
        entity = PACEntity(
            id=entity_id,
            name=name,
            delta=delta,
            byrefs=byrefs,
            level=2
        )
        
        self.entities[entity_id] = entity
        self.level_index[2].add(entity_id)
        self.name_to_id[name] = entity_id
        
        return entity_id
        
    def get_representation(self, name: str) -> Optional[np.ndarray]:
        """Get full representation of an entity (reconstructs from byrefs)"""
        if name not in self.name_to_id:
            return None
        entity_id = self.name_to_id[name]
        return self.entities[entity_id].get_full_representation(self.entities)
        
    def get_delta_only(self, name: str) -> Optional[np.ndarray]:
        """Get just the delta (what this entity adds)"""
        if name not in self.name_to_id:
            return None
        entity_id = self.name_to_id[name]
        return self.entities[entity_id].delta
        
    def find_similar(self, query: np.ndarray, level: int = None, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar entities by cosine similarity"""
        if query.shape[0] != self.dim:
            query = query[:self.dim] if query.shape[0] > self.dim else np.pad(query, (0, self.dim - query.shape[0]))
            
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        
        results = []
        for entity_id, entity in self.entities.items():
            if level is not None and entity.level != level:
                continue
                
            full_repr = entity.get_full_representation(self.entities)
            repr_norm = full_repr / (np.linalg.norm(full_repr) + 1e-8)
            
            sim = np.dot(query_norm, repr_norm)
            results.append((entity.name, float(sim)))
            
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
        
    def get_instances_of(self, category_name: str) -> List[str]:
        """Get all instances that a category refers to"""
        if category_name not in self.name_to_id:
            return []
            
        entity_id = self.name_to_id[category_name]
        entity = self.entities[entity_id]
        
        instances = []
        for ref in entity.byrefs:
            if ref.target_id in self.entities:
                instances.append(self.entities[ref.target_id].name)
                
        return instances
        
    def conservation_check(self, name: str) -> Dict[str, float]:
        """
        Verify PAC conservation for an entity.
        
        For categories: full_repr ≈ avg(byrefs) + delta
        """
        if name not in self.name_to_id:
            return {}
            
        entity_id = self.name_to_id[name]
        entity = self.entities[entity_id]
        
        if not entity.byrefs:
            return {'is_instance': True, 'conservation_error': 0.0}
            
        # Compute byref average
        weighted_sum = np.zeros(self.dim)
        total_weight = 0.0
        
        for ref in entity.byrefs:
            if ref.target_id in self.entities:
                target = self.entities[ref.target_id]
                target_repr = target.get_full_representation(self.entities)
                weighted_sum += ref.weight * target_repr
                total_weight += ref.weight
                
        if total_weight > 0:
            byref_avg = weighted_sum / total_weight
        else:
            byref_avg = np.zeros(self.dim)
            
        # Expected = byref_avg + delta
        expected = byref_avg + entity.delta
        
        # Actual
        actual = entity.get_full_representation(self.entities)
        
        # Conservation error
        error = np.linalg.norm(expected - actual)
        
        return {
            'is_instance': False,
            'num_byrefs': len(entity.byrefs),
            'delta_norm': float(np.linalg.norm(entity.delta)),
            'full_norm': float(np.linalg.norm(actual)),
            'conservation_error': float(error)
        }
        
    def print_hierarchy(self):
        """Print the hierarchical structure"""
        print("\n" + "="*50)
        print("ByRef PAC Tree Hierarchy")
        print("="*50)
        
        for level in sorted(self.level_index.keys(), reverse=True):
            level_name = {0: "Instances", 1: "Categories", 2: "Supercategories"}.get(level, f"Level {level}")
            print(f"\n{level_name} (Level {level}):")
            
            for entity_id in list(self.level_index[level])[:20]:  # Show first 20
                entity = self.entities[entity_id]
                delta_norm = np.linalg.norm(entity.delta)
                
                if entity.byrefs:
                    refs = [self.entities[r.target_id].name for r in entity.byrefs if r.target_id in self.entities]
                    print(f"  {entity.name} (δ={delta_norm:.3f})")
                    print(f"    → byrefs: {refs[:10]}{'...' if len(refs) > 10 else ''}")
                else:
                    print(f"  {entity.name} (δ={delta_norm:.3f})")
                    
            if len(self.level_index[level]) > 20:
                print(f"  ... and {len(self.level_index[level]) - 20} more")


class ByRefOracleDistillation:
    """
    Distill oracle knowledge into ByRef PAC trees.
    
    Process:
    1. Extract token embeddings as instances
    2. Discover clusters via oracle attention patterns
    3. Create categories with byref to instances + learned delta
    4. Build hierarchy up to concepts
    """
    
    def __init__(self, dim: int = 256):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim = dim
        self.pac_tree = ByRefPACTree(dim=dim)
        
        # Oracle models
        self.oracles = {}
        self._load_oracles()
        
        # Token confluence (from previous work)
        self.token_confluence = {}
        
    def _load_oracles(self):
        """Load oracle models"""
        print("\nLoading oracles for ByRef distillation...")
        
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            self.oracles['gpt2'] = {
                'model': GPT2LMHeadModel.from_pretrained('gpt2').to(self.device).eval(),
                'tokenizer': GPT2Tokenizer.from_pretrained('gpt2')
            }
            for p in self.oracles['gpt2']['model'].parameters():
                p.requires_grad = False
            print("  GPT-2: loaded")
        except Exception as e:
            print(f"  GPT-2 failed: {e}")
            
    def extract_token_instances(self, max_tokens: int = 5000):
        """Extract token embeddings as PAC instances"""
        print("\nExtracting token instances...")
        
        if 'gpt2' not in self.oracles:
            return
            
        tokenizer = self.oracles['gpt2']['tokenizer']
        model = self.oracles['gpt2']['model']
        
        # Get embeddings
        with torch.no_grad():
            embeddings = model.transformer.wte.weight.cpu().numpy()
            
        vocab = tokenizer.get_vocab()
        
        # Add tokens as instances
        count = 0
        for token, idx in vocab.items():
            if count >= max_tokens:
                break
                
            # Skip weird tokens
            clean = token.replace('Ġ', '').replace('▁', '').strip()
            if not clean or len(clean) < 2:
                continue
                
            # Project to our dimension
            emb = embeddings[idx]
            if emb.shape[0] > self.dim:
                emb = emb[:self.dim]
            else:
                emb = np.pad(emb, (0, self.dim - emb.shape[0]))
                
            self.pac_tree.add_instance(clean.lower(), emb)
            count += 1
            
        print(f"  Added {count} token instances")
        
    def discover_semantic_clusters(self):
        """Discover semantic clusters from embeddings"""
        print("\nDiscovering semantic clusters...")
        
        # Predefined semantic categories (could be learned)
        semantic_groups = {
            'animal': ['cat', 'dog', 'bird', 'fish', 'horse', 'mouse', 'lion', 'tiger', 'bear', 'elephant'],
            'color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple', 'pink', 'brown'],
            'number': ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'],
            'time': ['day', 'night', 'morning', 'evening', 'week', 'month', 'year', 'hour', 'minute', 'second'],
            'body': ['head', 'hand', 'eye', 'face', 'heart', 'arm', 'leg', 'foot', 'finger', 'ear'],
            'action': ['run', 'walk', 'jump', 'sit', 'stand', 'move', 'stop', 'go', 'come', 'take'],
            'emotion': ['happy', 'sad', 'angry', 'fear', 'love', 'hate', 'joy', 'hope', 'calm', 'excited'],
            'nature': ['water', 'fire', 'earth', 'air', 'sun', 'moon', 'star', 'tree', 'flower', 'rain'],
            'place': ['home', 'city', 'country', 'world', 'room', 'house', 'street', 'building', 'park', 'school'],
            'food': ['bread', 'meat', 'fruit', 'food', 'drink', 'water', 'milk', 'rice', 'fish', 'egg'],
        }
        
        for category, instances in semantic_groups.items():
            # Filter to instances we actually have
            available = [inst for inst in instances if inst in self.pac_tree.name_to_id]
            
            if len(available) >= 2:
                self.pac_tree.add_category(category, available)
                print(f"  {category}: {len(available)} instances → byref created")
                
    def build_supercategories(self):
        """Build higher-level supercategories"""
        print("\nBuilding supercategories...")
        
        supercats = {
            'living_thing': ['animal', 'body'],
            'physical': ['nature', 'place', 'food'],
            'abstract': ['emotion', 'time', 'number'],
            'quality': ['color', 'action'],
        }
        
        for supercat, categories in supercats.items():
            available = [cat for cat in categories if cat in self.pac_tree.name_to_id]
            if len(available) >= 2:
                self.pac_tree.add_supercategory(supercat, available)
                print(f"  {supercat}: {len(available)} categories → byref created")
                
    def learn_attention_byrefs(self, num_probes: int = 50):
        """
        Learn byref connections from oracle attention patterns.
        
        When oracle attention connects token A strongly to token B,
        that suggests a byref relationship.
        """
        print("\nLearning attention-based byrefs...")
        
        if 'gpt2' not in self.oracles:
            return
            
        tokenizer = self.oracles['gpt2']['tokenizer']
        model = self.oracles['gpt2']['model']
        
        # Probe with real sentences
        sentences = [
            "The cat sat on the mat.",
            "Dogs and cats are animals.",
            "Red and blue are colors.",
            "Monday is the first day.",
            "The heart pumps blood.",
            "Trees grow in the forest.",
            "Happy people smile often.",
            "Water flows in rivers.",
            "The sun is a star.",
            "Birds can fly in the sky.",
            "Fish live in water.",
            "Lions are wild animals.",
            "Morning comes before noon.",
            "Love is an emotion.",
            "Houses have rooms.",
            "Food gives us energy.",
        ]
        
        attention_links = defaultdict(lambda: defaultdict(float))
        
        for sentence in sentences:
            tokens = tokenizer.encode(sentence)
            input_ids = torch.tensor([tokens], device=self.device)
            
            with torch.no_grad():
                outputs = model(input_ids, output_attentions=True)
                
            # Average attention across layers and heads
            attentions = torch.stack(outputs.attentions)  # [layers, batch, heads, seq, seq]
            avg_attn = attentions.mean(dim=(0, 1, 2))  # [seq, seq]
            
            # Record strong attention links
            for i in range(len(tokens)):
                for j in range(len(tokens)):
                    if i != j and avg_attn[i, j] > 0.1:
                        token_i = tokenizer.decode([tokens[i]]).strip().lower()
                        token_j = tokenizer.decode([tokens[j]]).strip().lower()
                        
                        if token_i in self.pac_tree.name_to_id and token_j in self.pac_tree.name_to_id:
                            attention_links[token_i][token_j] += float(avg_attn[i, j])
                            
        # Add strongest attention links as byrefs
        links_added = 0
        for token_i, targets in attention_links.items():
            entity_id = self.pac_tree.name_to_id[token_i]
            entity = self.pac_tree.entities[entity_id]
            
            # Get top 3 attention targets
            sorted_targets = sorted(targets.items(), key=lambda x: -x[1])[:3]
            
            for token_j, weight in sorted_targets:
                if weight > 0.3:
                    target_id = self.pac_tree.name_to_id[token_j]
                    entity.add_byref(target_id, weight=weight, relation="attends_to")
                    links_added += 1
                    
        print(f"  Added {links_added} attention-based byrefs")
        
    def learn_token_confluence(self, num_probes: int = 100):
        """Learn token transitions for confluence"""
        print("\nLearning token confluence from oracle...")
        
        if 'gpt2' not in self.oracles:
            return
            
        tokenizer = self.oracles['gpt2']['tokenizer']
        model = self.oracles['gpt2']['model']
        
        # Common words for probing
        common_words = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "and", "but", "or", "if", "because", "that", "which", "who", "what",
            "this", "these", "those", "it", "they", "them", "their",
            "he", "she", "his", "her", "we", "our", "you", "your",
            "time", "year", "people", "way", "day", "man", "woman", "child", "world",
            "life", "hand", "part", "place", "case", "week", "company", "system",
        ]
        
        text_prompts = [
            "The cat sat on the mat.",
            "Scientists study the natural world.",
            "Language is a tool for communication.",
            "In the future, technology will advance.",
            "The ocean is vast and deep.",
            "Music brings people together.",
            "Books contain knowledge and wisdom.",
            "The sun shines brightly in the sky.",
            "Children learn through play.",
            "History teaches us important lessons.",
        ]
        
        for probe_idx in range(num_probes):
            # Alternate between text and common word probes
            if probe_idx % 2 == 0:
                text = text_prompts[probe_idx % len(text_prompts)]
                tokens_list = tokenizer.encode(text)
                if len(tokens_list) < 32:
                    tokens_list = tokens_list + [tokenizer.eos_token_id or 0] * (32 - len(tokens_list))
                tokens = torch.tensor([tokens_list[:32]], device=self.device)
            else:
                words = [random.choice(common_words) for _ in range(16)]
                text = ' '.join(words)
                tokens_list = tokenizer.encode(text)[:32]
                if len(tokens_list) < 32:
                    tokens_list = tokens_list + [tokenizer.eos_token_id or 0] * (32 - len(tokens_list))
                tokens = torch.tensor([tokens_list], device=self.device)
                
            with torch.no_grad():
                outputs = model(tokens)
                predictions = outputs.logits.argmax(dim=-1)
                
            # Store transitions
            for t in range(tokens.shape[1] - 1):
                next_token = predictions[0, t].item()
                
                for ctx_len in [5, 4, 3, 2]:
                    if t + 1 >= ctx_len:
                        context = tuple(tokens[0, t+1-ctx_len:t+1].cpu().tolist())
                        if context not in self.token_confluence:
                            self.token_confluence[context] = {}
                        self.token_confluence[context][next_token] = \
                            self.token_confluence[context].get(next_token, 0) + 1
                                
        print(f"  Learned {len(self.token_confluence)} confluence contexts")
        
    def distill(self):
        """Full distillation pipeline"""
        print("="*60)
        print("ByRef PAC Distillation")
        print("="*60)
        
        # 1. Extract tokens as instances - more for better semantic coverage
        self.extract_token_instances(max_tokens=10000)
        
        # 2. Discover semantic clusters
        self.discover_semantic_clusters()
        
        # 3. Build supercategories
        self.build_supercategories()
        
        # 4. Learn attention-based byrefs
        self.learn_attention_byrefs()
        
        # 5. Learn token confluence
        self.learn_token_confluence(num_probes=50)
        
        # Print results
        self.pac_tree.print_hierarchy()
        
        # Conservation check
        print("\n" + "="*50)
        print("PAC Conservation Check")
        print("="*50)
        
        for name in ['animal', 'color', 'living_thing', 'abstract']:
            if name in self.pac_tree.name_to_id:
                check = self.pac_tree.conservation_check(name)
                print(f"\n  {name}:")
                print(f"    byrefs: {check.get('num_byrefs', 0)}")
                print(f"    delta_norm: {check.get('delta_norm', 0):.4f}")
                print(f"    full_norm: {check.get('full_norm', 0):.4f}")
                print(f"    conservation_error: {check.get('conservation_error', 0):.6f}")
                
        return self
        
    def generate(self, prompt: str, max_tokens: int = 30, verbose: bool = False, 
                 temperature: float = 0.8) -> str:
        """Generate using PAC tree + confluence with temperature"""
        if 'gpt2' not in self.oracles:
            return prompt
            
        tokenizer = self.oracles['gpt2']['tokenizer']
        model = self.oracles['gpt2']['model']
        
        tokens = tokenizer.encode(prompt)
        hits = 0
        misses = 0
        recent_tokens = []  # Track recent tokens to avoid repetition
        
        for _ in range(max_tokens):
            # Try confluence
            found = False
            for ctx_len in [5, 4, 3, 2]:
                if len(tokens) >= ctx_len:
                    context = tuple(tokens[-ctx_len:])
                    if context in self.token_confluence:
                        candidates = self.token_confluence[context]
                        if candidates:
                            # Filter out recently used tokens to reduce repetition
                            filtered = {k: v for k, v in candidates.items() 
                                       if k not in recent_tokens[-3:]}
                            if not filtered:
                                filtered = candidates
                                
                            # Apply temperature
                            items = list(filtered.items())
                            weights = np.array([v for _, v in items], dtype=float)
                            
                            if temperature > 0:
                                weights = weights ** (1.0 / temperature)
                                
                            weights = weights / weights.sum()
                            
                            # Sample
                            idx = np.random.choice(len(items), p=weights)
                            next_token = items[idx][0]
                                
                            tokens.append(next_token)
                            recent_tokens.append(next_token)
                            found = True
                            hits += 1
                            break
                            
            if not found:
                misses += 1
                # Oracle fallback with top-k sampling
                input_ids = torch.tensor([tokens[-32:]], device=self.device)
                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits[0, -1] / temperature
                    
                    # Top-k sampling
                    top_k = 50
                    top_logits, top_indices = torch.topk(logits, top_k)
                    probs = torch.softmax(top_logits, dim=-1)
                    idx = torch.multinomial(probs, 1).item()
                    next_token = top_indices[idx].item()
                    
                tokens.append(next_token)
                recent_tokens.append(next_token)
                
                # Learn on the fly
                for ctx_len in [5, 4, 3, 2]:
                    if len(tokens) > ctx_len:
                        context = tuple(tokens[-(ctx_len+1):-1])
                        if context not in self.token_confluence:
                            self.token_confluence[context] = {}
                        self.token_confluence[context][next_token] = \
                            self.token_confluence[context].get(next_token, 0) + 1
                            
            # Stop on period
            if tokenizer.decode([tokens[-1]]).strip() == '.':
                break
                
        result = tokenizer.decode(tokens)
        
        if verbose:
            total = hits + misses
            hit_rate = hits / total * 100 if total > 0 else 0
            print(f"    [Confluence: {hits}/{total} = {hit_rate:.1f}%]")
            
        return result
        
    def query_category(self, category: str) -> Dict:
        """Query what we know about a category via byref traversal"""
        if category not in self.pac_tree.name_to_id:
            return {'error': f"Category '{category}' not found"}
            
        entity = self.pac_tree.entities[self.pac_tree.name_to_id[category]]
        
        # Get instances via byref
        instances = self.pac_tree.get_instances_of(category)
        
        # Get delta (what this category adds)
        delta = entity.delta
        
        # Get full representation
        full_repr = self.pac_tree.get_representation(category)
        
        # Find similar entities
        similar = self.pac_tree.find_similar(full_repr, top_k=5)
        
        return {
            'name': category,
            'level': entity.level,
            'instances': instances,
            'delta_norm': float(np.linalg.norm(delta)),
            'full_norm': float(np.linalg.norm(full_repr)),
            'similar': similar,
            'num_byrefs': len(entity.byrefs)
        }


def main():
    print("="*60)
    print("ByRef PAC System - True Conservation")
    print("="*60)
    print("Higher levels REFERENCE lower levels, store only DELTA")
    print("="*60)
    
    # Create and distill
    distiller = ByRefOracleDistillation(dim=256)
    distiller.distill()
    
    # Query examples
    print("\n" + "="*50)
    print("Category Queries (via ByRef)")
    print("="*50)
    
    for category in ['animal', 'color', 'living_thing']:
        info = distiller.query_category(category)
        print(f"\n{category}:")
        print(f"  Level: {info.get('level', 'N/A')}")
        print(f"  Instances: {info.get('instances', [])}")
        print(f"  Delta norm: {info.get('delta_norm', 0):.3f}")
        print(f"  Similar: {info.get('similar', [])[:3]}")
        
    # Generation test
    print("\n" + "="*50)
    print("Generation Test")
    print("="*50)
    
    prompts = ["The cat", "Animals are", "The color red", "In nature"]
    
    for prompt in prompts:
        print(f"\n'{prompt}' →")
        result = distiller.generate(prompt, max_tokens=25, verbose=True)
        print(f"    {result}")
        
    # Save
    results = {
        'instances': len(distiller.pac_tree.level_index[0]),
        'categories': len(distiller.pac_tree.level_index[1]),
        'supercategories': len(distiller.pac_tree.level_index[2]),
        'confluence_contexts': len(distiller.token_confluence),
        'clusters': list(distiller.pac_tree.clusters.keys())
    }
    
    output_path = Path(__file__).parent.parent / "results" / "byref_pac_system.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")
    
    return distiller


if __name__ == "__main__":
    distiller = main()
