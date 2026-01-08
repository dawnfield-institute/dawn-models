"""
Multi-Model Fusion Crystallization.

When multiple language models agree on predictions or embeddings,
those patterns are crystallized as highly reliable knowledge.

Key insight from Dawn Field Theory:
- Agreement = convergence point = high confidence
- Disagreement = entropy = needs more data
- Crystallization = phase transition when agreement exceeds threshold

This module implements:
1. Querying multiple models for predictions
2. Detecting agreement across models
3. Crystallizing agreed-upon patterns
4. Using disagreement to identify uncertain regions
"""

import torch
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
import math

from .pac_mesh import PACMeshSpace, MeshNode
from .physics_mesh import PhysicsMesh, XI, PHI, PHI_INV, LAMBDA_STAR, CollapseType


@dataclass
class ModelVote:
    """A model's vote for next token."""
    model_name: str
    token_id: int
    token_str: str
    probability: float
    embedding: Optional[torch.Tensor] = None


@dataclass
class FusionResult:
    """Result of fusing multiple model predictions."""
    agreed_tokens: List[str]
    agreement_scores: Dict[str, float]
    disagreement_tokens: List[str]
    crystallized: List[str]
    entropy: float


@dataclass
class ModelSource:
    """Metadata about a model source."""
    name: str
    embed_dim: int
    vocab_size: int
    trust_weight: float = 1.0  # How much to trust this model
    queries_made: int = 0
    agreements: int = 0
    
    @property
    def agreement_rate(self) -> float:
        if self.queries_made == 0:
            return 0.0
        return self.agreements / self.queries_made


class MultiModelFusion:
    """
    Fuse predictions from multiple language models.
    
    When models agree, crystallize the pattern.
    When they disagree, use ensemble voting.
    
    Usage:
        fusion = MultiModelFusion(physics)
        fusion.add_model('gpt2')
        fusion.add_model('EleutherAI/pythia-70m')
        
        result = fusion.fuse_predictions("The capital of France is")
        # result.agreed_tokens = ['Paris'] if both agree
    """
    
    # Agreement thresholds
    STRONG_AGREEMENT = 0.8   # 80% probability overlap
    WEAK_AGREEMENT = 0.5     # 50% overlap
    CRYSTALLIZATION_THRESHOLD = PHI_INV  # 0.618
    
    def __init__(self,
                 physics: PhysicsMesh,
                 device: str = 'cpu'):
        self.physics = physics
        self.mesh = physics.mesh
        self.device = device
        
        # Model registry
        self.models: Dict[str, ModelSource] = {}
        self.loaded_models: Dict[str, any] = {}
        self.tokenizers: Dict[str, any] = {}
        
        # Fusion memory
        self.agreement_history: List[Tuple[str, float]] = []
        self.crystallization_count = 0
    
    def add_model(self, 
                  model_name: str,
                  trust_weight: float = 1.0) -> None:
        """
        Add a model to the fusion ensemble.
        
        Model is loaded lazily when first queried.
        """
        self.models[model_name] = ModelSource(
            name=model_name,
            embed_dim=0,  # Set when loaded
            vocab_size=0,
            trust_weight=trust_weight
        )
    
    def _load_model(self, model_name: str) -> None:
        """Lazy load a model."""
        if model_name in self.loaded_models:
            return
            
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading {model_name} for fusion...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(self.device)
        model.eval()
        
        self.loaded_models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        
        # Update metadata
        self.models[model_name].embed_dim = model.config.hidden_size
        self.models[model_name].vocab_size = model.config.vocab_size
        
        print(f"  Loaded: {sum(p.numel() for p in model.parameters()):,} params")
    
    def query_model(self,
                    model_name: str,
                    context: str,
                    top_k: int = 10) -> List[ModelVote]:
        """
        Query a single model for predictions.
        
        Returns list of ModelVote with probabilities.
        """
        self._load_model(model_name)
        
        model = self.loaded_models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        with torch.no_grad():
            inputs = tokenizer(context, return_tensors='pt').to(self.device)
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=top_k)
            
            votes = []
            for prob, idx in zip(top_probs, top_indices):
                token_str = tokenizer.decode([idx.item()])
                votes.append(ModelVote(
                    model_name=model_name,
                    token_id=idx.item(),
                    token_str=token_str,
                    probability=prob.item()
                ))
        
        self.models[model_name].queries_made += 1
        return votes
    
    def fuse_predictions(self,
                         context: str,
                         top_k: int = 10,
                         crystallize: bool = True) -> FusionResult:
        """
        Query all models and fuse their predictions.
        
        1. Get top-k predictions from each model
        2. Compute agreement scores
        3. Crystallize highly-agreed patterns
        4. Return fusion result
        """
        if not self.models:
            return FusionResult(
                agreed_tokens=[],
                agreement_scores={},
                disagreement_tokens=[],
                crystallized=[],
                entropy=1.0
            )
        
        # Collect votes from all models
        all_votes: Dict[str, List[ModelVote]] = {}
        
        for model_name in self.models:
            votes = self.query_model(model_name, context, top_k)
            all_votes[model_name] = votes
        
        # Compute agreement
        token_scores: Dict[str, Dict[str, float]] = {}  # token -> {model -> prob}
        
        for model_name, votes in all_votes.items():
            trust = self.models[model_name].trust_weight
            for vote in votes:
                token = vote.token_str.strip()
                if token not in token_scores:
                    token_scores[token] = {}
                token_scores[token][model_name] = vote.probability * trust
        
        # Calculate agreement scores
        num_models = len(self.models)
        agreement_scores: Dict[str, float] = {}
        agreed_tokens: List[str] = []
        disagreement_tokens: List[str] = []
        
        for token, model_probs in token_scores.items():
            # Agreement = how many models have this token × average probability
            coverage = len(model_probs) / num_models
            avg_prob = sum(model_probs.values()) / len(model_probs)
            agreement = coverage * avg_prob
            
            agreement_scores[token] = agreement
            
            if coverage >= self.WEAK_AGREEMENT:
                agreed_tokens.append(token)
                # Update model agreement stats
                for model_name in model_probs:
                    self.models[model_name].agreements += 1
            else:
                disagreement_tokens.append(token)
        
        # Sort by agreement
        agreed_tokens.sort(key=lambda t: agreement_scores.get(t, 0), reverse=True)
        
        # Calculate entropy
        total_agreement = sum(agreement_scores.values())
        if total_agreement > 0:
            probs = [s / total_agreement for s in agreement_scores.values()]
            entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
        else:
            entropy = 1.0
        
        # Crystallize strongly agreed patterns
        crystallized = []
        if crystallize:
            for token in agreed_tokens:
                if agreement_scores[token] >= self.CRYSTALLIZATION_THRESHOLD:
                    self._crystallize_token(context, token, agreement_scores[token])
                    crystallized.append(token)
                    self.crystallization_count += 1
        
        # Record history
        if agreed_tokens:
            self.agreement_history.append((agreed_tokens[0], agreement_scores.get(agreed_tokens[0], 0)))
        
        return FusionResult(
            agreed_tokens=agreed_tokens,
            agreement_scores=agreement_scores,
            disagreement_tokens=disagreement_tokens,
            crystallized=crystallized,
            entropy=entropy
        )
    
    def _crystallize_token(self,
                           context: str,
                           token: str,
                           confidence: float) -> Optional[MeshNode]:
        """
        Crystallize an agreed-upon token into the mesh.
        
        Creates or strengthens a node and marks it as an attractor.
        """
        # Find or create node in mesh
        token_id = hash(token) % 1000000
        
        # Create simple embedding from hash (or use model embedding if available)
        emb = torch.zeros(self.mesh.embed_dim)
        for i, c in enumerate(token):
            emb[i % self.mesh.embed_dim] += ord(c) / 256
        emb = emb / (len(token) + 1)
        
        # Create root node (context handling would need context nodes)
        node = self.mesh.get_or_create_root(
            token_id, token, emb, "fusion"
        )
        
        # Mark as crystallized attractor
        node.confidence = min(1.0, node.confidence + confidence * 0.2)
        
        # Add an incoming path to make it a convergence point
        node.incoming_paths[f"fusion_{hash(context) % 10000}"] = 1
        
        # Add to physics attractors
        self.physics.attractors[node.node_id] = confidence
        
        # Add to collapse crystallized set
        self.physics.collapse.crystallized.add(node.node_id)
        
        return node
    
    def fuse_embeddings(self,
                        tokens: List[str],
                        method: str = 'average') -> torch.Tensor:
        """
        Fuse embeddings from multiple models.
        
        Methods:
        - 'average': Simple average of normalized embeddings
        - 'weighted': Trust-weighted average
        - 'concat': Concatenate all embeddings
        
        Note: Different models have different embedding dims,
        so we normalize to a common space.
        """
        if not self.models:
            return torch.zeros(self.mesh.embed_dim)
        
        # Load all models
        for model_name in self.models:
            self._load_model(model_name)
        
        all_embeddings: List[torch.Tensor] = []
        weights: List[float] = []
        
        for model_name in self.models:
            model = self.loaded_models[model_name]
            tokenizer = self.tokenizers[model_name]
            trust = self.models[model_name].trust_weight
            
            with torch.no_grad():
                # Get embeddings from model
                inputs = tokenizer(tokens, return_tensors='pt', padding=True).to(self.device)
                
                # Get hidden states
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]  # Last layer
                
                # Average over tokens
                emb = hidden.mean(dim=1).squeeze()
                
                # Normalize to mesh dim
                if emb.shape[-1] != self.mesh.embed_dim:
                    # Simple projection (could use learned projection)
                    if emb.shape[-1] > self.mesh.embed_dim:
                        emb = emb[:self.mesh.embed_dim]
                    else:
                        pad = torch.zeros(self.mesh.embed_dim - emb.shape[-1])
                        emb = torch.cat([emb, pad])
                
                all_embeddings.append(emb)
                weights.append(trust)
        
        # Fuse based on method
        if method == 'average':
            fused = torch.stack(all_embeddings).mean(dim=0)
        elif method == 'weighted':
            weights_tensor = torch.tensor(weights) / sum(weights)
            fused = sum(e * w for e, w in zip(all_embeddings, weights_tensor))
        elif method == 'concat':
            # This changes dimension - use with caution
            fused = torch.cat(all_embeddings)
        else:
            fused = all_embeddings[0]
        
        return fused
    
    def detect_uncertainty(self,
                           context: str,
                           threshold: float = 0.3) -> Dict[str, float]:
        """
        Detect tokens where models strongly disagree.
        
        High disagreement indicates uncertainty - useful for
        knowing when to be less confident or seek more data.
        """
        result = self.fuse_predictions(context, crystallize=False)
        
        uncertain = {}
        for token in result.disagreement_tokens:
            score = result.agreement_scores.get(token, 0)
            if score < threshold:
                uncertain[token] = 1.0 - score
        
        return uncertain
    
    def stats(self) -> Dict:
        """Get fusion statistics."""
        model_stats = {}
        for name, source in self.models.items():
            model_stats[name] = {
                'trust': source.trust_weight,
                'queries': source.queries_made,
                'agreement_rate': source.agreement_rate
            }
        
        return {
            'models': len(self.models),
            'crystallizations': self.crystallization_count,
            'agreement_history_len': len(self.agreement_history),
            'model_stats': model_stats
        }


class FusionGenerator:
    """
    Generator that uses multi-model fusion for prediction.
    
    At each step, queries multiple models and crystallizes agreements.
    """
    
    def __init__(self,
                 fusion: MultiModelFusion,
                 max_tokens: int = 50,
                 temperature: float = 0.8):
        self.fusion = fusion
        self.physics = fusion.physics
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def generate(self, prompt: str) -> Tuple[str, FusionResult]:
        """
        Generate text using multi-model fusion.
        
        At each step:
        1. Query all models for next token
        2. Fuse predictions (crystallize agreements)
        3. Sample from fused distribution
        4. Repeat
        """
        generated = []
        context = prompt
        total_crystallized = []
        total_entropy = 0.0
        
        for _ in range(self.max_tokens):
            # Fuse predictions
            result = self.fusion.fuse_predictions(context)
            
            if not result.agreed_tokens:
                break
            
            # Sample from agreed tokens (temperature-adjusted)
            scores = [result.agreement_scores.get(t, 0) for t in result.agreed_tokens]
            
            if self.temperature != 1.0:
                scores = [s ** (1.0 / self.temperature) for s in scores]
            
            total = sum(scores)
            if total <= 0:
                break
            
            probs = [s / total for s in scores]
            
            # Sample
            import random
            r = random.random()
            cumsum = 0.0
            chosen = result.agreed_tokens[0]
            for token, prob in zip(result.agreed_tokens, probs):
                cumsum += prob
                if r < cumsum:
                    chosen = token
                    break
            
            generated.append(chosen)
            context = context + chosen
            total_crystallized.extend(result.crystallized)
            total_entropy += result.entropy
            
            # Stop conditions
            if chosen in ['.', '!', '?', '\n']:
                break
        
        avg_entropy = total_entropy / max(1, len(generated))
        
        final_result = FusionResult(
            agreed_tokens=generated,
            agreement_scores={},
            disagreement_tokens=[],
            crystallized=total_crystallized,
            entropy=avg_entropy
        )
        
        return prompt + "".join(generated), final_result


# For testing without loading actual models
class MockModel:
    """Mock model for testing fusion without GPU/model loading."""
    
    def __init__(self, name: str, vocab: List[str], bias: Dict[str, float] = None):
        self.name = name
        self.vocab = vocab
        self.bias = bias or {}
        
    def predict(self, context: str, top_k: int = 10) -> List[ModelVote]:
        """Generate mock predictions based on hash and bias."""
        import hashlib
        
        votes = []
        base_hash = int(hashlib.md5(context.encode()).hexdigest()[:8], 16)
        
        for i, token in enumerate(self.vocab[:top_k]):
            # Create probability from hash + bias
            hash_prob = ((base_hash + i * 1337) % 100) / 100
            bias_prob = self.bias.get(token, 0)
            prob = min(1.0, hash_prob * 0.7 + bias_prob * 0.3)
            
            votes.append(ModelVote(
                model_name=self.name,
                token_id=i,
                token_str=token,
                probability=prob
            ))
        
        # Normalize
        total = sum(v.probability for v in votes)
        if total > 0:
            for v in votes:
                v.probability /= total
        
        return votes


class MockMultiModelFusion(MultiModelFusion):
    """Fusion using mock models for testing."""
    
    def __init__(self, physics: PhysicsMesh, device: str = 'cpu'):
        super().__init__(physics, device)
        self.mock_models: Dict[str, MockModel] = {}
    
    def add_mock_model(self,
                       name: str,
                       vocab: List[str],
                       bias: Dict[str, float] = None,
                       trust_weight: float = 1.0) -> None:
        """Add a mock model for testing."""
        self.models[name] = ModelSource(
            name=name,
            embed_dim=64,
            vocab_size=len(vocab),
            trust_weight=trust_weight
        )
        self.mock_models[name] = MockModel(name, vocab, bias)
    
    def query_model(self,
                    model_name: str,
                    context: str,
                    top_k: int = 10) -> List[ModelVote]:
        """Query mock model."""
        if model_name in self.mock_models:
            self.models[model_name].queries_made += 1
            return self.mock_models[model_name].predict(context, top_k)
        return super().query_model(model_name, context, top_k)
