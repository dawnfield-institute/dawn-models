"""
PAC Generator: Text generation using PAC tree + transitions.

Validated in:
- POC-019, 021: No backprop, pure counting
- POC-022: 65% hit rate at scale (100K vocab)
- POC-023: Reject-resample for +3.6% quality

Key insight: Generation is LOOKUP, not computation.
Build statistics, then sample from them.
"""

import torch
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from .pac_tree import PACTree
from .transitions import TransitionMatrix
from .concentration import ConcentrationMonitor, ConcentrationResult, PHI_INV


@dataclass
class GenerationResult:
    """Result of text generation."""
    tokens: torch.Tensor         # Generated token IDs
    text: str                    # Decoded text
    stats: Dict                  # Generation statistics


class PACGenerator:
    """
    Text generator using PAC tree transitions.
    
    Architecture:
    1. Look up n-gram context in transition matrix
    2. Get multi-depth predictions (depths 1-5)
    3. Check concentration (agreement across depths)
    4. If low concentration: reject and resample
    5. Return high-quality predictions
    
    NO NEURAL NETWORKS. NO GRADIENTS. PURE STATISTICS.
    
    Usage:
        gen = PACGenerator(tree, transitions)
        result = gen.generate(prompt="Once upon", max_tokens=100)
        print(result.text)
    """
    
    def __init__(
        self,
        pac_tree: PACTree,
        transition_matrix: TransitionMatrix,
        concentration_monitor: Optional[ConcentrationMonitor] = None,
        device: str = 'cuda',
        use_reject_resample: bool = True,
        max_resample_attempts: int = 5,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ):
        self.tree = pac_tree
        self.transitions = transition_matrix
        self.monitor = concentration_monitor or ConcentrationMonitor(device=device)
        self.device = device
        
        # Generation params
        self.use_reject_resample = use_reject_resample
        self.max_resample_attempts = max_resample_attempts
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        # Statistics
        self.stats = {
            'tokens_generated': 0,
            'resamples': 0,
            'fallbacks': 0,
            'cache_hits': 0,
        }
    
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_tokens: int = 100,
        min_tokens: int = 1,
        stop_tokens: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Generate tokens from prompt.
        
        Args:
            prompt_tokens: Token IDs for prompt [batch, seq] or [seq]
            max_tokens: Maximum tokens to generate
            min_tokens: Minimum tokens before stopping
            stop_tokens: Token IDs that trigger stopping
        
        Returns:
            (generated_tokens, generation_stats)
        """
        stop_tokens = set(stop_tokens or [])
        
        # Handle batch dimension
        if prompt_tokens.dim() == 1:
            prompt_tokens = prompt_tokens.unsqueeze(0)
        
        batch_size, prompt_len = prompt_tokens.shape
        device = prompt_tokens.device
        
        # Build output buffer
        output = prompt_tokens.clone()
        
        # Generate tokens one at a time
        for step in range(max_tokens):
            # Get context window
            context = output[:, -self.transitions.max_context_len:]
            
            # Get multi-depth predictions
            depth_predictions = self._get_depth_predictions(context)
            
            # Analyze concentration
            conc_result = self.monitor.analyze(depth_predictions)
            
            # Sample next token
            if self.use_reject_resample:
                next_token = self._sample_with_rejection(
                    depth_predictions, 
                    conc_result
                )
            else:
                next_token = self._sample_direct(depth_predictions)
            
            # Append token
            output = torch.cat([
                output, 
                next_token.unsqueeze(1)
            ], dim=1)
            
            self.stats['tokens_generated'] += 1
            
            # Check stopping conditions
            if step >= min_tokens - 1:
                if next_token.item() in stop_tokens:
                    break
        
        return output[:, prompt_len:], self._get_stats()
    
    def _get_depth_predictions(
        self, 
        context: torch.Tensor
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get predictions from multiple n-gram depths.
        
        Returns:
            {depth: (token_ids, probabilities)}
        """
        predictions = {}
        
        # Handle batch dimension
        if context.dim() == 2:
            context = context.squeeze(0)
        
        # Query different context lengths
        for depth in range(1, min(len(context) + 1, 6)):
            # Use last 'depth' tokens as context tuple
            ctx_tuple = tuple(context[-depth:].tolist())
            
            # Get prediction from transition matrix
            next_tokens, probs = self.transitions.predict(
                ctx_tuple,
                top_k=self.top_k
            )
            
            if len(next_tokens) > 0:
                predictions[depth] = (next_tokens, probs)
        
        return predictions
    
    def _sample_with_rejection(
        self,
        depth_predictions: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        conc_result: ConcentrationResult,
    ) -> torch.Tensor:
        """
        Sample with reject-resample for quality.
        
        POC-023: This gives +3.6% quality improvement.
        """
        for attempt in range(self.max_resample_attempts):
            # Sample from deepest available prediction
            if not depth_predictions:
                self.stats['fallbacks'] += 1
                return self._fallback_sample()
            
            max_depth = max(depth_predictions.keys())
            tokens, probs = depth_predictions[max_depth]
            
            # Apply temperature
            if self.temperature != 1.0:
                probs = (probs / self.temperature).softmax(dim=-1)
            
            # Apply top-p
            if self.top_p < 1.0:
                sorted_probs, sorted_indices = probs.sort(descending=True)
                cumsum = sorted_probs.cumsum(dim=-1)
                mask = cumsum <= self.top_p
                # Always include at least one token
                mask[0] = True
                probs = probs * mask.float()
                probs = probs / probs.sum()
            
            # Sample
            try:
                idx = torch.multinomial(probs, num_samples=1)
                sampled_token = tokens[idx].squeeze()
            except RuntimeError:
                # Fallback if multinomial fails
                sampled_token = tokens[0]
            
            # Check concentration for sampled token
            check_result = self.monitor.analyze(
                depth_predictions, 
                candidate_token=sampled_token.item()
            )
            
            if check_result.is_high_quality:
                return sampled_token.unsqueeze(0)
            
            self.stats['resamples'] += 1
        
        # Max attempts reached, use majority vote
        return torch.tensor([conc_result.predicted_token], device=self.device)
    
    def _sample_direct(
        self,
        depth_predictions: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Sample directly from deepest prediction (no rejection).
        """
        if not depth_predictions:
            self.stats['fallbacks'] += 1
            return self._fallback_sample()
        
        max_depth = max(depth_predictions.keys())
        tokens, probs = depth_predictions[max_depth]
        
        # Apply temperature
        if self.temperature != 1.0:
            probs = (probs / self.temperature).softmax(dim=-1)
        
        # Sample
        try:
            idx = torch.multinomial(probs, num_samples=1)
            return tokens[idx].squeeze().unsqueeze(0)
        except RuntimeError:
            return tokens[0].unsqueeze(0)
    
    def _fallback_sample(self) -> torch.Tensor:
        """
        Fallback when no predictions available.
        Sample uniformly from vocabulary.
        """
        vocab_size = self.transitions.vocab_size
        return torch.randint(0, vocab_size, (1,), device=self.device)
    
    def _get_stats(self) -> Dict:
        """Get generation statistics."""
        total = self.stats['tokens_generated']
        return {
            **self.stats,
            'resample_rate': self.stats['resamples'] / total if total > 0 else 0,
            'fallback_rate': self.stats['fallbacks'] / total if total > 0 else 0,
            'concentration_stats': self.monitor.get_statistics(),
        }
    
    def greedy_decode(
        self,
        prompt_tokens: torch.Tensor,
        max_tokens: int = 100,
    ) -> torch.Tensor:
        """
        Greedy decoding (always pick highest probability).
        
        Useful for deterministic outputs and testing.
        """
        if prompt_tokens.dim() == 1:
            prompt_tokens = prompt_tokens.unsqueeze(0)
        
        output = prompt_tokens.clone()
        
        for _ in range(max_tokens):
            context = output[:, -self.transitions.max_context_len:]
            
            # Get prediction
            next_tokens, probs = self.transitions.predict(context, top_k=1)
            
            if len(next_tokens) == 0:
                # Random fallback
                next_token = torch.randint(0, 50257, (1,), device=output.device)
            else:
                next_token = next_tokens[0].unsqueeze(0)
            
            output = torch.cat([output, next_token.unsqueeze(1)], dim=1)
        
        return output[:, prompt_tokens.shape[1]:]
    
    def beam_search(
        self,
        prompt_tokens: torch.Tensor,
        max_tokens: int = 100,
        beam_width: int = 5,
    ) -> List[Tuple[torch.Tensor, float]]:
        """
        Beam search decoding.
        
        Returns top beam_width sequences with their log-probabilities.
        """
        if prompt_tokens.dim() == 1:
            prompt_tokens = prompt_tokens.unsqueeze(0)
        
        # Initialize beams: (sequence, log_prob)
        beams = [(prompt_tokens.clone(), 0.0)]
        
        for step in range(max_tokens):
            all_candidates = []
            
            for seq, log_prob in beams:
                context = seq[:, -self.transitions.max_context_len:]
                
                # Get predictions
                next_tokens, probs = self.transitions.predict(context, top_k=beam_width)
                
                if len(next_tokens) == 0:
                    # Keep beam unchanged
                    all_candidates.append((seq, log_prob - 10.0))  # Penalty
                    continue
                
                # Extend beam with each candidate
                log_probs = (probs + 1e-10).log()
                
                for i, (tok, lp) in enumerate(zip(next_tokens, log_probs)):
                    new_seq = torch.cat([seq, tok.unsqueeze(0).unsqueeze(1)], dim=1)
                    new_log_prob = log_prob + lp.item()
                    all_candidates.append((new_seq, new_log_prob))
            
            # Keep top beam_width
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]
        
        return [(seq[:, prompt_tokens.shape[1]:], prob) for seq, prob in beams]


if __name__ == "__main__":
    # Quick test with mock objects
    print("PACGenerator module loaded successfully")
    print("Run with actual PACTree and TransitionMatrix for full test")
