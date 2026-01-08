"""
Concentration Monitor: Multi-scale agreement detection.

Validated in:
- POC-023: λ≈0.5 universal eigenvalue across PAC/GPT-2/primes
- POC-023: +3.6% quality, -48% collapses with reject-resample
- POC-024: φ threshold at critical transition (depth 4)

Key insight: When multiple prediction depths agree, quality is higher.
Low concentration = hallucination risk.
"""

import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


# Dawn Field Constants
PHI = 1.618033988749895
PHI_INV = 1 / PHI  # 0.618...
LAMBDA_HALF = 0.5  # Universal eigenvalue (POC-023)


@dataclass
class ConcentrationResult:
    """Result of concentration analysis."""
    concentration: float          # Fraction of depths agreeing
    predicted_token: int          # Token with highest agreement
    depth_votes: Dict[int, int]   # depth → predicted token
    confidence: float             # Based on vote margin
    is_high_quality: bool         # concentration >= threshold


class ConcentrationMonitor:
    """
    Monitor multi-scale prediction agreement.
    
    When predictions from different depths agree, the output is reliable.
    When they disagree, we're in uncertain territory (hallucination risk).
    
    POC-023 findings:
    - λ₃ ≈ 0.49 in PAC trees, 0.53 in GPT-2 layers
    - This eigenvalue is universal across domains
    - Concentration correlates with quality (r=+0.36)
    - Reject-resample intervention: +3.6% quality
    
    Usage:
        monitor = ConcentrationMonitor(transition_matrix)
        result = monitor.analyze(context, candidate_tokens)
        if result.is_high_quality:
            accept(result.predicted_token)
        else:
            resample()
    """
    
    def __init__(
        self,
        threshold: float = PHI_INV,  # 0.618 - critical threshold from POC-024
        max_depth: int = 5,
        device: str = 'cuda'
    ):
        self.threshold = threshold
        self.max_depth = max_depth
        self.device = device
        
        # Statistics
        self.stats = {
            'total_analyzed': 0,
            'high_quality': 0,
            'low_quality': 0,
            'concentration_sum': 0.0,
            'rejects': 0,
        }
    
    def analyze(
        self,
        depth_predictions: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        candidate_token: Optional[int] = None
    ) -> ConcentrationResult:
        """
        Analyze concentration across multiple prediction depths.
        
        Args:
            depth_predictions: depth → (token_ids, probs) from different n-gram levels
            candidate_token: If provided, check if this token has agreement
        
        Returns:
            ConcentrationResult with concentration score and recommendation
        """
        self.stats['total_analyzed'] += 1
        
        if not depth_predictions:
            return ConcentrationResult(
                concentration=0.0,
                predicted_token=-1,
                depth_votes={},
                confidence=0.0,
                is_high_quality=False
            )
        
        # Get top prediction from each depth
        depth_votes = {}
        for depth, (token_ids, probs) in depth_predictions.items():
            if len(token_ids) > 0:
                top_idx = probs.argmax().item()
                depth_votes[depth] = token_ids[top_idx].item()
        
        if not depth_votes:
            return ConcentrationResult(
                concentration=0.0,
                predicted_token=-1,
                depth_votes={},
                confidence=0.0,
                is_high_quality=False
            )
        
        # Count agreement
        vote_counts = {}
        for token in depth_votes.values():
            vote_counts[token] = vote_counts.get(token, 0) + 1
        
        # Find majority vote
        majority_token = max(vote_counts, key=vote_counts.get)
        majority_count = vote_counts[majority_token]
        total_votes = len(depth_votes)
        
        # Concentration = fraction agreeing with majority
        concentration = majority_count / total_votes
        
        # Confidence = margin over second place
        sorted_counts = sorted(vote_counts.values(), reverse=True)
        second_count = sorted_counts[1] if len(sorted_counts) > 1 else 0
        confidence = (majority_count - second_count) / total_votes
        
        # Check if high quality
        is_high_quality = concentration >= self.threshold
        
        # Update stats
        self.stats['concentration_sum'] += concentration
        if is_high_quality:
            self.stats['high_quality'] += 1
        else:
            self.stats['low_quality'] += 1
        
        # If candidate provided, check if it matches
        if candidate_token is not None:
            candidate_votes = vote_counts.get(candidate_token, 0)
            concentration = candidate_votes / total_votes
            is_high_quality = concentration >= self.threshold
        
        return ConcentrationResult(
            concentration=concentration,
            predicted_token=majority_token,
            depth_votes=depth_votes,
            confidence=confidence,
            is_high_quality=is_high_quality
        )
    
    def should_resample(self, result: ConcentrationResult) -> bool:
        """
        Decide whether to reject and resample.
        
        POC-023 showed this improves quality by 3.6%.
        """
        should = not result.is_high_quality
        if should:
            self.stats['rejects'] += 1
        return should
    
    def compute_transition_eigenvalue(
        self, 
        depth_agreement_matrix: torch.Tensor
    ) -> float:
        """
        Compute the third eigenvalue of depth transition matrix.
        
        POC-023 finding: λ₃ ≈ 0.5 universally.
        This is a signature of hierarchical prediction systems.
        
        Args:
            depth_agreement_matrix: (n_depths, n_depths) where
                M[i,j] = P(depth j agrees | depth i agrees)
        
        Returns:
            Third eigenvalue (expected to be ~0.5)
        """
        if depth_agreement_matrix.shape[0] < 3:
            return 0.0
        
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvals(depth_agreement_matrix.float())
        
        # Sort by magnitude
        magnitudes = eigenvalues.abs()
        sorted_indices = magnitudes.argsort(descending=True)
        
        # Return third eigenvalue (index 2)
        if len(sorted_indices) > 2:
            lambda_3 = eigenvalues[sorted_indices[2]].real.item()
            return lambda_3
        
        return 0.0
    
    def build_agreement_matrix(
        self,
        samples: List[Dict[int, int]],  # List of {depth → prediction}
    ) -> torch.Tensor:
        """
        Build depth agreement matrix from samples.
        
        Args:
            samples: List of dictionaries mapping depth → predicted token
        
        Returns:
            Agreement matrix M where M[i,j] = P(depths i,j agree)
        """
        # Find all depths
        depths = set()
        for s in samples:
            depths.update(s.keys())
        depths = sorted(depths)
        n_depths = len(depths)
        depth_to_idx = {d: i for i, d in enumerate(depths)}
        
        # Count agreements
        agreement_counts = torch.zeros(n_depths, n_depths)
        depth_counts = torch.zeros(n_depths)
        
        for sample in samples:
            for d1, pred1 in sample.items():
                idx1 = depth_to_idx[d1]
                depth_counts[idx1] += 1
                
                for d2, pred2 in sample.items():
                    idx2 = depth_to_idx[d2]
                    if pred1 == pred2:
                        agreement_counts[idx1, idx2] += 1
        
        # Normalize
        matrix = torch.zeros(n_depths, n_depths)
        for i in range(n_depths):
            if depth_counts[i] > 0:
                matrix[i] = agreement_counts[i] / depth_counts[i]
        
        return matrix
    
    def get_statistics(self) -> Dict:
        """Get monitoring statistics."""
        total = self.stats['total_analyzed']
        return {
            'total_analyzed': total,
            'high_quality_rate': self.stats['high_quality'] / total if total > 0 else 0,
            'low_quality_rate': self.stats['low_quality'] / total if total > 0 else 0,
            'mean_concentration': self.stats['concentration_sum'] / total if total > 0 else 0,
            'reject_rate': self.stats['rejects'] / total if total > 0 else 0,
            'threshold': self.threshold,
        }


if __name__ == "__main__":
    # Quick test
    monitor = ConcentrationMonitor(device='cpu')
    
    # Simulate depth predictions (all agree)
    depth_preds = {
        1: (torch.tensor([5, 3, 1]), torch.tensor([0.6, 0.3, 0.1])),
        2: (torch.tensor([5, 7, 2]), torch.tensor([0.7, 0.2, 0.1])),
        3: (torch.tensor([5, 8, 9]), torch.tensor([0.8, 0.15, 0.05])),
    }
    
    result = monitor.analyze(depth_preds)
    print(f"High agreement:")
    print(f"  Concentration: {result.concentration:.3f}")
    print(f"  Predicted: {result.predicted_token}")
    print(f"  High quality: {result.is_high_quality}")
    
    # Simulate disagreement
    depth_preds_bad = {
        1: (torch.tensor([5, 3, 1]), torch.tensor([0.6, 0.3, 0.1])),
        2: (torch.tensor([7, 5, 2]), torch.tensor([0.5, 0.3, 0.2])),
        3: (torch.tensor([9, 8, 5]), torch.tensor([0.6, 0.25, 0.15])),
    }
    
    result = monitor.analyze(depth_preds_bad)
    print(f"\nLow agreement:")
    print(f"  Concentration: {result.concentration:.3f}")
    print(f"  Predicted: {result.predicted_token}")
    print(f"  High quality: {result.is_high_quality}")
    print(f"  Should resample: {monitor.should_resample(result)}")
    
    print(f"\nStatistics: {monitor.get_statistics()}")
