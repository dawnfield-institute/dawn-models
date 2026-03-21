"""
PAC Oracle Distillation: Growing transformers using real model as loss function.

Key insight: Instead of extracted templates, use the REAL model as an oracle.
- Probe with broad signals (parameter sweep)
- Oracle gives "correct" outputs
- Student grows to match oracle behavior
- ByRef links = regions of input space mastered

This is online distillation with strategic probing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import sys

# Add fracton to path
fracton_path = Path(__file__).parent.parent.parent.parent / "fracton"
sys.path.insert(0, str(fracton_path))

from fracton.physics.constants import PHI, XI, PHI_XI

# Check if transformers available
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️ transformers not installed - oracle mode disabled")


@dataclass
class ProbeResult:
    """Result of probing a region of input space."""
    probe_type: str  # 'random', 'structured', 'interpolation', 'edge'
    input_region: str  # Description of input region
    oracle_loss: float  # How well student matches oracle
    mastered: bool  # Whether this region is mastered
    probe_id: int


@dataclass 
class MasteredRegion:
    """A region of input space the student has mastered (ByRef link)."""
    probe_type: str
    region_description: str
    mastery_score: float
    layer_idx: int  # Which layer learned this
    probe_ids: List[int] = field(default_factory=list)


class StudentLayer(nn.Module):
    """A single transformer layer that can be trained independently."""
    
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        # Track what this layer has learned
        self.mastered_regions: List[MasteredRegion] = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        
        # MLP
        h = self.norm2(x)
        h = self.mlp(h)
        return x + h


class ProbeGenerator:
    """Generates diverse probes to sweep the input space."""
    
    def __init__(self, vocab_size: int, device: str = 'cpu'):
        self.vocab_size = vocab_size
        self.device = device
        self.probe_counter = 0
    
    def random_tokens(self, batch_size: int = 32, seq_len: int = 32) -> Tuple[torch.Tensor, str]:
        """Random token sequences - broad coverage."""
        tokens = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        self.probe_counter += 1
        return tokens, f"random_{self.probe_counter}"
    
    def structured_tokens(self, batch_size: int = 32, seq_len: int = 32, 
                          pattern: str = 'repeat') -> Tuple[torch.Tensor, str]:
        """Structured patterns - test specific behaviors."""
        if pattern == 'repeat':
            # Repeating patterns
            base = torch.randint(0, self.vocab_size, (batch_size, seq_len // 4), device=self.device)
            tokens = base.repeat(1, 4)
        elif pattern == 'ascending':
            # Ascending sequences
            start = torch.randint(0, self.vocab_size - seq_len, (batch_size, 1), device=self.device)
            tokens = start + torch.arange(seq_len, device=self.device).unsqueeze(0)
        elif pattern == 'common':
            # Common tokens (low IDs tend to be frequent)
            tokens = torch.randint(0, min(1000, self.vocab_size), (batch_size, seq_len), device=self.device)
        else:
            tokens = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        
        self.probe_counter += 1
        return tokens, f"structured_{pattern}_{self.probe_counter}"
    
    def interpolation_tokens(self, anchor1: torch.Tensor, anchor2: torch.Tensor,
                             n_steps: int = 8) -> Tuple[torch.Tensor, str]:
        """Interpolate between two sequences - test smooth transitions."""
        # Token-level interpolation: mix tokens from both
        batch_size, seq_len = anchor1.shape
        alphas = torch.linspace(0, 1, n_steps, device=self.device)
        
        results = []
        for alpha in alphas:
            mask = torch.rand(batch_size, seq_len, device=self.device) < alpha
            mixed = torch.where(mask, anchor2, anchor1)
            results.append(mixed)
        
        tokens = torch.cat(results, dim=0)
        self.probe_counter += 1
        return tokens, f"interpolation_{self.probe_counter}"
    
    def edge_tokens(self, batch_size: int = 32, seq_len: int = 32) -> Tuple[torch.Tensor, str]:
        """Edge cases - rare tokens, long sequences, etc."""
        # High token IDs (rare tokens)
        tokens = torch.randint(
            self.vocab_size - 1000, self.vocab_size,
            (batch_size, seq_len), device=self.device
        )
        self.probe_counter += 1
        return tokens, f"edge_rare_{self.probe_counter}"
    
    def coherent_tokens(self, batch_size: int = 32, seq_len: int = 32,
                         tokenizer=None, oracle=None) -> Tuple[torch.Tensor, str]:
        """Generate coherent text by sampling from oracle, then use as training data."""
        if tokenizer is None or oracle is None:
            return self.random_tokens(batch_size, seq_len)
        
        # Start with common prompts
        prompts = [
            "The", "In a", "It was", "There is", "When the", 
            "After", "Before", "The meaning", "Scientists", "Today"
        ]
        
        all_tokens = []
        for i in range(batch_size):
            prompt = prompts[i % len(prompts)]
            tokens = tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # Generate from oracle to get coherent sequences
            with torch.no_grad():
                for _ in range(seq_len - tokens.shape[1]):
                    logits = oracle(tokens).logits
                    # Sample with temperature
                    probs = F.softmax(logits[0, -1] / 0.8, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            
            # Pad or truncate to seq_len
            if tokens.shape[1] < seq_len:
                tokens = F.pad(tokens, (0, seq_len - tokens.shape[1]))
            else:
                tokens = tokens[:, :seq_len]
            
            all_tokens.append(tokens)
        
        result = torch.cat(all_tokens, dim=0)
        self.probe_counter += 1
        return result, f"coherent_{self.probe_counter}"
    
    def parameter_sweep(self, n_probes: int = 10, batch_size: int = 32, 
                        seq_len: int = 32,
                        tokenizer=None, oracle=None) -> List[Tuple[torch.Tensor, str, str]]:
        """
        Generate diverse probes covering the parameter space.
        
        Returns list of (tokens, probe_id, probe_type)
        """
        probes = []
        
        # Mix of probe types - prioritize coherent when available
        for i in range(n_probes):
            if i % 6 == 0:
                tokens, pid = self.random_tokens(batch_size, seq_len)
                probes.append((tokens, pid, 'random'))
            elif i % 6 == 1:
                tokens, pid = self.structured_tokens(batch_size, seq_len, 'repeat')
                probes.append((tokens, pid, 'structured'))
            elif i % 6 == 2:
                tokens, pid = self.structured_tokens(batch_size, seq_len, 'common')
                probes.append((tokens, pid, 'common'))
            elif i % 6 == 3:
                tokens, pid = self.edge_tokens(batch_size, seq_len)
                probes.append((tokens, pid, 'edge'))
            elif i % 6 == 4 and tokenizer is not None and oracle is not None:
                # Coherent text from oracle
                tokens, pid = self.coherent_tokens(batch_size, seq_len, tokenizer, oracle)
                probes.append((tokens, pid, 'coherent'))
            else:
                # Interpolation between random anchors
                a1, _ = self.random_tokens(batch_size // 2, seq_len)
                a2, _ = self.random_tokens(batch_size // 2, seq_len)
                tokens, pid = self.interpolation_tokens(a1, a2)
                probes.append((tokens, pid, 'interpolation'))
        
        return probes


class GrowingStudent(nn.Module):
    """
    Transformer that grows by learning from an oracle model.
    
    The oracle (real Pythia) is used as the loss function.
    Student probes input space and grows to match oracle behavior.
    """
    
    def __init__(self,
                 vocab_size: int = 50304,
                 initial_dim: int = 64,
                 n_heads: int = 4,
                 device: str = 'cpu'):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.current_dim = initial_dim
        self.n_heads = n_heads
        self.device = device
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, initial_dim)
        self.pos_embedding = nn.Embedding(1024, initial_dim)
        
        # Student layers
        self.layers = nn.ModuleList([
            StudentLayer(initial_dim, n_heads)
        ])
        
        # Output
        self.output_norm = nn.LayerNorm(initial_dim)
        self.output_proj = nn.Linear(initial_dim, vocab_size)
        
        # Mastered regions (ByRef links to input space)
        self.mastered_regions: List[MasteredRegion] = []
        
        # Growth tracking
        self.growth_events = []
        
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - returns logits."""
        B, T = x.shape
        
        tok_emb = self.embedding(x)
        pos_emb = self.pos_embedding(torch.arange(T, device=self.device))
        h = tok_emb + pos_emb
        
        for layer in self.layers:
            h = layer(h)
        
        h = self.output_norm(h)
        return self.output_proj(h)
    
    def get_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """Get hidden states (for comparing with oracle)."""
        B, T = x.shape
        
        tok_emb = self.embedding(x)
        pos_emb = self.pos_embedding(torch.arange(T, device=self.device))
        h = tok_emb + pos_emb
        
        for layer in self.layers:
            h = layer(h)
        
        return self.output_norm(h)
    
    def grow(self, reason: str = ""):
        """Add layer and/or grow dimensions."""
        old_dim = self.current_dim
        new_dim = min(512, int(old_dim * PHI))
        new_dim = (new_dim // self.n_heads) * self.n_heads
        
        if new_dim <= old_dim:
            new_dim = old_dim + self.n_heads
        
        # Don't exceed 512 for memory
        if new_dim > 512:
            new_dim = 512
        
        grew_dim = new_dim > old_dim
        
        if grew_dim:
            print(f"  🌱 GROW: {old_dim} → {new_dim} dim")
            
            # Expand embedding
            new_embedding = nn.Embedding(self.vocab_size, new_dim, device=self.device)
            with torch.no_grad():
                new_embedding.weight[:, :old_dim] = self.embedding.weight
            self.embedding = new_embedding
            
            new_pos = nn.Embedding(1024, new_dim, device=self.device)
            with torch.no_grad():
                new_pos.weight[:, :old_dim] = self.pos_embedding.weight
            self.pos_embedding = new_pos
            
            # Expand existing layers
            for layer in self.layers:
                self._expand_layer(layer, old_dim, new_dim)
            
            # Expand output
            self.output_norm = nn.LayerNorm(new_dim).to(self.device)
            new_output = nn.Linear(new_dim, self.vocab_size, device=self.device)
            with torch.no_grad():
                new_output.weight[:, :old_dim] = self.output_proj.weight
            self.output_proj = new_output
            
            self.current_dim = new_dim
        
        # Add new layer
        print(f"  🌱 ADD LAYER: {len(self.layers)} → {len(self.layers)+1}")
        new_layer = StudentLayer(self.current_dim, self.n_heads).to(self.device)
        self.layers.append(new_layer)
        
        self.growth_events.append({
            'old_dim': old_dim,
            'new_dim': self.current_dim,
            'n_layers': len(self.layers),
            'reason': reason
        })
    
    def _expand_layer(self, layer: StudentLayer, old_dim: int, new_dim: int):
        """Expand a layer's dimensions."""
        layer.norm1 = nn.LayerNorm(new_dim).to(self.device)
        layer.norm2 = nn.LayerNorm(new_dim).to(self.device)
        layer.attn = nn.MultiheadAttention(new_dim, self.n_heads, batch_first=True).to(self.device)
        layer.mlp = nn.Sequential(
            nn.Linear(new_dim, new_dim * 4),
            nn.GELU(),
            nn.Linear(new_dim * 4, new_dim)
        ).to(self.device)
        layer.dim = new_dim


class OracleDistillation:
    """
    Distill from oracle model using probe-based parameter sweep.
    
    Strategy:
    1. Generate diverse probes (parameter sweep)
    2. Get oracle outputs for each probe
    3. Train student to match
    4. If student can't match, grow
    5. Record mastered regions as ByRef links
    """
    
    def __init__(self,
                 oracle_name: str = "EleutherAI/pythia-70m",
                 initial_dim: int = 64,
                 device: str = 'cuda'):
        self.device = device
        self.oracle_name = oracle_name
        
        # Load oracle
        print(f"Loading oracle: {oracle_name}")
        if HAS_TRANSFORMERS:
            self.oracle = AutoModelForCausalLM.from_pretrained(
                oracle_name,
                torch_dtype=torch.float32
            ).to(device)
            self.oracle.eval()
            for p in self.oracle.parameters():
                p.requires_grad = False
            
            self.tokenizer = AutoTokenizer.from_pretrained(oracle_name)
            vocab_size = self.oracle.config.vocab_size
            print(f"  Oracle loaded: {sum(p.numel() for p in self.oracle.parameters()):,} params")
            
            # Get oracle embedding for initialization
            self.oracle_embedding = self.oracle.gpt_neox.embed_in.weight.detach()
        else:
            self.oracle = None
            self.tokenizer = None
            self.oracle_embedding = None
            vocab_size = 50304
            print("  ⚠️ Using mock oracle (transformers not available)")
        
        # Create student
        self.student = GrowingStudent(
            vocab_size=vocab_size,
            initial_dim=initial_dim,
            n_heads=4,
            device=device
        )
        
        # Probe generator
        self.probe_gen = ProbeGenerator(vocab_size, device)
        
        # Initialize student embedding from oracle (projected)
        if self.oracle_embedding is not None:
            print("  Initializing student embeddings from oracle...")
            oracle_dim = self.oracle_embedding.shape[1]  # 512
            student_dim = self.student.current_dim  # 64
            
            # Simple projection: just take first student_dim dimensions
            with torch.no_grad():
                projected = self.oracle_embedding[:, :student_dim]
                self.student.embedding.weight.copy_(projected.to(device))
            print(f"    Truncated {oracle_dim} → {student_dim} dims")
        
        # Stats
        self.total_probes = 0
        self.regions_mastered = 0
    
    def get_oracle_output(self, tokens: torch.Tensor) -> torch.Tensor:
        """Get oracle's output for given tokens."""
        if self.oracle is None:
            # Mock oracle: just return random logits
            return torch.randn(tokens.shape[0], tokens.shape[1], self.student.vocab_size, device=self.device)
        
        with torch.no_grad():
            outputs = self.oracle(tokens)
            return outputs.logits
    
    def compute_distillation_loss(self, 
                                   student_logits: torch.Tensor,
                                   oracle_logits: torch.Tensor,
                                   input_tokens: torch.Tensor,
                                   temperature: float = 2.0) -> torch.Tensor:
        """
        Combined loss: KL divergence + causal language modeling.
        
        KL teaches distribution matching.
        CLM teaches sequence prediction.
        """
        # KL divergence (soft targets)
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        oracle_probs = F.softmax(oracle_logits / temperature, dim=-1)
        kl_loss = F.kl_div(student_probs, oracle_probs, reduction='batchmean') * (temperature ** 2)
        
        # Causal language modeling loss (predict next token)
        # Shift for next-token prediction
        shift_logits = student_logits[:, :-1, :].contiguous()
        shift_labels = input_tokens[:, 1:].contiguous()
        
        clm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean'
        )
        
        # Combined loss (weight CLM more for generation ability)
        return 0.5 * kl_loss + 0.5 * clm_loss
    
    def probe_and_train(self,
                        probe_tokens: torch.Tensor,
                        probe_id: str,
                        probe_type: str,
                        n_steps: int = 50,
                        mastery_threshold: float = 1.0) -> ProbeResult:
        """
        Probe a region of input space and train to match oracle.
        
        Returns probe result with mastery status.
        """
        self.total_probes += 1
        
        # Get oracle output
        oracle_logits = self.get_oracle_output(probe_tokens)
        
        # Train student
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=1e-3)
        
        losses = []
        for step in range(n_steps):
            student_logits = self.student(probe_tokens)
            loss = self.compute_distillation_loss(student_logits, oracle_logits, probe_tokens)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
        
        final_loss = sum(losses[-10:]) / 10 if len(losses) >= 10 else sum(losses) / len(losses)
        mastered = final_loss < mastery_threshold
        
        return ProbeResult(
            probe_type=probe_type,
            input_region=probe_id,
            oracle_loss=final_loss,
            mastered=mastered,
            probe_id=self.total_probes
        )
    
    def sweep_and_grow(self,
                       n_sweeps: int = 5,
                       probes_per_sweep: int = 10,
                       mastery_threshold: float = 1.0,
                       growth_trigger: float = 0.5) -> Dict:
        """
        Main training loop: sweep input space, grow as needed.
        
        Args:
            n_sweeps: Number of sweep iterations
            probes_per_sweep: Probes per sweep
            mastery_threshold: Loss below which region is "mastered"
            growth_trigger: Fraction of failed probes that triggers growth
        """
        print("\n" + "="*60)
        print("PARAMETER SWEEP DISTILLATION")
        print("="*60)
        
        initial_params = sum(p.numel() for p in self.student.parameters())
        
        results = {
            'sweeps': [],
            'growth_events': 0,
            'total_probes': 0,
            'regions_mastered': 0
        }
        
        for sweep_idx in range(n_sweeps):
            print(f"\n📊 Sweep {sweep_idx + 1}/{n_sweeps}")
            print("-" * 40)
            
            # Generate probes for this sweep
            probes = self.probe_gen.parameter_sweep(
                n_probes=probes_per_sweep,
                batch_size=16,
                seq_len=32,
                tokenizer=self.tokenizer,
                oracle=self.oracle
            )
            
            sweep_results = []
            mastered_count = 0
            
            for tokens, probe_id, probe_type in probes:
                result = self.probe_and_train(
                    tokens, probe_id, probe_type,
                    n_steps=100,  # More training per probe
                    mastery_threshold=mastery_threshold
                )
                
                sweep_results.append(result)
                
                if result.mastered:
                    mastered_count += 1
                    # Record as ByRef link
                    region = MasteredRegion(
                        probe_type=probe_type,
                        region_description=probe_id,
                        mastery_score=1.0 - result.oracle_loss / mastery_threshold,
                        layer_idx=len(self.student.layers) - 1,
                        probe_ids=[result.probe_id]
                    )
                    self.student.mastered_regions.append(region)
                    self.regions_mastered += 1
            
            # Report sweep results
            avg_loss = sum(r.oracle_loss for r in sweep_results) / len(sweep_results)
            mastery_rate = mastered_count / len(sweep_results)
            
            print(f"  Probes: {len(sweep_results)}")
            print(f"  Avg loss: {avg_loss:.4f}")
            print(f"  Mastery rate: {mastery_rate:.1%} ({mastered_count}/{len(sweep_results)})")
            
            results['sweeps'].append({
                'sweep_idx': sweep_idx,
                'avg_loss': avg_loss,
                'mastery_rate': mastery_rate,
                'n_probes': len(sweep_results)
            })
            
            # Check if we need to grow
            if mastery_rate < growth_trigger:
                print(f"  ⚠️ Mastery rate below {growth_trigger:.0%} - triggering growth")
                self.student.grow(reason=f"sweep_{sweep_idx}_mastery_{mastery_rate:.2f}")
                results['growth_events'] += 1
        
        # Final stats
        final_params = sum(p.numel() for p in self.student.parameters())
        
        results['total_probes'] = self.total_probes
        results['regions_mastered'] = self.regions_mastered
        results['initial_params'] = initial_params
        results['final_params'] = final_params
        results['final_dim'] = self.student.current_dim
        results['final_layers'] = len(self.student.layers)
        
        return results
    
    def coherent_training_phase(self, n_batches: int = 200, batch_size: int = 32, seq_len: int = 64):
        """
        Focused training on coherent text only.
        
        After sweep establishes architecture, this trains for generation quality.
        """
        print("\n" + "="*60)
        print("COHERENT TEXT TRAINING PHASE")
        print("="*60)
        
        if self.tokenizer is None or self.oracle is None:
            print("  ⚠️ Tokenizer/oracle not available, skipping")
            return
        
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_batches)
        
        losses = []
        
        for batch_idx in range(n_batches):
            # Generate coherent text from oracle
            tokens, _ = self.probe_gen.coherent_tokens(
                batch_size, seq_len, self.tokenizer, self.oracle
            )
            
            # Get oracle logits
            with torch.no_grad():
                oracle_logits = self.oracle(tokens).logits
            
            # Train student
            student_logits = self.student(tokens)
            loss = self.compute_distillation_loss(student_logits, oracle_logits, tokens)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
            
            if (batch_idx + 1) % 50 == 0:
                avg_loss = sum(losses[-50:]) / 50
                print(f"  Batch {batch_idx+1}/{n_batches}: loss={avg_loss:.4f}")
        
        final_loss = sum(losses[-20:]) / 20
        print(f"\n  Final loss: {final_loss:.4f}")
    
    def test_generation(self, prompt: str = "The meaning of life is") -> str:
        """Test student's generation ability."""
        if self.tokenizer is None:
            return "[tokenizer not available]"
        
        tokens = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate with student
        with torch.no_grad():
            for _ in range(20):
                logits = self.student(tokens)
                next_token = logits[0, -1].argmax()
                tokens = torch.cat([tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        return self.tokenizer.decode(tokens[0])


def main():
    """Demo: Grow transformers using oracle as loss function."""
    
    print("="*70)
    print("PAC ORACLE DISTILLATION")
    print("="*70)
    print("\n  Oracle = Real Pythia model (loss function)")
    print("  Probes = Parameter sweep (input space coverage)")
    print("  Student = Growing transformer")
    print("  ByRef Links = Mastered regions of input space")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📍 Device: {device}")
    
    # Create distillation system
    distiller = OracleDistillation(
        oracle_name="EleutherAI/pythia-70m",
        initial_dim=64,
        device=device
    )
    
    initial_params = sum(p.numel() for p in distiller.student.parameters())
    print(f"\n📊 Student initial: {initial_params:,} params, "
          f"{distiller.student.current_dim} dim, "
          f"{len(distiller.student.layers)} layers")
    
    # Run sweep-based training
    results = distiller.sweep_and_grow(
        n_sweeps=10,
        probes_per_sweep=15,
        mastery_threshold=8.0,  # Tighter threshold to force more growth
        growth_trigger=0.5  # Grow if < 50% mastery
    )
    
    # Focused coherent training for generation quality
    distiller.coherent_training_phase(n_batches=500, batch_size=32, seq_len=64)
    
    # Report
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    print(f"\n📊 RESULTS:")
    print(f"  Total probes: {results['total_probes']}")
    print(f"  Regions mastered: {results['regions_mastered']}")
    print(f"  Growth events: {results['growth_events']}")
    print(f"  Final dim: {results['final_dim']}")
    print(f"  Final layers: {results['final_layers']}")
    print(f"  Params: {results['initial_params']:,} → {results['final_params']:,} "
          f"({results['final_params']/results['initial_params']:.1f}x)")
    
    # Sweep history
    print(f"\n📈 SWEEP HISTORY:")
    for sweep in results['sweeps']:
        bar = "█" * int(sweep['mastery_rate'] * 20)
        print(f"  Sweep {sweep['sweep_idx']+1}: loss={sweep['avg_loss']:.3f}, "
              f"mastery={sweep['mastery_rate']:.0%} {bar}")
    
    # Mastered regions (ByRef links)
    print(f"\n🔗 BYREF LINKS (mastered regions):")
    by_type = {}
    for region in distiller.student.mastered_regions:
        by_type[region.probe_type] = by_type.get(region.probe_type, 0) + 1
    
    for ptype, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {ptype}: {count} regions")
    
    # Test generation
    print("\n" + "="*60)
    print("GENERATION TEST")
    print("="*60)
    
    prompts = [
        "The meaning of life is",
        "In a world where",
        "Scientists have discovered"
    ]
    
    for prompt in prompts:
        print(f"\n  Prompt: \"{prompt}\"")
        output = distiller.test_generation(prompt)
        print(f"  Student: \"{output}\"")
        
        # Also get oracle output for comparison
        if distiller.oracle is not None:
            tokens = distiller.tokenizer.encode(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                for _ in range(20):
                    logits = distiller.oracle(tokens).logits
                    next_token = logits[0, -1].argmax()
                    tokens = torch.cat([tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            oracle_out = distiller.tokenizer.decode(tokens[0])
            print(f"  Oracle:  \"{oracle_out}\"")
    
    print("\n" + "="*70)
    print("✅ ORACLE DISTILLATION COMPLETE")
    print("="*70)
    
    print("""
💡 KEY INSIGHTS:
  - Oracle provides GROUND TRUTH behavior (not approximations)
  - Parameter sweep covers diverse input space regions
  - Growth triggered by failure to master regions
  - ByRef links = which input regions are understood
  - Student learns the TRANSFORMATION, not just the weights

📐 PAC ORACLE DISTILLATION ARCHITECTURE:
  ┌──────────────────────────────────────────────────────┐
  │  ORACLE (Pythia-70M)                                 │
  │  ────────────────────                                │
  │  • Real model = ground truth                         │
  │  • Full behavior, not extracted templates            │
  │  • Emergent properties preserved                     │
  └────────────────────┬─────────────────────────────────┘
                       │ logits
                       ▼
  ┌──────────────────────────────────────────────────────┐
  │  PROBE GENERATOR                                     │
  │  ───────────────────                                 │
  │  • Random tokens (broad coverage)                    │
  │  • Structured patterns (specific behaviors)          │
  │  • Coherent text (generation quality)                │
  │  • Edge cases (rare tokens)                          │
  │  • Interpolations (smooth transitions)               │
  └────────────────────┬─────────────────────────────────┘
                       │ diverse inputs
                       ▼
  ┌──────────────────────────────────────────────────────┐
  │  GROWING STUDENT                                     │
  │  ───────────────────                                 │
  │  • Starts small (64 dim, 1 layer)                    │
  │  • Grows when mastery < threshold                    │
  │  • Each layer can specialize                         │
  │  • Records mastered regions as ByRef links           │
  └──────────────────────────────────────────────────────┘

🔗 BYREF LINKS = Mastered Input Regions:
  • "random_42" → Layer 2 mastered this input distribution
  • "coherent_17" → Layer 3 learned this text pattern  
  • Skills = which transformations the student has learned
""")


if __name__ == "__main__":
    main()
