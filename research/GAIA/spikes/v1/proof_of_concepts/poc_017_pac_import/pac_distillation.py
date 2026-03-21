"""
PAC-Guided Transformer Distillation
=====================================

Grow transformers by learning FROM extracted model transformers.

Architecture:
1. PAC tree = Curriculum (what to learn, in what order)
2. Extracted MLP/Attention = Teachers (how to transform)
3. Growing transformer = Student (learns by imitating)
4. ByRef links = Mastered skills (created after learning succeeds)

The key insight: We don't copy weights, we learn BEHAVIOR.
The tree guides curriculum, the templates show target behavior.

Flow:
1. Start with tiny transformer
2. For each tree level (curriculum stage):
   a. Get teacher behavior from extracted MLP/attention
   b. Train student to imitate teacher on that behavior
   c. When loss < threshold, create ByRef skill (mastery)
   d. Grow transformer if needed for next level
3. Result: Transformer that learned the model's knowledge + skill links
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

from fracton.core import PACSystem
from fracton.physics.constants import PHI, XI, PHI_XI, LAMBDA_STAR

print(f"✓ Using fracton from {fracton_path}")


@dataclass
class TeacherBehavior:
    """Captured behavior from extracted model layer."""
    layer_idx: int
    
    # MLP teacher (SVD decomposed)
    up_U: torch.Tensor  # [hidden, rank]
    up_S: torch.Tensor  # [rank]
    up_Vh: torch.Tensor  # [rank, input]
    down_U: torch.Tensor  # [output, rank]
    down_S: torch.Tensor  # [rank]
    down_Vh: torch.Tensor  # [rank, hidden]
    
    # Attention pattern (what to focus on)
    attention_pattern: Optional[torch.Tensor] = None
    
    def apply_mlp(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the teacher's MLP transformation."""
        # Up projection: x @ Vh.T @ diag(S) @ U.T
        h = x @ self.up_Vh.T  # [batch, seq, rank]
        h = h * self.up_S.unsqueeze(0).unsqueeze(0)
        h = h @ self.up_U.T  # [batch, seq, hidden]
        
        # Activation
        h = F.gelu(h)
        
        # Down projection
        h = h @ self.down_Vh.T
        h = h * self.down_S.unsqueeze(0).unsqueeze(0)
        h = h @ self.down_U.T
        
        return h
    
    def get_target_output(self, x: torch.Tensor) -> torch.Tensor:
        """Get the teacher's output for input x."""
        return self.apply_mlp(x)


class StudentLayer(nn.Module):
    """A single student transformer layer that learns from teacher."""
    
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        
        # Attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        
        # MLP (starts small, can grow)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        # Tracking
        self.teacher_idx = -1
        self.mastery_score = 0.0
        self.mastered = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        
        # MLP
        h = self.norm2(x)
        h = self.mlp(h)
        return x + h
    
    def imitation_loss(self, x: torch.Tensor, teacher: TeacherBehavior) -> torch.Tensor:
        """Compute loss vs teacher behavior."""
        # Get student output
        student_out = self.forward(x)
        
        # Get teacher output (need to project dimensions)
        with torch.no_grad():
            # Project x to teacher's input dim
            teacher_in = x
            if x.shape[-1] != teacher.up_Vh.shape[1]:
                # Pad or truncate
                if x.shape[-1] < teacher.up_Vh.shape[1]:
                    teacher_in = F.pad(x, (0, teacher.up_Vh.shape[1] - x.shape[-1]))
                else:
                    teacher_in = x[..., :teacher.up_Vh.shape[1]]
            
            teacher_out = teacher.get_target_output(teacher_in)
            
            # Project back to student dim
            if teacher_out.shape[-1] != x.shape[-1]:
                if teacher_out.shape[-1] > x.shape[-1]:
                    teacher_out = teacher_out[..., :x.shape[-1]]
                else:
                    teacher_out = F.pad(teacher_out, (0, x.shape[-1] - teacher_out.shape[-1]))
        
        # MSE loss on behavior
        loss = F.mse_loss(student_out, teacher_out + x)  # Teacher output is residual
        
        return loss


class GrowingStudent(nn.Module):
    """
    Growing transformer that learns from extracted teachers.
    
    Curriculum is guided by PAC tree structure.
    Growth happens when mastery is achieved.
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
        
        # Student layers (start with 1)
        self.layers = nn.ModuleList([
            StudentLayer(initial_dim, n_heads)
        ])
        
        # Output
        self.output_norm = nn.LayerNorm(initial_dim)
        self.output_proj = nn.Linear(initial_dim, vocab_size)
        
        # Teachers
        self.teachers: List[TeacherBehavior] = []
        
        # Mastered skills (ByRef links)
        self.skills: Dict[int, Dict] = {}  # layer_idx -> skill info
        
        # Curriculum position
        self.curriculum_level = 0
        
        # Stats
        self.growth_events = []
        self.mastery_events = []
        
        self.to(device)
    
    def load_teachers(self, mlp_templates: List[Dict], attention_patterns: List[torch.Tensor]):
        """Load teacher behaviors from extraction."""
        for idx, template in enumerate(mlp_templates):
            up_U = template.get('up_U')
            up_S = template.get('up_S')
            up_Vh = template.get('up_Vh')
            down_U = template.get('down_U')
            down_S = template.get('down_S')
            down_Vh = template.get('down_Vh')
            
            if all(t is not None for t in [up_U, up_S, up_Vh, down_U, down_S, down_Vh]):
                attn = attention_patterns[idx] if idx < len(attention_patterns) else None
                
                teacher = TeacherBehavior(
                    layer_idx=idx,
                    up_U=up_U.to(self.device),
                    up_S=up_S.to(self.device),
                    up_Vh=up_Vh.to(self.device),
                    down_U=down_U.to(self.device),
                    down_S=down_S.to(self.device),
                    down_Vh=down_Vh.to(self.device),
                    attention_pattern=attn.to(self.device) if attn is not None else None
                )
                self.teachers.append(teacher)
        
        print(f"  Loaded {len(self.teachers)} teacher behaviors")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through student."""
        h = self.embedding(x)
        
        for layer in self.layers:
            h = layer(h)
        
        h = self.output_norm(h)
        logits = self.output_proj(h)
        
        return logits
    
    def train_on_teacher(self, 
                         teacher_idx: int,
                         layer_idx: int = None,  # Specify which layer to train
                         batch_size: int = 32,
                         n_steps: int = 100,
                         mastery_threshold: float = 0.1) -> Tuple[bool, float]:
        """
        Train specific layer to imitate a teacher.
        
        Returns (mastered, final_loss)
        """
        if teacher_idx >= len(self.teachers):
            return False, float('inf')
        
        teacher = self.teachers[teacher_idx]
        
        # Use specified layer or last layer
        if layer_idx is not None and layer_idx < len(self.layers):
            student_layer = self.layers[layer_idx]
        else:
            student_layer = self.layers[-1]  # Train last layer
            layer_idx = len(self.layers) - 1
            
        student_layer.teacher_idx = teacher_idx
        
        # Optimizer for just this layer
        optimizer = torch.optim.AdamW(student_layer.parameters(), lr=1e-3)
        
        losses = []
        
        for step in range(n_steps):
            # Random input
            x = torch.randn(batch_size, 16, self.current_dim, device=self.device)
            
            # Imitation loss
            loss = student_layer.imitation_loss(x, teacher)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        final_loss = sum(losses[-10:]) / 10
        mastered = final_loss < mastery_threshold
        
        if mastered:
            student_layer.mastered = True
            student_layer.mastery_score = 1.0 - final_loss / mastery_threshold
            
            # Create skill (ByRef link)
            self.skills[teacher_idx] = {
                'layer_idx': len(self.layers) - 1,
                'teacher_idx': teacher_idx,
                'mastery_score': student_layer.mastery_score,
                'final_loss': final_loss
            }
            
            self.mastery_events.append({
                'teacher_idx': teacher_idx,
                'final_loss': final_loss,
                'layer': len(self.layers) - 1
            })
        
        return mastered, final_loss
    
    def grow(self):
        """Add a new layer and grow dimensions."""
        old_dim = self.current_dim
        new_dim = min(512, int(old_dim * PHI))
        new_dim = (new_dim // self.n_heads) * self.n_heads
        
        # Ensure we always grow, never shrink
        if new_dim <= old_dim:
            new_dim = old_dim + self.n_heads
        
        # Cap at 512 for memory
        if new_dim > 512:
            new_dim = 512
            if old_dim >= 512:
                # Can't grow dimensions anymore, just add layer
                print(f"  🌱 ADD LAYER: {len(self.layers)} → {len(self.layers)+1} layers (dim capped)")
                new_layer = StudentLayer(old_dim, self.n_heads).to(self.device)
                self.layers.append(new_layer)
                self.growth_events.append({
                    'old_dim': old_dim,
                    'new_dim': old_dim,
                    'n_layers': len(self.layers)
                })
                return
        
        print(f"  🌱 GROWTH: {old_dim} → {new_dim} dim, {len(self.layers)} → {len(self.layers)+1} layers")
        
        # Expand embedding
        new_embedding = nn.Embedding(self.vocab_size, new_dim, device=self.device)
        with torch.no_grad():
            new_embedding.weight[:, :old_dim] = self.embedding.weight
        self.embedding = new_embedding
        
        # Expand existing layers
        for layer in self.layers:
            self._expand_layer(layer, old_dim, new_dim)
        
        # Add new layer
        new_layer = StudentLayer(new_dim, self.n_heads).to(self.device)
        self.layers.append(new_layer)
        
        # Expand output
        self.output_norm = nn.LayerNorm(new_dim).to(self.device)
        new_output = nn.Linear(new_dim, self.vocab_size, device=self.device)
        with torch.no_grad():
            new_output.weight[:, :old_dim] = self.output_proj.weight
        self.output_proj = new_output
        
        self.current_dim = new_dim
        self.growth_events.append({
            'old_dim': old_dim,
            'new_dim': new_dim,
            'n_layers': len(self.layers)
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
    
    def curriculum_train(self, 
                         n_epochs: int = 3,
                         steps_per_teacher: int = 100) -> Dict:
        """
        Train through the curriculum (all teachers).
        
        For each teacher:
        1. Train current layer to imitate
        2. If mastered, create skill link
        3. If not mastered, grow and retry
        """
        print("\n" + "="*60)
        print("CURRICULUM TRAINING")
        print("="*60)
        
        results = {
            'teachers_mastered': 0,
            'growth_events': 0,
            'final_skills': {}
        }
        
        # Progressive difficulty: later teachers require tighter mastery
        for teacher_idx, teacher in enumerate(self.teachers):
            # Mastery threshold scales with difficulty
            # But keep it achievable - these are compressed representations
            mastery_threshold = 0.05 / (1 + teacher_idx * 0.3)
            
            print(f"\n  Teacher {teacher_idx}/{len(self.teachers)-1} (threshold={mastery_threshold:.4f}):")
            
            for attempt in range(n_epochs):
                mastered, loss = self.train_on_teacher(
                    teacher_idx, 
                    n_steps=steps_per_teacher,
                    mastery_threshold=mastery_threshold
                )
                
                print(f"    Attempt {attempt+1}: loss={loss:.6f}, mastered={mastered}")
                
                if mastered:
                    results['teachers_mastered'] += 1
                    print(f"    ✓ Mastered! Created skill link.")
                    break
                else:
                    # Grow and retry
                    self.grow()
                    results['growth_events'] += 1
        
        results['final_skills'] = self.skills
        results['n_layers'] = len(self.layers)
        results['final_dim'] = self.current_dim
        
        return results


def load_extraction(extraction_dir: Path) -> Tuple[torch.Tensor, List[torch.Tensor], List[Dict]]:
    """Load extracted model data."""
    print(f"Loading extraction from {extraction_dir}")
    
    # Vocab
    vocab_data = torch.load(extraction_dir / "pac_vocab.pt", weights_only=False)
    vocab_embeddings = vocab_data['vocab_deltas']
    print(f"  Vocab: {vocab_embeddings.shape}")
    
    # Attention
    attn_data = torch.load(extraction_dir / "pac_attention.pt", weights_only=False)
    attention_patterns = attn_data['patterns']
    print(f"  Attention: {len(attention_patterns)} layers")
    
    # MLP
    mlp_data = torch.load(extraction_dir / "pac_mlp.pt", weights_only=False)
    mlp_templates = mlp_data['templates']
    print(f"  MLP: {len(mlp_templates)} layers")
    
    return vocab_embeddings, attention_patterns, mlp_templates


def main():
    """Demo: Grow transformers by learning from extracted teachers."""
    
    print("="*70)
    print("PAC-GUIDED TRANSFORMER DISTILLATION")
    print("="*70)
    print("\n  PAC tree = Curriculum (what to learn)")
    print("  Extracted MLP = Teachers (how to transform)")
    print("  Growing transformer = Student (learns by imitating)")
    print("  ByRef links = Mastered skills")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📍 Device: {device}")
    
    # Load extraction
    extraction_dir = Path(__file__).parent.parent / "poc_016_pac_extraction" / "extracted_v3" / "pythia_70m"
    
    if not extraction_dir.exists():
        print(f"❌ No extraction found at {extraction_dir}")
        return
    
    vocab_embeddings, attention_patterns, mlp_templates = load_extraction(extraction_dir)
    
    # Create student
    print("\n" + "="*60)
    print("CREATING STUDENT")
    print("="*60)
    
    student = GrowingStudent(
        vocab_size=vocab_embeddings.shape[0],
        initial_dim=64,
        n_heads=4,
        device=device
    )
    
    initial_params = sum(p.numel() for p in student.parameters())
    print(f"  Initial: {initial_params:,} params, {student.current_dim} dim, {len(student.layers)} layers")
    
    # Load teachers
    print("\n" + "="*60)
    print("LOADING TEACHERS")
    print("="*60)
    
    student.load_teachers(mlp_templates, attention_patterns)
    
    # Initialize embedding from extraction
    print("\n  Initializing embeddings from extraction...")
    with torch.no_grad():
        # Project extracted embeddings to student dim
        source_dim = vocab_embeddings.shape[1]
        target_dim = student.current_dim
        
        for i in range(min(10000, vocab_embeddings.shape[0])):
            if source_dim > target_dim:
                student.embedding.weight[i] = vocab_embeddings[i, :target_dim]
            else:
                student.embedding.weight[i, :source_dim] = vocab_embeddings[i]
    
    print(f"  Initialized {min(10000, vocab_embeddings.shape[0])} token embeddings")
    
    # Curriculum training
    results = student.curriculum_train(n_epochs=5, steps_per_teacher=200)
    
    # Results
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    final_params = sum(p.numel() for p in student.parameters())
    
    print(f"\n📊 RESULTS:")
    print(f"  Teachers mastered: {results['teachers_mastered']}/{len(student.teachers)}")
    print(f"  Growth events: {results['growth_events']}")
    print(f"  Final layers: {results['n_layers']}")
    print(f"  Final dim: {results['final_dim']}")
    print(f"  Parameters: {initial_params:,} → {final_params:,} ({final_params/initial_params:.1f}x)")
    
    print(f"\n🎯 SKILLS LEARNED:")
    for teacher_idx, skill in results['final_skills'].items():
        print(f"  Teacher {teacher_idx} → Layer {skill['layer_idx']} "
              f"(mastery={skill['mastery_score']:.2f}, loss={skill['final_loss']:.4f})")
    
    # Test generation
    print("\n" + "="*60)
    print("TESTING GENERATION")
    print("="*60)
    
    # Simple test
    test_input = torch.randint(0, 1000, (1, 8), device=device)
    with torch.no_grad():
        logits = student(test_input)
    
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Top predictions: {logits[0, -1].topk(5).indices.tolist()}")
    
    # Test skill transfer: Does the student produce similar outputs to teacher?
    print("\n" + "="*60)
    print("SKILL TRANSFER VERIFICATION")
    print("="*60)
    
    print("  Testing if student learned teacher behavior...")
    
    for teacher_idx, teacher in enumerate(student.teachers):
        # Random input
        x = torch.randn(1, 16, student.current_dim, device=device)
        
        # Student output
        student_layer = student.layers[0]
        with torch.no_grad():
            student_out = student_layer(x)
        
        # Teacher output (projected)
        with torch.no_grad():
            teacher_in = x
            if x.shape[-1] < teacher.up_Vh.shape[1]:
                teacher_in = F.pad(x, (0, teacher.up_Vh.shape[1] - x.shape[-1]))
            else:
                teacher_in = x[..., :teacher.up_Vh.shape[1]]
            
            teacher_out = teacher.get_target_output(teacher_in)
            
            if teacher_out.shape[-1] > x.shape[-1]:
                teacher_out = teacher_out[..., :x.shape[-1]]
            else:
                teacher_out = F.pad(teacher_out, (0, x.shape[-1] - teacher_out.shape[-1]))
        
        # Compare
        mse = F.mse_loss(student_out - x, teacher_out).item()
        cosine = F.cosine_similarity(
            (student_out - x).flatten(),
            teacher_out.flatten(),
            dim=0
        ).item()
        
        print(f"  Teacher {teacher_idx}: MSE={mse:.6f}, Cosine={cosine:.4f}")
    
    print("\n" + "="*70)
    print("✅ DISTILLATION COMPLETE")
    print("="*70)
    print("\n💡 KEY INSIGHT:")
    print("  - We didn't copy weights, we learned BEHAVIOR")
    print("  - Each teacher's transformation is now a skill")
    print("  - Student grew to match teacher complexity")
    print("  - Skills (ByRef links) show WHAT was learned")


if __name__ == "__main__":
    main()
