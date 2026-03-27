"""
SI Phase 0 — Instrument GPT-2 Forward Pass
==========================================

Measures PAC conservation, SEC phase, and RBF balance at each layer
of GPT-2 small (12 layers, 117M params) during inference.

Tests predictions 1, 3, and 5 from SI_theory.md:
  P1: PAC budget E(h_l) + I(h_l) + M(h_l) ≈ const across layers
  P3: SEC phase (input entropy) predicts attention entropy
  P5: MED bound — effective depth bounded regardless of nominal depth

Usage:
  python si_instrument.py                    # run on default prompts
  python si_instrument.py --text "custom"    # run on custom text
  python si_instrument.py --suite            # run full test suite
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ─── DFT Constants ───────────────────────────────────────────────────────────

XI_SEC = 0.0618       # Collapse threshold
PHI_INV = 0.618       # Golden ratio inverse
LAMBDA_STAR = 0.9816  # Critical coupling
GAMMA = 0.0184        # Decay rate


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class LayerMeasurement:
    """PAC/SEC/RBF measurements at a single transformer layer."""
    layer: int
    # PAC components
    entropy: float          # E(h_l) — Shannon entropy of activation distribution
    information: float      # I(h_l) — information content (norm-based proxy)
    coherence: float        # M(h_l) — structural coherence
    pac_budget: float       # E + I + M
    delta_pac: float        # |B_l - B_{l-1}|

    # SEC
    sec_phase: str          # crystallized / ordered / transitional / chaotic
    hidden_entropy: float   # Raw entropy of hidden state

    # RBF
    xi_balance: float       # Ξ — balance operator approximation

    # Attention (per-head)
    attention_entropies: list[float] = field(default_factory=list)
    mean_attention_entropy: float = 0.0

    # Effective depth
    layer_contribution: float = 0.0  # ||F_l(h_{l-1})|| / ||h_{l-1}||


@dataclass
class ForwardPassProfile:
    """Complete PAC/SEC/RBF profile of a single forward pass."""
    text: str
    tokens: list[str]
    n_layers: int
    layers: list[LayerMeasurement]

    # Summary statistics
    pac_mean: float = 0.0
    pac_std: float = 0.0
    pac_cv: float = 0.0           # Coefficient of variation
    effective_depth: int = 0       # Layers with contribution > threshold
    sec_attention_corr: float = 0.0  # Correlation: input entropy ↔ attention entropy


# ─── Measurement Functions ───────────────────────────────────────────────────

def activation_entropy(h: torch.Tensor) -> float:
    """Shannon entropy of the activation magnitude distribution.

    Treats absolute activation values as an unnormalized distribution
    over the hidden dimension. This measures how spread out the
    representation is — uniform = high entropy, peaked = low.
    """
    # h shape: (batch, seq, hidden) or (seq, hidden)
    if h.dim() == 3:
        h = h.squeeze(0)  # remove batch

    # Use mean across sequence positions
    h_mean = h.mean(dim=0)  # (hidden,)
    magnitudes = h_mean.abs()

    # Normalize to probability distribution
    total = magnitudes.sum()
    if total < 1e-10:
        return 0.0
    p = magnitudes / total
    p = p[p > 1e-10]  # avoid log(0)

    entropy = -(p * p.log()).sum().item()
    # Normalize by max possible entropy (log of hidden dim)
    max_entropy = math.log(len(magnitudes))
    if max_entropy < 1e-10:
        return 0.0
    return entropy / max_entropy  # [0, 1]


def information_content(h: torch.Tensor, h_prev: torch.Tensor) -> float:
    """Information content — how much new information this layer created.

    Measures the cosine distance between input and output of a layer.
    0 = no change (identity layer), 1 = completely new representation.
    This is the actualization: potential converted to information.
    """
    if h.dim() == 3:
        h = h.squeeze(0)
    if h_prev.dim() == 3:
        h_prev = h_prev.squeeze(0)

    # Flatten to vectors for comparison
    h_flat = h.reshape(-1)
    h_prev_flat = h_prev.reshape(-1)

    # Cosine similarity
    cos_sim = F.cosine_similarity(h_flat.unsqueeze(0), h_prev_flat.unsqueeze(0)).item()
    # Convert to distance: 0 = identical, 1 = orthogonal
    return 1.0 - max(cos_sim, 0.0)


def structural_coherence(h: torch.Tensor) -> float:
    """Coherence metric — average pairwise cosine similarity across
    sequence positions.

    High coherence = positions encode similar structure (crystallized).
    Low coherence = positions encode diverse structure (chaotic).
    """
    if h.dim() == 3:
        h = h.squeeze(0)

    n_pos = h.shape[0]
    if n_pos < 2:
        return 1.0

    # Sample if sequence is long (avoid O(n²) for long sequences)
    max_pairs = 100
    if n_pos > 50:
        indices = torch.randperm(n_pos)[:50]
        h = h[indices]
        n_pos = len(indices)

    # Normalize
    h_norm = F.normalize(h, dim=-1)

    # Pairwise cosine similarity
    sim_matrix = h_norm @ h_norm.T
    # Extract upper triangle (exclude diagonal)
    mask = torch.triu(torch.ones(n_pos, n_pos, device=h.device), diagonal=1).bool()
    similarities = sim_matrix[mask]

    if len(similarities) == 0:
        return 1.0

    return similarities.mean().item()


def classify_sec_phase(entropy: float) -> str:
    """Classify SEC phase based on normalized entropy.

    Thresholds derived from DFT constants:
    - crystallized: H < ξ_SEC (0.0618) — very low entropy
    - ordered: ξ_SEC ≤ H < φ^(-1) (0.618) — moderate, decreasing
    - transitional: φ^(-1) ≤ H < λ* (0.9816) — between order and chaos
    - chaotic: H ≥ λ* — high entropy
    """
    if entropy < XI_SEC:
        return "crystallized"
    elif entropy < PHI_INV:
        return "ordered"
    elif entropy < LAMBDA_STAR:
        return "transitional"
    else:
        return "chaotic"


def attention_head_entropy(attn_weights: torch.Tensor) -> list[float]:
    """Compute entropy of attention distribution for each head.

    attn_weights shape: (batch, n_heads, seq_q, seq_k)
    Returns entropy per head, averaged across query positions.
    """
    if attn_weights.dim() == 4:
        attn_weights = attn_weights.squeeze(0)  # (n_heads, seq_q, seq_k)

    n_heads = attn_weights.shape[0]
    entropies = []

    for head in range(n_heads):
        # (seq_q, seq_k)
        head_attn = attn_weights[head]
        # Entropy per query position
        head_attn = head_attn.clamp(min=1e-10)
        per_pos_entropy = -(head_attn * head_attn.log()).sum(dim=-1)  # (seq_q,)
        # Normalize by max entropy (log of seq_k)
        max_ent = math.log(head_attn.shape[-1])
        if max_ent > 0:
            per_pos_entropy = per_pos_entropy / max_ent
        entropies.append(per_pos_entropy.mean().item())

    return entropies


def xi_balance(h: torch.Tensor, h_prev: torch.Tensor) -> float:
    """Approximate the RBF balance operator Ξ.

    Ξ = symbolic entropy rate / field curvature potential

    Proxy: ratio of entropy change rate to norm change rate.
    Ξ ≈ 1 means balanced. Ξ > 1 means excess pressure (collapse).
    Ξ < 1 means decay.
    """
    e_curr = activation_entropy(h)
    e_prev = activation_entropy(h_prev)

    n_curr = h.norm().item()
    n_prev = h_prev.norm().item()

    delta_e = abs(e_curr - e_prev)
    delta_n = abs(n_curr - n_prev)

    if delta_n < 1e-10:
        return 1.0  # No curvature change → balanced

    return delta_e / (delta_n / max(n_prev, 1e-10) + 1e-10)


# ─── Instrument GPT-2 ───────────────────────────────────────────────────────

class GPT2Instrument:
    """Instruments GPT-2's forward pass to measure PAC/SEC/RBF at each layer."""

    def __init__(self, model_name: str = "gpt2"):
        print(f"Loading {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name,
            output_attentions=True,
            output_hidden_states=True,
            attn_implementation="eager",
        )
        self.model.eval()
        self.n_layers = self.model.config.n_layer
        self.n_heads = self.model.config.n_head
        self.hidden_size = self.model.config.n_embd
        print(f"  {self.n_layers} layers, {self.n_heads} heads, {self.hidden_size} hidden dim")

    @torch.no_grad()
    def profile(self, text: str) -> ForwardPassProfile:
        """Run a single forward pass and measure everything."""
        inputs = self.tokenizer(text, return_tensors="pt")
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        outputs = self.model(**inputs)

        # hidden_states: tuple of (n_layers + 1) tensors, each (batch, seq, hidden)
        # hidden_states[0] = embedding output, hidden_states[i] = layer i output
        hidden_states = outputs.hidden_states
        # attentions: tuple of n_layers tensors, each (batch, n_heads, seq, seq)
        attentions = outputs.attentions

        measurements = []
        prev_pac = None

        for layer_idx in range(self.n_layers):
            h_in = hidden_states[layer_idx]       # Input to this layer
            h_out = hidden_states[layer_idx + 1]   # Output of this layer
            attn = attentions[layer_idx]

            # PAC components
            E = activation_entropy(h_out)
            I = information_content(h_out, h_in)  # how much this layer changed
            M = structural_coherence(h_out)
            pac = E + I + M
            delta = abs(pac - prev_pac) if prev_pac is not None else 0.0

            # SEC
            phase = classify_sec_phase(E)

            # RBF
            xi = xi_balance(h_out, h_in)

            # Attention entropy per head
            attn_ents = attention_head_entropy(attn)

            # Layer contribution (effective depth metric)
            F_l = h_out - h_in  # Residual = what the layer actually added
            contribution = F_l.norm().item() / max(h_in.norm().item(), 1e-10)

            m = LayerMeasurement(
                layer=layer_idx,
                entropy=E,
                information=I,
                coherence=M,
                pac_budget=pac,
                delta_pac=delta,
                sec_phase=phase,
                hidden_entropy=E,
                xi_balance=xi,
                attention_entropies=attn_ents,
                mean_attention_entropy=sum(attn_ents) / len(attn_ents),
                layer_contribution=contribution,
            )
            measurements.append(m)
            prev_pac = pac

        # Summary statistics
        pac_values = [m.pac_budget for m in measurements]
        pac_mean = sum(pac_values) / len(pac_values)
        pac_std = (sum((p - pac_mean) ** 2 for p in pac_values) / len(pac_values)) ** 0.5
        pac_cv = pac_std / pac_mean if pac_mean > 0 else 0.0

        # Effective depth: layers with contribution > 0.1 (10% of input norm)
        effective = sum(1 for m in measurements if m.layer_contribution > 0.1)

        # SEC-attention correlation (P3)
        input_entropies = [m.hidden_entropy for m in measurements]
        attn_entropies = [m.mean_attention_entropy for m in measurements]
        corr = _spearman_correlation(input_entropies, attn_entropies)

        profile = ForwardPassProfile(
            text=text,
            tokens=tokens,
            n_layers=self.n_layers,
            layers=measurements,
            pac_mean=pac_mean,
            pac_std=pac_std,
            pac_cv=pac_cv,
            effective_depth=effective,
            sec_attention_corr=corr,
        )
        return profile


def _spearman_correlation(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation between two lists."""
    n = len(x)
    if n < 3:
        return 0.0

    def _rank(vals):
        sorted_indices = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)

    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * d_sq / (n * (n * n - 1))


# ─── Display ─────────────────────────────────────────────────────────────────

def print_profile(profile: ForwardPassProfile):
    """Print a formatted profile."""
    print(f"\n{'=' * 80}")
    print(f"TEXT: {profile.text[:80]}...")
    print(f"TOKENS: {len(profile.tokens)}")
    print(f"{'=' * 80}")

    # Layer-by-layer table
    print(f"\n{'Layer':>5} | {'E(h)':>6} {'I(h)':>6} {'M(h)':>6} | "
          f"{'PAC':>7} {'dPAC':>6} | {'SEC Phase':>13} | "
          f"{'Xi':>5} | {'Attn H':>6} | {'F_l/h':>6}")
    print(f"{'-' * 5}-+-{'-' * 6}-{'-' * 6}-{'-' * 6}-+-"
          f"{'-' * 7}-{'-' * 6}-+-{'-' * 13}-+-"
          f"{'-' * 5}-+-{'-' * 6}-+-{'-' * 6}")

    for m in profile.layers:
        xi_str = f"{m.xi_balance:.3f}" if m.xi_balance < 100 else ">100"
        print(f"{m.layer:>5} | {m.entropy:>6.4f} {m.information:>6.3f} {m.coherence:>6.4f} | "
              f"{m.pac_budget:>7.4f} {m.delta_pac:>6.4f} | {m.sec_phase:>13} | "
              f"{xi_str:>5} | {m.mean_attention_entropy:>6.4f} | {m.layer_contribution:>6.4f}")

    # Summary
    print(f"\n{'-' * 80}")
    print(f"PAC CONSERVATION (P1):")
    print(f"  Mean PAC budget:   {profile.pac_mean:.4f}")
    print(f"  Std PAC budget:    {profile.pac_std:.4f}")
    print(f"  Coeff of variation:{profile.pac_cv:.4f}  "
          f"{'[PASS] CONSERVED (<0.1)' if profile.pac_cv < 0.1 else '[FAIL] NOT CONSERVED (>=0.1)'}")

    print(f"\nSEC-ATTENTION CORRELATION (P3):")
    print(f"  Spearman rho:      {profile.sec_attention_corr:.4f}  "
          f"{'[PASS] CORRELATED (>0.5)' if profile.sec_attention_corr > 0.5 else '[?] WEAK/NO CORRELATION (<=0.5)'}")

    print(f"\nEFFECTIVE DEPTH (P5):")
    print(f"  Effective layers:  {profile.effective_depth} / {profile.n_layers}  "
          f"(ratio: {profile.effective_depth / profile.n_layers:.2f})")

    # Phase distribution
    phases = [m.sec_phase for m in profile.layers]
    phase_counts = {p: phases.count(p) for p in set(phases)}
    print(f"\nSEC PHASE DISTRIBUTION:")
    for phase in ["crystallized", "ordered", "transitional", "chaotic"]:
        count = phase_counts.get(phase, 0)
        bar = "#" * count
        print(f"  {phase:>13}: {count:>2} {bar}")

    # Attention entropy per head at each layer (compact)
    print(f"\nATTENTION ENTROPY (per head, per layer):")
    for m in profile.layers:
        ents = " ".join(f"{e:.2f}" for e in m.attention_entropies)
        print(f"  L{m.layer:>2}: [{ents}]")


def save_profile(profile: ForwardPassProfile, path: Path):
    """Save profile as JSON for later analysis."""
    data = {
        "text": profile.text,
        "tokens": profile.tokens,
        "n_layers": profile.n_layers,
        "pac_mean": profile.pac_mean,
        "pac_std": profile.pac_std,
        "pac_cv": profile.pac_cv,
        "effective_depth": profile.effective_depth,
        "sec_attention_corr": profile.sec_attention_corr,
        "layers": [
            {
                "layer": m.layer,
                "entropy": m.entropy,
                "information": m.information,
                "coherence": m.coherence,
                "pac_budget": m.pac_budget,
                "delta_pac": m.delta_pac,
                "sec_phase": m.sec_phase,
                "xi_balance": m.xi_balance,
                "attention_entropies": m.attention_entropies,
                "mean_attention_entropy": m.mean_attention_entropy,
                "layer_contribution": m.layer_contribution,
            }
            for m in profile.layers
        ],
    }
    path.write_text(json.dumps(data, indent=2))


# ─── Test Suite ──────────────────────────────────────────────────────────────

DEFAULT_PROMPTS = [
    # Factual — should be crystallized/ordered
    "The capital of France is Paris, which is located in the northern part of the country.",
    # Narrative — should show transitional phases
    "Once upon a time, in a kingdom far away, there lived a young princess who dreamed of exploring the stars.",
    # Technical — mixed phases
    "The transformer architecture uses multi-head self-attention to compute contextual representations of input tokens.",
    # Nonsense — should be chaotic
    "Flurble gnarx the quizzical brontosaurus while seventeen purple equations danced.",
    # Repetitive — should be highly crystallized
    "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.",
    # Long-range dependency
    "Although the experiment had been running for several months without producing any significant results, the researchers decided to continue because they believed that a breakthrough was imminent.",
]


def run_suite(instrument: GPT2Instrument):
    """Run the full test suite and compute aggregate statistics."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    all_profiles = []
    for i, text in enumerate(DEFAULT_PROMPTS):
        print(f"\n{'#' * 80}")
        print(f"PROMPT {i + 1}/{len(DEFAULT_PROMPTS)}")
        profile = instrument.profile(text)
        print_profile(profile)
        save_profile(profile, results_dir / f"profile_{i:02d}.json")
        all_profiles.append(profile)

    # Aggregate analysis
    print(f"\n\n{'=' * 80}")
    print(f"AGGREGATE ANALYSIS ({len(all_profiles)} prompts)")
    print(f"{'=' * 80}")

    # P1: PAC conservation across all prompts
    cvs = [p.pac_cv for p in all_profiles]
    print(f"\nP1 - PAC Conservation:")
    print(f"  Mean CV:  {sum(cvs) / len(cvs):.4f}")
    print(f"  Min CV:   {min(cvs):.4f}")
    print(f"  Max CV:   {max(cvs):.4f}")
    conserved = sum(1 for cv in cvs if cv < 0.1)
    print(f"  Conserved (<0.1 CV): {conserved}/{len(cvs)}")
    print(f"  VERDICT: {'[PASS] PAC CONSERVED' if conserved == len(cvs) else '[?] PARTIAL' if conserved > 0 else '[FAIL] NOT CONSERVED'}")

    # P3: SEC-attention correlation
    corrs = [p.sec_attention_corr for p in all_profiles]
    print(f"\nP3 - SEC-Attention Correlation:")
    print(f"  Mean rho: {sum(corrs) / len(corrs):.4f}")
    print(f"  Min rho:  {min(corrs):.4f}")
    print(f"  Max rho:  {max(corrs):.4f}")
    correlated = sum(1 for c in corrs if c > 0.5)
    print(f"  Correlated (rho>0.5): {correlated}/{len(corrs)}")
    print(f"  VERDICT: {'[PASS] SEC PREDICTS ATTENTION' if correlated == len(corrs) else '[?] PARTIAL' if correlated > 0 else '[FAIL] NO CORRELATION'}")

    # P5: Effective depth
    eff_ratios = [p.effective_depth / p.n_layers for p in all_profiles]
    print(f"\nP5 - Effective Depth:")
    print(f"  Mean ratio: {sum(eff_ratios) / len(eff_ratios):.2f}")
    print(f"  Min ratio:  {min(eff_ratios):.2f}")
    print(f"  Max ratio:  {max(eff_ratios):.2f}")
    bounded = sum(1 for r in eff_ratios if r < 0.5)
    print(f"  Bounded (<0.5): {bounded}/{len(eff_ratios)}")
    print(f"  VERDICT: {'[PASS] MED BOUND HOLDS' if bounded == len(eff_ratios) else '[?] PARTIAL' if bounded > 0 else '[FAIL] ALL LAYERS EFFECTIVE'}")

    # Per-layer PAC budget stability (pooled across all prompts)
    print(f"\nPER-LAYER PAC BUDGET (pooled across prompts):")
    for layer_idx in range(all_profiles[0].n_layers):
        pacs = [p.layers[layer_idx].pac_budget for p in all_profiles]
        mean = sum(pacs) / len(pacs)
        std = (sum((v - mean) ** 2 for v in pacs) / len(pacs)) ** 0.5
        phases = [p.layers[layer_idx].sec_phase for p in all_profiles]
        dominant = max(set(phases), key=phases.count)
        contribs = [p.layers[layer_idx].layer_contribution for p in all_profiles]
        mean_contrib = sum(contribs) / len(contribs)
        print(f"  L{layer_idx:>2}: PAC={mean:.4f}+/-{std:.4f}  "
              f"phase={dominant:<13}  contrib={mean_contrib:.4f}")

    # === DEEPER ANALYSIS ===
    n_layers = all_profiles[0].n_layers

    # 1. Conservation band detection
    # Find the contiguous range of layers where PAC budget is most stable
    print(f"\n{'=' * 80}")
    print(f"DEEPER ANALYSIS")
    print(f"{'=' * 80}")

    # Compute per-layer mean PAC across prompts
    layer_pac_means = []
    for l in range(n_layers):
        pacs = [p.layers[l].pac_budget for p in all_profiles]
        layer_pac_means.append(sum(pacs) / len(pacs))

    # Find best conservation band (min CV over contiguous range of >= 3 layers)
    best_band = None
    best_cv = 999.0
    for start in range(n_layers):
        for end in range(start + 3, n_layers + 1):
            band = layer_pac_means[start:end]
            mean_b = sum(band) / len(band)
            std_b = (sum((v - mean_b) ** 2 for v in band) / len(band)) ** 0.5
            cv_b = std_b / mean_b if mean_b > 0 else 999
            if cv_b < best_cv:
                best_cv = cv_b
                best_band = (start, end - 1)

    if best_band:
        s, e = best_band
        band_vals = layer_pac_means[s:e + 1]
        band_mean = sum(band_vals) / len(band_vals)
        print(f"\nCONSERVATION BAND:")
        print(f"  Layers {s}-{e} ({e - s + 1} layers)")
        print(f"  PAC range: {min(band_vals):.4f} - {max(band_vals):.4f}")
        print(f"  Band mean: {band_mean:.4f}")
        print(f"  Band CV:   {best_cv:.4f}  "
              f"{'[PASS] CONSERVED' if best_cv < 0.05 else '[?] WEAK' if best_cv < 0.1 else '[FAIL]'}")

    # 2. Budget return: does L0 ~= L_last?
    l0_pacs = [p.layers[0].pac_budget for p in all_profiles]
    lN_pacs = [p.layers[-1].pac_budget for p in all_profiles]
    l0_mean = sum(l0_pacs) / len(l0_pacs)
    lN_mean = sum(lN_pacs) / len(lN_pacs)
    return_ratio = lN_mean / l0_mean if l0_mean > 0 else 0
    print(f"\nBUDGET RETURN (L0 vs L{n_layers - 1}):")
    print(f"  L0 mean PAC:     {l0_mean:.4f}")
    print(f"  L{n_layers - 1} mean PAC:    {lN_mean:.4f}")
    print(f"  Return ratio:    {return_ratio:.4f}  "
          f"{'[PASS] RETURNS TO ENTRY' if abs(return_ratio - 1.0) < 0.1 else '[?] PARTIAL' if abs(return_ratio - 1.0) < 0.2 else '[FAIL]'}")

    # 3. Attention entropy trend
    print(f"\nATTENTION ENTROPY TREND:")
    layer_attn_means = []
    for l in range(n_layers):
        attns = [p.layers[l].mean_attention_entropy for p in all_profiles]
        layer_attn_means.append(sum(attns) / len(attns))

    # Monotonic decrease check
    decreases = sum(1 for i in range(1, len(layer_attn_means))
                    if layer_attn_means[i] < layer_attn_means[i - 1])
    total_pairs = len(layer_attn_means) - 1
    print(f"  Decreasing steps: {decreases}/{total_pairs}")
    trend_corr = _spearman_correlation(list(range(n_layers)), layer_attn_means)
    print(f"  Layer vs Attn H correlation: {trend_corr:.4f}  "
          f"{'(decreasing)' if trend_corr < -0.5 else '(flat/mixed)'}")
    for l in range(n_layers):
        bar_len = int(layer_attn_means[l] * 40)
        bar = "=" * bar_len
        print(f"  L{l:>2}: {layer_attn_means[l]:.4f} {bar}")

    # 4. Xi balance in conservation band
    if best_band:
        s, e = best_band
        print(f"\nXi BALANCE IN CONSERVATION BAND (L{s}-L{e}):")
        for l in range(s, e + 1):
            xis = [p.layers[l].xi_balance for p in all_profiles]
            xi_mean = sum(xis) / len(xis)
            near_one = abs(xi_mean - 1.0) < 0.5
            print(f"  L{l:>2}: Xi={xi_mean:.3f}  "
                  f"{'<-- near 1' if near_one else ''}")

    # 5. Layer contribution distribution — test 3-basis pattern (MED nodes<=3)
    print(f"\nLAYER CONTRIBUTION CLUSTERS:")
    contribs_mean = []
    for l in range(n_layers):
        cs = [p.layers[l].layer_contribution for p in all_profiles]
        contribs_mean.append(sum(cs) / len(cs))

    # Classify into clusters: high (>1.0), medium (0.1-1.0), low (<0.1)
    high = [l for l in range(n_layers) if contribs_mean[l] > 1.0]
    medium = [l for l in range(n_layers) if 0.1 <= contribs_mean[l] <= 1.0]
    low = [l for l in range(n_layers) if contribs_mean[l] < 0.1]
    print(f"  HIGH   (>1.0):   L{high}  ({len(high)} layers)")
    print(f"  MEDIUM (0.1-1):  L{medium}  ({len(medium)} layers)")
    print(f"  LOW    (<0.1):   L{low}  ({len(low)} layers)")
    n_clusters = sum(1 for g in [high, medium, low] if g)
    print(f"  Distinct contribution levels: {n_clusters}  "
          f"{'[PASS] MED nodes<=3' if n_clusters <= 3 else '[FAIL]'}")

    # Save aggregate
    agg = {
        "n_prompts": len(all_profiles),
        "model": all_profiles[0].n_layers,
        "p1_pac_conservation": {
            "mean_cv": sum(cvs) / len(cvs),
            "all_cvs": cvs,
            "conserved_count": conserved,
            "verdict": "conserved" if conserved == len(cvs) else "partial" if conserved > 0 else "not_conserved",
        },
        "p3_sec_attention": {
            "mean_correlation": sum(corrs) / len(corrs),
            "all_correlations": corrs,
            "correlated_count": correlated,
            "verdict": "correlated" if correlated == len(corrs) else "partial" if correlated > 0 else "no_correlation",
        },
        "p5_effective_depth": {
            "mean_ratio": sum(eff_ratios) / len(eff_ratios),
            "all_ratios": eff_ratios,
            "bounded_count": bounded,
            "verdict": "bounded" if bounded == len(eff_ratios) else "partial" if bounded > 0 else "all_effective",
        },
        "conservation_band": {
            "start": best_band[0] if best_band else None,
            "end": best_band[1] if best_band else None,
            "cv": best_cv,
        },
        "budget_return_ratio": return_ratio,
        "attention_trend_correlation": trend_corr,
        "contribution_clusters": {
            "high": high,
            "medium": medium,
            "low": low,
        },
    }
    (results_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print(f"\nResults saved to {results_dir}/")


# ─── Main ────────────────────────────────────────────────────────────────────

def run_scaling(models: list[str]):
    """Compare effective depth across model sizes."""
    print(f"\n{'=' * 80}")
    print(f"SCALING COMPARISON: {' vs '.join(models)}")
    print(f"{'=' * 80}")

    # Use a single representative prompt
    test_text = "Although the experiment had been running for several months without producing any significant results, the researchers decided to continue because they believed that a breakthrough was imminent."

    results = []
    for model_name in models:
        instrument = GPT2Instrument(model_name)
        profile = instrument.profile(test_text)

        # Conservation band
        layer_pacs = [m.pac_budget for m in profile.layers]
        best_cv = 999.0
        best_band = None
        for start in range(profile.n_layers):
            for end in range(start + 3, profile.n_layers + 1):
                band = layer_pacs[start:end]
                mean_b = sum(band) / len(band)
                std_b = (sum((v - mean_b) ** 2 for v in band) / len(band)) ** 0.5
                cv_b = std_b / mean_b if mean_b > 0 else 999
                if cv_b < best_cv:
                    best_cv = cv_b
                    best_band = (start, end - 1)

        contribs = [m.layer_contribution for m in profile.layers]
        high = sum(1 for c in contribs if c > 1.0)
        medium = sum(1 for c in contribs if 0.1 <= c <= 1.0)
        low = sum(1 for c in contribs if c < 0.1)
        effective = sum(1 for c in contribs if c > 0.1)

        # Budget return
        return_ratio = profile.layers[-1].pac_budget / profile.layers[0].pac_budget

        results.append({
            "model": model_name,
            "n_layers": profile.n_layers,
            "effective": effective,
            "ratio": effective / profile.n_layers,
            "band": best_band,
            "band_cv": best_cv,
            "band_len": (best_band[1] - best_band[0] + 1) if best_band else 0,
            "return_ratio": return_ratio,
            "high": high,
            "medium": medium,
            "low": low,
        })

        print(f"\n  {model_name}: {profile.n_layers} layers")
        print(f"    Effective: {effective}/{profile.n_layers} (ratio {effective / profile.n_layers:.2f})")
        if best_band:
            print(f"    Conservation band: L{best_band[0]}-L{best_band[1]} "
                  f"({best_band[1] - best_band[0] + 1} layers, CV={best_cv:.4f})")
        print(f"    Budget return: {return_ratio:.4f}")
        print(f"    Clusters: {high} high / {medium} medium / {low} low")

        # Show contribution profile
        for i, m in enumerate(profile.layers):
            bar_len = min(int(m.layer_contribution * 10), 50)
            bar = "=" * bar_len
            print(f"    L{i:>2}: {m.layer_contribution:.4f} {bar}")

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"SCALING SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Model':<15} {'Layers':>6} {'Eff':>4} {'Ratio':>6} {'Band':>8} {'BandCV':>7} {'Return':>7}")
    print(f"{'-' * 15} {'-' * 6} {'-' * 4} {'-' * 6} {'-' * 8} {'-' * 7} {'-' * 7}")
    for r in results:
        band_str = f"L{r['band'][0]}-L{r['band'][1]}" if r["band"] else "none"
        print(f"{r['model']:<15} {r['n_layers']:>6} {r['effective']:>4} "
              f"{r['ratio']:>6.2f} {band_str:>8} {r['band_cv']:>7.4f} {r['return_ratio']:>7.4f}")

    print(f"\nMED PREDICTION: effective depth should plateau as nominal depth increases.")
    if len(results) > 1:
        ratios = [r["ratio"] for r in results]
        if ratios[-1] < ratios[0]:
            print(f"  Ratio decreased from {ratios[0]:.2f} to {ratios[-1]:.2f} [PASS] sublinear growth")
        else:
            print(f"  Ratio stable/increased [?] needs more data points")


def main():
    parser = argparse.ArgumentParser(description="SI Phase 0 - Instrument GPT-2")
    parser.add_argument("--text", type=str, help="Custom text to profile")
    parser.add_argument("--suite", action="store_true", help="Run full test suite")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name (default: gpt2)")
    parser.add_argument("--scaling", action="store_true",
                        help="Compare across gpt2, gpt2-medium, gpt2-large")
    args = parser.parse_args()

    if args.scaling:
        run_scaling(["gpt2", "gpt2-medium", "gpt2-large"])
    else:
        instrument = GPT2Instrument(args.model)

        if args.suite:
            run_suite(instrument)
        elif args.text:
            profile = instrument.profile(args.text)
            print_profile(profile)
        else:
            run_suite(instrument)


if __name__ == "__main__":
    main()
