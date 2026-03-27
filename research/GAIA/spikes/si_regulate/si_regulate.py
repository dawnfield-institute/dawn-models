"""
SI Phase 1 — Conservation-Gated Generation
============================================

Tests predictions 2 and 4 from SI_theory.md:
  P2: PAC violations predict hallucination
  P4: Conservation-gated generation reduces hallucination without killing fluency

Mechanism:
  - During autoregressive generation, sample top-k candidate tokens
  - For each candidate, run a full forward pass and measure PAC conservation
  - Score each candidate by "budget return" (how close L_last PAC is to L0 PAC)
  - Reject candidates with poor conservation scores
  - Compare factual accuracy: standard sampling vs conservation-gated sampling

Key insight from Phase 0: GPT-2 has a conservation loop where L0 PAC ~= L_last PAC
(return ratio 0.98 for gpt2-small). Tokens that break this loop may be hallucinations.

Usage:
  python si_regulate.py                    # run default factual completion test
  python si_regulate.py --compare          # side-by-side standard vs gated
  python si_regulate.py --sweep            # sweep conservation threshold
"""

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ─── DFT Constants ───

XI_SEC = 0.0618
PHI_INV = 0.618
LAMBDA_STAR = 0.9816
GAMMA = 0.0184


# ─── PAC Measurement (from si_instrument) ───

def activation_entropy(h: torch.Tensor) -> float:
    if h.dim() == 3:
        h = h.squeeze(0)
    h_mean = h.mean(dim=0)
    magnitudes = h_mean.abs()
    total = magnitudes.sum()
    if total < 1e-10:
        return 0.0
    p = magnitudes / total
    p = p[p > 1e-10]
    entropy = -(p * p.log()).sum().item()
    max_entropy = math.log(len(magnitudes))
    return entropy / max_entropy if max_entropy > 1e-10 else 0.0


def information_content(h: torch.Tensor, h_prev: torch.Tensor) -> float:
    if h.dim() == 3:
        h = h.squeeze(0)
    if h_prev.dim() == 3:
        h_prev = h_prev.squeeze(0)
    h_flat = h.reshape(-1)
    h_prev_flat = h_prev.reshape(-1)
    cos_sim = F.cosine_similarity(h_flat.unsqueeze(0), h_prev_flat.unsqueeze(0)).item()
    return 1.0 - max(cos_sim, 0.0)


def structural_coherence(h: torch.Tensor) -> float:
    if h.dim() == 3:
        h = h.squeeze(0)
    n_pos = h.shape[0]
    if n_pos < 2:
        return 1.0
    if n_pos > 50:
        indices = torch.randperm(n_pos)[:50]
        h = h[indices]
        n_pos = len(indices)
    h_norm = F.normalize(h, dim=-1)
    sim_matrix = h_norm @ h_norm.T
    mask = torch.triu(torch.ones(n_pos, n_pos, device=h.device), diagonal=1).bool()
    similarities = sim_matrix[mask]
    return similarities.mean().item() if len(similarities) > 0 else 1.0


def pac_budget(h_out: torch.Tensor, h_in: torch.Tensor) -> tuple[float, float, float, float]:
    """Returns (E, I, M, total_pac)."""
    E = activation_entropy(h_out)
    I = information_content(h_out, h_in)
    M = structural_coherence(h_out)
    return E, I, M, E + I + M


# ─── Conservation Score ───

def conservation_score(hidden_states: tuple, n_layers: int) -> dict:
    """Compute conservation metrics for a forward pass.

    Returns dict with:
      - budget_return: L_last PAC / L0 PAC (1.0 = perfect conservation loop)
      - band_cv: coefficient of variation in the conservation band (middle layers)
      - max_delta: largest PAC jump between consecutive layers
      - pac_profile: per-layer PAC values
    """
    pac_values = []
    for layer_idx in range(n_layers):
        h_in = hidden_states[layer_idx]
        h_out = hidden_states[layer_idx + 1]
        _, _, _, total = pac_budget(h_out, h_in)
        pac_values.append(total)

    # Budget return
    budget_return = pac_values[-1] / pac_values[0] if pac_values[0] > 1e-10 else 0.0

    # Conservation band (middle third of layers)
    band_start = max(2, n_layers // 4)
    band_end = min(n_layers - 2, 3 * n_layers // 4)
    band = pac_values[band_start:band_end]
    if len(band) >= 2:
        band_mean = sum(band) / len(band)
        band_std = (sum((v - band_mean) ** 2 for v in band) / len(band)) ** 0.5
        band_cv = band_std / band_mean if band_mean > 0 else 0
    else:
        band_cv = 0.0

    # Max delta
    deltas = [abs(pac_values[i] - pac_values[i - 1]) for i in range(1, len(pac_values))]
    max_delta = max(deltas) if deltas else 0.0

    return {
        "budget_return": budget_return,
        "band_cv": band_cv,
        "max_delta": max_delta,
        "pac_profile": pac_values,
    }


# ─── Generator ───

class ConservationGatedGenerator:
    """GPT-2 generator with conservation-gated token selection."""

    def __init__(self, model_name: str = "gpt2"):
        print(f"Loading {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            attn_implementation="eager",
        )
        self.model.eval()
        self.n_layers = self.model.config.n_layer
        print(f"  {self.n_layers} layers, {self.model.config.n_embd} hidden dim")

    @torch.no_grad()
    def generate_standard(self, prompt: str, max_tokens: int = 30,
                          temperature: float = 1.0, top_k: int = 50) -> dict:
        """Standard top-k sampling generation."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        generated = input_ids.clone()
        token_scores = []

        for step in range(max_tokens):
            outputs = self.model(generated, output_hidden_states=True)
            logits = outputs.logits[:, -1, :] / temperature

            # Top-k filtering
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            probs = F.softmax(top_k_logits, dim=-1)

            # Sample
            idx = torch.multinomial(probs, 1)
            next_token = top_k_indices[0, idx[0, 0]].unsqueeze(0).unsqueeze(0)

            # Measure conservation of the chosen token's forward pass
            score = conservation_score(outputs.hidden_states, self.n_layers)
            token_scores.append(score)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop at EOS or period
            token_str = self.tokenizer.decode(next_token[0])
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return {
            "text": text,
            "prompt": prompt,
            "generated": text[len(prompt):],
            "n_tokens": len(token_scores),
            "scores": token_scores,
            "method": "standard",
        }

    @torch.no_grad()
    def generate_gated(self, prompt: str, max_tokens: int = 30,
                       temperature: float = 1.0, top_k: int = 50,
                       conservation_threshold: float = 0.3,
                       max_resamples: int = 5) -> dict:
        """Conservation-gated generation.

        For each position:
        1. Sample top-k candidates
        2. Score each by conservation (budget return closeness to 1.0)
        3. Pick the best-conserving candidate above threshold
        4. If none pass, fall back to standard sampling
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        generated = input_ids.clone()
        token_scores = []
        rejections = 0
        total_candidates = 0
        gated_selections = 0

        for step in range(max_tokens):
            outputs = self.model(generated, output_hidden_states=True)
            logits = outputs.logits[:, -1, :] / temperature

            # Top-k candidates
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.shape[-1]))
            probs = F.softmax(top_k_logits, dim=-1)

            # Evaluate top candidates by conservation score
            # Only check top-N candidates (full forward pass per candidate is expensive)
            n_candidates = min(8, top_k)
            candidate_scores = []

            for c in range(n_candidates):
                candidate_token = top_k_indices[0, c].unsqueeze(0).unsqueeze(0)
                candidate_input = torch.cat([generated, candidate_token], dim=1)

                c_outputs = self.model(candidate_input, output_hidden_states=True)
                c_score = conservation_score(c_outputs.hidden_states, self.n_layers)
                c_prob = probs[0, c].item()

                # Conservation quality: how close is budget return to 1.0?
                conservation_quality = 1.0 - abs(c_score["budget_return"] - 1.0)

                candidate_scores.append({
                    "token_id": top_k_indices[0, c].item(),
                    "prob": c_prob,
                    "conservation_quality": conservation_quality,
                    "budget_return": c_score["budget_return"],
                    "band_cv": c_score["band_cv"],
                    "score": c_score,
                })

            total_candidates += n_candidates

            # Select best candidate that passes conservation threshold
            # Sort by conservation quality, pick highest-prob among those passing
            passing = [c for c in candidate_scores
                       if c["conservation_quality"] > (1.0 - conservation_threshold)]

            if passing:
                # Among passing candidates, pick highest probability
                best = max(passing, key=lambda c: c["prob"])
                gated_selections += 1
            else:
                # Fallback: pick highest probability regardless
                best = max(candidate_scores, key=lambda c: c["prob"])
                rejections += 1

            next_token = torch.tensor([[best["token_id"]]])
            token_scores.append(best["score"])
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return {
            "text": text,
            "prompt": prompt,
            "generated": text[len(prompt):],
            "n_tokens": len(token_scores),
            "scores": token_scores,
            "method": "conservation_gated",
            "rejections": rejections,
            "gated_selections": gated_selections,
            "total_candidates": total_candidates,
            "conservation_threshold": conservation_threshold,
        }

    @torch.no_grad()
    def score_continuation(self, prompt: str, continuation: str) -> dict:
        """Score a known continuation for conservation quality.

        Used to compare correct vs incorrect completions.
        """
        full_text = prompt + continuation
        input_ids = self.tokenizer.encode(full_text, return_tensors="pt")

        outputs = self.model(input_ids, output_hidden_states=True)
        score = conservation_score(outputs.hidden_states, self.n_layers)

        return {
            "text": full_text,
            "prompt": prompt,
            "continuation": continuation,
            "score": score,
        }


# ─── Test Cases ───

FACTUAL_TESTS = [
    {
        "prompt": "The capital of France is",
        "correct": " Paris",
        "incorrect": [" London", " Berlin", " Madrid"],
        "category": "geography",
    },
    {
        "prompt": "Water freezes at",
        "correct": " 0 degrees",
        "incorrect": [" 100 degrees", " 50 degrees", " 25 degrees"],
        "category": "science",
    },
    {
        "prompt": "The largest planet in our solar system is",
        "correct": " Jupiter",
        "incorrect": [" Mars", " Saturn", " Earth"],
        "category": "astronomy",
    },
    {
        "prompt": "Shakespeare wrote",
        "correct": " Hamlet",
        "incorrect": [" calculus", " symphonies", " Python"],
        "category": "literature",
    },
    {
        "prompt": "The speed of light is approximately",
        "correct": " 300,000 kilometers per second",
        "incorrect": [" 300 kilometers per second", " 1000 miles per hour"],
        "category": "physics",
    },
    {
        "prompt": "DNA stands for",
        "correct": " deoxyribonucleic acid",
        "incorrect": [" digital network access", " dynamic neural architecture"],
        "category": "biology",
    },
    {
        "prompt": "The chemical symbol for gold is",
        "correct": " Au",
        "incorrect": [" Go", " Gd", " Ag"],
        "category": "chemistry",
    },
    {
        "prompt": "Albert Einstein developed the theory of",
        "correct": " relativity",
        "incorrect": [" evolution", " gravity waves", " quantum computing"],
        "category": "physics",
    },
    {
        "prompt": "The Great Wall of China was built primarily to",
        "correct": " protect against invasions",
        "incorrect": [" generate electricity", " transport water"],
        "category": "history",
    },
    {
        "prompt": "Photosynthesis converts sunlight into",
        "correct": " chemical energy",
        "incorrect": [" nuclear energy", " electrical current", " sound waves"],
        "category": "biology",
    },
    {
        "prompt": "The Earth revolves around the",
        "correct": " Sun",
        "incorrect": [" Moon", " Mars", " North Star"],
        "category": "astronomy",
    },
    {
        "prompt": "The human heart has",
        "correct": " four chambers",
        "incorrect": [" two chambers", " six chambers", " eight chambers"],
        "category": "biology",
    },
]


def run_conservation_test(gen: ConservationGatedGenerator):
    """P2: Do correct continuations have better conservation scores than incorrect ones?"""
    print(f"\n{'=' * 80}")
    print("P2 - PAC VIOLATIONS PREDICT HALLUCINATION")
    print(f"{'=' * 80}")
    print(f"\nScoring {len(FACTUAL_TESTS)} factual prompts: correct vs incorrect continuations\n")

    correct_returns = []
    incorrect_returns = []
    correct_band_cvs = []
    incorrect_band_cvs = []
    wins = 0
    total = 0

    for test in FACTUAL_TESTS:
        prompt = test["prompt"]

        # Score correct continuation
        correct_result = gen.score_continuation(prompt, test["correct"])
        c_return = correct_result["score"]["budget_return"]
        c_cv = correct_result["score"]["band_cv"]
        correct_returns.append(c_return)
        correct_band_cvs.append(c_cv)

        # Score incorrect continuations
        inc_returns_local = []
        inc_cvs_local = []
        for inc in test["incorrect"]:
            inc_result = gen.score_continuation(prompt, inc)
            i_return = inc_result["score"]["budget_return"]
            i_cv = inc_result["score"]["band_cv"]
            incorrect_returns.append(i_return)
            incorrect_band_cvs.append(i_cv)
            inc_returns_local.append(i_return)
            inc_cvs_local.append(i_cv)

        # Does correct have better conservation?
        avg_inc_return = sum(inc_returns_local) / len(inc_returns_local)
        c_quality = 1.0 - abs(c_return - 1.0)
        i_quality = 1.0 - abs(avg_inc_return - 1.0)
        won = c_quality > i_quality
        if won:
            wins += 1
        total += 1

        marker = "[+]" if won else "[-]"
        print(f"  {marker} {prompt}")
        print(f"      Correct: '{test['correct'].strip()}' "
              f"return={c_return:.4f} quality={c_quality:.4f}")
        print(f"      Incorrect avg: return={avg_inc_return:.4f} quality={i_quality:.4f}")

    # Aggregate
    mean_correct_return = sum(correct_returns) / len(correct_returns)
    mean_incorrect_return = sum(incorrect_returns) / len(incorrect_returns)
    correct_quality = 1.0 - abs(mean_correct_return - 1.0)
    incorrect_quality = 1.0 - abs(mean_incorrect_return - 1.0)

    print(f"\n{'-' * 80}")
    print(f"RESULTS:")
    print(f"  Correct continuations:   mean return = {mean_correct_return:.4f}  "
          f"quality = {correct_quality:.4f}")
    print(f"  Incorrect continuations: mean return = {mean_incorrect_return:.4f}  "
          f"quality = {incorrect_quality:.4f}")
    print(f"  Separation: {abs(correct_quality - incorrect_quality):.4f}")
    print(f"  Correct wins: {wins}/{total} ({100 * wins / total:.0f}%)")
    print(f"  VERDICT: {'[PASS]' if wins > total * 0.6 else '[?] MARGINAL' if wins > total * 0.4 else '[FAIL]'} "
          f"conservation {'predicts' if wins > total * 0.6 else 'partially predicts' if wins > total * 0.4 else 'does not predict'} correctness")

    # Band CV comparison
    mean_correct_cv = sum(correct_band_cvs) / len(correct_band_cvs)
    mean_incorrect_cv = sum(incorrect_band_cvs) / len(incorrect_band_cvs)
    print(f"\n  Band CV (conservation band stability):")
    print(f"    Correct:   {mean_correct_cv:.6f}")
    print(f"    Incorrect: {mean_incorrect_cv:.6f}")
    print(f"    {'[PASS] correct has tighter band' if mean_correct_cv < mean_incorrect_cv else '[?] no clear separation'}")

    return {
        "wins": wins,
        "total": total,
        "correct_quality": correct_quality,
        "incorrect_quality": incorrect_quality,
        "separation": abs(correct_quality - incorrect_quality),
        "correct_returns": correct_returns,
        "incorrect_returns": incorrect_returns,
    }


def run_generation_compare(gen: ConservationGatedGenerator):
    """P4: Compare standard vs conservation-gated generation."""
    print(f"\n{'=' * 80}")
    print("P4 - CONSERVATION-GATED GENERATION")
    print(f"{'=' * 80}")

    prompts = [
        "The theory of evolution was developed by",
        "In quantum mechanics, the uncertainty principle states that",
        "The Roman Empire fell because",
        "Machine learning models work by",
        "The human brain contains approximately",
    ]

    thresholds = [0.1, 0.2, 0.3, 0.5]

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print(f"{'-' * 60}")

        # Standard generation
        t0 = time.time()
        std = gen.generate_standard(prompt, max_tokens=25, temperature=0.7, top_k=20)
        std_time = time.time() - t0
        std_returns = [s["budget_return"] for s in std["scores"]]
        std_mean_return = sum(std_returns) / len(std_returns) if std_returns else 0

        print(f"  Standard ({std_time:.1f}s): {std['generated'].strip()}")
        print(f"    Mean budget return: {std_mean_return:.4f}")

        # Gated generation at different thresholds
        for threshold in thresholds:
            t0 = time.time()
            gated = gen.generate_gated(prompt, max_tokens=25, temperature=0.7,
                                       top_k=20, conservation_threshold=threshold)
            gated_time = time.time() - t0
            gated_returns = [s["budget_return"] for s in gated["scores"]]
            gated_mean_return = sum(gated_returns) / len(gated_returns) if gated_returns else 0

            print(f"  Gated t={threshold} ({gated_time:.1f}s): {gated['generated'].strip()}")
            print(f"    Mean budget return: {gated_mean_return:.4f}  "
                  f"gated: {gated['gated_selections']}/{gated['n_tokens']}  "
                  f"fallbacks: {gated['rejections']}")


def run_threshold_sweep(gen: ConservationGatedGenerator):
    """Sweep conservation threshold to find quality-conservation tradeoff."""
    print(f"\n{'=' * 80}")
    print("THRESHOLD SWEEP")
    print(f"{'=' * 80}")

    prompt = "The theory of evolution was developed by"
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]
    n_runs = 3  # Average over multiple runs

    print(f"\nPrompt: '{prompt}'")
    print(f"Runs per threshold: {n_runs}")
    print(f"\n{'Threshold':>10} {'MeanReturn':>11} {'GatedPct':>9} {'Fallbacks':>10} {'Time':>6}")
    print(f"{'-' * 10} {'-' * 11} {'-' * 9} {'-' * 10} {'-' * 6}")

    for threshold in thresholds:
        returns_all = []
        gated_pcts = []
        fallbacks_all = []
        times = []

        for _ in range(n_runs):
            t0 = time.time()
            result = gen.generate_gated(prompt, max_tokens=20, temperature=0.7,
                                         top_k=20, conservation_threshold=threshold)
            elapsed = time.time() - t0
            times.append(elapsed)

            rets = [s["budget_return"] for s in result["scores"]]
            returns_all.extend(rets)
            gated_pcts.append(result["gated_selections"] / max(result["n_tokens"], 1))
            fallbacks_all.append(result["rejections"])

        mean_return = sum(returns_all) / len(returns_all) if returns_all else 0
        mean_gated = sum(gated_pcts) / len(gated_pcts) if gated_pcts else 0
        mean_fallback = sum(fallbacks_all) / len(fallbacks_all)
        mean_time = sum(times) / len(times)

        print(f"{threshold:>10.2f} {mean_return:>11.4f} {mean_gated:>8.0%} "
              f"{mean_fallback:>10.1f} {mean_time:>5.1f}s")


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description="SI Phase 1 - Conservation-Gated Generation")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--compare", action="store_true", help="Compare standard vs gated")
    parser.add_argument("--sweep", action="store_true", help="Sweep conservation threshold")
    parser.add_argument("--all", action="store_true", help="Run everything")
    args = parser.parse_args()

    gen = ConservationGatedGenerator(args.model)

    if args.all or (not args.compare and not args.sweep):
        # Default: run P2 test (does conservation predict correctness?)
        results = run_conservation_test(gen)

        # Save results
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        save_data = {k: v for k, v in results.items()
                     if not isinstance(v, list) or len(v) < 100}
        save_data["correct_returns"] = results["correct_returns"]
        save_data["incorrect_returns"] = results["incorrect_returns"]
        (results_dir / "p2_conservation_test.json").write_text(
            json.dumps(save_data, indent=2))

    if args.compare or args.all:
        run_generation_compare(gen)

    if args.sweep or args.all:
        run_threshold_sweep(gen)


if __name__ == "__main__":
    main()
