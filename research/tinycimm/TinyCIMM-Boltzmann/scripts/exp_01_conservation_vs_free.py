#!/usr/bin/env python3
"""
EXP 01: Conservation vs Free — The PAC Hallucination Test
==========================================================

Core Question:
  exp_12 showed LLMs violate PAC during hallucination (+9.6% uncompensated
  entropy, compensation ratio = 0.000 for GPT-2).

  Does ENFORCING conservation change how a model handles noise?

Design:
  4 conditions in a 2×2 matrix:

                     Factual         Noise (halluc-analogue)
  Conservation ON    learns well?    violation contained?
  Conservation OFF   learns well?    violation unconstrained?

  For each condition: 500 steps, 4 heads × 2 layers, same random seed.

Measurements:
  1. Task loss trajectory (can it still learn?)
  2. Budget violation over time
  3. Compensation ratio (cross-head entropy flow)
  4. Phase distribution (crystallized/ordered/transitional/chaotic)
  5. Budget stability (CV of total entropy over time)
  6. Early vs late dynamics (does violation grow or self-correct?)

Success Criteria:
  - Conservation ON should show lower violation under noise
  - Factual should show better conservation than noise (both modes)
  - If conservation helps: noise+conservation < noise+free violation
  - If PAC theory holds: violation magnitude correlates with task loss

Author: Dawn Field Institute
Date: 2026-02-14
"""

import sys, json, time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from scipy import stats as sp

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MODEL_DIR))

from tinycimm_boltzmann import (
    TinyCIMMBoltzmann,
    create_factual_stream,
    create_hallucination_stream,
    create_mixed_stream,
    create_fibonacci_ratio_stream,
    classify_sec_phase,
)

N_STEPS = 500
N_HEADS = 4
N_LAYERS = 2
HIDDEN = 32
SEEDS = [42, 137, 256, 314, 628]  # 5 seeds for significance testing
CONSERVATION_STRENGTHS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]


def run_condition(stream_fn, conservation_strength, seed, n_steps=N_STEPS):
    """Run a single condition and return metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mode = 'soft' if conservation_strength > 0 else 'none'

    model = TinyCIMMBoltzmann(
        input_size=1, hidden_size=HIDDEN, output_size=1,
        n_heads=N_HEADS, n_layers=N_LAYERS,
        conservation_mode=mode,
        conservation_strength=conservation_strength,
        learning_rate=0.01,
        device=device,
    )

    stream = stream_fn(n_steps)
    history = model.continuous_train(stream, max_steps=n_steps, log_interval=9999)

    # Extract trajectories
    task_losses = [m.task_loss for m in history]
    violations = [m.violation_pct for m in history]
    compensations = [m.compensation_ratio for m in history]
    budgets = [m.total_budget for m in history]
    entropies = [m.mean_entropy for m in history]
    cvs = [m.head_cv for m in history]

    # Phase counts over time
    phase_counts = {'crystallized': 0, 'ordered': 0, 'transitional': 0, 'chaotic': 0}
    for m in history:
        for p in m.head_phases:
            phase_counts[p] = phase_counts.get(p, 0) + 1
    total_phases = sum(phase_counts.values())
    phase_fracs = {k: v / max(total_phases, 1) for k, v in phase_counts.items()}

    # Early vs late
    split = len(history) // 2
    early_violation = float(np.mean([abs(v) for v in violations[:split]]))
    late_violation = float(np.mean([abs(v) for v in violations[split:]]))
    early_loss = float(np.mean(task_losses[:split]))
    late_loss = float(np.mean(task_losses[split:]))
    early_comp = float(np.mean(compensations[:split])) if compensations[:split] else 1.0
    late_comp = float(np.mean(compensations[split:])) if compensations[split:] else 1.0

    # Budget stability
    budget_std = float(np.std(budgets))
    budget_mean = float(np.mean(budgets))
    budget_cv = budget_std / max(budget_mean, 1e-8)

    summary = model.get_conservation_summary()

    return {
        'final_task_loss': task_losses[-1] if task_losses else 0,
        'mean_task_loss': float(np.mean(task_losses)),
        'final_violation': violations[-1] if violations else 0,
        'mean_abs_violation': float(np.mean([abs(v) for v in violations])),
        'final_compensation': compensations[-1] if compensations else 1.0,
        'mean_compensation': float(np.mean(compensations)) if compensations else 1.0,
        'early_violation': early_violation,
        'late_violation': late_violation,
        'violation_trend': 'growing' if late_violation > early_violation else 'shrinking',
        'early_loss': early_loss,
        'late_loss': late_loss,
        'early_compensation': early_comp,
        'late_compensation': late_comp,
        'budget_cv': budget_cv,
        'budget_stability': summary['budget_stability'],
        'phase_fracs': phase_fracs,
        'mean_entropy': float(np.mean(entropies)),
        'mean_head_cv': float(np.mean(cvs)),
    }


def main():
    print("=" * 70)
    print("  EXP 01: Conservation vs Free — The PAC Hallucination Test")
    print("  Does enforcing PAC conservation prevent hallucination-analogue?")
    print("=" * 70)

    t0 = time.time()
    results = {}

    # ── Part 1: 2×2 Matrix (conservation on/off × factual/noise) ──
    print("\n" + "─" * 60)
    print("  PART 1: 2×2 Conservation × Data Type Matrix")
    print("─" * 60)

    conditions = [
        ("factual_conserved",    create_factual_stream,       1.0),
        ("factual_free",         create_factual_stream,       0.0),
        ("noise_conserved",      create_hallucination_stream, 1.0),
        ("noise_free",           create_hallucination_stream, 0.0),
    ]

    matrix_results = {}
    for cond_name, stream_fn, strength in conditions:
        print(f"\n  Condition: {cond_name} (strength={strength})")
        seed_results = []
        for seed in SEEDS:
            r = run_condition(stream_fn, strength, seed)
            seed_results.append(r)
            print(f"    seed={seed}: loss={r['final_task_loss']:.6f}  "
                  f"violation={r['mean_abs_violation']:.1f}%  "
                  f"comp={r['mean_compensation']:.3f}")

        # Aggregate across seeds
        agg = {
            'mean_loss': float(np.mean([r['final_task_loss'] for r in seed_results])),
            'std_loss': float(np.std([r['final_task_loss'] for r in seed_results])),
            'mean_violation': float(np.mean([r['mean_abs_violation'] for r in seed_results])),
            'std_violation': float(np.std([r['mean_abs_violation'] for r in seed_results])),
            'mean_compensation': float(np.mean([r['mean_compensation'] for r in seed_results])),
            'mean_budget_cv': float(np.mean([r['budget_cv'] for r in seed_results])),
            'mean_stability': float(np.mean([r['budget_stability'] for r in seed_results])),
            'early_violation': float(np.mean([r['early_violation'] for r in seed_results])),
            'late_violation': float(np.mean([r['late_violation'] for r in seed_results])),
            'seed_results': seed_results,
        }
        matrix_results[cond_name] = agg

    results['matrix'] = matrix_results

    # Print matrix
    print(f"\n  {'─'*60}")
    print(f"  2×2 MATRIX RESULTS (mean ± std across {len(SEEDS)} seeds):")
    print(f"  {'Condition':25s} {'Loss':>12s} {'|Violation|%':>12s} "
          f"{'Compensation':>12s} {'Stability':>10s}")
    for name, agg in matrix_results.items():
        print(f"  {name:25s} "
              f"{agg['mean_loss']:8.6f}±{agg['std_loss']:.4f} "
              f"{agg['mean_violation']:8.1f}±{agg['std_violation']:.1f}% "
              f"{agg['mean_compensation']:12.3f} "
              f"{agg['mean_stability']:10.3f}")

    # Statistical tests
    print(f"\n  STATISTICAL TESTS:")

    # Test 1: Does conservation reduce violation under noise?
    noise_cons = [r['mean_abs_violation'] for r in matrix_results['noise_conserved']['seed_results']]
    noise_free = [r['mean_abs_violation'] for r in matrix_results['noise_free']['seed_results']]
    _, p1 = sp.mannwhitneyu(noise_cons, noise_free, alternative='two-sided')
    sig1 = "***" if p1 < 0.001 else "**" if p1 < 0.01 else "*" if p1 < 0.05 else "n.s."
    print(f"  Conservation reduces noise violation? p={p1:.6f} {sig1}")
    print(f"    Conserved: {np.mean(noise_cons):.1f}%  Free: {np.mean(noise_free):.1f}%")

    # Test 2: Does noise have more violation than factual?
    fact_cons = [r['mean_abs_violation'] for r in matrix_results['factual_conserved']['seed_results']]
    _, p2 = sp.mannwhitneyu(fact_cons, noise_cons, alternative='two-sided')
    sig2 = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < 0.05 else "n.s."
    print(f"  Noise > factual violation (conserved)? p={p2:.6f} {sig2}")

    # Test 3: Does conservation hurt factual learning?
    fact_cons_loss = [r['final_task_loss'] for r in matrix_results['factual_conserved']['seed_results']]
    fact_free_loss = [r['final_task_loss'] for r in matrix_results['factual_free']['seed_results']]
    _, p3 = sp.mannwhitneyu(fact_cons_loss, fact_free_loss, alternative='two-sided')
    sig3 = "***" if p3 < 0.001 else "**" if p3 < 0.01 else "*" if p3 < 0.05 else "n.s."
    print(f"  Conservation hurts factual learning? p={p3:.6f} {sig3}")
    print(f"    Conserved loss: {np.mean(fact_cons_loss):.6f}  "
          f"Free loss: {np.mean(fact_free_loss):.6f}")

    results['tests'] = {
        'conservation_reduces_noise_violation': {'p': float(p1), 'sig': sig1},
        'noise_more_violation_than_factual': {'p': float(p2), 'sig': sig2},
        'conservation_hurts_factual': {'p': float(p3), 'sig': sig3},
    }

    # ── Part 2: Strength Sweep ──
    print(f"\n{'─'*60}")
    print(f"  PART 2: Conservation Strength Sweep (noise data)")
    print(f"{'─'*60}")

    sweep_results = {}
    for strength in CONSERVATION_STRENGTHS:
        seed_results = []
        for seed in SEEDS:
            r = run_condition(create_hallucination_stream, strength, seed)
            seed_results.append(r)

        mean_violation = float(np.mean([r['mean_abs_violation'] for r in seed_results]))
        mean_loss = float(np.mean([r['final_task_loss'] for r in seed_results]))
        mean_comp = float(np.mean([r['mean_compensation'] for r in seed_results]))
        mean_stab = float(np.mean([r['budget_stability'] for r in seed_results]))

        sweep_results[str(strength)] = {
            'mean_violation': mean_violation,
            'mean_loss': mean_loss,
            'mean_compensation': mean_comp,
            'mean_stability': mean_stab,
        }
        print(f"  strength={strength:4.1f}: "
              f"violation={mean_violation:6.1f}%  "
              f"loss={mean_loss:.6f}  "
              f"comp={mean_comp:.3f}  "
              f"stability={mean_stab:.3f}")

    results['strength_sweep'] = sweep_results

    # ── Part 3: Mixed Stream (factual then noise) ──
    print(f"\n{'─'*60}")
    print(f"  PART 3: Mixed Stream (factual → noise transition)")
    print(f"{'─'*60}")

    mixed_results = {}
    for mode_name, strength in [("conserved", 1.0), ("free", 0.0)]:
        seed_results = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            mode = 'soft' if strength > 0 else 'none'

            model = TinyCIMMBoltzmann(
                input_size=1, hidden_size=HIDDEN, output_size=1,
                n_heads=N_HEADS, n_layers=N_LAYERS,
                conservation_mode=mode,
                conservation_strength=strength,
                device=device,
            )

            stream = create_mixed_stream(n_factual=300, n_halluc=200)
            history = model.continuous_train(stream, max_steps=500, log_interval=9999)

            # Split at transition point
            factual_phase = history[:300]
            noise_phase = history[300:]

            fact_violations = [abs(m.violation_pct) for m in factual_phase]
            noise_violations = [abs(m.violation_pct) for m in noise_phase]
            fact_losses = [m.task_loss for m in factual_phase]
            noise_losses = [m.task_loss for m in noise_phase]

            seed_results.append({
                'factual_violation': float(np.mean(fact_violations)),
                'noise_violation': float(np.mean(noise_violations)),
                'factual_loss': float(np.mean(fact_losses)),
                'noise_loss': float(np.mean(noise_losses)),
                'transition_shock': float(np.mean(noise_violations[:20])),
            })

        mixed_results[mode_name] = {
            'mean_factual_violation': float(np.mean([r['factual_violation'] for r in seed_results])),
            'mean_noise_violation': float(np.mean([r['noise_violation'] for r in seed_results])),
            'mean_transition_shock': float(np.mean([r['transition_shock'] for r in seed_results])),
            'seed_results': seed_results,
        }

        print(f"\n  {mode_name}:")
        print(f"    Factual phase violation: "
              f"{mixed_results[mode_name]['mean_factual_violation']:.1f}%")
        print(f"    Noise phase violation:   "
              f"{mixed_results[mode_name]['mean_noise_violation']:.1f}%")
        print(f"    Transition shock:        "
              f"{mixed_results[mode_name]['mean_transition_shock']:.1f}%")

    results['mixed_stream'] = mixed_results

    # Test: transition shock
    cons_shock = [r['transition_shock'] for r in mixed_results['conserved']['seed_results']]
    free_shock = [r['transition_shock'] for r in mixed_results['free']['seed_results']]
    _, p_shock = sp.mannwhitneyu(cons_shock, free_shock, alternative='two-sided')
    sig_shock = "***" if p_shock < 0.001 else "**" if p_shock < 0.01 else "*" if p_shock < 0.05 else "n.s."
    print(f"\n  Conservation reduces transition shock? p={p_shock:.6f} {sig_shock}")
    print(f"    Conserved: {np.mean(cons_shock):.1f}%  Free: {np.mean(free_shock):.1f}%")

    results['tests']['conservation_reduces_transition_shock'] = {
        'p': float(p_shock), 'sig': sig_shock
    }

    # ── Verdict ──
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    n_sig = sum(1 for t in results['tests'].values()
                if float(t['p']) < 0.05)
    print(f"  {n_sig}/{len(results['tests'])} tests significant at p < 0.05")

    for name, test in results['tests'].items():
        print(f"    {name}: p={test['p']:.6f} {test['sig']}")

    # Save
    results_dir = MODEL_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"exp_01_conservation_vs_free_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
