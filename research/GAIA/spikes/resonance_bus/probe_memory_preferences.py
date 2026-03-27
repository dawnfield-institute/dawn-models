"""Probe: Do modules form internal preferences even when bus weights are balanced?

The QBE bus produces near-uniform weights (-90% preference vs sequential).
But preferences might form INSIDE modules, not in the bus weights.

This probe runs the QBE bus (and others) through the preference scenario
and inspects Memory's PAC tree after each stimulus class:
  - Tree structure (nodes, roots, depth distribution)
  - Storage efficiency (delta compression ratio — similar patterns compress better)
  - Retrieval resonance (how strongly the tree resonates with each class)
  - Node clustering (do stimulus classes create distinct subtrees?)

If the QBE bus is doing its job, the Memory module's PAC tree should show
INTERNAL differentiation even though the bus weights are symmetric.
"""

from __future__ import annotations

import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_gaia_root = os.path.join(_here, "..", "..")
sys.path.insert(0, os.path.join(_gaia_root, "src"))
sys.path.insert(0, os.path.join(_here, "..", "..", "..", "..", "..", "fracton"))

import torch

from gaia.core.bus import ConservationBus
from gaia.core.types import FieldState
from gaia.modules.language import LanguageModule
from gaia.modules.memory import MemoryModule
from gaia.modules.observability import ObservabilityModule
from gaia.modules.reasoning import ReasoningModule
from gaia.modules.safety import SafetyModule
from gaia.body.environment import GridWorld
from gaia.body.senses import VisualChannel, ProprioceptiveChannel
from gaia.body.motor import GridMotorDecoder
from gaia.body.loop import BodyLoop

from spike_c_perspective import PerspectiveBus
from spike_d_continuous import ContinuousFieldBus
from spike_e_qbe_regulated import QBEFieldBus
from spike_f_coupled_fields import CoupledFieldsBus

INPUT_DIM = 22
SEED = 42
N_TICKS_PER_CLASS = 30


def make_modules():
    """Create fresh module set, returning the memory module separately for inspection."""
    modules = {
        "observability": ObservabilityModule(),
        "safety": SafetyModule(input_dim=INPUT_DIM),
        "reasoning": ReasoningModule(input_dim=INPUT_DIM),
        "memory": MemoryModule(),
        "language": LanguageModule(),
    }
    return modules


def make_bus_with_modules(bus_class, **kwargs):
    """Create a bus and register modules. Return (bus, memory_module)."""
    modules = make_modules()
    bus = bus_class(enforcement="soft", **kwargs)
    for mod in modules.values():
        bus.register_module(mod)
    return bus, modules["memory"]


def snapshot_memory(memory_module: MemoryModule, label: str) -> dict:
    """Capture memory module's internal state."""
    tree = memory_module.tree
    metrics = memory_module.metrics

    # Depth distribution
    depth_dist = tree.depth_distribution()

    # Node strengths by depth
    strengths = {}
    for node in tree._nodes.values():
        d = node.depth.name.lower()
        if d not in strengths:
            strengths[d] = []
        strengths[d].append(node.strength)

    mean_strengths = {
        d: sum(s) / len(s) for d, s in strengths.items() if s
    }

    # Storage ratio (compression efficiency)
    storage_ratio = tree.storage_ratio()

    # Root count and tree structure
    n_roots = len(tree._root_ids)
    n_children = sum(
        len(n.children_ids) for n in tree._nodes.values()
    )
    mean_children = n_children / max(len(tree._nodes), 1)

    return {
        "label": label,
        "n_nodes": tree.size,
        "n_roots": n_roots,
        "depth_distribution": depth_dist,
        "mean_strengths": mean_strengths,
        "storage_ratio": storage_ratio,
        "mean_children_per_node": mean_children,
    }


def probe_retrieval(memory_module: MemoryModule, envs: dict[str, GridWorld]) -> dict[str, dict]:
    """Test how strongly the memory tree resonates with each stimulus class.

    Creates test inputs from each environment and measures mean retrieval
    score against the tree — higher score = tree has stronger memory of
    that stimulus class.
    """
    tree = memory_module.tree
    vis = VisualChannel()
    prop = ProprioceptiveChannel()
    results = {}

    for class_name, env in envs.items():
        obs = env.reset()  # reset returns Observation
        scores = []
        for _ in range(5):  # 5 test inputs per class
            # Encode visual + proprioceptive into tensors, then combine
            vis_state = vis.encode(obs.visual)
            prop_state = prop.encode(obs.proprioceptive)
            combined = torch.cat([vis_state.tensor, prop_state.tensor], dim=-1)
            flat = combined.flatten()[:INPUT_DIM]
            if flat.shape[0] < INPUT_DIM:
                flat = torch.nn.functional.pad(flat, (0, INPUT_DIM - flat.shape[0]))

            matches = tree.retrieve(flat, top_k=5, threshold=0.0)
            if matches:
                best_score = matches[0][1]
            else:
                best_score = 0.0
            scores.append(best_score)

            # Step with a no-op action to get a slightly different observation
            from gaia.body.motor import Action
            obs = env.step(Action(direction=torch.zeros(2), magnitude=0.0))

        results[class_name] = {
            "mean_best_retrieval": sum(scores) / len(scores),
            "max_retrieval": max(scores),
            "min_retrieval": min(scores),
        }

    return results


def run_preference_probe(bus_class, bus_name: str, **kwargs):
    """Run 3 stimulus classes through the bus, inspect memory after each."""
    bus, memory = make_bus_with_modules(bus_class, **kwargs)
    channels = [VisualChannel(), ProprioceptiveChannel()]
    decoder = GridMotorDecoder()

    # Create 3 distinct environments (stimulus classes)
    envs = {
        "A": GridWorld(size=5, n_stimuli=3, seed=SEED),
        "B": GridWorld(size=5, n_stimuli=3, seed=SEED + 100),
        "C": GridWorld(size=5, n_stimuli=3, seed=SEED + 200),
    }

    print(f"\n{'='*60}")
    print(f"  {bus_name}")
    print(f"{'='*60}")

    snapshots = []

    # Run each class sequentially
    for class_name in ["A", "B", "C"]:
        env = envs[class_name]
        loop = BodyLoop(bus, channels, decoder, env)
        loop.run(N_TICKS_PER_CLASS)

        snap = snapshot_memory(memory, f"After class {class_name}")
        snapshots.append(snap)

        print(f"\n  After {N_TICKS_PER_CLASS} ticks of class {class_name}:")
        print(f"    Nodes: {snap['n_nodes']}, Roots: {snap['n_roots']}")
        print(f"    Storage ratio: {snap['storage_ratio']:.4f} "
              f"(lower = better compression)")
        print(f"    Depth dist: {snap['depth_distribution']}")
        print(f"    Mean children/node: {snap['mean_children_per_node']:.2f}")
        if snap['mean_strengths']:
            for d, s in sorted(snap['mean_strengths'].items()):
                print(f"    Mean strength [{d}]: {s:.4f}")

    # Probe retrieval resonance for each class
    print(f"\n  Retrieval resonance (how strongly tree resonates with each class):")
    retrieval = probe_retrieval(memory, envs)
    for class_name, scores in retrieval.items():
        print(f"    Class {class_name}: best={scores['mean_best_retrieval']:.4f} "
              f"(range: {scores['min_retrieval']:.4f} - {scores['max_retrieval']:.4f})")

    # Compute preference signal: how differently does memory respond to each class?
    best_scores = [r["mean_best_retrieval"] for r in retrieval.values()]
    if len(best_scores) >= 2:
        preference_range = max(best_scores) - min(best_scores)
        preference_std = (sum((s - sum(best_scores)/len(best_scores))**2
                             for s in best_scores) / len(best_scores)) ** 0.5
        print(f"\n  INTERNAL PREFERENCE SIGNAL:")
        print(f"    Retrieval range: {preference_range:.4f} "
              f"(>0 = tree differentiates classes)")
        print(f"    Retrieval std:   {preference_std:.4f}")

    # Tree growth pattern
    if len(snapshots) >= 2:
        growth_rates = []
        for i in range(1, len(snapshots)):
            growth = snapshots[i]["n_nodes"] - snapshots[i-1]["n_nodes"]
            growth_rates.append(growth)
        print(f"    Node growth per class: {growth_rates}")
        print(f"    Root growth: {[s['n_roots'] for s in snapshots]}")

    # Health report
    rbf = memory.health()
    print(f"\n  Final RBF health: E={rbf.energy:.4f} I={rbf.information:.4f} "
          f"M={rbf.memory:.4f} B={rbf.balance:.4f}")

    return snapshots, retrieval


def main():
    print("Memory Preference Probe")
    print(f"Testing whether modules form INTERNAL preferences")
    print(f"even when bus weights are balanced.")
    print(f"({N_TICKS_PER_CLASS} ticks per stimulus class, 3 classes)")

    all_results = {}

    # Test key buses
    configs = [
        (ConservationBus, "Sequential", {}),
        (PerspectiveBus, "Perspective (Spike C)", {}),
        (ContinuousFieldBus, "Continuous (Spike D)", {}),
        (QBEFieldBus, "QBE (Spike E)", {}),
        (CoupledFieldsBus, "Coupled (Spike F)", {}),
    ]

    for bus_class, name, kwargs in configs:
        snaps, retrieval = run_preference_probe(bus_class, name, **kwargs)
        all_results[name] = {
            "snapshots": snaps,
            "retrieval": retrieval,
        }

    # Cross-bus comparison
    print(f"\n{'='*60}")
    print(f"  CROSS-BUS COMPARISON")
    print(f"{'='*60}")

    print(f"\n  {'Bus':<30} {'Tree Size':>10} {'Roots':>8} {'Storage':>10} "
          f"{'Pref Range':>12} {'Pref Std':>10}")
    print(f"  {'-'*80}")

    for name, data in all_results.items():
        final_snap = data["snapshots"][-1]
        best_scores = [r["mean_best_retrieval"] for r in data["retrieval"].values()]
        pref_range = max(best_scores) - min(best_scores) if len(best_scores) >= 2 else 0
        pref_std = (sum((s - sum(best_scores)/len(best_scores))**2
                       for s in best_scores) / len(best_scores)) ** 0.5 if best_scores else 0

        print(f"  {name:<30} {final_snap['n_nodes']:>10} {final_snap['n_roots']:>8} "
              f"{final_snap['storage_ratio']:>10.4f} {pref_range:>12.4f} {pref_std:>10.4f}")


if __name__ == "__main__":
    main()
