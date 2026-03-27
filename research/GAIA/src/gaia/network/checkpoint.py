"""Checkpoint system for GAIA organisms and colonies.

Saves and restores the full state of a population: module weights,
learned transitions, PAC trees, resonance states, trust networks,
voice vectors -- everything needed to resume growing a colony.

Uses torch.save/load (pickle + tensor optimization). Non-invasive:
reads/writes module internals without modifying module source code.

Usage:
    from gaia.network.checkpoint import save_colony, load_colony

    # Save
    save_colony(colony, "checkpoints/colony_v1.pt")

    # Load (creates a new Colony with restored state)
    colony = load_colony("checkpoints/colony_v1.pt")
    colony.run(100, env_fn=...)  # continues from where it left off
"""

from __future__ import annotations

import copy
import logging
from collections import deque
from pathlib import Path
from typing import Any

import torch

from ..core.coupled_fields_bus import CoupledFieldsBus, CoupledFieldState
from ..core.types import FieldState, SECPhase

logger = logging.getLogger(__name__)


# ─── Module State Extraction ──────────────────────────────────────

def _save_safety_module(mod) -> dict:
    """Extract SafetyModule state."""
    state: dict[str, Any] = {
        "type": "safety",
        "step_count": mod._step_count,
        "layers": [],
    }
    for i, layer in enumerate(mod._layers):
        layer_state = {
            "state_dict": {k: v.cpu() for k, v in layer.state_dict().items()},
        }
        # Monitor state
        if i < len(mod._monitors):
            monitor = mod._monitors[i]
            layer_state["monitor"] = {
                "budget_history": list(monitor.budget_history),
                "violation_history": list(monitor.violation_history),
                "compensation_history": list(monitor.compensation_history),
            }
        state["layers"].append(layer_state)

    # Output projection
    if hasattr(mod, '_output_proj') and mod._output_proj is not None:
        state["output_proj"] = {
            k: v.cpu() for k, v in mod._output_proj.state_dict().items()
        }
    return state


def _load_safety_module(mod, state: dict) -> None:
    """Restore SafetyModule state."""
    mod._step_count = state["step_count"]
    for i, layer_state in enumerate(state["layers"]):
        if i < len(mod._layers):
            mod._layers[i].load_state_dict(layer_state["state_dict"])
            if "monitor" in layer_state and i < len(mod._monitors):
                monitor = mod._monitors[i]
                ms = layer_state["monitor"]
                monitor.budget_history = list(ms["budget_history"])
                monitor.violation_history = list(ms["violation_history"])
                monitor.compensation_history = list(ms["compensation_history"])
    if "output_proj" in state and hasattr(mod, '_output_proj') and mod._output_proj is not None:
        mod._output_proj.load_state_dict(state["output_proj"])


def _save_reasoning_module(mod) -> dict:
    """Extract ReasoningModule state."""
    state: dict[str, Any] = {
        "type": "reasoning",
        "step_count": mod._step_count,
        "layers": [],
    }
    for layer in mod._layers:
        state["layers"].append({
            k: v.cpu() for k, v in layer.state_dict().items()
        })

    # PhiAnchorMemory
    if hasattr(mod, '_anchor_memory') and mod._anchor_memory is not None:
        anchors = []
        for snap in mod._anchor_memory.anchors:
            snap_data = {"task": snap["task"]}
            if "params" in snap:
                snap_data["params"] = [
                    {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                     for k, v in p.items()}
                    for p in snap["params"]
                ]
            if "freq" in snap:
                snap_data["freq"] = snap["freq"]
            if "chord" in snap:
                snap_data["chord"] = snap["chord"]
            anchors.append(snap_data)
        state["anchor_memory"] = {
            "anchors": anchors,
            "current_task": mod._anchor_memory.current_task,
        }
    return state


def _load_reasoning_module(mod, state: dict) -> None:
    """Restore ReasoningModule state."""
    mod._step_count = state["step_count"]
    for i, layer_state in enumerate(state["layers"]):
        if i < len(mod._layers):
            mod._layers[i].load_state_dict(layer_state)

    if "anchor_memory" in state and hasattr(mod, '_anchor_memory') and mod._anchor_memory is not None:
        am = state["anchor_memory"]
        mod._anchor_memory.anchors = am["anchors"]
        mod._anchor_memory.current_task = am["current_task"]


def _save_memory_module(mod) -> dict:
    """Extract MemoryModule state."""
    state: dict[str, Any] = {
        "type": "memory",
        "step_count": mod._step_count,
        "last_stored_id": mod._last_stored_id,
    }

    # PACTree
    tree = mod._tree
    nodes = {}
    for nid, node in tree._nodes.items():
        nodes[nid] = {
            "id": node.id,
            "delta": node.delta.cpu().clone(),
            "parent_id": node.parent_id,
            "children_ids": list(node.children_ids),
            "strength": node.strength,
            "depth": node.depth.value if hasattr(node.depth, 'value') else node.depth,
            "access_count": node.access_count,
            "created_at": node.created_at,
            "label": node.label,
        }
    state["tree"] = {
        "nodes": nodes,
        "next_id": tree._next_id,
        "root_ids": list(tree._root_ids),
        "capacity": tree._capacity,
    }

    # TransitionTracker
    tt = mod._transitions
    state["transitions"] = {
        "transitions": dict(tt._transitions),
        "best_next": dict(tt._best_next) if hasattr(tt, '_best_next') else {},
        "decay_rate": tt._decay_rate if hasattr(tt, '_decay_rate') else 0.0,
    }
    return state


def _load_memory_module(mod, state: dict) -> None:
    """Restore MemoryModule state."""
    from ..modules.memory import MemoryNode, BifractalDepth

    mod._step_count = state["step_count"]
    mod._last_stored_id = state["last_stored_id"]

    # Restore PACTree
    tree = mod._tree
    tree._nodes.clear()
    tree._value_cache.clear()
    ts = state["tree"]
    for nid, ns in ts["nodes"].items():
        depth = ns["depth"]
        if isinstance(depth, int):
            depth = BifractalDepth(depth)
        elif isinstance(depth, str):
            depth = BifractalDepth[depth.upper()]
        node = MemoryNode(
            id=ns["id"],
            delta=ns["delta"],
            parent_id=ns["parent_id"],
            children_ids=list(ns["children_ids"]),
            strength=ns["strength"],
            depth=depth,
            access_count=ns["access_count"],
            created_at=ns["created_at"],
            label=ns["label"],
        )
        tree._nodes[int(nid)] = node
    tree._next_id = ts["next_id"]
    tree._root_ids = list(ts["root_ids"])
    tree._capacity = ts["capacity"]

    # Restore TransitionTracker (must use defaultdict to match original type)
    tt = mod._transitions
    tts = state["transitions"]
    tt._transitions.clear()
    for from_id, targets in tts["transitions"].items():
        for to_id, weight in targets.items():
            tt._transitions[int(from_id)][int(to_id)] = weight
    if hasattr(tt, '_best_next'):
        tt._best_next = {int(k): v for k, v in tts.get("best_next", {}).items()}


def _save_language_module(mod) -> dict:
    """Extract LanguageModule state."""
    state: dict[str, Any] = {
        "type": "language",
        "step_count": mod._step_count,
        "integer_mode": mod._integer_mode,
    }

    # Bin boundaries
    if mod._bin_boundaries is not None:
        state["bin_boundaries"] = mod._bin_boundaries.cpu().clone()

    # TransitionCounter
    counter = mod._counter
    state["counter"] = {
        "counts": dict(counter._counts),
        "totals": dict(counter._totals),
        "stats": {
            "total_transitions": counter.stats.total_transitions,
            "unique_contexts": counter.stats.unique_contexts,
            "unique_transitions": counter.stats.unique_transitions,
        },
    }

    # ConcentrationGate
    gate = mod._gate
    state["gate"] = {
        "total_analyzed": gate._total_analyzed,
        "high_quality_count": gate._high_quality_count,
        "concentration_sum": gate._concentration_sum,
    }

    # EmbeddingStore (if present)
    if mod._embeddings is not None:
        state["embeddings"] = {
            "tensor": mod._embeddings.embeddings.cpu().clone(),
            "vocab_size": mod._embeddings.vocab_size,
            "embed_dim": mod._embeddings.embed_dim,
        }

    return state


def _load_language_module(mod, state: dict) -> None:
    """Restore LanguageModule state."""
    from ..modules.language import TransitionStats

    mod._step_count = state["step_count"]
    mod._integer_mode = state["integer_mode"]

    if "bin_boundaries" in state:
        mod._bin_boundaries = state["bin_boundaries"]

    # Counter
    cs = state["counter"]
    mod._counter._counts = dict(cs["counts"])
    mod._counter._totals = dict(cs["totals"])
    mod._counter.stats = TransitionStats(
        total_transitions=cs["stats"]["total_transitions"],
        unique_contexts=cs["stats"]["unique_contexts"],
        unique_transitions=cs["stats"]["unique_transitions"],
    )

    # Gate
    gs = state["gate"]
    mod._gate._total_analyzed = gs["total_analyzed"]
    mod._gate._high_quality_count = gs["high_quality_count"]
    mod._gate._concentration_sum = gs["concentration_sum"]


def _save_observability_module(mod) -> dict:
    """Extract ObservabilityModule state."""
    state: dict[str, Any] = {
        "type": "observability",
        "step_count": mod._step_count,
    }

    # SCBFTracker
    tracker = mod._tracker
    state["tracker"] = {
        "step": tracker._step,
        "entropy_momentum": tracker._entropy_momentum,
        "raw_entropies": list(tracker._raw_entropies),
        "smoothed_entropies": list(tracker._smoothed_entropies),
        "stability_scores": list(tracker._stability_scores),
        "coherence_scores": list(tracker._coherence_scores),
        "recursion_scores": list(tracker._recursion_scores),
        "densities": list(tracker._densities),
        "top_consistency": list(tracker._top_consistency),
    }
    if tracker._prev_normalized is not None:
        state["tracker"]["prev_normalized"] = tracker._prev_normalized.cpu().clone()

    # QBEController
    qbe = mod._qbe
    state["qbe"] = {
        "momentum": qbe.momentum,
        "error_band": qbe.error_band,
        "energy_balance": qbe.energy_balance,
    }
    return state


def _load_observability_module(mod, state: dict) -> None:
    """Restore ObservabilityModule state."""
    mod._step_count = state["step_count"]

    ts = state["tracker"]
    tracker = mod._tracker
    tracker._step = ts["step"]
    tracker._entropy_momentum = ts["entropy_momentum"]
    tracker._raw_entropies = deque(ts["raw_entropies"], maxlen=tracker._window)
    tracker._smoothed_entropies = deque(ts["smoothed_entropies"], maxlen=tracker._window)
    tracker._stability_scores = deque(ts["stability_scores"], maxlen=tracker._window)
    tracker._coherence_scores = deque(ts["coherence_scores"], maxlen=tracker._window)
    tracker._recursion_scores = deque(ts["recursion_scores"], maxlen=tracker._window)
    tracker._densities = deque(ts["densities"], maxlen=tracker._window)
    tracker._top_consistency = deque(ts["top_consistency"], maxlen=tracker._window)
    if "prev_normalized" in ts:
        tracker._prev_normalized = ts["prev_normalized"]

    qs = state["qbe"]
    mod._qbe.momentum = qs["momentum"]
    mod._qbe.error_band = qs["error_band"]
    mod._qbe.energy_balance = qs["energy_balance"]


# ─── Module Dispatch ──────────────────────────────────────────────

_SAVE_DISPATCH = {
    "SafetyModule": _save_safety_module,
    "ReasoningModule": _save_reasoning_module,
    "MemoryModule": _save_memory_module,
    "LanguageModule": _save_language_module,
    "ObservabilityModule": _save_observability_module,
}

_LOAD_DISPATCH = {
    "safety": _load_safety_module,
    "reasoning": _load_reasoning_module,
    "memory": _load_memory_module,
    "language": _load_language_module,
    "observability": _load_observability_module,
}


def _save_module(mod) -> dict:
    """Save any GAIAModule's state."""
    cls_name = type(mod).__name__
    if cls_name in _SAVE_DISPATCH:
        return _SAVE_DISPATCH[cls_name](mod)
    # Unknown module -- save what we can
    return {"type": cls_name, "warning": "no custom serializer"}


def _load_module(mod, state: dict) -> None:
    """Load any GAIAModule's state."""
    mod_type = state.get("type", "")
    if mod_type in _LOAD_DISPATCH:
        _LOAD_DISPATCH[mod_type](mod, state)


# ─── Bus State ────────────────────────────────────────────────────

def _save_bus_state(bus: CoupledFieldsBus) -> dict:
    """Save CoupledFieldsBus internal state."""
    field_states = {}
    for name, fs in bus._field_states.items():
        field_states[name] = {
            "tensor": fs.tensor.cpu().clone(),
            "lens": fs.lens.cpu().clone(),
            "prediction_error": fs.prediction_error,
            "adaptation_rate": fs.adaptation_rate,
            "ticks_alive": fs.ticks_alive,
            "surprise_history": list(fs.surprise_history),
        }

    modules_state = {}
    for name, mod in bus._modules.items():
        modules_state[name] = _save_module(mod)

    return {
        "tick": bus._tick,
        "enforcement": bus._enforcement,
        "field_states": field_states,
        "modules": modules_state,
        "n_violations": len(bus._violation_log),
    }


def _load_bus_state(bus: CoupledFieldsBus, state: dict) -> None:
    """Restore CoupledFieldsBus internal state."""
    bus._tick = state["tick"]

    # Restore field states
    for name, fs_data in state["field_states"].items():
        bus._field_states[name] = CoupledFieldState(
            tensor=fs_data["tensor"],
            lens=fs_data["lens"],
            prediction_error=fs_data["prediction_error"],
            adaptation_rate=fs_data["adaptation_rate"],
            ticks_alive=fs_data["ticks_alive"],
            surprise_history=list(fs_data["surprise_history"]),
        )

    # Restore module internal state
    for name, mod_state in state["modules"].items():
        if name in bus._modules:
            _load_module(bus._modules[name], mod_state)


# ─── Cell/Colony Save/Load (v2 — GrowingColony with tree topology) ──

def _save_cell(cell) -> dict:
    """Save a single Cell's full state."""
    state = {
        "name": cell.name,
        "maturity": cell.maturity,  # IntEnum serializes as int
        "access_count": cell.access_count,
        "idle_ticks": cell.idle_ticks,
        "birth_tick": cell.birth_tick,
        "voice": cell.voice.cpu().clone(),
        "connections": dict(cell.connections),
        "spec_history": cell.spec_history,
        "activation_history": cell.activation_history,
        "total_activations": cell.total_activations,
        # Tree structure (names, not references)
        "parent_name": cell.parent.name if cell.parent is not None else None,
        "children_names": [c.name for c in cell.children],
        # Agent state
        "agent": {
            "name": cell.agent.name,
            "field_dim": cell.agent._identity.field_dim,
            "bus": _save_bus_state(cell.agent.bus),
            "identity": {
                "resonance_field": cell.agent.identity.resonance_field.cpu().clone(),
                "spectral_lens": cell.agent.identity.spectral_lens.cpu().clone(),
                "experience": cell.agent.identity.experience,
                "surprise_history": list(cell.agent.identity.surprise_history),
            },
        },
    }

    # Last signal
    if cell.last_signal is not None:
        state["last_signal"] = {
            "sender": cell.last_signal.sender,
            "tensor": cell.last_signal.tensor.cpu().clone(),
            "tick": cell.last_signal.tick,
        }

    return state


def save_colony(colony, path: str | Path) -> None:
    """Save entire GrowingColony state to a checkpoint file.

    Saves: all cell states (modules, bus, voice, connections, maturity,
    activation history), tree topology (parent/children), colony tick,
    growth log, population/lobe history.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "version": 2,
        "tick": colony.tick,
        "next_id": colony._next_id,
        "roots": list(colony.roots),
        "cells": {},
        "growth_log": colony.growth_log,
        "population_history": colony.population_history,
        "lobe_history": colony.lobe_history,
    }

    for name, cell in colony.cells.items():
        state["cells"][name] = _save_cell(cell)

    torch.save(state, path)
    n_cells = len(state["cells"])
    n_lobes = len(state["roots"])
    logger.info(f"Colony saved to {path} ({n_cells} cells, {n_lobes} lobes, tick {colony.tick})")


def load_colony(path: str | Path, cell_class, colony_class,
                make_organism_fn, signal_class=None):
    """Load a GrowingColony from a checkpoint file.

    Args:
        path: checkpoint file path
        cell_class: Cell class from the spike module
        colony_class: GrowingColony class
        make_organism_fn: function(name) -> GAIAAgent (creates fresh agent)
        signal_class: Signal dataclass (sender, tensor, tick)

    Returns:
        Restored GrowingColony with full tree topology and state.
    """
    path = Path(path)
    state = torch.load(path, weights_only=False)

    version = state.get("version", 1)
    if version < 2:
        raise ValueError(f"Checkpoint version {version} is not compatible with "
                         "GrowingColony. Use v2+ checkpoints.")

    # Pass 1: Create all cells (without tree links)
    cells: dict[str, Any] = {}
    for name, cell_state in state["cells"].items():
        cell = cell_class.__new__(cell_class)
        cell.agent = make_organism_fn(name)
        cell.parent = None
        cell.children = []
        cell.maturity = cell_state["maturity"]
        cell.access_count = cell_state["access_count"]
        cell.idle_ticks = cell_state["idle_ticks"]
        cell.birth_tick = cell_state["birth_tick"]
        cell.voice = cell_state["voice"]
        cell.connections = dict(cell_state["connections"])
        cell.spec_history = cell_state.get("spec_history", [])
        cell.activation_history = cell_state.get("activation_history", [])
        cell.total_activations = cell_state.get("total_activations", 0)

        # Restore agent bus state
        agent_state = cell_state["agent"]
        _load_bus_state(cell.agent.bus, agent_state["bus"])

        # Restore identity
        id_state = agent_state["identity"]
        cell.agent.identity.resonance_field = id_state["resonance_field"]
        cell.agent.identity.spectral_lens = id_state["spectral_lens"]
        cell.agent.identity.experience = id_state["experience"]
        cell.agent.identity.surprise_history = list(id_state["surprise_history"])

        # Restore last signal
        if "last_signal" in cell_state and signal_class is not None:
            ls = cell_state["last_signal"]
            cell.last_signal = signal_class(
                sender=ls["sender"], tensor=ls["tensor"], tick=ls["tick"],
            )
        else:
            cell.last_signal = None

        cells[name] = (cell, cell_state)

    # Pass 2: Wire tree topology (parent/children references)
    for name, (cell, cell_state) in cells.items():
        parent_name = cell_state.get("parent_name")
        if parent_name is not None and parent_name in cells:
            cell.parent = cells[parent_name][0]
        for child_name in cell_state.get("children_names", []):
            if child_name in cells:
                cell.children.append(cells[child_name][0])

    # Build colony
    colony = colony_class.__new__(colony_class)
    colony.cells = {name: cell for name, (cell, _) in cells.items()}
    colony.roots = list(state["roots"])
    colony.tick = state["tick"]
    colony._next_id = state["next_id"]
    colony.growth_log = state.get("growth_log", [])
    colony.population_history = state.get("population_history", [])
    colony.lobe_history = state.get("lobe_history", [])
    colony._last_activation_ratio = 1.0

    n_cells = len(colony.cells)
    n_lobes = len(colony.roots)
    logger.info(f"Colony loaded from {path} ({n_cells} cells, {n_lobes} lobes, tick {colony.tick})")
    return colony


def checkpoint_info(path: str | Path) -> dict:
    """Read checkpoint metadata without loading full state."""
    state = torch.load(path, weights_only=False)
    version = state.get("version", 0)
    info = {
        "version": version,
        "tick": state["tick"],
    }
    if version >= 2:
        # GrowingColony format
        info["n_cells"] = len(state.get("cells", {}))
        info["n_lobes"] = len(state.get("roots", []))
        info["cells"] = {}
        for name, cs in state.get("cells", {}).items():
            agent = cs.get("agent", {})
            bus = agent.get("bus", {})
            info["cells"][name] = {
                "maturity": cs.get("maturity", 0),
                "access_count": cs.get("access_count", 0),
                "total_activations": cs.get("total_activations", 0),
                "parent": cs.get("parent_name"),
                "n_children": len(cs.get("children_names", [])),
                "n_connections": len(cs.get("connections", {})),
                "modules": list(bus.get("modules", {}).keys()),
            }
        births = sum(1 for e in state.get("growth_log", []) if "birth" in e.get("event", ""))
        deaths = sum(1 for e in state.get("growth_log", [])
                     if e.get("event") in ("death", "subtree_death"))
        info["total_births"] = births
        info["total_deaths"] = deaths
    else:
        # Legacy format
        info["organisms"] = {}
        for name, org_state in state.get("organisms", {}).items():
            agent = org_state.get("agent", {})
            bus = agent.get("bus", {})
            info["organisms"][name] = {
                "modules": list(bus.get("modules", {}).keys()),
                "experience": agent.get("identity", {}).get("experience", 0),
            }
    return info
