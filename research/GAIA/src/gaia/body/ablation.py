"""Ablation framework — systematic module removal for behavioral analysis.

Runs all 4 behavioral scenarios with different module configurations
to reveal each module's contribution to behavioral coherence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from gaia.core.bus import ConservationBus
from gaia.core.coupled_fields_bus import CoupledFieldsBus
from gaia.core.resonance_bus import ResonanceBus
from gaia.modules.language import LanguageModule
from gaia.modules.memory import MemoryModule
from gaia.modules.observability import ObservabilityModule
from gaia.modules.reasoning import ReasoningModule
from gaia.modules.safety import SafetyModule

from .scenarios import run_adaptation, run_habituation, run_novelty, run_preference

# ─── Module Registry ──────────────────────────────────────────────

MODULE_REGISTRY: dict[str, Callable[[int], object]] = {
    "observability": lambda dim: ObservabilityModule(),
    "safety": lambda dim: SafetyModule(input_dim=dim),
    "reasoning": lambda dim: ReasoningModule(input_dim=dim),
    "memory": lambda dim: MemoryModule(),
    "language": lambda dim: LanguageModule(),
}

ALL_MODULES = list(MODULE_REGISTRY.keys())


def make_bus(
    module_names: list[str],
    input_dim: int = 22,
    enforcement: str = "soft",
) -> ConservationBus:
    """Create a ConservationBus with only the named modules registered."""
    bus = ConservationBus(enforcement=enforcement)
    for name in module_names:
        factory = MODULE_REGISTRY[name]
        bus.register_module(factory(input_dim))
    return bus


def make_resonance_bus(
    module_names: list[str],
    input_dim: int = 22,
    enforcement: str = "soft",
) -> ResonanceBus:
    """Create a ResonanceBus with only the named modules registered."""
    bus = ResonanceBus(enforcement=enforcement)
    for name in module_names:
        factory = MODULE_REGISTRY[name]
        bus.register_module(factory(input_dim))
    return bus


def make_coupled_fields_bus(
    module_names: list[str],
    input_dim: int = 22,
    enforcement: str = "soft",
    **kwargs,
) -> CoupledFieldsBus:
    """Create a CoupledFieldsBus with only the named modules registered."""
    bus = CoupledFieldsBus(enforcement=enforcement, **kwargs)
    for name in module_names:
        factory = MODULE_REGISTRY[name]
        bus.register_module(factory(input_dim))
    return bus


# ─── Configuration ────────────────────────────────────────────────


@dataclass
class ModuleConfig:
    """A named set of modules to register on the bus."""

    name: str
    modules: list[str]
    description: str = ""


def standard_configs() -> list[ModuleConfig]:
    """Generate the 14 standard ablation configurations."""
    configs: list[ModuleConfig] = []

    # Baseline: all 5 modules
    configs.append(ModuleConfig(
        name="full",
        modules=list(ALL_MODULES),
        description="All 5 modules (baseline)",
    ))

    # Leave-one-out (always keep observability)
    for mod in ["safety", "reasoning", "memory", "language"]:
        remaining = [m for m in ALL_MODULES if m != mod]
        configs.append(ModuleConfig(
            name=f"no_{mod}",
            modules=remaining,
            description=f"All modules except {mod}",
        ))

    # Control: observability only
    configs.append(ModuleConfig(
        name="observability_only",
        modules=["observability"],
        description="Identity baseline (observability only)",
    ))

    # Selected pairs (observability + 2)
    pairs = [
        ("safety", "reasoning"),
        ("memory", "language"),
        ("safety", "memory"),
        ("reasoning", "language"),
    ]
    for a, b in pairs:
        configs.append(ModuleConfig(
            name=f"{a}+{b}",
            modules=["observability", a, b],
            description=f"Observability + {a} + {b}",
        ))

    # Singles (observability + 1)
    for mod in ["safety", "reasoning", "memory", "language"]:
        configs.append(ModuleConfig(
            name=f"{mod}_only",
            modules=["observability", mod],
            description=f"Observability + {mod} only",
        ))

    return configs


# ─── Results ──────────────────────────────────────────────────────


@dataclass
class AblationResult:
    """Results from running all 4 scenarios with one module configuration."""

    config: ModuleConfig
    metrics: dict[str, float] = field(default_factory=dict)
    provenance_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class AblationReport:
    """Full ablation study results."""

    results: list[AblationResult] = field(default_factory=list)
    baseline: AblationResult | None = None

    def metric_matrix(self) -> dict[str, dict[str, float]]:
        """config_name -> metric_name -> value."""
        return {r.config.name: dict(r.metrics) for r in self.results}

    def delta_matrix(self) -> dict[str, dict[str, float]]:
        """config_name -> metric_name -> (value - baseline_value)."""
        if self.baseline is None:
            return {}
        deltas: dict[str, dict[str, float]] = {}
        for r in self.results:
            if r.config.name == self.baseline.config.name:
                continue
            row: dict[str, float] = {}
            for key, val in r.metrics.items():
                baseline_val = self.baseline.metrics.get(key, 0.0)
                row[key] = val - baseline_val
            deltas[r.config.name] = row
        return deltas


# ─── Runner ───────────────────────────────────────────────────────


def run_ablation_config(
    config: ModuleConfig,
    input_dim: int = 22,
    seed: int = 42,
) -> AblationResult:
    """Run all 4 behavioral scenarios with a given module configuration."""
    metrics: dict[str, float] = {}

    # Each scenario gets a fresh bus (matching bench_behavioral pattern)
    bus = make_bus(config.modules, input_dim=input_dim)
    metrics.update(run_habituation(bus, n_ticks=50, seed=seed))

    bus = make_bus(config.modules, input_dim=input_dim)
    metrics.update(run_novelty(bus, n_familiar=20, seed=seed))

    bus = make_bus(config.modules, input_dim=input_dim)
    metrics.update(run_adaptation(bus, n_before=25, n_after=25, seed=seed))

    bus = make_bus(config.modules, input_dim=input_dim)
    metrics.update(run_preference(bus, n_ticks_per_class=30, seed=seed))

    return AblationResult(config=config, metrics=metrics)


def run_full_ablation(
    configs: list[ModuleConfig] | None = None,
    input_dim: int = 22,
    seed: int = 42,
) -> AblationReport:
    """Run the complete ablation study across all configurations."""
    if configs is None:
        configs = standard_configs()

    report = AblationReport()
    for config in configs:
        result = run_ablation_config(config, input_dim=input_dim, seed=seed)
        report.results.append(result)
        if config.name == "full":
            report.baseline = result

    return report
