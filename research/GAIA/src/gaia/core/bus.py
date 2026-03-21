"""ConservationBus — PAC enforcement, SEC routing, RBF regulation."""

from __future__ import annotations

import logging
from typing import Optional

from .exceptions import ConservationViolation, ModuleRegistrationError
from .protocol import GAIAModule
from .sec_router import SECRouter
from .types import ConservationResult, FieldState, RBFBalance, SECPhase

logger = logging.getLogger(__name__)


class ConservationBus:
    """Central nervous system of GAIA v2.

    Receives FieldState, routes to modules via SEC phase classification,
    validates PAC conservation at every module boundary, and regulates
    module activity via RBF balance.

    Enforcement modes:
        "hard":    Raises ConservationViolation on PAC failure.
        "soft":    Logs violation, continues processing.
        "monitor": Logs violation, no correction. For development.
    """

    def __init__(
        self,
        enforcement: str = "hard",
        tolerance: float = 1e-6,
        rbf_lambda: float = 1.0,
        rbf_alpha: float = 0.1,
        rbf_suppression_threshold: float = 0.0,
    ) -> None:
        if enforcement not in ("hard", "soft", "monitor"):
            raise ValueError(f"enforcement must be 'hard', 'soft', or 'monitor', got '{enforcement}'")

        self._enforcement = enforcement
        self._tolerance = tolerance
        self._rbf_lambda = rbf_lambda
        self._rbf_alpha = rbf_alpha
        self._rbf_suppression_threshold = rbf_suppression_threshold
        self._router = SECRouter()
        self._modules: dict[str, GAIAModule] = {}
        self._violation_log: list[ConservationResult] = []

    def register_module(
        self,
        module: GAIAModule,
        phases: list[SECPhase] | None = None,
    ) -> None:
        """Register a module with the bus.

        Args:
            module: Module satisfying GAIAModule protocol.
            phases: SEC phases this module handles. None = all phases.

        Raises:
            ModuleRegistrationError: If module doesn't satisfy protocol.
        """
        if not isinstance(module, GAIAModule):
            raise ModuleRegistrationError(
                f"Object does not satisfy GAIAModule protocol: {type(module).__name__}"
            )
        self._modules[module.name] = module
        self._router.register(module, phases)

    def process(self, field_state: FieldState) -> FieldState:
        """Full bus pipeline: classify -> route -> regulate -> dispatch -> validate.

        Args:
            field_state: Input state to process.

        Returns:
            Processed FieldState (may be unchanged if no modules match).

        Raises:
            ConservationViolation: In hard mode, if any module violates PAC.
        """
        # 1. Classify SEC phase and update field state
        phase = self._router.classify(field_state.entropy)
        field_state = field_state.clone()
        field_state.phase = phase

        # 2. Route to modules
        modules = self._router.route(field_state)

        if not modules:
            return field_state

        # 3. Regulate — suppress unhealthy modules
        modules = self._regulate(modules)

        if not modules:
            return field_state

        # 4. Dispatch sequentially, validating PAC at each boundary
        current_state = field_state
        for module in modules:
            output_state = module.process(current_state)
            result = self._validate_conservation(current_state, output_state, module.name)

            if not result.conserved:
                self._handle_violation(result)

            current_state = output_state

        return current_state

    def _validate_conservation(
        self,
        input_state: FieldState,
        output_state: FieldState,
        module_name: str,
    ) -> ConservationResult:
        """Validate PAC conservation at a module boundary.

        Checks that total_energy is preserved within tolerance,
        accounting for any declared conservation_budget.
        """
        input_energy = input_state.total_energy()
        output_energy = output_state.total_energy()

        # Account for conservation budget (explicit entropy consumption/production)
        budget_delta = output_state.conservation_budget - input_state.conservation_budget
        adjusted_residual = abs(input_energy - output_energy) - abs(budget_delta)
        residual = max(0.0, adjusted_residual)

        # Relative tolerance: compare residual against energy magnitude
        energy_scale = max(abs(input_energy), abs(output_energy), 1e-10)
        conserved = (residual / energy_scale) < self._tolerance

        result = ConservationResult(
            conserved=conserved,
            input_energy=input_energy,
            output_energy=output_energy,
            residual=residual,
            module_name=module_name,
            violation_type=None if conserved else self._enforcement,
        )

        if not conserved:
            self._violation_log.append(result)

        return result

    def _regulate(self, modules: list[GAIAModule]) -> list[GAIAModule]:
        """Filter modules by RBF health. Suppress unhealthy modules."""
        healthy = []
        for module in modules:
            rbf = module.health()
            if rbf.balance >= self._rbf_suppression_threshold:
                healthy.append(module)
            else:
                logger.info(
                    "Module '%s' suppressed: RBF balance %.4f < threshold %.4f",
                    module.name, rbf.balance, self._rbf_suppression_threshold,
                )
        return healthy

    def _handle_violation(self, result: ConservationResult) -> None:
        """Handle a conservation violation based on enforcement mode."""
        msg = (
            f"PAC violation in '{result.module_name}': "
            f"residual={result.residual:.6e} "
            f"(in={result.input_energy:.6f}, out={result.output_energy:.6f})"
        )

        if self._enforcement == "hard":
            raise ConservationViolation(result, result.module_name)
        elif self._enforcement == "soft":
            logger.warning("SOFT: %s", msg)
        else:  # monitor
            logger.info("MONITOR: %s", msg)

    @property
    def violation_log(self) -> list[ConservationResult]:
        """All conservation violations recorded during processing."""
        return list(self._violation_log)

    @property
    def enforcement(self) -> str:
        """Current enforcement mode."""
        return self._enforcement

    def get_metrics(self) -> dict:
        """Bus operational metrics."""
        return {
            "modules_registered": len(self._modules),
            "module_names": list(self._modules.keys()),
            "enforcement": self._enforcement,
            "tolerance": self._tolerance,
            "total_violations": len(self._violation_log),
        }
