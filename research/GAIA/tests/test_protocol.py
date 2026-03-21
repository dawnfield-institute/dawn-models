"""Tests for the GAIAModule protocol."""

from gaia.core.protocol import GAIAModule
from gaia.core.types import SECPhase
from tests.conftest import IdentityModule, ScalingModule


class TestGAIAModuleProtocol:
    def test_identity_module_satisfies_protocol(self):
        module = IdentityModule()
        assert isinstance(module, GAIAModule)

    def test_scaling_module_satisfies_protocol(self):
        module = ScalingModule(factor=2.0)
        assert isinstance(module, GAIAModule)

    def test_plain_object_fails_protocol(self):
        class NotAModule:
            pass

        assert not isinstance(NotAModule(), GAIAModule)

    def test_partial_implementation_fails(self):
        class PartialModule:
            @property
            def name(self) -> str:
                return "partial"

            def process(self, field_state):
                return field_state
            # Missing phase() and health()

        assert not isinstance(PartialModule(), GAIAModule)

    def test_module_has_name(self):
        module = IdentityModule()
        assert module.name == "identity"

    def test_module_has_phase(self):
        module = IdentityModule()
        assert module.phase() == SECPhase.ORDERED
