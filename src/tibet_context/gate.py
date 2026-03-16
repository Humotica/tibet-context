"""
JIS Capability Gate — determines which layers a model can access.

Like AACS keys for Blu-ray: the gate checks model capability
and unlocks the appropriate context layers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .container import ContextContainer
from .layers import CapabilityProfile, Layer


# Known model capabilities (billions of parameters)
# Users can register custom models or override these
_DEFAULT_MODEL_REGISTRY: Dict[str, int] = {
    # Qwen family
    "qwen2.5:3b": 3,
    "qwen2.5:7b": 7,
    "qwen2.5:14b": 14,
    "qwen2.5:32b": 32,
    "qwen2.5:72b": 72,
    # Llama family
    "llama3.2:1b": 1,
    "llama3.2:3b": 3,
    "llama3.1:8b": 8,
    "llama3.1:70b": 70,
    "llama3.1:405b": 405,
    # Mistral family
    "mistral:7b": 7,
    "mixtral:8x7b": 56,
    "mistral-large": 123,
    # Claude family
    "claude-haiku": 20,
    "claude-sonnet": 70,
    "claude-opus": 200,
}


@dataclass
class CapabilityGate:
    """
    JIS capability gate for context layer access control.

    Checks model capability against profile thresholds
    to determine which layers can be unlocked.
    """
    profile: CapabilityProfile = field(default_factory=CapabilityProfile.default)
    model_registry: Dict[str, int] = field(default_factory=lambda: dict(_DEFAULT_MODEL_REGISTRY))

    def register_model(self, model_id: str, capability: int) -> None:
        """Register a model with its capability (B params)."""
        self.model_registry[model_id] = capability

    def check_capability(self, model_id: str) -> int:
        """
        Get capability for a model.

        Args:
            model_id: Model identifier (e.g., "qwen2.5:7b")

        Returns:
            Capability in billions of parameters, 0 if unknown

        Raises:
            ValueError: If model_id is not registered
        """
        cap = self.model_registry.get(model_id)
        if cap is None:
            raise ValueError(
                f"Unknown model: {model_id}. "
                f"Register it with gate.register_model('{model_id}', capability_B)"
            )
        return cap

    def unlock_layers(self, container: ContextContainer, model_id: str) -> List[Layer]:
        """
        Unlock accessible layers for a model.

        Args:
            container: The context container
            model_id: Model identifier

        Returns:
            List of accessible Layer objects, ordered by level
        """
        capability = self.check_capability(model_id)
        accessible = []
        for level in sorted(container.layers.keys()):
            if self.profile.can_access(level, capability):
                accessible.append(container.layers[level])
        return accessible

    def max_accessible_level(self, model_id: str) -> int:
        """Get the highest layer level a model can access."""
        capability = self.check_capability(model_id)
        levels = self.profile.accessible_levels(capability)
        return max(levels) if levels else -1

    def carbonara_test(self, model_id: str, required_level: int = 2) -> bool:
        """
        Carbonara test: can this model handle deep context?

        The "zakjapanner" test — a small model that can't be trusted
        with complex tasks. Returns True only if the model can access
        the required layer level (default: L2 deep context).

        Args:
            model_id: Model to test
            required_level: Minimum required layer level (default 2)

        Returns:
            True if model passes the carbonara test
        """
        capability = self.check_capability(model_id)
        return self.profile.can_access(required_level, capability)

    def gate_report(self, container: ContextContainer, model_id: str) -> Dict[str, Any]:
        """
        Generate a gate report for audit/debugging.

        Returns:
            Dict with capability, accessible levels, blocked levels, etc.
        """
        capability = self.check_capability(model_id)
        accessible = []
        blocked = []

        for level in sorted(container.layers.keys()):
            if self.profile.can_access(level, capability):
                accessible.append(level)
            else:
                blocked.append(level)

        return {
            "model_id": model_id,
            "capability": capability,
            "profile": self.profile.name,
            "accessible_levels": accessible,
            "blocked_levels": blocked,
            "carbonara_pass": self.carbonara_test(model_id),
            "total_accessible_tokens": sum(
                container.layers[l].token_count for l in accessible
            ),
        }
