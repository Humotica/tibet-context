"""
ContextReader — capability-filtered reading of containers.

Reads only the layers a model is authorized to access,
formatting them for injection into a model's context window.
"""

from typing import Optional

from .container import ContextContainer
from .gate import CapabilityGate
from .layers import CapabilityProfile, Layer


class ContextReader:
    """
    Reads context containers filtered by model capability.

    The reader acts as the bridge between the container and the model:
    it unlocks accessible layers and formats them for consumption.
    """

    def __init__(self, gate: Optional[CapabilityGate] = None):
        self.gate = gate or CapabilityGate()

    def read(self, container: ContextContainer, model_id: str) -> str:
        """
        Read container content filtered by model capability.

        Args:
            container: The context container
            model_id: Model identifier

        Returns:
            Formatted string with accessible context
        """
        layers = self.gate.unlock_layers(container, model_id)

        if not layers:
            return ""

        sections = []
        for layer in layers:
            header = _layer_header(layer.level)
            sections.append(f"=== {header} ===\n{layer.content}")

        return "\n\n".join(sections)

    def read_layer(
        self,
        container: ContextContainer,
        model_id: str,
        level: int,
    ) -> Optional[str]:
        """
        Read a specific layer if accessible.

        Args:
            container: The context container
            model_id: Model identifier
            level: Layer level to read

        Returns:
            Layer content string, or None if not accessible
        """
        capability = self.gate.check_capability(model_id)
        layer = container.get_layer(level)

        if layer is None:
            return None

        if layer.min_capability > capability:
            return None

        return layer.content

    def read_for_capability(self, container: ContextContainer, capability: int) -> str:
        """
        Read container using a raw capability value (no model lookup).

        Args:
            container: The context container
            capability: Model capability in B params

        Returns:
            Formatted string with accessible context
        """
        layers = container.available_layers(capability)

        if not layers:
            return ""

        sections = []
        for layer in layers:
            header = _layer_header(layer.level)
            sections.append(f"=== {header} ===\n{layer.content}")

        return "\n\n".join(sections)

    def accessible_token_count(self, container: ContextContainer, model_id: str) -> int:
        """Get total tokens accessible for a model."""
        layers = self.gate.unlock_layers(container, model_id)
        return sum(layer.token_count for layer in layers)

    def summary(self, container: ContextContainer, model_id: str) -> dict:
        """
        Get a reading summary for a model.

        Returns:
            Dict with accessible info, blocked layers, token counts
        """
        report = self.gate.gate_report(container, model_id)
        layers = self.gate.unlock_layers(container, model_id)

        return {
            "model_id": model_id,
            "capability": report["capability"],
            "accessible_levels": report["accessible_levels"],
            "blocked_levels": report["blocked_levels"],
            "accessible_tokens": sum(l.token_count for l in layers),
            "total_tokens": container.total_tokens(),
            "carbonara_pass": report["carbonara_pass"],
        }


def _layer_header(level: int) -> str:
    """Get human-readable header for a layer level."""
    headers = {
        0: "L0: Summary",
        1: "L1: Conversation",
        2: "L2: Deep Context",
    }
    return headers.get(level, f"L{level}: Custom")
