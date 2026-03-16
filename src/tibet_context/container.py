"""
ContextContainer — the core layered context unit.

A container holds multiple layers (L0, L1, L2) plus a reference
to the TIBET provenance chain that tracks its creation and usage.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .layers import Layer


@dataclass(frozen=True)
class ContextContainer:
    """
    Layered context container.

    Attributes:
        id: Unique container ID
        layers: Mapping of level -> Layer
        tibet_chain_id: Root token ID of the TIBET provenance chain
        created_at: ISO timestamp of creation
        source_session: Session that created this container
        metadata: Extensible metadata dict
    """
    id: str
    layers: Dict[int, Layer]
    tibet_chain_id: str
    created_at: str = ""
    source_session: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now().isoformat())

    def get_layer(self, level: int) -> Optional[Layer]:
        """Get a specific layer by level."""
        return self.layers.get(level)

    def available_layers(self, capability: int) -> List[Layer]:
        """Get all layers accessible at a given model capability."""
        return [
            layer for level, layer in sorted(self.layers.items())
            if layer.min_capability <= capability
        ]

    def token_count(self) -> Dict[int, int]:
        """Get token count per layer level."""
        return {level: layer.token_count for level, layer in self.layers.items()}

    def total_tokens(self) -> int:
        """Total tokens across all layers."""
        return sum(layer.token_count for layer in self.layers.values())

    def verify_integrity(self) -> bool:
        """Verify content hashes for all layers."""
        return all(layer.verify() for layer in self.layers.values())

    def layer_count(self) -> int:
        """Number of layers in this container."""
        return len(self.layers)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "layers": {str(k): v.to_dict() for k, v in self.layers.items()},
            "tibet_chain_id": self.tibet_chain_id,
            "created_at": self.created_at,
            "source_session": self.source_session,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextContainer":
        """Deserialize from dictionary."""
        layers = {
            int(k): Layer.from_dict(v) for k, v in data["layers"].items()
        }
        return cls(
            id=data["id"],
            layers=layers,
            tibet_chain_id=data["tibet_chain_id"],
            created_at=data.get("created_at", ""),
            source_session=data.get("source_session"),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        levels = sorted(self.layers.keys())
        total = self.total_tokens()
        return f"ContextContainer(id={self.id[:12]}..., layers={levels}, tokens={total})"
