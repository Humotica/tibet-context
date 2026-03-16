"""
Layer definitions and configurable capability profiles.

The Blu-ray model: each layer requires a minimum model capability to unlock.
Profiles are fully configurable — no hardcoded thresholds.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Layer:
    """
    A single context layer.

    Attributes:
        level: Layer level (0=summary, 1=conversation, 2=deep context)
        content: The actual content text
        token_count: Estimated token count
        min_capability: Minimum model size in billions of parameters
        content_hash: SHA-256 of content for integrity
        tibet_tokens: Token IDs that contributed to this layer
    """
    level: int
    content: str
    token_count: int
    min_capability: int
    content_hash: str = ""
    tibet_tokens: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.content_hash:
            computed = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
            object.__setattr__(self, "content_hash", computed)

    def verify(self) -> bool:
        """Verify content integrity."""
        expected = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        return self.content_hash == expected

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "content": self.content,
            "token_count": self.token_count,
            "min_capability": self.min_capability,
            "content_hash": self.content_hash,
            "tibet_tokens": list(self.tibet_tokens),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Layer":
        return cls(**data)


@dataclass(frozen=True)
class LayerSpec:
    """
    Specification for a layer within a profile.

    Attributes:
        level: Layer level
        min_capability: Minimum model capability (B params) to access
        max_tokens: Maximum tokens allowed in this layer
    """
    level: int
    min_capability: int
    max_tokens: int


@dataclass
class CapabilityProfile:
    """
    Configurable capability profile — no hardcoded thresholds.

    Defines what model size is needed to access each layer.
    Different model families can define their own profiles.
    """
    name: str
    layers: Dict[int, LayerSpec] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "CapabilityProfile":
        """HumoticaOS Qwen profile (reference)."""
        return cls(name="qwen", layers={
            0: LayerSpec(level=0, min_capability=3, max_tokens=512),
            1: LayerSpec(level=1, min_capability=14, max_tokens=4096),
            2: LayerSpec(level=2, min_capability=32, max_tokens=16384),
        })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapabilityProfile":
        """Load from config dict."""
        name = data.get("name", "custom")
        layers_data = data.get("layers", {})
        layers = {}
        for level_str, spec_data in layers_data.items():
            level = int(level_str)
            layers[level] = LayerSpec(
                level=level,
                min_capability=spec_data["min_capability"],
                max_tokens=spec_data["max_tokens"],
            )
        return cls(name=name, layers=layers)

    @classmethod
    def from_file(cls, path: str) -> "CapabilityProfile":
        """Load from .toml or .json config file."""
        file_path = Path(path)
        text = file_path.read_text(encoding="utf-8")

        if file_path.suffix == ".json":
            data = json.loads(text)
            profile_data = data.get("profile", data)
            return cls.from_dict(profile_data)

        if file_path.suffix == ".toml":
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore[no-redef]
            data = tomllib.loads(text)
            profile_data = data.get("profile", data)
            return cls.from_dict(profile_data)

        raise ValueError(f"Unsupported config format: {file_path.suffix}")

    def get_spec(self, level: int) -> Optional[LayerSpec]:
        """Get layer spec for a given level."""
        return self.layers.get(level)

    def can_access(self, level: int, capability: int) -> bool:
        """Check if a model with given capability can access a layer level."""
        spec = self.layers.get(level)
        if spec is None:
            return False
        return capability >= spec.min_capability

    def accessible_levels(self, capability: int) -> list[int]:
        """Get all layer levels accessible at a given capability."""
        return sorted(
            level for level, spec in self.layers.items()
            if capability >= spec.min_capability
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "layers": {
                str(level): {
                    "min_capability": spec.min_capability,
                    "max_tokens": spec.max_tokens,
                }
                for level, spec in self.layers.items()
            },
        }


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token."""
    return max(1, len(text) // 4)
