"""Tests for layers module — Layer, LayerSpec, CapabilityProfile."""

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from tibet_context.layers import (
    Layer,
    LayerSpec,
    CapabilityProfile,
    estimate_tokens,
)


class TestLayer:
    def test_create_layer(self):
        layer = Layer(level=0, content="Hello world", token_count=3, min_capability=3)
        assert layer.level == 0
        assert layer.content == "Hello world"
        assert layer.token_count == 3
        assert layer.min_capability == 3

    def test_auto_hash(self):
        layer = Layer(level=0, content="test content", token_count=3, min_capability=3)
        expected = hashlib.sha256(b"test content").hexdigest()
        assert layer.content_hash == expected

    def test_verify_ok(self):
        layer = Layer(level=0, content="test", token_count=1, min_capability=3)
        assert layer.verify() is True

    def test_verify_fail(self):
        layer = Layer(
            level=0, content="test", token_count=1,
            min_capability=3, content_hash="badhash"
        )
        assert layer.verify() is False

    def test_immutable(self):
        layer = Layer(level=0, content="test", token_count=1, min_capability=3)
        with pytest.raises(AttributeError):
            layer.level = 1  # type: ignore

    def test_to_dict(self):
        layer = Layer(level=1, content="data", token_count=1, min_capability=14)
        d = layer.to_dict()
        assert d["level"] == 1
        assert d["content"] == "data"
        assert d["min_capability"] == 14
        assert "content_hash" in d

    def test_from_dict_roundtrip(self):
        layer = Layer(level=2, content="deep", token_count=1, min_capability=32,
                      tibet_tokens=["tok1", "tok2"])
        d = layer.to_dict()
        restored = Layer.from_dict(d)
        assert restored.level == layer.level
        assert restored.content == layer.content
        assert restored.content_hash == layer.content_hash
        assert restored.tibet_tokens == ["tok1", "tok2"]

    def test_tibet_tokens_default_empty(self):
        layer = Layer(level=0, content="x", token_count=1, min_capability=3)
        assert layer.tibet_tokens == []


class TestLayerSpec:
    def test_create(self):
        spec = LayerSpec(level=0, min_capability=3, max_tokens=512)
        assert spec.level == 0
        assert spec.min_capability == 3
        assert spec.max_tokens == 512


class TestCapabilityProfile:
    def test_default_profile(self):
        profile = CapabilityProfile.default()
        assert profile.name == "qwen"
        assert len(profile.layers) == 3
        assert profile.layers[0].min_capability == 3
        assert profile.layers[1].min_capability == 14
        assert profile.layers[2].min_capability == 32

    def test_can_access(self):
        profile = CapabilityProfile.default()
        assert profile.can_access(0, 3) is True
        assert profile.can_access(0, 1) is False
        assert profile.can_access(1, 14) is True
        assert profile.can_access(1, 7) is False
        assert profile.can_access(2, 32) is True
        assert profile.can_access(2, 14) is False

    def test_accessible_levels(self):
        profile = CapabilityProfile.default()
        assert profile.accessible_levels(3) == [0]
        assert profile.accessible_levels(14) == [0, 1]
        assert profile.accessible_levels(32) == [0, 1, 2]
        assert profile.accessible_levels(1) == []

    def test_from_dict(self):
        data = {
            "name": "llama",
            "layers": {
                "0": {"min_capability": 1, "max_tokens": 512},
                "1": {"min_capability": 8, "max_tokens": 4096},
                "2": {"min_capability": 70, "max_tokens": 16384},
            },
        }
        profile = CapabilityProfile.from_dict(data)
        assert profile.name == "llama"
        assert profile.layers[0].min_capability == 1
        assert profile.layers[2].min_capability == 70

    def test_to_dict_roundtrip(self):
        profile = CapabilityProfile.default()
        d = profile.to_dict()
        restored = CapabilityProfile.from_dict(d)
        assert restored.name == profile.name
        assert len(restored.layers) == len(profile.layers)

    def test_from_json_file(self):
        data = {
            "profile": {
                "name": "test",
                "layers": {
                    "0": {"min_capability": 2, "max_tokens": 256},
                    "1": {"min_capability": 10, "max_tokens": 2048},
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            profile = CapabilityProfile.from_file(f.name)

        assert profile.name == "test"
        assert profile.layers[0].min_capability == 2

    def test_from_toml_file(self):
        toml_content = """
[profile]
name = "custom"

[profile.layers.0]
min_capability = 5
max_tokens = 1024

[profile.layers.1]
min_capability = 20
max_tokens = 8192
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            try:
                profile = CapabilityProfile.from_file(f.name)
                assert profile.name == "custom"
                assert profile.layers[0].min_capability == 5
            except ImportError:
                pytest.skip("tomllib/tomli not available")

    def test_get_spec(self):
        profile = CapabilityProfile.default()
        spec = profile.get_spec(1)
        assert spec is not None
        assert spec.min_capability == 14
        assert profile.get_spec(99) is None

    def test_nonexistent_level(self):
        profile = CapabilityProfile.default()
        assert profile.can_access(5, 100) is False


class TestEstimateTokens:
    def test_basic(self):
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("a" * 400) == 100

    def test_empty(self):
        assert estimate_tokens("") == 1  # min 1

    def test_unicode(self):
        # Unicode chars are still counted by len()
        result = estimate_tokens("日本語テスト")
        assert result >= 1
