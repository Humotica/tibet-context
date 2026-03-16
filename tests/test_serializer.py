"""Tests for serializer module — JSON and binary (.tctx) formats."""

import json
import tempfile
from pathlib import Path

import pytest

from tibet_context.serializer import (
    to_json,
    from_json,
    to_json_file,
    from_json_file,
    to_binary,
    from_binary,
    to_tctx_file,
    from_tctx_file,
    MAGIC,
)
from tibet_context.container import ContextContainer
from tibet_context.layers import Layer


def _make_container():
    layers = {
        0: Layer(level=0, content="Summary layer content.", token_count=5, min_capability=3),
        1: Layer(level=1, content="Conversation layer with more detail.", token_count=9, min_capability=14),
        2: Layer(level=2, content="Deep context layer with architecture decisions.", token_count=11, min_capability=32),
    }
    return ContextContainer(
        id="serial-test-001",
        layers=layers,
        tibet_chain_id="chain-serial",
        created_at="2025-06-15T12:00:00",
        source_session="sess-001",
        metadata={"test": True},
    )


class TestJsonSerialization:
    def test_to_json(self):
        container = _make_container()
        j = to_json(container)
        data = json.loads(j)
        assert data["id"] == "serial-test-001"
        assert "0" in data["layers"]

    def test_roundtrip(self):
        container = _make_container()
        j = to_json(container)
        restored = from_json(j)
        assert restored.id == container.id
        assert restored.layer_count() == 3
        assert restored.get_layer(0).content == "Summary layer content."
        assert restored.tibet_chain_id == "chain-serial"

    def test_json_file_roundtrip(self):
        container = _make_container()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        to_json_file(container, path)
        restored = from_json_file(path)
        assert restored.id == container.id
        assert restored.get_layer(2).content == container.get_layer(2).content

    def test_preserves_metadata(self):
        container = _make_container()
        j = to_json(container)
        restored = from_json(j)
        assert restored.metadata.get("test") is True
        assert restored.source_session == "sess-001"


class TestBinarySerialization:
    def test_magic_header(self):
        container = _make_container()
        data = to_binary(container)
        assert data[:4] == MAGIC
        assert data[-4:] == MAGIC

    def test_roundtrip(self):
        container = _make_container()
        binary = to_binary(container)
        restored = from_binary(binary)
        assert restored.id == container.id
        assert restored.layer_count() == 3
        assert restored.get_layer(0).content == "Summary layer content."
        assert restored.get_layer(1).content == "Conversation layer with more detail."
        assert restored.get_layer(2).content == "Deep context layer with architecture decisions."

    def test_tctx_file_roundtrip(self):
        container = _make_container()
        with tempfile.NamedTemporaryFile(suffix=".tctx", delete=False) as f:
            path = f.name

        to_tctx_file(container, path)
        restored = from_tctx_file(path)
        assert restored.id == container.id
        assert restored.get_layer(0).content == container.get_layer(0).content

    def test_invalid_magic(self):
        with pytest.raises(ValueError, match="Invalid magic"):
            from_binary(b"XXXX" + b"\x00" * 80)

    def test_invalid_footer(self):
        container = _make_container()
        binary = bytearray(to_binary(container))
        binary[-4:] = b"XXXX"
        with pytest.raises(ValueError, match="Invalid footer"):
            from_binary(bytes(binary))

    def test_too_short(self):
        with pytest.raises(ValueError, match="too short"):
            from_binary(b"TCTX")

    def test_binary_preserves_chain_id(self):
        container = _make_container()
        binary = to_binary(container)
        restored = from_binary(binary)
        assert restored.tibet_chain_id == "chain-serial"

    def test_compacted_flag(self):
        layers = {
            0: Layer(level=0, content="Short.", token_count=1, min_capability=3),
        }
        container = ContextContainer(
            id="flagtest",
            layers=layers,
            tibet_chain_id="chain-f",
            metadata={"compacted": True},
        )
        binary = to_binary(container)
        restored = from_binary(binary)
        assert restored.metadata.get("compacted") is True
