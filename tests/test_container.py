"""Tests for container module — ContextContainer."""

import pytest

from tibet_context.container import ContextContainer
from tibet_context.layers import Layer


def _make_layers():
    """Helper to create test layers."""
    return {
        0: Layer(level=0, content="Summary text", token_count=3, min_capability=3),
        1: Layer(level=1, content="Full conversation here with more detail", token_count=10, min_capability=14),
        2: Layer(level=2, content="Deep context with code and architecture decisions that are very detailed", token_count=18, min_capability=32),
    }


class TestContextContainer:
    def test_create(self):
        layers = _make_layers()
        c = ContextContainer(id="test-1", layers=layers, tibet_chain_id="chain-1")
        assert c.id == "test-1"
        assert c.tibet_chain_id == "chain-1"
        assert c.created_at != ""

    def test_get_layer(self):
        layers = _make_layers()
        c = ContextContainer(id="test-1", layers=layers, tibet_chain_id="chain-1")
        assert c.get_layer(0).content == "Summary text"
        assert c.get_layer(1).level == 1
        assert c.get_layer(99) is None

    def test_available_layers_3b(self):
        layers = _make_layers()
        c = ContextContainer(id="test-1", layers=layers, tibet_chain_id="chain-1")
        available = c.available_layers(3)
        assert len(available) == 1
        assert available[0].level == 0

    def test_available_layers_14b(self):
        layers = _make_layers()
        c = ContextContainer(id="test-1", layers=layers, tibet_chain_id="chain-1")
        available = c.available_layers(14)
        assert len(available) == 2

    def test_available_layers_32b(self):
        layers = _make_layers()
        c = ContextContainer(id="test-1", layers=layers, tibet_chain_id="chain-1")
        available = c.available_layers(32)
        assert len(available) == 3

    def test_token_count(self):
        layers = _make_layers()
        c = ContextContainer(id="test-1", layers=layers, tibet_chain_id="chain-1")
        counts = c.token_count()
        assert counts[0] == 3
        assert counts[1] == 10
        assert counts[2] == 18

    def test_total_tokens(self):
        layers = _make_layers()
        c = ContextContainer(id="test-1", layers=layers, tibet_chain_id="chain-1")
        assert c.total_tokens() == 31

    def test_verify_integrity(self):
        layers = _make_layers()
        c = ContextContainer(id="test-1", layers=layers, tibet_chain_id="chain-1")
        assert c.verify_integrity() is True

    def test_layer_count(self):
        layers = _make_layers()
        c = ContextContainer(id="test-1", layers=layers, tibet_chain_id="chain-1")
        assert c.layer_count() == 3

    def test_to_dict(self):
        layers = _make_layers()
        c = ContextContainer(id="test-1", layers=layers, tibet_chain_id="chain-1",
                             source_session="sess-1", metadata={"key": "val"})
        d = c.to_dict()
        assert d["id"] == "test-1"
        assert "0" in d["layers"]
        assert d["metadata"]["key"] == "val"

    def test_from_dict_roundtrip(self):
        layers = _make_layers()
        c = ContextContainer(id="test-1", layers=layers, tibet_chain_id="chain-1",
                             created_at="2025-01-01T00:00:00")
        d = c.to_dict()
        restored = ContextContainer.from_dict(d)
        assert restored.id == c.id
        assert restored.layer_count() == 3
        assert restored.get_layer(0).content == "Summary text"
        assert restored.created_at == "2025-01-01T00:00:00"

    def test_repr(self):
        layers = _make_layers()
        c = ContextContainer(id="test-container-123", layers=layers, tibet_chain_id="chain-1")
        r = repr(c)
        assert "test-contain" in r
        assert "tokens=" in r
