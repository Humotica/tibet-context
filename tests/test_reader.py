"""Tests for reader module — ContextReader."""

import pytest

from tibet_context.reader import ContextReader
from tibet_context.gate import CapabilityGate
from tibet_context.container import ContextContainer
from tibet_context.layers import Layer


def _make_container():
    layers = {
        0: Layer(level=0, content="This is the summary.", token_count=5, min_capability=3),
        1: Layer(level=1, content="Full conversation with details.", token_count=7, min_capability=14),
        2: Layer(level=2, content="Deep context with architecture.", token_count=7, min_capability=32),
    }
    return ContextContainer(id="reader-test", layers=layers, tibet_chain_id="chain-r")


class TestContextReader:
    def test_read_3b(self):
        reader = ContextReader()
        container = _make_container()
        result = reader.read(container, "qwen2.5:3b")
        assert "Summary" in result
        assert "This is the summary" in result
        assert "Full conversation" not in result

    def test_read_14b(self):
        reader = ContextReader()
        container = _make_container()
        result = reader.read(container, "qwen2.5:14b")
        assert "Summary" in result
        assert "Conversation" in result
        assert "Deep Context" not in result

    def test_read_32b(self):
        reader = ContextReader()
        container = _make_container()
        result = reader.read(container, "qwen2.5:32b")
        assert "Summary" in result
        assert "Conversation" in result
        assert "Deep Context" in result

    def test_read_layer_accessible(self):
        reader = ContextReader()
        container = _make_container()
        content = reader.read_layer(container, "qwen2.5:32b", 2)
        assert content == "Deep context with architecture."

    def test_read_layer_blocked(self):
        reader = ContextReader()
        container = _make_container()
        content = reader.read_layer(container, "qwen2.5:3b", 2)
        assert content is None

    def test_read_layer_nonexistent(self):
        reader = ContextReader()
        container = _make_container()
        content = reader.read_layer(container, "qwen2.5:32b", 99)
        assert content is None

    def test_read_for_capability(self):
        reader = ContextReader()
        container = _make_container()
        result = reader.read_for_capability(container, 14)
        assert "Summary" in result
        assert "Conversation" in result
        assert "Deep Context" not in result

    def test_accessible_token_count(self):
        reader = ContextReader()
        container = _make_container()
        assert reader.accessible_token_count(container, "qwen2.5:3b") == 5
        assert reader.accessible_token_count(container, "qwen2.5:14b") == 12
        assert reader.accessible_token_count(container, "qwen2.5:32b") == 19

    def test_summary(self):
        reader = ContextReader()
        container = _make_container()
        s = reader.summary(container, "qwen2.5:7b")
        assert s["model_id"] == "qwen2.5:7b"
        assert 0 in s["accessible_levels"]
        assert 2 in s["blocked_levels"]
        assert s["carbonara_pass"] is False

    def test_empty_container(self):
        reader = ContextReader()
        container = ContextContainer(id="empty", layers={}, tibet_chain_id="chain-e")
        result = reader.read(container, "qwen2.5:32b")
        assert result == ""
