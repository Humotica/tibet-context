"""Tests for compactor module — context compaction."""

import pytest

from tibet_context.compactor import Compactor, _default_summarizer
from tibet_context.container import ContextContainer
from tibet_context.layers import Layer, estimate_tokens


def _make_big_container():
    layers = {
        0: Layer(level=0, content="Short summary.", token_count=4, min_capability=3),
        1: Layer(level=1, content="A" * 2000, token_count=500, min_capability=14),
        2: Layer(level=2, content="B" * 4000, token_count=1000, min_capability=32),
    }
    return ContextContainer(id="compact-test", layers=layers, tibet_chain_id="chain-c")


class TestDefaultSummarizer:
    def test_short_text_unchanged(self):
        text = "This is short."
        result = _default_summarizer(text, 100)
        assert result == text

    def test_long_text_truncated(self):
        text = "Hello world. " * 100
        result = _default_summarizer(text, 10)
        assert estimate_tokens(result) <= 15  # some margin for sentence boundary

    def test_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence."
        result = _default_summarizer(text, 5)
        assert result.endswith(".")


class TestCompactor:
    def test_no_compaction_needed(self):
        compactor = Compactor()
        container = _make_big_container()
        result = compactor.compact(container, target_tokens=10000)
        # Should return same content since under budget
        assert result.total_tokens() == container.total_tokens()

    def test_compaction_reduces_tokens(self):
        compactor = Compactor()
        container = _make_big_container()
        result = compactor.compact(container, target_tokens=100)
        assert result.total_tokens() < container.total_tokens()

    def test_l0_preserved(self):
        compactor = Compactor()
        container = _make_big_container()
        result = compactor.compact(container, target_tokens=100, preserve_l0=True)
        assert result.get_layer(0).content == container.get_layer(0).content

    def test_l0_not_preserved(self):
        compactor = Compactor()
        container = _make_big_container()
        result = compactor.compact(container, target_tokens=5, preserve_l0=False)
        # L0 may be compacted
        assert result.total_tokens() <= container.total_tokens()

    def test_compacted_metadata(self):
        compactor = Compactor()
        container = _make_big_container()
        result = compactor.compact(container, target_tokens=100)
        assert result.metadata.get("compacted") is True

    def test_compact_single_layer(self):
        compactor = Compactor()
        layer = Layer(level=1, content="X" * 2000, token_count=500, min_capability=14)
        result = compactor.compact_layer(layer, max_tokens=50)
        assert result.token_count <= 500

    def test_custom_summarizer(self):
        def my_summarizer(text: str, max_tokens: int) -> str:
            return text[:20] + "..."

        compactor = Compactor(summarizer=my_summarizer)
        container = _make_big_container()
        result = compactor.compact(container, target_tokens=10)
        # Should use custom summarizer
        for level in [1, 2]:
            layer = result.get_layer(level)
            if layer and layer.content != container.get_layer(level).content:
                assert layer.content.endswith("...")
