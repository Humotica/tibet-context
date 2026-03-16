"""Tests for builder module — ContextBuilder."""

import pytest

from tibet_context.builder import ContextBuilder
from tibet_context.layers import CapabilityProfile


class TestContextBuilder:
    def test_from_conversation_basic(self):
        builder = ContextBuilder()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        container = builder.from_conversation(messages)
        assert container.layer_count() >= 2
        assert container.get_layer(0) is not None
        assert container.get_layer(1) is not None
        assert container.tibet_chain_id != ""

    def test_from_conversation_with_summary(self):
        builder = ContextBuilder()
        messages = [{"role": "user", "content": "Test"}]
        container = builder.from_conversation(
            messages,
            summary="Custom summary for L0",
        )
        assert "Custom summary" in container.get_layer(0).content

    def test_from_conversation_with_deep_context(self):
        builder = ContextBuilder()
        messages = [{"role": "user", "content": "Test"}]
        container = builder.from_conversation(
            messages,
            deep_context="Deep architectural context here",
        )
        assert container.layer_count() == 3
        assert "Deep architectural" in container.get_layer(2).content

    def test_from_conversation_session_id(self):
        builder = ContextBuilder()
        messages = [{"role": "user", "content": "Test"}]
        container = builder.from_conversation(messages, session_id="sess-123")
        assert container.source_session == "sess-123"

    def test_from_chain(self):
        builder = ContextBuilder()
        # Create some tokens first
        t1 = builder.provider.create(
            action="test.action",
            erin={"data": "hello"},
            erachter="Test chain token",
        )
        container = builder.from_chain(t1.token_id)
        assert container.layer_count() >= 2
        assert container.tibet_chain_id == t1.token_id

    def test_from_chain_empty(self):
        builder = ContextBuilder()
        with pytest.raises(ValueError, match="Empty chain"):
            builder.from_chain("nonexistent-id")

    def test_merge(self):
        builder = ContextBuilder()
        c1 = builder.from_conversation([{"role": "user", "content": "First"}])
        c2 = builder.from_conversation([{"role": "user", "content": "Second"}])
        merged = builder.merge([c1, c2])
        assert merged.layer_count() >= 2
        # Merged content should contain parts from both
        l1_content = merged.get_layer(1).content
        assert "First" in l1_content
        assert "Second" in l1_content

    def test_merge_single(self):
        builder = ContextBuilder()
        c1 = builder.from_conversation([{"role": "user", "content": "Only"}])
        result = builder.merge([c1])
        assert result.id == c1.id

    def test_merge_empty(self):
        builder = ContextBuilder()
        with pytest.raises(ValueError, match="Cannot merge empty"):
            builder.merge([])

    def test_compact(self):
        builder = ContextBuilder()
        messages = [{"role": "user", "content": "A" * 1000}]
        container = builder.from_conversation(
            messages,
            deep_context="B" * 5000,
        )
        original_tokens = container.total_tokens()
        compacted = builder.compact(container, target_tokens=50)
        assert compacted.total_tokens() <= original_tokens

    def test_metadata(self):
        builder = ContextBuilder()
        messages = [{"role": "user", "content": "Hello"}]
        container = builder.from_conversation(messages)
        assert container.metadata["builder"] == "from_conversation"
        assert container.metadata["message_count"] == 1
