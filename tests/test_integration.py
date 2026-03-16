"""Tests for integration layer — provider, kmbit, oomllama, llm."""

import json
import os
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from tibet_context.integration.provider import ContextProvider
from tibet_context.integration.kmbit import KmBiTBridge, INTENT_LAYER_MAP
from tibet_context.integration.oomllama import OomLlamaBridge
from tibet_context.integration.llm import (
    OllamaBackend,
    GeminiBackend,
    LLMResponse,
    get_backend,
)


# ============================================================
# ContextProvider tests
# ============================================================

class TestContextProvider:
    def test_build_from_conversation(self):
        cp = ContextProvider(actor="test")
        container = cp.build_from_conversation(
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert container.layer_count() >= 2
        assert cp.token_count > 0

    def test_read(self):
        cp = ContextProvider(actor="test")
        container = cp.build_from_conversation(
            messages=[{"role": "user", "content": "Test"}],
        )
        result = cp.read(container, "qwen2.5:32b")
        assert "Summary" in result

    def test_gate_check(self):
        cp = ContextProvider(actor="test")
        container = cp.build_from_conversation(
            messages=[{"role": "user", "content": "Test"}],
        )
        report = cp.gate_check(container, "qwen2.5:3b")
        assert report["capability"] == 3
        assert report["carbonara_pass"] is False

    def test_carbonara_test(self):
        cp = ContextProvider(actor="test")
        assert cp.carbonara_test("qwen2.5:3b") is False
        assert cp.carbonara_test("qwen2.5:32b") is True

    def test_escalation(self):
        cp = ContextProvider(actor="test")
        container = cp.build_from_conversation(
            messages=[{"role": "user", "content": "Pasta carbonara?"}],
            deep_context="Guanciale, pecorino, tempering...",
        )
        result = cp.escalate(container, "qwen2.5:3b", "qwen2.5:32b", "Quality fail")
        assert result["from_model"] == "qwen2.5:3b"
        assert result["to_model"] == "qwen2.5:32b"
        assert result["from_layers"] == 0
        assert result["to_layers"] == 2

    def test_record_completion(self):
        cp = ContextProvider(actor="test")
        container = cp.build_from_conversation(
            messages=[{"role": "user", "content": "Test"}],
        )
        token = cp.record_completion(container, "qwen2.5:32b", "Good answer", True)
        assert token.action == "context.completion"
        assert token.erin["quality_pass"] is True

    def test_export_chain(self):
        cp = ContextProvider(actor="test")
        cp.build_from_conversation(
            messages=[{"role": "user", "content": "Export test"}],
        )
        chain = cp.export_chain(format="dict")
        assert len(chain) > 0


# ============================================================
# KmBiTBridge tests
# ============================================================

class TestKmBiTBridge:
    def test_on_request_simple(self):
        bridge = KmBiTBridge()
        container = bridge.on_request(
            text="Hoe laat is het?",
            intent="simple",
            model_id="qwen2.5:3b",
        )
        assert container.layer_count() >= 2
        # Simple intent should not have L2
        assert container.get_layer(2) is None

    def test_on_request_complex(self):
        bridge = KmBiTBridge()
        container = bridge.on_request(
            text="Leg backpropagation uit",
            intent="complex",
            model_id="qwen2.5:32b",
        )
        assert container.layer_count() == 3
        assert container.get_layer(2) is not None

    def test_on_escalation(self):
        bridge = KmBiTBridge()
        conversation = [
            {"role": "user", "content": "Hoe maak ik carbonara?"},
            {"role": "assistant", "content": "Kook pasta, doe ei erop."},
        ]
        result = bridge.on_escalation(
            from_model="qwen2.5:3b",
            to_model="qwen2.5:32b",
            reason="Quality fail - rauw ei",
            conversation=conversation,
            failed_response="Kook pasta, doe ei erop.",
        )
        assert result.from_model == "qwen2.5:3b"
        assert result.to_model == "qwen2.5:32b"
        assert "rauw ei" in result.reason
        assert result.context_for_model != ""
        assert result.tibet_token_id != ""

    def test_on_completion(self):
        bridge = KmBiTBridge()
        container = bridge.on_request("Test", "simple", "qwen2.5:3b")
        # Should not raise
        bridge.on_completion(container, "qwen2.5:3b", "Antwoord", True)

    def test_should_escalate(self):
        bridge = KmBiTBridge()
        # 3B can handle simple (L0)
        assert bridge.should_escalate("qwen2.5:3b", "simple") is False
        # 3B cannot handle complex (L2)
        assert bridge.should_escalate("qwen2.5:3b", "complex") is True
        # 32B can handle complex (L2)
        assert bridge.should_escalate("qwen2.5:32b", "complex") is False

    def test_suggest_model_simple(self):
        bridge = KmBiTBridge()
        model = bridge.suggest_model("simple")
        assert model is not None
        # Should suggest smallest capable model
        cap = bridge.gate.check_capability(model)
        assert cap >= 3

    def test_suggest_model_complex(self):
        bridge = KmBiTBridge()
        model = bridge.suggest_model("complex")
        assert model is not None
        cap = bridge.gate.check_capability(model)
        assert cap >= 32

    def test_intent_layer_map(self):
        assert INTENT_LAYER_MAP["simple"] == 0
        assert INTENT_LAYER_MAP["complex"] == 2
        assert INTENT_LAYER_MAP["wakeword"] == 0


# ============================================================
# OomLlamaBridge tests (with temp SQLite database)
# ============================================================

def _create_test_db(path: str):
    """Create a test OomLlama database with sample data."""
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            idd_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_activity TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX idx_conv_idd ON conversations(idd_name)")
    conn.execute("""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            provider TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.execute("CREATE INDEX idx_msg_conv ON messages(conversation_id)")

    # Insert test data
    conn.execute("""
        INSERT INTO conversations VALUES
        ('conv-001', 'kit', '2025-12-25T10:00:00', '2025-12-25T10:05:00')
    """)
    conn.execute("""
        INSERT INTO conversations VALUES
        ('conv-002', 'kit', '2025-12-25T11:00:00', '2025-12-25T11:03:00')
    """)
    conn.execute("""
        INSERT INTO messages (conversation_id, role, content, provider, timestamp) VALUES
        ('conv-001', 'user', 'Hoe maak ik pasta carbonara?', NULL, '2025-12-25T10:00:00'),
        ('conv-001', 'assistant', 'Kook pasta, doe ei erop.', 'qwen2.5:3b', '2025-12-25T10:00:05'),
        ('conv-001', 'user', 'Dat klinkt niet goed...', NULL, '2025-12-25T10:01:00'),
        ('conv-002', 'user', 'Hoe laat is het?', NULL, '2025-12-25T11:00:00'),
        ('conv-002', 'assistant', 'Het is 11:00 uur.', 'qwen2.5:3b', '2025-12-25T11:00:02')
    """)
    conn.commit()
    conn.close()


class TestOomLlamaBridge:
    def setup_method(self):
        self.db_fd = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.db_fd.name
        self.db_fd.close()
        _create_test_db(self.db_path)
        self.bridge = OomLlamaBridge(db_path=self.db_path)

    def teardown_method(self):
        os.unlink(self.db_path)

    def test_list_conversations(self):
        convs = self.bridge.list_conversations()
        assert len(convs) == 2

    def test_list_conversations_by_idd(self):
        convs = self.bridge.list_conversations(idd_name="kit")
        assert len(convs) == 2

    def test_get_messages(self):
        msgs = self.bridge.get_messages("conv-001")
        assert len(msgs) == 3
        assert msgs[0].role == "user"
        assert "carbonara" in msgs[0].content

    def test_get_messages_with_limit(self):
        msgs = self.bridge.get_messages("conv-001", max_messages=2)
        assert len(msgs) == 2

    def test_from_conversation_memory(self):
        container = self.bridge.from_conversation_memory("conv-001")
        assert container.layer_count() >= 2
        assert container.source_session == "conv-001"
        # L1 should contain the conversation
        l1 = container.get_layer(1)
        assert "carbonara" in l1.content

    def test_from_conversation_memory_with_deep_context(self):
        container = self.bridge.from_conversation_memory(
            "conv-001",
            deep_context="Guanciale, pecorino romano, egg yolks...",
        )
        assert container.layer_count() == 3
        l2 = container.get_layer(2)
        assert "Guanciale" in l2.content

    def test_from_conversation_memory_empty(self):
        with pytest.raises(ValueError, match="No messages"):
            self.bridge.from_conversation_memory("nonexistent")

    def test_inject_context(self):
        container = self.bridge.from_conversation_memory("conv-001")
        ctx = self.bridge.inject_context(container, "qwen2.5:32b")
        assert "<context" in ctx
        assert "tibet-context" in ctx
        assert "carbonara" in ctx

    def test_inject_context_for_capability(self):
        container = self.bridge.from_conversation_memory("conv-001")
        ctx = self.bridge.inject_context_for_capability(container, 3)
        assert "<context" in ctx
        # Should only contain L0
        assert "Summary" in ctx

    def test_stats(self):
        stats = self.bridge.stats()
        assert stats["total_conversations"] == 2
        assert stats["total_messages"] == 5
        assert stats["active_idds"] == 1

    def test_db_not_found(self):
        bridge = OomLlamaBridge(db_path="/nonexistent/path.db")
        with pytest.raises(FileNotFoundError):
            bridge.list_conversations()


# ============================================================
# LLM Backend tests
# ============================================================

class TestOllamaBackend:
    def test_name(self):
        backend = OllamaBackend(model="qwen2.5:7b")
        assert backend.name == "ollama:qwen2.5:7b"

    @patch("tibet_context.integration.llm.requests")
    def test_generate(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "Test answer"}
        mock_resp.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_resp

        backend = OllamaBackend(model="qwen2.5:3b")
        result = backend.generate("Hello")
        assert result.text == "Test answer"
        assert result.provider == "ollama"
        assert result.model == "qwen2.5:3b"

    @patch("tibet_context.integration.llm.requests")
    def test_is_available(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_requests.get.return_value = mock_resp

        backend = OllamaBackend()
        assert backend.is_available() is True

    @patch("tibet_context.integration.llm.requests")
    def test_not_available(self, mock_requests):
        mock_requests.get.side_effect = Exception("Connection refused")
        backend = OllamaBackend()
        assert backend.is_available() is False


class TestGeminiBackend:
    def test_name(self):
        backend = GeminiBackend(model="gemini-2.0-flash", api_key="test")
        assert backend.name == "gemini:gemini-2.0-flash"

    def test_no_api_key(self):
        # Clear env vars for this test
        with patch.dict(os.environ, {}, clear=True):
            backend = GeminiBackend(api_key=None)
            assert backend.is_available() is False

    def test_has_api_key(self):
        backend = GeminiBackend(api_key="test-key")
        assert backend.is_available() is True

    @patch("tibet_context.integration.llm.requests")
    def test_generate(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{
                "content": {"parts": [{"text": "Gemini answer"}]}
            }]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_resp

        backend = GeminiBackend(api_key="test-key")
        result = backend.generate("Hello")
        assert result.text == "Gemini answer"
        assert result.provider == "gemini"

    def test_generate_no_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            backend = GeminiBackend(api_key=None)
            with pytest.raises(ValueError, match="API key"):
                backend.generate("test")


class TestGetBackend:
    def test_ollama(self):
        backend = get_backend("ollama", model="qwen2.5:7b")
        assert isinstance(backend, OllamaBackend)
        assert backend.model == "qwen2.5:7b"

    def test_gemini(self):
        backend = get_backend("gemini", api_key="test")
        assert isinstance(backend, GeminiBackend)

    def test_unknown(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_backend("openai")
