"""
tibet-context v0.2.0 — Integration Layer

Connects the context protocol with the real world:
- KmBiT orchestration (intent routing + escalation)
- OomLlama memory bridge (conversation history)
- LLM backends (Ollama local, Gemini API)
- tibet-core Provider integration
"""

from .provider import ContextProvider
from .kmbit import KmBiTBridge
from .oomllama import OomLlamaBridge
from .llm import LLMBackend, OllamaBackend, GeminiBackend

__all__ = [
    "ContextProvider",
    "KmBiTBridge",
    "OomLlamaBridge",
    "LLMBackend",
    "OllamaBackend",
    "GeminiBackend",
]
