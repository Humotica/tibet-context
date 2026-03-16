"""
LLM backends for tibet-context.

Provides a unified interface for calling different LLM backends:
- Ollama (local, P520)
- Gemini API (Google)

Used by the carbonara flow to test model responses and for
context-aware completions.
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

log = logging.getLogger("tibet-context.llm")


@dataclass
class LLMResponse:
    """Response from an LLM backend."""
    text: str
    model: str
    provider: str
    token_count: Optional[int] = None
    latency_ms: Optional[float] = None


class LLMBackend(ABC):
    """Abstract LLM backend."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Generate a response."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is reachable."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier."""
        ...


class OllamaBackend(LLMBackend):
    """
    Ollama backend for local model inference.

    Connects to the Ollama API (default: P520 at 192.168.4.85:11434).
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        host: str = "http://192.168.4.85:11434",
    ):
        self.model = model
        self.host = host.rstrip("/")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        start = time.time()
        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.time() - start) * 1000

            return LLMResponse(
                text=data.get("response", ""),
                model=self.model,
                provider="ollama",
                latency_ms=elapsed,
            )
        except Exception as e:
            log.error(f"Ollama error: {e}")
            raise ConnectionError(f"Ollama ({self.host}) error: {e}") from e

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"


class GeminiBackend(LLMBackend):
    """
    Google Gemini API backend.

    Uses the REST API directly (no SDK dependency).
    API key from GOOGLE_API_KEY environment variable or constructor.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:

        if not self.api_key:
            raise ValueError("Gemini API key not set. Set GOOGLE_API_KEY env var or pass api_key.")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:generateContent?key={self.api_key}"
        )

        body: Dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        start = time.time()
        try:
            resp = requests.post(url, json=body, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.time() - start) * 1000

            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )

            return LLMResponse(
                text=text,
                model=self.model,
                provider="gemini",
                latency_ms=elapsed,
            )
        except Exception as e:
            log.error(f"Gemini error: {e}")
            raise ConnectionError(f"Gemini API error: {e}") from e

    def is_available(self) -> bool:
        return bool(self.api_key)

    @property
    def name(self) -> str:
        return f"gemini:{self.model}"


def get_backend(provider: str, model: Optional[str] = None, **kwargs) -> LLMBackend:
    """
    Factory for LLM backends.

    Args:
        provider: "ollama" or "gemini"
        model: Optional model override
        **kwargs: Additional backend-specific args

    Returns:
        Configured LLMBackend
    """
    if provider == "ollama":
        return OllamaBackend(
            model=model or kwargs.get("ollama_model", "qwen2.5:3b"),
            host=kwargs.get("ollama_host", "http://192.168.4.85:11434"),
        )
    elif provider == "gemini":
        return GeminiBackend(
            model=model or kwargs.get("gemini_model", "gemini-2.0-flash"),
            api_key=kwargs.get("api_key"),
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'ollama' or 'gemini'.")
