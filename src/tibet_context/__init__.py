"""
tibet-context: Layered context container with TIBET provenance and JIS capability gating.

The Blu-ray model for AI context: same data, different access levels
based on model capability. Audit trail and AI context in one.

Usage:
    from tibet_context import ContextBuilder, ContextReader, CapabilityGate

    # Build context from conversation
    builder = ContextBuilder()
    container = builder.from_conversation(messages)

    # Read with capability filtering
    reader = ContextReader()
    context = reader.read(container, model_id="qwen2.5:32b")

    # Gate check
    gate = CapabilityGate()
    if gate.carbonara_test("qwen2.5:3b"):
        print("Model can handle deep context")
    else:
        print("Zakjapanner detected — escalate!")
"""

from .layers import Layer, LayerSpec, CapabilityProfile, estimate_tokens
from .container import ContextContainer
from .gate import CapabilityGate
from .builder import ContextBuilder
from .reader import ContextReader
from .compactor import Compactor
from . import serializer

__version__ = "0.1.0"
__all__ = [
    "Layer",
    "LayerSpec",
    "CapabilityProfile",
    "ContextContainer",
    "CapabilityGate",
    "ContextBuilder",
    "ContextReader",
    "Compactor",
    "serializer",
    "estimate_tokens",
]
