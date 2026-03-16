# tibet-context

Layered context container with TIBET provenance and JIS capability gating.

> *"Audit is context. Context is key. TIBET is the answer."*

[![PyPI version](https://badge.fury.io/py/tibet-context.svg)](https://pypi.org/project/tibet-context/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## The Blu-ray Model

Inspired by Blu-ray disk architecture: same data, different access levels. A JIS capability gate — like AACS keys — determines which model can read which layer.

The same TIBET chain serves two consumers:
- **Human / regulator**: audit trail (compliance, evidence)
- **AI model**: context window (memory, reasoning)

```
┌─────────────────────────────────────────────┐
│              tibet-context                   │
│  ┌───────────────────────────────────────┐  │
│  │  L0: Summary Layer (always readable)  │  │
│  │  - Compact summary (~512 tokens)      │  │
│  │  - Any model can read this (3B+)      │  │
│  ├───────────────────────────────────────┤  │
│  │  L1: Conversation Layer               │  │
│  │  - Full conversation context          │  │
│  │  - Requires: JIS capability >= 14B    │  │
│  ├───────────────────────────────────────┤  │
│  │  L2: Deep Context Layer               │  │
│  │  - Full codebase + cross-session mem  │  │
│  │  - Requires: JIS capability >= 32B    │  │
│  ├───────────────────────────────────────┤  │
│  │  TIBET Chain (through all layers)     │  │
│  │  - Provenance trail + integrity       │  │
│  └───────────────────────────────────────┘  │
│  JIS Capability Gate                        │
└─────────────────────────────────────────────┘
```

## Installation

```bash
pip install tibet-context
```

Requires Python 3.10+ and [tibet-core](https://pypi.org/project/tibet-core/) >= 0.3.0 (installed automatically). Zero other dependencies.

For LLM backend support (Ollama, Gemini API):
```bash
pip install tibet-context[llm]
```

## Quick Start

```python
from tibet_context import ContextBuilder, ContextReader, CapabilityGate

# Build a layered context from conversation
builder = ContextBuilder()
container = builder.from_conversation(
    messages=[
        {"role": "user", "content": "How do I make pasta carbonara?"},
        {"role": "assistant", "content": "Cook pasta, put egg on it."},
    ],
    deep_context="Carbonara requires guanciale, pecorino, egg yolks, black pepper...",
)

# Read with capability filtering
reader = ContextReader()
reader.read(container, model_id="qwen2.5:32b")  # All 3 layers
reader.read(container, model_id="qwen2.5:3b")   # Only L0 summary

# Carbonara test — can this model handle deep context?
gate = CapabilityGate()
gate.carbonara_test("qwen2.5:3b")   # False — zakjapanner!
gate.carbonara_test("qwen2.5:32b")  # True
```

## v0.1.0 — Core Engine

The core protocol: layered containers, capability gating, binary serialization, and TIBET provenance. Everything needed to prove the concept works.

### Core Modules

| Module | Purpose |
|--------|---------|
| `layers.py` | `Layer`, `LayerSpec`, `CapabilityProfile` — configurable layer definitions |
| `container.py` | `ContextContainer` — the core layered context unit |
| `gate.py` | `CapabilityGate` — JIS capability gate + carbonara test |
| `builder.py` | `ContextBuilder` — build from conversations, chains, or merge containers |
| `reader.py` | `ContextReader` — capability-filtered reading |
| `compactor.py` | `Compactor` — intelligent context compaction per layer |
| `serializer.py` | JSON + binary `.tctx` format with integrity verification |

### Capability Profiles

Profiles are fully configurable — no hardcoded thresholds. The default is tuned for the Qwen family:

```toml
# tibet-context.toml
[profile]
name = "qwen"

[profile.layers.0]
min_capability = 3    # Qwen 3B can read L0
max_tokens = 512

[profile.layers.1]
min_capability = 14   # Qwen 14B for L1
max_tokens = 4096

[profile.layers.2]
min_capability = 32   # Qwen 32B for L2
max_tokens = 16384
```

Create your own profile for any model family:

```python
from tibet_context import CapabilityProfile
from tibet_context.layers import LayerSpec

profile = CapabilityProfile(name="llama", layers={
    0: LayerSpec(level=0, min_capability=1, max_tokens=512),
    1: LayerSpec(level=1, min_capability=8, max_tokens=4096),
    2: LayerSpec(level=2, min_capability=70, max_tokens=16384),
})
```

### Binary `.tctx` Format

Compact binary format for efficient storage and transport:

```python
from tibet_context import serializer

serializer.to_tctx_file(container, "context.tctx")
restored = serializer.from_tctx_file("context.tctx")
assert restored.verify_integrity()
```

Format: `TCTX` magic header, version, layers with content hashes, `TCTX` footer verification.

### The Carbonara Test

The "zakjapanner" problem: a small model that gives superficially correct but actually wrong answers — like putting raw egg on pasta and calling it carbonara.

```python
gate = CapabilityGate()
gate.carbonara_test("qwen2.5:3b")   # False — needs escalation
gate.carbonara_test("qwen2.5:32b")  # True — can handle deep context
```

## v0.2.0 — Integration Layer

Connects the protocol with the real world: KmBiT orchestration, OomLlama memory, and LLM backends.

### Integration Modules

| Module | Purpose |
|--------|---------|
| `integration/provider.py` | `ContextProvider` — tibet-core Provider integration with full TIBET tracking |
| `integration/kmbit.py` | `KmBiTBridge` — KmBiT Overseer hooks for intent routing + escalation |
| `integration/oomllama.py` | `OomLlamaBridge` — reads conversation history from OomLlama's SQLite memory |
| `integration/llm.py` | `OllamaBackend`, `GeminiBackend` — LLM backend abstraction |
| `integration/carbonara.py` | End-to-end carbonara demo flow (mock + live modes) |

### KmBiT Escalation Flow

The bridge hooks into the KmBiT Overseer's intent routing pipeline:

```python
from tibet_context.integration import KmBiTBridge

bridge = KmBiTBridge()

# 1. KmBiT classifies intent -> routes to model
container = bridge.on_request(
    text="Hoe maak ik pasta carbonara?",
    intent="simple",
    model_id="qwen2.5:3b",
)

# 2. Small model fails quality check -> escalate
result = bridge.on_escalation(
    from_model="qwen2.5:3b",
    to_model="qwen2.5:32b",
    reason="Carbonara test FAIL",
    conversation=[...],
    failed_response="Kook pasta, doe ei erop.",
)

# 3. Big model gets full context package
print(result.context_for_model)  # L0 + L1 + L2 with escalation context
```

### OomLlama Memory Bridge

Reads directly from OomLlama's SQLite conversation database:

```python
from tibet_context.integration import OomLlamaBridge

bridge = OomLlamaBridge(db_path="/srv/jtel-stack/data/chimera_memory.db")

# List conversations
convs = bridge.list_conversations(idd_name="kit")

# Build context container from conversation history
container = bridge.from_conversation_memory("conv-001")

# Inject as prompt context for a model
context_str = bridge.inject_context(container, "qwen2.5:32b")
```

### LLM Backends

Unified interface for Ollama (local P520) and Gemini API:

```python
from tibet_context.integration.llm import OllamaBackend, GeminiBackend

# Local Ollama (P520 GPU)
small = OllamaBackend(model="qwen2.5:3b", host="http://192.168.4.85:11434")
response = small.generate("Hoe maak ik carbonara?", temperature=0.3)

# Google Gemini API
big = GeminiBackend(model="gemini-2.0-flash")  # uses GOOGLE_API_KEY env
response = big.generate("Explain carbonara properly", max_tokens=500)
```

### Carbonara Demo

Full end-to-end flow showing the escalation pipeline:

```python
from tibet_context.integration.carbonara import run_carbonara_mock, print_result

# Mock mode (no LLM needed — perfect for testing)
result = run_carbonara_mock()
print_result(result)

# Live mode (requires Ollama on P520 + Gemini API key)
from tibet_context.integration.carbonara import run_carbonara_live
result = run_carbonara_live()
print_result(result)
```

The flow:
1. User asks "Hoe maak ik pasta carbonara?"
2. KmBiT classifies intent -> routes to Qwen 3B (small, fast)
3. Qwen 3B answers with L0 context only -> "Kook pasta, doe ei erop"
4. Quality check **FAILS** (carbonara test — rauw ei op bord!)
5. KmBiT escalates to big model with full context package
6. Big model answers with L0+L1+L2 -> proper carbonara recipe
7. Quality check **PASSES**
8. TIBET chain records the complete provenance trail

### CLI (v0.2.0)

```bash
# Core commands (v0.1.0)
python -m tibet_context --version
python -m tibet_context info
python -m tibet_context demo
python -m tibet_context read context.tctx --model qwen2.5:32b
python -m tibet_context profile --file my-profile.toml

# Integration commands (v0.2.0)
python -m tibet_context carbonara-mock
python -m tibet_context carbonara-mock --question "Hoe maak ik risotto?"
python -m tibet_context carbonara-live
python -m tibet_context carbonara-live --small qwen2.5:7b --big gemini-2.0-flash
```

## ContextProvider — One-Stop Integration

The `ContextProvider` wraps everything into a single entry point:

```python
from tibet_context.integration import ContextProvider

cp = ContextProvider(actor="my-app")

# Build from conversation
container = cp.build_from_conversation(
    messages=[{"role": "user", "content": "Carbonara?"}],
    deep_context="Guanciale, pecorino, tempering...",
)

# Gate check
report = cp.gate_check(container, "qwen2.5:3b")
# {"capability": 3, "max_level": 0, "carbonara_pass": False, ...}

# Escalate
result = cp.escalate(container, "qwen2.5:3b", "qwen2.5:32b", "Quality fail")

# Record completion with TIBET provenance
cp.record_completion(container, "qwen2.5:32b", "Proper answer", quality_pass=True)

# Export full TIBET chain
chain = cp.export_chain(format="dict")
```

## Relation to TIBET Ecosystem

```
tibet-core (Token, Chain, Provider, FileStore)
    │ provides provenance
tibet-context (Container, Layers, Gate, Builder)
    │ feeds context to           │ hooks into
OomLlama (.oom chunks)     KmBiT (orchestration)
    │ runs on                    │ routes to
P520 GPU (Qwen 3B/7B/32B)  Gemini API (Flash)
```

tibet-context is the **glue** between audit (tibet) and AI (oomllama/kmbit). It transforms audit trail into actionable context.

## Tests

```bash
# Run all 126 tests
pytest tests/ -v

# Core engine only (88 tests)
pytest tests/ -v --ignore=tests/test_integration.py

# Integration layer only (38 tests)
pytest tests/test_integration.py -v
```

## License

MIT — [Humotica](https://humotica.com)
