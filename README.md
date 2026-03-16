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

### Modules

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

# Load from file
profile = CapabilityProfile.from_file("my-profile.toml")

# Or build programmatically
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

# Write
serializer.to_tctx_file(container, "context.tctx")

# Read
restored = serializer.from_tctx_file("context.tctx")
assert restored.verify_integrity()
```

Format: `TCTX` magic header, version, layers with content hashes, `TCTX` footer verification.

### The Carbonara Test

The "zakjapanner" problem: a small model that gives superficially correct but actually wrong answers — like putting raw egg on pasta and calling it carbonara.

```python
gate = CapabilityGate()

# Small model: can only see the summary
gate.carbonara_test("qwen2.5:3b")   # False — needs escalation
gate.carbonara_test("qwen2.5:7b")   # False

# Large model: can see deep context with the real technique
gate.carbonara_test("qwen2.5:32b")  # True — can handle it
gate.carbonara_test("qwen2.5:72b")  # True
```

### CLI

```bash
# Version
python -m tibet_context --version

# Package info + default profile
python -m tibet_context info

# Run the carbonara demo
python -m tibet_context demo

# Read a container file
python -m tibet_context read context.tctx --model qwen2.5:32b

# Show a capability profile
python -m tibet_context profile --file my-profile.toml
```

## Relation to TIBET Ecosystem

```
tibet-core (Token, Chain, Provider, FileStore)
    │ provides provenance
tibet-context (Container, Layers, Gate, Builder)
    │ feeds context to           │ hooks into
OomLlama (.oom chunks)     KmBiT (orchestration)
    │ runs on
P520 GPU (Qwen 3B/7B/32B)
```

tibet-context is the **glue** between audit (tibet) and AI (oomllama/kmbit). It transforms audit trail into actionable context.

## Roadmap

- **v0.1.0** — Core Engine *(current)* — protocol + gating + serialization
- **v0.2.0** — Integration Layer — KmBiT orchestration, OomLlama memory bridge, tibet-core Provider hooks

## License

MIT — [Humotica](https://humotica.com)
