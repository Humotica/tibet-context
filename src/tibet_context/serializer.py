"""
Serialization for ContextContainer — JSON and binary (.tctx) formats.

Binary format:
    Header:  TCTX (4B) + version (2B) + layer_count (1B) + flags (1B)
             + container_id (32B) + chain_id (32B) + timestamp (8B)
    Layers:  Per layer: level (1B) + min_cap (2B) + size (4B) + hash (32B) + content
    Footer:  TCTX (4B) verification
"""

import hashlib
import json
import struct
from typing import Any, Dict, Optional

from .container import ContextContainer
from .layers import Layer


# Binary format constants
MAGIC = b"TCTX"
FORMAT_VERSION = 1
FLAG_COMPACTED = 0x01
FLAG_ENCRYPTED = 0x02  # Reserved for future use


def to_json(container: ContextContainer, indent: int = 2) -> str:
    """Serialize container to JSON string."""
    return json.dumps(container.to_dict(), indent=indent, default=str)


def from_json(json_str: str) -> ContextContainer:
    """Deserialize container from JSON string."""
    data = json.loads(json_str)
    return ContextContainer.from_dict(data)


def to_json_file(container: ContextContainer, path: str) -> None:
    """Write container to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_json(container))


def from_json_file(path: str) -> ContextContainer:
    """Read container from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return from_json(f.read())


def to_binary(container: ContextContainer) -> bytes:
    """
    Serialize container to binary .tctx format.

    Format:
        Header (80 bytes):
            magic:       4B  "TCTX"
            version:     2B  uint16
            layer_count: 1B  uint8
            flags:       1B  uint8
            container_id:32B  zero-padded UTF-8
            chain_id:    32B  zero-padded UTF-8
            timestamp:   8B  double (unix timestamp as float)

        Per layer:
            level:       1B  uint8
            min_cap:     2B  uint16
            size:        4B  uint32 (content byte length)
            hash:        32B hex string as bytes
            content:     {size}B  UTF-8 content

        Footer:
            magic:       4B  "TCTX"
    """
    buf = bytearray()

    # Header
    flags = 0
    if container.metadata.get("compacted"):
        flags |= FLAG_COMPACTED

    buf.extend(MAGIC)
    buf.extend(struct.pack(">H", FORMAT_VERSION))
    buf.extend(struct.pack(">B", len(container.layers)))
    buf.extend(struct.pack(">B", flags))

    # Container ID (32 bytes, zero-padded)
    cid_bytes = container.id.encode("utf-8")[:32]
    buf.extend(cid_bytes.ljust(32, b"\x00"))

    # Chain ID (32 bytes, zero-padded)
    chain_bytes = container.tibet_chain_id.encode("utf-8")[:32]
    buf.extend(chain_bytes.ljust(32, b"\x00"))

    # Timestamp as double
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(container.created_at)
        ts = dt.timestamp()
    except (ValueError, TypeError):
        ts = 0.0
    buf.extend(struct.pack(">d", ts))

    # Layers (sorted by level)
    for level in sorted(container.layers.keys()):
        layer = container.layers[level]
        content_bytes = layer.content.encode("utf-8")
        hash_bytes = layer.content_hash.encode("ascii")[:32].ljust(32, b"\x00")

        buf.extend(struct.pack(">B", layer.level))
        buf.extend(struct.pack(">H", layer.min_capability))
        buf.extend(struct.pack(">I", len(content_bytes)))
        buf.extend(hash_bytes)
        buf.extend(content_bytes)

    # Footer
    buf.extend(MAGIC)

    return bytes(buf)


def from_binary(data: bytes) -> ContextContainer:
    """
    Deserialize container from binary .tctx format.

    Raises:
        ValueError: If data is corrupted or format is wrong
    """
    if len(data) < 84:  # Minimum: header (80) + footer (4)
        raise ValueError("Data too short for TCTX format")

    # Check magic
    if data[:4] != MAGIC:
        raise ValueError(f"Invalid magic: expected TCTX, got {data[:4]!r}")

    # Check footer
    if data[-4:] != MAGIC:
        raise ValueError("Invalid footer: TCTX verification failed")

    offset = 4

    # Version
    version = struct.unpack_from(">H", data, offset)[0]
    offset += 2
    if version > FORMAT_VERSION:
        raise ValueError(f"Unsupported version: {version} (max: {FORMAT_VERSION})")

    # Layer count
    layer_count = struct.unpack_from(">B", data, offset)[0]
    offset += 1

    # Flags
    flags = struct.unpack_from(">B", data, offset)[0]
    offset += 1

    # Container ID
    container_id = data[offset:offset + 32].rstrip(b"\x00").decode("utf-8")
    offset += 32

    # Chain ID
    chain_id = data[offset:offset + 32].rstrip(b"\x00").decode("utf-8")
    offset += 32

    # Timestamp
    ts = struct.unpack_from(">d", data, offset)[0]
    offset += 8

    from datetime import datetime, timezone
    if ts > 0:
        created_at = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    else:
        created_at = ""

    # Parse layers
    layers = {}
    for _ in range(layer_count):
        level = struct.unpack_from(">B", data, offset)[0]
        offset += 1

        min_cap = struct.unpack_from(">H", data, offset)[0]
        offset += 2

        content_size = struct.unpack_from(">I", data, offset)[0]
        offset += 4

        content_hash = data[offset:offset + 32].rstrip(b"\x00").decode("ascii")
        offset += 32

        content = data[offset:offset + content_size].decode("utf-8")
        offset += content_size

        from .layers import estimate_tokens
        layers[level] = Layer(
            level=level,
            content=content,
            token_count=estimate_tokens(content),
            min_capability=min_cap,
            content_hash=content_hash,
        )

    metadata = {}
    if flags & FLAG_COMPACTED:
        metadata["compacted"] = True

    return ContextContainer(
        id=container_id,
        layers=layers,
        tibet_chain_id=chain_id,
        created_at=created_at,
        metadata=metadata,
    )


def to_tctx_file(container: ContextContainer, path: str) -> None:
    """Write container to binary .tctx file."""
    with open(path, "wb") as f:
        f.write(to_binary(container))


def from_tctx_file(path: str) -> ContextContainer:
    """Read container from binary .tctx file."""
    with open(path, "rb") as f:
        return from_binary(f.read())
