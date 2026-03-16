"""
tibet-context CLI — python -m tibet_context

Quick verification and demo tool.
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="tibet-context",
        description="Layered context container with TIBET provenance and JIS capability gating",
    )
    parser.add_argument("--version", action="store_true", help="Show version")

    sub = parser.add_subparsers(dest="command")

    # Demo command
    demo_parser = sub.add_parser("demo", help="Run carbonara demo")

    # Info command
    info_parser = sub.add_parser("info", help="Show package info")

    # Read command
    read_parser = sub.add_parser("read", help="Read a .tctx or .json container file")
    read_parser.add_argument("file", help="Path to container file")
    read_parser.add_argument("--model", default=None, help="Model ID for capability filtering")
    read_parser.add_argument("--capability", type=int, default=None, help="Raw capability (B params)")

    # Profile command
    profile_parser = sub.add_parser("profile", help="Show or load a capability profile")
    profile_parser.add_argument("--file", default=None, help="Path to profile .toml/.json")

    args = parser.parse_args()

    if args.version:
        from . import __version__
        print(f"tibet-context {__version__}")
        return

    if args.command == "demo":
        _run_demo()
    elif args.command == "info":
        _show_info()
    elif args.command == "read":
        _read_file(args.file, args.model, args.capability)
    elif args.command == "profile":
        _show_profile(args.file)
    else:
        parser.print_help()


def _show_info():
    from . import __version__
    from .layers import CapabilityProfile

    profile = CapabilityProfile.default()
    print(f"tibet-context {__version__}")
    print(f"Default profile: {profile.name}")
    print("Layers:")
    for level, spec in sorted(profile.layers.items()):
        print(f"  L{level}: min {spec.min_capability}B params, max {spec.max_tokens} tokens")


def _run_demo():
    from .builder import ContextBuilder
    from .reader import ContextReader
    from .gate import CapabilityGate

    print("=== Carbonara Test Demo ===\n")

    messages = [
        {"role": "user", "content": "Hoe maak ik pasta carbonara?"},
        {"role": "assistant", "content": "Kook pasta, doe ei erop."},
        {"role": "system", "content": "Quality check: FAIL — rauw ei op bord. Escalation needed."},
    ]

    builder = ContextBuilder()
    container = builder.from_conversation(
        messages,
        session_id="carbonara-demo",
        deep_context=(
            "Pasta carbonara requires: guanciale (not bacon), pecorino romano, "
            "egg yolks, black pepper. The key technique is tempering the egg mixture "
            "with hot pasta water to create a creamy sauce without scrambling. "
            "Never add cream — that's not carbonara. The pasta (rigatoni or spaghetti) "
            "should be cooked al dente and tossed with rendered guanciale fat."
        ),
    )

    print(f"Container: {container.id}")
    print(f"Layers: {container.layer_count()}")
    print(f"Total tokens: {container.total_tokens()}")
    print()

    gate = CapabilityGate()
    reader = ContextReader(gate)

    # Test with 3B model (zakjapanner)
    print("--- Qwen 3B (zakjapanner) ---")
    print(f"Carbonara test: {'PASS' if gate.carbonara_test('qwen2.5:3b') else 'FAIL'}")
    context_3b = reader.read(container, "qwen2.5:3b")
    print(f"Accessible: {reader.accessible_token_count(container, 'qwen2.5:3b')} tokens")
    print(context_3b[:200])
    print()

    # Test with 32B model
    print("--- Qwen 32B ---")
    print(f"Carbonara test: {'PASS' if gate.carbonara_test('qwen2.5:32b') else 'FAIL'}")
    context_32b = reader.read(container, "qwen2.5:32b")
    print(f"Accessible: {reader.accessible_token_count(container, 'qwen2.5:32b')} tokens")
    print(context_32b[:500])
    print()

    print("=== Demo complete ===")


def _read_file(path: str, model_id: str | None, capability: int | None):
    from . import serializer
    from .reader import ContextReader

    if path.endswith(".tctx"):
        container = serializer.from_tctx_file(path)
    else:
        container = serializer.from_json_file(path)

    print(f"Container: {container.id}")
    print(f"Chain: {container.tibet_chain_id}")
    print(f"Created: {container.created_at}")
    print(f"Layers: {container.layer_count()}")
    print(f"Total tokens: {container.total_tokens()}")
    print()

    if model_id:
        reader = ContextReader()
        content = reader.read(container, model_id)
        print(content)
    elif capability:
        reader = ContextReader()
        content = reader.read_for_capability(container, capability)
        print(content)
    else:
        for level, layer in sorted(container.layers.items()):
            print(f"--- L{level} ({layer.token_count} tokens, min {layer.min_capability}B) ---")
            print(layer.content[:300])
            if len(layer.content) > 300:
                print(f"  ... ({len(layer.content)} chars total)")
            print()


def _show_profile(file_path: str | None):
    from .layers import CapabilityProfile

    if file_path:
        profile = CapabilityProfile.from_file(file_path)
    else:
        profile = CapabilityProfile.default()

    print(f"Profile: {profile.name}")
    print("Layers:")
    for level, spec in sorted(profile.layers.items()):
        print(f"  L{level}: min_capability={spec.min_capability}B, max_tokens={spec.max_tokens}")


if __name__ == "__main__":
    main()
