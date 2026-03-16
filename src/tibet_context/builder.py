"""
ContextBuilder — builds ContextContainers from various sources.

Creates layered context from conversations, TIBET chains,
or by merging existing containers.
"""

import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional

from tibet_core import Provider, Chain, Token

from .container import ContextContainer
from .layers import CapabilityProfile, Layer, estimate_tokens
from .compactor import Compactor


def _generate_container_id() -> str:
    """Generate a unique container ID."""
    ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
    rand = hashlib.sha256(f"{ts}{id(object())}".encode()).hexdigest()[:8]
    return f"tctx_{ts}_{rand}"


class ContextBuilder:
    """
    Builds ContextContainers from different sources.

    Uses a CapabilityProfile to determine layer structure
    and a tibet-core Provider for provenance tracking.
    """

    def __init__(
        self,
        provider: Optional[Provider] = None,
        profile: Optional[CapabilityProfile] = None,
        compactor: Optional[Compactor] = None,
    ):
        self.provider = provider or Provider(actor="tibet-context:builder")
        self.profile = profile or CapabilityProfile.default()
        self.compactor = compactor or Compactor()

    def from_conversation(
        self,
        messages: List[Dict[str, str]],
        session_id: Optional[str] = None,
        summary: Optional[str] = None,
        deep_context: Optional[str] = None,
    ) -> ContextContainer:
        """
        Build a container from conversation messages.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            session_id: Optional session identifier
            summary: Optional pre-computed summary for L0.
                     If None, auto-generates from messages.
            deep_context: Optional deep context for L2.

        Returns:
            ContextContainer with populated layers
        """
        # Track with TIBET
        root_token = self.provider.create(
            action="context.build.conversation",
            erin={"message_count": len(messages), "session_id": session_id},
            erachter="Building context container from conversation",
        )

        # L0: Summary
        if summary is None:
            summary = self._auto_summary(messages)
        l0_spec = self.profile.get_spec(0)
        l0_max = l0_spec.max_tokens if l0_spec else 512
        l0_content = summary
        if estimate_tokens(l0_content) > l0_max:
            l0_content = self.compactor.summarizer(l0_content, l0_max)

        l0_token = self.provider.create(
            action="context.layer.create",
            erin={"level": 0, "tokens": estimate_tokens(l0_content)},
            erachter="Created L0 summary layer",
        )

        # L1: Full conversation
        l1_content = self._format_conversation(messages)
        l1_spec = self.profile.get_spec(1)
        l1_max = l1_spec.max_tokens if l1_spec else 4096
        if estimate_tokens(l1_content) > l1_max:
            l1_content = self.compactor.summarizer(l1_content, l1_max)

        l1_token = self.provider.create(
            action="context.layer.create",
            erin={"level": 1, "tokens": estimate_tokens(l1_content)},
            erachter="Created L1 conversation layer",
        )

        layers: Dict[int, Layer] = {
            0: Layer(
                level=0,
                content=l0_content,
                token_count=estimate_tokens(l0_content),
                min_capability=l0_spec.min_capability if l0_spec else 3,
                tibet_tokens=[root_token.token_id, l0_token.token_id],
            ),
            1: Layer(
                level=1,
                content=l1_content,
                token_count=estimate_tokens(l1_content),
                min_capability=l1_spec.min_capability if l1_spec else 14,
                tibet_tokens=[root_token.token_id, l1_token.token_id],
            ),
        }

        # L2: Deep context (if provided)
        if deep_context is not None:
            l2_spec = self.profile.get_spec(2)
            l2_max = l2_spec.max_tokens if l2_spec else 16384
            l2_content = deep_context
            if estimate_tokens(l2_content) > l2_max:
                l2_content = self.compactor.summarizer(l2_content, l2_max)

            l2_token = self.provider.create(
                action="context.layer.create",
                erin={"level": 2, "tokens": estimate_tokens(l2_content)},
                erachter="Created L2 deep context layer",
            )
            layers[2] = Layer(
                level=2,
                content=l2_content,
                token_count=estimate_tokens(l2_content),
                min_capability=l2_spec.min_capability if l2_spec else 32,
                tibet_tokens=[root_token.token_id, l2_token.token_id],
            )

        return ContextContainer(
            id=_generate_container_id(),
            layers=layers,
            tibet_chain_id=root_token.token_id,
            source_session=session_id,
            metadata={"builder": "from_conversation", "message_count": len(messages)},
        )

    def from_chain(self, chain_id: str) -> ContextContainer:
        """
        Build a container from a TIBET chain.

        Reconstructs context from the provenance trail:
        - L0: Chain summary (actions, actors, timeline)
        - L1: Full chain details (all token data)
        - L2: Cross-references and deep analysis

        Args:
            chain_id: Root token ID of the chain

        Returns:
            ContextContainer representing the chain
        """
        chain = Chain(self.provider.store)
        tokens = chain.trace(chain_id)

        if not tokens:
            raise ValueError(f"Empty chain: {chain_id}")

        root_token = self.provider.create(
            action="context.build.chain",
            erin={"chain_id": chain_id, "token_count": len(tokens)},
            erachter="Building context container from TIBET chain",
        )

        # L0: Chain summary
        chain_summary = chain.summary(chain_id)
        l0_lines = [
            f"Chain: {chain_id[:16]}... ({chain_summary['length']} tokens)",
            f"Actors: {', '.join(chain_summary.get('actors', []))}",
            f"Actions: {', '.join(chain_summary.get('actions', [])[:10])}",
            f"Period: {chain_summary.get('start', '?')} - {chain_summary.get('end', '?')}",
            f"Integrity: {'OK' if chain_summary.get('valid') else 'FAIL'}",
        ]
        l0_content = "\n".join(l0_lines)

        # L1: Full token details
        l1_lines = []
        for t in tokens:
            l1_lines.append(
                f"[{t.timestamp}] {t.action} by {t.actor}: "
                f"{t.erachter} (erin={t.erin})"
            )
        l1_content = "\n".join(l1_lines)

        # L2: Deep analysis with cross-references
        l2_lines = ["=== Deep Chain Analysis ==="]
        for t in tokens:
            l2_lines.append(f"\n--- Token {t.token_id[:16]} ---")
            l2_lines.append(f"Action: {t.action}")
            l2_lines.append(f"Actor: {t.actor}")
            l2_lines.append(f"ERIN: {t.erin}")
            l2_lines.append(f"ERAAN: {t.eraan}")
            l2_lines.append(f"EROMHEEN: {t.eromheen}")
            l2_lines.append(f"ERACHTER: {t.erachter}")
            l2_lines.append(f"Parent: {t.parent_id or 'root'}")
            l2_lines.append(f"Hash valid: {t.verify()}")
        l2_content = "\n".join(l2_lines)

        token_ids = [t.token_id for t in tokens]
        l0_spec = self.profile.get_spec(0)
        l1_spec = self.profile.get_spec(1)
        l2_spec = self.profile.get_spec(2)

        layers = {
            0: Layer(
                level=0,
                content=l0_content,
                token_count=estimate_tokens(l0_content),
                min_capability=l0_spec.min_capability if l0_spec else 3,
                tibet_tokens=[root_token.token_id] + token_ids[:3],
            ),
            1: Layer(
                level=1,
                content=l1_content,
                token_count=estimate_tokens(l1_content),
                min_capability=l1_spec.min_capability if l1_spec else 14,
                tibet_tokens=[root_token.token_id] + token_ids,
            ),
            2: Layer(
                level=2,
                content=l2_content,
                token_count=estimate_tokens(l2_content),
                min_capability=l2_spec.min_capability if l2_spec else 32,
                tibet_tokens=[root_token.token_id] + token_ids,
            ),
        }

        return ContextContainer(
            id=_generate_container_id(),
            layers=layers,
            tibet_chain_id=chain_id,
            metadata={"builder": "from_chain", "chain_length": len(tokens)},
        )

    def merge(self, containers: List[ContextContainer]) -> ContextContainer:
        """
        Merge multiple containers into one.

        Concatenates content per layer level.

        Args:
            containers: Containers to merge

        Returns:
            Merged ContextContainer
        """
        if not containers:
            raise ValueError("Cannot merge empty list")

        if len(containers) == 1:
            return containers[0]

        root_token = self.provider.create(
            action="context.merge",
            erin={"container_count": len(containers)},
            eraan=[c.id for c in containers],
            erachter="Merging multiple context containers",
        )

        # Collect all layers by level
        all_levels: set[int] = set()
        for c in containers:
            all_levels.update(c.layers.keys())

        merged_layers: Dict[int, Layer] = {}
        for level in sorted(all_levels):
            contents = []
            tibet_ids: list[str] = [root_token.token_id]

            for c in containers:
                if level in c.layers:
                    layer = c.layers[level]
                    contents.append(layer.content)
                    tibet_ids.extend(layer.tibet_tokens)

            merged_content = "\n\n---\n\n".join(contents)
            spec = self.profile.get_spec(level)

            merged_layers[level] = Layer(
                level=level,
                content=merged_content,
                token_count=estimate_tokens(merged_content),
                min_capability=spec.min_capability if spec else 3,
                tibet_tokens=tibet_ids,
            )

        return ContextContainer(
            id=_generate_container_id(),
            layers=merged_layers,
            tibet_chain_id=root_token.token_id,
            metadata={
                "builder": "merge",
                "source_containers": [c.id for c in containers],
            },
        )

    def compact(self, container: ContextContainer, target_tokens: int) -> ContextContainer:
        """Compact a container to target token budget."""
        token = self.provider.create(
            action="context.compact",
            erin={"container_id": container.id, "target_tokens": target_tokens},
            erachter=f"Compacting container to {target_tokens} tokens",
        )
        return self.compactor.compact(container, target_tokens)

    def _auto_summary(self, messages: List[Dict[str, str]]) -> str:
        """Generate automatic summary from messages."""
        if not messages:
            return "Empty conversation."

        lines = [f"Conversation with {len(messages)} messages."]

        # First and last message hints
        if messages:
            first = messages[0]
            lines.append(f"Started: {first.get('role', '?')}: {first.get('content', '')[:100]}")
        if len(messages) > 1:
            last = messages[-1]
            lines.append(f"Latest: {last.get('role', '?')}: {last.get('content', '')[:100]}")

        # Unique roles
        roles = set(m.get("role", "unknown") for m in messages)
        lines.append(f"Participants: {', '.join(sorted(roles))}")

        return "\n".join(lines)

    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into readable conversation text."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"[{role}]: {content}")
        return "\n\n".join(lines)
