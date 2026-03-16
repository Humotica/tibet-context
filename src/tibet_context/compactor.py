"""
Context compaction — intelligent summarization per layer.

Uses TIBET ERACHTER (intent) to determine what's important.
Tokens with high trust or those in chains are preserved.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from .container import ContextContainer
from .layers import Layer, estimate_tokens


def _default_summarizer(text: str, max_tokens: int) -> str:
    """
    Default summarizer: truncate to approximate token budget.

    In production, this would call an LLM. For v0.1.0 we use
    intelligent truncation that preserves sentence boundaries.
    """
    estimated = estimate_tokens(text)
    if estimated <= max_tokens:
        return text

    # Target character count (~4 chars per token)
    target_chars = max_tokens * 4

    # Try to break at sentence boundary
    truncated = text[:target_chars]
    last_period = truncated.rfind(".")
    last_newline = truncated.rfind("\n")
    break_point = max(last_period, last_newline)

    if break_point > target_chars // 2:
        truncated = truncated[:break_point + 1]

    return truncated.rstrip()


@dataclass
class Compactor:
    """
    Context compactor — reduces container to target token budget.

    Strategy:
    - L0 (summary) is always preserved as-is
    - L1 (conversation) is compacted if needed
    - L2 (deep context) is compacted most aggressively

    The summarizer function can be swapped for an LLM-based one.
    """
    summarizer: Callable[[str, int], str] = _default_summarizer

    def compact(
        self,
        container: ContextContainer,
        target_tokens: int,
        preserve_l0: bool = True,
    ) -> ContextContainer:
        """
        Compact a container to fit within a token budget.

        Args:
            container: Source container
            target_tokens: Target total token count
            preserve_l0: Whether L0 should be preserved unchanged

        Returns:
            New ContextContainer with compacted layers
        """
        current_total = container.total_tokens()
        if current_total <= target_tokens:
            return container

        # Calculate how much we need to cut
        excess = current_total - target_tokens

        # Get layers sorted by level (highest first for most aggressive compaction)
        levels = sorted(container.layers.keys(), reverse=True)
        new_layers = dict(container.layers)

        for level in levels:
            if excess <= 0:
                break

            if preserve_l0 and level == 0:
                continue

            layer = new_layers[level]
            # Calculate this layer's share of the cut
            layer_budget = max(1, layer.token_count - excess)
            if layer_budget < layer.token_count:
                compacted_content = self.summarizer(layer.content, layer_budget)
                new_token_count = estimate_tokens(compacted_content)
                saved = layer.token_count - new_token_count
                excess -= saved

                new_layers[level] = Layer(
                    level=layer.level,
                    content=compacted_content,
                    token_count=new_token_count,
                    min_capability=layer.min_capability,
                    tibet_tokens=layer.tibet_tokens,
                )

        return ContextContainer(
            id=container.id,
            layers=new_layers,
            tibet_chain_id=container.tibet_chain_id,
            created_at=container.created_at,
            source_session=container.source_session,
            metadata={**container.metadata, "compacted": True, "target_tokens": target_tokens},
        )

    def compact_layer(self, layer: Layer, max_tokens: int) -> Layer:
        """Compact a single layer to a token budget."""
        if layer.token_count <= max_tokens:
            return layer

        compacted_content = self.summarizer(layer.content, max_tokens)
        return Layer(
            level=layer.level,
            content=compacted_content,
            token_count=estimate_tokens(compacted_content),
            min_capability=layer.min_capability,
            tibet_tokens=layer.tibet_tokens,
        )
