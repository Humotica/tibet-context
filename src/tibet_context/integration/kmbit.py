"""
KmBiT orchestration hooks for tibet-context.

Integrates with the KmBiT Overseer's intent routing and escalation pipeline.
When a small model fails the carbonara test, KmBiT escalates with
a tibet-context package that gives the big model proper context.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from tibet_core import Provider

from ..builder import ContextBuilder
from ..container import ContextContainer
from ..gate import CapabilityGate
from ..layers import CapabilityProfile
from ..reader import ContextReader

log = logging.getLogger("tibet-context.kmbit")

# Maps KmBiT intent -> minimum required layer level
INTENT_LAYER_MAP = {
    "wakeword": 0,
    "simple": 0,
    "complex": 2,
    "command": 1,
    "unknown": 0,
}


@dataclass
class EscalationResult:
    """Result of a KmBiT escalation."""
    container: ContextContainer
    from_model: str
    to_model: str
    reason: str
    context_for_model: str
    tibet_token_id: str


class KmBiTBridge:
    """
    Bridge between KmBiT Overseer and tibet-context.

    Hooks into the escalation pipeline:
    1. KmBiT classifies intent (SIMPLE/COMPLEX)
    2. If SIMPLE -> 3B answers with L0 context
    3. If quality fails -> escalate to 32B with full context package
    4. TIBET chain tracks everything
    """

    def __init__(
        self,
        provider: Optional[Provider] = None,
        profile: Optional[CapabilityProfile] = None,
        kmbit_url: str = "http://192.168.4.85:5002",
    ):
        self.provider = provider or Provider(actor="tibet-context:kmbit")
        self.profile = profile or CapabilityProfile.default()
        self.gate = CapabilityGate(profile=self.profile)
        self.builder = ContextBuilder(provider=self.provider, profile=self.profile)
        self.reader = ContextReader(gate=self.gate)
        self.kmbit_url = kmbit_url

    def on_request(
        self,
        text: str,
        intent: str,
        model_id: str,
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> ContextContainer:
        """
        Handle an incoming request from KmBiT.

        Builds a context container appropriate for the routed model.

        Args:
            text: User's input text
            intent: KmBiT classified intent ("simple", "complex", etc.)
            model_id: Model being routed to
            session_id: Optional session identifier
            conversation_history: Optional previous messages

        Returns:
            ContextContainer ready for the model
        """
        self.provider.create(
            action="kmbit.request",
            erin={"text": text[:200], "intent": intent, "model_id": model_id},
            erachter=f"KmBiT request: intent={intent}, routed to {model_id}",
        )

        messages = conversation_history or []
        messages.append({"role": "user", "content": text})

        # Determine if we need deep context based on intent
        required_level = INTENT_LAYER_MAP.get(intent, 0)
        needs_deep = required_level >= 2

        container = self.builder.from_conversation(
            messages,
            session_id=session_id,
            deep_context=self._build_deep_context(text, intent) if needs_deep else None,
        )

        return container

    def on_escalation(
        self,
        from_model: str,
        to_model: str,
        reason: str,
        conversation: List[Dict[str, str]],
        failed_response: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> EscalationResult:
        """
        Handle an escalation from KmBiT.

        This is the carbonara flow: the small model failed, so we build
        a rich context package for the big model.

        Args:
            from_model: Model that failed
            to_model: Model being escalated to
            reason: Why escalation happened
            conversation: Full conversation history
            failed_response: The response that failed quality check
            session_id: Optional session identifier

        Returns:
            EscalationResult with container and context string
        """
        # Record escalation
        escalation_token = self.provider.create(
            action="kmbit.escalation",
            erin={
                "from_model": from_model,
                "to_model": to_model,
                "reason": reason,
                "failed_response": failed_response[:200] if failed_response else None,
            },
            erachter=f"Escalation: {from_model} -> {to_model}. {reason}",
        )

        # Build deep context including the failed attempt
        deep_lines = [
            f"=== Escalation Context ===",
            f"Previous model ({from_model}) failed with reason: {reason}",
        ]
        if failed_response:
            deep_lines.append(f"Failed response: {failed_response}")
        deep_lines.append(f"Escalated to {to_model} for deeper analysis.")
        deep_context = "\n".join(deep_lines)

        # Build container with all layers
        container = self.builder.from_conversation(
            conversation,
            session_id=session_id,
            deep_context=deep_context,
        )

        # Get context string for the target model
        context_str = self.reader.read(container, to_model)

        return EscalationResult(
            container=container,
            from_model=from_model,
            to_model=to_model,
            reason=reason,
            context_for_model=context_str,
            tibet_token_id=escalation_token.token_id,
        )

    def on_completion(
        self,
        container: ContextContainer,
        model_id: str,
        response: str,
        quality_pass: bool,
    ) -> None:
        """
        Record a completion event.

        Args:
            container: The context container used
            model_id: Model that generated the response
            response: The model's response
            quality_pass: Whether the response passed quality check
        """
        self.provider.create(
            action="kmbit.completion",
            erin={
                "container_id": container.id,
                "model_id": model_id,
                "response_length": len(response),
                "quality_pass": quality_pass,
            },
            eraan=[container.tibet_chain_id],
            erachter=f"Completion by {model_id}: {'PASS' if quality_pass else 'FAIL - may escalate'}",
        )

    def should_escalate(self, model_id: str, intent: str) -> bool:
        """
        Check if a model needs escalation for a given intent.

        Returns True if the model can't handle the required context depth.
        """
        required_level = INTENT_LAYER_MAP.get(intent, 0)
        return not self.gate.carbonara_test(model_id, required_level=required_level)

    def suggest_model(self, intent: str) -> Optional[str]:
        """
        Suggest the best model for an intent based on capability profile.

        Returns the smallest model that can handle the required layer.
        """
        required_level = INTENT_LAYER_MAP.get(intent, 0)
        spec = self.profile.get_spec(required_level)
        if spec is None:
            return None

        # Find smallest registered model that meets the requirement
        candidates = [
            (model_id, cap) for model_id, cap in self.gate.model_registry.items()
            if cap >= spec.min_capability
        ]
        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def _build_deep_context(self, text: str, intent: str) -> str:
        """Build deep context for complex intents."""
        return (
            f"=== Deep Context ===\n"
            f"Intent: {intent}\n"
            f"Request requires deep analysis.\n"
            f"Full context available for model with L2 access."
        )
