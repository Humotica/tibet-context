"""
tibet-core Provider integration for tibet-context.

Every context operation (build, read, compact, gate check) creates
TIBET tokens for full provenance tracking.
"""

from typing import Any, Dict, List, Optional

from tibet_core import Provider, Token

from ..builder import ContextBuilder
from ..container import ContextContainer
from ..gate import CapabilityGate
from ..reader import ContextReader
from ..layers import CapabilityProfile


class ContextProvider:
    """
    TIBET-aware context provider.

    Wraps ContextBuilder, ContextReader, and CapabilityGate
    with automatic TIBET token creation for every operation.
    """

    def __init__(
        self,
        actor: str = "tibet-context",
        provider: Optional[Provider] = None,
        profile: Optional[CapabilityProfile] = None,
    ):
        self.provider = provider or Provider(actor=actor)
        self.profile = profile or CapabilityProfile.default()
        self.gate = CapabilityGate(profile=self.profile)
        self.builder = ContextBuilder(provider=self.provider, profile=self.profile)
        self.reader = ContextReader(gate=self.gate)

    def build_from_conversation(
        self,
        messages: List[Dict[str, str]],
        session_id: Optional[str] = None,
        summary: Optional[str] = None,
        deep_context: Optional[str] = None,
    ) -> ContextContainer:
        """Build container from conversation with TIBET tracking."""
        return self.builder.from_conversation(
            messages, session_id=session_id,
            summary=summary, deep_context=deep_context,
        )

    def build_from_chain(self, chain_id: str) -> ContextContainer:
        """Build container from TIBET chain."""
        return self.builder.from_chain(chain_id)

    def read(self, container: ContextContainer, model_id: str) -> str:
        """Read container with capability filtering + TIBET tracking."""
        token = self.provider.create(
            action="context.read",
            erin={"container_id": container.id, "model_id": model_id},
            erachter=f"Reading context for model {model_id}",
        )
        return self.reader.read(container, model_id)

    def gate_check(self, container: ContextContainer, model_id: str) -> Dict[str, Any]:
        """Run gate check with TIBET tracking."""
        report = self.gate.gate_report(container, model_id)
        self.provider.create(
            action="context.gate_check",
            erin=report,
            erachter=f"Gate check for {model_id}: carbonara={'PASS' if report['carbonara_pass'] else 'FAIL'}",
        )
        return report

    def carbonara_test(self, model_id: str) -> bool:
        """Run carbonara test with TIBET tracking."""
        result = self.gate.carbonara_test(model_id)
        self.provider.create(
            action="context.carbonara_test",
            erin={"model_id": model_id, "pass": result},
            erachter=f"Carbonara test for {model_id}: {'PASS' if result else 'FAIL (zakjapanner)'}",
        )
        return result

    def escalate(
        self,
        container: ContextContainer,
        from_model: str,
        to_model: str,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Record an escalation event in the TIBET chain.

        This is the core carbonara flow: a small model failed,
        so we escalate to a bigger model with more context.
        """
        token = self.provider.create(
            action="context.escalate",
            erin={
                "container_id": container.id,
                "from_model": from_model,
                "to_model": to_model,
            },
            eraan=[container.tibet_chain_id],
            erachter=f"Escalation: {from_model} -> {to_model}. Reason: {reason}",
        )
        return {
            "token_id": token.token_id,
            "from_model": from_model,
            "to_model": to_model,
            "reason": reason,
            "from_layers": self.gate.max_accessible_level(from_model),
            "to_layers": self.gate.max_accessible_level(to_model),
        }

    def record_completion(
        self,
        container: ContextContainer,
        model_id: str,
        response: str,
        quality_pass: bool,
    ) -> Token:
        """Record a model completion with quality assessment."""
        return self.provider.create(
            action="context.completion",
            erin={
                "container_id": container.id,
                "model_id": model_id,
                "response_length": len(response),
                "quality_pass": quality_pass,
            },
            eraan=[container.tibet_chain_id],
            erachter=f"Completion by {model_id}: quality={'PASS' if quality_pass else 'FAIL'}",
        )

    @property
    def token_count(self) -> int:
        """Number of TIBET tokens created."""
        return self.provider.count

    def export_chain(self, format: str = "dict") -> Any:
        """Export all TIBET tokens."""
        return self.provider.export(format=format)
