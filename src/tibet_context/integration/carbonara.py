"""
End-to-end Carbonara Demo — the full flow.

Demonstrates the complete tibet-context pipeline:
1. User asks a question
2. KmBiT classifies intent -> routes to small model
3. Small model answers with L0 context only
4. Quality check fails (carbonara test)
5. KmBiT escalates to big model with full context package
6. Big model answers with L0+L1+L2 context
7. Quality check passes
8. TIBET chain records everything

Can run in two modes:
- Mock mode (no LLM needed, for testing)
- Live mode (requires Ollama and/or Gemini API)
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

from tibet_core import Provider, Chain

from ..builder import ContextBuilder
from ..container import ContextContainer
from ..gate import CapabilityGate
from ..layers import CapabilityProfile
from ..reader import ContextReader
from .kmbit import KmBiTBridge
from .provider import ContextProvider
from .llm import LLMBackend, OllamaBackend, GeminiBackend, LLMResponse

log = logging.getLogger("tibet-context.carbonara")


@dataclass
class CarbonaraResult:
    """Full result of a carbonara flow run."""
    question: str
    # Phase 1: Small model
    small_model: str
    small_response: str
    small_quality_pass: bool
    small_context_tokens: int
    # Phase 2: Escalation (if needed)
    escalated: bool
    big_model: Optional[str]
    big_response: Optional[str]
    big_quality_pass: Optional[bool]
    big_context_tokens: Optional[int]
    # Provenance
    tibet_token_count: int
    container_id: str
    chain_id: str


def run_carbonara_mock(
    question: str = "Hoe maak ik pasta carbonara?",
    profile: Optional[CapabilityProfile] = None,
) -> CarbonaraResult:
    """
    Run the carbonara demo in mock mode (no LLM needed).

    Perfect for testing and CI. Uses hardcoded responses.
    """
    ctx_provider = ContextProvider(
        actor="carbonara-demo",
        profile=profile,
    )
    bridge = KmBiTBridge(
        provider=ctx_provider.provider,
        profile=ctx_provider.profile,
    )

    small_model = "qwen2.5:3b"
    big_model = "qwen2.5:32b"

    # --- Phase 1: Small model attempt ---
    ctx_provider.provider.create(
        action="carbonara.start",
        erin={"question": question},
        erachter="Starting carbonara demo flow",
    )

    # KmBiT classifies: simple intent
    container = bridge.on_request(
        text=question, intent="simple", model_id=small_model,
    )

    # Small model "answers" with only L0 context
    small_context = ctx_provider.read(container, small_model)
    small_response = "Kook pasta, doe ei erop."  # Mock: bad answer
    small_quality_pass = False  # Carbonara test FAIL

    bridge.on_completion(container, small_model, small_response, small_quality_pass)

    small_tokens = ctx_provider.reader.accessible_token_count(container, small_model)

    # --- Phase 2: Escalation ---
    escalation = bridge.on_escalation(
        from_model=small_model,
        to_model=big_model,
        reason="Carbonara test FAIL: rauw ei op bord",
        conversation=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": small_response},
            {"role": "system", "content": f"Quality FAIL: {small_response}"},
        ],
        failed_response=small_response,
    )

    # Big model "answers" with full context (L0+L1+L2)
    big_response = (
        "Voor echte pasta carbonara heb je nodig: guanciale (geen bacon!), "
        "pecorino romano, eigeel, zwarte peper en spaghetti of rigatoni. "
        "Het geheim is tempering: meng eigeel met geraspte pecorino, "
        "voeg langzaam heet pastawater toe zodat je een romige saus krijgt "
        "zonder dat het ei stolt. Bak de guanciale knapperig uit, "
        "meng met de al dente pasta en roer dan van het vuur af "
        "het ei-kaasmengsel erdoor. Nooit room toevoegen!"
    )
    big_quality_pass = True  # Carbonara test PASS

    bridge.on_completion(
        escalation.container, big_model, big_response, big_quality_pass,
    )

    big_tokens = ctx_provider.reader.accessible_token_count(
        escalation.container, big_model,
    )

    ctx_provider.provider.create(
        action="carbonara.complete",
        erin={
            "small_pass": small_quality_pass,
            "big_pass": big_quality_pass,
            "escalated": True,
        },
        erachter="Carbonara demo complete: escalation successful",
    )

    return CarbonaraResult(
        question=question,
        small_model=small_model,
        small_response=small_response,
        small_quality_pass=small_quality_pass,
        small_context_tokens=small_tokens,
        escalated=True,
        big_model=big_model,
        big_response=big_response,
        big_quality_pass=big_quality_pass,
        big_context_tokens=big_tokens,
        tibet_token_count=ctx_provider.token_count,
        container_id=escalation.container.id,
        chain_id=escalation.container.tibet_chain_id,
    )


def run_carbonara_live(
    question: str = "Hoe maak ik pasta carbonara?",
    small_backend: Optional[LLMBackend] = None,
    big_backend: Optional[LLMBackend] = None,
    quality_checker: Optional[LLMBackend] = None,
    profile: Optional[CapabilityProfile] = None,
) -> CarbonaraResult:
    """
    Run the carbonara demo with real LLM backends.

    Args:
        question: The question to ask
        small_backend: Backend for small model (default: Ollama qwen2.5:3b)
        big_backend: Backend for big model (default: Gemini 2.0 Flash)
        quality_checker: Backend for quality assessment (default: same as big)
        profile: Optional capability profile

    Returns:
        CarbonaraResult with real responses
    """
    small = small_backend or OllamaBackend(model="qwen2.5:3b")
    big = big_backend or GeminiBackend()

    ctx_provider = ContextProvider(actor="carbonara-live", profile=profile)
    bridge = KmBiTBridge(
        provider=ctx_provider.provider,
        profile=ctx_provider.profile,
    )

    small_model = small.name
    big_model = big.name

    # Register models in gate
    ctx_provider.gate.register_model(small_model, 3)
    ctx_provider.gate.register_model(big_model, 200)  # External API = high cap

    # --- Phase 1: Small model ---
    ctx_provider.provider.create(
        action="carbonara.start",
        erin={"question": question, "small_backend": small_model, "big_backend": big_model},
        erachter="Starting live carbonara flow",
    )

    container = bridge.on_request(
        text=question, intent="simple", model_id=small_model,
    )

    small_context = ctx_provider.read(container, small_model)
    log.info(f"Small model context ({small_model}): {len(small_context)} chars")

    small_llm_response = small.generate(
        prompt=question,
        system_prompt="Beantwoord kort en bondig in het Nederlands.",
        max_tokens=150,
        temperature=0.3,
    )
    small_response = small_llm_response.text

    # Quality check: does the response mention key carbonara techniques?
    small_quality_pass = _quality_check(small_response)

    bridge.on_completion(container, small_model, small_response, small_quality_pass)

    small_tokens = ctx_provider.reader.accessible_token_count(container, small_model)

    if small_quality_pass:
        # Small model passed — no escalation needed
        ctx_provider.provider.create(
            action="carbonara.complete",
            erin={"small_pass": True, "escalated": False},
            erachter="Small model passed carbonara test — no escalation needed",
        )
        return CarbonaraResult(
            question=question,
            small_model=small_model,
            small_response=small_response,
            small_quality_pass=True,
            small_context_tokens=small_tokens,
            escalated=False,
            big_model=None,
            big_response=None,
            big_quality_pass=None,
            big_context_tokens=None,
            tibet_token_count=ctx_provider.token_count,
            container_id=container.id,
            chain_id=container.tibet_chain_id,
        )

    # --- Phase 2: Escalation ---
    log.info(f"Quality FAIL — escalating to {big_model}")

    escalation = bridge.on_escalation(
        from_model=small_model,
        to_model=big_model,
        reason=f"Carbonara test FAIL: response lacks key techniques",
        conversation=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": small_response},
        ],
        failed_response=small_response,
    )

    # Big model gets full context
    big_prompt = (
        f"{escalation.context_for_model}\n\n"
        f"De vorige poging was onvoldoende: \"{small_response}\"\n"
        f"Geef nu een correct en gedetailleerd antwoord.\n\n"
        f"Vraag: {question}"
    )

    big_llm_response = big.generate(
        prompt=big_prompt,
        system_prompt="Je bent een culinair expert. Geef gedetailleerde, correcte antwoorden in het Nederlands.",
        max_tokens=500,
        temperature=0.5,
    )
    big_response = big_llm_response.text
    big_quality_pass = _quality_check(big_response)

    bridge.on_completion(
        escalation.container, big_model, big_response, big_quality_pass,
    )

    big_tokens = ctx_provider.reader.accessible_token_count(
        escalation.container, big_model,
    )

    ctx_provider.provider.create(
        action="carbonara.complete",
        erin={
            "small_pass": False,
            "big_pass": big_quality_pass,
            "escalated": True,
        },
        erachter=f"Carbonara flow complete: big model {'PASS' if big_quality_pass else 'FAIL'}",
    )

    return CarbonaraResult(
        question=question,
        small_model=small_model,
        small_response=small_response,
        small_quality_pass=False,
        small_context_tokens=small_tokens,
        escalated=True,
        big_model=big_model,
        big_response=big_response,
        big_quality_pass=big_quality_pass,
        big_context_tokens=big_tokens,
        tibet_token_count=ctx_provider.token_count,
        container_id=escalation.container.id,
        chain_id=escalation.container.tibet_chain_id,
    )


def _quality_check(response: str) -> bool:
    """
    Simple carbonara quality check.

    Checks if the response mentions at least 2 key carbonara elements.
    In production this would use an LLM for assessment.
    """
    keywords = [
        "guanciale", "pecorino", "eigeel", "egg yolk",
        "tempering", "temperen", "peper", "pepper",
        "al dente", "geen room", "no cream", "spaghetti", "rigatoni",
    ]
    response_lower = response.lower()
    matches = sum(1 for kw in keywords if kw in response_lower)
    return matches >= 2


def print_result(result: CarbonaraResult) -> None:
    """Pretty-print a carbonara result."""
    print("=" * 60)
    print("CARBONARA FLOW RESULT")
    print("=" * 60)
    print(f"\nQuestion: {result.question}")
    print(f"\n--- Phase 1: {result.small_model} ---")
    print(f"Context tokens: {result.small_context_tokens}")
    print(f"Response: {result.small_response[:200]}")
    print(f"Quality: {'PASS' if result.small_quality_pass else 'FAIL'}")

    if result.escalated:
        print(f"\n--- Phase 2: {result.big_model} (escalated) ---")
        print(f"Context tokens: {result.big_context_tokens}")
        print(f"Response: {result.big_response[:300]}")
        print(f"Quality: {'PASS' if result.big_quality_pass else 'FAIL'}")
    else:
        print("\nNo escalation needed.")

    print(f"\n--- TIBET Provenance ---")
    print(f"Tokens created: {result.tibet_token_count}")
    print(f"Container: {result.container_id}")
    print(f"Chain: {result.chain_id}")
    print("=" * 60)
