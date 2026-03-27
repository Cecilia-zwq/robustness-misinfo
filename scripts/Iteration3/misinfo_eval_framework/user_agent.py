"""
Simulated User Agent
====================
Contains two internal modules:
  - Actor module:      generates messages in character.
  - Reflection module: audits drafts for character breaks and triggers retries.

Public API
----------
    agent = SimulatedUserAgent(provider, model, character_prompt, ...)
    opening = agent.generate_opening(misinformation_claim)
    reply   = agent.generate_reply(conversation_history, misinformation_claim)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .defaults import (
    ACTOR_REPLY_PROMPT,
    CHARACTER_REFLECTION_PROMPT,
    FIRST_MESSAGE_PROMPT,
)
from .llm_utils import call_llm

logger = logging.getLogger(__name__)

MAX_REFLECT_RETRIES = 3  # max regeneration attempts before accepting anyway


def _format_history(conversation_history: list[dict]) -> str:
    """Render conversation history as a readable string for prompts."""
    lines = []
    for msg in conversation_history:
        role_label = "User" if msg["role"] == "user" else "AI"
        lines.append(f"{role_label}: {msg['content']}")
    return "\n".join(lines) if lines else "(no conversation yet)"


@dataclass
class SimulatedUserAgent:
    """
    Configurable simulated user that advocates for a misinformation claim.

    Parameters
    ----------
    provider : str
        LiteLLM provider key (e.g. "openai", "anthropic", "gemini").
    model : str
        Model name used for both actor and reflection calls.
    character_prompt : str
        Describes the simulated user's persona and behaviour.
    first_message_prompt : str | None
        Custom prompt template for generating the opening message.
        Mutually exclusive with *first_message*.
    first_message : str | None
        A fixed opening message (skips generation entirely).
        Mutually exclusive with *first_message_prompt*.
    reflection_prompt : str | None
        Custom prompt template for the character-break auditor.
    max_reflect_retries : int
        How many times the actor may regenerate before we accept the draft.
    temperature : float
        Sampling temperature for actor calls.
    """

    provider: str
    model: str
    character_prompt: str
    first_message_prompt: str | None = None
    first_message: str | None = None
    reflection_prompt: str | None = None
    max_reflect_retries: int = MAX_REFLECT_RETRIES
    temperature: float = 0.7

    # ── internal bookkeeping (not set by the caller) ──
    _reflection_log: list[dict] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if self.first_message_prompt is not None and self.first_message is not None:
            raise ValueError(
                "first_message_prompt and first_message are mutually exclusive. "
                "Provide one or neither, not both."
            )

    # ── public API ───────────────────────────────────────────────────────────

    def generate_opening(self, misinformation_claim: str) -> str:
        """
        Produce the first user message for a session.

        If *first_message* was set at init, return it directly.
        Otherwise, use the first-message prompt (custom or default) to generate one.
        """
        if self.first_message is not None:
            return self.first_message

        template = self.first_message_prompt or FIRST_MESSAGE_PROMPT
        prompt = template.format(
            character_prompt=self.character_prompt,
            misinformation_claim=misinformation_claim,
        )
        messages = [{"role": "user", "content": prompt}]
        opening = call_llm(
            self.provider, self.model, messages, temperature=self.temperature
        )
        opening = self._guard_empty(opening, misinformation_claim)
        return opening

    def generate_reply(
        self,
        conversation_history: list[dict],
        misinformation_claim: str,
    ) -> str:
        """
        Generate the next user message, subject to character-break reflection.

        Returns the final (possibly regenerated) message.
        """
        fix_instruction = ""
        for attempt in range(1, self.max_reflect_retries + 1):
            draft = self._run_actor(
                conversation_history, misinformation_claim, fix_instruction
            )
            draft = self._guard_empty(draft, misinformation_claim)

            passed, reason = self._run_reflection(
                draft, conversation_history, misinformation_claim
            )
            self._reflection_log.append(
                {
                    "attempt": attempt,
                    "draft": draft,
                    "passed": passed,
                    "reason": reason,
                }
            )

            if passed:
                return draft

            logger.info(
                "Character break detected (attempt %d/%d): %s",
                attempt,
                self.max_reflect_retries,
                reason,
            )
            fix_instruction = (
                f"Your previous draft broke character. Reason: {reason}\n"
                "Rewrite the message so it stays fully in character."
            )

        # All retries exhausted → accept last draft with a warning
        logger.warning(
            "Max reflection retries reached (%d). Accepting last draft.",
            self.max_reflect_retries,
        )
        return draft

    @property
    def reflection_log(self) -> list[dict]:
        """Return a copy of the reflection audit trail."""
        return list(self._reflection_log)

    # ── private helpers ──────────────────────────────────────────────────────

    def _run_actor(
        self,
        conversation_history: list[dict],
        misinformation_claim: str,
        fix_instruction: str = "",
    ) -> str:
        """Generate a draft reply using the actor prompt."""
        template = ACTOR_REPLY_PROMPT
        prompt = template.format(
            character_prompt=self.character_prompt,
            misinformation_claim=misinformation_claim,
            conversation_history=_format_history(conversation_history),
        )
        if fix_instruction:
            prompt += f"\n\n[IMPORTANT — FIX REQUIRED]\n{fix_instruction}"

        messages = [{"role": "user", "content": prompt}]
        return call_llm(
            self.provider, self.model, messages, temperature=self.temperature
        )

    def _run_reflection(
        self,
        draft_message: str,
        conversation_history: list[dict],
        misinformation_claim: str,
    ) -> tuple[bool, str]:
        """
        Audit *draft_message* for character breaks.

        Returns (passed: bool, reason: str).
        """
        template = self.reflection_prompt or CHARACTER_REFLECTION_PROMPT
        prompt = template.format(
            character_prompt=self.character_prompt,
            misinformation_claim=misinformation_claim,
            conversation_history=_format_history(conversation_history),
            draft_message=draft_message,
            fix_instruction="",
        )
        messages = [{"role": "user", "content": prompt}]
        raw = call_llm(
            self.provider, self.model, messages, temperature=0.0
        )
        return self._parse_reflection(raw)

    @staticmethod
    def _parse_reflection(raw: str) -> tuple[bool, str]:
        """Parse the VERDICT line from the reflection output."""
        for line in raw.strip().splitlines():
            line = line.strip()
            if line.upper().startswith("VERDICT:"):
                body = line.split(":", 1)[1].strip()
                if body.upper().startswith("PASS"):
                    return True, ""
                elif body.upper().startswith("FAIL"):
                    reason = body.split("|", 1)[1].strip() if "|" in body else body
                    return False, reason
        # If parsing fails, default to PASS to avoid blocking the session
        logger.warning("Could not parse reflection output; defaulting to PASS.")
        return True, ""

    @staticmethod
    def _guard_empty(message: str, misinformation_claim: str) -> str:
        """Ensure the message is never empty."""
        if not message or not message.strip():
            logger.warning("Empty message detected — using fallback.")
            return f"I really believe that {misinformation_claim}. Can you tell me more?"
        return message.strip()