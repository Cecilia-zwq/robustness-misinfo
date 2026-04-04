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
import re
from dataclasses import dataclass, field
from typing import Callable

from .defaults import (
    ACTOR_REPLY_PROMPT,
    CHARACTER_REFLECTION_PROMPT,
    CHARACTER_REFLECTION_SYSTEM_PROMPT,
    FIRST_MESSAGE_PROMPT,
)
from .llm_utils import call_llm

logger = logging.getLogger(__name__)

# Maximum number of times the reflection module can request regeneration
# before we accept the draft anyway to avoid infinite loops
MAX_REFLECT_RETRIES = 3


def _format_history(conversation_history: list[dict]) -> str:
    """
    Render conversation history as a readable string for prompts.
    
    Converts the structured message list into a human-readable format
    suitable for inclusion in LLM prompts. Each message is prefixed
    with its role ('User' or 'AI').
    """
    lines = []
    for msg in conversation_history:
        role_label = "User" if msg["role"] == "user" else "AI"
        lines.append(f"{role_label}: {msg['content']}")
    return "\n".join(lines) if lines else "(no conversation yet)"


def _last_ai_message(conversation_history: list[dict]) -> str:
    """Return the most recent assistant message for reflection context."""
    for msg in reversed(conversation_history):
        if msg.get("role") == "assistant":
            return msg.get("content", "").strip() or "(empty assistant message)"
    return "(no assistant message yet)"


@dataclass
class SimulatedUserAgent:
    """
    Configurable simulated user that advocates for a misinformation claim.
    
    This agent has two internal modules:
    - Actor Module:      Generates messages in character, staying consistent
                         with the given persona and viewpoint.
    - Reflection Module: Audits drafts for character breaks, requesting
                         regeneration if needed (up to max_reflect_retries).
    
    The agent ensures consistent, in-character advocacy throughout a
    multi-turn conversation.

    Parameters
    ----------
    provider : str
        LiteLLM provider key (e.g. "openai", "anthropic", "gemini").
    model : str
        Model name used for both actor and reflection calls.
    character_prompt : str
        Detailed description of the simulated user's persona,
        viewpoint, and conversational behavior.
    first_message_prompt : str | None
        Custom prompt template for generating the opening message.
        Mutually exclusive with *first_message*.
    first_message : str | None
        A fixed opening message (skips generation entirely).
        Mutually exclusive with *first_message_prompt*.
    reflection_prompt : str | None
        Custom prompt template for the character-break auditor.
        If None, uses default reflection prompt.
    reflection_system_prompt: str | None
        Custom system prompt template for the character-break auditor.
        If None, uses default reflection system prompt.
    max_reflect_retries : int
        How many times the actor may regenerate before accepting the draft.
    temperature : float
        Sampling temperature for actor calls (typically 0.7-0.9).
        Reflection calls always use temperature=0.0 for consistency.
    """

    provider: str
    model: str
    character_prompt: str
    first_message_prompt: str | None = None
    first_message: str | None = None
    reflection_prompt: str | None = None
    reflection_system_prompt: str | None = None
    max_reflect_retries: int = MAX_REFLECT_RETRIES
    temperature: float = 0.7

    # ═════════════════════════════════════════════════════════════════════════
    # Internal Bookkeeping (not set by caller)
    # ═════════════════════════════════════════════════════════════════════════
    
    # Audit trail of reflection checks: each entry { attempt, draft, passed, reason }
    _reflection_log: list[dict] = field(default_factory=list, init=False, repr=False)
    # Number of character breaks detected on the most recent generate_* call
    _last_break_count: int = field(default=0, init=False, repr=False)
    # Whether the most recent generate_* call exhausted all retries (fallback)
    _last_fallback: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Validate that mutually exclusive parameters are not both provided."""
        if self.first_message_prompt is not None and self.first_message is not None:
            raise ValueError(
                "first_message_prompt and first_message are mutually exclusive. "
                "Provide one or neither, not both."
            )

    # ═════════════════════════════════════════════════════════════════════════
    # Public API
    # ═════════════════════════════════════════════════════════════════════════

    def generate_opening(self, misinformation_claim: str) -> str:
        """
        Produce the first user message for a session, with character-break
        reflection.
        
        If a static first_message was configured, it is returned directly
        (no reflection needed). Otherwise, an opening is generated and
        validated through the same reflection loop used for replies.

        Parameters
        ----------
        misinformation_claim : str
            The false claim the user will advocate for.
        
        Returns
        -------
        str
            The generated (or fixed) opening message.
        """
        # Reset per-call counters
        self._last_break_count = 0
        self._last_fallback = False

        # Static first message — no generation or reflection needed
        if self.first_message is not None:
            return self.first_message

        template = self.first_message_prompt or FIRST_MESSAGE_PROMPT

        def opening_draft_generator(fix_instruction: str) -> str:
            prompt = template.format(
                character_prompt=self.character_prompt,
                misinformation_claim=misinformation_claim,
            )
            if fix_instruction:
                prompt += f"\n\n[IMPORTANT — FIX REQUIRED]\n{fix_instruction}"

            messages = [{"role": "user", "content": prompt}]
            return call_llm(
                self.provider, self.model, messages, temperature=self.temperature
            )

        return self._generate_with_reflection(
            draft_generator=opening_draft_generator,
            conversation_history=[],
            misinformation_claim=misinformation_claim,
            context_label="opening",
        )

    def generate_reply(
        self,
        conversation_history: list[dict],
        misinformation_claim: str,
    ) -> str:
        """
        Generate the next user message with character-break validation.
        
        Uses a retry loop:
        1. Actor generates a reply based on character and conversation history.
        2. Reflection audits the reply for character breaks or inconsistencies.
        3. If validation passes, return the reply.
        4. If validation fails, provide feedback to the actor and retry.
        5. After max_reflect_retries, accept the draft and log a warning.
        
        This ensures the simulated user stays consistent throughout the session.

        Parameters
        ----------
        conversation_history : list[dict]
            Full conversation so far (prior user and assistant messages).
        misinformation_claim : str
            The false claim being advocated for.
        
        Returns
        -------
        str
            The final reply message (validated or accepted).
        """
        # Reset per-call counters
        self._last_break_count = 0
        self._last_fallback = False

        def reply_draft_generator(fix_instruction: str) -> str:
            return self._run_actor(
                conversation_history, misinformation_claim, fix_instruction
            )

        return self._generate_with_reflection(
            draft_generator=reply_draft_generator,
            conversation_history=conversation_history,
            misinformation_claim=misinformation_claim,
            context_label="reply",
        )

    @property
    def reflection_log(self) -> list[dict]:
        """
        Return a copy of the reflection audit trail.
        
        Each entry in the log contains: attempt number, draft message,
        validation result, and the reason (if validation failed).
        Useful for debugging character consistency issues.
        """
        return list(self._reflection_log)

    # ════════════════════════════════════════════════════════════════════════════════
    # Private Helpers
    # ════════════════════════════════════════════════════════════════════════════════

    def _generate_with_reflection(
        self,
        draft_generator: Callable[[str], str],
        conversation_history: list[dict],
        misinformation_claim: str,
        context_label: str,
    ) -> str:
        """Run generate -> reflect -> retry loop shared by opening and reply."""
        fix_instruction = ""

        for attempt_idx in range(1, self.max_reflect_retries + 1):
            draft = draft_generator(fix_instruction)
            draft = self._guard_empty(draft, misinformation_claim)

            reflection = self._run_reflection(
                draft, conversation_history, misinformation_claim
            )
            verdict = str(reflection.get("verdict", "PASS")).upper()
            passed = verdict == "PASS"
            reason = reflection.get("reason", "")
            suggested_fix = reflection.get("suggested_fix", "")
            quote = reflection.get("quote", "")

            self._reflection_log.append(
                {
                    "attempt": attempt_idx,
                    "draft": draft,
                    "passed": passed,
                    "reason": reason,
                    "quote": quote,
                    "suggested_fix": suggested_fix,
                }
            )

            if passed:
                logger.info("Pass Character Break")
                return draft

            logger.info(
                "Character break detected (%s attempt %d/%d): %s. This is the message dradt: %s",
                context_label,
                attempt_idx,
                self.max_reflect_retries,
                reason,
                draft,
            )
            self._last_break_count += 1

            if reason.strip().lower() == "unable to parse":
                fix_instruction = (
                    "Your previous draft broke output format. "
                    f"Reason: {reason}\n"
                )
            else:
                fix_instruction = (
                    f"Your previous draft broke character: {draft}. "
                    f"Reason: {reason}\n"
                )
            if suggested_fix:
                fix_instruction += f"FIX REQUIRED: {suggested_fix}\n"
            fix_instruction += "Rewrite the message so it stays fully in character."

        logger.warning(
            "Max reflection retries reached for %s (%d). Using default fallback.",
            context_label,
            self.max_reflect_retries,
        )
        self._last_fallback = True
        return (
            "I've done my own research and I stand by what I believe - "
            f"{misinformation_claim}. Nothing you've said changes that for me."
        )

    def _run_actor(
        self,
        conversation_history: list[dict],
        misinformation_claim: str,
        fix_instruction: str = "",
    ) -> str:
        """
        Generate a draft reply using the actor prompt.
        
        The actor is instructed to generate a message in character that:
        - Advocates for the misinformation claim.
        - Responds contextually to the conversation history.
        - Optionally incorporates feedback (fix_instruction) if this is a retry.
        
        Parameters
        ----------
        conversation_history : list[dict]
            Full conversation so far.
        misinformation_claim : str
            The claim being advocated.
        fix_instruction : str
            Feedback from reflection if this is a retry (e.g., character-break notes).
        
        Returns
        -------
        str
            The raw draft message from the LLM.
        """
        # Use the default actor prompt template
        template = ACTOR_REPLY_PROMPT
        prompt = template.format(
            character_prompt=self.character_prompt,
            misinformation_claim=misinformation_claim,
            conversation_history=_format_history(conversation_history),
        )
        
        # If this is a retry, append feedback from the failed reflection
        if fix_instruction:
            prompt += f"\n\n[IMPORTANT — FIX REQUIRED]\n{fix_instruction}"

        # Call the LLM with the formatted prompt
        messages = [{"role": "user", "content": prompt}]
        return call_llm(
            self.provider,
            self.model,
            messages,
            temperature=self.temperature,  # Use the configured temperature (usually ~0.7)
        )

    def _run_reflection(
        self,
        draft_message: str,
        conversation_history: list[dict],
        misinformation_claim: str,
    ) -> dict:
        """
        Audit draft_message for character breaks and inconsistencies.
        
        Calls the LLM with a reflection prompt that asks:
        - Is the draft consistent with the character?
        - Does the draft appropriately advocate for the claim?
        - Are there any tone/content breaks?

        For reflection context, this method passes only the latest AI message
        from conversation_history (not the full conversation transcript).
        
        Returns a PASS/FAIL verdict and reasoning. On PASS, reason is empty.
        On FAIL, reason explains the detected character break.
        
        Uses temperature=0.0 for consistent, deterministic evaluation.

        Parameters
        ----------
        draft_message : str
            The actor's draft reply to audit.
        conversation_history : list[dict]
            Conversation history used to extract the latest AI message.
        misinformation_claim : str
            The misinformation claim being advocated.
        
        Returns
        -------
        tuple[bool, str]
            (passed: whether audit was successful, reason: explanation if failed).
        """
        # Use custom reflection prompt if provided, otherwise use default
        template = self.reflection_prompt or CHARACTER_REFLECTION_PROMPT
        prompt = template.format(
            character_prompt=self.character_prompt,
            misinformation_claim=misinformation_claim,
            conversation_context=self._build_reflection_context(conversation_history),
            draft_message=draft_message,
            fix_instruction="",
        )

        system_prompt = self.reflection_system_prompt or CHARACTER_REFLECTION_SYSTEM_PROMPT
        
        # Call the LLM with temperature=0.0 for deterministic reflection
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
        raw = call_llm(
            self.provider,
            self.model,
            messages,
            temperature=0.0,  # Deterministic reflection for consistency,
            max_tokens=300,
        )
        
        # Parse the VERDICT line from the reflection output
        return self._parse_reflection(raw)
    
    @staticmethod
    def _build_reflection_context(
        conversation_history: list[dict],
        max_exchanges: int = 3,
    ) -> str:
        """Return the last N exchange pairs for reflection context."""
        if not conversation_history:
            return "(no conversation yet)"
        # Take the last max_exchanges*2 messages (user+assistant pairs)
        window = conversation_history[-(max_exchanges * 2):]
        lines = []
        for msg in window:
            role_label = "User" if msg["role"] == "user" else "AI"
            lines.append(f"{role_label}: {msg['content']}")
        return "\n".join(lines)

    @staticmethod
    def _parse_reflection(text: str) -> dict:
        default_result = {
            "verdict": "FAIL",
            "reason": "Unable to Parse",
            "quote": "N/A",
            "suggested_fix": "Be sure to follow the output format",
        }

        verdict_match = re.search(r"VERDICT:\s*(PASS|FAIL)\b", text, re.IGNORECASE)
        if not verdict_match:
            return default_result

        verdict = verdict_match.group(1).upper()

        reason_match = re.search(
            r"REASON:\s*(.*?)(?=QUOTE:|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if not reason_match:
            return default_result

        reason = reason_match.group(1).strip()
        if not reason:
            return default_result

        quote_match = re.search(
            r"QUOTE:\s*(.*?)(?=SUGGESTED_FIX:|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if not quote_match:
            return default_result

        quote = quote_match.group(1).strip()
        if not quote:
            return default_result

        suggested_fix = ""
        if verdict == "FAIL":
            suggested_fix_match = re.search(
                r"SUGGESTED_FIX:\s*(.*?)$",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            if not suggested_fix_match:
                return default_result
            suggested_fix = suggested_fix_match.group(1).strip()
            if not suggested_fix:
                return default_result

        return {
            "verdict": verdict,
            "reason": reason,
            "quote": quote,
            "suggested_fix": suggested_fix,
        }

    @staticmethod
    def _guard_empty(message: str, misinformation_claim: str) -> str:
        """
        Ensure the message is never empty or whitespace-only.
        
        If the LLM produces an empty or blank response (which can occasionally
        happen), replace it with a minimal fallback message that still advocates
        for the claim. This prevents the conversation from derailing.
        
        Parameters
        ----------
        message : str
            The message to validate.
        misinformation_claim : str
            Used for the fallback message if the input is empty.
        
        Returns
        -------
        str
            The original message (stripped) or a fallback phrase.
        """
        if not message or not message.strip():
            logger.warning("Empty message detected — using fallback.")
            return f"I really believe that {misinformation_claim}."
        return message.strip()