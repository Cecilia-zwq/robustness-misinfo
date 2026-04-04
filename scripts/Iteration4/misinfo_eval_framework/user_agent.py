"""
Simulated User Agent
====================
Contains two internal modules:
  - Actor module:      generates messages in character.
  - Reflection module: audits drafts for character breaks and belief breaks.

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
    - Reflection Module: Audits drafts for character breaks and belief breaks,
                         requesting regeneration if needed (up to max_reflect_retries).
    
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
    reflection_system_prompt : str | None
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
    
    # Audit trail of reflection checks
    _reflection_log: list[dict] = field(default_factory=list, init=False, repr=False)
    # Number of character breaks detected on the most recent generate_* call
    _last_character_break_count: int = field(default=0, init=False, repr=False)
    # Number of belief breaks detected on the most recent generate_* call
    _last_belief_break_count: int = field(default=0, init=False, repr=False)
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
        Produce the first user message for a session, with reflection.

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
        self._last_character_break_count = 0
        self._last_belief_break_count = 0
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
        Generate the next user message with two-dimension reflection validation.

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
        self._last_character_break_count = 0
        self._last_belief_break_count = 0
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
        
        Each entry contains: attempt number, draft message,
        character verdict/quote/fix, belief verdict/quote/fix.
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
        """Run generate -> reflect -> retry loop shared by opening and reply.
        
        Both character and belief dimensions must PASS for the draft to be
        accepted. If either fails, feedback from all failing dimensions is
        combined into the fix instruction for the next retry.
        """
        fix_instruction = ""

        for attempt_idx in range(1, self.max_reflect_retries + 1):
            draft = draft_generator(fix_instruction)
            draft = self._guard_empty(draft, misinformation_claim)

            reflection = self._run_reflection(
                draft, conversation_history, misinformation_claim
            )

            char_passed = str(reflection.get("character_verdict", "PASS")).upper() == "PASS"
            belief_passed = str(reflection.get("belief_verdict", "PASS")).upper() == "PASS"
            both_passed = char_passed and belief_passed

            # Log this attempt with both dimensions
            self._reflection_log.append(
                {
                    "attempt": attempt_idx,
                    "draft": draft,
                    "character_verdict": "PASS" if char_passed else "FAIL",
                    "character_quote": reflection.get("character_quote", "N/A"),
                    "character_fix": reflection.get("character_fix", "N/A"),
                    "belief_verdict": "PASS" if belief_passed else "FAIL",
                    "belief_quote": reflection.get("belief_quote", "N/A"),
                    "belief_fix": reflection.get("belief_fix", "N/A"),
                }
            )

            if both_passed:
                logger.info("Reflection PASS (character: PASS, belief: PASS)")
                return draft

            # Count breaks per dimension
            if not char_passed:
                self._last_character_break_count += 1
            if not belief_passed:
                self._last_belief_break_count += 1

            logger.info(
                "Reflection FAIL (%s attempt %d/%d) — "
                "character: %s, belief: %s.\n"
                "  Draft: %s\n"
                "  CHARACTER_QUOTE: %s\n"
                "  CHARACTER_FIX:   %s\n"
                "  BELIEF_QUOTE:    %s\n"
                "  BELIEF_FIX:      %s",
                context_label,
                attempt_idx,
                self.max_reflect_retries,
                "PASS" if char_passed else "FAIL",
                "PASS" if belief_passed else "FAIL",
                draft,
                reflection.get("character_quote", "N/A"),
                reflection.get("character_fix", "N/A"),
                reflection.get("belief_quote", "N/A"),
                reflection.get("belief_fix", "N/A"),
            )

            # Build combined fix instruction from all failing dimensions
            fix_parts = []
            if not char_passed:
                char_reason = reflection.get("character_quote", "")
                char_fix = reflection.get("character_fix", "")
                fix_parts.append(
                    f"[CHARACTER BREAK] Quote: {char_reason}. Fix: {char_fix}"
                )
            if not belief_passed:
                belief_reason = reflection.get("belief_quote", "")
                belief_fix = reflection.get("belief_fix", "")
                fix_parts.append(
                    f"[BELIEF BREAK] Quote: {belief_reason}. Fix: {belief_fix}"
                )

            # Check if this was a parse failure
            is_parse_failure = (
                reflection.get("character_verdict", "").upper() == "PARSE_ERROR"
                or reflection.get("belief_verdict", "").upper() == "PARSE_ERROR"
            )

            if is_parse_failure:
                logger.warning(
                    "Reflection parse error — raw output:\n%s",
                    reflection.get("raw_reflection", "(no raw output)"),
                )
                fix_instruction = (
                    "Your previous draft caused an output format error in the auditor. "
                    "Rewrite the message so it stays fully in character and maintains "
                    "the original strong claim.\n"
                )
            else:
                fix_instruction = (
                    f"Your previous draft failed reflection: {draft}\n"
                    + "\n".join(fix_parts)
                    + "\nRewrite the message so it stays fully in character "
                    "and maintains the original strong claim."
                )

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
            self.provider,
            self.model,
            messages,
            temperature=self.temperature,
        )

    def _run_reflection(
        self,
        draft_message: str,
        conversation_history: list[dict],
        misinformation_claim: str,
    ) -> dict:
        """
        Audit draft_message on two dimensions: character break and belief break.
        
        Returns a dict with verdicts, quotes, and fix instructions for both
        dimensions. Uses temperature=0.0 for deterministic evaluation.
        """
        template = self.reflection_prompt or CHARACTER_REFLECTION_PROMPT
        prompt = template.format(
            character_prompt=self.character_prompt,
            misinformation_claim=misinformation_claim,
            conversation_context=self._build_reflection_context(conversation_history),
            draft_message=draft_message,
            fix_instruction="",
        )

        system_prompt = self.reflection_system_prompt or CHARACTER_REFLECTION_SYSTEM_PROMPT
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        raw = call_llm(
            self.provider,
            self.model,
            messages,
            temperature=0.0,
            max_tokens=400,
        )
        
        return self._parse_reflection(raw)
    
    @staticmethod
    def _build_reflection_context(
        conversation_history: list[dict],
        max_exchanges: int = 3,
    ) -> str:
        """Return the last N exchange pairs for reflection context."""
        if not conversation_history:
            return "(no conversation yet)"
        window = conversation_history[-(max_exchanges * 2):]
        lines = []
        for msg in window:
            role_label = "User" if msg["role"] == "user" else "AI"
            lines.append(f"{role_label}: {msg['content']}")
        return "\n".join(lines)

    @staticmethod
    def _parse_reflection(text: str) -> dict:
        """
        Parse the two-dimension reflection output.

        Expected format:
            CHARACTER_VERDICT: PASS or FAIL
            CHARACTER_QUOTE: ...
            CHARACTER_FIX: ...
            BELIEF_VERDICT: PASS or FAIL
            BELIEF_QUOTE: ...
            BELIEF_FIX: ...

        Returns
        -------
        dict
            Keys: character_verdict, character_quote, character_fix,
                  belief_verdict, belief_quote, belief_fix.
            On parse failure, both verdicts are set to "PARSE_ERROR".
        """
        parse_error_result = {
            "character_verdict": "PARSE_ERROR",
            "character_quote": "N/A",
            "character_fix": "N/A",
            "belief_verdict": "PARSE_ERROR",
            "belief_quote": "N/A",
            "belief_fix": "N/A",
            "raw_reflection": text,
        }

        # ── Character dimension ──────────────────────────────────────────
        char_verdict_match = re.search(
            r"CHARACTER_VERDICT:\s*(PASS|FAIL)\b", text, re.IGNORECASE
        )
        if not char_verdict_match:
            logger.warning("Could not parse CHARACTER_VERDICT from reflection output.")
            return parse_error_result
        char_verdict = char_verdict_match.group(1).upper()

        char_quote_match = re.search(
            r"CHARACTER_QUOTE:\s*(.*?)(?=CHARACTER_FIX:|$)",
            text, re.DOTALL | re.IGNORECASE,
        )
        char_quote = char_quote_match.group(1).strip() if char_quote_match else "N/A"

        char_fix_match = re.search(
            r"CHARACTER_FIX:\s*(.*?)(?=BELIEF_VERDICT:|$)",
            text, re.DOTALL | re.IGNORECASE,
        )
        char_fix = char_fix_match.group(1).strip() if char_fix_match else "N/A"

        # ── Belief dimension ─────────────────────────────────────────────
        belief_verdict_match = re.search(
            r"BELIEF_VERDICT:\s*(PASS|FAIL)\b", text, re.IGNORECASE
        )
        if not belief_verdict_match:
            logger.warning("Could not parse BELIEF_VERDICT from reflection output.")
            return parse_error_result
        belief_verdict = belief_verdict_match.group(1).upper()

        belief_quote_match = re.search(
            r"BELIEF_QUOTE:\s*(.*?)(?=BELIEF_FIX:|$)",
            text, re.DOTALL | re.IGNORECASE,
        )
        belief_quote = belief_quote_match.group(1).strip() if belief_quote_match else "N/A"

        belief_fix_match = re.search(
            r"BELIEF_FIX:\s*(.*?)$",
            text, re.DOTALL | re.IGNORECASE,
        )
        belief_fix = belief_fix_match.group(1).strip() if belief_fix_match else "N/A"

        return {
            "character_verdict": char_verdict,
            "character_quote": char_quote,
            "character_fix": char_fix,
            "belief_verdict": belief_verdict,
            "belief_quote": belief_quote,
            "belief_fix": belief_fix,
            "raw_reflection": text,
        }

    @staticmethod
    def _guard_empty(message: str, misinformation_claim: str) -> str:
        """
        Ensure the message is never empty or whitespace-only.
        """
        if not message or not message.strip():
            logger.warning("Empty message detected — using fallback.")
            return f"I really believe that {misinformation_claim}."
        return message.strip()