"""
Target LLM
==========
The model under evaluation. Receives conversation history and produces a
response. By default, no system prompt is applied (reflecting real-world
unconstrained usage).

Public API
----------
    target = TargetLLM(provider, model)
    response, is_empty = target.respond(conversation_history)

The two-value return is the protocol contract used by
``core/conversation.py``. ``is_empty`` is True when the provider
returned no text content (typically a refusal returned as a non-text
content block); in that case ``response`` is the stable placeholder
``llm_utils.EMPTY_TARGET_PLACEHOLDER`` and the conversation loop records
``target_empty=True`` on the corresponding turn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .llm_utils import EMPTY_TARGET_PLACEHOLDER, EmptyLLMResponse, call_llm


logger = logging.getLogger(__name__)


@dataclass
class TargetLLM:
    """
    Wrapper around the model being evaluated.

    Parameters
    ----------
    provider : str
        LiteLLM provider key (e.g. "openai", "anthropic", "gemini").
    model : str
        Model name (e.g. "gpt-4.1-mini-2025-04-14").
    system_prompt : str | None
        Optional system prompt. Default is None (no system prompt) to
        reflect real-world unconstrained usage.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum response length.
    """

    provider: str
    model: str
    system_prompt: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None

    def respond(self, conversation_history: list[dict]) -> tuple[str, bool]:
        """
        Generate a response given the full conversation history.

        Parameters
        ----------
        conversation_history : list[dict]
            OpenAI-style message list with alternating user/assistant roles.

        Returns
        -------
        (text, is_empty) : tuple[str, bool]
            ``text`` is the model's response, or
            ``llm_utils.EMPTY_TARGET_PLACEHOLDER`` if the provider returned
            no content. ``is_empty`` is True iff the provider produced no
            text. The conversation loop persists this flag per turn.
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(conversation_history)

        try:
            text = call_llm(
                self.provider,
                self.model,
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return text, False
        except EmptyLLMResponse as exc:
            logger.warning(
                "Target %s/%s returned empty content after %d attempt(s) "
                "(finish_reason=%r, native_stop_reason=%r); "
                "substituting placeholder and continuing the session.",
                self.provider,
                self.model,
                exc.attempts,
                exc.finish_reason,
                exc.native_stop_reason,
            )
            return EMPTY_TARGET_PLACEHOLDER, True
