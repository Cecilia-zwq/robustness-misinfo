"""
Target LLM
==========
The model under evaluation. Receives conversation history and produces a
response. By default, no system prompt is applied (reflecting real-world
unconstrained usage).

Public API
----------
    target = TargetLLM(provider, model)
    response = target.respond(conversation_history)
"""

from __future__ import annotations

from dataclasses import dataclass

from .llm_utils import call_llm


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
    max_tokens: int = 1024

    def respond(self, conversation_history: list[dict]) -> str:
        """
        Generate a response given the full conversation history.

        Parameters
        ----------
        conversation_history : list[dict]
            OpenAI-style message list with alternating user/assistant roles.

        Returns
        -------
        str
            The model's response text.
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(conversation_history)

        return call_llm(
            self.provider,
            self.model,
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )