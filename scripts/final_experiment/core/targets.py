"""
core/targets.py
===============
Target-LLM configuration spec, generalised beyond the original
`TargetLLM(provider, model, system_prompt, temperature, max_tokens)`
shape so that future experiments can vary:

  - thinking on/off (Anthropic extended_thinking, OpenAI reasoning, Gemini thinking_budget)
  - system prompts (the Claude system-prompts release-notes feature)
  - external tool/search use
  - any other provider-specific knob LiteLLM exposes

A `TargetConfig` is a small data record that captures everything needed
to invoke one specific target-LLM configuration. The runner instantiates
a fresh one per session; the artifact embeds it verbatim under
`config.target_config` so re-running with the same record reproduces the
call exactly.

Why a dict-shaped extras field instead of new dataclass fields per knob?
Because every provider invents new ones (LiteLLM gets `reasoning_effort`,
`thinking`, `extra_body`, `tools`, ...) and we don't want to chase them.
The contract is: anything in `extras` is forwarded to `litellm.completion`
as a kwarg. Adding a new RQ1.3 condition is a new `TargetConfig(...)`
value; no code change to the runner or artifact schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import litellm

from misinfo_eval_framework.llm_utils import build_model_string


@dataclass
class TargetConfig:
    """One target-LLM invocation spec.

    Attributes
    ----------
    provider, model
        Routed through LiteLLM via `build_model_string`.
    label
        Filename-safe slug for use in `session_id`. Caller chooses something
        descriptive ("claude-sonnet-4.6" vs "claude-sonnet-4.6-thinking")
        so two configs of the same model don't collide on disk.
    system_prompt
        Prepended to the conversation as a system message. None = no system
        prompt (the Iter-5 default, "real-world unconstrained usage").
    temperature, max_tokens
        Standard sampling controls.
    extras
        Provider-specific kwargs forwarded to `litellm.completion`. This is
        where thinking modes, reasoning effort, tool configs, etc. live.
        Examples (illustrative; check current LiteLLM docs):
          {"thinking": {"type": "enabled", "budget_tokens": 8000}}    # Anthropic
          {"reasoning_effort": "high"}                                # OpenAI
          {"tools": [...], "tool_choice": "auto"}                     # any
    """
    provider: str
    model: str
    label: str
    system_prompt: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = 600
    extras: dict[str, Any] = field(default_factory=dict)

    def respond(self, conversation_history: list[dict]) -> str:
        """Generate a response given conversation history.

        Mirrors the public API of misinfo_eval_framework.target_llm.TargetLLM
        so the conversation loop stays identical.
        """
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(conversation_history)

        response = litellm.completion(
            model=build_model_string(self.provider, self.model),
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.extras,
        )
        return response.choices[0].message.content.strip()

    def to_dict(self) -> dict:
        """JSON-friendly representation for embedding in artifacts.

        `extras` is included as-is — it's already JSON-friendly by
        construction (the caller built it with primitives).
        """
        return asdict(self)


# ════════════════════════════════════════════════════════════════════════════
# Convenience constructors for common configurations
# ════════════════════════════════════════════════════════════════════════════
# These are not exhaustive — RQ1.3 will add more. Each is a one-liner that
# produces a TargetConfig. The benefit over a string-based factory is that
# typos are caught at import time rather than at run time.

def plain(provider: str, model: str, label: str, **kw) -> TargetConfig:
    """No system prompt, no thinking, no tools — the default."""
    return TargetConfig(provider=provider, model=model, label=label, **kw)


def with_system_prompt(
    provider: str, model: str, label: str, system_prompt: str, **kw,
) -> TargetConfig:
    """Same model, with a system prompt prepended.

    Use case (RQ1.3): compare a model with and without its provider's
    default system prompt to test whether the system prompt itself drives
    robustness.
    """
    return TargetConfig(
        provider=provider, model=model, label=label,
        system_prompt=system_prompt, **kw,
    )


def anthropic_thinking(
    model: str, label: str, *, budget_tokens: int = 8000, **kw,
) -> TargetConfig:
    """Anthropic model with extended-thinking enabled.

    Note the parameter shape may evolve — check LiteLLM's current
    Anthropic docs at https://docs.litellm.ai/docs/providers/anthropic.
    """
    return TargetConfig(
        provider="anthropic",
        model=model,
        label=label,
        extras={
            "thinking": {"type": "enabled", "budget_tokens": budget_tokens},
        },
        **kw,
    )


def openai_reasoning(
    model: str, label: str, *, reasoning_effort: str = "medium", **kw,
) -> TargetConfig:
    """OpenAI reasoning model (o-series) with reasoning_effort control."""
    return TargetConfig(
        provider="openai",
        model=model,
        label=label,
        extras={"reasoning_effort": reasoning_effort},
        **kw,
    )