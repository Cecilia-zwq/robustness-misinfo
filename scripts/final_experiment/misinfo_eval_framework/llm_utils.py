"""
Thin wrapper around LiteLLM for consistent API calling across components.

Empty-response handling
-----------------------
Some providers (notably ``claude-sonnet-4.6`` via OpenRouter on
pandemic / bio-conspiracy prompts) return a chat completion whose
``message.content`` is None or empty whitespace — typically because the
model issued a refusal as a non-text content block. LiteLLM normalises
``finish_reason`` to OpenAI-style ``"stop"`` for these, so the real
signal lives at ``choice.provider_specific_fields["native_finish_reason"]``
(e.g. Anthropic's ``"refusal"`` vs ``"end_turn"``). This wrapper:

  - Captures both the normalised ``finish_reason`` and the native
    ``stop_reason`` and exposes them on ``EmptyLLMResponse``.
  - Retries once (configurable via ``retry_on_empty``) on empty content,
    since a non-trivial fraction of refusals are stochastic at T>0.
  - Raises ``EmptyLLMResponse`` (a ``ValueError`` subclass) when retries
    are exhausted. Callers that *must* have text (the simulated user
    agent, the rubric evaluator) propagate the error and the runner
    records the session as failed.
  - Exposes ``EMPTY_TARGET_PLACEHOLDER``, a stable sentinel string the
    target wrappers (``TargetLLM.respond`` / ``TargetConfig.respond``)
    substitute in place of the empty content so the conversation loop
    can continue. Each turn carries an explicit ``target_empty`` flag in
    the artifact, so analysis filters on the flag rather than the
    sentinel string.
"""

import logging

import litellm


logger = logging.getLogger(__name__)


# Default number of automatic retries when the provider returns empty
# content. Empirically, ~17% of Claude Sonnet 4.6 refusals on
# pandemic/bio prompts succeed on a single retry at T=0.7; the rest are
# deterministic policy refusals that no amount of retrying will fix.
DEFAULT_RETRY_ON_EMPTY = 1


# ════════════════════════════════════════════════════════════════════════════
# Empty-response signalling
# ════════════════════════════════════════════════════════════════════════════
# Subclassing ValueError keeps backward-compat with any caller that
# catches the broader exception type (none in-tree at the time of
# writing, but cheap insurance).

class EmptyLLMResponse(ValueError):
    """The provider returned a message with no text content.

    Attributes
    ----------
    model : str
        The fully-qualified model string passed to LiteLLM.
    finish_reason : str | None
        LiteLLM's OpenAI-normalised finish_reason. For refusals this is
        usually ``"stop"`` and therefore not very informative on its own.
    native_stop_reason : str | None
        The provider's raw stop_reason when LiteLLM exposes it (e.g.
        Anthropic's ``"refusal"``, ``"end_turn"``, ``"max_tokens"``).
        This is the field analysts actually want.
    attempts : int
        How many requests were made (1 = no retries, 2 = one retry, …).
    """

    def __init__(
        self,
        model: str,
        finish_reason: str | None,
        native_stop_reason: str | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(
            f"Empty content from {model} after {attempts} attempt(s) "
            f"(finish_reason={finish_reason!r}, "
            f"native_stop_reason={native_stop_reason!r})."
        )
        self.model = model
        self.finish_reason = finish_reason
        self.native_stop_reason = native_stop_reason
        self.attempts = attempts


# Stable sentinel string the target wrappers substitute when the
# provider returns empty content. The simulated user agent sees this in
# the conversation history; downstream analysis should rely on the
# per-turn ``target_empty`` boolean (set by run_conversation) rather
# than string-matching against the sentinel.
EMPTY_TARGET_PLACEHOLDER = "[The assistant did not provide a response.]"


def build_model_string(provider: str, model: str) -> str:
    """
    Construct the model identifier that LiteLLM expects.

    - OpenAI models are passed as-is  (e.g. "gpt-4.1-mini-2025-04-14")
    - Other providers are prefixed     (e.g. "anthropic/claude-sonnet-4-20250514")
    """
    provider = provider.strip().lower()
    if provider in ("openai", ""):
        return model
    return f"{provider}/{model}"


def extract_native_stop_reason(choice, response) -> str | None:
    """Pull the provider's raw stop_reason out of a LiteLLM choice.

    LiteLLM stuffs Anthropic's ``stop_reason`` (and analogous fields for
    other providers) into ``provider_specific_fields["native_finish_reason"]``
    on the choice. Older LiteLLM versions occasionally surface it as a
    top-level attribute instead, so we probe a few locations.
    """
    psf = getattr(choice, "provider_specific_fields", None)
    if isinstance(psf, dict):
        v = psf.get("native_finish_reason") or psf.get("stop_reason")
        if v:
            return v
    msg_psf = getattr(getattr(choice, "message", None),
                      "provider_specific_fields", None)
    if isinstance(msg_psf, dict):
        v = msg_psf.get("native_finish_reason") or msg_psf.get("stop_reason")
        if v:
            return v
    return (
        getattr(choice, "stop_reason", None)
        or getattr(response, "stop_reason", None)
    )


def call_llm(
    provider: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int | None = None,
    retry_on_empty: int = DEFAULT_RETRY_ON_EMPTY,
) -> str:
    """
    Send *messages* to the specified model and return the assistant's text.

    Parameters
    ----------
    provider : str
        LiteLLM provider key (e.g. "openai", "anthropic", "gemini").
    model : str
        Model name (e.g. "gpt-4.1-mini-2025-04-14").
    messages : list[dict]
        OpenAI-style message list.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens in the response.
    retry_on_empty : int
        How many additional requests to issue if the first one comes back
        empty. ``0`` disables retrying. Defaults to
        :data:`DEFAULT_RETRY_ON_EMPTY` (currently 1). Each retry uses the
        same temperature; sampling stochasticity is what makes the retry
        meaningful for non-deterministic refusals.

    Returns
    -------
    str
        The model's response text.

    Raises
    ------
    EmptyLLMResponse
        If every attempt (initial + retries) returned a chat completion
        with ``message.content`` equal to ``None`` or only whitespace.
        The exception carries both the OpenAI-normalised
        ``finish_reason`` and the provider-native ``stop_reason``
        (e.g. Anthropic's ``"refusal"``).
    """
    model_string = build_model_string(provider, model)
    max_attempts = max(1, retry_on_empty + 1)

    last_finish_reason: str | None = None
    last_native_stop: str | None = None

    for attempt in range(1, max_attempts + 1):
        response = litellm.completion(
            model=model_string,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        content = choice.message.content
        last_finish_reason = getattr(choice, "finish_reason", None)
        last_native_stop = extract_native_stop_reason(choice, response)

        if content is not None and content.strip():
            if attempt > 1:
                logger.info(
                    "Recovered empty response from %s on attempt %d/%d "
                    "(prior native_stop_reason=%r).",
                    model_string, attempt, max_attempts, last_native_stop,
                )
            return content.strip()

        if attempt < max_attempts:
            logger.warning(
                "Empty content from %s on attempt %d/%d "
                "(finish_reason=%r, native_stop_reason=%r); retrying.",
                model_string, attempt, max_attempts,
                last_finish_reason, last_native_stop,
            )

    raise EmptyLLMResponse(
        model=model_string,
        finish_reason=last_finish_reason,
        native_stop_reason=last_native_stop,
        attempts=max_attempts,
    )
