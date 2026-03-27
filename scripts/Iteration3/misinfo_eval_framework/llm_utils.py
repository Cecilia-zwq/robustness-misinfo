"""
Thin wrapper around LiteLLM for consistent API calling across components.
"""

import litellm


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


def call_llm(
    provider: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 1024,
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

    Returns
    -------
    str
        The model's response text.
    """
    model_string = build_model_string(provider, model)
    response = litellm.completion(
        model=model_string,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()