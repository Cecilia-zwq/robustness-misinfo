"""
llm_dialogue.py — Two LLMs talking to each other.

Each LLM maintains its own perspective on the shared chat history:
  - Its own past messages appear as "assistant"
  - The other LLM's messages appear as "user"

Usage example (matches the demo setup):
  python llm_dialogue.py \
    --provider-a ollama --model-a llama2-uncensored \
    --provider-b openai --model-b gpt-4.1-mini-2025-04-14 \
    --first-speaker a \
    --first-message "Hello! I want to debate whether pineapple belongs on pizza. I say it absolutely does. Do you agree?" \
    --system-a "You are a passionate advocate. Argue your position strongly and ask follow-up questions." \
    --system-b "You are a thoughtful critic. Challenge ideas and provide counter-arguments." \
    --temperature-a 1.0
    --temperature-b None
    --max-turns 5
"""

import argparse
import sys
import textwrap
from typing import Literal
from openai import OpenAI
import ollama

# ── providers ────────────────────────────────────────────────────────────────

def call_openai(model: str, messages: list, system_prompt, temperature: float = None) -> str:
    client = OpenAI()
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    kwargs = {"model": model, "messages": full_messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


def call_ollama(model: str, messages: list, system_prompt, temperature: float = None) -> str:
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    kwargs = {"model": model, "messages": full_messages}
    if temperature is not None:
        kwargs["options"] = {"temperature": temperature}
    response = ollama.chat(**kwargs)
    return response.message.content.strip()


PROVIDERS = {
    "openai": call_openai,
    "ollama": call_ollama,
}

# ── history helpers ───────────────────────────────────────────────────────────

def build_messages_for(llm_id: str, shared_history: list) -> list:
    """
    Translate the shared history into a role-correct message list for one LLM.

    The convention from each LLM's point of view:
      - Its own past utterances  → role "assistant"
      - The other LLM's messages → role "user"

    Both LLMs therefore read the same sequence of content but with
    'user' and 'assistant' roles that correctly reflect who said what
    from their individual perspective.
    """
    return [
        {
            "role": "assistant" if entry["speaker"] == llm_id else "user",
            "content": entry["content"],
        }
        for entry in shared_history
    ]


# ── pretty printing ───────────────────────────────────────────────────────────

COLORS = {
    "a":     "\033[94m",   # blue
    "b":     "\033[92m",   # green
    "reset": "\033[0m",
    "dim":   "\033[2m",
    "bold":  "\033[1m",
}

def print_turn(turn: int, speaker: str, label: str, content: str) -> None:
    color = COLORS["a"] if speaker == "a" else COLORS["b"]
    header = f"{color}{COLORS['bold']}[Turn {turn}] {label}{COLORS['reset']}"
    wrapped = textwrap.fill(content, width=90, subsequent_indent="  ")
    print(f"\n{header}")
    print(f"  {wrapped}")
    print(f"{COLORS['dim']}{'─' * 92}{COLORS['reset']}")


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Let two LLMs hold a conversation with each other.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # LLM A
    parser.add_argument("--provider-a", default="ollama",
                        choices=list(PROVIDERS),
                        help="Provider for LLM A  [ollama | openai]  (default: ollama)")
    parser.add_argument("--model-a", default="llama2-uncensored",
                        help="Model name/tag for LLM A (default: llama2-uncensored)")
    parser.add_argument("--temperature-a", type=float, default=None,
                        help="Temperature for LLM A (default: None, use provider's default)")
    parser.add_argument("--system-a", default=None,
                        help="System prompt for LLM A")
    parser.add_argument("--label-a", default="Simulated user",
                        help='Display label for LLM A (default: "Simulated user")')

    # LLM B
    parser.add_argument("--provider-b", default="openai",
                        choices=list(PROVIDERS),
                        help="Provider for LLM B  [ollama | openai]  (default: openai)")
    parser.add_argument("--model-b", default="gpt-4.1-mini-2025-04-14",
                        help="Model name for LLM B (default: gpt-4.1-mini-2025-04-14)")
    parser.add_argument("--temperature-b", type=float, default=None,
                        help="Temperature for LLM B (default: None, use provider's default)")
    parser.add_argument("--system-b", default=None,
                        help="System prompt for LLM B")
    parser.add_argument("--label-b", default="Evaluated LLM",
                        help='Display label for LLM B (default: "Evaluated LLM")')

    # Conversation settings
    parser.add_argument("--first-speaker", default="a", choices=["a", "b"],
                        help="Which LLM delivers the opening message (default: a)")
    parser.add_argument("--first-message", required=True,
                        help="Predefined opening message injected into the conversation")
    parser.add_argument("--max-turns", type=int, default=6,
                        help="Maximum total turns including the opening message (default: 6)")

    return parser.parse_args()


# ── main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    llm_config = {
        "a": {
            "provider": args.provider_a,
            "model":    args.model_a,
            "temperature": args.temperature_a,
            "system":   args.system_a,
            "label":    f"{args.label_a}  [{args.provider_a.upper()} / {args.model_a}]",
        },
        "b": {
            "provider": args.provider_b,
            "model":    args.model_b,
            "temperature": args.temperature_b,
            "system":   args.system_b,
            "label":    f"{args.label_b}  [{args.provider_b.upper()} / {args.model_b}]",
        },
    }

    # Shared history: ordered list of {"speaker": "a"|"b", "content": str}
    # Both LLMs read from this same list, but build_messages_for() flips
    # the roles so each LLM sees itself as "assistant" and the other as "user".
    shared_history: list = []

    first  = args.first_speaker           # "a" or "b"
    second = "b" if first == "a" else "a"

    print(f"\n{COLORS['bold']}{'═' * 92}")
    print("  LLM DIALOGUE")
    print(f"  {llm_config['a']['label']}")
    print(f"       ⟷")
    print(f"  {llm_config['b']['label']}")
    print(f"  First speaker : {llm_config[first]['label']}")
    print(f"  Max turns     : {args.max_turns}")
    print(f"{'═' * 92}{COLORS['reset']}")

    # ── Turn 1: inject the predefined opening message (no API call needed)
    shared_history.append({"speaker": first, "content": args.first_message})
    print_turn(1, first, llm_config[first]["label"], args.first_message)

    turn = 2
    current_speaker = second  # the other LLM responds next

    while turn <= args.max_turns:
        # Get llm_config from current speaker
        cfg    = llm_config[current_speaker]
        # Get the cosreponding LLM calling function (call_ollama|call_openai)
        call_fn = PROVIDERS[cfg["provider"]]

        # Build a role-correct view of history for the current speaker
        messages = build_messages_for(current_speaker, shared_history)

        print(f"\n{COLORS['dim']}  [{cfg['label']} is thinking…]{COLORS['reset']}",
              end="", flush=True)

        try:
            reply = call_fn(cfg["model"], messages, cfg["system"], cfg["temperature"])
        except Exception as exc:
            print(f"\n\033[91m  Error calling {cfg['label']}: {exc}\033[0m")
            sys.exit(1)

        # Append the new reply to the shared history
        shared_history.append({"speaker": current_speaker, "content": reply})
        print_turn(turn, current_speaker, cfg["label"], reply)

        # Alternate the speaker
        current_speaker = "a" if current_speaker == "b" else "b"
        turn += 1

    print(f"\n{COLORS['bold']}{'═' * 92}")
    print(f"  Conversation complete — {len(shared_history)} total messages.")
    print(f"{'═' * 92}{COLORS['reset']}\n")


if __name__ == "__main__":
    main()