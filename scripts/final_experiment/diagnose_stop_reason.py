"""
diagnose_stop_reason.py
=======================
One-off diagnostic: replay just the *first* turn that produced an empty
target response in each failed session and capture the provider's
finish_reason / stop_reason. Does NOT continue the conversation.

For each session artifact under
``results/final_experiment/main_user_IVs/<run>/conversations/`` whose
``turns[*].target_empty == True``:

  1. Reconstruct the message list the target actually saw on the first
     empty turn (system prompt + alternating user/assistant up to and
     including the offending user turn).
  2. Issue a single ``litellm.completion`` with the same model and
     params recorded in the artifact (``temperature_target``,
     ``max_tokens_target``).
  3. Print:
       - finish_reason          (LiteLLM-normalised)
       - native stop_reason     (raw provider value, when exposed)
       - content presence       (None / empty / non-empty preview)
       - any refusal / safety field on the message

Usage
-----
    cd scripts/final_experiment
    python diagnose_stop_reason.py \
        --run-dir ../../results/final_experiment/main_user_IVs/20260427_165233 \
        --filter conspiracy-0014 conspiracy-0033 climate-0030

Add ``--all-empty-turns`` to replay every empty turn in each session
instead of only the first.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import litellm

# Make the package importable so we reuse build_model_string and the
# system prompt the runner used.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from main_user_IVs import config as cfg  # noqa: E402
from misinfo_eval_framework.llm_utils import (  # noqa: E402
    EMPTY_TARGET_PLACEHOLDER,
)


logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("LiteLLM").setLevel(logging.ERROR)


# ════════════════════════════════════════════════════════════════════════════
# Reconstruction
# ════════════════════════════════════════════════════════════════════════════

def reconstruct_messages(
    artifact: dict, up_to_turn: int, system_prompt: str | None
) -> list[dict]:
    """
    Build the exact message list the target saw on ``up_to_turn``.

    The runner's loop appends user_i then calls target.respond, so the
    target on turn N sees: [system?, user_1, assistant_1, ..., user_N].
    For turns where target_empty was True the assistant slot is the
    EMPTY_TARGET_PLACEHOLDER sentinel — we keep it as recorded so the
    replay matches what the model actually saw, character-for-character.
    """
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for t in artifact["turns"]:
        if t["turn"] > up_to_turn:
            break
        messages.append({"role": "user", "content": t["user_message"]})
        if t["turn"] < up_to_turn:
            messages.append(
                {"role": "assistant", "content": t["target_response"]}
            )
    return messages


# ════════════════════════════════════════════════════════════════════════════
# Probe
# ════════════════════════════════════════════════════════════════════════════

def probe_one(messages: list[dict], model_str: str, temperature: float,
              max_tokens: int | None) -> dict[str, Any]:
    """Issue a single call and pull every signal we can out of the response."""
    resp = litellm.completion(
        model=model_str,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = resp.choices[0]
    msg = choice.message

    content = getattr(msg, "content", None)
    finish_reason = getattr(choice, "finish_reason", None)

    # LiteLLM stores provider-native fields here when normalising.
    provider_specific = getattr(choice, "provider_specific_fields", None) \
        or getattr(msg, "provider_specific_fields", None)
    hidden = getattr(resp, "_hidden_params", None)

    # Anthropic via OpenRouter sometimes surfaces stop_reason directly,
    # sometimes inside provider_specific_fields.stop_reason. Probe both.
    native_stop = None
    if isinstance(provider_specific, dict):
        native_stop = provider_specific.get("stop_reason")
    if native_stop is None:
        native_stop = getattr(choice, "stop_reason", None)
    if native_stop is None:
        native_stop = getattr(resp, "stop_reason", None)

    # Refusal field (OpenAI-style structured refusal, also forwarded by
    # some OpenRouter providers).
    refusal = getattr(msg, "refusal", None)

    if content is None:
        content_state = "None"
        preview = ""
    elif not content.strip():
        content_state = "empty/whitespace"
        preview = repr(content)
    else:
        content_state = "non-empty"
        preview = content.strip().replace("\n", " ")[:160]

    return {
        "finish_reason": finish_reason,
        "native_stop_reason": native_stop,
        "refusal": refusal,
        "content_state": content_state,
        "content_preview": preview,
        "provider_specific_fields": provider_specific,
        "hidden_params": (
            {k: v for k, v in (hidden or {}).items()
             if k in ("response_cost", "model_id", "custom_llm_provider")}
        ),
    }


# ════════════════════════════════════════════════════════════════════════════
# Driver
# ════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to results/.../<timestamp> directory with conversations/.",
    )
    parser.add_argument(
        "--filter",
        nargs="*",
        default=None,
        help="Only consider session files whose name contains any of these "
             "substrings (e.g. 'conspiracy-0014 climate-0030').",
    )
    parser.add_argument(
        "--all-empty-turns",
        action="store_true",
        help="Probe every empty turn in each session, not just the first.",
    )
    parser.add_argument(
        "--system-prompt",
        default=cfg.SYS_TARGET,
        help="System prompt to inject; defaults to the one used by the runner.",
    )
    args = parser.parse_args()

    conv_dir = args.run_dir / "conversations"
    if not conv_dir.is_dir():
        print(f"[error] {conv_dir} does not exist", file=sys.stderr)
        return 2

    sessions = sorted(conv_dir.glob("*.json"))
    if args.filter:
        sessions = [
            p for p in sessions if any(f in p.name for f in args.filter)
        ]

    print(f"Scanning {len(sessions)} session file(s) for target_empty turns…")
    print()

    n_probed = 0
    for path in sessions:
        artifact = json.loads(path.read_text())
        empty_turns = [t for t in artifact["turns"] if t.get("target_empty")]
        if not empty_turns:
            continue
        if not args.all_empty_turns:
            empty_turns = empty_turns[:1]

        target_model_str = artifact["models"]["target_llm"]
        temperature = artifact["config"].get("temperature_target", 0.7) or 0.7
        max_tokens = artifact["config"].get("max_tokens_target")

        print("=" * 88)
        print(f"session: {artifact['session_id']}")
        print(f"target : {target_model_str}  (T={temperature}, "
              f"max_tokens={max_tokens})")
        print(f"belief : [{artifact['belief']['category']}/"
              f"{artifact['belief']['subtype']}] "
              f"{artifact['belief']['content'][:120]}")

        for t in empty_turns:
            turn_idx = t["turn"]
            messages = reconstruct_messages(
                artifact, up_to_turn=turn_idx,
                system_prompt=args.system_prompt,
            )
            n_user = sum(1 for m in messages if m["role"] == "user")
            n_asst = sum(1 for m in messages if m["role"] == "assistant")
            print(f"\n  turn {turn_idx}  (replaying {n_user} user / "
                  f"{n_asst} assistant prior msgs; assistant slots may "
                  f"contain the placeholder)")

            try:
                info = probe_one(messages, target_model_str,
                                 temperature, max_tokens)
            except Exception as exc:  # noqa: BLE001
                print(f"    [ERROR] {type(exc).__name__}: {exc}")
                continue

            print(f"    finish_reason     : {info['finish_reason']!r}")
            print(f"    native stop_reason: {info['native_stop_reason']!r}")
            print(f"    refusal           : {info['refusal']!r}")
            print(f"    content           : {info['content_state']}"
                  + (f"  | preview={info['content_preview']!r}"
                     if info['content_preview'] else ""))
            if info["provider_specific_fields"]:
                print(f"    provider_specific : "
                      f"{info['provider_specific_fields']}")
            if info["hidden_params"]:
                print(f"    hidden_params     : {info['hidden_params']}")
            n_probed += 1

    n_sessions_with_empty = sum(
        1 for p in sessions
        if any(t.get("target_empty") for t in json.loads(p.read_text())["turns"])
    )
    print()
    print(f"Done. Probed {n_probed} empty turn(s) across "
          f"{n_sessions_with_empty} session(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
