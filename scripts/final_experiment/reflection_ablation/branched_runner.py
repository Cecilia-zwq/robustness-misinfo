"""
reflection_ablation/branched_runner.py
======================================
Run a single ablation session: replay the pre-break prefix verbatim from
the source artifact, inject the rejected draft as the user message at
the first break turn, get a fresh target response to it, then continue
the conversation with the reflection module disabled on the user agent.

The result is a ConversationArtifact mirroring the source schema, so
downstream analysis and scoring work unchanged.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime

from core.conversation import TargetLike, UserSimulation, format_belief_for_agent
from core.storage import (
    SCHEMA_VERSION,
    ConversationArtifact,
    ReflectionAttempt,
    TurnArtifact,
)

logger = logging.getLogger(__name__)


def _dedupe_attempt_drafts(turn: TurnArtifact) -> None:
    """Blank out attempt drafts that duplicate ``turn.user_message``.

    The accepted attempt's draft is by definition equal to user_message,
    so storing both wastes space (turn-1 messages can be 500–1500
    chars). We blank the redundant draft to an empty string and leave
    the verdict/quote/fix fields intact for audit. Downstream code that
    needs the accepted text reads ``turn.user_message``.
    """
    for att in turn.reflection_attempts:
        if att.draft == turn.user_message and att.accepted:
            att.draft = ""


def run_branched_conversation(
    *,
    session_id: str,
    experiment: str,
    source: ConversationArtifact,
    break_turn: int,
    branching_message: str,
    user_simulation: UserSimulation,
    target: TargetLike,
    n_turns: int,
    config_metadata: dict | None = None,
) -> ConversationArtifact:
    """Branch off `source` at `break_turn` with `branching_message` injected.

    Pre-break turns (1..break_turn-1) are copied verbatim from the
    source artifact — same user message, same target response — so the
    shared prefix is bit-identical to the original conversation.

    At ``break_turn``, the broken draft is sent to the target and a
    fresh response is generated. From ``break_turn+1`` onward the
    user_simulation (typically NoReflectionSimulation) drives the
    conversation against the same target. The target is called fresh
    every post-branch turn, so the conversation can diverge arbitrarily.

    Reflection metadata for the branch turn is recorded so the artifact
    still carries enough provenance to be auditable:

      * ``reflection_attempts`` contains a single entry (the injected
        broken draft), with both verdicts set to ``"INJECTED"``.
      * ``is_fallback`` is False (the message was deliberately chosen,
        not produced by the reflection-exhaustion fallback path).
      * ``n_character_breaks`` and ``n_belief_breaks`` are 0 (the
        reflection module did not evaluate this draft in this run).
    """
    if break_turn < 1:
        raise ValueError(f"break_turn must be >= 1, got {break_turn}")
    if break_turn > n_turns:
        raise ValueError(f"break_turn {break_turn} exceeds n_turns {n_turns}")
    if len(source.turns) < break_turn:
        raise ValueError(
            f"Source artifact has {len(source.turns)} turns but break_turn "
            f"is {break_turn}; cannot extract a pre-break prefix."
        )

    misinformation_belief = format_belief_for_agent(source.belief)

    # ── metadata for the produced artifact ───────────────────────────────
    user_meta: dict = {}
    if hasattr(user_simulation, "model_id"):
        user_meta["user_agent"] = user_simulation.model_id
    if hasattr(user_simulation, "temperature"):
        user_meta["temperature_user"] = user_simulation.temperature
    if hasattr(user_simulation, "max_reflect_retries"):
        user_meta["max_reflect_retries"] = user_simulation.max_reflect_retries

    artifact = ConversationArtifact(
        schema_version=SCHEMA_VERSION,
        session_id=session_id,
        experiment=experiment,
        cell=deepcopy(source.cell),
        belief=deepcopy(source.belief),
        models={
            "user_agent": user_meta.get("user_agent", "unknown"),
            "target_llm": f"{target.provider}/{target.model}",
        },
        config={
            "n_turns": n_turns,
            "temperature_target": getattr(target, "temperature", None),
            "max_tokens_target": getattr(target, "max_tokens", None),
            **{k: v for k, v in user_meta.items() if k != "user_agent"},
            "source_session_id": source.session_id,
            "branch_turn": break_turn,
            "branching_message_origin": "source.reflection_attempts[0].draft",
            **(config_metadata or {}),
        },
        turns=[],
    )

    conversation_history: list[dict] = []

    # ── Step 1: copy pre-break prefix verbatim ───────────────────────────
    for src_turn in source.turns[: break_turn - 1]:
        # rebuild history from source content
        conversation_history.append(
            {"role": "user", "content": src_turn.user_message}
        )
        conversation_history.append(
            {"role": "assistant", "content": src_turn.target_response}
        )
        # carry the source turn forward (deep-copied attempts to avoid
        # cross-artifact mutation)
        new_turn = TurnArtifact(
            turn=src_turn.turn,
            user_message=src_turn.user_message,
            target_response=src_turn.target_response,
            reflection_attempts=[
                ReflectionAttempt(**vars(a)) for a in src_turn.reflection_attempts
            ],
            is_fallback=src_turn.is_fallback,
            target_empty=src_turn.target_empty,
            n_character_breaks=src_turn.n_character_breaks,
            n_belief_breaks=src_turn.n_belief_breaks,
        )
        _dedupe_attempt_drafts(new_turn)
        artifact.turns.append(new_turn)

    # ── Step 2: branch turn — inject broken draft, fresh target response ─
    conversation_history.append({"role": "user", "content": branching_message})
    branch_target_resp, branch_target_empty = target.respond(conversation_history)
    conversation_history.append(
        {"role": "assistant", "content": branch_target_resp}
    )
    branch_turn_obj = TurnArtifact(
        turn=break_turn,
        user_message=branching_message,
        target_response=branch_target_resp,
        reflection_attempts=[
            ReflectionAttempt(
                attempt=1,
                draft=branching_message,
                character_verdict="INJECTED",
                character_quote="N/A",
                character_fix="N/A",
                belief_verdict="INJECTED",
                belief_quote="N/A",
                belief_fix="N/A",
                accepted=True,
            )
        ],
        is_fallback=False,
        target_empty=branch_target_empty,
        n_character_breaks=0,
        n_belief_breaks=0,
    )
    _dedupe_attempt_drafts(branch_turn_obj)
    artifact.turns.append(branch_turn_obj)

    # ── Step 3: continue post-branch turns with no-reflection simulation ─
    for turn_idx in range(break_turn + 1, n_turns + 1):
        gen = user_simulation.generate(
            turn_idx=turn_idx,
            conversation_history=conversation_history,
            misinformation_belief=misinformation_belief,
        )
        conversation_history.append({"role": "user", "content": gen.user_message})
        target_resp, target_empty = target.respond(conversation_history)
        conversation_history.append(
            {"role": "assistant", "content": target_resp}
        )

        post_turn = TurnArtifact(
            turn=turn_idx,
            user_message=gen.user_message,
            target_response=target_resp,
            reflection_attempts=gen.reflection_attempts,
            is_fallback=gen.is_fallback,
            target_empty=target_empty,
            n_character_breaks=gen.n_character_breaks,
            n_belief_breaks=gen.n_belief_breaks,
        )
        _dedupe_attempt_drafts(post_turn)
        artifact.turns.append(post_turn)

        logger.info(
            "[ablation] turn %d/%d  char=%d  belief=%d  fb=%s  empty=%s  | %s",
            turn_idx, n_turns,
            gen.n_character_breaks, gen.n_belief_breaks,
            gen.is_fallback, target_empty,
            session_id,
        )

    artifact.completed_at = datetime.now().isoformat()
    return artifact
