"""
core/conversation.py
====================
Run one multi-turn conversation, capturing every reflection draft along
the way, and return a ConversationArtifact ready to be persisted.

Replaces misinfo_eval_framework.session.run_session. Differences:

  - Scoring is removed. Scoring is a separate phase (core/scoring.py).
  - Reflection drafts are captured. Every rejected attempt is preserved
    alongside the accepted draft.
  - Pluggable user simulation. The default uses SimulatedUserAgent
    (RQ1, RQ1.3) but RQ2 (static replay) and RQ3 (no-reflection runs)
    plug in different simulations without touching this file.
  - Pluggable target. Accepts anything with `respond(history)` — the
    framework's TargetLLM and core.targets.TargetConfig both qualify.

This file deliberately has no `misinfo_eval_framework` import. It works
with any UserSimulation that satisfies the protocol below; the
framework-dependent simulations live in core/users.py.
Parallelism is handled one level up by core/runner.py, so this module
stays simple and testable in isolation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from .storage import (
    SCHEMA_VERSION,
    ConversationArtifact,
    ReflectionAttempt,
    TurnArtifact,
)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Protocols (duck-typed: framework TargetLLM and core.targets.TargetConfig
# both satisfy TargetLike; UserSimulation is implemented in
# core/users.py)
# ════════════════════════════════════════════════════════════════════════════

class TargetLike(Protocol):
    """Anything that can respond to a conversation history.

    Implemented by:
      - misinfo_eval_framework.target_llm.TargetLLM (legacy; still used
        by main_user_IVs/run_conversations.py)
      - core.targets.TargetConfig (RQ1.3 — adds thinking/system-prompt/tools)
    """
    provider: str
    model: str

    def respond(self, conversation_history: list[dict]) -> str: ...


@dataclass
class TurnGenerationResult:
    """What a UserSimulation returns for one turn.

    A static replay simulation would set `reflection_attempts=[]` and
    `is_fallback=False`. The full agent simulation fills all fields. RQ3's
    no-reflection simulation returns the first-draft as `user_message`
    with a single recorded attempt for audit.
    """
    user_message: str
    reflection_attempts: list[ReflectionAttempt]
    is_fallback: bool
    n_character_breaks: int
    n_belief_breaks: int


class UserSimulation(Protocol):
    """Generate the next user message given the conversation so far.

    The runner calls this once per turn. Implementations live in
    core/users.py:
      - AgentSimulation                — RQ1, RQ1.3
      - StaticReplaySimulation         — RQ2
      - NoReflectionSimulation         — RQ3 (live)
      - CounterfactualReplaySimulation — RQ3 (offline / cheap)

    The simulation owns whatever stateful context it needs (an LLM
    agent, a pre-recorded message list, etc.). The runner is stateless
    w.r.t. the simulation between calls — only `conversation_history`
    and `turn_idx` flow in.

    Optional metadata properties (`model_id`, `temperature`,
    `max_reflect_retries`) are read by run_conversation when present
    and embedded in the artifact's `models`/`config` block. They're
    not part of the required protocol — a minimal simulation only needs
    `generate`.
    """

    def generate(
        self,
        *,
        turn_idx: int,
        conversation_history: list[dict],
        misinformation_belief: str,
    ) -> TurnGenerationResult: ...


# ════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════

def run_conversation(
    *,
    session_id: str,
    experiment: str,
    cell: dict,
    belief: dict,
    user_simulation: UserSimulation,
    target: TargetLike,
    n_turns: int,
    config_metadata: dict | None = None,
) -> ConversationArtifact:
    """Run one full session and return its artifact.

    Parameters
    ----------
    session_id : str
        Unique identifier (see storage.build_session_id).
    experiment : str
        Name of the parent experiment, embedded in the artifact.
    cell : dict
        Condition metadata as a JSON-friendly dict. For the IV1 × IV2
        experiments this is `Condition.to_dict()`. For other experiments
        the cell is whatever shape that experiment defines —
        run_conversation does not inspect it.
    belief : dict
        Full belief record. Embedded verbatim in the artifact.
    user_simulation : UserSimulation
        Generates the user message per turn.
    target : TargetLike
        Anything with a `.respond(history)` method (TargetLLM or
        TargetConfig).
    n_turns : int
        Number of conversational turns.
    config_metadata : dict | None
        Extra config to embed (e.g. target's full TargetConfig.to_dict()).

    Returns
    -------
    ConversationArtifact
        Fully populated, with completed_at set on success. No IO is
        performed; the caller persists the result.
    """
    misinformation_belief = _format_belief_for_agent(belief)

    # Best-effort metadata pulls from the simulation. AgentSimulation
    # exposes these properties; static/counterfactual simulations may
    # not. The artifact tolerates missing entries.
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
        cell=cell,
        belief=dict(belief),
        models={
            "user_agent": user_meta.get("user_agent", "unknown"),
            "target_llm": f"{target.provider}/{target.model}",
        },
        config={
            "n_turns": n_turns,
            "temperature_target": getattr(target, "temperature", None),
            "max_tokens_target": getattr(target, "max_tokens", None),
            **{k: v for k, v in user_meta.items() if k != "user_agent"},
            **(config_metadata or {}),
        },
        turns=[],
    )

    conversation_history: list[dict] = []

    for turn_idx in range(1, n_turns + 1):
        # ── Step 1: user message ─────────────────────────────────────────
        gen = user_simulation.generate(
            turn_idx=turn_idx,
            conversation_history=conversation_history,
            misinformation_belief=misinformation_belief,
        )
        conversation_history.append({"role": "user", "content": gen.user_message})

        # ── Step 2: target response ──────────────────────────────────────
        target_response = target.respond(conversation_history)
        conversation_history.append(
            {"role": "assistant", "content": target_response}
        )

        artifact.turns.append(
            TurnArtifact(
                turn=turn_idx,
                user_message=gen.user_message,
                target_response=target_response,
                reflection_attempts=gen.reflection_attempts,
                is_fallback=gen.is_fallback,
                n_character_breaks=gen.n_character_breaks,
                n_belief_breaks=gen.n_belief_breaks,
            )
        )

        logger.info(
            "Turn %d/%d | char_breaks=%d belief_breaks=%d fallback=%s | %s",
            turn_idx,
            n_turns,
            gen.n_character_breaks,
            gen.n_belief_breaks,
            gen.is_fallback,
            session_id,
        )

    artifact.completed_at = datetime.now().isoformat()
    return artifact


# ════════════════════════════════════════════════════════════════════════════
# Belief formatting helper
# ════════════════════════════════════════════════════════════════════════════

def format_belief_for_agent(belief: dict) -> str:
    """Render a belief record as the string the simulation + evaluator expect.

    Mirrors the convention in experiment3.py::build_belief_and_claim_fields:
      - short-claim: just the content
      - long-text:   "[{title}]\\n{long_text}"

    Exposed as a module-level helper because UserSimulation implementations
    may need to format belief strings themselves (e.g. for prompt building).
    """
    if belief.get("is_long_text"):
        title = belief.get("content", "").strip()
        body = belief.get("long_text", "").strip()
        return f"[{title}]\n{body}"
    return belief["content"].strip()


# Internal alias kept for module-private callers; publicly use the un-prefixed name.
_format_belief_for_agent = format_belief_for_agent
