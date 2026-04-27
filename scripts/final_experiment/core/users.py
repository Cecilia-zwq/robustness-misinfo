"""
core/users.py
========================
UserSimulation implementations.

A UserSimulation is one strategy for producing the next user message in
a conversation. Each strategy answers a different research question:

  - AgentSimulation         — the standard reflection-driven simulated
                              user (RQ1 main effects, RQ1.3 target
                              configurations).
  - StaticReplaySimulation  — replays user messages from an existing
                              artifact against a fresh target. Isolates
                              the contribution of *interactivity* (RQ2).
  - NoReflectionSimulation  — runs the standard agent but disables the
                              reflection retry loop, so the first draft
                              is always accepted. Live counterfactual
                              for RQ3.
  - CounterfactualReplaySimulation — replays the *first reflection
                              draft* of each turn from an existing
                              artifact, against a fresh target. Cheap
                              offline counterfactual for RQ3.

Adding a new simulation is one new class implementing the
UserSimulation protocol (one method: `generate`). No changes to
run_conversation or any orchestrator beyond instantiating the new class.

Framework dependency
--------------------
This is the only file in `core/` that imports from
misinfo_eval_framework. AgentSimulation and NoReflectionSimulation wrap
SimulatedUserAgent. The other two simulations only need stored data
and could in principle move to a framework-free module — they live
here for cohesion (one place for "how the user side is simulated").
"""

from __future__ import annotations

from misinfo_eval_framework import SimulatedUserAgent

from .conversation import TurnGenerationResult
from .storage import ConversationArtifact, ReflectionAttempt


# ════════════════════════════════════════════════════════════════════════════
# Default — RQ1, RQ1.3
# ════════════════════════════════════════════════════════════════════════════

class AgentSimulation:
    """UserSimulation that delegates to a SimulatedUserAgent.

    The default behaviour: an LLM-driven user with a character prompt,
    advocating for a misinformation belief, with a reflection retry loop
    that audits each draft for character/belief breaks and regenerates
    if either dimension fails. This is what RQ1's main IV-structure
    study uses, and what RQ1.3's target-configuration variants compose
    against.

    Owns the agent and tracks the reflection-log slice per turn so
    every draft is correctly attributed to the turn it was generated
    for. The agent's reflection log is appended across turns; the slice
    `agent.reflection_log[log_len_before:]` picks out only the entries
    produced for the current turn.
    """

    def __init__(self, agent: SimulatedUserAgent) -> None:
        self._agent = agent

    @property
    def model_id(self) -> str:
        return f"{self._agent.provider}/{self._agent.model}"

    @property
    def temperature(self) -> float:
        return self._agent.temperature

    @property
    def max_reflect_retries(self) -> int:
        return self._agent.max_reflect_retries

    def generate(
        self,
        *,
        turn_idx: int,
        conversation_history: list[dict],
        misinformation_belief: str,
    ) -> TurnGenerationResult:
        # Snapshot the agent's reflection log length BEFORE this turn,
        # so we can slice out only the attempts produced for this turn
        # after the call returns.
        log_len_before = len(self._agent.reflection_log)

        if turn_idx == 1:
            user_msg = self._agent.generate_opening(misinformation_belief)
        else:
            user_msg = self._agent.generate_reply(
                conversation_history, misinformation_belief
            )

        n_char = self._agent._last_character_break_count
        n_belief = self._agent._last_belief_break_count
        is_fb = self._agent._last_fallback

        new_log = self._agent.reflection_log[log_len_before:]
        attempts = _build_reflection_attempts(new_log, accepted_draft=user_msg)

        return TurnGenerationResult(
            user_message=user_msg,
            reflection_attempts=attempts,
            is_fallback=is_fb,
            n_character_breaks=n_char,
            n_belief_breaks=n_belief,
        )


# ════════════════════════════════════════════════════════════════════════════
# RQ2 — Static replay
# ════════════════════════════════════════════════════════════════════════════

class StaticReplaySimulation:
    """Replays user messages from an existing ConversationArtifact.

    The replay is stateless w.r.t. the target's responses: turn 1 emits
    the same first message regardless of what the target said in any
    prior session. This isolates the contribution of *interactivity* —
    the comparison condition (interactive) is just AgentSimulation on
    the same belief.

    Usage::

        source_artifact = read_conversation(paths_rq1, sid)
        sim = StaticReplaySimulation(source_artifact)
        # Pair with a fresh target (possibly a different model than the
        # original) and call run_conversation.
    """

    def __init__(self, source_artifact: ConversationArtifact) -> None:
        self._messages = [t.user_message for t in source_artifact.turns]
        # Carry forward the original reflection trail so the replay
        # artifact preserves provenance — useful when auditing later.
        self._original_attempts = [
            t.reflection_attempts for t in source_artifact.turns
        ]
        self._source_session_id = source_artifact.session_id

    @property
    def model_id(self) -> str:
        return f"static-replay({self._source_session_id})"

    def generate(
        self,
        *,
        turn_idx: int,
        conversation_history: list[dict],
        misinformation_belief: str,
    ) -> TurnGenerationResult:
        if turn_idx > len(self._messages):
            raise ValueError(
                f"StaticReplaySimulation has only {len(self._messages)} "
                f"messages but turn {turn_idx} was requested. The replay "
                f"experiment must use n_turns <= source artifact's n_turns."
            )
        msg = self._messages[turn_idx - 1]
        return TurnGenerationResult(
            user_message=msg,
            # Replays don't run reflection in this session — the recorded
            # attempts are from the source artifact, copied for provenance.
            reflection_attempts=self._original_attempts[turn_idx - 1],
            is_fallback=False,
            n_character_breaks=0,
            n_belief_breaks=0,
        )


# ════════════════════════════════════════════════════════════════════════════
# RQ3 — No reflection (live)
# ════════════════════════════════════════════════════════════════════════════

class NoReflectionSimulation:
    """Like AgentSimulation but bypasses the reflection retry loop.

    Implementation: wraps a SimulatedUserAgent that was constructed with
    `max_reflect_retries=0`. The agent's first draft is always accepted
    immediately because the reflect-and-retry loop runs zero times.

    This lets RQ3 measure the causal contribution of reflection to the
    user-simulation pipeline against a true counterfactual, with the
    target LLM responding to non-reflected drafts in real time (so the
    conversation history diverges from the with-reflection condition).

    Construction note: the caller passes a SimulatedUserAgent with
    `max_reflect_retries=0`. We could enforce that here, but doing it
    at the call site keeps construction explicit and visible in
    orchestrator code.
    """

    def __init__(self, agent: SimulatedUserAgent) -> None:
        if agent.max_reflect_retries != 0:
            raise ValueError(
                "NoReflectionSimulation expects an agent with "
                "max_reflect_retries=0, got "
                f"{agent.max_reflect_retries}. Construct the agent with "
                "max_reflect_retries=0 to bypass the reflection loop."
            )
        self._agent = agent

    @property
    def model_id(self) -> str:
        return f"{self._agent.provider}/{self._agent.model} (no-reflection)"

    @property
    def temperature(self) -> float:
        return self._agent.temperature

    @property
    def max_reflect_retries(self) -> int:
        return 0

    def generate(
        self,
        *,
        turn_idx: int,
        conversation_history: list[dict],
        misinformation_belief: str,
    ) -> TurnGenerationResult:
        # Even with max_reflect_retries=0 the agent still appends one
        # reflection-log entry per turn (the audit of the only draft).
        # We capture it for forensics but the verdict is irrelevant —
        # the draft was accepted by virtue of the retry budget being 0.
        log_len_before = len(self._agent.reflection_log)

        if turn_idx == 1:
            user_msg = self._agent.generate_opening(misinformation_belief)
        else:
            user_msg = self._agent.generate_reply(
                conversation_history, misinformation_belief
            )

        new_log = self._agent.reflection_log[log_len_before:]
        attempts = [
            ReflectionAttempt(
                attempt=e["attempt"],
                draft=e["draft"],
                character_verdict=e["character_verdict"],
                character_quote=e["character_quote"],
                character_fix=e["character_fix"],
                belief_verdict=e["belief_verdict"],
                belief_quote=e["belief_quote"],
                belief_fix=e["belief_fix"],
                accepted=(e["draft"] == user_msg),
            )
            for e in new_log
        ]

        return TurnGenerationResult(
            user_message=user_msg,
            reflection_attempts=attempts,
            # Counters from the agent are still meaningful as audit
            # signal even though they no longer drive retries.
            is_fallback=self._agent._last_fallback,
            n_character_breaks=self._agent._last_character_break_count,
            n_belief_breaks=self._agent._last_belief_break_count,
        )


# ════════════════════════════════════════════════════════════════════════════
# RQ3 (cheaper, offline) — Counterfactual replay of first drafts
# ════════════════════════════════════════════════════════════════════════════

class CounterfactualReplaySimulation:
    """Replays the *first reflection draft* from an existing artifact.

    Use case: existing RQ1 artifacts already contain
    `turns[].reflection_attempts[0].draft` for every turn — the message
    the agent would have sent had reflection not intervened. Pairing
    this with a fresh target gives a no-reflection counterfactual at
    the cost of just the target's responses (no user-side LLM calls).

    Caveat: this is *not* the same as a true NoReflectionSimulation
    run. The original reflection_attempts[0] was generated against the
    original conversation history, which itself was shaped by
    reflection earlier in the session. So the counterfactual is "what
    would the target say if it received the first-draft of each turn,
    taken from the with-reflection trajectory" — informative but not
    the same as "rerun the whole agent with reflection off."

    For a paper, run NoReflectionSimulation as the headline RQ3
    condition and use this as a robustness check or for cheap
    exploratory analysis.
    """

    def __init__(self, source_artifact: ConversationArtifact) -> None:
        self._first_drafts: list[str] = []
        for t in source_artifact.turns:
            if not t.reflection_attempts:
                # Edge case: a fallback turn with no recorded attempts.
                # Use the accepted user_message as fallback.
                self._first_drafts.append(t.user_message)
            else:
                self._first_drafts.append(t.reflection_attempts[0].draft)
        self._source_session_id = source_artifact.session_id

    @property
    def model_id(self) -> str:
        return f"counterfactual-first-draft({self._source_session_id})"

    def generate(
        self,
        *,
        turn_idx: int,
        conversation_history: list[dict],
        misinformation_belief: str,
    ) -> TurnGenerationResult:
        if turn_idx > len(self._first_drafts):
            raise ValueError(
                f"CounterfactualReplaySimulation has only "
                f"{len(self._first_drafts)} drafts but turn {turn_idx} "
                f"was requested."
            )
        return TurnGenerationResult(
            user_message=self._first_drafts[turn_idx - 1],
            reflection_attempts=[],
            is_fallback=False,
            n_character_breaks=0,
            n_belief_breaks=0,
        )


# ════════════════════════════════════════════════════════════════════════════
# Internal helper
# ════════════════════════════════════════════════════════════════════════════

def _build_reflection_attempts(
    log_entries: list[dict],
    *,
    accepted_draft: str,
) -> list[ReflectionAttempt]:
    """Convert raw agent reflection-log entries into ReflectionAttempt objects.

    The accepted draft is identified by string-matching against the
    most recent entry whose draft equals the message that was actually
    sent. On fallback (max retries exceeded), the agent returns a
    hardcoded fallback string that is *not* in the log — in that case
    no attempt is marked accepted, which is the correct semantics.
    """
    attempts = [
        ReflectionAttempt(
            attempt=entry["attempt"],
            draft=entry["draft"],
            character_verdict=entry["character_verdict"],
            character_quote=entry["character_quote"],
            character_fix=entry["character_fix"],
            belief_verdict=entry["belief_verdict"],
            belief_quote=entry["belief_quote"],
            belief_fix=entry["belief_fix"],
            accepted=False,
        )
        for entry in log_entries
    ]

    # Mark the accepted attempt by matching draft text. Search backwards:
    # the accepted draft is normally the last passing one. The agent
    # has subtle whitespace handling (_guard_empty does .strip()), so
    # we compare with strip-equality as a fallback to exact match.
    for att in reversed(attempts):
        if att.draft == accepted_draft or att.draft.strip() == accepted_draft.strip():
            att.accepted = True
            break

    return attempts