"""
Session Runner
==============
Orchestrates a multi-turn conversation between the SimulatedUserAgent and
the TargetLLM, scoring each turn with the Evaluator.

Public API
----------
    result = run_session(
        user_agent, target_llm, evaluator,
        misinformation_claim, n_turns=8,
    )
    # result is a SessionResult dataclass with full logs and aggregate metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .evaluator import Evaluator
from .target_llm import TargetLLM
from .user_agent import SimulatedUserAgent

logger = logging.getLogger(__name__)


@dataclass
class TurnRecord:
    """Record for a single conversational turn."""

    turn: int
    user_message: str
    target_response: str
    scores: dict


@dataclass
class SessionResult:
    """Complete result of a multi-turn evaluation session."""

    misinformation_claim: str
    n_turns: int
    turns: list[TurnRecord] = field(default_factory=list)
    conversation_history: list[dict] = field(default_factory=list)

    # ── aggregate metrics ────────────────────────────────────────────────────

    @property
    def mean_correction(self) -> float:
        vals = [t.scores.get("correction", 0) for t in self.turns]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def mean_rebuttal(self) -> float:
        vals = [t.scores.get("rebuttal", 0) for t in self.turns]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def score_trajectory(self) -> list[dict]:
        """Per-turn scores (without raw_output) for easy plotting."""
        return [
            {
                "turn": t.turn,
                "correction": t.scores.get("correction"),
                "rebuttal": t.scores.get("rebuttal"),
            }
            for t in self.turns
        ]

    def summary(self) -> dict:
        """Return a compact summary dict for logging / analysis."""
        return {
            "misinformation_claim": self.misinformation_claim,
            "n_turns": self.n_turns,
            "mean_correction": round(self.mean_correction, 3),
            "mean_rebuttal": round(self.mean_rebuttal, 3),
            "score_trajectory": self.score_trajectory,
        }


def run_session(
    user_agent: SimulatedUserAgent,
    target_llm: TargetLLM,
    evaluator: Evaluator,
    misinformation_claim: str,
    n_turns: int = 8,
) -> SessionResult:
    """
    Run a complete multi-turn evaluation session.

    Parameters
    ----------
    user_agent : SimulatedUserAgent
        The adversarial simulated user.
    target_llm : TargetLLM
        The model under evaluation.
    evaluator : Evaluator
        Scores each target-LLM response.
    misinformation_claim : str
        The false claim the simulated user advocates for.
    n_turns : int
        Number of conversational turns (user message → LLM response pairs).

    Returns
    -------
    SessionResult
        Full logs and aggregate metrics for the session.
    """
    result = SessionResult(
        misinformation_claim=misinformation_claim,
        n_turns=n_turns,
    )
    conversation_history: list[dict] = []

    for turn_idx in range(1, n_turns + 1):
        logger.info("──── Turn %d / %d ────", turn_idx, n_turns)

        # 1. Simulated user generates a message
        if turn_idx == 1:
            user_msg = user_agent.generate_opening(misinformation_claim)
        else:
            user_msg = user_agent.generate_reply(
                conversation_history, misinformation_claim
            )

        conversation_history.append({"role": "user", "content": user_msg})
        logger.info("USER: %s", user_msg[:120])

        # 2. Target LLM responds
        target_response = target_llm.respond(conversation_history)
        conversation_history.append({"role": "assistant", "content": target_response})
        logger.info("TARGET: %s", target_response[:120])

        # 3. Evaluator scores the response
        scores = evaluator.evaluate(
            user_message=user_msg,
            response=target_response,
            misinformation_claim=misinformation_claim,
        )
        logger.info(
            "SCORES — correction: %.0f, rebuttal: %.0f",
            scores.get("correction", -1),
            scores.get("rebuttal", -1),
        )

        result.turns.append(
            TurnRecord(
                turn=turn_idx,
                user_message=user_msg,
                target_response=target_response,
                scores=scores,
            )
        )

    result.conversation_history = conversation_history
    return result