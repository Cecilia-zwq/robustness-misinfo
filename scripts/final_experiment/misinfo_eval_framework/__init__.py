"""
misinfo_eval_framework
======================
Modular framework for evaluating LLM robustness to misinformation
in multi-turn conversations.

Components
----------
SimulatedUserAgent  — adversarial user agent (actor + reflection)
TargetLLM           — the model under evaluation
Evaluator           — scores target-LLM responses

Quick start
-----------
    from misinfo_eval_framework import (
        SimulatedUserAgent, TargetLLM, Evaluator,
    )

    agent = SimulatedUserAgent(
        provider="openai",
        model="gpt-4.1-mini-2025-04-14",
        character_prompt="You are a confident person who believes ...",
    )
    target = TargetLLM(provider="openai", model="gpt-4.1-mini-2025-04-14")
    evaluator = Evaluator(provider="openai", model="gpt-4.1-mini-2025-04-14")
"""

from .evaluator import Evaluator
# from .session import SessionResult, TurnRecord, run_session
from .target_llm import TargetLLM
from .user_agent import SimulatedUserAgent

__all__ = [
    "SimulatedUserAgent",
    "TargetLLM",
    "Evaluator",
    # "run_session",
    # "SessionResult",
    # "TurnRecord",
]