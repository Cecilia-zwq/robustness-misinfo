"""
core
====
Reusable building blocks for the NeurIPS-extension experiment series.

Layered structure
-----------------
beliefs.py            — load and validate the belief pool
conditions.py         — IV1 × IV2 condition definitions, prompt assembly
storage.py            — disk layout, atomic IO, ConversationArtifact schema
targets.py            — TargetConfig: target-LLM spec with thinking/system/tool support
conversation.py       — generate one conversation, return artifact (pluggable user/target)
users.py   — UserSimulation implementations (Agent, StaticReplay, NoReflection, CounterfactualReplay)
scoring.py            — score one conversation under a Rubric, return ScoreArtifact
runner.py             — parallel pool with resume + Ctrl+C drain

Each layer depends only on layers below it. The only file in `core/`
that imports from misinfo_eval_framework is user_simulations.py — every
other module is framework-independent and works against the protocols
defined in conversation.py.

Orchestrator scripts in sibling folders (main-user-IVs/,
evaluator-validation/, etc.) compose these into specific experiments.
"""

from .beliefs import load_beliefs
from .conditions import (
    IV1_LEVELS,
    IV2_LEVELS,
    Condition,
    stage1_main_effect_conditions,
    stage2_interaction_conditions,
)
from .conversation import (
    TargetLike,
    TurnGenerationResult,
    UserSimulation,
    format_belief_for_agent,
    run_conversation,
)
from .runner import Job, JobResult, run_jobs
from .scoring import (
    MISINFO_V0,
    MISINFO_V1,
    RUBRICS,
    Rubric,
    ScoreArtifact,
    TurnScores,
    score_conversation,
    score_run,
    write_score_artifact,
)
from .storage import (
    SCHEMA_VERSION,
    ConversationArtifact,
    ReflectionAttempt,
    RunPaths,
    TurnArtifact,
    atomic_write_json,
    atomic_write_text,
    build_session_id,
    list_completed_conversation_ids,
    make_run_paths,
    read_conversation,
    read_manifest,
    write_conversation,
    write_manifest,
)
from .targets import (
    TargetConfig,
    anthropic_thinking,
    openai_reasoning,
    plain,
    with_system_prompt,
)
from .users import (
    AgentSimulation,
    CounterfactualReplaySimulation,
    NoReflectionSimulation,
    StaticReplaySimulation,
)

__all__ = [
    # beliefs
    "load_beliefs",
    # conditions
    "Condition",
    "IV1_LEVELS",
    "IV2_LEVELS",
    "stage1_main_effect_conditions",
    "stage2_interaction_conditions",
    # storage
    "SCHEMA_VERSION",
    "ConversationArtifact",
    "TurnArtifact",
    "ReflectionAttempt",
    "RunPaths",
    "make_run_paths",
    "build_session_id",
    "atomic_write_text",
    "atomic_write_json",
    "write_conversation",
    "read_conversation",
    "list_completed_conversation_ids",
    "write_manifest",
    "read_manifest",
    # targets
    "TargetConfig",
    "plain",
    "with_system_prompt",
    "anthropic_thinking",
    "openai_reasoning",
    # conversation (orchestration loop + protocols + helper)
    "run_conversation",
    "UserSimulation",
    "TargetLike",
    "TurnGenerationResult",
    "format_belief_for_agent",
    # user_simulations (concrete UserSimulation implementations)
    "AgentSimulation",
    "StaticReplaySimulation",
    "NoReflectionSimulation",
    "CounterfactualReplaySimulation",
    # scoring
    "Rubric",
    "RUBRICS",
    "MISINFO_V0",
    "MISINFO_V1",
    "ScoreArtifact",
    "TurnScores",
    "score_conversation",
    "write_score_artifact",
    "score_run",
    # runner
    "Job",
    "JobResult",
    "run_jobs",
]