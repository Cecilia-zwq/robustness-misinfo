"""
main_user_IVs/config.py
=======================
RQ1 configuration: which conditions, which beliefs, which models,
which paths. Keep all the dials here so the orchestrator scripts
(run_conversations.py, run_scoring.py) stay short.

Values follow the spec in description.md.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "main_user_IVs"

REPO_ROOT = Path("/home/wzhan969/robustness-misinfo")
BELIEFS_PATH = REPO_ROOT / "data" / "dataset" / "sampled_beliefs.json"
RESULTS_DIR = REPO_ROOT / "results"

# ── Generation config ────────────────────────────────────────────────────────
N_TURNS = 8
TEMPERATURE_USER = 0.7
TEMPERATURE_TARGET = 0.7
MAX_TOKENS_TARGET = 600
MAX_REFLECT_RETRIES = 3

# ── Models ───────────────────────────────────────────────────────────────────
USER_AGENT_PROVIDER = "openrouter"
USER_AGENT_MODEL = "openai/gpt-4.1-mini"

# Each entry: (provider, model, short_slug_for_session_id)
TARGET_LLMS: list[tuple[str, str, str]] = [
    ("anthropic", "claude-sonnet-4.6", "claude-sonnet-4.6"),
    ("openai", "gpt-5.3-chat", "gpt-5.3-chat"),
    ("google", "gemini-3-flash-preview", "gemini-3-flash"),
    ("deepseek", "deepseek-v3.2", "deepseek-v3.2"),
]

# ── Scoring config ───────────────────────────────────────────────────────────
# Two-evaluator agreement is the default protocol from description.md.
# If they disagree, the orchestrator can escalate (handled in run_scoring.py
# or a follow-up script — out of scope for the conversation phase).

EVALUATOR_PRIMARY = ("openai", "gpt-5-mini")
EVALUATOR_SECONDARY = ("google", "gemini-3-flash-preview")
EVALUATOR_ESCALATION_NONTHINKING = ("anthropic", "claude-sonnet-4.6")
# The thinking variant is selected via API parameter, not model string —
# this constant is the slug recorded in the score artifact.
EVALUATOR_ESCALATION_THINKING = ("anthropic", "claude-sonnet-4.6-thinking")

EVALUATOR_TEMPERATURE = 0.0
DEFAULT_RUBRIC_NAME = "misinfo_v1"