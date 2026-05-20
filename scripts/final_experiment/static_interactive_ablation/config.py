"""
static_interactive_ablation/config.py
======================================
Configuration for the static (opening-only) interactivity ablation.

Goal: re-run the same sampled sessions as the reflection ablation, but
replace the adaptive user simulation with a static script of 8 messages
pre-generated using generate_opening only. The target LLM, belief,
character prompt, and all other variables are held constant; only the
user's ability to adapt to the target's replies is removed.

Outputs live in a sibling directory of the source run::

    results/final_experiment/main_user_IVs/20260427_165233/
      conversations/            (original interactive source)
      conversations_static/     (this experiment)
      scores_static/            (scored separately with run_scoring.py)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from main_user_IVs import config as main_cfg

# ── Source run selection ─────────────────────────────────────────────────────
_DEFAULT_SOURCE_RUN = main_cfg.RESULTS_DIR / "main_user_IVs" / "20260427_165233"
_SOURCE_RUN_ENV = os.environ.get("STATIC_ABLATION_SOURCE_RUN")
SOURCE_RUN_DIR = (
    Path(_SOURCE_RUN_ENV).expanduser().resolve()
    if _SOURCE_RUN_ENV
    else _DEFAULT_SOURCE_RUN
)

# ── Paths ────────────────────────────────────────────────────────────────────
SOURCE_CONV_DIR      = SOURCE_RUN_DIR / "conversations"
OUTPUT_CONV_DIR      = SOURCE_RUN_DIR / "conversations_static"
SAMPLE_INDEX_PATH    = SOURCE_RUN_DIR / "ablation_sample_index.json"
CHECKPOINT_PATH      = SOURCE_RUN_DIR / "checkpoint_static_ablation.json"
MANIFEST_PATH        = SOURCE_RUN_DIR / "manifest_static_ablation.json"

# ── Experiment identity ──────────────────────────────────────────────────────
EXPERIMENT_NAME    = "main_user_IVs__static_ablation"
SESSION_ID_SUFFIX  = "__static"

# ── Generation config (mirror source run) ────────────────────────────────────
N_TURNS            = main_cfg.N_TURNS
TEMPERATURE_USER   = main_cfg.TEMPERATURE_USER
TEMPERATURE_TARGET = main_cfg.TEMPERATURE_TARGET
SYS_TARGET         = main_cfg.SYS_TARGET
MAX_REFLECT_RETRIES = main_cfg.MAX_REFLECT_RETRIES


def _resolve_user_agent() -> tuple[str, str]:
    """Read user agent from source run manifest; fall back to main config."""
    manifest = SOURCE_RUN_DIR / "manifest.json"
    try:
        with manifest.open("r", encoding="utf-8") as fh:
            user_agent = str(json.load(fh)["user_agent"])
        provider, model = user_agent.split("/", 1)
        return provider, model
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return main_cfg.USER_AGENT_PROVIDER, main_cfg.USER_AGENT_MODEL


USER_AGENT_PROVIDER, USER_AGENT_MODEL = _resolve_user_agent()
