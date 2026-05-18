"""
reflection_ablation/config.py
=============================
Configuration for the reflection-module ablation experiment.

Goal: take a stratified ~12% subsample of the conversations that had at
least one reflection break, replay them up to the first break turn,
inject the rejected draft (the "broken" message) as the user message,
and then continue the conversation with the reflection module disabled
on the user agent.

Output mirrors the original conversation schema and lives in a sibling
folder of the source run::

    results/final_experiment/main_user_IVs/20260427_165233/
      conversations/                  (source)
      conversations_none_reflection/  (this experiment)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Reuse main study config for shared knobs.
from main_user_IVs import config as main_cfg

# ── Source run selection ─────────────────────────────────────────────────────
# By default the ablation targets the original gpt-4.1-mini study run.
# Set the REFLECTION_ABLATION_SOURCE_RUN environment variable to point the
# whole pipeline (run_ablation, embed_trajectory_similarities,
# embed_reflection_pairs) at a different source run — e.g. a corpus
# regenerated with a local llama2-uncensored user simulator::
#
#     REFLECTION_ABLATION_SOURCE_RUN=results/final_experiment/main_user_IVs/<ts> \
#         python -m reflection_ablation.run_ablation ...
#
# All ablation artifacts (sample index, no-reflection conversations,
# checkpoint, manifest, trajectory CSV) then live inside that run dir.
_DEFAULT_SOURCE_RUN = main_cfg.RESULTS_DIR / "main_user_IVs" / "20260427_165233"
_SOURCE_RUN_ENV = os.environ.get("REFLECTION_ABLATION_SOURCE_RUN")
SOURCE_RUN_DIR = (
    Path(_SOURCE_RUN_ENV).expanduser().resolve()
    if _SOURCE_RUN_ENV
    else _DEFAULT_SOURCE_RUN
)

# ── Paths ────────────────────────────────────────────────────────────────────
SOURCE_CONV_DIR = SOURCE_RUN_DIR / "conversations"
OUTPUT_DIR_NAME = "conversations_none_reflection"
OUTPUT_CONV_DIR = SOURCE_RUN_DIR / OUTPUT_DIR_NAME
CHECKPOINT_PATH = SOURCE_RUN_DIR / "checkpoint_ablation.json"
MANIFEST_PATH = SOURCE_RUN_DIR / "manifest_ablation.json"
SAMPLE_INDEX_PATH = SOURCE_RUN_DIR / "ablation_sample_index.json"
REFLECTION_PAIR_DISTANCES_PATH = SOURCE_RUN_DIR / "reflection_pair_distances.csv"

# ── Experiment identity ──────────────────────────────────────────────────────
EXPERIMENT_NAME = "main_user_IVs__reflection_ablation"
SESSION_ID_SUFFIX = "__noref"

# ── Sampling parameters ──────────────────────────────────────────────────────
SAMPLE_FRACTION = 0.12           # flat 12% of each cell; cells yielding 0 are skipped
TURN1_RATIO = 0.7                # 70% of cell-quota from turn-1 breaks
SAMPLING_SEED = 42

# ── Generation config (mirror the source run) ────────────────────────────────
# Reflection is OFF for the user agent: max_reflect_retries=0 forces the
# NoReflectionSimulation path. Target LLM, temperatures and n_turns
# mirror the source so the only changed variable is the user-agent
# reflection module.
N_TURNS = main_cfg.N_TURNS
TEMPERATURE_USER = main_cfg.TEMPERATURE_USER
TEMPERATURE_TARGET = main_cfg.TEMPERATURE_TARGET
SYS_TARGET = main_cfg.SYS_TARGET

# The no-reflection conversations must use the SAME user simulator model
# as the source corpus — otherwise the reflection ON/OFF comparison is
# confounded by a model change. We therefore read the user agent from the
# source run's manifest.json (written by run_conversations.py as
# "<provider>/<model>"); if that is unavailable we fall back to the main
# study default (openrouter/openai/gpt-4.1-mini).
def _resolve_user_agent() -> tuple[str, str]:
    manifest = SOURCE_RUN_DIR / "manifest.json"
    try:
        with manifest.open("r", encoding="utf-8") as fh:
            user_agent = str(json.load(fh)["user_agent"])
        provider, model = user_agent.split("/", 1)
        return provider, model
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return main_cfg.USER_AGENT_PROVIDER, main_cfg.USER_AGENT_MODEL


USER_AGENT_PROVIDER, USER_AGENT_MODEL = _resolve_user_agent()
