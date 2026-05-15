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

from pathlib import Path

# Reuse main study config for shared knobs.
from main_user_IVs import config as main_cfg

# ── Paths ────────────────────────────────────────────────────────────────────
SOURCE_RUN_DIR = main_cfg.RESULTS_DIR / "main_user_IVs" / "20260427_165233"
SOURCE_CONV_DIR = SOURCE_RUN_DIR / "conversations"
OUTPUT_DIR_NAME = "conversations_none_reflection"
OUTPUT_CONV_DIR = SOURCE_RUN_DIR / OUTPUT_DIR_NAME
CHECKPOINT_PATH = SOURCE_RUN_DIR / "checkpoint_ablation.json"
MANIFEST_PATH = SOURCE_RUN_DIR / "manifest_ablation.json"
SAMPLE_INDEX_PATH = SOURCE_RUN_DIR / "ablation_sample_index.json"

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

USER_AGENT_PROVIDER = main_cfg.USER_AGENT_PROVIDER
USER_AGENT_MODEL = main_cfg.USER_AGENT_MODEL
