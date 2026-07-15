"""
response_diversity/config.py
============================
Paths and knobs for the response-diversity analysis. Mirrors the source
run selection convention used by reflection_ablation so the analysis can
target any main_user_IVs run directory.
"""

from __future__ import annotations

import os
from pathlib import Path

# Reuse main study config for shared knobs.
from main_user_IVs import config as main_cfg

# ── Source run selection ─────────────────────────────────────────────────────
# Same convention as reflection_ablation: override with the
# RESPONSE_DIVERSITY_SOURCE_RUN env var to point at a different run.
_DEFAULT_SOURCE_RUN = main_cfg.RESULTS_DIR / "main_user_IVs" / "20260427_165233"
_SOURCE_RUN_ENV = os.environ.get("RESPONSE_DIVERSITY_SOURCE_RUN")
SOURCE_RUN_DIR = (
    Path(_SOURCE_RUN_ENV).expanduser().resolve()
    if _SOURCE_RUN_ENV
    else _DEFAULT_SOURCE_RUN
)

# ── Paths ────────────────────────────────────────────────────────────────────
SOURCE_CONV_DIR = SOURCE_RUN_DIR / "conversations"
PER_TURN_OUTPUT_PATH = SOURCE_RUN_DIR / "response_diversity_per_turn.csv"
SESSION_OUTPUT_PATH = SOURCE_RUN_DIR / "response_diversity_session.csv"
SAMPLE_INDEX_PATH = SOURCE_RUN_DIR / "response_diversity_sample_index.json"

# ── Analysis knobs ───────────────────────────────────────────────────────────
N_TURNS = main_cfg.N_TURNS  # 8

# Only these models are analyzed; Claude is excluded per the spec.
# Keys match the short model slug embedded in session_id (…__model-<slug>.json)
# as well as `models.target_llm` after normalization.
INCLUDED_TARGET_MODELS: tuple[str, ...] = (
    "gpt-5.3-chat",
    "gemini-3-flash",
    "deepseek-v3.2",
)

# ── Sampling ────────────────────────────────────────────────────────────────
# Stratified per (target_llm, iv1, belief_category) cell. Unlike
# reflection_ablation, we do NOT filter to break conversations — diversity
# is a property of every session, so the pool is every valid session for
# the included target models.
SAMPLE_FRACTION = 0.12
SAMPLING_SEED = 42
