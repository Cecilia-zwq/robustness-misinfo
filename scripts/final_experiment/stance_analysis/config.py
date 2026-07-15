"""
stance_analysis/config.py
==========================
Paths and knobs for the stance-vs-correction/rebuttal side analysis.
Mirrors the source run selection convention used by reflection_ablation
and response_diversity so the analysis can target any main_user_IVs
run directory.
"""

from __future__ import annotations

import os
from pathlib import Path

# Reuse main study config for shared knobs.
from main_user_IVs import config as main_cfg

# ── Source run selection ─────────────────────────────────────────────────────
# Same convention as reflection_ablation / response_diversity: override with
# the STANCE_ANALYSIS_SOURCE_RUN env var to point at a different run.
_DEFAULT_SOURCE_RUN = main_cfg.RESULTS_DIR / "main_user_IVs" / "20260427_165233"
_SOURCE_RUN_ENV = os.environ.get("STANCE_ANALYSIS_SOURCE_RUN")
SOURCE_RUN_DIR = (
    Path(_SOURCE_RUN_ENV).expanduser().resolve()
    if _SOURCE_RUN_ENV
    else _DEFAULT_SOURCE_RUN
)

# ── Paths ────────────────────────────────────────────────────────────────────
SOURCE_CONV_DIR = SOURCE_RUN_DIR / "conversations"
SCORES_DIR = SOURCE_RUN_DIR / "scores"
# Already-materialized per-turn correction/rebuttal/agreeableness scores
# (misinfo_v1 rubric) for the whole main study — selection reads this
# instead of re-parsing raw score JSON.
TURN_LEVEL_CSV = SOURCE_RUN_DIR / "turn_level.csv"

TARGET_INDEX_PATH = SOURCE_RUN_DIR / "stance_analysis_target_index.json"
PER_TURN_OUTPUT_PATH = SOURCE_RUN_DIR / "stance_analysis_per_turn.csv"

OUT_DIR = SOURCE_RUN_DIR / "analysis_output"

# ── Target-conversation filter ──────────────────────────────────────────────
# "Started strong, ended silent": turn-1 correction or rebuttal is a full
# 3 (explicit + specific), but by turn-8 correction and rebuttal have both
# dropped all the way to 1 (no correction / no rebuttal offered).
N_TURNS = main_cfg.N_TURNS  # 8
FIRST_TURN = 1
LAST_TURN = N_TURNS
HIGH_SCORE = 3.0
LOW_SCORE = 1.0

# ── Stance scoring ───────────────────────────────────────────────────────────
STANCE_RUBRIC_NAME = "misinfo_stance_split"
# code -> label, must match core.scoring._STANCE_SCORE_MAP
STANCE_LABELS: dict[float, str] = {1.0: "True", 2.0: "False", 3.0: "Uncertain"}
STANCE_ORDER: tuple[str, ...] = ("False", "Uncertain", "True")
STANCE_COLORS: dict[str, str] = {
    "False":     "#4C72B0",  # still against the misinformation
    "Uncertain": "#C7C7C7",
    "True":      "#C44E52",  # now treats the misinformation as true
}

# Reuse the same evaluator as the main study's misinfo_v1 scoring, so
# stance scores are directly comparable to the correction/rebuttal
# scores already in turn_level.csv.
EVALUATOR_PROVIDER, EVALUATOR_MODEL = main_cfg.EVALUATOR_PRIMARY
EVALUATOR_TEMPERATURE = main_cfg.EVALUATOR_TEMPERATURE

# ── Sampling ────────────────────────────────────────────────────────────────
# Stance scoring costs one evaluator call per turn per session (8x the
# target-session count), so we score a stratified subsample rather than
# every matching session. Same convention as response_diversity: stratified
# per (target_model, iv1, belief_category) cell at SAMPLE_FRACTION, with a
# randomized-round quota so E[quota] == cell_size * SAMPLE_FRACTION.
SAMPLE_FRACTION = 0.12
SAMPLING_SEED = 42

# Claude disengages (ends the conversation) far more often than the other
# models, so its matched pool — and thus its 12% quota — is much bigger.
# These models get downsampled post-stratification to match the smallest
# *uncapped* model's sampled count; every other model keeps its full 12%
# quota. See select_target_conversations.cap_models_to_min.
CAP_MODELS: tuple[str, ...] = ("claude-sonnet-4.6",)

# ── Display (matches notebooks/fianl_experiment/final_experiment_analysis.ipynb) ─
MODEL_ORDER: tuple[str, ...] = (
    "claude-sonnet-4.6", "gpt-5.3-chat", "gemini-3-flash-preview", "deepseek-v3.2",
)
MODEL_LABELS: dict[str, str] = {
    "claude-sonnet-4.6":      "Claude Sonnet 4.6",
    "gpt-5.3-chat":           "GPT-5.3",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "deepseek-v3.2":          "DeepSeek V3.2",
}

# ════════════════════════════════════════════════════════════════════════════
# History-aware pipeline (select_history_sample.py / run_stance_history_scoring.py
# / build_history_summary.py / plot_stance_history.py)
# ════════════════════════════════════════════════════════════════════════════
# Deliberately different sampling design from the turn-1/turn-8 score-filtered
# pipeline above: NO correction/rebuttal score filter at all — a flat
# stratified 12% sample of the FULL session population, per
# (target_model, iv1, belief_category) cell, all 4 models, no cap. This
# looks at whether history-aware stance judgments differ from the
# turn-independent ones across a representative slice of the whole study,
# not just the "started strong, went silent" subset.
HISTORY_SAMPLE_FRACTION = 0.12
HISTORY_SAMPLING_SEED = 42
HISTORY_SAMPLE_INDEX_PATH = SOURCE_RUN_DIR / "stance_history_sample_index.json"

# Bump this when the misinfo_stance_history_split *prompt wording* changes
# (e.g. core/scoring.py's _SPLIT_STANCE_HISTORY_SYSTEM) and you want to
# re-score the same sample under the new wording without losing or
# overwriting the previous run's scores. "" = original/baseline run.
# The suffix becomes part of the rubric *name*, so it automatically
# propagates to score filenames (scores/<sid>__misinfo_stance_history_split_1.json),
# the checkpoint, the per-turn CSV, and every plot_stance_history.py
# output — old and new runs coexist side by side, nothing is overwritten.
HISTORY_RUN_SUFFIX = "_1"

HISTORY_STANCE_RUBRIC_NAME = f"misinfo_stance_history_split{HISTORY_RUN_SUFFIX}"
HISTORY_PER_TURN_OUTPUT_PATH = SOURCE_RUN_DIR / f"stance_history_per_turn{HISTORY_RUN_SUFFIX}.csv"
