"""
evaluator_validation/config.py
==============================
Configuration for RQ3 evaluator-validation.

The PRIMARY evaluator is implicit — it's whichever model produced the
existing unsuffixed score files (``<sid>__<rubric>.json``) under the
target run dir. We don't re-score with it.

The SECONDARY evaluator is run here on the stratified 10% sample with
the same rubric. Its scores are written with the evaluator slug
appended to the filename so they live side-by-side with the primary
files in the run's ``scores/`` directory.
"""

from __future__ import annotations


# ── Sampling ────────────────────────────────────────────────────────────────
# 10% of conversations, stratified by (belief.category, belief.subtype) so
# every (dataset, sub-category) combination is represented.
SAMPLE_FRACTION = 0.10

# Random seed for reproducibility. Same seed + same conversation pool
# always yields the same sampled session_ids.
SAMPLE_SEED = 42

# Floor so rare strata (small subtypes) are not silently dropped from the
# validation set. With 10% and ceil(), strata of size 1-9 already round
# up to 1, but this makes the contract explicit.
MIN_PER_STRATUM = 1

# Stratification keys, in order. Both are read from each conversation's
# embedded `belief` block at sampling time.
STRATIFICATION_KEYS = ("category", "subtype")


# ── Rubric ──────────────────────────────────────────────────────────────────
# Must match the rubric used for the primary scoring pass. The rubric
# definition itself lives in core.scoring.RUBRICS — we just name it.
RUBRIC_NAME = "misinfo_v1_split"


# ── Evaluators ──────────────────────────────────────────────────────────────
# Primary is what produced the unsuffixed score files in <run_dir>/scores/.
# Recorded here for documentation only — this script does not call it.
PRIMARY_EVALUATOR = ("openrouter", "openai/gpt-4.1-mini")

# Secondary, per experiment-doc.md. Overridable at the CLI via
# --secondary-provider / --secondary-model.
SECONDARY_EVALUATOR = ("openrouter", "openai/gpt-4o-mini")

# Deterministic scoring — match the primary pass.
EVALUATOR_TEMPERATURE = 0.0


# ── Output layout under <run_dir>/ ──────────────────────────────────────────
# The validation outputs are namespaced inside the existing run dir so
# they travel with the conversations they describe.
#
#   <run_dir>/
#     ├── conversations/        (existing; read-only here)
#     ├── scores/               (existing; secondary score files added)
#     │     ├── <sid>__misinfo_v1_split.json                          ← primary
#     │     └── <sid>__misinfo_v1_split__<secondary-slug>.json        ← new
#     └── evaluator_validation/                                        ← new tree
#           ├── sample_manifest.json
#           ├── coverage_report.csv
#           └── analysis_output/   (populated by analyze_agreement.py)
#                 ├── tables/
#                 └── plots/
VALIDATION_SUBDIR = "evaluator_validation"
SAMPLE_MANIFEST_NAME = "sample_manifest.json"
COVERAGE_REPORT_NAME = "coverage_report.csv"