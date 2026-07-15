"""
stance_analysis/scoring_progress.py
====================================
Shared incremental-scoring bookkeeping for run_stance_scoring.py (the
turn-independent misinfo_stance_split rubric) and
run_stance_history_scoring.py (the history-aware misinfo_stance_history_split
rubric): given a session's (possibly partial) score file on disk, which
turns are already done, and a pre-flight scan of how many evaluator
calls actually remain across a whole target/sample set.

Both scoring scripts write into the same scores/ directory, one file
per (session_id, rubric_name), incrementally after every turn (see
each script's module docstring for the resume contract). This module
is the parsing/scanning logic they share; parameterized by
``rubric_name`` since that's the only thing that differs between them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from core import TurnScores

from . import config as cfg

logger = logging.getLogger(__name__)


def score_path(session_id: str, rubric_name: str) -> Path:
    return cfg.SCORES_DIR / f"{session_id}__{rubric_name}.json"


def load_existing_turns(
    path: Path, *, expected_evaluator: str,
) -> dict[int, TurnScores]:
    """Turn-number -> TurnScores already on disk for this session, if any.

    Returns {} if there's no file, the file is unreadable, or the file
    was scored with a *different* evaluator (mixing two evaluators'
    results into one file would silently corrupt the artifact — safer
    to warn and rescore from turn 1 than to guess).
    """
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}

    if data.get("evaluator_model") != expected_evaluator:
        logger.warning(
            "%s: existing score file was written with evaluator=%s, not "
            "the requested %s — ignoring it and rescoring from turn 1 "
            "(this will overwrite the file with the new evaluator's results).",
            path.stem, data.get("evaluator_model"), expected_evaluator,
        )
        return {}

    return {int(t["turn"]): TurnScores(**t) for t in data.get("turns", [])}


def is_fully_scored(
    session_id: str, rubric_name: str, *, expected_evaluator: str, n_turns: int,
) -> bool:
    existing = load_existing_turns(
        score_path(session_id, rubric_name), expected_evaluator=expected_evaluator,
    )
    return len(existing) >= n_turns


def scan_progress(
    entries: list[dict], rubric_name: str, *, expected_evaluator: str, n_turns: int,
) -> dict:
    """Pre-flight scan: exactly how many evaluator calls remain, before any run.

    Reads each session's (possibly partial) score file once and reports
    both session-level and individual-call-level counts. Since scoring
    is incremental, a partially-done session needs fewer calls than a
    fresh one, so "N sessions remaining" alone understates or overstates
    true remaining cost.
    """
    n_done = n_partial = n_not_started = 0
    n_calls_remaining = 0
    for e in entries:
        n_existing = len(load_existing_turns(
            score_path(e["session_id"], rubric_name), expected_evaluator=expected_evaluator,
        ))
        if n_existing >= n_turns:
            n_done += 1
        elif n_existing > 0:
            n_partial += 1
            n_calls_remaining += n_turns - n_existing
        else:
            n_not_started += 1
            n_calls_remaining += n_turns
    return {
        "n_sessions":         len(entries),
        "n_done":             n_done,
        "n_partial":          n_partial,
        "n_not_started":      n_not_started,
        "n_calls_remaining":  n_calls_remaining,
    }
