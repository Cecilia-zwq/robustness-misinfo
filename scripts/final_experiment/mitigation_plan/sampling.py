"""
mitigation_plan/sampling.py
============================
Stratified selection of existing main_user_IVs sessions to re-run under
the mitigation system prompt (see config.MITIGATION_STATEMENT).

Same randomized-round quota algorithm as
stance_analysis/select_target_conversations.stratified_sample_session_ids
and response_diversity/sampling.py, re-implemented locally here (repo
convention: each analysis folder owns its own copy rather than importing
across sibling packages).

quota = randomized_round(cell_size * fraction), where
randomized_round(x) = floor(x)+1 with probability equal to the
fractional part of x, else floor(x) — deterministic given `seed`, with
E[quota] == cell_size * fraction.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

STRATA_COLS: tuple[str, ...] = ("cell_id", "target_model")


def _load_meta(path: Path) -> dict | None:
    """Return session metadata if the artifact is complete, else None."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None

    if not data.get("turns") or data.get("completed_at") is None:
        return None

    sid = str(data.get("session_id", path.stem))
    if "__model-" not in sid:
        return None
    target_model = sid.rsplit("__model-", 1)[1]

    cell = data.get("cell") or {}
    belief = data.get("belief") or {}
    return {
        "session_id":      sid,
        "source_path":     str(path),
        "cell_id":         str(cell.get("cell_id", "unknown")),
        "iv1":             str(cell.get("iv1", "unknown")),
        "iv2":             str(cell.get("iv2", "unknown")),
        "belief_category": str(belief.get("category", "unknown")),
        "belief_subtype":  str(belief.get("subtype", "unknown")),
        "target_model":    target_model,
    }


def build_matched_sample(
    source_conv_dir: Path,
    *,
    sample_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Stratified sample of completed sessions under `source_conv_dir`.

    Returns (entries, cell_stats). Strata are STRATA_COLS
    (cell_id, target_model) — this covers whichever IV cells actually
    have data in the source run (main_user_IVs' iv1 sweep at the time of
    writing; iv2 sweep cells simply won't appear if absent).
    """
    if not source_conv_dir.exists():
        raise FileNotFoundError(f"source_conv_dir not found: {source_conv_dir}")
    if not (0.0 <= sample_fraction <= 1.0):
        raise ValueError(f"sample_fraction must be in [0, 1], got {sample_fraction!r}")

    rng = random.Random(seed)

    by_stratum: dict[tuple, list[dict]] = defaultdict(list)
    for path in sorted(source_conv_dir.glob("*.json")):
        meta = _load_meta(path)
        if meta is None:
            continue
        key = tuple(meta[c] for c in STRATA_COLS)
        by_stratum[key].append(meta)

    entries: list[dict] = []
    cell_stats: list[dict] = []
    for key in sorted(by_stratum.keys()):
        pool = by_stratum[key]
        cell_size = len(pool)

        expected = cell_size * sample_fraction
        quota = int(expected)
        if rng.random() < (expected - quota):
            quota += 1
        quota = min(quota, cell_size)
        if quota <= 0:
            continue

        rng.shuffle(pool)
        entries.extend(pool[:quota])
        cell_stats.append({
            **dict(zip(STRATA_COLS, key)),
            "cell_size": cell_size,
            "quota": quota,
        })

    entries.sort(key=lambda e: e["session_id"])
    return entries, cell_stats


def write_sample_index(
    out_path: Path,
    *,
    entries: list[dict],
    cell_stats: list[dict],
    params: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": params,
        "cell_stats": cell_stats,
        "n_sessions": len(entries),
        "entries": entries,
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def read_sample_index(in_path: Path) -> list[dict]:
    with in_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)["entries"]
