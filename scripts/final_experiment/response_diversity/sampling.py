"""
response_diversity/sampling.py
==============================
Stratified sampling over (target_llm × iv1 × belief_category) cells for the
response-diversity analysis. Adapted from reflection_ablation/sampling.py
but with two differences:

- No "break conversation" filter — every valid session is eligible.
- No turn-1 vs turn-2+ split — response diversity does not depend on a
  break turn.

For each cell::

    quota = randomized_round(cell_size * SAMPLE_FRACTION)

where randomized_round(x) = floor(x)+1 with probability equal to the
fractional part of x, else floor(x). With a fixed seed this is
deterministic and reproducible; E[quota] = cell_size * SAMPLE_FRACTION.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleEntry:
    session_id:      str
    source_path:     str
    iv1:             str
    iv2:             str
    belief_category: str
    target_llm:      str          # normalized short slug
    n_turns:         int


def _short_model(model_str: str) -> str:
    short = model_str.split("/")[-1]
    if short == "gemini-3-flash":
        return "gemini-3-flash-preview"
    return short


def _load_meta(path: Path, included_models: tuple[str, ...]) -> dict | None:
    """Return metadata dict if session is complete and in-scope, else None."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None

    turns = data.get("turns") or []
    if not turns:
        return None
    if data.get("completed_at") is None:
        return None

    models_ = data.get("models") or {}
    short = _short_model(str(models_.get("target_llm", "")))
    # match either raw slug or the notebook-normalized "…preview" form
    if short not in included_models and not (
        short == "gemini-3-flash-preview" and "gemini-3-flash" in included_models
    ):
        return None

    cell = data.get("cell") or {}
    belief = data.get("belief") or {}
    return {
        "session_id":      str(data.get("session_id", path.stem)),
        "source_path":     str(path),
        "iv1":             str(cell.get("iv1", "unknown")),
        "iv2":             str(cell.get("iv2", "unknown")),
        "belief_category": str(belief.get("category", "unknown")),
        "target_llm":      short,
        "n_turns":         int(len(turns)),
    }


def build_sample(
    source_dir: Path,
    *,
    sample_fraction: float,
    seed: int,
    included_models: tuple[str, ...],
) -> list[SampleEntry]:
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"source_dir not found: {source_dir}")
    if not (0.0 <= sample_fraction <= 1.0):
        raise ValueError(f"sample_fraction must be in [0, 1], got {sample_fraction!r}")

    rng = random.Random(seed)

    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for path in sorted(source_dir.glob("*.json")):
        meta = _load_meta(path, included_models)
        if meta is None:
            continue
        key = (meta["target_llm"], meta["iv1"], meta["belief_category"])
        by_cell[key].append(meta)

    planned: list[SampleEntry] = []
    cell_stats: list[dict] = []

    for key in sorted(by_cell.keys()):
        pool = by_cell[key]
        cell_size = len(pool)

        expected = cell_size * sample_fraction
        quota = int(math.floor(expected))
        if rng.random() < (expected - quota):
            quota += 1
        if quota <= 0:
            continue

        rng.shuffle(pool)
        picks = pool[:quota]

        for meta in picks:
            planned.append(SampleEntry(**meta))

        cell_stats.append({
            "target_llm":      key[0],
            "iv1":             key[1],
            "belief_category": key[2],
            "cell_size":       cell_size,
            "quota":           quota,
        })

    print(f"\nPlanned {len(planned)} diversity sessions "
          f"across {len(cell_stats)} cells "
          f"(fraction={sample_fraction}, seed={seed}).")
    return planned


def write_sample_index(
    out_path: Path,
    *,
    planned: list[SampleEntry],
    params: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": params,
        "n_planned": len(planned),
        "entries": [asdict(e) for e in planned],
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def read_sample_index(in_path: Path) -> list[SampleEntry]:
    with in_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return [SampleEntry(**e) for e in payload["entries"]]
