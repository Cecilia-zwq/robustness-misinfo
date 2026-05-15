"""
reflection_ablation/sampling.py
===============================
Stratified sampling over (iv1 × belief_category × target_llm) cells.

For each cell:
  quota = max(floor(cell_size * SAMPLE_FRACTION), min(cell_size, MIN_PER_CELL))

Within each cell, sample `round(quota * TURN1_RATIO)` from turn-1 breaks
and the remainder from turn-2+ breaks. If one pool is short, the other
fills the gap. If both combined < quota we take all of them.

A "break conversation" is one whose first break turn is identified by
the same rule the analysis notebook uses: the lowest-numbered turn
whose ``reflection_attempts`` list has length > 1. Conversations with
no break are skipped entirely.

The script is read-only over the source corpus — it only produces a
deterministic plan, written to ``ablation_sample_index.json`` for
reproducibility and resume.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SampleEntry:
    """One planned ablation session."""
    source_session_id: str
    source_path: str
    iv1: str
    belief_category: str
    target_llm: str
    first_break_turn: int       # 1-indexed
    n_turns: int                # original conversation length


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _first_break_turn(turns: list[dict]) -> int | None:
    """Return 1-indexed turn number of the first break, or None."""
    for idx, turn in enumerate(turns, start=1):
        attempts = turn.get("reflection_attempts") or []
        try:
            n = len(attempts)
        except TypeError:
            n = 0
        if n > 1:
            return int(turn.get("turn", idx))
    return None


def _load_conversation_meta(path: Path) -> tuple[dict, int] | None:
    """Return (meta_dict, first_break_turn) for a single conversation file.

    Skips files that don't parse, lack a break turn, or look incomplete.
    """
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

    first_brk = _first_break_turn(turns)
    if first_brk is None:
        return None

    cell = data.get("cell") or {}
    belief = data.get("belief") or {}
    models_ = data.get("models") or {}

    meta = {
        "source_session_id": str(data.get("session_id", path.stem)),
        "source_path": str(path),
        "iv1": str(cell.get("iv1", "unknown")),
        "belief_category": str(belief.get("category", "unknown")),
        "target_llm": str(models_.get("target_llm", "unknown")),
        "n_turns": int(len(turns)),
    }
    return meta, first_brk


# ════════════════════════════════════════════════════════════════════════════
# Sampling
# ════════════════════════════════════════════════════════════════════════════

def build_sample(
    source_dir: Path,
    *,
    sample_fraction: float,
    min_per_cell: int,
    turn1_ratio: float,
    seed: int,
) -> list[SampleEntry]:
    """Scan source_dir, group break conversations into cells, sample per spec.

    Returns a list of SampleEntry, one per planned ablation session.
    """
    rng = random.Random(seed)

    # collect break conversations per cell
    by_cell: dict[tuple, list[tuple[dict, int]]] = defaultdict(list)
    for path in sorted(source_dir.glob("*.json")):
        loaded = _load_conversation_meta(path)
        if loaded is None:
            continue
        meta, first_brk = loaded
        key = (meta["iv1"], meta["belief_category"], meta["target_llm"])
        by_cell[key].append((meta, first_brk))

    planned: list[SampleEntry] = []
    cell_stats: list[dict] = []

    for key in sorted(by_cell.keys()):
        pool = by_cell[key]
        cell_size = len(pool)

        # quota for this cell
        quota = max(
            int(math.floor(cell_size * sample_fraction)),
            min(cell_size, min_per_cell),
        )
        if quota <= 0:
            continue

        # split pool by turn-1 vs turn-2+ breaks
        turn1 = [(m, t) for (m, t) in pool if t == 1]
        turn2p = [(m, t) for (m, t) in pool if t >= 2]

        # desired split honoring turn1_ratio
        n_turn1 = int(round(quota * turn1_ratio))
        n_turn2 = quota - n_turn1

        # rebalance against availability
        if n_turn2 > len(turn2p):
            n_turn2 = len(turn2p)
            n_turn1 = min(quota - n_turn2, len(turn1))
        if n_turn1 > len(turn1):
            n_turn1 = len(turn1)
            n_turn2 = min(quota - n_turn1, len(turn2p))

        rng.shuffle(turn1)
        rng.shuffle(turn2p)
        picks = turn1[:n_turn1] + turn2p[:n_turn2]

        for meta, first_brk in picks:
            planned.append(SampleEntry(
                source_session_id=meta["source_session_id"],
                source_path=meta["source_path"],
                iv1=meta["iv1"],
                belief_category=meta["belief_category"],
                target_llm=meta["target_llm"],
                first_break_turn=first_brk,
                n_turns=meta["n_turns"],
            ))

        cell_stats.append({
            "iv1": key[0],
            "belief_category": key[1],
            "target_llm": key[2],
            "cell_size": cell_size,
            "quota": quota,
            "picked_turn1": n_turn1,
            "picked_turn2plus": n_turn2,
            "available_turn1": len(turn1),
            "available_turn2plus": len(turn2p),
        })

    print(f"\nPlanned {len(planned)} ablation sessions "
          f"across {len(cell_stats)} cells.")
    print(f"  turn-1 picks total : {sum(c['picked_turn1']    for c in cell_stats):,}")
    print(f"  turn-2+ picks total: {sum(c['picked_turn2plus'] for c in cell_stats):,}")
    return planned


def write_sample_index(
    out_path: Path,
    *,
    planned: list[SampleEntry],
    params: dict,
) -> None:
    """Persist the sampling plan for reproducibility and resume."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": params,
        "n_planned": len(planned),
        "entries": [asdict(e) for e in planned],
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def read_sample_index(in_path: Path) -> list[SampleEntry]:
    """Load a previously persisted sampling plan."""
    with in_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return [SampleEntry(**e) for e in payload["entries"]]
