r"""
human_evaluation/make_rater_blocks.py
=====================================
Cut the task-1 master item set into rater blocks and assign them.

Consumes ``task1_annotation_items.csv`` (from ``make_annotation_csv.py``)
and produces the fielding plan: which rater sees which items, in which
session, so that every item collects ``RATERS_PER_ITEM`` independent
ratings.

Blocking — equal reading load, mixed lengths
--------------------------------------------
Item count is a poor proxy for effort here: totals run from under 150 to
over 1400 words, and the binary ``is_long_text`` flag does not separate
them cleanly (length varies widely *within* both levels). Blocks are
therefore built by **banding on word count**:

1. Rank all N items by total words (belief + user message + response).
2. Cut the ranking into ``BLOCK_SIZE`` equal bands — with 600 items in
   blocks of 10, that is 10 bands of 60.
3. Give every block exactly one item from every band.

Each rater's block then spans the whole length distribution — one of the
shortest items, one of the longest, and one from each decile between —
so total reading time is near-identical across raters *and* no one faces
a block of uniformly punishing text. Mixing long and short is the point:
it is what keeps fatigue comparable rather than merely making the
averages match.

Within a band, which of the 60 items goes to which of the 60 blocks is
chosen greedily to spread ``BLOCK_BALANCE_KEYS`` (condition, model, turn,
category), so length balance is achieved *and* no rater's items are
concentrated in one experimental cell.

Assignment — rotation
---------------------
Blocks are dealt to raters by rotation: with B blocks, R raters per item
and q blocks per rater, block ``b`` goes to raters at successive offsets
so that each block is seen by exactly R distinct raters and each rater
holds exactly q distinct blocks. Requires ``q`` to divide ``B``; the
resulting rater count is ``R * B / q``.

At the defaults (600 items, blocks of 10, 2 blocks per rater, 3 raters
per item): 60 blocks, 90 raters, 20 items each.

Sessions
--------
Each rater's q blocks map one-to-one onto q sessions with a break between
them, so the break always falls on a block boundary and the two halves
are length-matched by construction.

Usage
-----
::

    cd scripts/final_experiment

    python -m human_evaluation.make_rater_blocks \
        --run-dir ../../results/final_experiment/main_user_IVs/20260427_165233

    # Shorter sessions: blocks of 6, 2 per rater -> 12 items, 150 raters
    python -m human_evaluation.make_rater_blocks \
        --run-dir <run-dir> --block-size 6

    # Inspect the plan without writing per-block CSVs
    python -m human_evaluation.make_rater_blocks \
        --run-dir <run-dir> --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from . import config as cfg  # noqa: E402
from .make_annotation_csv import parse_session_id  # noqa: E402


_TAG_RE = re.compile(r"<[^>]+>")
_TEXT_COLUMNS = ("user_message", "misinformation_belief", "ai_message")


# ════════════════════════════════════════════════════════════════════════════
# Loading and measurement
# ════════════════════════════════════════════════════════════════════════════

def count_words(row: dict) -> int:
    """Words a rater actually reads for one item.

    HTML tags are stripped first — the cells are rendered markup, so
    ``<strong>`` and ``<br>`` are not words on screen. Covers all three
    stimulus columns because the belief is shown alongside the exchange.
    """
    total = 0
    for col in _TEXT_COLUMNS:
        total += len(_TAG_RE.sub(" ", str(row.get(col, ""))).split())
    return total


def load_items(csv_path: Path, encoding: str) -> list[dict]:
    """Read the master CSV and attach word counts and design factors."""
    if not csv_path.exists():
        raise SystemExit(
            f"Master item file not found: {csv_path}\n"
            "Build it first:\n"
            "  python -m human_evaluation.make_annotation_csv --run-dir <run-dir>"
        )
    with open(csv_path, "r", encoding=encoding, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{csv_path} is empty.")

    items: list[dict] = []
    for i, row in enumerate(rows):
        meta = parse_session_id(row["session_id"])
        items.append({
            **row,
            **meta,
            "turn": int(row["turn"]),
            "is_long_text": int(row.get("is_long_text", 0)),
            "n_words": count_words(row),
            "_ord": i,
        })
    return items


# ════════════════════════════════════════════════════════════════════════════
# Blocking
# ════════════════════════════════════════════════════════════════════════════

def build_blocks(
    items: list[dict],
    *,
    block_size: int,
    balance_keys: tuple[str, ...],
    seed: int,
) -> list[list[dict]]:
    """Partition items into equal-size, length-banded, factor-balanced blocks.

    Banding (step 1-3 in the module docstring) fixes the length profile of
    every block: each takes one item per band, so block word totals differ
    only by within-band variation. The greedy step then decides *which*
    item from each band lands in each block, minimising a factor-imbalance
    cost so that a rater does not end up with, say, four hostile-condition
    items out of ten.

    Cost for placing an item in a block is the count of items already in
    that block sharing each balance key. Ties are broken randomly, so the
    assignment is seeded rather than order-dependent.
    """
    n = len(items)
    if n % block_size:
        raise SystemExit(
            f"{n} items is not divisible by --block-size {block_size}. "
            f"Pick a divisor of {n} (e.g. "
            f"{', '.join(str(k) for k in range(4, 21) if n % k == 0)})."
        )
    n_blocks = n // block_size
    rng = random.Random(seed)

    # Rank by reading load. The random second key keeps equal-length items
    # from always landing in the same relative position.
    ranked = sorted(items, key=lambda it: (it["n_words"], rng.random()))

    blocks: list[list[dict]] = [[] for _ in range(n_blocks)]
    # Per-block tallies of each (key, value) pair, for the imbalance cost.
    tallies: list[Counter] = [Counter() for _ in range(n_blocks)]

    for band_start in range(0, n, n_blocks):
        band = ranked[band_start:band_start + n_blocks]
        # Every block receives exactly one item from this band.
        open_blocks = set(range(n_blocks))
        # Place the most "constrained" items first — those whose factor
        # combination is rarest in the band — so common items absorb the
        # leftover slots rather than the other way round.
        band_profile: Counter = Counter()
        for it in band:
            for k in balance_keys:
                band_profile[(k, it[k])] += 1
        band = sorted(
            band,
            key=lambda it: (
                min(band_profile[(k, it[k])] for k in balance_keys),
                rng.random(),
            ),
        )
        for it in band:
            best_b, best_cost = None, None
            for b in open_blocks:
                cost = sum(tallies[b][(k, it[k])] for k in balance_keys)
                key = (cost, rng.random())
                if best_cost is None or key < best_cost:
                    best_b, best_cost = b, key
            blocks[best_b].append(it)
            for k in balance_keys:
                tallies[best_b][(k, it[k])] += 1
            open_blocks.discard(best_b)

    return blocks


def order_within_block(
    blocks: list[list[dict]], *, mode: str, seed: int,
) -> None:
    """Set presentation order inside each block, in place."""
    rng = random.Random(seed + 1)
    for block in blocks:
        if mode == "ascending":
            block.sort(key=lambda it: it["n_words"])
        elif mode == "shuffle":
            rng.shuffle(block)
        else:
            raise ValueError(f"Unknown block item order: {mode!r}")


# ════════════════════════════════════════════════════════════════════════════
# Rotation: blocks -> raters
# ════════════════════════════════════════════════════════════════════════════

def assign_blocks_to_raters(
    n_blocks: int,
    *,
    raters_per_item: int,
    blocks_per_rater: int,
) -> dict[int, list[int]]:
    """Rotate blocks onto raters.

    Builds ``raters_per_item`` passes over the block list, each pass
    rotated by a stride, then deals the concatenated slots to raters in
    contiguous runs of ``blocks_per_rater``. Because a run sits inside a
    single pass (guaranteed when ``blocks_per_rater`` divides ``n_blocks``),
    the blocks in a run are always distinct.

    Returns {rater_id: [block_id, ...]}, and verifies both regularity
    conditions before returning — a silently malformed design would only
    surface after the data were collected.
    """
    if n_blocks % blocks_per_rater:
        raise SystemExit(
            f"--blocks-per-rater {blocks_per_rater} must divide the number "
            f"of blocks ({n_blocks})."
        )
    n_raters = raters_per_item * n_blocks // blocks_per_rater

    slots: list[int] = []
    for p in range(raters_per_item):
        # Stride the passes apart so a block's raters are spread through
        # the rater index rather than clustered.
        offset = (p * n_blocks) // raters_per_item
        slots.extend((b + offset) % n_blocks for b in range(n_blocks))

    assignment = {
        r: slots[r * blocks_per_rater:(r + 1) * blocks_per_rater]
        for r in range(n_raters)
    }

    # Verify: every block seen by exactly R distinct raters; every rater
    # holds q distinct blocks.
    seen: dict[int, set[int]] = defaultdict(set)
    for rater, block_ids in assignment.items():
        if len(set(block_ids)) != blocks_per_rater:
            raise SystemExit(
                f"Rotation produced a repeated block for rater {rater}: "
                f"{block_ids}. Adjust --block-size / --blocks-per-rater."
            )
        for b in block_ids:
            seen[b].add(rater)
    bad = {b: len(rs) for b, rs in seen.items() if len(rs) != raters_per_item}
    if bad or len(seen) != n_blocks:
        raise SystemExit(
            f"Rotation is not {raters_per_item}-regular over blocks: {bad}"
        )
    return assignment


# ════════════════════════════════════════════════════════════════════════════
# Timing estimate
# ════════════════════════════════════════════════════════════════════════════

def estimate_session_minutes(block: list[dict]) -> float:
    """Reading + rating minutes for one block, from its actual word count."""
    words = sum(it["n_words"] for it in block)
    return (words / cfg.WORDS_PER_MINUTE
            + len(block) * cfg.MINUTES_PER_ITEM_RATING)


# ════════════════════════════════════════════════════════════════════════════
# Output
# ════════════════════════════════════════════════════════════════════════════

def _write_csv(path: Path, rows: list[dict], columns: list[str],
               encoding: str) -> None:
    with open(path, "w", encoding=encoding, newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _balance_report(blocks: list[list[dict]]) -> dict:
    """Per-block spread of the things a rater's experience depends on."""
    words = [sum(it["n_words"] for it in b) for b in blocks]
    mins = [estimate_session_minutes(b) for b in blocks]
    longs = [sum(it["is_long_text"] for it in b) for b in blocks]
    mean_w = sum(words) / len(words)
    spread = {
        "block_words_min": min(words),
        "block_words_max": max(words),
        "block_words_mean": round(mean_w, 1),
        "block_words_cv": round(
            (sum((w - mean_w) ** 2 for w in words) / len(words)) ** 0.5
            / mean_w, 4,
        ),
        "session_minutes_min": round(min(mins), 1),
        "session_minutes_max": round(max(mins), 1),
        "long_text_per_block_min": min(longs),
        "long_text_per_block_max": max(longs),
    }
    for key in cfg.BLOCK_BALANCE_KEYS:
        per_block_max = [
            max(Counter(it[key] for it in b).values()) for b in blocks
        ]
        spread[f"max_same_{key}_in_a_block"] = max(per_block_max)
    return spread


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--task", type=str, default=cfg.TASK_ID)
    p.add_argument("--block-size", type=int, default=cfg.BLOCK_SIZE)
    p.add_argument("--blocks-per-rater", type=int, default=cfg.BLOCKS_PER_RATER)
    p.add_argument("--raters-per-item", type=int, default=cfg.RATERS_PER_ITEM)
    p.add_argument(
        "--item-order", choices=("shuffle", "ascending"),
        default=cfg.BLOCK_ITEM_ORDER,
        help="Presentation order within a block. shuffle (default) keeps "
             "position effects from correlating with item length.",
    )
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--encoding", type=str, default=cfg.DEFAULT_ENCODING)
    p.add_argument(
        "--dry-run", action="store_true",
        help="Report the plan and balance without writing block CSVs.",
    )
    args = p.parse_args()

    if args.block_size < 2:
        p.error("--block-size must be >= 2")
    if args.blocks_per_rater < 1:
        p.error("--blocks-per-rater must be >= 1")
    if args.raters_per_item < 1:
        p.error("--raters-per-item must be >= 1")

    stem = cfg.csv_stem(args.task)
    out_dir = args.run_dir / cfg.HUMAN_SUBDIR
    master = out_dir / f"{stem}.csv"

    items = load_items(master, args.encoding)
    print(f"\n[load] {len(items)} items from {master.name}")
    words = sorted(it["n_words"] for it in items)
    print(f"[load] words/item: min {words[0]}, median {words[len(words)//2]}, "
          f"p90 {words[int(.9*len(words))]}, max {words[-1]}")

    blocks = build_blocks(
        items,
        block_size=args.block_size,
        balance_keys=cfg.BLOCK_BALANCE_KEYS,
        seed=args.seed,
    )
    order_within_block(blocks, mode=args.item_order, seed=args.seed)
    assignment = assign_blocks_to_raters(
        len(blocks),
        raters_per_item=args.raters_per_item,
        blocks_per_rater=args.blocks_per_rater,
    )

    report = _balance_report(blocks)

    # Report the typical rater and the range. Quoting the slowest block as
    # "the" session length overstates it — with banded blocks the spread is
    # a couple of minutes, which is the whole point of balancing on words.
    def _total(session_min: float) -> float:
        return (cfg.ONBOARDING_MINUTES
                + args.blocks_per_rater * session_min
                + (args.blocks_per_rater - 1) * cfg.BREAK_MINUTES)

    session_means = [estimate_session_minutes(b) for b in blocks]
    per_session = sum(session_means) / len(session_means)
    total_min = _total(per_session)
    total_lo = _total(report["session_minutes_min"])
    total_hi = _total(report["session_minutes_max"])

    print(f"\n[design] {len(blocks)} blocks of {args.block_size}")
    print(f"[design] {len(assignment)} raters x {args.blocks_per_rater} "
          f"blocks = {args.block_size * args.blocks_per_rater} items each")
    print(f"[design] {args.raters_per_item} ratings per item "
          f"({len(items) * args.raters_per_item} total)")
    print(f"\n[balance] block words   : {report['block_words_min']}-"
          f"{report['block_words_max']} (mean {report['block_words_mean']}, "
          f"CV {report['block_words_cv']:.1%})")
    print(f"[balance] session length : {report['session_minutes_min']}-"
          f"{report['session_minutes_max']} min")
    print(f"[balance] long-text/block: {report['long_text_per_block_min']}-"
          f"{report['long_text_per_block_max']} of {args.block_size}")
    for key in cfg.BLOCK_BALANCE_KEYS:
        print(f"[balance] max same {key:14s}: "
              f"{report[f'max_same_{key}_in_a_block']} of {args.block_size}")
    print(f"\n[time] onboarding {cfg.ONBOARDING_MINUTES:.0f} + "
          f"{args.blocks_per_rater} sessions x {per_session:.1f} min + "
          f"{args.blocks_per_rater - 1} x {cfg.BREAK_MINUTES} min break")
    print(f"[time] ~{total_min:.0f} min per rater "
          f"(range {total_lo:.0f}-{total_hi:.0f} across blocks)")

    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return

    # ── Write ───────────────────────────────────────────────────────────
    blocks_dir = out_dir / f"{stem.replace('_items', '')}_blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    for old in blocks_dir.glob("block_*.csv"):
        old.unlink()

    qualtrics_cols = list(cfg.COLUMNS)
    for b_id, block in enumerate(blocks, 1):
        _write_csv(blocks_dir / f"block_{b_id:03d}.csv", block,
                   qualtrics_cols, args.encoding)

    # Master with block_id attached, for the join back from Qualtrics.
    block_of = {it["session_id"]: b_id
                for b_id, block in enumerate(blocks, 1) for it in block}
    pos_of = {it["session_id"]: i
              for block in blocks for i, it in enumerate(block, 1)}
    for it in items:
        it["block_id"] = block_of[it["session_id"]]
        it["position_in_block"] = pos_of[it["session_id"]]
    _write_csv(out_dir / f"{stem}_with_blocks.csv",
               sorted(items, key=lambda it: (it["block_id"],
                                             it["position_in_block"])),
               qualtrics_cols + ["block_id", "position_in_block", "n_words"],
               args.encoding)

    # Fielding plan: rater -> session -> block.
    plan_rows = [
        {"rater_id": f"R{rater + 1:03d}", "session": s, "block_id": b_id + 1,
         "n_items": len(blocks[b_id]),
         "est_minutes": round(estimate_session_minutes(blocks[b_id]), 1)}
        for rater, block_ids in sorted(assignment.items())
        for s, b_id in enumerate(block_ids, 1)
    ]
    _write_csv(out_dir / f"{stem.replace('_items', '')}_rater_assignment.csv",
               plan_rows,
               ["rater_id", "session", "block_id", "n_items", "est_minutes"],
               args.encoding)

    manifest = {
        "task": args.task,
        "builder": "human_evaluation/make_rater_blocks",
        "master_csv": master.name,
        "n_items": len(items),
        "n_blocks": len(blocks),
        "block_size": args.block_size,
        "blocks_per_rater": args.blocks_per_rater,
        "raters_per_item": args.raters_per_item,
        "n_raters": len(assignment),
        "items_per_rater": args.block_size * args.blocks_per_rater,
        "total_ratings": len(items) * args.raters_per_item,
        "sessions_per_rater": args.blocks_per_rater,
        "break_minutes": cfg.BREAK_MINUTES,
        "onboarding_minutes": cfg.ONBOARDING_MINUTES,
        "est_minutes_per_rater": round(total_min, 1),
        "est_minutes_per_rater_range": [round(total_lo, 1), round(total_hi, 1)],
        "est_minutes_per_session": round(per_session, 1),
        "block_balance_keys": list(cfg.BLOCK_BALANCE_KEYS),
        "item_order": args.item_order,
        "seed": args.seed,
        "balance_report": report,
        "assignment": {f"R{r + 1:03d}": [b + 1 for b in bs]
                       for r, bs in sorted(assignment.items())},
        "created_at": datetime.now().isoformat(),
    }
    manifest_path = out_dir / f"{stem.replace('_items', '')}_blocks_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print(f"\n[write] {blocks_dir.name}/block_001..{len(blocks):03d}.csv")
    print(f"[write] {stem}_with_blocks.csv")
    print(f"[write] {stem.replace('_items', '')}_rater_assignment.csv "
          f"({len(plan_rows)} rater-sessions)")
    print(f"[write] {manifest_path.name}")


if __name__ == "__main__":
    main()
