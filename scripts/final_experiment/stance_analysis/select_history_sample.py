"""
stance_analysis/select_history_sample.py
=========================================
Select the sample for the *history-aware* stance side analysis
(misinfo_stance_history_split — see core/scoring.py).

Unlike select_target_conversations.py, which filters to a specific
"started strong, went silent" subset (turn-1 correction|rebuttal == 3
AND turn-8 correction|rebuttal == 1), this samples directly from the
FULL population of main-study sessions — no correction/rebuttal score
filter at all. Flat stratified 12% per (target_model, iv1,
belief_category) cell, all 4 models, no cap — same randomized-round
algorithm as select_target_conversations.stratified_sample_session_ids
(reused directly here) and response_diversity/sampling.py.

Usage::

    cd scripts/final_experiment
    python -m stance_analysis.select_history_sample
    python -m stance_analysis.select_history_sample --sample-fraction 0.2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from . import config as cfg  # noqa: E402
from .select_target_conversations import (  # noqa: E402
    STRATA_COLS,
    build_target_index,
    stratified_sample_session_ids,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stratified-sample the full session population for "
                    "history-aware stance scoring (no score filter)."
    )
    p.add_argument("--sample-fraction", type=float, default=cfg.HISTORY_SAMPLE_FRACTION)
    p.add_argument("--sampling-seed", type=int, default=cfg.HISTORY_SAMPLING_SEED)
    args = p.parse_args()

    if not cfg.TURN_LEVEL_CSV.exists():
        raise SystemExit(f"turn_level.csv not found: {cfg.TURN_LEVEL_CSV}")

    turn_level = pd.read_csv(cfg.TURN_LEVEL_CSV)
    by_session = turn_level.drop_duplicates("session_id")
    n_total = len(by_session)

    print(f"Loaded {n_total:,} sessions from {cfg.TURN_LEVEL_CSV} "
          f"(full population — no correction/rebuttal score filter)")
    print("\nBy target_model:")
    print(by_session.groupby("target_model").size().to_string())
    print("\nBy iv1:")
    print(by_session.groupby("iv1").size().to_string())

    sampled_ids, cell_stats = stratified_sample_session_ids(
        by_session,
        group_cols=STRATA_COLS,
        fraction=args.sample_fraction,
        seed=args.sampling_seed,
    )
    print(
        f"\nStratified sample @ {args.sample_fraction:.0%} "
        f"(seed={args.sampling_seed}, strata={STRATA_COLS}): "
        f"{len(sampled_ids):,} / {n_total:,} sessions"
    )
    sampled_by_session = by_session[by_session["session_id"].isin(sampled_ids)]
    print("\nSampled — by target_model:")
    print(sampled_by_session.groupby("target_model").size().to_string())

    target_sampled = turn_level[turn_level["session_id"].isin(sampled_ids)]
    entries = build_target_index(target_sampled)

    cfg.HISTORY_SAMPLE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": {
            "score_filter":          None,
            "source_turn_level_csv": str(cfg.TURN_LEVEL_CSV),
            "n_population":          n_total,
            "sample_fraction":       args.sample_fraction,
            "sampling_seed":         args.sampling_seed,
            "strata_cols":           STRATA_COLS,
        },
        "cell_stats": cell_stats,
        "n_sessions": len(entries),
        "entries": entries,
    }
    with cfg.HISTORY_SAMPLE_INDEX_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote history-aware sample index ({len(entries)} sessions) "
          f"→ {cfg.HISTORY_SAMPLE_INDEX_PATH}")


if __name__ == "__main__":
    main()
