"""
mitigation_plan/select_sample.py
=================================
Select a matched subsample of existing main_user_IVs sessions to re-run
under the mitigation system prompt.

Stratified 5% sample (by cell_id x target_model, randomized-round quota
— see sampling.py) of every completed session in
config.SOURCE_CONV_DIR. Because run_conversations.py re-uses each
selected session's belief + IV condition + target model unchanged (only
the target's system prompt differs), the result is a paired before/after
design: downstream analysis can diff a mitigation session's scores
against its source session's already-scored misinfo_v1 / stance scores
by stripping config.SESSION_ID_SUFFIX from the session_id.

Usage::

    cd scripts/final_experiment
    python -m mitigation_plan.select_sample
    python -m mitigation_plan.select_sample --sample-fraction 0.1
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from . import config as cfg  # noqa: E402
from . import sampling  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-fraction", type=float, default=cfg.SAMPLE_FRACTION)
    p.add_argument("--sampling-seed", type=int, default=cfg.SAMPLING_SEED)
    args = p.parse_args()

    print(f"\nSource conversations: {cfg.SOURCE_CONV_DIR}")
    entries, cell_stats = sampling.build_matched_sample(
        cfg.SOURCE_CONV_DIR,
        sample_fraction=args.sample_fraction,
        seed=args.sampling_seed,
    )

    n_pool = sum(c["cell_size"] for c in cell_stats)
    print(
        f"Stratified sample @ {args.sample_fraction:.0%} "
        f"(seed={args.sampling_seed}, strata={sampling.STRATA_COLS}): "
        f"{len(entries):,} / {n_pool:,} sessions"
    )

    by_cell = collections.Counter(e["cell_id"] for e in entries)
    by_model = collections.Counter(e["target_model"] for e in entries)
    print("\nBy cell:")
    for k in sorted(by_cell):
        print(f"  {k:<28} {by_cell[k]}")
    print("\nBy target model:")
    for k in sorted(by_model):
        print(f"  {k:<24} {by_model[k]}")

    sampling.write_sample_index(
        cfg.SAMPLE_INDEX_PATH,
        entries=entries,
        cell_stats=cell_stats,
        params={
            "source_conv_dir": str(cfg.SOURCE_CONV_DIR),
            "sample_fraction": args.sample_fraction,
            "sampling_seed": args.sampling_seed,
            "strata_cols": list(sampling.STRATA_COLS),
        },
    )
    print(f"\nWrote sample index ({len(entries)} sessions) -> {cfg.SAMPLE_INDEX_PATH}")


if __name__ == "__main__":
    main()
