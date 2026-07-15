"""
stance_analysis/select_target_conversations.py
================================================
Select the target conversations for the stance side analysis: sessions
that started with a strong correction or rebuttal at turn 1 but had
dropped to *no* correction/rebuttal by turn 8.

Filter (see config.py for the thresholds)::

    (correction[turn=1] == 3 or rebuttal[turn=1] == 3)
    and
    (correction[turn=8] == 1 or rebuttal[turn=8] == 1)

Reads main_user_IVs' turn_level.csv (already-scored misinfo_v1
correction/rebuttal per turn) rather than the raw score JSON — it is
the single already-materialized source for those two dimensions across
the whole study.

Because stance scoring costs one evaluator call per turn per session,
the matched sessions are then stratified-sampled down to
``--sample-fraction`` (default config.SAMPLE_FRACTION = 12%) per
(target_model, iv1, belief_category) cell — same randomized-round
algorithm as response_diversity/sampling.py, so E[sampled] == matched
* fraction per cell. Pass ``--no-sample`` to write out every matched
session instead.

Claude has by far the largest matched pool (it disengages — ends the
conversation rather than continuing to correct/rebut — more often than
the other models), so its 12% stratified quota would dominate the
stance-scored set. After stratified sampling, Claude's sampled sessions
are therefore further capped down to the smallest *other* model's
sampled count (config.CAP_MODELS, default ``("claude-sonnet-4.6",)``);
the other models keep their full 12% quota. Pass ``--no-equalize`` to
skip the cap, or ``--cap-models`` to cap a different subset.

Usage::

    cd scripts/final_experiment
    python -m stance_analysis.select_target_conversations
    python -m stance_analysis.select_target_conversations --sample-fraction 0.2
    python -m stance_analysis.select_target_conversations --no-sample
    python -m stance_analysis.select_target_conversations --no-equalize
    python -m stance_analysis.select_target_conversations --cap-models claude-sonnet-4.6 deepseek-v3.2

Topping up an existing selection
---------------------------------
``--top-up-to N`` grows an *existing* target index to N sessions per
target_model, **preserving every already-selected session** (so
already-scored sessions in scores/ stay valid and re-running
run_stance_scoring.py only pays for the newly-added ones — it never
rescores anything). For each model with fewer than N sessions
selected, additional sessions are drawn uniformly at random (seeded,
no replacement) from that model's matched pool, excluding whatever is
already selected. Models already at or above N are left untouched (no
sessions are ever removed in this mode). Ignores --sample-fraction /
--no-sample / --cap-models / --no-equalize::

    python -m stance_analysis.select_target_conversations --top-up-to 73
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from . import config as cfg  # noqa: E402

META_COLS = [
    "session_id", "target_model", "iv1", "iv2",
    "category", "subtype", "is_control",
]

STRATA_COLS = ["target_model", "iv1", "category"]  # belief_category


def select_target_sessions(turn_level: pd.DataFrame) -> pd.DataFrame:
    """Return the full turn_level rows (all turns) for target sessions."""
    t1 = turn_level[turn_level["turn"] == cfg.FIRST_TURN].set_index("session_id")
    t8 = turn_level[turn_level["turn"] == cfg.LAST_TURN].set_index("session_id")
    common = t1.index.intersection(t8.index)
    t1, t8 = t1.loc[common], t8.loc[common]

    cond_high = (t1["correction"] == cfg.HIGH_SCORE) | (t1["rebuttal"] == cfg.HIGH_SCORE)
    cond_low = (t8["correction"] == cfg.LOW_SCORE) | (t8["rebuttal"] == cfg.LOW_SCORE)
    target_ids = common[(cond_high & cond_low).to_numpy()]

    return turn_level[turn_level["session_id"].isin(target_ids)].copy()


def stratified_sample_session_ids(
    by_session: pd.DataFrame,
    *,
    group_cols: list[str],
    fraction: float,
    seed: int,
) -> tuple[list[str], list[dict]]:
    """Stratified sample of session_ids, quota = randomized_round(cell_size * fraction).

    randomized_round(x) = floor(x)+1 with probability equal to the
    fractional part of x, else floor(x) — deterministic given `seed`,
    with E[quota] == cell_size * fraction. Same algorithm as
    response_diversity/sampling.py.
    """
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"fraction must be in [0, 1], got {fraction!r}")

    rng = random.Random(seed)
    picked: list[str] = []
    cell_stats: list[dict] = []

    for key, grp in by_session.groupby(group_cols, sort=True):
        pool = grp["session_id"].tolist()
        cell_size = len(pool)

        expected = cell_size * fraction
        quota = int(expected)
        if rng.random() < (expected - quota):
            quota += 1
        quota = min(quota, cell_size)
        if quota <= 0:
            continue

        rng.shuffle(pool)
        picked.extend(pool[:quota])
        cell_stats.append({
            **dict(zip(group_cols, key if isinstance(key, tuple) else (key,))),
            "cell_size": cell_size,
            "quota": quota,
        })

    return picked, cell_stats


def cap_models_to_min(
    sampled_ids: list[str],
    by_session: pd.DataFrame,
    *,
    model_col: str,
    models_to_cap: tuple[str, ...],
    seed: int,
) -> tuple[list[str], int, list[dict]]:
    """Downsample only `models_to_cap` to the smallest *uncapped* model's count.

    Every model not in `models_to_cap` keeps its full stratified quota
    unchanged. Models in `models_to_cap` (default: just Claude — it
    disengages, ending the conversation, far more often than the other
    models, so its matched pool and thus its 12% quota is much bigger)
    are randomly downsampled to match. Downsampling is a plain random
    draw per model (post stratified-sampling, so within-model
    iv1/belief_category balance is only approximately preserved).
    """
    id_to_model = dict(zip(by_session["session_id"], by_session[model_col]))
    by_model: dict[str, list[str]] = defaultdict(list)
    for sid in sampled_ids:
        by_model[id_to_model[sid]].append(sid)

    uncapped_counts = [
        len(ids) for model, ids in by_model.items() if model not in models_to_cap
    ]
    if not uncapped_counts:
        raise ValueError(
            "cap_models_to_min: every sampled model is in models_to_cap — "
            "no uncapped model left to size the cap from."
        )
    min_count = min(uncapped_counts)

    rng = random.Random(seed)
    capped: list[str] = []
    cap_stats: list[dict] = []
    for model in sorted(by_model):
        ids = sorted(by_model[model])
        if model in models_to_cap and len(ids) > min_count:
            rng.shuffle(ids)
            kept = ids[:min_count]
        else:
            kept = ids
        capped.extend(kept)
        cap_stats.append({
            "target_model":  model,
            "capped":        model in models_to_cap,
            "before_equalize": len(ids),
            "after_equalize":  len(kept),
        })

    return capped, min_count, cap_stats


def top_up_to_count(
    existing_ids: set[str],
    by_session: pd.DataFrame,
    *,
    model_col: str,
    target_n: int,
    seed: int,
) -> tuple[list[str], list[dict]]:
    """Grow `existing_ids` so every model has >= target_n sessions selected.

    Never removes anything: a model already at or above target_n is left
    as-is. For a model below target_n, additional sessions are drawn
    uniformly at random (no replacement) from that model's matched pool
    minus whatever's already selected — uniform sampling over individual
    sessions distributes proportionally across iv1/belief_category cells
    in expectation, same effect as explicit stratification without the
    apportionment-rounding complexity.
    """
    rng = random.Random(seed)
    by_model: dict[str, list[str]] = defaultdict(list)
    for row in by_session.itertuples(index=False):
        by_model[getattr(row, model_col)].append(row.session_id)

    final_ids = set(existing_ids)
    topup_stats: list[dict] = []

    for model in sorted(by_model):
        pool = set(by_model[model])
        existing_for_model = pool & existing_ids
        n_before = len(existing_for_model)
        deficit = target_n - n_before

        candidates = sorted(pool - existing_ids)
        n_added = 0
        if deficit > 0 and candidates:
            n_to_pick = min(deficit, len(candidates))
            picks = rng.sample(candidates, k=n_to_pick)
            final_ids.update(picks)
            n_added = n_to_pick

        topup_stats.append({
            "target_model": model,
            "pool_size":    len(pool),
            "before":       n_before,
            "added":        n_added,
            "after":        n_before + n_added,
            "capped_by_pool": deficit > 0 and n_added < deficit,
        })

    return sorted(final_ids), topup_stats


def build_target_index(turn_level_target: pd.DataFrame) -> list[dict]:
    """One entry per session with metadata + conversation source path."""
    per_session = (
        turn_level_target[META_COLS]
        .drop_duplicates(subset="session_id")
        .sort_values("session_id")
    )
    entries = []
    for row in per_session.itertuples(index=False):
        entries.append({
            "session_id":      row.session_id,
            "source_path":     str(cfg.SOURCE_CONV_DIR / f"{row.session_id}.json"),
            "target_model":    row.target_model,
            "iv1":             row.iv1,
            "iv2":             row.iv2,
            "belief_category": row.category,
            "belief_subtype":  row.subtype,
            "is_control":      bool(row.is_control),
        })
    return entries


def main() -> None:
    p = argparse.ArgumentParser(
        description="Select (and stratified-sample) target sessions for stance scoring."
    )
    p.add_argument("--sample-fraction", type=float, default=cfg.SAMPLE_FRACTION)
    p.add_argument("--sampling-seed", type=int, default=cfg.SAMPLING_SEED)
    p.add_argument(
        "--no-sample", action="store_true",
        help="Write out every matched session instead of a stratified sample.",
    )
    p.add_argument(
        "--no-equalize", action="store_true",
        help="Skip the post-sampling cap "
             "(keeps every model's full stratified quota, incl. Claude's).",
    )
    p.add_argument(
        "--cap-models", type=str, nargs="+", default=list(cfg.CAP_MODELS),
        help="target_model value(s) to downsample to the smallest "
             "*uncapped* model's sampled count. Default: config.CAP_MODELS.",
    )
    p.add_argument(
        "--top-up-to", type=int, default=None,
        help="Grow the EXISTING target index so every target_model has at "
             "least this many sessions selected, preserving every "
             "already-selected (and possibly already-scored) session. "
             "Ignores --sample-fraction/--no-sample/--cap-models/--no-equalize.",
    )
    args = p.parse_args()

    if not cfg.TURN_LEVEL_CSV.exists():
        raise SystemExit(f"turn_level.csv not found: {cfg.TURN_LEVEL_CSV}")

    turn_level = pd.read_csv(cfg.TURN_LEVEL_CSV)
    target = select_target_sessions(turn_level)
    by_session = target.drop_duplicates("session_id")
    n_matched = len(by_session)

    print(f"Loaded {turn_level['session_id'].nunique():,} sessions from {cfg.TURN_LEVEL_CSV}")
    print(
        f"Matched sessions "
        f"(turn{cfg.FIRST_TURN} correction|rebuttal == {cfg.HIGH_SCORE:g}  AND  "
        f"turn{cfg.LAST_TURN} correction|rebuttal == {cfg.LOW_SCORE:g}): "
        f"{n_matched:,}"
    )
    print("\nBy target_model:")
    print(by_session.groupby("target_model").size().to_string())
    print("\nBy iv1:")
    print(by_session.groupby("iv1").size().to_string())

    if args.top_up_to is not None:
        if not cfg.TARGET_INDEX_PATH.exists():
            raise SystemExit(
                f"--top-up-to requires an existing target index: "
                f"{cfg.TARGET_INDEX_PATH} not found. Run a normal "
                f"(non-top-up) selection first."
            )
        with cfg.TARGET_INDEX_PATH.open("r", encoding="utf-8") as fh:
            existing_payload = json.load(fh)
        existing_ids = {e["session_id"] for e in existing_payload["entries"]}
        print(f"\nExisting target index: {len(existing_ids)} sessions "
              f"(preserved — none will be removed)")

        final_ids, topup_stats = top_up_to_count(
            existing_ids, by_session,
            model_col="target_model", target_n=args.top_up_to,
            seed=args.sampling_seed,
        )
        print(f"\nTop-up to {args.top_up_to}/model (seed={args.sampling_seed}):")
        for row in topup_stats:
            flag = "  ** capped by matched pool size **" if row["capped_by_pool"] else ""
            print(f"  {row['target_model']:<24} pool={row['pool_size']:<5} "
                  f"before={row['before']:<4} +{row['added']:<4} -> {row['after']}{flag}")

        target_final = target[target["session_id"].isin(final_ids)]
        entries = build_target_index(target_final)

        cfg.TARGET_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "params": {
                **existing_payload.get("params", {}),
                "topped_up_to":   args.top_up_to,
                "topup_seed":     args.sampling_seed,
            },
            "cell_stats":    existing_payload.get("cell_stats", []),
            "equalize_stats": existing_payload.get("equalize_stats", []),
            "topup_stats":   topup_stats,
            "n_sessions":    len(entries),
            "entries":       entries,
        }
        with cfg.TARGET_INDEX_PATH.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"\nWrote target index ({len(entries)} sessions, "
              f"{len(entries) - len(existing_ids)} new) → {cfg.TARGET_INDEX_PATH}")
        return

    equalize_min_count: int | None = None
    equalize_stats: list[dict] = []

    if args.no_sample:
        sampled_ids = by_session["session_id"].tolist()
        cell_stats: list[dict] = []
        print(f"\n--no-sample: writing all {len(sampled_ids):,} matched sessions.")
    else:
        sampled_ids, cell_stats = stratified_sample_session_ids(
            by_session,
            group_cols=STRATA_COLS,
            fraction=args.sample_fraction,
            seed=args.sampling_seed,
        )
        print(
            f"\nStratified sample @ {args.sample_fraction:.0%} "
            f"(seed={args.sampling_seed}, strata={STRATA_COLS}): "
            f"{len(sampled_ids):,} / {n_matched:,} sessions"
        )
        sampled_by_session = by_session[by_session["session_id"].isin(sampled_ids)]
        print("\nSampled — by target_model:")
        print(sampled_by_session.groupby("target_model").size().to_string())

        if not args.no_equalize:
            sampled_ids, equalize_min_count, equalize_stats = cap_models_to_min(
                sampled_ids, by_session,
                model_col="target_model",
                models_to_cap=tuple(args.cap_models),
                seed=args.sampling_seed,
            )
            print(
                f"\nCapped {args.cap_models} to {equalize_min_count} "
                f"(smallest uncapped model's sampled count); "
                f"other models keep their full quota: {len(sampled_ids):,} sessions"
            )
            equalized_by_session = by_session[by_session["session_id"].isin(sampled_ids)]
            print("\nAfter cap — by target_model:")
            print(equalized_by_session.groupby("target_model").size().to_string())

    target_sampled = target[target["session_id"].isin(sampled_ids)]
    entries = build_target_index(target_sampled)

    cfg.TARGET_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": {
            "first_turn":            cfg.FIRST_TURN,
            "last_turn":             cfg.LAST_TURN,
            "high_score":            cfg.HIGH_SCORE,
            "low_score":             cfg.LOW_SCORE,
            "source_turn_level_csv": str(cfg.TURN_LEVEL_CSV),
            "n_matched":             n_matched,
            "sampled":               not args.no_sample,
            "sample_fraction":       None if args.no_sample else args.sample_fraction,
            "sampling_seed":         None if args.no_sample else args.sampling_seed,
            "strata_cols":           STRATA_COLS,
            "capped":                not args.no_sample and not args.no_equalize,
            "cap_models":            None if args.no_sample or args.no_equalize else args.cap_models,
            "cap_count":             equalize_min_count,
        },
        "cell_stats": cell_stats,
        "equalize_stats": equalize_stats,
        "n_sessions": len(entries),
        "entries": entries,
    }
    with cfg.TARGET_INDEX_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote target index ({len(entries)} sessions) → {cfg.TARGET_INDEX_PATH}")


if __name__ == "__main__":
    main()
