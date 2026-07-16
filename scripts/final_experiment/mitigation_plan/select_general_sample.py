"""
mitigation_plan/select_general_sample.py
==========================================
General (non-failure-mode-targeted) pilot sample: N sessions per target
model, drawn uniformly at random from stance_analysis's existing
history-sample pool (config.HISTORY_SAMPLE_INDEX_PATH — a flat
stratified 12% of the full population, ~170 sessions/model across all 4
target models, no correction/rebuttal score filter — see
stance_analysis/select_history_sample.py).

Unlike select_failure_mode_pilot.py, this does NOT target disengagement
or affective-validation sessions specifically. It exists to check
whether select_failure_mode_pilot.py's weak/mixed effect (correction and
rebuttal only nudging up, agreeableness only nudging down — especially
for DeepSeek) was an artifact of cherry-picking already-broken sessions,
or holds on an ordinary representative sample too.

--reuse-index / --reuse-models
-------------------------------
For a model whose "failure mode" pool covers most of its population
anyway (Claude: 1144/1425 = 80% of sessions match the disengagement
filter — it isn't a narrow edge case, it's close to Claude's typical
behavior in this study), it's wasteful to generate a second, mostly-
overlapping-in-character set of mitigated sessions. Pass
--reuse-index <failure_mode_sample_index.json> --reuse-models
claude-sonnet-4.6 to source up to --n-per-model of that model's entries
from the already-generated failure-mode sample instead of drawing fresh
ones. If the reuse pool has fewer than --n-per-model entries for that
model, the remainder is topped up with a fresh draw from
--source-index (excluding anything already reused) — e.g. going from
35 to 70 per model reuses all 35 already-generated Claude sessions
(zero new generation) and draws 35 more from the unfiltered pool, so
the combined 70 isn't 100% disengagement-pattern like a pure reuse
would be. Every entry keeps a "source" field recording where it
actually came from (reused vs. freshly drawn) and is written with
failure_mode="general" (not "disengagement") since it's standing in for
the unfiltered population here — this substitution stays auditable
rather than silently blurring the two samples together.

Usage::

    cd scripts/final_experiment
    python -m mitigation_plan.select_general_sample
    python -m mitigation_plan.select_general_sample --n-per-model 35 --seed 45
    python -m mitigation_plan.select_general_sample \\
        --reuse-index results/final_experiment/main_user_IVs/20260427_165233/mitigation_sample_index_failure_modes.json \\
        --reuse-models claude-sonnet-4.6
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from . import config as cfg  # noqa: E402
from . import sampling  # noqa: E402


def _entry_target_model(entry: dict) -> str:
    """Short slug matching the session_id's `__model-{slug}` suffix
    (e.g. "gemini-3-flash", not the turn_level.csv-style
    "gemini-3-flash-preview") — same convention as sampling.py."""
    sid = entry["session_id"]
    return sid.rsplit("__model-", 1)[1] if "__model-" in sid else entry["target_model"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-per-model", type=int, default=35)
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--source-index", type=Path, default=cfg.HISTORY_SAMPLE_INDEX_PATH)
    p.add_argument(
        "--reuse-index", type=Path, default=None,
        help="An existing sample index (e.g. an already-generated "
             "failure-mode sample) whose entries should be reused verbatim "
             "for --reuse-models instead of drawing fresh ones.",
    )
    p.add_argument(
        "--reuse-models", nargs="+", default=[],
        help="target_model slug(s) to source from --reuse-index instead of "
             "--source-index.",
    )
    args = p.parse_args()

    if not args.source_index.exists():
        raise SystemExit(
            f"Source sample index not found: {args.source_index}\n"
            "Run `python -m stance_analysis.select_history_sample` first."
        )

    with args.source_index.open("r", encoding="utf-8") as fh:
        pool_entries = json.load(fh)["entries"]

    by_model: dict[str, list[dict]] = defaultdict(list)
    for e in pool_entries:
        by_model[_entry_target_model(e)].append(e)

    print(f"Source pool: {args.source_index}")
    for model in sorted(by_model):
        print(f"  {model:<24} pool={len(by_model[model])}")

    reuse_by_model: dict[str, list[dict]] = defaultdict(list)
    if args.reuse_index is not None:
        if not args.reuse_index.exists():
            raise SystemExit(f"--reuse-index not found: {args.reuse_index}")
        with args.reuse_index.open("r", encoding="utf-8") as fh:
            reuse_entries = json.load(fh)["entries"]
        for e in reuse_entries:
            reuse_by_model[_entry_target_model(e)].append(e)
        print(f"\nReuse pool ({args.reuse_index.name}), for {args.reuse_models}:")
        for model in args.reuse_models:
            print(f"  {model:<24} pool={len(reuse_by_model.get(model, []))}")

    rng = random.Random(args.seed)
    entries = []
    cell_stats = []
    for model in sorted(by_model):
        picked_with_source: list[tuple[dict, str]] = []

        if model in args.reuse_models:
            candidates = reuse_by_model.get(model, [])
            n_reuse = min(args.n_per_model, len(candidates))
            reused = candidates if n_reuse == len(candidates) else rng.sample(candidates, n_reuse)
            picked_with_source += [(e, f"reused from {args.reuse_index.name}") for e in reused]

            n_remaining = args.n_per_model - n_reuse
            if n_remaining > 0:
                reused_ids = {e["session_id"] for e in reused}
                topup_pool = [e for e in by_model[model] if e["session_id"] not in reused_ids]
                n_topup = min(n_remaining, len(topup_pool))
                topped_up = rng.sample(topup_pool, n_topup)
                picked_with_source += [
                    (e, f"topped up from {args.source_index.name}") for e in topped_up
                ]
        else:
            pool = by_model[model]
            n = min(args.n_per_model, len(pool))
            picked_with_source = [
                (e, f"drawn from {args.source_index.name}") for e in rng.sample(pool, n)
            ]

        for e, source in picked_with_source:
            entries.append({
                "session_id":      e["session_id"],
                "source_path":     e["source_path"],
                "cell_id":         f"iv1-{e['iv1']}__iv2-{e['iv2']}",
                "iv1":             e["iv1"],
                "iv2":             e["iv2"],
                "belief_category": e["belief_category"],
                "belief_subtype":  e.get("belief_subtype", "unknown"),
                "target_model":    model,
                "failure_mode":    "general",
                "source":          source,
            })

        by_source = defaultdict(int)
        for _, source in picked_with_source:
            by_source[source] += 1
        cell_stats.append({
            "target_model": model,
            "n_sampled": len(picked_with_source),
            "by_source": dict(by_source),
        })

    entries.sort(key=lambda e: e["session_id"])

    print(f"\nSampled (seed={args.seed}, n_per_model={args.n_per_model}): {len(entries)} sessions total")
    for stat in cell_stats:
        breakdown = ", ".join(f"{v} {k}" for k, v in stat["by_source"].items())
        print(f"  {stat['target_model']:<24} {stat['n_sampled']}  [{breakdown}]")

    sampling.write_sample_index(
        cfg.GENERAL_SAMPLE_INDEX_PATH,
        entries=entries,
        cell_stats=cell_stats,
        params={
            "source_index": str(args.source_index),
            "reuse_index": str(args.reuse_index) if args.reuse_index else None,
            "reuse_models": args.reuse_models,
            "n_per_model": args.n_per_model,
            "seed": args.seed,
            "note": "uniform per-model draw from the stance-analysis history "
                    "sample pool, no failure-mode targeting, except "
                    "--reuse-models which reuse an already-generated sample",
        },
    )
    print(f"\nWrote general sample index ({len(entries)} sessions) "
          f"-> {cfg.GENERAL_SAMPLE_INDEX_PATH}")


if __name__ == "__main__":
    main()
