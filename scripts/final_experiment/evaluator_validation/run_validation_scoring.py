"""
evaluator_validation/run_validation_scoring.py
==============================================
RQ3 — Evaluator validation pipeline (sampling + secondary scoring).

Phase 1 — Stratified sample (beliefs, not conversations)
--------------------------------------------------------
The sampling unit is the **belief**, not the conversation. We load
``data/dataset/sampled_beliefs.json`` (the file produced by
sample_beliefs.py — already stratified once at dataset-construction
time), group beliefs by (category, subtype), and within each stratum
draw ``ceil(SAMPLE_FRACTION * n_stratum)`` beliefs with a fixed seed
(floor: MIN_PER_STRATUM, so rare subtypes are never dropped).

For each sampled belief, **all** its conversations are added to the
validation set: every (cell × target_model) combination that ran
against that belief. This guarantees full cell × model coverage per
belief, which is what the per-subgroup agreement breakdowns in
analyze_agreement.py need.

With 285 beliefs, 10% gives ~29 beliefs. At 5 cells × 4 models per
belief that's ~580 conversations, ~4640 turn-level scores — the
"≈570 conversations / 4560 responses" budget in experiment-doc.md.

The manifest at ``<run_dir>/evaluator_validation/sample_manifest.json``
records the sampled belief identifiers (the primary sampling unit) and
the resolved session_ids (the unit downstream scoring operates on),
plus the seed so the draw is reproducible.

Phase 2 — Secondary scoring
---------------------------
Score every sampled conversation with the SECONDARY evaluator
(default: ``google/gemini-3.1-flash-lite-preview``) using the same
``misinfo_v1_split`` rubric the primary evaluator used. Score files are
written with ``--suffix-evaluator`` semantics so they coexist with the
primary score files in ``<run_dir>/scores/``::

    <sid>__misinfo_v1_split.json
        ↑ primary (already on disk; not touched)
    <sid>__misinfo_v1_split__<secondary-slug>.json
        ↑ produced by this script

Re-runs are safe: any session whose secondary score file already exists
is skipped. A separate scoring checkpoint is kept under
``<run_dir>/checkpoint_validation_*.json`` so this phase resumes
independently of the primary scoring checkpoint.

Agreement analysis is intentionally a separate file —
``evaluator_validation/analyze_agreement.py`` — so it can be re-run
without touching the API.

Usage
-----
::

    cd scripts/final_experiment

    # Sample (if not yet done) + score with the default secondary evaluator
    python -m evaluator_validation.run_validation_scoring \\
        --run-dir results/final_experiment/main_user_IVs/20260427_165233 \\
        --workers 16

    # Sample only — useful for inspecting coverage before paying for
    # the secondary evaluator's API calls.
    python -m evaluator_validation.run_validation_scoring \\
        --run-dir <run-dir> --sample-only

    # Force resampling (overwrites the manifest; rare — use a different
    # seed via --seed instead if you want a different draw).
    python -m evaluator_validation.run_validation_scoring \\
        --run-dir <run-dir> --resample

    # Use a different secondary evaluator
    python -m evaluator_validation.run_validation_scoring \\
        --run-dir <run-dir> \\
        --secondary-provider openrouter \\
        --secondary-model anthropic/claude-sonnet-4.6
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Make `core` importable when this file is run as a module from
# scripts/final_experiment/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import (  # noqa: E402
    Job,
    JobResult,
    RUBRICS,
    RunPaths,
    atomic_write_json,
    load_beliefs,
    read_conversation,
    run_jobs,
    safe_slug,
    score_conversation,
    write_score_artifact,
)

from main_user_IVs import config as main_cfg  # noqa: E402

from . import config as cfg  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(threadName)-12s | %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Subtype normalisation
# ════════════════════════════════════════════════════════════════════════════

def _normalize_subtype(raw) -> str:
    """Flatten subtype to a stable string for stratification.

    The subtype field is sometimes a plain string (``"gender"``), sometimes
    a python-list-as-string (``"['gender']"``), sometimes an actual list,
    and sometimes empty. We normalise to a sorted, comma-joined string so
    two records with the same set of subtypes always end up in the same
    stratum regardless of source format. This mirrors the convention used
    in main_user_IVs/final_experiment_analysis.py::_coerce_subtype.
    """
    if isinstance(raw, list):
        return ",".join(sorted(str(x) for x in raw))
    if isinstance(raw, str) and raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw.replace("'", '"'))
            if isinstance(parsed, list):
                return ",".join(sorted(str(x) for x in parsed))
        except json.JSONDecodeError:
            pass
        return raw.strip("[]'\" ")
    return (raw or "").strip()


# ════════════════════════════════════════════════════════════════════════════
# Phase 1 — Belief-level stratified sampling
# ════════════════════════════════════════════════════════════════════════════
#
# We sample beliefs (the unit produced by sample_beliefs.py) and then
# pull every conversation that used a sampled belief. session_ids are
# built by core.storage.build_session_id as
#
#     cell-{cell_id}__belief-{category}-{belief_index:04d}__model-{slug}
#
# so a sampled belief at (category=c, belief_index=i) maps to the set of
# session_ids whose name contains ``__belief-{c}-{i:04d}__``.

def _belief_token(category: str, belief_index: int) -> str:
    """Return the substring that uniquely identifies a belief in session_ids.

    Anchored on both sides with ``__`` so the match cannot bleed across
    the cell or model fields, and zero-padded to 4 digits to match
    ``build_session_id``.
    """
    return f"__belief-{category}-{int(belief_index):04d}__"


def _stratum_key(belief: dict, keys: tuple[str, ...]) -> tuple[str, ...]:
    """Compute the stratum tuple for one belief.

    `subtype` is normalised through ``_normalize_subtype``; everything
    else is read raw and coerced to str so missing fields land in a
    well-defined "" stratum rather than crashing.
    """
    parts: list[str] = []
    for k in keys:
        v = belief.get(k, "")
        if k == "subtype":
            v = _normalize_subtype(v)
        parts.append(str(v))
    return tuple(parts)


def _stratified_sample_beliefs(
    beliefs: list[dict],
    *,
    fraction: float,
    seed: int,
    min_per_stratum: int,
    stratification_keys: tuple[str, ...] = cfg.STRATIFICATION_KEYS,
) -> tuple[list[dict], list[dict]]:
    """Stratified belief sample by ``stratification_keys``.

    Sampling rule per stratum::

        n_sample = min(n_pop, max(min_per_stratum, ceil(fraction * n_pop)))

    Beliefs within a stratum are sorted by ``belief_index_global`` before
    sampling so the seed produces the same draw regardless of the
    iteration order in the input list.

    Returns:
        (sampled_beliefs, per_stratum_stats)

    sampled_beliefs is a list of belief dicts (same shape as the input,
    incl. belief_index and belief_index_global). per_stratum_stats is a
    list of summary dicts for the manifest.
    """
    rng = random.Random(seed)

    by_stratum: dict[tuple[str, ...], list[dict]] = {}
    for b in beliefs:
        by_stratum.setdefault(_stratum_key(b, stratification_keys), []).append(b)

    sampled: list[dict] = []
    stats: list[dict] = []
    for key in sorted(by_stratum):
        pool = sorted(by_stratum[key], key=lambda b: b["belief_index_global"])
        n_pop = len(pool)
        n_sample = max(min_per_stratum, math.ceil(fraction * n_pop))
        n_sample = min(n_sample, n_pop)
        chosen = rng.sample(pool, n_sample)
        chosen.sort(key=lambda b: b["belief_index_global"])
        sampled.extend(chosen)
        # Stratum descriptor: one column per stratification key, plus
        # the population/sample counts. This shape is what the manifest
        # and the printed table consume — kept generic so adding a key
        # to `stratification_keys` doesn't break either.
        stratum_record = dict(zip(stratification_keys, key))
        stratum_record.update({
            "n_population": n_pop,
            "n_sampled": n_sample,
            "actual_fraction": round(n_sample / n_pop, 4),
            "sampled_belief_indices": [b["belief_index"] for b in chosen],
        })
        stats.append(stratum_record)

    sampled.sort(key=lambda b: b["belief_index_global"])
    return sampled, stats


def _resolve_session_ids(
    conv_dir: Path,
    sampled_beliefs: list[dict],
    *,
    stratification_keys: tuple[str, ...] = cfg.STRATIFICATION_KEYS,
) -> tuple[list[str], list[dict]]:
    """Map sampled beliefs to existing session_ids on disk.

    For each sampled belief, find every completed conversation artifact
    whose filename contains the belief's identifying substring. Returns
    the sorted union of session_ids and a per-belief diagnostic listing
    how many conversations each belief contributed.
    """
    # Cache the on-disk artifact filenames once.
    on_disk: list[str] = []
    for path in sorted(conv_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Skipping unreadable artifact: %s", path.name)
            continue
        if not data.get("completed_at"):
            continue
        on_disk.append(path.stem)

    sampled_ids: set[str] = set()
    per_belief: list[dict] = []
    for b in sampled_beliefs:
        token = _belief_token(b["category"], b["belief_index"])
        matched = sorted(sid for sid in on_disk if token in sid)
        sampled_ids.update(matched)
        record = {
            "category": b["category"],
            "belief_index": b["belief_index"],
            "belief_index_global": b["belief_index_global"],
            "content_preview": (b.get("content", "") or "")[:120],
            "n_conversations": len(matched),
        }
        # Include subtype for traceability even when not used as a
        # stratifier — it's useful diagnostic detail when inspecting
        # the manifest later. Skip if it's already covered above.
        if "subtype" not in record:
            record["subtype"] = _normalize_subtype(b.get("subtype", ""))
        per_belief.append(record)

    return sorted(sampled_ids), per_belief


def _build_coverage_report(
    paths: RunPaths,
    sampled_ids: set[str],
) -> pd.DataFrame:
    """Cross-tab of sample counts by (cell_id, target_model).

    Diagnostic only. Because sampling happens at the belief level, the
    per-cell and per-model counts are uniform up to the structure of
    the IV grid: each sampled belief contributes exactly one
    conversation per (cell × target_model). A cell × model bucket with
    zero sampled sessions therefore means that combination simply
    wasn't run for any sampled belief — usually because the
    conversation is missing from disk (failed session, partial run).
    """
    rows: list[dict] = []
    for path in sorted(paths.conversations.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue
        if not data.get("completed_at"):
            continue
        sid = data["session_id"]
        cell_id = data.get("cell", {}).get("cell_id", "")
        target_full = data.get("models", {}).get("target_llm", "")
        rows.append({
            "session_id": sid,
            "cell_id": cell_id,
            "target_model": target_full.split("/")[-1] if target_full else "",
            "sampled": int(sid in sampled_ids),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    coverage = (
        df.groupby(["cell_id", "target_model"], as_index=False)
          .agg(n_population=("session_id", "count"),
               n_sampled=("sampled", "sum"))
    )
    coverage["actual_fraction"] = (
        coverage["n_sampled"] / coverage["n_population"]
    ).round(4)
    return coverage.sort_values(["cell_id", "target_model"]).reset_index(drop=True)


def _do_sampling(
    *,
    paths: RunPaths,
    validation_dir: Path,
    beliefs_path: Path,
    fraction: float,
    seed: int,
    resample: bool,
) -> list[str]:
    """Phase 1: produce or load the belief-level stratified sample manifest."""
    manifest_path = validation_dir / cfg.SAMPLE_MANIFEST_NAME

    if manifest_path.exists() and not resample:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        print(
            f"[sampling] reusing existing manifest: {manifest_path.name}\n"
            f"           {manifest['n_beliefs_sampled']} of "
            f"{manifest['n_beliefs_population']} beliefs, "
            f"{manifest['n_sessions_sampled']} resolved sessions "
            f"(fraction={manifest['fraction']}, seed={manifest['seed']}). "
            f"Pass --resample to regenerate."
        )
        return list(manifest["sampled_session_ids"])

    if not beliefs_path.exists():
        raise SystemExit(
            f"Beliefs file not found: {beliefs_path}. "
            "Run scripts/final_experiment/sample_beliefs.py first, "
            "or pass --beliefs-path."
        )
    beliefs = load_beliefs(beliefs_path)
    print(f"[sampling] loaded {len(beliefs)} beliefs from {beliefs_path}")

    # Sanity-check: which categories does the run dir's conversations
    # actually cover? Beliefs whose category isn't represented can't
    # contribute conversations and would just appear as zero-match
    # entries below. Drop them up front and tell the user.
    # session_id format (build_session_id):
    #     cell-...__belief-{category}-NNNN__model-...
    # Categories may contain hyphens (e.g. "fake_news"), so we anchor on
    # the trailing -NNNN__model- to peel the category off cleanly.
    categories_on_disk: set[str] = set()
    for p in paths.conversations.glob("*.json"):
        parts = p.stem.split("__belief-", 1)
        if len(parts) != 2:
            continue
        before_model = parts[1].split("__model-", 1)[0]
        # before_model = "{category}-NNNN"; rsplit on the last hyphen.
        cat, _, idx = before_model.rpartition("-")
        if cat and idx.isdigit():
            categories_on_disk.add(cat)
    if categories_on_disk:
        n_before = len(beliefs)
        beliefs = [b for b in beliefs if b["category"] in categories_on_disk]
        if len(beliefs) != n_before:
            dropped = n_before - len(beliefs)
            print(
                f"[sampling] note: dropped {dropped} belief(s) whose category "
                f"is not present in {paths.conversations.name}/ "
                f"(present categories: {sorted(categories_on_disk)})."
            )

    sampled_beliefs, strata_stats = _stratified_sample_beliefs(
        beliefs,
        fraction=fraction,
        seed=seed,
        min_per_stratum=cfg.MIN_PER_STRATUM,
    )
    print(
        f"[sampling] sampled {len(sampled_beliefs)} beliefs across "
        f"{len(strata_stats)} (category, subtype) strata "
        f"(fraction={fraction}, seed={seed})"
    )

    sampled_ids, per_belief_stats = _resolve_session_ids(
        paths.conversations, sampled_beliefs,
    )
    n_zero = sum(1 for r in per_belief_stats if r["n_conversations"] == 0)
    if n_zero:
        print(
            f"[sampling] ⚠ {n_zero} sampled belief(s) matched 0 "
            f"conversations on disk — likely failed sessions or a partial "
            f"run. They are kept in the manifest for traceability."
        )
    print(f"[sampling] resolved to {len(sampled_ids)} session_id(s) "
          f"({sum(r['n_conversations'] for r in per_belief_stats)} total "
          f"belief × cell × model combinations).")

    # Manifest. ``sampled_session_ids`` is the field analyze_agreement.py
    # consumes; the belief-level fields are kept for provenance.
    manifest = {
        "sampling_unit": "belief",
        "fraction": fraction,
        "seed": seed,
        "min_per_stratum": cfg.MIN_PER_STRATUM,
        "stratification_keys": list(cfg.STRATIFICATION_KEYS),
        "beliefs_path": str(beliefs_path),
        "n_beliefs_population": len(beliefs),
        "n_beliefs_sampled": len(sampled_beliefs),
        "n_sessions_sampled": len(sampled_ids),
        "n_strata": len(strata_stats),
        "sampled_beliefs": [
            {
                "category": b["category"],
                "subtype": _normalize_subtype(b.get("subtype", "")),
                "belief_index": b["belief_index"],
                "belief_index_global": b["belief_index_global"],
                "is_long_text": bool(b.get("is_long_text", False)),
                "content_preview": (b.get("content", "") or "")[:200],
            }
            for b in sampled_beliefs
        ],
        "per_stratum_stats": strata_stats,
        "per_belief_resolution": per_belief_stats,
        "sampled_session_ids": sampled_ids,
        "created_at": datetime.now().isoformat(),
    }
    atomic_write_json(manifest_path, manifest)
    print(f"[sampling] wrote {manifest_path.relative_to(paths.root)}")

    coverage = _build_coverage_report(paths, set(sampled_ids))
    coverage_path = validation_dir / cfg.COVERAGE_REPORT_NAME
    coverage.to_csv(coverage_path, index=False)
    print(f"[sampling] wrote {coverage_path.relative_to(paths.root)}")

    print("\n[sampling] per-stratum draw:")
    strata_df = pd.DataFrame([
        {k: v for k, v in s.items() if k != "sampled_belief_indices"}
        for s in strata_stats
    ])
    print(strata_df.to_string(index=False))

    print("\n[sampling] coverage by (cell × target_model):")
    if coverage.empty:
        print("  (no completed conversations on disk)")
    else:
        print(coverage.to_string(index=False))

    return sampled_ids


# ════════════════════════════════════════════════════════════════════════════
# Phase 2 — Secondary scoring
# ════════════════════════════════════════════════════════════════════════════

def _scoring_worker(payload: dict) -> JobResult:
    """One scoring job: read conversation, score, write secondary score file."""
    sid: str = payload["session_id"]
    paths: RunPaths = payload["paths"]
    rubric = payload["rubric"]
    eval_provider: str = payload["eval_provider"]
    eval_model: str = payload["eval_model"]

    artifact = read_conversation(paths, sid)
    score_art = score_conversation(
        artifact=artifact,
        rubric=rubric,
        evaluator_provider=eval_provider,
        evaluator_model=eval_model,
        temperature=cfg.EVALUATOR_TEMPERATURE,
    )
    # suffix_with_evaluator=True so we never overwrite the primary file.
    write_score_artifact(paths, score_art, suffix_with_evaluator=True)

    # Compact summary for progress display.
    means: dict = {}
    n_parse_fail = 0
    for dim in rubric.dimensions:
        valid = [t.scores.get(dim, -1.0) for t in score_art.turns
                 if t.scores.get(dim, -1.0) >= 0]
        means[dim] = round(sum(valid) / len(valid), 2) if valid else float("nan")
        n_parse_fail += sum(
            1 for t in score_art.turns if t.scores.get(dim, -1.0) < 0
        )

    return JobResult(
        job_id=sid,
        status="ok",
        info={**means, "parse_fail": n_parse_fail},
    )


def _do_scoring(
    *,
    paths: RunPaths,
    sampled_ids: list[str],
    eval_provider: str,
    eval_model: str,
    workers: int,
) -> None:
    """Phase 2: score the sampled sessions with the secondary evaluator."""
    rubric = RUBRICS[cfg.RUBRIC_NAME]
    eval_slug = safe_slug(f"{eval_provider}/{eval_model}")
    secondary_filename_part = f"{rubric.name}__{eval_slug}"

    print(f"\n[scoring] rubric    : {rubric.name} (dims: {rubric.dimensions})")
    print(f"[scoring] evaluator : {eval_provider}/{eval_model}")
    print(f"[scoring] file slug : __{eval_slug}.json")

    # Enumerate jobs. Skip sessions that already have the secondary score
    # file. Also count sessions whose primary score file is missing — the
    # agreement analysis will skip those, so it's worth flagging early.
    jobs: list[Job] = []
    n_already_scored = 0
    n_missing_primary = 0
    for sid in sampled_ids:
        if not paths.score_path(sid, rubric.name).exists():
            n_missing_primary += 1
        if paths.score_path(sid, secondary_filename_part).exists():
            n_already_scored += 1
            continue
        jobs.append(Job(
            job_id=sid,
            payload={
                "session_id": sid,
                "paths": paths,
                "rubric": rubric,
                "eval_provider": eval_provider,
                "eval_model": eval_model,
            },
        ))

    if n_missing_primary:
        print(
            f"\n[scoring] ⚠ {n_missing_primary} sampled session(s) lack the "
            f"primary score file (<sid>__{rubric.name}.json). "
            f"Run main_user_IVs/run_scoring.py with --rubric {rubric.name} "
            f"to produce them — the agreement analysis script will skip "
            f"sessions where either side is missing.\n"
        )

    print(
        f"[scoring] {len(jobs)} session(s) to score, "
        f"{n_already_scored} already scored — "
        f"total sample = {len(sampled_ids)}."
    )

    if not jobs:
        print("[scoring] nothing to do.")
        return

    # Distinct checkpoint name so multiple scoring passes (e.g. with
    # different secondary evaluators) on the same run dir don't overwrite
    # each other's progress files.
    run_jobs(
        jobs=jobs,
        worker=_scoring_worker,
        paths=paths,
        n_workers=workers,
        is_done=lambda j: False,  # already filtered above
        progress_label="validation-scoring",
        checkpoint_name=f"checkpoint_validation_{secondary_filename_part}.json",
    )


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run-dir", type=Path, required=True,
        help="Existing main_user_IVs run directory "
             "(results/final_experiment/main_user_IVs/<timestamp>).",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--fraction", type=float, default=cfg.SAMPLE_FRACTION,
        help=f"Sampling fraction (default {cfg.SAMPLE_FRACTION}).",
    )
    p.add_argument("--seed", type=int, default=cfg.SAMPLE_SEED)
    p.add_argument(
        "--resample", action="store_true",
        help="Force regeneration of sample_manifest.json. Use with care: "
             "any secondary score files produced from the previous sample "
             "remain on disk but won't be analysed unless their session "
             "appears in the new sample.",
    )
    p.add_argument(
        "--sample-only", action="store_true",
        help="Run Phase 1 only; skip Phase 2 (scoring). Useful for "
             "inspecting coverage before paying for API calls.",
    )
    p.add_argument(
        "--beliefs-path", type=Path,
        default=main_cfg.BELIEFS_PATH,
        help=f"Path to sampled_beliefs.json (default: {main_cfg.BELIEFS_PATH}).",
    )
    p.add_argument(
        "--secondary-provider", type=str,
        default=cfg.SECONDARY_EVALUATOR[0],
    )
    p.add_argument(
        "--secondary-model", type=str,
        default=cfg.SECONDARY_EVALUATOR[1],
    )
    args = p.parse_args()

    if args.workers < 1:
        p.error("--workers must be >= 1")
    if not (0.0 < args.fraction <= 1.0):
        p.error("--fraction must be in (0, 1]")
    if not args.run_dir.exists():
        p.error(f"--run-dir does not exist: {args.run_dir}")
    if not (args.run_dir / "conversations").exists():
        p.error(
            f"{args.run_dir}/conversations not found — pass a "
            "main_user_IVs run dir with completed conversations."
        )

    paths = RunPaths(root=args.run_dir)
    paths.ensure_dirs()
    validation_dir = paths.root / cfg.VALIDATION_SUBDIR
    validation_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRun dir           : {paths.root}")
    print(f"Conversations dir : {paths.conversations}")
    print(f"Scores dir        : {paths.scores}")
    print(f"Validation dir    : {validation_dir}")

    sampled_ids = _do_sampling(
        paths=paths,
        validation_dir=validation_dir,
        beliefs_path=args.beliefs_path,
        fraction=args.fraction,
        seed=args.seed,
        resample=args.resample,
    )

    if args.sample_only:
        print("\n[done] --sample-only set; skipping scoring phase.")
        return

    _do_scoring(
        paths=paths,
        sampled_ids=sampled_ids,
        eval_provider=args.secondary_provider,
        eval_model=args.secondary_model,
        workers=args.workers,
    )

    print(
        "\n[done] Phase 1 + Phase 2 complete.\n"
        "       Next: python -m evaluator_validation.analyze_agreement "
        f"--run-dir {paths.root}"
    )


if __name__ == "__main__":
    main()