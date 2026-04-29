"""
main_user_IVs/run_scoring.py
============================
Stage 2 of RQ1: score every conversation with the chosen rubric and
evaluator. Read-only against conversations/, write-only to scores/.

Single-evaluator mode (default): scores every conversation under one
(rubric, evaluator) pair. Re-run with a different evaluator to add a
second score file per session — files are namespaced by evaluator slug
when --suffix-evaluator is set.

Two-evaluator agreement is built on top of this script: run twice with
two different evaluators and --suffix-evaluator, then a downstream
analysis step compares the two score files per session.

Rubric kinds
------------
This script supports both single-prompt rubrics (``misinfo_v1``,
``misinfo_v0``) and multi-prompt rubrics (``misinfo_v1_split``). The
multi-prompt variant calls the evaluator three times per turn (once
per dimension) — about 3× the cost and runtime of the single-prompt
variant. See core/scoring.py for the rubric definitions.

The two rubric kinds drop side-by-side files into the same scores/
folder, distinguished by the rubric name in the filename:

  scores/<sid>__misinfo_v1.json        ← single-prompt
  scores/<sid>__misinfo_v1_split.json  ← multi-prompt

The rubric_name field inside each artifact is the primary
disambiguator; the filename suffix mirrors it.

Note: the Anthropic "thinking" escalation variant from config.py is
reserved for a follow-up script that passes provider-specific thinking
kwargs; this script currently runs the non-thinking evaluator set.

Usage
-----
  cd scripts/final_experiment

  # Single-prompt rubric (default), full corpus:
  python -m main_user_IVs.run_scoring \\
      --run-dir results/main_user_IVs/<timestamp> \\
      --evaluator primary

  # Multi-prompt rubric, control-cell only (iv1=none, iv2=none):
  python -m main_user_IVs.run_scoring \\
      --run-dir results/main_user_IVs/<timestamp> \\
      --evaluator primary \\
      --rubric misinfo_v1_split \\
      --cell iv1-none__iv2-none

  # Restrict to one or more target models (slugs from config.TARGET_LLMS):
  python -m main_user_IVs.run_scoring \\
      --run-dir results/main_user_IVs/<timestamp> \\
      --evaluator primary \\
      --target gemini-3-flash deepseek-v3.2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import (  # noqa: E402
    AnyRubric,
    Job,
    JobResult,
    RUBRICS,
    RunPaths,
    read_conversation,
    run_jobs,
    safe_slug,
    score_conversation,
    write_score_artifact,
)

from . import config as cfg  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(threadName)-12s | %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


# ════════════════════════════════════════════════════════════════════════════
# Evaluator pick
# ════════════════════════════════════════════════════════════════════════════

EVALUATOR_REGISTRY = {
    "primary":   cfg.EVALUATOR_PRIMARY,
    "secondary": cfg.EVALUATOR_SECONDARY,
    "claude":    cfg.EVALUATOR_ESCALATION_NONTHINKING,
}


# ════════════════════════════════════════════════════════════════════════════
# Worker
# ════════════════════════════════════════════════════════════════════════════

def _worker(payload: dict) -> JobResult:
    sid: str = payload["session_id"]
    paths: RunPaths = payload["paths"]
    rubric: AnyRubric = payload["rubric"]
    eval_provider: str = payload["eval_provider"]
    eval_model: str = payload["eval_model"]
    suffix_evaluator: bool = payload["suffix_evaluator"]

    artifact = read_conversation(paths, sid)
    score_art = score_conversation(
        artifact=artifact,
        rubric=rubric,
        evaluator_provider=eval_provider,
        evaluator_model=eval_model,
        temperature=cfg.EVALUATOR_TEMPERATURE,
    )
    write_score_artifact(paths, score_art, suffix_with_evaluator=suffix_evaluator)

    # Quick summary for progress display.
    means = {
        dim: round(
            sum(t.scores.get(dim, -1) for t in score_art.turns
                if t.scores.get(dim, -1) >= 0)
            / max(1, sum(1 for t in score_art.turns
                         if t.scores.get(dim, -1) >= 0)),
            2,
        )
        for dim in rubric.dimensions
    }
    n_parse_fail = sum(
        1 for t in score_art.turns
        for v in t.scores.values() if v == -1.0
    )

    return JobResult(
        job_id=sid,
        status="ok",
        info={**means, "parse_fail": n_parse_fail},
    )


# ════════════════════════════════════════════════════════════════════════════
# Cell filter
# ════════════════════════════════════════════════════════════════════════════

def _matches_cell(session_id: str, cell_id: str) -> bool:
    """Return True iff the session_id belongs to the given cell.

    session_ids are built as
      ``cell-{cell_id}__belief-{cat}-{idx}__model-{slug}``
    so the cell is the substring between ``cell-`` and the first
    ``__belief-``. We anchor on the literal ``__belief-`` so partial
    cell ids like ``iv1-warm__iv2-n`` don't false-match
    ``iv1-warm__iv2-none`` or ``iv1-warm__iv2-norms``. Doing it by
    string-matching is faster than reading each artifact, which matters
    for large runs.
    """
    prefix = f"cell-{cell_id}__belief-"
    return session_id.startswith(prefix)


def _matches_target(session_id: str, targets: list[str]) -> bool:
    """Return True iff the session_id's target model matches one of ``targets``.

    session_ids end with ``__model-{slug}`` (the short slug from
    config.TARGET_LLMS, e.g. ``gemini-3-flash``). We anchor on the
    literal ``__model-`` so partial slugs like ``gpt-5`` don't
    false-match ``gpt-5.3-chat``.
    """
    for tgt in targets:
        if session_id.endswith(f"__model-{tgt}"):
            return True
    return False


_TARGET_SLUGS = [slug for _, _, slug in cfg.TARGET_LLMS]


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir", type=Path, required=True,
        help="Existing results dir (results/main_user_IVs/<timestamp>).",
    )
    p.add_argument(
        "--evaluator", choices=list(EVALUATOR_REGISTRY), default="primary",
        help="Which evaluator from config.py to use.",
    )
    p.add_argument(
        "--rubric", default=cfg.DEFAULT_RUBRIC_NAME,
        choices=list(RUBRICS),
        help="Rubric name. 'misinfo_v1' = single-prompt 3-dim. "
             "'misinfo_v1_split' = three independent prompts (3x cost). "
             "'misinfo_v0' = legacy 2-dim.",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--suffix-evaluator", action="store_true",
        help="Append evaluator slug to score filenames. Use this when "
             "scoring the same conversations with multiple evaluators.",
    )
    p.add_argument(
        "--cell", type=str, default=None,
        help="Restrict scoring to one cell (e.g. 'iv1-none__iv2-none' "
             "for the pure-control cell). Match is by session_id prefix.",
    )
    p.add_argument(
        "--target", type=str, nargs="+", default=None,
        choices=_TARGET_SLUGS,
        help="Restrict scoring to one or more target-model groups, "
             "identified by the short slug from config.TARGET_LLMS "
             f"(e.g. {' '.join(_TARGET_SLUGS)}). Match is on the "
             "'__model-{slug}' suffix of the session_id.",
    )
    args = p.parse_args()

    if args.workers < 1:
        p.error("--workers must be >= 1")
    if not args.run_dir.exists():
        p.error(f"--run-dir does not exist: {args.run_dir}")

    paths = RunPaths(root=args.run_dir)
    paths.ensure_dirs()
    rubric = RUBRICS[args.rubric]
    eval_provider, eval_model = EVALUATOR_REGISTRY[args.evaluator]

    print(f"\nRun dir   : {paths.root}")
    print(f"Rubric    : {rubric.name} (dims: {rubric.dimensions})")
    print(f"Evaluator : {eval_provider}/{eval_model}")
    print(f"Suffix    : {args.suffix_evaluator}")
    if args.cell:
        print(f"Cell      : {args.cell}")
    if args.target:
        print(f"Target    : {', '.join(args.target)}")
    print()

    # Build jobs from conversations on disk. Skip ones whose score file
    # already exists. Multi-prompt rubric output goes to a different
    # filename (rubric.name differs), so it never collides with a
    # single-prompt rubric's output for the same session.
    eval_slug = safe_slug(f"{eval_provider}/{eval_model}")

    def already_scored(sid: str) -> bool:
        if args.suffix_evaluator:
            return paths.score_path(sid, f"{rubric.name}__{eval_slug}").exists()
        return paths.score_path(sid, rubric.name).exists()

    jobs: list[Job] = []
    n_filtered_out = 0
    for path in sorted(paths.conversations.glob("*.json")):
        sid = path.stem
        if args.cell and not _matches_cell(sid, args.cell):
            n_filtered_out += 1
            continue
        if args.target and not _matches_target(sid, args.target):
            n_filtered_out += 1
            continue
        if already_scored(sid):
            continue
        jobs.append(Job(
            job_id=sid,
            payload={
                "session_id": sid,
                "paths": paths,
                "rubric": rubric,
                "eval_provider": eval_provider,
                "eval_model": eval_model,
                "suffix_evaluator": args.suffix_evaluator,
            },
        ))

    if args.cell or args.target:
        print(f"Filters excluded {n_filtered_out} session(s).")

    if not jobs:
        if (args.cell or args.target) and n_filtered_out > 0:
            print(
                "All matching sessions already scored, or filters "
                "matched 0 conversations. Nothing to do."
            )
        else:
            print("All conversations already scored. Nothing to do.")
        return

    # Distinct checkpoint name per (rubric, evaluator, cell-filter,
    # target-filter) so multiple scoring passes on the same run dir
    # don't overwrite each other's progress files.
    checkpoint_suffix = f"{rubric.name}__{eval_slug}"
    if args.cell:
        checkpoint_suffix += f"__cell-{args.cell}"
    if args.target:
        checkpoint_suffix += f"__target-{'-'.join(args.target)}"

    run_jobs(
        jobs=jobs,
        worker=_worker,
        paths=paths,
        n_workers=args.workers,
        is_done=lambda job: False,  # already filtered above
        progress_label="scoring",
        checkpoint_name=f"checkpoint_scoring__{checkpoint_suffix}.json",
    )


if __name__ == "__main__":
    main()