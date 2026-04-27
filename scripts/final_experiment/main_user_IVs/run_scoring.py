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

Usage
-----
  cd scripts/final-experiment
  python -m main_user_IVs.run_scoring \
      --run-dir results/main_user_IVs/<timestamp> \
      --evaluator primary
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import (  # noqa: E402
    Job,
    JobResult,
    RUBRICS,
    Rubric,
    RunPaths,
    read_conversation,
    run_jobs,
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
    rubric: Rubric = payload["rubric"]
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
    p.add_argument("--rubric", default=cfg.DEFAULT_RUBRIC_NAME)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--suffix-evaluator", action="store_true",
        help="Append evaluator slug to score filenames. Use this when "
             "scoring the same conversations with multiple evaluators.",
    )
    args = p.parse_args()

    if args.workers < 1:
        p.error("--workers must be >= 1")
    if not args.run_dir.exists():
        p.error(f"--run-dir does not exist: {args.run_dir}")
    if args.rubric not in RUBRICS:
        p.error(f"--rubric must be one of {list(RUBRICS)}")

    paths = RunPaths(root=args.run_dir)
    paths.ensure_dirs()
    rubric = RUBRICS[args.rubric]
    eval_provider, eval_model = EVALUATOR_REGISTRY[args.evaluator]

    print(f"\nRun dir   : {paths.root}")
    print(f"Rubric    : {rubric.name} (dims: {rubric.dimensions})")
    print(f"Evaluator : {eval_provider}/{eval_model}")
    print(f"Suffix    : {args.suffix_evaluator}\n")

    # Build jobs from conversations on disk. Skip ones whose score file
    # already exists.
    from core.storage import _safe
    eval_slug = _safe(f"{eval_provider}/{eval_model}")

    def already_scored(sid: str) -> bool:
        if args.suffix_evaluator:
            return paths.score_path(sid, f"{rubric.name}__{eval_slug}").exists()
        return paths.score_path(sid, rubric.name).exists()

    jobs: list[Job] = []
    for path in sorted(paths.conversations.glob("*.json")):
        sid = path.stem
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

    if not jobs:
        print("All conversations already scored. Nothing to do.")
        return

    run_jobs(
        jobs=jobs,
        worker=_worker,
        paths=paths,
        n_workers=args.workers,
        is_done=lambda job: False,  # already filtered above
        progress_label="scoring",
        checkpoint_name=f"checkpoint_scoring__{rubric.name}__{eval_slug}.json",
    )


if __name__ == "__main__":
    main()