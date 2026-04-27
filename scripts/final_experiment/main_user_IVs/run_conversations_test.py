"""
main_user_IVs/run_conversations_test.py
=======================================
Test runner for Stage 1 of RQ1: generate a random subset of conversation
sessions instead of the full grid.

The subset size is controlled by --n-sessions. Results are written under
an experiment folder that includes both "test" and the sampled session
count, e.g.:

  results/final_experiment/main_user_IVs_test_25sessions/<timestamp>/

Usage
-----
  cd scripts/final_experiment
  python -m main_user_IVs.run_conversations_test \
      --n-sessions 32 \
      --seed 42 \
      --workers 32
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Allow `python -m main_user_IVs.run_conversations_test` from the parent
# directory. Adds scripts/final_experiment to sys.path so `core` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import (  # noqa: E402
    Job,
    list_completed_conversation_ids,
    make_run_paths,
    read_manifest,
    run_jobs,
    write_manifest,
)

from . import config as cfg  # noqa: E402
from .run_conversations import _build_jobs, _worker  # noqa: E402


def _test_experiment_name(n_sessions: int) -> str:
    return f"{cfg.EXPERIMENT_NAME}_test_{n_sessions}sessions"


def _resolve_sample(
    *,
    all_jobs: list[Job],
    n_sessions: int,
    seed: int,
    existing_manifest: dict | None,
) -> tuple[list[Job], list[str], bool]:
    """Pick sampled jobs or reuse a previously-saved sample on resume.

    Returns:
      (sampled_jobs, sampled_session_ids, reused_existing_sample)
    """
    job_by_id = {job.job_id: job for job in all_jobs}

    if existing_manifest and existing_manifest.get("sampled_session_ids"):
        sampled_ids = list(existing_manifest["sampled_session_ids"])
        missing = [sid for sid in sampled_ids if sid not in job_by_id]
        if missing:
            raise RuntimeError(
                "Resume manifest references unknown session IDs "
                f"(first missing: {missing[0]})."
            )
        sampled_jobs = [job_by_id[sid] for sid in sampled_ids]
        return sampled_jobs, sampled_ids, True

    if n_sessions > len(all_jobs):
        raise ValueError(
            f"--n-sessions ({n_sessions}) exceeds total session pool "
            f"({len(all_jobs)})."
        )

    rng = random.Random(seed)
    sampled_ids = rng.sample([job.job_id for job in all_jobs], n_sessions)
    sampled_jobs = [job_by_id[sid] for sid in sampled_ids]
    return sampled_jobs, sampled_ids, False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-sessions", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    if args.n_sessions < 1:
        p.error("--n-sessions must be >= 1")
    if args.workers < 1:
        p.error("--workers must be >= 1")

    experiment_name = _test_experiment_name(args.n_sessions)
    paths = make_run_paths(
        cfg.RESULTS_DIR,
        experiment_name,
        resume_dir=args.resume,
    )
    print(f"\nResults dir: {paths.root}")
    print(f"Conversations: {paths.conversations}")

    all_jobs = _build_jobs(paths)
    existing_manifest = read_manifest(paths)
    jobs, sampled_ids, reused_existing = _resolve_sample(
        all_jobs=all_jobs,
        n_sessions=args.n_sessions,
        seed=args.seed,
        existing_manifest=existing_manifest,
    )

    if reused_existing:
        print(
            f"Reusing sampled session list from existing manifest "
            f"({len(sampled_ids)} sessions)."
        )
    else:
        print(
            f"Sampled {len(sampled_ids)} sessions from {len(all_jobs)} total "
            f"(seed={args.seed})."
        )

    completed_ids = list_completed_conversation_ids(paths)

    write_manifest(paths, {
        "experiment": experiment_name,
        "phase": "conversations_test",
        "is_test_run": True,
        "n_sessions_pool": len(all_jobs),
        "n_sessions_sampled": len(jobs),
        "seed": args.seed,
        "sampled_session_ids": sampled_ids,
        "user_agent": f"{cfg.USER_AGENT_PROVIDER}/{cfg.USER_AGENT_MODEL}",
        "target_llms": [f"{p}/{m}" for p, m, _ in cfg.TARGET_LLMS],
        "n_turns": cfg.N_TURNS,
        "temperature_user": cfg.TEMPERATURE_USER,
        "temperature_target": cfg.TEMPERATURE_TARGET,
    })

    run_jobs(
        jobs=jobs,
        worker=_worker,
        paths=paths,
        n_workers=args.workers,
        is_done=lambda job: job.job_id in completed_ids,
        progress_label="test-session",
        checkpoint_name=f"checkpoint_conversations_test_{len(jobs)}sessions.json",
    )


if __name__ == "__main__":
    main()
