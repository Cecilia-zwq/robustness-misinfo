"""
reflection_ablation/run_ablation.py
===================================
Driver for the reflection-module ablation experiment.

Workflow per session:

  1. Sample ~12% of break conversations, stratified by
     (iv1 × belief_category × target_llm) cell, with 70% turn-1 / 30%
     turn-2+ break splits. The full sample plan is written to
     ``ablation_sample_index.json`` so the experiment is reproducible
     and resumable.
  2. For each sampled session, load the source artifact, build a
     SimulatedUserAgent with ``max_reflect_retries=0`` (so reflection is
     bypassed), wrap it in NoReflectionSimulation, and run
     ``run_branched_conversation``.
  3. Write the result to
     ``<source-run>/conversations_none_reflection/<session_id>__noref.json``.

Usage::

    cd scripts/final_experiment
    python -m reflection_ablation.run_ablation [--workers 16] [--rebuild-sample]

Resume: the script skips sessions whose output file already exists.
Re-running picks up where it left off. Use ``--rebuild-sample`` to
re-derive the sampling plan (e.g. if the source corpus changed); by
default the persisted plan in ``ablation_sample_index.json`` is reused.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make sibling packages importable when run as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import (  # noqa: E402
    ConversationArtifact,
    Job,
    JobResult,
    NoReflectionSimulation,
    RunPaths,
    atomic_write_json,
    atomic_write_text,
)
from misinfo_eval_framework import SimulatedUserAgent, TargetLLM  # noqa: E402

from . import config as cfg  # noqa: E402
from .branched_runner import run_branched_conversation  # noqa: E402
from .sampling import (  # noqa: E402
    SampleEntry,
    build_sample,
    read_sample_index,
    write_sample_index,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(threadName)-12s | %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Session-id helpers
# ════════════════════════════════════════════════════════════════════════════

def _ablation_session_id(source_session_id: str) -> str:
    """Suffix the source session id so the ablation output is distinguishable."""
    return f"{source_session_id}{cfg.SESSION_ID_SUFFIX}"


def _output_path(session_id: str) -> Path:
    return cfg.OUTPUT_CONV_DIR / f"{session_id}.json"


# ════════════════════════════════════════════════════════════════════════════
# Worker
# ════════════════════════════════════════════════════════════════════════════

def _worker(payload: dict) -> JobResult:
    """Run one ablation session: load source → branch + no-reflect → persist."""
    entry: SampleEntry = payload["entry"]
    new_session_id: str = payload["new_session_id"]
    output_path: Path = payload["output_path"]

    # ── Load source artifact ─────────────────────────────────────────────
    with open(entry.source_path, "r", encoding="utf-8") as fh:
        source = ConversationArtifact.from_dict(json.load(fh))

    # Sanity: the break turn must have a recorded rejected draft.
    src_break_turn = source.turns[entry.first_break_turn - 1]
    if not src_break_turn.reflection_attempts:
        raise ValueError(
            f"Source turn {entry.first_break_turn} has no reflection_attempts; "
            f"cannot extract a broken draft."
        )
    branching_message = src_break_turn.reflection_attempts[0].draft
    if not branching_message or not branching_message.strip():
        raise ValueError(
            f"Source turn {entry.first_break_turn} attempt-1 draft is empty."
        )

    # ── Build the no-reflection user agent ───────────────────────────────
    character_prompt = source.cell.get("character_prompt", "")
    is_long_text = bool(source.belief.get("is_long_text", False))

    agent = SimulatedUserAgent(
        provider=cfg.USER_AGENT_PROVIDER,
        model=cfg.USER_AGENT_MODEL,
        character_prompt=character_prompt,
        is_long_text=is_long_text,
        max_reflect_retries=0,           # <- reflection OFF
        temperature=cfg.TEMPERATURE_USER,
    )
    user_sim = NoReflectionSimulation(agent)

    # ── Build the target (same model as the source session) ──────────────
    raw_target = source.models.get("target_llm", "")
    if "/" not in raw_target:
        raise ValueError(
            f"Cannot parse target_llm={raw_target!r} from source artifact."
        )
    provider, model = raw_target.split("/", 1)
    target = TargetLLM(
        provider=provider,
        model=model,
        temperature=cfg.TEMPERATURE_TARGET,
        system_prompt=cfg.SYS_TARGET,
    )

    # ── Run the branched conversation ────────────────────────────────────
    n_turns = int(source.config.get("n_turns", cfg.N_TURNS))
    artifact = run_branched_conversation(
        session_id=new_session_id,
        experiment=cfg.EXPERIMENT_NAME,
        source=source,
        break_turn=entry.first_break_turn,
        branching_message=branching_message,
        user_simulation=user_sim,
        target=target,
        n_turns=n_turns,
    )

    # ── Persist atomically ───────────────────────────────────────────────
    atomic_write_text(output_path, artifact.to_json())

    post_branch = artifact.turns[entry.first_break_turn:]
    n_char  = sum(t.n_character_breaks for t in post_branch)
    n_bel   = sum(t.n_belief_breaks    for t in post_branch)
    n_empty = sum(1 for t in post_branch if t.target_empty)

    return JobResult(
        job_id=new_session_id,
        status="ok",
        info={
            "brk_turn": entry.first_break_turn,
            "post_char": n_char,
            "post_bel":  n_bel,
            "post_empty": n_empty,
        },
    )


# ════════════════════════════════════════════════════════════════════════════
# Sample plan + manifest
# ════════════════════════════════════════════════════════════════════════════

def _load_or_build_sample(rebuild: bool) -> list[SampleEntry]:
    if (not rebuild) and cfg.SAMPLE_INDEX_PATH.exists():
        planned = read_sample_index(cfg.SAMPLE_INDEX_PATH)
        print(f"\nLoaded {len(planned)} planned sessions from "
              f"{cfg.SAMPLE_INDEX_PATH.name}")
        return planned

    print(f"\nBuilding sample plan from {cfg.SOURCE_CONV_DIR} ...")
    planned = build_sample(
        cfg.SOURCE_CONV_DIR,
        sample_fraction=cfg.SAMPLE_FRACTION,
        min_per_cell=cfg.MIN_PER_CELL,
        turn1_ratio=cfg.TURN1_RATIO,
        seed=cfg.SAMPLING_SEED,
    )
    write_sample_index(
        cfg.SAMPLE_INDEX_PATH,
        planned=planned,
        params={
            "sample_fraction": cfg.SAMPLE_FRACTION,
            "min_per_cell": cfg.MIN_PER_CELL,
            "turn1_ratio": cfg.TURN1_RATIO,
            "seed": cfg.SAMPLING_SEED,
            "source_conv_dir": str(cfg.SOURCE_CONV_DIR),
        },
    )
    print(f"Wrote sample plan: {cfg.SAMPLE_INDEX_PATH}")
    return planned


def _build_jobs(planned: list[SampleEntry]) -> list[Job]:
    jobs: list[Job] = []
    for entry in planned:
        new_sid = _ablation_session_id(entry.source_session_id)
        out = _output_path(new_sid)
        jobs.append(Job(
            job_id=new_sid,
            payload={
                "entry": entry,
                "new_session_id": new_sid,
                "output_path": out,
            },
        ))
    return jobs


def _list_completed_outputs() -> set[str]:
    """Session ids whose ablation artifact is fully written on disk."""
    if not cfg.OUTPUT_CONV_DIR.exists():
        return set()
    done: set[str] = set()
    for p in cfg.OUTPUT_CONV_DIR.glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if (
            data.get("completed_at") is not None
            and data.get("session_id") == p.stem
        ):
            done.add(p.stem)
    return done


def _write_manifest(*, n_planned: int, n_workers: int) -> None:
    atomic_write_json(cfg.MANIFEST_PATH, {
        "experiment": cfg.EXPERIMENT_NAME,
        "phase": "ablation_conversations",
        "source_run": str(cfg.SOURCE_RUN_DIR),
        "source_conversations_dir": str(cfg.SOURCE_CONV_DIR),
        "output_conversations_dir": str(cfg.OUTPUT_CONV_DIR),
        "n_sessions_planned": n_planned,
        "n_workers": n_workers,
        "sampling": {
            "sample_fraction": cfg.SAMPLE_FRACTION,
            "min_per_cell": cfg.MIN_PER_CELL,
            "turn1_ratio": cfg.TURN1_RATIO,
            "seed": cfg.SAMPLING_SEED,
        },
        "user_agent": f"{cfg.USER_AGENT_PROVIDER}/{cfg.USER_AGENT_MODEL}",
        "user_agent_max_reflect_retries": 0,
        "temperature_user": cfg.TEMPERATURE_USER,
        "temperature_target": cfg.TEMPERATURE_TARGET,
        "n_turns": cfg.N_TURNS,
    })


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8,
                   help="Number of parallel worker threads (default 8).")
    p.add_argument("--rebuild-sample", action="store_true",
                   help="Re-derive the sampling plan from the source corpus, "
                        "overwriting ablation_sample_index.json.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build the sample plan and print stats, but do not "
                        "run any sessions.")
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be >= 1")

    # Ensure the source run actually exists.
    if not cfg.SOURCE_CONV_DIR.exists():
        raise SystemExit(f"Source dir does not exist: {cfg.SOURCE_CONV_DIR}")
    cfg.OUTPUT_CONV_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nSource conversations: {cfg.SOURCE_CONV_DIR}")
    print(f"Output conversations: {cfg.OUTPUT_CONV_DIR}")

    planned = _load_or_build_sample(args.rebuild_sample)
    jobs = _build_jobs(planned)

    if args.dry_run:
        completed = _list_completed_outputs()
        remaining = [j for j in jobs if j.job_id not in completed]
        print(f"\nDry run: {len(jobs)} jobs planned, "
              f"{len(completed)} already complete, "
              f"{len(remaining)} would be executed.")
        return

    _write_manifest(n_planned=len(jobs), n_workers=args.workers)

    completed_ids = _list_completed_outputs()

    # The shared runner expects RunPaths only for its checkpoint location;
    # we point it at the source run dir so checkpoint_ablation.json lives
    # next to the existing run artifacts.
    paths = RunPaths(root=cfg.SOURCE_RUN_DIR)
    paths.ensure_dirs()  # safe no-op for existing dirs

    # Import run_jobs late so the rest of the script can be imported in
    # tests without spinning up the executor module.
    from core.runner import run_jobs  # noqa: E402

    run_jobs(
        jobs=jobs,
        worker=_worker,
        paths=paths,
        n_workers=args.workers,
        is_done=lambda job: job.job_id in completed_ids,
        progress_label="ablation",
        checkpoint_name=cfg.CHECKPOINT_PATH.name,
    )


if __name__ == "__main__":
    main()
