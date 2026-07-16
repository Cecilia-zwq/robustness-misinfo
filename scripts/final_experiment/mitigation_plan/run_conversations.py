"""
mitigation_plan/run_conversations.py
=====================================
Generate the mitigation conversation corpus: for every session in
config.SAMPLE_INDEX_PATH (see select_sample.py), replay the same belief,
IV condition, target model, and user agent through a fresh *adaptive*
conversation — the only change is the target's system prompt, which
carries config.MITIGATION_STATEMENT on top of the baseline SYS_TARGET.

Unlike static_interactive_ablation (which deliberately makes the user
non-adaptive to isolate interactivity), the user agent here stays fully
adaptive, since the question under test is how the target's behavior —
and the user's downstream reactions to it — shift under the mitigated
system prompt, not whether interactivity itself matters.

Output: an independent, timestamped run directory
results/final_experiment/mitigation_plan/<timestamp>/conversations/,
scored afterwards by reusing main_user_IVs/run_scoring.py unmodified
(see package docstring for the exact commands).

Usage
-----
  cd scripts/final_experiment
  python -m mitigation_plan.run_conversations [--workers 8] [--resume <dir>]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import (  # noqa: E402
    AgentSimulation,
    ConversationArtifact,
    Job,
    JobResult,
    list_completed_conversation_ids,
    make_run_paths,
    run_conversation,
    run_jobs,
    write_conversation,
    write_manifest,
)

from . import config as cfg  # noqa: E402
from . import sampling  # noqa: E402

from misinfo_eval_framework import SimulatedUserAgent, TargetLLM  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(threadName)-12s | %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Session-id helpers
# ════════════════════════════════════════════════════════════════════════════

def _mitigation_session_id(source_session_id: str) -> str:
    return f"{source_session_id}{cfg.SESSION_ID_SUFFIX}"


# ════════════════════════════════════════════════════════════════════════════
# Worker
# ════════════════════════════════════════════════════════════════════════════

def _worker(payload: dict) -> JobResult:
    """One session: load the source artifact, replay with the mitigated
    system prompt, persist the new artifact."""
    source_path: Path = payload["source_path"]
    new_session_id: str = payload["new_session_id"]
    paths = payload["paths"]

    with source_path.open("r", encoding="utf-8") as fh:
        source = ConversationArtifact.from_dict(json.load(fh))

    character_prompt = source.cell.get("character_prompt", "")
    is_long_text = bool(source.belief.get("is_long_text", False))

    agent = SimulatedUserAgent(
        provider=cfg.USER_AGENT_PROVIDER,
        model=cfg.USER_AGENT_MODEL,
        character_prompt=character_prompt,
        is_long_text=is_long_text,
        max_reflect_retries=cfg.MAX_REFLECT_RETRIES,
        temperature=cfg.TEMPERATURE_USER,
    )

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

    artifact = run_conversation(
        session_id=new_session_id,
        experiment=cfg.EXPERIMENT_NAME,
        cell=source.cell,
        belief=source.belief,
        user_simulation=AgentSimulation(agent),
        target=target,
        n_turns=cfg.N_TURNS,
        config_metadata={"source_session_id": source.session_id},
    )
    write_conversation(paths, artifact)

    n_char = sum(t.n_character_breaks for t in artifact.turns)
    n_belief = sum(t.n_belief_breaks for t in artifact.turns)
    n_fb = sum(1 for t in artifact.turns if t.is_fallback)

    return JobResult(
        job_id=new_session_id,
        status="ok",
        info={"char_brk": n_char, "bel_brk": n_belief, "fb": n_fb},
    )


# ════════════════════════════════════════════════════════════════════════════
# Job builder
# ════════════════════════════════════════════════════════════════════════════

def _build_jobs(entries: list[dict], paths) -> list[Job]:
    jobs: list[Job] = []
    for entry in entries:
        source_sid = entry["session_id"]
        new_sid = _mitigation_session_id(source_sid)
        jobs.append(Job(
            job_id=new_sid,
            payload={
                "source_path": Path(entry["source_path"]),
                "new_session_id": new_sid,
                "paths": paths,
            },
        ))
    return jobs


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--resume", type=str, default=None,
        help="Path to an existing mitigation_plan/<timestamp> run dir to "
             "continue. Already-written conversations are skipped; only "
             "missing/failed sessions are (re)run.",
    )
    p.add_argument(
        "--sample-index", type=Path, default=cfg.SAMPLE_INDEX_PATH,
        help="Path to a mitigation_sample_index.json written by "
             "select_sample.py (default: config.SAMPLE_INDEX_PATH).",
    )
    p.add_argument(
        "--max-same-error", type=int, default=5,
        help="Stop early once the same error message has recurred this many "
             "times (catches systematic failures — bad API key, wrong model "
             "slug, exhausted credits — before burning through the whole "
             "sample). 0 disables this gate. Default: 5.",
    )
    p.add_argument(
        "--max-failure-rate", type=float, default=None,
        help="Stop early once failed/consumed reaches this rate, e.g. 0.5. "
             "Disabled by default.",
    )
    p.add_argument(
        "--max-failed", type=int, default=None,
        help="Stop early once this many jobs have failed (absolute count). "
             "Disabled by default.",
    )
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be >= 1")

    if not args.sample_index.exists():
        raise SystemExit(
            f"Sample index not found: {args.sample_index}\n"
            "Run `python -m mitigation_plan.select_sample` first."
        )

    entries = sampling.read_sample_index(args.sample_index)

    paths = make_run_paths(
        cfg.RESULTS_DIR,
        cfg.EXPERIMENT_NAME,
        resume_dir=args.resume,
    )
    print(f"\nResults dir  : {paths.root}")
    print(f"Conversations: {paths.conversations}")
    print(f"Source run   : {cfg.SOURCE_RUN_DIR}")
    print(f"Sample index : {args.sample_index}  ({len(entries)} sessions)")
    print(f"User agent   : {cfg.USER_AGENT_PROVIDER}/{cfg.USER_AGENT_MODEL}")
    print(f"Workers      : {args.workers}")
    print(f"Fail-fast    : max_same_error={args.max_same_error or 'off'}  "
          f"max_failure_rate={args.max_failure_rate or 'off'}  "
          f"max_failed={args.max_failed or 'off'}")
    print(f"\nMitigated system prompt:\n{cfg.SYS_TARGET}\n")

    jobs = _build_jobs(entries, paths)
    completed_ids = list_completed_conversation_ids(paths)

    n_todo = sum(1 for j in jobs if j.job_id not in completed_ids)
    # Each turn costs 1 target call plus >=1 user-agent call (actor +
    # possible reflection retries, up to max_reflect_retries); this is a
    # rough per-session call-count estimate for sizing expectations before
    # the first real ETA appears from run_jobs below.
    est_calls_per_session = cfg.N_TURNS * (1 + 1)
    print(
        f"\nTo run: {n_todo}/{len(jobs)} sessions "
        f"({len(jobs) - n_todo} already complete on disk). "
        f"~{est_calls_per_session} LLM calls/session minimum "
        f"({cfg.N_TURNS} turns x [1 target + >=1 user-agent] call), "
        f"more if reflection retries fire. Per-job elapsed/ETA prints below "
        f"once the first few sessions finish; re-run this exact command to "
        f"resume after any interruption or failure.\n"
    )

    write_manifest(paths, {
        "experiment": cfg.EXPERIMENT_NAME,
        "phase": "conversations",
        "source_run": str(cfg.SOURCE_RUN_DIR),
        "sample_index": str(args.sample_index),
        "n_sessions_planned": len(jobs),
        "user_agent": f"{cfg.USER_AGENT_PROVIDER}/{cfg.USER_AGENT_MODEL}",
        "max_reflect_retries": cfg.MAX_REFLECT_RETRIES,
        "n_turns": cfg.N_TURNS,
        "temperature_user": cfg.TEMPERATURE_USER,
        "temperature_target": cfg.TEMPERATURE_TARGET,
        "sys_target_baseline": cfg.SYS_TARGET_BASELINE,
        "mitigation_statement": cfg.MITIGATION_STATEMENT,
        "sys_target_mitigation": cfg.SYS_TARGET,
    })

    try:
        completed, failed = run_jobs(
            jobs=jobs,
            worker=_worker,
            paths=paths,
            n_workers=args.workers,
            is_done=lambda job: job.job_id in completed_ids,
            progress_label="session",
            checkpoint_name="checkpoint_conversations.json",
            max_failed=args.max_failed,
            max_failure_rate=args.max_failure_rate,
            max_same_error=args.max_same_error or None,
        )
    except KeyboardInterrupt:
        # run_jobs already drained in-flight completions and persisted the
        # checkpoint before re-raising; nothing left to do but tell the
        # user how to continue.
        print(
            f"\nStopped by user. Re-run with --resume {paths.root} to "
            "continue — already-written conversations are skipped."
        )
        raise

    print(f"\nCompleted: {len(completed)}  Failed: {len(failed)}")
    if failed:
        print("\nFailed sessions (re-run the same command, or with "
              f"--resume {paths.root}, to retry these):")
        for r in failed:
            err = str(r.info.get("error", "")).strip()
            print(f"  - {r.job_id}: {err[:160]}")

    n_untried = len(jobs) - len(completed_ids) - len(completed) - len(failed)
    if n_untried > 0:
        print(
            f"\n{n_untried} session(s) were never attempted (fail-fast "
            "likely stopped the run early). Investigate the errors above, "
            f"then re-run with --resume {paths.root} to continue."
        )


if __name__ == "__main__":
    main()
