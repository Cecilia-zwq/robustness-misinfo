"""
static_interactive_ablation/run_conversation_static.py
=======================================================
Static (opening-only) interactivity ablation.

For each session in the reflection ablation sample index, pre-generates
N_TURNS messages using generate_opening (the opening prompt, no
conversation history), wraps them in StaticReplaySimulation, and runs
a fresh conversation against the same target LLM as the source session.

The only variable changed from the original interactive run is user
*adaptivity*: the target sees the same character and belief but the user
messages are a fixed script rather than replies shaped by the target's
prior responses.

Outputs: conversations_static/<session_id>__static.json
Scores:  run main_user_IVs/run_scoring.py --conversations-dir conversations_static
         --scores-dir scores_static after this script completes.

Usage
-----
  cd scripts/final_experiment
  python -m static_interactive_ablation.run_conversation_static [--workers 8]
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
    RunPaths,
    StaticReplaySimulation,
    atomic_write_json,
    atomic_write_text,
    format_belief_for_agent,
    run_conversation,
)
from misinfo_eval_framework import SimulatedUserAgent, TargetLLM  # noqa: E402

from . import config as cfg  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(threadName)-12s | %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Session-id helpers
# ════════════════════════════════════════════════════════════════════════════

def _static_session_id(source_session_id: str) -> str:
    return f"{source_session_id}{cfg.SESSION_ID_SUFFIX}"


def _output_path(session_id: str) -> Path:
    return cfg.OUTPUT_CONV_DIR / f"{session_id}.json"


# ════════════════════════════════════════════════════════════════════════════
# Worker
# ════════════════════════════════════════════════════════════════════════════

def _worker(payload: dict) -> JobResult:
    """One session: load source → pre-generate opening script → run → persist."""
    source_path: Path = payload["source_path"]
    new_session_id: str = payload["new_session_id"]
    output_path: Path = payload["output_path"]

    # ── Load source artifact for cell / belief / target metadata ────────────
    with source_path.open("r", encoding="utf-8") as fh:
        source = ConversationArtifact.from_dict(json.load(fh))

    character_prompt = source.cell.get("character_prompt", "")
    is_long_text = bool(source.belief.get("is_long_text", False))
    n_turns = int(source.config.get("n_turns", cfg.N_TURNS))

    # ── Build the user agent ─────────────────────────────────────────────────
    agent = SimulatedUserAgent(
        provider=cfg.USER_AGENT_PROVIDER,
        model=cfg.USER_AGENT_MODEL,
        character_prompt=character_prompt,
        is_long_text=is_long_text,
        max_reflect_retries=cfg.MAX_REFLECT_RETRIES,
        temperature=cfg.TEMPERATURE_USER,
    )

    # ── Pre-generate N opening messages (no conversation history) ────────────
    misinformation_belief = format_belief_for_agent(source.belief)
    agent_sim = AgentSimulation(agent)
    messages, all_attempts = [], []
    for _ in range(n_turns):
        result = agent_sim.generate(
            turn_idx=1,
            conversation_history=[],
            misinformation_belief=misinformation_belief,
        )
        messages.append(result.user_message)
        all_attempts.append(result.reflection_attempts)

    source_label = f"{cfg.USER_AGENT_PROVIDER}/{cfg.USER_AGENT_MODEL} (opening-only)"
    user_sim = StaticReplaySimulation.from_messages(
        messages, source_label=source_label, reflection_attempts=all_attempts
    )

    # ── Build the target (same model as source session) ──────────────────────
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

    # ── Run fresh conversation (from turn 1, not branched) ───────────────────
    artifact = run_conversation(
        session_id=new_session_id,
        experiment=cfg.EXPERIMENT_NAME,
        cell=source.cell,
        belief=source.belief,
        user_simulation=user_sim,
        target=target,
        n_turns=n_turns,
    )

    atomic_write_text(output_path, artifact.to_json())

    n_char = sum(t.n_character_breaks for t in artifact.turns)
    n_bel  = sum(t.n_belief_breaks    for t in artifact.turns)
    n_fb   = sum(1 for t in artifact.turns if t.is_fallback)

    return JobResult(
        job_id=new_session_id,
        status="ok",
        info={"char_brk": n_char, "bel_brk": n_bel, "fb": n_fb},
    )


# ════════════════════════════════════════════════════════════════════════════
# Job builder
# ════════════════════════════════════════════════════════════════════════════

def _load_sample_index(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        index = json.load(fh)
    return index.get("entries", [])


def _list_completed_outputs() -> set[str]:
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


def _build_jobs(entries: list[dict]) -> list[Job]:
    jobs: list[Job] = []
    for entry in entries:
        source_sid = entry["source_session_id"]
        new_sid    = _static_session_id(source_sid)
        src_path   = Path(entry["source_path"])
        out_path   = _output_path(new_sid)
        jobs.append(Job(
            job_id=new_sid,
            payload={
                "source_path":    src_path,
                "new_session_id": new_sid,
                "output_path":    out_path,
            },
        ))
    return jobs


# ════════════════════════════════════════════════════════════════════════════
# Manifest
# ════════════════════════════════════════════════════════════════════════════

def _write_manifest(*, n_planned: int, n_workers: int) -> None:
    atomic_write_json(cfg.MANIFEST_PATH, {
        "experiment":              cfg.EXPERIMENT_NAME,
        "phase":                   "static_ablation_conversations",
        "source_run":              str(cfg.SOURCE_RUN_DIR),
        "source_conversations_dir": str(cfg.SOURCE_CONV_DIR),
        "output_conversations_dir": str(cfg.OUTPUT_CONV_DIR),
        "sample_index":            str(cfg.SAMPLE_INDEX_PATH),
        "n_sessions_planned":      n_planned,
        "n_workers":               n_workers,
        "user_agent":              f"{cfg.USER_AGENT_PROVIDER}/{cfg.USER_AGENT_MODEL}",
        "user_agent_mode":         "opening-only (static)",
        "max_reflect_retries":     cfg.MAX_REFLECT_RETRIES,
        "temperature_user":        cfg.TEMPERATURE_USER,
        "temperature_target":      cfg.TEMPERATURE_TARGET,
        "n_turns":                 cfg.N_TURNS,
    })


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel worker threads (default 8).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print job counts without running any sessions.")
    p.add_argument("--max-failed", type=int, default=None)
    p.add_argument("--max-failure-rate", type=float, default=None)
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be >= 1")

    if not cfg.SAMPLE_INDEX_PATH.exists():
        raise SystemExit(f"Sample index not found: {cfg.SAMPLE_INDEX_PATH}")
    if not cfg.SOURCE_CONV_DIR.exists():
        raise SystemExit(f"Source conversations dir not found: {cfg.SOURCE_CONV_DIR}")

    cfg.OUTPUT_CONV_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nSource conversations : {cfg.SOURCE_CONV_DIR}")
    print(f"Sample index        : {cfg.SAMPLE_INDEX_PATH}")
    print(f"Output conversations: {cfg.OUTPUT_CONV_DIR}")
    print(f"User agent          : {cfg.USER_AGENT_PROVIDER}/{cfg.USER_AGENT_MODEL}")

    entries = _load_sample_index(cfg.SAMPLE_INDEX_PATH)
    jobs    = _build_jobs(entries)
    print(f"Sessions planned    : {len(jobs)}")

    if args.dry_run:
        completed = _list_completed_outputs()
        remaining = [j for j in jobs if j.job_id not in completed]
        print(f"\nDry run: {len(completed)} already done, {len(remaining)} to run.")
        return

    _write_manifest(n_planned=len(jobs), n_workers=args.workers)
    completed_ids = _list_completed_outputs()

    paths = RunPaths(root=cfg.SOURCE_RUN_DIR)

    from core.runner import run_jobs  # noqa: E402

    run_jobs(
        jobs=jobs,
        worker=_worker,
        paths=paths,
        n_workers=args.workers,
        is_done=lambda job: job.job_id in completed_ids,
        progress_label="static-ablation",
        checkpoint_name=cfg.CHECKPOINT_PATH.name,
        max_failed=args.max_failed,
        max_failure_rate=args.max_failure_rate,
    )


if __name__ == "__main__":
    main()
