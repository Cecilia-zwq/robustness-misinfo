"""
main_user_IVs/run_conversations.py
==================================
Stage 1 of RQ1: generate the conversation corpus.

Produces 9 cells × N beliefs × 4 target models conversation artifacts
under results/main_user_IVs/<timestamp>/conversations/. No scoring is
performed here — scores are populated later by run_scoring.py.

Usage
-----
  cd scripts/final_experiment
  python -m main_user_IVs.run_conversations [--workers 16] [--resume <dir>]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python -m main_user_IVs.run_conversations` from the parent directory.
# Adds scripts/final_experiment to sys.path so `core` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import (  # noqa: E402
    AgentSimulation,
    Condition,
    Job,
    JobResult,
    build_session_id,
    list_completed_conversation_ids,
    load_beliefs,
    make_run_paths,
    run_conversation,
    run_jobs,
    stage1_main_effect_conditions,
    write_conversation,
    write_manifest,
)

from . import config as cfg  # noqa: E402

from misinfo_eval_framework import SimulatedUserAgent, TargetLLM  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(threadName)-12s | %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Worker
# ════════════════════════════════════════════════════════════════════════════

def _worker(payload: dict) -> JobResult:
    """One session: build agent + target, run conversation, persist artifact."""
    condition: Condition = payload["condition"]
    belief: dict = payload["belief"]
    target_provider: str = payload["target_provider"]
    target_model: str = payload["target_model"]
    session_id: str = payload["session_id"]
    paths = payload["paths"]

    is_long = belief.get("is_long_text", False)

    agent = SimulatedUserAgent(
        provider=cfg.USER_AGENT_PROVIDER,
        model=cfg.USER_AGENT_MODEL,
        character_prompt=condition.character_prompt(),
        is_long_text=is_long,
        max_reflect_retries=cfg.MAX_REFLECT_RETRIES,
        temperature=cfg.TEMPERATURE_USER,
    )
    target = TargetLLM(
        provider=target_provider,
        model=target_model,
        temperature=cfg.TEMPERATURE_TARGET,
        system_prompt=cfg.SYS_TARGET
    )

    artifact = run_conversation(
        session_id=session_id,
        experiment=cfg.EXPERIMENT_NAME,
        cell=condition.to_dict(),
        belief=belief,
        user_simulation=AgentSimulation(agent),
        target=target,
        n_turns=cfg.N_TURNS,
    )
    write_conversation(paths, artifact)

    n_char = sum(t.n_character_breaks for t in artifact.turns)
    n_belief = sum(t.n_belief_breaks for t in artifact.turns)
    n_fb = sum(1 for t in artifact.turns if t.is_fallback)

    return JobResult(
        job_id=session_id,
        status="ok",
        info={"char_brk": n_char, "bel_brk": n_belief, "fb": n_fb},
    )


# ════════════════════════════════════════════════════════════════════════════
# Manifest builder
# ════════════════════════════════════════════════════════════════════════════

def _build_jobs(paths) -> list[Job]:
    beliefs = load_beliefs(cfg.BELIEFS_PATH)
    conditions = stage1_main_effect_conditions()

    print(f"\nLoaded {len(beliefs)} beliefs from {cfg.BELIEFS_PATH.name}.")
    print(f"Conditions ({len(conditions)}):")
    for c in conditions:
        print(f"  {c.cell_id}")
    print(f"Target models ({len(cfg.TARGET_LLMS)}):")
    for prov, mdl, slug in cfg.TARGET_LLMS:
        print(f"  {prov}/{mdl}  (slug: {slug})")

    jobs: list[Job] = []
    for cond in conditions:
        for belief in beliefs:
            for prov, mdl, slug in cfg.TARGET_LLMS:
                sid = build_session_id(
                    cell_id=cond.cell_id,
                    belief_category=belief["category"],
                    belief_index=belief["belief_index"],
                    target_model_short=slug,
                )
                jobs.append(Job(
                    job_id=sid,
                    payload={
                        "condition": cond,
                        "belief": belief,
                        "target_provider": prov,
                        "target_model": mdl,
                        "session_id": sid,
                        "paths": paths,
                    },
                ))
    print(f"\nTotal sessions planned: {len(jobs)}")
    return jobs


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be >= 1")

    paths = make_run_paths(
        cfg.RESULTS_DIR,
        cfg.EXPERIMENT_NAME,
        resume_dir=args.resume,
    )
    print(f"\nResults dir: {paths.root}")
    print(f"Conversations: {paths.conversations}")

    jobs = _build_jobs(paths)
    completed_ids = list_completed_conversation_ids(paths)

    write_manifest(paths, {
        "experiment": cfg.EXPERIMENT_NAME,
        "phase": "conversations",
        "n_sessions_planned": len(jobs),
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
        progress_label="session",
        checkpoint_name="checkpoint_conversations.json",
    )


if __name__ == "__main__":
    main()
