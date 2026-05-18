"""
main_user_IVs/run_conversations.py
==================================
Stage 1 of RQ1: generate the conversation corpus.

Produces 10 cells × N beliefs × 4 target models conversation artifacts
under results/main_user_IVs/<timestamp>/conversations/. No scoring is
performed here — scores are populated later by run_scoring.py.

Usage
-----
  cd scripts/final_experiment
  python -m main_user_IVs.run_conversations [--workers 16] [--resume <dir>]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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
    user_provider: str = payload["user_provider"]
    user_model: str = payload["user_model"]
    max_reflect_retries: int = payload["max_reflect_retries"]
    session_id: str = payload["session_id"]
    paths = payload["paths"]

    is_long = belief.get("is_long_text", False)

    agent = SimulatedUserAgent(
        provider=user_provider,
        model=user_model,
        character_prompt=condition.character_prompt(),
        is_long_text=is_long,
        max_reflect_retries=max_reflect_retries,
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

def _load_sample_index_ids(sample_index_path: Path) -> set[str]:
    """Read an ablation_sample_index.json and return its source session ids."""
    with open(sample_index_path, "r", encoding="utf-8") as fh:
        index = json.load(fh)
    entries = index.get("entries") or []
    return {str(e["source_session_id"]) for e in entries}


def _build_jobs(
    paths,
    *,
    user_provider: str,
    user_model: str,
    max_reflect_retries: int,
    iv1_filter: str | None = None,
    iv2_filter: str | None = None,
    sample_index_path: Path | None = None,
) -> list[Job]:
    beliefs = load_beliefs(cfg.BELIEFS_PATH)
    conditions = stage1_main_effect_conditions()

    if iv1_filter is not None:
        before = len(conditions)
        conditions = [c for c in conditions if c.iv1 == iv1_filter]
        print(f"\n--iv1 {iv1_filter!r} filter: {before} → {len(conditions)} conditions.")
    if iv2_filter is not None:
        before = len(conditions)
        conditions = [c for c in conditions if c.iv2 == iv2_filter]
        print(f"\n--iv2 {iv2_filter!r} filter: {before} → {len(conditions)} conditions.")
    if not conditions:
        raise SystemExit(
            "No conditions left after IV filtering — check --iv1/--iv2 values."
        )

    print(f"\nLoaded {len(beliefs)} beliefs from {cfg.BELIEFS_PATH.name}.")
    print(f"User agent: {user_provider}/{user_model}")
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
                        "user_provider": user_provider,
                        "user_model": user_model,
                        "max_reflect_retries": max_reflect_retries,
                        "session_id": sid,
                        "paths": paths,
                    },
                ))

    # Restrict to the sessions named in an ablation_sample_index.json.
    # Used to regenerate just the ablation subsample (e.g. with a
    # different user-agent model) instead of the whole cross-product.
    if sample_index_path is not None:
        wanted = _load_sample_index_ids(sample_index_path)
        before = len(jobs)
        jobs = [j for j in jobs if j.job_id in wanted]
        matched = {j.job_id for j in jobs}
        missing = wanted - matched
        print(f"\n--sample-index filter: {before} → {len(jobs)} jobs "
              f"({len(wanted)} requested).")
        if missing:
            print(f"  WARNING: {len(missing)} sample-index session id(s) did not "
                  f"match any planned job (belief pool may have changed). "
                  f"Examples: {sorted(missing)[:3]}")
        if not jobs:
            raise SystemExit(
                "No jobs left after --sample-index filtering — check that the "
                "index was built from a compatible belief pool / condition set."
            )

    print(f"\nTotal sessions planned: {len(jobs)}")
    return jobs


def _apply_limit(jobs: list[Job], limit: int | None, seed: int = 0) -> list[Job]:
    if limit is None or limit <= 0:
        return jobs
    if limit < len(jobs):
        rng = random.Random(seed)
        sampled = rng.sample(jobs, limit)
        print(f"\n--limit {limit} (seed={seed}): sampled {limit} of {len(jobs)} sessions.")
        return sampled
    return jobs


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument(
        "--iv1", default=None,
        help="Restrict to conditions whose IV1 level equals this value "
             "(e.g. 'none'). Omit to keep all IV1 levels.",
    )
    p.add_argument(
        "--iv2", default=None,
        help="Restrict to conditions whose IV2 level equals this value "
             "(e.g. 'none' to run only the IV2-none cohort, then resume "
             "without the flag for the remaining IV2 cells). Omit to keep "
             "all IV2 levels.",
    )
    p.add_argument(
        "--user-provider", default=None,
        help=f"Override the user-agent provider (default cfg.USER_AGENT_PROVIDER="
             f"{cfg.USER_AGENT_PROVIDER!r}). Use 'ollama_chat' for a local "
             f"Ollama model served at localhost:11434.",
    )
    p.add_argument(
        "--user-model", default=None,
        help=f"Override the user-agent model (default cfg.USER_AGENT_MODEL="
             f"{cfg.USER_AGENT_MODEL!r}). E.g. 'llama2-uncensored' together "
             f"with --user-provider ollama_chat.",
    )
    p.add_argument(
        "--sample-index", default=None,
        help="Path to an ablation_sample_index.json. When given, only the "
             "source sessions listed in that file are (re)generated; the "
             "rest of the condition x belief x target cross-product is skipped.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Randomly sample N sessions after all other filtering. "
             "Useful for quick smoke tests (e.g. --limit 10).",
    )
    p.add_argument(
        "--limit-seed", type=int, default=0,
        help="Random seed for --limit sampling (default 0). Change to get a "
             "different random subset.",
    )
    p.add_argument(
        "--max-reflect-retries", type=int, default=None,
        help=f"Override cfg.MAX_REFLECT_RETRIES ({cfg.MAX_REFLECT_RETRIES}). "
             f"Increase for local models that need more attempts to pass the "
             f"reflection auditor (e.g. --max-reflect-retries 6).",
    )
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be >= 1")
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be >= 1")
    if args.max_reflect_retries is not None and args.max_reflect_retries < 0:
        p.error("--max-reflect-retries must be >= 0")

    user_provider = args.user_provider or cfg.USER_AGENT_PROVIDER
    user_model = args.user_model or cfg.USER_AGENT_MODEL
    max_reflect_retries = (
        args.max_reflect_retries if args.max_reflect_retries is not None
        else cfg.MAX_REFLECT_RETRIES
    )

    sample_index_path: Path | None = None
    if args.sample_index is not None:
        sample_index_path = Path(args.sample_index)
        if not sample_index_path.exists():
            p.error(f"--sample-index file not found: {sample_index_path}")

    paths = make_run_paths(
        cfg.RESULTS_DIR,
        cfg.EXPERIMENT_NAME,
        resume_dir=args.resume,
    )
    print(f"\nResults dir: {paths.root}")
    print(f"Conversations: {paths.conversations}")

    jobs = _build_jobs(
        paths,
        user_provider=user_provider,
        user_model=user_model,
        max_reflect_retries=max_reflect_retries,
        iv1_filter=args.iv1,
        iv2_filter=args.iv2,
        sample_index_path=sample_index_path,
    )
    jobs = _apply_limit(jobs, args.limit, seed=args.limit_seed)
    completed_ids = list_completed_conversation_ids(paths)

    write_manifest(paths, {
        "experiment": cfg.EXPERIMENT_NAME,
        "phase": "conversations",
        "n_sessions_planned": len(jobs),
        "user_agent": f"{user_provider}/{user_model}",
        "max_reflect_retries": max_reflect_retries,
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
