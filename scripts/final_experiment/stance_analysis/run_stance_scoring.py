"""
stance_analysis/run_stance_scoring.py
======================================
Score every turn (1-8) of the target sessions (see
select_target_conversations.py) with the ``misinfo_stance_split``
rubric — does the model's response treat the misinformation as true,
false, or take no clear position?

Writes into the *same* scores/ directory as the main study
(``scores/<session_id>__misinfo_stance_split.json``), so downstream
code can join stance scores against the existing misinfo_v1
correction/rebuttal scores by session_id without any extra
bookkeeping.

This calls a real evaluator LLM once per turn per target session (8
calls/session) — check the target-session count from
select_target_conversations.py before running on the full set.

Usage::

    cd scripts/final_experiment
    python -m stance_analysis.run_stance_scoring [--workers 8]

Resume
------
Scoring is turn-independent (each turn's evaluator call only ever sees
that one turn's user_message/response — never the rest of the
conversation), and the worker now exploits that: it writes the
session's score file **incrementally after every turn**, not once at
the end. So if turn 5 of 8 throws (network blip, rate limit, ...), the
already-scored turns 1-4 are safely on disk — a resume only pays for
turns 5-8, not the whole session over again.

Re-running the exact same command:
  - Skips any session whose score file already has all N_TURNS turns
    (fully done).
  - For a session with a *partial* score file (some turns scored,
    some not — left behind by an interrupted prior run), only scores
    the missing turns and merges them in.
  - Every write is atomic (temp file + rename; see
    core.storage.atomic_write_json), so a file on disk is always
    either fully written for the turns present, or absent — never
    corrupted mid-write.

On Ctrl+C, core.runner.run_jobs drains in-flight completions, writes
them out, and persists checkpoint_stance_scoring.json before
re-raising — just re-run the same command to continue from where you
stopped. A turn that fails to parse after retries is written with a
-1.0 placeholder (see core.scoring) and counted in ``parse_fail`` — it
will NOT be auto-retried since the file already carries a (failed)
result for that turn; delete the session's score file to force a
full re-score.

Fail-fast: by default the run stops early if the same error message
recurs 5 times (``--max-same-error``), so a systematic problem (bad
API key, wrong model slug, exhausted credits) doesn't silently burn
through the whole target set before you notice. Override or disable
(``--max-same-error 0``) via the CLI flags below.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import (  # noqa: E402
    ConversationArtifact,
    Job,
    JobResult,
    RUBRICS,
    RunPaths,
    ScoreArtifact,
    TurnArtifact,
    TurnScores,
    atomic_write_json,
    run_jobs,
    score_conversation,
)

from main_user_IVs import config as main_cfg  # noqa: E402

from . import config as cfg  # noqa: E402
from . import scoring_progress as prog  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(threadName)-12s | %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

STANCE_RUBRIC = RUBRICS[cfg.STANCE_RUBRIC_NAME]

EVALUATOR_REGISTRY = {
    "primary":   main_cfg.EVALUATOR_PRIMARY,
    "secondary": main_cfg.EVALUATOR_SECONDARY,
}


def _score_path(session_id: str) -> Path:
    return prog.score_path(session_id, STANCE_RUBRIC.name)


def _worker(payload: dict) -> JobResult:
    sid: str = payload["session_id"]
    conv_path: Path = payload["conv_path"]
    eval_provider: str = payload["eval_provider"]
    eval_model: str = payload["eval_model"]
    evaluator_slug = f"{eval_provider}/{eval_model}"

    # NOTE: deliberately no session id / path in these messages. run_jobs'
    # --max-same-error fail-fast gate dedupes on the raw error string
    # (core.runner._error_signature) — a per-session unique message (like
    # embedding sid/path) would make every failure its own "signature" and
    # silently defeat that gate. The session id is already visible via
    # job_id in run_jobs' own FAIL line and in the failed-sessions summary
    # below, so nothing is lost by keeping the message generic.
    if not conv_path.exists():
        raise FileNotFoundError("conversation file missing")

    try:
        with conv_path.open("r", encoding="utf-8") as fh:
            conv_data = json.load(fh)
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"malformed conversation file: {type(e).__name__}: {e}") from e

    score_path = _score_path(sid)
    existing_turns = prog.load_existing_turns(score_path, expected_evaluator=evaluator_slug)
    turns_data = conv_data["turns"]
    n_total = len(turns_data)

    if existing_turns:
        logger.info(
            "%s: resuming — %d/%d turn(s) already scored, %d remaining.",
            sid, len(existing_turns), n_total, n_total - len(existing_turns),
        )

    all_turns: list[TurnScores] = []
    for turn_data in turns_data:
        turn_num = int(turn_data["turn"])

        if turn_num in existing_turns:
            all_turns.append(existing_turns[turn_num])
            continue

        # Scoring is turn-independent (see module docstring) — this call
        # only ever sees this one turn's user_message/response, never the
        # rest of the conversation, so scoring it in isolation here is
        # identical to what score_conversation would do inline.
        mini_artifact = ConversationArtifact(
            schema_version=conv_data["schema_version"],
            session_id=sid,
            experiment=conv_data["experiment"],
            cell=conv_data["cell"],
            belief=conv_data["belief"],
            models=conv_data["models"],
            config=conv_data["config"],
            turns=[TurnArtifact(
                turn=turn_data["turn"],
                user_message=turn_data["user_message"],
                target_response=turn_data["target_response"],
            )],
        )
        one_turn_art = score_conversation(
            artifact=mini_artifact,
            rubric=STANCE_RUBRIC,
            evaluator_provider=eval_provider,
            evaluator_model=eval_model,
            temperature=cfg.EVALUATOR_TEMPERATURE,
        )
        turn_scores = one_turn_art.turns[0]
        all_turns.append(turn_scores)

        stance_label = cfg.STANCE_LABELS.get(turn_scores.scores.get("stance", -1.0), "unparsed")
        logger.info(
            "%s: scored turn %d/%d  stance=%s", sid, turn_num, n_total, stance_label,
        )

        # Persist after every turn — a failure on a later turn (network
        # blip, rate limit, ...) then only costs a retry of the remaining
        # turns, not this session's already-paid-for calls.
        all_turns.sort(key=lambda t: t.turn)
        is_complete = len(all_turns) == n_total
        partial_artifact = ScoreArtifact(
            schema_version="1.0",
            session_id=sid,
            rubric_name=STANCE_RUBRIC.name,
            rubric_dimensions=list(STANCE_RUBRIC.dimensions),
            rubric_kind="multi_prompt",
            evaluator_model=evaluator_slug,
            evaluator_temperature=cfg.EVALUATOR_TEMPERATURE,
            turns=all_turns,
            completed_at=datetime.now().isoformat() if is_complete else None,
        )
        atomic_write_json(score_path, asdict(partial_artifact))

    n_parse_fail = sum(
        1 for t in all_turns for v in t.scores.values() if v == -1.0
    )
    if n_parse_fail:
        logger.warning(
            "%s: %d/%d turn(s) unparsed after retries (written with -1.0 "
            "placeholders — file exists, so it will NOT auto-retry; "
            "delete the score file to force a re-score).",
            sid, n_parse_fail, len(all_turns),
        )
    return JobResult(job_id=sid, status="ok", info={"parse_fail": n_parse_fail})


def main() -> None:
    p = argparse.ArgumentParser(
        description="Score target sessions with the misinfo_stance_split rubric."
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--evaluator", choices=list(EVALUATOR_REGISTRY), default="primary")
    p.add_argument(
        "--target-index", type=Path, default=cfg.TARGET_INDEX_PATH,
        help="Target session index written by select_target_conversations.py.",
    )
    p.add_argument(
        "--max-same-error", type=int, default=5,
        help="Stop early once the same error message has recurred this many "
             "times (catches systematic failures — bad key, wrong model "
             "slug, exhausted credits — before they burn through the whole "
             "target set). 0 disables this gate. Default: 5.",
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

    if not args.target_index.exists():
        raise SystemExit(
            f"Target index not found: {args.target_index}\n"
            "Run `python -m stance_analysis.select_target_conversations` first."
        )

    with args.target_index.open("r", encoding="utf-8") as fh:
        entries = json.load(fh)["entries"]

    eval_provider, eval_model = EVALUATOR_REGISTRY[args.evaluator]
    evaluator_slug = f"{eval_provider}/{eval_model}"
    cfg.SCORES_DIR.mkdir(parents=True, exist_ok=True)

    def _fully_scored(session_id: str) -> bool:
        return prog.is_fully_scored(
            session_id, STANCE_RUBRIC.name,
            expected_evaluator=evaluator_slug, n_turns=main_cfg.N_TURNS,
        )

    progress = prog.scan_progress(
        entries, STANCE_RUBRIC.name,
        expected_evaluator=evaluator_slug, n_turns=main_cfg.N_TURNS,
    )
    n_already_scored = progress["n_done"]

    print(f"Target sessions   : {progress['n_sessions']}  "
          f"({progress['n_done']} fully scored, {progress['n_partial']} partial, "
          f"{progress['n_not_started']} not started)")
    print(f"Evaluator calls   : {progress['n_calls_remaining']} remaining "
          f"(1 per unscored turn, {main_cfg.N_TURNS} turns/session)")
    print(f"Rubric            : {STANCE_RUBRIC.name} (dims: {STANCE_RUBRIC.dimensions})")
    print(f"Evaluator         : {eval_provider}/{eval_model}")
    print(f"Scores out        : {cfg.SCORES_DIR}")
    print(f"Workers           : {args.workers}")
    print(f"Fail-fast         : max_same_error={args.max_same_error or 'off'}  "
          f"max_failure_rate={args.max_failure_rate or 'off'}  "
          f"max_failed={args.max_failed or 'off'}\n")

    jobs = [
        Job(
            job_id=e["session_id"],
            payload={
                "session_id":   e["session_id"],
                "conv_path":    Path(e["source_path"]),
                "eval_provider": eval_provider,
                "eval_model":    eval_model,
            },
        )
        for e in entries
    ]

    paths = RunPaths(root=cfg.SOURCE_RUN_DIR)
    try:
        completed, failed = run_jobs(
            jobs=jobs,
            worker=_worker,
            paths=paths,
            n_workers=args.workers,
            is_done=lambda j: _fully_scored(j.job_id),
            progress_label="stance-scoring",
            checkpoint_name="checkpoint_stance_scoring.json",
            max_failed=args.max_failed,
            max_failure_rate=args.max_failure_rate,
            max_same_error=args.max_same_error or None,
        )
    except KeyboardInterrupt:
        # run_jobs already drained + persisted the checkpoint before
        # re-raising; nothing left to do but tell the user how to continue.
        print(
            "\nStopped by user. Re-run the exact same command to resume — "
            "already-scored sessions (score file on disk) are skipped."
        )
        raise

    n_parse_fail = sum(r.info.get("parse_fail", 0) for r in completed)
    print(f"\nCompleted: {len(completed)}  Failed: {len(failed)}  "
          f"(turns with parse failures: {n_parse_fail})")

    if failed:
        print("\nFailed sessions (re-run the same command to retry these):")
        for r in failed:
            err = str(r.info.get("error", "")).strip()
            print(f"  - {r.job_id}: {err[:160]}")

    n_untried = len(entries) - n_already_scored - len(completed) - len(failed)
    if n_untried > 0:
        # Fail-fast aborted the run before consuming every queued job.
        print(
            f"\n{n_untried} session(s) were never attempted (fail-fast likely "
            f"stopped the run early). Investigate the errors above, then "
            f"re-run the same command to continue."
        )


if __name__ == "__main__":
    main()
