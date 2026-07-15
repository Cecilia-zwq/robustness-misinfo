"""
stance_analysis/run_stance_history_scoring.py
================================================
Score every turn of the sampled sessions (see select_history_sample.py)
with the ``misinfo_stance_history_split`` rubric (current prompt
wording in core/scoring.py) — the history-aware variant of
misinfo_stance_split: the evaluator sees every prior turn as context,
but scores only the current turn.

Writes into the *same* scores/ directory as the rest of the study
(``scores/<session_id>__<rubric-name>.json``, where rubric-name is
``misinfo_stance_history_split`` + config.HISTORY_RUN_SUFFIX, e.g.
``misinfo_stance_history_split_1``) — distinct filename both from the
turn-independent ``misinfo_stance_split`` rubric and from any earlier
history-run suffix, so old and new runs (and the turn-independent
rubric) all coexist and can be compared per session/turn. Bump
config.HISTORY_RUN_SUFFIX whenever you change the prompt wording and
want a fresh, non-overwriting re-score of the same sample — see that
constant's docstring.

This calls a real evaluator LLM once per turn per sampled session (8
calls/session) — check the sample-session count from
select_history_sample.py before running on the full set.

Usage::

    cd scripts/final_experiment
    python -m stance_analysis.run_stance_history_scoring [--workers 8]

Resume
------
Unlike the turn-independent rubric, a history-aware turn's evaluator
call depends on the *real* prior turns' text — but not on their
*scores*. That text is always fully known upfront from the
conversation file, regardless of how much of this session has been
scored so far, so turns can still be scored one at a time and persisted
incrementally with **zero wasted calls**: scoring turn 5 never requires
re-scoring turns 1-4 to reconstruct their history, it just reads their
real user_message/target_response directly off the conversation
artifact (see core.scoring.score_one_turn).

So the resume contract is identical to run_stance_scoring.py's:
  - Skips any session whose score file already has all N_TURNS turns.
  - For a session with a *partial* score file, only scores the missing
    turns and merges them in.
  - Every write is atomic; a file on disk is always either fully
    written for the turns present, or absent — never corrupted mid-write.
  - On Ctrl+C, core.runner.run_jobs drains in-flight completions and
    persists a checkpoint before re-raising; re-run the same command to
    continue.
  - A turn that fails to parse after retries is written with a -1.0
    placeholder and counted in ``parse_fail`` — it will NOT be
    auto-retried since the file already carries a result for that turn;
    delete the session's score file to force a full re-score.

Fail-fast: by default the run stops early if the same error message
recurs 5 times (``--max-same-error``). Override or disable
(``--max-same-error 0``) via the CLI flags below.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, replace as _dc_replace
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
    TurnScores,
    atomic_write_json,
    run_jobs,
    score_one_turn,
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

# The rubric registered in core.scoring always carries the CURRENT prompt
# wording; cfg.HISTORY_STANCE_RUBRIC_NAME may have a run suffix appended
# (e.g. "_1") so this run's score files don't collide with an earlier
# run's. dataclasses.replace swaps just the `name` — prompt content is
# untouched, so this always scores with whatever's in core/scoring.py
# right now, under a filename that won't be resume-skipped against an
# older run.
HISTORY_STANCE_RUBRIC = _dc_replace(
    RUBRICS["misinfo_stance_history_split"], name=cfg.HISTORY_STANCE_RUBRIC_NAME,
)

EVALUATOR_REGISTRY = {
    "primary":   main_cfg.EVALUATOR_PRIMARY,
    "secondary": main_cfg.EVALUATOR_SECONDARY,
}


def _score_path(session_id: str) -> Path:
    return prog.score_path(session_id, HISTORY_STANCE_RUBRIC.name)


def _worker(payload: dict) -> JobResult:
    sid: str = payload["session_id"]
    conv_path: Path = payload["conv_path"]
    eval_provider: str = payload["eval_provider"]
    eval_model: str = payload["eval_model"]
    evaluator_slug = f"{eval_provider}/{eval_model}"

    # NOTE: deliberately no session id / path in these messages — see
    # run_stance_scoring.py's identical note on why (--max-same-error
    # dedupes on the raw error string).
    if not conv_path.exists():
        raise FileNotFoundError("conversation file missing")

    try:
        with conv_path.open("r", encoding="utf-8") as fh:
            artifact = ConversationArtifact.from_dict(json.load(fh))
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"malformed conversation file: {type(e).__name__}: {e}") from e

    score_path = _score_path(sid)
    existing_turns = prog.load_existing_turns(score_path, expected_evaluator=evaluator_slug)
    n_total = len(artifact.turns)

    if existing_turns:
        logger.info(
            "%s: resuming — %d/%d turn(s) already scored, %d remaining.",
            sid, len(existing_turns), n_total, n_total - len(existing_turns),
        )

    all_turns: list[TurnScores] = []
    for turn_index, turn in enumerate(artifact.turns):
        turn_num = int(turn.turn)

        if turn_num in existing_turns:
            all_turns.append(existing_turns[turn_num])
            continue

        # score_one_turn derives history from the REAL artifact's prior
        # turns (already-known text — free), not from previously-scored
        # results, so this is correct and costs exactly 1 call
        # regardless of resume state. See module docstring.
        scores, raw, n_attempts = score_one_turn(
            artifact=artifact,
            turn_index=turn_index,
            rubric=HISTORY_STANCE_RUBRIC,
            evaluator_provider=eval_provider,
            evaluator_model=eval_model,
            temperature=cfg.EVALUATOR_TEMPERATURE,
        )
        turn_scores = TurnScores(
            turn=turn_num, scores=scores, raw_output=raw, n_parse_attempts=n_attempts,
        )
        all_turns.append(turn_scores)

        stance_label = cfg.STANCE_LABELS.get(turn_scores.scores.get("stance", -1.0), "unparsed")
        logger.info(
            "%s: scored turn %d/%d  stance=%s", sid, turn_num, n_total, stance_label,
        )

        # Persist after every turn — a failure on a later turn only
        # costs a retry of the remaining turns, not this session's
        # already-paid-for calls.
        all_turns.sort(key=lambda t: t.turn)
        is_complete = len(all_turns) == n_total
        partial_artifact = ScoreArtifact(
            schema_version="1.0",
            session_id=sid,
            rubric_name=HISTORY_STANCE_RUBRIC.name,
            rubric_dimensions=list(HISTORY_STANCE_RUBRIC.dimensions),
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
        description="Score sampled sessions with the misinfo_stance_history_split rubric."
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--evaluator", choices=list(EVALUATOR_REGISTRY), default="primary")
    p.add_argument(
        "--sample-index", type=Path, default=cfg.HISTORY_SAMPLE_INDEX_PATH,
        help="Sample index written by select_history_sample.py.",
    )
    p.add_argument(
        "--max-same-error", type=int, default=5,
        help="Stop early once the same error message has recurred this many "
             "times. 0 disables this gate. Default: 5.",
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

    if not args.sample_index.exists():
        raise SystemExit(
            f"Sample index not found: {args.sample_index}\n"
            "Run `python -m stance_analysis.select_history_sample` first."
        )

    with args.sample_index.open("r", encoding="utf-8") as fh:
        entries = json.load(fh)["entries"]

    eval_provider, eval_model = EVALUATOR_REGISTRY[args.evaluator]
    evaluator_slug = f"{eval_provider}/{eval_model}"
    cfg.SCORES_DIR.mkdir(parents=True, exist_ok=True)

    def _fully_scored(session_id: str) -> bool:
        return prog.is_fully_scored(
            session_id, HISTORY_STANCE_RUBRIC.name,
            expected_evaluator=evaluator_slug, n_turns=main_cfg.N_TURNS,
        )

    progress = prog.scan_progress(
        entries, HISTORY_STANCE_RUBRIC.name,
        expected_evaluator=evaluator_slug, n_turns=main_cfg.N_TURNS,
    )
    n_already_scored = progress["n_done"]

    print(f"Sample sessions   : {progress['n_sessions']}  "
          f"({progress['n_done']} fully scored, {progress['n_partial']} partial, "
          f"{progress['n_not_started']} not started)")
    print(f"Evaluator calls   : {progress['n_calls_remaining']} remaining "
          f"(1 per unscored turn, {main_cfg.N_TURNS} turns/session)")
    print(f"Rubric            : {HISTORY_STANCE_RUBRIC.name} (dims: {HISTORY_STANCE_RUBRIC.dimensions})")
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
                "session_id":    e["session_id"],
                "conv_path":     Path(e["source_path"]),
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
            progress_label="stance-history-scoring",
            checkpoint_name=f"checkpoint_stance_history_scoring{cfg.HISTORY_RUN_SUFFIX}.json",
            max_failed=args.max_failed,
            max_failure_rate=args.max_failure_rate,
            max_same_error=args.max_same_error or None,
        )
    except KeyboardInterrupt:
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
        print(
            f"\n{n_untried} session(s) were never attempted (fail-fast likely "
            f"stopped the run early). Investigate the errors above, then "
            f"re-run the same command to continue."
        )


if __name__ == "__main__":
    main()
