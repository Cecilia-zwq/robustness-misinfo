"""
mitigation_plan/score_history_stance.py
=========================================
Score the MITIGATED conversations with the misinfo_stance_history_split
rubric — the same history-aware rubric (evaluator sees every prior turn
as context, but still scores only the current turn) that
stance_analysis/plot_stance_history.py uses for its baseline population
figures. compare_to_baseline.py originally used misinfo_stance_split
(turn-independent, no history) for both sides, which is NOT what
plot_stance_history.py's figures are built from — this script (mitigated
side) plus reusing stance_analysis's own scorer (baseline side) closes
that gap so the two are a fair comparison.

For the BASELINE side, reuse stance_analysis's own scorer directly,
pointed at this study's sample index instead of its own — it already
writes into the same scores/ dir compare_to_baseline.py reads for
BASELINE_SCORES_DIR, and skips anything already scored (most sessions
overlap with stance_analysis's own 680-session history-sample pool and
are already done; only sessions sourced from elsewhere, e.g. the
reused Claude disengagement pool, need fresh calls):

    python -m stance_analysis.run_stance_history_scoring \\
        --sample-index <mitigation_sample_index.json>

This script is the MITIGATED-side counterpart: same rubric, same
incremental/resumable per-turn design (copied from
stance_analysis/run_stance_history_scoring.py), but pointed at the
mitigation run's own conversations/scores dirs and mitigated session ids
(<source_session_id><config.SESSION_ID_SUFFIX>). Written under the SAME
name as the baseline side — config.HISTORY_STANCE_RUBRIC_NAME, sourced
directly from stance_analysis.config.HISTORY_STANCE_RUBRIC_NAME (its
"_1"-suffixed current-prompt-wording convention). Both sides are freshly
scored under the current prompt wording, so there's no reason for the
names to differ — and stance_analysis's own scores/ dir already has an
*unsuffixed* "misinfo_stance_history_split" from an older, since-changed
prompt wording that was explicitly excluded from every current analysis,
so reusing that unsuffixed name here would misleadingly suggest this is
the same kind of stale run.

Usage::

    cd scripts/final_experiment
    python -m mitigation_plan.score_history_stance
    python -m mitigation_plan.score_history_stance --sample-index <path> --run-dir <path>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from dataclasses import replace as _dc_replace
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(threadName)-12s | %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# dataclasses.replace swaps just the `name` (matching
# cfg.HISTORY_STANCE_RUBRIC_NAME, same name the baseline side uses) —
# prompt content is untouched.
HISTORY_STANCE_RUBRIC = _dc_replace(
    RUBRICS["misinfo_stance_history_split"], name=cfg.HISTORY_STANCE_RUBRIC_NAME,
)

EVALUATOR_REGISTRY = {
    "primary":   main_cfg.EVALUATOR_PRIMARY,
    "secondary": main_cfg.EVALUATOR_SECONDARY,
}


def _latest_run_dir() -> Path | None:
    root = cfg.RESULTS_DIR / cfg.EXPERIMENT_NAME
    if not root.exists():
        return None
    candidates = sorted(p for p in root.iterdir() if p.is_dir())
    return candidates[-1] if candidates else None


def _score_path(scores_dir: Path, session_id: str) -> Path:
    return scores_dir / f"{session_id}__{HISTORY_STANCE_RUBRIC.name}.json"


def _load_existing_turns(score_path: Path, expected_evaluator: str) -> dict[int, TurnScores]:
    if not score_path.exists():
        return {}
    try:
        with score_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}
    if data.get("evaluator_model") != expected_evaluator:
        return {}
    return {
        int(t["turn"]): TurnScores(
            turn=int(t["turn"]), scores=t["scores"],
            raw_output=t.get("raw_output", ""), n_parse_attempts=t.get("n_parse_attempts", 1),
        )
        for t in data.get("turns", [])
    }


def _is_fully_scored(score_path: Path, expected_evaluator: str, n_turns: int) -> bool:
    existing = _load_existing_turns(score_path, expected_evaluator)
    return len(existing) >= n_turns


def _worker(payload: dict) -> JobResult:
    sid: str = payload["session_id"]
    conv_path: Path = payload["conv_path"]
    scores_dir: Path = payload["scores_dir"]
    eval_provider: str = payload["eval_provider"]
    eval_model: str = payload["eval_model"]
    evaluator_slug = f"{eval_provider}/{eval_model}"

    if not conv_path.exists():
        raise FileNotFoundError("conversation file missing")

    try:
        with conv_path.open("r", encoding="utf-8") as fh:
            artifact = ConversationArtifact.from_dict(json.load(fh))
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"malformed conversation file: {type(e).__name__}: {e}") from e

    score_path = _score_path(scores_dir, sid)
    existing_turns = _load_existing_turns(score_path, expected_evaluator=evaluator_slug)
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
        # results, so this costs exactly 1 call regardless of resume
        # state — same incremental design as
        # stance_analysis/run_stance_history_scoring.py.
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
        logger.info("%s: scored turn %d/%d  stance=%s", sid, turn_num, n_total, stance_label)

        all_turns.sort(key=lambda t: t.turn)
        is_complete = len(all_turns) == n_total
        partial_artifact = ScoreArtifact(
            schema_version="1.0",
            session_id=sid,
            rubric_name=HISTORY_STANCE_RUBRIC.name,
            rubric_dimensions=list(HISTORY_STANCE_RUBRIC.dimensions),
            rubric_kind="multi_prompt",
            evaluator_model=evaluator_slug,
            evaluator_temperature=main_cfg.EVALUATOR_TEMPERATURE,
            turns=all_turns,
            completed_at=datetime.now().isoformat() if is_complete else None,
        )
        atomic_write_json(score_path, asdict(partial_artifact))

    n_parse_fail = sum(1 for t in all_turns for v in t.scores.values() if v == -1.0)
    if n_parse_fail:
        logger.warning(
            "%s: %d/%d turn(s) unparsed after retries.", sid, n_parse_fail, len(all_turns),
        )
    return JobResult(job_id=sid, status="ok", info={"parse_fail": n_parse_fail})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--evaluator", choices=list(EVALUATOR_REGISTRY), default="primary")
    p.add_argument(
        "--sample-index", type=Path, default=cfg.GENERAL_SAMPLE_INDEX_PATH,
        help="Sample index (default: config.GENERAL_SAMPLE_INDEX_PATH).",
    )
    p.add_argument(
        "--run-dir", type=Path, default=None,
        help="Mitigation run dir. Defaults to the most recently created run "
             "under results/final_experiment/mitigation_plan/, or the "
             "MITIGATION_RUN_DIR env var if set.",
    )
    p.add_argument("--max-same-error", type=int, default=5)
    p.add_argument("--max-failure-rate", type=float, default=None)
    p.add_argument("--max-failed", type=int, default=None)
    args = p.parse_args()

    if not args.sample_index.exists():
        raise SystemExit(f"Sample index not found: {args.sample_index}")

    run_dir = args.run_dir
    if run_dir is None:
        env_run_dir = os.environ.get("MITIGATION_RUN_DIR")
        run_dir = Path(env_run_dir).expanduser().resolve() if env_run_dir else _latest_run_dir()
    if run_dir is None or not run_dir.exists():
        raise SystemExit(
            f"No mitigation run dir found under {cfg.RESULTS_DIR / cfg.EXPERIMENT_NAME}. "
            "Pass --run-dir or set MITIGATION_RUN_DIR."
        )

    conv_dir = run_dir / "conversations"
    scores_dir = run_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    with args.sample_index.open("r", encoding="utf-8") as fh:
        entries = json.load(fh)["entries"]

    eval_provider, eval_model = EVALUATOR_REGISTRY[args.evaluator]
    evaluator_slug = f"{eval_provider}/{eval_model}"

    jobs = []
    n_missing_conv = 0
    for e in entries:
        mit_sid = f"{e['session_id']}{cfg.SESSION_ID_SUFFIX}"
        conv_path = conv_dir / f"{mit_sid}.json"
        if not conv_path.exists():
            n_missing_conv += 1
            continue
        jobs.append(Job(
            job_id=mit_sid,
            payload={
                "session_id":   mit_sid,
                "conv_path":    conv_path,
                "scores_dir":   scores_dir,
                "eval_provider": eval_provider,
                "eval_model":    eval_model,
            },
        ))

    n_already_scored = sum(
        1 for j in jobs
        if _is_fully_scored(
            _score_path(scores_dir, j.job_id), evaluator_slug, main_cfg.N_TURNS,
        )
    )

    print(f"Run dir           : {run_dir}")
    print(f"Sample index      : {args.sample_index}  ({len(entries)} entries)")
    if n_missing_conv:
        print(f"  {n_missing_conv} entrie(s) have no mitigated conversation on disk yet — "
              f"skipped (run mitigation_plan.run_conversations first).")
    print(f"Sessions to score : {len(jobs)}  ({n_already_scored} already fully scored)")
    print(f"Rubric            : {HISTORY_STANCE_RUBRIC.name} (dims: {HISTORY_STANCE_RUBRIC.dimensions})")
    print(f"Evaluator         : {eval_provider}/{eval_model}")
    print(f"Scores out        : {scores_dir}")
    print(f"Workers           : {args.workers}")
    print(f"Fail-fast         : max_same_error={args.max_same_error or 'off'}  "
          f"max_failure_rate={args.max_failure_rate or 'off'}  "
          f"max_failed={args.max_failed or 'off'}\n")

    paths = RunPaths(root=run_dir)
    try:
        completed, failed = run_jobs(
            jobs=jobs,
            worker=_worker,
            paths=paths,
            n_workers=args.workers,
            is_done=lambda j: _is_fully_scored(
                _score_path(scores_dir, j.job_id), evaluator_slug, main_cfg.N_TURNS,
            ),
            progress_label="stance-history-scoring-mitigated",
            checkpoint_name="checkpoint_stance_history_scoring_mitigated.json",
            max_failed=args.max_failed,
            max_failure_rate=args.max_failure_rate,
            max_same_error=args.max_same_error or None,
        )
    except KeyboardInterrupt:
        print("\nStopped by user. Re-run the exact same command to resume — "
              "already-scored sessions are skipped.")
        raise

    n_parse_fail = sum(r.info.get("parse_fail", 0) for r in completed)
    print(f"\nCompleted: {len(completed)}  Failed: {len(failed)}  "
          f"(turns with parse failures: {n_parse_fail})")
    if failed:
        print("\nFailed sessions (re-run the same command to retry these):")
        for r in failed:
            err = str(r.info.get("error", "")).strip()
            print(f"  - {r.job_id}: {err[:160]}")


if __name__ == "__main__":
    main()
