"""
stance_analysis/spot_check_stance.py
=====================================
Sanity-check the misinfo_stance_split evaluator before running it at
scale: sample a handful of individual (session, turn) responses from
the target index and score just that one turn each, printing the
user message, target response, and the evaluator's stance label +
reasoning side-by-side with the already-known correction/rebuttal
scores for that turn.

Deliberately scores single turns (not whole 8-turn conversations) so
this stays cheap (--n calls total, default 5) and lets you eyeball the
prompt/parsing before committing to the full run_stance_scoring.py
pass.

Usage::

    cd scripts/final_experiment
    python -m stance_analysis.spot_check_stance
    python -m stance_analysis.spot_check_stance --n 10 --seed 1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from core import (  # noqa: E402
    ConversationArtifact,
    RUBRICS,
    TurnArtifact,
    format_belief_for_agent,
    score_conversation,
)

from main_user_IVs import config as main_cfg  # noqa: E402

from . import config as cfg  # noqa: E402

STANCE_RUBRIC = RUBRICS[cfg.STANCE_RUBRIC_NAME]


def _wrap(text: str, width: int = 100, indent: str = "    ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def main() -> None:
    p = argparse.ArgumentParser(description="Spot-check the stance evaluator on a few turns.")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--target-index", type=Path, default=cfg.TARGET_INDEX_PATH,
        help="Target session index written by select_target_conversations.py.",
    )
    args = p.parse_args()

    if not args.target_index.exists():
        raise SystemExit(
            f"Target index not found: {args.target_index}\n"
            "Run `python -m stance_analysis.select_target_conversations` first."
        )
    with args.target_index.open("r", encoding="utf-8") as fh:
        entries = json.load(fh)["entries"]

    turn_level = pd.read_csv(cfg.TURN_LEVEL_CSV)
    target_ids = {e["session_id"] for e in entries}
    turn_level = turn_level[turn_level["session_id"].isin(target_ids)]

    # Pool of every (session_id, turn) among the target sessions.
    pool = list(turn_level[["session_id", "turn"]].itertuples(index=False, name=None))
    rng = random.Random(args.seed)
    sample = rng.sample(pool, k=min(args.n, len(pool)))

    print(f"Sampling {len(sample)} individual responses from {len(target_ids)} target sessions "
          f"(seed={args.seed}).\n")
    print(f"Rubric    : {STANCE_RUBRIC.name}")
    print(f"Evaluator : {main_cfg.EVALUATOR_PRIMARY[0]}/{main_cfg.EVALUATOR_PRIMARY[1]}\n")
    print("=" * 100)

    for sid, turn in sample:
        conv_path = cfg.SOURCE_CONV_DIR / f"{sid}.json"
        with conv_path.open("r", encoding="utf-8") as fh:
            conv_data = json.load(fh)

        turn_data = next(t for t in conv_data["turns"] if int(t["turn"]) == int(turn))

        # Minimal single-turn artifact — exercises the real score_conversation
        # code path (same one run_stance_scoring.py uses) without paying for
        # the other 7 turns.
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

        score_art = score_conversation(
            artifact=mini_artifact,
            rubric=STANCE_RUBRIC,
            evaluator_provider=main_cfg.EVALUATOR_PRIMARY[0],
            evaluator_model=main_cfg.EVALUATOR_PRIMARY[1],
            temperature=cfg.EVALUATOR_TEMPERATURE,
        )
        turn_scores = score_art.turns[0]
        stance_code = turn_scores.scores.get("stance", -1.0)
        stance_label = cfg.STANCE_LABELS.get(stance_code, "unparsed")

        row = turn_level[(turn_level["session_id"] == sid) & (turn_level["turn"] == turn)].iloc[0]

        print(f"\nsession_id : {sid}")
        print(f"turn       : {turn}   target_model: {row['target_model']}")
        print(f"known correction={row['correction']:g}  rebuttal={row['rebuttal']:g}"
              f"  agreeableness={row['agreeableness']:g}")
        print(f"stance     : {stance_label}  (code={stance_code:g})")
        print("\nmisinformation belief (ground truth, false claim):")
        print(_wrap(format_belief_for_agent(conv_data["belief"])))
        print("\nuser_message:")
        print(_wrap(turn_data["user_message"]))
        print("\ntarget_response:")
        print(_wrap(turn_data["target_response"]))
        print("\nevaluator raw output:")
        print(textwrap.indent(turn_scores.raw_output.strip(), "    "))
        print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
