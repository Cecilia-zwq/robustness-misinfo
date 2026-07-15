"""
stance_analysis/spot_check_stance_history.py
==============================================
Sanity-check the misinfo_stance_history_split evaluator before running
it at scale: sample a handful of individual (session, turn) responses
from the history sample index and score just that one turn each (with
its real conversation history as context), printing the history, user
message, target response, and the evaluator's stance label + reasoning
side-by-side with the already-known correction/rebuttal scores.

Deliberately scores single turns (not whole 8-turn conversations) so
this stays cheap (--n calls total, default 5). Turn indices >= 2 are
weighted into the sample on purpose (see --min-turn) since turn 1 has
no history and doesn't exercise what's actually new about this rubric.

Usage::

    cd scripts/final_experiment
    python -m stance_analysis.spot_check_stance_history
    python -m stance_analysis.spot_check_stance_history --n 10 --seed 1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from dataclasses import replace as _dc_replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from core import (  # noqa: E402
    ConversationArtifact,
    RUBRICS,
    format_belief_for_agent,
    score_one_turn,
)

from main_user_IVs import config as main_cfg  # noqa: E402

from . import config as cfg  # noqa: E402

# See run_stance_history_scoring.py's identical note: the registered
# rubric always carries the current prompt wording; cfg's name may carry
# a run suffix so this spot-check is labeled consistently with whichever
# run it's checking.
HISTORY_STANCE_RUBRIC = _dc_replace(
    RUBRICS["misinfo_stance_history_split"], name=cfg.HISTORY_STANCE_RUBRIC_NAME,
)


def _wrap(text: str, width: int = 100, indent: str = "    ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Spot-check the history-aware stance evaluator on a few turns."
    )
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--min-turn", type=int, default=2,
        help="Only sample turns >= this (default 2 — turn 1 has no history, "
             "so it doesn't exercise what's new about this rubric).",
    )
    p.add_argument(
        "--sample-index", type=Path, default=cfg.HISTORY_SAMPLE_INDEX_PATH,
        help="Sample index written by select_history_sample.py.",
    )
    args = p.parse_args()

    if not args.sample_index.exists():
        raise SystemExit(
            f"Sample index not found: {args.sample_index}\n"
            "Run `python -m stance_analysis.select_history_sample` first."
        )
    with args.sample_index.open("r", encoding="utf-8") as fh:
        entries = json.load(fh)["entries"]

    turn_level = pd.read_csv(cfg.TURN_LEVEL_CSV)
    sample_ids = {e["session_id"] for e in entries}
    turn_level = turn_level[
        turn_level["session_id"].isin(sample_ids) & (turn_level["turn"] >= args.min_turn)
    ]

    pool = list(turn_level[["session_id", "turn"]].itertuples(index=False, name=None))
    rng = random.Random(args.seed)
    sample = rng.sample(pool, k=min(args.n, len(pool)))

    print(f"Sampling {len(sample)} individual responses (turn >= {args.min_turn}) "
          f"from {len(sample_ids)} sampled sessions (seed={args.seed}).\n")
    print(f"Rubric    : {HISTORY_STANCE_RUBRIC.name}")
    print(f"Evaluator : {main_cfg.EVALUATOR_PRIMARY[0]}/{main_cfg.EVALUATOR_PRIMARY[1]}\n")
    print("=" * 100)

    for sid, turn in sample:
        conv_path = cfg.SOURCE_CONV_DIR / f"{sid}.json"
        with conv_path.open("r", encoding="utf-8") as fh:
            conv_data = json.load(fh)
        artifact = ConversationArtifact.from_dict(conv_data)
        turn_index = int(turn) - 1
        turn_data = conv_data["turns"][turn_index]

        # Full real artifact, not a mini one — score_one_turn derives
        # history from the real prior turns, and only pays for 1 call
        # regardless of how many turns precede it.
        scores, raw, n_attempts = score_one_turn(
            artifact=artifact,
            turn_index=turn_index,
            rubric=HISTORY_STANCE_RUBRIC,
            evaluator_provider=main_cfg.EVALUATOR_PRIMARY[0],
            evaluator_model=main_cfg.EVALUATOR_PRIMARY[1],
            temperature=cfg.EVALUATOR_TEMPERATURE,
        )
        stance_code = scores.get("stance", -1.0)
        stance_label = cfg.STANCE_LABELS.get(stance_code, "unparsed")

        row = turn_level[(turn_level["session_id"] == sid) & (turn_level["turn"] == turn)].iloc[0]

        print(f"\nsession_id : {sid}")
        print(f"turn       : {turn}   target_model: {row['target_model']}")
        print(f"known correction={row['correction']:g}  rebuttal={row['rebuttal']:g}"
              f"  agreeableness={row['agreeableness']:g}")
        print(f"stance     : {stance_label}  (code={stance_code:g})")
        print("\nmisinformation belief (ground truth, false claim):")
        print(_wrap(format_belief_for_agent(conv_data["belief"])))
        print(f"\nconversation history (turns 1-{turn - 1}):")
        for prior in conv_data["turns"][:turn_index]:
            print(f"    -- turn {prior['turn']} --")
            print(_wrap(f"User: {prior['user_message']}", indent="      "))
            print(_wrap(f"AI:   {prior['target_response']}", indent="      "))
        print(f"\ncurrent turn ({turn}) user_message:")
        print(_wrap(turn_data["user_message"]))
        print(f"\ncurrent turn ({turn}) target_response:")
        print(_wrap(turn_data["target_response"]))
        print("\nevaluator raw output:")
        print(textwrap.indent(raw.strip(), "    "))
        print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
