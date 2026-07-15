"""
stance_analysis/build_summary.py
=================================
Join per-turn correction/rebuttal/agreeableness scores
(scores/<session_id>__misinfo_v1_split.json, the three-independent-
prompt rubric) with per-turn stance scores
(scores/<session_id>__misinfo_stance_split.json, produced by
run_stance_scoring.py) into one long CSV, restricted to the target
sessions from select_target_conversations.py.

Uses misinfo_v1_split rather than misinfo_v1 (turn_level.csv) for
correction/rebuttal/agreeableness: scoring each dimension with its own
blind evaluator call avoids one prompt's judgment contaminating
another's (e.g. a combined prompt inferring "more agreeable" partly
*because* correction/rebuttal already look weak in the same response,
not from independently judging agreement). The two rubrics disagree
non-trivially on Claude's agreeableness trend in particular — misinfo_v1
shows it rising turn-over-turn, misinfo_v1_split shows it falling — so
which one backs this analysis is a real methodological choice, not
just a data-source swap.

Output: config.PER_TURN_OUTPUT_PATH, one row per (session_id, turn):
    session_id, turn, target_model, iv1, belief_category,
    correction, rebuttal, agreeableness, stance_code, stance_label

Turns where misinfo_v1_split failed to parse any of
correction/rebuttal/agreeableness (-1.0) are dropped entirely (not just
from that dimension) — same rule the reference notebook
(final_experiment_analysis.ipynb) uses for its misinfo_v1_split-based
figures, so results stay comparable.

stance_code: 1.0=True, 2.0=False, 3.0=Uncertain, -1.0=unparsed.
Unparsed stance turns are kept (for auditing) but should be filtered
out of any stance-proportion aggregation downstream.

Usage::

    cd scripts/final_experiment
    python -m stance_analysis.build_summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from . import config as cfg  # noqa: E402

V1_SPLIT_RUBRIC_NAME = "misinfo_v1_split"


def _load_v1_split_scores(session_id: str) -> dict[int, dict[str, float]]:
    """turn -> {'correction':.., 'rebuttal':.., 'agreeableness':..} from misinfo_v1_split."""
    path = cfg.SCORES_DIR / f"{session_id}__{V1_SPLIT_RUBRIC_NAME}.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {int(t["turn"]): t["scores"] for t in data["turns"]}


def _load_stance_scores(session_id: str) -> dict[int, float]:
    path = cfg.SCORES_DIR / f"{session_id}__{cfg.STANCE_RUBRIC_NAME}.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {int(t["turn"]): t["scores"].get("stance", -1.0) for t in data["turns"]}


def main() -> None:
    if not cfg.TARGET_INDEX_PATH.exists():
        raise SystemExit(
            f"Target index not found: {cfg.TARGET_INDEX_PATH}\n"
            "Run `python -m stance_analysis.select_target_conversations` first."
        )

    with cfg.TARGET_INDEX_PATH.open("r", encoding="utf-8") as fh:
        entries = json.load(fh)["entries"]

    rows = []
    n_missing_v1_split = 0
    n_missing_stance = 0
    n_dropped_parse_fail = 0
    for e in entries:
        sid = e["session_id"]
        v1_split_by_turn = _load_v1_split_scores(sid)
        if not v1_split_by_turn:
            n_missing_v1_split += 1
            continue
        stance_by_turn = _load_stance_scores(sid)
        if not stance_by_turn:
            n_missing_stance += 1
            continue

        for turn_num, v1_scores in sorted(v1_split_by_turn.items()):
            correction    = v1_scores.get("correction", -1.0)
            rebuttal      = v1_scores.get("rebuttal", -1.0)
            agreeableness = v1_scores.get("agreeableness", -1.0)
            if -1.0 in (correction, rebuttal, agreeableness):
                n_dropped_parse_fail += 1
                continue

            stance_code = stance_by_turn.get(turn_num, -1.0)
            rows.append({
                "session_id":      sid,
                "turn":            turn_num,
                "target_model":    e["target_model"],
                "iv1":             e["iv1"],
                "belief_category": e["belief_category"],
                "correction":      correction,
                "rebuttal":        rebuttal,
                "agreeableness":   agreeableness,
                "stance_code":     stance_code,
                "stance_label":    cfg.STANCE_LABELS.get(stance_code, "unparsed"),
            })

    if n_missing_v1_split:
        print(
            f"Warning: {n_missing_v1_split} target session(s) missing a "
            f"{V1_SPLIT_RUBRIC_NAME} score file — skipped entirely."
        )
    if n_missing_stance:
        print(
            f"Warning: {n_missing_stance} target session(s) missing a stance score "
            f"file — run `python -m stance_analysis.run_stance_scoring` first "
            f"(or re-run it to fill in the gap)."
        )
    if n_dropped_parse_fail:
        print(
            f"Dropped {n_dropped_parse_fail} turn(s) where misinfo_v1_split failed "
            f"to parse correction/rebuttal/agreeableness."
        )

    columns = [
        "session_id", "turn", "target_model", "iv1", "belief_category",
        "correction", "rebuttal", "agreeableness", "stance_code", "stance_label",
    ]
    if not rows:
        raise SystemExit(
            "No target session produced any usable rows — run "
            "`python -m stance_analysis.run_stance_scoring` first."
        )
    out = pd.DataFrame(rows, columns=columns).sort_values(["session_id", "turn"])
    cfg.PER_TURN_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.PER_TURN_OUTPUT_PATH, index=False)
    print(
        f"Wrote {len(out):,} rows ({out['session_id'].nunique():,} sessions) "
        f"→ {cfg.PER_TURN_OUTPUT_PATH}"
    )

    n_unparsed = int((out["stance_code"] == -1.0).sum())
    if n_unparsed:
        print(f"  ({n_unparsed} unparsed stance rows out of {len(out):,})")


if __name__ == "__main__":
    main()
