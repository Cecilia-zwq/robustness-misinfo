"""
multi_evaluator_scoring.py
==========================
Runs two additional LLM evaluators on all turn-level data, joining
turn_level.csv (scores/metadata) with summary.json (conversation text).

Usage:
    cd Iteration4

    # Step 1: Dry run — prints what would be sent to each evaluator
    python multi_evaluator_scoring.py \
        --turn-csv  results/.../turn_level.csv \
        --summary   results/.../summary.json \
        --output    results/.../turn_level_multi_eval.csv \
        --dry-run

    # Step 2: Real run — calls the two evaluators
    python multi_evaluator_scoring.py \
        --turn-csv  results/.../turn_level.csv \
        --summary   results/.../summary.json \
        --output    results/.../turn_level_multi_eval.csv

    # Step 3: Resume if interrupted
    python multi_evaluator_scoring.py \
        --turn-csv  results/.../turn_level.csv \
        --summary   results/.../summary.json \
        --output    results/.../turn_level_multi_eval.csv \
        --resume

    # Step 4: Report only (skip scoring, just compute agreement)
    python multi_evaluator_scoring.py \
        --output    results/.../turn_level_multi_eval.csv \
        --report-only

Environment:
    export OPENROUTER_API_KEY="sk-or-..."
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════════

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

ADDITIONAL_MODELS = [
    {
        "name": "gpt-5.4-mini",
        "provider": "openrouter",
        "model": "openai/gpt-5.4-mini",
    },
    {
        "name": "claude-haiku-4.5",
        "provider": "openrouter",
        "model": "anthropic/claude-haiku-4.5",
    },
]

DIMENSIONS = ["correction", "rebuttal"]

# Rate-limit safety: pause between API calls (seconds)
PAUSE_BETWEEN_CALLS = 0.3


# ════════════════════════════════════════════════════════════════════════════════
# Load and join data sources
# ════════════════════════════════════════════════════════════════════════════════


def load_and_join(turn_csv: Path, summary_json: Path) -> pd.DataFrame:
    """
    Load turn_level.csv and summary.json, then join conversation text
    (user_message, target_response, misinformation_claim) onto each turn row.

    The conversation_history in summary.json is an alternating list:
        [{"role": "user", ...}, {"role": "assistant", ...}, ...]
    So turn N (1-indexed) corresponds to:
        user_message    = conversation_history[(N-1)*2]["content"]
        target_response = conversation_history[(N-1)*2 + 1]["content"]
    """
    df = pd.read_csv(turn_csv)
    logger.info("Loaded %d rows from %s", len(df), turn_csv)

    with open(summary_json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Build lookup: session_id -> {claim, conversation_history}
    session_lookup = {}
    for r in summary["results"]:
        session_lookup[r["session_id"]] = {
            "claim": r["claim"],
            "conversation_history": r.get("conversation_history", []),
        }

    logger.info("Loaded %d sessions from %s", len(session_lookup), summary_json)

    # Join text onto each row
    user_messages = []
    target_responses = []
    claims = []
    missing_count = 0

    for _, row in df.iterrows():
        sid = row["session_id"]
        turn = int(row["turn"])

        info = session_lookup.get(sid)
        if info is None:
            user_messages.append(None)
            target_responses.append(None)
            claims.append(None)
            missing_count += 1
            continue

        history = info["conversation_history"]
        user_idx = (turn - 1) * 2
        asst_idx = user_idx + 1

        if user_idx < len(history) and asst_idx < len(history):
            user_messages.append(history[user_idx]["content"])
            target_responses.append(history[asst_idx]["content"])
        else:
            user_messages.append(None)
            target_responses.append(None)
            missing_count += 1

        claims.append(info["claim"])

    df["user_message"] = user_messages
    df["target_response"] = target_responses
    df["misinformation_claim"] = claims

    if missing_count > 0:
        logger.warning(
            "%d rows could not be joined (missing session or turn in history)",
            missing_count,
        )

    return df


# ════════════════════════════════════════════════════════════════════════════════
# Dry-run: print what would be sent to the evaluator
# ════════════════════════════════════════════════════════════════════════════════


def dry_run(df: pd.DataFrame, n_samples: int = 5):
    """Print a few sample evaluator inputs to verify the join is correct."""
    print("\n" + "=" * 70)
    print("DRY RUN — Showing evaluator inputs for first %d rows" % n_samples)
    print("=" * 70)

    valid = df[df["user_message"].notna() & df["target_response"].notna()]
    sample = valid.head(n_samples)

    for i, (_, row) in enumerate(sample.iterrows()):
        print(f"\n{'-' * 60}")
        print(f"Row {i+1} | session={row['session_id']} | turn={int(row['turn'])}")

        # Show the original score column name (after rename)
        orig_corr_col = "correction_gpt-4.1-mini"
        orig_rebt_col = "rebuttal_gpt-4.1-mini"
        if orig_corr_col in df.columns:
            print(f"  Original scores: correction={row[orig_corr_col]}, rebuttal={row[orig_rebt_col]}")
        else:
            print(f"  Original scores: correction={row.get('correction','?')}, rebuttal={row.get('rebuttal','?')}")

        print(f"  Category: {row['category']} | Subtype: {row['subtype']}")
        print(f"  Persona: {row['persona']} | Rep: {row['rep']}")
        print(f"{'-' * 60}")
        print(f"  MISINFORMATION_CLAIM:")
        claim_text = str(row['misinformation_claim'])
        print(f"    {claim_text[:200]}")
        print(f"\n  USER_MESSAGE:")
        user_text = str(row['user_message'])
        print(f"    {user_text[:300]}")
        print(f"\n  TARGET_RESPONSE:")
        target_text = str(row['target_response'])
        print(f"    {target_text[:300]}")
        print()

    # Summary stats
    n_total = len(df)
    n_valid = len(valid)
    n_missing = n_total - n_valid

    print(f"\n{'=' * 70}")
    print("DRY RUN SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total rows:      {n_total}")
    print(f"  Valid (joinable): {n_valid}")
    print(f"  Missing text:    {n_missing}")

    # Show score distribution from original evaluator
    orig_corr_col = "correction_gpt-4.1-mini"
    if orig_corr_col in df.columns:
        print(f"\n  Original score distribution (gpt-4.1-mini):")
        for dim in DIMENSIONS:
            col = f"{dim}_gpt-4.1-mini"
            print(f"    {dim}:")
            counts = df[col].value_counts().sort_index()
            for score, count in counts.items():
                print(f"      {score}: {count}")

    print(f"\n  Models to run: {[m['name'] for m in ADDITIONAL_MODELS]}")
    print(f"  API calls needed: {n_valid} rows x {len(ADDITIONAL_MODELS)} models "
          f"= {n_valid * len(ADDITIONAL_MODELS)} calls")
    est_minutes = n_valid * len(ADDITIONAL_MODELS) * (PAUSE_BETWEEN_CALLS + 0.5) / 60
    print(f"  Estimated time: ~{est_minutes:.0f} minutes")
    print(f"\n  To proceed with scoring, remove --dry-run flag.")


# ════════════════════════════════════════════════════════════════════════════════
# Scoring logic
# ════════════════════════════════════════════════════════════════════════════════


def score_all_rows(
    df: pd.DataFrame,
    model_cfg: dict,
    output_path: Path,
    correction_col: str,
    rebuttal_col: str,
) -> pd.DataFrame:
    """
    Run one evaluator model on every row that still needs scoring.
    Checkpoints every 20 rows.
    """
    # Import here so dry-run doesn't require litellm installed
    from misinfo_eval_framework.evaluator import Evaluator

    evaluator = Evaluator(
        provider=model_cfg["provider"],
        model=model_cfg["model"],
        temperature=0.0,
    )

    # Find rows that need scoring AND have valid text
    needs_scoring = (
        (df[correction_col].isna() | df[rebuttal_col].isna())
        & df["user_message"].notna()
        & df["target_response"].notna()
    )
    remaining = needs_scoring.sum()
    logger.info(
        "Model %s: %d / %d rows need scoring.",
        model_cfg["name"],
        remaining,
        len(df),
    )

    scored = 0
    errors = 0

    for idx in df.index[needs_scoring]:
        row = df.loc[idx]

        try:
            scores = evaluator.evaluate(
                user_message=str(row["user_message"]),
                response=str(row["target_response"]),
                misinformation_claim=str(row["misinformation_claim"]),
            )
            df.at[idx, correction_col] = scores.get("correction", -1.0)
            df.at[idx, rebuttal_col] = scores.get("rebuttal", -1.0)
            scored += 1

        except Exception as e:
            logger.warning("Error at row %d (session=%s, turn=%s): %s",
                           idx, row.get("session_id", "?"), row.get("turn", "?"), e)
            df.at[idx, correction_col] = -1.0
            df.at[idx, rebuttal_col] = -1.0
            errors += 1

        # Progress log every 50 rows
        if (scored + errors) % 50 == 0:
            logger.info(
                "  [%s] Progress: %d/%d scored, %d errors",
                model_cfg["name"], scored, remaining, errors,
            )

        # Checkpoint every 20 rows
        if (scored + errors) % 20 == 0:
            df.to_csv(output_path, index=False)

        time.sleep(PAUSE_BETWEEN_CALLS)

    # Final save
    df.to_csv(output_path, index=False)
    logger.info(
        "Model %s done. Scored: %d, Errors: %d.",
        model_cfg["name"], scored, errors,
    )
    return df


# ════════════════════════════════════════════════════════════════════════════════
# Agreement analysis
# ════════════════════════════════════════════════════════════════════════════════


def compute_agreement(df: pd.DataFrame) -> dict:
    """
    For each dimension, classify each row as:
      - three_agree: all 3 models gave the same score
      - two_agree:   2 of 3 match
      - no_agree:    all 3 differ
    """
    results = {}

    for dim in DIMENSIONS:
        col_a = f"{dim}_gpt-4.1-mini"
        col_b = f"{dim}_gpt-5.4-mini"
        col_c = f"{dim}_claude-haiku-4.5"

        # Check columns exist
        for col in [col_a, col_b, col_c]:
            if col not in df.columns:
                logger.error("Missing column: %s", col)
                return {}

        # Filter out error rows (-1) and NaN
        valid = df[
            (df[col_a] != -1.0) & (df[col_b] != -1.0) & (df[col_c] != -1.0)
            & df[col_a].notna() & df[col_b].notna() & df[col_c].notna()
        ].copy()

        three_agree = []
        two_agree = []
        no_agree = []

        for idx, row in valid.iterrows():
            scores = [row[col_a], row[col_b], row[col_c]]
            unique = len(set(scores))

            if unique == 1:
                three_agree.append(idx)
            elif unique == 2:
                two_agree.append(idx)
            else:
                no_agree.append(idx)

        # Break down three-way agreements by score level
        three_agree_by_score = {}
        for s in [1.0, 2.0, 3.0]:
            count = sum(1 for idx in three_agree if valid.loc[idx, col_a] == s)
            three_agree_by_score[int(s)] = count

        # Break down two-way agreements by majority score
        two_agree_by_majority = {}
        for idx in two_agree:
            scores = [valid.loc[idx, col_a], valid.loc[idx, col_b], valid.loc[idx, col_c]]
            majority = Counter(scores).most_common(1)[0][0]
            key = int(majority)
            two_agree_by_majority[key] = two_agree_by_majority.get(key, 0) + 1

        # Break down no-agreement by the score spread
        no_agree_details = []
        for idx in no_agree:
            scores = sorted([valid.loc[idx, col_a], valid.loc[idx, col_b], valid.loc[idx, col_c]])
            no_agree_details.append(tuple(int(s) for s in scores))

        results[dim] = {
            "total_valid": len(valid),
            "three_agree": len(three_agree),
            "two_agree": len(two_agree),
            "no_agree": len(no_agree),
            "three_agree_by_score": three_agree_by_score,
            "two_agree_by_majority": two_agree_by_majority,
            "no_agree_spread": Counter(no_agree_details),
        }

    return results


def print_agreement_report(results: dict):
    """Print a formatted agreement summary."""
    print("\n" + "=" * 70)
    print("MULTI-EVALUATOR AGREEMENT REPORT")
    print("=" * 70)

    for dim in DIMENSIONS:
        r = results[dim]
        total = r["total_valid"]
        if total == 0:
            print(f"\n  {dim.upper()}: no valid rows")
            continue

        print(f"\n{'-' * 50}")
        print(f"  {dim.upper()}")
        print(f"{'-' * 50}")
        print(f"  Valid rows:         {total}")
        print(f"  Three-way agree:    {r['three_agree']:>4}  ({r['three_agree']/total*100:.1f}%)")
        print(f"  Two-way agree:      {r['two_agree']:>4}  ({r['two_agree']/total*100:.1f}%)")
        print(f"  No agreement:       {r['no_agree']:>4}  ({r['no_agree']/total*100:.1f}%)")

        print(f"\n  Three-way agreements by score:")
        for score in sorted(r["three_agree_by_score"]):
            count = r["three_agree_by_score"][score]
            print(f"    Score {score}: {count:>4}")

        print(f"\n  Two-way agreements by majority score:")
        for score in sorted(r["two_agree_by_majority"]):
            count = r["two_agree_by_majority"][score]
            print(f"    Score {score}: {count:>4}")

        if r["no_agree_spread"]:
            print(f"\n  No-agreement score spreads:")
            for spread, count in r["no_agree_spread"].most_common():
                print(f"    {spread}: {count:>4}")

    print("\n" + "=" * 70)


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple evaluators on turn-level data."
    )
    parser.add_argument(
        "--turn-csv", type=str,
        help="Path to turn_level.csv.",
    )
    parser.add_argument(
        "--summary", type=str,
        help="Path to summary.json (contains conversation_history).",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output CSV path for the multi-eval results.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sample evaluator inputs without calling any APIs.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output file (skip already-scored rows).",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Skip scoring; just compute agreement from existing output file.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    # ── Report-only mode ─────────────────────────────────────────────────
    if args.report_only:
        if not output_path.exists():
            logger.error("Output file not found: %s", output_path)
            sys.exit(1)
        df = pd.read_csv(output_path)
        results = compute_agreement(df)
        if results:
            print_agreement_report(results)

            summary_path = output_path.with_name("agreement_summary.json")
            json_results = {}
            for dim in DIMENSIONS:
                r = results[dim].copy()
                r["no_agree_spread"] = {str(k): v for k, v in r["no_agree_spread"].items()}
                json_results[dim] = r
            with open(summary_path, "w") as f:
                json.dump(json_results, f, indent=2)
            logger.info("Agreement summary saved to %s", summary_path)
        return

    # ── Validate inputs ──────────────────────────────────────────────────
    if not args.turn_csv or not args.summary:
        logger.error("--turn-csv and --summary are required (unless --report-only).")
        sys.exit(1)

    turn_csv = Path(args.turn_csv)
    summary_json = Path(args.summary)

    if not turn_csv.exists():
        logger.error("Turn CSV not found: %s", turn_csv)
        sys.exit(1)
    if not summary_json.exists():
        logger.error("Summary JSON not found: %s", summary_json)
        sys.exit(1)

    # ── Load or resume ───────────────────────────────────────────────────
    if args.resume and output_path.exists():
        logger.info("Resuming from %s", output_path)
        df = pd.read_csv(output_path)
    else:
        df = load_and_join(turn_csv, summary_json)

        # Rename original scores with suffix
        df.rename(
            columns={
                "correction": "correction_gpt-4.1-mini",
                "rebuttal": "rebuttal_gpt-4.1-mini",
            },
            inplace=True,
        )

        # Add empty columns for new models
        for model_cfg in ADDITIONAL_MODELS:
            name = model_cfg["name"]
            for dim in DIMENSIONS:
                col = f"{dim}_{name}"
                if col not in df.columns:
                    df[col] = pd.NA

    # ── Dry-run mode ─────────────────────────────────────────────────────
    if args.dry_run:
        dry_run(df)
        return

    # ── Score with each additional model ─────────────────────────────────
    for model_cfg in ADDITIONAL_MODELS:
        name = model_cfg["name"]
        correction_col = f"correction_{name}"
        rebuttal_col = f"rebuttal_{name}"

        df = score_all_rows(
            df=df,
            model_cfg=model_cfg,
            output_path=output_path,
            correction_col=correction_col,
            rebuttal_col=rebuttal_col,
        )

    # ── Agreement analysis ───────────────────────────────────────────────
    results = compute_agreement(df)
    if results:
        print_agreement_report(results)

        summary_path = output_path.with_name("agreement_summary.json")
        json_results = {}
        for dim in DIMENSIONS:
            r = results[dim].copy()
            r["no_agree_spread"] = {str(k): v for k, v in r["no_agree_spread"].items()}
            json_results[dim] = r
        with open(summary_path, "w") as f:
            json.dump(json_results, f, indent=2)
        logger.info("Agreement summary saved to %s", summary_path)

    # ── Final save ───────────────────────────────────────────────────────
    df.to_csv(output_path, index=False)
    logger.info("Final output saved to %s", output_path)
    logger.info("Total rows: %d", len(df))


if __name__ == "__main__":
    main()