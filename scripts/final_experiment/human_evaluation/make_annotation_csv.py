r"""
human_evaluation/make_annotation_csv.py
=======================================
Task 1 — build the Qualtrics annotation source CSV for validating the
ORIGINAL metrics (the ``misinfo_v1_split`` dimensions the LLM evaluator
already scores: correction / rebuttal / agreeableness).

Per Readme.md, task 1's stimulus is an **extracted local response**: one
isolated (user_message, ai_message) pair shown without conversation
context, coded against the same instructions the LLM judge received. That
is one CSV row per rated unit, which is what this script emits.

Design
------
The sample is *not* redrawn — it reads the 600 session_ids that
``evaluator_validation/run_validation_scoring.py`` already stratified, so
both arms score identical (session_id, turn) units and the agreement join
is exact.

From each of those 600 conversations we take **exactly one turn**. Turns
within a conversation are strongly correlated (ICC ~= 0.57 on correction),
so all 8 turns of one conversation are worth only ~1.6 independent items;
600 conversations x 1 turn buys the precision that 600 x 8 = 4800 items
would nominally suggest but not deliver. Turn position is assigned by a
rotation over the belief x (cell x model) grid so all 8 positions are
evenly represented — which matters because turn drives the score hard
(57% of turn-1 responses score 3 on correction vs 25% at turn 8).

This script emits the full 600-item master set only. Cutting it into rater
blocks — balancing reading load, and rotating blocks so every item is rated
by three independent people — is ``make_rater_blocks.py``.

Qualtrics notes
---------------
* The Loop & Merge "import from file" parser treats a newline as a row
  terminator **even inside a quoted field**. Every emitted cell is
  therefore newline-free under the default ``--text-format html``; raw
  newlines become ``<br>``. ``--text-format raw`` lifts that guarantee
  and is only for pandas-side use.
* Target responses are markdown-heavy (``**bold**``, ``- `` bullets,
  ``### headers``). Qualtrics renders merged text as HTML, not markdown,
  so the html formatter converts the common constructs; otherwise
  annotators read literal asterisks.
* Column order is load-bearing — Loop & Merge addresses fields
  positionally. See ``config.COLUMNS``.
* Pair this with ``export_rubric_instructions.py``, which emits the
  participant-facing rubric text straight out of ``core.scoring.RUBRICS``.

Usage
-----
::

    cd scripts/final_experiment

    # The task-1 design: 600 conversations, 1 balanced turn each.
    python -m human_evaluation.make_annotation_csv \
        --run-dir ../../results/final_experiment/main_user_IVs/20260427_165233

    # Alternatives: i.i.d. turn draw, or a lighter budget.
    python -m human_evaluation.make_annotation_csv \
        --run-dir <run-dir> --turn-assignment random --max-sessions 400

Then cut it into rater blocks::

    python -m human_evaluation.make_rater_blocks --run-dir <run-dir>
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Make `core` importable when this file is run as a module from
# scripts/final_experiment/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import RunPaths  # noqa: E402

from . import config as cfg  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
# session_id parsing
# ════════════════════════════════════════════════════════════════════════════
#
# build_session_id (core/storage.py) composes:
#     cell-{cell_id}__belief-{category}-{belief_index:04d}__model-{slug}
# where cell_id is itself "{iv1}__{iv2}", e.g.
#     cell-iv1-emotional__iv2-none__belief-bias-0003__model-claude-sonnet-4.6
#
# Categories contain underscores ("fake_news") and model slugs contain
# hyphens and dots, so the pattern anchors on the literal "__belief-" and
# "__model-" separators and peels the zero-padded index off the end of the
# category field with a greedy category match.

_SESSION_ID_RE = re.compile(
    r"^cell-(?P<cell_id>.+?)"
    r"__belief-(?P<category>.+)-(?P<belief_index>\d{4})"
    r"__model-(?P<target_model>.+)$"
)


def parse_session_id(session_id: str) -> dict[str, str]:
    """Recover experimental provenance from a session_id.

    Kept as a module-level helper (rather than inlined) so the downstream
    human-vs-LLM agreement analysis can import it and reconstruct the
    condition / belief / model columns from the Qualtrics export, which
    carries only session_id and turn back.

    Returns cell_id, iv1, iv2, category, belief_index and target_model.
    Raises ValueError on an unparseable id rather than returning partial
    junk — a silent miss here would corrupt every subgroup breakdown.
    """
    m = _SESSION_ID_RE.match(session_id)
    if not m:
        raise ValueError(f"Unparseable session_id: {session_id!r}")
    cell_id = m.group("cell_id")
    iv1, _, iv2 = cell_id.partition("__")
    return {
        "cell_id": cell_id,
        "iv1": iv1.removeprefix("iv1-"),
        "iv2": iv2.removeprefix("iv2-"),
        "category": m.group("category"),
        "belief_index": m.group("belief_index"),
        "target_model": m.group("target_model"),
    }


# ════════════════════════════════════════════════════════════════════════════
# Text formatting for Qualtrics
# ════════════════════════════════════════════════════════════════════════════

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_WS_RE = re.compile(r"\s+")


def _to_html(text: str) -> str:
    """Render one message as a single-line HTML fragment.

    Deliberately minimal: escape first (so a literal ``<`` in a model
    response can't inject markup into the survey), then convert only the
    markdown constructs that actually appear in these transcripts —
    ``**bold**``, ATX headers, and dash/star bullets. Anything else is
    left alone; a stray character reads better than a mangled one.

    The result contains no newline, which is what makes it safe for the
    Qualtrics Loop & Merge file importer.
    """
    escaped = html.escape(text, quote=False)
    escaped = _BOLD_RE.sub(r"<strong>\1</strong>", escaped)

    lines: list[str] = []
    for raw_line in escaped.split("\n"):
        line = raw_line.rstrip()
        header = _HEADER_RE.match(line)
        if header:
            lines.append(f"<strong>{header.group(1).strip()}</strong>")
            continue
        bullet = _BULLET_RE.match(line)
        if bullet:
            lines.append(f"&bull; {bullet.group(2).strip()}")
            continue
        lines.append(line.strip())

    # Collapse runs of blank lines to a single paragraph break so a cell
    # doesn't open with a stack of empty <br>s.
    out: list[str] = []
    blank_run = 0
    for line in lines:
        if line:
            out.append(line)
            blank_run = 0
        else:
            blank_run += 1
            if blank_run == 1 and out:
                out.append("")
    while out and not out[-1]:
        out.pop()
    return "<br>".join(out)


def _to_plain(text: str) -> str:
    """Collapse all whitespace to single spaces. Newline-free by construction."""
    return _WS_RE.sub(" ", text).strip()


def format_text(text: str, text_format: str) -> str:
    """Dispatch on --text-format. 'raw' is the only mode that can emit newlines."""
    text = text or ""
    if text_format == "html":
        return _to_html(text)
    if text_format == "plain":
        return _to_plain(text)
    if text_format == "raw":
        return text
    raise ValueError(f"Unknown text format: {text_format!r}")


def format_belief(belief: dict) -> str:
    """Render the belief exactly as the LLM judge saw it.

    Mirrors ``core.scoring._format_belief_for_evaluator``: long-text
    beliefs are shown as ``[title]`` followed by the body, plain ones as
    the bare claim. Duplicated rather than imported for the same reason
    that function duplicates the agent's formatter — but it must stay in
    sync, or humans and the judge are reading different stimuli.
    """
    if belief.get("is_long_text"):
        title = (belief.get("content") or "").strip()
        body = (belief.get("long_text") or "").strip()
        return f"[{title}]\n{body}"
    return (belief.get("content") or "").strip()


# ════════════════════════════════════════════════════════════════════════════
# Turn assignment — one turn per conversation, spread across 1..8
# ════════════════════════════════════════════════════════════════════════════

def assign_turns(
    session_ids: list[str],
    *,
    mode: str,
    n_turns: int,
    seed: int,
) -> dict[str, int]:
    """Choose which single turn each conversation contributes.

    ``rotation`` (default) lays the conversations out on the
    belief x (cell x model) grid the run was built from and assigns

        turn = (belief_row + bucket_col) mod n_turns + 1

    Both indices are taken over *sorted* keys, so the assignment is
    deterministic and reproducible without an RNG. On this run's 30 x 20
    grid it yields 74-76 conversations per turn, 14-16 per
    (turn x condition), 17-20 per (turn x model), and every belief
    covering all 8 positions — margins an i.i.d. draw would not hit at
    n=600.

    ``random`` draws each turn independently from a seeded RNG. Simpler to
    describe in a methods section; accepts chance imbalance in the margins.
    """
    if mode == "random":
        rng = random.Random(seed)
        return {sid: rng.randint(1, n_turns) for sid in sorted(session_ids)}
    if mode != "rotation":
        raise ValueError(f"Unknown turn assignment mode: {mode!r}")

    parsed = {sid: parse_session_id(sid) for sid in session_ids}
    beliefs = sorted({(p["category"], p["belief_index"]) for p in parsed.values()})
    buckets = sorted({(p["cell_id"], p["target_model"]) for p in parsed.values()})
    belief_row = {b: i for i, b in enumerate(beliefs)}
    bucket_col = {b: j for j, b in enumerate(buckets)}

    out: dict[str, int] = {}
    for sid, p in parsed.items():
        i = belief_row[(p["category"], p["belief_index"])]
        j = bucket_col[(p["cell_id"], p["target_model"])]
        out[sid] = ((i + j) % n_turns) + 1
    return out


# ════════════════════════════════════════════════════════════════════════════
# Sample loading and session selection
# ════════════════════════════════════════════════════════════════════════════

def _load_sampled_session_ids(validation_dir: Path) -> list[str]:
    """Read the evaluator_validation manifest's session_ids. Never re-samples."""
    manifest_path = validation_dir / cfg.SAMPLE_MANIFEST_NAME
    if not manifest_path.exists():
        raise SystemExit(
            f"Sample manifest not found: {manifest_path}\n"
            "The human set must reuse the evaluator_validation sample so both "
            "arms score identical units. Produce it first with:\n"
            "  python -m evaluator_validation.run_validation_scoring "
            "--run-dir <run-dir> --sample-only"
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    sids = list(manifest.get("sampled_session_ids", []))
    if not sids:
        raise SystemExit(f"{manifest_path} contains no sampled_session_ids.")
    print(
        f"[source] {manifest_path.name}: {len(sids)} session(s) "
        f"from {manifest.get('n_beliefs_sampled', '?')} belief(s) "
        f"(fraction={manifest.get('fraction')}, seed={manifest.get('seed')})"
    )
    return sids


def _largest_remainder(
    strata: dict[tuple, list],
    total: int,
    n_target: int,
    rng: random.Random,
) -> dict[tuple, int]:
    """Proportional allocation of ``n_target`` across strata.

    Each stratum's exact quota is ``n_target * n_stratum / total``. Floors
    are handed out first, then the leftover slots go to the largest
    fractional remainders. Ties are broken by a random key rather than by
    stratum name, so when quotas are uniformly fractional (which is the
    normal case here — ~3-4 items per stratum at a 20% rate) the choice of
    which strata contribute is random rather than alphabetical.
    """
    quotas: dict[tuple, int] = {}
    remainders: list[tuple[float, float, tuple]] = []
    for key, members in strata.items():
        exact = n_target * len(members) / total
        quotas[key] = int(exact)
        remainders.append((exact - int(exact), rng.random(), key))
    leftover = n_target - sum(quotas.values())
    remainders.sort(key=lambda r: (-r[0], r[1]))
    for _, _, key in remainders[:leftover]:
        quotas[key] += 1
    return quotas


def _select_sessions(
    session_ids: list[str],
    *,
    max_sessions: int | None,
    seed: int,
) -> list[str]:
    """Optionally cut the session list down to a lighter budget.

    Stratified by (cell_id, target_model) so the balanced factorial
    structure survives the downsample — an unstratified draw would leave
    some cell x model buckets thin and make the per-subgroup agreement
    estimates noisy in different places than the LLM arm's.
    """
    if max_sessions is None or max_sessions >= len(session_ids):
        return sorted(session_ids)

    by_bucket: dict[tuple, list[str]] = defaultdict(list)
    for sid in session_ids:
        p = parse_session_id(sid)
        by_bucket[(p["cell_id"], p["target_model"])].append(sid)

    rng = random.Random(seed)
    quotas = _largest_remainder(by_bucket, len(session_ids), max_sessions, rng)
    chosen: list[str] = []
    for bucket in sorted(by_bucket):
        pool = sorted(by_bucket[bucket])
        chosen.extend(rng.sample(pool, min(quotas[bucket], len(pool))))

    print(
        f"[select] downsampled {len(session_ids)} -> {len(chosen)} session(s), "
        f"stratified across {len(by_bucket)} (cell x model) bucket(s)."
    )
    return sorted(chosen)


# ════════════════════════════════════════════════════════════════════════════
# Row construction
# ════════════════════════════════════════════════════════════════════════════

def build_rows(
    paths: RunPaths,
    session_ids: list[str],
    turn_of: dict[str, int],
    *,
    text_format: str,
) -> tuple[list[dict[str, object]], Counter]:
    """Flatten each conversation to its one assigned turn.

    Rows come out in deterministic (session_id, turn) order. A conversation
    whose assigned turn is missing or has an empty response is dropped and
    counted — silently substituting a neighbouring turn would quietly break
    the balance the rotation was built to guarantee.
    """
    rows: list[dict[str, object]] = []
    stats: Counter = Counter()

    for sid in session_ids:
        path = paths.conversation_path(sid)
        if not path.exists():
            stats["missing_conversation"] += 1
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                conv = json.load(f)
        except json.JSONDecodeError:
            stats["unreadable_conversation"] += 1
            continue
        if not conv.get("completed_at"):
            stats["incomplete_conversation"] += 1
            continue

        tn = turn_of[sid]
        turn = next((t for t in conv.get("turns", []) if t.get("turn") == tn), None)
        if turn is None:
            stats["missing_turn"] += 1
            continue
        user_message = (turn.get("user_message") or "").strip()
        ai_message = (turn.get("target_response") or "").strip()
        if not ai_message:
            stats["empty_response"] += 1
            continue

        rows.append({
            "session_id": sid,
            "turn": tn,
            "user_message": format_text(user_message, text_format),
            "misinformation_belief": format_text(
                format_belief(conv.get("belief", {})), text_format,
            ),
            "ai_message": format_text(ai_message, text_format),
            "is_long_text": int(bool(conv.get("belief", {}).get("is_long_text"))),
            "belief_category": parse_session_id(sid)["category"],
        })
        stats["rows"] += 1

    rows.sort(key=lambda r: (r["session_id"], r["turn"]))
    return rows, stats


# ════════════════════════════════════════════════════════════════════════════
# Overlap core
# ════════════════════════════════════════════════════════════════════════════

def mark_overlap_core(
    rows: list[dict[str, object]],
    *,
    fraction: float,
    seed: int,
) -> list[dict[str, object]]:
    """Flag a proportional random subset as the every-annotator core.

    Strata are (condition x target_model x turn) per
    ``cfg.OVERLAP_STRATA_KEYS``. At a 20% rate each stratum holds only
    ~3-4 of this run's 600 items, so nearly every quota lands in the
    fractional range and the draw reduces to "one item from a random
    proportional subset of strata" — which is the intent: random within
    the stratified set, spread across all three margins.

    Returns the core rows (the flag is written in place on ``rows``).
    """
    if fraction <= 0:
        return []
    n_target = round(fraction * len(rows))
    if n_target <= 0:
        return []

    strata: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        p = parse_session_id(str(row["session_id"]))
        key = tuple(
            row["turn"] if k == "turn" else p[k]
            for k in cfg.OVERLAP_STRATA_KEYS
        )
        strata[key].append(row)

    rng = random.Random(seed)
    quotas = _largest_remainder(strata, len(rows), n_target, rng)

    core: list[dict[str, object]] = []
    for key in sorted(strata):
        pool = strata[key]
        n = min(quotas[key], len(pool))
        for row in rng.sample(pool, n):
            row["overlap_core"] = 1
            core.append(row)

    core.sort(key=lambda r: (r["session_id"], r["turn"]))
    print(
        f"[overlap] flagged {len(core)} of {len(rows)} item(s) "
        f"({len(core) / len(rows):.1%}) across {len(strata)} "
        f"({' x '.join(cfg.OVERLAP_STRATA_KEYS)}) strata."
    )
    return core


# ════════════════════════════════════════════════════════════════════════════
# Output
# ════════════════════════════════════════════════════════════════════════════

def _write_csv(path: Path, rows: list[dict[str, object]], encoding: str) -> None:
    """Write one CSV with cfg.COLUMNS in order.

    ``newline=""`` plus csv's default CRLF terminator is what the module
    expects on every platform, and CRLF is what the Qualtrics importer is
    happiest with.
    """
    with open(path, "w", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cfg.COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _assert_newline_free(rows: list[dict[str, object]], text_format: str) -> None:
    """Guard the Qualtrics invariant: no newline inside any emitted cell.

    A single leaked newline shifts every subsequent Loop & Merge row by
    one — silent, and unrecoverable once annotation has started.
    """
    if text_format == "raw":
        return
    text_cols = ("user_message", "misinformation_belief", "ai_message")
    for i, row in enumerate(rows):
        for col in text_cols:
            if "\n" in str(row[col]) or "\r" in str(row[col]):
                raise SystemExit(
                    f"Internal error: newline survived formatting in row {i} "
                    f"({row['session_id']} turn {row['turn']}, column {col}). "
                    "Refusing to write a CSV that would corrupt a Qualtrics "
                    "Loop & Merge import."
                )


def _margin_table(rows: list[dict[str, object]], key: str) -> dict:
    """Count rows per level of one design factor, for the printed summary."""
    counts: Counter = Counter()
    for row in rows:
        if key == "turn":
            counts[row["turn"]] += 1
        else:
            counts[parse_session_id(str(row["session_id"]))[key]] += 1
    return dict(sorted(counts.items(), key=lambda kv: str(kv[0])))


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run-dir", type=Path, required=True,
        help="main_user_IVs run directory containing conversations/ and "
             "evaluator_validation/sample_manifest.json.",
    )
    p.add_argument(
        "--turn-assignment", choices=("rotation", "random"),
        default=cfg.TURN_ASSIGNMENT,
        help="How each conversation's single turn is chosen. rotation "
             "(default) balances turns 1-8 across the belief x cell x model "
             "grid; random draws i.i.d. from --seed.",
    )
    p.add_argument(
        "--max-sessions", type=int, default=None,
        help="Cap the number of conversations, stratified across "
             "(cell x target_model). Omit to keep all 600.",
    )
    p.add_argument(
        "--text-format", choices=("html", "plain", "raw"),
        default=cfg.DEFAULT_TEXT_FORMAT,
        help="html (default): escape, render markdown, newlines -> <br>. "
             "plain: collapse whitespace. raw: verbatim, multi-line cells — "
             "NOT safe for Qualtrics import.",
    )
    p.add_argument(
        "--shuffle", action="store_true",
        help="Shuffle row order (seeded). Off by default so the first "
             "annotation file stays in deterministic (session_id, turn) order.",
    )
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument(
        "--task", type=str, default=cfg.TASK_ID,
        help=f"Task label prefixed to every output file (default: "
             f"{cfg.TASK_ID}).",
    )
    p.add_argument(
        "--stem", type=str, default=None,
        help="Override the full filename stem. Default is derived from "
             "--task as '<task>_annotation_items'.",
    )
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help=f"Output directory (default: <run-dir>/{cfg.HUMAN_SUBDIR}).",
    )
    p.add_argument("--encoding", type=str, default=cfg.DEFAULT_ENCODING)
    args = p.parse_args()

    if not args.run_dir.exists():
        p.error(f"--run-dir does not exist: {args.run_dir}")
    if not (args.run_dir / "conversations").exists():
        p.error(f"{args.run_dir}/conversations not found.")
    if args.max_sessions is not None and args.max_sessions < 1:
        p.error("--max-sessions must be >= 1")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.task):
        p.error("--task must be filename-safe: letters, digits, '_' or '-'.")

    stem = args.stem or cfg.csv_stem(args.task)
    paths = RunPaths(root=args.run_dir)
    validation_dir = args.run_dir / cfg.VALIDATION_SUBDIR
    out_dir = args.out_dir or (args.run_dir / cfg.HUMAN_SUBDIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTask              : {args.task} (rubric: {cfg.RUBRIC_NAME})")
    print(f"Run dir           : {paths.root}")
    print(f"Output dir        : {out_dir}")

    session_ids = _load_sampled_session_ids(validation_dir)
    session_ids = _select_sessions(
        session_ids, max_sessions=args.max_sessions, seed=args.seed,
    )
    turn_of = assign_turns(
        session_ids,
        mode=args.turn_assignment,
        n_turns=cfg.N_TURNS_PER_CONVERSATION_POOL,
        seed=args.seed,
    )
    print(f"[turns] {args.turn_assignment}: one turn per conversation, "
          f"drawn from 1..{cfg.N_TURNS_PER_CONVERSATION_POOL}")

    rows, stats = build_rows(
        paths, session_ids, turn_of, text_format=args.text_format,
    )
    if not rows:
        raise SystemExit("No rows produced — check the run directory.")

    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
        print(f"[order] shuffled with seed={args.seed}")
    else:
        print("[order] deterministic (session_id, turn)")

    _assert_newline_free(rows, args.text_format)

    # One master file. Cutting it into rater blocks is make_rater_blocks.py's
    # job — it needs word counts and the rotation, neither of which belongs
    # in the flattening step.
    written: list[Path] = []
    path = out_dir / f"{stem}.csv"
    _write_csv(path, rows, args.encoding)
    written.append(path)

    # Sidecar manifest: enough to reproduce this exact CSV, and to tell the
    # agreement analysis which units and which rubric the ratings belong to.
    manifest = {
        "task": args.task,
        "builder": "human_evaluation/make_annotation_csv",
        "stimulus_unit": "extracted_local_response",
        "run_dir": str(paths.root),
        "source_manifest": str(validation_dir / cfg.SAMPLE_MANIFEST_NAME),
        "rubric_name": cfg.RUBRIC_NAME,
        "columns": list(cfg.COLUMNS),
        "hidden_columns": list(cfg.HIDDEN_COLUMNS),
        "turns_per_conversation": cfg.TURNS_PER_CONVERSATION,
        "turn_assignment": args.turn_assignment,
        "max_sessions": args.max_sessions,
        "text_format": args.text_format,
        "encoding": args.encoding,
        "shuffled": bool(args.shuffle),
        "seed": args.seed,
        "n_sessions": len(session_ids),
        "n_rows": len(rows),
        "files": [f.name for f in written],
        "skipped": {k: v for k, v in sorted(stats.items()) if k != "rows"},
        "created_at": datetime.now().isoformat(),
    }
    manifest_path = out_dir / f"{stem}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Summary: the margins are the point of the design, so print them.
    print(f"\n[build] {len(rows)} item(s) from {len(session_ids)} conversation(s)")
    for key, label in (("turn", "turn"), ("iv1", "condition"),
                       ("target_model", "model"), ("category", "category")):
        print(f"[margin] {label:10s} {_margin_table(rows, key)}")
    skipped = {k: v for k, v in stats.items() if k != "rows"}
    if skipped:
        print(f"[build] ⚠ skipped: {skipped}")

    print()
    for f_ in written:
        n = sum(1 for _ in open(f_, encoding=args.encoding)) - 1
        print(f"[write] {f_.relative_to(out_dir.parent)}  ({n} rows, "
              f"{f_.stat().st_size / 1024:.0f} KB)")
    print(f"[write] {manifest_path.relative_to(out_dir.parent)}")

    print(
        "\n[qualtrics] Loop & Merge field order:\n"
        + "\n".join(
            f"              ${{lm://Field/{i}}} = {c}"
            + ("   (hidden)" if c in cfg.HIDDEN_COLUMNS else "")
            for i, c in enumerate(cfg.COLUMNS, 1)
        )
        + "\n\n[next] python -m human_evaluation.make_rater_blocks "
          f"--run-dir {paths.root}"
          "\n       python -m human_evaluation.export_rubric_instructions "
          f"--out-dir {out_dir}"
    )


if __name__ == "__main__":
    main()
