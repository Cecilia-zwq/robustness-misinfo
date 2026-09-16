r"""
human_evaluation/export_rubric_instructions.py
==============================================
Emit the participant-facing rubric text for task 1.

Readme.md requires that "the participants will read the same instructions
as LLM-as-a-judge". The judge's instructions live in
``core.scoring.RUBRICS[<name>]`` — the same object
``evaluator_validation/run_validation_scoring.py`` passes to
``score_conversation`` — so this script derives the human instructions
from that object rather than letting someone re-type them into Qualtrics,
where they would silently drift the moment the rubric is edited.

What is stripped
----------------
The judge's prompt contains three parts. Only the middle one is
instruction; the other two are machine plumbing that would confuse or
mislead a human:

1. **The stimulus header** — ``User message: {user_message}`` and friends.
   Qualtrics renders the stimulus itself from the CSV's Loop & Merge
   fields, so the placeholder lines are dropped. Everything up to and
   including the last ``{placeholder}`` line goes.

2. **The question and its 1-3 scale** — kept verbatim. This is what the
   participant must read.

3. **The output-format directives** — ``Output your response in the
   format below``, ``#scores``, ``#overall_reasoning``, ``a <reasoning>``,
   ``b <answer>``, and the "always provide a numeric score" tips. A human
   answers with a radio button, so these are dropped: everything from the
   first ``Output your response(s) in the format`` line (or the first
   ``#directive`` line) onward.

The per-dimension system prompt is kept, since it is the framing the judge
was given, but it is reworded from "You are a helpful red teaming
assistant" only if ``--rewrite-persona`` is passed — by default it is
reproduced verbatim so the two arms provably read the same text.

Usage
-----
::

    cd scripts/final_experiment

    python -m human_evaluation.export_rubric_instructions \
        --out-dir ../../results/final_experiment/main_user_IVs/20260427_165233/human_evaluation

    # Inspect without writing
    python -m human_evaluation.export_rubric_instructions --stdout
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import RUBRICS, MultiPromptRubric  # noqa: E402

from . import config as cfg  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
# Stripping
# ════════════════════════════════════════════════════════════════════════════

# Every rubric's format tail opens with this sentence (singular or plural
# "response"); see core/scoring.py _SPLIT_FORMAT_TAIL and the MISINFO_V1
# templates. A leading "#directive" line is the fallback marker for any
# future template that omits the sentence.
_FORMAT_TAIL_RE = re.compile(
    r"^\s*(?:Output your responses? in the format\b|#\w+)", re.MULTILINE,
)

# A line carrying a str.format placeholder, e.g. "User message: {user_message}".
_PLACEHOLDER_RE = re.compile(r"\{[a-z_]+\}")


def strip_to_instructions(user_template: str) -> str:
    """Reduce a judge prompt template to the human-readable instruction.

    Drops the stimulus header (everything through the last placeholder
    line) and the output-format directives (everything from the format
    tail onward), leaving the question and its scale definitions.
    """
    # 1. Cut the format tail.
    tail = _FORMAT_TAIL_RE.search(user_template)
    body = user_template[: tail.start()] if tail else user_template

    # 2. Cut the stimulus header: find the last placeholder line and keep
    #    only what follows it. Searching for the *last* one (rather than
    #    splitting on a known header constant) keeps this working for
    #    templates that interpolate {history} or other extra fields.
    lines = body.split("\n")
    last_placeholder = -1
    for i, line in enumerate(lines):
        if _PLACEHOLDER_RE.search(line):
            last_placeholder = i
    instruction = "\n".join(lines[last_placeholder + 1:])

    # 3. Tidy: collapse 3+ blank lines, trim.
    return re.sub(r"\n{3,}", "\n\n", instruction).strip()


def split_scale(instruction: str) -> tuple[str, list[tuple[str, str]]]:
    """Separate the question from its numbered scale points.

    Returns (question, [(level, definition), ...]). Scale lines look like
    ``1 = Does not indicate anything is wrong with the claim.``; anything
    before the first such line is the question. If no scale lines are
    found the whole instruction is returned as the question with an empty
    scale, so an unusual rubric degrades to "show the text as-is" rather
    than losing content.
    """
    scale_re = re.compile(r"^\s*(\d+)\s*=\s*(.+)$")
    question_lines: list[str] = []
    scale: list[tuple[str, str]] = []
    current: list[str] | None = None

    for line in instruction.split("\n"):
        m = scale_re.match(line)
        if m:
            current = [m.group(2).strip()]
            scale.append((m.group(1), ""))
            scale[-1] = (m.group(1), current[0])
            continue
        if current is not None and line.strip():
            # Continuation of a wrapped scale definition.
            current.append(line.strip())
            scale[-1] = (scale[-1][0], " ".join(current))
            continue
        if current is None:
            question_lines.append(line)

    return "\n".join(question_lines).strip(), scale


# ════════════════════════════════════════════════════════════════════════════
# Rendering
# ════════════════════════════════════════════════════════════════════════════

def _dimension_records(rubric_name: str) -> list[dict]:
    """Pull one instruction record per rubric dimension."""
    rubric = RUBRICS.get(rubric_name)
    if rubric is None:
        raise SystemExit(
            f"Unknown rubric {rubric_name!r}. Available: {sorted(RUBRICS)}"
        )
    if not isinstance(rubric, MultiPromptRubric):
        raise SystemExit(
            f"{rubric_name} is a single-prompt rubric; task 1 expects the "
            "multi-prompt split rubric where each dimension has its own "
            "instruction."
        )

    records = []
    for dim in rubric.dimensions:
        spec = rubric.prompts[dim]
        instruction = strip_to_instructions(spec.user_template)
        question, scale = split_scale(instruction)
        records.append({
            "dimension": dim,
            "system_prompt": spec.system_prompt.strip(),
            "question": question,
            "scale": [{"level": lv, "definition": d} for lv, d in scale],
            "score_range": list(rubric.score_range),
        })
    return records


def render_markdown(rubric_name: str, records: list[dict]) -> str:
    """Human-readable instruction sheet."""
    out = [
        f"# Annotation instructions — `{rubric_name}`",
        "",
        "You will see an isolated exchange: a message from a user who holds "
        "a false belief, the belief itself, and one AI model's reply. Rate "
        "**that reply only**. You will not see the rest of the conversation.",
        "",
        "Rate every response on all three dimensions below. The scales are "
        "reproduced verbatim from the instructions given to the automated "
        "evaluator whose ratings yours are being compared against.",
        "",
    ]
    for rec in records:
        lo, hi = rec["score_range"]
        out += [
            f"## {rec['dimension'].capitalize()}",
            "",
            f"*Framing:* {rec['system_prompt']}",
            "",
            rec["question"],
            "",
        ]
        for point in rec["scale"]:
            out.append(f"- **{point['level']}** — {point['definition']}")
        if not rec["scale"]:
            out.append(f"_(scale {lo}-{hi}; see question text)_")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_html(rubric_name: str, records: list[dict]) -> str:
    """Single-line-per-block HTML for pasting into a Qualtrics question."""
    parts = [f"<h3>Annotation instructions — {html.escape(rubric_name)}</h3>"]
    for rec in records:
        parts.append(f"<h4>{html.escape(rec['dimension'].capitalize())}</h4>")
        parts.append(f"<p><em>{html.escape(rec['system_prompt'])}</em></p>")
        parts.append(f"<p>{html.escape(rec['question']).replace(chr(10), '<br>')}</p>")
        if rec["scale"]:
            items = "".join(
                f"<li><strong>{html.escape(p['level'])}</strong> — "
                f"{html.escape(p['definition'])}</li>"
                for p in rec["scale"]
            )
            parts.append(f"<ul>{items}</ul>")
    return "\n".join(parts) + "\n"


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--rubric", type=str, default=cfg.RUBRIC_NAME)
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Directory for the instruction files. Omit with --stdout.",
    )
    p.add_argument("--task", type=str, default=cfg.TASK_ID)
    p.add_argument(
        "--stdout", action="store_true",
        help="Print the markdown to stdout instead of writing files.",
    )
    args = p.parse_args()

    records = _dimension_records(args.rubric)
    markdown = render_markdown(args.rubric, records)

    if args.stdout or args.out_dir is None:
        print(markdown)
        if args.out_dir is None and not args.stdout:
            print("(no --out-dir given; nothing written)", file=sys.stderr)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.task}_rubric_instructions"

    md_path = args.out_dir / f"{stem}.md"
    md_path.write_text(markdown, encoding="utf-8")

    html_path = args.out_dir / f"{stem}.html"
    html_path.write_text(render_html(args.rubric, records), encoding="utf-8")

    # Structured form, so a survey-generation step (or the agreement
    # analysis, when labelling its output) can consume the scale points
    # without re-parsing prose.
    json_path = args.out_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(
            {
                "task": args.task,
                "rubric_name": args.rubric,
                "source": "core.scoring.RUBRICS",
                "stripped": ["stimulus_placeholders", "output_format_directives"],
                "dimensions": records,
                "created_at": datetime.now().isoformat(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    for path in (md_path, html_path, json_path):
        print(f"[write] {path.name}")
    print(f"\n[dims] {', '.join(r['dimension'] for r in records)}")


if __name__ == "__main__":
    main()
