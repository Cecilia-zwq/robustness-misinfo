"""
human_evaluation/config.py
==========================
Configuration for the human arm of evaluator validation.

Defaults here are deliberately aligned with ``evaluator_validation/config.py``
so the human annotation set is drawn from *exactly* the same sample the
secondary-LLM agreement analysis uses. Changing a value here without
changing its counterpart there breaks the join back to the LLM scores.
"""

from __future__ import annotations


# ── Provenance: where the sample comes from ─────────────────────────────────
# The human set is not re-sampled. It reads the manifest produced by
#   evaluator_validation/run_validation_scoring.py --sample-only
# so both arms score the identical (session_id, turn) units.
VALIDATION_SUBDIR = "evaluator_validation"
SAMPLE_MANIFEST_NAME = "sample_manifest.json"

# Rubric the humans code against — the same one the LLM judge used, so the
# instructions handed to participants are generated from this name by
# export_rubric_instructions.py rather than transcribed by hand.
RUBRIC_NAME = "misinfo_v1_split"


# ── Tasks ───────────────────────────────────────────────────────────────────
# Readme.md defines three human evaluations. Their *stimulus unit* differs,
# which is what decides whether they can share a CSV builder:
#
#   task1 — original framework metrics (correction / rebuttal /
#           agreeableness). Stimulus = an EXTRACTED LOCAL RESPONSE: one
#           isolated (user_message, ai_message) pair, no conversation
#           context, coded against the same instructions the LLM judge got.
#           One CSV row per rated unit -> make_annotation_csv.py.
#   task2 — epistemic endorsement vs affective validation. Stimulus = a
#           FULL CONVERSATION, still rated per turn. One row must carry a
#           whole transcript, so it needs its own builder, not this one.
#   task3 — in-person thematic analysis of conversation transcripts.
#           No Qualtrics CSV; transcripts are exported for coding.
#
# The task id prefixes every output filename so the tasks' CSVs and
# manifests coexist in one directory without collision, and so an exported
# Qualtrics response file can be traced back to the task that produced it.
TASK_ID = "task1"

# ── Output layout under <run_dir>/ ──────────────────────────────────────────
#   <run_dir>/
#     └── human_evaluation/
#           ├── task1_annotation_items.csv          ← Qualtrics L&M source
#           ├── task1_annotation_items_block01.csv  ← optional block splits
#           ├── task1_annotation_items_manifest.json
#           └── task1_rubric_instructions.md        ← participant-facing text
HUMAN_SUBDIR = "human_evaluation"
CSV_STEM_SUFFIX = "annotation_items"


def csv_stem(task_id: str = TASK_ID) -> str:
    """Filename stem for a task's outputs, e.g. 'task1_annotation_items'."""
    return f"{task_id}_{CSV_STEM_SUFFIX}"


# ── Item selection: one turn per conversation ───────────────────────────────
# Turns within a conversation are strongly correlated (ICC ~= 0.57 on
# correction for this run), so m turns drawn from one conversation are worth
# only 1 + (m - 1) * ICC independent items. All 8 turns of a conversation buy
# ~1.6 items' worth of precision, not 8. Taking exactly ONE turn from each of
# the 600 sampled conversations therefore maximises effective sample size per
# unit of annotator time, while keeping the full 30 beliefs x 5 conditions x
# 4 target models coverage of the LLM-arm sample.
TURNS_PER_CONVERSATION = 1

# Turn position is the single strongest predictor of score (on correction,
# 57% of turn-1 responses score 3 vs 25% at turn 8), so which turn each
# conversation contributes must be spread across 1-8 rather than fixed.
#
# "rotation" (default) assigns turn = (belief_i + bucket_j) mod 8 + 1 over the
# belief x (cell x model) grid. Deterministic and near-perfectly balanced:
# 74-76 conversations per turn overall, 14-16 per (turn x condition), 17-20
# per (turn x model), and every belief covers all 8 positions.
# "random" draws turns i.i.d. per conversation from a seeded RNG — simpler to
# describe, but leaves margins uneven by chance at this n.
TURN_ASSIGNMENT = "rotation"
N_TURNS_PER_CONVERSATION_POOL = 8

# ── Rater redundancy and blocking ───────────────────────────────────────────
# Every item is rated by RATERS_PER_ITEM independent crowdworkers, and the
# per-item human score is their consensus. This supersedes the earlier
# "overlap core" design (a subset double-coded, the rest coded once), which
# was a concession to expensive expert coders: with Prolific, full redundancy
# is affordable and strictly better — each item's human score is a consensus
# rather than one person's opinion, and inter-rater reliability is estimable
# across the whole set and within every subgroup, not just on a core.
RATERS_PER_ITEM = 3

# Items are shuffled, cut into blocks of BLOCK_SIZE, and each block is
# rotationally assigned to raters, BLOCKS_PER_RATER blocks each.
#
# Feasibility constraints: with N items and B = N / BLOCK_SIZE blocks, the
# rotation needs BLOCKS_PER_RATER to divide B, and yields
#     n_raters = RATERS_PER_ITEM * B / BLOCKS_PER_RATER
#
# At N=600: blocks of 15, 1 block per rater -> 40 blocks, 120 raters, 15
# items each, one ~42-minute sitting with no break. 15 is a divisor of 600,
# so every block is full and banding (below) gives 15 length bands of 40.
BLOCK_SIZE = 15
BLOCKS_PER_RATER = 1

# ── Session structure ───────────────────────────────────────────────────────
# Consent, instructions, practice items and demographics are a fixed ~7 min
# cost paid once per rater. One block of 15 is a single sitting: long enough
# to amortise that overhead, short enough to need no break.
#
# Measured from this run's text (419 words/item mean: 473 for long-text
# beliefs, 382 for short) at 200-220 wpm plus ~20s to rate three dimensions:
#
#     onboarding      ~7 min
#     session         15 items  ~33-37 min
#     ------------------------------------
#     total           15 items  ~40-44 min
#
# Blocks are banded on word count precisely so this estimate holds for
# every rater rather than on average. Raising BLOCKS_PER_RATER to 2 would
# restore a two-session design with a BREAK_MINUTES break between blocks.
SESSIONS_PER_RATER = 1
BREAK_MINUTES = 5
ONBOARDING_MINUTES = 7.0
# Reading speed and per-item rating overhead used for the printed estimate.
WORDS_PER_MINUTE = 210
MINUTES_PER_ITEM_RATING = 0.33

# ── Block composition ───────────────────────────────────────────────────────
# Blocks are built so that every rater's workload is equivalent in *reading
# time*, not just in item count.
#
# The binary is_long_text flag is too coarse for this on its own: item length
# varies enormously within both levels (AI responses alone run 4 to 1060
# words), so two blocks with the same long/short split can still differ by
# tens of minutes. Instead, items are ranked by total word count and cut into
# BLOCK_SIZE bands of equal size; each block takes exactly one item from each
# band (15 bands of 40 at the defaults). That gives every rater
#   * near-identical total reading load (one item per length decile), and
#   * a mix spanning the shortest to the longest items, so no one gets a
#     block of uniformly punishing text — which is the fatigue risk.
#
# Within each band, WHICH item goes to WHICH block is then chosen greedily to
# spread the design factors below, so length balance and factor balance are
# optimised together rather than trading off.
BLOCK_BALANCE_KEYS = ("cell_id", "target_model", "turn", "category")

# The balance key the swap-repair pass weights most heavily. Category gets
# priority because it is the belief sample's stratum and because it is
# entangled with length (the long-text beliefs are all fake_health or
# fake_news), so the length bands alone would skew it. With 600 items in
# 40 blocks the proportional block is exactly 4/2/3/2/4, which the repair
# reaches for every block.
BLOCK_PRIORITY_KEY = "category"

# Item order inside a block. "shuffle" (default) randomises so that position
# effects (fatigue, drift in severity) are not confounded with item length.
# "ascending" eases raters in with the shortest item first, at the cost of
# making position and length collinear.
BLOCK_ITEM_ORDER = "shuffle"

# Seed for shuffling, blocking, rotation offsets, and --turn-assignment
# random. Distinct from evaluator_validation's SAMPLE_SEED (42): that seed
# decides which units are in the sample, this one only allocation within it.
SEED = 7


# ── Qualtrics formatting ────────────────────────────────────────────────────
# Qualtrics' Loop & Merge "import from file" parser treats a newline as a
# row terminator **even inside a quoted field**, so every cell we emit must be
# newline-free. "html" satisfies that (newlines become <br>) and also
# renders the markdown the target models emit; "plain" collapses whitespace
# to single spaces; "raw" keeps the text verbatim (multi-line cells —
# only safe if you are loading the CSV with pandas, not Qualtrics).
DEFAULT_TEXT_FORMAT = "html"

# Qualtrics writes the CSV's bytes straight into the survey; a UTF-8 BOM
# is what its importer (and Excel, if the file is eyeballed first) expects.
DEFAULT_ENCODING = "utf-8-sig"

# The whole schema. Loop & Merge references fields positionally as
# ${lm://Field/1}, ${lm://Field/2}, ... so this order is load-bearing.
#
# Columns 3-5 are the STIMULUS, and they are ordered to mirror the judge's
# prompt header (core.scoring._SPLIT_HEADER) exactly:
#     User message: {user_message}
#     The misinformation believed by the user is: {misinformation_belief}
#     AI model response: {response}
# The belief is in the CSV because the LLM judge is shown it; Readme.md
# requires participants read the same instructions, so they must see the
# same stimulus. Long-text beliefs are rendered "[title] body", matching
# core.scoring._format_belief_for_evaluator.
#
# session_id is a structured key —
# cell-{iv1}__{iv2}__belief-{category}-{index}__model-{slug} — so condition,
# belief and target model are recoverable by parsing it. belief_category is
# carried explicitly as well, because it is the stratum the belief sample
# was drawn on and the first thing a per-block or per-response breakdown
# needs; exporting it saves every downstream consumer from re-parsing.
# It is appended LAST so Field/1-6 keep their positions in any survey
# already wired to them.
COLUMNS = (
    "session_id",             # Field/1 — hidden, join key
    "turn",                   # Field/2 — hidden, join key
    "user_message",           # Field/3
    "misinformation_belief",  # Field/4
    "ai_message",             # Field/5
    "is_long_text",           # Field/6 — hidden, block-balancing key
    "belief_category",        # Field/7 — hidden, sampling stratum
)

# Columns an annotator should never be shown in the rendered question.
# Recorded here so the survey-building step has a single source of truth.
HIDDEN_COLUMNS = ("session_id", "turn", "is_long_text", "belief_category")
