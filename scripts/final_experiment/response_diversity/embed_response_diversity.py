"""
response_diversity/embed_response_diversity.py
==============================================
Embed every target-LLM response across the 8 turns of each session and
compute the per-turn diversity div_{s,t} and session-level diversity D_s
described in the module docstring.

Definitions
-----------
For a session s with target-response embeddings e_{s,1}, …, e_{s,8} and
t in [2, 8]::

    div_{s,t} = 1 - (1/(t-1)) * sum_{i=1}^{t-1} cos(e_{s,i}, e_{s,t})

    D_s = (1/7) * sum_{t=2}^{8} div_{s,t}

Outputs
-------
Two CSVs written under SOURCE_RUN_DIR (paths configurable via CLI):

- response_diversity_per_turn.csv
    One row per (session, turn) for turn in [2, 8].
    Columns: session_id, target_model, iv1, iv2, belief_category,
             is_control, turn, div

- response_diversity_session.csv
    One row per session.
    Columns: session_id, target_model, iv1, iv2, belief_category,
             is_control, D_s

Claude sessions are skipped (see config.INCLUDED_TARGET_MODELS).

Sessions are stratified-sampled per (target_llm, iv1, belief_category)
cell at ``SAMPLE_FRACTION`` (see config.py). The sampling plan is written
to ``response_diversity_sample_index.json`` inside the source run so the
plan is reproducible and the same subset can be resumed after an
interrupt. Pass ``--resample`` to draw a fresh plan.

Usage::

    cd scripts/final_experiment
    python -m response_diversity.embed_response_diversity \\
        [--workers 8] [--batch-size 64] \\
        [--sample-fraction 0.12] [--sampling-seed 42] \\
        [--resample] [--rebuild]

Resume:
- If the per-turn output CSV already exists and --rebuild is not passed,
  the script exits immediately.
- If the sample index already exists and --resample is not passed, the
  existing plan is reused (so re-runs embed the same subset).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import litellm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from . import config as cfg  # noqa: E402
from . import sampling  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(threadName)-12s | %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_EMBED_MODEL = "openrouter/openai/text-embedding-3-large"
DEFAULT_PER_TURN_OUTPUT = cfg.PER_TURN_OUTPUT_PATH
DEFAULT_SESSION_OUTPUT = cfg.SESSION_OUTPUT_PATH
DEFAULT_SAMPLE_INDEX_PATH = cfg.SAMPLE_INDEX_PATH
DEFAULT_SAMPLE_FRACTION = cfg.SAMPLE_FRACTION
DEFAULT_SAMPLING_SEED = cfg.SAMPLING_SEED
DEFAULT_BATCH_SIZE = 64
DEFAULT_WORKERS = 8

EXPECTED_TURNS = cfg.N_TURNS  # 8


# ════════════════════════════════════════════════════════════════════════════
# Data types
# ════════════════════════════════════════════════════════════════════════════

class SessionMeta(NamedTuple):
    session_id:        str
    target_model:      str    # short slug: gpt-5.3-chat, gemini-3-flash-preview, deepseek-v3.2
    iv1:               str
    iv2:               str
    belief_category:   str
    is_control:        bool
    target_responses:  list[str]   # length == N_TURNS


# ════════════════════════════════════════════════════════════════════════════
# Loading
# ════════════════════════════════════════════════════════════════════════════

def _short_model(model_str: str) -> str:
    """Normalize the target_llm string to the short slug used elsewhere."""
    short = model_str.split("/")[-1]
    # Match the notebook's normalization: gemini-3-flash → gemini-3-flash-preview
    if short == "gemini-3-flash":
        return "gemini-3-flash-preview"
    return short


def _model_matches_included(short_slug: str) -> bool:
    """Match either the raw slug or the notebook-normalized slug."""
    if short_slug in cfg.INCLUDED_TARGET_MODELS:
        return True
    if short_slug == "gemini-3-flash-preview" and "gemini-3-flash" in cfg.INCLUDED_TARGET_MODELS:
        return True
    return False


def _load_session(path: Path) -> SessionMeta | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None

    turns = data.get("turns") or []
    if not turns:
        return None

    cell = data.get("cell") or {}
    belief = data.get("belief") or {}
    models = data.get("models") or {}

    target_model_full = str(models.get("target_llm", ""))
    short_slug = _short_model(target_model_full)
    if not _model_matches_included(short_slug):
        return None

    responses = [str(t.get("target_response") or "").strip() for t in turns]
    if not all(responses):
        return None

    return SessionMeta(
        session_id=str(data.get("session_id", path.stem)),
        target_model=short_slug,
        iv1=str(cell.get("iv1", "unknown")),
        iv2=str(cell.get("iv2", "unknown")),
        belief_category=str(belief.get("category", "unknown")),
        is_control=bool(cell.get("is_control", False)),
        target_responses=responses,
    )


def load_sessions_from_sample(
    sample_entries: list[sampling.SampleEntry],
) -> list[SessionMeta]:
    """Load only the sessions listed in the sample plan."""
    sessions: list[SessionMeta] = []
    skipped_missing = 0
    skipped_turns = 0

    for entry in sample_entries:
        path = Path(entry.source_path)
        if not path.exists():
            skipped_missing += 1
            logger.warning("Missing conversation file: %s", path)
            continue

        meta = _load_session(path)
        if meta is None:
            skipped_missing += 1
            continue

        if len(meta.target_responses) != EXPECTED_TURNS:
            skipped_turns += 1
            logger.warning(
                "Skipping %s: has %d turns (expected %d).",
                path.name, len(meta.target_responses), EXPECTED_TURNS,
            )
            continue

        sessions.append(meta)

    logger.info(
        "Loaded %d sampled sessions. Skipped: %d missing/invalid, %d wrong turn count.",
        len(sessions), skipped_missing, skipped_turns,
    )
    return sessions


# ════════════════════════════════════════════════════════════════════════════
# Embedding helpers (same style as embed_trajectory_similarities.py)
# ════════════════════════════════════════════════════════════════════════════

def _embed_batch(texts: list[str], model: str) -> list[list[float]]:
    response = litellm.embedding(model=model, input=texts)
    return [
        item["embedding"] if isinstance(item, dict) else item.embedding
        for item in response.data
    ]


def embed_texts(
    texts: list[str],
    *,
    model: str,
    batch_size: int,
    n_workers: int,
    label: str = "texts",
) -> np.ndarray:
    """Embed a list of texts in parallel batches; returns (N, D) float32."""
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    logger.info(
        "Embedding %d %s → %d batch(es), %d worker(s).",
        len(texts), label, len(batches), n_workers,
    )

    results: dict[int, list[list[float]]] = {}
    errors: list[tuple[int, str]] = []

    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="emb") as pool:
        future_to_idx = {
            pool.submit(_embed_batch, batch, model): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
                done = len(results)
                if done % max(1, len(batches) // 10) == 0 or done == len(batches):
                    logger.info("  %s: %d/%d batches", label, done, len(batches))
            except Exception as exc:
                errors.append((idx, traceback.format_exc()))
                logger.error("Batch %d of '%s' failed: %s", idx, label, exc)

    if errors:
        raise RuntimeError(
            f"{len(errors)} batch(es) failed for '{label}'. "
            f"First error:\n{errors[0][1]}"
        )

    ordered = [emb for _, emb in sorted(results.items())]
    flat = [emb for batch_embs in ordered for emb in batch_embs]
    return np.array(flat, dtype=np.float32)


def _dedup(texts: list[str]) -> tuple[list[str], list[int]]:
    """Return (unique_texts, indices) s.t. texts[i] == unique_texts[indices[i]]."""
    seen: dict[str, int] = {}
    unique: list[str] = []
    indices: list[int] = []
    for t in texts:
        if t not in seen:
            seen[t] = len(unique)
            unique.append(t)
        indices.append(seen[t])
    return unique, indices


def _normalize(A: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.where(norms > 0, norms, 1.0)


# ════════════════════════════════════════════════════════════════════════════
# Diversity computation
# ════════════════════════════════════════════════════════════════════════════

def compute_session_diversity(embeddings: np.ndarray) -> np.ndarray:
    """Compute per-turn diversity for one session.

    embeddings: (T, D) matrix, T == N_TURNS.
    Returns: (T-1,) vector giving div_{s,t} for t in [2..T] (1-indexed),
             i.e. div[0] corresponds to turn 2.
    """
    normed = _normalize(embeddings)             # (T, D)
    # Cosine similarity matrix (T, T); sim[i, j] = cos(e_i, e_j)
    sim = normed @ normed.T                     # (T, T)

    T = embeddings.shape[0]
    div = np.zeros(T - 1, dtype=np.float64)
    for t in range(2, T + 1):  # 1-indexed turn 2..T
        # mean cosine with all prior turns 1..t-1
        prior_sims = sim[t - 1, : t - 1]       # (t-1,)
        div[t - 2] = 1.0 - float(prior_sims.mean())
    return div


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Embed target-LLM responses and compute per-session "
                    "response diversity D_s."
    )
    p.add_argument("--workers",     type=int,  default=DEFAULT_WORKERS)
    p.add_argument("--batch-size",  type=int,  default=DEFAULT_BATCH_SIZE)
    p.add_argument("--model",       default=DEFAULT_EMBED_MODEL)
    p.add_argument("--per-turn-output", type=Path, default=DEFAULT_PER_TURN_OUTPUT)
    p.add_argument("--session-output",  type=Path, default=DEFAULT_SESSION_OUTPUT)
    p.add_argument("--sample-index",    type=Path, default=DEFAULT_SAMPLE_INDEX_PATH,
                   help="Path where the sampling plan is (re)used from.")
    p.add_argument("--sample-fraction", type=float, default=DEFAULT_SAMPLE_FRACTION,
                   help="Stratified per-cell sampling fraction "
                        "(ignored when re-using an existing --sample-index).")
    p.add_argument("--sampling-seed",   type=int,   default=DEFAULT_SAMPLING_SEED)
    p.add_argument("--resample", action="store_true",
                   help="Rebuild the sample index even if it already exists.")
    p.add_argument("--rebuild", action="store_true",
                   help="Re-run embedding even if output CSVs already exist.")
    args = p.parse_args()

    if args.per_turn_output.exists() and args.session_output.exists() and not args.rebuild:
        print(
            f"Outputs already exist:\n  {args.per_turn_output}\n  {args.session_output}\n"
            "Pass --rebuild to overwrite."
        )
        return

    conv_dir = cfg.SOURCE_CONV_DIR
    if not conv_dir.exists():
        raise SystemExit(f"Conversations dir not found: {conv_dir}")

    # ── 1. Build (or load) the stratified sample ─────────────────────────
    print(f"\nConversations dir: {conv_dir}")

    if args.sample_index.exists() and not args.resample:
        print(f"Reusing existing sample index: {args.sample_index}")
        sample_entries = sampling.read_sample_index(args.sample_index)
        print(f"  {len(sample_entries)} planned sessions from index.")
    else:
        sample_entries = sampling.build_sample(
            conv_dir,
            sample_fraction=args.sample_fraction,
            seed=args.sampling_seed,
            included_models=cfg.INCLUDED_TARGET_MODELS,
        )
        sampling.write_sample_index(
            args.sample_index,
            planned=sample_entries,
            params={
                "sample_fraction": args.sample_fraction,
                "seed":            args.sampling_seed,
                "included_models": list(cfg.INCLUDED_TARGET_MODELS),
                "n_turns":         EXPECTED_TURNS,
            },
        )
        print(f"Wrote sample index → {args.sample_index}")

    if not sample_entries:
        raise SystemExit("Sample is empty — check fraction and model filter.")

    # ── 2. Load sampled sessions ──────────────────────────────────────────
    sessions = load_sessions_from_sample(sample_entries)
    if not sessions:
        raise SystemExit("No sessions loaded from sample index.")
    print(f"Sessions loaded: {len(sessions)}")

    # ── 3. Build flat text list and dedupe ────────────────────────────────
    flat_texts: list[str] = []
    session_flat_ranges: list[tuple[int, int]] = []   # (start, end) into flat_texts per session
    for sess in sessions:
        start = len(flat_texts)
        flat_texts.extend(sess.target_responses)
        session_flat_ranges.append((start, len(flat_texts)))

    uniq_texts, idx_map = _dedup(flat_texts)
    print(f"Unique target responses: {len(uniq_texts):,}  "
          f"(from {len(flat_texts):,} total responses)")

    # ── 4. Embed ──────────────────────────────────────────────────────────
    emb_u = embed_texts(
        uniq_texts,
        model=args.model,
        batch_size=args.batch_size,
        n_workers=args.workers,
        label="target_responses",
    )
    emb_flat = emb_u[np.array(idx_map)]  # (n_sessions * N_TURNS, D)

    # ── 5. Compute per-turn diversity and D_s per session ────────────────
    print("\nComputing per-turn diversity and D_s ...")
    args.per_turn_output.parent.mkdir(parents=True, exist_ok=True)
    args.session_output.parent.mkdir(parents=True, exist_ok=True)

    per_turn_fields = [
        "session_id", "target_model", "iv1", "iv2", "belief_category",
        "is_control", "turn", "div",
    ]
    session_fields = [
        "session_id", "target_model", "iv1", "iv2", "belief_category",
        "is_control", "D_s",
    ]

    n_per_turn = 0
    n_sessions = 0

    with args.per_turn_output.open("w", encoding="utf-8", newline="") as pt_fh, \
         args.session_output.open("w", encoding="utf-8", newline="") as ss_fh:

        pt_writer = csv.DictWriter(pt_fh, fieldnames=per_turn_fields)
        ss_writer = csv.DictWriter(ss_fh, fieldnames=session_fields)
        pt_writer.writeheader()
        ss_writer.writeheader()

        for sess, (start, end) in zip(sessions, session_flat_ranges):
            emb_session = emb_flat[start:end]          # (T, D)
            div_vec = compute_session_diversity(emb_session)   # (T-1,) for t=2..T
            d_s = float(div_vec.mean())

            for offset, div_val in enumerate(div_vec):
                pt_writer.writerow({
                    "session_id":      sess.session_id,
                    "target_model":    sess.target_model,
                    "iv1":             sess.iv1,
                    "iv2":             sess.iv2,
                    "belief_category": sess.belief_category,
                    "is_control":      sess.is_control,
                    "turn":            offset + 2,      # turns 2..T
                    "div":             round(float(div_val), 6),
                })
                n_per_turn += 1

            ss_writer.writerow({
                "session_id":      sess.session_id,
                "target_model":    sess.target_model,
                "iv1":             sess.iv1,
                "iv2":             sess.iv2,
                "belief_category": sess.belief_category,
                "is_control":      sess.is_control,
                "D_s":             round(d_s, 6),
            })
            n_sessions += 1

    print(f"\nSaved {n_per_turn:,} per-turn rows → {args.per_turn_output}")
    print(f"Saved {n_sessions:,} session rows → {args.session_output}")

    # ── 6. Quick summary ──────────────────────────────────────────────────
    import pandas as pd
    df = pd.read_csv(args.session_output)
    print("\nMean D_s by target_model:")
    print(df.groupby("target_model")["D_s"].agg(["count", "mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
