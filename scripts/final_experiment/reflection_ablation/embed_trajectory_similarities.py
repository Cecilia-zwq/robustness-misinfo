"""
reflection_ablation/embed_trajectory_similarities.py
=====================================================
For each matched (original, no-reflection) session pair, embed every
user message across all 8 turns and compute per-turn cosine similarity
to the session's false belief claim and character instruction.

The output CSV has one row per (pair_id × condition × turn) and is the
primary input for the PERMANOVA / Wilcoxon trajectory analysis in the
notebook.

Schema
------
pair_id          str   source session id (without __noref suffix)
condition        str   "original" | "noref"
branch_turn      int   turn number where the conversation branched
iv1              str   e.g. "emotional", "logical", "none"
belief_category  str   e.g. "bias", "climate"
target_llm       str   full model string
is_control       bool  True when iv1 == "none" (no character prompt)
turn             int   1-indexed turn number
sim_belief       float cosine similarity of user_message to false_belief
sim_char         float cosine similarity of user_message to char_instr
                       (NaN for control sessions with no character prompt)

Usage::

    cd scripts/final_experiment
    python -m reflection_ablation.embed_trajectory_similarities \\
        [--workers 8] [--batch-size 64] [--rebuild]

Resume: if the output CSV already exists and --rebuild is not passed the
script exits immediately.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(threadName)-12s | %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_EMBED_MODEL = "openrouter/openai/text-embedding-3-large"
DEFAULT_OUTPUT_PATH = cfg.SOURCE_RUN_DIR / "trajectory_similarities.csv"
DEFAULT_BATCH_SIZE  = 64
DEFAULT_WORKERS     = 8

NOREF_DIR = cfg.OUTPUT_CONV_DIR


# ════════════════════════════════════════════════════════════════════════════
# Data types
# ════════════════════════════════════════════════════════════════════════════

class SessionMeta(NamedTuple):
    pair_id:         str
    condition:       str          # "original" | "noref"
    branch_turn:     int
    iv1:             str
    belief_category: str
    target_llm:      str
    is_control:      bool
    user_messages:   list[str]    # one per turn (length == n_turns)
    false_belief:    str
    char_instr:      str          # "" for control sessions


# ════════════════════════════════════════════════════════════════════════════
# Loading
# ════════════════════════════════════════════════════════════════════════════

def _load_session(data: dict, path: Path, condition: str, branch_turn: int) -> SessionMeta | None:
    """Build a SessionMeta from a pre-loaded JSON dict."""
    turns = data.get("turns") or []
    if not turns:
        return None

    cell   = data.get("cell")   or {}
    belief = data.get("belief") or {}
    models = data.get("models") or {}

    src_id = data.get("config", {}).get("source_session_id") or data.get("session_id", path.stem)
    pair_id = src_id.removesuffix(cfg.SESSION_ID_SUFFIX) if src_id.endswith(cfg.SESSION_ID_SUFFIX) else src_id

    user_msgs = [str(t.get("user_message") or "").strip() for t in turns]
    if not any(user_msgs):
        return None

    return SessionMeta(
        pair_id=pair_id,
        condition=condition,
        branch_turn=branch_turn,
        iv1=str(cell.get("iv1", "unknown")),
        belief_category=str(belief.get("category", "unknown")),
        target_llm=str(models.get("target_llm", "unknown")),
        is_control=bool(cell.get("is_control", False)),
        user_messages=user_msgs,
        false_belief=str(belief.get("content") or "").strip(),
        char_instr=str(cell.get("character_prompt") or "").strip(),
    )


def _is_fallback(turns: list[dict], turn_1indexed: int) -> bool:
    """Return True if turns[turn_1indexed - 1] has is_fallback set."""
    idx = turn_1indexed - 1
    if idx < 0 or idx >= len(turns):
        return False
    v = turns[idx].get("is_fallback", False)
    return v if isinstance(v, bool) else str(v).lower() == "true"


EXPECTED_TURNS = cfg.N_TURNS   # 8; every session must have exactly this many turns


def load_sessions(noref_dir: Path, orig_dir: Path) -> list[SessionMeta]:
    """Load all matched (original, noref) session pairs.

    Acceptance criteria (pair is skipped with a warning if any fails):
    - Original file exists on disk.
    - The original session's branch_turn is NOT a fallback turn.
      Fallback turns mean reflection exhausted all retries and sent a default
      message; the original user_message no longer represents the reflection
      module's output, so including such pairs would measure "fallback vs
      rejected-draft" rather than "reflection vs no-reflection".
    - Both sessions have exactly EXPECTED_TURNS turns.
    """
    sessions: list[SessionMeta] = []
    missing_orig     = 0
    skipped_fallback = 0
    skipped_turns    = 0

    for noref_path in sorted(noref_dir.glob("*.json")):
        try:
            with noref_path.open("r", encoding="utf-8") as fh:
                noref_data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        bt     = int(noref_data.get("config", {}).get("branch_turn", 1))
        src_id = noref_data.get("config", {}).get("source_session_id", "")
        orig_path = orig_dir / f"{src_id}.json"

        if not orig_path.exists():
            missing_orig += 1
            logger.warning("Missing original for %s", noref_path.name)
            continue

        try:
            with orig_path.open("r", encoding="utf-8") as fh:
                orig_data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        # Skip pairs where the original's branch_turn is a fallback.
        if _is_fallback(orig_data.get("turns") or [], bt):
            skipped_fallback += 1
            logger.warning(
                "Skipping pair %s: original branch_turn=%d is a fallback turn "
                "(reflection exhausted retries; user_message is the default, "
                "not a reflection-passing draft).",
                src_id, bt,
            )
            continue

        noref_meta = _load_session(noref_data, noref_path, "noref",     bt)
        orig_meta  = _load_session(orig_data,  orig_path,  "original",  bt)

        if noref_meta is None or orig_meta is None:
            continue

        n_noref = len(noref_meta.user_messages)
        n_orig  = len(orig_meta.user_messages)
        if n_noref != EXPECTED_TURNS or n_orig != EXPECTED_TURNS:
            skipped_turns += 1
            logger.warning(
                "Skipping pair %s: noref has %d turns, original has %d turns "
                "(expected %d).",
                src_id, n_noref, n_orig, EXPECTED_TURNS,
            )
            continue

        sessions.append(noref_meta)
        sessions.append(orig_meta)

    logger.info(
        "Loaded %d sessions (%d pairs). "
        "Skipped: %d missing originals, %d fallback branch turns, %d wrong turn count.",
        len(sessions), len(sessions) // 2,
        missing_orig, skipped_fallback, skipped_turns,
    )
    return sessions


# ════════════════════════════════════════════════════════════════════════════
# Embedding helpers (same as embed_reflection_pairs.py)
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
    errors:  list[tuple[int, str]] = []

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
    flat    = [emb for batch_embs in ordered for emb in batch_embs]
    return np.array(flat, dtype=np.float32)


def _dedup(texts: list[str]) -> tuple[list[str], list[int]]:
    """Return (unique_texts, indices) s.t. texts[i] == unique_texts[indices[i]]."""
    seen:    dict[str, int] = {}
    unique:  list[str]      = []
    indices: list[int]      = []
    for t in texts:
        if t not in seen:
            seen[t] = len(unique)
            unique.append(t)
        indices.append(seen[t])
    return unique, indices


def _normalize(A: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.where(norms > 0, norms, 1.0)


def row_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two (N, D) arrays."""
    return (_normalize(A) * _normalize(B)).sum(axis=1)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Embed per-turn user messages and compute trajectory similarities."
    )
    p.add_argument("--workers",    type=int,  default=DEFAULT_WORKERS)
    p.add_argument("--batch-size", type=int,  default=DEFAULT_BATCH_SIZE)
    p.add_argument("--model",      default=DEFAULT_EMBED_MODEL)
    p.add_argument("--output",     type=Path, default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--rebuild",    action="store_true",
                   help="Re-run even if output CSV already exists.")
    args = p.parse_args()

    if args.output.exists() and not args.rebuild:
        print(f"Output already exists: {args.output}\nPass --rebuild to overwrite.")
        return

    orig_dir  = cfg.SOURCE_CONV_DIR
    noref_dir = NOREF_DIR

    if not orig_dir.exists():
        raise SystemExit(f"Original conversations dir not found: {orig_dir}")
    if not noref_dir.exists():
        raise SystemExit(f"No-reflection dir not found: {noref_dir}")

    # ── 1. Load session metadata ──────────────────────────────────────────
    print(f"\nOriginal conversations : {orig_dir}")
    print(f"No-reflection sessions : {noref_dir}")
    sessions = load_sessions(noref_dir, orig_dir)
    if not sessions:
        raise SystemExit("No sessions loaded — check directory paths.")
    print(f"Sessions loaded: {len(sessions)} ({len(sessions)//2} pairs)")

    # ── 2. Build flat text lists for embedding ────────────────────────────
    # Flatten user messages: (session_idx, turn_idx) → flat index
    flat_msgs:   list[str] = []
    session_turn_idx: list[tuple[int, int]] = []  # maps flat_idx → (sess, turn)

    for si, sess in enumerate(sessions):
        for ti, msg in enumerate(sess.user_messages):
            flat_msgs.append(msg)
            session_turn_idx.append((si, ti))

    beliefs = [s.false_belief for s in sessions]

    uniq_msgs,   idx_msgs   = _dedup(flat_msgs)
    uniq_beliefs, idx_bels  = _dedup(beliefs)
    # Deduplicate char instrs only for non-control sessions
    non_ctrl_mask   = np.array([not s.is_control for s in sessions])
    non_ctrl_chars  = [s.char_instr for s in sessions if not s.is_control]
    uniq_chars, idx_chars_local = _dedup(non_ctrl_chars) if non_ctrl_chars else ([], [])

    print(f"\nUnique user messages : {len(uniq_msgs):,}  (from {len(flat_msgs):,} turns)")
    print(f"Unique beliefs       : {len(uniq_beliefs):,}")
    print(f"Unique char instrs   : {len(uniq_chars):,}  "
          f"({int(non_ctrl_mask.sum()):,} non-control sessions)")

    # ── 3. Embed ──────────────────────────────────────────────────────────
    model = args.model
    bs    = args.batch_size
    nw    = args.workers

    emb_msgs_u    = embed_texts(uniq_msgs,    model=model, batch_size=bs, n_workers=nw, label="user_messages")
    emb_beliefs_u = embed_texts(uniq_beliefs, model=model, batch_size=bs, n_workers=nw, label="beliefs")

    # Expand to full session count
    emb_beliefs = emb_beliefs_u[np.array(idx_bels)]  # (n_sessions, D)

    if uniq_chars:
        emb_chars_u = embed_texts(uniq_chars, model=model, batch_size=bs, n_workers=nw, label="char_instrs")
        D = emb_chars_u.shape[1]
        emb_chars = np.zeros((len(sessions), D), dtype=np.float32)
        local_pos = 0
        for si, sess in enumerate(sessions):
            if not sess.is_control:
                emb_chars[si] = emb_chars_u[idx_chars_local[local_pos]]
                local_pos += 1
    else:
        emb_chars = None

    # Expand flat user-message embeddings back to (n_sessions, n_turns, D)
    emb_msgs_flat = emb_msgs_u[np.array(idx_msgs)]  # (total_turns, D)

    # ── 4. Compute per-turn similarities ─────────────────────────────────
    print("\nComputing per-turn cosine similarities ...")
    fieldnames = [
        "pair_id", "condition", "branch_turn", "turn",
        "iv1", "belief_category", "target_llm", "is_control",
        "sim_belief", "sim_char",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0

    # Reconstruct per-session turn ranges in flat_msgs
    # Build offset map: session_idx → list of flat indices
    sess_flat_idx: list[list[int]] = [[] for _ in sessions]
    for flat_i, (si, _) in enumerate(session_turn_idx):
        sess_flat_idx[si].append(flat_i)

    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for si, sess in enumerate(sessions):
            sess_flat_indices = sess_flat_idx[si]
            n_t = len(sess_flat_indices)

            # Per-turn user message embeddings for this session
            emb_turn_msgs = emb_msgs_flat[np.array(sess_flat_indices)]  # (n_t, D)

            # Belief embedding broadcast to (n_t, D)
            bel_rep = np.tile(emb_beliefs[si], (n_t, 1))
            sim_bel = row_cosine(emb_turn_msgs, bel_rep)

            # Char similarity: NaN for control
            if not sess.is_control and emb_chars is not None:
                char_rep = np.tile(emb_chars[si], (n_t, 1))
                sim_char = row_cosine(emb_turn_msgs, char_rep)
            else:
                sim_char = np.full(n_t, np.nan)

            for ti in range(n_t):
                writer.writerow({
                    "pair_id":         sess.pair_id,
                    "condition":       sess.condition,
                    "branch_turn":     sess.branch_turn,
                    "turn":            ti + 1,
                    "iv1":             sess.iv1,
                    "belief_category": sess.belief_category,
                    "target_llm":      sess.target_llm,
                    "is_control":      sess.is_control,
                    "sim_belief":      round(float(sim_bel[ti]), 6),
                    "sim_char": (
                        "" if np.isnan(sim_char[ti])
                        else round(float(sim_char[ti]), 6)
                    ),
                })
                n_rows += 1

    print(f"\nSaved {n_rows:,} rows → {args.output}")

    # ── 5. Quick summary ──────────────────────────────────────────────────
    import pandas as pd
    df = pd.read_csv(args.output)
    for cond in ["original", "noref"]:
        sub = df[df["condition"] == cond]
        sb = sub["sim_belief"].mean()
        sc = sub["sim_char"].dropna().mean()
        print(f"  {cond:8s}  mean sim_belief={sb:.4f}  mean sim_char={sc:.4f}")


if __name__ == "__main__":
    main()
