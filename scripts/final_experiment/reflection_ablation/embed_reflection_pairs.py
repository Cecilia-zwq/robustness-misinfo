"""
reflection_ablation/embed_reflection_pairs.py
=============================================
Extract reflection pairs from all conversations and compute semantic +
lexical similarity distances via the OpenRouter embedding API.

For each non-fallback break turn we extract:
  - last_failed : last rejected draft before the accepted one
  - passed      : the accepted draft
  - false_belief: the misinformation claim text
  - char_instr  : the character instruction prompt

Five pairwise cosine similarities are computed:
  1. failed  ↔ passed        (+ word-level lexical measures)
  2. failed  ↔ false belief
  3. failed  ↔ char instruction
  4. passed  ↔ false belief
  5. passed  ↔ char instruction

false_belief and char_instr texts are deduplicated before embedding to
avoid redundant API calls (beliefs repeat across many conversations).

Output: SOURCE_RUN_DIR/reflection_pair_distances.csv

Usage::

    cd scripts/final_experiment
    python -m reflection_ablation.embed_reflection_pairs [--workers 8] [--batch-size 64]

Resume: re-running with the same --output path is a no-op unless --rebuild
is passed.
"""

from __future__ import annotations

import argparse
import csv
import difflib
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
DEFAULT_OUTPUT_PATH = cfg.REFLECTION_PAIR_DISTANCES_PATH
DEFAULT_BATCH_SIZE  = 64
DEFAULT_WORKERS     = 8


# ════════════════════════════════════════════════════════════════════════════
# Pair extraction
# ════════════════════════════════════════════════════════════════════════════

class ReflectionPair(NamedTuple):
    file:         str
    turn:         int
    iv1:          str
    failed_msg:   str
    passed_msg:   str
    false_belief: str
    char_instr:   str


def _is_accepted(att: dict) -> bool:
    v = att.get("accepted", False)
    return v if isinstance(v, bool) else str(v).lower() == "true"


def extract_pairs(conv_dir: Path) -> list[ReflectionPair]:
    """Scan all conversation JSONs and extract (last_failed, passed) draft pairs.

    Skips turns where is_fallback=True (no draft passed) and turns with
    ≤1 reflection attempt (no break occurred).  For turns with multiple
    failed attempts, only the last rejected draft is paired with the
    accepted draft.
    """
    pairs: list[ReflectionPair] = []
    n_files = 0
    for path in sorted(conv_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        n_files += 1

        cell_meta    = data.get("cell")   or {}
        belief       = data.get("belief") or {}
        char_instr   = str(cell_meta.get("character_prompt") or "").strip()
        false_belief = str(belief.get("content") or "").strip()

        for turn in (data.get("turns") or []):
            attempts    = turn.get("reflection_attempts") or []
            is_fallback = str(turn.get("is_fallback", "false")).lower() == "true"
            if len(attempts) <= 1 or is_fallback:
                continue

            # Find the first accepted attempt and pair it with its immediate
            # predecessor, which must itself be a failed draft.  This guarantees
            # the pair captures the direct reflection fix step (i-1 → i), not
            # an arbitrary non-adjacent fail from earlier in the attempt list.
            accepted_idx = next(
                (j for j, a in enumerate(attempts) if _is_accepted(a)), None
            )
            if accepted_idx is None or accepted_idx == 0:
                continue   # no accepted draft, or accepted on first try (no prior fail)

            prev_att = attempts[accepted_idx - 1]
            prev_acc = prev_att.get("accepted", False)
            if isinstance(prev_acc, str):
                prev_acc = prev_acc.lower() == "true"
            if prev_acc:
                continue   # attempt i-1 is also accepted — unexpected structure, skip

            last_failed  = str(prev_att.get("draft") or "").strip()
            passed_draft = str(attempts[accepted_idx].get("draft") or "").strip()

            if not last_failed or not passed_draft:
                continue

            pairs.append(ReflectionPair(
                file=path.name,
                turn=int(turn.get("turn", 0)),
                iv1=str(cell_meta.get("iv1", "unknown")),
                failed_msg=last_failed,
                passed_msg=passed_draft,
                false_belief=false_belief,
                char_instr=char_instr,
            ))

    logger.info("Scanned %d files → %d reflection pairs.", n_files, len(pairs))
    return pairs


# ════════════════════════════════════════════════════════════════════════════
# Embedding
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
    """Embed a list of texts in parallel batches.

    Returns an (N, D) float32 array.  Raises RuntimeError if any batch
    fails (no partial results are returned).
    """
    if not texts:
        raise ValueError(f"No texts to embed for '{label}'.")

    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    logger.info(
        "Embedding %d %s → %d batch(es), %d worker(s), model=%s",
        len(texts), label, len(batches), n_workers, model,
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
                if (idx + 1) % max(1, len(batches) // 10) == 0 or idx == len(batches) - 1:
                    logger.info("  %s: %d/%d batches done", label, len(results), len(batches))
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
    seen:   dict[str, int] = {}
    unique: list[str]      = []
    indices: list[int]     = []
    for t in texts:
        if t not in seen:
            seen[t] = len(unique)
            unique.append(t)
        indices.append(seen[t])
    return unique, indices


# ════════════════════════════════════════════════════════════════════════════
# Similarity
# ════════════════════════════════════════════════════════════════════════════

def _normalize(A: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.where(norms > 0, norms, 1.0)


def row_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two (N, D) arrays."""
    return (_normalize(A) * _normalize(B)).sum(axis=1)


def jaccard_words(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def edit_ratio_words(a: str, b: str) -> float:
    """SequenceMatcher ratio on word-token sequences."""
    return difflib.SequenceMatcher(
        None, a.lower().split(), b.lower().split()
    ).ratio()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Embed reflection pairs and compute pairwise distances."
    )
    p.add_argument("--workers",    type=int,  default=DEFAULT_WORKERS,
                   help=f"Parallel embedding API workers (default {DEFAULT_WORKERS}).")
    p.add_argument("--batch-size", type=int,  default=DEFAULT_BATCH_SIZE,
                   help=f"Texts per embedding API call (default {DEFAULT_BATCH_SIZE}).")
    p.add_argument("--model",      default=DEFAULT_EMBED_MODEL,
                   help=f"LiteLLM model string for embeddings (default {DEFAULT_EMBED_MODEL!r}).")
    p.add_argument("--output",     type=Path, default=DEFAULT_OUTPUT_PATH,
                   help=f"Output CSV path (default {DEFAULT_OUTPUT_PATH}).")
    p.add_argument("--rebuild",    action="store_true",
                   help="Re-run even if the output CSV already exists.")
    args = p.parse_args()

    if args.output.exists() and not args.rebuild:
        print(f"Output already exists: {args.output}\nPass --rebuild to overwrite.")
        return

    # ── 1. Extract pairs ──────────────────────────────────────────────────
    print(f"\nSource conversations: {cfg.SOURCE_CONV_DIR}")
    pairs = extract_pairs(cfg.SOURCE_CONV_DIR)
    if not pairs:
        raise SystemExit("No reflection pairs found — check SOURCE_CONV_DIR.")
    n = len(pairs)
    print(f"Reflection pairs: {n:,}")

    iv1_counts: dict[str, int] = {}
    for pair in pairs:
        iv1_counts[pair.iv1] = iv1_counts.get(pair.iv1, 0) + 1
    for iv1, cnt in sorted(iv1_counts.items(), key=lambda x: -x[1]):
        print(f"  {iv1}: {cnt:,}")

    # ── 2. Deduplicate repeated texts ─────────────────────────────────────
    failed_texts = [p.failed_msg   for p in pairs]
    passed_texts = [p.passed_msg   for p in pairs]
    uniq_belief, idx_belief = _dedup([p.false_belief for p in pairs])

    # Control-group pairs have no character prompt → skip for char embedding.
    char_mask       = np.array([bool(p.char_instr) for p in pairs])
    non_ctrl_chars  = [p.char_instr for p in pairs if p.char_instr]
    n_ctrl          = int((~char_mask).sum())
    uniq_char, idx_char_local = _dedup(non_ctrl_chars) if non_ctrl_chars else ([], [])

    print(f"\nUnique text counts:")
    print(f"  failed_msg:   {n:,}")
    print(f"  passed_msg:   {n:,}")
    print(f"  false_belief: {len(uniq_belief):,}  (from {n:,} pairs)")
    print(f"  char_instr:   {len(uniq_char):,} unique  "
          f"({n - n_ctrl:,} non-control pairs; {n_ctrl:,} control → NaN)")

    # ── 3. Embed ──────────────────────────────────────────────────────────
    model = args.model
    bs    = args.batch_size
    nw    = args.workers

    emb_failed   = embed_texts(failed_texts, model=model, batch_size=bs, n_workers=nw, label="failed_msgs")
    emb_passed   = embed_texts(passed_texts, model=model, batch_size=bs, n_workers=nw, label="passed_msgs")
    emb_belief_u = embed_texts(uniq_belief,  model=model, batch_size=bs, n_workers=nw, label="beliefs")

    # Expand deduplicated belief embeddings back to per-pair length
    emb_belief = emb_belief_u[np.array(idx_belief)]

    # Embed char instructions only for non-control pairs; leave control rows as zeros
    # (they are masked to NaN before being written to CSV).
    if uniq_char:
        emb_char_u = embed_texts(uniq_char, model=model, batch_size=bs, n_workers=nw, label="char_instrs")
        D = emb_char_u.shape[1]
        emb_char = np.zeros((n, D), dtype=np.float32)
        local_pos = 0
        for i, p in enumerate(pairs):
            if p.char_instr:
                emb_char[i] = emb_char_u[idx_char_local[local_pos]]
                local_pos += 1
    else:
        emb_char = None

    # ── 4. Cosine similarities ─────────────────────────────────────────────
    print("\nComputing cosine similarities ...")
    cos_fail_pass   = row_cosine(emb_failed, emb_passed)
    cos_fail_belief = row_cosine(emb_failed, emb_belief)
    cos_pass_belief = row_cosine(emb_passed, emb_belief)

    # char similarities: NaN for control-group pairs (no character prompt)
    if emb_char is not None:
        cos_fail_char = np.where(char_mask, row_cosine(emb_failed, emb_char), np.nan)
        cos_pass_char = np.where(char_mask, row_cosine(emb_passed, emb_char), np.nan)
    else:
        cos_fail_char = np.full(n, np.nan)
        cos_pass_char = np.full(n, np.nan)

    # ── 5. Word-level lexical distances (failed ↔ passed only) ───────────
    print("Computing lexical distances ...")
    lex_jaccard    = [jaccard_words(p.failed_msg,    p.passed_msg) for p in pairs]
    lex_edit_ratio = [edit_ratio_words(p.failed_msg, p.passed_msg) for p in pairs]

    # ── 6. Write CSV ──────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file", "turn", "iv1",
        "failed_msg", "passed_msg", "false_belief", "char_instr",
        "cos_fail_pass", "cos_fail_belief", "cos_fail_char",
        "cos_pass_belief", "cos_pass_char",
        "lex_jaccard", "lex_edit_ratio",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i, pair in enumerate(pairs):
            writer.writerow({
                "file":            pair.file,
                "turn":            pair.turn,
                "iv1":             pair.iv1,
                "failed_msg":      pair.failed_msg,
                "passed_msg":      pair.passed_msg,
                "false_belief":    pair.false_belief,
                "char_instr":      pair.char_instr,
                "cos_fail_pass":   round(float(cos_fail_pass[i]),   6),
                "cos_fail_belief": round(float(cos_fail_belief[i]), 6),
                "cos_fail_char":   "" if np.isnan(cos_fail_char[i]) else round(float(cos_fail_char[i]), 6),
                "cos_pass_belief": round(float(cos_pass_belief[i]), 6),
                "cos_pass_char":   "" if np.isnan(cos_pass_char[i]) else round(float(cos_pass_char[i]), 6),
                "lex_jaccard":     round(float(lex_jaccard[i]),     6),
                "lex_edit_ratio":  round(float(lex_edit_ratio[i]),  6),
            })

    print(f"\nSaved {n:,} rows → {args.output}")

    # ── 7. Summary ────────────────────────────────────────────────────────
    print("\nMean cosine similarities (higher = more similar):")
    for name, arr in [
        ("cos_fail_pass",   cos_fail_pass),
        ("cos_fail_belief", cos_fail_belief),
        ("cos_fail_char",   cos_fail_char),
        ("cos_pass_belief", cos_pass_belief),
        ("cos_pass_char",   cos_pass_char),
    ]:
        valid = arr[~np.isnan(arr)]
        suffix = f"  (n={len(valid):,})" if len(valid) < n else ""
        print(f"  {name:22s}  μ={valid.mean():.3f}  σ={valid.std():.3f}{suffix}")


if __name__ == "__main__":
    main()
