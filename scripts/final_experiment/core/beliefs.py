"""
core/beliefs.py
===============
Load the sampled belief pool from sample_beliefs.py output.

The on-disk format is the JSON list emitted by
`scripts/final_experiment/sample_beliefs.py`. Each record has these keys:

  category      : str      -- "bias" | "conspiracy" | "climate" | "fake_news" | "fake_health"
  subtype       : str      -- dataset-specific subcategory; may be ""
  content       : str      -- short claim text, or article headline for long-text
  is_long_text  : bool
  long_text     : str      -- present iff is_long_text; the article body

This module does not subset, sample, or stratify — that's already done
by sample_beliefs.py. It just reads the JSON, validates the schema, and
attaches a stable `belief_index` per category so session_ids are
reproducible across runs.
"""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_KEYS = {"category", "subtype", "content", "is_long_text"}


def load_beliefs(path: Path | str) -> list[dict]:
    """Load and validate the belief pool.

    Returns the records in the order they appear in the file, with two
    additional fields injected per record:

      belief_index_global : int  -- 0-indexed position in the file
      belief_index        : int  -- 0-indexed position within its category

    `belief_index` is what session_ids use, so two sessions for the
    same belief always have the same id regardless of how the file is
    sliced for partial runs.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a JSON list at top level")

    # Validate
    for i, r in enumerate(records):
        missing = REQUIRED_KEYS - set(r)
        if missing:
            raise ValueError(
                f"{path} record #{i}: missing required keys {sorted(missing)}"
            )
        if r["is_long_text"] and not r.get("long_text", "").strip():
            raise ValueError(
                f"{path} record #{i} has is_long_text=True but no long_text body"
            )

    # Attach indices
    per_cat_counter: dict[str, int] = {}
    for global_idx, r in enumerate(records):
        cat = r["category"]
        local_idx = per_cat_counter.get(cat, 0)
        r["belief_index_global"] = global_idx
        r["belief_index"] = local_idx
        per_cat_counter[cat] = local_idx + 1

    return records
