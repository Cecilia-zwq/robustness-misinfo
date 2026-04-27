"""Sample misinformation beliefs for the final experiment.

Output: data/dataset/sampled_beliefs.json — a JSON list of belief records, each
shaped to match the entry format consumed by the Iteration 5 experiment loader
(`scripts/Iteration5/experiment3.py::build_belief_and_claim_fields`):

    {
        "category":     "<dataset family>",       # e.g. "bias", "fake_news"
        "subtype":      "<value of the type col>", # may be "" if NaN
        "content":      "<short claim or headline>",
        "is_long_text": true | false,
        "long_text":    "<body>"  # only present when is_long_text is true
    }

Sampling policy:
  - ds_bias        : keep ALL rows (72)
  - ds_conspiracy  : keep ALL rows (59, including 3 with NaN type)
  - ds_fibvid      : 80 rows, stratified by `type` (largest-remainder allocation)
  - ds_climatefever: 80 rows, stratified by `type`
  - ds_fakenews    : 80 rows, stratified by `type`   (long-text)
  - ds_fakehealth  : 80 rows, stratified by `type`   (long-text)

Total: 72 + 59 + 80 + 80 + 80 + 80 = 451 beliefs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "dataset"
OUT_PATH = DATA_DIR / "sampled_beliefs.json"

SEED = 42
N_SAMPLE = 80


def largest_remainder_allocation(counts: pd.Series, n: int) -> dict:
    """Allocate `n` samples across categories proportional to `counts`.

    Uses the largest-remainder method. Categories may receive 0 if their
    population is small relative to others; this keeps every allocation
    quota <= the available population.
    """
    if n >= int(counts.sum()):
        return counts.astype(int).to_dict()

    total = counts.sum()
    exact = counts * n / total
    floor_alloc = np.floor(exact).astype(int)
    remainder = n - int(floor_alloc.sum())
    if remainder > 0:
        fractional = exact - floor_alloc
        top_up = fractional.nlargest(remainder).index
        floor_alloc[top_up] += 1
    return floor_alloc.to_dict()


def stratified_sample(df: pd.DataFrame, type_col: str, n: int, seed: int,
                      label: str) -> pd.DataFrame:
    """Stratified sample of `n` rows from `df`, balanced by `type_col`."""
    counts = df[type_col].value_counts()
    allocation = largest_remainder_allocation(counts, n)

    print(f"\n[{label}] stratified sample: {n} from {len(df)} rows "
          f"({df[type_col].nunique()} subtypes)")
    for cat, k in sorted(allocation.items(), key=lambda x: -x[1]):
        print(f"    {cat:<35s}  pop={counts[cat]:>4d}  sample={k}")

    frames = []
    for cat, k in allocation.items():
        if k > 0:
            frames.append(df[df[type_col] == cat].sample(n=k, random_state=seed))
    return pd.concat(frames, ignore_index=True)


def _safe_str(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def build_short_claim_records(df: pd.DataFrame, category: str,
                              type_col: str = "type") -> list[dict]:
    """Convert short-claim rows to belief records."""
    return [
        {
            "category":     category,
            "subtype":      _safe_str(row[type_col]),
            "content":      _safe_str(row["content"]),
            "is_long_text": False,
        }
        for _, row in df.iterrows()
    ]


def build_long_text_records(df: pd.DataFrame, category: str) -> list[dict]:
    """Convert long-text rows (schema: title / content / type) to belief records.

    Mirrors Iteration 5 `load_fakenews`: the dict's `content` field carries the
    headline (used as display label) and `long_text` carries the body.
    """
    return [
        {
            "category":     category,
            "subtype":      _safe_str(row["type"]),
            "content":      _safe_str(row["title"]),
            "is_long_text": True,
            "long_text":    _safe_str(row["content"]),
        }
        for _, row in df.iterrows()
    ]


def main() -> None:
    print(f"Reading from : {DATA_DIR}")
    print(f"Seed         : {SEED}")
    print(f"N per dataset (sampled): {N_SAMPLE}")

    ds_bias = pd.read_csv(DATA_DIR / "ds_bias.csv")
    ds_conspiracy = pd.read_csv(DATA_DIR / "ds_conspiracy.csv")
    ds_fibvid = pd.read_csv(DATA_DIR / "ds_fibvid.csv")
    ds_climate = pd.read_csv(DATA_DIR / "ds_climatefever.csv")
    ds_fakenews = pd.read_csv(DATA_DIR / "ds_fakenews.csv")
    ds_fakehealth = pd.read_csv(DATA_DIR / "ds_fakehealth.csv")

    # Long-text datasets: drop rows with no usable body or headline. The
    # Iteration 5 loader (`load_fakenews`) does the same filter, so any rows
    # we sample here that have a blank body would be silently dropped at
    # experiment time. Filter once, up front, so the final 80-per-dataset
    # count is honest.
    for name, df in [("ds_fakenews", ds_fakenews), ("ds_fakehealth", ds_fakehealth)]:
        n_before = len(df)
        df.dropna(subset=["title", "content"], inplace=True)
        df.drop(df[df["content"].astype(str).str.strip() == ""].index, inplace=True)
        df.drop(df[df["title"].astype(str).str.strip() == ""].index, inplace=True)
        df.reset_index(drop=True, inplace=True)
        if len(df) != n_before:
            print(f"[{name}] dropped {n_before - len(df)} rows with empty title/body "
                  f"(source has {n_before}, usable for sampling = {len(df)})")

    fibvid_sample = stratified_sample(ds_fibvid, "type", N_SAMPLE, SEED, "fibvid")
    climate_sample = stratified_sample(ds_climate, "type", N_SAMPLE, SEED, "climatefever")
    fakenews_sample = stratified_sample(ds_fakenews, "type", N_SAMPLE, SEED, "fakenews")
    fakehealth_sample = stratified_sample(ds_fakehealth, "type", N_SAMPLE, SEED, "fakehealth")

    records: list[dict] = []
    records += build_short_claim_records(ds_bias, "bias", type_col="bias_type")
    records += build_short_claim_records(ds_conspiracy, "conspiracy")
    records += build_short_claim_records(fibvid_sample, "fibvid")
    records += build_short_claim_records(climate_sample, "climate")
    records += build_long_text_records(fakenews_sample, "fake_news")
    records += build_long_text_records(fakehealth_sample, "fake_health")

    print("\nFinal record counts by category:")
    cat_counts: dict[str, int] = {}
    for r in records:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1
    for cat, c in cat_counts.items():
        print(f"    {cat:<14s}  {c}")
    print(f"    {'TOTAL':<14s}  {len(records)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(records)} belief records -> {OUT_PATH}")


if __name__ == "__main__":
    main()
