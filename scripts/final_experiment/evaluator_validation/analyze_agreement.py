# %% [markdown]
# # Evaluator validation: three-evaluator agreement
#
# Loads paired score files for the primary evaluator (`gpt-4.1-mini`)
# and two secondary evaluators (`gpt-5.4-mini`, `gpt-5.4`) on the
# stratified validation subset, then computes:
#
# 1. **Per-dimension Krippendorff's α** across all three judges
#    (multi-rater agreement headline number).
# 2. **Pairwise quadratic-weighted Cohen's κ + Spearman ρ + signed
#    bias** for each of the three (judge × judge) pairs, on each pair's
#    intersection of co-scored turns.
# 3. **Score-distribution + bias diagnostics** by judge.
#
# Outputs land under `<RUN_DIR>/evaluator_validation/analysis_output/`.

# %% [markdown]
# # Package and configuration

# %%
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix


# %%
# ── Paths ───────────────────────────────────────────────────────────────────
RUN_DIR = Path(__file__).resolve().parents[3] / "results" / "final_experiment" / "main_user_IVs" / "20260427_165233"
CONV_DIR = RUN_DIR / "conversations"
SCORE_DIR = RUN_DIR / "scores"
VALIDATION_DIR = RUN_DIR / "evaluator_validation"
SAMPLE_MANIFEST_PATH = VALIDATION_DIR / "sample_manifest.json"

OUT_DIR = VALIDATION_DIR / "analysis_output"
TABLE_DIR = OUT_DIR / "tables"
PLOT_DIR = OUT_DIR / "plots"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# %%
# ── Rubric + evaluator identities ───────────────────────────────────────────
RUBRIC_NAME = "misinfo_v1_split"

# Order matters for display: the primary judge first, then secondaries.
# `slug` is what `core.storage.safe_slug` writes after `__` in score
# filenames; None means the file is unsuffixed (`<sid>__<rubric>.json`).
# Update slugs to match what your scoring runs actually wrote.
EVALUATORS: list[dict] = [
    {
        "key":   "gpt-4.1-mini",
        "label": "GPT-4.1-mini",
        "slug":  None,                                   # primary, unsuffixed
        "color": "#4F2683",
    },
    {
        "key":   "gpt-5.4-mini",
        "label": "GPT-5.4-mini",
        "slug":  "openrouter-openai-gpt-5.4-mini",
        "color": "#00c9ac",
    },
    {
        "key":   "gpt-5.4",
        "label": "GPT-5.4",
        "slug":  "openrouter-openai-gpt-5.4",
        "color": "#e07b54",
    },
]
EVAL_KEYS = [e["key"] for e in EVALUATORS]
EVAL_LABEL = {e["key"]: e["label"] for e in EVALUATORS}
EVAL_COLOR = {e["key"]: e["color"] for e in EVALUATORS}

# ── Score dimensions and 1–3 scale ──────────────────────────────────────────
SCORE_DIMS = ["correction", "rebuttal", "agreeableness"]
SCORE_RANGE = (1, 3)
SCORE_LABELS = list(range(SCORE_RANGE[0], SCORE_RANGE[1] + 1))

# ── Visual conventions — matches the headline analysis notebook ─────────────
model_palette = {
    "claude-sonnet-4.6":      "#CC785C",
    "gpt-5.3-chat":           "#10A37F",
    "gemini-3-flash-preview": "#4285F4",
    "deepseek-v3.2":          "#7C3AED",
}
model_labels = {
    "claude-sonnet-4.6":      "Claude Sonnet 4.6",
    "gpt-5.3-chat":           "GPT-5.3",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "deepseek-v3.2":          "DeepSeek V3.2",
}
IV1_ORDER = ["none", "warm", "emotional", "hostile", "logical"]
IV1_LABEL = {
    "none": "Control", "warm": "Warm", "emotional": "Emotional",
    "hostile": "Hostile", "logical": "Logical",
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "axes.edgecolor": "black",
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
})


# %% [markdown]
# # Load data
#
# For each sampled session, load every available evaluator's score
# file. Build a wide-format DataFrame: one row per (session, turn,
# dimension), one column per evaluator. NaN where a score is missing
# (file not yet produced) or marked as a parse failure (-1).
#
# Pairwise analyses below restrict to the per-pair intersection of
# valid scores. Multi-rater Krippendorff's α handles missingness
# natively — units rated by only 2 of 3 judges still contribute.

# %%
def _short_model(model_str: str) -> str:
    short = model_str.split("/")[-1]
    if short == "gemini-3-flash":
        return "gemini-3-flash-preview"
    return short


def _coerce_subtype(raw):
    if isinstance(raw, list):
        return ",".join(sorted(str(x) for x in raw))
    if isinstance(raw, str) and raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw.replace("'", '"'))
            if isinstance(parsed, list):
                return ",".join(sorted(parsed))
        except json.JSONDecodeError:
            pass
        return raw.strip("[]'\" ")
    return raw or ""


def _score_path(score_dir: Path, sid: str, rubric: str, slug: str | None) -> Path:
    suffix = f"{rubric}__{slug}" if slug else rubric
    return score_dir / f"{sid}__{suffix}.json"


def _load_scores(score_dir: Path, sid: str, rubric: str, slug: str | None) -> dict[int, dict]:
    """Return {turn: {dim: score}} for one (session, evaluator) pair, or {} if missing."""
    path = _score_path(score_dir, sid, rubric, slug)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {t["turn"]: t["scores"] for t in data["turns"]}


# %%
# ── Load sample manifest ────────────────────────────────────────────────────
with open(SAMPLE_MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)
sampled_ids = list(manifest["sampled_session_ids"])
print(f"Sample: {manifest.get('n_beliefs_sampled', '?')} beliefs of "
      f"{manifest.get('n_beliefs_population', '?')}, "
      f"{len(sampled_ids)} resolved sessions "
      f"(seed={manifest.get('seed')}, fraction={manifest.get('fraction')}).")


# %%
# ── Build wide-format paired table ──────────────────────────────────────────
# One row per (session, turn, dimension); one score column per evaluator.
# Rows where ALL three evaluators are NaN/-1 are skipped.
rows: list[dict] = []
n_skipped_no_conv = 0
parse_fails_per_eval = {k: 0 for k in EVAL_KEYS}

for sid in sampled_ids:
    conv_path = CONV_DIR / f"{sid}.json"
    if not conv_path.exists():
        n_skipped_no_conv += 1
        continue
    with open(conv_path, "r", encoding="utf-8") as f:
        conv = json.load(f)
    belief = conv.get("belief", {})
    cell = conv.get("cell", {})
    meta = {
        "session_id":   sid,
        "iv1":          cell.get("iv1", ""),
        "iv2":          cell.get("iv2", ""),
        "target_model": _short_model(conv.get("models", {}).get("target_llm", "")),
        "category":     belief.get("category", ""),
        "subtype":      _coerce_subtype(belief.get("subtype", "")),
        "is_long_text": bool(belief.get("is_long_text", False)),
    }

    eval_scores = {
        e["key"]: _load_scores(SCORE_DIR, sid, RUBRIC_NAME, e["slug"])
        for e in EVALUATORS
    }
    all_turns = sorted(set().union(*[set(s) for s in eval_scores.values()]))

    for turn in all_turns:
        for dim in SCORE_DIMS:
            row = {**meta, "turn": turn, "dimension": dim}
            any_valid = False
            for k in EVAL_KEYS:
                v = eval_scores[k].get(turn, {}).get(dim, None)
                if v is None:
                    row[f"score_{k}"] = np.nan
                elif v < 0:
                    row[f"score_{k}"] = np.nan
                    parse_fails_per_eval[k] += 1
                else:
                    row[f"score_{k}"] = float(v)
                    any_valid = True
            if any_valid:
                rows.append(row)

paired = pd.DataFrame(rows)
print(f"\nLoaded {len(paired)} rows × {len(EVAL_KEYS)} evaluator columns "
      f"from {paired['session_id'].nunique()} session(s).")
if n_skipped_no_conv:
    print(f"  ⚠ skipped {n_skipped_no_conv} session(s) missing conversation artifact.")
for k, n_fail in parse_fails_per_eval.items():
    if n_fail:
        print(f"  ⚠ {k}: {n_fail} parse-failure (-1) score(s) treated as missing.")

# Per-evaluator coverage: how many rows actually have a score from each judge?
print("\nPer-evaluator coverage (non-NaN rows):")
for k in EVAL_KEYS:
    n_valid = paired[f"score_{k}"].notna().sum()
    print(f"  {EVAL_LABEL[k]:<14}  {n_valid:>5} / {len(paired)}  "
          f"({n_valid/max(len(paired),1):.2%})")

paired.to_csv(TABLE_DIR / "per_turn_three_eval.csv", index=False)
display(paired.head(3))


# %% [markdown]
# # Multi-rater agreement: Krippendorff's α
#
# **Krippendorff's α** is the standard inter-rater reliability statistic
# for >2 raters. It generalises Cohen's κ, handles missing data
# natively (a turn rated by only 2 of 3 judges still contributes), and
# accepts an ordinal distance function appropriate for our 1–3 scale.
#
# We use ordinal α (Krippendorff 2011): the disagreement metric weights
# rating differences by their rank distance, equivalent in spirit to
# the quadratic weighting in $\kappa_q$. Conventional thresholds
# (Krippendorff 2004): $\alpha \geq 0.667$ marginal, $\geq 0.80$
# acceptable for tentative conclusions, $\geq 0.90$ acceptable for
# firm conclusions.
#
# This function is a from-scratch ordinal-α implementation following
# Hayes & Krippendorff's reference formula (no external krippendorff
# package dependency).

# %%
def krippendorff_alpha_ordinal(ratings: np.ndarray, value_domain: list[int]) -> float:
    """Krippendorff's α with ordinal disagreement metric.

    Parameters
    ----------
    ratings : np.ndarray, shape (n_units, n_raters)
        Ratings per unit per rater. NaN = missing.
    value_domain : list[int]
        All possible rating values (e.g. [1, 2, 3]).

    Returns
    -------
    float
        Krippendorff's α. NaN if no informative units.
    """
    ratings = np.asarray(ratings, dtype=float)

    # Only "informative" units contribute — those with >= 2 valid ratings.
    n_per_unit = (~np.isnan(ratings)).sum(axis=1)
    informative = n_per_unit >= 2
    if informative.sum() == 0:
        return float("nan")

    R = ratings[informative]
    n_per_unit = n_per_unit[informative]

    # Ordinal disagreement (Krippendorff 2011, Eq. 9):
    #   delta_ord(c, k) = ( sum_{g=c..k} n_g - (n_c + n_k)/2 )^2
    # where n_g is the marginal count of rating g across all valid
    # ratings on informative units.
    flat = R[~np.isnan(R)]
    marginals = {v: int(np.sum(flat == v)) for v in value_domain}
    n_total = int(flat.size)

    def delta_ord(c: int, k: int) -> float:
        if c == k:
            return 0.0
        lo, hi = (c, k) if c < k else (k, c)
        between = sum(marginals[g] for g in value_domain if lo <= g <= hi)
        return (between - (marginals[c] + marginals[k]) / 2.0) ** 2

    # Observed disagreement: sum over units of pairwise ordinal
    # disagreements, normalised by within-unit pairs.
    obs_num = 0.0
    obs_denom = 0.0
    for row, n_u in zip(R, n_per_unit):
        valid = row[~np.isnan(row)].astype(int)
        n_u = int(n_u)
        if n_u < 2:
            continue
        for i in range(n_u):
            for j in range(n_u):
                if i == j:
                    continue
                obs_num += delta_ord(int(valid[i]), int(valid[j]))
        obs_denom += n_u * (n_u - 1)

    # Expected disagreement: marginal-weighted ordinal disagreement.
    exp_num = 0.0
    for c in value_domain:
        for k in value_domain:
            if c == k:
                continue
            exp_num += marginals[c] * marginals[k] * delta_ord(c, k)
    exp_denom = n_total * (n_total - 1)

    if obs_denom == 0 or exp_denom == 0 or exp_num == 0:
        return float("nan")
    obs = obs_num / obs_denom
    exp = exp_num / exp_denom
    return 1.0 - obs / exp


def _alpha_table(df: pd.DataFrame) -> pd.DataFrame:
    """Krippendorff's α per dimension across all three judges."""
    rows = []
    eval_cols = [f"score_{k}" for k in EVAL_KEYS]
    for dim in SCORE_DIMS:
        d = df[df["dimension"] == dim]
        ratings = d[eval_cols].to_numpy()
        n_informative = (~np.isnan(ratings)).sum(axis=1) >= 2
        alpha = krippendorff_alpha_ordinal(ratings, SCORE_LABELS)
        rows.append({
            "dimension": dim,
            "n_units_informative": int(n_informative.sum()),
            "n_units_total": len(d),
            "alpha_ordinal": alpha,
        })
    return pd.DataFrame(rows)


alpha_overall = _alpha_table(paired)
alpha_overall.to_csv(TABLE_DIR / "alpha_overall.csv", index=False)
print("Krippendorff's α (ordinal, three-rater)")
display(alpha_overall.round(6))


# %% [markdown]
# # Pairwise agreement
#
# For each (judge A, judge B) pair, restrict to rows where both A and
# B have a valid score (the secondary judges may cover different
# session subsets). Compute κ_q, Spearman ρ, exact agreement,
# within-1, and signed bias A − B on the per-pair intersection.

# %%
def _pairwise_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    mask = (~np.isnan(a)) & (~np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 5:
        return {"n": n, "kappa_q": np.nan, "spearman_rho": np.nan,
                "exact": np.nan, "within_1": np.nan, "signed_diff": np.nan}
    a_int, b_int = a.astype(int), b.astype(int)
    try:
        kq = cohen_kappa_score(a_int, b_int, weights="quadratic", labels=SCORE_LABELS)
    except (ValueError, TypeError):
        kq = np.nan
    try:
        rho, _ = stats.spearmanr(a, b)
    except ValueError:
        rho = np.nan
    return {
        "n": n,
        "kappa_q": float(kq),
        "spearman_rho": float(rho),
        "exact": float(np.mean(a == b)),
        "within_1": float(np.mean(np.abs(a - b) <= 1)),
        "signed_diff": float(np.mean(a - b)),
    }


def _kappa_label(k: float) -> str:
    if np.isnan(k):
        return "—"
    if k < 0.21:
        return "slight"
    if k < 0.41:
        return "fair"
    if k < 0.61:
        return "moderate"
    if k < 0.81:
        return "substantial"
    return "almost perfect"


def _pairwise_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a_key, b_key in combinations(EVAL_KEYS, 2):
        for dim in SCORE_DIMS:
            d = df[df["dimension"] == dim]
            m = _pairwise_metrics(d[f"score_{a_key}"], d[f"score_{b_key}"])
            rows.append({
                "judge_a": EVAL_LABEL[a_key],
                "judge_b": EVAL_LABEL[b_key],
                "dimension": dim,
                **m,
                "kappa_label": _kappa_label(m["kappa_q"]),
            })
    return pd.DataFrame(rows)


pairwise = _pairwise_table(paired)
pairwise.to_csv(TABLE_DIR / "pairwise_overall.csv", index=False)
display(pairwise.round(3))


# %% [markdown]
# # Score distributions per evaluator
#
# How does each judge use the 1–3 scale? Side-by-side bars per
# dimension. Calibration differences (one judge skewing to "1" while
# others spread out) are far more interpretable here than in the κ
# table alone.

# %%
fig, axes = plt.subplots(1, len(SCORE_DIMS),
                         figsize=(5 * len(SCORE_DIMS), 4.0),
                         sharey=True)
if len(SCORE_DIMS) == 1:
    axes = [axes]

x = np.array(SCORE_LABELS)
bar_w = 0.78 / len(EVAL_KEYS)

for ax, dim in zip(axes, SCORE_DIMS):
    d = paired[paired["dimension"] == dim]
    for i, k in enumerate(EVAL_KEYS):
        col = d[f"score_{k}"].dropna()
        if col.empty:
            continue
        dist = col.value_counts(normalize=True).reindex(x, fill_value=0)
        offset = (i - (len(EVAL_KEYS) - 1) / 2) * bar_w
        ax.bar(x + offset, dist.values, bar_w,
               label=EVAL_LABEL[k], color=EVAL_COLOR[k], alpha=0.88)
    ax.set_xticks(x)
    ax.set_title(dim)
    ax.set_xlabel("Score (1–3)")
    if ax is axes[0]:
        ax.set_ylabel("Proportion")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("Score-value distribution by evaluator (turn-level)",
             fontsize=13, y=1.04)
plt.tight_layout()
plt.savefig(PLOT_DIR / "fig_score_distributions_three_eval.png")
plt.show()


# %% [markdown]
# # Bias by target model and pair
#
# Mean signed difference (judge A − judge B) per (target_model,
# dimension), one heatmap per pair. Constant offset uniform across
# target models = pure calibration difference (safe). Offset that
# varies with target model = self-preference signal worth flagging.

# %%
n_pairs = len(list(combinations(EVAL_KEYS, 2)))
fig, axes = plt.subplots(
    n_pairs, 1,
    figsize=(7, 0.6 * 4 * n_pairs + 1.5),
    constrained_layout=True,
)
if n_pairs == 1:
    axes = [axes]

for ax, (a_key, b_key) in zip(axes, combinations(EVAL_KEYS, 2)):
    bias = paired.copy()
    bias["signed_diff"] = bias[f"score_{a_key}"] - bias[f"score_{b_key}"]
    pivot = (
        bias.dropna(subset=["signed_diff"])
            .groupby(["target_model", "dimension"])["signed_diff"]
            .mean().unstack().reindex(columns=list(SCORE_DIMS))
    )
    if pivot.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        continue
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, ax=ax, linewidths=0.4,
        cbar_kws={"label": f"mean({EVAL_LABEL[a_key]} − {EVAL_LABEL[b_key]})"},
    )
    ax.set_title(f"{EVAL_LABEL[a_key]} − {EVAL_LABEL[b_key]}")
    ax.set_xlabel("")
    ax.set_ylabel("Target model")

plt.savefig(PLOT_DIR / "fig_bias_by_target_three_eval.png")
plt.show()


# %% [markdown]
# # Trajectory overlay — all three evaluators
#
# Mean score per turn for each judge, sample-averaged. The "do
# headline RQ1/RQ2 patterns survive evaluator substitution?" check.

# %%
fig, axes = plt.subplots(1, len(SCORE_DIMS),
                         figsize=(5 * len(SCORE_DIMS), 4.4),
                         sharey=True)
if len(SCORE_DIMS) == 1:
    axes = [axes]

for ax, dim in zip(axes, SCORE_DIMS):
    d = paired[paired["dimension"] == dim]
    if d.empty:
        ax.set_title(dim); continue
    for k in EVAL_KEYS:
        col = f"score_{k}"
        m = d.groupby("turn")[col].mean()
        sem = d.groupby("turn")[col].sem()
        ax.plot(m.index, m.values, "o-",
                label=EVAL_LABEL[k], color=EVAL_COLOR[k],
                linewidth=2, markersize=5)
        ax.fill_between(m.index, m - 1.96 * sem, m + 1.96 * sem,
                        alpha=0.13, color=EVAL_COLOR[k])
    ax.set_title(dim)
    ax.set_xlabel("Turn")
    if ax is axes[0]:
        ax.set_ylabel("Mean score (1–3)")
    ax.set_ylim(0.9, 3.1)
    ax.set_xticks(sorted(d["turn"].unique()))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    "Score trajectories by turn — three evaluators "
    "(sample average; shaded = 95% CI)",
    fontsize=13, y=1.04,
)
plt.tight_layout()
plt.savefig(PLOT_DIR / "fig_trajectory_three_eval.png")
plt.show()


# %% [markdown]
# # Constant-offset analysis: is the disagreement pure additive bias?
#
# Visually, the trajectory plots show three curves with the same shape
# but different vertical levels on correction and rebuttal. This cell
# tests that observation formally with three nested claims, each
# stronger than the last:
#
# 1. **Offset is non-zero.** One-sample Wilcoxon signed-rank on per-row
#    signed differences (primary − secondary), separately for each
#    pair × dimension. Reports the median offset and a 95% bootstrap
#    CI for transparency.
# 2. **Rank ordering preserved despite offset.** Already covered by
#    Spearman ρ in the pairwise table — restated here as a one-line
#    sanity check.
# 3. **Offset is approximately constant across substantive factors.**
#    Tests that the per-cell offsets (one per target model × dimension,
#    and one per IV1 persona × dimension) cluster tightly around a
#    single constant rather than depending on the factor. We use two
#    diagnostics: a Kruskal-Wallis test on the signed-difference
#    distributions across cells (large p ⇒ the offset is consistent),
#    and the SD of the per-cell offsets (small SD relative to the
#    headline offset ⇒ the additive-bias model fits well).
#
# Together these establish: "GPT-4.1-mini scores correction and
# rebuttal systematically higher than the secondaries, but this
# offset does not interact with any of the factors that drive
# RQ1/RQ2, so all relative comparisons in the paper are
# evaluator-invariant up to that constant."

# %%
def _bootstrap_median_ci(diffs: np.ndarray,
                         n_resamples: int = 2000,
                         seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the median of `diffs`. NaN if too few."""
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    medians = np.empty(n_resamples)
    n = len(diffs)
    for i in range(n_resamples):
        medians[i] = np.median(diffs[rng.integers(0, n, n)])
    return (float(np.percentile(medians, 2.5)),
            float(np.percentile(medians, 97.5)))


def _offset_significance(df: pd.DataFrame) -> pd.DataFrame:
    """One-sample Wilcoxon (H0: median offset = 0) per pair × dimension."""
    rows = []
    for a_key, b_key in combinations(EVAL_KEYS, 2):
        for dim in SCORE_DIMS:
            d = df[df["dimension"] == dim]
            diffs = (d[f"score_{a_key}"] - d[f"score_{b_key}"]).to_numpy()
            diffs = diffs[~np.isnan(diffs)]
            if len(diffs) < 10:
                rows.append({
                    "pair": f"{EVAL_LABEL[a_key]} − {EVAL_LABEL[b_key]}",
                    "dimension": dim, "n": len(diffs),
                    "mean_offset": np.nan, "median_offset": np.nan,
                    "ci95_lo": np.nan, "ci95_hi": np.nan,
                    "wilcoxon_p": np.nan,
                })
                continue
            mean_off = float(np.mean(diffs))
            median_off = float(np.median(diffs))
            ci_lo, ci_hi = _bootstrap_median_ci(diffs)
            try:
                # zero_method='wilcox' drops zero-difference pairs (the
                # standard convention); 'pratt' keeps them. We use
                # 'wilcox' so the p-value isn't artificially deflated
                # by the heavy mass of agreeing pairs.
                _, p = stats.wilcoxon(diffs, zero_method="wilcox")
            except ValueError:
                p = np.nan
            rows.append({
                "pair": f"{EVAL_LABEL[a_key]} − {EVAL_LABEL[b_key]}",
                "dimension": dim, "n": len(diffs),
                "mean_offset": mean_off, "median_offset": median_off,
                "ci95_lo": ci_lo, "ci95_hi": ci_hi,
                "wilcoxon_p": p,
            })
    return pd.DataFrame(rows)


offset_sig = _offset_significance(paired)
offset_sig.to_csv(TABLE_DIR / "offset_significance.csv", index=False)
print("Step 1 — Offset is non-zero?  (Wilcoxon H0: median offset = 0)")
display(offset_sig.round(4))


# %%
# ── Step 3: per-cell offsets — does the offset depend on target_model? ───
def _offset_by_factor(df: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Per-cell mean offset for each (pair × dimension × factor level).

    Plus diagnostics:
      * SD of the per-cell offsets across factor levels (small ⇒ constant
        offset; large ⇒ the factor modulates the offset).
      * Kruskal-Wallis p-value on the per-row signed-difference
        distributions across factor levels (large p ⇒ offset
        distribution is consistent; small p ⇒ at least one level
        differs).
    """
    rows = []
    for a_key, b_key in combinations(EVAL_KEYS, 2):
        for dim in SCORE_DIMS:
            d = df[df["dimension"] == dim].copy()
            d["diff"] = d[f"score_{a_key}"] - d[f"score_{b_key}"]
            d = d.dropna(subset=["diff", factor])
            if d.empty:
                continue
            per_level = d.groupby(factor)["diff"].mean()
            # Kruskal-Wallis across the per-row distributions.
            groups = [g["diff"].to_numpy() for _, g in d.groupby(factor)]
            groups = [g for g in groups if len(g) >= 5]
            if len(groups) < 2:
                kw_p = np.nan
            else:
                try:
                    _, kw_p = stats.kruskal(*groups)
                except ValueError:
                    kw_p = np.nan
            rows.append({
                "pair": f"{EVAL_LABEL[a_key]} − {EVAL_LABEL[b_key]}",
                "dimension": dim,
                "factor": factor,
                "headline_offset": float(d["diff"].mean()),
                "per_level_min": float(per_level.min()),
                "per_level_max": float(per_level.max()),
                "per_level_sd": float(per_level.std(ddof=1))
                                 if len(per_level) > 1 else np.nan,
                "kruskal_p": kw_p,
                "n_levels": int(len(per_level)),
                "constant_offset_ok": (
                    (per_level.std(ddof=1) < 0.10) if len(per_level) > 1
                    else True
                ),
            })
    return pd.DataFrame(rows)


offset_by_target = _offset_by_factor(paired, "target_model")
offset_by_iv1 = _offset_by_factor(paired, "iv1")
offset_by_turn = _offset_by_factor(paired, "turn")

offset_by_target.to_csv(TABLE_DIR / "offset_by_target_model.csv", index=False)
offset_by_iv1.to_csv(TABLE_DIR / "offset_by_iv1.csv", index=False)
offset_by_turn.to_csv(TABLE_DIR / "offset_by_turn.csv", index=False)

print("\nStep 3a — Offset constant across target models?")
display(offset_by_target.round(3))

print("\nStep 3b — Offset constant across IV1 personas?")
display(offset_by_iv1.round(3))

print("\nStep 3c — Offset constant across turns?")
display(offset_by_turn.round(3))


# %%
# ── Visual: per-cell offsets clustered around a single constant? ───────────
# One panel per dimension. X-axis = factor level; y-axis = mean offset
# for that cell. Reference line at the headline offset. If the points
# cluster tightly around the line, the additive-offset model fits.

def _plot_constant_offset(factor: str, factor_order=None):
    fig, axes = plt.subplots(
        1, len(SCORE_DIMS),
        figsize=(5 * len(SCORE_DIMS), 4.0),
        sharey=True,
    )
    if len(SCORE_DIMS) == 1:
        axes = [axes]

    for ax, dim in zip(axes, SCORE_DIMS):
        for a_key, b_key in combinations(EVAL_KEYS, 2):
            d = paired[paired["dimension"] == dim].copy()
            d["diff"] = d[f"score_{a_key}"] - d[f"score_{b_key}"]
            d = d.dropna(subset=["diff", factor])
            per_level = d.groupby(factor)["diff"].mean()
            if factor_order is not None:
                per_level = per_level.reindex(
                    [k for k in factor_order if k in per_level.index]
                )
            headline = float(d["diff"].mean())
            label = f"{EVAL_LABEL[a_key]} − {EVAL_LABEL[b_key]}"

            # Use the colour of the secondary judge (the one being
            # subtracted) so the legend reads naturally.
            col = EVAL_COLOR[b_key]
            xs = np.arange(len(per_level))
            ax.plot(xs, per_level.values, "o-",
                    label=label, color=col,
                    linewidth=1.6, markersize=6, alpha=0.85)
            ax.axhline(headline, color=col, linestyle=":",
                       linewidth=1, alpha=0.5)

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(dim)
        ax.set_xlabel(factor)
        if ax is axes[0]:
            ax.set_ylabel("Mean offset (judge A − judge B)")
        if factor_order is not None:
            ax.set_xticks(np.arange(len([k for k in factor_order
                                         if k in per_level.index])))
            ax.set_xticklabels(
                [IV1_LABEL.get(k, k)
                 for k in factor_order if k in per_level.index],
                rotation=20,
            )
        else:
            ax.set_xticks(np.arange(len(per_level)))
            ax.set_xticklabels(per_level.index, rotation=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        f"Per-{factor} mean offset, with headline offset shown as dotted line.\n"
        "Tight clustering around the dotted line ⇒ pure additive bias "
        "(no interaction).",
        fontsize=12, y=1.06,
    )
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"fig_constant_offset_by_{factor}.png")
    plt.show()


_plot_constant_offset("target_model")
_plot_constant_offset("iv1", factor_order=IV1_ORDER)
_plot_constant_offset("turn")


# %% [markdown]
# # What the constant-offset analysis tells you
#
# Read the three tables and the figures together:
#
# 1. **Wilcoxon p-values** in `offset_significance` confirm that the
#    correction and rebuttal offsets between GPT-4.1-mini and the two
#    secondary judges are statistically non-zero — i.e. the offset is
#    a real signal, not noise. Agreeableness offsets should be near
#    zero with non-significant or borderline p-values, matching what
#    we already saw in the bias table.
#
# 2. **Spearman ρ from the pairwise table** (above) confirms that
#    despite the offset, the rank ordering of scores is preserved
#    across judges. RQ1 / RQ2's relative comparisons depend only on
#    rank ordering, not on absolute level, so an additive offset
#    doesn't threaten them.
#
# 3. **Kruskal-Wallis p-values** in `offset_by_target_model`,
#    `offset_by_iv1`, and `offset_by_turn`, plus the SD columns, are
#    the strong claim. If `kruskal_p > 0.05` (and ideally if
#    `per_level_sd < 0.10`) for correction and rebuttal across all
#    three factors, then the offset between judges is a single
#    constant, not a function of the substantive variables. This
#    licenses the paper sentence: "the calibration difference between
#    judges is constant across target models, IV1 personas, and
#    conversational turns."
#
# Caveat: with this much data (n ≈ 4,800–14,400 paired rows), even
# trivially small interactions will produce significant Kruskal-Wallis
# p-values. The SD column is more informative than the p-value here.
# An SD of 0.05 across target models when the headline offset is 0.30
# means the offset varies ±0.05 around 0.30 — small enough to call
# "approximately constant" in the paper.


# %% [markdown]
# # Headline summary

# %%
print("=" * 82)
print("HEADLINE — three-evaluator validation")
print("=" * 82)
print(f"Sample        : {len(sampled_ids)} sessions, "
      f"{paired['session_id'].nunique()} actually loaded")
print(f"Rubric        : {RUBRIC_NAME}")
print(f"Evaluators    : {', '.join(EVAL_LABEL[k] for k in EVAL_KEYS)}")
print(f"Output dir    : {OUT_DIR}")
print()
print("Krippendorff's α (3-rater, ordinal):")
for _, row in alpha_overall.iterrows():
    print(f"  {row['dimension']:<14}  α={row['alpha_ordinal']:.3f}  "
          f"(n_informative={row['n_units_informative']})")
print()
print("Pairwise (κ_q | ρ | bias):")
for (a, b) in combinations(EVAL_KEYS, 2):
    print(f"  {EVAL_LABEL[a]} ↔ {EVAL_LABEL[b]}")
    sub = pairwise[(pairwise["judge_a"] == EVAL_LABEL[a])
                   & (pairwise["judge_b"] == EVAL_LABEL[b])]
    for _, row in sub.iterrows():
        sign = "+" if row["signed_diff"] >= 0 else ""
        print(f"    {row['dimension']:<14}  n={int(row['n']):>5}  "
              f"κ_q={row['kappa_q']:.3f} ({row['kappa_label']})  "
              f"ρ={row['spearman_rho']:.3f}  "
              f"Δ={sign}{row['signed_diff']:.3f}")

print()
print("Constant-offset diagnostic — per-cell offset SD:")
print("  (small SD relative to headline offset ⇒ additive-bias model fits)")
for factor_name, ofs_df in [
    ("target_model", offset_by_target),
    ("iv1",          offset_by_iv1),
    ("turn",         offset_by_turn),
]:
    print(f"  By {factor_name}:")
    for _, row in ofs_df.iterrows():
        print(f"    {row['pair']:<32} {row['dimension']:<14}  "
              f"headline={row['headline_offset']:+.3f}  "
              f"per-cell SD={row['per_level_sd']:.3f}  "
              f"range=[{row['per_level_min']:+.2f}, {row['per_level_max']:+.2f}]"
              f"  KW p={row['kruskal_p']:.3g}")