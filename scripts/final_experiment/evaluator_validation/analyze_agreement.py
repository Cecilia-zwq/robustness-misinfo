# %% [markdown]
# # Evaluator validation: inter-evaluator agreement
#
# Reads the sample manifest written by `run_validation_scoring.py`,
# loads the paired score files (primary + secondary evaluator) for every
# sampled session, and computes per-dimension and per-subgroup
# agreement metrics with corresponding plots.
#
# Outputs land under `<RUN_DIR>/evaluator_validation/analysis_output/`.

# %% [markdown]
# # Package and configuration

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix


# %%
# ── Paths ───────────────────────────────────────────────────────────────────
RUN_DIR = Path(
    "/home/wzhan969/robustness-misinfo/results/final_experiment/"
    "main_user_IVs/20260427_165233"
)
CONV_DIR = RUN_DIR / "conversations"
SCORE_DIR = RUN_DIR / "scores"
VALIDATION_DIR = RUN_DIR / "evaluator_validation"
SAMPLE_MANIFEST_PATH = VALIDATION_DIR / "sample_manifest.json"

OUT_DIR = VALIDATION_DIR / "analysis_output"
TABLE_DIR = OUT_DIR / "tables"
PLOT_DIR = OUT_DIR / "plots"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Rubric + evaluator identity ─────────────────────────────────────────────
# Primary scores live in <SCORE_DIR>/<sid>__<RUBRIC_NAME>.json. Set
# PRIMARY_SLUG only if your primary scoring pass used --suffix-evaluator
# (in that case primary files are <sid>__<RUBRIC_NAME>__<slug>.json).
RUBRIC_NAME = "misinfo_v1_split"
PRIMARY_SLUG: str | None = None

# Secondary slug — must match what run_validation_scoring.py wrote.
# Built by core.storage.safe_slug from "openrouter/google/gemini-3.1-flash-lite-preview".
SECONDARY_SLUG = "openrouter-google-gemini-3.1-flash-lite-preview"

# ── Score dimensions and 1–3 scale ──────────────────────────────────────────
SCORE_DIMS = ["correction", "rebuttal", "agreeableness"]
SCORE_RANGE = (1, 3)
SCORE_LABELS = list(range(SCORE_RANGE[0], SCORE_RANGE[1] + 1))

metrics = [
    ("correction", "Correction Quality"),
    ("rebuttal", "Rebuttal Quality"),
    ("agreeableness", "Agreeableness"),
]

# ── Visual conventions — matches notebooks/final_experiment/final_experiment_analysis.py ─
# Same palettes / labels / plt.rcParams so figures from this notebook and
# the headline analysis notebook can sit side-by-side in the paper without
# restyling.

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
IV1_PALETTE = {
    "Control": "#7f7f7f", "Warm": "#e07b54", "Emotional": "#00c9ac",
    "Hostile": "#4F2683", "Logical": "#4a90d9",
}

# Evaluator colour pair — used in the trajectory overlay and the
# distribution plot. Keep these consistent across figures.
PRIMARY_COLOUR = "#4F2683"
SECONDARY_COLOUR = "#00c9ac"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


# %% [markdown]
# # Load data
#
# For each sampled session, pair the primary and secondary score files
# and pull subgroup metadata (target model, IV1 persona, belief
# category) from the conversation artifact. Drop (turn × dimension)
# pairs where either evaluator emitted a parse-failure score (-1).

# %%
def _short_model(model_str: str) -> str:
    """Match the convention in final_experiment_analysis.py."""
    short = model_str.split("/")[-1]
    if short == "gemini-3-flash":
        return "gemini-3-flash-preview"
    return short


def _coerce_subtype(raw):
    """Flatten subtype to a stable string (handles list, '['x']', etc.)."""
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
    """Score files: <sid>__<rubric>.json or <sid>__<rubric>__<slug>.json."""
    suffix = f"{rubric}__{slug}" if slug else rubric
    return score_dir / f"{sid}__{suffix}.json"


# %%
# ── Load sample manifest ────────────────────────────────────────────────────
with open(SAMPLE_MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)

sampled_ids = list(manifest["sampled_session_ids"])
print(
    f"Sample manifest:\n"
    f"  beliefs sampled : {manifest.get('n_beliefs_sampled', '?')} of "
    f"{manifest.get('n_beliefs_population', '?')} "
    f"(fraction={manifest.get('fraction')}, seed={manifest.get('seed')})\n"
    f"  resolved sessions : {len(sampled_ids)}"
)


# %%
# ── Load paired scores ──────────────────────────────────────────────────────
turn_rows: list[dict] = []
n_skipped_missing = 0
n_dropped_parse_fail = 0

for sid in sampled_ids:
    primary_path = _score_path(SCORE_DIR, sid, RUBRIC_NAME, PRIMARY_SLUG)
    secondary_path = _score_path(SCORE_DIR, sid, RUBRIC_NAME, SECONDARY_SLUG)
    conv_path = CONV_DIR / f"{sid}.json"

    if not (primary_path.exists() and secondary_path.exists() and conv_path.exists()):
        n_skipped_missing += 1
        continue

    with open(primary_path, "r", encoding="utf-8") as f:
        primary = json.load(f)
    with open(secondary_path, "r", encoding="utf-8") as f:
        secondary = json.load(f)
    with open(conv_path, "r", encoding="utf-8") as f:
        conv = json.load(f)

    belief = conv.get("belief", {})
    cell = conv.get("cell", {})
    meta = {
        "session_id":   sid,
        "cell_id":      cell.get("cell_id", ""),
        "iv1":          cell.get("iv1", ""),
        "iv2":          cell.get("iv2", ""),
        "target_model": _short_model(conv.get("models", {}).get("target_llm", "")),
        "category":     belief.get("category", ""),
        "subtype":      _coerce_subtype(belief.get("subtype", "")),
        "is_long_text": bool(belief.get("is_long_text", False)),
    }

    primary_by_turn = {t["turn"]: t["scores"] for t in primary["turns"]}
    secondary_by_turn = {t["turn"]: t["scores"] for t in secondary["turns"]}

    for turn in sorted(set(primary_by_turn) & set(secondary_by_turn)):
        ps = primary_by_turn[turn]
        ss = secondary_by_turn[turn]
        for dim in SCORE_DIMS:
            p_val = ps.get(dim, -1.0)
            s_val = ss.get(dim, -1.0)
            if p_val < 0 or s_val < 0:
                n_dropped_parse_fail += 1
                continue
            turn_rows.append({
                **meta,
                "turn": turn,
                "dimension": dim,
                "score_primary": float(p_val),
                "score_secondary": float(s_val),
            })

paired = pd.DataFrame(turn_rows)
print(f"\nLoaded {len(paired)} paired (session × turn × dimension) rows "
      f"from {paired['session_id'].nunique()} session(s).")
if n_skipped_missing:
    print(f"  ⚠ skipped {n_skipped_missing} session(s) with missing files.")
if n_dropped_parse_fail:
    print(f"  ⚠ dropped {n_dropped_parse_fail} (turn, dim) row(s) with -1 score.")

paired.to_csv(TABLE_DIR / "per_turn_diff.csv", index=False)
display(paired.head(3))


# %% [markdown]
# # Agreement metrics
#
# **Quadratic-weighted Cohen's κ** is the headline metric — it's what
# reviewers expect for ordinal LLM-as-judge validation. We also report
# Spearman ρ (rank order), Pearson r, exact-agreement proportion,
# within-1 proportion, mean signed difference (primary − secondary, the
# bias check), and mean absolute difference.
#
# Landis & Koch (1977) verbal labels: κ < 0.21 slight, 0.21–0.40 fair,
# 0.41–0.60 moderate, 0.61–0.80 substantial, > 0.80 almost perfect.

# %%
def _agreement_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """All agreement stats for one column of paired ratings."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = (~np.isnan(a)) & (~np.isnan(b))
    a = a[mask]
    b = b[mask]
    n = int(len(a))
    if n < 5:
        return {
            "n": n, "kappa_quadratic": np.nan,
            "spearman_rho": np.nan, "spearman_p": np.nan,
            "pearson_r": np.nan, "pearson_p": np.nan,
            "exact_agreement": np.nan, "within_1": np.nan,
            "mean_signed_diff": np.nan, "mean_abs_diff": np.nan,
        }

    a_int, b_int = a.astype(int), b.astype(int)

    try:
        kappa = cohen_kappa_score(
            a_int, b_int, weights="quadratic", labels=SCORE_LABELS,
        )
    except (ValueError, TypeError):
        kappa = np.nan

    # SciPy raises on constant input — trap and return NaN.
    try:
        rho, rho_p = stats.spearmanr(a, b)
    except ValueError:
        rho, rho_p = np.nan, np.nan
    try:
        r, r_p = stats.pearsonr(a, b)
    except ValueError:
        r, r_p = np.nan, np.nan

    return {
        "n": n,
        "kappa_quadratic": float(kappa),
        "spearman_rho": float(rho), "spearman_p": float(rho_p),
        "pearson_r": float(r), "pearson_p": float(r_p),
        "exact_agreement": float(np.mean(a == b)),
        "within_1": float(np.mean(np.abs(a - b) <= 1)),
        "mean_signed_diff": float(np.mean(a - b)),  # primary − secondary
        "mean_abs_diff": float(np.mean(np.abs(a - b))),
    }


def _kappa_label(k: float) -> str:
    """Landis & Koch (1977) verbal interpretation."""
    if np.isnan(k):
        return "—"
    if k < 0:
        return "poor"
    if k < 0.21:
        return "slight"
    if k < 0.41:
        return "fair"
    if k < 0.61:
        return "moderate"
    if k < 0.81:
        return "substantial"
    return "almost perfect"


def _agreement_table(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Apply _agreement_metrics per (group_cols, dimension) cell."""
    rows: list[dict] = []
    if group_cols:
        for keys, sub in df.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            for dim in SCORE_DIMS:
                d = sub[sub["dimension"] == dim]
                m = _agreement_metrics(d["score_primary"], d["score_secondary"])
                row = dict(zip(group_cols, keys))
                row["dimension"] = dim
                row.update(m)
                row["kappa_label"] = _kappa_label(m["kappa_quadratic"])
                rows.append(row)
    else:
        for dim in SCORE_DIMS:
            d = df[df["dimension"] == dim]
            m = _agreement_metrics(d["score_primary"], d["score_secondary"])
            rows.append({
                "dimension": dim, **m,
                "kappa_label": _kappa_label(m["kappa_quadratic"]),
            })
    return pd.DataFrame(rows)


# %% [markdown]
# ## Overall agreement
#
# Pooled across all sessions. This is the headline number — what to
# quote in the abstract / methods section.

# %%
overall = _agreement_table(paired, group_cols=None)
overall.to_csv(TABLE_DIR / "agreement_overall.csv", index=False)
display(overall.round(3))


# %% [markdown]
# ## Agreement by target model
#
# Breakdown by target model is the **self-preference check**: if
# `gpt-4.1-mini` (primary evaluator) scores `gpt-5.3-chat` outputs more
# leniently than Gemini does, that's a confound for RQ1's "GPT-5.3 is
# most stable" claim. A model-correlated offset on `mean_signed_diff`
# is the signal to look for.

# %%
by_target = _agreement_table(paired, ["target_model"])
by_target.to_csv(TABLE_DIR / "agreement_by_target_model.csv", index=False)
display(by_target.round(3))


# %% [markdown]
# ## Agreement by IV1 persona
#
# Breakdown by adversarial user style — the "hard-cases check". If
# agreement is uniform across personas, the framework's conclusions
# about persona effects (RQ2) are evaluator-robust. If agreement
# craters specifically on the emotional persona, that's a flag worth
# reporting in the limitations section.

# %%
by_iv1 = _agreement_table(paired, ["iv1"])
# Reorder rows to match the canonical IV1 ordering.
by_iv1["iv1"] = pd.Categorical(by_iv1["iv1"], categories=IV1_ORDER, ordered=True)
by_iv1 = by_iv1.sort_values(["iv1", "dimension"]).reset_index(drop=True)
by_iv1.to_csv(TABLE_DIR / "agreement_by_iv1.csv", index=False)
display(by_iv1.round(3))


# %% [markdown]
# ## Agreement by belief category
#
# Are some categories (e.g. long-text fake_news vs short-claim bias)
# harder for the evaluators to agree on than others?

# %%
by_category = _agreement_table(paired, ["category"])
by_category.to_csv(TABLE_DIR / "agreement_by_category.csv", index=False)
display(by_category.round(3))


# %% [markdown]
# ## Confusion matrices (long-format CSV)
#
# Saved separately so downstream tools can pivot however they need.
# The plotted version is below.

# %%
def _confusion_matrix_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dim in SCORE_DIMS:
        d = df[df["dimension"] == dim]
        if d.empty:
            continue
        cm = confusion_matrix(
            d["score_primary"].astype(int),
            d["score_secondary"].astype(int),
            labels=SCORE_LABELS,
        )
        for i, p_label in enumerate(SCORE_LABELS):
            for j, s_label in enumerate(SCORE_LABELS):
                rows.append({
                    "dimension": dim,
                    "primary_score": p_label,
                    "secondary_score": s_label,
                    "count": int(cm[i, j]),
                })
    return pd.DataFrame(rows)


cm_long = _confusion_matrix_long(paired)
cm_long.to_csv(TABLE_DIR / "confusion_matrices.csv", index=False)
display(cm_long.head(9))


# %% [markdown]
# # Plots

# %% [markdown]
# ## Confusion matrices (one per dimension)
#
# Counts annotated; colour is row-normalised so the diagonal isn't
# drowned out by class imbalance (which is heavy — most agreeableness
# scores are 1, etc.).

# %%
fig, axes = plt.subplots(1, len(SCORE_DIMS), figsize=(5 * len(SCORE_DIMS), 4.4))
if len(SCORE_DIMS) == 1:
    axes = [axes]
for ax, dim in zip(axes, SCORE_DIMS):
    d = paired[paired["dimension"] == dim]
    if d.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(dim); continue
    cm = confusion_matrix(
        d["score_primary"].astype(int),
        d["score_secondary"].astype(int),
        labels=SCORE_LABELS,
    )
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sums
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="Blues",
        xticklabels=SCORE_LABELS, yticklabels=SCORE_LABELS,
        cbar_kws={"label": "row-normalised proportion"},
        vmin=0, vmax=1, ax=ax, linewidths=0.4,
    )
    m = _agreement_metrics(d["score_primary"], d["score_secondary"])
    ax.set_title(
        f"{dim}\nκ_q = {m['kappa_quadratic']:.3f} "
        f"({_kappa_label(m['kappa_quadratic'])}), n={m['n']}"
    )
    ax.set_xlabel("Secondary score")
    ax.set_ylabel("Primary score" if ax is axes[0] else "")

fig.suptitle(
    "Inter-evaluator confusion matrices  "
    "(counts annotated; colour = row-normalised proportion)",
    fontsize=13, y=1.04,
)
plt.tight_layout()
plt.savefig(PLOT_DIR / "fig_confusion_matrices.png")
plt.show()


# %% [markdown]
# ## Agreement (κ) by subgroup
#
# Three panels: target_model / iv1 / category. Reference lines at 0.6
# (substantial) and 0.8 (almost perfect) make subgroup degradation
# easy to spot.

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)

panels = [
    (by_target, "target_model", "By target model", None),
    (by_iv1, "iv1", "By IV1 persona", IV1_ORDER),
    (by_category, "category", "By belief category", None),
]

for ax, (df_grp, group_col, title, order) in zip(axes, panels):
    if df_grp.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title); continue
    pivot = df_grp.pivot_table(
        index=group_col, columns="dimension", values="kappa_quadratic",
    )
    # Preserve canonical dimension column order.
    pivot = pivot.reindex(columns=list(SCORE_DIMS))
    if order is not None:
        pivot = pivot.reindex([k for k in order if k in pivot.index])
    pivot.plot(kind="bar", ax=ax, width=0.78, edgecolor="white")
    ax.set_title(title)
    ax.set_ylabel("Quadratic-weighted κ" if ax is axes[0] else "")
    ax.set_ylim(0, 1)
    ax.axhline(0.6, color="grey", linestyle=":", linewidth=1,
               label="substantial (κ ≥ 0.6)")
    ax.axhline(0.8, color="black", linestyle=":", linewidth=1,
               label="almost perfect (κ ≥ 0.8)")
    ax.tick_params(axis="x", rotation=25)
    ax.set_xlabel("")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("Inter-evaluator agreement (κ) across subgroups",
             fontsize=13, y=1.04)
plt.tight_layout()
plt.savefig(PLOT_DIR / "fig_kappa_by_subgroup.png")
plt.show()


# %% [markdown]
# ## Score-value distributions
#
# Side-by-side bars of the score distribution for each evaluator. A
# divergent shape (e.g. secondary skews to 1 while primary is mostly
# 2) is a more interpretable diagnostic than κ alone for spotting
# calibration differences.

# %%
fig, axes = plt.subplots(1, len(SCORE_DIMS),
                         figsize=(5 * len(SCORE_DIMS), 4.0),
                         sharey=True)
if len(SCORE_DIMS) == 1:
    axes = [axes]
for ax, dim in zip(axes, SCORE_DIMS):
    d = paired[paired["dimension"] == dim]
    if d.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(dim); continue
    x = np.array(SCORE_LABELS)
    primary_dist = d["score_primary"].value_counts(normalize=True).reindex(x, fill_value=0)
    secondary_dist = d["score_secondary"].value_counts(normalize=True).reindex(x, fill_value=0)
    w = 0.38
    ax.bar(x - w / 2, primary_dist.values, w,
           label="Primary", color=PRIMARY_COLOUR, alpha=0.88)
    ax.bar(x + w / 2, secondary_dist.values, w,
           label="Secondary", color=SECONDARY_COLOUR, alpha=0.88)
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
plt.savefig(PLOT_DIR / "fig_score_distributions.png")
plt.show()


# %% [markdown]
# ## Per-target-model bias
#
# Mean signed difference (primary − secondary) for each target × dim
# combination. Non-zero ⇒ systematic offset. The interesting case is
# when the offset correlates with the target model — that's a
# self-preference signal (e.g. gpt-4.1-mini scoring gpt-5.3-chat
# outputs more leniently than gemini does), worth flagging in the
# limitations section.

# %%
bias_df = paired.copy()
bias_df["signed_diff"] = bias_df["score_primary"] - bias_df["score_secondary"]
bias_pivot = (
    bias_df.groupby(["target_model", "dimension"])["signed_diff"]
           .mean()
           .unstack()
           .reindex(columns=list(SCORE_DIMS))
)
display(bias_pivot.round(3))

if not bias_pivot.empty:
    fig, ax = plt.subplots(figsize=(6, 0.6 * max(len(bias_pivot), 1) + 1.8))
    sns.heatmap(
        bias_pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, ax=ax, linewidths=0.4,
        cbar_kws={"label": "mean(primary − secondary)"},
    )
    ax.set_title(
        "Per-target-model evaluator bias\n"
        "(non-zero ⇒ systematic offset; "
        "model-correlated ⇒ self-preference signal)"
    )
    ax.set_xlabel("")
    ax.set_ylabel("Target model")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "fig_bias_by_target_model.png")
    plt.show()


# %% [markdown]
# ## Trajectory overlay — primary vs secondary
#
# Mean score per turn for both evaluators, on the validation sample.
# **This is the "do the headline RQ1/RQ2 patterns survive evaluator
# substitution?" check.** If the two trajectories track each other,
# the framework's qualitative claims are evaluator-robust on this
# sample.

# %%
fig, axes = plt.subplots(1, len(SCORE_DIMS),
                         figsize=(5 * len(SCORE_DIMS), 4.4),
                         sharey=True)
if len(SCORE_DIMS) == 1:
    axes = [axes]
for ax, dim in zip(axes, SCORE_DIMS):
    d = paired[paired["dimension"] == dim]
    if d.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(dim); continue
    for col, label, colour in [
        ("score_primary", "Primary", PRIMARY_COLOUR),
        ("score_secondary", "Secondary", SECONDARY_COLOUR),
    ]:
        m = d.groupby("turn")[col].mean()
        sem = d.groupby("turn")[col].sem()
        ax.plot(m.index, m.values, "o-",
                label=label, color=colour, linewidth=2, markersize=5)
        ax.fill_between(m.index, m - 1.96 * sem, m + 1.96 * sem,
                        alpha=0.15, color=colour)
    ax.set_title(dim)
    ax.set_xlabel("Turn")
    if ax is axes[0]:
        ax.set_ylabel("Mean score (1–3)")
    ax.set_ylim(0.9, 3.1)
    ax.set_xticks(sorted(d["turn"].unique()))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
fig.suptitle(
    "Score trajectories by turn — primary vs secondary evaluator "
    "(sample average; shaded = 95% CI)",
    fontsize=13, y=1.04,
)
plt.tight_layout()
plt.savefig(PLOT_DIR / "fig_trajectory_evaluator_overlay.png")
plt.show()


# %% [markdown]
# ## Trajectory overlay by target model
#
# Same as above, broken out per target model. This is the strongest
# version of the evaluator-robustness check: if Claude's
# correction/rebuttal collapse around turn 4 shows up under both
# evaluators, that finding is robust; if only the primary sees it,
# the limitation has to be flagged.

# %%
models_present = sorted(paired["target_model"].unique())
n_rows = len(models_present)
n_cols = len(SCORE_DIMS)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(4.6 * n_cols, 3.0 * n_rows),
    sharex=True, sharey="col",
)
if n_rows == 1:
    axes = np.array([axes])
if n_cols == 1:
    axes = axes[:, None]

for r, m_name in enumerate(models_present):
    for c, dim in enumerate(SCORE_DIMS):
        ax = axes[r, c]
        d = paired[(paired["target_model"] == m_name)
                   & (paired["dimension"] == dim)]
        if d.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes); continue
        for col, label, colour in [
            ("score_primary", "Primary", PRIMARY_COLOUR),
            ("score_secondary", "Secondary", SECONDARY_COLOUR),
        ]:
            mean = d.groupby("turn")[col].mean()
            sem = d.groupby("turn")[col].sem()
            ax.plot(mean.index, mean.values, "o-",
                    label=label, color=colour, linewidth=1.7, markersize=4)
            ax.fill_between(mean.index, mean - 1.96 * sem, mean + 1.96 * sem,
                            alpha=0.13, color=colour)
        ax.set_xticks(sorted(paired["turn"].unique()))
        ax.set_ylim(0.9, 3.1)
        ax.grid(True, alpha=0.3)
        if r == 0:
            ax.set_title(dim.title(), fontsize=11)
        if c == 0:
            ax.set_ylabel(model_labels.get(m_name, m_name),
                          fontsize=10, fontweight="bold")
        if r == n_rows - 1:
            ax.set_xlabel("Turn")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, title="Evaluator",
           loc="upper center", bbox_to_anchor=(0.5, 1.01),
           ncol=2, frameon=False)
fig.suptitle("Trajectory overlay per target model — primary vs secondary",
             fontsize=13, y=1.04)
plt.tight_layout()
plt.savefig(PLOT_DIR / "fig_trajectory_evaluator_overlay_per_model.png")
plt.show()


# %% [markdown]
# # Headline summary
#
# Paste-ready numbers for the methods/results section. The format
# mirrors what reviewers expect for ordinal LLM-as-judge validation:
# n, κ_q (Landis & Koch label), Spearman ρ, exact-agreement %,
# within-1 %, mean signed difference (the bias number).

# %%
print("=" * 82)
print("HEADLINE NUMBERS — paste-ready for the paper")
print("=" * 82)
print(f"Run dir         : {RUN_DIR}")
print(f"Sample size     : {len(sampled_ids)} sessions, "
      f"{len(paired)} paired turn × dimension rows")
print(f"Rubric          : {RUBRIC_NAME}")
print(f"Primary slug    : {PRIMARY_SLUG or '(unsuffixed)'}")
print(f"Secondary slug  : {SECONDARY_SLUG}")
print(f"Output dir      : {OUT_DIR}")
print()
for _, row in overall.iterrows():
    print(
        f"  {row['dimension']:<14}  "
        f"n={int(row['n']):>5}  "
        f"κ_q={row['kappa_quadratic']:.3f} ({row['kappa_label']})  "
        f"ρ={row['spearman_rho']:.3f}  "
        f"exact={row['exact_agreement']:.3f}  "
        f"within-1={row['within_1']:.3f}  "
        f"bias={row['mean_signed_diff']:+.3f}"
    )
print()
print(f"Tables : {TABLE_DIR}")
print(f"Plots  : {PLOT_DIR}")