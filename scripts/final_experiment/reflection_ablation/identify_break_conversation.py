# %% [markdown]
# # Break Conversation Analysis
#
# Identify and analyse conversations where a **reflection break** occurred.
# A break is the first turn whose `reflection_attempts` list has length > 1
# (at least one failed attempt before a successful one).
#
# **Factors examined:**
# `iv1` · Belief category · Target LLM · Break type · Break turn number
# — plus pairwise / three-way interactions and post-hoc tests.

# %%
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu

# %% [markdown]
# ## Configuration

# %%
# --- path resolution: works both as a script and inside a notebook ----
try:
    _REPO_ROOT = Path(__file__).resolve().parents[3]
except NameError:
    _REPO_ROOT = Path.cwd()
    while not (_REPO_ROOT / "results").exists() and _REPO_ROOT != _REPO_ROOT.parent:
        _REPO_ROOT = _REPO_ROOT.parent

CONV_DIR = _REPO_ROOT / "results/final_experiment/main_user_IVs/20260427_165233/conversations"

# Abbreviations for long model IDs
LLM_ABBREV: dict[str, str] = {
    "openrouter/anthropic/claude-sonnet-4.6":      "Claude-S",
    "openrouter/deepseek/deepseek-v3.2":           "DeepSeek",
    "openrouter/google/gemini-3-flash-preview":    "Gemini-F",
    "openrouter/openai/gpt-5.3-chat":              "GPT-5.3",
}

BREAK_TYPE_COLORS = {
    "character": "#4C72B0",
    "belief":    "#DD8452",
    "both":      "#55A868",
    "unknown":   "#C7C7C7",
}

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110

# %% [markdown]
# ## Helper Functions

# %%
def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def find_break_turn(
    turns: Iterable[Dict[str, Any]],
) -> Optional[tuple[int, Dict[str, Any]]]:
    for idx, turn in enumerate(turns, start=1):
        attempts = turn.get("reflection_attempts") or []
        try:
            n = len(attempts)
        except TypeError:
            n = 0
        if n > 1:
            return int(turn.get("turn", idx)), turn
    return None


def classify_break_type(turn: Dict[str, Any]) -> str:
    def _int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    n_char = _int(turn.get("n_character_breaks", 0))
    n_bel  = _int(turn.get("n_belief_breaks",    0))
    if n_char > 0 and n_bel > 0:
        return "both"
    if n_char > 0:
        return "character"
    if n_bel > 0:
        return "belief"
    return "unknown"


def sig_str(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def kruskal_result(groups: list) -> tuple[float, float]:
    """Kruskal-Wallis across groups; returns (H, p). Needs ≥2 non-empty groups."""
    non_empty = [g for g in groups if len(g) >= 2]
    if len(non_empty) < 2:
        return float("nan"), float("nan")
    return kruskal(*non_empty)


def pairwise_mwu_bonferroni(
    data: pd.DataFrame, group_col: str, value_col: str
) -> dict[tuple[str, str], tuple[float, float]]:
    """Mann-Whitney U, Bonferroni-corrected. Returns {(a,b): (U, p_adj)}."""
    levels = sorted(data[group_col].dropna().unique())
    pairs = list(combinations(levels, 2))
    results = {}
    for a, b in pairs:
        ga = data.loc[data[group_col] == a, value_col].dropna()
        gb = data.loc[data[group_col] == b, value_col].dropna()
        if len(ga) < 2 or len(gb) < 2:
            continue
        U, p = mannwhitneyu(ga, gb, alternative="two-sided")
        results[(a, b)] = (U, min(p * len(pairs), 1.0))
    return results


def add_sig_brackets(
    ax: plt.Axes,
    x_order: list[str],
    pair_pvals: dict[tuple[str, str], tuple[float, float]],
    y_top: float,
    y_step: float,
) -> None:
    """Draw significance brackets for significant pairs on a categorical axis."""
    pos = {lbl: i for i, lbl in enumerate(x_order)}
    sig_pairs = [
        ((a, b), p)
        for (a, b), (_, p) in pair_pvals.items()
        if sig_str(p) != "ns" and a in pos and b in pos
    ]
    # draw shortest spans first to reduce crossing
    sig_pairs.sort(key=lambda x: abs(pos[x[0][0]] - pos[x[0][1]]))
    current_y = y_top
    for (a, b), p in sig_pairs:
        x1, x2 = pos[a], pos[b]
        current_y += y_step
        bar_y = current_y
        ax.plot([x1, x1, x2, x2],
                [bar_y, bar_y + y_step * 0.3, bar_y + y_step * 0.3, bar_y],
                lw=0.9, c="black")
        ax.text((x1 + x2) / 2, bar_y + y_step * 0.35,
                sig_str(p), ha="center", va="bottom", fontsize=8)
    if sig_pairs:
        ax.set_ylim(top=current_y + y_step * 1.5)

# %% [markdown]
# ## Load & Flatten Conversations

# %%
records = []
for path in sorted(CONV_DIR.glob("*.json")):
    data = load_json(path)
    if not data:
        continue

    turns      = data.get("turns")  or []
    break_info = find_break_turn(turns)
    cell       = data.get("cell")   or {}
    belief     = data.get("belief") or {}
    models     = data.get("models") or {}

    raw_llm = str(models.get("target_llm", "unknown"))
    records.append({
        "file":            path.name,
        "has_break":       break_info is not None,
        "break_turn":      break_info[0] if break_info else None,
        "break_type":      classify_break_type(break_info[1]) if break_info else None,
        "iv1":             str(cell.get("iv1",        "unknown")),
        "belief_category": str(belief.get("category", "unknown")),
        "target_llm":      LLM_ABBREV.get(raw_llm, raw_llm),
    })

if not records:
    raise FileNotFoundError(
        f"No JSON files found in:\n  {CONV_DIR}\n"
        "Check that CONV_DIR points to the correct conversations directory."
    )

df = pd.DataFrame(records)
print(f"Loaded {len(df):,} conversations from {CONV_DIR}")
df.head()

# %% [markdown]
# ## Overview Statistics

# %%
total      = len(df)
n_breaks   = int(df["has_break"].sum())
break_rate = n_breaks / total if total else 0.0

print(f"Total conversations     : {total:,}")
print(f"Conversations with breaks: {n_breaks:,}  ({break_rate:.1%})")

df_breaks = df[df["has_break"]].copy()

# %% [markdown]
# ## Break Turn Distribution

# %%
fig, ax = plt.subplots(figsize=(8, 4))
(
    df_breaks["break_turn"]
    .value_counts()
    .sort_index()
    .plot.bar(ax=ax, color="steelblue")
)
ax.set_xlabel("Turn number")
ax.set_ylabel("Count")
ax.set_title("Distribution of turn where first break occurs")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Break Rate by Individual Factor
#
# Fraction of conversations with a break within each level — removes
# sampling-imbalance bias vs. raw counts.

# %%
def break_rate_table(col: str) -> pd.DataFrame:
    return (
        df.groupby(col)["has_break"]
        .agg(total="count", n_breaks="sum")
        .assign(break_rate=lambda x: x["n_breaks"] / x["total"])
        .sort_values("break_rate", ascending=False)
    )


for col in ["iv1", "belief_category", "target_llm"]:
    print(f"\nBreak rate by {col}:")
    print(break_rate_table(col).to_string())

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, title in zip(
    axes,
    ["iv1", "belief_category", "target_llm"],
    ["iv1", "Belief category", "Target LLM"],
):
    br = break_rate_table(col)
    bars = ax.bar(range(len(br)), br["break_rate"], color="coral")
    ax.set_xticks(range(len(br)))
    ax.set_xticklabels(br.index, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Break rate by {title}")
    ax.set_ylabel("Break rate")
    for i, v in enumerate(br["break_rate"]):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", va="bottom", fontsize=8)
fig.suptitle("Break rate by individual factors")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Break Type Composition — Detailed
#
# For each factor level: absolute counts, percentage within level, and a
# chi-squared test of whether break-type proportions differ across levels.

# %%
def break_type_detail(col: str) -> None:
    pivot = (
        df_breaks.groupby([col, "break_type"])
        .size()
        .unstack("break_type")
        .fillna(0)
        .astype(int)
    )
    ordered = [t for t in BREAK_TYPE_COLORS if t in pivot.columns]
    pivot = pivot[ordered]

    # row-normalised percentages
    pct = pivot.div(pivot.sum(axis=1), axis=0).mul(100).round(1)
    combined = pivot.astype(str) + " (" + pct.astype(str) + "%)"
    print(f"\nBreak-type counts (% within {col}):")
    print(combined.to_string())

    # chi-squared test
    if pivot.shape[0] >= 2 and pivot.shape[1] >= 2:
        chi2, p, dof, _ = chi2_contingency(pivot)
        print(f"  χ²={chi2:.2f}  df={dof}  p={p:.4f}  {sig_str(p)}")


for col in ["iv1", "belief_category", "target_llm"]:
    break_type_detail(col)

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for col, ax_abs, ax_pct in zip(
    ["iv1", "belief_category", "target_llm"],
    axes[0], axes[1],
):
    pivot = (
        df_breaks.groupby([col, "break_type"])
        .size()
        .unstack("break_type")
        .fillna(0)
        .astype(int)
    )
    ordered = [t for t in BREAK_TYPE_COLORS if t in pivot.columns]
    pivot = pivot[ordered]
    colors = [BREAK_TYPE_COLORS[t] for t in ordered]

    # absolute counts
    pivot.plot.bar(stacked=True, ax=ax_abs, color=colors, legend=False)
    ax_abs.set_title(f"Abs. count by {col}")
    ax_abs.set_ylabel("Count")
    ax_abs.tick_params(axis="x", rotation=30)

    # row-normalised (%)
    pct = pivot.div(pivot.sum(axis=1), axis=0)
    pct.plot.bar(stacked=True, ax=ax_pct, color=colors, legend=False)
    ax_pct.set_title(f"Proportion (%) by {col}")
    ax_pct.set_ylabel("Fraction")
    ax_pct.set_ylim(0, 1)
    ax_pct.tick_params(axis="x", rotation=30)

    # significance label
    if pivot.shape[0] >= 2 and pivot.shape[1] >= 2:
        _, p, dof, _ = chi2_contingency(pivot)
        ax_abs.set_xlabel(f"χ²-test  p={p:.4f} {sig_str(p)}", fontsize=8)

legend_handles = [
    mpatches.Patch(color=BREAK_TYPE_COLORS[t], label=t)
    for t in BREAK_TYPE_COLORS if t != "unknown"
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Break type composition by factor  (top: counts · bottom: proportions)")
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.show()

# %% [markdown]
# ## Pairwise Factor Interactions — Break Rate Heatmaps

# %%
PAIRS = [
    ("iv1",             "target_llm"),
    ("iv1",             "belief_category"),
    ("belief_category", "target_llm"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (r, c) in zip(axes, PAIRS):
    pivot = (
        df.groupby([r, c])["has_break"]
        .mean()
        .unstack(c)
        .fillna(0)
    )
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".0%",
        cmap="YlOrRd", vmin=0, vmax=1, linewidths=0.5,
    )
    ax.set_title(f"Break rate: {r} × {c}")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
fig.suptitle("Pairwise factor interactions — break rate")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Chi-squared Independence Tests (all factor pairs)
#
# Codes: `***` p<0.001 · `**` p<0.01 · `*` p<0.05 · `ns`

# %%
print("Chi-squared tests — break counts by factor pairs:\n")
for col_a, col_b in PAIRS:
    ct = pd.crosstab(df[col_a], df[col_b],
                     values=df["has_break"], aggfunc="sum").fillna(0)
    chi2, p, dof, _ = chi2_contingency(ct)
    print(f"  {col_a:20s} × {col_b:20s}  χ²={chi2:7.2f}  df={dof}  p={p:.4f}  {sig_str(p)}")

# %% [markdown]
# ## Deep Dive: iv1 × belief_category  (**significant**)
#
# This is the only significant pair (χ²=149.92, p<0.001).
# We unpack it with standardised residuals and post-hoc pairwise tests.

# %%
ct_raw = pd.crosstab(df["iv1"], df["belief_category"],
                     values=df["has_break"], aggfunc="sum").fillna(0)
chi2_val, p_val, dof_val, expected = chi2_contingency(ct_raw)

std_resid = (ct_raw - expected) / (expected ** 0.5)

print(f"iv1 × belief_category  χ²={chi2_val:.2f}  df={dof_val}  p={p_val:.4e}")
print("\nStandardised residuals (|z| > 2 → cell drives significance):")
print(std_resid.round(2).to_string())

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# left: break count heatmap
sns.heatmap(ct_raw.astype(int), ax=axes[0], annot=True, fmt="d",
            cmap="Blues", linewidths=0.5)
axes[0].set_title("Break counts: iv1 × belief_category")
axes[0].tick_params(axis="x", rotation=30)
axes[0].tick_params(axis="y", rotation=0)

# right: standardised residuals — diverging palette
vmax = max(abs(std_resid.values.max()), abs(std_resid.values.min()))
sns.heatmap(std_resid.round(2), ax=axes[1], annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax, linewidths=0.5)
axes[1].set_title("Standardised residuals\n(red = more breaks than expected)")
axes[1].tick_params(axis="x", rotation=30)
axes[1].tick_params(axis="y", rotation=0)

fig.suptitle(f"iv1 × belief_category  χ²={chi2_val:.2f}  df={dof_val}  p={p_val:.2e} ***")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Post-hoc: which iv1 levels differ in break rate?
#
# For each belief category separately: pairwise Fisher's exact / chi-squared
# tests on iv1 levels, Bonferroni-corrected.

# %%
print("Post-hoc pairwise chi-squared — break rate of iv1 pairs within each belief category")
print("(Bonferroni-corrected; showing only p < 0.05)\n")

iv1_levels = sorted(df["iv1"].unique())
n_pairs = len(list(combinations(iv1_levels, 2)))

for cat in sorted(df["belief_category"].unique()):
    sub = df[df["belief_category"] == cat]
    sig_found = False
    lines = []
    for a, b in combinations(iv1_levels, 2):
        ct2 = pd.crosstab(
            sub[sub["iv1"].isin([a, b])]["iv1"],
            sub[sub["iv1"].isin([a, b])]["has_break"],
        )
        if ct2.shape != (2, 2):
            continue
        chi2, p, *_ = chi2_contingency(ct2)
        p_adj = min(p * n_pairs, 1.0)
        if p_adj < 0.05:
            lines.append(f"    {a:12s} vs {b:12s}  χ²={chi2:.2f}  p_adj={p_adj:.4f}  {sig_str(p_adj)}")
            sig_found = True
    if sig_found:
        print(f"  belief_category = {cat}")
        print("\n".join(lines))

# %% [markdown]
# ### Break rate profile: iv1 × belief_category (line plot)

# %%
rate_grid = (
    df.groupby(["iv1", "belief_category"])["has_break"]
    .mean()
    .reset_index(name="break_rate")
)

fig, ax = plt.subplots(figsize=(9, 5))
cats = sorted(df["belief_category"].unique())
palette = sns.color_palette("tab10", len(cats))
for cat, color in zip(cats, palette):
    sub = rate_grid[rate_grid["belief_category"] == cat].set_index("iv1")
    sub = sub.reindex(sorted(iv1_levels))
    ax.plot(sub.index, sub["break_rate"], marker="o", label=cat, color=color)

ax.set_ylabel("Break rate")
ax.set_xlabel("iv1")
ax.set_ylim(0, 1)
ax.set_title("Break rate by iv1 — lines per belief category")
ax.legend(title="belief_category", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Break Turn Number by Factor — Significance Tests
#
# Kruskal-Wallis overall + pairwise Mann-Whitney U (Bonferroni-corrected).
# Box plots show IQR; jittered strip overlaid so sparse groups remain visible.

# %%
FACTORS = ["iv1", "belief_category", "target_llm"]

for col in FACTORS:
    levels = sorted(df_breaks[col].dropna().unique())
    groups = [df_breaks.loc[df_breaks[col] == lv, "break_turn"].dropna().values
              for lv in levels]
    H, p_kw = kruskal_result(groups)
    print(f"\nBreak turn ~ {col}")
    print(f"  Kruskal-Wallis  H={H:.2f}  p={p_kw:.4f}  {sig_str(p_kw)}")

    pairs = pairwise_mwu_bonferroni(df_breaks, col, "break_turn")
    sig_pairs = [(ab, pv) for ab, (_, pv) in pairs.items() if sig_str(pv) != "ns"]
    if sig_pairs:
        print("  Significant pairwise comparisons (Bonferroni-corrected):")
        for (a, b), pv in sorted(sig_pairs, key=lambda x: x[1]):
            print(f"    {a} vs {b}  p_adj={pv:.4f}  {sig_str(pv)}")
    else:
        print("  No significant pairwise comparisons after Bonferroni correction.")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for ax, col in zip(axes, FACTORS):
    order = sorted(df_breaks[col].dropna().unique())
    groups = [df_breaks.loc[df_breaks[col] == lv, "break_turn"].dropna().values
              for lv in order]

    # box plot (no outlier markers — strip covers them)
    sns.boxplot(
        data=df_breaks, x=col, y="break_turn", order=order, ax=ax,
        palette="pastel", showfliers=False,
        boxprops=dict(alpha=0.6),
    )
    # strip overlay so sparse groups are always visible
    sns.stripplot(
        data=df_breaks, x=col, y="break_turn", order=order, ax=ax,
        color="0.3", size=3, alpha=0.5, jitter=True,
    )

    H, p_kw = kruskal_result(groups)
    ax.set_title(f"Break turn by {col}\nKW H={H:.1f}  p={p_kw:.3f} {sig_str(p_kw)}")
    ax.set_xlabel(col)
    ax.set_ylabel("Break turn")
    ax.tick_params(axis="x", rotation=30)

    # significance brackets for significant pairs
    y_top = df_breaks["break_turn"].max()
    y_step = max(y_top * 0.07, 0.5)
    pairs = pairwise_mwu_bonferroni(df_breaks, col, "break_turn")
    add_sig_brackets(ax, order, pairs, y_top=y_top, y_step=y_step)

fig.suptitle("When breaks occur — by factor  (box+strip; brackets = sig. pairs, Bonferroni)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Three-way Interaction: iv1 × target_llm × break_type

# %%
three_way = (
    df_breaks.groupby(["iv1", "target_llm", "break_type"])
    .size()
    .reset_index(name="count")
)

iv1_order = sorted(three_way["iv1"].unique())
llm_order  = sorted(three_way["target_llm"].unique())

g = sns.FacetGrid(three_way, col="iv1", col_order=iv1_order, height=4, aspect=0.9, sharey=False)
g.map_dataframe(
    sns.barplot,
    x="target_llm",
    y="count",
    hue="break_type",
    palette=BREAK_TYPE_COLORS,
    order=llm_order,
)
g.add_legend(title="Break type")
g.set_axis_labels("Target LLM", "Count")
g.set_titles(col_template="iv1 = {col_name}")
for ax in g.axes.flat:
    ax.tick_params(axis="x", rotation=30)
g.figure.suptitle("Three-way: iv1 × target_llm × break_type", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Three-way: iv1 × belief_category per target_LLM
#
# Break-rate heatmap faceted by LLM.

# %%
llms = sorted(df["target_llm"].unique())
ncols = len(llms)
fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
if ncols == 1:
    axes = [axes]
for ax, llm in zip(axes, llms):
    sub = df[df["target_llm"] == llm]
    pivot = (
        sub.groupby(["iv1", "belief_category"])["has_break"]
        .mean()
        .unstack("belief_category")
        .fillna(0)
    )
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".0%",
        cmap="YlOrRd", vmin=0, vmax=1, linewidths=0.5,
    )
    ax.set_title(f"{llm}")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
fig.suptitle("Break rate: iv1 × belief_category — faceted by target_LLM")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Full Summary Table

# %%
summary = (
    df.groupby(["iv1", "belief_category", "target_llm"])["has_break"]
    .agg(total="count", n_breaks="sum")
    .assign(break_rate=lambda x: (x["n_breaks"] / x["total"]).round(3))
    .reset_index()
    .sort_values("break_rate", ascending=False)
)
print(summary.to_string(index=False))
