# %% [markdown]
# # Stance vs. correction/rebuttal — target sessions
#
# For sessions that opened with a strong turn-1 correction or rebuttal
# (score 3) but had gone silent on both by turn 8 (score 1), does the
# model's underlying *stance* toward the misinformation actually drift
# toward agreement — or is it just not *restating* an earlier
# correction/rebuttal while still implicitly treating the claim as
# false?
#
# One figure: turn on the x-axis (1-8), two lines for mean
# correction/rebuttal quality (right axis), and a stacked bar per turn
# showing the proportion of sessions whose stance toward the
# misinformation is True / False / Uncertain (left axis, %).
#
# Usage::
#
#     cd scripts/final_experiment
#     python -m stance_analysis.plot_stance_vs_correction
#
# Also runnable cell-by-cell (VS Code "Run Cell" / Jupyter) — the import
# below falls back to a non-relative import when there's no package
# context, which is the case when a cell is exec'd directly.

# %%
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/final_experiment

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as _stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

try:
    from . import config as cfg  # noqa: E402
except ImportError:
    # No parent package (e.g. running this cell-by-cell rather than via
    # `python -m stance_analysis.plot_stance_vs_correction`) — fall back
    # to importing the sibling module directly.
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # stance_analysis/
    import config as cfg  # type: ignore[no-redef]  # noqa: E402

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# %% [markdown]
# ## Load

# %%
if not cfg.PER_TURN_OUTPUT_PATH.exists():
    raise SystemExit(
        f"Per-turn summary not found: {cfg.PER_TURN_OUTPUT_PATH}\n"
        "Run the pipeline first:\n"
        "  python -m stance_analysis.select_target_conversations\n"
        "  python -m stance_analysis.run_stance_scoring\n"
        "  python -m stance_analysis.build_summary"
    )

df = pd.read_csv(cfg.PER_TURN_OUTPUT_PATH)
n_sessions = df["session_id"].nunique()
print(f"Loaded {len(df):,} rows, {n_sessions:,} target sessions, "
      f"turns {sorted(df['turn'].unique())}")

# Drop unparsed stance turns from proportion/correlation calcs (kept in
# the CSV itself for auditing — see build_summary.py).
df_parsed = df[df["stance_label"] != "unparsed"].copy()
n_unparsed = len(df) - len(df_parsed)
if n_unparsed:
    print(f"Excluding {n_unparsed} unparsed stance row(s) from proportions.")

turns = sorted(df["turn"].unique())

# %% [markdown]
# ## Per-turn aggregates

# %%
mean_correction    = df.groupby("turn")["correction"].mean().reindex(turns)
mean_rebuttal      = df.groupby("turn")["rebuttal"].mean().reindex(turns)
mean_agreeableness = df.groupby("turn")["agreeableness"].mean().reindex(turns)

stance_counts = (
    df_parsed.groupby(["turn", "stance_label"]).size()
    .unstack("stance_label")
    .reindex(index=turns, columns=list(cfg.STANCE_ORDER))
    .fillna(0)
)
stance_pct = stance_counts.div(stance_counts.sum(axis=1), axis=0) * 100

turn_table = pd.DataFrame({
    "mean_correction":    mean_correction,
    "mean_rebuttal":      mean_rebuttal,
    "mean_agreeableness": mean_agreeableness,
    "n":                  stance_counts.sum(axis=1),
    **{f"pct_{lbl}": stance_pct[lbl] for lbl in cfg.STANCE_ORDER},
})
print("\nPer-turn summary:")
print(turn_table.round(2).to_string())

(cfg.OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
turn_table.round(4).to_csv(cfg.OUT_DIR / "tables" / "stance_per_turn_summary.csv")
print(f"\nSaved → {cfg.OUT_DIR / 'tables' / 'stance_per_turn_summary.csv'}")

# %% [markdown]
# ## Figure: correction/rebuttal/agreeableness lines + stance-proportion bars

# %%
def _stance_pct_and_means(sub_df: pd.DataFrame, sub_df_parsed: pd.DataFrame, turn_range):
    """Per-turn (mean_correction, mean_rebuttal, mean_agreeableness, stance_pct) for one slice of df."""
    mean_c = sub_df.groupby("turn")["correction"].mean().reindex(turn_range)
    mean_r = sub_df.groupby("turn")["rebuttal"].mean().reindex(turn_range)
    mean_a = sub_df.groupby("turn")["agreeableness"].mean().reindex(turn_range)
    counts = (
        sub_df_parsed.groupby(["turn", "stance_label"]).size()
        .unstack("stance_label")
        .reindex(index=turn_range, columns=list(cfg.STANCE_ORDER))
        .fillna(0)
    )
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    return mean_c, mean_r, mean_a, pct


def _draw_stance_panel(
    ax_bar, turn_range, mean_c, mean_r, mean_a, pct, *, bar_labels: bool = False,
):
    """Draw one stacked-bar + correction/rebuttal/agreeableness-line panel onto ax_bar.

    Returns ax_line so the caller can pull legend handles off it (and
    ax_bar) to build a single shared legend instead of one per panel.
    """
    bottom = np.zeros(len(turn_range))
    for label in cfg.STANCE_ORDER:
        vals = pct[label].to_numpy()
        ax_bar.bar(
            turn_range, vals, bottom=bottom, width=0.6,
            color=cfg.STANCE_COLORS[label], alpha=0.85,
            label=f"Stance: {label}" if bar_labels else None,
        )
        bottom += vals

    ax_bar.set_xlabel("Turn")
    ax_bar.set_ylabel("Stance proportion (%)")
    ax_bar.set_ylim(0, 100)
    ax_bar.set_xticks(turn_range)
    ax_bar.grid(True, axis="y", alpha=0.25)

    ax_line = ax_bar.twinx()
    ax_line.plot(turn_range, mean_c.values, "o-", color="black",
                 linewidth=2, markersize=6,
                 label="Correction (mean)" if bar_labels else None)
    ax_line.plot(turn_range, mean_r.values, "s--", color="#8B0000",
                 linewidth=2, markersize=6,
                 label="Rebuttal (mean)" if bar_labels else None)
    ax_line.plot(turn_range, mean_a.values, "^:", color="#1F5FA8",
                 linewidth=2, markersize=6,
                 label="Agreeableness (mean)" if bar_labels else None)
    ax_line.set_ylabel("Mean quality score (1-3)")
    ax_line.set_ylim(0.8, 3.2)

    return ax_line


fig, ax_bar = plt.subplots(figsize=(9, 6))
ax_line = _draw_stance_panel(
    ax_bar, turns, mean_correction, mean_rebuttal, mean_agreeableness, stance_pct, bar_labels=True,
)

h_bar, l_bar = ax_bar.get_legend_handles_labels()
h_line, l_line = ax_line.get_legend_handles_labels()
ax_bar.legend(
    h_bar + h_line, l_bar + l_line,
    loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=True,
)

ax_bar.set_title(
    f"Stance toward misinformation vs. correction/rebuttal/agreeableness quality\n"
    f"Target sessions: turn-1 correction|rebuttal = 3  →  "
    f"turn-8 correction|rebuttal = 1  (N={n_sessions:,})"
)

plt.tight_layout()
(cfg.OUT_DIR / "plots").mkdir(parents=True, exist_ok=True)
out_path = cfg.OUT_DIR / "plots" / "fig_stance_vs_correction_rebuttal.png"
plt.savefig(out_path)
plt.show()
print(f"\nSaved → {out_path}")

# %% [markdown]
# ## Figure: same breakdown, faceted by target model
#
# Same panel as above, one per target model, so you can see whether the
# stance-drift pattern holds consistently across models or is driven by
# one of them (Claude and GPT-5.3 only have N=11 each after the sampling
# cap — see select_target_conversations.py — so their panels are noisier).

# %%
present_models = [m for m in cfg.MODEL_ORDER if m in df["target_model"].unique()]

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
for ax_m, model in zip(axes.flat, present_models):
    sub = df[df["target_model"] == model]
    sub_parsed = df_parsed[df_parsed["target_model"] == model]
    n_model = sub["session_id"].nunique()

    m_mean_c, m_mean_r, m_mean_a, m_pct = _stance_pct_and_means(sub, sub_parsed, turns)
    ax_m_line = _draw_stance_panel(ax_m, turns, m_mean_c, m_mean_r, m_mean_a, m_pct)
    ax_m.set_title(f"{cfg.MODEL_LABELS.get(model, model)}  (N={n_model})")

# One shared legend for the whole grid (panels themselves carry no labels).
proxy_bars = [
    plt.Rectangle((0, 0), 1, 1, color=cfg.STANCE_COLORS[lbl], alpha=0.85)
    for lbl in cfg.STANCE_ORDER
]
proxy_lines = [
    plt.Line2D([0], [0], color="black", marker="o", linewidth=2, label="Correction (mean)"),
    plt.Line2D([0], [0], color="#8B0000", marker="s", linestyle="--", linewidth=2, label="Rebuttal (mean)"),
    plt.Line2D([0], [0], color="#1F5FA8", marker="^", linestyle=":", linewidth=2, label="Agreeableness (mean)"),
]
fig.legend(
    proxy_bars + proxy_lines,
    [f"Stance: {lbl}" for lbl in cfg.STANCE_ORDER]
    + ["Correction (mean)", "Rebuttal (mean)", "Agreeableness (mean)"],
    loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=True,
)

fig.suptitle(
    "Stance toward misinformation vs. correction/rebuttal/agreeableness quality, by target model\n"
    "Target sessions: turn-1 correction|rebuttal = 3  →  turn-8 correction|rebuttal = 1",
    y=1.0,
)

plt.tight_layout(rect=[0, 0.04, 1, 0.97])
out_path_by_model = cfg.OUT_DIR / "plots" / "fig_stance_vs_correction_rebuttal_by_model.png"
plt.savefig(out_path_by_model)
plt.show()
print(f"\nSaved → {out_path_by_model}")

# %% [markdown]
# ## Correlation: is a "False" (still-against) stance associated with
# higher correction/rebuttal quality?
#
# Nominal stance labels aren't ordinal, so instead of correlating the
# arbitrary 1/2/3 codes we use a binary indicator per (session, turn):
# ``is_against = 1`` iff stance == "False", else 0 (unparsed turns
# dropped).

# %%
df_parsed["is_against"] = (df_parsed["stance_label"] == "False").astype(float)

# %% [markdown]
# ### Turn-level: does the association hold at every turn, or only in
# the session-level average?
#
# Computed *separately per turn* (144 sessions at each t) rather than
# pooling all (session, turn) rows into one correlation — pooling would
# treat a session's 8 turns as 8 independent observations when they
# aren't (same session, same target model, same belief), which inflates
# significance. Stratifying by turn keeps each test's N independent
# across sessions and additionally shows *where* the association is
# strongest — e.g. if it only emerges once correction/rebuttal has
# already collapsed, that's a different story than it holding from
# turn 1. BH-FDR corrects for the 8 turns x 2 comparisons = 16
# simultaneous tests.

# %%
turn_corr_rows = []
for t in turns:
    sub = df_parsed[df_parsed["turn"] == t]
    for col in ["correction", "rebuttal"]:
        rho, p = _stats.spearmanr(sub["is_against"], sub[col])
        turn_corr_rows.append({
            "turn": t,
            "comparison": f"is_against vs {col}",
            "n": len(sub),
            "spearman_rho": rho,
            "p": p,
        })
turn_corr_df = pd.DataFrame(turn_corr_rows)

for col in ["correction", "rebuttal"]:
    mask = turn_corr_df["comparison"] == f"is_against vs {col}"
    _, p_fdr, _, _ = multipletests(turn_corr_df.loc[mask, "p"], method="fdr_bh")
    turn_corr_df.loc[mask, "p_fdr"] = p_fdr

print("\nTurn-level Spearman correlation (is-still-against-misinfo vs. quality, per turn):")
print(turn_corr_df.round(5).to_string(index=False))

turn_corr_out = cfg.OUT_DIR / "tables" / "stance_vs_quality_correlation_by_turn.csv"
turn_corr_df.round(5).to_csv(turn_corr_out, index=False)
print(f"\nSaved → {turn_corr_out}")

# %% [markdown]
# ### Session-level: overall summary (each session collapsed to one
# point via its mean across turns 1-8) — the headline number, at the
# cost of losing the turn-by-turn detail above.

# %%
session_stats = df_parsed.groupby("session_id").agg(
    mean_is_against=("is_against", "mean"),
    mean_correction=("correction", "mean"),
    mean_rebuttal=("rebuttal", "mean"),
).reset_index()

corr_rows = []
for col in ["mean_correction", "mean_rebuttal"]:
    rho, p = _stats.spearmanr(session_stats["mean_is_against"], session_stats[col])
    corr_rows.append({
        "comparison": f"mean_is_against vs {col}",
        "n": len(session_stats),
        "spearman_rho": rho,
        "p": p,
    })
corr_df = pd.DataFrame(corr_rows)
print("\nSession-level Spearman correlation (is-still-against-misinfo vs. quality):")
print(corr_df.round(5).to_string(index=False))

corr_df.round(5).to_csv(cfg.OUT_DIR / "tables" / "stance_vs_quality_correlation.csv", index=False)
print(f"\nSaved → {cfg.OUT_DIR / 'tables' / 'stance_vs_quality_correlation.csv'}")
