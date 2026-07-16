# %% [markdown]
# # Mitigation vs. baseline score comparison
#
# Paired before/after comparison: for every mitigation session, diff its
# `misinfo_v1_split` (correction/rebuttal/agreeableness, scored as three
# independent single-dimension prompts) and `misinfo_stance_history_split`
# (stance toward the misinformation, evaluator sees every prior turn as
# context — same rubric `stance_analysis/plot_stance_history.py` uses,
# NOT the turn-independent `misinfo_stance_split` this script used
# originally) scores against its paired baseline session — same belief,
# IV condition, and target model, differing only in the target's system
# prompt (see `run_conversations.py`). Answers "did the mitigation
# system prompt actually change these sessions?"
#
# Both sides read from real score JSON files, same rubric filename
# (`config.HISTORY_STANCE_RUBRIC_NAME`, sourced from
# `stance_analysis.config.HISTORY_STANCE_RUBRIC_NAME`, currently
# "misinfo_stance_history_split_1" — stance_analysis's scores/ dir also
# has an *unsuffixed* "misinfo_stance_history_split" from an older,
# since-changed prompt wording that's explicitly excluded from every
# current analysis, deliberately not used here).
#
# Requires all three scoring passes already on disk for the mitigation run:
#
#     python -m main_user_IVs.run_scoring --run-dir <mitigation_run_dir> --rubric misinfo_v1_split
#     python -m stance_analysis.run_stance_history_scoring --sample-index <this study's sample index>
#     python -m mitigation_plan.score_history_stance
#
# Runnable as a script (`python -m mitigation_plan.compare_to_baseline`)
# or cell-by-cell (VS Code "Run Cell" / Jupyter / `jupytext --to notebook
# compare_to_baseline.py`) — the import below falls back to a
# non-relative import when there's no package context, which is the
# case when a cell is exec'd directly.

# %%
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/final_experiment

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.distance import pdist, squareform  # noqa: E402

try:
    from . import config as cfg  # noqa: E402
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # mitigation_plan/
    import config as cfg  # type: ignore[no-redef]  # noqa: E402

pd.set_option("display.width", 120)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# %% [markdown]
# ## Parameters — edit these and re-run
#
# `RUN_DIR` defaults to the most recently created run under
# `results/final_experiment/mitigation_plan/`; override with the
# `MITIGATION_RUN_DIR` env var (same convention as `SOURCE_RUN_DIR` in
# `stance_analysis`/`response_diversity`/`static_interactive_ablation`)
# or just reassign the variable in this cell before re-running.
#
# `SAMPLE_INDEX_PATH` picks which sample this analysis covers — several
# sample indices can point at conversations living in the *same*
# `RUN_DIR` (e.g. the failure-mode-targeted 75-session pilot and the
# general 35/model pilot both landed in the same run dir, see
# select_failure_mode_pilot.py / select_general_sample.py). Defaults to
# the general sample; override with the `MITIGATION_SAMPLE_INDEX` env
# var or by reassigning the variable below. Outputs are namespaced by
# `SAMPLE_LABEL` (the index file's stem) so switching samples never
# overwrites another sample's comparison.csv / plots / tables.

# %%
def _latest_run_dir() -> Path | None:
    root = cfg.RESULTS_DIR / cfg.EXPERIMENT_NAME
    if not root.exists():
        return None
    candidates = sorted(p for p in root.iterdir() if p.is_dir())
    return candidates[-1] if candidates else None


_RUN_DIR_ENV = os.environ.get("MITIGATION_RUN_DIR")
RUN_DIR = Path(_RUN_DIR_ENV).expanduser().resolve() if _RUN_DIR_ENV else _latest_run_dir()

_SAMPLE_INDEX_ENV = os.environ.get("MITIGATION_SAMPLE_INDEX")
SAMPLE_INDEX_PATH = (
    Path(_SAMPLE_INDEX_ENV).expanduser().resolve() if _SAMPLE_INDEX_ENV
    else cfg.GENERAL_SAMPLE_INDEX_PATH
)
SAMPLE_LABEL = SAMPLE_INDEX_PATH.stem  # e.g. "mitigation_sample_index_general"

BASELINE_SCORES_DIR = cfg.SOURCE_RUN_DIR / "scores"
OUT_CSV = (RUN_DIR / f"comparison__{SAMPLE_LABEL}.csv") if RUN_DIR else None

# Same rubric filenames on both sides.
RUBRIC_NAMES = ("misinfo_v1_split", cfg.HISTORY_STANCE_RUBRIC_NAME)
STANCE_LABELS = cfg.STANCE_LABELS
PLOTS_DIR = (RUN_DIR / "plots" / SAMPLE_LABEL) if RUN_DIR else None

if RUN_DIR is None or not RUN_DIR.exists():
    raise SystemExit(
        f"No mitigation run dir found under {cfg.RESULTS_DIR / cfg.EXPERIMENT_NAME}. "
        "Set MITIGATION_RUN_DIR or reassign RUN_DIR above."
    )
if not (RUN_DIR / "scores").exists():
    raise SystemExit(
        f"No scores/ dir under {RUN_DIR} — run main_user_IVs.run_scoring "
        "(misinfo_v1_split) and mitigation_plan.score_history_stance against "
        "this run dir first."
    )
if not SAMPLE_INDEX_PATH.exists():
    raise SystemExit(
        f"Sample index not found: {SAMPLE_INDEX_PATH}\n"
        "Set MITIGATION_SAMPLE_INDEX or reassign SAMPLE_INDEX_PATH above."
    )

print(f"Mitigation run       : {RUN_DIR}")
print(f"Sample index         : {SAMPLE_INDEX_PATH}")
print(f"Sample label         : {SAMPLE_LABEL}")
print(f"Baseline scores dir  : {BASELINE_SCORES_DIR}")
print(f"Rubrics (same name both sides) : {RUBRIC_NAMES}")

# %% [markdown]
# ## Helper: per-session mean scores

# %%
def _session_means(score_path: Path) -> dict[str, float] | None:
    """Per-dimension mean over turns, skipping -1.0 parse failures."""
    if not score_path.exists():
        return None
    with score_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    dims = data["rubric_dimensions"]
    sums = {d: 0.0 for d in dims}
    counts = {d: 0 for d in dims}
    for turn in data["turns"]:
        for d, v in turn["scores"].items():
            if v == -1.0:
                continue
            sums[d] += v
            counts[d] += 1
    return {d: (sums[d] / counts[d] if counts[d] else float("nan")) for d in dims}


def _print_group_table(df: pd.DataFrame, group_cols: list[str]) -> None:
    dims = [c[len("diff_"):] for c in df.columns if c.startswith("diff_")]
    agg = {}
    for dim in dims:
        agg[f"baseline_{dim}"] = "mean"
        agg[f"mitigated_{dim}"] = "mean"
        agg[f"diff_{dim}"] = "mean"
    grouped = df.groupby(group_cols).agg(agg)
    grouped["n"] = df.groupby(group_cols).size()
    print(grouped.round(2).to_string())

# %% [markdown]
# ## Build the paired comparison table
#
# One row per sampled session: baseline_* / mitigated_* / diff_* for
# each rubric dimension present in both score files.

# %%
with SAMPLE_INDEX_PATH.open("r", encoding="utf-8") as fh:
    entries = json.load(fh)["entries"]

mit_scores_dir = RUN_DIR / "scores"

rows = []
n_missing_baseline = 0
n_missing_mitigated = 0
for entry in entries:
    source_sid = entry["session_id"]
    mit_sid = f"{source_sid}{cfg.SESSION_ID_SUFFIX}"

    row = {
        "source_session_id": source_sid,
        "failure_mode": entry.get("failure_mode", "n/a"),
        "target_model": entry["target_model"],
        "iv1": entry.get("iv1"),
        "belief_category": entry.get("belief_category"),
    }
    any_baseline, any_mitigated = False, False
    for rubric in RUBRIC_NAMES:
        base = _session_means(BASELINE_SCORES_DIR / f"{source_sid}__{rubric}.json")
        mit = _session_means(mit_scores_dir / f"{mit_sid}__{rubric}.json")
        if base is not None:
            any_baseline = True
            for dim, v in base.items():
                row[f"baseline_{dim}"] = v
        if mit is not None:
            any_mitigated = True
            for dim, v in mit.items():
                row[f"mitigated_{dim}"] = v

    if not any_baseline:
        n_missing_baseline += 1
        continue
    if not any_mitigated:
        n_missing_mitigated += 1
        continue
    rows.append(row)

if n_missing_baseline:
    print(f"  (skipped {n_missing_baseline} session(s) with no baseline score file — "
          "run stance_analysis.run_stance_history_scoring first)")
if n_missing_mitigated:
    print(f"  (skipped {n_missing_mitigated} session(s) with no mitigated score file — "
          "run main_user_IVs.run_scoring and mitigation_plan.score_history_stance first)")

df = pd.DataFrame(rows)
for dim in ("correction", "rebuttal", "agreeableness", "stance"):
    b, m = f"baseline_{dim}", f"mitigated_{dim}"
    if b in df.columns and m in df.columns:
        df[f"diff_{dim}"] = df[m] - df[b]

if df.empty:
    raise SystemExit("No paired sessions with both baseline and mitigated scores found.")

print(f"\nPaired sessions with both baseline + mitigated scores: {len(df)}")
df.head()

# %% [markdown]
# ## By failure_mode x target_model

# %%
_print_group_table(df, ["failure_mode", "target_model"])

# %% [markdown]
# ## By failure_mode (pooled across models)

# %%
_print_group_table(df, ["failure_mode"])

# %% [markdown]
# ## Overall

# %%
df["_all"] = "all sessions"
_print_group_table(df, ["_all"])
df.drop(columns="_all", inplace=True)

# %% [markdown]
# ## Stance label shift (baseline -> mitigated), per session

# %%
if "diff_stance" in df.columns:
    for col, label in (("baseline_stance", "baseline"), ("mitigated_stance", "mitigated")):
        counts = df[col].round().map(STANCE_LABELS).value_counts()
        print(f"\n{label}:")
        print(counts.to_string())
else:
    print("No stance scores found for these sessions — run "
          "stance_analysis.run_stance_history_scoring (baseline) and "
          "mitigation_plan.score_history_stance (mitigated) first.")

# %% [markdown]
# ## Save per-session comparison to CSV

# %%
if OUT_CSV is not None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote per-session comparison ({len(df)} rows) -> {OUT_CSV}")

# %% [markdown]
# ## Per-turn data (baseline vs. mitigated)
#
# Same shape as `stance_analysis/plot_stance_history.py`'s per-turn
# dataframe, but long over `condition` (baseline / mitigated) so each
# sampled session contributes two 8-turn trajectories instead of one —
# needed for the turn-by-turn history plots below (the session-level
# means above collapse turns 1-8 into a single number per session).

# %%
def _load_turn_scores(score_path: Path) -> dict[int, dict[str, float]]:
    """turn -> {dim: raw value}, straight off one rubric's score file."""
    if not score_path.exists():
        return {}
    with score_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {int(t["turn"]): dict(t["scores"]) for t in data["turns"]}


turn_rows = []
for entry in entries:
    source_sid = entry["session_id"]
    mit_sid = f"{source_sid}{cfg.SESSION_ID_SUFFIX}"
    target_model = entry["target_model"]
    failure_mode = entry.get("failure_mode", "n/a")
    iv1 = entry.get("iv1")

    for condition, sid, scores_dir in (
        ("baseline", source_sid, BASELINE_SCORES_DIR),
        ("mitigated", mit_sid, mit_scores_dir),
    ):
        v1_turns = _load_turn_scores(scores_dir / f"{sid}__misinfo_v1_split.json")
        stance_turns = _load_turn_scores(scores_dir / f"{sid}__{cfg.HISTORY_STANCE_RUBRIC_NAME}.json")
        for t in sorted(set(v1_turns) | set(stance_turns)):
            v1_t = v1_turns.get(t, {})
            stance_val = stance_turns.get(t, {}).get("stance", -1.0)
            turn_rows.append({
                "source_session_id": source_sid,
                "condition":         condition,
                "target_model":      target_model,
                "failure_mode":      failure_mode,
                "iv1":               iv1,
                "turn":              t,
                "correction":        v1_t.get("correction", -1.0),
                "rebuttal":          v1_t.get("rebuttal", -1.0),
                "agreeableness":     v1_t.get("agreeableness", -1.0),
                "stance_label":      cfg.STANCE_LABELS.get(stance_val, "unparsed"),
            })

turn_df = pd.DataFrame(turn_rows)
for _dim in ("correction", "rebuttal", "agreeableness"):
    turn_df.loc[turn_df[_dim] == -1.0, _dim] = float("nan")

turn_df_parsed = turn_df[turn_df["stance_label"] != "unparsed"].copy()
turns = sorted(turn_df["turn"].unique())
print(
    f"Per-turn rows: {len(turn_df)}  "
    f"({turn_df['source_session_id'].nunique()} sessions x 2 conditions, turns {turns})"
)

# %% [markdown]
# ## Double-check: does each target_model map to exactly one failure_mode?
#
# By construction (`select_failure_mode_pilot.py`), claude-sonnet-4.6
# sessions are all `disengagement` and gemini-3-flash / deepseek-v3.2
# sessions are all `affective_validation`. Verifying that holds before
# reading the per-model plots below as if they were pure
# per-failure-mode plots — if a model shows more than one failure_mode
# here, the sample composition changed and the plot titles below would
# be misleading.

# %%
model_to_modes = turn_df.groupby("target_model")["failure_mode"].unique()
for model, modes in model_to_modes.items():
    status = "OK — pure" if len(modes) == 1 else "MIXED — plot title below overstates purity"
    label = cfg.MODEL_LABELS.get(model, model)
    print(f"  {label:<20} ({model:<20}) -> {list(modes)}   [{status}]")

MODEL_FAILURE_MODE = {
    model: modes[0] if len(modes) == 1 else "mixed"
    for model, modes in model_to_modes.items()
}

# %% [markdown]
# ## Plotting helpers (same as stance_analysis/plot_stance_history.py)

# %%
def _stance_pct_and_means(sub_df: pd.DataFrame, sub_df_parsed: pd.DataFrame, turn_range):
    """Per-turn (mean_correction, mean_rebuttal, mean_agreeableness, stance_pct) for one slice."""
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


def _draw_stance_panel(ax_bar, turn_range, mean_c, mean_r, mean_a, pct, *, bar_labels: bool = False):
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


proxy_bars = [
    plt.Rectangle((0, 0), 1, 1, color=cfg.STANCE_COLORS[lbl], alpha=0.85)
    for lbl in cfg.STANCE_ORDER
]
proxy_lines = [
    plt.Line2D([0], [0], color="black", marker="o", linewidth=2, label="Correction (mean)"),
    plt.Line2D([0], [0], color="#8B0000", marker="s", linestyle="--", linewidth=2, label="Rebuttal (mean)"),
    plt.Line2D([0], [0], color="#1F5FA8", marker="^", linestyle=":", linewidth=2, label="Agreeableness (mean)"),
]

if PLOTS_DIR is not None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Figure: one plot per target model, baseline vs. mitigated side by side
#
# Three figures (one per model present in the sample). Each figure has
# two panels — left = baseline, right = mitigated — sharing one legend,
# so the turn-by-turn shift in stance proportions and
# correction/rebuttal/agreeableness quality under the mitigation prompt
# is directly comparable to the unmitigated baseline for that model /
# failure mode.

# %%
present_models = [m for m in cfg.MODEL_LABELS if m in turn_df["target_model"].unique()]
model_condition_stats: dict[str, dict[str, dict]] = {}

for model in present_models:
    model_condition_stats[model] = {}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)

    for ax, condition in zip(axes, ("baseline", "mitigated")):
        sub = turn_df[(turn_df["target_model"] == model) & (turn_df["condition"] == condition)]
        sub_parsed = turn_df_parsed[
            (turn_df_parsed["target_model"] == model) & (turn_df_parsed["condition"] == condition)
        ]
        n_sessions = sub["source_session_id"].nunique()

        mean_c, mean_r, mean_a, pct = _stance_pct_and_means(sub, sub_parsed, turns)
        model_condition_stats[model][condition] = {
            "n": n_sessions, "mean_c": mean_c, "mean_r": mean_r, "mean_a": mean_a, "pct": pct,
        }
        _draw_stance_panel(ax, turns, mean_c, mean_r, mean_a, pct)
        ax.set_title(f"{condition.capitalize()}  (N={n_sessions})")

    fig.legend(
        proxy_bars + proxy_lines,
        [f"Stance: {lbl}" for lbl in cfg.STANCE_ORDER]
        + ["Correction (mean)", "Rebuttal (mean)", "Agreeableness (mean)"],
        loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=True,
    )

    fig.suptitle(
        f"{cfg.MODEL_LABELS.get(model, model)} — baseline vs. mitigated\n"
        f"failure_mode = {MODEL_FAILURE_MODE.get(model, 'n/a')}",
        y=1.02,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    if PLOTS_DIR is not None:
        out_path = PLOTS_DIR / f"fig_baseline_vs_mitigated_{model}.png"
        plt.savefig(out_path)
        print(f"Saved -> {out_path}")
    plt.show()

# %% [markdown]
# ## Snapshot tables (T1 / T5 / T8), one per model
#
# Reads directly out of `model_condition_stats` (the same mean_c/mean_r/
# mean_a/pct the figures above just plotted), so these numbers can't
# drift from what's shown above. Each table has a `condition` column
# (baseline / mitigated) so the before/after shift is visible turn by
# turn, not just in the session-level averages further up.

# %%
SNAPSHOT_TURNS = [t for t in (1, 5, 8) if t in turns]

snapshot_tables: dict[str, pd.DataFrame] = {}
for model in present_models:
    model_rows = []
    for condition in ("baseline", "mitigated"):
        stats = model_condition_stats[model][condition]
        for t in SNAPSHOT_TURNS:
            if t not in stats["mean_c"].index:
                continue
            model_rows.append({
                "target_model":       model,
                "failure_mode":       MODEL_FAILURE_MODE.get(model, "n/a"),
                "condition":          condition,
                "n":                  stats["n"],
                "turn":               t,
                "mean_correction":    stats["mean_c"].loc[t],
                "mean_rebuttal":      stats["mean_r"].loc[t],
                "mean_agreeableness": stats["mean_a"].loc[t],
                **{f"pct_{lbl}": stats["pct"].loc[t, lbl] for lbl in cfg.STANCE_ORDER},
            })
    snapshot_tables[model] = pd.DataFrame(model_rows)

    print(f"\n{'=' * 78}")
    print(f"{cfg.MODEL_LABELS.get(model, model)}  "
          f"(failure_mode={MODEL_FAILURE_MODE.get(model, 'n/a')})  "
          f"— snapshot at turns {SNAPSHOT_TURNS}")
    print("=" * 78)
    for row in snapshot_tables[model].itertuples(index=False):
        print(
            f"  [{row.condition:<9}] T{row.turn}: c={row.mean_correction:.2f} "
            f"r={row.mean_rebuttal:.2f} a={row.mean_agreeableness:.2f}  |  "
            f"False={row.pct_False:.1f}%  Uncertain={row.pct_Uncertain:.1f}%  True={row.pct_True:.1f}%"
        )

# %% [markdown]
# ## Save the three per-model snapshot tables

# %%
if RUN_DIR is not None:
    tables_dir = RUN_DIR / "tables" / SAMPLE_LABEL
    tables_dir.mkdir(parents=True, exist_ok=True)
    combined_snapshot = pd.concat(snapshot_tables.values(), ignore_index=True)
    out_path = tables_dir / "snapshot_by_model_baseline_vs_mitigated.csv"
    combined_snapshot.round(2).to_csv(out_path, index=False)
    print(f"Wrote combined snapshot table ({len(combined_snapshot)} rows) -> {out_path}")

# %% [markdown]
# ## Effect size 1: PERMANOVA R² on 8-turn trajectories (baseline vs. mitigated)
#
# Same method as `notebooks/fianl_experiment/final_experiment_analysis.ipynb`'s
# "Category significance analysis" cell (PERMANOVA on a precomputed
# Euclidean distance matrix, Anderson 2001) — `_ss_decomp` / `permanova`
# copied verbatim from there. Unit of analysis is the same too: one
# 8-dim vector per session ([s_t1, ..., s_t8] for a given metric),
# sessions with any missing turn dropped. The only change is what
# defines the groups being compared: there it was belief category
# within one target model; here it's `condition` (baseline vs.
# mitigated) within one target model — i.e. "does the whole 8-turn
# trajectory shape differ under the mitigation prompt?", not just the
# turn-1/5/8 snapshot means already printed above.
#
# R² is the fraction of total trajectory variance explained by
# condition (baseline vs. mitigated) — 0 = no shape difference, 1 =
# complete separation. p is a permutation p-value (999 label
# shuffles), not from a parametric distribution.

# %%
PERMANOVA_N_PERM = 999
PERMANOVA_SEED = 0
PERMANOVA_METRICS = ("correction", "rebuttal", "agreeableness")


def _ss_decomp(D2: np.ndarray, groups: np.ndarray) -> tuple[float, float, float]:
    n = D2.shape[0]
    SS_T = D2.sum() / (2 * n)
    SS_W = 0.0
    for u in np.unique(groups):
        idx = np.where(groups == u)[0]
        if len(idx) < 2:
            continue
        SS_W += D2[np.ix_(idx, idx)].sum() / (2 * len(idx))
    return SS_T, SS_T - SS_W, SS_W


def permanova(
    D: np.ndarray, groups: np.ndarray, n_perm: int = PERMANOVA_N_PERM, seed: int = PERMANOVA_SEED,
) -> tuple[float, float, float]:
    """Returns (pseudo_F, R2, permutation_p). NaNs if a group has <2 members."""
    groups = np.asarray(groups)
    n, a = len(groups), len(np.unique(groups))
    if a < 2 or n - a < 1:
        return np.nan, np.nan, np.nan

    D2 = D ** 2
    SS_T, SS_A, SS_W = _ss_decomp(D2, groups)
    F_obs = (SS_A / (a - 1)) / (SS_W / (n - a))
    R2 = SS_A / SS_T if SS_T > 0 else np.nan

    rng = np.random.default_rng(seed)
    perm = groups.copy()
    n_ge = 1
    for _ in range(n_perm):
        rng.shuffle(perm)
        _, SS_A_p, SS_W_p = _ss_decomp(D2, perm)
        F_p = (SS_A_p / (a - 1)) / (SS_W_p / (n - a))
        if F_p >= F_obs:
            n_ge += 1
    return F_obs, R2, n_ge / (n_perm + 1)


def _build_wide_trajectory(df: pd.DataFrame, metric: str, turn_range: list[int]):
    """One row per (session, condition), columns = turn 1..N values of `metric`."""
    wide = df.pivot_table(
        index=["source_session_id", "condition", "target_model"],
        columns="turn", values=metric, aggfunc="first",
    )
    turn_cols = [t for t in turn_range if t in wide.columns]
    wide = wide.reset_index()
    return wide.dropna(subset=turn_cols).copy(), turn_cols


def _fmt_p(p: float) -> str:
    return f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"


permanova_rows = []
for metric in PERMANOVA_METRICS:
    wide, turn_cols = _build_wide_trajectory(turn_df, metric, turns)

    for model_key, model_df in [("All models (pooled)", wide)] + list(wide.groupby("target_model")):
        groups = model_df["condition"].to_numpy()
        if len(np.unique(groups)) < 2 or len(model_df) < 4:
            continue
        X = model_df[turn_cols].to_numpy()
        D = squareform(pdist(X, metric="euclidean"))
        F, R2, p = permanova(D, groups)
        n_baseline = int((groups == "baseline").sum())
        n_mitigated = int((groups == "mitigated").sum())
        permanova_rows.append({
            "target_model": model_key,
            "metric": metric,
            "pseudo_F": F,
            "R2": R2,
            "p_value": p,
            "n_baseline": n_baseline,
            "n_mitigated": n_mitigated,
            "mean_level_baseline": float(X[groups == "baseline"].mean()),
            "mean_level_mitigated": float(X[groups == "mitigated"].mean()),
            "significant": p < 0.05,
        })

permanova_df = pd.DataFrame(permanova_rows)

print("=" * 90)
print(f"PERMANOVA: 8-turn trajectory shape, baseline vs. mitigated "
      f"({PERMANOVA_N_PERM} permutations)")
print("=" * 90)
for _, r in permanova_df.iterrows():
    star = "*" if r["significant"] else " "
    print(
        f"{star} {r['target_model']:<24s} {r['metric']:<14s} "
        f"F={r['pseudo_F']:6.2f}  R²={r['R2']:.3f}  p={_fmt_p(r['p_value'])}  "
        f"n_base={r['n_baseline']:<3d} n_mit={r['n_mitigated']:<3d}  "
        f"level {r['mean_level_baseline']:.2f} -> {r['mean_level_mitigated']:.2f}"
    )

if RUN_DIR is not None:
    out_path = tables_dir / "permanova_trajectory_baseline_vs_mitigated.csv"
    permanova_df.round(4).to_csv(out_path, index=False)
    print(f"\nWrote PERMANOVA table ({len(permanova_df)} rows) -> {out_path}")

# %% [markdown]
# ## Effect size 2: Cohen's h on % False stance (baseline vs. mitigated)
#
# `cohen_h` copied verbatim from the same notebook's break-rate-by-factor
# analysis (`|h| < 0.2 negligible · 0.2-0.5 small · 0.5-0.8 medium · >=0.8
# large`). Computed per target_model (+ an all-models-pooled row) at four
# granularities: pooled across all 8 turns, and at T1/T5/T8 individually
# (matching SNAPSHOT_TURNS above) — pooled gives the overall picture, the
# per-turn breakdown shows whether the effect is front-loaded, sustained,
# or only shows up late.

# %%
def cohen_h(p1: float, p2: float) -> float:
    """Effect size for two proportions. |h|<0.2 small, ~0.5 medium, >0.8 large."""
    return abs(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def _h_label(h: float) -> str:
    if h >= 0.8:
        return "large"
    if h >= 0.5:
        return "medium"
    if h >= 0.2:
        return "small"
    return "negligible"


def _pct_by_label(sub_df: pd.DataFrame) -> tuple[dict[str, float], int]:
    """Proportion of each stance label (False/Uncertain/True) over parsed
    turn-observations in sub_df, plus the shared n (== len(parsed) for all
    three, since they're proportions of the same denominator)."""
    parsed = sub_df[sub_df["stance_label"] != "unparsed"]
    if parsed.empty:
        return {lbl: float("nan") for lbl in cfg.STANCE_ORDER}, 0
    n = len(parsed)
    return {lbl: float((parsed["stance_label"] == lbl).mean()) for lbl in cfg.STANCE_ORDER}, n


stance_effect_rows = []
for model_key in ["All models (pooled)"] + present_models:
    model_turn_df = turn_df if model_key == "All models (pooled)" else turn_df[turn_df["target_model"] == model_key]

    granularities = [("pooled (all turns)", model_turn_df)]
    for t in SNAPSHOT_TURNS:
        granularities.append((f"T{t}", model_turn_df[model_turn_df["turn"] == t]))

    for label, gdf in granularities:
        base_pcts, n_base = _pct_by_label(gdf[gdf["condition"] == "baseline"])
        mit_pcts, n_mit = _pct_by_label(gdf[gdf["condition"] == "mitigated"])
        if any(np.isnan(v) for v in base_pcts.values()) or any(np.isnan(v) for v in mit_pcts.values()):
            continue
        row = {
            "target_model": model_key,
            "granularity": label,
            "n_baseline": n_base,
            "n_mitigated": n_mit,
        }
        for lbl in cfg.STANCE_ORDER:
            row[f"pct_{lbl}_baseline"] = base_pcts[lbl] * 100
            row[f"pct_{lbl}_mitigated"] = mit_pcts[lbl] * 100
            row[f"diff_pct_{lbl}"] = (mit_pcts[lbl] - base_pcts[lbl]) * 100
        row["cohen_h_False"] = cohen_h(base_pcts["False"], mit_pcts["False"])
        row["h_label_False"] = _h_label(row["cohen_h_False"])
        stance_effect_rows.append(row)

stance_effect_df = pd.DataFrame(stance_effect_rows)

print("\n" + "=" * 90)
print("Stance label shift: % False / Uncertain / True, baseline vs. mitigated")
print("  (n = turn-observations pooled across sessions, NOT session count — see")
print("   note above the raw-diff summary below for why this differs from the")
print("   correction/rebuttal/agreeableness N)")
print("  Cohen's h (False only): |h| < 0.2 negligible · 0.2-0.5 small · 0.5-0.8 medium · >=0.8 large")
print("=" * 90)
for model_key in ["All models (pooled)"] + present_models:
    sub = stance_effect_df[stance_effect_df["target_model"] == model_key]
    if sub.empty:
        continue
    print(f"\n[{cfg.MODEL_LABELS.get(model_key, model_key)}]")
    for _, r in sub.iterrows():
        print(
            f"  {r['granularity']:<20s} "
            f"False: {r['pct_False_baseline']:5.1f}%->{r['pct_False_mitigated']:5.1f}% "
            f"({r['diff_pct_False']:+.1f}pp, h={r['cohen_h_False']:.2f} {r['h_label_False']})  "
            f"Uncertain: {r['pct_Uncertain_baseline']:5.1f}%->{r['pct_Uncertain_mitigated']:5.1f}% "
            f"({r['diff_pct_Uncertain']:+.1f}pp)  "
            f"True: {r['pct_True_baseline']:5.1f}%->{r['pct_True_mitigated']:5.1f}% "
            f"({r['diff_pct_True']:+.1f}pp)  "
            f"n_base={r['n_baseline']:<3d} n_mit={r['n_mitigated']:<3d}"
        )

if RUN_DIR is not None:
    out_path = tables_dir / "cohens_h_stance_false_baseline_vs_mitigated.csv"
    stance_effect_df.round(3).to_csv(out_path, index=False)
    print(f"\nWrote stance shift table ({len(stance_effect_df)} rows) -> {out_path}")

# %% [markdown]
# ## Summary: raw mean differences (mitigated - baseline), no significance gate
#
# PERMANOVA/Cohen's h test whether the mitigation prompt reliably shifts the
# *distribution* — useful for ruling out noise, but with an underpowered
# sample (see the misinfo_v1_split coverage gap flagged above — several
# models are currently well under their full N) a real, consistent shift
# can fail to clear a significance threshold without being fake. This
# table reports the plain mean difference per model with NO significance
# gate, so a small-but-real effect isn't hidden behind "n.s.".
#
# Computed the same way as the stance diffs above: pair each session's
# baseline and mitigated value AT EACH TURN individually (turn_df
# already has both), take the per-turn-observation difference, THEN
# average across all turn-observations — not "average each session's 8
# turns into one number, then diff the two session-level means" (what
# an earlier version of this table did). The two aren't numerically
# identical whenever a turn is missing on one side but not the other
# (a -1.0 parse failure), and this version's N is now directly
# comparable to `n_stance` — both count turn-observations, not
# sessions, so e.g. 70 fully-scored sessions -> n=560 on every column.

# %%
_wide_turn = turn_df.pivot_table(
    index=["source_session_id", "turn", "target_model"],
    columns="condition", values=["correction", "rebuttal", "agreeableness"],
)
_wide_turn.columns = [f"{metric}_{condition}" for metric, condition in _wide_turn.columns]
_wide_turn = _wide_turn.reset_index()
for dim in ("correction", "rebuttal", "agreeableness"):
    b, m = f"{dim}_baseline", f"{dim}_mitigated"
    if b in _wide_turn.columns and m in _wide_turn.columns:
        _wide_turn[f"diff_{dim}"] = _wide_turn[m] - _wide_turn[b]

summary_rows = []
for model_key in ["All models (pooled)"] + present_models:
    d = _wide_turn if model_key == "All models (pooled)" else _wide_turn[_wide_turn["target_model"] == model_key]
    stance_row = stance_effect_df[
        (stance_effect_df["target_model"] == model_key)
        & (stance_effect_df["granularity"] == "pooled (all turns)")
    ]
    row = {"target_model": model_key}
    for dim in ("correction", "rebuttal", "agreeableness"):
        col = f"diff_{dim}"
        row[f"n_{dim}"] = int(d[col].count()) if col in d else 0
        row[f"diff_{dim}"] = d[col].mean() if col in d else float("nan")
    if not stance_row.empty:
        row["n_stance"] = int(stance_row["n_baseline"].iloc[0])
        for lbl in cfg.STANCE_ORDER:
            row[f"diff_pct_{lbl}"] = float(stance_row[f"diff_pct_{lbl}"].iloc[0])
    else:
        row["n_stance"] = 0
        for lbl in cfg.STANCE_ORDER:
            row[f"diff_pct_{lbl}"] = float("nan")
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

print("\n" + "=" * 90)
print("Raw mean differences (mitigated - baseline), no significance gate")
print("Per-turn paired diff, averaged across turn-observations (n matches n_stance)")
print("=" * 90)
for _, r in summary_df.iterrows():
    print(
        f"  {cfg.MODEL_LABELS.get(r['target_model'], r['target_model']):<20s}  "
        f"correction {r['diff_correction']:+.2f} (n={r['n_correction']:.0f})   "
        f"rebuttal {r['diff_rebuttal']:+.2f} (n={r['n_rebuttal']:.0f})   "
        f"agreeableness {r['diff_agreeableness']:+.2f} (n={r['n_agreeableness']:.0f})   "
        f"%False {r['diff_pct_False']:+.1f}pp  %Uncertain {r['diff_pct_Uncertain']:+.1f}pp  "
        f"%True {r['diff_pct_True']:+.1f}pp  (n_stance={r['n_stance']:.0f})"
    )

if RUN_DIR is not None:
    out_path = tables_dir / "raw_diff_summary_baseline_vs_mitigated.csv"
    summary_df.round(3).to_csv(out_path, index=False)
    print(f"\nWrote raw diff summary ({len(summary_df)} rows) -> {out_path}")
