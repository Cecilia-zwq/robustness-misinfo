"""
evaluator_validation/analyze_agreement.py
=========================================
RQ3 — Compute inter-evaluator agreement on the validated subset.

Reads the sample manifest written by run_validation_scoring.py, then for
every sampled session loads two score files::

    primary    : <sid>__<rubric>.json
    secondary  : <sid>__<rubric>__<secondary-slug>.json

Pairs the per-turn scores and computes, for each rubric dimension
(correction / rebuttal / agreeableness):

  - Quadratic-weighted Cohen's κ          (headline metric)
  - Spearman rank correlation
  - Pearson correlation
  - Exact-agreement proportion
  - Within-1 proportion (≥ 1-band agreement on the 1–3 scale)
  - Mean signed difference   (primary − secondary; bias check)
  - Mean absolute difference
  - 3 × 3 confusion matrix

The same metrics are also broken out by ``target_model``, ``iv1``, and
``category`` to test whether agreement is uniform across the design
factors that drive RQ1/RQ2 conclusions.

A trajectory plot overlays the two evaluators' mean scores per turn —
this is the "do the headline RQ1/RQ2 patterns survive evaluator
substitution?" check, in figure form.

Outputs land under ``<run_dir>/evaluator_validation/analysis_output/``::

    tables/
      ├── agreement_overall.csv
      ├── agreement_by_target_model.csv
      ├── agreement_by_iv1.csv
      ├── agreement_by_category.csv
      ├── confusion_matrices.csv
      └── per_turn_diff.csv          # raw paired turn-level scores
    plots/
      ├── fig_confusion_matrices.png
      ├── fig_kappa_by_subgroup.png
      ├── fig_score_distributions.png
      ├── fig_bias_by_target_model.png
      └── fig_trajectory_evaluator_overlay.png
    agreement_log.txt                 # mirrored stdout for the paper

Usage
-----
::

    cd scripts/final_experiment
    python -m evaluator_validation.analyze_agreement \\
        --run-dir results/final_experiment/main_user_IVs/<timestamp>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import RunPaths, safe_slug  # noqa: E402

from . import config as cfg  # noqa: E402

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Visual conventions — matches main_user_IVs/final_experiment_analysis.py
# so figures from this script and the headline analysis script can sit
# side-by-side in the paper without restyling.
# ════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.facecolor": "white",
})

IV1_ORDER = ["none", "warm", "emotional", "hostile", "logical"]
IV1_LABEL = {
    "none": "Control", "warm": "Warm", "emotional": "Emotional",
    "hostile": "Hostile", "logical": "Logical",
}
IV1_PALETTE = {
    "Control": "#7f7f7f", "Warm": "#e07b54", "Emotional": "#00c9ac",
    "Hostile": "#4F2683", "Logical": "#4a90d9",
}

SCORE_DIMS = ("correction", "rebuttal", "agreeableness")
SCORE_RANGE = (1, 3)
SCORE_LABELS = list(range(SCORE_RANGE[0], SCORE_RANGE[1] + 1))

# Evaluator colour pair — used in the trajectory overlay and the
# distribution plot. Keep these consistent across figures.
PRIMARY_COLOUR = "#4F2683"
SECONDARY_COLOUR = "#00c9ac"


# ════════════════════════════════════════════════════════════════════════════
# Helpers — mirrors of stuff in final_experiment_analysis.py
# ════════════════════════════════════════════════════════════════════════════

def _short_model(model_str: str) -> str:
    """``openrouter/openai/gpt-4.1-mini`` → ``gpt-4.1-mini``."""
    return model_str.split("/")[-1]


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


class _Tee:
    """Mirror stdout to a log file. Lifted from the headline analysis."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


# ════════════════════════════════════════════════════════════════════════════
# Load paired scores
# ════════════════════════════════════════════════════════════════════════════

def _load_paired_scores(
    *,
    paths: RunPaths,
    sampled_ids: list[str],
    rubric_name: str,
    secondary_slug: str,
    primary_slug: str | None,
) -> pd.DataFrame:
    """Long-format DF: one row per (session, turn, dimension).

    Columns: session_id, turn, dimension, score_primary, score_secondary,
             cell_id, iv1, iv2, target_model, category, subtype, is_long_text.

    Rows where either score is -1.0 (parse failure) are dropped — they
    can't contribute to agreement metrics. The drop-count is reported so
    the eventual paper can quote it.
    """
    primary_suffix = (
        f"{rubric_name}__{primary_slug}" if primary_slug else rubric_name
    )
    secondary_suffix = f"{rubric_name}__{secondary_slug}"

    rows: list[dict] = []
    n_skipped_missing = 0
    n_dropped_parse_fail = 0

    for sid in sampled_ids:
        primary_path = paths.score_path(sid, primary_suffix)
        secondary_path = paths.score_path(sid, secondary_suffix)
        if not primary_path.exists() or not secondary_path.exists():
            n_skipped_missing += 1
            continue

        with open(primary_path, "r", encoding="utf-8") as f:
            primary = json.load(f)
        with open(secondary_path, "r", encoding="utf-8") as f:
            secondary = json.load(f)

        # Pull subgroup metadata from the conversation artifact.
        conv_path = paths.conversation_path(sid)
        if not conv_path.exists():
            n_skipped_missing += 1
            continue
        with open(conv_path, "r", encoding="utf-8") as f:
            conv = json.load(f)
        belief = conv.get("belief", {})
        cell = conv.get("cell", {})
        meta = {
            "session_id": sid,
            "cell_id": cell.get("cell_id", ""),
            "iv1": cell.get("iv1", ""),
            "iv2": cell.get("iv2", ""),
            "target_model": _short_model(conv.get("models", {}).get("target_llm", "")),
            "category": belief.get("category", ""),
            "subtype": _coerce_subtype(belief.get("subtype", "")),
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
                rows.append({
                    **meta,
                    "turn": turn,
                    "dimension": dim,
                    "score_primary": float(p_val),
                    "score_secondary": float(s_val),
                })

    if n_skipped_missing:
        logger.warning(
            "Skipped %d session(s) missing one of: primary score, "
            "secondary score, or conversation artifact.",
            n_skipped_missing,
        )
    if n_dropped_parse_fail:
        logger.warning(
            "Dropped %d (turn, dimension) pair(s) with parse-failure scores.",
            n_dropped_parse_fail,
        )

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Agreement metrics
# ════════════════════════════════════════════════════════════════════════════

def _agreement_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """All agreement stats for one column of paired ratings."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = (~np.isnan(a)) & (~np.isnan(b))
    a = a[mask]
    b = b[mask]
    n = int(len(a))
    blank = {
        "n": n, "kappa_quadratic": np.nan,
        "spearman_rho": np.nan, "spearman_p": np.nan,
        "pearson_r": np.nan, "pearson_p": np.nan,
        "exact_agreement": np.nan, "within_1": np.nan,
        "mean_signed_diff": np.nan, "mean_abs_diff": np.nan,
    }
    if n < 5:
        return blank

    a_int, b_int = a.astype(int), b.astype(int)

    try:
        kappa = cohen_kappa_score(
            a_int, b_int, weights="quadratic", labels=SCORE_LABELS,
        )
    except (ValueError, TypeError):
        kappa = np.nan

    # SciPy raises if all values are constant (no variance). Trap and
    # return NaN — this happens occasionally in tiny subgroups.
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


def _agreement_table(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Apply ``_agreement_metrics`` per (group_cols, dimension) cell."""
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


def _confusion_matrix_table(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format confusion-matrix CSV for downstream tools."""
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


# ════════════════════════════════════════════════════════════════════════════
# Plots
# ════════════════════════════════════════════════════════════════════════════

def _plot_confusion_matrices(df: pd.DataFrame, out_dir: Path) -> None:
    """One 3 × 3 confusion matrix per dimension. Counts annotated, colour
    is row-normalised so the diagonal isn't drowned out by class imbalance.
    """
    fig, axes = plt.subplots(
        1, len(SCORE_DIMS), figsize=(5 * len(SCORE_DIMS), 4.4),
    )
    if len(SCORE_DIMS) == 1:
        axes = [axes]
    for ax, dim in zip(axes, SCORE_DIMS):
        d = df[df["dimension"] == dim]
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
    plt.savefig(out_dir / "fig_confusion_matrices.png")
    plt.close()


def _plot_kappa_by_subgroup(
    by_target: pd.DataFrame,
    by_iv1: pd.DataFrame,
    by_category: pd.DataFrame,
    out_dir: Path,
) -> None:
    """3 panels of grouped bars: κ by target_model / iv1 / category."""
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
        # Preserve the canonical dimension column order.
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
    fig.suptitle(
        "Inter-evaluator agreement (κ) across subgroups",
        fontsize=13, y=1.04,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "fig_kappa_by_subgroup.png")
    plt.close()


def _plot_score_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    """Side-by-side bars of the score-value distribution per evaluator.

    A divergent shape (e.g. secondary skews to 1 while primary is mostly
    2) is a more interpretable diagnostic than κ alone for spotting
    calibration differences.
    """
    fig, axes = plt.subplots(
        1, len(SCORE_DIMS),
        figsize=(5 * len(SCORE_DIMS), 4.0),
        sharey=True,
    )
    if len(SCORE_DIMS) == 1:
        axes = [axes]
    for ax, dim in zip(axes, SCORE_DIMS):
        d = df[df["dimension"] == dim]
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
    fig.suptitle(
        "Score-value distribution by evaluator (turn-level)",
        fontsize=13, y=1.04,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "fig_score_distributions.png")
    plt.close()


def _plot_bias_by_target_model(df: pd.DataFrame, out_dir: Path) -> None:
    """Mean signed difference (primary − secondary) per target × dim.

    Non-zero ⇒ systematic offset. The interesting case is when the offset
    correlates with the target model itself — that's a self-preference
    signal (e.g. gpt-4.1-mini scoring gpt-5.3-chat outputs more
    leniently than gemini does). Flagged in the results section.
    """
    df = df.copy()
    df["signed_diff"] = df["score_primary"] - df["score_secondary"]
    pivot = (
        df.groupby(["target_model", "dimension"])["signed_diff"]
          .mean()
          .unstack()
          .reindex(columns=list(SCORE_DIMS))
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 0.6 * max(len(pivot), 1) + 1.8))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, ax=ax, linewidths=0.4,
        cbar_kws={"label": "mean(primary − secondary)"},
    )
    ax.set_title(
        "Per-target-model evaluator bias\n"
        "(non-zero ⇒ systematic offset; "
        "model-correlated ⇒ self-preference signal)"
    )
    ax.set_xlabel(""); ax.set_ylabel("Target model")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_bias_by_target_model.png")
    plt.close()


def _plot_trajectory_overlay(df: pd.DataFrame, out_dir: Path) -> None:
    """Mean score per turn, both evaluators overlaid, one panel per dim.

    This is the "do the headline RQ1/RQ2 patterns survive evaluator
    substitution?" check. If the two trajectories track each other, the
    framework's qualitative claims are evaluator-robust on this sample.
    """
    fig, axes = plt.subplots(
        1, len(SCORE_DIMS),
        figsize=(5 * len(SCORE_DIMS), 4.4),
        sharey=True,
    )
    if len(SCORE_DIMS) == 1:
        axes = [axes]
    for ax, dim in zip(axes, SCORE_DIMS):
        d = df[df["dimension"] == dim]
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
    plt.savefig(out_dir / "fig_trajectory_evaluator_overlay.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def _section(title: str, char: str = "═") -> None:
    bar = char * 78
    print(f"\n{bar}\n{title}\n{bar}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--secondary-provider", type=str,
        default=cfg.SECONDARY_EVALUATOR[0],
    )
    p.add_argument(
        "--secondary-model", type=str,
        default=cfg.SECONDARY_EVALUATOR[1],
    )
    p.add_argument("--rubric", default=cfg.RUBRIC_NAME)
    p.add_argument(
        "--primary-slug", default=None,
        help="If primary scores were written with --suffix-evaluator, pass "
             "that slug here (e.g. 'openrouter-openai-gpt-4.1-mini'). "
             "Default assumes the primary file is unsuffixed.",
    )
    args = p.parse_args()

    if not args.run_dir.exists():
        p.error(f"--run-dir does not exist: {args.run_dir}")

    paths = RunPaths(root=args.run_dir)
    validation_dir = paths.root / cfg.VALIDATION_SUBDIR
    if not validation_dir.exists():
        p.error(
            f"Validation directory not found: {validation_dir}. "
            "Run run_validation_scoring.py first."
        )

    manifest_path = validation_dir / cfg.SAMPLE_MANIFEST_NAME
    if not manifest_path.exists():
        p.error(f"Sample manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    sampled_ids = list(manifest["sampled_session_ids"])

    secondary_slug = safe_slug(f"{args.secondary_provider}/{args.secondary_model}")

    out_dir = validation_dir / "analysis_output"
    table_dir = out_dir / "tables"
    plot_dir = out_dir / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "agreement_log.txt"
    log_fp = open(log_path, "w")
    sys.stdout = _Tee(sys.__stdout__, log_fp)

    print(f"Run dir         : {paths.root}")
    print(f"Sample size     : {len(sampled_ids)} sessions "
          f"(seed={manifest.get('seed')}, "
          f"fraction={manifest.get('fraction')})")
    print(f"Rubric          : {args.rubric}")
    print(f"Primary slug    : {args.primary_slug or '(unsuffixed)'}")
    print(f"Secondary slug  : {secondary_slug}")
    print(f"Output dir      : {out_dir}")

    df = _load_paired_scores(
        paths=paths,
        sampled_ids=sampled_ids,
        rubric_name=args.rubric,
        secondary_slug=secondary_slug,
        primary_slug=args.primary_slug,
    )
    if df.empty:
        print("\nNo paired scores found; nothing to analyze.")
        sys.stdout = sys.__stdout__
        log_fp.close()
        return

    print(
        f"\nLoaded {len(df)} paired (turn × dimension) rows from "
        f"{df['session_id'].nunique()} session(s)."
    )
    df.to_csv(table_dir / "per_turn_diff.csv", index=False)

    # ── Overall ─────────────────────────────────────────────────────────
    _section("OVERALL AGREEMENT")
    overall = _agreement_table(df, group_cols=None)
    overall.to_csv(table_dir / "agreement_overall.csv", index=False)
    print(overall.round(3).to_string(index=False))

    # ── By target model ─────────────────────────────────────────────────
    _section("AGREEMENT BY TARGET MODEL  (self-preference check)")
    by_target = _agreement_table(df, ["target_model"])
    by_target.to_csv(
        table_dir / "agreement_by_target_model.csv", index=False,
    )
    print(by_target.round(3).to_string(index=False))

    # ── By IV1 ──────────────────────────────────────────────────────────
    _section("AGREEMENT BY IV1 PERSONA  (hard-cases check)")
    by_iv1 = _agreement_table(df, ["iv1"])
    by_iv1.to_csv(table_dir / "agreement_by_iv1.csv", index=False)
    print(by_iv1.round(3).to_string(index=False))

    # ── By belief category ──────────────────────────────────────────────
    _section("AGREEMENT BY BELIEF CATEGORY")
    by_category = _agreement_table(df, ["category"])
    by_category.to_csv(
        table_dir / "agreement_by_category.csv", index=False,
    )
    print(by_category.round(3).to_string(index=False))

    # ── Confusion matrices (long format) ────────────────────────────────
    cm_long = _confusion_matrix_table(df)
    cm_long.to_csv(table_dir / "confusion_matrices.csv", index=False)

    # ── Plots ───────────────────────────────────────────────────────────
    _section("WRITING PLOTS")
    _plot_confusion_matrices(df, plot_dir)
    print("  ✓ fig_confusion_matrices.png")
    _plot_kappa_by_subgroup(by_target, by_iv1, by_category, plot_dir)
    print("  ✓ fig_kappa_by_subgroup.png")
    _plot_score_distributions(df, plot_dir)
    print("  ✓ fig_score_distributions.png")
    _plot_bias_by_target_model(df, plot_dir)
    print("  ✓ fig_bias_by_target_model.png")
    _plot_trajectory_overlay(df, plot_dir)
    print("  ✓ fig_trajectory_evaluator_overlay.png")

    # ── Headline summary block ──────────────────────────────────────────
    _section("HEADLINE NUMBERS  (paste-ready for the paper)")
    headline_lines = []
    for _, row in overall.iterrows():
        headline_lines.append(
            f"  {row['dimension']:<14}  "
            f"n={int(row['n']):>5}  "
            f"κ_q={row['kappa_quadratic']:.3f} ({row['kappa_label']})  "
            f"ρ={row['spearman_rho']:.3f}  "
            f"exact={row['exact_agreement']:.3f}  "
            f"within-1={row['within_1']:.3f}  "
            f"bias={row['mean_signed_diff']:+.3f}"
        )
    print("\n".join(headline_lines))
    print(
        f"\nTables : {table_dir}\n"
        f"Plots  : {plot_dir}\n"
        f"Log    : {log_path}\n"
    )

    sys.stdout = sys.__stdout__
    log_fp.close()
    print(f"Done. Outputs under {out_dir}/")


if __name__ == "__main__":
    main()