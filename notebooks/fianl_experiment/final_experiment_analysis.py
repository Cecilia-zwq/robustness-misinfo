# %% [markdown]
# # Package and configuration

# %%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import json
from scipy import stats
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.multitest import multipletests
import seaborn as sns


# %%
# ── Configuration ───────────────────────────────────────────────────────────

# (column, display title, direction note)
SCORE_DIMS = ["correction", "rebuttal", "agreeableness"]
metrics = [
    ('correction',    'Correction Quality',  ),
    ('rebuttal',      'Rebuttal Quality', ),
    ('agreeableness', 'Agreeableness',),
]

# Path
DATA_DIR = Path('/home/wzhan969/robustness-misinfo/results/final_experiment/main_user_IVs/20260427_165233')
OUT_DIR = Path('/home/wzhan969/robustness-misinfo/results/final_experiment/main_user_IVs/20260427_165233/analysis_output')
OUT_DIR.mkdir(exist_ok=True)

# Brand-inspired but tuned for contrast on a white background
model_palette = {
    'claude-sonnet-4.6':      '#CC785C',  # Anthropic clay
    'gpt-5.3-chat':           '#10A37F',  # OpenAI teal
    'gemini-3-flash-preview': '#4285F4',  # Google blue
    'deepseek-v3.2':          '#7C3AED',  # Violet
}

model_labels = {
    'claude-sonnet-4.6':      'Claude Sonnet 4.6',
    'gpt-5.3-chat':           'GPT-5.3',
    'gemini-3-flash-preview': 'Gemini 3 Flash',
    'deepseek-v3.2':          'DeepSeek V3.2',
}

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    # Add these to fix text color
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.edgecolor': 'black',
    # Keep your other settings
    'figure.dpi': 150, 
    'savefig.dpi': 150, 
    'savefig.bbox': 'tight',
    'font.size': 11, 
    'axes.titlesize': 13, 
    'axes.labelsize': 12,
})

# %% [markdown]
# # Load data

# %%
# # ── Load and filter ─────────────────────────────────────────────────────────
# tl = pd.read_csv( DATA_DIR / 'turn_level.csv')

# # Drop control sessions (no persona) — only adversarial users
# tl_control = tl[tl['iv1'] == 'none'].copy()

# %%
def _short_model(model_str: str) -> str:
    """Normalize model names for plotting keys."""
    short = model_str.split('/')[-1]
    if short == 'gemini-3-flash':
        return 'gemini-3-flash-preview'
    return short


def _parse_cell_id(session_id: str) -> dict[str, str]:
    """Extract iv1/iv2 from session ids like `cell-iv1-none__iv2-none__...`."""
    cell_id = session_id.split('__belief-', 1)[0]
    parts: dict[str, str] = {}
    for part in cell_id.removeprefix('cell-').split('__'):
        key, value = part.split('-', 1)
        parts[key] = value
    return parts


score_dir = DATA_DIR / 'scores'
score_paths = sorted(score_dir.glob('*__misinfo_v1_split.json'))
print(f'Found {len(score_paths)} misinfo_v1_split score files under {score_dir}')

summary = pd.read_csv(DATA_DIR / 'summary.csv')

turn_rows: list[dict] = []
for sp in score_paths:
    sc = json.loads(sp.read_text())
    session_id = sc['session_id']
    cell_meta = _parse_cell_id(session_id)
    mask = summary['session_id'] == session_id
    category_val, subtype_val = summary.loc[mask, ['category', 'subtype']].iloc[0].tolist()
    for turn_data in sc.get('turns', []):
        scores = turn_data.get('scores', {})
        turn_rows.append({
            'session_id': session_id,
            'cell_id': session_id.split('__belief-', 1)[0],
            'belief_id': session_id.split('belief-')[1].split('__model')[0],
            'iv1': cell_meta.get('iv1', ''),
            'iv2': cell_meta.get('iv2', ''),
            'is_control': cell_meta.get('iv1') == 'none' and cell_meta.get('iv2') == 'none',
            'target_model': _short_model(session_id.rsplit('__model-', 1)[-1]),
            'rubric_name': sc.get('rubric_name', 'misinfo_v1_split'),
            'turn': turn_data['turn'],
            'correction': scores.get('correction', np.nan),
            'rebuttal': scores.get('rebuttal', np.nan),
            'agreeableness': scores.get('agreeableness', np.nan),
            'category': category_val,
            'subtype': subtype_val
        })

tl = pd.DataFrame(turn_rows)
print(f'Loaded {len(tl)} turn rows')

tl = tl.loc[
    ~tl[['correction', 'rebuttal', 'agreeableness']].eq(-1).any(axis=1)
].copy()
print(f'Loaded {len(tl)} turn rows after dropping rows with any -1 score')

tl_control = tl[tl['is_control']]

# %%
display(tl_control.head(2))
display(tl.head(2))

# %% [markdown]
# # Model comparison (control group)
# 
# ## 3 scores

# %%
# ── Plot: score trajectory by turn × model (1×3 panel) ─────────────────────
turns_present = sorted(tl_control['turn'].unique())

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, (metric, title) in zip(axes, metrics):
    for model_key, label in model_labels.items():
        subset = tl_control[tl_control['target_model'] == model_key]
        if subset.empty:
            continue

        grouped = subset.groupby('turn')[metric]
        means = grouped.mean()
        sems  = grouped.sem()
        color = model_palette[model_key]

        ax.plot(means.index, means.values, 'o-', label=label,
                color=color, linewidth=2, markersize=6)
        ax.fill_between(means.index,
                        means - 1.96 * sems,
                        means + 1.96 * sems,
                        alpha=0.15, color=color)

    ax.set_title(f'{title}')
    ax.set_xticks(turns_present)
    ax.set_ylim(1, 3.0)
    ax.grid(True, alpha=0.3)

# Shared axis labels
fig.supxlabel('Turn')
fig.supylabel('Mean Score', x=0.01)

# Shared legend — pull handles from the first axis
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Target Model',
           loc='lower center', ncol=len(model_labels),
           frameon=True, bbox_to_anchor=(0.5, 0.94))

fig.suptitle('', fontsize=14, y=1.09)
plt.tight_layout()
plt.savefig(OUT_DIR/ 'plots' / 'fig_scores_by_turn_model.png',
            dpi=200, bbox_inches='tight')
plt.show()

# %% [markdown]
# - GPT-5.3 demonstrates the greatest stability. Its three metrics remain consistent across the interaction, yielding the highest overall correction and rebuttal quality while maintaining relatively low agreeableness.
# - Claude Sonnet 4.6 exhibits a sharp decline in correction and rebuttal quality starting around the fourth turn. Interestingly, despite this failure to actively correct or rebut the user's misinformed beliefs, the model also displays extremely low agreeableness, with scores tending to converge toward 1. A qualitative review of the conversation logs reveals the reason for this pattern: as the dialogue progresses, the model simply shuts down the discussion with responses such as, "I've said what I have to say on this topic clearly and repeatedly. I'm not going to keep responding to the same argument. Is there something else I can help you with?"
# - Gemini 3 Flash and DeepSeek V3.2, conversely, display similar behaviour. As the conversation advances, their correction and rebuttal quality steadily decreases while their agreeableness continuously climbs. This pattern clearly illustrates the exact sycophantic behavior our framework is designed to measure.

# %%
# # ── Per-row composite sycophancy score ─────────────────────────────────────
# tl_control['robustness'] = (tl_control['correction'] + tl_control['rebuttal']) / ( tl_control['agreeableness'] * 2)

# # ── Plot: sycophancy trajectory by turn × model ─────────────────────────────
# turns_present = sorted(tl_control['turn'].unique())

# fig, ax = plt.subplots(figsize=(7, 5))

# for model_key, label in model_labels.items():
#     subset = tl_control[tl_control['target_model'] == model_key]
#     if subset.empty:
#         continue

#     grouped = subset.groupby('turn')['robustness']
#     means = grouped.mean()
#     sems  = grouped.sem()
#     color = model_palette[model_key]

#     ax.plot(means.index, means.values, 'o-', label=label,
#             color=color, linewidth=2, markersize=6)
#     ax.fill_between(means.index,
#                     means - 1.96 * sems,
#                     means + 1.96 * sems,
#                     alpha=0.15, color=color)

# ax.set_xlabel('Turn')
# ax.set_ylabel('Mean Robustness Score')
# # ax.set_title('Robustness Severity Over Turns')
# ax.set_xticks(turns_present)
# ax.grid(True, alpha=0.3)

# ax.legend(handles=[plt.Line2D([0], [0], color=model_palette[k],
#                               marker='o', linewidth=2, markersize=6,
#                               label=lbl)
#                    for k, lbl in model_labels.items()],
#           title='Target Model',
#           loc='lower center', ncol=len(model_labels),
#           frameon=True, bbox_to_anchor=(0.5, 1.00))

# plt.tight_layout()
# # plt.savefig(OUT_DIR / 'fig_robustness_by_model.png',
# #             dpi=200, bbox_inches='tight')
# plt.show()


# %% [markdown]
# ## Bias type comparison

# %%
category_values = list(tl_control['category'].unique())
category_values

# %%
# ── Plot: score trajectory by turn × category, one figure per model ─────────
category_palette = dict(zip(category_values, plt.cm.tab20.colors[:len(category_values)]))

for model_key, model_title in model_labels.items():
    model_data = tl_control[tl_control['target_model'] == model_key]
    if model_data.empty:
        continue

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, (metric, title) in zip(axes, metrics):
        for category in category_values:
            subset = model_data[model_data['category'] == category]
            if subset.empty:
                continue

            grouped = subset.groupby('turn')[metric]
            means = grouped.mean()
            sems = grouped.sem().fillna(0)
            color = category_palette[category]

            ax.plot(means.index, means.values, 'o-', label=category,
                    color=color, linewidth=2, markersize=6)
            ax.fill_between(means.index,
                            means - 1.96 * sems,
                            means + 1.96 * sems,
                            alpha=0.15, color=color)

        ax.set_title(title)
        ax.set_xticks(turns_present)
        ax.set_ylim(1, 3.0)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Turn')

    axes[0].set_ylabel('Mean Score')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Category',
               loc='upper center', ncol=min(len(category_values), 4),
               frameon=True, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(f'{model_title} Trajectories by Category', fontsize=14, y=1.08)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

# %% [markdown]
# ## Category significance analysis (multivariate, per-turn vectors)
# 
# Unit of analysis: per-session 8-dim trajectory vector. For each session × metric, build [s_t1, s_t2, ..., s_t8]. Drop sessions missing any turn (after the -1 filter). 
# 
# Test choice — PERMANOVA (Anderson 2001). It's the natural multivariate non-parametric analog of Kruskal-Wallis
# 
# Two-stage testing per (model × metric):
# 
# 1. Omnibus PERMANOVA across all categories. Report pseudo-F, R², permutation p, n.
# 2. Pairwise PERMANOVA for category pairs, with Holm-Bonferroni correction within each (model × metric) family. Report per-pair pseudo-F, R², raw and Holm-adjusted p, plus the per-group mean trajectory level for direction interpretation.

# %%
METRICS = ['correction', 'rebuttal', 'agreeableness']
ALPHA = 0.05
N_PERM = 1999
RNG_SEED = 0
EXPECTED_TURNS = sorted(tl_control['turn'].unique())

# ── PERMANOVA (Anderson 2001) on a precomputed Euclidean distance matrix ──────
def _ss_decomp(D2, groups):
    n = D2.shape[0]
    SS_T = D2.sum() / (2 * n)
    SS_W = 0.0
    for u in np.unique(groups):
        idx = np.where(groups == u)[0]
        if len(idx) < 2:
            continue
        SS_W += D2[np.ix_(idx, idx)].sum() / (2 * len(idx))
    return SS_T, SS_T - SS_W, SS_W

def permanova(D, groups, n_perm=N_PERM, seed=RNG_SEED):
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

# ── Build per-metric session vectors (one row = one session, 8 turn columns) ──
def build_wide(df, metric, turns):
    wide = df.pivot_table(
        index=['session_id', 'target_model', 'category'],
        columns='turn', values=metric, aggfunc='first'
    )
    turn_cols = [t for t in turns if t in wide.columns]
    wide = wide.reset_index()
    full = wide.dropna(subset=turn_cols).copy()
    return full, turn_cols

# ── Run omnibus + pairwise PERMANOVA per (model × metric) ─────────────────────
omnibus_rows, pairwise_rows = [], []

for metric in METRICS:
    wide, turn_cols = build_wide(tl_control, metric, EXPECTED_TURNS)
    n_dropped = (
        tl_control[['session_id', 'target_model', 'category']]
        .drop_duplicates().shape[0] - wide.shape[0]
    )
    print(f'[{metric}] {wide.shape[0]} complete {len(turn_cols)}-turn vectors '
          f'({n_dropped} sessions dropped for incomplete trajectories)')

    for model_key, model_df in wide.groupby('target_model'):
        cats = sorted(model_df['category'].dropna().unique())
        if len(cats) < 2 or len(model_df) < 4:
            continue

        X = model_df[turn_cols].to_numpy()
        groups = model_df['category'].to_numpy()
        D = squareform(pdist(X, metric='euclidean'))

        F, R2, p = permanova(D, groups)
        omnibus_rows.append({
            'target_model': model_key, 'metric': metric,
            'pseudo_F': F, 'R2': R2, 'p_value': p,
            'n_sessions': len(groups), 'n_categories': len(cats),
            'significant': p < ALPHA,
        })

        # Pairwise
        comps, pvals = [], []
        for c1, c2 in combinations(cats, 2):
            mask = np.isin(groups, [c1, c2])
            X_p, g_p = X[mask], groups[mask]
            if len(np.unique(g_p)) < 2:
                continue
            D_p = squareform(pdist(X_p, metric='euclidean'))
            F_p, R2_p, p_raw = permanova(D_p, g_p, seed=RNG_SEED + hash((c1, c2)) % 10_000)

            comps.append({
                'target_model': model_key, 'metric': metric,
                'cat_a': c1, 'cat_b': c2,
                'n_a': int((g_p == c1).sum()), 'n_b': int((g_p == c2).sum()),
                'mean_level_a': float(X_p[g_p == c1].mean()),
                'mean_level_b': float(X_p[g_p == c2].mean()),
                'pseudo_F': F_p, 'R2': R2_p, 'p_raw': p_raw,
            })
            pvals.append(p_raw)

        if pvals:
            _, p_holm, _, _ = multipletests(pvals, alpha=ALPHA, method='holm')
            for comp, ph in zip(comps, p_holm):
                comp['p_holm'] = ph
                comp['significant'] = ph < ALPHA
                pairwise_rows.append(comp)

omnibus_df = pd.DataFrame(omnibus_rows)
pairwise_df = pd.DataFrame(pairwise_rows)

# %%
# ── Pretty print ──────────────────────────────────────────────────────────────
def _fmt_p(p):
    return f'{p:.2e}' if p < 1e-4 else f'{p:.4f}'

print('\n' + '=' * 82)
print('OMNIBUS: PERMANOVA on 8-turn trajectory vectors (categories within model × metric)')
print('=' * 82)
for _, r in omnibus_df.iterrows():
    star = '*' if r['significant'] else ' '
    print(f"{star} {r['target_model']:35s} {r['metric']:14s} "
          f"F={r['pseudo_F']:6.2f}  R²={r['R2']:.3f}  "
          f"p={_fmt_p(r['p_value'])}  n={r['n_sessions']}  k={r['n_categories']}")

print('\n' + '=' * 82)
print('POST-HOC: pairwise PERMANOVA with Holm correction (significant pairs only)')
print('=' * 82)
sig_omni = omnibus_df.loc[omnibus_df['significant'], ['target_model', 'metric']]
for _, key in sig_omni.iterrows():
    sub = pairwise_df[
        (pairwise_df['target_model'] == key['target_model']) &
        (pairwise_df['metric'] == key['metric'])
    ]
    sig_pairs = sub[sub['significant']]
    if sig_pairs.empty:
        continue
    print(f"\n[{key['target_model']} | {key['metric']}]")
    for _, r in sig_pairs.iterrows():
        direction = '>' if r['mean_level_a'] > r['mean_level_b'] else '<'
        print(f"  {r['cat_a']:20s} {direction} {r['cat_b']:20s}  "
              f"F={r['pseudo_F']:5.2f}  R²={r['R2']:.3f}  "
              f"p_holm={_fmt_p(r['p_holm'])}  "
              f"(level {r['mean_level_a']:.2f} vs {r['mean_level_b']:.2f})")

# %%
# ── Diagnostic heatmap: R² across model × metric ──────────────────────────────
heatmap = (
    omnibus_df.pivot(index='target_model', columns='metric', values='R2')
    .reindex(columns=METRICS)
)
sig_mask = (
    omnibus_df.pivot(index='target_model', columns='metric', values='significant')
    .reindex(columns=METRICS).reindex(index=heatmap.index)
)

fig, ax = plt.subplots(figsize=(6, 0.6 * len(heatmap) + 1.5))
sns.heatmap(
    heatmap, annot=True, fmt='.3f', cmap='magma_r',
    vmin=0, vmax=max(0.1, heatmap.max().max()),
    cbar_kws={'label': 'Category effect size'},
    linewidths=0.5, ax=ax
)
for i, model in enumerate(heatmap.index):
    for j, metric in enumerate(heatmap.columns):
        if not bool(sig_mask.loc[model, metric]):
            ax.text(j + 0.5, i + 0.85, 'n.s.', ha='center', va='center',
                    fontsize=8, color='gray')
# ax.set_title('Category effect')
ax.set_xlabel('Score dimension')
ax.set_ylabel('Model')
plt.tight_layout()
plt.show()

# %% [markdown]
# # User type effect (vs. CONTROL)
# 
# ## Merge all into one
# 
# What the figure shows
# 
# One subplot per target turn (default: 1, 5, 8). Within each subplot:
#   x-axis  = target model
#   y-axis  = paired effect (user_type − control), in score points (0–3)
#   color   = score dimension (correction / rebuttal / agreeableness)
#   marker  = user type (warm / emotional / hostile / logical)
# 
# How the effect is computed
# 
# For each (model, turn t, metric, user_type):
#   1. Restrict tl to that (model, turn).
#   2. Pivot to claim_idx × iv1 (mean over reps if any).
#   3. effect_per_claim = score_user_type − score_control
#   4. Report mean(diffs) and 1.96·SE/√n as the 95% CI.
# This is paired-by-claim — no session means, no pooling across turns.
# 
# Polarity reminder
# 
# - Correction / Rebuttal: positive effect = user type IMPROVES robustness
# - Agreeableness: positive effect = user type WORSENS robustness (more sycophancy)

# %%
from scipy import stats as _stats
from matplotlib.lines import Line2D
 
# ── Knobs ───────────────────────────────────────────────────────────────
TURN_TARGETS = [1, 5, 8]      # which turns to display side-by-side

CONTROL_KEY  = "none"         # IV1 reference level
TREAT_ORDER = ["emotional", "warm",  "logical", "hostile",]
TV1_ORDER = ["none", "emotional", "warm",  "logical", "hostile" ]
IV1_LABEL = {
    "none": "Control",
    "warm": "Warm",
    "emotional": "Emotional", 
    "hostile": "Hostile", 
    "logical": "Logical"
}

IV1_PALETTE = {
    "Control":   "#7f7f7f",
    "Warm":      "#e07b54",
    "Emotional": "#00c9ac",   # ← memory-pinned: soft / emotional
    "Hostile":   "#4F2683",   # ← memory-pinned: hard / hostile
    "Logical":   "#4a90d9",
}

USER_TYPE_MARKERS = {
    "warm":      "o",
    "emotional": "s",
    "hostile":   "^",
    "logical":   "D",
}
METRIC_COLORS = {              # colorblind-safe (Wong 2011)
    "correction":    "#cdb96a",   # blue
    "rebuttal":      "#89a33f",   # green
    "agreeableness": "#00a08e",   # vermilion
}

models_present = sorted(tl["target_model"].unique())



# %%
def _paired_effect(tl, model, metric, user_type, turn,
                   control=CONTROL_KEY):
    """Mean paired effect (user_type − control) across claims, plus 95% CI.
 
    Returns
    -------
    (mean_effect, ci_lo, ci_hi, n_pairs)
    """
    sub = tl[(tl["target_model"] == model) & (tl["turn"] == turn)]
    if sub.empty:
        return np.nan, np.nan, np.nan, 0
    wide = sub.pivot_table(
        index="belief_id", columns="iv1",
        values=metric, aggfunc="mean",
    )
    if user_type not in wide.columns or control not in wide.columns:
        return np.nan, np.nan, np.nan, 0
    paired = wide[[user_type, control]].dropna()
    if len(paired) < 2:
        return np.nan, np.nan, np.nan, len(paired)
    diffs   = paired[user_type] - paired[control]
    mean_e  = float(diffs.mean())
    sem     = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
    half_ci = 1.96 * sem
    return mean_e, mean_e - half_ci, mean_e + half_ci, len(diffs)

# %%
# ── Build tidy effect table ────────────────────────────────────────────
effect_rows = []
for m_name in models_present:
    for t in TURN_TARGETS:
        for dim in SCORE_DIMS:
            for ut in TREAT_ORDER:
                eff, lo, hi, n = _paired_effect(tl, m_name, dim, ut, t)
                effect_rows.append({
                    "model":     m_name,
                    "turn":      t,
                    "metric":    dim,
                    "user_type": ut,
                    "effect":    eff,
                    "ci_lo":     lo,
                    "ci_hi":     hi,
                    "n_pairs":   n,
                })
effect_df = pd.DataFrame(effect_rows)
# effect_df.to_csv(TABLE_DIR / "iv1_effects_per_turn.csv", index=False)
print(f"  ✓ iv1_effects_per_turn.csv  ({len(effect_df)} rows)")
display(effect_df.head(4))

# %%
# ── Plot ────────────────────────────────────────────────────────────────
n_panels = len(TURN_TARGETS)
fig, axes = plt.subplots(
    1, n_panels, figsize=(5.8 * n_panels, 5.0), sharey=True,
)
if n_panels == 1:
    axes = [axes]
 
# Layout per model:
#   metric column offsets at ±METRIC_GAP, 0
#   inside each metric column, user types jittered at ±USER_JITTER
METRIC_GAP  = 0.22
USER_JITTER = 0.00
 
metric_offsets = {
    dim: (i - (len(SCORE_DIMS) - 1) / 2) * METRIC_GAP
    for i, dim in enumerate(SCORE_DIMS)
}
user_jitters = {
    ut: (i - (len(TREAT_ORDER) - 1) / 2) * USER_JITTER
    for i, ut in enumerate(TREAT_ORDER)
}
 
for ax, t in zip(axes, TURN_TARGETS):
    panel = effect_df[effect_df["turn"] == t]
    for x_idx, m_name in enumerate(models_present):
        for dim in SCORE_DIMS:
            for ut in TREAT_ORDER:
                row = panel[
                    (panel["model"] == m_name)
                    & (panel["metric"] == dim)
                    & (panel["user_type"] == ut)
                ]
                if row.empty or pd.isna(row["effect"].iloc[0]):
                    continue
                eff = row["effect"].iloc[0]
                lo  = row["ci_lo"].iloc[0]
                hi  = row["ci_hi"].iloc[0]
                x   = x_idx + metric_offsets[dim] + user_jitters[ut]
 
                # vertical CI bar
                ax.vlines(x, lo, hi,
                          color=METRIC_COLORS[dim],
                          linewidth=1.4, alpha=0.85, zorder=1)
                # point
                ax.scatter(
                    x, eff,
                    marker=USER_TYPE_MARKERS[ut],
                    facecolors=METRIC_COLORS[dim],
                    edgecolors="white",
                    linewidths=0.8, s=72, zorder=3,
                )
        # faint vertical separators between models
        if x_idx < len(models_present) - 1:
            ax.axvline(x_idx + 0.5, color="gray",
                       linewidth=0.5, alpha=0.3, zorder=0)
 
    ax.axhline(0, color="black", linewidth=0.9, alpha=0.6, zorder=0)
    ax.set_xticks(range(len(models_present)))
    ax.set_xticklabels(
        [model_labels.get(m, m) for m in models_present], ha="center", fontsize=9
    )
    ax.set_title(f"Turn {t}", fontsize=12)
    ax.grid(True, axis="y", alpha=0.25)
 
axes[0].set_ylabel(
    "Treatment Score - Control Score",
    fontsize=10,
)

# ── Two-part legend (color = metric, shape = user type) ────────────────
metric_handles = [
    Line2D([0], [0], marker="s", linestyle="",
           markerfacecolor=METRIC_COLORS[d], markeredgecolor="white",
           markersize=10, label=d.title())
    for d in SCORE_DIMS
]
user_handles = [
    Line2D([0], [0], marker=USER_TYPE_MARKERS[ut], linestyle="",
           markerfacecolor="#555555", markeredgecolor="white",
           markersize=9, label=IV1_LABEL[ut])
    for ut in TREAT_ORDER
]
fig.legend(
    metric_handles, [h.get_label() for h in metric_handles],
    title="Score", loc="upper center",
    bbox_to_anchor=(0.30, 1.05), ncol=len(SCORE_DIMS), frameon=False,
)
fig.legend(
    user_handles, [h.get_label() for h in user_handles],
    title="User Type", loc="upper center",
    bbox_to_anchor=(0.72, 1.05), ncol=len(TREAT_ORDER), frameon=False,
)
 
fig.suptitle(
    "",
    fontsize=13, y=1.12,
)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_iv1_effect_per_turn.png",
            dpi=200, bbox_inches="tight")
plt.show()
print("  ✓ fig_iv1_effect_per_turn.png")

# %% [markdown]
# We calculate the difference between the scores of the user type treatment groups and the control group, along with their 95% confidence intervals (CI). The 0 on the Y-axis and the corresponding grey line represent the baseline control group. Colors represent the three scoring dimensions, while the shapes of the dots represent the four user persona types.
# 
# **All four models are most susceptible to emotional users**
# 
# - Across all models, interacting with the emotional user causes a significant decrease in correction and rebuttal quality, paired with a large increase in agreeableness.
# - This effect is most pronounced in GPT-5.3, standing in contrast to earlier observations where GPT-5.3 demonstrated the greatest stability when interacting with control group users (i.e., those without a specific persona).
# - Even though Claude Sonnet 4.6 exhibits strong resistance in the control study, facing an emotional user still triggers an increase in the model's agreeableness.
# - These findings suggest that current LLMs struggle to maintain resistance when users employ strong emotional framing around misinformation.
# 
# **Model behavior diverges across the remaining three user types**
# 
# - Gemini 3 Flash shows strong resistance to the hostile user, displaying an increase in correction and rebuttal scores alongside a decrease in agreeableness. Hostile users similarly reduce the agreeableness of DeepSeek V3.2.
# - GPT-5.3 is also quite sensitive to warm users, showcasing a behavioral vulnerability similar to its interactions with emotional users.
# - DeepSeek and Claude exhibit a bidirectional change in effect across conversational turns. For the three non-emotional user types, the impact of the persona initially peaks around (we can see from the 5th turn figure), but the effect diminishes as the conversation progresses to the 8th turn.
# 
# Overall, the emotional user persona raises the most significant red flags for evaluating model robustness. Furthermore, while GPT, Gemini, and DeepSeek are more resistant to hostile users, Claude demonstrates greater robustness when interacting with the highly analytical, logical persona. A more detailed turn-by-turn results of user effects across all 8 turns can be found in the Appendix.
# 

# %% [markdown]
# ## Per turn
# 

# %%
# models_present = ["gemini-3-flash-preview"]

ALL_TURNS = list(range(1, 9))

# Build effect-trajectory table: one row per (model, metric, user_type, turn)
eff_traj_rows = []
for m_name in models_present:
    for dim in SCORE_DIMS:
        for ut in TREAT_ORDER:
            for t in ALL_TURNS:
                eff, lo, hi, n = _paired_effect(tl, m_name, dim, ut, t)
                eff_traj_rows.append({
                    "model":     m_name,
                    "metric":    dim,
                    "user_type": ut,
                    "turn":      t,
                    "effect":    eff,
                    "ci_lo":     lo,
                    "ci_hi":     hi,
                    "n_pairs":   n,
                })
eff_traj_df = pd.DataFrame(eff_traj_rows)
# eff_traj_df.to_csv(TABLE_DIR / "iv1_effect_trajectory.csv", index=False)
print(f"  ✓ iv1_effect_trajectory.csv  ({len(eff_traj_df)} rows)")


# ── Plot ────────────────────────────────────────────────────────────────
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

        # Control reference line at y=0
        ax.axhline(0, color="black", linewidth=0.9, alpha=0.6, zorder=0)

        for ut in TREAT_ORDER:
            sub = eff_traj_df[
                (eff_traj_df["model"]     == m_name)
                & (eff_traj_df["metric"]  == dim)
                & (eff_traj_df["user_type"] == ut)
            ].sort_values("turn")
            if sub.empty or sub["effect"].isna().all():
                continue
            label = IV1_LABEL[ut]
            color = IV1_PALETTE[label]
            ax.plot(
                sub["turn"], sub["effect"], "o-",
                label=label, color=color,
                linewidth=1.7, markersize=4,
            )
            ax.fill_between(
                sub["turn"], sub["ci_lo"], sub["ci_hi"],
                alpha=0.13, color=color,
            )

        ax.set_xticks(range(1, 9))
        ax.grid(True, alpha=0.3)
        if r == 0:
            ax.set_title(f"{dim.title()}", fontsize=11)
        if c == 0:
            ax.set_ylabel(m_name, fontsize=10, fontweight="bold")
        if r == n_rows - 1:
            ax.set_xlabel("Turn")

# single legend at the top — only treatment user types (control = y=0 line)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels, title="User type",
    loc="upper center", bbox_to_anchor=(0.5, 1.01),
    ncol=len(TREAT_ORDER), frameon=False,
)

fig.suptitle(
    "Effect trajectory: IV1 vs. Control  (Δ score, 0–3 scale, 95% CI)",
    fontsize=13, y=1.04,
)
fig.supylabel(
    "Effect vs. Control  (Δ score, 0–3 scale)",
    fontsize=10, x=-0.01,
)
plt.tight_layout()
# plt.savefig(PLOT_DIR / "fig_iv1_effect_trajectory_grid.png",
#             dpi=200, bbox_inches="tight")
plt.show()
print("  ✓ fig_iv1_effect_trajectory_grid.png")

# %%
def _ci_band_local(g, metric):
    """Mean trajectory plus 95% CI band (mirrors the helper in Cell 6)."""
    m  = g.groupby("turn")[metric].mean()
    se = g.groupby("turn")[metric].sem()
    return m, m - 1.96 * se, m + 1.96 * se
 
 

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
        for k in IV1_ORDER:
            sub = tl[(tl["target_model"] == m_name) & (tl["iv1"] == k)]
            if sub.empty:
                continue
            mean, lo, hi = _ci_band_local(sub, dim)
            label = IV1_LABEL[k]
            color = IV1_PALETTE[label]
            ax.plot(mean.index, mean.values, "o-",
                    label=label, color=color,
                    linewidth=1.7, markersize=4)
            ax.fill_between(mean.index, lo.values, hi.values,
                            alpha=0.13, color=color)
 
        ax.set_xticks(range(1, 9))
        ax.set_ylim(-0.1, 3.2)
        ax.grid(True, alpha=0.3)
        if r == 0:
            ax.set_title(f"{dim.title()}", fontsize=11)
        if c == 0:
            ax.set_ylabel(m_name, fontsize=10, fontweight="bold")
        if r == n_rows - 1:
            ax.set_xlabel("Turn")
 
# single legend at the top
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels, title="User type",
    loc="upper center", bbox_to_anchor=(0.5, 1.01),
    ncol=len(TV1_ORDER), frameon=False,
)
 
fig.suptitle("Score trajectories: Model × Metric × IV1",
             fontsize=13, y=1.04)
plt.tight_layout()
# plt.savefig(PLOT_DIR / "fig_trajectory_grid_model_x_metric.png",
#             dpi=200, bbox_inches="tight")
plt.show()
print("  ✓ fig_trajectory_grid_model_x_metric.png")

# %% [markdown]
# ## Slope-as-feature: per-session OLS slope + paired Wilcoxon
# 
# For each session, fit a linear slope (np.polyfit deg=1) to the per-turn
# scores. Two things to keep in mind:
# 
# - The slope is the BEST LINEAR APPROXIMATION to the trajectory. We are
#     not assuming the data is a line — we are summarizing the linear
#     component of whatever shape it has. For monotone-ish degradation
#     curves it is a faithful one-number summary; for sharp inflections
#     (Claude's shutdown), it under-represents curvature but still
#     captures average direction.
# - Equivalent simpler summary: (score_T − score_1) / (T − 1) is a
#     two-point version of slope. We use OLS on all turns because it
#     uses the full trajectory and is more robust to one anomalous turn.
# 
# Test design: paired Wilcoxon over claim_idx — for each (model, metric),
# compare slope(user_type) vs slope(control) per matched claim. Wilcoxon
# is non-parametric and uses the matched-claim design. BH-FDR correction
# within each (model, metric) family of 4 tests.

# %%
from statsmodels.stats.multitest import multipletests
 
 
def _session_slope(group, metric):
    """OLS slope of metric over turn for one session. NaN if <2 valid points."""
    sub = group
    if "is_fallback" in sub.columns:
        sub = sub[~sub["is_fallback"]]
    sub = sub.dropna(subset=["turn", metric])
    if len(sub) < 2:
        return np.nan
    return float(np.polyfit(
        sub["turn"].astype(float),
        sub[metric].astype(float),
        deg=1,
    )[0])

# %%
# ── Build session-level slope table ─────────────────────────────────────
slope_rows = []
for sid, group in tl.groupby("session_id"):
    base = group.iloc[0]
    row = {
        "session_id":   sid,
        "belief_id":    base["belief_id"],
        "iv1":          base["iv1"],
        "target_model": base["target_model"],
    }
    for dim in SCORE_DIMS:
        row[f"slope_{dim}"] = _session_slope(group, dim)
    slope_rows.append(row)
slope_df = pd.DataFrame(slope_rows)
# slope_df.to_csv(TABLE_DIR / "session_slopes.csv", index=False)

display(slope_df.head(5))

# %%
# ── Paired Wilcoxon: slope(user_type) vs slope(control) per claim ──────
slope_test_rows = []
for m_name in models_present:
    for dim in SCORE_DIMS:
        col = f"slope_{dim}"
        sub = slope_df[slope_df["target_model"] == m_name]
        wide = sub.pivot_table(
            index="belief_id", columns="iv1",
            values=col, aggfunc="mean",
        )
        if CONTROL_KEY not in wide.columns:
            continue
        for ut in TREAT_ORDER:
            if ut not in wide.columns:
                continue
            paired = wide[[ut, CONTROL_KEY]].dropna()
            if len(paired) < 6:
                continue
            diffs = paired[ut] - paired[CONTROL_KEY]
            try:
                stat, p = _stats.wilcoxon(diffs, zero_method="wilcox")
            except ValueError:
                stat, p = np.nan, np.nan
            slope_test_rows.append({
                "model":                 m_name,
                "metric":                dim,
                "user_type":             ut,
                "n":                     len(paired),
                "mean_slope_treatment":  float(paired[ut].mean()),
                "mean_slope_control":    float(paired[CONTROL_KEY].mean()),
                "mean_slope_difference": float(diffs.mean()),
                "wilcoxon_W":            float(stat) if not np.isnan(stat) else np.nan,
                "p":                     float(p)    if not np.isnan(p)    else np.nan,
            })
 
slope_tests = pd.DataFrame(slope_test_rows)

def _bh(p_vec):
    p_arr = np.asarray(p_vec, dtype=float)
    mask  = ~np.isnan(p_arr)
    out   = np.full_like(p_arr, np.nan, dtype=float)
    if mask.any():
        out[mask] = multipletests(p_arr[mask], method="fdr_bh")[1]
    return out
 
 
# BH-FDR within each (model, metric) family of 4 tests
slope_tests["p_fdr"] = (
    slope_tests
    .groupby(["model", "metric"], group_keys=False)["p"]
    .transform(_bh)
)
slope_tests["sig"] = slope_tests["p_fdr"].apply(
    lambda x: "***" if (pd.notna(x) and x < 0.001)
              else "**"  if (pd.notna(x) and x < 0.01)
              else "*"   if (pd.notna(x) and x < 0.05)
              else ""
)
# slope_tests.to_csv(TABLE_DIR / "slope_paired_tests.csv", index=False)
print("\nSlope tests (treatment vs. control, paired Wilcoxon, "
      "BH-FDR within model × metric):\n")
print(slope_tests.round(4).to_string(index=False))
print("\n  ✓ slope_paired_tests.csv")

# %%



