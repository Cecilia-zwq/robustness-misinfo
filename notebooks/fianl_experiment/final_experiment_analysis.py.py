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

tl_control = pd.DataFrame(turn_rows)
print(f'Loaded {len(tl_control)} turn rows')

tl_control = tl_control.loc[
    ~tl_control[['correction', 'rebuttal', 'agreeableness']].eq(-1).any(axis=1)
].copy()
print(f'Loaded {len(tl_control)} turn rows after dropping rows with any -1 score')

# %%
tl_control.head(2)

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
           frameon=True, bbox_to_anchor=(0.5, 0.91))

fig.suptitle('LLM Robustness Trajectories by Model', fontsize=14, y=1.09)
plt.tight_layout()
# plt.savefig(OUT_DIR / 'fig_trajectory_by_model.png',
#             dpi=200, bbox_inches='tight')
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
# # User type effect

# %%



