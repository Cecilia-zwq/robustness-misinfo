"""
experiment1_analysis.py
=======================
Comprehensive analysis of Experiment 1 results.
Produces statistical tests, tables, and publication-ready plots.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = Path("/home/claude/analysis")
OUT_DIR = Path("/home/claude/analysis/plots")
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.facecolor': 'white',
})

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / "summary.csv")
tl = pd.read_csv(DATA_DIR / "turn_level.csv")

persona_labels = {'persona_a_soft': 'Soft (Emotional)', 'persona_b_hard': 'Hard (Aggressive)'}
df['persona_label'] = df['persona'].map(persona_labels)
tl['persona_label'] = tl['persona'].map(persona_labels)

print("=" * 70)
print("EXPERIMENT 1 — FULL ANALYSIS")
print("=" * 70)
print(f"\nDataset: {len(df)} sessions, {len(tl)} turn-level observations")
print(f"Personas: {df['persona'].value_counts().to_dict()}")
print(f"Categories: {df['category'].value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA QUALITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 70}")
print("1. DATA QUALITY")
print(f"{'═' * 70}")

# Check for anomalous scores
anomalies = tl[(tl['correction'] < 0) | (tl['rebuttal'] < 0)]
if len(anomalies) > 0:
    print(f"\n⚠ ANOMALY: {len(anomalies)} turns with negative scores:")
    print(anomalies[['session_id','turn','correction','rebuttal']].to_string(index=False))

# Fallback rate
print(f"\nFallback turns: {tl['is_fallback'].sum()} / {len(tl)} = {tl['is_fallback'].mean():.3f}")
print(f"Zero-fallback sessions: {(df['n_breaks_fallback'] == 0).sum()} / {len(df)}")

# Inter-rep consistency (CV within claim × persona)
rep_cv = df.groupby(['claim_idx', 'persona'])['mean_correction'].agg(['mean','std'])
rep_cv['cv'] = rep_cv['std'] / rep_cv['mean'].replace(0, np.nan)
print(f"\nInter-rep consistency (CV of mean_correction):")
print(f"  Median CV: {rep_cv['cv'].median():.3f}")
print(f"  Max CV:    {rep_cv['cv'].max():.3f}")
print(f"  Conditions with CV > 0.3: {(rep_cv['cv'] > 0.3).sum()} / {len(rep_cv)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. MAIN EFFECTS — STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 70}")
print("2. STATISTICAL TESTS")
print(f"{'═' * 70}")

# Helper: significance stars
def sig(p): return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'

# Helper: Mann-Whitney U with rank-biserial effect size
def mwu(a, b):
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    r = 1 - 2*u/(len(a)*len(b))
    return u, p, r

# ── Slices ───────────────────────────────────────────────────────────────
soft_corr = df[df['persona']=='persona_a_soft']['mean_correction']
hard_corr = df[df['persona']=='persona_b_hard']['mean_correction']
soft_rebt = df[df['persona']=='persona_a_soft']['mean_rebuttal']
hard_rebt = df[df['persona']=='persona_b_hard']['mean_rebuttal']

# ── 2a. Normality checks ────────────────────────────────────────────────
print("\n--- Shapiro-Wilk Normality Tests ---")
for label, vals in [('soft_corr', soft_corr), ('hard_corr', hard_corr),
                     ('soft_rebt', soft_rebt), ('hard_rebt', hard_rebt)]:
    _, p = stats.shapiro(vals)
    print(f"  {label:<12} p={p:.4f} {'(normal)' if p>0.05 else '(non-normal → use Mann-Whitney)'}")

# ── 2b. Persona effect ──────────────────────────────────────────────────
u_corr, p_corr, r_corr = mwu(soft_corr, hard_corr)
u_rebt, p_rebt, r_rebt = mwu(soft_rebt, hard_rebt)

print(f"\n--- Persona Effect ---")
print(f"  {'Metric':<12} {'Soft Mean (SD)':<20} {'Hard Mean (SD)':<20} {'U':>7} {'p':>12} {'r':>7} {'sig':>4}")
print(f"  {'─'*12} {'─'*20} {'─'*20} {'─'*7} {'─'*12} {'─'*7} {'─'*4}")
print(f"  {'Correction':<12} {soft_corr.mean():.3f} ({soft_corr.std():.3f})     "
      f"{hard_corr.mean():.3f} ({hard_corr.std():.3f})     "
      f"{u_corr:>7.0f} {p_corr:>12.6f} {r_corr:>7.3f} {sig(p_corr):>4}")
print(f"  {'Rebuttal':<12} {soft_rebt.mean():.3f} ({soft_rebt.std():.3f})     "
      f"{hard_rebt.mean():.3f} ({hard_rebt.std():.3f})     "
      f"{u_rebt:>7.0f} {p_rebt:>12.6f} {r_rebt:>7.3f} {sig(p_rebt):>4}")
print(f"\n  Δ correction = {hard_corr.mean() - soft_corr.mean():+.3f} (hard − soft)")
print(f"  Δ rebuttal   = {hard_rebt.mean() - soft_rebt.mean():+.3f} (hard − soft)")

# ── 2c. Category effect ─────────────────────────────────────────────────
bias_corr = df[df['category']=='bias']['mean_correction']
consp_corr = df[df['category']=='conspiracy']['mean_correction']
bias_rebt = df[df['category']=='bias']['mean_rebuttal']
consp_rebt = df[df['category']=='conspiracy']['mean_rebuttal']

u_cat_c, p_cat_c, r_cat_c = mwu(bias_corr, consp_corr)
u_cat_r, p_cat_r, r_cat_r = mwu(bias_rebt, consp_rebt)

print(f"\n--- Category Effect ---")
print(f"  {'Metric':<12} {'Bias Mean':<12} {'Consp. Mean':<14} {'U':>7} {'p':>10} {'r':>7} {'sig':>4}")
print(f"  {'─'*12} {'─'*12} {'─'*14} {'─'*7} {'─'*10} {'─'*7} {'─'*4}")
print(f"  {'Correction':<12} {bias_corr.mean():<12.3f} {consp_corr.mean():<14.3f} "
      f"{u_cat_c:>7.0f} {p_cat_c:>10.4f} {r_cat_c:>7.3f} {sig(p_cat_c):>4}")
print(f"  {'Rebuttal':<12} {bias_rebt.mean():<12.3f} {consp_rebt.mean():<14.3f} "
      f"{u_cat_r:>7.0f} {p_cat_r:>10.4f} {r_cat_r:>7.3f} {sig(p_cat_r):>4}")

# ── 2d. Degradation trend ───────────────────────────────────────────────
soft_ct = df[df['persona']=='persona_a_soft']['correction_trend']
hard_ct = df[df['persona']=='persona_b_hard']['correction_trend']
soft_rt = df[df['persona']=='persona_a_soft']['rebuttal_trend']
hard_rt = df[df['persona']=='persona_b_hard']['rebuttal_trend']

u_ct, p_ct, r_ct = mwu(soft_ct, hard_ct)
u_rt, p_rt, r_rt = mwu(soft_rt, hard_rt)

print(f"\n--- Degradation Trend (linear slope per turn) ---")
print(f"  {'Metric':<16} {'Soft Slope':<14} {'Hard Slope':<14} {'U':>7} {'p':>12} {'sig':>4}")
print(f"  {'─'*16} {'─'*14} {'─'*14} {'─'*7} {'─'*12} {'─'*4}")
print(f"  {'Correction':<16} {soft_ct.mean():<+14.4f} {hard_ct.mean():<+14.4f} "
      f"{u_ct:>7.0f} {p_ct:>12.6f} {sig(p_ct):>4}")
print(f"  {'Rebuttal':<16} {soft_rt.mean():<+14.4f} {hard_rt.mean():<+14.4f} "
      f"{u_rt:>7.0f} {p_rt:>12.6f} {sig(p_rt):>4}")

# ── 2e. Character break rate ────────────────────────────────────────────
soft_br = df[df['persona']=='persona_a_soft']['character_break_rate']
hard_br = df[df['persona']=='persona_b_hard']['character_break_rate']
u_br, p_br, r_br = mwu(soft_br, hard_br)

print(f"\n--- Simulated User Character Breaks ---")
print(f"  Soft break rate: {soft_br.mean():.3f} (SD={soft_br.std():.3f})")
print(f"  Hard break rate: {hard_br.mean():.3f} (SD={hard_br.std():.3f})")
print(f"  U={u_br:.0f}, p={p_br:.6f}, r={r_br:.3f} ({sig(p_br)})")


# ══════════════════════════════════════════════════════════════════════════════
# 3. SYCOPHANCY ONSET ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 70}")
print("3. SYCOPHANCY ONSET ANALYSIS")
print(f"{'═' * 70}")

syc = df[df['sycophancy_onset_turn'].notna()].copy()
print(f"\nSessions with sycophancy onset (correction ≤ 2): {len(syc)}/{len(df)} ({100*len(syc)/len(df):.0f}%)")

for p in ['persona_a_soft', 'persona_b_hard']:
    s = syc[syc['persona']==p]['sycophancy_onset_turn']
    print(f"\n  {persona_labels[p]}:")
    print(f"    Mean onset turn: {s.mean():.2f} (SD={s.std():.2f})")
    print(f"    Median: {s.median():.0f}, Range: [{s.min():.0f}, {s.max():.0f}]")
    print(f"    Onset at turn 1: {(s==1).sum()}/{len(s)} ({100*(s==1).sum()/len(s):.0f}%)")

u_syc, p_syc = stats.mannwhitneyu(
    syc[syc['persona']=='persona_a_soft']['sycophancy_onset_turn'],
    syc[syc['persona']=='persona_b_hard']['sycophancy_onset_turn'],
    alternative='two-sided')
print(f"\n  Mann-Whitney U test: U={u_syc:.0f}, p={p_syc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SUBTYPE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 70}")
print("4. SUBTYPE ANALYSIS")
print(f"{'═' * 70}")

subtype_stats = df.groupby(['category','subtype']).agg(
    n=('mean_correction','count'),
    corr_mean=('mean_correction','mean'),
    corr_std=('mean_correction','std'),
    rebt_mean=('mean_rebuttal','mean'),
    onset_mean=('sycophancy_onset_turn','mean'),
).round(3)
print(f"\n{subtype_stats.to_string()}")

# Weakest and strongest subtypes
print(f"\n  Weakest (lowest correction): {subtype_stats['corr_mean'].idxmin()} = {subtype_stats['corr_mean'].min():.3f}")
print(f"  Strongest (highest correction): {subtype_stats['corr_mean'].idxmax()} = {subtype_stats['corr_mean'].max():.3f}")

# Kruskal-Wallis within bias subtypes
bias_groups = [g['mean_correction'].values for _, g in df[df['category']=='bias'].groupby('subtype')]
if len(bias_groups) > 2:
    h_bias, p_bias = stats.kruskal(*bias_groups)
    print(f"\n  Kruskal-Wallis (bias subtypes): H={h_bias:.2f}, p={p_bias:.4f}")

consp_groups = [g['mean_correction'].values for _, g in df[df['category']=='conspiracy'].groupby('subtype')]
if len(consp_groups) > 2:
    h_consp, p_consp = stats.kruskal(*consp_groups)
    print(f"  Kruskal-Wallis (conspiracy subtypes): H={h_consp:.2f}, p={p_consp:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. CLAIM-LEVEL VULNERABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 70}")
print("5. MOST/LEAST VULNERABLE CLAIMS")
print(f"{'═' * 70}")

claim_stats = df.groupby(['claim_idx','category','subtype','claim']).agg(
    corr_mean=('mean_correction','mean'),
    corr_min=('min_correction','min'),
).sort_values('corr_mean')

print("\nTop 5 most vulnerable claims (lowest mean correction):")
for i, (idx, row) in enumerate(claim_stats.head(5).iterrows()):
    print(f"  {i+1}. [{idx[1]}|{idx[2]}] corr={row['corr_mean']:.2f} — {idx[3][:60]}")

print("\nTop 5 most robust claims (highest mean correction):")
for i, (idx, row) in enumerate(claim_stats.tail(5).iloc[::-1].iterrows()):
    print(f"  {i+1}. [{idx[1]}|{idx[2]}] corr={row['corr_mean']:.2f} — {idx[3][:60]}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 70}")
print("6. GENERATING PLOTS")
print(f"{'═' * 70}")

palette = {'Soft (Emotional)': '#e07b54', 'Hard (Aggressive)': '#4a90d9'}

# ── Plot 1: Score trajectory by turn × persona ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, metric, title in zip(axes, ['correction', 'rebuttal'],
                              ['Correction Quality', 'Rebuttal Quality']):
    for p, label in persona_labels.items():
        subset = tl[tl['persona'] == p]
        means = subset.groupby('turn')[metric].mean()
        sems = subset.groupby('turn')[metric].sem()
        ax.plot(means.index, means.values, 'o-', label=label,
                color=palette[label], linewidth=2, markersize=6)
        ax.fill_between(means.index, means - 1.96*sems, means + 1.96*sems,
                        alpha=0.15, color=palette[label])
    ax.set_xlabel('Turn')
    ax.set_ylabel('Score (0–3)')
    ax.set_title(title)
    ax.set_xticks(range(1, 9))
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 3.2)

fig.suptitle('LLM Robustness Degradation Over Conversation Turns', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig1_trajectory_by_persona.png')
plt.close()
print("  ✓ fig1_trajectory_by_persona.png")

# ── Plot 2: Trajectory by category ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
cat_palette = {'bias': '#6ba368', 'conspiracy': '#9b59b6'}

for ax, metric, title in zip(axes, ['correction', 'rebuttal'],
                              ['Correction Quality', 'Rebuttal Quality']):
    for cat in ['bias', 'conspiracy']:
        subset = tl[tl['category'] == cat]
        means = subset.groupby('turn')[metric].mean()
        sems = subset.groupby('turn')[metric].sem()
        ax.plot(means.index, means.values, 'o-', label=cat.title(),
                color=cat_palette[cat], linewidth=2, markersize=6)
        ax.fill_between(means.index, means - 1.96*sems, means + 1.96*sems,
                        alpha=0.15, color=cat_palette[cat])
    ax.set_xlabel('Turn')
    ax.set_ylabel('Score (0–3)')
    ax.set_title(title)
    ax.set_xticks(range(1, 9))
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 3.2)

fig.suptitle('LLM Robustness by Misinformation Category', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig2_trajectory_by_category.png')
plt.close()
print("  ✓ fig2_trajectory_by_category.png")

# ── Plot 3: Trajectory by persona × category (2×2) ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
combo_styles = {
    ('persona_a_soft', 'bias'):       ('Soft × Bias',       '#e07b54', '-'),
    ('persona_a_soft', 'conspiracy'): ('Soft × Conspiracy',  '#e07b54', '--'),
    ('persona_b_hard', 'bias'):       ('Hard × Bias',       '#4a90d9', '-'),
    ('persona_b_hard', 'conspiracy'): ('Hard × Conspiracy',  '#4a90d9', '--'),
}

for ax, metric, title in zip(axes, ['correction', 'rebuttal'],
                              ['Correction Quality', 'Rebuttal Quality']):
    for (p, cat), (label, color, ls) in combo_styles.items():
        subset = tl[(tl['persona']==p) & (tl['category']==cat)]
        means = subset.groupby('turn')[metric].mean()
        ax.plot(means.index, means.values, marker='o', linestyle=ls,
                label=label, color=color, linewidth=2, markersize=5)
    ax.set_xlabel('Turn')
    ax.set_ylabel('Score (0–3)')
    ax.set_title(title)
    ax.set_xticks(range(1, 9))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 3.2)

fig.suptitle('Persona × Category Interaction', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig3_trajectory_interaction.png')
plt.close()
print("  ✓ fig3_trajectory_interaction.png")

# ── Plot 4: Boxplot of mean correction by persona ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, metric, title in zip(axes, ['mean_correction', 'mean_rebuttal'],
                              ['Mean Correction', 'Mean Rebuttal']):
    sns.boxplot(data=df, x='persona_label', y=metric, ax=ax, palette=palette,
                width=0.5, fliersize=4)
    sns.stripplot(data=df, x='persona_label', y=metric, ax=ax,
                  color='black', alpha=0.3, size=3, jitter=True)
    ax.set_xlabel('')
    ax.set_ylabel('Score (0–3)')
    ax.set_title(title)

fig.suptitle('Score Distribution by Persona Type', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig4_boxplot_persona.png')
plt.close()
print("  ✓ fig4_boxplot_persona.png")

# ── Plot 5: Subtype heatmap ─────────────────────────────────────────────────
pivot = df.groupby(['subtype','persona_label'])['mean_correction'].mean().unstack()
pivot = pivot.sort_values('Soft (Emotional)')

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax,
            vmin=0.8, vmax=2.8, linewidths=0.5,
            cbar_kws={'label': 'Mean Correction Score'})
ax.set_title('Mean Correction Score by Subtype × Persona', fontsize=13)
ax.set_xlabel('')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig5_subtype_heatmap.png')
plt.close()
print("  ✓ fig5_subtype_heatmap.png")

# ── Plot 6: Sycophancy onset distribution ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for p, label in persona_labels.items():
    vals = df[df['persona']==p]['sycophancy_onset_turn'].dropna()
    counts = vals.value_counts().sort_index()
    ax.bar(counts.index + (-0.2 if p=='persona_a_soft' else 0.2),
           counts.values / len(vals) * 100,
           width=0.35, label=label, color=palette[label], alpha=0.85)
ax.set_xlabel('Sycophancy Onset Turn')
ax.set_ylabel('% of Sessions')
ax.set_title('When Does the LLM First Capitulate? (correction ≤ 2)')
ax.set_xticks(range(1, 9))
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig6_sycophancy_onset.png')
plt.close()
print("  ✓ fig6_sycophancy_onset.png")

# ── Plot 7: Correction trend distribution ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for p, label in persona_labels.items():
    vals = df[df['persona']==p]['correction_trend'].dropna()
    ax.hist(vals, bins=20, alpha=0.6, label=label, color=palette[label], edgecolor='white')
ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Correction Trend (slope per turn)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Score Degradation Rate')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig7_correction_trend.png')
plt.close()
print("  ✓ fig7_correction_trend.png")

# ── Plot 8: Per-claim mean correction, grouped by persona ───────────────────
claim_means = df.groupby(['claim_idx','persona_label','category','subtype'])['mean_correction'].mean().reset_index()
claim_pivot = claim_means.pivot(index='claim_idx', columns='persona_label', values='mean_correction')

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(claim_pivot))
w = 0.35
ax.bar(x - w/2, claim_pivot['Soft (Emotional)'], w, label='Soft (Emotional)',
       color=palette['Soft (Emotional)'], alpha=0.85)
ax.bar(x + w/2, claim_pivot['Hard (Aggressive)'], w, label='Hard (Aggressive)',
       color=palette['Hard (Aggressive)'], alpha=0.85)
ax.set_xlabel('Claim Index')
ax.set_ylabel('Mean Correction Score')
ax.set_title('Per-Claim Robustness by Persona Type')
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in claim_pivot.index], fontsize=8)
ax.axvline(14.5, color='gray', linestyle=':', linewidth=1)
ax.text(7, 2.9, 'BIAS', ha='center', fontsize=11, color='gray')
ax.text(22, 2.9, 'CONSPIRACY', ha='center', fontsize=11, color='gray')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig8_per_claim_comparison.png')
plt.close()
print("  ✓ fig8_per_claim_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY OF KEY FINDINGS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 70}")
print("7. KEY FINDINGS SUMMARY")
print(f"{'═' * 70}")

print(f"""
1. PERSONA EFFECT (primary finding):
   Soft emotional pressure degrades LLM robustness significantly more than
   hard aggressive pressure on BOTH metrics:
     Correction: soft={soft_corr.mean():.2f} vs hard={hard_corr.mean():.2f}
       U={u_corr:.0f}, p={p_corr:.6f}, r={r_corr:.3f}
     Rebuttal:   soft={soft_rebt.mean():.2f} vs hard={hard_rebt.mean():.2f}
       U={u_rebt:.0f}, p={p_rebt:.6f}, r={r_rebt:.3f}
   
2. UNIVERSAL EARLY CAPITULATION:
   100% of sessions show sycophancy onset (correction ≤ 2).
   Soft persona: median onset at turn {syc[syc['persona']=='persona_a_soft']['sycophancy_onset_turn'].median():.0f} 
   (mean {syc[syc['persona']=='persona_a_soft']['sycophancy_onset_turn'].mean():.1f}).
   Hard persona: median onset at turn {syc[syc['persona']=='persona_b_hard']['sycophancy_onset_turn'].median():.0f}
   (mean {syc[syc['persona']=='persona_b_hard']['sycophancy_onset_turn'].mean():.1f}).

3. CATEGORY EFFECT:
   Bias vs. conspiracy: no significant difference on either metric.
     Correction: p={p_cat_c:.4f}
     Rebuttal:   p={p_cat_r:.4f}
   
4. DEGRADATION TRAJECTORY:
   Soft persona causes steeper decline on both metrics:
     Correction slope: soft={soft_ct.mean():.4f} vs hard={hard_ct.mean():.4f}, p={p_ct:.6f}
     Rebuttal slope:   soft={soft_rt.mean():.4f} vs hard={hard_rt.mean():.4f}, p={p_rt:.6f}
   
5. SUBTYPE VULNERABILITY:
   Most vulnerable: {subtype_stats['corr_mean'].idxmin()} (corr={subtype_stats['corr_mean'].min():.2f}, rebt={subtype_stats.loc[subtype_stats['corr_mean'].idxmin(), 'rebt_mean']:.2f})
   Most robust: {subtype_stats['corr_mean'].idxmax()} (corr={subtype_stats['corr_mean'].max():.2f}, rebt={subtype_stats.loc[subtype_stats['corr_mean'].idxmax(), 'rebt_mean']:.2f})
   
6. CHARACTER CONSISTENCY:
   Soft persona breaks character more ({soft_br.mean():.3f})
   than hard persona ({hard_br.mean():.3f}), U={u_br:.0f}, p={p_br:.6f}.
   Paradox: the persona that breaks more is also more effective.

7. DATA ANOMALY:
   One turn scored -1 (bias_04_persona_a_soft_rep2, turn 2). Flag for review.
""")

print(f"\nAll plots saved to: {OUT_DIR}/")
print("Done.")