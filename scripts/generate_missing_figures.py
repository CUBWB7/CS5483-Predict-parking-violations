"""
Generate 5 missing figures for the video presentation.
Run from the project root directory:
    conda run -n parking python scripts/generate_missing_figures.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Ensure output directory exists
os.makedirs('figures', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Figure 1: Score Progression (P0)
# ─────────────────────────────────────────────────────────────
def plot_score_progression():
    labels = ['v1', 'v2', 'v3', 'v7', 'Exp C', 'Exp I-A']
    oof    = [0.5880, 0.6012, 0.6408, 0.6429, 0.6464, 0.6478]
    plat   = [0.5222, 0.5338, 0.5620, 0.5636, 0.5698, 0.5705]

    annotations = {
        'v3':     ('Optuna\nTuning',       'oof'),
        'v7':     ('Sample\nWeighting',    'oof'),
        'Exp C':  ('Rank-Target\nTraining','plat'),
        'Exp I-A':('More Iterations\n(BEST: 0.5705)', 'plat'),
    }

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, oof,  'o-', color='steelblue',  linewidth=2, markersize=7, label='OOF Spearman')
    ax.plot(x, plat, 's--', color='darkorange', linewidth=2, markersize=7, label='Platform Spearman')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Spearman Correlation', fontsize=11)
    ax.set_title('Score Progression: v1 → Exp I-A', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0.50, 0.68)
    ax.axhline(y=0.197, color='gray', linestyle=':', alpha=0.7, label='Official Baseline (0.197)')

    for i, lbl in enumerate(labels):
        if lbl in annotations:
            text, ref = annotations[lbl]
            y_val = plat[i] if ref == 'plat' else oof[i]
            offset = -0.028 if ref == 'plat' else 0.012
            ax.annotate(
                text,
                xy=(i, y_val),
                xytext=(i, y_val + offset),
                ha='center', fontsize=8.5,
                color='darkorange' if ref == 'plat' else 'steelblue',
                arrowprops=dict(arrowstyle='->', color='darkorange' if ref == 'plat' else 'steelblue', lw=1)
            )

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/score_progression.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] figures/score_progression.png")


# ─────────────────────────────────────────────────────────────
# Figure 2: Rank-Target Diagram (P0)
# ─────────────────────────────────────────────────────────────
def plot_rank_target_diagram():
    from scipy.stats import rankdata

    # Try to load actual training data; fall back to synthetic
    try:
        import pandas as pd
        df = pd.read_parquet('data/train_features_tier2.parquet', columns=['invalid_ratio'])
        y_raw = df['invalid_ratio'].values
        print("  rank_target_diagram: using actual training data")
    except Exception:
        np.random.seed(42)
        y_raw = np.concatenate([
            np.zeros(3000),
            np.ones(2000),
            np.random.beta(0.5, 3, 15000)
        ])
        print("  rank_target_diagram: using synthetic data (actual data not found)")

    y_rank = rankdata(y_raw) / len(y_raw)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: raw target
    axes[0].hist(y_raw, bins=60, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].set_title('Original Target\n(invalid_ratio)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Count')
    peak_count = np.histogram(y_raw, bins=60)[0].max()
    axes[0].annotate('Highly skewed\ndistribution',
                     xy=(0.6, peak_count * 0.6),
                     fontsize=9, color='steelblue',
                     bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', alpha=0.5))

    # Middle arrow
    fig.text(0.5, 0.52,
             '→\nrankdata(y) / N',
             ha='center', va='center', fontsize=12, fontweight='bold', color='black',
             transform=fig.transFigure)

    # Right: rank target
    axes[1].hist(y_rank, bins=60, color='darkorange', edgecolor='white', alpha=0.85)
    axes[1].set_title('Rank-Transformed Target\nrankdata(y) / N', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Count')
    axes[1].annotate('Uniform distribution\n→ Aligns with Spearman',
                     xy=(0.3, len(y_raw) / 60 * 0.7),
                     fontsize=9, color='darkorange',
                     bbox=dict(boxstyle='round,pad=0.3', fc='moccasin', alpha=0.5))

    fig.suptitle('Rank-Target Transformation: Directly Optimizing Spearman Ranking',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/rank_target_diagram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] figures/rank_target_diagram.png")


# ─────────────────────────────────────────────────────────────
# Figure 3: Experiment Summary Chart (P0)
# ─────────────────────────────────────────────────────────────
def plot_experiment_summary():
    from matplotlib.patches import Patch

    experiments = [
        'v1 (Initial)',
        'v3 (+Optuna)',
        'v7 (+Weighting)',
        'Exp E (TabM)',
        'Exp H (Noise Filter)',
        'Exp G (Pseudo-label)',
        'Exp C (Rank-Target)',
        'Exp I-A (More Iter.)',
    ]
    oof_scores  = [0.5880, 0.6408, 0.6429, 0.4445, 0.6442, 0.6463, 0.6464, 0.6478]
    plat_scores = [0.5222, 0.5620, 0.5636, None,   0.5613, None,   0.5698, 0.5705]
    status      = ['baseline','baseline','baseline','failed','failed','null','success','best']

    color_map = {
        'baseline': '#90CAF9',
        'failed':   '#EF9A9A',
        'null':     '#FFE082',
        'success':  '#A5D6A7',
        'best':     '#2E7D32',
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    y_pos = np.arange(len(experiments))

    for i, (exp, oof, plat, st) in enumerate(zip(experiments, oof_scores, plat_scores, status)):
        color = color_map[st]
        bar = ax.barh(i - 0.2, oof, height=0.35, color=color, alpha=0.9)
        ax.text(oof + 0.002, i - 0.2, f'{oof:.4f}', va='center', fontsize=8, color='black')
        if plat is not None:
            ax.barh(i + 0.2, plat, height=0.35, color=color, alpha=0.55, hatch='//')
            ax.text(plat + 0.002, i + 0.2, f'{plat:.4f}', va='center', fontsize=8, color='dimgray')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(experiments, fontsize=10)
    ax.set_xlabel('Spearman Correlation', fontsize=11)
    ax.set_title('Experiment Summary: OOF and Platform Scores', fontsize=13, fontweight='bold')
    ax.axvline(x=0.5636, color='gray',    linestyle=':',  alpha=0.7, linewidth=1.5)
    ax.axvline(x=0.5705, color='#2E7D32', linestyle='--', alpha=0.9, linewidth=1.5)
    ax.text(0.5636 + 0.001, 0.3, 'v7 platform\n(0.5636)', fontsize=7.5, color='gray',      transform=ax.get_xaxis_transform())
    ax.text(0.5705 + 0.001, 0.3, 'BEST platform\n(0.5705)', fontsize=7.5, color='#2E7D32', transform=ax.get_xaxis_transform())

    legend_elements = [
        Patch(facecolor='#90CAF9', label='Baseline'),
        Patch(facecolor='#A5D6A7', label='Success'),
        Patch(facecolor='#2E7D32', label='Best Result'),
        Patch(facecolor='#EF9A9A', label='Failed'),
        Patch(facecolor='#FFE082', label='Null Result'),
        Patch(facecolor='white', edgecolor='gray', hatch='//', label='Platform score (hatched)'),
        Patch(facecolor='gray',  alpha=0.5,                   label='OOF score (solid)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    ax.set_xlim(0.40, 0.685)
    plt.tight_layout()
    plt.savefig('figures/experiment_summary_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] figures/experiment_summary_chart.png")


# ─────────────────────────────────────────────────────────────
# Figure 4: Feature Engineering Pipeline (P1)
# ─────────────────────────────────────────────────────────────
def plot_feature_engineering_pipeline():
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Box style helpers
    box_kw  = dict(boxstyle='round,pad=0.5', fc='#E3F2FD', ec='#1565C0', lw=1.5)
    arr_kw  = dict(arrowstyle='->', color='#1565C0', lw=1.5)
    side_kw = dict(boxstyle='round,pad=0.4', fc='#FFF9C4', ec='#F57F17', lw=1.2)
    feat_kw = dict(boxstyle='round,pad=0.5', fc='#E8F5E9', ec='#2E7D32', lw=1.8)

    steps = [
        (5, 9.2, 'Raw Features (10)',             box_kw),
        (5, 7.7, 'Coordinate Binning',            box_kw),
        (5, 6.2, 'K-Fold Target Encoding (k=5)', box_kw),
        (5, 4.7, 'Cyclic Encoding',               box_kw),
        (5, 3.2, 'Cross Features',                box_kw),
        (5, 1.7, 'Sample Weighting',              box_kw),
        (5, 0.3, 'Tier2 Feature Set (~20 feats)', feat_kw),
    ]

    side_notes = [
        (8.2, 7.7, '→ grid_x, grid_y, grid_id'),
        (8.2, 6.2, '→ grid_te, period_te, …'),
        (8.2, 4.7, '→ sin/cos(hour, month)'),
        (8.2, 3.2, '→ total_count × grid_te'),
        (8.2, 1.7, '→ log1p(total_count)'),
    ]

    for x, y, text, style in steps:
        ax.text(x, y, text, ha='center', va='center', fontsize=11,
                fontweight='bold' if 'Tier2' in text or 'Raw' in text else 'normal',
                bbox=style, transform=ax.transData)

    # Arrows between boxes
    for i in range(len(steps) - 1):
        _, y1, _, _ = steps[i]
        _, y2, _, _ = steps[i+1]
        ax.annotate('', xy=(5, y2 + 0.35), xytext=(5, y1 - 0.35),
                    arrowprops=arr_kw)

    # Side notes
    for x, y, text in side_notes:
        ax.text(x, y, text, ha='left', va='center', fontsize=9,
                bbox=side_kw, color='#5D4037')
        ax.annotate('', xy=(x - 0.15, y), xytext=(6.2, y),
                    arrowprops=dict(arrowstyle='->', color='#F57F17', lw=1.0))

    ax.set_title('Feature Engineering Pipeline', fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig('figures/feature_engineering_pipeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] figures/feature_engineering_pipeline.png")


# ─────────────────────────────────────────────────────────────
# Figure 5: OOF-Platform Gap (P1)
# ─────────────────────────────────────────────────────────────
def plot_oof_platform_gap():
    versions  = ['v1', 'v3', 'v7', 'Exp H', 'Exp C', 'Exp I-A']
    oof_vals  = [0.5880, 0.6408, 0.6429, 0.6442, 0.6464, 0.6478]
    plat_vals = [0.5222, 0.5620, 0.5636, 0.5613, 0.5698, 0.5705]
    gaps      = [o - p for o, p in zip(oof_vals, plat_vals)]

    x = np.arange(len(versions))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    # Top: OOF vs Platform lines
    ax1.plot(x, oof_vals,  'o-', color='steelblue',  linewidth=2, markersize=7, label='OOF Spearman')
    ax1.plot(x, plat_vals, 's--', color='darkorange', linewidth=2, markersize=7, label='Platform Spearman')
    ax1.fill_between(x, plat_vals, oof_vals, alpha=0.12, color='gray', label='Gap region')
    for xi, (o, p) in enumerate(zip(oof_vals, plat_vals)):
        ax1.annotate(f'{o:.4f}', (xi, o), textcoords='offset points', xytext=(0, 6),
                     ha='center', fontsize=8, color='steelblue')
        ax1.annotate(f'{p:.4f}', (xi, p), textcoords='offset points', xytext=(0, -13),
                     ha='center', fontsize=8, color='darkorange')
    ax1.set_ylabel('Spearman Correlation', fontsize=11)
    ax1.set_title('OOF vs Platform Score — Gap Stability Analysis', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_ylim(0.50, 0.68)
    ax1.grid(axis='y', alpha=0.3)

    # Bottom: gap bars
    bar_colors = ['#90CAF9'] * 3 + ['#EF9A9A'] + ['#A5D6A7'] + ['#2E7D32']
    ax2.bar(x, gaps, color=bar_colors, alpha=0.85, edgecolor='white')
    ax2.axhline(np.mean(gaps), color='dimgray', linestyle='--', linewidth=1.2,
                label=f'Mean gap = {np.mean(gaps):.4f}')
    for xi, g in enumerate(gaps):
        ax2.text(xi, g + 0.001, f'{g:.4f}', ha='center', fontsize=9)
    ax2.set_ylabel('OOF − Platform', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(versions, fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 0.12)
    ax2.grid(axis='y', alpha=0.3)

    fig.text(0.5, -0.01,
             'Stable gap (~0.077) indicates distribution shift, NOT overfitting',
             ha='center', fontsize=10, style='italic', color='dimgray')

    plt.tight_layout()
    plt.savefig('figures/oof_platform_gap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] figures/oof_platform_gap.png")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating missing figures...")
    plot_score_progression()
    plot_rank_target_diagram()
    plot_experiment_summary()
    plot_feature_engineering_pipeline()
    plot_oof_platform_gap()
    print("\nAll done. Check figures/ directory.")
