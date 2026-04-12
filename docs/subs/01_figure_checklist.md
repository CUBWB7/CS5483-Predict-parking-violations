# Figure Checklist for Video Presentation

## Existing Figures (23 total)

### docs/figures/ — EDA Phase (8 files)

| File | Content | PPT Usability | Recommended Section |
|------|---------|--------------|---------------------|
| fig1_target_distribution.png | Distribution of `invalid_ratio` target | ✅ Use | Sec 1 Intro / Sec 2 EDA |
| fig2_totalcount_vs_violation.png | `total_count` vs violation rate | ✅ Use (key finding) | Sec 2 EDA |
| fig3_feature_distributions.png | Distribution of all features | ⚠️ Optional | Sec 2 EDA (background) |
| fig4_spearman_correlation.png | Feature Spearman correlations | ✅ Use | Sec 2 EDA / Sec 3 FE |
| fig5_spatial_violation.png | Geographic heatmap of violations | ✅ Use (visual impact) | Sec 1 Intro |
| fig6_time_patterns.png | Temporal patterns (hour/day/month) | ✅ Use | Sec 2 EDA |
| fig7_missing_values.png | Missing value heatmap | ⚠️ Skip | — |
| fig_h_noise_diagnosis.png | Label noise diagnosis (tc=1 subset) | ✅ Use (supports Sec 2) | Sec 2 EDA |

### figures/ — Model Training Phase (15 files)

| File | Content | PPT Usability | Recommended Section |
|------|---------|--------------|---------------------|
| lgbm_feature_importance.png | LightGBM feature importance | ✅ Use | Sec 2/3 Feature Engineering |
| shap_summary.png | SHAP beeswarm plot | ✅ Use (model explanation) | Sec 5 Experiments |
| shap_bar.png | Mean absolute SHAP values | ✅ Use | Sec 5 Experiments |
| shap_dep_total_count.png | SHAP dependence on total_count | ✅ Use (key insight) | Sec 2 EDA |
| shap_dep_grid_te.png | SHAP dependence on grid_te | ⚠️ Optional | — |
| shap_dep_grid_period_te.png | SHAP dependence on period TE | ⚠️ Optional | — |
| pred_vs_actual.png | Predicted vs actual scatter | ✅ Use | Sec 3/4 Evaluation |
| prediction_distribution.png | Distribution of model predictions | ⚠️ Optional | — |
| model_comparison.png | Model performance comparison | ✅ Use | Sec 3 Baseline |
| ablation_study.png | Ablation study results | ✅ Use | Sec 2/3 Feature Engineering |
| grouped_spearman.png | Grouped Spearman analysis | ✅ Use | Sec 5 Experiments |
| tabm_correlation.png | TabM correlation matrix | ⚠️ Optional (Exp E) | Sec 5 (if mentioning DL) |
| te_distribution_shift.png | Train-test TE distribution shift | ✅ Use | Sec 3/4 Gap Analysis |
| av_feature_importance.png | Adversarial validation feature importance | ✅ Use | Sec 3 Gap Diagnosis |
| av_probability_distribution.png | Adversarial validation probability | ✅ Use | Sec 3 Gap Diagnosis |

---

## Missing Figures — Need to Generate (5 total)

### P0: Critical — Must Have

#### 1. `figures/score_progression.png`
**Purpose:** Show the full journey from v1 to Exp I-A, visualizing both OOF and Platform improvement.  
**Used in:** Section 1 (Intro teaser) and Section 3 (Baseline Development).

**Data:**
```
Label         OOF      Platform   Key annotation
v1            0.5880   0.5222     "Initial LGB+XGB"
v2            0.6012   0.5338     "+n_estimators"
v3            0.6408   0.5620     "+Optuna tuning"
v7            0.6429   0.5636     "+log1p(tc) weighting"
Exp C         0.6464   0.5698     "Rank-Target Training"
Exp I-A       0.6478   0.5705     "+More Iterations (BEST)"
```

**Code sketch (add to notebook Section I or a new cell):**
```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

labels = ['v1', 'v2', 'v3', 'v7', 'Exp C', 'Exp I-A']
oof    = [0.5880, 0.6012, 0.6408, 0.6429, 0.6464, 0.6478]
plat   = [0.5222, 0.5338, 0.5620, 0.5636, 0.5698, 0.5705]

annotations = {
    'v3': 'Optuna\nTuning',
    'v7': 'Sample\nWeighting',
    'Exp C': 'Rank-Target\nTraining',
    'Exp I-A': 'More Iterations\n(BEST: 0.5705)',
}

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, oof, 'o-', color='steelblue', linewidth=2, markersize=7, label='OOF Spearman')
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
        ax.annotate(annotations[lbl], xy=(i, plat[i]), xytext=(i, plat[i]-0.025),
                    ha='center', fontsize=8.5, color='darkorange',
                    arrowprops=dict(arrowstyle='->', color='darkorange', lw=1))

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/score_progression.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

#### 2. `figures/rank_target_diagram.png`
**Purpose:** Visually explain WHY rank-target training works — aligning training with Spearman.  
**Used in:** Section 4 Key Innovation (cannot be skipped).

**Code sketch:**
```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

# Load actual y values from training data
# y = train_df['invalid_ratio'].values  # actual data

# For illustration only, use synthetic data with similar distribution
np.random.seed(42)
y_raw = np.concatenate([
    np.zeros(3000),  # tc=1 boundary mass
    np.ones(2000),
    np.random.beta(0.5, 3, 15000)  # main distribution (right-skewed)
])
y_rank = rankdata(y_raw) / len(y_raw)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: raw target
axes[0].hist(y_raw, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].set_title('Original Target\n(invalid_ratio)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Count')
axes[0].annotate('Skewed\ndistribution', xy=(0.5, 3000), fontsize=9, color='gray')

# Right: rank target
axes[1].hist(y_rank, bins=50, color='darkorange', edgecolor='white', alpha=0.85)
axes[1].set_title('Rank-Transformed Target\nrankdata(y) / N', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Count')
axes[1].annotate('Uniform\ndistribution\n→ Aligns with Spearman', xy=(0.3, 1500), fontsize=9, color='darkorange')

fig.suptitle('Rank-Target Transformation: Directly Optimizing Spearman Ranking',
             fontsize=13, fontweight='bold', y=1.02)

# Add arrow between subplots
fig.text(0.5, 0.5, '→\ny_rank = rankdata(y) / N',
         ha='center', va='center', fontsize=12, color='black',
         transform=fig.transFigure)

plt.tight_layout()
plt.savefig('figures/rank_target_diagram.png', dpi=150, bbox_inches='tight')
plt.show()
```

> **Note:** Replace the synthetic data with actual `train_df['invalid_ratio']` values for accuracy.

---

#### 3. `figures/experiment_summary_chart.png`
**Purpose:** Side-by-side comparison of all experiments, showing what worked and what did not.  
**Used in:** Section 5 Experiment Summary.

**Data:**
```
Experiment    OOF      Platform   Status
v1            0.5880   0.5222     baseline
v3            0.6408   0.5620     baseline (Optuna)
v7            0.6429   0.5636     baseline (best pre-sprint)
Exp D (AV)    —        —          diagnostic only
Exp E (TabM)  0.4445   —          failed (DL)
Exp H (Noise) 0.6442   0.5613     failed (OOF gain, plat drop)
Exp G (PL)    0.6463   —          null result
Exp C (Rank)  0.6464   0.5698     SUCCESS
Exp I-A       0.6478   0.5705     BEST
```

**Code sketch:**
```python
import matplotlib.pyplot as plt
import numpy as np

experiments = ['v1 (Initial)', 'v3 (+Optuna)', 'v7 (+Weighting)',
               'Exp E (TabM)', 'Exp H (Noise)', 'Exp G (Pseudo-label)',
               'Exp C (Rank-Target)', 'Exp I-A (More Iter.)']
oof_scores  = [0.5880, 0.6408, 0.6429, 0.4445, 0.6442, 0.6463, 0.6464, 0.6478]
plat_scores = [0.5222, 0.5620, 0.5636, None,   0.5613, None,   0.5698, 0.5705]
status      = ['baseline','baseline','baseline','failed','failed','null','success','best']

color_map = {'baseline': '#90CAF9', 'failed': '#EF9A9A', 'null': '#FFE082',
             'success': '#A5D6A7', 'best': '#2E7D32'}

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(experiments))

for i, (exp, oof, plat, st) in enumerate(zip(experiments, oof_scores, plat_scores, status)):
    color = color_map[st]
    ax.barh(i - 0.2, oof, height=0.35, color=color, alpha=0.9)
    if plat is not None:
        ax.barh(i + 0.2, plat, height=0.35, color=color, alpha=0.6, hatch='//')

ax.set_yticks(y_pos)
ax.set_yticklabels(experiments, fontsize=10)
ax.set_xlabel('Spearman Correlation', fontsize=11)
ax.set_title('Experiment Summary: OOF and Platform Scores', fontsize=13, fontweight='bold')
ax.axvline(x=0.5636, color='gray', linestyle=':', alpha=0.7, label='v7 baseline (Platform 0.5636)')
ax.axvline(x=0.5705, color='#2E7D32', linestyle='--', alpha=0.8, label='Best (Exp I-A Platform 0.5705)')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#90CAF9', label='Baseline'),
    Patch(facecolor='#A5D6A7', label='Success'),
    Patch(facecolor='#2E7D32', label='Best Result'),
    Patch(facecolor='#EF9A9A', label='Failed'),
    Patch(facecolor='#FFE082', label='Null Result'),
    Patch(facecolor='gray', alpha=0.5, hatch='//', label='Platform score'),
    Patch(facecolor='gray', alpha=0.5, label='OOF score'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
ax.set_xlim(0.40, 0.68)
plt.tight_layout()
plt.savefig('figures/experiment_summary_chart.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

### P1: Recommended (Adds completeness)

#### 4. `figures/feature_engineering_pipeline.png`
**Purpose:** Show the Tier2 feature engineering flow visually.  
**Used in:** Section 2 Feature Engineering.

Can be created as a simple text-based flowchart in matplotlib or drawn in PowerPoint/Canva and exported. The flow is:
```
Raw 10 Features
  ↓
[Coordinate Binning]  → grid_x, grid_y, grid_id
  ↓
[K-Fold Target Encoding (k=5)]  → grid_te, period_te, …
  ↓
[Cyclic Encoding]  → sin/cos(hour), sin/cos(month)
  ↓
[Cross Features]  → total_count × grid_te
  ↓
Tier2 Feature Set (~20 features)
  ↓
[Sample Weighting]  → log1p(total_count)
```

#### 5. `figures/oof_platform_gap.png`
**Purpose:** Show that the OOF-Platform gap is stable (~0.077), proving the issue is distribution shift (not overfitting).  
**Used in:** Section 3 Gap Diagnosis.

| Version | OOF    | Platform | Gap    |
|---------|--------|----------|--------|
| v1      | 0.5880 | 0.5222   | 0.0658 |
| v3      | 0.6408 | 0.5620   | 0.0788 |
| v7      | 0.6429 | 0.5636   | 0.0793 |
| Exp C   | 0.6464 | 0.5698   | 0.0766 |
| Exp H   | 0.6442 | 0.5613   | 0.0829 |
| Exp I-A | 0.6478 | 0.5705   | 0.0773 |

---

## Summary

| Priority | Figure | Status |
|----------|--------|--------|
| P0 | score_progression.png | ✅ Generated (2026-04-12) |
| P0 | rank_target_diagram.png | ✅ Generated (2026-04-12, actual data) |
| P0 | experiment_summary_chart.png | ✅ Generated (2026-04-12) |
| P1 | feature_engineering_pipeline.png | ✅ Generated (2026-04-12) |
| P1 | oof_platform_gap.png | ✅ Generated (2026-04-12) |
| ✅ | fig5_spatial_violation.png | Already exists |
| ✅ | fig2_totalcount_vs_violation.png | Already exists |
| ✅ | av_probability_distribution.png | Already exists |
| ✅ | shap_summary.png | Already exists |
| ✅ | lgbm_feature_importance.png | Already exists |
| ✅ | pred_vs_actual.png | Already exists |
| ✅ | ablation_study.png | Already exists |
